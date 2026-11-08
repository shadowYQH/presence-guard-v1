package com.presenceguard.app.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ConcurrentCamera
import androidx.camera.core.ConcurrentCamera.SingleCameraConfig
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCase
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ExperimentalCameraProviderConfiguration
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import com.presenceguard.app.MainActivity
import com.presenceguard.app.R
import com.presenceguard.app.audio.AlertPlayer
import com.presenceguard.app.audio.AlertTone
import com.presenceguard.app.core.FrameResult
import com.presenceguard.app.core.Roi
import com.presenceguard.app.core.Sensitivity
import com.presenceguard.app.data.Settings
import com.presenceguard.app.detect.FrameAnalyzer
import com.presenceguard.app.detect.PoseEngine
import com.presenceguard.app.detect.PresenceFusionEngine
import com.presenceguard.app.detect.TflitePersonEngine
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * 常驻检测服务。
 *
 * 把相机管线和两个推理引擎放在前台服务里，而不是 Activity 里，是因为：
 *  - Activity 一旦不可见 / 屏幕熄灭，绑定在 Activity 生命周期上的 CameraX 就会被解绑，
 *    检测直接停止。绑到本服务的 [LifecycleOwner] 上，息屏、切后台都能继续跑。
 *  - 前台服务声明 `foregroundServiceType="camera"` 后，系统允许在后台持有相机。
 *  - 配合 PARTIAL_WAKE_LOCK，熄屏后 CPU 不停机，推理和提示音才不会被挂起。
 */
class DetectionService : Service(), LifecycleOwner {

    enum class Phase { INITIALIZING, RUNNING, PAUSED, NO_PERMISSION, ERROR }

    data class Status(
        val phase: Phase,
        val present: Boolean = false,
        val personCount: Int = 0,
        val confidence: Float = 0f,
        val message: String? = null
    )

    interface Listener {
        /** 每帧回调，来自分析线程；UI 层需自行切主线程 */
        fun onFrame(result: FrameResult, outcome: PresenceFusionEngine.Outcome)

        fun onStatus(status: Status)

        fun onError(message: String)

        /**
         * 巡检结果回调。和 [onFrame] 完全分开，因为巡检**不参与主判定**：
         * 它只是「另一面 / 更广的视野里有没有人」的一个旁路观察，用来补漏。
         *
         * 给默认实现是为了不逼着已有实现类都改一遍。
         */
        fun onInspection(present: Boolean, score: Float, source: String) {}
    }

    inner class LocalBinder : Binder() {
        val service: DetectionService get() = this@DetectionService
    }

    private val binder = LocalBinder()
    private val lifecycleRegistry = LifecycleRegistry(this)
    override val lifecycle: Lifecycle get() = lifecycleRegistry

    private lateinit var settings: Settings
    private lateinit var alertPlayer: AlertPlayer
    private lateinit var cameraExecutor: ExecutorService

    private val fusion = PresenceFusionEngine()
    private val listeners = CopyOnWriteArrayList<Listener>()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var poseEngine: PoseEngine? = null
    private var personEngine: TflitePersonEngine? = null
    private var analyzer: FrameAnalyzer? = null
    private var analysisUseCase: ImageAnalysis? = null
    private var previewUseCase: Preview? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var wakeLock: PowerManager.WakeLock? = null

    /**
     * 第二颗摄像头（与当前镜头朝向相反）的巡检分析器。
     *
     * 它和 [analyzer] 共用同一套引擎实例 —— 不是省事，是必须：
     * TFLite 的 Interpreter 不是线程安全的，ML Kit 的检测器也不适合多实例并存。
     * 两者都挂在同一个单线程 [cameraExecutor] 上，推理天然串行，
     * 「同时获取」在取流层是真的并发，「同时运算」在推理层是背靠背排队。
     * 对 2 秒一次的巡检来说这个差别可以忽略，却省掉一整类竞态崩溃。
     */
    private var secondEye: FrameAnalyzer? = null

    private var secondAnalysisUseCase: ImageAnalysis? = null

    /** 前后双摄是否真的并发绑上了。多数机型不支持，此时降级为单镜头巡检 */
    @Volatile
    private var concurrentActive = false

    /**
     * Activity 传入的预览 SurfaceProvider。
     *
     * 相机管线在服务里，但「画面显示到哪」只有 Activity 知道，所以这里存一份，
     * 等 [previewUseCase] 创建好再补上。界面不可见时置空，Preview 就不再往屏幕送帧，
     * 省掉这一路的显示功耗——检测走的是 ImageAnalysis，和预览互不影响。
     */
    private var pendingSurfaceProvider: Preview.SurfaceProvider? = null

    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var sensitivity = Sensitivity.MEDIUM
    private var paused = false
    private var initializing = false

    /** 当前绑定的相机，变焦要经过它 */
    private var camera: Camera? = null

    /** 变焦倍率。CameraX 的 zoom 对预览与分析流同时生效 */
    private var zoomRatio = 1f

    /** 后置物理镜头列表。有些机型会把广角/长焦都暴露出来，可以手动切 */
    private var backInfos: List<CameraInfo> = emptyList()
    private var lensIndex = 0

    /**
     * 预览是否应当输出画面。界面不可见（息屏 / 切后台）时置 false，
     * 此时 [bindCamera] 干脆不绑 Preview 用例——不是「绑了但不送帧」，
     * 是连那条相机输出流都不创建，从根上省掉这部分功耗。
     */
    private var previewVisible = false

    private var status = Status(Phase.INITIALIZING)
    private var lastAlertAt = 0L
    private var lastNotifyAt = 0L

    // ------------------------------------------------------------ 生命周期

    override fun onCreate() {
        super.onCreate()
        settings = Settings(this)
        alertPlayer = AlertPlayer(this)
        cameraExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "camera-analysis") }

        lensFacing = settings.lensFacing
        sensitivity = settings.sensitivity
        zoomRatio = settings.zoomRatio
        lensIndex = settings.lensIndex
        alertPlayer.setTone(settings.tone)
        alertPlayer.setVibrateEnabled(true)

        createChannel()
        acquireWakeLock()

        // 服务活着就认为相机应当工作
        lifecycleRegistry.currentState = Lifecycle.State.RESUMED

        ensureForeground(Status(Phase.INITIALIZING))
        bootstrap()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopSelf()
                return START_NOT_STICKY
            }
            ACTION_TOGGLE_PAUSE -> togglePause()
        }
        ensureForeground(status)
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        super.onDestroy()
        lifecycleRegistry.currentState = Lifecycle.State.DESTROYED
        runCatching { analysisUseCase?.clearAnalyzer() }
        runCatching { secondAnalysisUseCase?.clearAnalyzer() }
        runCatching { cameraProvider?.unbindAll() }

        val engine = poseEngine
        val detector = personEngine
        val analyzerRef = analyzer
        val secondRef = secondEye
        // 在途帧可能还卡在 ML Kit 回调里，立刻 close 会让它重复完成 Task 而崩溃
        mainHandler.postDelayed({
            runCatching { engine?.close() }
            runCatching { detector?.close() }
            runCatching { analyzerRef?.shutdown() }
            runCatching { secondRef?.shutdown() }
        }, 600L)

        runCatching { alertPlayer.release() }
        cameraExecutor.shutdown()
        releaseWakeLock()
    }

    // ------------------------------------------------------------ 对外接口

    fun addListener(listener: Listener) {
        if (!listeners.contains(listener)) listeners.add(listener)
        listener.onStatus(status)
    }

    fun removeListener(listener: Listener) {
        listeners.remove(listener)
    }

    fun currentStatus(): Status = status

    fun currentSensitivity(): Sensitivity = sensitivity

    fun setSensitivity(value: Sensitivity) {
        sensitivity = value
        settings.sensitivity = value
        analyzer?.config = value.toConfig()
        analyzer?.tilesPerFrame = tilesForSensitivity(value)
        fusion.reset()
        publishStatus(status.copy())
    }

    /**
     * 精细路强度随灵敏度变化：越灵敏，每帧多扫几片瓦片，远处的小目标越快被发现。
     * 单次推理实测约 80ms，所以这不是免费的——每多一片就多 80ms 帧耗时。
     */
    private fun tilesForSensitivity(value: Sensitivity): Int {
        if (settings.powerSave) return 0
        return when (value) {
            Sensitivity.HIGH -> 3
            Sensitivity.MEDIUM -> 2
            Sensitivity.LOW -> 1
        }
    }

    fun setAudioEnabled(enabled: Boolean) {
        settings.audioEnabled = enabled
        publishStatus(status.copy())
    }

    fun isAudioEnabled(): Boolean = settings.audioEnabled

    fun currentTone(): AlertTone = alertPlayer.currentTone()

    fun cycleTone(): AlertTone {
        val next = alertPlayer.currentTone().next()
        alertPlayer.setTone(next)
        settings.tone = next
        alertPlayer.play()
        return next
    }

    fun setAlertInterval(millis: Long) {
        settings.alertIntervalMs = millis
    }

    fun previewTone() = alertPlayer.play()

    fun isAlarmAudible(): Boolean = alertPlayer.isAlarmAudible()

    fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        settings.lensFacing = lensFacing
        fusion.reset()
        mainHandler.post { bindCamera() }
    }

    // ------------------------------------------------------------ 焦段 / 镜头

    fun currentZoom(): Float = zoomRatio

    /** 当前镜头支持的变焦区间；相机还没就绪时给一个保守的默认值 */
    fun zoomRange(): Pair<Float, Float> {
        val state = camera?.cameraInfo?.zoomState?.value
            ?: return 1f to Settings.MAX_ZOOM
        return state.minZoomRatio to state.maxZoomRatio
    }

    /**
     * 设置变焦倍率。
     *
     * 走 CameraX 的 zoom 而不是「裁切预览画面」——前者是传感器级的区域裁切（部分机型
     * 还会自动切换到长焦镜头），分析流拿到的就是放大后的真实像素，检测精度跟着提升；
     * 后者只是把已经很模糊的一小块拉大，对识别没有任何帮助。
     */
    fun setZoomRatio(ratio: Float) {
        zoomRatio = ratio.coerceIn(1f, Settings.MAX_ZOOM)
        settings.zoomRatio = zoomRatio
        applyZoom()
    }

    private fun applyZoom() {
        val cam = camera ?: return
        val state = cam.cameraInfo.zoomState.value ?: return
        val target = zoomRatio.coerceIn(state.minZoomRatio, state.maxZoomRatio)
        runCatching { cam.cameraControl.setZoomRatio(target) }
            .onFailure { Log.w(TAG, "setZoomRatio failed", it) }
        Log.i(TAG, "zoom=$target (range ${state.minZoomRatio}..${state.maxZoomRatio})")
    }

    /** 后置可切换的物理镜头数量。多数机型只暴露一个，此时镜头发切换没有意义 */
    fun lensCount(): Int = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
        backInfos.size.coerceAtLeast(1)
    } else {
        1
    }

    // ------------------------------------------------------------ 双指缩放（数字裁切）

    fun currentCropScale(): Float = settings.cropScale

    /**
     * 设置数字裁切倍数（双指缩放）。
     *
     * 这是**纯软件裁切**：只改送进模型的那块像素，相机参数一律不动，预览画面也不变。
     * 它和 [setZoomRatio] 是两条完全独立的路：
     *
     *  - [setZoomRatio] -> 相机传感器级区域裁切，部分机型自动切长焦，预览同步放大，
     *    对识别是**真增益**（拿到的是真实像素）。
     *  - 本方法 -> 从已经采到的帧里裁中心一块送模型，等效加焦距，但像素没有增加。
     */
    fun setCropScale(scale: Float) {
        val next = scale.coerceIn(1f, Settings.MAX_CROP_SCALE)
        if (next == settings.cropScale) return
        settings.cropScale = next
        analyzer?.cropScale = next
        Log.i(TAG, "cropScale=$next")
    }

    fun currentLensIndex(): Int = lensIndex

    /**
     * 切换到下一个后置物理镜头。
     *
     * 通过 [CameraSelector.Builder.addCameraFilter] 把候选相机列表裁到只剩目标那一颗，
     * 这样 CameraX 就会绑定到具体的物理镜头，而不是系统给的默认逻辑相机。
     */
    fun cycleLens() {
        if (lensCount() <= 1) return
        lensIndex = (lensIndex + 1) % backInfos.size
        settings.lensIndex = lensIndex
        fusion.reset()
        Log.i(TAG, "switch to back lens #$lensIndex of ${backInfos.size}")
        mainHandler.post { bindCamera() }
    }

    // ------------------------------------------------------------ 选区

    fun currentRois(): List<Roi> = settings.rois

    /**
     * 更新选区。非空即开启「选区模式」：只在这些多边形里判定，
     * 区域内做放大识别，区域外完全不看。
     */
    fun setRois(list: List<Roi>) {
        settings.rois = list
        analyzer?.rois = list
        analyzer?.roiPerFrame = roiBudgetFor(list.size)
        fusion.reset()
        Log.i(TAG, "rois=${list.size} budget=${roiBudgetFor(list.size)}")
    }

    private fun roiBudgetFor(count: Int): Int {
        if (count <= 0) return 0
        // 一个选区 ≈ 检测 + 姿态两次推理。每帧多扫一个就多一份固定成本，
        // 所以上限卡在 2，更多选区靠游标轮转覆盖。
        return count.coerceAtMost(if (settings.powerSave) 1 else 2)
    }

    fun setPowerSave(enabled: Boolean) {
        settings.powerSave = enabled
        analyzer?.minFrameIntervalMs = if (enabled) POWER_SAVE_FRAME_INTERVAL else 0L
        // 省电模式下关掉瓦片精细路：远处的小目标会漏，但功耗能降一大截
        analyzer?.tilesPerFrame = if (enabled) 0 else tilesForSensitivity(sensitivity)
        // 选区路是「有明确目标」时才开的，省电下保留但减到每帧 1 个
        analyzer?.roiPerFrame = roiBudgetFor(analyzer?.rois?.size ?: 0)
        // 副摄那条输出流是这一档最大的开销，省电时必须摘掉；主摄巡检保留
        if (dualEyeEnabled() != (secondAnalysisUseCase != null)) {
            mainHandler.post { bindCamera() }
        }
    }

    fun isPowerSave(): Boolean = settings.powerSave

    fun togglePause(): Boolean {
        paused = !paused
        if (paused) {
            analysisUseCase?.clearAnalyzer()
            secondAnalysisUseCase?.clearAnalyzer()
        } else {
            analyzer?.let { analysisUseCase?.setAnalyzer(cameraExecutor, it) }
            // 副摄用例可能因为暂停时没建出来，恢复时直接重绑一次最省心
            if (dualEyeEnabled() && secondEye?.inspectionOnly == true) {
                mainHandler.post { bindCamera() }
            }
        }
        fusion.reset()
        secondEye?.let { runCatching { it.rebind(sensitivity.toConfig(), false) } }
        publishStatus(status.copy(phase = if (paused) Phase.PAUSED else Phase.RUNNING, present = false, personCount = 0))
        return paused
    }

    fun isPaused(): Boolean = paused

    /**
     * 接上画面预览。传 `null` 表示界面不可见，停掉预览渲染。
     *
     * 必须在主线程调用（CameraX 的 setSurfaceProvider 有这个要求）。
     * Activity 可能在相机就绪之前就调用，所以这里先缓存，等 Preview 建好再补。
     */
    fun setPreviewSurfaceProvider(surfaceProvider: Preview.SurfaceProvider?) {
        pendingSurfaceProvider = surfaceProvider
        runCatching { previewUseCase?.setSurfaceProvider(surfaceProvider) }
    }

    /**
     * 界面是否可见。
     *
     * 这里做的是「重新绑相机」而不是「设个标志位」：
     * CameraX 的 Preview 用例只要还绑在相机上，就会持续占用一条相机输出流 + 显示合成，
     * 即便 surface 已经被摘掉。把 Preview 从绑定的用例列表里去掉，[bindCamera] 就只绑
     * ImageAnalysis，那条输出流根本不会被创建——这是唯一能让预览功耗真正归零的做法。
     *
     * 代价是一次相机的关闭与重开（几百毫秒）。只在息屏 / 切后台时发生，可以接受。
     */
    fun setPreviewVisible(visible: Boolean) {
        if (previewVisible == visible) return
        previewVisible = visible
        Log.i(TAG, "previewVisible=$visible")
        if (analyzer == null) return
        mainHandler.post { bindCamera() }
    }

    fun isPreviewVisible(): Boolean = previewVisible

    // ------------------------------------------------------------ 初始化

    private fun bootstrap() {
        if (initializing) return
        initializing = true

        cameraExecutor.execute {
            val engine = runCatching { PoseEngine() }.getOrNull()
            val detector = runCatching {
                // 用固定的低阈值做粗筛：真正的判定门槛在融合层，而且灵敏度可以
                // 中途切换，而 TFLite 的 threshold 在构造时就固化了，不能跟着变。
                TflitePersonEngine(applicationContext, DETECTOR_PRE_FILTER)
            }.getOrNull()

            mainHandler.post {
                if (engine == null) {
                    updateStatus(Status(Phase.ERROR, message = "姿态模型加载失败"))
                    return@post
                }
                poseEngine = engine
                personEngine = detector
                analyzer = FrameAnalyzer(engine, detector, fusion, frameListener).apply {
                    config = sensitivity.toConfig()
                    mirrored = lensFacing == CameraSelector.LENS_FACING_FRONT
                    minFrameIntervalMs = if (settings.powerSave) POWER_SAVE_FRAME_INTERVAL else 0L
                    tilesPerFrame = tilesForSensitivity(sensitivity)
                    rois = settings.rois
                    roiPerFrame = roiBudgetFor(settings.rois.size)
                    cropScale = settings.cropScale
                    inspectionIntervalMs = if (inspectionEnabled()) INSPECTION_INTERVAL_MS else 0L
                    onInspectionHit = { present, score -> onInspectionResult(present, score, "主摄") }
                }
                // 副摄巡检器：只做整帧粗判，自己的融合状态机，绝不投票给主判定
                secondEye = FrameAnalyzer(engine, detector, PresenceFusionEngine(), silentListener)
                    .apply {
                        config = sensitivity.toConfig()
                        inspectionOnly = true
                        inspectionIntervalMs = INSPECTION_INTERVAL_MS
                        onInspectionHit = { present, score ->
                            onInspectionResult(present, score, oppositeLensName())
                        }
                    }
                bindCamera()
                updateStatus(Status(Phase.RUNNING))
            }
        }
    }

    private fun bindCamera() {
        val analyzerRef = analyzer ?: run {
            Log.w(TAG, "bindCamera skipped: analyzer not ready")
            return
        }
        Log.i(TAG, "bindCamera() previewVisible=$previewVisible facing=$lensFacing")

        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = runCatching { future.get() }.getOrNull()
            if (provider == null) {
                updateStatus(Status(Phase.ERROR, message = "相机初始化失败"))
                return@addListener
            }
            Log.i(TAG, "camera provider ready")
            cameraProvider = provider
            provider.unbindAll()
            camera = null
            concurrentActive = false
            backInfos = collectBackCameras(provider)

            // unbindAll 只是解绑，分析器还挂在旧用例上；不摘掉的话旧管线会把帧
            // 继续送进同一个分析器，和新建的用例抢着关 ImageProxy，是必崩的写法。
            runCatching { analysisUseCase?.clearAnalyzer() }
            runCatching { secondAnalysisUseCase?.clearAnalyzer() }
            secondAnalysisUseCase = null

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelectorFor(ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analyzerRef.rebind(
                config = sensitivity.toConfig(),
                mirrored = lensFacing == CameraSelector.LENS_FACING_FRONT
            )
            analyzerRef.minFrameIntervalMs = if (settings.powerSave) POWER_SAVE_FRAME_INTERVAL else 0L
            analyzerRef.tilesPerFrame = tilesForSensitivity(sensitivity)
            analyzerRef.rois = settings.rois
            analyzerRef.roiPerFrame = roiBudgetFor(settings.rois.size)
            analyzerRef.cropScale = settings.cropScale
            analyzerRef.inspectionIntervalMs = if (inspectionEnabled()) INSPECTION_INTERVAL_MS else 0L

            analysisUseCase = analysis
            if (!paused) analysis.setAnalyzer(cameraExecutor, analyzerRef)

            var preview: Preview? = null
            if (previewVisible) {
                preview = Preview.Builder().build()
                previewUseCase = preview
                // 关键：不接 SurfaceProvider 的话，相机照样开、检测照样跑，但屏幕上永远没画面
                pendingSurfaceProvider?.let { preview.setSurfaceProvider(it) }
            } else {
                // 界面不可见：干脆不绑 Preview，省掉整条预览输出流
                previewUseCase = null
            }

            val secondary = buildSecondEye()
            var bound = bindCameras(provider, analysis, preview, secondary)
            if (!bound) {
                Log.w(TAG, "bind failed, retry with other lens")
                lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                    CameraSelector.LENS_FACING_FRONT
                } else {
                    CameraSelector.LENS_FACING_BACK
                }
                analyzerRef.mirrored = lensFacing == CameraSelector.LENS_FACING_FRONT
                bound = bindCameras(provider, analysis, preview, secondary)
                if (!bound) updateStatus(Status(Phase.ERROR, message = "无法打开摄像头"))
            }

            if (bound) {
                applyZoom()
                Log.i(
                    TAG,
                    "bound facing=$lensFacing lens=$lensIndex/${backInfos.size} " +
                        "preview=$previewVisible dual=$concurrentActive inspect=${inspectionEnabled()}"
                )
                // 并发双摄成功与否要等这一刻才知道，推一次状态让界面把按钮文案改对
                publishStatus(status.copy())
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun resolutionSelectorFor(width: Int, height: Int) = ResolutionSelector.Builder()
        .setResolutionStrategy(
            ResolutionStrategy(
                Size(width, height),
                // 精细路靠瓦片放大视野，但采集本身分辨率太低时，放大的只是模糊块。
                // 1080p 是 865 上性价比最高的折中。
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
            )
        )
        .build()

    /**
     * 第二颗摄像头（另一面）的巡检用例。
     *
     * 分辨率刻意压到 720p：并发双摄的**流组合**是由相机 HAL 白名单决定的，
     * 两路都按 1080p 去要，很多机型会直接判定组合不合法而拒绝绑定。
     * 巡检只判断"那一面有没有人"，720p 足够；换来的是并发成功率更高。
     */
    private fun buildSecondEye(): ImageAnalysis? {
        val eye = secondEye ?: return null
        if (!dualEyeEnabled() || paused) return null

        val useCase = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelectorFor(INSPECTION_WIDTH, INSPECTION_HEIGHT))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        eye.config = sensitivity.toConfig()
        eye.inspectionIntervalMs = INSPECTION_INTERVAL_MS
        useCase.setAnalyzer(cameraExecutor, eye)
        secondAnalysisUseCase = useCase
        return useCase
    }

    /**
     * 绑定相机。优先尝试前后双摄并发，失败则退回单镜头。
     *
     * 并发双摄不是"调个开关就有"的东西：CameraX 的 [ProcessCameraProvider.bindToLifecycle]
     * 接受多组 [SingleCameraConfig]，但底层要求相机 HAL 声明支持
     * `LOGICAL_MULTI_CAMERA` 的并发流组合。绝大多数中低端机、以及不少旗舰的
     * 「前+后」组合都会直接抛异常。所以这里的流程是「试 -> 失败就降级」，
     * 而不是「先探测再决定」——探测一次同样要开相机，成本比试一次还高。
     */
    private fun bindCameras(
        provider: ProcessCameraProvider,
        analysis: ImageAnalysis,
        preview: Preview?,
        secondary: ImageAnalysis?
    ): Boolean {
        if (secondary != null) {
            val attempt = runCatching { bindConcurrent(provider, analysis, preview, secondary) }
            val concurrent = attempt.getOrNull()
            // getCameras() 的顺序和传入的 config 顺序一致，第 0 个就是主摄
            val primary = concurrent?.cameras?.firstOrNull()

            if (primary != null) {
                concurrentActive = true
                camera = primary
                Log.i(TAG, "concurrent bind ok: ${concurrent.cameras.size} cameras")
                return true
            }

            concurrentActive = false
            Log.w(
                TAG,
                "concurrent bind unsupported, degrade to single camera: ${attempt.exceptionOrNull()?.message}"
            )
            // 并发失败时可能已经绑了一半，先清干净再走单路
            runCatching { secondAnalysisUseCase?.clearAnalyzer() }
            secondAnalysisUseCase = null
            runCatching { provider.unbindAll() }
            camera = null
        }

        concurrentActive = false
        val useCases = ArrayList<UseCase>(2)
        preview?.let { useCases += it }
        useCases += analysis
        return runCatching {
            camera = provider.bindToLifecycle(
                this, selectorFor(lensFacing), *useCases.toTypedArray()
            )
        }.onFailure { Log.w(TAG, "single bind failed", it) }.isSuccess
    }

    @androidx.annotation.OptIn(ExperimentalCameraProviderConfiguration::class)
    private fun bindConcurrent(
        provider: ProcessCameraProvider,
        analysis: ImageAnalysis,
        preview: Preview?,
        secondary: ImageAnalysis
    ): ConcurrentCamera {
        val primaryGroup = UseCaseGroup.Builder().apply {
            addUseCase(analysis)
            preview?.let { addUseCase(it) }
        }.build()
        val secondaryGroup = UseCaseGroup.Builder().addUseCase(secondary).build()

        val opposite = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }

        return provider.bindToLifecycle(
            listOf(
                SingleCameraConfig(selectorFor(lensFacing), primaryGroup, this),
                SingleCameraConfig(selectorFor(opposite), secondaryGroup, this)
            )
        )
    }

    /**
     * 收集可用的后置物理镜头。
     *
     * 用 [CameraSelector.DEFAULT_BACK_CAMERA.filter] 反查，而不是读 Camera2 的私有字段：
     * 前者是公开 API，对「逻辑相机 + 物理相机」的差异由 CameraX 自己处理。
     */
    private fun collectBackCameras(provider: ProcessCameraProvider): List<CameraInfo> {
        val infos = runCatching {
            provider.availableCameraInfos.filter { info ->
                CameraSelector.DEFAULT_BACK_CAMERA.filter(listOf(info)).isNotEmpty()
            }
        }.getOrDefault(emptyList())
        if (infos.isNotEmpty()) {
            lensIndex = lensIndex.coerceIn(0, infos.size - 1)
        } else {
            lensIndex = 0
        }
        return infos
    }

    /**
     * 构造相机选择器。后置有多颗物理镜头时，把候选列表裁到只剩用户选中的那颗。
     */
    private fun selectorFor(facing: Int): CameraSelector {
        val builder = CameraSelector.Builder().requireLensFacing(facing)
        if (facing == CameraSelector.LENS_FACING_BACK && backInfos.size > 1) {
            val target = backInfos[lensIndex.coerceIn(0, backInfos.size - 1)]
            builder.addCameraFilter { infos ->
                infos.filter { it == target }.ifEmpty { infos }
            }
        }
        return builder.build()
    }

    // ------------------------------------------------------------ 帧回调

    private val frameListener = object : FrameAnalyzer.Listener {
        override fun onResult(result: FrameResult, outcome: PresenceFusionEngine.Outcome) {
            val now = SystemClock.elapsedRealtime()

            // 持续报警：只要画面里还有人，就按设定的间隔反复提示，
            // 而不是只在「刚出现」那一下响一次。
            if (outcome.present) {
                if (settings.audioEnabled && now - lastAlertAt >= settings.alertIntervalMs) {
                    lastAlertAt = now
                    alertPlayer.play()
                }
            } else {
                // 人离开后清零，下次出现能立刻响
                lastAlertAt = 0L
            }

            val newStatus = status.copy(
                phase = if (paused) Phase.PAUSED else Phase.RUNNING,
                present = outcome.present,
                personCount = outcome.personCount,
                confidence = outcome.smoothedScore
            )
            if (newStatus != status) {
                status = newStatus
                notifyListeners { it.onStatus(status) }
                if (now - lastNotifyAt > NOTIFY_REFRESH_MS) {
                    lastNotifyAt = now
                    refreshNotification()
                }
            }

            notifyListeners { it.onFrame(result, outcome) }
        }

        override fun onError(message: String) {
            notifyListeners { it.onError(message) }
        }
    }

    /**
     * 副摄分析器用的空监听器。
     *
     * [FrameAnalyzer.inspectionOnly] 打开后它永远不会 publish，这两个回调只是签名占位。
     */
    private val silentListener = object : FrameAnalyzer.Listener {
        override fun onResult(result: FrameResult, outcome: PresenceFusionEngine.Outcome) = Unit
        override fun onError(message: String) = Unit
    }

    // ------------------------------------------------------------ 2 秒巡检

    /** 2 秒一次的全图巡检（含主摄广视野 + 副摄另一面） */
    private fun inspectionEnabled(): Boolean = settings.dualInspect

    /**
     * 前后双摄并发巡检。
     *
     * 省电模式下只留主摄巡检：多开一条相机输出流是这一个功能里最贵的一笔，
     * 而 2 秒一次的主摄广视野巡检已经能兜住「人走到侧面」的漏检。
     */
    private fun dualEyeEnabled(): Boolean = settings.dualInspect && !settings.powerSave

    /** 副摄在界面上叫什么。它永远对着当前镜头的反面 */
    private fun oppositeLensName(): String =
        if (lensFacing == CameraSelector.LENS_FACING_BACK) "前摄" else "后摄"

    fun isDualInspect(): Boolean = settings.dualInspect

    /** 前后双摄是否真的并发上了。false 表示机型不支持，已降级为单镜头巡检 */
    fun isConcurrentInspection(): Boolean = concurrentActive

    fun setDualInspect(enabled: Boolean) {
        if (settings.dualInspect == enabled) return
        settings.dualInspect = enabled
        analyzer?.inspectionIntervalMs = if (enabled) INSPECTION_INTERVAL_MS else 0L
        fusion.reset()
        publishStatus(status.copy())
        // 副摄用例要不要存在变了，得重绑一次相机
        mainHandler.post { bindCamera() }
    }

    /**
     * 巡检命中处理。
     *
     * 巡检结果是「主判定之外的旁路证据」：**绝不**写回 [status] / [fusion]，
     * 但命中就响 —— 只在主判定已经报警时才响的话，等于什么也没补上。
     */
    private fun onInspectionResult(present: Boolean, score: Float, source: String) {
        notifyListeners { it.onInspection(present, score, source) }
        if (!present || paused) return
        if (!settings.audioEnabled) return

        val now = SystemClock.elapsedRealtime()
        if (now - lastAlertAt < settings.alertIntervalMs) return
        lastAlertAt = now
        Log.i(TAG, "inspection hit [$source] score=$score")
        alertPlayer.play()
    }

    private inline fun notifyListeners(block: (Listener) -> Unit) {
        for (listener in listeners) {
            runCatching { block(listener) }
        }
    }

    private fun updateStatus(newStatus: Status) {
        status = newStatus
        notifyListeners { it.onStatus(newStatus) }
        refreshNotification()
    }

    private fun publishStatus(newStatus: Status) {
        status = newStatus
        notifyListeners { it.onStatus(newStatus) }
    }

    // ------------------------------------------------------------ 前台通知

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (manager.getNotificationChannel(CHANNEL_ID) != null) return

        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notify_channel_name),
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = getString(R.string.notify_channel_desc)
            setShowBadge(false)
            setSound(null, null)
            enableVibration(false)
        }
        manager.createNotificationChannel(channel)
    }

    private fun ensureForeground(current: Status) {
        val notification = buildNotification(current)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun refreshNotification() {
        runCatching {
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.notify(NOTIFICATION_ID, buildNotification(status))
        }
    }

    private fun buildNotification(current: Status): Notification {
        val title = when {
            current.phase == Phase.ERROR -> getString(R.string.status_error)
            current.phase == Phase.INITIALIZING -> getString(R.string.status_booting)
            current.phase == Phase.PAUSED -> getString(R.string.status_paused)
            current.present -> getString(R.string.notify_present, current.personCount)
            else -> getString(R.string.status_idle)
        }

        val text = when (current.phase) {
            Phase.ERROR -> current.message ?: getString(R.string.status_error)
            Phase.INITIALIZING -> getString(R.string.status_booting_detail)
            Phase.PAUSED -> getString(R.string.notify_paused_detail)
            else -> getString(R.string.notify_running_detail)
        }

        val openIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val stopIntent = PendingIntent.getService(
            this,
            1,
            Intent(this, DetectionService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text)
            .setContentIntent(openIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .addAction(0, getString(R.string.notify_action_stop), stopIntent)
            .build()
    }

    // ------------------------------------------------------------ 电源

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        val pm = getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "PresenceGuard::detect").apply {
            setReferenceCounted(false)
            acquire()
        }
    }

    private fun releaseWakeLock() {
        runCatching { if (wakeLock?.isHeld == true) wakeLock?.release() }
        wakeLock = null
    }

    companion object {
        private const val TAG = "DetectionService"
        const val CHANNEL_ID = "presence_monitor"
        const val NOTIFICATION_ID = 1001
        const val ACTION_STOP = "com.presenceguard.app.action.STOP"
        const val ACTION_TOGGLE_PAUSE = "com.presenceguard.app.action.TOGGLE_PAUSE"
        private const val NOTIFY_REFRESH_MS = 5000L
        /** 省电模式下的分析间隔：5fps 足够判断「有没有人」 */
        private const val POWER_SAVE_FRAME_INTERVAL = 200L

        /** 全图巡检周期。2 秒是「看得见人走过」和「别把 CPU 烧了」之间的折中 */
        private const val INSPECTION_INTERVAL_MS = 2000L

        /** 副摄巡检流的分辨率。压到 720p 既省转换开销，也提高并发绑定被 HAL 接受的概率 */
        private const val INSPECTION_WIDTH = 1280
        private const val INSPECTION_HEIGHT = 720

        /** 分析流分辨率。瓦片放大需要足够的原始像素，720p 是折中值 */
        private const val ANALYSIS_WIDTH = 1920
        private const val ANALYSIS_HEIGHT = 1080

        /** 目标检测引擎的粗筛置信度。压得比较低，交给融合层决定是否采信 */
        private const val DETECTOR_PRE_FILTER = 0.22f

        fun start(context: Context) {
            val intent = Intent(context, DetectionService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }
    }
}
