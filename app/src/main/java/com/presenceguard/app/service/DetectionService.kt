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
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
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
        runCatching { cameraProvider?.unbindAll() }

        val engine = poseEngine
        val detector = personEngine
        val analyzerRef = analyzer
        // 在途帧可能还卡在 ML Kit 回调里，立刻 close 会让它重复完成 Task 而崩溃
        mainHandler.postDelayed({
            runCatching { engine?.close() }
            runCatching { detector?.close() }
            runCatching { analyzerRef?.shutdown() }
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

    fun setPowerSave(enabled: Boolean) {
        settings.powerSave = enabled
        analyzer?.minFrameIntervalMs = if (enabled) POWER_SAVE_FRAME_INTERVAL else 0L
        // 省电模式下关掉瓦片精细路：远处的小目标会漏，但功耗能降一大截
        analyzer?.tilesPerFrame = if (enabled) 0 else tilesForSensitivity(sensitivity)
    }

    fun isPowerSave(): Boolean = settings.powerSave

    fun togglePause(): Boolean {
        paused = !paused
        if (paused) {
            analysisUseCase?.clearAnalyzer()
        } else {
            analyzer?.let { analysisUseCase?.setAnalyzer(cameraExecutor, it) }
        }
        fusion.reset()
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
                }
                bindCamera()
                updateStatus(Status(Phase.RUNNING))
            }
        }
    }

    private fun bindCamera() {
        val analyzerRef = analyzer ?: return

        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = runCatching { future.get() }.getOrNull()
            if (provider == null) {
                updateStatus(Status(Phase.ERROR, message = "相机初始化失败"))
                return@addListener
            }
            cameraProvider = provider
            provider.unbindAll()

            val preview = Preview.Builder().build()
            previewUseCase = preview
            // 关键：不接 SurfaceProvider 的话，相机照样开、检测照样跑，但屏幕上永远没画面
            pendingSurfaceProvider?.let { preview.setSurfaceProvider(it) }
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        // 精细路靠瓦片放大视野，但采集本身分辨率太低时，放大的只是模糊块。
                        // 720p 是 865 上性价比最高的折中。
                        Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                    )
                )
                .build()

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analyzerRef.rebind(
                config = sensitivity.toConfig(),
                mirrored = lensFacing == CameraSelector.LENS_FACING_FRONT
            )
            analyzerRef.minFrameIntervalMs = if (settings.powerSave) POWER_SAVE_FRAME_INTERVAL else 0L
            analyzerRef.tilesPerFrame = tilesForSensitivity(sensitivity)

            analysisUseCase?.clearAnalyzer()
            analysisUseCase = analysis

            if (!paused) analysis.setAnalyzer(cameraExecutor, analyzerRef)

            val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
            try {
                provider.bindToLifecycle(this, selector, preview, analysis)
            } catch (t: Throwable) {
                Log.w(TAG, "bind failed, retry with other lens", t)
                lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                    CameraSelector.LENS_FACING_FRONT
                } else {
                    CameraSelector.LENS_FACING_BACK
                }
                runCatching {
                    val fallback = CameraSelector.Builder().requireLensFacing(lensFacing).build()
                    provider.bindToLifecycle(this, fallback, preview, analysis)
                    analyzerRef.mirrored = lensFacing == CameraSelector.LENS_FACING_FRONT
                }.onFailure {
                    updateStatus(Status(Phase.ERROR, message = "无法打开摄像头"))
                }
            }
        }, ContextCompat.getMainExecutor(this))
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
