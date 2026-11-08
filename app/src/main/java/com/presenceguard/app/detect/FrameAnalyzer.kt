package com.presenceguard.app.detect

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.presenceguard.app.core.DetectionConfig
import com.presenceguard.app.core.FrameResult
import com.presenceguard.app.core.PersonCandidate
import com.presenceguard.app.core.Roi
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.min

/**
 * CameraX 分析器：双路合成检测。
 *
 * 一条路线不可能同时做到「便宜」和「看得清远处」，所以拆成两条：
 *
 *  - 快速路：每帧对整帧跑一次 320×320 推理，负责近处大目标，成本固定且很低；
 *  - 精细路：把画面切成 [TileGrid] 片，每帧只滚 [tilesPerFrame] 片，若干帧覆盖全图。
 *    瓦片把远处的小目标放大约 1.5 倍，是唯一能捞到几十米外行人的手段。
 *
 * 两路与姿态引擎并行跑在各自的线程上，用 CountDownLatch 汇合。
 * ImageProxy 必须等三路都结束才能 close，否则 ML Kit 内部的 Image 引用会失效。
 */
class FrameAnalyzer(
    private val poseEngine: PoseEngine,
    private val personEngine: TflitePersonEngine?,
    private val fusion: PresenceFusionEngine,
    private val listener: Listener
) : ImageAnalysis.Analyzer {

    interface Listener {
        fun onResult(result: FrameResult, outcome: PresenceFusionEngine.Outcome)
        fun onError(message: String)
    }

    @Volatile
    var config: DetectionConfig = DetectionConfig()

    /** 前置摄像头时叠加层需要镜像 */
    @Volatile
    var mirrored: Boolean = false

    /**
     * 省电模式下的最小帧间隔（毫秒）。0 表示不限。
     * 判断「画面里有没有人」不需要高帧率，5fps 已经足够，功耗却能降一大截。
     */
    @Volatile
    var minFrameIntervalMs: Long = 0

    /** 精细路的瓦片网格 */
    @Volatile
    var tileGrid: TileGrid = TileGrid.DEFAULT

    /** 精细路每帧滚几片。0 表示关闭精细路（省电模式）。 */
    @Volatile
    var tilesPerFrame: Int = DEFAULT_TILES_PER_FRAME

    /**
     * 用户框选的多边形选区。**非空即切换到「选区模式」**：
     * 判定只在选区内进行，区域外的画面完全不参与推理。
     */
    @Volatile
    var rois: List<Roi> = emptyList()

    /**
     * 选区模式下每帧扫几个选区。默认 1：一个选区的成本是「裁切 + 检测 + 姿态」两次推理，
     * 多了帧率会塌。多个选区靠游标轮转覆盖，跨帧轨迹会把证据攒起来。
     */
    @Volatile
    var roiPerFrame: Int = DEFAULT_ROI_PER_FRAME

    /** 选区轮转游标 */
    private var roiCursor = 0

    /** 最近一帧的分段耗时，供界面与日志核对 */
    @Volatile
    var lastConvertMs: Long = 0
        private set

    @Volatile
    var lastPoseMs: Long = 0
        private set

    @Volatile
    var lastDetectMs: Long = 0
        private set

    /**
     * 切换摄像头 / 改灵敏度时更新参数。
     *
     * 刻意不销毁重建分析器：这里的两个线程池是 ML Kit 回调的执行场所，
     * 一旦 shutdown，ML Kit 在内部异常分支里会对已经完成的 Task 再 setException，
     * 触发 DuplicateTaskCompletionException 直接闪退。
     */
    fun rebind(config: DetectionConfig, mirrored: Boolean) {
        this.config = config
        this.mirrored = mirrored
        lastFrameAt = 0L
        smoothedFps = 0f
    }

    private val poseExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "pose-engine").apply { priority = Thread.NORM_PRIORITY }
    }

    private val detectExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "detector-engine").apply { priority = Thread.NORM_PRIORITY }
    }

    private var lastFrameAt = 0L
    private var lastAcceptedAt = 0L
    private var smoothedFps = 0f
    private var diagTick = 0

    /** 上一帧的判定结果，用来决定这一帧要不要花算力做瓦片扫描 */
    @Volatile
    private var lastPresent = false

    // ML Kit 需要 android.media.Image，CameraX 目前把 getImage() 标为实验 API
    @OptIn(markerClass = [ExperimentalGetImage::class])
    override fun analyze(imageProxy: ImageProxy) {
        val begin = SystemClock.elapsedRealtime()

        // 省电模式：主动丢帧，避免每个相机帧都跑两次推理
        val interval = minFrameIntervalMs
        if (interval > 0 && begin - lastAcceptedAt < interval) {
            imageProxy.close()
            return
        }
        lastAcceptedAt = begin

        val rotation = imageProxy.imageInfo.rotationDegrees
        val mediaImage = imageProxy.image

        // 先转成已旋转的 Bitmap：两条检测路线都需要它
        val bitmap = try {
            val raw = imageProxy.toBitmap()
            rotate(raw, rotation)
        } catch (t: Throwable) {
            Log.w(TAG, "frame conversion failed", t)
            imageProxy.close()
            return
        }
        lastConvertMs = SystemClock.elapsedRealtime() - begin

        val cfg = config
        val activeRois = rois
        val roiMode = activeRois.isNotEmpty()

        // RGBA 输出格式下没有 android.media.Image，退回用位图构造（坐标同样是已旋转的）
        val inputImage = if (mediaImage != null) {
            InputImage.fromMediaImage(mediaImage, rotation)
        } else {
            InputImage.fromBitmap(bitmap, 0)
        }

        val frameWidth = if (rotation % 180 == 0) imageProxy.width else imageProxy.height
        val frameHeight = if (rotation % 180 == 0) imageProxy.height else imageProxy.width

        // 选区模式下 fusion 层要换一套门槛：用户已经用多边形声明了「这块区域值得看」，
        // 再拿「占整帧面积太小」去卡他，等于把选区这个先验扔掉了。
        val fusionCfg = if (roiMode) {
            cfg.copy(minAreaRatio = ROI_MIN_AREA_RATIO, smallTargetAreaRatio = 0f)
        } else {
            cfg
        }

        // 镜头被遮挡 / 贴着桌面时，画面几乎是纯色，模型会在这种无纹理图上疯狂误报。
        // 先用平均亮度把这类帧挡掉：既省掉几次推理，也避免「有人」卡死。
        val luma = averageLuma(bitmap)
        if (luma < MIN_VALID_LUMA || luma > MAX_VALID_LUMA) {
            bitmap.recycle()
            imageProxy.close()
            val now = SystemClock.elapsedRealtime()
            val outcome = fusion.update(emptyList(), emptyList(), frameWidth, frameHeight, now, fusionCfg)
            listener.onResult(
                FrameResult(
                    candidates = emptyList(),
                    sourceWidth = frameWidth,
                    sourceHeight = frameHeight,
                    mirrored = mirrored,
                    processingMs = SystemClock.elapsedRealtime() - begin,
                    fps = smoothedFps
                ),
                outcome
            )
            return
        }

        val latch: CountDownLatch
        var poseCandidates: List<PersonCandidate> = emptyList()
        var detectorCandidates: List<PersonCandidate> = emptyList()
        var poseFailed: Throwable? = null

        if (roiMode) {
            // ---- 选区模式
            // 只扫选区，而且两个引擎都在**放大后的裁切图**上跑：
            //  - 放大：选区外接矩形缩放到模型的 320×320 输入，等于给画面加了一台长焦；
            //  - 双引擎：放大之后置信度上来了，交叉印证才有意义，能压住「放大后噪点变人」。
            // 整帧检测在这条路径下完全跳过——用户已经指定了要盯的地方，别的地方不看。
            latch = CountDownLatch(1)
            val take = min(roiPerFrame.coerceAtLeast(1), activeRois.size)
            val start = roiCursor % activeRois.size
            val selected = (0 until take).map { activeRois[(start + it) % activeRois.size] }
            roiCursor = (start + take) % activeRois.size

            val detectBegin = SystemClock.elapsedRealtime()
            detectExecutor.execute {
                try {
                    val poseOut = ArrayList<PersonCandidate>(4)
                    val detOut = ArrayList<PersonCandidate>(4)
                    for (roi in selected) {
                        scanRegion(bitmap, roi, cfg, frameWidth, frameHeight, poseOut, detOut)
                    }
                    poseCandidates = poseOut
                    detectorCandidates = detOut
                } catch (t: Throwable) {
                    Log.w(TAG, "roi scan failed", t)
                } finally {
                    lastDetectMs = SystemClock.elapsedRealtime() - detectBegin
                    latch.countDown()
                }
            }
        } else {
            // ---- 常规模式：整帧双路合成
            latch = CountDownLatch(if (personEngine == null) 1 else 2)

            val poseStartedAt = SystemClock.elapsedRealtime()
            poseEngine.detect(
                image = inputImage,
                executor = poseExecutor,
                onResult = { poses ->
                    poseCandidates = PoseScorer.toCandidates(poses, cfg.keypointThreshold, frameHeight)
                },
                onFailure = { poseFailed = it },
                onComplete = {
                    lastPoseMs = SystemClock.elapsedRealtime() - poseStartedAt
                    latch.countDown()
                }
            )

            personEngine?.let { engine ->
                val grid = tileGrid
                // 画面里已经有人时不做瓦片扫描：报警已经触发，瓦片的意义是「找人」，
                // 稳态下省掉这几次推理，帧率能明显提升。空场景才是需要全力搜索的时候。
                val tiles = if (lastPresent) 0 else tilesPerFrame
                val detectBegin = SystemClock.elapsedRealtime()
                detectExecutor.execute {
                    try {
                        val collected = ArrayList<PersonCandidate>(tiles + 2)
                        // 瓦片先跑：整帧推理会让 TFLite 的 MlImage 在关闭时回收底层位图，
                        // 之后再裁瓦片就会拿到 recycled bitmap。顺序反过来才是安全的。
                        if (tiles > 0) {
                            collected += engine.detectTiles(bitmap, grid, tiles)
                        }
                        collected += engine.detectFull(bitmap)
                        detectorCandidates = collected
                    } catch (t: Throwable) {
                        Log.w(TAG, "detector failed", t)
                    } finally {
                        lastDetectMs = SystemClock.elapsedRealtime() - detectBegin
                        latch.countDown()
                    }
                }
            }
        }

        var finished = false
        try {
            finished = latch.await(FRAME_TIMEOUT_MS, TimeUnit.MILLISECONDS)
            if (!finished) Log.w(TAG, "frame timed out")
        } catch (interrupted: InterruptedException) {
            Thread.currentThread().interrupt()
        } finally {
            // 超时说明检测线程可能还在读这张 bitmap，此时回收会踩到野指针，
            // 宁可交给 GC 也别冒这个险
            if (finished && !bitmap.isRecycled) bitmap.recycle()
            imageProxy.close()
        }

        poseFailed?.let { listener.onError("姿态引擎异常：${it.javaClass.simpleName}") }

        val now = SystemClock.elapsedRealtime()

        val outcome = fusion.update(poseCandidates, detectorCandidates, frameWidth, frameHeight, now, fusionCfg)
        lastPresent = outcome.present


        // 诊断：分数、耗时、两路来源
        if (++diagTick % 10 == 0) {
            val area = (frameWidth * frameHeight).toFloat().coerceAtLeast(1f)
            val detInfo = detectorCandidates.joinToString(" | ") { c ->
                String.format(Locale.US, "%.2f[a%.4f]", c.score, c.area() / area)
            }
            Log.d(
                TAG,
                String.format(
                    Locale.US,
                    "cv=%dms pose=%dms det=%dms accel=%s src=%dx%d roi=%d/%d(z%d) " +
                        "pose=%d det=[%s] -> n=%d fs=%.2f ss=%.2f v=%d %s",
                    lastConvertMs, lastPoseMs, lastDetectMs,
                    personEngine?.accelerator ?: "-", frameWidth, frameHeight,
                    if (roiMode) activeRois.size else 0, activeRois.size, roiCursor,
                    poseCandidates.size, detInfo,
                    outcome.candidates.size, outcome.frameScore, outcome.smoothedScore,
                    outcome.votes, if (outcome.present) "PRESENT" else "none"
                )
            )
        }

        val delta = now - lastFrameAt
        lastFrameAt = now
        val instantFps = if (delta in 1..2000) 1000f / delta else 0f
        smoothedFps = if (smoothedFps <= 0f) instantFps else smoothedFps * 0.8f + instantFps * 0.2f

        listener.onResult(
            FrameResult(
                candidates = outcome.candidates,
                sourceWidth = frameWidth,
                sourceHeight = frameHeight,
                mirrored = mirrored,
                processingMs = now - begin,
                fps = smoothedFps
            ),
            outcome
        )
    }

    // ------------------------------------------------------------ 选区放大识别

    /**
     * 扫一个选区：裁出外接矩形 -> 姿态 -> 目标检测 -> 坐标平移回原图 -> 按多边形过滤。
     *
     * 顺序不能换：ML Kit 的 [InputImage.fromBitmap] 不持有位图所有权，跑完就放手；
     * 而 TFLite 的 MlImage 关闭时会**回收**传入的位图。所以姿态必须在检测之前。
     */
    private fun scanRegion(
        source: Bitmap,
        roi: Roi,
        cfg: DetectionConfig,
        sourceWidth: Int,
        sourceHeight: Int,
        poseOut: MutableList<PersonCandidate>,
        detOut: MutableList<PersonCandidate>
    ) {
        val region = regionOf(roi, source.width, source.height) ?: return
        val crop = runCatching {
            Bitmap.createBitmap(source, region.left, region.top, region.width(), region.height())
        }.onFailure { Log.w(TAG, "roi crop failed", it) }.getOrNull() ?: return

        try {
            val poseBegin = SystemClock.elapsedRealtime()
            for (candidate in poseOnCrop(crop, cfg)) {
                remap(candidate, region.left, region.top, roi, sourceWidth, sourceHeight)?.let { poseOut += it }
            }
            lastPoseMs = SystemClock.elapsedRealtime() - poseBegin

            val detected = personEngine?.detectCrop(crop, region.left, region.top).orEmpty()
            for (candidate in detected) {
                remap(candidate, region.left, region.top, roi, sourceWidth, sourceHeight)?.let { detOut += it }
            }
        } finally {
            if (!crop.isRecycled) crop.recycle()
        }
    }

    /**
     * 在裁切图上同步跑一次姿态。
     *
     * 分析线程和检测线程是分开的，这里从检测线程阻塞等待姿态结果，最多等
     * [POSE_TIMEOUT_MS]；超时只是少一份证据，不影响这一帧继续往下走。
     */
    private fun poseOnCrop(crop: Bitmap, cfg: DetectionConfig): List<PersonCandidate> {
        val done = CountDownLatch(1)
        var output: List<PersonCandidate> = emptyList()
        poseEngine.detect(
            image = InputImage.fromBitmap(crop, 0),
            executor = poseExecutor,
            onResult = { poses ->
                // 关键点门限按裁切图高度算：放大之后肩宽也随之放大，
                // 沿用整帧高度会把「够宽」的人重新判成过窄
                output = PoseScorer.toCandidates(poses, cfg.keypointThreshold, crop.height)
            },
            onFailure = { Log.w(TAG, "roi pose failed", it) },
            onComplete = { done.countDown() }
        )
        if (!done.await(POSE_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
            Log.w(TAG, "roi pose timed out")
        }
        return output
    }

    /** 选区的送检矩形（像素）。外扩一点，避免人站在选区边缘时框被裁掉半个身子 */
    private fun regionOf(roi: Roi, width: Int, height: Int): Rect? {
        val b = roi.bounds()
        val padX = (b.width() * REGION_PAD).coerceAtLeast(MIN_REGION_PAD)
        val padY = (b.height() * REGION_PAD).coerceAtLeast(MIN_REGION_PAD)

        val left = ((b.left - padX) * width).toInt().coerceIn(0, width - 1)
        val top = ((b.top - padY) * height).toInt().coerceIn(0, height - 1)
        val right = ((b.right + padX) * width).toInt().coerceIn(left + 1, width)
        val bottom = ((b.bottom + padY) * height).toInt().coerceIn(top + 1, height)

        if (right - left < MIN_REGION_PX || bottom - top < MIN_REGION_PX) return null
        return Rect(left, top, right, bottom)
    }

    /**
     * 把裁切图坐标系里的结果平移回整帧坐标系，并判断它是否真的落在选区内。
     *
     * 判定用「中心点是否在多边形内」，另外放一条重叠面积的退路：人站在选区边缘时，
     * 框中心很容易落到多边形外，但身体明明压在里面，这种情况不能丢。
     */
    private fun remap(
        candidate: PersonCandidate,
        offsetX: Int,
        offsetY: Int,
        roi: Roi,
        sourceWidth: Int,
        sourceHeight: Int
    ): PersonCandidate? {
        val box = RectF(
            candidate.box.left + offsetX,
            candidate.box.top + offsetY,
            candidate.box.right + offsetX,
            candidate.box.bottom + offsetY
        )
        val landmarks = if (candidate.landmarks.isEmpty()) {
            candidate.landmarks
        } else {
            candidate.landmarks.map { it.copy(x = it.x + offsetX, y = it.y + offsetY) }
        }
        val moved = candidate.copy(box = box, landmarks = landmarks)

        val cx = box.centerX() / sourceWidth
        val cy = box.centerY() / sourceHeight
        if (roi.contains(cx, cy)) return moved

        val bounds = roi.boundsPx(sourceWidth, sourceHeight)
        val left = max(box.left, bounds.left)
        val top = max(box.top, bounds.top)
        val right = min(box.right, bounds.right)
        val bottom = min(box.bottom, bounds.bottom)
        if (right <= left || bottom <= top) return null

        val inter = (right - left) * (bottom - top)
        val own = box.width() * box.height()
        return if (own > 0f && inter >= own * ROI_OVERLAP_ACCEPT) moved else null
    }

    /**
     * 采样估算平均亮度（0~255）。每 64 个像素取一个，耗时约 3~5ms，
     * 相对于两次模型推理可以忽略。
     */
    private fun averageLuma(bitmap: Bitmap): Int {
        val w = bitmap.width
        val h = bitmap.height
        if (w <= 0 || h <= 0) return 0
        val stepX = (w / SAMPLES_PER_SIDE).coerceAtLeast(1)
        val stepY = (h / SAMPLES_PER_SIDE).coerceAtLeast(1)

        var sum = 0L
        var count = 0
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val color = bitmap.getPixel(x, y)
                sum += (299 * Color.red(color) + 587 * Color.green(color) + 114 * Color.blue(color)) / 1000
                count++
                x += stepX
            }
            y += stepY
        }
        return if (count == 0) 0 else (sum / count).toInt()
    }

    fun shutdown() {
        poseExecutor.shutdownNow()
        detectExecutor.shutdownNow()
    }

    private fun rotate(source: Bitmap, degrees: Int): Bitmap {
        val normalized = ((degrees % 360) + 360) % 360
        if (normalized == 0) return source
        val matrix = Matrix().apply { postRotate(normalized.toFloat()) }
        val rotated = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
        if (rotated !== source && !source.isRecycled) source.recycle()
        return rotated
    }

    companion object {
        private const val TAG = "FrameAnalyzer"
        private const val FRAME_TIMEOUT_MS = 2000L

        /**
         * 精细路每帧滚动的瓦片数。24 片全图，每帧 4 片 ⇒ 6 帧覆盖一遍，
         * 按 10fps 算约 0.6 秒，既够快又不至于把整帧耗时推高。
         */
        const val DEFAULT_TILES_PER_FRAME = 4

        /** 选区模式下每帧扫几个选区。1 个 ≈ 两次推理，再多帧率就撑不住了 */
        const val DEFAULT_ROI_PER_FRAME = 1

        /** 选区送检时外接矩形外扩的比例，避免压线的人被裁掉半个身子 */
        private const val REGION_PAD = 0.06f

        /** 外扩的绝对下限（归一化），小选区上按比例算出来的外扩太少 */
        private const val MIN_REGION_PAD = 0.012f

        /**
         * 送检矩形的最小像素边长。低于这个值裁出来全是模糊块，放大到 320 只是把噪声放大，
         * 白白浪费一次推理。
         */
        private const val MIN_REGION_PX = 40

        /** 框与选区的重叠达到框自身面积的这个比例，就认为人确实在选区内 */
        private const val ROI_OVERLAP_ACCEPT = 0.45f

        /**
         * 选区模式下的面积下限。整帧模式下 0.003 是为了过滤远处噪点；
         * 选区是用户亲手划的，这个先验已经足够强，面积门限可以放到极低。
         */
        private const val ROI_MIN_AREA_RATIO = 0.0002f

        /** 选区模式下等待单次姿态推理的上限，超时就少一份证据继续走 */
        private const val POSE_TIMEOUT_MS = 900L

        /** 采样网格边长，64×64 ≈ 4096 个采样点，足够反映整体明暗 */
        private const val SAMPLES_PER_SIDE = 64

        /** 低于这个亮度（几乎纯黑）视为镜头被挡住，不做推理 */
        private const val MIN_VALID_LUMA = 10

        /** 高于这个亮度（几乎纯白，如镜头贴纸/强光过曝）同样跳过 */
        private const val MAX_VALID_LUMA = 248
    }
}
