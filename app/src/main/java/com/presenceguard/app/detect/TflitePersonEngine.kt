package com.presenceguard.app.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import com.google.android.odml.image.BitmapMlImageBuilder
import com.presenceguard.app.core.DetectionSource
import com.presenceguard.app.core.Evidence
import com.presenceguard.app.core.PersonCandidate
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 瓦片网格。
 *
 * 为什么需要它：EfficientDet-Lite0 的输入固定 320×320，整帧缩放后远处的人只剩
 * 十几像素，特征彻底消失；**提高采集分辨率没有用**，因为缩放会把增益抵消掉。
 * 唯一有效的手段是缩小送进模型的视野——把画面切成瓦片，人在这片里占的比例被放大
 * 约 [放大倍数] 倍。放大倍数只取决于瓦片数量，与采集分辨率无关（分辨率只决定
 * 放大后是否清晰）。
 *
 * [overlap] 是相邻瓦片的重叠比例，避免人正好跨在接缝上被切成两半。
 */
data class TileGrid(
    val cols: Int,
    val rows: Int,
    val overlap: Float = 0.25f
) {
    fun rects(width: Int, height: Int): List<Rect> {
        if (width <= 0 || height <= 0 || cols <= 0 || rows <= 0) return emptyList()

        val tileW = (width.toFloat() / (cols - (cols - 1) * overlap))
            .roundToInt().coerceIn(1, width)
        val tileH = (height.toFloat() / (rows - (rows - 1) * overlap))
            .roundToInt().coerceIn(1, height)

        val stepX = if (cols > 1) (width - tileW).toFloat() / (cols - 1) else 0f
        val stepY = if (rows > 1) (height - tileH).toFloat() / (rows - 1) else 0f

        val out = ArrayList<Rect>(cols * rows)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val left = (c * stepX).roundToInt().coerceIn(0, width - tileW)
                val top = (r * stepY).roundToInt().coerceIn(0, height - tileH)
                out += Rect(left, top, left + tileW, top + tileH)
            }
        }
        return out
    }

    val tileCount: Int get() = cols * rows

    companion object {
        /** 默认：6 列 × 4 行，相邻重叠 25%，共 24 片 */
        val DEFAULT = TileGrid(cols = 6, rows = 4)
    }
}

/**
 * 推理加速器。实测决定用哪个，不靠猜。
 */
enum class Accel { CPU, GPU, NNAPI }

/**
 * 副引擎：TFLite EfficientDet-Lite0（COCO 数据集，含 person 类）。
 *
 * 提供三种用法，对应「双路合成」的两条路线：
 *  - [detectFull]  整帧一次推理。便宜，负责近处的大目标，每帧都跑。
 *  - [detectTiles] 瓦片滚动扫描，每帧只跑指定的几片。负责远处的小目标，
 *                  全图覆盖一遍需要若干帧，靠跨帧轨迹把证据攒起来。
 *  - [detectSweep] 一次扫完全部瓦片。离屏验收（探针）用，实时不走这条。
 *
 * 加速器按 [preferred] 依次尝试，失败自动退到 CPU。
 */
class TflitePersonEngine(
    context: Context,
    private val scoreThreshold: Float,
    private val preferred: Accel = Accel.GPU,
    private val modelAsset: String = MODEL_ASSET
) {

    private val appContext = context.applicationContext

    private var detector: ObjectDetector? = null

    /** 实际生效的加速器，供日志与界面显示 */
    @Volatile
    var accelerator: String = "CPU"
        private set

    /** 最近一次单张推理耗时，用于性能核对 */
    @Volatile
    var lastInferenceMs: Long = 0
        private set

    private var tileCursor = 0
    private var diagTick = 0

    init {
        val order = when (preferred) {
            Accel.GPU -> listOf(Accel.GPU, Accel.CPU)
            Accel.NNAPI -> listOf(Accel.NNAPI, Accel.CPU)
            Accel.CPU -> listOf(Accel.CPU)
        }
        for (mode in order) {
            val created = createDetector(mode)
            if (created != null) {
                detector = created
                accelerator = mode.name
                break
            }
        }
        Log.i(TAG, "detector ready, accelerator=$accelerator model=$modelAsset (preferred=$preferred)")
    }

    private fun createDetector(mode: Accel): ObjectDetector? = runCatching {
        val base = BaseOptions.builder().setNumThreads(NUM_THREADS)
        when (mode) {
            Accel.GPU -> base.useGpu()
            Accel.NNAPI -> base.useNnapi()
            Accel.CPU -> Unit
        }
        ObjectDetector.createFromFileAndOptions(
            appContext,
            modelAsset,
            ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(base.build())
                .setScoreThreshold(scoreThreshold)
                .setMaxResults(MAX_RESULTS)
                .build()
        )
    }.onFailure {
        Log.w(TAG, "create detector failed (mode=$mode, model=$modelAsset)", it)
    }.getOrNull()

    // ---------------------------------------------------------------- 对外

    /** 整帧一次推理，负责近处大目标 */
    fun detectFull(bitmap: Bitmap): List<PersonCandidate> {
        val engine = detector ?: return emptyList()
        if (bitmap.isRecycled) return emptyList()
        return infer(engine, bitmap, 0, 0, 1f)
    }

    /**
     * 在**已经裁好的**位图上做一次推理，结果坐标按 (offsetX, offsetY) 平移回原图。
     *
     * 这是「多边形内放大识别」的入口：调用方把选区的外接矩形裁出来（可选外扩），
     * 送进固定 320×320 的模型后，选区里的人在模型视野里被放大了 1/选区占比 倍，
     * 相当于凭空多出一台长焦镜头。
     *
     * 注意：TFLite 的 MlImage 在关闭时会**回收传入的位图**，所以本方法返回后
     * [crop] 就不能再用了。调用方如果需要同一块裁切图做别的事（比如再跑一次姿态），
     * 必须在调用本方法**之前**做完。
     */
    fun detectCrop(
        crop: Bitmap,
        offsetX: Int,
        offsetY: Int,
        scoreDiscount: Float = 1f
    ): List<PersonCandidate> {
        val engine = detector ?: return emptyList()
        if (crop.isRecycled) return emptyList()
        return infer(engine, crop, offsetX, offsetY, scoreDiscount)
    }

    /**
     * 瓦片滚动扫描：每帧只跑 [limit] 片，游标轮转，若干帧覆盖全图。
     * 只回传「小目标」——瓦片本来就是为了捞远处的小东西，大目标交给整帧那一路，
     * 否则同一个人会被多个瓦片各框一次，徒增重复。
     */
    fun detectTiles(
        bitmap: Bitmap,
        grid: TileGrid,
        limit: Int,
        maxAreaRatio: Float = DEFAULT_TILE_TARGET_MAX_AREA
    ): List<PersonCandidate> {
        val engine = detector ?: return emptyList()
        if (bitmap.isRecycled) return emptyList()

        val tiles = grid.rects(bitmap.width, bitmap.height)
        if (tiles.isEmpty()) return emptyList()

        val take = min(limit, tiles.size)
        val collected = ArrayList<PersonCandidate>(take * 2)
        for (i in 0 until take) {
            val tile = tiles[(tileCursor + i) % tiles.size]
            collected += inferTile(engine, bitmap, tile, maxAreaRatio)
        }
        tileCursor = (tileCursor + take) % tiles.size
        return collected
    }

    /** 一次扫完全部瓦片，离屏验收用 */
    fun detectSweep(
        bitmap: Bitmap,
        grid: TileGrid,
        maxAreaRatio: Float = 1f
    ): List<PersonCandidate> {
        val engine = detector ?: return emptyList()
        if (bitmap.isRecycled) return emptyList()

        val frameArea = (bitmap.width * bitmap.height).toFloat().coerceAtLeast(1f)
        val collected = ArrayList<PersonCandidate>(grid.tileCount * 2)
        for (tile in grid.rects(bitmap.width, bitmap.height)) {
            collected += inferTile(engine, bitmap, tile, maxAreaRatio)
        }
        if (++diagTick % 10 == 0) {
            Log.d(
                TAG,
                String.format(
                    Locale.US, "sweep tiles=%d raw=%d accel=%s",
                    grid.tileCount, collected.size, accelerator
                )
            )
        }
        return nonMaximumSuppression(collected, frameArea)
    }

    /**
     * 基准测试：在给定图上跑 [iterations] 次整帧推理，返回中位耗时（毫秒）。
     *
     * 每次都用副本，因为 TFLite 的 MlImage 关闭时会回收传入的位图。
     * 用中位数而不是平均值，避开首帧的委托预热。
     */
    fun benchmark(source: Bitmap, iterations: Int): Double {
        if (source.isRecycled) return -1.0
        val times = ArrayList<Long>(iterations)
        repeat(iterations) {
            val copy = source.copy(Bitmap.Config.ARGB_8888, false) ?: return@repeat
            val started = android.os.SystemClock.elapsedRealtime()
            detectFull(copy)
            times += android.os.SystemClock.elapsedRealtime() - started
        }
        if (times.isEmpty()) return -1.0
        times.sort()
        return times[times.size / 2].toDouble()
    }

    // ---------------------------------------------------------------- 内部

    private fun inferTile(
        engine: ObjectDetector,
        source: Bitmap,
        tile: Rect,
        maxAreaRatio: Float
    ): List<PersonCandidate> {
        val crop = runCatching {
            Bitmap.createBitmap(source, tile.left, tile.top, tile.width(), tile.height())
        }.getOrNull() ?: return emptyList()

        return try {
            val frameArea = (source.width * source.height).toFloat().coerceAtLeast(1f)
            infer(engine, crop, tile.left, tile.top, TILE_SCORE_DISCOUNT)
                // 只保留小目标：瓦片路径存在的意义就是找远处的人
                .filter { it.area() / frameArea <= maxAreaRatio }
        } finally {
            if (!crop.isRecycled) crop.recycle()
        }
    }

    private fun infer(
        engine: ObjectDetector,
        bitmap: Bitmap,
        offsetX: Int,
        offsetY: Int,
        scoreDiscount: Float
    ): List<PersonCandidate> {
        val started = android.os.SystemClock.elapsedRealtime()
        val image = BitmapMlImageBuilder(bitmap).build()
        return try {
            engine.detect(image).mapNotNull { detection ->
                // 只关心人；EfficientDet 的 COCO 标签里 person 是第 0 类。
                val category = detection.categories.firstOrNull() ?: return@mapNotNull null
                val label = category.label?.trim().orEmpty()
                val isPerson = if (label.isEmpty()) {
                    category.index == 0
                } else {
                    label.equals(PERSON_LABEL, ignoreCase = true)
                }
                if (!isPerson) return@mapNotNull null

                val src = detection.boundingBox
                if (src.width() <= 0 || src.height() <= 0) return@mapNotNull null

                val score = (category.score * scoreDiscount).coerceIn(0f, 1f)
                PersonCandidate(
                    box = RectF(
                        (src.left + offsetX).toFloat(),
                        (src.top + offsetY).toFloat(),
                        (src.right + offsetX).toFloat(),
                        (src.bottom + offsetY).toFloat()
                    ),
                    score = score,
                    source = DetectionSource.DETECTOR,
                    evidence = Evidence(detectorScore = score)
                )
            }
        } finally {
            image.close()
            lastInferenceMs = android.os.SystemClock.elapsedRealtime() - started
        }
    }

    private fun nonMaximumSuppression(
        input: List<PersonCandidate>,
        frameArea: Float
    ): List<PersonCandidate> {
        if (input.size <= 1) return input
        val sorted = input.sortedByDescending { it.score }
        val kept = ArrayList<PersonCandidate>(sorted.size)
        for (candidate in sorted) {
            val duplicate = kept.any { iou(it.box, candidate.box) > NMS_IOU }
            if (!duplicate) kept += candidate
            if (kept.size >= MAX_CANDIDATES) break
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        if (right <= left || bottom <= top) return 0f
        val inter = (right - left) * (bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() {
        runCatching { detector?.close() }
        detector = null
    }

    companion object {
        private const val TAG = "TflitePersonEngine"
        const val MODEL_ASSET = "efficientdet_lite0.tflite"
        private const val PERSON_LABEL = "person"
        private const val MAX_RESULTS = 10
        private const val NUM_THREADS = 4
        private const val MAX_CANDIDATES = 16

        private const val TILE_SCORE_DISCOUNT = 0.95f
        private const val NMS_IOU = 0.55f

        /** 瓦片路径只收面积占整帧 12% 以下的目标 */
        const val DEFAULT_TILE_TARGET_MAX_AREA = 0.12f
    }
}
