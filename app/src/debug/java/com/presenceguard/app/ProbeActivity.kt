package com.presenceguard.app

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import com.presenceguard.app.core.PersonCandidate
import com.presenceguard.app.detect.Accel
import com.presenceguard.app.detect.TflitePersonEngine
import com.presenceguard.app.detect.TileGrid
import java.util.Locale

/**
 * 离屏验收探针（仅 debug 构建）。
 *
 * 两件事：
 *  1. 实测各模型的单次推理耗时，选型不靠猜；
 *  2. 在两张「远处小目标」样例图上跑瓦片扫描，输出候选框，
 *     客观回答「远处的人到底能不能识别」。
 *
 * 注意：TFLite Task 的原生库是进程级共享的，**关闭最后一个 detector 会把库卸载**，
 * 之后再创建就会报 "Error getting native address of native library"。
 * 所以这里所有引擎先建好、最后统一关闭，中途不 close。
 *
 * 用法：
 *   adb shell am start -n com.presenceguard.app/.ProbeActivity
 *   adb logcat -s ProbeActivity TflitePersonEngine
 */
class ProbeActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread {
            runCatching { runProbe() }.onFailure { Log.e(TAG, "probe crashed", it) }
            Log.i(TAG, "===== probe finished =====")
        }.start()
        finish()
    }

    private fun runProbe() {
        // 一个进程只测一个模型：TFLite Task 的原生库在同一进程里二次创建 detector
        // 会报 "Error getting native address of native library"，所以按模型分进程跑。
        val model = intent?.getStringExtra(EXTRA_MODEL) ?: MODELS.first()
        val accel = runCatching {
            Accel.valueOf(intent?.getStringExtra(EXTRA_ACCEL) ?: Accel.GPU.name)
        }.getOrDefault(Accel.GPU)
        val engine = TflitePersonEngine(this, SCORE_THRESHOLD, accel, model)
        try {
            val sample = load(SAMPLES.first())
            val patch = sample?.let { Bitmap.createBitmap(it, 600, 400, 320, 320) }
            if (patch != null) {
                val ms = engine.benchmark(patch, BENCH_ITERATIONS)
                Log.i(TAG, String.format(Locale.US, "bench | %-32s %-4s -> %.1f ms",
                    model, engine.accelerator, ms))
            }
            accuracy(engine, model)
            if (patch != null && !patch.isRecycled) patch.recycle()
            if (sample != null && !sample.isRecycled) sample.recycle()
        } finally {
            engine.close()
        }
    }

    private fun accuracy(engine: TflitePersonEngine, model: String) {
        for (asset in SAMPLES) {
            val bitmap = load(asset) ?: continue
            for (grid in GRIDS) {
                val started = SystemClock.elapsedRealtime()
                val found = engine.detectSweep(bitmap, grid)
                val cost = SystemClock.elapsedRealtime() - started
                Log.i(TAG, String.format(Locale.US, "acc | %-32s %-18s %dx%d %5dms n=%d %s",
                    model, asset, grid.cols, grid.rows, cost, found.size, describe(found, bitmap)))
            }
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    private fun describe(found: List<PersonCandidate>, bitmap: Bitmap): String {
        if (found.isEmpty()) return ""
        return found.sortedByDescending { it.score }.take(6).joinToString(" ") { c ->
            val box: RectF = c.box
            String.format(
                Locale.US, "[%.2f %.0fx%.0f @%.0f,%.0f]",
                c.score, box.width(), box.height(), box.centerX(), box.centerY()
            )
        }
    }

    private fun load(asset: String): Bitmap? = runCatching {
        assets.open(asset).use { BitmapFactory.decodeStream(it) }
    }.getOrNull()

    private companion object {
        const val TAG = "ProbeActivity"
        const val EXTRA_MODEL = "model"
        const val EXTRA_ACCEL = "accel"
        const val SCORE_THRESHOLD = 0.18f
        const val BENCH_ITERATIONS = 8

        /** 目前只有 float32 版能被 Task 库加载；量化版实测两次都报 UnsatisfiedLinkError */
        val MODELS = listOf("efficientdet_lite0.tflite")
        val SAMPLES = listOf("probe_far_lake.jpg", "probe_far_plaza.jpg")
        val GRIDS = listOf(
            TileGrid(cols = 6, rows = 4),
            TileGrid(cols = 8, rows = 6)
        )
    }
}
