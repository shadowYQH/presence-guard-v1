package com.presenceguard.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.mlkit.vision.pose.PoseLandmark
import com.presenceguard.app.core.DetectionSource
import com.presenceguard.app.core.FrameResult
import com.presenceguard.app.core.PersonCandidate
import kotlin.math.max
import kotlin.math.min

/**
 * 在预览画面上叠加检测结果。
 *
 * 坐标转换：检测在「旋转校正后的图像坐标」里做，预览用的是 FILL_CENTER（裁剪填满），
 * 所以这里用同一套等比缩放 + 居中偏移的映射，前置摄像头再做一次水平翻转，
 * 保证框和骨架严丝合缝地贴在画面上。
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private data class Snapshot(
        val candidates: List<PersonCandidate>,
        val srcW: Int,
        val srcH: Int,
        val mirrored: Boolean,
        val present: Boolean
    )

    private var snapshot = Snapshot(emptyList(), 0, 0, false, false)

    /** 关掉骨架只留框，画面更干净 */
    var showSkeleton: Boolean = true
        set(value) {
            field = value
            invalidate()
        }

    private val density = resources.displayMetrics.density

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
    }
    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2.5f)
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#E6FFFFFF")
    }
    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#F2FFFFFF")
    }
    private val labelBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val labelText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = sp(11f)
        isFakeBoldText = true
    }
    private val edgeGlow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(5f)
        color = Color.parseColor("#99FF3B30")
    }

    private val boxRect = RectF()
    private val path = Path()

    fun render(result: FrameResult, present: Boolean) {
        if (result.sourceWidth <= 0 || result.sourceHeight <= 0) return
        snapshot = Snapshot(result.candidates, result.sourceWidth, result.sourceHeight, result.mirrored, present)
        invalidate()
    }

    fun clear() {
        snapshot = Snapshot(emptyList(), 0, 0, false, false)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val snap = snapshot
        if (snap.srcW <= 0 || snap.srcH <= 0) return

        val scale = max(width / snap.srcW.toFloat(), height / snap.srcH.toFloat())
        val offsetX = (width - snap.srcW * scale) / 2f
        val offsetY = (height - snap.srcH * scale) / 2f

        if (snap.present) {
            // 状态确认时给画面加一圈红边，余光也能看到
            val inset = edgeGlow.strokeWidth / 2f
            canvas.drawRect(inset, inset, width - inset, height - inset, edgeGlow)
        }

        for (candidate in snap.candidates) {
            val left = mapX(candidate.box.left, scale, offsetX, snap.mirrored)
            val right = mapX(candidate.box.right, scale, offsetX, snap.mirrored)
            val top = mapY(candidate.box.top, scale, offsetY)
            val bottom = mapY(candidate.box.bottom, scale, offsetY)

            boxRect.set(min(left, right), top, max(left, right), bottom)
            // 边界外扩的框可能超出画面，裁一裁再画，避免圆角变形
            if (!boxRect.intersect(0f, 0f, width.toFloat(), height.toFloat())) continue

            drawCandidate(canvas, candidate, boxRect, scale, offsetX, offsetY, snap.mirrored)
        }
    }

    private fun drawCandidate(
        canvas: Canvas,
        candidate: PersonCandidate,
        rect: RectF,
        scale: Float,
        offsetX: Float,
        offsetY: Float,
        mirrored: Boolean
    ) {
        val accent = accentOf(candidate.source)
        val radius = dp(10f)

        fillPaint.color = withAlpha(accent, 34)
        canvas.drawRoundRect(rect, radius, radius, fillPaint)

        strokePaint.color = accent
        canvas.drawRoundRect(rect, radius, radius, strokePaint)

        if (showSkeleton && candidate.landmarks.isNotEmpty()) {
            drawSkeleton(canvas, candidate, scale, offsetX, offsetY, mirrored)
        }

        val text = buildLabel(candidate)
        val textWidth = labelText.measureText(text)
        val padH = dp(6f)
        val padV = dp(4f)
        val labelHeight = labelText.textSize + padV * 2
        val labelTop = (rect.top - labelHeight - dp(2f)).coerceAtLeast(0f)
        val labelLeft = rect.left.coerceAtMost(width - textWidth - padH * 2)

        labelBg.color = accent
        canvas.drawRoundRect(
            labelLeft, labelTop, labelLeft + textWidth + padH * 2, labelTop + labelHeight,
            dp(4f), dp(4f), labelBg
        )
        canvas.drawText(text, labelLeft + padH, labelTop + labelHeight - padV - dp(1f), labelText)
    }

    private fun drawSkeleton(
        canvas: Canvas,
        candidate: PersonCandidate,
        scale: Float,
        offsetX: Float,
        offsetY: Float,
        mirrored: Boolean
    ) {
        val points = HashMap<Int, Pair<Float, Float>>(candidate.landmarks.size)
        for (landmark in candidate.landmarks) {
            points[landmark.type] = Pair(
                mapX(landmark.x, scale, offsetX, mirrored),
                mapY(landmark.y, scale, offsetY)
            )
        }

        for ((a, b) in BONES) {
            val pa = points[a] ?: continue
            val pb = points[b] ?: continue
            canvas.drawLine(pa.first, pa.second, pb.first, pb.second, bonePaint)
        }

        for ((x, y) in points.values) {
            canvas.drawCircle(x, y, dp(2.6f), jointPaint)
        }
    }

    private fun buildLabel(candidate: PersonCandidate): String {
        val tag = when (candidate.source) {
            DetectionSource.FUSED -> "确认"
            DetectionSource.POSE -> "骨架"
            DetectionSource.DETECTOR -> "轮廓"
        }
        return "$tag ${(candidate.score * 100).toInt()}%"
    }

    private fun accentOf(source: DetectionSource): Int = when (source) {
        DetectionSource.FUSED -> Color.parseColor("#FF34C759")
        DetectionSource.POSE -> Color.parseColor("#FF0A84FF")
        DetectionSource.DETECTOR -> Color.parseColor("#FFFF9F0A")
    }

    private fun mapX(x: Float, scale: Float, offsetX: Float, mirrored: Boolean): Float {
        val mapped = x * scale + offsetX
        return if (mirrored) width - mapped else mapped
    }

    private fun mapY(y: Float, scale: Float, offsetY: Float): Float = y * scale + offsetY

    private fun withAlpha(color: Int, alpha: Int): Int =
        Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color))

    private fun dp(value: Float): Float = value * density

    private fun sp(value: Float): Float = value * resources.displayMetrics.scaledDensity

    companion object {
        /** BlazePose 的关键点连线（只取上半身与四肢主干，够表达姿态又不显杂乱） */
        private val BONES: List<Pair<Int, Int>> = listOf(
            PoseLandmark.LEFT_SHOULDER to PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_SHOULDER to PoseLandmark.LEFT_ELBOW,
            PoseLandmark.LEFT_ELBOW to PoseLandmark.LEFT_WRIST,
            PoseLandmark.RIGHT_SHOULDER to PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.RIGHT_ELBOW to PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_SHOULDER to PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_SHOULDER to PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_HIP to PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_HIP to PoseLandmark.LEFT_KNEE,
            PoseLandmark.LEFT_KNEE to PoseLandmark.LEFT_ANKLE,
            PoseLandmark.RIGHT_HIP to PoseLandmark.RIGHT_KNEE,
            PoseLandmark.RIGHT_KNEE to PoseLandmark.RIGHT_ANKLE,
            PoseLandmark.NOSE to PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.NOSE to PoseLandmark.RIGHT_SHOULDER
        )
    }
}
