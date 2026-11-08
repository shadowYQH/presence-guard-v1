package com.presenceguard.app.ui

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import com.google.mlkit.vision.pose.PoseLandmark
import com.presenceguard.app.core.DetectionSource
import com.presenceguard.app.core.FrameResult
import com.presenceguard.app.core.PersonCandidate
import com.presenceguard.app.core.Roi
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * 在预览画面上叠加检测结果，并承担多边形选区的绘制与取点。
 *
 * 坐标转换：检测在「旋转校正后的图像坐标」里做，预览用的是 FILL_CENTER（裁剪填满），
 * 所以这里用同一套等比缩放 + 居中偏移的映射，前置摄像头再做一次水平翻转，
 * 保证框和骨架严丝合缝地贴在画面上。
 *
 * 选区取点是这套映射的**逆变换**：屏幕点 -> 归一化坐标（0~1）。存归一化值而不是像素，
 * 是为了让选区在换分辨率、换镜头、切前后摄之后依然落在同一个物理位置上。
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    /** 画面的几何信息。与检测结果分开保存：点「清空」时不该把坐标系一起丢掉 */
    private data class Geometry(val srcW: Int, val srcH: Int, val mirrored: Boolean)

    private data class Snapshot(
        val candidates: List<PersonCandidate>,
        val present: Boolean
    )

    private var geometry: Geometry? = null
    private var snapshot = Snapshot(emptyList(), false)

    /** 关掉骨架只留框，画面更干净 */
    var showSkeleton: Boolean = true
        set(value) {
            field = value
            invalidate()
        }

    /** 已确认的多边形选区。由外部（Activity）负责持久化，本视图只负责画和改。 */
    var rois: List<Roi> = emptyList()
        private set

    /** 选区变化回调，参数是变化后的完整列表 */
    var onRoisChange: ((List<Roi>) -> Unit)? = null

    /** 正在绘制中的多边形顶点（归一化坐标） */
    private val draft = ArrayList<PointF>()

    /**
     * 取点模式。开启后吃掉全部触摸事件，改由自己解释为「加一个顶点」，
     * 关闭时一律返回 false，把事件让给下面层的控件。
     */
    var editMode: Boolean = false
        set(value) {
            if (field == value) return
            field = value
            if (!value && draft.size < Roi.MIN_POINTS) draft.clear()
            if (value) closeHintReady = false
            invalidate()
        }

    private var closeHintReady = false

    /**
     * 数字裁切倍数（双指缩放）。1 表示不裁。
     *
     * 这**只影响送进模型的那块像素**，不动任何相机参数 —— 所以它和「焦段」是两回事：
     * 焦段是换镜头（或传感器级裁切），这里的裁切是纯软件放大，预览画面不会变。
     * 为了让人不至于迷糊，画面上会画出裁切框并标注「框外不识别」。
     */
    var cropScale: Float = 1f
        private set

    /** 裁切倍数变化回调 */
    var onCropScaleChange: ((Float) -> Unit)? = null

    private val scaleDetector = ScaleGestureDetector(
        context,
        object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            private var base = 1f

            override fun onScaleBegin(detector: ScaleGestureDetector): Boolean {
                base = cropScale
                return true
            }

            override fun onScale(detector: ScaleGestureDetector): Boolean {
                // 用倍乘而不是累加：连续捏合时累加会越捏越迟钝
                val next = (base * detector.scaleFactor).coerceIn(1f, MAX_CROP_SCALE)
                if (abs(next - cropScale) < 0.005f) return true
                cropScale = next
                onCropScaleChange?.invoke(cropScale)
                invalidate()
                return true
            }

            /**
             * 收到接近 1 的地方就吸附到 1。
             *
             * 捏合手势永远不可能精确回到 1.00，差 0.05 就意味着「还在裁切」，
             * 用户会以为自己已经复位了却一直少看一圈画面。
             */
            override fun onScaleEnd(detector: ScaleGestureDetector) {
                if (cropScale < SNAP_TO_ONE_BELOW) resetCrop()
            }
        }
    ).apply {
        // 关掉「双击快速缩放」：双击在这个界面里太容易误触
        isQuickScaleEnabled = false
    }

    /** 把裁切倍数恢复到 1（不裁） */
    fun resetCrop() {
        if (cropScale == 1f) return
        cropScale = 1f
        onCropScaleChange?.invoke(1f)
        invalidate()
    }

    fun setCropScale(scale: Float) {
        cropScale = scale.coerceIn(1f, MAX_CROP_SCALE)
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

    // ---- 选区专用画笔
    private val roiFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#3322D3EE")
    }
    private val roiStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
        color = Color.parseColor("#FF22D3EE")
    }
    private val draftStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
        color = Color.parseColor("#FFFFD60A")
        pathEffect = DashPathEffect(floatArrayOf(dp(7f), dp(5f)), 0f)
    }
    private val vertexPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#FFFFD60A")
    }
    private val vertexFirstPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2.5f)
        color = Color.parseColor("#FFFF9F0A")
    }
    private val hintBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#CC000000")
    }
    private val hintText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFF2F2F7")
        textSize = sp(12f)
    }

    private val boxRect = RectF()
    private val path = Path()
    private val roiPath = Path()
    private val pointScratch = PointF()

    /** 数字裁切框外的遮罩：告诉用户这块不参与识别 */
    private val cropMaskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#99000000")
    }
    private val cropBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
        color = Color.parseColor("#B3FFFFFF")
        pathEffect = DashPathEffect(floatArrayOf(dp(9f), dp(7f)), 0f)
    }

    // ------------------------------------------------------------ 对外接口

    fun render(result: FrameResult, present: Boolean) {
        if (result.sourceWidth <= 0 || result.sourceHeight <= 0) return
        geometry = Geometry(result.sourceWidth, result.sourceHeight, result.mirrored)
        snapshot = Snapshot(result.candidates, present)
        invalidate()
    }

    /** 清空检测结果（不动坐标系，也不动选区） */
    fun clear() {
        snapshot = Snapshot(emptyList(), false)
        invalidate()
    }

    fun setRois(list: List<Roi>) {
        rois = list.filter { it.isValid }
        draft.clear()
        invalidate()
    }

    /**
     * 撤销一步：优先删掉正在画的最后一个顶点；没有草稿就删掉最后一个已确认的选区。
     */
    fun undo() {
        when {
            draft.isNotEmpty() -> draft.removeAt(draft.size - 1)
            rois.isNotEmpty() -> {
                rois = rois.dropLast(1)
                onRoisChange?.invoke(rois)
            }
            else -> return
        }
        closeHintReady = false
        invalidate()
    }

    fun clearRois() {
        if (rois.isEmpty() && draft.isEmpty()) return
        rois = emptyList()
        draft.clear()
        closeHintReady = false
        onRoisChange?.invoke(rois)
        invalidate()
    }

    /**
     * 结束当前多边形：顶点数够就闭合收下，不够就丢弃草稿。
     * @return 是否成功收下一个选区
     */
    fun finishDraft(): Boolean {
        if (draft.size < Roi.MIN_POINTS) {
            draft.clear()
            closeHintReady = false
            invalidate()
            return false
        }
        val candidate = Roi(draft.toList())
        draft.clear()
        closeHintReady = false
        if (candidate.areaRatio() < Roi.MIN_AREA_RATIO) {
            invalidate()
            return false
        }
        rois = rois + candidate
        onRoisChange?.invoke(rois)
        invalidate()
        return true
    }

    fun draftPointCount(): Int = draft.size

    // ------------------------------------------------------------ 触摸取点

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (editMode) {
            // 画选区时只吃单指取点，不响应缩放：双指落点很容易被当成两个顶点，
            // 用户会得到一个莫名其妙的三角形。要缩放先退出选区模式。
            if (event.actionMasked != MotionEvent.ACTION_DOWN) return true
            handleTap(event.x, event.y)
            return true
        }

        scaleDetector.onTouchEvent(event)
        if (event.actionMasked == MotionEvent.ACTION_UP) performClick()
        // 一直返回 true：叠加层下面是 PreviewView，不需要触摸事件。
        // 上面的控制栏 z 序更高，按钮照常可点。
        //
        // 注意：这会让根布局上的 OnTouchListener 收不到事件 —— 所以「暗屏时点一下
        // 恢复亮度」必须放在 Activity.dispatchTouchEvent 里，不能挂在根布局上。
        return true
    }

    private fun handleTap(viewX: Float, viewY: Float) {
        val point = toNormalized(viewX, viewY) ?: return

        // 离上一个顶点太近 = 手指抖动，忽略，否则多边形会退化成一堆重合点
        if (draft.isNotEmpty()) {
            val last = draft.last()
            if (hypot(point.x - last.x, point.y - last.y) < Roi.MIN_VERTEX_GAP) return
        }

        // 点回起点即闭合
        if (draft.size >= Roi.MIN_POINTS) {
            val first = draft.first()
            if (hypot(point.x - first.x, point.y - first.y) < CLOSE_GAP) {
                finishDraft()
                return
            }
        }

        draft += point
        closeHintReady = draft.size >= Roi.MIN_POINTS
        invalidate()
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    /** 屏幕坐标 -> 归一化图像坐标（FILL_CENTER 映射的逆变换） */
    private fun toNormalized(viewX: Float, viewY: Float): PointF? {
        val geo = geometry ?: return null
        if (geo.srcW <= 0 || geo.srcH <= 0 || width <= 0 || height <= 0) return null

        val scale = max(width / geo.srcW.toFloat(), height / geo.srcH.toFloat())
        val offsetX = (width - geo.srcW * scale) / 2f
        val offsetY = (height - geo.srcH * scale) / 2f

        val px = if (geo.mirrored) width - viewX else viewX
        val nx = ((px - offsetX) / scale) / geo.srcW
        val ny = ((viewY - offsetY) / scale) / geo.srcH
        return PointF(nx.coerceIn(0f, 1f), ny.coerceIn(0f, 1f))
    }

    // ------------------------------------------------------------ 绘制

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val geo = geometry
        val snap = snapshot

        if (geo != null && geo.srcW > 0 && geo.srcH > 0) {
            val scale = max(width / geo.srcW.toFloat(), height / geo.srcH.toFloat())
            val offsetX = (width - geo.srcW * scale) / 2f
            val offsetY = (height - geo.srcH * scale) / 2f

            if (snap.present) {
                // 状态确认时给画面加一圈红边，余光也能看到
                val inset = edgeGlow.strokeWidth / 2f
                canvas.drawRect(inset, inset, width - inset, height - inset, edgeGlow)
            }

            for (candidate in snap.candidates) {
                val left = mapX(candidate.box.left, scale, offsetX, geo.mirrored)
                val right = mapX(candidate.box.right, scale, offsetX, geo.mirrored)
                val top = mapY(candidate.box.top, scale, offsetY)
                val bottom = mapY(candidate.box.bottom, scale, offsetY)

                boxRect.set(min(left, right), top, max(left, right), bottom)
                // 边界外扩的框可能超出画面，裁一裁再画，避免圆角变形
                if (!boxRect.intersect(0f, 0f, width.toFloat(), height.toFloat())) continue

                drawCandidate(canvas, candidate, boxRect, scale, offsetX, offsetY, geo.mirrored)
            }

            drawRois(canvas, geo, scale, offsetX, offsetY)
            drawDraft(canvas, geo, scale, offsetX, offsetY)
        }

        if (editMode) drawEditHint(canvas)
        // 裁切遮罩画在最后：它压住所有检测框，才能传达「框外的东西不算数」
        drawCropMask(canvas)
    }

    /**
     * 数字裁切的范围提示。
     *
     * 裁切只发生在模型输入上，预览画面是不动的 —— 如果不标出来，用户会看到
     * 画面里明明有个人却一直不报警（因为他在裁切框外）。所以框外压一层遮罩，
     * 框上写清楚倍数和「框外不识别」。
     */
    private fun drawCropMask(canvas: Canvas) {
        val s = cropScale
        if (s <= 1.01f || width <= 0 || height <= 0) return

        val w = width / s
        val h = height / s
        val left = (width - w) / 2f
        val top = (height - h) / 2f

        canvas.drawRect(0f, 0f, width.toFloat(), top, cropMaskPaint)
        canvas.drawRect(0f, top + h, width.toFloat(), height.toFloat(), cropMaskPaint)
        canvas.drawRect(0f, top, left, top + h, cropMaskPaint)
        canvas.drawRect(left + w, top, width.toFloat(), top + h, cropMaskPaint)
        canvas.drawRect(left, top, left + w, top + h, cropBorderPaint)

        val label = "数字 ${String.format(java.util.Locale.US, "%.1f", s)}x · 框外不识别"
        val tw = hintText.measureText(label)
        val padH = dp(8f)
        val padV = dp(4f)
        val boxH = hintText.textSize + padV * 2
        canvas.drawRoundRect(
            left + dp(6f), top + dp(6f),
            left + dp(6f) + tw + padH * 2, top + dp(6f) + boxH,
            dp(6f), dp(6f), hintBg
        )
        canvas.drawText(label, left + dp(6f) + padH, top + dp(6f) + boxH - padV - dp(1f), hintText)
    }

    private fun drawRois(canvas: Canvas, geo: Geometry, scale: Float, offsetX: Float, offsetY: Float) {
        if (rois.isEmpty() && draft.isEmpty() && !editMode) return

        rois.forEachIndexed { index, roi ->
            roiPath.reset()
            roi.points.forEachIndexed { i, point ->
                toScreen(point, geo, scale, offsetX, offsetY)
                if (i == 0) roiPath.moveTo(pointScratch.x, pointScratch.y)
                else roiPath.lineTo(pointScratch.x, pointScratch.y)
            }
            roiPath.close()

            canvas.drawPath(roiPath, roiFill)
            canvas.drawPath(roiPath, roiStroke)

            // 编号标在第一个顶点旁边，方便和多边形一一对应
            val first = roi.points.first()
            toScreen(first, geo, scale, offsetX, offsetY)
            val text = "${index + 1}"
            val tx = pointScratch.x.coerceIn(dp(4f), width - dp(18f))
            val ty = (pointScratch.y - dp(6f)).coerceIn(dp(16f), height - dp(4f))
            canvas.drawCircle(tx, ty, dp(9f), roiStroke)
            canvas.drawText(text, tx - dp(3f), ty + dp(4f), labelText)
        }
    }

    private fun drawDraft(canvas: Canvas, geo: Geometry, scale: Float, offsetX: Float, offsetY: Float) {
        if (draft.isEmpty()) return

        path.reset()
        draft.forEachIndexed { i, point ->
            toScreen(point, geo, scale, offsetX, offsetY)
            if (i == 0) path.moveTo(pointScratch.x, pointScratch.y)
            else path.lineTo(pointScratch.x, pointScratch.y)
        }
        canvas.drawPath(path, draftStroke)

        // 顶点之间已经连成线，再给每个顶点画个实心点，看清自己点到哪了
        draft.forEach { point ->
            toScreen(point, geo, scale, offsetX, offsetY)
            canvas.drawCircle(pointScratch.x, pointScratch.y, dp(3.5f), vertexPaint)
        }

        // 首点加个圈：提示「点这里可以闭合」
        toScreen(draft.first(), geo, scale, offsetX, offsetY)
        canvas.drawCircle(
            pointScratch.x, pointScratch.y,
            dp(if (closeHintReady) 11f else 7f), vertexFirstPaint
        )

        if (closeHintReady) {
            // 虚线连回起点，让「还差一步闭合」这件事在视觉上成立
            toScreen(draft.last(), geo, scale, offsetX, offsetY)
            path.reset()
            path.moveTo(pointScratch.x, pointScratch.y)
            toScreen(draft.first(), geo, scale, offsetX, offsetY)
            path.lineTo(pointScratch.x, pointScratch.y)
            canvas.drawPath(path, draftStroke)
        }
    }

    private fun drawEditHint(canvas: Canvas) {
        val text = when {
            draft.isEmpty() -> "点按画面添加多边形顶点"
            draft.size < Roi.MIN_POINTS -> "继续点，至少 ${Roi.MIN_POINTS} 个点（已 ${draft.size}）"
            else -> "再点 ${Roi.MIN_POINTS} 点以上可继续加，点回起点闭合"
        }
        val textWidth = hintText.measureText(text)
        val padH = dp(10f)
        val padV = dp(6f)
        val boxHeight = hintText.textSize + padV * 2
        val left = (width - textWidth - padH * 2) / 2f
        val top = height - boxHeight - dp(96f)

        canvas.drawRoundRect(
            left, top, left + textWidth + padH * 2, top + boxHeight,
            dp(6f), dp(6f), hintBg
        )
        canvas.drawText(text, left + padH, top + boxHeight - padV - dp(1f), hintText)
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

    /** 归一化坐标 -> 屏幕坐标（就地写入 [pointScratch]，避免每帧产生大量临时对象） */
    private fun toScreen(point: PointF, geo: Geometry, scale: Float, offsetX: Float, offsetY: Float) {
        val sx = point.x * geo.srcW
        val sy = point.y * geo.srcH
        var vx = sx * scale + offsetX
        if (geo.mirrored) vx = width - vx
        pointScratch.x = vx
        pointScratch.y = sy * scale + offsetY
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
        /** 点回起点的判定半径（归一化）。太小手指够不到，太大会误闭合 */
        private const val CLOSE_GAP = 0.035f

        /** 数字裁切上限，和 FrameAnalyzer.MAX_CROP_SCALE 保持一致 */
        private const val MAX_CROP_SCALE = 6f

        /** 捏合松手时低于这个倍数就吸附回 1.0（不裁） */
        private const val SNAP_TO_ONE_BELOW = 1.12f

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
