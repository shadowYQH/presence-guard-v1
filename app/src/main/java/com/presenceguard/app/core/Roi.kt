package com.presenceguard.app.core

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs

/**
 * 用户框选的多边形感兴趣区域。
 *
 * 坐标是**归一化的**（0~1，相对分析画面），不是像素。这样切换分辨率、切换摄像头、
 * 甚至改采集尺寸之后，选区依然落在同一个物理位置上，不需要跟着重算。
 *
 * 为什么用多边形而不是矩形：实际场景里要盯的区域往往不是方的——走廊是长条、
 * 门口是斜的、阳台栏杆是细长一条。矩形会把大片无关背景卷进来，多边形能贴着边界框。
 */
data class Roi(
    val points: List<PointF>,
    /** 选区在画面里的显示名次，用于叠加层标注 */
    val label: String = ""
) {
    val isValid: Boolean get() = points.size >= MIN_POINTS

    /** 外接矩形（归一化），用于裁剪送检 */
    fun bounds(): RectF {
        if (points.isEmpty()) return RectF()
        var left = 1f
        var top = 1f
        var right = 0f
        var bottom = 0f
        for (p in points) {
            if (p.x < left) left = p.x
            if (p.x > right) right = p.x
            if (p.y < top) top = p.y
            if (p.y > bottom) bottom = p.y
        }
        return RectF(
            left.coerceIn(0f, 1f),
            top.coerceIn(0f, 1f),
            right.coerceIn(0f, 1f),
            bottom.coerceIn(0f, 1f)
        )
    }

    /** 外接矩形换算成指定画面的像素坐标 */
    fun boundsPx(width: Int, height: Int): RectF {
        val b = bounds()
        return RectF(b.left * width, b.top * height, b.right * width, b.bottom * height)
    }

    /** 射线法判断点是否在多边形内。坐标为归一化值。 */
    fun contains(x: Float, y: Float): Boolean {
        if (points.size < MIN_POINTS) return false
        var inside = false
        var j = points.size - 1
        for (i in points.indices) {
            val pi = points[i]
            val pj = points[j]
            val crosses = (pi.y > y) != (pj.y > y)
            if (crosses) {
                val t = (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y) + pi.x
                if (x < t) inside = !inside
            }
            j = i
        }
        return inside
    }

    /**
     * 选区相对整幅画面的面积占比。太小（<0.02%）的选区放大后全是噪点，
     * 送检没有意义，反而增加耗时。
     */
    fun areaRatio(): Float {
        if (points.size < MIN_POINTS) return 0f
        var sum = 0f
        var j = points.size - 1
        for (i in points.indices) {
            sum += (points[j].x * points[i].y - points[i].x * points[j].y)
            j = i
        }
        return abs(sum) / 2f
    }

    fun encode(): String = points.joinToString(";") { "${it.x},${it.y}" }

    companion object {
        /** 三角形是最少的多边形，两点只能画一条线 */
        const val MIN_POINTS = 3

        /** 相邻顶点间距过近视为误触，加不进选区 */
        const val MIN_VERTEX_GAP = 0.012f

        /** 选区面积下限（占画面比例），低于此值送检没有意义 */
        const val MIN_AREA_RATIO = 0.0002f

        fun decode(raw: String): Roi? {
            if (raw.isBlank()) return null
            val pts = raw.split(";").mapNotNull { token ->
                val parts = token.split(",")
                if (parts.size != 2) return@mapNotNull null
                val x = parts[0].toFloatOrNull() ?: return@mapNotNull null
                val y = parts[1].toFloatOrNull() ?: return@mapNotNull null
                PointF(x.coerceIn(0f, 1f), y.coerceIn(0f, 1f))
            }
            return if (pts.size < MIN_POINTS) null else Roi(pts)
        }
    }
}
