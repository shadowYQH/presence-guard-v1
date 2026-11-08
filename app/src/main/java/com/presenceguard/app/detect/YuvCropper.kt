package com.presenceguard.app.detect

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import androidx.camera.core.ImageProxy

/**
 * 直接从 YUV_420_888 里挖出一块矩形并转成 ARGB 位图，**不碰矩形外的像素**。
 *
 * 为什么需要它：`ImageProxy.toBitmap()` 会把整帧 1080×1440 全部转一遍（实测 25~28ms，
 * 还要分配一块 6MB 的位图）。而选区模式下我们真正要看的可能只有 400×300 ——
 * 剩下 95% 的像素转完就被扔掉了，纯浪费。这个类只遍历目标矩形内的像素。
 *
 * 坐标系约定（很容易搞错，写在这里免得下次再推一遍）：
 *
 *  - `ImageProxy` 的 `width/height` 是**传感器方向**；
 *  - 界面、叠加层、选区多边形用的都是**正立方向**（旋转 `rotationDegrees` 度顺时针之后）；
 *  - 所以本类接收的是正立坐标系的矩形，内部再反查回传感器坐标。
 *
 * `ImageProxy.toBitmap()` 本身**不做旋转**（走 `ImageUtil.createBitmapFromPlane`，
 * 直接按 width×height 出图），旋转是调用方自己做的 —— 所以正立方向的换算只有这一处，
 * 不会和别处重复。
 */
object YuvCropper {

    /**
     * @param rect 正立坐标系下的矩形，会被裁剪到画面内
     * @return 已经旋转到正立方向的位图；格式不支持或参数非法时返回 null
     */
    fun crop(proxy: ImageProxy, rect: Rect): Bitmap? {
        if (proxy.format != ImageFormat.YUV_420_888) return null

        val sensorW = proxy.width
        val sensorH = proxy.height
        if (sensorW <= 0 || sensorH <= 0) return null

        val rotation = ((proxy.imageInfo.rotationDegrees % 360) + 360) % 360
        val turned = rotation % 180 != 0
        val orientedW = if (turned) sensorH else sensorW
        val orientedH = if (turned) sensorW else sensorH

        val left = rect.left.coerceIn(0, orientedW - 1)
        val top = rect.top.coerceIn(0, orientedH - 1)
        val right = rect.right.coerceIn(left + 1, orientedW)
        val bottom = rect.bottom.coerceIn(top + 1, orientedH)
        val outW = right - left
        val outH = bottom - top
        if (outW <= 0 || outH <= 0) return null

        val planes = proxy.planes
        if (planes.size < 3) return null
        val yBuf = planes[0].buffer
        val uBuf = planes[1].buffer
        val vBuf = planes[2].buffer
        val yRow = planes[0].rowStride
        val yPix = planes[0].pixelStride
        val uRow = planes[1].rowStride
        val uPix = planes[1].pixelStride
        val vRow = planes[2].rowStride
        val vPix = planes[2].pixelStride

        val pixels = IntArray(outW * outH)
        var at = 0

        for (oy in 0 until outH) {
            // 正立行号换算成传感器行号，整行只算一次
            val gy = top + oy
            when (rotation) {
                90 -> {
                    // dest(ox,oy) = src(x = oy, y = H-1-ox)
                    val sx = gy
                    for (ox in 0 until outW) {
                        val sy = sensorH - 1 - (left + ox)
                        pixels[at++] = yuvToArgb(
                            yBuf, uBuf, vBuf, yRow, yPix, uRow, uPix, vRow, vPix, sx, sy
                        )
                    }
                }
                180 -> {
                    val sy = sensorH - 1 - gy
                    for (ox in 0 until outW) {
                        val sx = sensorW - 1 - (left + ox)
                        pixels[at++] = yuvToArgb(
                            yBuf, uBuf, vBuf, yRow, yPix, uRow, uPix, vRow, vPix, sx, sy
                        )
                    }
                }
                270 -> {
                    // dest(ox,oy) = src(x = W-1-oy, y = ox)
                    val sx = sensorW - 1 - gy
                    for (ox in 0 until outW) {
                        val sy = left + ox
                        pixels[at++] = yuvToArgb(
                            yBuf, uBuf, vBuf, yRow, yPix, uRow, uPix, vRow, vPix, sx, sy
                        )
                    }
                }
                else -> {
                    val sy = gy
                    for (ox in 0 until outW) {
                        val sx = left + ox
                        pixels[at++] = yuvToArgb(
                            yBuf, uBuf, vBuf, yRow, yPix, uRow, uPix, vRow, vPix, sx, sy
                        )
                    }
                }
            }
        }

        return runCatching {
            Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888).apply {
                setPixels(pixels, 0, outW, 0, 0, outW, outH)
            }
        }.getOrNull()
    }

    /**
     * 单像素 YUV → ARGB。BT.601 limited range（相机输出基本都是这个）：
     * `R = 1.164(Y-16) + 1.596(V-128)` 等，这里用整数近似（>>8）省掉浮点。
     */
    private fun yuvToArgb(
        yBuf: java.nio.ByteBuffer, uBuf: java.nio.ByteBuffer, vBuf: java.nio.ByteBuffer,
        yRow: Int, yPix: Int, uRow: Int, uPix: Int, vRow: Int, vPix: Int,
        sx: Int, sy: Int
    ): Int {
        val y = yBuf.get(sy * yRow + sx * yPix).toInt() and 0xFF
        val uvX = sx shr 1
        val uvY = sy shr 1
        val u = uBuf.get(uvY * uRow + uvX * uPix).toInt() and 0xFF
        val v = vBuf.get(uvY * vRow + uvX * vPix).toInt() and 0xFF

        val yy = (y - 16).coerceAtLeast(0) * 298
        val uu = u - 128
        val vv = v - 128

        val r = (yy + 409 * vv) shr 8
        val g = (yy - 100 * uu - 208 * vv) shr 8
        val b = (yy + 516 * uu) shr 8

        return (0xFF shl 24) or
            (r.coerceIn(0, 255) shl 16) or
            (g.coerceIn(0, 255) shl 8) or
            b.coerceIn(0, 255)
    }
}
