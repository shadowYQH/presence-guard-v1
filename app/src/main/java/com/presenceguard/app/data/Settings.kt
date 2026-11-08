package com.presenceguard.app.data

import android.content.Context
import android.content.SharedPreferences
import com.presenceguard.app.audio.AlertTone
import com.presenceguard.app.core.Roi
import com.presenceguard.app.core.Sensitivity

/**
 * 用户设置的持久化。用 SharedPreferences 而不是 DataStore：
 * 这里只有六七个标量，读写在主线程瞬时完成，没必要引入协程流。
 */
class Settings(context: Context) {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(NAME, Context.MODE_PRIVATE)

    var sensitivity: Sensitivity
        get() = runCatching { Sensitivity.valueOf(prefs.getString(KEY_SENSITIVITY, Sensitivity.MEDIUM.name)!!) }
            .getOrDefault(Sensitivity.MEDIUM)
        set(value) {
            prefs.edit().putString(KEY_SENSITIVITY, value.name).apply()
        }

    var tone: AlertTone
        get() = AlertTone.fromName(prefs.getString(KEY_TONE, AlertTone.DINGDONG.name))
        set(value) {
            prefs.edit().putString(KEY_TONE, value.name).apply()
        }

    /** 是否开启声音报警 */
    var audioEnabled: Boolean
        get() = prefs.getBoolean(KEY_AUDIO, true)
        set(value) {
            prefs.edit().putBoolean(KEY_AUDIO, value).apply()
        }

    /** 持续报警的重复间隔（毫秒） */
    var alertIntervalMs: Long
        get() = prefs.getLong(KEY_INTERVAL, DEFAULT_INTERVAL).coerceIn(500L, 10000L)
        set(value) {
            prefs.edit().putLong(KEY_INTERVAL, value.coerceIn(500L, 10000L)).apply()
        }

    /** 摄像头朝向：CameraSelector.LENS_FACING_* */
    var lensFacing: Int
        get() = prefs.getInt(KEY_LENS, 1)
        set(value) {
            prefs.edit().putInt(KEY_LENS, value).apply()
        }

    /** 省电模式：降低分析帧率、约束 CPU 唤醒 */
    var powerSave: Boolean
        get() = prefs.getBoolean(KEY_POWER_SAVE, false)
        set(value) {
            prefs.edit().putBoolean(KEY_POWER_SAVE, value).apply()
        }

    /**
     * 变焦倍率。CameraX 的 zoom 是传感器级裁切 + 缩放，**同时作用于预览和分析流**，
     * 所以把远处的人拉近之后，检测拿到的是真实的放大像素，不是数码插值。
     */
    var zoomRatio: Float
        get() = prefs.getFloat(KEY_ZOOM, 1f).coerceIn(1f, MAX_ZOOM)
        set(value) {
            prefs.edit().putFloat(KEY_ZOOM, value.coerceIn(1f, MAX_ZOOM)).apply()
        }

    /** 后置多摄时用户选中的物理镜头序号（0 起）；-1 表示用系统的默认逻辑相机 */
    var lensIndex: Int
        get() = prefs.getInt(KEY_LENS_INDEX, 0)
        set(value) {
            prefs.edit().putInt(KEY_LENS_INDEX, value).apply()
        }

    /**
     * 用户框选的多边形区域。
     *
     * 存成一个字符串：多个多边形用 `|` 分隔，每个多边形的顶点用 `;` 分隔，
     * 顶点坐标是 `x,y` 归一化值。不用 JSON 是因为这里就一个固定结构，
     * 手写编解码比引入序列化库更省心，也便于用 adb 直接看配置。
     */
    var rois: List<Roi>
        get() {
            val raw = prefs.getString(KEY_ROIS, "") ?: ""
            if (raw.isBlank()) return emptyList()
            return raw.split('|').mapNotNull { Roi.decode(it) }
        }
        set(value) {
            prefs.edit()
                .putString(KEY_ROIS, value.filter { it.isValid }.joinToString("|") { it.encode() })
                .apply()
        }

    companion object {
        private const val NAME = "presence_guard_settings"
        private const val KEY_SENSITIVITY = "sensitivity"
        private const val KEY_TONE = "tone"
        private const val KEY_AUDIO = "audio_enabled"
        private const val KEY_INTERVAL = "alert_interval"
        private const val KEY_LENS = "lens_facing"
        private const val KEY_POWER_SAVE = "power_save"
        private const val KEY_ZOOM = "zoom_ratio"
        private const val KEY_LENS_INDEX = "lens_index"
        private const val KEY_ROIS = "rois"

        /** 变焦上限。远超过这个值只是纯数码插值，没有任何新信息 */
        const val MAX_ZOOM = 8f

        /** 默认 1.5 秒重复一次，既不吵又能持续提醒 */
        const val DEFAULT_INTERVAL = 1500L
    }
}
