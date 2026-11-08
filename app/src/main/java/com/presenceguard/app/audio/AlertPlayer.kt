package com.presenceguard.app.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.ToneGenerator
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin

/**
 * 提示音色。全部用程序合成 PCM，不依赖任何音频素材文件，
 * 方便按需调整参数，也让 APK 不必携带额外资源。
 */
enum class AlertTone(val displayName: String) {
    DINGDONG("叮咚"),
    BEEP("蜂鸣"),
    ALARM("警报"),
    SIREN("警笛"),
    HEARTBEAT("心跳");

    fun next(): AlertTone {
        val all = entries
        return all[(ordinal + 1) % all.size]
    }

    companion object {
        fun fromName(name: String?): AlertTone =
            entries.firstOrNull { it.name == name } ?: DINGDONG
    }
}

/**
 * 提示音播放器。
 *
 * 走 **ALARM 通道**（USAGE_ALARM + STREAM_ALARM），手机静音或媒体音量为 0 时依然能响，
 * 这是看护 / 安防场景的必要行为。
 *
 * 内部单线程串行播放：持续报警时每 1~2 秒触发一次，串行队列能保证不会并发写
 * AudioTrack 导致状态错乱。
 */
class AlertPlayer(private val context: Context) {

    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "alert-player").apply { priority = Thread.MAX_PRIORITY }
    }

    @Volatile
    private var tone: AlertTone = AlertTone.DINGDONG

    @Volatile
    private var vibrateEnabled: Boolean = true

    private var track: AudioTrack? = null
    private var trackTone: AlertTone? = null

    fun setTone(newTone: AlertTone) {
        tone = newTone
    }

    fun currentTone(): AlertTone = tone

    fun setVibrateEnabled(enabled: Boolean) {
        vibrateEnabled = enabled
    }

    /** 播一次当前音色。异步执行，不阻塞调用线程。 */
    fun play() {
        executor.execute {
            try {
                playTone()
            } catch (t: Throwable) {
                Log.w(TAG, "AudioTrack playback failed, falling back to ToneGenerator", t)
                fallbackBeep()
            }
            if (vibrateEnabled) vibrateOnce()
        }
    }

    /** 闹钟通道是否被调成 0，用来提示用户「响了但听不见」 */
    fun isAlarmAudible(): Boolean {
        val am = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return true
        return try {
            am.getStreamVolume(AudioManager.STREAM_ALARM) > 0
        } catch (t: Throwable) {
            true
        }
    }

    private fun playTone() {
        val desired = tone
        if (track == null || trackTone != desired) {
            releaseTrack()
            track = createTrack(synthesize(desired))
            trackTone = desired
        }
        val local = track ?: return
        local.stop()
        local.reloadStaticData()
        local.play()
    }

    private fun createTrack(pcm: ShortArray): AudioTrack {
        val attributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ALARM)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()

        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(SAMPLE_RATE)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()

        val created = AudioTrack.Builder()
            .setAudioAttributes(attributes)
            .setAudioFormat(format)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .setBufferSizeInBytes(pcm.size * 2)
            .build()

        created.write(pcm, 0, pcm.size)
        return created
    }

    private fun releaseTrack() {
        runCatching {
            track?.stop()
            track?.release()
        }
        track = null
        trackTone = null
    }

    private fun fallbackBeep() {
        runCatching {
            val tg = ToneGenerator(AudioManager.STREAM_ALARM, 95)
            tg.startTone(ToneGenerator.TONE_PROP_ACK, 350)
            Thread.sleep(420)
            tg.release()
        }
    }

    private fun vibrateOnce() {
        try {
            val vibrator: Vibrator = (if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val manager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
                manager?.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
            }) ?: return

            if (!vibrator.hasVibrator()) return
            vibrator.vibrate(VibrationEffect.createOneShot(120, VibrationEffect.DEFAULT_AMPLITUDE))
        } catch (t: Throwable) {
            Log.w(TAG, "vibration failed", t)
        }
    }

    fun release() {
        executor.execute { releaseTrack() }
        executor.shutdown()
    }

    // ------------------------------------------------------------ 合成

    private data class Segment(
        val freqStart: Double,
        val freqEnd: Double,
        val durationMs: Int,
        val amplitude: Double,
        val square: Boolean = false
    )

    private fun synthesize(which: AlertTone): ShortArray {
        val segments = when (which) {
            // 经典「叮—咚」，温和但穿透力够
            AlertTone.DINGDONG -> listOf(
                Segment(880.0, 880.0, 120, 0.55),
                Segment(0.0, 0.0, 55, 0.0),
                Segment(1320.0, 1320.0, 190, 0.65)
            )
            // 短促高频蜂鸣，嘈杂环境更醒目
            AlertTone.BEEP -> listOf(
                Segment(1800.0, 1800.0, 90, 0.60, square = true),
                Segment(0.0, 0.0, 60, 0.0),
                Segment(1800.0, 1800.0, 90, 0.60, square = true)
            )
            // 报警器式双音交替
            AlertTone.ALARM -> buildList {
                repeat(3) {
                    add(Segment(1000.0, 1000.0, 110, 0.62, square = true))
                    add(Segment(1400.0, 1400.0, 110, 0.62, square = true))
                }
            }
            // 扫频警笛
            AlertTone.SIREN -> listOf(
                Segment(700.0, 1400.0, 320, 0.55),
                Segment(1400.0, 700.0, 320, 0.55)
            )
            // 低频双击，像心跳，不刺耳
            AlertTone.HEARTBEAT -> listOf(
                Segment(70.0, 55.0, 110, 0.75),
                Segment(0.0, 0.0, 70, 0.0),
                Segment(65.0, 50.0, 140, 0.60)
            )
        }

        val total = segments.sumOf { it.durationMs * SAMPLE_RATE / 1000 }
        val out = ShortArray(total)
        var cursor = 0

        for (segment in segments) {
            val count = segment.durationMs * SAMPLE_RATE / 1000
            if (count <= 0) continue
            if (segment.amplitude <= 0.0) {
                cursor += count
                continue
            }

            val attack = (0.006 * SAMPLE_RATE).toInt().coerceAtLeast(1)
            val decay = if (segment.square) 1.5 else 3.0
            var phase = 0.0

            for (i in 0 until count) {
                val progress = i.toDouble() / count
                val freq = segment.freqStart + (segment.freqEnd - segment.freqStart) * progress
                phase += 2.0 * PI * freq / SAMPLE_RATE

                val envelope = if (i < attack) {
                    i.toDouble() / attack
                } else {
                    exp(-decay * ((i - attack).toDouble() / (count - attack).coerceAtLeast(1)))
                }

                val wave = if (segment.square) {
                    if (sin(phase) >= 0) 1.0 else -1.0
                } else {
                    sin(phase)
                }

                val value = wave * segment.amplitude * envelope
                out[cursor + i] = (value * Short.MAX_VALUE).toInt()
                    .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                    .toShort()
            }
            cursor += count
        }
        return out
    }

    companion object {
        private const val TAG = "AlertPlayer"
        const val SAMPLE_RATE = 44100
    }
}
