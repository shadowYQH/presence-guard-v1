package com.presenceguard.app

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.presenceguard.app.core.FrameResult
import com.presenceguard.app.core.Sensitivity
import com.presenceguard.app.databinding.ActivityMainBinding
import com.presenceguard.app.detect.PresenceFusionEngine
import com.presenceguard.app.service.DetectionService
import java.util.Locale

/**
 * 只负责界面。相机与推理全部在 [DetectionService] 里，
 * 这样息屏、切后台、Activity 重建都不会打断检测。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private var service: DetectionService? = null
    private var bound = false
    private var dimmed = false

    private var lastPresent = false
    private var lastCount = -1
    private var metricTick = 0
    private var cachedSensitivity = Sensitivity.MEDIUM

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { result ->
        if (result[Manifest.permission.CAMERA] == true) {
            startAndBind()
        } else {
            showPermissionPanel()
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val local = binder as? DetectionService.LocalBinder ?: return
            service = local.service
            bound = true
            local.service.addListener(frameListener)
            attachPreview()
            syncButtons()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
            bound = false
        }
    }

    private val frameListener = object : DetectionService.Listener {
        override fun onFrame(result: FrameResult, outcome: PresenceFusionEngine.Outcome) {
            runOnUiThread {
                if (!dimmed) binding.overlayView.render(result, outcome.present)

                if (outcome.present != lastPresent || outcome.personCount != lastCount) {
                    lastPresent = outcome.present
                    lastCount = outcome.personCount
                    applyStatus()
                }

                if (++metricTick % 5 == 0 && !dimmed) {
                    val score = String.format(Locale.US, "%.2f", outcome.smoothedScore)
                    binding.metricsText.text =
                        "${result.fps.toInt()} fps · ${result.processingMs} ms · 分值 $score · ${outcome.votes}"
                    binding.metricsText.visibility = View.VISIBLE
                }
            }
        }

        override fun onStatus(status: DetectionService.Status) {
            runOnUiThread { applyStatus(status) }
        }

        override fun onError(message: String) {
            runOnUiThread { Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show() }
        }
    }

    // ------------------------------------------------------------ 生命周期

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setupControls()

        if (hasCameraPermission()) {
            startAndBind()
        } else {
            showPermissionPanel()
            requestPermissions()
        }
    }

    override fun onStart() {
        super.onStart()
        if (bound) {
            service?.addListener(frameListener)
            attachPreview()
        }
    }

    override fun onStop() {
        super.onStop()
        // 界面不可见时摘掉回调，避免后台堆积 UI 消息；检测本身仍在服务里跑
        if (bound) service?.removeListener(frameListener)
        // 顺便断开预览渲染：息屏 / 切后台时不必再往屏幕送帧
        if (bound) service?.setPreviewSurfaceProvider(null)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (bound) {
            runCatching { service?.removeListener(frameListener) }
            runCatching { unbindService(serviceConnection) }
            bound = false
        }
    }

    // ------------------------------------------------------------ 服务

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun requestPermissions() {
        val permissions = mutableListOf(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions += Manifest.permission.POST_NOTIFICATIONS
        }
        permissionLauncher.launch(permissions.toTypedArray())
    }

    private fun startAndBind() {
        binding.permissionPanel.visibility = View.GONE
        DetectionService.start(this)
        val intent = Intent(this, DetectionService::class.java)
        runCatching { bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE) }
    }

    private fun showPermissionPanel() {
        binding.permissionPanel.visibility = View.VISIBLE
        binding.btnGrant.setOnClickListener { requestPermissions() }
    }

    /** 把 PreviewView 的画面通道交给服务里的 Preview 用例 */
    private fun attachPreview() {
        service?.setPreviewSurfaceProvider(binding.previewView.surfaceProvider)
    }

    // ------------------------------------------------------------ 界面

    private fun setupControls() {
        binding.btnSwitch.setOnClickListener { service?.switchCamera() }

        binding.btnSensitivity.setOnClickListener {
            val svc = service ?: return@setOnClickListener
            val next = when (cachedSensitivity) {
                Sensitivity.HIGH -> Sensitivity.MEDIUM
                Sensitivity.MEDIUM -> Sensitivity.LOW
                Sensitivity.LOW -> Sensitivity.HIGH
            }
            cachedSensitivity = next
            svc.setSensitivity(next)
            updateSensitivityButton(next)
            binding.overlayView.clear()
            lastPresent = false
            lastCount = -1
        }

        binding.btnTone.setOnClickListener {
            val svc = service ?: return@setOnClickListener
            val tone = svc.cycleTone()
            binding.btnTone.text = getString(R.string.btn_tone_fmt, tone.displayName)
            if (!svc.isAlarmAudible()) {
                Toast.makeText(this, R.string.hint_alarm_muted, Toast.LENGTH_LONG).show()
            }
        }

        binding.btnSound.setOnClickListener {
            val svc = service ?: return@setOnClickListener
            val enabled = !svc.isAudioEnabled()
            svc.setAudioEnabled(enabled)
            updateSoundButton(enabled)
        }

        binding.btnPause.setOnClickListener {
            val svc = service ?: return@setOnClickListener
            val paused = svc.togglePause()
            binding.btnPause.setText(if (paused) R.string.btn_resume else R.string.btn_pause)
            binding.overlayView.clear()
        }

        binding.btnTest.setOnClickListener {
            service?.previewTone()
            if (service?.isAlarmAudible() == false) {
                Toast.makeText(this, R.string.hint_alarm_muted, Toast.LENGTH_LONG).show()
            }
        }

        binding.btnDim.setOnClickListener { setDimmed(!dimmed) }

        binding.btnPower.setOnClickListener {
            val svc = service ?: return@setOnClickListener
            val next = !svc.isPowerSave()
            svc.setPowerSave(next)
            updatePowerButton(next)
            Toast.makeText(
                this,
                if (next) R.string.hint_power_on else R.string.hint_power_off,
                Toast.LENGTH_SHORT
            ).show()
        }

        // 暗屏状态下任意触摸即恢复亮度，避免用户找不到退出方式
        binding.root.setOnTouchListener { _, _ ->
            if (dimmed) {
                setDimmed(false)
                true
            } else {
                false
            }
        }

        updateSensitivityButton(cachedSensitivity)
        updateSoundButton(true)
    }

    private fun updateSensitivityButton(value: Sensitivity) {
        val textRes = when (value) {
            Sensitivity.HIGH -> R.string.btn_sensitivity_high
            Sensitivity.MEDIUM -> R.string.btn_sensitivity_medium
            Sensitivity.LOW -> R.string.btn_sensitivity_low
        }
        binding.btnSensitivity.setText(textRes)
    }

    private fun updateSoundButton(enabled: Boolean) {
        binding.btnSound.setText(if (enabled) R.string.btn_sound_on else R.string.btn_sound_off)
    }

    private fun updatePowerButton(enabled: Boolean) {
        binding.btnPower.setText(if (enabled) R.string.btn_power_on else R.string.btn_power_off)
    }

    private fun syncButtons() {
        val svc = service ?: return
        updateSoundButton(svc.isAudioEnabled())
        updatePowerButton(svc.isPowerSave())
        binding.btnTone.text = getString(R.string.btn_tone_fmt, svc.currentTone().displayName)
        binding.btnPause.setText(if (svc.isPaused()) R.string.btn_resume else R.string.btn_pause)
        applyStatus(svc.currentStatus())
    }

    private fun applyStatus(status: DetectionService.Status? = service?.currentStatus()) {
        val current = status ?: return
        when (current.phase) {
            DetectionService.Phase.INITIALIZING -> {
                binding.statusDot.setBackgroundResource(R.drawable.dot_idle)
                binding.statusTitle.setText(R.string.status_booting)
                binding.statusDetail.setText(R.string.status_booting_detail)
            }

            DetectionService.Phase.ERROR -> {
                binding.statusDot.setBackgroundResource(R.drawable.dot_alert)
                binding.statusTitle.setText(R.string.status_error)
                binding.statusDetail.text = current.message.orEmpty()
            }

            DetectionService.Phase.PAUSED -> {
                binding.statusDot.setBackgroundResource(R.drawable.dot_idle)
                binding.statusTitle.setText(R.string.status_paused)
                binding.statusDetail.setText(R.string.notify_paused_detail)
            }

            DetectionService.Phase.NO_PERMISSION -> showPermissionPanel()

            DetectionService.Phase.RUNNING -> {
                binding.statusTitle.setTextColor(ContextCompat.getColor(this, R.color.text_primary))
                if (current.present) {
                    binding.statusDot.setBackgroundResource(R.drawable.dot_alert)
                    binding.statusTitle.setText(R.string.status_present)
                    binding.statusTitle.setTextColor(ContextCompat.getColor(this, R.color.accent_red))
                    val confidence = (current.confidence * 100).toInt().coerceIn(0, 100)
                    binding.statusDetail.text = "画面中 ${current.personCount} 人 · 把握 $confidence%"
                } else {
                    binding.statusDot.setBackgroundResource(R.drawable.dot_active)
                    binding.statusTitle.setText(R.string.status_idle)
                    binding.statusDetail.setText(R.string.status_running_detail)
                    lastPresent = false
                    lastCount = 0
                }
            }
        }
    }

    private fun setDimmed(dim: Boolean) {
        dimmed = dim
        val params = window.attributes
        params.screenBrightness = if (dim) DIM_BRIGHTNESS else WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
        window.attributes = params

        if (dim) {
            binding.overlayView.clear()
            binding.statusCard.alpha = 0.35f
            binding.controlBar.alpha = 0.35f
            binding.metricsText.visibility = View.GONE
        } else {
            binding.statusCard.alpha = 1f
            binding.controlBar.alpha = 1f
        }
        binding.btnDim.setText(if (dim) R.string.btn_brighten else R.string.btn_dim)
    }

    private companion object {
        /**
         * 暗屏亮度。0.01 基本等于全黑，背光确实最省电，但屏幕上一片漆黑，
         * 分不清「在检测」还是「没画面」。0.12 已经把背光功耗压掉八成以上，
         * 同时还能看清画面里有没有人。
         */
        const val DIM_BRIGHTNESS = 0.12f
    }
}
