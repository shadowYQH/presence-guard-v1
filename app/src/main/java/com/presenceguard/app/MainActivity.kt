package com.presenceguard.app

import android.Manifest
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import com.presenceguard.app.core.FrameResult
import com.presenceguard.app.core.Roi
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
    private var rois: List<Roi> = emptyList()

    /** 「数字裁切」这件事只提示一次，别每帧弹 */
    private var cropHintShown = false

    /** 最近一次巡检的人话结论，挂在指标行第二行 */
    private var inspectionLine = ""

    /** 指标行第一行（帧率 / 耗时 / 分值），巡检刷新时不能被覆盖掉 */
    private var metricsHead = ""

    /** 屏幕是否亮着。熄屏时要主动摘掉预览，见 [screenReceiver] 的注释 */
    private var screenOn = true

    /**
     * 屏幕开关广播。
     *
     * 为什么不能只靠 onStop：实测在 Android 13 上按下电源键熄屏后，Activity 不一定
     * 立刻收到 onStop（系统只是让显示进入 doze，进程仍被视作可见），于是预览流会继续跑。
     * ACTION_SCREEN_OFF 是确定会来的，用它兜底。
     */
    private val screenReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val action = intent?.action ?: return
            android.util.Log.i("MainActivity", "screen broadcast: $action")
            when (action) {
                Intent.ACTION_SCREEN_OFF -> {
                    screenOn = false
                    service?.setPreviewVisible(false)
                }
                Intent.ACTION_SCREEN_ON -> {
                    screenOn = true
                    if (lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) attachPreview()
                }
            }
        }
    }

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
            // 选区是持久化的，服务里那份才是权威值
            rois = local.service.currentRois()
            binding.overlayView.setRois(rois)
            cachedSensitivity = local.service.currentSensitivity()
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
                    metricsHead =
                        "${result.fps.toInt()} fps · ${result.processingMs} ms · 分值 $score · ${outcome.votes}"
                    renderMetrics()
                }
            }
        }

        /**
         * 巡检结果。
         *
         * 巡检不走 [onFrame]（它不参与主判定，也不 publish），所以指标行不会因为
         * 巡检自己刷新 —— 这里补一次，否则用户看不到「另一面有没有人」这个信息。
         */
        override fun onInspection(present: Boolean, score: Float, source: String) {
            runOnUiThread {
                val pct = (score * 100).toInt().coerceIn(0, 100)
                inspectionLine = if (present) {
                    "巡检·$source 有人 · 把握 $pct%"
                } else {
                    "巡检·$source 无人"
                }
                renderMetrics()
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
        registerScreenReceiver()

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

    /**
     * 在这里绑定预览。
     *
     * 用 onResume/onPause 而不是 onStart/onStop，是因为实测（MIUI Android 13）熄屏后
     * Activity 只走到 paused，onStop 迟迟不来；单靠 onStop 摘预览，熄屏时那条相机
     * 输出流会一直跑。onPause 一定会被调用，和 onResume 成对，行为可预期。
     */
    override fun onResume() {
        super.onResume()
        if (bound) attachPreview()
    }

    override fun onPause() {
        super.onPause()
        // 界面不在前台：把 Preview 用例从相机上摘掉，只留 ImageAnalysis
        android.util.Log.i("MainActivity", "onPause: detach preview")
        service?.setPreviewVisible(false)
    }

    override fun onStop() {
        super.onStop()
        // 界面不可见时摘掉回调，避免后台堆积 UI 消息；检测本身仍在服务里跑
        if (bound) service?.removeListener(frameListener)
        // 关键：让服务把 Preview 用例整个解绑。只是摘掉 SurfaceProvider 的话，
        // 相机的那条预览输出流还在跑，息屏后这部分功耗不会停。
        if (bound) service?.setPreviewVisible(false)
    }

    override fun onDestroy() {
        super.onDestroy()
        runCatching { unregisterReceiver(screenReceiver) }
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

    /** 把 PreviewView 的画面通道交给服务里的 Preview 用例，并让它把预览绑回去 */
    private fun attachPreview() {
        val svc = service ?: return
        svc.setPreviewSurfaceProvider(binding.previewView.surfaceProvider)
        // 屏幕还没亮就别急着绑预览
        if (screenOn) svc.setPreviewVisible(true)
    }

    private fun registerScreenReceiver() {
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_SCREEN_ON)
        }
        ContextCompat.registerReceiver(this, screenReceiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)
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

        // ---------------------------------------------------------- 焦段 / 镜头
        binding.btnZoom.setOnClickListener { cycleZoom() }

        binding.btnLens.setOnClickListener {
            service?.cycleLens()
            updateLensButton()
        }

        // ---------------------------------------------------------- 多边形选区
        binding.btnRoi.setOnClickListener { toggleRoiEdit() }

        binding.btnRoiUndo.setOnClickListener {
            val overlay = binding.overlayView
            // 编辑中 = 撤销一步（先撤顶点、再撤整个选区）；非编辑中 = 一键清空
            if (overlay.editMode) overlay.undo() else overlay.clearRois()
            updateRoiButton()
        }

        binding.overlayView.onRoisChange = { list ->
            rois = list
            service?.setRois(list)
            updateRoiButton()
            Toast.makeText(
                this,
                if (list.isEmpty()) getString(R.string.hint_roi_off)
                else getString(R.string.hint_roi_on, list.size),
                Toast.LENGTH_SHORT
            ).show()
        }

        // ---------------------------------------------------------- 双指缩放（数字裁切）
        //
        // 手势在 OverlayView 里识别（它盖在预览之上，能吃到所有触摸），
        // 这里只负责把倍率转给服务、刷新按钮文字、并在第一次缩放时提示一次
        // 「这是数字裁切，不是换镜头」—— 这两件事用户很容易搞混。
        binding.overlayView.onCropScaleChange = { scale ->
            service?.setCropScale(scale)
            if (!cropHintShown && scale > 1.01f) {
                cropHintShown = true
                Toast.makeText(
                    this,
                    getString(R.string.hint_crop_on, String.format(Locale.US, "%.1f", scale)),
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

        // ---------------------------------------------------------- 2 秒全图巡检
        binding.btnInspect.setOnClickListener {
            val svc = service ?: return@setOnClickListener
            val next = !svc.isDualInspect()
            svc.setDualInspect(next)
            updateInspectButton()
            // 并发绑定的结果是异步回来的，这里先按"双摄"报，等状态刷新后按钮会自己纠正成"巡检 单"
            Toast.makeText(
                this,
                if (next) R.string.hint_inspect_dual else R.string.hint_inspect_off,
                Toast.LENGTH_SHORT
            ).show()
        }

        // 暗屏状态下任意触摸即恢复亮度，避免用户找不到退出方式。
        //
        // 放在 dispatchTouchEvent 而不是 View.setOnTouchListener：叠加层（OverlayView）
        // 要实现双指缩放，必须吃掉全部触摸（非编辑模式下一律返回 true），
        // 挂在根布局上的 listener 在事件被下层消费后根本收不到 —— 那样暗屏就点不回来了。
        // 在分发之前拦掉，无论谁消费事件都能恢复亮度。
        updateSensitivityButton(cachedSensitivity)
        updateSoundButton(true)
        updateRoiButton()
        updateInspectButton()
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (dimmed) {
            setDimmed(false)
            return true
        }
        return super.dispatchTouchEvent(ev)
    }

    // ------------------------------------------------------------ 焦段 / 镜头

    /**
     * 循环切换变焦档位。
     *
     * 只列当前镜头确实支持的倍率（不同机型 maxZoomRatio 差别很大），
     * 切到最大档之后再点一次回到 1x。
     */
    private fun cycleZoom() {
        val svc = service ?: return
        val (minRatio, maxRatio) = svc.zoomRange()
        val presets = listOf(1f, 2f, 3f, 5f, 8f)
            .filter { it >= minRatio - 0.01f && it <= maxRatio + 0.01f }
            .ifEmpty { listOf(1f) }
        val current = svc.currentZoom()
        val next = presets.firstOrNull { it > current + 0.05f } ?: presets.first()
        svc.setZoomRatio(next)
        updateZoomButton(next)
    }

    private fun updateZoomButton(ratio: Float) {
        binding.btnZoom.text = getString(
            R.string.btn_zoom_fmt,
            String.format(Locale.US, "%.1fx", ratio)
        )
    }

    /** 后置只有一颗镜头时把按钮藏掉，避免点了没反应 */
    private fun updateLensButton() {
        val svc = service
        val count = svc?.lensCount() ?: 1
        binding.btnLens.visibility = if (count > 1) View.VISIBLE else View.GONE
        if (count > 1) {
            binding.btnLens.text = getString(R.string.btn_lens_fmt, (svc?.currentLensIndex() ?: 0) + 1)
        }
    }

    // ------------------------------------------------------------ 指标行

    /**
     * 指标行。
     *
     * 第一行是常规性能数据，第二行是巡检结论。两行分开维护，谁刷新都不会擦掉对方。
     * 只有**画面不可见**（暗屏 / 息屏）时才不渲染，因为那时候没人看。
     */
    private fun renderMetrics() {
        if (dimmed) return
        val text = listOf(metricsHead, inspectionLine)
            .filter { it.isNotEmpty() }
            .joinToString("\n")
        if (text.isEmpty()) return
        binding.metricsText.text = text
        binding.metricsText.visibility = View.VISIBLE
    }

    // ------------------------------------------------------------ 双指缩放

    /**
     * 双指缩放只改数字裁切，和「焦段」按钮的物理变焦是两件事。
     *
     * 按钮文字上不混在一起显示：焦段按钮显示相机 zoom（1.0x/2.0x…），
     * 双指缩放的倍率由 OverlayView 自己在取景框上标出来（"数字 2.3× · 框外不识别"）。
     * 这样用户一眼就能分清"我拉近的是镜头，还是只是裁了一块"。
     */
    private fun updateInspectButton() {
        val svc = service
        val on = svc?.isDualInspect() ?: true
        binding.btnInspect.setText(
            when {
                !on -> R.string.btn_inspect_off
                svc?.isConcurrentInspection() == true -> R.string.btn_inspect_dual
                else -> R.string.btn_inspect_single
            }
        )
    }

    // ------------------------------------------------------------ 选区

    private fun toggleRoiEdit() {
        val overlay = binding.overlayView
        val entering = !overlay.editMode
        // 暗屏下取点看不清画面，先恢复亮度
        if (entering && dimmed) setDimmed(false)
        overlay.editMode = entering
        if (!entering && overlay.finishDraft()) {
            Toast.makeText(this, R.string.hint_roi_added, Toast.LENGTH_SHORT).show()
        }
        updateRoiButton()
    }

    private fun updateRoiButton() {
        val overlay = binding.overlayView
        val count = overlay.rois.size
        binding.btnRoi.text = when {
            overlay.editMode -> getString(R.string.btn_roi_done)
            count > 0 -> getString(R.string.btn_roi_count_fmt, count)
            else -> getString(R.string.btn_roi)
        }
        binding.btnRoiUndo.setText(
            if (overlay.editMode) R.string.btn_roi_undo else R.string.btn_roi_clear
        )
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
        updateSensitivityButton(cachedSensitivity)
        updateZoomButton(svc.currentZoom())
        updateLensButton()
        updateRoiButton()
        updateInspectButton()
        // 数字裁切是持久化的，重建界面时把倍率和取景框一起还原
        binding.overlayView.setCropScale(svc.currentCropScale())
        binding.btnTone.text = getString(R.string.btn_tone_fmt, svc.currentTone().displayName)
        binding.btnPause.setText(if (svc.isPaused()) R.string.btn_resume else R.string.btn_pause)
        applyStatus(svc.currentStatus())
    }

    private fun applyStatus(status: DetectionService.Status? = service?.currentStatus()) {
        val current = status ?: return
        // 并发双摄是否成功要等相机绑完才知道，状态每次变都可能变，顺手刷一下
        updateInspectButton()
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
