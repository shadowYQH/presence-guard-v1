package com.presenceguard.app.detect

import android.graphics.RectF
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
import com.presenceguard.app.core.DetectionSource
import com.presenceguard.app.core.Evidence
import com.presenceguard.app.core.PersonCandidate
import com.presenceguard.app.core.PosePoint
import java.util.concurrent.Executor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * 主引擎：ML Kit 姿态检测（accurate / bundled 版）。
 *
 * 选它作为「人是否在场」的主判据，是因为它判断的是**人体骨架**而不是脸：
 * 背对镜头、侧身、戴帽子口罩、蹲在桌下，只要有可见的躯干或四肢轮廓，
 * 都能给出 33 个关键点。这正好对应「不露脸也能识别」的要求。
 *
 * bundled 版本把模型打进 APK（assets/mlkit_pose），不依赖 Google Play 服务，
 * 国产机型、无 GMS 设备也能离线跑。
 */
class PoseEngine {

    private val detector: PoseDetector

    init {
        // 分两步写，避开 Builder 泛型桥接方法在 Kotlin 里的返回类型推断问题
        val builder = AccuratePoseDetectorOptions.Builder()
        builder.setDetectorMode(PoseDetectorOptionsBase.STREAM_MODE)
        // 显式锁定 CPU 推理。不指定的话 ML Kit 会 fork 一个
        // ":mlkit_acceleration_mini_benchmark" 子进程跑加速器基准测试，
        // 该子进程在部分机型上会因 MediaPipe 的 JNI 字段查找失败而 SIGABRT，
        // 连带主进程抛 DuplicateTaskCompletionException 闪退。
        builder.setPreferredHardwareConfigs(PoseDetectorOptionsBase.CPU)
        detector = PoseDetection.getClient(builder.build())
    }

    /**
     * 提交一帧做推理。
     *
     * 注意 [PoseDetector.process] 在这个版本上的泛型签名与文档不一致（签名写的是单个
     * [Pose]，实际多人场景返回的是 `List<Pose>`）。这里用 [normalize] 同时兼容两种
     * 形态，避免被库的签名坑到。
     *
     * [onComplete] 一定会被调用，调用方靠它做汇合。
     */
    fun detect(
        image: InputImage,
        executor: Executor,
        onResult: (List<Pose>) -> Unit,
        onFailure: (Throwable) -> Unit,
        onComplete: () -> Unit
    ) {
        detector.process(image).addOnCompleteListener(executor) { task ->
            if (task.isSuccessful) {
                val raw: Any? = task.result
                onResult(normalize(raw))
            } else {
                onFailure(task.exception ?: IllegalStateException("pose inference failed"))
            }
            onComplete()
        }
    }

    private fun normalize(raw: Any?): List<Pose> = when (raw) {
        null -> emptyList()
        is List<*> -> raw.filterIsInstance<Pose>()
        is Pose -> listOf(raw)
        else -> emptyList()
    }

    fun close() {
        runCatching { detector.close() }
    }
}

/**
 * 把 ML Kit 的原始 [Pose] 翻译成带可信度的候选目标。
 *
 * ML Kit 的 `Pose` 只提供关键点，没有包围盒，所以这里自己算；同时按关键点
 * 部位给不同权重做加权打分 —— 躯干（双肩双髋）是最强的「这是一个直立人体」证据，
 * 而单独一只手腕、一只脚踝完全可能是误匹配。
 */
object PoseScorer {

    /** 部位权重：躯干 > 头部 > 上臂大腿 > 末端肢体 */
    private val WEIGHT: Map<Int, Float> = mapOf(
        PoseLandmark.LEFT_SHOULDER to 1.0f,
        PoseLandmark.RIGHT_SHOULDER to 1.0f,
        PoseLandmark.LEFT_HIP to 1.0f,
        PoseLandmark.RIGHT_HIP to 1.0f,
        PoseLandmark.NOSE to 0.6f,
        PoseLandmark.LEFT_EYE to 0.4f,
        PoseLandmark.RIGHT_EYE to 0.4f,
        PoseLandmark.LEFT_EAR to 0.4f,
        PoseLandmark.RIGHT_EAR to 0.4f,
        PoseLandmark.LEFT_ELBOW to 0.3f,
        PoseLandmark.RIGHT_ELBOW to 0.3f,
        PoseLandmark.LEFT_WRIST to 0.25f,
        PoseLandmark.RIGHT_WRIST to 0.25f,
        PoseLandmark.LEFT_KNEE to 0.3f,
        PoseLandmark.RIGHT_KNEE to 0.3f,
        PoseLandmark.LEFT_ANKLE to 0.2f,
        PoseLandmark.RIGHT_ANKLE to 0.2f
    )

    private val TORSO = intArrayOf(
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_HIP,
        PoseLandmark.RIGHT_HIP
    )

    /** 加权后达到这个值就算「证据充分」，大约等于躯干全部命中 */
    private const val FULL_EVIDENCE = 5.0f

    fun toCandidates(
        poses: List<Pose>,
        keypointThreshold: Float,
        frameHeight: Int
    ): List<PersonCandidate> {
        // 肩宽门限随画面高度自适应：固定 14px 会把 15 米外的人直接判死，
        // 而 480p 与 720p 下「多少像素算太小」本来就该不一样。
        val minShoulderWidth = max(MIN_SHOULDER_WIDTH_PX, frameHeight * MIN_SHOULDER_HEIGHT_RATIO)
        return poses.mapNotNull { pose -> scoreOne(pose, keypointThreshold, minShoulderWidth) }
    }

    private fun scoreOne(pose: Pose, threshold: Float, minShoulderWidth: Float): PersonCandidate? {
        val all = pose.allPoseLandmarks
        if (all.isEmpty()) return null

        val points = ArrayList<PosePoint>(all.size)
        var weightSum = 0f
        var likelihoodSum = 0f
        var torsoHits = 0
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        var counted = 0

        for (landmark in all) {
            val type = landmark.landmarkType
            val w = WEIGHT[type] ?: continue
            val likelihood = landmark.inFrameLikelihood
            if (likelihood < threshold) continue

            val p = landmark.position
            points += PosePoint(p.x, p.y, likelihood, type)

            weightSum += w
            likelihoodSum += likelihood
            counted++
            if (torsoHits < 4 && type in TORSO) torsoHits++

            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minY = min(minY, p.y); maxY = max(maxY, p.y)
        }

        if (counted == 0) return null

        val avgLikelihood = likelihoodSum / counted
        val normalized = min(1f, weightSum / FULL_EVIDENCE)
        var score = 0.70f * normalized + 0.30f * avgLikelihood

        // 完全没有躯干证据时大幅降权：可能只是手臂挥进画面，或模型把杂物认成了肢体
        if (torsoHits == 0) score *= 0.55f

        val box = RectF(minX, minY, maxX, maxY)
        if (box.width() <= 0f || box.height() <= 0f) return null

        // 关键点包围盒天然比真实人体小（头顶、脚底、外侧手脚常超出），向外放一点
        val padX = max(6f, box.width() * 0.10f)
        val padY = max(6f, box.height() * 0.10f)
        box.inset(-padX, -padY)

        // 几何合理性校验：模型偶尔会在插座、玩偶、立牌这类「有躯干轮廓」的杂物上
        // 也吐出一整套关键点，而且置信度很高。真实人体的骨架比例有硬约束，
        // 用这些约束把明显不是人的结果压下去。
        val plausibility = geometricPlausibility(points, minShoulderWidth)
        val finalScore = (score * plausibility).coerceIn(0f, 1f)

        return PersonCandidate(
            box = box,
            score = finalScore,
            landmarks = points,
            source = DetectionSource.POSE,
            evidence = Evidence(
                keypointCount = counted,
                avgKeypointScore = avgLikelihood,
                torsoHits = torsoHits,
                poseScore = finalScore,
                plausibility = plausibility
            )
        )
    }

    /**
     * 人体比例合理性，返回 0~1 的系数乘到姿态分上。
     *
     * 只使用**与朝向无关**的约束。早期版本用过「双肩要水平」「髋必须在肩下方」，
     * 结果把躺着、侧身、以及手机被横过来拍的人全判成了误检 —— 人躺沙发上、
     * 或镜头本身就是斜的，这两条约束会无条件失效。真正稳定的是比例关系：
     *
     *  1. 躯干长度 / 肩宽 —— 用两点距离算，旋转不改变长度
     *  2. 髋宽 / 肩宽 —— 两条身体横截线的比值，同样与朝向无关
     *  3. 头部离肩中点的距离不能太远
     *
     * 各项都有下限截断（[MIN_PLAUSIBILITY]），避免单条不满足就把真人压到 0。
     */
    private fun geometricPlausibility(points: List<PosePoint>, minShoulderWidth: Float): Float {
        if (points.isEmpty()) return 0f

        val map = HashMap<Int, PosePoint>(points.size)
        for (point in points) map[point.type] = point

        val leftShoulder = map[PoseLandmark.LEFT_SHOULDER]
        val rightShoulder = map[PoseLandmark.RIGHT_SHOULDER]

        // 连双肩都没有，说明只有零散肢体，证据不足
        if (leftShoulder == null || rightShoulder == null) return 0.5f

        val shoulderWidth = distance(leftShoulder, rightShoulder)
        // 肩宽过窄只说明人离得远、关键点精度下降，并不等于误检。
        // 这里只温和降权，把「是不是人」交给跨帧轨迹去判断。
        if (shoulderWidth < minShoulderWidth) return 0.55f

        var factor = 1f

        val midShoulderX = (leftShoulder.x + rightShoulder.x) / 2f
        val midShoulderY = (leftShoulder.y + rightShoulder.y) / 2f

        val leftHip = map[PoseLandmark.LEFT_HIP]
        val rightHip = map[PoseLandmark.RIGHT_HIP]

        if (leftHip != null && rightHip != null) {
            val midHipX = (leftHip.x + rightHip.x) / 2f
            val midHipY = (leftHip.y + rightHip.y) / 2f

            // 约束 1：躯干长度与肩宽的比例。真人约为 1.0~2.5，
            // 超出这个区间说明这堆关键点没有拼成一个合理的人形。
            val torsoLength = hypot(midHipX - midShoulderX, midHipY - midShoulderY)
            val torsoRatio = torsoLength / shoulderWidth
            if (torsoRatio < 0.40f || torsoRatio > 3.5f) factor *= 0.60f

            // 约束 2：髋宽与肩宽的比例
            val hipWidth = distance(leftHip, rightHip)
            val hipShoulderRatio = hipWidth / shoulderWidth
            if (hipShoulderRatio < 0.30f || hipShoulderRatio > 3.0f) factor *= 0.65f
        } else {
            // 只有肩、没有髋：可能是上半身被挡住，也可能是误检，适度降权
            factor *= 0.78f
        }

        // 约束 3：头不该离躯干太远
        val nose = map[PoseLandmark.NOSE]
        if (nose != null) {
            val headGap = hypot(nose.x - midShoulderX, nose.y - midShoulderY) / shoulderWidth
            if (headGap > 2.0f) factor *= 0.65f
        }

        return factor.coerceIn(MIN_PLAUSIBILITY, 1f)
    }

    private fun distance(a: PosePoint, b: PosePoint): Float =
        hypot(a.x - b.x, a.y - b.y)

    private const val MIN_SHOULDER_WIDTH_PX = 9f

    /** 肩宽门限随画面高度缩放：640 高时约 11.5px，相当于 13 米外的人 */
    private const val MIN_SHOULDER_HEIGHT_RATIO = 0.018f

    /** 几何校验的下限。宁可让可疑目标进入后续的跨引擎与投票环节，也不要一票否决 */
    private const val MIN_PLAUSIBILITY = 0.35f
}
