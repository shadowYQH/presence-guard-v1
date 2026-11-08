package com.presenceguard.app.core

/**
 * 灵敏度档位。三档只是同一套算法的不同阈值组合，不存在「另一套逻辑」，
 * 这样现场调参时行为可预期。
 *
 * 关于「远处的人」：判定小目标不能再靠单帧高分——离得越远，模型的原始置信度
 * 天然越低，任何「先把分数打折、再比门槛」的做法都会把人越推越远。所以小目标
 * （面积占比低于 [DetectionConfig.smallTargetAreaRatio]）走另一条判定路径：
 * 门槛降到 [DetectionConfig.smallTargetMinScore]，但必须在同一位置连续命中
 * [DetectionConfig.smallTargetMinHits] 帧。用时间证据换空间置信度。
 */
enum class Sensitivity(val displayName: String, val hint: String) {
    HIGH("灵敏", "远处的小个子、只露出半个身子也会报，代价是偶尔误报"),
    MEDIUM("均衡", "室内正常距离的人都能稳定识别，误报很少（推荐）"),
    LOW("稳重", "只在人足够大、特征足够明确时才报，几乎不误报");

    fun toConfig(): DetectionConfig = when (this) {
        HIGH -> DetectionConfig(
            enterScore = 0.44f,
            exitScore = 0.32f,
            votesToEnter = 3,
            votesToExit = 12,
            minAreaRatio = 0.0012f,
            keypointThreshold = 0.40f,
            poseOnlyMinScore = 0.46f,
            detectorOnlyMinScore = 0.42f,
            smallTargetAreaRatio = 0.018f,
            smallTargetMinScore = 0.34f,
            smallTargetMinHits = 2,
            smallTargetScoreFloor = 0.52f
        )
        MEDIUM -> DetectionConfig(
            enterScore = 0.52f,
            exitScore = 0.38f,
            votesToEnter = 4,
            votesToExit = 14,
            minAreaRatio = 0.0030f,
            keypointThreshold = 0.45f,
            poseOnlyMinScore = 0.56f,
            detectorOnlyMinScore = 0.52f,
            smallTargetAreaRatio = 0.015f,
            smallTargetMinScore = 0.42f,
            smallTargetMinHits = 3,
            smallTargetScoreFloor = 0.58f
        )
        LOW -> DetectionConfig(
            enterScore = 0.66f,
            exitScore = 0.50f,
            votesToEnter = 5,
            votesToExit = 16,
            minAreaRatio = 0.010f,
            keypointThreshold = 0.50f,
            poseOnlyMinScore = 0.70f,
            detectorOnlyMinScore = 0.64f,
            smallTargetAreaRatio = 0.012f,
            smallTargetMinScore = 0.55f,
            smallTargetMinHits = 5,
            smallTargetScoreFloor = 0.68f
        )
    }
}

/**
 * 检测参数。默认值即「均衡」档。
 *
 * 判定链路：单帧证据分 -> 双引擎交叉印证 -> 小目标轨迹持久化 -> 滑动窗口投票 -> 迟滞状态机。
 * 多层叠加是为了压住两类错误：把衣架当人（误报）和人走过时不响（漏报）。
 */
data class DetectionConfig(
    /** 单帧分数达到该值即视为「本帧有人」 */
    val enterScore: Float = 0.52f,
    /** 平滑分数低于该值才可能退出；比 [enterScore] 低，构成迟滞，避免边界抖动 */
    val exitScore: Float = 0.38f,
    /** 滑动窗口帧数（分析流约 10~15fps，8 帧≈0.6 秒） */
    val windowSize: Int = 8,
    /** 窗口内至少这么多帧命中才确认有人 */
    val votesToEnter: Int = 4,
    /** 连续这么多帧无目标才解除 */
    val votesToExit: Int = 14,
    /** 目标框面积占画面比例下限，过滤远处噪点与墙角误检 */
    val minAreaRatio: Float = 0.0030f,
    /** 关键点被视为有效的置信度门槛 */
    val keypointThreshold: Float = 0.45f,
    /** 目标检测引擎的原始置信度门槛（送进模型前先粗筛） */
    val detectorScoreThreshold: Float = 0.22f,
    /** 只有姿态引擎命中、没有目标检测印证时，需要更高的分数才采信 */
    val poseOnlyMinScore: Float = 0.56f,
    /** 只有目标检测命中、没有姿态印证时，同样需要更高分数 */
    val detectorOnlyMinScore: Float = 0.52f,

    // ---------------------------------------------------------------- 小目标
    /**
     * 面积占比低于该值视为「小目标」，改用跨帧轨迹判定。
     * 480×640 下 0.015 ≈ 4600 像素，约等于 8 米外人体的占屏比例。
     */
    val smallTargetAreaRatio: Float = 0.015f,
    /** 小目标的单帧原始分下限（比 detectorOnlyMinScore 低得多，靠持续命中兜底） */
    val smallTargetMinScore: Float = 0.42f,
    /** 小目标需要在同一位置连续命中这么多帧才采信 */
    val smallTargetMinHits: Int = 3,
    /** 轨迹确认后给予的分数下限，让它能正常推动状态机 */
    val smallTargetScoreFloor: Float = 0.58f,

    /** 两次报警之间的最短间隔 */
    val cooldownMs: Long = 3500L,
    /** 已在场的人数变多时是否再响一次 */
    val alertOnCountIncrease: Boolean = true
)
