package com.presenceguard.app.detect

import android.graphics.RectF
import com.presenceguard.app.core.DetectionConfig
import com.presenceguard.app.core.DetectionSource
import com.presenceguard.app.core.PersonCandidate
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 双引擎融合 + 时序状态机。这是整套方案里决定「准不准」的部分。
 *
 * 三级过滤，逐级消除不同类型的错误：
 *
 * 1) 空间融合（消除单帧误报）
 *    姿态结果与检测结果做 IoU / 包含关系匹配。两个引擎在同一位置都报，才采信为
 *    「融合目标」，分数按概率或合成（1-(1-a)(1-b)）；只有一个引擎报时，必须额外
 *    越过更高的单引擎门槛（poseOnlyMinScore / detectorOnlyMinScore）。这样衣架被
 *    EfficientDet 误判成人时，因为姿态引擎不出骨架，分数过不了门，直接丢弃。
 *
 * 2) 面积门限（消除远处噪点）
 *    框面积占画面比例低于 minAreaRatio 的候选丢掉，避免把远处的挂钟、花瓶、墙角
 *    阴影放大成「人」。
 *
 * 3) 滑动窗口投票 + 迟滞（消除瞬时抖动）
 *    单帧命中不算数。最近 windowSize 帧里至少 votesToEnter 帧命中才翻成「有人」；
 *    退出更苛刻，需要连续 votesToExit 帧无目标且平滑分低于 exitScore。进入门槛高于
 *    退出门槛构成迟滞（hysteresis），人站在阈值边缘时不会反复报警。
 */
class PresenceFusionEngine {

    enum class Event { NONE, APPEARED, COUNT_INCREASED, CLEARED }

    data class Outcome(
        /** 状态机判定：画面里是否有人 */
        val present: Boolean,
        /** 本帧识别到的人数（已融合去重） */
        val personCount: Int,
        /** 本帧最高分 */
        val frameScore: Float,
        /** 平滑后的分数，UI 上用它展示「把握程度」 */
        val smoothedScore: Float,
        /** 窗口命中率 0~1 */
        val hitRate: Float,
        /** 窗口内命中帧数 */
        val votes: Int,
        val event: Event,
        /** 本次是否需要响提示音（已应用冷却） */
        val shouldAlert: Boolean,
        val candidates: List<PersonCandidate>
    )

    private val window = ArrayDeque<Boolean>()
    private var smoothed = 0f
    private var present = false
    private var lastCount = 0
    private var emptyStreak = 0
    private var lastAlertAt = 0L

    /** 小目标的短期轨迹记忆，用于跨帧确认 */
    private val smallTracks = ArrayList<SmallTrack>()

    fun reset() {
        window.clear()
        smoothed = 0f
        present = false
        lastCount = 0
        emptyStreak = 0
        smallTracks.clear()
    }

    fun update(
        poseCandidates: List<PersonCandidate>,
        detectorCandidates: List<PersonCandidate>,
        sourceWidth: Int,
        sourceHeight: Int,
        now: Long,
        cfg: DetectionConfig
    ): Outcome {
        val frameArea = (sourceWidth * sourceHeight).toFloat().coerceAtLeast(1f)
        val fused = fuse(poseCandidates, detectorCandidates, sourceWidth, sourceHeight, frameArea, now, cfg)

        val frameScore = fused.maxOfOrNull { it.score } ?: 0f
        val hit = frameScore >= cfg.enterScore

        window.addLast(hit)
        while (window.size > cfg.windowSize) window.removeFirst()

        val votes = window.count { it }
        val hitRate = votes.toFloat() / window.size
        emptyStreak = if (hit) 0 else emptyStreak + 1

        // EMA 平滑：单帧分数突然拉高不会立刻触发，必须持续有证据
        smoothed = if (smoothed <= 0f && frameScore <= 0f) 0f else smoothed * 0.62f + frameScore * 0.38f

        var event = Event.NONE
        if (!present) {
            val enoughVotes = votes >= cfg.votesToEnter
            val enoughScore = smoothed >= cfg.enterScore
            if (enoughVotes && enoughScore) {
                present = true
                lastCount = fused.size
                event = Event.APPEARED
            }
        } else {
            if (emptyStreak >= cfg.votesToExit && smoothed < cfg.exitScore) {
                present = false
                lastCount = 0
                event = Event.CLEARED
            } else if (cfg.alertOnCountIncrease && fused.size > lastCount && fused.isNotEmpty()) {
                lastCount = fused.size
                event = Event.COUNT_INCREASED
            }
        }

        var shouldAlert = false
        if (event == Event.APPEARED || event == Event.COUNT_INCREASED) {
            if (now - lastAlertAt >= cfg.cooldownMs) {
                shouldAlert = true
                lastAlertAt = now
            }
        }

        val displayCount = if (present) max(1, fused.size) else fused.size

        return Outcome(
            present = present,
            personCount = displayCount,
            frameScore = frameScore,
            smoothedScore = smoothed,
            hitRate = hitRate,
            votes = votes,
            event = event,
            shouldAlert = shouldAlert,
            candidates = fused
        )
    }

    // ---------------------------------------------------------------- 融合

    private fun fuse(
        pose: List<PersonCandidate>,
        detector: List<PersonCandidate>,
        sourceWidth: Int,
        sourceHeight: Int,
        frameArea: Float,
        now: Long,
        cfg: DetectionConfig
    ): List<PersonCandidate> {
        val used = BooleanArray(detector.size)
        val merged = ArrayList<PersonCandidate>(pose.size + detector.size)

        for (p in pose) {
            val pArea = p.area() / frameArea
            if (pArea < cfg.minAreaRatio) continue

            var bestIndex = -1
            var bestIoU = 0f
            for (i in detector.indices) {
                if (used[i]) continue
                val value = iou(p.box, detector[i].box)
                if (value > bestIoU) {
                    bestIoU = value
                    bestIndex = i
                }
            }

            val matched = bestIndex >= 0 && (
                bestIoU >= MIN_IOU_FOR_MATCH ||
                    contains(detector[bestIndex].box, p.box) ||
                    contains(p.box, detector[bestIndex].box)
                )

            if (matched) {
                used[bestIndex] = true
                val d = detector[bestIndex]
                val dArea = d.area() / frameArea
                if (dArea < cfg.minAreaRatio * 0.5f) continue

                // 两个引擎互相印证 —— 可信度按「至少一个为真」的概率合成。
                // 检测引擎分数略打折，因为它单独更容易误报。
                val combined = 1f - (1f - p.score) * (1f - d.score * 0.90f)

                // 显示框用目标检测的回归框（更贴合人体外轮廓），
                // 但若姿态框明显更大（举手、张开腿），取并集避免截断
                val box = if (p.area() > d.area() * 1.6f) union(p.box, d.box) else RectF(d.box)

                merged += PersonCandidate(
                    box = box,
                    score = combined.coerceIn(0f, 1f),
                    landmarks = p.landmarks,
                    source = DetectionSource.FUSED,
                    evidence = p.evidence.copy(
                        detectorScore = d.score,
                        crossEngineAgree = true,
                        areaRatio = max(pArea, dArea)
                    )
                )
            } else if (
                p.score >= cfg.poseOnlyMinScore ||
                (pArea < cfg.smallTargetAreaRatio && p.score >= cfg.smallTargetMinScore)
            ) {
                // 只有骨架证据，没有外观证据：门槛更高才采信。
                // 远处的小目标放宽到 smallTargetMinScore，由后面
                // 的轨迹持久化补上可信度。
                merged += p.copy(
                    evidence = p.evidence.copy(areaRatio = pArea)
                )
            }
        }

        for (i in detector.indices) {
            if (used[i]) continue
            val d = detector[i]
            val areaRatio = d.area() / frameArea
            if (areaRatio < cfg.minAreaRatio) continue

            // 目标检测单独命中：按目标大小折扣。远处小框即使分数高也可能是杂物。
            val adjusted = d.score * sizeFactor(areaRatio)
            val isSmall = areaRatio < cfg.smallTargetAreaRatio

            // 小目标不再走「打折后比高门槛」这条路 —— 离得越远原始分越低，
            // 双重折扣会让它永远过不了门。改成低门槛收下，交给轨迹持久化确认。
            val accepted = if (isSmall) {
                d.score >= cfg.smallTargetMinScore
            } else {
                adjusted >= cfg.detectorOnlyMinScore
            }
            if (!accepted) continue

            merged += d.copy(
                score = if (isSmall) d.score else adjusted,
                evidence = d.evidence.copy(areaRatio = areaRatio)
            )
        }

        val kept = nonMaximumSuppression(
            merged
                .map { it.copy(box = clampToFrame(it.box, sourceWidth, sourceHeight)) }
                .filter { candidate ->
                    val ratio = candidate.area() / frameArea
                    if (ratio > 1f) return@filter false
                    // 单个引擎给出几乎覆盖全画面的框，多半是纯色墙面 / 桌面壁纸被当成了人；
                    // 但两个引擎都同意时，那就是镜头前真实的人贴着拍，不能丢。
                    candidate.source == DetectionSource.FUSED || ratio <= MAX_SINGLE_ENGINE_AREA_RATIO
                }
        )

        return gateSmallTargets(kept, frameArea, now, cfg)
    }

    /**
     * 小目标轨迹持久化。
     *
     * 远处的人单帧置信度天然偏低，用「门槛 + 单帧分数」几乎无法与杂物区分。
     * 但两者在时间维度上差别很大：真人会连续出现在同一位置，而墙上的图案、
     * 衣架这类误检往往是偶发、位置飘忽的。所以对小目标改判据——
     * 单帧只需越过较低的门槛，但必须在同一位置连续命中若干帧。
     */
    private fun gateSmallTargets(
        candidates: List<PersonCandidate>,
        frameArea: Float,
        now: Long,
        cfg: DetectionConfig
    ): List<PersonCandidate> {
        smallTracks.removeAll { now - it.lastSeen > SMALL_TRACK_TTL_MS }
        if (candidates.isEmpty()) return candidates

        val passed = ArrayList<PersonCandidate>(candidates.size)
        for (candidate in candidates) {
            val ratio = candidate.area() / frameArea
            if (ratio >= cfg.smallTargetAreaRatio) {
                passed += candidate
                continue
            }

            val track = smallTracks.firstOrNull { iou(it.box, candidate.box) >= SMALL_TRACK_IOU }
            if (track == null) {
                // 首次出现先记账不采信，等后续帧来印证
                smallTracks += SmallTrack(RectF(candidate.box), 1, now)
                continue
            }

            track.hits++
            track.box.set(candidate.box)
            track.lastSeen = now
            if (track.hits >= cfg.smallTargetMinHits) {
                // 确认是小目标后给一个分数下限，让它能正常推动状态机
                passed += candidate.copy(score = max(candidate.score, cfg.smallTargetScoreFloor))
            }
        }
        return passed
    }

    /** 一条小目标轨迹：位置 + 连续命中次数 + 最后出现时间 */
    private class SmallTrack(var box: RectF, var hits: Int, var lastSeen: Long)

    /** 检测框可能超出画面边界（模型回归不受约束），先裁齐再算面积占比 */
    private fun clampToFrame(box: RectF, width: Int, height: Int): RectF {
        val w = width.toFloat()
        val h = height.toFloat()
        return RectF(
            box.left.coerceIn(0f, w),
            box.top.coerceIn(0f, h),
            box.right.coerceIn(0f, w),
            box.bottom.coerceIn(0f, h)
        )
    }

    /**
     * 超出基准占比后的折扣。基准从 4% 下调到 3%、下限从 0.55 提到 0.72：
     * 原来的曲线对小目标太狠，叠加上 detectorOnlyMinScore 之后，
     * 远处的人算出来永远过不了门。
     */
    private fun sizeFactor(areaRatio: Float): Float {
        val t = (areaRatio / 0.03f).coerceIn(0f, 1f)
        return 0.72f + 0.28f * sqrt(t)
    }

    private fun nonMaximumSuppression(input: List<PersonCandidate>): List<PersonCandidate> {
        if (input.size <= 1) return input
        val sorted = input.sortedByDescending { it.score }
        val kept = ArrayList<PersonCandidate>(sorted.size)
        for (candidate in sorted) {
            if (kept.none { iou(it.box, candidate.box) > NMS_IOU }) kept += candidate
        }
        return kept
    }

    // ---------------------------------------------------------------- 几何

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        if (right <= left || bottom <= top) return 0f
        val inter = (right - left) * (bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    /** inner 的中心点落在 outer 内部（容忍一定外扩），用于处理「骨架框 ≪ 检测框」的情况 */
    private fun contains(outer: RectF, inner: RectF): Boolean {
        val cx = inner.centerX()
        val cy = inner.centerY()
        val margin = outer.width() * 0.12f
        return cx >= outer.left - margin && cx <= outer.right + margin &&
            cy >= outer.top - margin && cy <= outer.bottom + margin
    }

    private fun union(a: RectF, b: RectF): RectF {
        val out = RectF(a)
        out.union(b)
        return out
    }

    companion object {
        private const val MIN_IOU_FOR_MATCH = 0.22f
        private const val NMS_IOU = 0.60f

        /**
         * 单引擎目标框的面积占比上限。EfficientDet 在纯色墙面、桌面壁纸这类无纹理
         * 画面上，经常吐出一个覆盖几乎整幅图的人形框；真实人体单靠一个引擎难以
         * 与这种误报区分，所以单引擎超过这个比例就丢弃。
         * 双引擎一致的融合结果不受此限 —— 镜头前贴着拍的人本来就该占满画面。
         */
        private const val MAX_SINGLE_ENGINE_AREA_RATIO = 0.85f

        /** 小目标轨迹的匹配阈值与存活时间：松一点，容忍远处目标的抖动 */
        private const val SMALL_TRACK_IOU = 0.20f
        private const val SMALL_TRACK_TTL_MS = 900L
    }
}
