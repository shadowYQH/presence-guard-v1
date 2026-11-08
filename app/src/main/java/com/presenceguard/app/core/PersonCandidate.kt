package com.presenceguard.app.core

import android.graphics.RectF

/**
 * 检测结果的来源引擎。
 *
 * 采用双引擎的原因是两者的失效场景互补：
 *  - [POSE] ML Kit 姿态检测：对「人是否在场」这件事几乎不依赖人脸，背对、侧身、
 *           蹲坐、被桌椅遮挡一半时仍能给出躯干关键点；但它对远距离小目标、只露出
 *           一条腿/一只手的情形容易漏检。
 *  - [DETECTOR] TFLite EfficientDet：整体外观目标检测，远距离小目标更敏感；但它
 *           单帧可能把衣架、海报、雕像误判为人。
 * 两个引擎同时同意时几乎不会误报，因此融合分的权重远高于单引擎。
 */
enum class DetectionSource { POSE, DETECTOR, FUSED }

/** 一个人体关键点。坐标为旋转校正后的图像像素坐标。 */
data class PosePoint(
    val x: Float,
    val y: Float,
    val score: Float,
    val type: Int
)

/** 打分依据，用于界面上解释「为什么判定为有人」，也便于现场调参。 */
data class Evidence(
    /** 有效关键点数量（置信度超过阈值） */
    val keypointCount: Int = 0,
    /** 有效关键点平均置信度 */
    val avgKeypointScore: Float = 0f,
    /** 躯干关键点（双肩 / 双髋）命中数量，人体存在的最强证据 */
    val torsoHits: Int = 0,
    /** 姿态引擎单独给出的得分 */
    val poseScore: Float = 0f,
    /** 几何合理性系数（1 = 完全符合人体比例，越低越可疑） */
    val plausibility: Float = 1f,
    /** 目标检测引擎单独给出的得分 */
    val detectorScore: Float = 0f,
    /** 两个引擎在同一位置互相印证 */
    val crossEngineAgree: Boolean = false,
    /** 目标框面积占画面比例 */
    val areaRatio: Float = 0f
)

/**
 * 一个人体候选目标。[box] 与 [landmarks] 均为旋转校正后图像的像素坐标。
 */
data class PersonCandidate(
    val box: RectF,
    val score: Float,
    val landmarks: List<PosePoint> = emptyList(),
    val source: DetectionSource = DetectionSource.FUSED,
    val evidence: Evidence = Evidence()
) {
    val centerX: Float get() = box.centerX()
    val centerY: Float get() = box.centerY()

    fun area(): Float = if (box.isEmpty) 0f else box.width() * box.height()
}

/** 一帧的完整检测结果，直接交给渲染层使用。 */
data class FrameResult(
    val candidates: List<PersonCandidate>,
    /** 旋转校正后图像尺寸，用于把坐标映射到 View */
    val sourceWidth: Int,
    val sourceHeight: Int,
    /** 是否来自前置摄像头（叠加层需要水平镜像才能与预览对齐） */
    val mirrored: Boolean,
    val processingMs: Long,
    val fps: Float = 0f
)
