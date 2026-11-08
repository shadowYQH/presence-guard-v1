# 人影守望 · 开发文档

> 版本：v1（双引擎 + 双路合成 + 多边形选区）
> 目标设备：Xiaomi M2007J1SC（骁龙 865 / Adreno 650 / Android 13）
> 文档定位：把一个"为什么这么写"讲清楚，而不只是"这里写了什么"。所有参数、阈值、实测数据都可以直接拿来调。

---

## 1. 项目是什么

一个**纯本地**的人体存在检测器：相机持续看着某个区域，画面里出现人就用声音持续提醒。
不录像、不上传、不联网。典型用途是看店、看仓库、看门口、看有没有人靠近某台设备。

它要同时满足四个互相拉扯的约束：

| 约束 | 具体表现 |
|---|---|
| 不靠脸 | 背对、侧身、戴口罩帽子、蹲着、只露半个身子都要能认出来 |
| 能看见远处的人 | 十几米外只占画面千分之一的人体也要报 |
| 不能误报 | 衣架、海报、雕像、椅子、墙上的影子不能被当成人 |
| 要能跑在手机上 | 长时间运行、息屏也要跑、发热和耗电要可控 |

单靠任何一个模型都做不到。所以整个系统的核心不是"用了什么模型"，而是**怎么把证据组织起来**。

---

## 2. 目录结构

```
presence-guard-v1/
├── app/src/main/
│   ├── assets/
│   │   ├── efficientdet_lite0.tflite      副引擎模型（COCO person）
│   │   ├── probe_far_lake.jpg             验收图：湖边长焦远景
│   │   └── probe_far_plaza.jpg            验收图：车站广场密集小目标
│   ├── java/com/presenceguard/app/
│   │   ├── MainActivity.kt                界面、按钮、选区取点调度、屏幕广播
│   │   ├── audio/
│   │   │   ├── AlertPlayer.kt             提示音播放（PCM 合成 → AudioTrack）
│   │   │   └── AlertTone.kt               音色定义与波形参数
│   │   ├── core/
│   │   │   ├── DetectionConfig.kt         灵敏度三档 → 全部阈值的映射
│   │   │   ├── PersonCandidate.kt         候选目标 / 证据 / 帧结果数据模型
│   │   │   └── Roi.kt                     多边形选区（归一化坐标、内含判定、面积）
│   │   ├── data/
│   │   │   └── Settings.kt                SharedPreferences 持久化
│   │   ├── detect/
│   │   │   ├── FrameAnalyzer.kt           分析器：双路合成 + 选区放大 + 帧调度
│   │   │   ├── PoseEngine.kt              主引擎 ML Kit 姿态 + PoseScorer 打分
│   │   │   ├── PresenceFusionEngine.kt    融合判定 + 时序状态机
│   │   │   └── TflitePersonEngine.kt      副引擎 TFLite + 瓦片/裁切推理
│   │   ├── service/
│   │   │   └── DetectionService.kt        前台服务：相机、引擎、报警、通知、电源
│   │   ├── ui/
│   │   │   └── OverlayView.kt             叠加层：框、骨架、选区绘制与触摸取点
│   │   └── ProbeActivity.kt (debug)       离屏验收探针
│   └── res/                               布局与文案
├── docs/
│   ├── DEVELOPMENT.md                     本文档
│   └── PERFORMANCE.md                     性能实测与加速器结论
└── build.gradle.kts / README.md / .gitignore
```

> `ProbeActivity` 只在 debug 变体里存在（`app/src/debug/`），release 包不会包含任何验收代码。

---

## 3. 整体架构

相机管线跑在**前台服务**里而不是 Activity 里，这是整个设计的第一个前提：

- Activity 一旦不可见、屏幕熄灭，绑定在 Activity 生命周期上的 CameraX 会被解绑，检测直接停；
- 绑到服务自己的 `LifecycleOwner` 上，息屏、切后台都能继续；
- 服务声明 `foregroundServiceType="camera"`，系统才允许后台持有相机；
- 配合 `PARTIAL_WAKE_LOCK`，熄屏后 CPU 不停机，推理和提示音不会被挂起。

数据流：

```
CameraX ImageAnalysis (1080×1440, KEEP_ONLY_LATEST)
        │
        ├─ 转 Bitmap（含旋转校正）
        ├─ 亮度门（纯黑 / 纯白帧直接跳过，不做推理）
        │
        ├──[常规模式]──> 姿态引擎（整帧）        ┐
        │               目标检测引擎（瓦片 + 整帧）├─> PresenceFusionEngine
        │                                        ┘
        └──[选区模式]──> 每个选中选区裁切放大 ──> 姿态 + 检测（都在裁切图上）
                                                 │
                                                 ▼
                                   五层判定 → Outcome
                                                 │
                          ┌──────────────────────┼───────────────────┐
                          ▼                      ▼                   ▼
                    OverlayView            AlertPlayer         通知栏
                   （框/骨架/选区）        （持续报警）        （状态刷新）
```

---

## 4. 相机与预览层

### 4.1 分辨率

请求 1920×1080，规则 `FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER`。
实测该机 `ImageAnalysis` 实际给出 **1080×1440（4:3，旋转后竖向）**——传感器的原生比例，
不是 16:9。这一点很关键：

- 后续所有坐标、面积占比都以这个尺寸为基准；
- 预览用 `FILL_CENTER`（等比放大裁剪填满屏幕），所以屏幕上看到的画面只是分析画面的**一部分**；
- 叠加层必须复现同一套"等比缩放 + 居中偏移"才能让框贴在画面上（见 4.3）。

### 4.2 变焦（`setZoomRatio`）

三个档位按钮循环切换 1.0x → 2.0x → 3.0x → 5.0x → 8.0x，实际按当前镜头的
`minZoomRatio..maxZoomRatio` 过滤（本机 1.0–10.0）。持久化在 `Settings.zoomRatio`。

**为什么用 CameraX 的 zoom 而不是自己裁切画面**：

- CameraX 的 zoom 是**传感器级区域裁切**，部分机型还会自动切到长焦镜头，`ImageAnalysis`
  拿到的就是放大后的真实像素——检测精度跟着提升；
- 自己在 Bitmap 上裁切放大，只是把已经模糊的一小块插值拉大，对识别没有任何帮助，
  还会白白多一次拷贝。

因为 CameraX 的 zoom 对**同一个相机上的所有 use case 同时生效**，预览和分析流始终视野一致，
联合调试时看到的框和画面不会错位。

### 4.3 坐标映射

`OverlayView` 里的正向映射（图像像素 → 屏幕像素）：

```kotlin
val scale   = max(width / srcW, height / srcH)      // FILL_CENTER
val offsetX = (width  - srcW * scale) / 2f
val offsetY = (height - srcH * scale) / 2f
fun mapX(x) = if (mirrored) width - (x * scale + offsetX) else x * scale + offsetX
fun mapY(y) = y * scale + offsetY
```

选区取点用的是它的**逆变换**（屏幕像素 → 归一化 0~1）：

```kotlin
var px = if (mirrored) width - viewX else viewX
val nx = ((px - offsetX) / scale) / srcW
val ny = ((viewY - offsetY) / scale) / srcH
```

`mirrored` 只在**前置摄像头**时为真。后置不镜像，前置镜像，两者都实测对齐。

### 4.4 预览的生命周期与功耗

这是本项目一个容易踩坑的地方，结论先写：

> **`Preview.setSurfaceProvider(null)` 并不能停掉预览的功耗。**
> 只要 Preview 用例还绑在相机上，相机就一直有这么一条输出流在跑，只是没人显示。
> 真正能让它归零的做法是**把 Preview 从绑定的用例列表里去掉**。

实现：`DetectionService.setPreviewVisible(visible)` → 重建绑定，`useCases` 里
视情况是否包含 `Preview`：

```kotlin
val useCases = ArrayList<UseCase>(2)
if (previewVisible) {
    val preview = Preview.Builder().build()
    pendingSurfaceProvider?.let { preview.setSurfaceProvider(it) }
    useCases += preview
}
useCases += analysis
camera = provider.bindToLifecycle(this, selectorFor(lensFacing), *useCases.toTypedArray())
```

触发时机有个坑：**Android 13 / MIUI 上熄屏时 Activity 只走到 `onPause`，`onStop` 迟迟不来**。
实测按住电源键熄屏后，`dumpsys activity` 里 Activity 仍是 `mLastPausedActivity`，
`onStop` 没有触发。所以只靠 `onStop` 摘预览是不生效的。

最终用了三道保险：

1. `onPause()` → `setPreviewVisible(false)`（一定会被调用）；
2. `onResume()` → 重新绑定；
3. `ACTION_SCREEN_OFF` / `ACTION_SCREEN_ON` 广播兜底（`ContextCompat.registerReceiver`，
   `RECEIVER_NOT_EXPORTED`）。这个广播实测是稳定到达的。

实测数据（`top -o PID,%CPU`，单位是一个核心的百分比）：

| 状态 | 进程 CPU |
|---|---|
| 检测 + 预览（选区模式，8fps） | 106–130% |
| 仅预览（暂停检测） | 13–17% |
| 熄屏 + 暂停 | **0.0%** |

也就是说预览本身约占 15% 的单核开销，熄屏后这部分被完整移除，而检测不受影响。

### 4.5 多镜头

后置可切换的物理镜头通过公开 API 反查，不读 Camera2 私有字段：

```kotlin
provider.availableCameraInfos.filter { info ->
    CameraSelector.DEFAULT_BACK_CAMERA.filter(listOf(info)).isNotEmpty()
}
```

选中某一颗时用 `CameraSelector.Builder.addCameraFilter { infos -> infos.filter { it == target } }`
把候选裁到只剩目标。**本机实测只暴露 1 颗后置相机（日志 `lens=0/1`）**，
所以界面上"镜头"按钮自动隐藏（`lensCount() <= 1` 时 `GONE`），这时候靠 zoom 就够了。
多摄机型上按钮会自动出现。

---

## 5. 检测引擎层

### 5.1 主引擎：ML Kit 姿态（为什么用它当主判据）

`PoseEngine` 用 `AccuratePoseDetectorOptions` + `STREAM_MODE` + **bundled** 模型：

- bundled 版把模型打进 APK，不依赖 Google Play 服务，国产无 GMS 机型也能离线跑；
- 判断的是**人体骨架**而不是脸——背对、侧身、口罩、蹲坐、被桌椅挡一半，只要躯干或四肢可见就能给点；
- 显式锁定 `CPU` 推理。不指定的话 ML Kit 会 fork 一个
  `:mlkit_acceleration_mini_benchmark` 子进程跑加速基准，该子进程在部分机型上会因
  MediaPipe 的 JNI 字段查找失败而 SIGABRT，连带主进程抛 `DuplicateTaskCompletionException` 闪退。

`PoseEngine.detect()` 有个库层面的坑：`PoseDetector.process()` 的泛型签名写的是返回单个
`Pose`，但多人场景实际返回 `List<Pose>`。代码里用 `normalize()` 同时兼容两种形态。

### 5.2 PoseScorer：从关键点到"这是个人"

ML Kit 只给关键点，没有包围盒。`PoseScorer` 负责把它翻译成一个带分数的候选目标：

**打分**：按部位加权，躯干（双肩双髋）权重最高，末端肢体最低。

```
normalized = min(1, Σ权重 / 5.0)
score      = 0.70 * normalized + 0.30 * 平均关键点置信度
若无任何躯干关键点：score *= 0.55
```

**几何合理性校验**（`geometricPlausibility`）——这一步专治"模型在插座、玩偶、立牌、
抱枕上吐出一整套高置信度关键点"。

早期版本用过"双肩要水平""髋必须在肩下方"这两条约束，结果是**把现实中的人判成了误检**：
人躺沙发上、侧身站着、或者手机本身被横过来拍，这两条无条件失效。

现在只用**与朝向无关**的比例关系（用两点距离，旋转不改变长度）：

| 约束 | 判据 | 不满足时 |
|---|---|---|
| 躯干长 / 肩宽 | 0.40 ~ 3.5 | ×0.60 |
| 髋宽 / 肩宽 | 0.30 ~ 3.0 | ×0.65 |
| 鼻到肩中点距离 / 肩宽 | ≤ 2.0 | ×0.65 |
| 只有肩没有髋 | — | ×0.78 |
| 连双肩都没有 | — | 直接返回 0.50 |
| 肩宽 < 门限 | — | 返回 0.55（温和降权，不否决） |

系数下限 `MIN_PLAUSIBILITY = 0.35`：**宁可让可疑目标进入后面的跨引擎印证和投票环节，
也不要在这里一票否决**。单点判死是漏报的主要来源。

肩宽门限随画面高度自适应：`max(9px, frameHeight * 0.018)`。
固定像素值会让 480p 和 720p 下的判定标准完全不一样。

### 5.3 副引擎：TFLite EfficientDet-Lite0

COCO 数据集，取 `person` 类（标签为空时退回 `category.index == 0`）。
输入固定 320×320，`MAX_RESULTS = 10`，`NUM_THREADS = 4`。

加速器按 `Accel.GPU → CPU` 依次尝试，失败自动降级，实际生效的记在 `accelerator` 里供日志显示。
（GPU 实测并不比 CPU 快，详见 `docs/PERFORMANCE.md`——但仍然保留这条路，
因为不同机型/驱动差异很大。）

这里有三个必须知道的事实：

**（1）提高采集分辨率没用。**
模型输入固定 320×320，整帧缩放后远处的人只剩十几像素，特征彻底消失。
把采集从 720p 提到 1080p，缩放会把这个增益完全抵消掉。**唯一有效的手段是缩小送进模型的视野。**

**（2）`BitmapMlImage.close()` 会回收底层 Bitmap。**
`infer()` 里 `BitmapMlImageBuilder(bitmap).build()` 之后 `image.close()`，
传进去的那张 Bitmap 就被 recycle 了。所以：

- 必须先把所有栽切图裁出来，再做任何整帧推理；
- 否则第二步会拿到 `recycled bitmap`，抛 `IllegalArgumentException: cannot use a recycled source`；
- 选区模式下用同一张裁切图先跑姿态（`InputImage.fromBitmap` 不接管所有权）再跑检测，
  顺序反了就会踩这个坑。

**（3）int8 / float16 模型加载不了。**
`tensorflow-lite-task-vision:0.4.4` 的 JNI 会抛 `UnsatisfiedLinkError`（被
`TaskJniUtils.createHandleFromLibrary` 包了一层）。MediaPipe bucket 和 TF Hub 的量化模型都试过，
文件头是合法的 TFL3，就是加载失败；同源的 float32 模型正常。

### 5.4 瓦片扫描（常规模式下的"长焦"）

`TileGrid` 把画面切成 `cols × rows` 的瓦片（默认 6×4 = 24 片，相邻重叠 25%），
每帧只滚 `tilesPerFrame` 片，游标轮转，若干帧覆盖全图。

放大倍数只取决于瓦片数量，与采集分辨率无关：

```
放大倍数 ≈ sqrt(总瓦片数)   // 24 片 ≈ 每片覆盖 1/5 的边长 → 约 4.9 倍
```

为什么每帧只扫几片：单片推理约 70–90ms，24 片全扫就是 2 秒一帧，实时性没了。
滚动扫描用"时间换空间"——单帧只覆盖局部，靠跨帧轨迹把小目标的证据攒起来。

瓦片路径**只回传小目标**（面积占整帧 ≤ 12%），大目标交给整帧那一路，否则同一个人会被
多个瓦片各框一次，徒增重复。

还有一个刻意的优化：**画面里已经有人时不做瓦片扫描**。报警已经触发，瓦片的使命是"找人"，
稳态下省掉这几次推理，帧率明显提升。空场景才是需要全力搜索的时候。

### 5.5 选区放大识别（选区模式）

见第 7 章。

---

## 6. 融合判定层（决定准不准的部分）

`PresenceFusionEngine` 五层过滤，逐层消除不同类型的错误。

### 第 1 层：空间融合

姿态框与检测框做 IoU / 包含关系匹配（`MIN_IOU_FOR_MATCH = 0.22`，比较松——
两个引擎的框风格差异很大，姿态框常常比检测框小一圈）。

- **匹配上** → `FUSED`，分数按"至少一个为真"的概率合成：
  `combined = 1 - (1 - poseScore) * (1 - detectorScore * 0.90)`
  （检测分打 9 折，因为它单独更容易误报）。显示框取检测的回归框，若姿态框明显更大则取并集。
- **只有骨架** → 门槛更高：`poseScore >= poseOnlyMinScore`；
  若面积小于 `smallTargetAreaRatio`，放宽到 `smallTargetMinScore`，交给第 4 层补可信度。
- **只有检测框** → 先按面积折扣 `sizeFactor`，再和 `detectorOnlyMinScore` 比；
  小目标走另一条路（见第 4 层），**不做二次折扣**。

`sizeFactor` 曲线（`0.72 + 0.28 * sqrt(areaRatio / 0.03)`，上限 1.0）调过两轮：
第一版下限 0.55 对小目标太狠，叠加上 `detectorOnlyMinScore` 之后远处的人算出来永远过不了门。

### 第 2 层：面积门限

`minAreaRatio = 0.0030`（中等灵敏度）。低于此比例的候选丢掉，用来过滤远处的挂钟、
花瓶、墙角阴影。**这一层在选区模式下会被放开到 0.0002**，原因见 7.4。

另有一条"反向"限制：**单引擎给出几乎覆盖全画面的框**（> 85%）直接丢弃——
EfficientDet 在纯色墙面、桌面壁纸这类无纹理画面上经常吐出巨型人形框。
但两个引擎都同意时不受此限（`FUSED` 豁免）：镜头前贴着拍的人本来就该占满画面。
框出界时先 `clampToFrame()` 裁齐再算面积，否则面积占比会算出 > 1 的荒谬值。

### 第 3 层：NMS

`NMS_IOU = 0.60`，按分数降序保留。

### 第 4 层：小目标轨迹持久化

这是提升"远处的人"召回最关键的一层。

远处的人单帧置信度天然偏低，用"门槛 + 单帧分数"几乎无法与杂物区分。但两者在**时间维度**上
差别很大：真人会连续出现在同一位置，而墙上的图案、衣架这类误检往往偶发、位置飘忽。

所以对小目标改判据：

```
首次出现       → 记账（hits = 1），不采信
同位置再次命中 → hits++，位置更新
hits >= smallTargetMinHits → 采信，并给一个分数下限 smallTargetScoreFloor
```

轨迹匹配用 `IoU >= 0.20`（很松，容忍远处目标的抖动），存活期 `900ms`。
用**时间证据换空间置信度**，这是唯一能在不提高误报的前提下把远处的人捞出来的办法。

### 第 5 层：滑动窗口投票 + 迟滞

```
window:          最近 8 帧是否命中
进入条件:        votes >= votesToEnter(4) 且 smoothed >= enterScore
退出条件:        连续 votesToExit(14) 帧无目标 且 smoothed < exitScore
smoothed:        EMA，新值权重 0.38
```

进入门槛高于退出门槛构成**迟滞**（hysteresis），人站在阈值边缘时不会反复报警。
退出比进入苛刻得多（14 帧 vs 4 帧），因为"漏报一瞬间就解除"比"多等一秒"糟糕得多。

---

## 7. 多边形选区

### 7.1 为什么是归一化坐标 + 多边形

坐标存的是**归一化值（0~1，相对分析画面）**，不是像素。这样切换分辨率、切换摄像头、
甚至改采集尺寸之后，选区依然落在同一个物理位置上，不需要跟着重算。

用多边形而不是矩形，是因为实际场景里要盯的区域往往不是方的：走廊是长条、
门口是斜的、阳台栏杆是细长一条。矩形会把大片无关背景卷进来，多边形能贴着边界框。

`Roi` 提供的能力：

| 方法 | 用途 |
|---|---|
| `bounds()` / `boundsPx(w,h)` | 外接矩形，用于裁切送检 |
| `contains(x, y)` | 射线法判断点是否在多边形内 |
| `areaRatio()` | 鞋带公式算面积占画面比例 |
| `encode()` / `decode()` | 持久化成 `x,y;x,y;x,y` 字符串 |

相关常量：`MIN_POINTS = 3`（三角形是最少的多边形）、
`MIN_VERTEX_GAP = 0.012`（相邻顶点过近视为误触）、
`MIN_AREA_RATIO = 0.0002`（更小的选区放大后全是噪点，送检没意义）。

### 7.2 取点交互

在 `OverlayView` 里实现，`editMode = true` 时吃掉全部触摸事件：

1. 点按画面 → 加一个顶点（屏幕坐标经 4.3 的逆变换转成归一化坐标）；
2. 离上一个顶点太近 → 忽略（防手抖把多边形画成一堆重合点）；
3. 顶点数 ≥ 3 后，**点回起点**（半径 `CLOSE_GAP = 0.035`）即闭合，收下一个选区；
4. 闭合后可以继续画下一个，**选区数量不设上限**；
5. `撤销` 先删草稿顶点，草稿空了就删最后一个已确认的选区；`清空` 一次性删光。

绘制约定：已确认选区是青色描边 + 半透明填充，左上第一个顶点旁有编号圆标（1、2、3…），
和持久化列表一一对应；正在画的草稿是黄色虚线 + 顶点圆点，首点带一个圈提示"点这里闭合"，
屏幕下方有随状态变化的操作提示。

### 7.3 放大识别：怎么做的

`FrameAnalyzer.scanRegion()`：

```
1. regionOf(roi)          取外接矩形，外扩 6%（最少 1.2%），夹到画面内
                          任一边小于 40px 直接放弃（裁出来全是模糊块）
2. Bitmap.createBitmap    裁出这一块
3. poseOnCrop(crop)       姿态引擎在裁切图上跑 → 关键点/框平移到整帧坐标
4. engine.detectCrop()    检测引擎在裁切图上跑 → 框平移到整帧坐标
5. remap(...)             按多边形过滤：中心点在内 或 与选区的重叠 ≥ 框面积的 45%
6. crop.recycle()
```

**为什么这就等于长了焦距**：模型输入固定 320×320。整帧送进去，一个 30px 高的人被缩到
约 7px，特征消失；把这个选区裁出来送进去，同一个人被放大到 180px。
放大倍数 ≈ `1 / 选区边长占比`，一个占画面 1/8 边长的选区就是 8 倍，比任何数码变焦都真实。

**为什么姿态也要在裁切图上跑**：放大之后关键点置信度和肩宽都上来了，
`PoseScorer` 的几何校验才有意义。同时因为两个引擎都在同一张放大图上，
第 1 层的跨引擎印证才能真正压住"放大后噪点变成人"的风险——实测对一个音箱喇叭特写
做 4 倍放大，两个引擎都没有误报。

**过滤的宽松处理**：只判"框中心是否在多边形内"会漏掉站在选区边缘的人（框中心落到多边形外，
但身体明明压在里面）。所以加了一条退路：框与选区外接矩形的交集 ≥ 框自身面积的 45% 也算。

### 7.4 同时用多个小多边形

**可以，而且这是设计里就支持的。**

多个选区的成本是线性的（每个 ≈ 检测 + 姿态两次推理），所以不能每帧全扫。做法是**游标轮转**：

```kotlin
val take     = min(roiPerFrame, activeRois.size)
val start    = roiCursor % activeRois.size
val selected = (0 until take).map { activeRois[(start + it) % activeRois.size] }
roiCursor    = (start + take) % activeRois.size
```

`roiPerFrame` 的取值：`min(选区数, 2)`，省电模式下压到 1。

以 2 个选区、每次 1 个、8fps 为例：每个选区每 250ms 被扫一次，
配合第 4 层的小目标轨迹（存活 900ms），单帧证据不足的远处目标在 900ms 内能攒到约 3 次命中，
刚好够确认。**轮转 + 跨帧轨迹这两个机制是配套的**，去掉任何一个多选区都不可用。

### 7.5 选区模式下 fusion 门槛为什么要改

用户亲手划的多边形本身就是极强的空间先验。如果还拿"占整帧面积太小"去卡他，
等于把选区这个先验扔掉了。所以 `FrameAnalyzer` 在选区模式下这么调：

```kotlin
val fusionCfg = cfg.copy(
    minAreaRatio = ROI_MIN_AREA_RATIO,   // 0.0030 → 0.0002
    smallTargetAreaRatio = 0f            // 关掉小目标延迟确认
)
```

`smallTargetAreaRatio = 0` 的效果是所有候选都走进"非小目标"分支，即直接要求
`detectorOnlyMinScore` / `poseOnlyMinScore`。配合放大后的高置信度，不需要再攒帧。

**选区模式下整帧检测和瓦片扫描全部跳过。** 用户已经指定了要盯的地方，别的地方不看——
这既是语义上的正确，也把每帧成本从"瓦片 + 整帧"压到"1 个选区"。

---

## 8. 前台服务与息屏

### 8.1 生命周期

服务自己实现 `LifecycleOwner`（`LifecycleRegistry`），`onCreate` 里直接置 `RESUMED`：
"服务活着就认为相机应当工作"。这跟 Activity 的可见性完全解耦。

`onDestroy` 里有个细节：引擎不能立刻 close。

```kotlin
mainHandler.postDelayed({ engine?.close(); detector?.close(); analyzer?.shutdown() }, 600L)
```

在途帧可能还卡在 ML Kit 的回调里，立刻 close 会让它重复完成 Task 而崩溃
（`DuplicateTaskCompletionException`）。

### 8.2 电源

- `PARTIAL_WAKE_LOCK`（`PresenceGuard::detect`）：熄屏后 CPU 不停机；
- 省电模式：`minFrameIntervalMs = 200`（5fps）+ 关闭瓦片扫描。

5fps 对"有没有人"这个判断完全够用，功耗却能降一大截。

### 8.3 通知

`startForeground(id, notification, FOREGROUND_SERVICE_TYPE_CAMERA)`，
`IMPORTANCE_LOW` 通道（不响不震），状态变化时限频刷新（`NOTIFY_REFRESH_MS = 5000`），
避免每秒 notify 一次。

---

## 9. 提示音

不引入音频资源文件，直接**程序合成 PCM**：

- 每个 `AlertTone` 是一组参数（频率、时长、包络、谐波），`AlertPlayer` 用 `AudioTrack`
  `MODE_STATIC` 一次性写入，`play()` 只是 seek(0) + play，延迟极低；
- 走 **ALARM 音频通道**（`STREAM_ALARM` / `AudioAttributes.USAGE_ALARM`）——
  这是唯一一个不会被系统"媒体音量"静音掉、且不需要前台焦点就能出声的通道；
- 静音时给 UI 提示（`isAlarmAudible()` 检查闹钟音量），否则用户会以为是应用坏了；
- **持续报警**：`frameListener` 里判断 `now - lastAlertAt >= settings.alertIntervalMs`
  （默认 1500ms）就再响一次，人离开后立即清零 `lastAlertAt`，下次出现能立刻响。

---

## 10. 参数速查

### 10.1 灵敏度三档

| 参数 | 高（灵敏） | 中（均衡，默认） | 低（稳重） |
|---|---|---|---|
| `enterScore` | 0.44 | 0.52 | 0.66 |
| `exitScore` | 0.32 | 0.38 | 0.50 |
| `votesToEnter` | 3 | 4 | 5 |
| `votesToExit` | 12 | 14 | 16 |
| `minAreaRatio` | 0.0012 | 0.0030 | 0.010 |
| `keypointThreshold` | 0.40 | 0.45 | 0.50 |
| `poseOnlyMinScore` | 0.46 | 0.56 | 0.70 |
| `detectorOnlyMinScore` | 0.42 | 0.52 | 0.64 |
| `smallTargetAreaRatio` | 0.018 | 0.015 | 0.012 |
| `smallTargetMinScore` | 0.34 | 0.42 | 0.55 |
| `smallTargetMinHits` | 2 | 3 | 5 |
| `smallTargetScoreFloor` | 0.52 | 0.58 | 0.68 |
| 每帧瓦片数 | 3 | 2 | 1 |

`windowSize = 8`、`detectorScoreThreshold = 0.22`、`cooldownMs = 3500` 三档一致。

### 10.2 结构性常量

| 位置 | 常量 | 值 | 说明 |
|---|---|---|---|
| FrameAnalyzer | `FRAME_TIMEOUT_MS` | 2000 | 单帧等待上限 |
| FrameAnalyzer | `DEFAULT_TILES_PER_FRAME` | 4 | 服务会按灵敏度覆盖为 1–3 |
| FrameAnalyzer | `MIN_VALID_LUMA` / `MAX_VALID_LUMA` | 10 / 248 | 纯黑/纯白帧跳过推理 |
| FrameAnalyzer | `SAMPLES_PER_SIDE` | 64 | 亮度采样网格 |
| FrameAnalyzer | `REGION_PAD` | 0.06 | 选区外扩比例 |
| FrameAnalyzer | `MIN_REGION_PX` | 40 | 送检裁切最小边长 |
| FrameAnalyzer | `ROI_MIN_AREA_RATIO` | 0.0002 | 选区模式面积下限 |
| FrameAnalyzer | `ROI_OVERLAP_ACCEPT` | 0.45 | 压线人体的接受门槛 |
| FrameAnalyzer | `POSE_TIMEOUT_MS` | 900 | 选区内等姿态的上限 |
| Fusion | `MIN_IOU_FOR_MATCH` | 0.22 | 跨引擎匹配 |
| Fusion | `NMS_IOU` | 0.60 | 去重 |
| Fusion | `MAX_SINGLE_ENGINE_AREA_RATIO` | 0.85 | 单引擎巨型框上限 |
| Fusion | `SMALL_TRACK_IOU` | 0.20 | 小目标轨迹匹配 |
| Fusion | `SMALL_TRACK_TTL_MS` | 900 | 轨迹存活期 |
| Tflite | `TILE_SCORE_DISCOUNT` | 0.95 | 瓦片分数折扣 |
| Tflite | `DEFAULT_TILE_TARGET_MAX_AREA` | 0.12 | 瓦片只回传小目标 |
| PoseScorer | `FULL_EVIDENCE` | 5.0 | 加权满分基准 |
| PoseScorer | `MIN_SHOULDER_HEIGHT_RATIO` | 0.018 | 肩宽门限随高度缩放 |
| PoseScorer | `MIN_PLAUSIBILITY` | 0.35 | 几何校验下限 |
| Roi | `MIN_POINTS` / `MIN_VERTEX_GAP` / `MIN_AREA_RATIO` | 3 / 0.012 / 0.0002 | |
| Service | `ANALYSIS_WIDTH/HEIGHT` | 1920 / 1080 | 实际得到 1080×1440 |
| Service | `DETECTOR_PRE_FILTER` | 0.22 | 检测引擎构造时固化的粗筛 |
| Service | `POWER_SAVE_FRAME_INTERVAL` | 200 | 省电 5fps |
| Settings | `MAX_ZOOM` | 8.0 | 变焦上限 |
| Settings | `DEFAULT_INTERVAL` | 1500 | 报警重复间隔 |

---

## 11. 实测性能

设备：骁龙 865 / Adreno 650 / Android 13 / 1080×1440 分析流。

### 11.1 单帧耗时（`adb logcat -s FrameAnalyzer`）

| 模式 | 转换 | 姿态 | 检测 | 帧间隔 | 帧率 |
|---|---|---|---|---|---|
| 常规 · 空场景（2 瓦片 + 整帧） | 40–50ms | 60–75ms | 210–230ms | ~280ms | ≈3.5fps |
| 常规 · 已有人在（仅整帧） | ~45ms | ~70ms | ~80ms | ~200ms | ≈5fps |
| 选区 · 1 个选区 | 37–45ms | 40–75ms | 85–160ms（含姿态） | 128–133ms | **≈7.5–8fps** |

结论：**选区模式比常规模式快一倍以上，而且精度更高**。这不是巧合——
常规模式为了"看见整幅画面"必须付瓦片扫描的钱，选区模式把钱花在用户真正关心的那块。

### 11.2 CPU

| 状态 | 进程 CPU |
|---|---|
| 选区模式 + 预览 | 106–130% |
| 仅预览（暂停检测） | 13–17% |
| 熄屏 + 暂停 | 0.0% |

### 11.3 加速器

| 后端 | 单帧推理 | 结论 |
|---|---|---|
| CPU（4 线程） | 82ms | 基线 |
| GPU delegate (OpenCL) | 84ms | 委托确认生效（266/266 节点 offload），但**没有更快** |
| NNAPI | 69ms | 略快，稳定性一般 |
| int8 量化模型 | — | `UnsatisfiedLinkError`，加载失败 |

输入尺寸扫描（160px → 75ms，320px → 74ms）说明这 ~80ms 是模型自身的计算量，
不是预处理开销。设备没有热降频（thermal status 0，大核 2.24/2.42GHz）。
详见 `docs/PERFORMANCE.md`。

### 11.4 包体积

`abiFilters` 只保留 `arm64-v8a` 后 85MB → **56MB**。GPU 委托带了两个 ABI 的原生库，
去掉 32 位省了约 35MB。

---

## 12. 构建与调试

### 12.1 环境

```
JDK      ~/.jdks/jbr-21.0.11
Gradle   8.11.1（本地分发，无需下载）
AGP      8.5.2
Kotlin   2.0.21
SDK      compileSdk 34 / minSdk 26 / targetSdk 34
```

### 12.2 命令

```bash
# 编译检查（最快）
JAVA_HOME=~/.jdks/jbr-21.0.11 \
~/.gradle/wrapper/dists/gradle-8.11.1-bin/*/gradle-8.11.1/bin/gradle \
  :app:compileDebugKotlin --console=plain

# 打包 + 安装
... :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 12.3 日志字段解读

主诊断行（每 10 帧一条）：

```
cv=43ms pose=68ms det=82ms accel=GPU src=1080x1440 roi=1/1(z1) pose=1 det=[...] -> n=1 fs=0.69 ss=0.69 v=8 PRESENT
```

| 字段 | 含义 |
|---|---|
| `cv` | `ImageProxy` → Bitmap（含旋转）耗时 |
| `pose` | 姿态推理耗时（选区模式下为裁切图上的姿态） |
| `det` | 检测线程总耗时（选区模式含姿态，因为两者串行在同一线程） |
| `accel` | 实际生效的加速器 |
| `src` | 旋转校正后的分析画面尺寸 |
| `roi=x/y(z)` | 选区总数 x / 本帧选中 y 个 / 游标位置 z；`0/0` 表示常规模式 |
| `pose` / `det=[...]` | 两路各给出几个候选，方括号内是 `分数[面积占比]` |
| `n` / `fs` / `ss` / `v` | 融合后候选数 / 单帧最高分 / 平滑分 / 窗口命中数 |

服务侧关键日志：

```
DetectionService: previewVisible=false          # 预览是否绑定
DetectionService: bindCamera() previewVisible=.. # 每次重建绑定
DetectionService: bound facing=1 lens=0/1 preview=false
DetectionService: zoom=2.0 (range 1.0..10.0)
DetectionService: rois=2 budget=2
```

### 12.4 离屏验收探针

`ProbeActivity`（debug 专用，`Theme.NoDisplay`）可以直接对两张验收图跑全瓦片扫描并输出 JSON，
用来做"算法改动前后"的客观对比，不靠肉眼。

```bash
adb shell am start -n com.presenceguard.app/.ProbeActivity \
  --es model efficientdet_lite0.tflite --es accel GPU
```

注意：**一个进程只能测一个模型**。TFLite Task 的原生库是进程级共享的，
关掉最后一个 detector 会卸载它，同进程再建新的会抛
`Error getting native address of native library: task_vision_jni`。
换模型必须 `am force-stop` 之后重来。

### 12.5 常用 adb 操作

```bash
# 看/改配置
adb shell run-as com.presenceguard.app cat /data/data/com.presenceguard.app/shared_prefs/presence_guard_settings.xml

# 点亮 / 熄屏 / 解锁
adb shell input keyevent 224   # WAKEUP
adb shell input keyevent 223   # SLEEP
adb shell input keyevent 82    # MENU（解锁）

# 量 CPU
adb shell top -n 1 -b -o PID,%CPU,ARGS | grep presenceguard
```

---

## 13. 已知限制

1. **广场级的密集小目标**：`probe_far_plaza.jpg`（25–45px 的人）在我们的瓦片扫描下
   分数只能到 0.41–0.43，这是 EfficientDet-Lite0 的物理极限。同图的湖景远景能到
   0.70–0.75。要再往上必须换模型或换 640 输入的方案（见 v2 计划）。
2. **GPU 委托在 865 上不划算**：见 11.3，实际默认跑 CPU 更稳。
3. **本机只暴露 1 颗后置相机**：物理镜头切换按钮会自动隐藏，只能靠 zoom。
4. **变焦与选区的关系**：选区是**图像空间**的坐标，改变变焦后画面内容会变，
   选区仍停在屏幕上的同一位置。设计上视为"盯住这个方向"，不是"盯住这个物体"。
5. **预览重建有几百毫秒黑屏**：`setPreviewVisible(true)` 会重开相机，
   解锁回前台时有一瞬间没有画面。这是"彻底关掉预览流"的代价。
6. **姿态引擎不支持并发太多**：选区内跑姿态是同步阻塞等待（上限 900ms），
   选区数量多时帧率会线性下降，所以 `roiPerFrame` 上限卡在 2。

---

## 14. 调参指南

**远处的人认不出来**

1. 先画选区——这是性价比最高的一步，放大倍数直接翻几倍；
2. 没有明确区域时提高灵敏度（`smallTargetMinHits` 会从 3 降到 2，确认更快）；
3. 瓦片只在小目标上生效，检查目标的面积占比是否已超过 `DEFAULT_TILE_TARGET_MAX_AREA = 0.12`；
4. 最后才考虑降 `minAreaRatio`，因为它同时会放进更多噪声。

**误报太多**

1. 先看日志里的 `fs` 和 `ss`：如果分数很高但还是误报，说明是模型层面认错了，画选区隔离背景更有效；
2. 提高灵敏度档位（低档 `poseOnlyMinScore = 0.70`，几乎只认非常确定的人形）；
3. 检查是不是纯色/无纹理画布被检测引擎当成了巨型人——这条已被
   `MAX_SINGLE_ENGINE_AREA_RATIO = 0.85` 挡住，若仍有，把候选框打印出来看面积；
4. `plausibility` 日志字段低于 0.5 的，基本都是几何上不像人的东西。

**帧率太低**

1. 用选区模式（8fps vs 3.5fps）；
2. 开省电模式（5fps 上限 + 关瓦片）；
3. `roiPerFrame` 降到 1；
4. 别指望 GPU 委托，这台机器上没用。

---

## 15. 版本演进

| 阶段 | 做了什么 | 结果 |
|---|---|---|
| 初版 | 单姿态引擎 | 能认人，但远处漏检严重、插座/立牌常误报 |
| 加检测引擎 | 双引擎 + 概率合成 | 误报大幅下降 |
| 几何约束修正 | 去掉"双肩水平""髋在肩下"，改旋转不变比例 | 躺/侧身/横屏不再被判死 |
| 小目标体系 | 瓦片扫描 + 轨迹持久化 + 重平衡折扣 | 远处召回大幅提升 |
| 选区体系（当前） | 多点多边形 + 选区内放大 + 预览功耗治理 | 精度和帧率同时提升 |
| v2 计划 | YOLO 640 输入 + 自建 TFLite Interpreter 层 | 目标：广场级小目标 |

---

## 附：几个"看起来奇怪其实有原因"的写法

| 写法 | 原因 |
|---|---|
| 检测引擎的 threshold 在构造函数里固化 | TFLite Task 的 `scoreThreshold` 不可变，灵敏度切换只能靠融合层 |
| 瓦片先跑、整帧后跑 | 整帧推理会 recycle 底层 Bitmap，顺序反了就踩雷 |
| 姿态先跑、检测后跑（选区） | 同上，`InputImage` 不接管所有权，`MlImage` 会 |
| `rebind()` 而不是重建 `FrameAnalyzer` | 线程池一旦 shutdown，ML Kit 会在异常分支里重复完成 Task 而闪退 |
| 引擎延迟 600ms 再 close | 在途帧还卡在回调里 |
| 瓦片只回传小目标 | 否则一个人被多个瓦片各框一次 |
| 画面里有人时不做瓦片扫描 | 报警已触发，瓦片的使命是"找人" |
| `smallTargetAreaRatio = 0` 用于选区模式 | 让所有候选跳过"攒帧确认"，直接按分数判 |
| 几何校验只在极度离谱时才否决 | 单点判死是漏报的主要来源 |
