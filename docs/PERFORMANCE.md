# 性能实测与加速器结论

设备：Xiaomi M2007J1SC（骁龙 865 / Adreno 650），Android 13，分析流 1080×1440。
所有数字都是用 `docs/DEVELOPMENT.md` 第 12.4 节的离屏探针和实机日志量出来的，
不是估算。

---

## 1. 加速器对比

同一个 float32 EfficientDet-Lite0 模型，同一张图，各跑 10 次取中位数：

| 后端 | 单帧推理 | 备注 |
|---|---|---|
| CPU（4 线程） | **82ms** | 基线 |
| GPU delegate（`useGpu()`） | **84ms** | 委托确认生效，但没有更快 |
| NNAPI（`useNnapi()`） | **69ms** | 略快，但稳定性一般 |
| int8 量化模型 | 加载失败 | `UnsatisfiedLinkError` |

GPU 委托生效的证据（日志里 266 个节点全部 offload 到 `TfLiteGpuDelegateV2`，走 OpenCL）：

```
TfLiteGpuDelegateV2: Created new OpenCL context
GpuDelegate: Delegate created, 266/266 nodes delegated
```

**为什么 GPU 没有更快**：EfficientDet-Lite0 的 backbone 是深度可分离卷积，算子细碎、
每层数据量小，GPU 的调度/同步开销吃掉了并行收益。这类"小而碎"的模型在移动 GPU 上
经常跑不过 4 线程 CPU，属于正常现象，不是配置错误。

**排除预处理干扰**：把模型输入从 320 缩到 160 再测：

| 输入尺寸 | 单帧推理 |
|---|---|
| 160 × 160 | 75ms |
| 320 × 320 | 74ms |

输入缩小 4 倍耗时几乎不变 ⇒ 这 ~80ms 是**模型自身的计算量**，
不是图像缩放、拷贝、内存分配这些周边开销。想在 865 上把单帧压到 20ms，
换后端没用，只能换更小的模型或减少推理次数。

**排除热降频**：测量时 `thermal status = 0`（无节流），大核 2.24 / 2.42GHz，
与小核的频率比正常。

---

## 2. 量化模型为什么加载不了

试过三个来源的 int8 / float16 版本：

| 来源 | 文件 | 结果 |
|---|---|---|
| MediaPipe models bucket | `efficientdet_lite0/int8/1/*.tflite` | `UnsatisfiedLinkError` |
| TF Hub | `lite-model/efficientdet/lite0/int8/1`（4,561,184 字节，TFL3 头合法） | `UnsatisfiedLinkError` |
| 同源 float32 | `efficientdet_lite0/float32/1/*.tflite` | 正常加载 |

异常被 `TaskJniUtils.createHandleFromLibrary` 包了一层，栈顶是
`Error getting native address of native library`。文件头校验通过、大小正常，
说明不是下载损坏，而是 `tensorflow-lite-task-vision:0.4.4` 的 JNI 不接受量化模型。

结论：这个版本要么用 float32，要么绕开 Task 库自己写 `Interpreter`（v2 走这条路）。

另有一个相关的坑：**TFLite Task 的原生库是进程级共享的**。关掉最后一个 detector
会卸载它，同进程再创建新的 detector 就必然失败。所以离屏探针一次只能测一个模型，
换模型必须 `am force-stop` 之后重来。

---

## 3. 端到端帧耗时

| 模式 | 转换 | 姿态 | 检测 | 帧间隔 | 帧率 |
|---|---|---|---|---|---|
| 常规 · 空场景（2 瓦片 + 整帧） | 40–50ms | 60–75ms | 210–230ms | ~280ms | ≈3.5fps |
| 常规 · 已有人在（仅整帧） | ~45ms | ~70ms | ~80ms | ~200ms | ≈5fps |
| 选区 · 1 个选区（放大识别） | 37–45ms | 40–75ms | 85–160ms（含姿态） | 128–133ms | **≈7.5–8fps** |
| 省电模式 | 同上 | 同上 | 同上 | ≥200ms（主动丢帧） | 5fps 上限 |

**选区模式比常规模式快一倍以上**：常规模式为了覆盖整幅画面必须付瓦片扫描的钱，
选区模式把这笔预算花在用户真正关心的那块区域上，同时精度更高。

---

## 4. CPU 占用（含预览与显示合成）

`adb shell top -n 1 -b -o PID,%CPU,ARGS`，单位是一个核心的百分比：

| 状态 | 进程 CPU |
|---|---|
| 选区模式检测 + 预览（8fps） | 106–130% |
| 仅预览（暂停检测） | 13–17% |
| 熄屏 + 暂停 | **0.0%** |

推导：

- 预览（相机输出流 + 显示合成）约占 **13–17%** 单核；
- 检测（1 个选区的姿态 + 检测，8fps）约占 **90–115%** 单核；
- 熄屏后把 Preview 用例从相机上摘掉，这部分完整归零，检测不受影响。

**注意**：只调用 `Preview.setSurfaceProvider(null)` 是不会归零的——用例还绑在相机上，
那条输出流照样在跑。必须把它从 `bindToLifecycle` 的用例列表里去掉。
详见 `DEVELOPMENT.md` 4.4。

---

## 5. 识别质量（离屏验收）

用两张刻意刁难的样图做基准，`detectSweep()` 全瓦片扫描后的最高分数：

| 样图 | 目标尺寸 | 6×4 网格 | 4×3 网格 | 说明 |
|---|---|---|---|---|
| `probe_far_lake.jpg`（湖边长焦远景，含行人/儿童/骑行者） | ~40–80px | 0.70–0.75 | 0.70–0.72 | 可靠识别 |
| `probe_far_plaza.jpg`（车站广场，密集小目标） | ~25–45px | — | 0.41–0.43 | 模型物理极限 |

广场图到不了可靠阈值这件事，是 **EfficientDet-Lite0 在 320×320 输入下的能力边界**，
不是参数没调好：同一个模型在湖景图上能到 0.75，说明流程没问题；广场图的目标只有
25–45px，缩放后不足 10px，特征不存在。要突破必须换 640 输入的 YOLO 类模型（v2 计划）。

---

## 6. 包体积

| 项 | 变化 |
|---|---|
| 初始（含 armeabi-v7a + arm64-v8a） | 85MB |
| 去掉 armeabi-v7a | **56MB** |

GPU 委托插件自带两个 ABI 的 `libtensorflowlite_gpu_jni.so`，去掉 32 位省了约 35MB。
`x86_64` 只对模拟器有意义，`armeabi-v7a` 是为 32 位老机型准备的，本项目的目标设备
全部是 arm64。
