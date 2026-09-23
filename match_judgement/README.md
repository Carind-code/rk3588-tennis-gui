# RK3588 Tennis Detection

RK3588 板端网球轨迹、场地映射、落地判别、基础得分判罚与阿里云上报工程。

## 功能

- 三个 RKNNLite2 worker 分别绑定 RK3588 NPU 核心 0、1、2。
- V3.7 TrackNet FP 模型处理连续三帧，输出球心热力图。
- 使用 V3.7 原生种子确认、强弱置信度门控、动态速度门、突变确认、短片段/尖峰清理和受限插值。
- 启动时扫描视频，首次可靠识别球场后固定单应矩阵，不再重复运行场地模型。
- 默认按 TENNIIISSSS V5.2 使用 MediaPipe 在上下场 ROI 内逐帧采样球员上下文，用于过滤击球和发球抛球造成的假落地；未安装时自动降级为纯轨迹/球场坐标规则。
- 基于图像坐标和标准球场坐标的速度、加速度、方向变化识别落地。
- 支持单打/双打 IN/OUT、己方场地落地和二次落地的基础得分判罚。
- 输出带实时比分栏的 V5.2 reference 风格标注视频、轨迹 CSV、落地事件 CSV 和判罚 JSON。
- 可通过 Unix Domain Socket 交给上云进程，上传到阿里云 OSS 并通过 MQTT 通知结果。

## 快速使用

本地测试，不上云：

```bash
bash start_test.sh
```

推理并上云：

```bash
bash start_all.sh
```

默认输入为 `input.mp4`，默认按照单打边界判罚。

默认会尝试启用 V5.2 MediaPipe 球员上下文，采样间隔为 `--mediapipe-stride 1`。如果板端未安装 MediaPipe，程序不会中断，但终端统计会显示 `MediaPipe球员上下文：未启用/不可用`，落地过滤效果会弱于完整 V5.2。

禁用 MediaPipe 性能兜底：

```bash
bash start_test.sh --no-mediapipe
```

降低 MediaPipe 采样频率：

```bash
bash start_test.sh --mediapipe-stride 3
```

板端尝试安装 MediaPipe：

```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mediapipe==0.10.14
python3 -c "import mediapipe as mp; print(mp.__version__)"
```

如果 RK3588 aarch64 环境找不到可用 wheel，不建议现场编译 MediaPipe；直接使用 `--no-mediapipe` 或提高 `--mediapipe-stride`，保证主链路稳定。

## 输出

- `tracknet_rknn_output.mp4`：左上角实时比分、球点尾迹、右上角 mini-court 映射、落地/击球事件和判罚标注视频。
- `tracknet_rknn_output.csv`：图像坐标、标准球场坐标、置信度和各阶段耗时。
- `bounce_events.csv`：落地帧、落地坐标、事件置信度和时间修正信息。
- `judgement.json`：IN/OUT、最后击球方、得分方和判罚原因。

比分按 `top`（画面上方球员）和 `bottom`（画面下方球员）累计。每条得分判罚包含 `score_before`、`score_after` 和 `score_changed`；最终 `point_summary` 会随判罚 JSON 上传 OSS，并在 MQTT 完成消息的 `score` 字段中直接上报。

同一回合只允许记一分：终止性判罚得分后锁定比分，检测到新的球员触球后才开启下一回合。被保留用于分析但不重复计分的判罚会标记为 `score_suppressed: true`。

标准球场坐标为 `1000 x 2168`：左上角为原点，`x` 向右，`y` 向下。单打边线约为 `x=125..875`，双打边线为 `x=0..1000`。

默认视频可视化对齐 TENNIIISSSS V5.2：主画面显示红色当前球点、黄色历史轨迹尾巴、落地/击球事件标记，右上角显示 mini-court 球场映射。默认不在主画面绘制整场投影线；如需调试场地标定，加：

```bash
bash start_test.sh --draw-court-lines
```

## 模型

- `model_v37_360x640_b1_sigmoid_fp.rknn`：V3.7 网球追踪 FP 模型。
- `court_detector_fp.rknn`：球场关键点 FP 模型，仅在启动标定阶段运行。
- `broadcast_court_only.json`：MediaPipe 上下场球员检测区域，不是模型权重；用于避免检测观众、球童、广告牌和记分牌。

ONNX 到 RKNN 的可复现转换脚本位于 `tools/convert_v37_fp.py`。

## 性能口径

- 追踪输入固定为 `1x360x640x9` NHWC，三个 worker 分别绑定 NPU 核心 0、1、2。
- 追踪 ONNX 约 `175 KB`，FP RKNN 约 `3.0 MB`；球场 RKNN 约 `34.1 MB`。
- 球场模型只运行到首个可靠场地被检出，随后释放，整段视频复用固定映射。
- 程序结束时分别输出前处理、单核单帧推理、三核总吞吐、后处理、场地标定、MediaPipe 球员上下文、事件处理和端到端 FPS。不同视频编码、是否写 MP4、是否启用 MediaPipe 会显著影响端到端 FPS。

## 注意

当前得分模块覆盖落点出界、击球后落在己方场地、二次落地和上下场得分次数汇总。发球区、第一发/第二发、抢七和整局计分需要在比赛规则状态机中继续扩展。正式判罚前应使用实际机位视频人工标注并评估落地事件 Precision、Recall 和 F1。
