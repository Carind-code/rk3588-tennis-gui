# RK3588实时人体骨架显示

本工程用于在RK3588桌面端接入USB摄像头，对镜头前的单人实时绘制人体骨架。它是独立的实时姿态显示工程，不使用MediaPipe，也不包含动作分类、网球追踪或视频写入链路。

模型采用Rockchip官方RKNN Model Zoo的YOLOv8n-pose RK3588 INT8转换流程：一个模型同时输出人框和17个COCO骨骼关键点，运行时绑定RK3588的`NPU_CORE_0_1_2`。摄像头线程始终只提交最新帧，推理短暂变慢时会主动丢弃积压帧，以降低画面滞后。

## 适用范围与预期

- **适合**：USB摄像头清晰画面中的单人全身站立、举手、下蹲、弓步、摆臂和较大幅度挥拍动作；建议人物高度不低于约300像素，镜头光线均匀，身体尽量完整进入画面。
- **不等同于动作生物力学测量**：本模型输出17个COCO关键点，不是MediaPipe的33点精细手部/足部标记；球拍、手臂交叉、多人遮挡、人物被画面裁切、逆光或高速模糊会降低腕、肘、踝等关键点稳定性。
- **准确性参考**：上游YOLOv8n-pose在COCO Pose验证集上的`mAP50-95=50.4`、`mAP50=80.1`。该指标用于通用人体姿态，并非USB摄像头或网球专项实测结果；本工程未在板端实机运行，测试时应以本README的验收方法记录实际FPS和骨架稳定性。
- **性能预期**：RKNN Model Zoo公开的RK3588单核原始模型基准为约`55.9 FPS`。实际摄像头端到端FPS会受USB摄像头实际输出帧率、图像预处理、桌面显示和CPU频率影响，不能把该原始NPU指标直接当作实测结果。

## 工程结构

```text
Pose/
├─ pose_live.py                 # 摄像头/本地视频实时骨架主程序
├─ camera_capture.py             # V4L2最新帧采集，避免缓冲积压
├─ models/
│  └─ yolov8n_pose_i8.rknn      # 已转换的RK3588 INT8模型
├─ tools/
│  └─ convert_yolov8_pose_i8.py # 虚拟机端官方转换流程备份
├─ outputs/                      # 可选视频输出目录
└─ requirements.txt
```

模型文件SHA256：`9a968d62fe3768e2c0576c1aac2bbdd0f6b6a6808c5f6b3674c8f1dcc9bb1c2b`

## 板端依赖

板端原有RKNNLite2环境无需重复安装。确认下列命令成功即可：

```bash
python3 -c "from rknnlite.api import RKNNLite; import cv2, numpy; print('ok')"
v4l2-ctl --list-devices
```

若系统缺少OpenCV或NumPy，且板端已联网：

```bash
python3 -m pip install --user -r requirements.txt
```

## 一键实时骨架显示

将整个`Pose`目录复制到板端后，在RK3588本机桌面终端运行：

```bash
cd ~/0test/Pose
python3 pose_live.py
```

默认参数是`/dev/video21`、`1280x720`、请求`30 FPS`。屏幕会出现`QiuWu AI | RK3588 Live Pose`窗口；按`q`或`Esc`退出。

若从SSH启动、但仍希望窗口显示在板端桌面：

```bash
cd ~/0test/Pose
DISPLAY=:0 python3 pose_live.py
```

更换摄像头节点或采集尺寸：

```bash
python3 pose_live.py --camera /dev/video21 --camera-width 1280 --camera-height 720 --camera-fps 30
```

## 本地视频验证

用于先验证骨架效果，或生成带骨架的视频：

```bash
cd ~/0test/Pose
python3 pose_live.py --source video --video ./input.mp4 --out-video ./outputs/pose_overlay.mp4
```

只测推理吞吐、不显示窗口也不写视频：

```bash
python3 pose_live.py --source video --video ./input.mp4 --no-display
```

程序结束时会输出两项互不混淆的指标：

- `Average NPU inference / theoretical pose FPS`：仅RKNN调用阶段的平均时间和理论推理FPS。
- `End-to-end visual FPS`：读取、预处理、后处理、骨架绘制、显示和可选写视频后的真实完整FPS。

摄像头模式每5秒输出`LIVE`行，其中`display`为实际显示FPS，`pose`和`avg_npu`为最近30次姿态推理的平均性能。

## USB摄像头测试与验收

开始程序前，先单独验证摄像头实际帧率。以下命令只测试V4L2，不调用NPU：

```bash
v4l2-ctl -d /dev/video21 \
  --set-fmt-video=width=1280,height=720,pixelformat=MJPG \
  --stream-mmap=4 --stream-count=300 --stream-poll
```

验收建议：

1. 摄像头独立测试稳定达到目标帧率，且无`select timeout`。
2. 单人完整站入画面，依次做举手、下蹲、弓步、正手/反手挥拍；肩、肘、腕、髋、膝、踝应连续跟随，不应长期漂移到背景。
3. 记录终端的`LIVE`和最终统计。若视觉FPS明显低于30，先排查USB摄像头实际帧率，再检查NPU。
4. 出现无画面、超时或枚举错误时，收集：

```bash
lsusb | grep -i "1bcf:2d50" && v4l2-ctl -d /dev/video21 --list-formats-ext
```

本工程不会绕过摄像头问题。若V4L2独立测试不稳定，应先排查USB连接/供电/UVC枚举，再评估姿态模型。

## 运行机制

```text
USB摄像头 V4L2最新帧
        │
        ▼
640x640 Letterbox + BGR转RGB
        │
        ▼
YOLOv8n-pose INT8 RKNN（三核掩码）
        │
        ▼
人框 + 17关键点 + NMS
        │
        ▼
骨架叠加到最新相机帧并显示
```

输入张量为`1x640x640x3`、RGB、`uint8`，模型配置按`/255`归一化。输出是三组检测头和一组17点关键点张量。后处理使用DFL框解码、置信度筛选、NMS和letterbox反变换。默认只保留置信度最高的一人；多人调试时可使用`--max-persons 2`。

## 模型转换复现

已在转换虚拟机中，使用RKNN-Toolkit2.3.2和官方`rknn_model_zoo/examples/yolov8_pose/python/convert.py`成功转换。`tools/convert_yolov8_pose_i8.py`保留了同一官方Hybrid INT8配置，供后续复现；不要在板端运行它。

转换需要官方YOLOv8-pose ONNX和官方标定集。示例：

```bash
cd /home/elf/work/yolov8_pose_rk3588
/home/elf/miniconda3/envs/RKNN-Toolkit2-2.3.2/bin/python \
  /path/to/Pose/tools/convert_yolov8_pose_i8.py \
  --onnx ./yolov8n-pose.onnx \
  --dataset /home/elf/rknn_model_zoo-main/examples/yolov8_pose/model/dataset.txt \
  --rknn ./yolov8n_pose_i8.rknn
```

## 后续扩展边界

此工程只负责实时人体骨架显示。若需要“发球/正手/反手”等动作分类，应将稳定的17点时序特征交给独立动作模型；若需要网球训练分析，应与侧视TrackNet工程并行运行或在上层调度，避免把多个模型硬塞入本实时预览线程。

上游参考：[Rockchip RKNN Model Zoo YOLOv8 Pose](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8_pose)、[Ultralytics YOLOv8 Pose](https://docs.ultralytics.com/models/yolov8/)。
