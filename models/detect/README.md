# RK3588近景网球检测演示

本工程仅做一件事：对本地视频或USB摄像头画面中的网球进行逐帧检测，并以荧光绿色框、球心和短轨迹显示结果。

不使用TrackNet，不包含球场标定、判罚、人体动作、CSV、云端或IPC逻辑。它用于近距离单球演示和摄像头联调。

## 模型与边界

模型为`models/yolo_tennis_ball_fp.rknn`：基于公开YOLOv8n网球检测权重转换的RK3588 FP RKNN模型，输入为`1x3x640x640`。模型在NPU三核上单Runtime推理，画面处理采用最新帧机制，避免积压导致显示延迟。

程序采用两级近景检测：YOLO负责通用网球语义检测；HSV黄绿色阈值、面积、圆度和长宽比约束负责在单球近景画面中补偿短时漏检，并输出`YOLO`、`YOLO+COLOR`或`COLOR`来源标记。默认允许`COLOR`启动，以保证手持黄绿色网球在干净背景下的实时演示；在复杂黄绿色背景中可加`--no-color-bootstrap`禁用该补偿，只保留YOLO结果。

公开权重的主要训练场景偏比赛/训练视频，**尚未用USB摄像头近距离手持球数据验证精度**。若目标是稳定演示，应采集本机位、本光照和本背景下的近景帧，标注后微调轻量检测模型，再以同场景图像进行INT8校准。

## 板端运行

### 本地视频验证

默认写入带荧光绿轨迹的视频，并分别输出不含写视频的模型FPS和完整端到端FPS：

```bash
cd ~/0trest/detect
DISPLAY=:0 python3 tennis_ball_detector.py --source video --video input.mp4
```

不写视频、只测模型与画面显示吞吐：

```bash
DISPLAY=:0 python3 tennis_ball_detector.py --source video --video input.mp4 --no-write
```

### USB摄像头实时画面

```bash
cd ~/0trest/detect
DISPLAY=:0 python3 tennis_ball_detector.py --source camera --camera /dev/video21 --camera-width 1280 --camera-height 720 --camera-fps 30
```

按`q`或`Esc`退出。实时模式不写视频，以降低CPU编码负担。窗口日志中的`visual FPS`是摄像头采集、画面显示与短轨迹的帧率，`YOLO inference FPS`是语义检测吞吐，两者应分别看待。

## 摄像头接入说明

默认V4L2节点为`/dev/video21`。程序会请求`1280x720@30`，实际帧率由USB UVC链路决定。先确认摄像头链路确实有图像输出：

```bash
v4l2-ctl -d /dev/video21 --set-fmt-video=width=1280,height=720,pixelformat=MJPG --stream-mmap=4 --stream-count=120 --stream-poll
lsusb | grep -i "1bcf:2d50" && v4l2-ctl -d /dev/video21 --list-formats-ext
```

若出现`Unexpected sensor id(000000)`、`VIDIOC_STREAMON ... Operation not permitted`或持续`select timeout`，问题在摄像头USB连接、供电或UVC枚举，尚未进入模型程序。先修复该链路，再测试检测效果。

## 给后续开发者

- 模型预处理必须保持BGR画面等比缩放至640x640、填充值114；RKNN输入使用NHWC缓冲区。
- 运行时使用`NPU_CORE_0_1_2`是单个模型实例使用三核，不要复制成三个YOLO实例，否则会增加内存和调度竞争。
- 若近距离小球漏检，先采集并标注目标机位数据，优先微调模型；不要仅靠放宽`--confidence`解决，因为误检会明显增加。
- `--confidence`、`--min-area`、`--color-gate`可针对实拍环境小幅调参。复杂背景可加`--no-color-bootstrap`；颜色检测适合单球近景演示，不代替针对本机位数据的模型训练。

## 目录

```text
detect/
  tennis_ball_detector.py   检测、荧光绿轨迹、视频/摄像头入口
  camera_capture.py         V4L2最新帧采集
  models/yolo_tennis_ball_fp.rknn
  tools/convert_yolo_tennis_i8.py
  input.mp4                 本地验证输入
  outputs/                  结果视频
```

模型来源：[RJTPP/tennis-ball-detection](https://huggingface.co/RJTPP/tennis-ball-detection)。
