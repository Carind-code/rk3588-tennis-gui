# RK3588纯网球追踪

独立的RK3588三核TrackNet追踪工程。仅完成网球逐帧追踪和红点短尾迹视频输出，不包含场地标定、出界/落地判罚、人体动作分析、CSV、上云或IPC。

默认模型是当前侧视训练模型`side_tracknet_360x640_sigmoid_fp.rknn`，输入为三帧`640x360`NHWC时序图像；三个独立RKNNLite2 Runtime分别绑定NPU核心`0,1,2`并行推理。

## 本地视频测试

将待测视频命名为`input.mp4`放到本目录，板端执行：

```bash
cd ~/track
python3 tracknet_ball_tracker.py
```

输出为`track_output.mp4`。视频中只有当前网球红点、白色描边和短黄色历史轨迹。

日志同时给出：

- `不含写视频：纯追踪FPS`：解码、前处理、三核推理、热图候选提取和轨迹平滑的完整追踪吞吐；不含视频编码。
- `包含写视频：完整FPS`：加上第二次读取源视频、轨迹绘制和MP4写入后的真实端到端吞吐。

只测追踪性能、不生成视频：

```bash
python3 tracknet_ball_tracker.py --no-write
```

## 摄像头接口

命令接口已预留为：

```bash
python3 tracknet_ball_tracker.py --camera /dev/video11 --camera-width 1024 --camera-height 576
```

该路径直接使用V4L2摄像头，三核并行推理后按帧序写出追踪视频。建议使用`--camera-frames 600`固定采集600帧后自动结束，以确保MP4文件正常封装。摄像头模式采用低延迟轻量关联，不引入离线视频的全局平滑，不影响本地视频验证路径。

## 依赖

板端原有环境需具备：`python3`、`opencv-python`、`numpy`、`rknn-toolkit-lite2`。
