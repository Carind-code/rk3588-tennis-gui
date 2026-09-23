# RK3588 网球智能分析系统

当前交付包含两个独立功能模块，共用同一块 RK3588 开发板，按需启动即可，无需同时运行。

## 功能目录

- `match_judgement/`：比赛视角网球轨迹追踪、球场标定、落地检测、基础判罚和阿里云上报。
- `body_action/`：人体姿态提取与发球、正手、反手、背景动作识别。

## 快速测试

比赛视角追踪与判罚，不上云：

```bash
cd ~/rk3588_tennis_system
bash run_match_local.sh
```

比赛视角追踪、判罚并上云：

```bash
bash run_match_cloud.sh
```

人体动作识别并生成标注视频：

```bash
bash run_body_action.sh
```

人体动作识别但不写标注视频：

```bash
bash run_body_action_fast.sh
```

## 默认输入与输出

比赛视角默认输入：`match_judgement/input.mp4`。

比赛视角输出：

- `match_judgement/tracknet_rknn_output.mp4`
- `match_judgement/tracknet_rknn_output.csv`
- `match_judgement/bounce_events.csv`
- `match_judgement/judgement.json`

人体动作默认输入：`body_action/samples/sample_video.mp4`。

人体动作输出位于 `body_action/outputs/`。

## 环境检查

板端需能导入以下依赖：

```bash
python3 -c "import cv2, numpy, mediapipe; from rknnlite.api import RKNNLite; print('环境正常')"
```

上云还需要有效的 `match_judgement/cloud/config.json` 和可用网络。
