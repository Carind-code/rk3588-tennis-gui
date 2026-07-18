# RK3588 网球智能分析 GUI（球悟AI / QiuWu AI）

基于 PyQt5 的 RK3588 网球智能分析系统图形界面，运行于 RK3588 开发板（Wayland/Qt），支持比赛智判、个人智练、双视全跑三种模式，集成视频可视化、AI 叠加、比分记录、动作时间线、NPU 监控与云端上报等功能。

## 功能特性

- **三模式驱动**：俯视比赛智判 / 侧视个人智练 / 双视一键全跑
- **实时视频叠加**：AI 轨迹、落地点、骨骼、动作标签叠加显示
- **球场标定 & ROI 编辑**：可视化拖拽标定比赛球场与玩家检测区域
- **比分 & 事件表**：自动记录判罚事件与比分变化
- **动作时间线**：侧视模式下展示发球/正手/反手动作片段
- **云端上报**：本地/云端双链路，云状态实时指示
- **NPU 监控**：实时显示 RK3588 NPU 占用率
- **摄像头录制**：内置 camera_recorder，支持录制回放
- **暗色科幻 HUD**：自带暗色 QSS 主题 + 科技感边框叠层
- **Qt 兼容层**：主用 PyQt5，可回退 PySide6

## 目录结构

```
.
├── gui/
│   ├── tennis_gui.py          # 主窗口入口（QiuWu AI 主程序）
│   ├── app.py                 # QApplication 引导（主题/字体/高DPI）
│   ├── modes.py               # 三模式定义与脚本映射
│   ├── qt_compat.py           # PyQt5 / PySide6 兼容层
│   ├── camera_recorder.py     # 摄像头录制模块
│   ├── panels/                # 布局面板
│   │   ├── top_bar.py         #   顶部状态栏
│   │   ├── left_panel.py      #   左侧模式选择
│   │   ├── center_panel.py    #   中央可视化区
│   │   ├── right_panel.py     #   右侧数据区
│   │   ├── bottom_console.py  #   底部控制台
│   │   └── roi_editor.py      #   ROI 可视化编辑
│   ├── widgets/               # 功能组件
│   │   ├── video_widget.py    #   视频显示
│   │   ├── ai_overlay.py      #   AI 叠加层
│   │   ├── mini_court.py      #   迷你球场
│   │   ├── score_box.py       #   比分框
│   │   ├── event_table.py     #   事件表
│   │   ├── action_timeline.py #   动作时间线
│   │   ├── cloud_indicator.py #   云状态指示
│   │   ├── npu_monitor.py     #   NPU 监控
│   │   ├── progress_bar.py    #   进度条
│   │   └── camera_preprocess.py # 摄像头预处理
│   ├── backend/               # 后端桥接
│   │   ├── process_manager.py #   子进程管理（启动/停止 backend 脚本）
│   │   ├── file_watcher.py    #   输出文件监视
│   │   └── data_parser.py     #   结果数据解析
│   └── resources/             # 资源
│       ├── style.qss          #   暗色主题样式
│       └── *.png              #   皮肤/背景图
├── roi_editor.py              # 独立 ROI 编辑工具（tkinter）
├── run_gui.sh                 # 一键启动脚本
└── README.md
```

## 三种模式

| 模式 | 名称 | 说明 | 本地脚本 | 云端脚本 |
|------|------|------|----------|----------|
| A | 俯视 · 比赛智判 | 轨迹追踪 · 边界判罚 · 落地检测 · 比分记录 | `run_match_local.sh` | `run_match_cloud.sh` |
| B | 侧视 · 个人智练 | 骨骼提取 · 发球/正手/反手 · 动作分析 | `run_side_local.sh` | `run_side_cloud.sh` |
| C | 双视 · 一键全跑 | 俯视比赛 + 侧视训练 · 全功能流水线 | `run_all_local.sh` | `run_all_cloud.sh` |

> 注：上述 shell 脚本由上层项目 `rk3588_tennis_system` 提供，调用 `match_judgement/`、`body_action/`、`side_training/` 等后端模块。本仓库仅包含 GUI 层代码。

## 环境依赖

```bash
# RK3588 板端
sudo apt install python3-pyqt5
pip install opencv-python numpy
# 后端推理（由上层项目提供）
pip install rknnlite mediapipe
```

环境自检：

```bash
python3 -c "import cv2, numpy, PyQt5; print('GUI 环境正常')"
```

## 快速启动

```bash
# 窗口模式（1280x800）
bash run_gui.sh

# 全屏模式
bash run_gui.sh --fullscreen

# 无边框 kiosk 模式
bash run_gui.sh --kiosk
```

启动脚本会自动：
- 检查 `gui/` 目录与 PyQt5 依赖
- 终止已运行的 GUI 实例
- 设置 Wayland 平台与 Qt 插件路径（修复 OpenCV 与系统 Qt 插件冲突）
- 进入 `gui/` 目录执行 `python3 tennis_gui.py`

## 架构概览

```
┌───────────────────────────────────────────────┐
│                 tennis_gui.py                  │
│              (QiuWuAI MainWindow)              │
├───────┬───────────────────────────┬───────────┤
│ Left  │        Center             │  Right    │
│ Mode  │   video + AI overlay      │  Data     │
│ Panel │   + mini court/timeline   │  Panel    │
├───────┴───────────────────────────┴───────────┤
│              Bottom Console                    │
├───────────────────────────────────────────────┤
│  Backend: ProcessManager / FileWatcher /      │
│           DataParser  ──►  shell scripts       │
│                  ──►  match_judgement /        │
│                      body_action / side_train  │
└───────────────────────────────────────────────┘
```

GUI 通过 `ProcessManager` 以子进程方式启动后端推理脚本，通过 `FileWatcher` 监视输出文件变化，由 `DataParser` 解析后实时更新到各 Widget。子进程 stdout/stderr 通过 Qt 信号实时输出到底部控制台。

## 运行平台

- **目标平台**：RK3588 开发板（aarch64，Wayland）
- **Qt 平台插件**：`/usr/lib/aarch64-linux-gnu/qt5/plugins`
- **显示**：默认 Wayland，可切 EGLFS 直渲
- **分辨率**：1280×800 窗口 / 全屏自适应

## 相关模块

本 GUI 为 `rk3588_tennis_system` 的界面层，后端推理模块（不含于本仓库）：

- `match_judgement/` — 比赛视角轨迹追踪、球场标定、落地检测、判罚、阿里云上报
- `body_action/` — 人体姿态提取与发球/正手/反手/背景动作识别
- `side_training/` — 侧视训练分析
- `models/` — RKNN 模型文件
