# -*- coding: utf-8 -*-
"""
Scene Mode definitions for QiuWu AI — 球悟AI.

Three modes map to shell scripts:
  Scene A: Top-down match judgement  (俯视·比赛智判)
  Scene B: Side-view personal training (侧视·个人智练)
  Scene C: Dual-view full pipeline    (双视·一键全跑)

The mode determines:
  - Which script to run
  - Which widgets are visible in center panel
  - Which data is shown in right panel
  - Whether court calibration / ROI buttons are enabled
"""

# ── Mode constants ───────────────────────────────────────────────

MODE_A = "match"      # 俯视·比赛智判
MODE_B = "side"       # 侧视·个人智练
MODE_C = "dual"       # 双视·一键全跑

# ── Mode metadata ────────────────────────────────────────────────

MODES = {
    MODE_A: {
        "name": "俯视 · 比赛智判",
        "subtitle": "轨迹追踪 · 边界判罚 · 落地检测 · 比分记录",
        "icon": "🎾",
        "color": "#ccff00",
        "script_local": "run_match_local.sh",
        "script_cloud": "run_match_cloud.sh",
        "output_video": "match_judgement/tracknet_rknn_output.mp4",
        "output_csv": "match_judgement/tracknet_rknn_output.csv",
        "output_events": "match_judgement/bounce_events.csv",
        "output_judgement": "match_judgement/judgement.json",
        "mini_court_visible": True,
        "timeline_visible": False,
        "court_calib_enabled": True,
        "roi_enabled": True,
        "right_panel": "judgement",   # score + events table
        "task_name": "match_judgement",
    },
    MODE_B: {
        "name": "侧视 · 个人智练",
        "subtitle": "骨骼提取 · 发球/正手/反手 · 动作分析",
        "icon": "🏃",
        "color": "#58a6ff",
        "script_local": "run_side_local.sh",
        "script_cloud": "run_side_cloud.sh",
        "output_video": "side_training/side_training_output.mp4",
        "output_csv": "side_training/side_tracknet_output.csv",
        "output_prediction": "side_training/body_action_prediction.txt",
        "mini_court_visible": False,
        "timeline_visible": True,
        "court_calib_enabled": False,
        "roi_enabled": False,
        "right_panel": "training",     # action status + pie chart
        "task_name": "side_training",
    },
    MODE_C: {
        "name": "双视 · 一键全跑",
        "subtitle": "俯视比赛 + 侧视训练 · 全功能流水线",
        "icon": "⚡",
        "color": "#f0a050",
        "script_local": "run_all_local.sh",
        "script_cloud": "run_all_cloud.sh",
        "output_video": "match_judgement/tracknet_rknn_output.mp4",
        "output_csv": "match_judgement/tracknet_rknn_output.csv",
        "output_events": "match_judgement/bounce_events.csv",
        "output_judgement": "match_judgement/judgement.json",
        "output_prediction": "side_training/body_action_prediction.txt",
        "mini_court_visible": True,
        "timeline_visible": True,
        "court_calib_enabled": True,
        "roi_enabled": True,
        "right_panel": "combined",     # score + action + summary
        "task_name": "dual_pipeline",
    },
}


def get_mode_info(mode_key):
    """Return mode metadata dict for the given key."""
    return MODES.get(mode_key, MODES[MODE_A])


def get_script(mode_key, cloud=False):
    """Return the shell script name for the given mode and cloud flag."""
    info = MODES.get(mode_key, MODES[MODE_A])
    return info["script_cloud"] if cloud else info["script_local"]
