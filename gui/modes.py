# -*- coding: utf-8 -*-
"""Scene metadata for the three QiuWu AI operating modes."""

# The three entries deliberately sit at the same navigation level.  MODE_C is
# retained only as a compatibility constant for older saved GUI state; it is
# no longer offered as a scene mode.
MODE_LIVE = "live"
MODE_A = "match"
MODE_B = "side"
MODE_C = "dual"


MODES = {
    MODE_LIVE: {
        "name": "实时演示",
        "subtitle": "摄像头实时预览 · TrackNet 网球检测 · Pose 人体骨架",
        "icon": "●",
        "color": "#7ee7ff",
        "script_local": "",
        "script_cloud": "",
        "mini_court_visible": False,
        "timeline_visible": False,
        "court_calib_enabled": False,
        "roi_enabled": False,
        "right_panel": "live",
        "task_name": "live_preview",
    },
    MODE_A: {
        "name": "俯视 · 比赛智判",
        "subtitle": "轨迹跟踪 · 边界判罚 · 落地检测 · 比分记录",
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
        "right_panel": "judgement",
        "task_name": "match_judgement",
    },
    MODE_B: {
        "name": "侧视 · 个人智练",
        "subtitle": "骨架提取 · 发球/正手/反手 · 动作分析",
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
        "right_panel": "training",
        "task_name": "side_training",
    },
}


def get_mode_info(mode_key):
    """Return metadata, falling back to the match workflow."""
    return MODES.get(mode_key, MODES[MODE_A])


def get_script(mode_key, cloud=False):
    """Return the corresponding offline or cloud script name."""
    info = get_mode_info(mode_key)
    return info["script_cloud"] if cloud else info["script_local"]
