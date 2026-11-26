from __future__ import annotations
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = HERE / "swimmer_thresholds.json"


def load_thresholds(path=None) -> dict:
    path = Path(path) if path else DEFAULT_THRESHOLDS
    if not path.exists():
        raise RuntimeError(f"Swimmer calibration file not found: {path}. Run the calibration script first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "CALIBRATED":
        raise RuntimeError(
            f"Swimmer thresholds are not calibrated yet: status={data.get('status')!r}. "
            "Run calibration/calibrate_swimmer_joint_speed.py first."
        )
    needed = ["action_threshold", "action_scale", "joint_speed_threshold", "joint_speed_scale"]
    missing = [k for k in needed if k not in data]
    if missing:
        raise RuntimeError(f"Swimmer threshold file missing keys: {missing}")
    try:
        vals = np.array([float(data[k]) for k in needed], dtype=float)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Swimmer thresholds are not calibrated: {data}") from e
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        raise RuntimeError(f"Invalid Swimmer thresholds: {dict(zip(needed, vals))}")
    return data


def swimmer_components(action, qvel, thresholds: dict) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    qvel = np.asarray(qvel, dtype=np.float32).reshape(-1)
    if qvel.size < 5:
        raise ValueError(f"Expected Swimmer qvel with >=3 entries, got {qvel}")
    action_peak = float(np.max(np.abs(action)))
    joint_speed_peak = float(np.max(np.abs(qvel[3:])))
    c_action = max(action_peak - float(thresholds["action_threshold"]), 0.0) / float(thresholds["action_scale"])
    c_joint = max(joint_speed_peak - float(thresholds["joint_speed_threshold"]), 0.0) / float(thresholds["joint_speed_scale"])
    return np.array([c_action, c_joint], dtype=np.float32)
