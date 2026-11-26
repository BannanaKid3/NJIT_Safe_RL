import numpy as np

from envs.half_cheetah import HalfCheetahWithPos
from configs.halfcheetah_safety_thresholds import PITCH_THRESHOLD, PITCH_SCALE


COST_VERSION = "wrapped_inversion_90deg_v3"


class HalfCheetahPitchCost(HalfCheetahWithPos):
    def __init__(self):
        if PITCH_THRESHOLD is None or PITCH_SCALE is None or PITCH_SCALE <= 0:
            raise ValueError("Finalize PITCH_THRESHOLD and PITCH_SCALE in configs/halfcheetah_safety_thresholds.py")
        expected = np.pi / 2.0
        if not np.isclose(float(PITCH_THRESHOLD), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"PITCH_THRESHOLD must be pi/2 for {COST_VERSION}, got {PITCH_THRESHOLD}")
        if not np.isclose(float(PITCH_SCALE), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"PITCH_SCALE must be pi/2 for {COST_VERSION}, got {PITCH_SCALE}")
        super().__init__()

    def step(self, action):
        observation, reward, _, truncated, terminated, info = super().step(action)
        pitch_raw = float(self.sim.data.qpos[2])
        pitch = float(np.arctan2(np.sin(pitch_raw), np.cos(pitch_raw)))
        pitch_abs = abs(pitch)
        excess = max(pitch_abs - float(PITCH_THRESHOLD), 0.0)
        cost = excess / float(PITCH_SCALE)
        info = dict(info)
        info.update(
            {
                "torso_pitch_raw": pitch_raw,
                "torso_pitch": pitch,
                "torso_pitch_abs": pitch_abs,
                "pitch_threshold": float(PITCH_THRESHOLD),
                "pitch_excess": excess,
                "pitch_scale": float(PITCH_SCALE),
                "pitch_cost": float(cost),
                "cost_version": COST_VERSION,
            }
        )
        return observation, reward, float(cost), truncated, terminated, info