import numpy as np

from envs.half_cheetah import HalfCheetahWithPos
from configs.halfcheetah_safety_thresholds import JOINT_SPEED_THRESHOLDS, JOINT_SPEED_SCALES, COST_VERSION


class HalfCheetahJointSpeedCost(HalfCheetahWithPos):
    def __init__(self):
        if JOINT_SPEED_THRESHOLDS is None or JOINT_SPEED_SCALES is None:
            raise ValueError("Finalize JOINT_SPEED_THRESHOLDS and JOINT_SPEED_SCALES in configs/halfcheetah_safety_thresholds.py")
        self.joint_speed_thresholds = np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64)
        self.joint_speed_scales = np.asarray(JOINT_SPEED_SCALES, dtype=np.float64)
        if self.joint_speed_thresholds.shape != (6,) or self.joint_speed_scales.shape != (6,) or np.any(self.joint_speed_scales <= 0):
            raise ValueError("Joint thresholds and scales must each contain six positive values")
        super().__init__()

    def step(self, action):
        observation, reward, _, truncated, terminated, info = super().step(action)
        velocities = np.asarray(self.sim.data.qvel[3:9], dtype=np.float64).copy()
        speeds = np.abs(velocities)
        excess = np.maximum(speeds - self.joint_speed_thresholds, 0.0)
        normalized_excess = excess / self.joint_speed_scales
        cost = float(np.max(normalized_excess))
        info = dict(info)
        info.update(
            {
                "joint_velocities": velocities,
                "joint_speeds": speeds,
                "joint_speed_thresholds": self.joint_speed_thresholds.copy(),
                "joint_speed_excess": excess,
                "joint_speed_normalized_excess": normalized_excess,
                "joint_speed_scales": self.joint_speed_scales.copy(),
                "joint_speed_cost": cost,
                "fastest_joint_index": int(np.argmax(speeds)),
                "maximum_joint_speed": float(np.max(speeds)),
                "cost_version": COST_VERSION,
            }
        )
        return observation, reward, cost, truncated, terminated, info