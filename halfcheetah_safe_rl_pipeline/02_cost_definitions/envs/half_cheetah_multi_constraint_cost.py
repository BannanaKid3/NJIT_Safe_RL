import numpy as np

from envs.half_cheetah_combined_max_cost import HalfCheetahCombinedMaxCost


MULTI_CONSTRAINT_COST_VERSION = "multi_constraint_pitch_joint_v1"
COST_NAMES = ("pitch", "joint_speed")


class HalfCheetahMultiConstraintCost(HalfCheetahCombinedMaxCost):
    cost_dim = 2

    def step(self, action):
        observation, reward, _, truncated, terminated, info = super().step(action)
        cost = np.asarray(
            [info["pitch_cost"], info["joint_speed_cost"]],
            dtype=np.float32,
        )
        info = dict(info)
        info["cost_vector"] = cost.copy()
        info["cost_names"] = COST_NAMES
        info["cost_version"] = MULTI_CONSTRAINT_COST_VERSION
        return observation, reward, cost, truncated, terminated, info
