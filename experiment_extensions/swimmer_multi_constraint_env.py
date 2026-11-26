from __future__ import annotations
import numpy as np
from envs.swimmer import SwimmerWithPosPerturbed
from .swimmer_costs import load_thresholds, swimmer_components


class _BaseSwimmerSafety(SwimmerWithPosPerturbed):
    def __init__(self, sigma_viscosity: float = 0.0, max_steps: int = 1000, thresholds_path=None, **kwargs):
        self._safety_thresholds = load_thresholds(thresholds_path)
        super().__init__(sigma_viscosity=sigma_viscosity, max_steps=max_steps, **kwargs)

    def _components(self, action):
        qvel = np.asarray(self.sim.data.qvel.flat, dtype=np.float32)
        return swimmer_components(action, qvel, self._safety_thresholds)


class SwimmerMultiConstraintEnv(_BaseSwimmerSafety):
    def step(self, action):
        ob, reward, _legacy_cost, truncated, terminated, info = super().step(action)
        components = self._components(action)
        info = dict(info)
        info["multi_cost"] = components.copy()
        info["combined_cost"] = float(np.max(components))
        info["action_constraint_cost"] = float(components[0])
        info["joint_speed_constraint_cost"] = float(components[1])
        return ob, reward, components, truncated, terminated, info


class SwimmerCombinedEnv(_BaseSwimmerSafety):
    def step(self, action):
        ob, reward, _legacy_cost, truncated, terminated, info = super().step(action)
        components = self._components(action)
        combined = float(np.max(components))
        info = dict(info)
        info["multi_cost"] = components.copy()
        info["combined_cost"] = combined
        info["action_constraint_cost"] = float(components[0])
        info["joint_speed_constraint_cost"] = float(components[1])
        return ob, reward, combined, truncated, terminated, info
