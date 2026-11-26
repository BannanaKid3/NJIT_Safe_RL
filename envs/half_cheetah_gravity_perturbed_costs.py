import numpy as np

from envs.half_cheetah_combined_max_cost import HalfCheetahCombinedMaxCost
from envs.half_cheetah_multi_constraint_cost import HalfCheetahMultiConstraintCost


class _StepwiseGravityPerturbationMixin:
    def _configure_gravity_perturbation(self, sigma_gravity):
        sigma_gravity = float(sigma_gravity)
        if not np.isfinite(sigma_gravity) or sigma_gravity < 0.0:
            raise ValueError("sigma_gravity must be a finite nonnegative number")
        self.sigma_gravity = sigma_gravity
        self._gravity_axis = 2
        self._base_gravity_z = float(self.model.opt.gravity[self._gravity_axis])
        self._last_gravity_noise = 0.0
        self._last_gravity_z = self._base_gravity_z

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=self.model.nq,
        )
        qvel = self.init_qvel + self.np_random.normal(
            loc=0.0,
            scale=0.1,
            size=self.model.nv,
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self, seed=None, **kwargs):
        result = super().reset(seed=seed, **kwargs)
        self.model.opt.gravity[self._gravity_axis] = self._base_gravity_z
        self._last_gravity_noise = 0.0
        self._last_gravity_z = self._base_gravity_z
        return result

    def _sample_and_apply_gravity(self):
        if self.sigma_gravity == 0.0:
            noise = 0.0
        else:
            noise = float(self.np_random.normal(0.0, self.sigma_gravity))
        gravity_z = self._base_gravity_z + noise
        self.model.opt.gravity[self._gravity_axis] = gravity_z
        self._last_gravity_noise = noise
        self._last_gravity_z = gravity_z

    def _gravity_info(self):
        return {
            "base_gravity_z": self._base_gravity_z,
            "gravity_noise": self._last_gravity_noise,
            "gravity_z": self._last_gravity_z,
            "sigma_gravity": self.sigma_gravity,
            "gravity_resampling": "every_step_before_simulation",
        }


class HalfCheetahCombinedMaxCostGravityPerturbed(
    _StepwiseGravityPerturbationMixin,
    HalfCheetahCombinedMaxCost,
):
    def __init__(self, sigma_gravity=0.5):
        super().__init__()
        self._configure_gravity_perturbation(sigma_gravity)

    def step(self, action):
        self._sample_and_apply_gravity()
        observation, reward, cost, truncated, terminated, info = super().step(action)
        info = dict(info)
        info.update(self._gravity_info())
        return observation, reward, cost, truncated, terminated, info


class HalfCheetahMultiConstraintCostGravityPerturbed(
    _StepwiseGravityPerturbationMixin,
    HalfCheetahMultiConstraintCost,
):
    def __init__(self, sigma_gravity=0.5):
        super().__init__()
        self._configure_gravity_perturbation(sigma_gravity)

    def step(self, action):
        self._sample_and_apply_gravity()
        observation, reward, cost, truncated, terminated, info = super().step(action)
        info = dict(info)
        info.update(self._gravity_info())
        return observation, reward, cost, truncated, terminated, info
