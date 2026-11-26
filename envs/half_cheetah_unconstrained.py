import os

import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100
REWARD_TYPE = "old"
ABS_PATH = os.path.abspath(os.path.dirname(__file__))
HALF_CHEETAH_JOINT_NAMES = (
    "back_thigh",
    "back_shin",
    "back_foot",
    "front_thigh",
    "front_shin",
    "front_foot",
)


def add_halfcheetah_diagnostics(sim, info):
    torso_pitch = float(sim.data.qpos[2])
    joint_velocities = np.asarray(sim.data.qvel[3:9], dtype=np.float64).copy()
    joint_speeds = np.abs(joint_velocities)
    fastest_joint_index = int(np.argmax(joint_speeds))
    result = dict(info)
    result.update(
        {
            "torso_pitch": torso_pitch,
            "torso_pitch_abs": abs(torso_pitch),
            "joint_velocities": joint_velocities,
            "joint_speeds": joint_speeds,
            "maximum_joint_speed": float(joint_speeds[fastest_joint_index]),
            "fastest_joint_index": fastest_joint_index,
            "fastest_joint_name": HALF_CHEETAH_JOINT_NAMES[fastest_joint_index],
        }
    )
    return result


def unwrap_reset_observation(reset_result):
    result = reset_result
    while isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float64)


class HalfCheetahTest(HalfCheetahEnv):
    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.current_timestep = 0
        self.violations = 0
        return observation

    def step(self, action):
        result = super().step(action)
        if len(result) == 5:
            next_observation, reward, terminated, truncated, info = result
        else:
            next_observation, reward, done, info = result
            terminated = bool(done)
            truncated = False
        self.current_timestep = getattr(self, "current_timestep", 0) + 1
        if np.any(np.abs(action) > ACTION_TORQUE_THRESHOLD):
            self.violations = getattr(self, "violations", 0) + 1
        if self.violations > VIOLATIONS_ALLOWED:
            terminated = True
            reward = 0.0
        if len(result) == 5:
            return next_observation, reward, terminated, truncated, info
        return next_observation, reward, terminated or truncated, info


class HalfCheetahWithObstacle(HalfCheetahEnv):
    def __init__(self, xml_file=ABS_PATH + "/xmls/half_cheetah_obstacle.xml"):
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            dtype=np.float32,
        )

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        observation = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
        }
        return observation, reward, False, info

    def camera_setup(self):
        super().camera_setup()
        self.camera._render_camera.distance = 5.0

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )


class HalfCheetahEqual(HalfCheetahEnv):
    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        observation = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
        }
        return observation, reward, False, info


class HalfCheetahBackward(HalfCheetahEnv):
    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        observation = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = -(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
        }
        return observation, reward, False, info


class HalfCheetahWithPos(HalfCheetahEnv):
    OBS_DIM = 18
    max_steps = 1000

    def __init__(self):
        super().__init__()
        observation_high = np.full(self.OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-observation_high,
            high=observation_high,
            dtype=np.float32,
        )
        self._elapsed_steps = 0

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        reset_result = super().reset()
        self._elapsed_steps = 0
        return reset_result, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

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

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "xpos": xposafter,
        }
        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt
        reward = reward_dist + reward_ctrl
        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "reward_dist": reward_dist,
            "xpos": xposafter,
        }
        return reward, info

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        observation = self._get_obs()
        if REWARD_TYPE == "new":
            reward, info = self.new_reward(xposbefore, xposafter, action)
        else:
            reward, info = self.old_reward(xposbefore, xposafter, action)
        info = add_halfcheetah_diagnostics(self.sim, info)
        self._elapsed_steps += 1
        cost = float(
            np.maximum(
                np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
                0.0,
            )
        )
        truncated = self._elapsed_steps >= self.max_steps
        terminated = False
        return observation, reward, cost, truncated, terminated, info

class HalfCheetahUnconstrained(HalfCheetahEnv):
    OBS_DIM = 18
    max_steps = 1000

    def __init__(self):
        super().__init__()
        observation_high = np.full(self.OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-observation_high,
            high=observation_high,
            dtype=np.float32,
        )
        self._elapsed_steps = 0

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        reset_result = super().reset()
        self._elapsed_steps = 0
        return reset_result, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

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

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "xpos": xposafter,
        }
        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt
        reward = reward_dist + reward_ctrl
        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "reward_dist": reward_dist,
            "xpos": xposafter,
        }
        return reward, info

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        observation = self._get_obs()
        if REWARD_TYPE == "new":
            reward, info = self.new_reward(xposbefore, xposafter, action)
        else:
            reward, info = self.old_reward(xposbefore, xposafter, action)
        info = add_halfcheetah_diagnostics(self.sim, info)
        self._elapsed_steps += 1
        cost = 0.0
        truncated = self._elapsed_steps >= self.max_steps
        terminated = False
        return observation, reward, cost, truncated, terminated, info


class HalfCheetahWithPosPerturbed(HalfCheetahEnv):
    OBS_DIM = 18
    max_steps = 1000

    def __init__(self, sigma_gravity=0.7, max_steps=1000):
        super().__init__()
        observation_high = np.full(self.OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-observation_high,
            high=observation_high,
            dtype=np.float32,
        )
        self._elapsed_steps = 0
        self.sigma_gravity = float(sigma_gravity)
        self.max_steps = int(max_steps)
        self._gravity_axis = 2
        self._base_gravity = float(self.model.opt.gravity[self._gravity_axis])

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        reset_result = super().reset()
        self._elapsed_steps = 0
        self.model.opt.gravity[self._gravity_axis] = self._base_gravity
        return reset_result, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

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

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "xpos": xposafter,
        }
        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt
        reward = reward_dist + reward_ctrl
        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "reward_dist": reward_dist,
            "xpos": xposafter,
        }
        return reward, info

    def step(self, action):
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._gravity_axis] = (
                self._base_gravity
                + self.np_random.normal(
                    loc=0.0,
                    scale=self.sigma_gravity,
                )
            )
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        observation = self._get_obs()
        if REWARD_TYPE == "new":
            reward, info = self.new_reward(xposbefore, xposafter, action)
        else:
            reward, info = self.old_reward(xposbefore, xposafter, action)
        info = add_halfcheetah_diagnostics(self.sim, info)
        self._elapsed_steps += 1
        cost = float(
            np.maximum(
                np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
                0.0,
            )
        )
        truncated = self._elapsed_steps >= self.max_steps
        terminated = False
        return observation, reward, cost, truncated, terminated, info