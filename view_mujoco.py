import os

# Must be set before importing Gym or MuJoCo.
os.environ.pop("MUJOCO_GL", None)

# Use Mesa's CPU renderer instead of the problematic WSL GPU GLX context.
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3COMPAT"

import time
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


def main() -> None:
    env = HalfCheetahEnv(render_mode="human")

    try:
        observation, info = env.reset(seed=1)

        for _ in range(2000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

            time.sleep(0.01)
    finally:
        env.close()


if __name__ == "__main__":
    main()