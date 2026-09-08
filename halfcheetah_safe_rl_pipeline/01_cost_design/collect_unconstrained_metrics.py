import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from envs.half_cheetah_unconstrained import HalfCheetahUnconstrained


JOINT_NAMES = (
    "back_thigh",
    "back_shin",
    "back_foot",
    "front_thigh",
    "front_shin",
    "front_foot",
)


class ActorGaussian(nn.Module):
    def __init__(self, state_dim, hidden_width, action_dim, max_action):
        super().__init__()
        self.max_action = float(max_action)
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.activate_func = nn.ReLU()

    def forward(self, state):
        state = self.activate_func(self.fc1(state))
        state = self.activate_func(self.fc2(state))
        state = self.activate_func(self.fc3(state))
        return self.max_action * torch.tanh(self.mean_layer(state))


def unwrap_reset(reset_result):
    result = reset_result
    while isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float64)


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def resolve_actor(model_prefix):
    prefix = Path(model_prefix)
    if prefix.exists():
        return prefix
    actor = Path(str(prefix) + "_actor")
    if actor.exists():
        return actor
    raise FileNotFoundError(f"{prefix}\n{actor}")


def collect(run, episodes, seed, model_prefix, output):
    env = HalfCheetahUnconstrained()
    actor_path = resolve_actor(model_prefix)
    state = torch.load(actor_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    if "actor_state_dict" in state:
        state = state["actor_state_dict"]

    actor = ActorGaussian(
        int(state["fc1.weight"].shape[1]),
        int(state["fc1.weight"].shape[0]),
        int(state["mean_layer.weight"].shape[0]),
        float(env.action_space.high[0]),
    )
    actor.load_state_dict(state)
    actor.eval()

    episode = []
    reset_seed = []
    step = []
    reward = []
    cost = []
    xpos = []
    torso_pitch_raw = []
    torso_pitch = []
    torso_pitch_abs = []
    joint_velocities = []
    joint_speeds = []
    maximum_joint_speed = []
    fastest_joint_index = []
    actions = []
    maximum_action_abs = []
    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    episode_net_displacements = []
    episode_absolute_displacements = []
    episode_traveled_distances = []

    for ep in range(episodes):
        ep_seed = seed + ep
        obs = unwrap_reset(env.reset(seed=ep_seed))
        start_x = float(env.sim.data.qpos[0])
        previous_x = start_x
        total_reward = 0.0
        total_cost = 0.0
        traveled = 0.0
        t = 0
        done = False

        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor(x).cpu().numpy().reshape(-1)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs_next, r, c, truncated, terminated, info = env.step(action)
            done = bool(truncated or terminated)

            current_x = float(info.get("xpos", env.sim.data.qpos[0]))
            traveled += abs(current_x - previous_x)
            previous_x = current_x

            pitch_raw = float(env.sim.data.qpos[2])
            pitch = wrap_angle(pitch_raw)

            velocities = np.asarray(
                info.get("joint_velocities", env.sim.data.qvel[3:9]),
                dtype=np.float64,
            ).copy()
            speeds = np.asarray(
                info.get("joint_speeds", np.abs(velocities)),
                dtype=np.float64,
            ).copy()
            fastest = int(np.argmax(speeds))

            episode.append(ep + 1)
            reset_seed.append(ep_seed)
            step.append(t)
            reward.append(float(r))
            cost.append(float(c))
            xpos.append(current_x)
            torso_pitch_raw.append(pitch_raw)
            torso_pitch.append(pitch)
            torso_pitch_abs.append(abs(pitch))
            joint_velocities.append(velocities)
            joint_speeds.append(speeds)
            maximum_joint_speed.append(float(speeds[fastest]))
            fastest_joint_index.append(fastest)
            actions.append(np.asarray(action, dtype=np.float64).copy())
            maximum_action_abs.append(float(np.max(np.abs(action))))

            total_reward += float(r)
            total_cost += float(c)
            obs = obs_next
            t += 1

        end_x = float(env.sim.data.qpos[0])
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_lengths.append(t)
        episode_net_displacements.append(end_x - start_x)
        episode_absolute_displacements.append(abs(end_x - start_x))
        episode_traveled_distances.append(traveled)

        print(
            f"Episode {ep + 1}/{episodes} seed={ep_seed} "
            f"reward={total_reward:.3f} cost={total_cost:.6f} "
            f"displacement={end_x - start_x:.3f}"
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        run_number=np.asarray(run, dtype=np.int64),
        first_seed=np.asarray(seed, dtype=np.int64),
        number_of_episodes=np.asarray(episodes, dtype=np.int64),
        joint_names=np.asarray(JOINT_NAMES),
        episode=np.asarray(episode, dtype=np.int64),
        reset_seed=np.asarray(reset_seed, dtype=np.int64),
        step=np.asarray(step, dtype=np.int64),
        reward=np.asarray(reward, dtype=np.float64),
        cost=np.asarray(cost, dtype=np.float64),
        xpos=np.asarray(xpos, dtype=np.float64),
        torso_pitch_raw=np.asarray(torso_pitch_raw, dtype=np.float64),
        torso_pitch=np.asarray(torso_pitch, dtype=np.float64),
        torso_pitch_abs=np.asarray(torso_pitch_abs, dtype=np.float64),
        joint_velocities=np.asarray(joint_velocities, dtype=np.float64),
        joint_speeds=np.asarray(joint_speeds, dtype=np.float64),
        maximum_joint_speed=np.asarray(maximum_joint_speed, dtype=np.float64),
        fastest_joint_index=np.asarray(fastest_joint_index, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.float64),
        maximum_action_abs=np.asarray(maximum_action_abs, dtype=np.float64),
        episode_rewards=np.asarray(episode_rewards, dtype=np.float64),
        episode_costs=np.asarray(episode_costs, dtype=np.float64),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
        episode_net_displacements=np.asarray(episode_net_displacements, dtype=np.float64),
        episode_absolute_displacements=np.asarray(episode_absolute_displacements, dtype=np.float64),
        episode_traveled_distances=np.asarray(episode_traveled_distances, dtype=np.float64),
    )

    print()
    print("Saved:", output)
    print("Actor:", actor_path)
    print("Episodes:", episodes)
    print("Timesteps:", len(step))
    print("Mean episode reward:", float(np.mean(episode_rewards)))
    print("Maximum recorded cost:", float(np.max(cost)))
    print("Wrapped pitch P99:", float(np.percentile(torso_pitch_abs, 99)))
    print("Maximum wrapped pitch:", float(np.max(torso_pitch_abs)))
    print("Joint-speed P99:", np.percentile(np.asarray(joint_speeds), 99, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--model-prefix", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    model_prefix = (
        Path(args.model_prefix)
        if args.model_prefix
        else project / "models" / "HalfCheetahUnconstrained" / f"run{args.run}" / "Best_Unconstrained"
    )
    output = (
        Path(args.output)
        if args.output
        else project / "data_train" / "HalfCheetahUnconstrained" / f"run{args.run}" / "metric_calibration" / "baseline_metrics.npz"
    )

    if not model_prefix.is_absolute():
        model_prefix = project / model_prefix
    if not output.is_absolute():
        output = project / output

    collect(args.run, args.episodes, args.seed, model_prefix, output)


if __name__ == "__main__":
    main()