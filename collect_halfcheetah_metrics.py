import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from analysis.halfcheetah_metric_utils import save_metrics
from code_ipm_rcmdp_rcrl_max_hc import (
    Actor_Beta,
    Actor_Gaussian,
    Normalization,
    RewardScaling,
    RunningMeanStd,
)
from envs.half_cheetah import (
    HalfCheetahWithPos,
    HalfCheetahWithPosPerturbed,
    unwrap_reset_observation,
)


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        replacements = {
            "Normalization": Normalization,
            "RewardScaling": RewardScaling,
            "RunningMeanStd": RunningMeanStd,
        }
        if module == "__main__" and name in replacements:
            return replacements[name]
        return super().find_class(module, name)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return CompatibleUnpickler(handle).load()


def resolve_model_prefix(project_root, environment_name, run_number, requested):
    if requested is not None:
        prefix = Path(requested)
        if not prefix.is_absolute():
            prefix = project_root / prefix
        return prefix
    model_directory = (
        project_root
        / "models"
        / environment_name
        / f"run{run_number}"
    )
    best_prefix = model_directory / "Best_RCAC"
    if Path(f"{best_prefix}_actor").exists():
        return best_prefix
    actor_files = sorted(model_directory.glob("*_actor"))
    if not actor_files:
        raise FileNotFoundError(
            f"No actor model was found in {model_directory}"
        )
    return Path(str(actor_files[-1])[:-6])


def build_environment(environment_name, sigma_gravity):
    if environment_name == "HalfCheetahWithPos":
        return HalfCheetahWithPos()
    if environment_name == "HalfCheetahWithPosPerturbed":
        return HalfCheetahWithPosPerturbed(
            sigma_gravity=sigma_gravity
        )
    raise ValueError(f"Unsupported environment: {environment_name}")


def build_actor(
    policy_dist,
    state_dim,
    action_dim,
    max_action,
    hidden_width,
):
    actor_args = SimpleNamespace(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        hidden_width=hidden_width,
        use_tanh=True,
        use_orthogonal_init=False,
    )
    if policy_dist == "Gaussian":
        return Actor_Gaussian(actor_args)
    if policy_dist == "Beta":
        return Actor_Beta(actor_args)
    raise ValueError("Only Gaussian and Beta policies are supported")


def deterministic_action(actor, policy_dist, state, max_action):
    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
    ).unsqueeze(0)
    with torch.no_grad():
        if policy_dist == "Gaussian":
            action = actor(state_tensor)
        else:
            action = actor.mean(state_tensor)
            action = 2.0 * (action - 0.5) * max_action
    return action.cpu().numpy().reshape(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument(
        "--env",
        type=str,
        default="HalfCheetahWithPos",
        choices=(
            "HalfCheetahWithPos",
            "HalfCheetahWithPosPerturbed",
        ),
    )
    parser.add_argument(
        "--policy-dist",
        type=str,
        default="Gaussian",
        choices=("Gaussian", "Beta"),
    )
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sigma-gravity", type=float, default=0.0)
    parser.add_argument("--model-prefix", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    environment = build_environment(
        args.env,
        args.sigma_gravity,
    )
    environment.action_space.seed(args.seed)
    initial_state = unwrap_reset_observation(
        environment.reset(seed=args.seed)
    )
    state_dim = int(initial_state.shape[0])
    action_dim = int(environment.action_space.shape[0])
    max_action = float(environment.action_space.high[0])
    model_prefix = resolve_model_prefix(
        project_root,
        args.env,
        args.run,
        args.model_prefix,
    )
    actor_path = Path(f"{model_prefix}_actor")
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor file not found: {actor_path}")
    actor = build_actor(
        args.policy_dist,
        state_dim,
        action_dim,
        max_action,
        args.hidden_width,
    )
    actor.load_state_dict(
        torch.load(
            actor_path,
            map_location=torch.device("cpu"),
        )
    )
    actor.eval()
    state_norm_path = Path(f"{model_prefix}_state_norm")
    state_norm = (
        load_pickle(state_norm_path)
        if state_norm_path.exists()
        else None
    )
    torso_pitch = []
    joint_velocities = []
    joint_speeds = []
    actions = []
    rewards = []
    costs = []
    episode_ids = []
    step_ids = []
    episode_rewards = []
    episode_costs = []
    episode_max_costs = []
    for episode_index in range(args.episodes):
        state = unwrap_reset_observation(
            environment.reset(seed=args.seed + episode_index)
        )
        if state_norm is not None:
            state = state_norm(state, update=False)
        done = False
        step_index = 0
        total_reward = 0.0
        total_cost = 0.0
        maximum_cost = float("-inf")
        while not done:
            action = deterministic_action(
                actor,
                args.policy_dist,
                state,
                max_action,
            )
            (
                next_state,
                reward,
                cost,
                truncated,
                terminated,
                info,
            ) = environment.step(action)
            done = bool(truncated or terminated)
            torso_pitch.append(float(info["torso_pitch"]))
            joint_velocities.append(
                np.asarray(
                    info["joint_velocities"],
                    dtype=np.float64,
                ).copy()
            )
            joint_speeds.append(
                np.asarray(
                    info["joint_speeds"],
                    dtype=np.float64,
                ).copy()
            )
            actions.append(action.copy())
            rewards.append(float(reward))
            costs.append(float(cost))
            episode_ids.append(episode_index)
            step_ids.append(step_index)
            total_reward += float(reward)
            total_cost += float(cost)
            maximum_cost = max(maximum_cost, float(cost))
            state = np.asarray(next_state, dtype=np.float64)
            if state_norm is not None:
                state = state_norm(state, update=False)
            step_index += 1
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_max_costs.append(maximum_cost)
    environment.close()
    if args.output is None:
        output_path = (
            project_root
            / "data_train"
            / args.env
            / f"run{args.run}"
            / "metric_calibration"
            / "baseline_metrics.npz"
        )
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    saved_path = save_metrics(
        output_path=output_path,
        torso_pitch=torso_pitch,
        joint_velocities=joint_velocities,
        joint_speeds=joint_speeds,
        actions=actions,
        rewards=rewards,
        costs=costs,
        episode_id=episode_ids,
        step_in_episode=step_ids,
        episode_rewards=episode_rewards,
        episode_costs=episode_costs,
        episode_max_costs=episode_max_costs,
        model_prefix=model_prefix,
        environment_name=args.env,
        seed=args.seed,
    )
    print(saved_path)


if __name__ == "__main__":
    main()