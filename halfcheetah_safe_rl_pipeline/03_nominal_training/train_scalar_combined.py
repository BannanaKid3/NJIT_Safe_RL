import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import code_ipm_rcmdp_rcrl_max_hc as base
from envs.half_cheetah_combined_max_cost import HalfCheetahCombinedMaxCost
from configs.halfcheetah_combined_safety import (
    PITCH_THRESHOLD,
    PITCH_SCALE,
    JOINT_NAMES,
    JOINT_SPEED_THRESHOLDS,
    JOINT_SPEED_SCALES,
    PERSISTENT_EPS,
    COMBINED_COST_VERSION,
)


ENV_NAME = "HalfCheetahCombinedMaxCost"
BEST_NAME = "Best_CombinedMaxCost"


def save_histories(data_dir, plot_dir, evaluate_rewards, evaluate_costs, evaluate_max_costs, episode_rewards, episode_costs, episode_max_costs):
    np.save(data_dir / "evaluate_rewards.npy", np.asarray(evaluate_rewards, dtype=np.float64))
    np.save(data_dir / "evaluate_costs.npy", np.asarray(evaluate_costs, dtype=np.float64))
    np.save(data_dir / "evaluate_max_costs.npy", np.asarray(evaluate_max_costs, dtype=np.float64))
    np.save(plot_dir / "episode_rewards.npy", np.asarray(episode_rewards, dtype=np.float64))
    np.save(plot_dir / "episode_costs.npy", np.asarray(episode_costs, dtype=np.float64))
    np.save(plot_dir / "episode_max_costs.npy", np.asarray(episode_max_costs, dtype=np.float64))


def main(args):
    if args.seed != 5:
        raise ValueError("Official matched combined run uses training seed 5")
    if args.max_train_steps != 2000:
        raise ValueError("Official matched combined run uses exactly 2000 training episodes")
    if not np.isclose(args.persistent_eps, PERSISTENT_EPS, rtol=0.0, atol=1e-12):
        raise ValueError(f"persistent_eps must be {PERSISTENT_EPS}")

    base.HalfCheetahWithPos = HalfCheetahCombinedMaxCost
    args.env = "HalfCheetahWithPos"

    env = HalfCheetahCombinedMaxCost()
    env_evaluate = HalfCheetahCombinedMaxCost()

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    env_evaluate.reset(seed=args.seed)
    env_evaluate.action_space.seed(args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env.max_steps

    model_dir = Path("models") / ENV_NAME / COMBINED_COST_VERSION / f"run{args.run}"
    data_dir = Path("data_train") / ENV_NAME / COMBINED_COST_VERSION / f"run{args.run}"
    plot_dir = Path("plot_data") / ENV_NAME / COMBINED_COST_VERSION / f"run{args.run}"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    existing_actor = model_dir / f"{BEST_NAME}_actor"
    if existing_actor.exists():
        raise FileExistsError(
            f"{existing_actor} already exists. Move or delete this combined run folder before starting a fresh official run."
        )

    metadata = {
        "environment": ENV_NAME,
        "cost_version": COMBINED_COST_VERSION,
        "aggregation": "max",
        "training_seed": int(args.seed),
        "training_episodes": int(args.max_train_steps),
        "pitch_threshold": float(PITCH_THRESHOLD),
        "pitch_scale": float(PITCH_SCALE),
        "joint_names": list(JOINT_NAMES),
        "joint_speed_thresholds": np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64).tolist(),
        "joint_speed_scales": np.asarray(JOINT_SPEED_SCALES, dtype=np.float64).tolist(),
        "evaluate_freq": int(args.evaluate_freq),
        "persistent_eps": float(args.persistent_eps),
        "beta": float(args.beta),
        "K_epochs": int(args.K_epochs),
        "warm_start_episode": int(args.warm_start_episode),
        "batch_size": int(args.batch_size),
        "mini_batch_size": int(args.mini_batch_size),
        "hidden_width": int(args.hidden_width),
        "lr_a": float(args.lr_a),
        "lr_c": float(args.lr_c),
        "lr_cost": float(args.lr_cost),
        "gamma": float(args.gamma),
        "lamda": float(args.lamda),
        "epsilon": float(args.epsilon),
        "entropy_coef": float(args.entropy_coef),
        "weight_reg": float(args.weight_reg),
        "lambda_": int(args.lambda_),
        "policy_dist": str(args.policy_dist),
    }
    (data_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    replay_buffer = base.ReplayBuffer(args)
    agent = base.Robust_RCAC_NPG(args)
    writer = SummaryWriter(
        log_dir=f"runs/RNAC/env_{ENV_NAME}_{COMBINED_COST_VERSION}_Gaussian_run{args.run}_seed_{args.seed}"
    )

    state_norm = base.Normalization(shape=args.state_dim) if args.use_state_norm else None
    reward_norm = base.Normalization(shape=1) if args.use_reward_norm else None
    reward_scaling = base.RewardScaling(shape=1, gamma=args.gamma) if args.use_reward_scaling else None

    evaluate_rewards = []
    evaluate_costs = []
    evaluate_max_costs = []
    episode_rewards = []
    episode_costs = []
    episode_max_costs = []
    best_reward = float("-inf")
    feasible_checkpoint_found = False

    for total_steps in tqdm(range(args.max_train_steps)):
        s = env.reset()[0][0]
        if state_norm is not None:
            s = state_norm(s)
        if reward_scaling is not None:
            reward_scaling.reset()

        episode_steps = 0
        done = False
        total_reward = 0.0
        total_cost = 0.0
        max_cost = 0.0
        agent.beta = args.beta
        agent.warm_start_flag = int(total_steps > args.warm_start_episode)

        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            action = a
            s_next, r, c, truncated, terminated, info = env.step(action)
            done = bool(truncated or terminated)

            total_reward += float(r)
            total_cost += float(c)
            max_cost = max(max_cost, float(c))

            if state_norm is not None:
                s_next = state_norm(s_next)

            training_reward = r
            if reward_norm is not None:
                training_reward = reward_norm(training_reward)
            elif reward_scaling is not None:
                training_reward = reward_scaling(training_reward)

            dw = bool(done and episode_steps != args.max_episode_steps)
            replay_buffer.store(
                s,
                a,
                a_logprob,
                training_reward,
                c,
                s_next,
                dw,
                done,
            )
            s = copy.deepcopy(s_next)

            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_max_costs.append(max_cost)

        if total_steps % args.evaluate_freq == 0:
            evaluate_reward, evaluate_cost, evaluate_max_cost = base.evaluate_policy(
                args,
                env_evaluate,
                agent,
                state_norm=state_norm,
                reward_scaling=reward_scaling,
            )
            evaluate_rewards.append(evaluate_reward)
            evaluate_costs.append(evaluate_cost)
            evaluate_max_costs.append(evaluate_max_cost)

            writer.add_scalar("evaluation_reward", evaluate_reward, total_steps)
            writer.add_scalar("evaluation_cost", evaluate_cost, total_steps)
            writer.add_scalar("evaluation_max_cost", evaluate_max_cost, total_steps)

            save_histories(
                data_dir,
                plot_dir,
                evaluate_rewards,
                evaluate_costs,
                evaluate_max_costs,
                episode_rewards,
                episode_costs,
                episode_max_costs,
            )

            print(
                f"episode={total_steps} "
                f"evaluation={len(evaluate_rewards)} "
                f"reward={evaluate_reward:.6f} "
                f"cost={evaluate_cost:.6f} "
                f"max_cost={evaluate_max_cost:.6f}"
            )

            if evaluate_reward > best_reward and evaluate_max_cost <= args.persistent_eps:
                best_reward = evaluate_reward
                feasible_checkpoint_found = True
                base.save_agent(
                    agent,
                    str(model_dir / BEST_NAME),
                    state_norm=state_norm,
                    reward_scaling=reward_scaling,
                )

    save_histories(
        data_dir,
        plot_dir,
        evaluate_rewards,
        evaluate_costs,
        evaluate_max_costs,
        episode_rewards,
        episode_costs,
        episode_max_costs,
    )

    base.save_agent(
        agent,
        str(model_dir / "Final_CombinedMaxCost"),
        state_norm=state_norm,
        reward_scaling=reward_scaling,
    )

    writer.close()
    env.close()
    env_evaluate.close()

    print("Training complete")
    print("Cost version:", COMBINED_COST_VERSION)
    print("Aggregation: max")
    print("Training seed:", args.seed)
    print("Training episodes:", args.max_train_steps)
    print("Best feasible checkpoint found:", feasible_checkpoint_found)
    print("Model directory:", model_dir)
    print("Data directory:", data_dir)

    if not feasible_checkpoint_found:
        raise RuntimeError(
            "No evaluation checkpoint satisfied evaluate_max_cost <= persistent_eps. Final_CombinedMaxCost was saved for diagnostics, but do not use it as the official combined result."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--evaluate_freq", type=int, default=100)
    parser.add_argument("--persistent_eps", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=30000.0)
    parser.add_argument("--K_epochs", type=int, default=5)
    parser.add_argument("--warm_start_episode", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--hidden_width", type=int, default=64)
    parser.add_argument("--lr_a", type=float, default=1e-3)
    parser.add_argument("--lr_c", type=float, default=5e-3)
    parser.add_argument("--lr_cost", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--weight_reg", type=float, default=0.001)
    parser.add_argument("--lambda_", type=int, default=50)
    parser.add_argument("--warm_start_flag", type=int, default=0)
    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--use_adv_norm", type=bool, default=True)
    parser.add_argument("--use_state_norm", type=bool, default=False)
    parser.add_argument("--use_reward_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=False)
    parser.add_argument("--use_lr_decay", type=bool, default=True)
    parser.add_argument("--use_grad_clip", type=bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=bool, default=True)
    parser.add_argument("--set_adam_eps", type=bool, default=True)
    parser.add_argument("--use_tanh", type=int, default=1)
    parser.add_argument("--adaptive_alpha", type=bool, default=False)
    parser.add_argument("--GAMMA", type=str, default="0")
    main(parser.parse_args())
