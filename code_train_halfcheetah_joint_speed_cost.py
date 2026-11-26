import argparse
import copy
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import code_ipm_rcmdp_rcrl_max_hc as base
from envs.half_cheetah_jointspeed_cost import HalfCheetahJointSpeedCost


ENV_NAME = "HalfCheetahJointSpeedCost"
BEST_NAME = "Best_JointSpeedCost"



def main(args):
    base.HalfCheetahWithPos = HalfCheetahJointSpeedCost
    args.env = "HalfCheetahWithPos"

    env = HalfCheetahJointSpeedCost()
    env_evaluate = HalfCheetahJointSpeedCost()

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

    model_dir = f"./models/{ENV_NAME}/run{args.run}"
    data_dir = f"./data_train/{ENV_NAME}/run{args.run}"
    plot_dir = f"./plot_data/{ENV_NAME}/run{args.run}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    replay_buffer = base.ReplayBuffer(args)
    agent = base.Robust_RCAC_NPG(args)
    writer = SummaryWriter(
        log_dir=f"runs/RNAC/env_{ENV_NAME}_Gaussian_run{args.run}_seed_{args.seed}"
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

            np.save(f"{data_dir}/evaluate_rewards.npy", np.asarray(evaluate_rewards))
            np.save(f"{data_dir}/evaluate_costs.npy", np.asarray(evaluate_costs))
            np.save(f"{data_dir}/evaluate_max_costs.npy", np.asarray(evaluate_max_costs))
            np.save(f"{plot_dir}/episode_rewards.npy", np.asarray(episode_rewards))
            np.save(f"{plot_dir}/episode_costs.npy", np.asarray(episode_costs))
            np.save(f"{plot_dir}/episode_max_costs.npy", np.asarray(episode_max_costs))

            print(
                f"evaluation={len(evaluate_rewards)} "
                f"reward={evaluate_reward:.6f} "
                f"cost={evaluate_cost:.6f} "
                f"max_cost={evaluate_max_cost:.6f}"
            )

            if evaluate_reward > best_reward and evaluate_max_cost <= args.persistent_eps:
                best_reward = evaluate_reward
                base.save_agent(
                    agent,
                    f"{model_dir}/{BEST_NAME}",
                    state_norm=state_norm,
                    reward_scaling=reward_scaling,
                )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--evaluate_freq", type=int, default=25)
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