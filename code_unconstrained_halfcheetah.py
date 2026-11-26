import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm

from envs.half_cheetah_unconstrained import HalfCheetahUnconstrained


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ActorGaussian(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = nn.ReLU()

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        s = self.activate_func(self.fc3(s))
        return self.max_action * torch.tanh(self.mean_layer(s))

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand(mean.shape[0], -1)
        return Normal(mean, torch.exp(log_std))

    def save(self, filename):
        torch.save(self.state_dict(), filename)


class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc4 = nn.Linear(args.hidden_width, 1)
        self.activate_func = nn.ReLU()

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        s = self.activate_func(self.fc3(s))
        return self.fc4(s)

    def save(self, filename):
        torch.save(self.state_dict(), filename)


class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim), dtype=np.float32)
        self.a = np.zeros((args.batch_size, args.action_dim), dtype=np.float32)
        self.a_logprob = np.zeros((args.batch_size, args.action_dim), dtype=np.float32)
        self.r = np.zeros((args.batch_size, 1), dtype=np.float32)
        self.s_ = np.zeros((args.batch_size, args.state_dim), dtype=np.float32)
        self.dw = np.zeros((args.batch_size, 1), dtype=np.float32)
        self.done = np.zeros((args.batch_size, 1), dtype=np.float32)
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def tensors(self):
        return (
            torch.tensor(self.s, dtype=torch.float32),
            torch.tensor(self.a, dtype=torch.float32),
            torch.tensor(self.a_logprob, dtype=torch.float32),
            torch.tensor(self.r, dtype=torch.float32),
            torch.tensor(self.s_, dtype=torch.float32),
            torch.tensor(self.dw, dtype=torch.float32),
            torch.tensor(self.done, dtype=torch.float32),
        )


class PPOAgent:
    def __init__(self, args):
        self.actor = ActorGaussian(args)
        self.critic = Critic(args)
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.use_adv_norm = args.use_adv_norm
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.weight_reg = args.weight_reg

        eps = 1e-5 if args.set_adam_eps else 1e-8
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr_a,
            eps=eps,
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr_c,
            eps=eps,
        )

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()
            a = torch.clamp(a, -self.max_action, self.max_action)
            a_logprob = dist.log_prob(a)
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def evaluate(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        with torch.no_grad():
            return self.actor(s).numpy().flatten()

    def update(self, replay_buffer, training_episode):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.tensors()

        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs

            adv = []
            gae = 0.0
            for delta, d in zip(
                reversed(deltas.flatten().numpy()),
                reversed(done.flatten().numpy()),
            ):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)

            adv = torch.tensor(adv, dtype=torch.float32).view(-1, 1)

            last_linear = self.critic.fc4
            reg_norm = torch.norm(last_linear.weight, p=2)
            v_target = adv + vs + self.weight_reg * reg_norm

            if self.use_adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        for _ in range(self.K_epochs):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.batch_size)),
                self.mini_batch_size,
                False,
            ):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                a_logprob_now = dist_now.log_prob(a[index])

                ratios = torch.exp(
                    a_logprob_now.sum(1, keepdim=True)
                    - a_logprob[index].sum(1, keepdim=True)
                )

                surr1 = ratios * adv[index]
                surr2 = (
                    torch.clamp(
                        ratios,
                        1 - self.epsilon,
                        1 + self.epsilon,
                    )
                    * adv[index]
                )

                actor_loss = (
                    -torch.min(surr1, surr2)
                    - self.entropy_coef * dist_entropy
                ).mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        0.5,
                    )

                self.optimizer_actor.step()

                value = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], value)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        0.5,
                    )

                self.optimizer_critic.step()

        if self.use_lr_decay:
            fraction = 1.0 - training_episode / self.max_train_steps
            lr_a_now = self.lr_a * fraction
            lr_c_now = self.lr_c * fraction

            for group in self.optimizer_actor.param_groups:
                group["lr"] = lr_a_now

            for group in self.optimizer_critic.param_groups:
                group["lr"] = lr_c_now


def unwrap_reset(reset_result):
    result = reset_result
    while isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float64)


def evaluate_policy(env, agent, episodes):
    rewards = []
    costs = []

    for _ in range(episodes):
        s = unwrap_reset(env.reset())
        done = False
        episode_reward = 0.0
        episode_cost = 0.0

        while not done:
            action = agent.evaluate(s)
            s, reward, cost, truncated, terminated, _ = env.step(action)
            done = bool(truncated or terminated)
            episode_reward += float(reward)
            episode_cost += float(cost)

        rewards.append(episode_reward)
        costs.append(episode_cost)

    return float(np.mean(rewards)), float(np.mean(costs))


def save_agent(agent, model_dir):
    prefix = os.path.join(model_dir, "Best_Unconstrained")
    agent.actor.save(prefix + "_actor")
    agent.critic.save(prefix + "_Rcritic")


def main(args):
    env = HalfCheetahUnconstrained()
    env_evaluate = HalfCheetahUnconstrained()

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

    model_dir = f"./models/HalfCheetahUnconstrained/run{args.run}"
    data_dir = f"./data_train/HalfCheetahUnconstrained/run{args.run}"
    plot_dir = f"./plot_data/HalfCheetahUnconstrained/run{args.run}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    replay_buffer = ReplayBuffer(args)
    agent = PPOAgent(args)

    evaluate_rewards = []
    evaluate_costs = []
    episode_rewards = []
    episode_costs = []
    best_reward = float("-inf")

    print("env=HalfCheetahUnconstrained")
    print(f"run={args.run}")
    print(f"seed={args.seed}")
    print(f"state_dim={args.state_dim}")
    print(f"action_dim={args.action_dim}")
    print(f"max_action={args.max_action}")
    print(f"max_episode_steps={args.max_episode_steps}")

    for training_episode in tqdm(range(args.max_train_steps)):
        s = unwrap_reset(env.reset())
        done = False
        episode_steps = 0
        total_reward = 0.0
        total_cost = 0.0

        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)

            s_, reward, cost, truncated, terminated, _ = env.step(a)
            done = bool(truncated or terminated)

            if abs(float(cost)) > 1e-12:
                raise RuntimeError(
                    f"HalfCheetahUnconstrained returned nonzero cost: {cost}"
                )

            dw = bool(
                done
                and episode_steps != args.max_episode_steps
            )

            replay_buffer.store(
                s,
                a,
                a_logprob,
                reward,
                s_,
                dw,
                done,
            )

            s = np.asarray(s_, dtype=np.float64)
            total_reward += float(reward)
            total_cost += float(cost)

            if replay_buffer.count == args.batch_size:
                agent.update(
                    replay_buffer,
                    training_episode,
                )
                replay_buffer.count = 0

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)

        if training_episode % args.evaluate_freq == 0:
            evaluate_reward, evaluate_cost = evaluate_policy(
                env_evaluate,
                agent,
                args.evaluate_episodes,
            )

            evaluate_rewards.append(evaluate_reward)
            evaluate_costs.append(evaluate_cost)

            print(
                f"evaluation={len(evaluate_rewards)} "
                f"episode={training_episode} "
                f"reward={evaluate_reward:.6f} "
                f"cost={evaluate_cost:.6f}"
            )

            if abs(evaluate_cost) > 1e-12:
                raise RuntimeError(
                    f"Unconstrained evaluation returned nonzero cost: {evaluate_cost}"
                )

            if evaluate_reward > best_reward:
                best_reward = evaluate_reward
                save_agent(agent, model_dir)
                print(
                    f"Saved Best_Unconstrained_actor "
                    f"with reward={evaluate_reward:.6f}"
                )

            np.save(
                os.path.join(data_dir, "evaluate_rewards.npy"),
                np.asarray(evaluate_rewards),
            )
            np.save(
                os.path.join(data_dir, "evaluate_costs.npy"),
                np.asarray(evaluate_costs),
            )
            np.save(
                os.path.join(plot_dir, "episode_rewards.npy"),
                np.asarray(episode_rewards),
            )
            np.save(
                os.path.join(plot_dir, "episode_costs.npy"),
                np.asarray(episode_costs),
            )

    np.save(
        os.path.join(data_dir, "evaluate_rewards.npy"),
        np.asarray(evaluate_rewards),
    )
    np.save(
        os.path.join(data_dir, "evaluate_costs.npy"),
        np.asarray(evaluate_costs),
    )
    np.save(
        os.path.join(plot_dir, "episode_rewards.npy"),
        np.asarray(episode_rewards),
    )
    np.save(
        os.path.join(plot_dir, "episode_costs.npy"),
        np.asarray(episode_costs),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=16000)
    parser.add_argument("--evaluate_freq", type=int, default=100)
    parser.add_argument("--evaluate_episodes", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--hidden_width", type=int, default=64)
    parser.add_argument("--lr_a", type=float, default=1e-3)
    parser.add_argument("--lr_c", type=float, default=5e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=5)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--weight_reg", type=float, default=0.001)
    parser.add_argument("--use_adv_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_lr_decay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_grad_clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_orthogonal_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--set_adam_eps", action=argparse.BooleanOptionalAction, default=True)

    main(parser.parse_args())