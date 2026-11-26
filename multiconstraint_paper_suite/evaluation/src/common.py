from pathlib import Path
import csv
import json
import pickle
import re

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape)

    def __call__(self, x, update=False):
        return (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "RunningMeanStd":
            return RunningMeanStd
        if name == "Normalization":
            return Normalization
        return super().find_class(module, name)


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

    def forward(self, x):
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        x = self.activate_func(self.fc3(x))
        return self.max_action * torch.tanh(self.mean_layer(x))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def torch_load(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "actor_state_dict" in state:
        state = state["actor_state_dict"]
    out = {}
    for key, value in state.items():
        name = key
        for prefix in ("module.", "actor."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        out[name] = value
    return out


def load_actor(path, env, device="cpu"):
    state = clean_state_dict(torch_load(path, device))
    needed = {"fc1.weight", "fc2.weight", "fc3.weight", "mean_layer.weight"}
    missing = needed - set(state)
    if missing:
        raise ValueError(f"Unexpected actor checkpoint {path}: {sorted(missing)}")
    state_dim = int(state["fc1.weight"].shape[1])
    hidden = int(state["fc1.weight"].shape[0])
    action_dim = int(state["mean_layer.weight"].shape[0])
    env_state = int(np.prod(env.observation_space.shape))
    env_action = int(np.prod(env.action_space.shape))
    if state_dim != env_state or action_dim != env_action:
        raise ValueError(f"Checkpoint dimensions {(state_dim, action_dim)} do not match env {(env_state, env_action)}")
    max_action = float(np.asarray(env.action_space.high).reshape(-1)[0])
    actor = ActorGaussian(state_dim, hidden, action_dim, max_action)
    actor.load_state_dict(state, strict=True)
    actor.to(device)
    actor.eval()
    return actor


def normalizer_candidates(actor_path):
    actor_path = Path(actor_path)
    names = [actor_path.name.replace("_actor", "_state_norm", 1)]
    names.append(re.sub(r"_state_norm_\d+$", "_state_norm", names[0]))
    return [actor_path.with_name(name) for name in dict.fromkeys(names)]


def load_normalizer(actor_path):
    for path in normalizer_candidates(actor_path):
        if path.exists():
            with path.open("rb") as handle:
                return CompatUnpickler(handle).load()
    return None


def resolve_checkpoint(directory):
    directory = Path(directory)
    for name in ("Best_RCAC_actor", "Best_CombinedMaxCost_actor"):
        path = directory / name
        if path.is_file():
            return path
    numbered = []
    for path in directory.glob("Best_RCAC_actor_*"):
        match = re.search(r"_(\d+)$", path.name)
        if path.is_file() and match:
            numbered.append((int(match.group(1)), path))
    if numbered:
        return max(numbered, key=lambda item: item[0])[1]
    candidates = sorted(path for path in directory.glob("Best*_actor*") if path.is_file())
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No Best actor checkpoint in {directory}")


def reset_obs(env, seed):
    out = env.reset(seed=seed)
    while isinstance(out, tuple):
        out = out[0]
    return np.asarray(out, dtype=np.float32).reshape(-1)


def step_env(env, action):
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
        return np.asarray(obs, dtype=np.float32).reshape(-1), float(reward), bool(done), info, None
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return np.asarray(obs, dtype=np.float32).reshape(-1), float(reward), bool(terminated or truncated), info, None
    if len(out) == 6:
        obs, reward, cost, truncated, terminated, info = out
        return np.asarray(obs, dtype=np.float32).reshape(-1), float(reward), bool(terminated or truncated), info, cost
    raise RuntimeError(f"Unsupported env.step output length: {len(out)}")


def policy_action(actor, obs, normalizer, device="cpu"):
    x = obs if normalizer is None else normalizer(obs, update=False)
    x = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return actor(x).cpu().numpy().reshape(-1)


def save_seed_result(root, label, seed, result):
    safe = label.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
    out = Path(root) / "raw" / safe / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for key in ("rewards", "component_max_costs", "component_total_costs"):
        np.save(out / f"{key}.npy", np.asarray(result[key]))
    return out


def write_summary(root, methods, threshold):
    rows = []
    for label, entries in methods.items():
        for entry in entries:
            rewards = np.asarray(entry["result"]["rewards"], dtype=float)
            components = np.asarray(entry["result"]["component_max_costs"], dtype=float)
            max_costs = np.max(components, axis=1)
            rows.append({
                "method": label,
                "training_seed": entry["seed"],
                "checkpoint": str(entry["checkpoint"]),
                "episodes": len(rewards),
                "reward_mean": float(np.mean(rewards)),
                "reward_sd": float(np.std(rewards, ddof=0)),
                "max_cost_mean": float(np.mean(max_costs)),
                "max_cost_sd": float(np.std(max_costs, ddof=0)),
                "feasible_episode_fraction": float(np.mean(max_costs <= threshold)),
            })
    path = Path(root) / "seed_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    grouped = []
    for label in methods:
        subset = [row for row in rows if row["method"] == label]
        grouped.append({
            "method": label,
            "training_seeds": len(subset),
            "reward_mean_across_seeds": float(np.mean([row["reward_mean"] for row in subset])),
            "reward_sd_across_seeds": float(np.std([row["reward_mean"] for row in subset], ddof=0)),
            "max_cost_mean_across_seeds": float(np.mean([row["max_cost_mean"] for row in subset])),
            "max_cost_sd_across_seeds": float(np.std([row["max_cost_mean"] for row in subset], ddof=0)),
            "feasible_fraction_across_seeds": float(np.mean([row["feasible_episode_fraction"] for row in subset])),
        })
    path2 = Path(root) / "method_summary.csv"
    with path2.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0].keys()))
        writer.writeheader()
        writer.writerows(grouped)
    return path, path2
