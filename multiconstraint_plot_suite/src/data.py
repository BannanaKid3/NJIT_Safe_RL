from pathlib import Path
import json
import numpy as np


def load_run(path, max_episodes=None):
    p = Path(path)
    r = np.load(p / "episode_rewards.npy").astype(float).reshape(-1)
    c = np.load(p / "episode_max_costs.npy").astype(float)
    if max_episodes:
        r = r[:max_episodes]
        c = c[:max_episodes]
    return r, c


def scalar_cost(c):
    c = np.asarray(c, dtype=float)
    if c.ndim == 1:
        return c
    return np.max(c, axis=1)


def align(xs):
    n = min(len(x) for x in xs)
    return [np.asarray(x)[:n] for x in xs]


def stats(xs):
    xs = align(xs)
    a = np.stack(xs, axis=0)
    return a.mean(axis=0), a.std(axis=0, ddof=0), len(xs)


def run_info(path):
    p = Path(path)
    r, c = load_run(p)
    out = {"path": str(p), "episodes": int(len(r)), "reward_shape": list(r.shape), "max_cost_shape": list(c.shape)}
    e = p.parent.parent.parent / "data_train" if False else None
    return out


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
