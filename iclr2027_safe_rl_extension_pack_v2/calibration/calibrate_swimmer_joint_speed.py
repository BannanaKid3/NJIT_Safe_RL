from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys
import numpy as np
import torch


def unwrap_obs(obj):
    """Unwrap the nested reset formats used by the repo's mixed Gym versions."""
    cur = obj
    for _ in range(5):
        if isinstance(cur, tuple):
            cur = cur[0]
            continue
        break
    arr = np.asarray(cur, dtype=np.float32)
    if arr.ndim == 0:
        raise RuntimeError(f"Reset unwrapped to a scalar instead of a state vector: {obj!r}")
    return arr.reshape(-1)


def discover_prefix(repo: Path) -> Path:
    candidates = sorted((repo / "models" / "SwimmerWithPos").glob("run*/Best_RCAC_actor"))
    if len(candidates) != 1:
        shown = "\n".join(f"  {p}" for p in candidates[:20]) or "  (none)"
        raise SystemExit(
            "Could not uniquely auto-select a nominal Swimmer Best_RCAC actor. "
            "Pass --model-prefix explicitly. Candidates:\n" + shown
        )
    actor = candidates[0]
    return Path(str(actor)[:-len("_actor")])


def main():
    ap = argparse.ArgumentParser(description="Calibrate Swimmer internal-joint-speed safety threshold from a nominal trained policy.")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model-prefix", default=None, help="Example: models/SwimmerWithPos/run2/Best_RCAC")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--threshold-quantile", type=float, default=0.99)
    ap.add_argument("--scale-quantile", type=float, default=0.995)
    ap.add_argument("--target-cost-at-scale-quantile", type=float, default=0.1, help="Makes the scale quantile map to this normalized joint-speed cost")
    ap.add_argument("--hidden-width", type=int, default=64)
    ap.add_argument("--action-threshold", type=float, default=0.5)
    ap.add_argument("--action-scale", type=float, default=1.0, help="1.0 preserves the existing repo Swimmer action-excess cost magnitude")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from envs.swimmer import SwimmerWithPos
    from code_ipm_rcmdp_rcrl_max_swimmer import Actor_Gaussian

    prefix = Path(args.model_prefix) if args.model_prefix else discover_prefix(repo)
    if not prefix.is_absolute():
        prefix = repo / prefix
    actor_file = Path(str(prefix) + "_actor")
    if not actor_file.exists():
        raise SystemExit(f"Actor checkpoint does not exist: {actor_file}")

    env = SwimmerWithPos(sigma_viscosity=0.0, max_steps=1000)
    raw_reset = env.reset(seed=args.seed)
    obs = unwrap_obs(raw_reset)

    actor_args = SimpleNamespace(
        state_dim=int(obs.size),
        hidden_width=args.hidden_width,
        action_dim=int(env.action_space.shape[0]),
        max_action=float(env.action_space.high[0]),
        use_tanh=True,
        use_orthogonal_init=False,
    )
    actor = Actor_Gaussian(actor_args)
    actor.load(str(actor_file), device="cpu")
    actor.eval()

    joint_speed_peaks = []
    action_peaks = []
    rewards = []
    total_steps = 0

    for ep in range(args.episodes):
        obs = unwrap_obs(env.reset(seed=args.seed + ep))
        done = False
        ep_reward = 0.0
        while not done:
            state = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = actor.forward(state).cpu().numpy().reshape(-1)
            step = env.step(action)
            if len(step) != 6:
                raise RuntimeError(f"Expected 6-return Swimmer step, got {len(step)} values")
            next_obs, reward, _legacy_cost, truncated, terminated, _info = step
            qvel = np.asarray(env.sim.data.qvel.flat, dtype=np.float64).reshape(-1)
            if qvel.size < 5:
                raise RuntimeError(f"Expected qvel with >=3 entries, got {qvel}")
            joint_speed_peaks.append(float(np.max(np.abs(qvel[3:]))))
            action_peaks.append(float(np.max(np.abs(action))))
            ep_reward += float(reward)
            total_steps += 1
            done = bool(truncated or terminated)
            obs = unwrap_obs(next_obs)
        rewards.append(ep_reward)

    speeds = np.asarray(joint_speed_peaks, dtype=np.float64)
    if speeds.size < 100:
        raise RuntimeError(f"Only collected {speeds.size} samples; calibration is too small")
    threshold = float(np.quantile(speeds, args.threshold_quantile))
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError(f"Invalid calibrated joint-speed threshold: {threshold}")
    # Match the HalfCheetah normalization philosophy:
    # normalized positive excess divided by the behavior-derived threshold.
    scale = threshold

    out = repo / "experiment_extensions" / "swimmer_thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "CALIBRATED",
        "action_threshold": float(args.action_threshold),
        "action_scale": float(args.action_scale),
        "joint_speed_threshold": threshold,
        "joint_speed_scale": scale,
        "calibration": {
            "method": "nominal deterministic policy rollout",
            "threshold_quantile": args.threshold_quantile,
            "scale_definition": "joint_speed_scale = joint_speed_threshold; normalized positive excess / threshold",
            "episodes": args.episodes,
            "samples": int(speeds.size),
            "seed_start": args.seed,
            "checkpoint_prefix": str(prefix),
            "joint_speed_q50": float(np.quantile(speeds, 0.50)),
            "joint_speed_q95": float(np.quantile(speeds, 0.95)),
            "joint_speed_q995": float(np.quantile(speeds, 0.995)),
            "joint_speed_max": float(np.max(speeds)),
            "action_peak_q95": float(np.quantile(action_peaks, 0.95)),
            "action_peak_max": float(np.max(action_peaks)),
            "mean_episode_reward": float(np.mean(rewards)),
            "total_steps": total_steps
        }
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("CALIBRATION COMPLETE")
    print("checkpoint:", prefix)
    print("joint_speed_threshold:", threshold)
    print("joint_speed_scale:", scale)
    print("wrote:", out)


if __name__ == "__main__":
    main()
