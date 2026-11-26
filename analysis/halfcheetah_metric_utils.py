from pathlib import Path

import numpy as np


JOINT_NAMES = (
    "back_thigh",
    "back_shin",
    "back_foot",
    "front_thigh",
    "front_shin",
    "front_foot",
)


def summarize_metrics(
    torso_pitch,
    joint_speeds,
    episode_id,
    step_in_episode,
):
    torso_pitch = np.asarray(torso_pitch, dtype=np.float64)
    torso_pitch_abs = np.abs(torso_pitch)
    joint_speeds = np.asarray(joint_speeds, dtype=np.float64)
    episode_id = np.asarray(episode_id, dtype=np.int64)
    step_in_episode = np.asarray(step_in_episode, dtype=np.int64)
    if torso_pitch.ndim != 1:
        raise ValueError("torso_pitch must be one-dimensional")
    if joint_speeds.ndim != 2 or joint_speeds.shape[1] != 6:
        raise ValueError("joint_speeds must have shape (N, 6)")
    if torso_pitch.shape[0] != joint_speeds.shape[0]:
        raise ValueError("pitch and joint-speed sample counts differ")
    if episode_id.shape != torso_pitch.shape:
        raise ValueError("episode_id shape does not match pitch")
    if step_in_episode.shape != torso_pitch.shape:
        raise ValueError("step_in_episode shape does not match pitch")
    fastest_joint_index = np.argmax(joint_speeds, axis=1)
    fastest_joint_counts = np.bincount(
        fastest_joint_index,
        minlength=len(JOINT_NAMES),
    )
    return {
        "sample_count": np.asarray(torso_pitch.size, dtype=np.int64),
        "pitch_mean_signed": np.asarray(np.mean(torso_pitch)),
        "pitch_mean_abs": np.asarray(np.mean(torso_pitch_abs)),
        "pitch_std_abs": np.asarray(np.std(torso_pitch_abs)),
        "pitch_p50_abs": np.asarray(np.percentile(torso_pitch_abs, 50)),
        "pitch_p90_abs": np.asarray(np.percentile(torso_pitch_abs, 90)),
        "pitch_p95_abs": np.asarray(np.percentile(torso_pitch_abs, 95)),
        "pitch_p99_abs": np.asarray(np.percentile(torso_pitch_abs, 99)),
        "pitch_max_abs": np.asarray(np.max(torso_pitch_abs)),
        "joint_mean_abs": np.mean(joint_speeds, axis=0),
        "joint_std_abs": np.std(joint_speeds, axis=0),
        "joint_p50_abs": np.percentile(joint_speeds, 50, axis=0),
        "joint_p90_abs": np.percentile(joint_speeds, 90, axis=0),
        "joint_p95_abs": np.percentile(joint_speeds, 95, axis=0),
        "joint_p99_abs": np.percentile(joint_speeds, 99, axis=0),
        "joint_max_abs": np.max(joint_speeds, axis=0),
        "step_max_joint_speed_mean": np.asarray(
            np.mean(np.max(joint_speeds, axis=1))
        ),
        "step_max_joint_speed_p95": np.asarray(
            np.percentile(np.max(joint_speeds, axis=1), 95)
        ),
        "step_max_joint_speed_p99": np.asarray(
            np.percentile(np.max(joint_speeds, axis=1), 99)
        ),
        "step_max_joint_speed_max": np.asarray(
            np.max(np.max(joint_speeds, axis=1))
        ),
        "fastest_joint_counts": fastest_joint_counts,
    }


def save_metrics(
    output_path,
    torso_pitch,
    joint_velocities,
    joint_speeds,
    actions,
    rewards,
    costs,
    episode_id,
    step_in_episode,
    episode_rewards,
    episode_costs,
    episode_max_costs,
    model_prefix,
    environment_name,
    seed,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torso_pitch = np.asarray(torso_pitch, dtype=np.float64)
    joint_velocities = np.asarray(joint_velocities, dtype=np.float64)
    joint_speeds = np.asarray(joint_speeds, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    episode_id = np.asarray(episode_id, dtype=np.int64)
    step_in_episode = np.asarray(step_in_episode, dtype=np.int64)
    episode_rewards = np.asarray(episode_rewards, dtype=np.float64)
    episode_costs = np.asarray(episode_costs, dtype=np.float64)
    episode_max_costs = np.asarray(episode_max_costs, dtype=np.float64)
    summary = summarize_metrics(
        torso_pitch,
        joint_speeds,
        episode_id,
        step_in_episode,
    )
    payload = {
        "joint_names": np.asarray(JOINT_NAMES, dtype="U32"),
        "torso_pitch": torso_pitch,
        "torso_pitch_abs": np.abs(torso_pitch),
        "joint_velocities": joint_velocities,
        "joint_speeds": joint_speeds,
        "actions": actions,
        "rewards": rewards,
        "costs": costs,
        "episode_id": episode_id,
        "step_in_episode": step_in_episode,
        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
        "episode_max_costs": episode_max_costs,
        "model_prefix": np.asarray(str(model_prefix)),
        "environment_name": np.asarray(str(environment_name)),
        "seed": np.asarray(seed, dtype=np.int64),
    }
    payload.update({f"summary_{key}": value for key, value in summary.items()})
    np.savez_compressed(output_path, **payload)
    return output_path