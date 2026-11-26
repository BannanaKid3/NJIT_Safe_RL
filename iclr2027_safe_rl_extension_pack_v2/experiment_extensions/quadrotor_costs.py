from __future__ import annotations
import numpy as np

EXPECTED_COST_DIM = 4


def quadrotor_components_from_info(info: dict, cost_scale: float = 100.0) -> np.ndarray:
    if "constraint_values" not in info:
        raise KeyError(f"Missing info['constraint_values']; keys={list(info.keys())}")
    raw = np.asarray(info["constraint_values"], dtype=np.float32).reshape(-1)
    if raw.size != EXPECTED_COST_DIM:
        raise ValueError(
            f"Expected {EXPECTED_COST_DIM} Quadrotor constraint residuals, "
            f"got shape={raw.shape}, values={raw}"
        )
    return float(cost_scale) * np.maximum(raw, 0.0)


def quadrotor_combined_cost_from_info(info: dict, cost_scale: float = 100.0):
    components = quadrotor_components_from_info(info, cost_scale=cost_scale)
    return float(np.max(components)), components


def actuator_group_diagnostics(components: np.ndarray) -> np.ndarray:
    c = np.asarray(components, dtype=np.float32).reshape(-1)
    if c.size != 4:
        raise ValueError(f"Expected four components, got {c}")
    return np.array([max(c[0], c[2]), max(c[1], c[3])], dtype=np.float32)
