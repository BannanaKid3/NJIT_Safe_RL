from __future__ import annotations


def assert_nonempty_disturbances(env, label: str = "environment") -> None:
    disturbances = getattr(env, "disturbances", None)
    if disturbances is None:
        raise RuntimeError(f"{label}: env.disturbances is None; refusing to label this run perturbed/P+R")
    try:
        empty = len(disturbances) == 0
    except TypeError:
        empty = not bool(disturbances)
    if empty:
        raise RuntimeError(f"{label}: env.disturbances is empty; refusing to label this run perturbed/P+R")
    print(f"[PERTURBATION CHECK] {label} disturbances active: {disturbances}")
