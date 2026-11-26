from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent


def complete(paths, repo, n):
    out = []
    for rel in paths:
        p = repo / rel
        try:
            r = np.load(p / "episode_rewards.npy", mmap_mode="r")
            c = np.load(p / "episode_max_costs.npy", mmap_mode="r")
        except Exception:
            return None
        if len(r) < n or len(c) < n:
            return None
        out.append(str(p))
    return out


def pick(candidates, repo, n):
    for paths in candidates:
        x = complete(paths, repo, n)
        if x:
            return x
    return None


def auto_namespace(repo, names, n):
    best = []
    for name in names:
        root = repo / "plot_data" / name
        if not root.exists():
            continue
        cur = []
        max_len = 0
        for p in sorted(root.glob("run*")):
            try:
                r = np.load(p / "episode_rewards.npy", mmap_mode="r")
                c = np.load(p / "episode_max_costs.npy", mmap_mode="r")
            except Exception:
                continue
            k = min(len(r), len(c))
            if k > max_len:
                max_len = k
                cur = [str(p)]
            elif k == max_len:
                cur.append(str(p))
        if max_len >= n and cur:
            best = cur
            break
    return best or None


def run_plot(repo, env, name, a, b, threshold, n, components=False, fixed=None):
    out = repo / "multiconstraint_figures" / env / name
    cmd = [sys.executable, str(HERE / "plot_training.py"), "--method", a[0], *a[1], "--method", b[0], *b[1], "--threshold", str(threshold), "--max-episodes", str(n), "--window", "80", "--outdir", str(out), "--prefix", name]
    if components:
        cmd.append("--components")
    if fixed:
        cmd += ["--shade-min", str(fixed[0]), "--shade-max", str(fixed[1])]
    subprocess.run(cmd, check=True, cwd=repo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--only", choices=["all", "ablation", "robustness"], default="all")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    n = 10000
    q_m_np = pick([["plot_data/Quadrotor/run9212", "plot_data/Quadrotor/run9215"]], repo, n)
    q_c_np = pick([["plot_data/QuadrotorCombined/run9312", "plot_data/QuadrotorCombined/run9315"], ["plot_data/QuadrotorCombined/run201", "plot_data/QuadrotorCombined/run205"]], repo, n)
    q_m_pr = pick([["plot_data/QuadrotorMultiPerturbedRobust/run9412", "plot_data/QuadrotorMultiPerturbedRobust/run9415"], ["plot_data/QuadrotorMultiPerturbedRobust/run301", "plot_data/QuadrotorMultiPerturbedRobust/run305"]], repo, n)
    q_c_pr = pick([["plot_data/QuadrotorCombinedPerturbedRobust/run9512", "plot_data/QuadrotorCombinedPerturbedRobust/run9515"], ["plot_data/QuadrotorCombinedPerturbedRobust/run401", "plot_data/QuadrotorCombinedPerturbedRobust/run405"]], repo, n)
    s_m_np = pick([["plot_data/SwimmerMulti/run11612", "plot_data/SwimmerMulti/run11615"], ["plot_data/SwimmerMulti/run501", "plot_data/SwimmerMulti/run505"]], repo, n)
    s_c_np = pick([["plot_data/SwimmerCombined/run11712", "plot_data/SwimmerCombined/run11715"], ["plot_data/SwimmerCombined/run601", "plot_data/SwimmerCombined/run605"]], repo, n)
    s_m_pr = pick([["plot_data/SwimmerMultiPerturbedRobust/run11812", "plot_data/SwimmerMultiPerturbedRobust/run11815"], ["plot_data/SwimmerMultiPerturbedRobust/run701", "plot_data/SwimmerMultiPerturbedRobust/run705"]], repo, n)
    s_c_pr = pick([["plot_data/SwimmerCombinedPerturbedRobust/run11912", "plot_data/SwimmerCombinedPerturbedRobust/run11915"], ["plot_data/SwimmerCombinedPerturbedRobust/run801", "plot_data/SwimmerCombinedPerturbedRobust/run805"]], repo, n)
    h_m_np = auto_namespace(repo, ["HalfCheetahMultiConstraintCost"], n)
    h_c_np = auto_namespace(repo, ["HalfCheetahCombinedMaxCost"], n)
    h_m_pr = auto_namespace(repo, ["HalfCheetahMultiConstraintCostPerturbed", "HalfCheetahMultiPerturbedRobust", "HalfCheetahMultiConstraintPerturbed"], n)
    h_c_pr = auto_namespace(repo, ["HalfCheetahCombinedMaxCostPerturbed", "HalfCheetahCombinedPerturbedRobust"], n)
    groups = {
        "quadrotor": {"threshold": 0.5, "fixed": None, "m_np": q_m_np, "c_np": q_c_np, "m_pr": q_m_pr, "c_pr": q_c_pr},
        "halfcheetah": {"threshold": 0.1, "fixed": (-1.0, 12.5), "m_np": h_m_np, "c_np": h_c_np, "m_pr": h_m_pr, "c_pr": h_c_pr},
        "swimmer": {"threshold": 0.1, "fixed": (-1.0, 12.5), "m_np": s_m_np, "c_np": s_c_np, "m_pr": s_m_pr, "c_pr": s_c_pr},
    }
    selected = {}
    for env, g in groups.items():
        selected[env] = {k: v for k, v in g.items() if k not in {"fixed"}}
        if a.only in {"all", "ablation"}:
            if g["m_np"] and g["c_np"]:
                run_plot(repo, env, "ablation_np", ("Multi Constraint", g["m_np"]), ("Combined Cost", g["c_np"]), g["threshold"], n, True, g["fixed"])
            if g["m_pr"] and g["c_pr"]:
                run_plot(repo, env, "ablation_pr", ("Multi Constraint", g["m_pr"]), ("Combined Cost", g["c_pr"]), g["threshold"], n, True, g["fixed"])
        if a.only in {"all", "robustness"} and g["m_np"] and g["m_pr"]:
            run_plot(repo, env, "multi_np_vs_pr", ("Multi Constraint (P+R)", g["m_pr"]), ("Multi Constraint (NP)", g["m_np"]), g["threshold"], n, False, g["fixed"])
    out = repo / "multiconstraint_figures" / "selected_runs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
