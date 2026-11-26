from pathlib import Path
import argparse
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="multiconstraint_figures/inventory.json")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    rows = []
    root = repo / "plot_data"
    if root.exists():
        for p in sorted(root.glob("*/run*")):
            rp = p / "episode_rewards.npy"
            cp = p / "episode_max_costs.npy"
            if not rp.exists() or not cp.exists():
                continue
            try:
                r = np.load(rp, mmap_mode="r")
                c = np.load(cp, mmap_mode="r")
            except Exception:
                continue
            rows.append({"namespace": p.parent.name, "run": p.name, "episodes": int(len(r)), "reward_shape": list(r.shape), "max_cost_shape": list(c.shape), "path": str(p.relative_to(repo))})
    out = repo / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for x in rows:
        print(f"{x['namespace']:<38} {x['run']:<12} ep={x['episodes']:<6} cost={tuple(x['max_cost_shape'])}")
    print(out)


if __name__ == "__main__":
    main()
