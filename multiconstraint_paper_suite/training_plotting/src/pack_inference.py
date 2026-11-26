from pathlib import Path
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards", required=True)
    ap.add_argument("--max-costs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = np.load(a.rewards)
    c = np.load(a.max_costs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rewards=r, max_costs=c)
    print(out.resolve())


if __name__ == "__main__":
    main()
