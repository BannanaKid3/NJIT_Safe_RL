from pathlib import Path
import sys
import tempfile
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_extensions.quadrotor_costs import quadrotor_components_from_info, quadrotor_combined_cost_from_info
from experiment_extensions.swimmer_costs import swimmer_components, load_thresholds
from paper_plotting.paper_style_plots import smooth_preserve_prefix


def main():
    info = {"constraint_values": [-0.1, 0.02, 0.03, 0.04]}
    c = quadrotor_components_from_info(info, 100.0)
    assert np.allclose(c, [0, 2, 3, 4])
    scalar, parts = quadrotor_combined_cost_from_info(info, 100.0)
    assert scalar == 4.0 and np.allclose(parts, c)

    th = {"action_threshold": 0.5, "action_scale": 1.0, "joint_speed_threshold": 2.0, "joint_speed_scale": 1.0}
    s = swimmer_components([0.75, 0.1], [9, 8, 2.5, -1.0, 0.2], th)
    assert np.allclose(s, [0.25, 0.5])

    x = np.arange(20, dtype=float)
    y = smooth_preserve_prefix(x, window=10, prefix=10)
    assert np.allclose(y[:10], x[:10])
    assert np.isclose(y[10], np.mean(x[1:11]))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.json"
        p.write_text(json.dumps({"status":"CALIBRATED", **th}), encoding="utf-8")
        assert load_thresholds(p)["joint_speed_threshold"] == 2.0

    print("extension logic tests passed")


if __name__ == "__main__":
    main()
