import argparse
from functools import partial
from pathlib import Path

import code_ipm_rcmdp_rcrl_max_hc as base
from envs.half_cheetah_gravity_perturbed_costs import (
    HalfCheetahCombinedMaxCostGravityPerturbed,
)
from gravity_robustness_common import (
    add_training_arguments,
    ensure_output_roots,
    scalar_gravity_version,
    validate_training_arguments,
    write_robust_config,
)


ENVIRONMENT_NAME = "HalfCheetahCombinedMaxCost"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train the scalar combined-cost RCAC policy with independently "
            "resampled Gaussian gravity perturbations before every MuJoCo step."
        )
    )
    add_training_arguments(parser, multi_constraint=False)
    return validate_training_arguments(parser.parse_args())


def main():
    args = parse_arguments()
    args.env = ENVIRONMENT_NAME
    version = scalar_gravity_version(args.sigma_gravity, args.rho)
    environment_factory = partial(
        HalfCheetahCombinedMaxCostGravityPerturbed,
        sigma_gravity=args.sigma_gravity,
    )
    base.CUSTOM_ENV_FACTORIES[ENVIRONMENT_NAME] = environment_factory
    base.CUSTOM_ENV_METADATA[ENVIRONMENT_NAME] = {
        "cost_version": version,
        "cost_type": "pitch_and_joint_speed",
        "aggregation": "maximum_normalized_violation",
    }
    ensure_output_roots()
    config_path = write_robust_config(
        Path.cwd(),
        ENVIRONMENT_NAME,
        version,
        args,
        method="scalar_combined",
    )
    print(
        f"run={args.run} seed={args.seed} sigma_gravity={args.sigma_gravity} "
        f"rho={args.rho} version={version}"
    )
    print(f"robust configuration: {config_path}")
    base.main(args, run_number=args.run)


if __name__ == "__main__":
    main()

