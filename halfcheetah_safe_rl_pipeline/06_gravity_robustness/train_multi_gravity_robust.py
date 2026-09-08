import argparse
from functools import partial
from pathlib import Path

import code_ipm_rcmdp_rcrl_max_hc_multi_constraint as base
from envs.half_cheetah_gravity_perturbed_costs import (
    HalfCheetahMultiConstraintCostGravityPerturbed,
)
from gravity_robustness_common import (
    add_training_arguments,
    ensure_output_roots,
    multi_gravity_version,
    validate_training_arguments,
    write_robust_config,
)


ENVIRONMENT_NAME = "HalfCheetahMultiConstraintCost"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train the separate pitch/joint RCAC policy with independently "
            "resampled Gaussian gravity perturbations before every MuJoCo step."
        )
    )
    add_training_arguments(parser, multi_constraint=True)
    return validate_training_arguments(parser.parse_args())


def main():
    args = parse_arguments()
    args.env = ENVIRONMENT_NAME
    version = multi_gravity_version(args.sigma_gravity, args.rho)
    base.HalfCheetahMultiConstraintCost = partial(
        HalfCheetahMultiConstraintCostGravityPerturbed,
        sigma_gravity=args.sigma_gravity,
    )
    base.MULTI_CONSTRAINT_COST_VERSION = version
    ensure_output_roots()
    config_path = write_robust_config(
        Path.cwd(),
        ENVIRONMENT_NAME,
        version,
        args,
        method="multi_constraint",
    )
    print(
        f"run={args.run} seed={args.seed} sigma_gravity={args.sigma_gravity} "
        f"rho={args.rho} version={version}"
    )
    print(f"robust configuration: {config_path}")
    base.main(args, run_number=args.run)


if __name__ == "__main__":
    main()
