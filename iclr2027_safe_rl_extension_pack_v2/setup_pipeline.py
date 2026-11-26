from __future__ import annotations
from pathlib import Path
import argparse
import re
import shutil
import json

PACK = Path(__file__).resolve().parent


def copy_extensions(repo: Path):
    dst = repo / 'experiment_extensions'
    dst.mkdir(parents=True, exist_ok=True)
    for src in (PACK / 'experiment_extensions').iterdir():
        if not src.is_file():
            continue
        target = dst / src.name
        # A calibrated threshold file is experimental output. Never destroy it
        # just because setup is re-run to switch a Quadrotor profile.
        if src.name == 'swimmer_thresholds.json' and target.exists():
            try:
                old = json.loads(target.read_text(encoding='utf-8'))
                if old.get('status') == 'CALIBRATED':
                    print('PRESERVED calibrated', target)
                    continue
            except Exception:
                pass
        shutil.copy2(src, target)


def replace_required(text: str, old: str, new: str, expected: int, label: str) -> str:
    n = text.count(old)
    if n != expected:
        raise RuntimeError(f'{label}: expected {expected} exact matches, found {n}. Upstream source likely changed.')
    return text.replace(old, new)


def add_import(text: str, import_line: str) -> str:
    anchor = 'from safe_control_gym.utils.registration import make\n'
    if import_line + '\n' in text:
        return text
    if anchor not in text:
        raise RuntimeError('Could not find Safe-Control-Gym import anchor')
    return text.replace(anchor, anchor + import_line + '\n', 1)


def namespace_outputs(text: str, tag: str) -> str:
    text = text.replace('model_dir = f"./models/{args.env}/run{run_number}/"', f'model_dir = f"./models/{tag}/run{{run_number}}/"')
    text = text.replace('data_train_dir = f"./data_train/{args.env}/run{run_number}/"', f'data_train_dir = f"./data_train/{tag}/run{{run_number}}/"')
    text = text.replace('plot_data_dir = f"./plot_data/{args.env}/run{run_number}/"', f'plot_data_dir = f"./plot_data/{tag}/run{{run_number}}/"')
    text = text.replace('log_dir=f"runs/RNAC/env_{args.env}_', f'log_dir=f"runs/RNAC/env_{tag}_')
    return text


def patch_quad_combined(src: str, perturbed: bool) -> str:
    t = add_import(src, 'from experiment_extensions.quadrotor_costs import quadrotor_combined_cost_from_info')
    # Match both current active weighted-sum expressions, one in evaluation and one in training.
    pattern = re.compile(
        r'^(?P<indent>[ \t]*)c\s*=\s*args\.omega1\s*\*\s*max\(0,\s*info\[(?:"|\')constraint_values(?:"|\')\]\[0\]\)\s*\+\s*args\.omega2\s*\*\s*max\(0,\s*info\[(?:"|\')constraint_values(?:"|\')\]\[1\]\).*\n'
        r'[ \t]*#shilpa cost scale\n'
        r'[ \t]*c\s*=\s*c\s*\*\s*args\.cost_scale\s*$',
        re.M,
    )
    def combined_repl(match):
        indent = match.group('indent')
        return indent + 'c, component_costs = quadrotor_combined_cost_from_info(info, args.cost_scale)'
    t, n = pattern.subn(combined_repl, t)
    if n != 2:
        raise RuntimeError(f'Quad Combined cost patch expected 2 active weighted-sum blocks, found {n}')
    tag = 'QuadrotorCombinedPerturbedRobust' if perturbed else 'QuadrotorCombined'
    t = namespace_outputs(t, tag)
    if perturbed:
        t = replace_required(t,
            "overrides=['./envs/env_configs/constrained_tracking_reset.yaml']",
            "overrides=['./envs/env_configs/constrained_tracking_train_perturbed.yaml']", 1,
            'Quad Combined robust train YAML')
        t = replace_required(t,
            "overrides=['./envs/env_configs/constrained_tracking_eval.yaml']",
            "overrides=['./envs/env_configs/constrained_tracking_eval_perturbed.yaml']", 1,
            'Quad Combined robust eval YAML')
        t = add_import(t, 'from experiment_extensions.quadrotor_disturbance_guard import assert_nonempty_disturbances')
        anchor = "        env_reset = make('quadrotor', **config.quadrotor_config)\n"
        guard = anchor + "        assert_nonempty_disturbances(env, 'Quadrotor Combined training')\n        assert_nonempty_disturbances(env_evaluate, 'Quadrotor Combined evaluation')\n"
        if anchor not in t:
            raise RuntimeError('Quad Combined guard anchor missing')
        t = t.replace(anchor, guard, 1)
    return t


def patch_quad_multi_perturbed(src: str) -> str:
    t = replace_required(src,
        "overrides=['./envs/env_configs/constrained_tracking_reset.yaml']",
        "overrides=['./envs/env_configs/constrained_tracking_train_perturbed.yaml']", 1,
        'Quad Multi robust train YAML')
    t = replace_required(t,
        "overrides=['./envs/env_configs/constrained_tracking_eval.yaml']",
        "overrides=['./envs/env_configs/constrained_tracking_eval_perturbed.yaml']", 1,
        'Quad Multi robust eval YAML')
    t = add_import(t, 'from experiment_extensions.quadrotor_disturbance_guard import assert_nonempty_disturbances')
    anchor = "        env_reset = make('quadrotor', **config.quadrotor_config)\n"
    guard = anchor + "        assert_nonempty_disturbances(env, 'Quadrotor Multi training')\n        assert_nonempty_disturbances(env_evaluate, 'Quadrotor Multi evaluation')\n"
    if anchor not in t:
        raise RuntimeError('Quad Multi guard anchor missing')
    t = t.replace(anchor, guard, 1)
    return namespace_outputs(t, 'QuadrotorMultiPerturbedRobust')


def patch_swimmer_multi(src: str, perturbed: bool) -> str:
    t = replace_required(
        src,
        '# from envs.swimmer import SwimmerWithPos\n',
        'from experiment_extensions.swimmer_multi_constraint_env import SwimmerMultiConstraintEnv as SwimmerWithPos\n',
        1,
        'Swimmer Multi import',
    )

    # Evaluation: current Quad trainer expects a 4-return step and rebuilds c from info.
    old_eval = '''            # s_, r, c, truncated, terminated, info = env.step(action)\n            s_, r, done, info = env.step(action)\n            # print("Rendering environment...")\n            env.render()\n\n            # c = np.max(info.get('constraint_values', 0.))\n            # c= 0.0\n            # c = args.omega1*max(0, info["constraint_values"][0]) + args.omega2*max(0, info["constraint_values"][1])  # Assuming info contains constraint values\n            #shilpa multi constraint\n            # c = info["constraint_values"]  # Assuming info contains constraint values\n            c = args.cost_scale * (np.asarray(info["constraint_values"], dtype=np.float32).reshape(-1))\n            c = np.maximum(c, 0.0)'''
    new_eval = '''            s_, r, c, truncated, terminated, info = env.step(action)\n            done = bool(truncated or terminated)\n            c = np.asarray(c, dtype=np.float32).reshape(-1)\n            if c.size != args.cost_dim:\n                raise ValueError(f"Expected {args.cost_dim} Swimmer costs, got {c}")'''
    t = replace_required(t, old_eval, new_eval, 1, 'Swimmer Multi evaluation step')

    old_train = '''                s_, r, done, info = env.step(action)\n                # c = np.max(info.get('constraint_values', 0.))\n                # print(info['constraint_values'].shape)  \n                # print(info['constraint_values'])\n                # c = 0.0\n                # config[quadrotor_config]["constraints"][0]["lower_bounds"] and config[quadrotor_config]["constraints"][0]["upper_bounds"]\n                # c  = args.omega1 * max(0, info['constraint_values'][0])+ args.omega2 * max(0, info['constraint_values'][1])\n                c = info["constraint_values"] # Assuming info contains constraint values\n                # print("training cost =", c)\n                total_reward += r\n                #shilps multi constraint\n                # total_cost += c\n                # max_cost = max(max_cost, c)\n                c = args.cost_scale*( np.asarray(info["constraint_values"], dtype=np.float32).reshape(-1))\n                c = np.maximum(c, 0.0)\n                total_cost += c\n                max_cost = np.maximum(max_cost, c)'''
    new_train = '''                s_, r, c, truncated, terminated, info = env.step(action)\n                done = bool(truncated or terminated)\n                c = np.asarray(c, dtype=np.float32).reshape(-1)\n                if c.size != args.cost_dim:\n                    raise ValueError(f"Expected {args.cost_dim} Swimmer costs, got {c}")\n                total_reward += r\n                total_cost += c\n                max_cost = np.maximum(max_cost, c)'''
    t = replace_required(t, old_train, new_train, 1, 'Swimmer Multi training step')

    t = t.replace('    print("DISTURBANCE_MODES:", env.DISTURBANCE_MODES)\n    print("disturbances:", env.disturbances)\n',
                  '    print("sigma_viscosity:", args.sigma_viscosity)\n')
    t = t.replace('default="Quadrotor",', 'default="SwimmerWithPos",', 1)
    t = t.replace('"--cost_dim", type=int, default=4', '"--cost_dim", type=int, default=2', 1)

    if '--sigma_viscosity' not in t:
        gravity = '    parser.add_argument(\n        "--gravity_std", type=float, default=0.5, help="gravity perturbation"\n    )\n'
        if gravity not in t:
            raise RuntimeError('Could not insert Swimmer sigma_viscosity parser option')
        t = t.replace(gravity, gravity + '    parser.add_argument("--sigma_viscosity", type=float, default=0.0, help="Swimmer viscosity perturbation")\n', 1)

    t = t.replace('"--sigma_viscosity", type=float, default=0.7', '"--sigma_viscosity", type=float, default=0.0')
    if perturbed:
        t = t.replace('"--sigma_viscosity", type=float, default=0.0', '"--sigma_viscosity", type=float, default=0.7', 1)
        return namespace_outputs(t, 'SwimmerMultiPerturbedRobust')
    return namespace_outputs(t, 'SwimmerMulti')


def patch_swimmer_combined(src: str, perturbed: bool) -> str:
    t = replace_required(
        src,
        'from envs.swimmer import SwimmerWithPos, SwimmerWithPosPerturbed\n',
        'from experiment_extensions.swimmer_multi_constraint_env import SwimmerCombinedEnv as SwimmerWithPos, SwimmerCombinedEnv as SwimmerWithPosPerturbed\n',
        1,
        'Swimmer Combined import',
    )
    if perturbed:
        t = t.replace('default="SwimmerWithPos",', 'default="SwimmerWithPosPerturbed",', 1)
        t = t.replace('"--sigma_viscosity", type=float, default=0.0', '"--sigma_viscosity", type=float, default=0.7', 1)
        return namespace_outputs(t, 'SwimmerCombinedPerturbedRobust')
    t = t.replace('default="SwimmerWithPosPerturbed",', 'default="SwimmerWithPos",', 1)
    t = t.replace('"--sigma_viscosity", type=float, default=0.7', '"--sigma_viscosity", type=float, default=0.0', 1)
    return namespace_outputs(t, 'SwimmerCombined')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--quad-profile', default='white_noise_0p01')
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    req = {
        'quad_single': repo / 'code_ipm_rcmdp_rcrl_max_quadrotor_single_constraint.py',
        'quad_multi': repo / 'code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint.py',
        'quad_robust': repo / 'code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_robust.py',
        'swimmer_scalar': repo / 'code_ipm_rcmdp_rcrl_max_swimmer.py',
        'swimmer_env': repo / 'envs/swimmer.py',
    }
    missing = [str(p) for p in req.values() if not p.exists()]
    if missing:
        raise SystemExit('Run this against the updated Shilpa repo. Missing:\n  ' + '\n  '.join(missing))

    copy_extensions(repo)

    prof = PACK / 'env_profiles' / 'quadrotor' / args.quad_profile
    if not prof.exists():
        choices = sorted(p.name for p in (PACK/'env_profiles'/'quadrotor').iterdir() if p.is_dir())
        raise SystemExit(f'Unknown profile {args.quad_profile}. Choices: {choices}')
    cfg = repo / 'envs' / 'env_configs'
    cfg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prof / 'train.yaml', cfg / 'constrained_tracking_train_perturbed.yaml')
    shutil.copy2(prof / 'eval.yaml', cfg / 'constrained_tracking_eval_perturbed.yaml')
    (cfg / 'ACTIVE_PERTURBATION_PROFILE.txt').write_text(args.quad_profile + '\n', encoding='utf-8')

    quad_single = req['quad_single'].read_text(encoding='utf-8')
    quad_multi = req['quad_multi'].read_text(encoding='utf-8')
    quad_robust = req['quad_robust'].read_text(encoding='utf-8')
    swimmer_scalar = req['swimmer_scalar'].read_text(encoding='utf-8')

    outputs = {
        'code_ipm_rcmdp_rcrl_max_quadrotor_combined.py': patch_quad_combined(quad_single, False),
        'code_ipm_rcmdp_rcrl_max_quadrotor_combined_robust.py': patch_quad_combined(quad_single, True),
        'code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py': patch_quad_multi_perturbed(quad_robust),
        'code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py': patch_swimmer_multi(quad_multi, False),
        'code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py': patch_swimmer_multi(quad_multi, True),
        'code_ipm_rcmdp_rcrl_max_swimmer_combined.py': patch_swimmer_combined(swimmer_scalar, False),
        'code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py': patch_swimmer_combined(swimmer_scalar, True),
    }
    for name, content in outputs.items():
        (repo / name).write_text(content, encoding='utf-8')
        print('GENERATED', name)

    print('ACTIVE QUAD PROFILE', args.quad_profile)
    print('Swimmer calibration remains intentionally blocked until swimmer_thresholds.json is calibrated.')


if __name__ == '__main__':
    main()
