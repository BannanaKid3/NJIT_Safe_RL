#!/usr/bin/env python3
"""
Safe Quadrotor training I/O optimizer.

What it changes, and ONLY what it changes:
1) Disables active env.render() calls inside trainer/evaluation code.
2) Changes expensive per-episode training-history np.save()/plot_metrics()
   work to run once every 100 episodes, plus the final episode.

What it DOES NOT change:
- environment physics
- rewards
- constraints
- cost computation
- actor/critic updates
- seeds
- batch sizes
- learning rates
- evaluation frequency
- checkpoint selection/saving logic
- disturbance configuration

The script creates .pre_io_opt.bak backups before modifying files and
compiles each patched Python file afterward. It is idempotent.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import py_compile
import re
import shutil
import sys

PRIMARY_TARGETS = [
    "code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint.py",
    "code_ipm_rcmdp_rcrl_max_quadrotor_combined.py",
    "code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py",
    "code_ipm_rcmdp_rcrl_max_quadrotor_combined_robust.py",
]

OPTIONAL_TARGETS = [
    # Source/template robust trainer. Patch it too if present so a direct run
    # does not pay the same unnecessary diagnostics overhead.
    "code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_robust.py",
]

MARKER = "save_training_diagnostics = ((total_steps + 1) % 100 == 0) or (total_steps == args.max_train_steps - 1)"


def disable_render(text: str) -> tuple[str, int]:
    """Comment active standalone env.render() calls only."""
    pat = re.compile(r"^(?P<indent>[ \t]*)env\.render\(\)\s*(?P<comment>#.*)?$", re.M)

    def repl(m: re.Match) -> str:
        indent = m.group("indent")
        suffix = (" " + m.group("comment")) if m.group("comment") else ""
        return f"{indent}# env.render() disabled for headless training speed{suffix}"

    return pat.subn(repl, text)


def find_active_plot_call_end(lines: list[str], search_start: int) -> int:
    """
    Find the active plot_metrics(...) call after search_start and return the
    inclusive final line index of that function call.
    """
    call_start = None
    for i in range(search_start, len(lines)):
        s = lines[i].lstrip()
        if s.startswith("plot_metrics("):
            call_start = i
            break
    if call_start is None:
        raise RuntimeError("Could not find active plot_metrics(...) call after per-episode np.save block.")

    balance = 0
    seen_open = False
    for i in range(call_start, len(lines)):
        # The target source does not contain strings with unmatched parentheses
        # in this call block, so simple counting is appropriate and transparent.
        balance += lines[i].count("(")
        balance -= lines[i].count(")")
        if "(" in lines[i]:
            seen_open = True
        if seen_open and balance <= 0:
            return i

    raise RuntimeError("Could not determine end of active plot_metrics(...) call.")


def gate_training_diagnostics(text: str) -> tuple[str, bool]:
    """
    Wrap the expensive end-of-episode training-history save/plot block so it
    executes every 100 episodes and at the final episode.
    """
    if MARKER in text:
        return text, False

    lines = text.splitlines(keepends=True)

    # Find the specific per-episode training-history save that follows
    # episode_rewards.append(total_reward). We do not touch evaluation saves.
    start = None
    append_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "episode_rewards.append(total_reward)":
            append_idx = i
            # Search locally after this append for the training-history save.
            for j in range(i + 1, min(i + 60, len(lines))):
                if 'np.save(f"{plot_data_dir}/episode_rewards.npy"' in lines[j]:
                    start = j
                    break
            if start is not None:
                break

    if append_idx is None:
        raise RuntimeError("Could not find episode_rewards.append(total_reward).")
    if start is None:
        raise RuntimeError(
            "Could not find per-episode plot_data episode_rewards np.save block "
            "near episode_rewards.append(total_reward)."
        )

    end = find_active_plot_call_end(lines, start)

    # Determine indentation from the np.save line. In the current Shilpa-derived
    # trainers this is 8 spaces inside main().
    indent_match = re.match(r"^([ \t]*)", lines[start])
    indent = indent_match.group(1)
    child_indent = indent + "    "

    original_block = lines[start : end + 1]

    # Wrap the existing block unchanged. Only indentation is added.
    wrapped = [
        f"{indent}{MARKER}\n",
        f"{indent}if save_training_diagnostics:\n",
    ]
    for line in original_block:
        if line.strip():
            wrapped.append(child_indent + line[len(indent):] if line.startswith(indent) else child_indent + line)
        else:
            wrapped.append(line)

    lines[start : end + 1] = wrapped
    return "".join(lines), True


def validate_patch(path: Path, text: str) -> None:
    active_render = [
        (i + 1, line)
        for i, line in enumerate(text.splitlines())
        if re.match(r"^[ \t]*env\.render\(\)\s*(#.*)?$", line)
    ]
    if active_render:
        raise RuntimeError(f"{path.name}: active env.render() remains at {active_render}")

    if MARKER not in text:
        raise RuntimeError(f"{path.name}: diagnostics gating marker missing after patch.")

    if 'episode_rewards.append(total_reward)' not in text:
        raise RuntimeError(f"{path.name}: expected training-history append missing.")

    # Syntax validation.
    py_compile.compile(str(path), doraise=True)


def patch_one(path: Path) -> tuple[int, bool]:
    text = path.read_text(encoding="utf-8")
    original = text

    text, render_count = disable_render(text)
    text, gated_now = gate_training_diagnostics(text)

    if text == original:
        print(f"UNCHANGED {path.name} (already optimized)")
        validate_patch(path, text)
        return render_count, False

    backup = path.with_name(path.name + ".pre_io_opt.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"BACKUP   {backup.name}")
    else:
        print(f"BACKUP   existing {backup.name} preserved")

    path.write_text(text, encoding="utf-8")
    validate_patch(path, text)
    print(
        f"PATCHED  {path.name} | disabled render calls={render_count} | "
        f"gated diagnostics={'yes' if gated_now else 'already'}"
    )
    return render_count, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="RCRL_on_RCMDP repository root")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        print(f"ERROR: {repo} does not look like the Git repository root (.git missing).")
        return 2

    missing = [name for name in PRIMARY_TARGETS if not (repo / name).exists()]
    if missing:
        print("ERROR: required generated/current Quadrotor trainer files are missing:")
        for name in missing:
            print("  -", name)
        print("\nRun setup_pipeline.py first, then run this optimizer.")
        return 3

    print("Repo:", repo)
    print("Patch policy: render OFF; per-episode training diagnostics every 100 episodes + final")
    print()

    changed = 0
    for name in PRIMARY_TARGETS:
        _, did_change = patch_one(repo / name)
        changed += int(did_change)

    for name in OPTIONAL_TARGETS:
        path = repo / name
        if path.exists():
            _, did_change = patch_one(path)
            changed += int(did_change)
        else:
            print(f"SKIP     optional {name} not present")

    print()
    print(f"SUCCESS: Quadrotor I/O optimization validated. Files changed this run: {changed}")
    print("Learning/environment logic was not modified by this patcher.")
    print("Backups use the suffix: .pre_io_opt.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
