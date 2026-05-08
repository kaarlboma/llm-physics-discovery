"""
Iterative Prompt Optimization for NewtonBench
==============================================

Run N trials of a given module/model/difficulty.  After each failed trial,
Claude diagnoses the agent's reasoning failures and rewrites the system prompt
to be more effective — without overfitting to the specific law just attempted.

After every iteration the runner prints:
  • RMSLE and accuracy
  • Diagnosed failure points
  • Specific prompt changes made (and the generalizable principle behind each)

Usage
-----
python run_iterative_optimization.py \\
    --module m8_sound_speed \\
    --model_name ch45 \\
    --equation_difficulty easy \\
    --law_version v1 \\
    --iterations 3

Results land in:
    iterative_optimization_results/<model>/<module>/<backend>/<difficulty>/<law>/<timestamp>/
"""

import os
import sys
import json
import time
import argparse
import importlib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from utils.code_assisted_agent import (
    conduct_code_assisted_exploration,
    create_code_assisted_system_prompt,
)
from utils.vanilla_agent import conduct_exploration
from utils.prompt_optimizer import generate_optimized_prompt

MAX_TURNS   = 10
JUDGE_MODEL = "ch45"

DIVIDER     = "─" * 62
THICK       = "═" * 62


# ── trial runner ──────────────────────────────────────────────────────────────

def run_single_trial(module, model_name, noise_level, difficulty, system,
                     law_version, iter_dir, trial_id, agent_backend,
                     custom_system_prompt=None):
    trial_info = {"trial_id": trial_id, "trial_dir": iter_dir}

    if agent_backend == "code_assisted_agent":
        exploration = conduct_code_assisted_exploration(
            module=module, model_name=model_name, noise_level=noise_level,
            difficulty=difficulty, system=system, law_version=law_version,
            trial_info=trial_info, custom_system_prompt=custom_system_prompt,
        )
    else:
        exploration = conduct_exploration(
            module=module, model_name=model_name, noise_level=noise_level,
            difficulty=difficulty, system=system, law_version=law_version,
            trial_info=trial_info, custom_system_prompt=custom_system_prompt,
        )

    evaluation = module.evaluate_law(
        exploration["submitted_law"],
        param_description=module.PARAM_DESCRIPTION,
        difficulty=difficulty, law_version=law_version,
        judge_model_name=JUDGE_MODEL, trial_info=trial_info,
    )
    return exploration, evaluation


# ── pretty printers ──────────────────────────────────────────────────────────

def print_trial_result(iteration, total, acc, rmsle, rounds, experiments, elapsed):
    print(f"\n{DIVIDER}")
    print(f"  ITERATION {iteration}/{total} — TRIAL RESULT")
    print(DIVIDER)
    print(f"  Exact Accuracy : {'✓  YES (correct functional form)' if acc == 1.0 else '✗  NO'}")
    print(f"  RMSLE          : {rmsle:.4f}")
    print(f"  Rounds used    : {rounds} / {MAX_TURNS}")
    print(f"  Experiments    : {experiments}")
    print(f"  Time           : {elapsed:.1f}s")


def print_analysis(analysis: dict):
    fp = analysis.get("failure_points", "").strip()
    pc = analysis.get("prompt_changes", "").strip()

    print(f"\n{DIVIDER}")
    print("  FAILURE POINTS IDENTIFIED")
    print(DIVIDER)
    if fp:
        for line in fp.splitlines():
            print(f"  {line}")
    else:
        print("  (none parsed)")

    print(f"\n{DIVIDER}")
    print("  PROMPT CHANGES MADE  (generalizable principles only)")
    print(DIVIDER)
    if pc:
        for line in pc.splitlines():
            print(f"  {line}")
    else:
        print("  (none parsed)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Iterative prompt optimization for NewtonBench"
    )
    parser.add_argument("--module",              type=str, default="m8_sound_speed")
    parser.add_argument("--model_name",          type=str, default="ch45")
    parser.add_argument("--optimizer_model",     type=str, default="cs46",
                        help="Model used to rewrite the prompt (default: cs46)")
    parser.add_argument("-n", "--noise",         type=float, default=0.0)
    parser.add_argument("-d", "--equation_difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("-m", "--model_system",  type=str, default="vanilla_equation",
                        choices=["vanilla_equation", "simple_system", "complex_system"])
    parser.add_argument("-l", "--law_version",   type=str, default="v1")
    parser.add_argument("-b", "--agent_backend", type=str, default="code_assisted_agent",
                        choices=["code_assisted_agent", "vanilla_agent"])
    parser.add_argument("-i", "--iterations",    type=int, default=3)
    parser.add_argument("--initial_prompt_file", type=str, default=None,
                        help="Path to a text file to use as the starting system prompt instead of the default.")
    parser.add_argument("--optimize_on_success", action="store_true",
                        help="Run the optimizer even when the trial succeeds")
    args = parser.parse_args()

    # ── load module ──────────────────────────────────────────────────────────
    try:
        module = importlib.import_module(f"modules.{args.module}")
        print(f"Loaded module: {args.module}")
    except ImportError:
        print(f"ERROR: Module '{args.module}' not found.")
        sys.exit(1)

    # ── output dir ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        "iterative_optimization_results",
        args.model_name, args.module, args.agent_backend,
        args.equation_difficulty, args.law_version, timestamp,
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{THICK}")
    print(f"  ITERATIVE PROMPT OPTIMIZATION")
    print(THICK)
    print(f"  Module      : {args.module}")
    print(f"  Agent model : {args.model_name}")
    print(f"  Optimizer   : {args.optimizer_model}")
    print(f"  Difficulty  : {args.equation_difficulty}  |  Law: {args.law_version}")
    print(f"  Iterations  : {args.iterations}")
    print(f"  Output      : {out_dir}")
    print(THICK)

    # ── starting prompt ──────────────────────────────────────────────────────
    if args.initial_prompt_file:
        with open(args.initial_prompt_file, 'r') as f:
            current_prompt = f.read()
        print(f"  Starting from prompt: {args.initial_prompt_file}")
    else:
        current_prompt = create_code_assisted_system_prompt(
            module, args.equation_difficulty, args.model_system, MAX_TURNS
        )

    log = []

    for iteration in range(1, args.iterations + 1):
        print(f"\n{THICK}")
        print(f"  ITERATION {iteration} / {args.iterations}")
        print(THICK)

        iter_dir = os.path.join(out_dir, f"iteration_{iteration:02d}")
        os.makedirs(iter_dir, exist_ok=True)

        with open(os.path.join(iter_dir, "system_prompt_used.txt"), "w") as f:
            f.write(current_prompt)

        # ── run trial ────────────────────────────────────────────────────────
        label = "baseline prompt" if iteration == 1 else "optimized prompt"
        print(f"\n  Running trial with {label} …")
        t0 = time.time()

        try:
            exploration, evaluation = run_single_trial(
                module=module, model_name=args.model_name,
                noise_level=args.noise, difficulty=args.equation_difficulty,
                system=args.model_system, law_version=args.law_version,
                iter_dir=iter_dir, trial_id=iteration,
                agent_backend=args.agent_backend,
                custom_system_prompt=current_prompt if iteration > 1 else None,
            )
        except Exception as e:
            import traceback
            print(f"  Trial raised an exception: {e}")
            traceback.print_exc()
            break

        elapsed = time.time() - t0
        acc     = evaluation.get("exact_accuracy", 0.0)
        rmsle   = evaluation.get("rmsle", float("nan"))
        rounds  = exploration.get("rounds", "N/A")
        nexps   = exploration.get("num_experiments", "N/A")

        print_trial_result(iteration, args.iterations, acc, rmsle, rounds, nexps, elapsed)

        # ── save record ──────────────────────────────────────────────────────
        record = {
            "iteration":        iteration,
            "exact_accuracy":   acc,
            "rmsle":            rmsle,
            "rounds":           rounds,
            "num_experiments":  nexps,
            "submitted_law":    exploration.get("submitted_law"),
            "ground_truth_law": evaluation.get("ground_truth_law"),
            "time_seconds":     round(elapsed, 2),
        }

        # ── optimize ─────────────────────────────────────────────────────────
        is_last        = (iteration == args.iterations)
        trial_failed   = acc < 1.0
        should_optimize = (trial_failed or args.optimize_on_success) and not is_last

        analysis = {}
        if should_optimize:
            print(f"\n  Running prompt optimizer (model: {args.optimizer_model}) …")
            opt_info = {"trial_id": f"optimizer_iter{iteration}", "trial_dir": iter_dir}

            try:
                new_prompt, analysis = generate_optimized_prompt(
                    current_system_prompt=current_prompt,
                    trial_result=exploration,
                    evaluation_result=evaluation,
                    max_rounds=MAX_TURNS,
                    optimizer_model=args.optimizer_model,
                    trial_info=opt_info,
                )
                print_analysis(analysis)

                # Save optimizer output
                with open(os.path.join(iter_dir, "optimized_prompt.txt"), "w") as f:
                    f.write(new_prompt)
                with open(os.path.join(iter_dir, "optimizer_analysis.json"), "w") as f:
                    json.dump({
                        "failure_points": analysis.get("failure_points", ""),
                        "prompt_changes": analysis.get("prompt_changes", ""),
                    }, f, indent=2)

                print(f"\n  Prompt updated: {len(current_prompt)} → {len(new_prompt)} chars")
                current_prompt = new_prompt

            except Exception as e:
                print(f"  Optimizer failed ({e}). Keeping current prompt.")

        elif not trial_failed:
            print(f"\n  Trial succeeded — carrying winning prompt forward.")

        record["failure_points"] = analysis.get("failure_points", "")
        record["prompt_changes"] = analysis.get("prompt_changes", "")
        log.append(record)

        with open(os.path.join(iter_dir, "result.json"), "w") as f:
            json.dump(record, f, indent=2)

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n{THICK}")
    print("  OPTIMIZATION COMPLETE — SUMMARY")
    print(THICK)
    header = f"  {'Iter':>4}  {'Accuracy':>8}  {'RMSLE':>8}  {'Rounds':>6}  {'Exps':>5}"
    print(header)
    print(f"  {'-'*50}")
    for r in log:
        rmsle_s = f"{r['rmsle']:.4f}" if isinstance(r['rmsle'], float) else str(r['rmsle'])
        acc_s   = "YES" if r['exact_accuracy'] == 1.0 else "NO"
        print(f"  {r['iteration']:>4}  {acc_s:>8}  {rmsle_s:>8}  {str(r['rounds']):>6}  {str(r['num_experiments']):>5}")

    # Save summary + final prompt
    summary = {"config": vars(args), "iterations": log, "final_system_prompt": current_prompt}
    with open(os.path.join(out_dir, "optimization_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "final_prompt.txt"), "w") as f:
        f.write(current_prompt)

    print(f"\n  All results saved to: {out_dir}")


if __name__ == "__main__":
    main()
