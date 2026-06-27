"""
Perfect trajectory generator for GRPO training data.

For each power-law module, generates programmatically-correct chat transcripts
by running the actual experiment simulator, computing exponents via log-ratio,
and deriving the constant — no LLM involved.

Supported modules (power-law only):
  m0_gravity, m1_coulomb_force, m2_magnetic_force, m3_fourier_law,
  m8_sound_speed, m9_hooke_law, m11_heat_transfer
"""

import json
import math
import random
import importlib
import os
from typing import List, Dict, Any, Optional

from utils.vanilla_agent import BASE_PROMPT

MAX_TURNS = 10

# ─── Per-module configuration ─────────────────────────────────────────────────
#
# exp_variables : kwargs passed to run_experiment_for_module
# law_variables : parameter names in the discovered_law function signature
#                 (differs from exp_variables only for m8_sound_speed)
# defaults      : held-fixed values when isolating another variable
# explore_sets  : 3 interchangeable value sets per variable;
#                 variants pick different sets to diversify trajectories

MODULE_CONFIGS = {
    "m0_gravity": {
        "module_path": "modules.m0_gravity",
        "exp_variables": ["mass1", "mass2", "distance"],
        "law_variables": ["mass1", "mass2", "distance"],
        "defaults": {"mass1": 1.0, "mass2": 1.0, "distance": 1.0},
        "explore_sets": {
            "mass1":    [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "mass2":    [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "distance": [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
        },
    },
    "m1_coulomb_force": {
        "module_path": "modules.m1_coulomb_force",
        "exp_variables": ["q1", "q2", "distance"],
        "law_variables": ["q1", "q2", "distance"],
        "defaults": {"q1": 1.0, "q2": 1.0, "distance": 1.0},
        "explore_sets": {
            "q1":       [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "q2":       [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "distance": [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
        },
    },
    "m2_magnetic_force": {
        "module_path": "modules.m2_magnetic_force",
        "exp_variables": ["current1", "current2", "distance"],
        "law_variables": ["current1", "current2", "distance"],
        "defaults": {"current1": 1.0, "current2": 1.0, "distance": 1.0},
        "explore_sets": {
            "current1": [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "current2": [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "distance": [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
        },
    },
    "m3_fourier_law": {
        "module_path": "modules.m3_fourier_law",
        "exp_variables": ["k", "A", "delta_T", "d"],
        "law_variables": ["k", "A", "delta_T", "d"],
        "defaults": {"k": 1.0, "A": 1.0, "delta_T": 1.0, "d": 1.0},
        "explore_sets": {
            "k":       [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "A":       [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "delta_T": [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "d":       [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
        },
    },
    "m8_sound_speed": {
        "module_path": "modules.m8_sound_speed",
        # experiment runner uses these kwarg names:
        "exp_variables": ["adiabatic_index", "temperature", "molar_mass"],
        # discovered_law function uses these parameter names:
        "law_variables": ["gamma", "T", "M"],
        "defaults": {"adiabatic_index": 2.0, "temperature": 300.0, "molar_mass": 0.04},
        "explore_sets": {
            "adiabatic_index": [[1.0, 4.0, 16.0], [0.5, 2.0, 8.0],  [2.0, 8.0, 32.0]],
            "temperature":     [[100.0, 400.0, 1600.0], [50.0, 200.0, 800.0], [200.0, 800.0, 3200.0]],
            "molar_mass":      [[0.01, 0.04, 0.16], [0.005, 0.02, 0.08], [0.02, 0.08, 0.32]],
        },
    },
    "m9_hooke_law": {
        "module_path": "modules.m9_hooke_law",
        "exp_variables": ["x"],
        "law_variables": ["x"],
        "defaults": {"x": 1.0},
        "explore_sets": {
            "x": [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
        },
    },
    "m11_heat_transfer": {
        "module_path": "modules.m11_heat_transfer",
        "exp_variables": ["m", "c", "delta_T"],
        "law_variables": ["m", "c", "delta_T"],
        "defaults": {"m": 1.0, "c": 1.0, "delta_T": 1.0},
        "explore_sets": {
            "m":       [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "c":       [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
            "delta_T": [[1.0, 5.0, 25.0], [2.0, 10.0, 50.0], [3.0, 15.0, 75.0]],
        },
    },
}

# ─── Core helpers ─────────────────────────────────────────────────────────────

def _run_exp(run_fn, var_dict: Dict, difficulty: str, law_version: str) -> float:
    return run_fn(
        noise_level=0.0, difficulty=difficulty,
        system="vanilla_equation", law_version=law_version,
        **var_dict
    )


def _compute_exponent(values: List[float], outputs: List[float]) -> float:
    """Average log-ratio exponent across all consecutive pairs."""
    exps = []
    for i in range(len(values) - 1):
        v1, o1, v2, o2 = values[i], outputs[i], values[i+1], outputs[i+1]
        if v1 > 0 and v2 > 0 and o1 > 0 and o2 > 0:
            exps.append(math.log(o2 / o1) / math.log(v2 / v1))
    return sum(exps) / len(exps) if exps else 0.0


def _product(items):
    result = 1.0
    for x in items:
        result *= x
    return result


# ─── Trajectory generation ────────────────────────────────────────────────────

def generate_trajectory(
    module_name: str,
    difficulty: str,
    law_version: str,
    seed: int = 0,
) -> Dict:
    """
    Generate one perfect trajectory for the given module/difficulty/law_version.

    The seed controls variable exploration order and which explore_set is used
    per variable, producing distinct-but-correct variant trajectories.

    Returns a dict with chat_history and submitted_law matching the trial.json format.
    """
    random.seed(seed)

    config = MODULE_CONFIGS[module_name]
    mod = importlib.import_module(config["module_path"])
    run_fn = mod.run_experiment_for_module

    exp_vars = config["exp_variables"]
    law_vars = config["law_variables"]
    defaults = dict(config["defaults"])
    exp_to_law = dict(zip(exp_vars, law_vars))

    # Randomize variable exploration order for diversity across seeds
    var_order = list(exp_vars)
    random.shuffle(var_order)

    chat = [
        {"role": "system", "content": BASE_PROMPT.format(max_turns=MAX_TURNS)},
        {"role": "user",   "content": mod.get_task_prompt("vanilla_equation", noise_level=0.0)},
    ]

    exponents: Dict[str, float] = {}

    # ── Phase 1: isolate each variable ───────────────────────────────────────
    for var in var_order:
        set_idx = random.randint(0, len(config["explore_sets"][var]) - 1)
        explore_vals = config["explore_sets"][var][set_idx]

        experiments = []
        for v in explore_vals:
            pt = dict(defaults)
            pt[var] = v
            experiments.append(pt)

        outputs = [_run_exp(run_fn, pt, difficulty, law_version) for pt in experiments]
        exponents[var] = _compute_exponent(explore_vals, outputs)

        out_strs = ["{:.15e}".format(o) for o in outputs]
        chat.append({
            "role": "assistant",
            "content": f"<run_experiment>\n{json.dumps(experiments, indent=2)}\n</run_experiment>",
        })
        chat.append({
            "role": "user",
            "content": f"<experiment_output>\n{json.dumps(out_strs)}\n</experiment_output>",
        })

    # ── Phase 2: derive constant from defaults ────────────────────────────────
    ref_out = _run_exp(run_fn, defaults, difficulty, law_version)
    denom = _product(defaults[v] ** exponents[v] for v in exp_vars)
    constant = ref_out / denom

    # ── Phase 3: verification round (2 unseen points) ────────────────────────
    verify_factors = [2.5, 3.7, 6.1, 8.3, 4.4]
    verify_pts = []
    for i in range(2):
        factor = verify_factors[i % len(verify_factors)]
        pt = {v: defaults[v] * factor for v in exp_vars}
        verify_pts.append(pt)

    verify_outs = [_run_exp(run_fn, pt, difficulty, law_version) for pt in verify_pts]
    verify_strs = ["{:.15e}".format(o) for o in verify_outs]
    chat.append({
        "role": "assistant",
        "content": f"<run_experiment>\n{json.dumps(verify_pts, indent=2)}\n</run_experiment>",
    })
    chat.append({
        "role": "user",
        "content": f"<experiment_output>\n{json.dumps(verify_strs)}\n</experiment_output>",
    })

    # ── Phase 4: submit final law ─────────────────────────────────────────────
    terms = " * ".join(f"{exp_to_law[v]}**{exponents[v]:.6g}" for v in exp_vars)
    func_sig = mod.FUNCTION_SIGNATURE
    submitted_law = f"{func_sig}\n    C = {constant:.6g}\n    return C * {terms}"

    chat.append({
        "role": "assistant",
        "content": f"<final_law>\n{submitted_law}\n</final_law>",
    })

    return {
        "module_name": module_name,
        "difficulty": difficulty,
        "law_version": law_version,
        "seed": seed,
        "source": "programmatic_perfect",
        "chat_history": chat,
        "submitted_law": submitted_law,
        "computed_exponents": {exp_to_law[v]: round(exponents[v], 6) for v in exp_vars},
        "computed_constant": constant,
    }


def generate_all_trajectories(
    output_dir: str = "perfect_trajectories",
    n_variants: int = 3,
    difficulties: Optional[List[str]] = None,
    law_versions: Optional[List[str]] = None,
    modules: Optional[List[str]] = None,
) -> None:
    """
    Generate perfect trajectories for all power-law modules and save as JSON files.

    Output layout:
        {output_dir}/{module_name}/{difficulty}/{law_version}/trajectory_seed{N}.json
    """
    if difficulties is None:
        difficulties = ["easy"]
    if law_versions is None:
        law_versions = ["v0", "v1", "v2"]
    if modules is None:
        modules = list(MODULE_CONFIGS.keys())

    os.makedirs(output_dir, exist_ok=True)
    total = 0

    for module_name in modules:
        for difficulty in difficulties:
            for law_version in law_versions:
                out_dir = os.path.join(output_dir, module_name, difficulty, law_version)
                os.makedirs(out_dir, exist_ok=True)
                for seed in range(n_variants):
                    try:
                        traj = generate_trajectory(module_name, difficulty, law_version, seed)
                        path = os.path.join(out_dir, f"trajectory_seed{seed}.json")
                        with open(path, "w") as f:
                            json.dump(traj, f, indent=2)
                        total += 1
                        print(f"✓ {module_name}/{difficulty}/{law_version}/seed{seed}"
                              f"  exponents={traj['computed_exponents']}")
                    except Exception as e:
                        print(f"✗ {module_name}/{difficulty}/{law_version}/seed{seed}: {e}")

    print(f"\nTotal trajectories generated: {total}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate perfect trajectories for GRPO training")
    parser.add_argument("--output_dir", default="perfect_trajectories")
    parser.add_argument("--n_variants", type=int, default=3)
    parser.add_argument("--difficulties", nargs="+", default=["easy"])
    parser.add_argument("--law_versions", nargs="+", default=["v0", "v1", "v2"])
    parser.add_argument("--modules", nargs="+", default=None,
                        help="Subset of modules to generate (default: all power-law modules)")
    args = parser.parse_args()

    generate_all_trajectories(
        output_dir=args.output_dir,
        n_variants=args.n_variants,
        difficulties=args.difficulties,
        law_versions=args.law_versions,
        modules=args.modules,
    )
