"""
GRPO reward function for NewtonBench trajectories.

Blended reward:
    reward = alpha * accuracy_score + beta * exploration_score

    accuracy_score  = exp(-rmsle)        in [0, 1]; 1 = perfect law
    exploration_score                    in [0, 1]; 1 = every variable perfectly isolated

Both alpha and beta are configurable; defaults are alpha=0.7, beta=0.3.
"""

import io
import math
import importlib
import contextlib
from typing import Dict, List, Optional

from utils.vanilla_agent import parse_experiment_request
from utils.trajectory_generator import MODULE_CONFIGS


# ─── Exploration score ────────────────────────────────────────────────────────

def _is_constant(values: List[float]) -> bool:
    if len(values) < 2:
        return True
    ref = values[0]
    tol = 1e-9 * (abs(ref) + 1e-15)
    return all(abs(v - ref) <= tol for v in values)


def compute_exploration_score(chat_history: List[Dict], exp_variables: List[str]) -> float:
    """
    Score how systematically the model isolated variables one at a time.

    For each experiment batch in the chat:
      - isolation quality: fraction of non-target variables held constant
        (1.0 = exactly one variable varies, 0.0 = everything varies together)

    Final score = 0.5 * avg_isolation_quality + 0.5 * coverage
      where coverage = fraction of variables isolated at least once perfectly.

    Returns float in [0, 1].
    """
    n_vars = len(exp_variables)
    if n_vars == 0:
        return 0.0

    batches = []
    for msg in chat_history:
        if msg['role'] == 'assistant' and '<run_experiment>' in msg['content']:
            exps = parse_experiment_request(msg['content'])
            if exps and len(exps) >= 2:
                batches.append(exps)

    if not batches:
        return 0.0

    isolation_scores = []
    isolated_vars: set = set()

    for batch in batches:
        var_constant: Dict[str, bool] = {}
        for var in exp_variables:
            vals = [exp[var] for exp in batch if var in exp]
            var_constant[var] = _is_constant(vals) if len(vals) >= 2 else True

        n_constant = sum(1 for c in var_constant.values() if c)
        n_varying = n_vars - n_constant

        if n_vars == 1:
            score = 1.0
            isolated_vars.update(exp_variables)
        elif n_varying == 1:
            score = 1.0
            varying = next(v for v in exp_variables if not var_constant[v])
            isolated_vars.add(varying)
        elif n_varying == 0:
            score = 0.0  # all held constant — uninformative
        else:
            # partial credit proportional to how many non-target vars were held
            score = n_constant / (n_vars - 1)

        isolation_scores.append(score)

    avg_isolation = sum(isolation_scores) / len(isolation_scores)
    coverage = len(isolated_vars) / n_vars

    return 0.5 * avg_isolation + 0.5 * coverage


# ─── Accuracy score ───────────────────────────────────────────────────────────

def compute_rmsle(
    module_name: str,
    submitted_law: str,
    difficulty: str,
    law_version: str,
) -> float:
    """
    Evaluate submitted_law against the ground truth, returning RMSLE.
    Skips the LLM symbolic judge to avoid API calls during training.
    Returns nan on failure.
    """
    if not submitted_law or module_name not in MODULE_CONFIGS:
        return float('nan')

    config = MODULE_CONFIGS[module_name]
    try:
        mod = importlib.import_module(config['module_path'])
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            result = mod.evaluate_law(
                submitted_law,
                mod.PARAM_DESCRIPTION,
                difficulty=difficulty,
                law_version=law_version,
                judge_model_name=None,  # skip LLM judge
            )
        return result.get('rmsle', float('nan'))
    except Exception:
        return float('nan')


# ─── Blended reward ───────────────────────────────────────────────────────────

def compute_reward(
    chat_history: List[Dict],
    submitted_law: str,
    module_name: str,
    difficulty: str,
    law_version: str,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> Dict:
    """
    Compute blended GRPO reward for a single trajectory.

    Args:
        chat_history:  list of {role, content} message dicts
        submitted_law: the Python function string submitted as <final_law>
        module_name:   e.g. 'm0_gravity'
        difficulty:    'easy' / 'medium' / 'hard'
        law_version:   'v0' / 'v1' / 'v2'
        alpha:         weight on accuracy  (default 0.7)
        beta:          weight on exploration (default 0.3)

    Returns dict with keys:
        reward, accuracy_score, exploration_score, rmsle, alpha, beta
    """
    exp_variables = MODULE_CONFIGS.get(module_name, {}).get('exp_variables', [])

    exploration_score = compute_exploration_score(chat_history, exp_variables)

    rmsle = compute_rmsle(module_name, submitted_law, difficulty, law_version)
    accuracy_score = 0.0 if math.isnan(rmsle) else math.exp(-rmsle)

    reward = alpha * accuracy_score + beta * exploration_score

    return {
        'reward': reward,
        'accuracy_score': accuracy_score,
        'exploration_score': exploration_score,
        'rmsle': rmsle,
        'alpha': alpha,
        'beta': beta,
    }


def reward_from_trajectory(trajectory: Dict, alpha: float = 0.7, beta: float = 0.3) -> Dict:
    """Convenience wrapper that takes a trajectory dict directly."""
    return compute_reward(
        chat_history=trajectory['chat_history'],
        submitted_law=trajectory.get('submitted_law', ''),
        module_name=trajectory['module_name'],
        difficulty=trajectory['difficulty'],
        law_version=trajectory['law_version'],
        alpha=alpha,
        beta=beta,
    )


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import json

    print("=== Perfect trajectory (gravity easy v0 seed0) ===")
    with open('perfect_trajectories/m0_gravity/easy/v0/trajectory_seed0.json') as f:
        perfect = json.load(f)
    r = reward_from_trajectory(perfect)
    print(f"  reward={r['reward']:.4f}  accuracy={r['accuracy_score']:.4f}"
          f"  exploration={r['exploration_score']:.4f}  rmsle={r['rmsle']:.2e}")

    print("\n=== Qwen2.5-7b trial (gravity easy v0) ===")
    import glob
    trial_files = glob.glob('evaluation_results/qwen2.5-7b/m0_gravity/vanilla_agent/easy/v0/**/trial*.json',
                            recursive=True)
    for path in trial_files[:3]:
        with open(path) as f:
            trial = json.load(f)
        traj = {
            'chat_history': trial['chat_history'],
            'submitted_law': trial['submitted_law'],
            'module_name': 'm0_gravity',
            'difficulty': 'easy',
            'law_version': 'v0',
        }
        r = reward_from_trajectory(traj)
        print(f"  [{path.split('/')[-1]}] reward={r['reward']:.4f}"
              f"  accuracy={r['accuracy_score']:.4f}"
              f"  exploration={r['exploration_score']:.4f}"
              f"  rmsle={r['rmsle']:.4f}")
