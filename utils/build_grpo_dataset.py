"""
GRPO dataset builder.

Combines perfect trajectories and model rollouts into a HuggingFace Dataset
where each row is one (prompt, completion, reward) triplet.

Rows sharing the same group_id (module/difficulty/law_version) form one GRPO
group. The training script normalises rewards within each group to compute
advantages.

Output columns:
    group_id            str   e.g. "m0_gravity_easy_v0"
    prompt_messages     list  [system_msg, user_msg]
    completion_messages list  all messages after the prompt
    submitted_law       str   extracted final law
    reward              float blended reward in [0, 1]
    accuracy_score      float exp(-rmsle)
    exploration_score   float variable isolation quality
    source              str   "perfect" | "rollout"
    module_name         str
    difficulty          str
    law_version         str
"""

import json
import glob
import os
from typing import List, Dict, Optional

from datasets import Dataset

from utils.grpo_reward import reward_from_trajectory
from utils.trajectory_generator import MODULE_CONFIGS


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_perfect_trajectories(
    perfect_dir: str = "perfect_trajectories",
    modules: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    law_versions: Optional[List[str]] = None,
) -> List[Dict]:
    modules     = modules     or list(MODULE_CONFIGS.keys())
    difficulties = difficulties or ["easy"]
    law_versions = law_versions or ["v0", "v1", "v2"]

    rows = []
    pattern = os.path.join(perfect_dir, "**", "trajectory_seed*.json")
    for path in glob.glob(pattern, recursive=True):
        with open(path) as f:
            t = json.load(f)
        if t["module_name"] not in modules:
            continue
        if t["difficulty"] not in difficulties:
            continue
        if t["law_version"] not in law_versions:
            continue
        rows.append(t)

    print(f"Loaded {len(rows)} perfect trajectories from {perfect_dir}/")
    return rows


def load_model_rollouts(
    results_dir: str = "evaluation_results",
    model_name: str = "qwen2.5-7b",
    modules: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    law_versions: Optional[List[str]] = None,
) -> List[Dict]:
    modules      = modules      or list(MODULE_CONFIGS.keys())
    difficulties  = difficulties  or ["easy"]
    law_versions  = law_versions  or ["v0", "v1", "v2"]

    rows = []
    base = os.path.join(results_dir, model_name)
    pattern = os.path.join(base, "**", "trial*.json")

    for path in glob.glob(pattern, recursive=True):
        if "_fail" in path:
            continue  # skip failed trials — submitted law is a nan stub

        with open(path) as f:
            t = json.load(f)

        if t.get("module_name") not in modules:
            continue
        if t.get("equation_difficulty") not in difficulties:
            continue
        if t.get("law_version") not in law_versions:
            continue
        if not t.get("chat_history"):
            continue

        rows.append({
            "module_name": t["module_name"],
            "difficulty":  t["equation_difficulty"],
            "law_version": t["law_version"],
            "source":      "rollout",
            "chat_history":   t["chat_history"],
            "submitted_law":  t.get("submitted_law", ""),
        })

    print(f"Loaded {len(rows)} model rollouts from {base}/")
    return rows


# ─── Row builder ─────────────────────────────────────────────────────────────

def _build_row(traj: Dict, source: str) -> Dict:
    """Convert a trajectory dict into one dataset row."""
    chat = traj["chat_history"]

    # The prompt is always the first two messages: system + user mission.
    # Everything after is the completion (experiment rounds + final law).
    prompt_messages     = chat[:2]
    completion_messages = chat[2:]

    reward_info = reward_from_trajectory(traj)

    return {
        "group_id":           f"{traj['module_name']}_{traj['difficulty']}_{traj['law_version']}",
        "prompt_messages":    prompt_messages,
        "completion_messages": completion_messages,
        "submitted_law":      traj.get("submitted_law", ""),
        "reward":             reward_info["reward"],
        "accuracy_score":     reward_info["accuracy_score"],
        "exploration_score":  reward_info["exploration_score"],
        "source":             source,
        "module_name":        traj["module_name"],
        "difficulty":         traj["difficulty"],
        "law_version":        traj["law_version"],
    }


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_dataset(
    perfect_dir:  str = "perfect_trajectories",
    results_dir:  str = "evaluation_results",
    model_name:   str = "qwen2.5-7b",
    output_dir:   str = "grpo_dataset",
    modules:      Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    law_versions: Optional[List[str]] = None,
    alpha: float = 0.7,
    beta:  float = 0.3,
) -> Dataset:
    """
    Build the GRPO training dataset and save to output_dir.

    Returns a HuggingFace Dataset.
    """
    perfect  = load_perfect_trajectories(perfect_dir, modules, difficulties, law_versions)
    rollouts = load_model_rollouts(results_dir, model_name, modules, difficulties, law_versions)

    all_trajs = [(t, "perfect") for t in perfect] + [(t, "rollout") for t in rollouts]

    print(f"\nScoring {len(all_trajs)} trajectories...")
    rows = []
    for i, (traj, source) in enumerate(all_trajs):
        row = _build_row(traj, source)
        rows.append(row)
        print(f"  [{i+1}/{len(all_trajs)}] {row['group_id']} ({source})"
              f"  reward={row['reward']:.3f}"
              f"  acc={row['accuracy_score']:.3f}"
              f"  expl={row['exploration_score']:.3f}")

    dataset = Dataset.from_list(rows)

    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)

    # Summary
    print(f"\n{'='*50}")
    print(f"Dataset saved to {output_dir}/")
    print(f"  Total rows:  {len(dataset)}")
    print(f"  Perfect:     {sum(1 for r in rows if r['source'] == 'perfect')}")
    print(f"  Rollouts:    {sum(1 for r in rows if r['source'] == 'rollout')}")
    print(f"  Groups:      {len(set(r['group_id'] for r in rows))}")
    print(f"  Avg reward (perfect):  {_avg([r['reward'] for r in rows if r['source'] == 'perfect']):.3f}")
    print(f"  Avg reward (rollouts): {_avg([r['reward'] for r in rows if r['source'] == 'rollout']):.3f}")

    return dataset


def _avg(vals):
    return sum(vals) / len(vals) if vals else float('nan')


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build GRPO training dataset")
    parser.add_argument("--perfect_dir",  default="perfect_trajectories")
    parser.add_argument("--results_dir",  default="evaluation_results")
    parser.add_argument("--model_name",   default="qwen2.5-7b")
    parser.add_argument("--output_dir",   default="grpo_dataset")
    parser.add_argument("--modules",      nargs="+", default=None)
    parser.add_argument("--difficulties", nargs="+", default=["easy"])
    parser.add_argument("--law_versions", nargs="+", default=["v0", "v1", "v2"])
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta",  type=float, default=0.3)
    args = parser.parse_args()

    build_dataset(
        perfect_dir=args.perfect_dir,
        results_dir=args.results_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        modules=args.modules,
        difficulties=args.difficulties,
        law_versions=args.law_versions,
        alpha=args.alpha,
        beta=args.beta,
    )
