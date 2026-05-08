"""
Iterative prompt optimizer for NewtonBench.

After each trial, feeds the agent's reasoning, submitted law, and evaluation
outcome to Claude, which:
  1. Diagnoses specific failure points
  2. Describes what it is changing in the prompt and why
  3. Returns the improved system prompt

All improvements must be GENERALIZABLE — they should help the agent discover
any physical law, not overfit to the specific law just attempted.
"""

import math
import re
from utils.call_llm_api import call_llm_api

# ---------------------------------------------------------------------------
# Meta-optimizer prompts
# ---------------------------------------------------------------------------

_SYSTEM = """You are an expert in prompt engineering for AI scientific discovery agents.
The agent's job is to discover physical laws in a simulated universe by designing
experiments and fitting equations to data.  You improve the agent's system prompt
based on how it performed in a trial."""

_USER = """Below is everything you need to diagnose a trial failure and produce an improved system prompt.

══════════════════════════════════════════════
TRIAL PERFORMANCE
══════════════════════════════════════════════
• RMSLE          : {rmsle}  (0.0 = perfect; higher = worse)
• Exact Accuracy : {exact_accuracy}  (1.0 = correct functional form; 0.0 = wrong)
• Rounds used    : {rounds} / {max_rounds}
• Experiments run: {num_experiments}
• Status         : {status}

══════════════════════════════════════════════
SUBMITTED LAW (what the agent produced)
══════════════════════════════════════════════
```python
{submitted_law}
```

══════════════════════════════════════════════
GROUND TRUTH (correct functional form)
══════════════════════════════════════════════
{ground_truth_formula}

══════════════════════════════════════════════
KEY MOMENTS FROM THE AGENT'S REASONING
══════════════════════════════════════════════
{condensed_history}

══════════════════════════════════════════════
CURRENT SYSTEM PROMPT
══════════════════════════════════════════════
{current_system_prompt}

══════════════════════════════════════════════
YOUR TASK
══════════════════════════════════════════════
Respond in EXACTLY this three-section format (use the exact headers):

## FAILURE POINTS
A numbered list of specific, concrete reasoning failures observed in this trial.
Focus on *what the agent did wrong*, not *what the correct answer was*.
Examples of good failure points:
  - "Submitted after only 2 experiments without verifying predictions numerically"
  - "Assumed a variable had no effect without explicitly testing it in isolation"
  - "Did not cross-check the fitted law against held-out data points"

## PROMPT CHANGES
A numbered list of the specific changes you are making to the system prompt,
and the generalizable reasoning principle behind each change.
Format each as: "<what changed> — because <why it helps any law discovery task>"

## IMPROVED PROMPT
The full improved system prompt text, ready to be used as-is.

STRICT RULES for the improved prompt:
• GENERALIZABLE — changes must help the agent discover *any* physical law, not
  just this one. Do NOT mention the specific law, variables, or correct functional
  form from this trial.
• TARGETED — only add or refine rules/strategies that address the diagnosed
  failures. Do not pad with unrelated content.
• PRESERVE STRUCTURE — keep existing sections; only modify or append where needed.
• NO OVERFITTING — do not encode domain-specific physics knowledge."""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _condense_history(chat_history: list, chars_per_turn: int = 700) -> str:
    """Return a condensed view of the most informative assistant turns."""
    assistant_msgs = [m for m in chat_history if m.get("role") == "assistant"]
    if not assistant_msgs:
        return "(no assistant messages)"

    selected = []

    first = assistant_msgs[0]["content"][:chars_per_turn]
    selected.append(f"[Turn 1 — initial approach]\n{first}" +
                    ("…" if len(assistant_msgs[0]["content"]) > chars_per_turn else ""))

    if len(assistant_msgs) > 3:
        mid = len(assistant_msgs) // 2
        snippet = assistant_msgs[mid]["content"][:chars_per_turn]
        selected.append(f"[Mid-exploration]\n{snippet}" +
                        ("…" if len(assistant_msgs[mid]["content"]) > chars_per_turn else ""))

    for msg in assistant_msgs[-2:]:
        snippet = msg["content"][:chars_per_turn]
        selected.append(f"[Near-final / submission]\n{snippet}" +
                        ("…" if len(msg["content"]) > chars_per_turn else ""))

    return "\n\n---\n\n".join(selected)


def _parse_response(response: str) -> dict:
    """
    Parse the structured optimizer response into its three components.
    Returns dict with keys: failure_points, prompt_changes, improved_prompt.
    """
    result = {"failure_points": "", "prompt_changes": "", "improved_prompt": ""}

    fp_match = re.search(
        r"##\s*FAILURE POINTS\s*(.*?)(?=##\s*PROMPT CHANGES|$)",
        response, re.DOTALL | re.IGNORECASE
    )
    pc_match = re.search(
        r"##\s*PROMPT CHANGES\s*(.*?)(?=##\s*IMPROVED PROMPT|$)",
        response, re.DOTALL | re.IGNORECASE
    )
    ip_match = re.search(
        r"##\s*IMPROVED PROMPT\s*(.*?)$",
        response, re.DOTALL | re.IGNORECASE
    )

    if fp_match:
        result["failure_points"] = fp_match.group(1).strip()
    if pc_match:
        result["prompt_changes"] = pc_match.group(1).strip()
    if ip_match:
        result["improved_prompt"] = ip_match.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_optimized_prompt(
    current_system_prompt: str,
    trial_result: dict,
    evaluation_result: dict,
    max_rounds: int = 10,
    optimizer_model: str = "cs46",
    trial_info: dict = None,
) -> tuple:
    """
    Analyse a trial result and return (improved_prompt, analysis_dict).

    analysis_dict contains:
      - failure_points (str)
      - prompt_changes (str)
      - improved_prompt (str)
      - raw_response (str)

    Falls back to (current_system_prompt, {}) on error.
    """
    condensed = _condense_history(trial_result.get("chat_history", []))

    rmsle = evaluation_result.get("rmsle", float("nan"))
    rmsle_str = f"{rmsle:.4f}" if not math.isnan(rmsle) else "N/A"

    user_msg = _USER.format(
        rmsle=rmsle_str,
        exact_accuracy=evaluation_result.get("exact_accuracy", 0.0),
        rounds=trial_result.get("rounds", "N/A"),
        max_rounds=max_rounds,
        num_experiments=trial_result.get("num_experiments", "N/A"),
        status=trial_result.get("status", "N/A"),
        submitted_law=trial_result.get("submitted_law", "N/A"),
        ground_truth_formula=evaluation_result.get("ground_truth_law") or "Not available",
        condensed_history=condensed,
        current_system_prompt=current_system_prompt,
    )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    try:
        response, _, _ = call_llm_api(
            messages, model_name=optimizer_model, temperature=0.5, trial_info=trial_info
        )
        if response and response.strip():
            parsed = _parse_response(response)
            parsed["raw_response"] = response
            improved = parsed["improved_prompt"] or current_system_prompt
            return improved, parsed
    except Exception as e:
        print(f"[PromptOptimizer] call_llm_api failed: {e}")

    return current_system_prompt, {}
