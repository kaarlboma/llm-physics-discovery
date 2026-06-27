"""
Colab rollout generator for GRPO training data.

Run this on Google Colab Pro (A100) to generate Qwen2.5-7B-Instruct rollouts
for all power-law NewtonBench modules.

─── Colab setup (paste into a cell and run first) ────────────────────────────

    from google.colab import drive
    drive.mount('/content/drive')

    # Clone repo (or adjust path if already on Drive)
    !git clone https://github.com/kaarlboma/llm-physics-discovery.git
    %cd llm-physics-discovery
    !pip install transformers bitsandbytes accelerate datasets -q

    # Then run this script:
    !python generate_rollouts_colab.py

──────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import importlib
import torch
from datetime import datetime
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.vanilla_agent import BASE_PROMPT, parse_experiment_request, _extract_final_law
from utils.trajectory_generator import MODULE_CONFIGS
from utils.grpo_reward import reward_from_trajectory

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
MODEL_KEY  = "qwen2.5-7b-hf"          # used in output directory paths
DIFFICULTIES  = ["easy"]
LAW_VERSIONS  = ["v0", "v1", "v2"]
N_ROLLOUTS    = 3                       # rollouts per (module, difficulty, law_version)
MAX_TURNS     = 10
NOISE_LEVEL   = 0.0
TEMPERATURE   = 0.7                     # slightly higher than eval for diverse rollouts
MAX_NEW_TOKENS = 2048
OUTPUT_BASE   = "evaluation_results"
RUN_VERSION   = 1                       # increment each Colab iteration


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print("Model ready.\n")
    return model, tokenizer


# ─── Inference ────────────────────────────────────────────────────────────────

def generate_response(model, tokenizer, messages: List[Dict]) -> tuple:
    """Run one forward pass. Returns (response_text, n_tokens_generated)."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    return response, len(new_ids)


# ─── Single rollout ───────────────────────────────────────────────────────────

def run_single_rollout(
    model, tokenizer,
    module_name: str,
    difficulty: str,
    law_version: str,
    trial_id: int,
) -> Dict:
    config = MODULE_CONFIGS[module_name]
    mod    = importlib.import_module(config["module_path"])

    messages = [
        {"role": "system", "content": BASE_PROMPT.format(max_turns=MAX_TURNS)},
        {"role": "user",   "content": mod.get_task_prompt("vanilla_equation", noise_level=NOISE_LEVEL)},
    ]

    total_tokens   = 0
    num_experiments = 0
    rounds          = 0

    for _ in range(MAX_TURNS):
        response_text, n_tokens = generate_response(model, tokenizer, messages)
        total_tokens += n_tokens
        rounds       += 1
        messages.append({"role": "assistant", "content": response_text})

        if "<final_law>" in response_text:
            break

        experiments = parse_experiment_request(response_text)
        if experiments:
            results = []
            for exp in experiments:
                try:
                    result = mod.run_experiment_for_module(
                        noise_level=NOISE_LEVEL,
                        difficulty=difficulty,
                        system="vanilla_equation",
                        law_version=law_version,
                        **exp,
                    )
                    results.append("{:.15e}".format(result))
                except Exception:
                    results.append("nan")
                num_experiments += 1

            output_str = f"<experiment_output>\n{json.dumps(results)}\n</experiment_output>"
            messages.append({"role": "user", "content": output_str})
        else:
            messages.append({
                "role": "user",
                "content": "Please use <run_experiment> to gather data or <final_law> to submit your law.",
            })

    # If final law was never submitted, force it
    if "<final_law>" not in messages[-1]["content"]:
        messages.append({
            "role": "user",
            "content": "You have used all experiment turns. Submit your final law now using <final_law>.",
        })
        response_text, n_tokens = generate_response(model, tokenizer, messages)
        total_tokens += n_tokens
        rounds       += 1
        messages.append({"role": "assistant", "content": response_text})

    _, submitted_law = _extract_final_law(messages[-1]["content"], mod.FUNCTION_SIGNATURE)

    return {
        "trial_id":          trial_id,
        "module_name":       module_name,
        "noise_level":       NOISE_LEVEL,
        "model_name":        MODEL_KEY,
        "equation_difficulty": difficulty,
        "model_system":      "vanilla_equation",
        "law_version":       law_version,
        "retry_attempts":    0,
        "LLM judge":         None,
        "agent_backend":     "vanilla_agent",
        "retry_history":     [],
        "status":            "completed",
        "submitted_law":     submitted_law,
        "rounds":            rounds,
        "total_tokens":      total_tokens,
        "num_experiments":   num_experiments,
        "chat_history":      messages,
        "evaluation":        {"rmsle": None, "exact_accuracy": 0.0},
    }


# ─── Save ─────────────────────────────────────────────────────────────────────

def _trial_path(module_name, difficulty, law_version, trial_id):
    trial_dir = os.path.join(
        OUTPUT_BASE, MODEL_KEY, module_name, "vanilla_agent",
        difficulty, law_version,
        f"vanilla_equation_noise0_0_v{RUN_VERSION}", "trials",
    )
    os.makedirs(trial_dir, exist_ok=True)
    return os.path.join(trial_dir, f"trial{trial_id}.json")


def save_rollout(trial, module_name, difficulty, law_version):
    path = _trial_path(module_name, difficulty, law_version, trial["trial_id"])
    with open(path, "w") as f:
        json.dump(trial, f, indent=2)
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_all_rollouts():
    model, tokenizer = load_model()

    combos = [
        (m, d, lv)
        for m in MODULE_CONFIGS
        for d in DIFFICULTIES
        for lv in LAW_VERSIONS
    ]

    total = len(combos) * N_ROLLOUTS
    print(f"Generating {N_ROLLOUTS} rollouts × {len(combos)} combos = {total} total rollouts\n")

    done = 0
    for module_name, difficulty, law_version in combos:
        print(f"── {module_name}/{difficulty}/{law_version} ──")
        for trial_id in range(N_ROLLOUTS):

            # Skip if already generated (resume-safe)
            path = _trial_path(module_name, difficulty, law_version, trial_id)
            if os.path.exists(path):
                print(f"  trial {trial_id}: already exists, skipping")
                done += 1
                continue

            try:
                trial = run_single_rollout(
                    model, tokenizer,
                    module_name, difficulty, law_version, trial_id,
                )
                path = save_rollout(trial, module_name, difficulty, law_version)

                traj = {
                    "chat_history":  trial["chat_history"],
                    "submitted_law": trial["submitted_law"],
                    "module_name":   module_name,
                    "difficulty":    difficulty,
                    "law_version":   law_version,
                }
                r = reward_from_trajectory(traj)
                print(f"  trial {trial_id}: reward={r['reward']:.3f}"
                      f"  acc={r['accuracy_score']:.3f}"
                      f"  expl={r['exploration_score']:.3f}"
                      f"  → {path}")

            except Exception as e:
                print(f"  trial {trial_id}: FAILED — {e}")

            done += 1
            print(f"  Progress: {done}/{total}")

    print(f"\nDone. Results in {OUTPUT_BASE}/{MODEL_KEY}/")
    print("Next: re-run build_grpo_dataset.py to include these rollouts.")


if __name__ == "__main__":
    generate_all_rollouts()
