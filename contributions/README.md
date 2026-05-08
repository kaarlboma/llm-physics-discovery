# Contributions — Karl Boma

This folder contains the four contributions I made to the NewtonBench benchmark
as part of research at Boston University. Each subfolder is self-contained and
maps directly onto the original repo's directory structure, so any file can be
dropped in place to extend a NewtonBench installation.

---

## 1. Multi-Provider LLM API Routing (`llm_api_routing/`)

**File:** `utils/call_llm_api.py`

A unified API layer that routes calls across four providers — OpenAI (direct),
OpenRouter, Google Gemini, and Anthropic — automatically selecting the provider
based on which keys are present and which providers support the requested model.

**Model registry** (shorthand → provider/model-id):

| Shorthand | Provider | Full ID |
|---|---|---|
| `gpt41mini` | OpenAI / OpenRouter | gpt-4.1-mini-2025-04-14 |
| `gpt41` | OpenAI / OpenRouter | gpt-4.1-2025-04-14 |
| `o4mini` | OpenAI / OpenRouter | o4-mini-2025-04-16 |
| `gem25f` | Gemini / OpenRouter | gemini-2.5-flash |
| `gem25p` | Gemini / OpenRouter | gemini-2.5-pro |
| `dsv3` | OpenRouter | deepseek-chat-v3-0324 |
| `dsr1` | OpenRouter | deepseek-r1-0528 |
| `qwen3-235b` | OpenRouter | qwen3-235b-a22b |
| `cs46` | Anthropic | claude-sonnet-4-6 |
| `ch45` | Anthropic | claude-haiku-4-5-20251001 |

**Key functions:** `call_llm_api()`, `resolve_model_and_source()`, `robust_json_parse()`

Also includes a layered JSON repair fallback (`fix-busted-json` → custom regex →
manual parsing → raw text) for handling malformed structured outputs from LLMs.

---

## 2. Enhanced Evaluation Engine (`evaluation_engine/`)

**File:** `modules/common/evaluation.py`

Extends the original exact-match-only evaluation with two new metrics:

- **RMSLE** — Root Mean Squared Log Error, a continuous numeric score measuring
  how closely the submitted law's predictions match ground truth. Robust to
  sign differences via absolute-value transform before log.

- **LLM-as-Judge Symbolic Equivalence** — calls a configurable judge model to
  determine whether the submitted formula is mathematically equivalent to the
  ground truth. Handles cases where forms look different but are algebraically
  identical (e.g., `a*b^2` vs `b^2*a`). Uses AST-based formula extraction
  (`extract_formula_from_function`) so the judge sees a clean expression rather
  than raw Python source.

**Key functions:** `evaluate_law()`, `llm_symbolic_equivalence_judge()`,
`calculate_rmsle()`, `extract_formula_from_function()`

---

## 3. Module-Agnostic Experiment Runner (`experiment_runner/`)

**File:** `run_experiments.py`

Replaces the original hardcoded evaluation scripts with a general-purpose runner:

- Discovers any module by name at runtime via `importlib` — no per-module wiring
- Selects agent backend at the CLI (`--agent_backend vanilla_agent|code_assisted_agent`)
- Accepts a `--custom_prompt_file` to inject an optimized system prompt
- Parallelizes trials across CPU cores using `multiprocessing.Pool`
- Distributes trials evenly across law versions when `--law_version all` is set
- Writes structured JSON per trial (including retry history) and a rolled-up
  `aggregated_results.json` with RMSLE, accuracy, token usage, and retry stats
- Handles failures with a structured fail record rather than crashing the run

**Usage:**
```bash
python run_experiments.py \
  --module m0_gravity \
  --model_name gpt41mini \
  --equation_difficulty easy \
  --law_version all \
  --trials 12 \
  --agent_backend vanilla_agent
```

---

## 4. Iterative Prompt Optimization (`iterative_prompt_optimization/`)

**Files:** `run_iterative_optimization.py`, `utils/prompt_optimizer.py`,
`prompts/friend_prompt_v9.txt`

A self-improving pipeline for agent system prompts. After each trial, the
optimizer receives the agent's full reasoning trace, submitted law, and evaluation
outcome, then:

1. Diagnoses specific reasoning failures (e.g., "submitted after 2 experiments
   without verifying predictions numerically")
2. Lists each prompt change and its generalizable principle
3. Returns a complete improved system prompt, ready to use as-is

All improvements are constrained to be **generalizable** — they must help the
agent discover any physical law, not overfit to the specific trial just run.
Optimized prompts are versioned and saved per iteration.

`prompts/friend_prompt_v9.txt` is the ninth-generation optimized system prompt
produced by this pipeline.

**Usage:**
```bash
python run_iterative_optimization.py \
  --module m8_sound_speed \
  --model_name ch45 \
  --optimizer_model cs46 \
  --equation_difficulty easy \
  --law_version v1 \
  --iterations 5
```
