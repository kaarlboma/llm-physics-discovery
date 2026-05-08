# NewtonBench Extensions — Karl Boma

[NewtonBench](https://github.com/HKUST-KnowComp/NewtonBench) is an LLM benchmark
from HKUST that tests whether language models can discover physical laws from
experimental data. Models are placed in a simulated "alien universe" with unknown
physics, must design experiments, and submit a mathematical formula capturing the
underlying law. This repository extends NewtonBench with four contributions built
as part of research at Boston University.

---

## My Contributions

### 1. Multi-Provider LLM API Routing

**File:** `utils/call_llm_api.py` · [view in contributions](contributions/llm_api_routing/utils/call_llm_api.py)

A unified API layer that routes calls across OpenAI (direct), OpenRouter, Google
Gemini, and Anthropic — automatically selecting the provider based on available
keys and model support. Covers GPT-4.1, o4-mini, Gemini 2.5 Flash/Pro,
DeepSeek R1/V3, Qwen3-235B, Nemotron-Ultra, and Claude Sonnet/Haiku.

Includes a layered JSON repair fallback (`fix-busted-json` → custom regex →
manual parsing → raw text) for handling malformed structured outputs from LLMs.

**Key functions:** `call_llm_api()`, `resolve_model_and_source()`, `robust_json_parse()`

---

### 2. Enhanced Evaluation Engine

**File:** `modules/common/evaluation.py` · [view in contributions](contributions/evaluation_engine/modules/common/evaluation.py)

Extends the original exact-match-only evaluation with two new metrics:

- **RMSLE** — a continuous numeric score measuring how closely the submitted
  law's predictions match ground truth, even when the symbolic form differs.
- **LLM-as-Judge Symbolic Equivalence** — calls a configurable judge model to
  determine whether two formulas are mathematically equivalent (catching cases
  where forms look different but are algebraically identical). Uses AST-based
  formula extraction so the judge sees a clean expression rather than raw source.

**Key functions:** `evaluate_law()`, `llm_symbolic_equivalence_judge()`, `calculate_rmsle()`

---

### 3. Module-Agnostic Experiment Runner with Parallelization

**File:** `run_experiments.py` · [view in contributions](contributions/experiment_runner/run_experiments.py)

Replaces the original hardcoded evaluation scripts with a general-purpose runner:

- Discovers any module by name at runtime — no per-module wiring required
- Selects agent backend via `--agent_backend` flag
- Accepts `--custom_prompt_file` to inject an optimized system prompt
- Parallelizes trials across CPU cores with `multiprocessing.Pool`
- Distributes trials evenly across law versions when `--law_version all` is set
- Writes structured JSON per trial (with retry history) and a rolled-up
  `aggregated_results.json` with RMSLE, accuracy, token usage, and retry stats

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

### 4. Iterative Prompt Optimization

**Files:** `run_iterative_optimization.py`, `utils/prompt_optimizer.py`, `prompts/` · [view in contributions](contributions/iterative_prompt_optimization/)

A self-improving pipeline for agent system prompts. After each trial, the
optimizer receives the agent's full reasoning trace, submitted law, and evaluation
outcome, then:

1. Diagnoses specific reasoning failures (e.g., "submitted after 2 experiments
   without numerically verifying predictions")
2. Lists each prompt change with its generalizable principle
3. Returns a complete improved system prompt, ready to use as-is

All improvements are constrained to be **generalizable** — changes must help the
agent discover any physical law, not overfit to the specific trial just run.
`prompts/friend_prompt_v9.txt` is the ninth-generation prompt produced by this
pipeline.

```bash
python run_iterative_optimization.py \
  --module m8_sound_speed \
  --model_name ch45 \
  --optimizer_model cs46 \
  --equation_difficulty easy \
  --law_version v1 \
  --iterations 5
```

---

## Results

| Experiment | Metric | Before | After |
|---|---|---|---|
| Bernoulli's Principle (easy, `ch45`) | Exact Accuracy | 0% | **67%** |
| Bernoulli's Principle (easy, `ch45`) | RMSLE | 1.18 | **0.17** |
| Sound Speed (easy, `ch45`) | RMSLE | 10.86 | **0.00** |

Prompt optimization on the sound speed module reduced RMSLE from 10.86 to 0.00
in a single optimization cycle. On Bernoulli's Principle, iterative prompt
refinement lifted accuracy from 0% to 67% and cut RMSLE by 6.7×.

---

## Quick Start

**1. Clone and install**
```bash
git clone <this-repo>
cd NewtonBench
pip install -r requirements.txt
```

**2. Set API keys**
```bash
cp .env.example .env
# Fill in at least one of: OPENROUTER_API_KEY or OPENAI_API_KEY
```

**3. Run a benchmark experiment**
```bash
python run_experiments.py \
  --module m0_gravity \
  --model_name gpt41mini \
  --equation_difficulty easy \
  --trials 3
```

**4. Run iterative prompt optimization**
```bash
python run_iterative_optimization.py \
  --module m0_gravity \
  --model_name gpt41mini \
  --equation_difficulty easy \
  --iterations 5
```

Results are written to `evaluation_results/` and `iterative_optimization_results/`
(both excluded from version control).

---

## Repository Structure

```
.
├── README.md                ← this file
├── contributions/           ← isolated copies of my 4 contributions
│   ├── llm_api_routing/
│   ├── evaluation_engine/
│   ├── experiment_runner/
│   └── iterative_prompt_optimization/
├── modules/                 ← NewtonBench physics modules
├── utils/                   ← agent backends and shared utilities
├── configs/                 ← model lists
├── requirements.txt
└── .env.example
```

---

## Contact

Karl Boma · karlboma@bu.edu · Boston University

**Supervisor:** Siddharth Mishra-Sharma · Boston University

**Research Partner:** Ryden Tamura
