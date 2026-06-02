# Systematic Failure Modes in Small Open-Weight LLMs

**COMP6242 Deep Learning: Australian National University (June 2026)**

**Authors:** Adithya Rama (u8085375), Arnav Mahajan, Tanisha Sharma

**Written report:** [`report/final_report_draft.pdf`](report/final_report_draft.pdf) (source: [`report/final_report_draft.tex`](report/final_report_draft.tex))

This repository contains the full experiment pipeline, raw and processed results, analysis code, and the final report for our group project on **why small open-weight language models fail on reasoning tasks**, and **whether prompting (especially error-targeted in-context learning) can recover those failures without retraining**.

---

## Table of contents

1. [What we studied](#what-we-studied)
2. [Main findings (evaluator summary)](#main-findings-evaluator-summary)
3. [Repository layout](#repository-layout)
4. [Models evaluated](#models-evaluated)
5. [Benchmarks and sample sizes](#benchmarks-and-sample-sizes)
6. [Prompting strategies (conditions)](#prompting-strategies-conditions)
7. [Error taxonomy and metrics](#error-taxonomy-and-metrics)
8. [How we ran experiments](#how-we-ran-experiments)
9. [Setup and dependencies](#setup-and-dependencies)
10. [Running experiments locally (Ollama, ~1B–4B)](#running-experiments-locally-ollama-1b4b)
11. [Running 7B experiments on Google Colab](#running-7b-experiments-on-google-colab)
12. [Processing results and generating figures](#processing-results-and-generating-figures)
13. [Raw result file format](#raw-result-file-format)
14. [Reproducibility notes and limitations](#reproducibility-notes-and-limitations)
15. [Related work and citations](#related-work-and-citations)
16. [Team contributions and AI use](#team-contributions-and-ai-use)

---

## What we studied

Small open-weight LLMs are useful for local, low-cost reasoning experiments, but **a single accuracy score does not explain why a model failed** or whether a failure can be fixed with prompting alone.

We therefore:

1. Run **eight** Gemma, Llama, and Qwen models on **seven** reasoning benchmarks (200 examples per benchmark where available; **183** for FOLIO).
2. Apply a shared set of **prompting strategies** (zero-shot controls, few-shot CoT, **error-targeted ICL**, and self-consistency).
3. Assign each **wrong zero-shot (CoT) answer** an operational error label via rule-based code.
4. Measure **recovery**: among questions that failed under the baseline reasoning prompt with error type *e*, what fraction become correct under strategy *s*?

**Research questions (from the report):**

1. Which failure modes dominate after a baseline reasoning prompt?
2. Which failure types are recovered by error-targeted ICL and its ablations?
3. Which irregular model-size effects explain why larger models do not always win?

**Central claim:** Recovery is **error-dependent**. Prompting helps most for superficial failures (format/extraction, off-topic); deeper failures (drifting, hallucination, arithmetic slip, negation) are much harder to repair without stronger models, better decoding, or external tools.

---

## Main findings (evaluator summary)

These numbers are taken from the processed results summarized in the final report. See the PDF for tables, figures, and discussion.

| Result | Detail |
|--------|--------|
| **Highest mean accuracy** | Self-consistency (~50.4% mean over processed model–benchmark cells) |
| **Highest mean recovery over baseline failures** | Random few-shot CoT (~32.9%) |
| **Most recoverable error types** | Off-topic responses; format/extraction (`format_correct_wrong_answer` in code) |
| **Least recoverable error types** | Arithmetic slips, drifting, hallucination, negation |
| **Dominant zero-shot errors** | Drifting (~43%); format/extraction (~32%) |
| **Specialization example** | Qwen2-Math-1.5B strong on GSM8K/GSM-IC, weak on BBH/FOLIO |
| **Size irregularity** | Gemma 3 4B substantially outperforms older Gemma 1 7B-IT despite fewer parameters |
| **Prompting can hurt** | Some models drop when CoT or long ICL is added (e.g. Qwen 2.5 7B on GSM-Symbolic under zero-shot CoT) |

**Novel method — error-targeted ICL:** Few-shot prompts that show a **predicted** error pattern, an incorrect trace, and a corrected trace (with ablations: random error class, correct-only without wrong trace). Full prompts live in `prompts/errorTargetedPrompt.py` (`EXEMPLAR_BANK`).

**Not run (compute limit):** Planned 9B models and LLaVA 7B multimodal runs. Documented as a limitation in the report, not missing-at-random.

---

## Repository layout

```
llm-failure/
├── README.md                    # This file
├── report/                      # Final PDF + LaTeX report
├── config.py                    # Global experiment settings
├── run_experiment.py            # Main experiment runner (all conditions per question)
├── colab_7b_experiments.ipynb   # Colab GPU workflow for 7B Hugging Face models
├── requirements.txt
├── data/
│   └── data_loader.py           # Hugging Face dataset loaders
├── models/
│   └── model_loader.py          # Ollama API backend (default); patched in Colab for HF
├── prompts/                     # All prompting strategies + exemplar banks
├── evaluation/
│   ├── scorer.py                # Answer extraction + exact-match scoring
│   └── taxonomy.py              # Post-hoc error labeling on wrong answers
├── results/
│   ├── raw/<benchmark>/         # Per-run JSON (incremental saves)
│   └── processed/               # summary.csv, error CSVs (from processResults)
├── analysis/
│   └── heatmap.py               # Figures for report (heatmap, radar, cumulative, etc.)
├── llm-failure-7b-results/      # Colab logs, manifest, status, copied raw JSON
└── prompts/example_bank/        # Example JSONs from early pilot runs
```

---

## Models evaluated

Eight model configurations appear in the **processed** results used for the report (`results/processed/summary.csv`). They were run in two execution environments (see [How we ran experiments](#how-we-ran-experiments)).

| Hugging Face / config ID | Family | Approx. size | How run |
|--------------------------|--------|--------------|---------|
| `Gemma/Gemma3-1B` | Gemma 3 | 1B | Local Ollama (`gemma3:1b`) |
| `Gemma/Gemma3-4B` | Gemma 3 | 4B | Local Ollama (`gemma3:4b`) |
| `google/gemma-7b-it` | Gemma 1 | 7B | Colab GPU (`colab_7b_experiments.ipynb`) |
| `Llama/Llama3.2-1.5B` | Llama 3.2 | 1.5B | Local Ollama (`llama3.2:1b`) |
| `Llama/Llama3.2-3B` | Llama 3.2 | 3B | Local Ollama (`llama3.2:3b`) |
| `Qwen/Qwen2-math-1.5B` | Qwen 2 Math | 1.5B | Local Ollama (`qwen2-math:1.5b`) |
| `Qwen/Qwen2.5-3B` | Qwen 2.5 | 3B | Local Ollama (`qwen2.5:3b`) |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen 2.5 | 7B | Colab GPU |

**Ollama name mapping** is defined in `models/model_loader.py` (`MODEL_NAME_MAP`). To add a model: pull it in Ollama, add a map entry, and add the HF-style ID under `MODELS` in `config.py`.

**Planned but not executed:** 9B-scale models; LLaVA 7B (multimodal).

**Known data issue:** Some `Qwen/Qwen2.5-3B` benchmark pairs appear twice in `summary.csv` (duplicate raw runs). The report **averages duplicate rows** for numerical summaries.

---

## Benchmarks and sample sizes

| Config key | Dataset (Hugging Face) | Split / subset | Category tag | Eval samples |
|------------|-------------------------|----------------|--------------|--------------|
| `gsm8k` | `openai/gsm8k` | `main` train (first N) | `arithmetic_word_problem` | 200 |
| `gsm_symbolic` | `apple/GSM-Symbolic` | `test` | `arithmetic_symbolic` | 200 |
| `gsm_plus` | `qintongli/GSM-Plus` | `test` | `arithmetic_perturbed` | 200 |
| `gsm_ic` | `voidful/GSM-IC` | `validation` | `arithmetic_word_problem` | 200 |
| `bigbench_hard` | `lukaemon/bbh` | `logical_deduction_five_objects` | `logical_deduction` | 200 |
| `bigbench_hard_tracking` | `lukaemon/bbh` | `tracking_shuffled_objects_five_objects` | `object_tracking` | 200 |
| `folio` | `yale-nlp/FOLIO` | `validation` | `formal_logic` | **183** (full validation split used by loader) |

**Few-shot pool:** The first `FEW_SHOT_POOL_SIZE` (default **20**) loaded examples are reserved for random few-shot sampling; evaluation uses the next `NUM_SAMPLES` (default **200**) questions. Implemented in `run_experiment.py` → `runbenchmarkOnModel()`.

**Scoring:** Exact match after normalization (`evaluation/scorer.py`):

- GSM-style: numeric extraction (last number or `Answer:` line)
- BBH: multiple-choice option `(A)`–`(E)`
- FOLIO: `True` / `False` / `Uncertain`

---

## Prompting strategies (conditions)

All conditions are evaluated **on the same questions** within each raw JSON file. Condition names match `config.py` → `CONDITIONS`.

| Condition key | Description |
|---------------|-------------|
| `zero_shot_baseline` | Task instruction + required answer format; **no** explicit chain-of-thought (S0 control for accuracy tables) |
| `zero_shot` | Adds step-by-step / CoT-style instructions (`prompts/zeroShot.py`) — **baseline for error coding and recovery** |
| `targeted_few_shot_answer_only` | 3 hand-selected examples by question type; final answers only (no CoT in demos) |
| `random_few_shot` | 3 examples sampled from held-out pool (`prompts/randomFewShot.py`) |
| `targeted_few_shot` | 3 targeted examples with CoT (`prompts/targetedFewShots.py`, k=3) |
| `targeted_few_shot_k5` | Same as targeted few-shot with **5** examples |
| `error_targeted_icl` | Error-targeted ICL: wrong trace + corrected trace for **predicted** error class (`showError=True`) |
| `error_targeted_icl_random` | Same format but **random** error class (ablation: does targeting matter?) |
| `error_targeted_icl_correct_only` | Targeted class but **only** correct demonstrations (`showError=False`) |
| `self_consistency` | 5 samples at temperature 0.7 from zero-shot CoT prompt; **majority vote** on extracted answers |

**Error-targeted class prediction (before inference):** `prompts/errorTargetedPrompt.py` → `classifyErrorType(question)` uses simple keyword rules (e.g. distractor cues, negation words) — not the post-hoc taxonomy on model output.

**Self-consistency cost:** 5× inference per question (`N_SAMPLES_S6=5`, `TEMPERATURE_S6=0.7`).

---

## Error taxonomy and metrics

### Post-hoc labels (after a wrong answer)

Implemented in `evaluation/taxonomy.py` on the model **response** when `zero_shot` is incorrect. Labels include:

| Code label | Report name (approx.) |
|------------|------------------------|
| `format_correct_wrong_answer` | Format / extraction error |
| `drifting` | Drifting (wrong mid-reasoning) |
| `hallucination` | Hallucination |
| `distractor` | Distractor capture |
| `arithmetic_slip` | Arithmetic slip |
| `negation` | Negation error |
| `off_topic` | Off-topic response |
| `premise_order_sensitivity` | Premise-order sensitivity (rare) |
| `confidently_wrong`, `multi_hop`, `unknown` | Defined but rare or unused in dominant counts |

**Important:** This is an **automatic, heuristic** taxonomy — useful for exploration, not manually validated with inter-annotator agreement (see report limitations).

### Metrics

- **Accuracy(condition)** = fraction of examples where `majority_correct` is true for that condition.
- **Recovery(strategy, error type)** = among questions wrong under `zero_shot` with that `error_type`, fraction that become correct under the strategy (paired at question index — see report Method).
- **Gain(strategy)** = accuracy(strategy) − accuracy(`zero_shot_baseline`).

Recovery and error distributions in `results/processResults.py` use **`zero_shot`** as the failure baseline (includes reasoning trace for labeling). Table-style baseline accuracy in the report may use `zero_shot_baseline` where noted.

---

## How we ran experiments

We used **two pipelines** that share the same `run_experiment.py` logic and JSON schema.

### Pipeline A — Local smaller models (Ollama)

**Who / what:** Gemma 3 1B/4B, Llama 3.2 1.5B/3B, Qwen2-Math 1.5B, Qwen 2.5 3B (and pilot runs on subsets).

**Environment:**

- Machine with [Ollama](https://ollama.com) installed and `ollama serve` running (default `http://localhost:11434`).
- Python 3 with `pip install -r requirements.txt`.
- Hugging Face `datasets` downloads benchmarks on first run (network required).

**Steps:**

1. Pull models, e.g. `ollama pull gemma3:1b`, `ollama pull qwen2.5:3b` (see `MODEL_NAME_MAP` in `models/model_loader.py`).
2. Edit `config.py` → `MODELS` to select families to run.
3. Smoke test: `python run_experiment.py --smoke`
4. Run by family: `python run_experiment.py --model gemma` (runs **all** models in that family).
5. Outputs: `results/raw/<benchmark>/<Model>__<benchmark>__<timestamp>.json` (saved incrementally after each question).

**Inference:** `models/model_loader.py` calls Ollama `/api/chat` (temperature 0 for single-sample conditions). Qwen models append `\n/no_think` to reduce thinking-mode tokens.

### Pipeline B — 7B models on Google Colab (Hugging Face + 4-bit)

**Who / what:** `Qwen/Qwen2.5-7B-Instruct` and `google/gemma-7b-it` (Gemma 1 7B-IT).

**Notebook:** [`colab_7b_experiments.ipynb`](colab_7b_experiments.ipynb)

**Environment:**

- Colab GPU runtime.
- Google Drive for codebase, HF cache, and persistent outputs.
- Colab secret **`HF_TOKEN`** (Gemma/Llama licenses on Hugging Face).

**What the notebook does:**

1. Mount Drive; set `CODEBASE_DIR` and `OUTPUT_ROOT` (default under `COMP6242/Project/llm-failure`).
2. **Patch** runtime `config.py` and replace `models/model_loader.py` with a Hugging Face `transformers` + `bitsandbytes` 4-bit loader (repo default remains Ollama-only).
3. Run smoke cells, then full grid: each enabled model × each dataset × each condition (manifest-driven).
4. Copy completed raw JSON from `results/raw/` to `llm-failure-7b-results/raw/`; write logs and `manifest.json` for resume.
5. On disconnect: rerun setup, set `RUN_FULL_EXPERIMENTS = True`; completed manifest entries are **skipped** unless `FORCE_RERUN = True`.

**Scale (default full grid):** 2×7×10 = **140** condition-level jobs per model pair, with self-consistency costing 5 generations per question.

**Artifacts:** `llm-failure-7b-results/logs/`, `status/`, `manifest.json`, `manifest.csv`, `robustness_ratio.csv`, `error_distribution.csv`.

### End-to-end workflow (analysis)

```
run_experiment.py  →  results/raw/**/*.json
        ↓
python -m results.processResults [files...]  →  results/processed/*.csv
        ↓
python -m analysis.heatmap [--all | -<model> | --<benchmark>]  →  results/processed/charts/
```

---

## Setup and dependencies

```bash
pip install -r requirements.txt
```

**Core packages:** `torch`, `transformers`, `accelerate`, `datasets`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`, `huggingface-hub`.

**For Colab 7B runs:** also need GPU, `bitsandbytes`, and accepted HF licenses for gated models.

**Ollama (local):**

1. Install Ollama from https://ollama.com
2. Verify: `ollama --version`
3. Start server if needed: `ollama serve`
4. Pull each model listed in `MODEL_NAME_MAP`

---

## Running experiments locally (Ollama, ~1B–4B)

### Configuration (`config.py`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `SMOKE_TEST` | `False` | Use smoke sample count when true |
| `SMOKE_TEST_SAMPLES` | `2` | Samples per benchmark in smoke mode |
| `NUM_SAMPLES` | `200` | Evaluation questions per benchmark |
| `MAX_NEW_TOKENS` | `8192` | Max tokens per Ollama response |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base |
| `RESULTS_DIR` | `results/raw` | Raw JSON output directory |
| `PROCESSED_DIR` | `results/processed` | Processed CSV output |
| `MODELS` | see file | Dict: family → list of HF-style model IDs |
| `BENCHMARKS` | 7 benchmarks | Set of benchmark keys |
| `CONDITIONS` | 10 strategies | List of condition keys to run |
| `NUM_EXAMPLES` | `3` | Few-shot k (default) |
| `NUM_EXAMPLES_K5` | `5` | Targeted few-shot k=5 |
| `N_SAMPLES_S6` | `5` | Self-consistency samples |
| `TEMPERATURE_S6` | `0.7` | Self-consistency temperature |
| `FEW_SHOT_POOL_SIZE` | `20` | Reserved exemplar pool size |
| `NUM_RUNS` | `1` | Repeated runs per question (majority vote if >1) |

### CLI (`run_experiment.py`)

```bash
# Quick sanity check (2 samples per benchmark)
python run_experiment.py --smoke

# One model family (all models in that family in config.MODELS)
python run_experiment.py --model qwen

# One benchmark
python run_experiment.py --benchmark folio

# One prompting condition
python run_experiment.py --condition zero_shot

# Full configured run
python run_experiment.py
```

Arguments can be **combined**, e.g. `python run_experiment.py --model gemma --benchmark gsm8k --condition error_targeted_icl`.

**Runtime:** A full run (all models, benchmarks, conditions) takes **many hours** depending on hardware. Recommended order: smoke test → one family → one benchmark before full grid.

**Note:** `--model` takes a **family key** (`qwen`, `gemma`, `llama`), not a single HF model path.

---

## Running 7B experiments on Google Colab

1. Upload or sync this repo to Google Drive (see `CODEBASE_DIR` in the notebook).
2. Open [`colab_7b_experiments.ipynb`](colab_7b_experiments.ipynb) in Colab with **GPU** runtime.
3. Set `HF_TOKEN` as a Colab secret (required for `google/gemma-7b-it`).
4. Keep `RUN_FULL_EXPERIMENTS = False` until smoke cells pass.
5. Run setup cells → smoke test → set `RUN_FULL_EXPERIMENTS = True` → run full experiment cell.
6. After runs, merge/copy 7B raw JSON into `results/raw/` if needed for unified `processResults` (some merged filenames use pattern `*__merged_*.json`).

Default model order in notebook: **Qwen 7B first**, then **Gemma 7B-IT**.

---

## Processing results and generating figures

### Process raw JSON → CSV summaries

```bash
# All JSON under results/raw/
python -m results.processResults

# Specific files (under results/raw/<benchmark>/)
python -m results.processResults Gemma_Gemma3-1B__gsm_symbolic__20260511_2100.json
```

**Outputs** (`results/processed/`):

| File | Content |
|------|---------|
| `summary.csv` | Per model–benchmark accuracies and recovery rates |
| `error_distribution.csv` | Error counts from wrong `zero_shot` runs |
| `error_strategy_recovery.csv` | Recovery heatmap input (error × strategy) |

### Generate report figures

```bash
# Core figures: heatmap (Fig 1), family comparison (Fig 2), strategy progression (Fig 3)
python -m analysis.heatmap

# Everything: all per-model and per-benchmark charts
python -m analysis.heatmap --all

# Per-model radar + cumulative (Fig 4–5)
python -m analysis.heatmap -Llama/Llama3.2-3B

# Per-benchmark cumulative, error bars, family radars (Fig 6–9)
python -m analysis.heatmap --gsm8k
```

Charts are written under `results/processed/charts/` (regenerate if the folder is empty in your clone).

**Report figures** referenced in LaTeX use paths like `results/processed/charts/figures/fig1_error_strategy_heatmap.png`.

---

## Raw result file format

Each file is a JSON **list** of question records. Example structure:

```json
{
  "question": "...",
  "ground_truth": "...",
  "benchmark_category": "arithmetic_word_problem",
  "conditions": {
    "zero_shot": {
      "majority_correct": false,
      "correct_count": 0,
      "total_runs": 1,
      "runs": [{
        "run": 0,
        "condition": "zero_shot",
        "correct": false,
        "predicted": "...",
        "ground_truth": "...",
        "error_type": "format_correct_wrong_answer",
        "full_response": "..."
      }]
    },
    "random_few_shot": { "...": "..." }
  }
}
```

Filename convention: `{ModelWithSlashes}__{benchmark}__{YYYYMMDD_HHMM}.json` under `results/raw/{benchmark}/`.

---

## Reproducibility notes and limitations

1. **Compute scope:** No 9B or LLaVA 7B results; conclusions apply to included ≤7B text models only.
2. **Duplicate rows:** Average duplicate `Qwen/Qwen2.5-3B` cells when reproducing report tables.
3. **Automatic errors:** Taxonomy is rule-based; ambiguous multi-error answers may be mislabeled; strict parsing can inflate format errors.
4. **Prompt fairness:** Targeted/error exemplars are hand-written; random few-shot uses benchmark pool — prompt lengths differ; small models may be hurt by long ICL.
5. **Error-targeted targeting:** Pre-inference keyword classifier ≠ post-hoc error label on failed outputs.
6. **Self-consistency cost:** 5× inference; not compared to cheaper verification in this repo.
7. **Generative AI:** Used for debugging, some exemplar drafting, and report/LaTeX refinement — see [Team contributions and AI use](#team-contributions-and-ai-use).

---

## Related work and citations

Key references (full list in report bibliography):

- Chain-of-thought prompting (Wei et al., 2022)
- Zero-shot CoT (Kojima et al., 2022)
- Self-consistency (Wang et al., 2022)
- Benchmarks: GSM8K, GSM-Symbolic, GSM-Plus, GSM-IC, BIG-Bench Hard, FOLIO

---

## Team contributions and AI use

| Member | Role (high level) |
|--------|-------------------|
| **Adithya Rama** (u8085375) | Experiment planning, error-taxonomy framing, Qwen 2.5 7B + Gemma 7B-IT runs, interim analysis, initial LaTeX report draft, results/conclusion presentation |
| **Arnav Mahajan** | *(see report Contribution Declaration — to be completed)* |
| **Tanisha Sharma** | *(see report Contribution Declaration — to be completed)* |

**Generative AI declaration (project-wide):** Tools (including OpenAI Codex) assisted with debugging, some targeted prompt examples, grammar/LaTeX refinement, and README/report wording. **All numerical results, claims, and figures must be verified against raw/processed outputs** before grading. AI was not used to fabricate experimental results.

---

## Quick reference for evaluators

| I want to… | Go to… |
|------------|--------|
| Read the full study | `report/final_report_draft.pdf` |
| See aggregated numbers | `results/processed/summary.csv`, `error_strategy_recovery.csv` |
| Inspect per-question outputs | `results/raw/` and `llm-failure-7b-results/raw/` |
| Understand prompting code | `prompts/`, especially `errorTargetedPrompt.py` |
| Re-run scoring/labels | `evaluation/scorer.py`, `evaluation/taxonomy.py` |
| Reproduce figures | `python -m results.processResults` then `python -m analysis.heatmap --all` |
| Re-run 7B experiments | `colab_7b_experiments.ipynb` |
| Re-run local ≤4B experiments | Ollama + `python run_experiment.py` |

If anything in this README disagrees with the PDF report, **treat the report and processed CSVs as authoritative** and open an issue or check for an updated commit.
