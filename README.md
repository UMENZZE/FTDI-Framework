<p align="center">
  <h2 align="center">
  FTDI: A Budget-Aware Self-Healing Framework for Resilient LLM Multi-Agent Code Generation
  </h2>
  <p align="center">
    <a><strong>Sixue Men</strong></a><sup>1</sup>
    ·
    <a><strong>Qinyue Tong</strong></a><sup>1</sup>
    ·
    <a><strong>Rui Zuo</strong></a><sup>1</sup>
    ·
    <a><strong>Zheming Lu</strong></a><sup>1*</sup>
    <br>
    <sup>1</sup>School of Aeronautics and Astronautics, Zhejiang University, Hangzhou 310027, Zhejiang, China
    <br>
    <sup>*</sup>Corresponding author: zheminglu@zju.edu.cn
    <br>
    <br>
    <div align="center">
      <a href='https://github.com/UMENZZE/FTDI-Framework'><img src='https://img.shields.io/badge/GitHub-FTDI-black?logo=github'></a>
      <a href='#'><img src='https://img.shields.io/badge/License-MIT-blue'></a>
      <img src='https://img.shields.io/badge/Python-3.9+-green'>
    </div>
  </p>
</p>

<p align="center">
  <img src="images/Introduction.png" width="85%">
</p>

---

## :mega: News

- **2026.03**: We released the FTDI codebase and sample experiment results.
- **2026.04**: README updated to match paper v4.3 (method details, metrics, and deployment workflow).

---

## :memo: ToDo List

- [ ] Release full experiment result data.
- [x] Release FTDI framework source code.
- [x] Release sample results on HumanEval, HumanEval+, and MBPP.

---

## Overview

**FTDI** is a budget-aware self-healing framework for LLM-based multi-agent code generation. When faults corrupt the Coder output in a Planner→Coder→Tester pipeline—whether caused by semantic perturbations, runtime errors, or structural injections—FTDI intercepts, diagnoses, and repairs compromised outputs through a tiered repair strategy, spending budget only where diagnostic evidence justifies it.

The implementation is built on top of the **AutoAgents** runtime. In the main paper setting, all agents share the same backbone model (Llama3-8B) and the same decoding protocol so that performance differences mainly come from diagnosis and repair policies.



<p align="center">
  <img src="images/FTDI_Framework_Overview.png" width="90%">
</p>

The figure below shows a concrete example of a clean code message, its injected (attacked) version, and the repaired output produced by FTDI:

<p align="center">
  <img src="images/Clean_Injected_Repaired.png" width="88%">
</p>

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| **AutoInject** | `fault_injection/auto_inject.py` | Parameterized fault injector — injects syntax/semantic/text faults at message level ($P_m$) and line level ($P_e$); produces a deterministic **manifest** (seed + line spans + diff) for reproducible replay |
| **Auditor** | `ftdi/auditor.py` | Zero-overhead static scorer — five diagnostic dimensions: Deviation ($w$=0.35), Consistency ($w$=0.25), Uncertainty ($w$=0.20), Boundary ($w$=0.20), Off-by-one ($w$=0.20); outputs anomaly score $s$, `fail_type`, `suspect_span` |
| **Hook** | `ftdi/hook.py` | Environment-level interceptor — instruments `publish_message` to trigger Auditor and budget gate, then dispatches to tiered repair |
| **Tiered Repair** | `ftdi/tiered_repair.py` | Cost-benefit tier dispatcher (Algorithm 1): T0 regex fast-patch → T1 constrained local edit → T2 deep regeneration; stops early when budget is exhausted |
| **Repair Strategy** | `ftdi/repair_strategy.py` | `fail_type` → repair tier mapping; |
| **Repair Agent** | `ftdi/repair_agent.py` | LLM-powered repair engine; retrieves typed repair priors from the knowledge base as prompting evidence |
| **Knowledge Base** | `knowledge_base/` | `distilled_cures.json` — failure-type-indexed repair priors distilled from historical trajectories; `cure_library.json` — generic fallback |
| **Defense Baselines** | `ftdi/defense_baseline.py` | Challenger & Inspector baselines (Huang et al., 2024) |

### Fault Injection Model

Injection is controlled by two probabilities:
- **$P_m$** — probability that a given message is selected for injection
- **$P_e$** — probability that a selected line within that message is mutated
- **`max_lines`** — cap on the number of modified lines per message

Error types: `syntax`, `semantic`, `text`, `translate`

Each run writes a **manifest** (`seed` + `corrupted_spans` + `code_diff`) so the exact same faults can be replayed across all compared methods for fair evaluation. Pre-generated manifests for the paper's main experiments are in `fault_injection/manifests/`.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Pass@1** | Functional correctness on hidden test suite |
| **Resilience** | $S_\text{FTDI} / S_\text{clean} \times 100\%$ |
| **RecoveryRatio** | $(S_\text{FTDI} - S_\text{faulty}) / (S_\text{clean} - S_\text{faulty}) \times 100\%$; can exceed 100% when FTDI surpasses the clean baseline |
| **Tok./1%Rec.** | Average tokens per 1 percentage-point of recovery — lower is better |

---

## Getting Started

### Deployment Order (Important)

To reproduce the paper pipeline in this repository, deploy in the following order:

1. **Deploy AutoAgents backbone**
2. **Deploy local LLM-7B service (vLLM / OpenAI-compatible API)**
3. **Deploy FTDI components and run experiments**

### Step 1. Deploy AutoAgents Backbone

```bash
cd /data/10T/Msx/project/LLMfix

# 1) Create AutoAgents runtime env
conda env create -f autoagent_conda.yml
conda activate autoagent

# 2) Install AutoAgents dependencies and package
pip install -r autoagent_pip.txt
pip install -e ./AutoAgents-main
```

### Step 2. Deploy Local LLM-7B Model (Recommended)

```bash
cd /data/10T/Msx/project/LLMfix

# 1) Create vLLM/runtime env
conda env create -f llmfix_conda.yml
conda activate llmfix
pip install -r llmfix_pip.txt

# 2) Start local OpenAI-compatible endpoint
cd AutoAgents-main
MODEL_PATH=/path/to/your/7b-model \
MODEL_NAME=qwen2.5-7b-instruct \
PORT=8000 \
bash scripts/run_vllm.sh
```

Quick check:

```bash
curl http://localhost:8000/v1/models
```

### Step 3. Deploy FTDI Code

```bash
cd /data/10T/Msx/project/LLMfix
conda activate autoagent

cd FTDI
pip install -r requirements.txt

# Ensure AutoAgents can be imported by FTDI scripts
export PYTHONPATH="/data/10T/Msx/project/LLMfix/AutoAgents-main:${PYTHONPATH}"

# Optional: make script path layout consistent
[ -e examples ] || ln -s ../AutoAgents-main/examples examples
```

### Credential / Endpoint Configuration

Create a `.env.inference` file in `FTDI/`:

```bash
# Main multi-agent flow
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
OPENAI_API_MODEL=qwen2.5-7b-instruct

# Repair agent (can be same local model or a stronger remote model)
REPAIR_BASE_URL=https://your-api-endpoint/v1
REPAIR_API_KEY=your-repair-key
REPAIR_MODEL=deepseek-v3
```

---

## Running Experiments

All scripts below can run with a local model endpoint by setting `USE_XIAOAI=off`.

### Healthy Baseline (no attack)

```bash
USE_XIAOAI=off bash scripts/run_healthy.sh
```

### Attack Only (no repair)

```bash
USE_XIAOAI=off PM=0.8 PE=0.3 bash scripts/run_attack.sh
```

### FTDI Full Defense (attack + tiered repair)

```bash
USE_XIAOAI=off PM=0.8 PE=0.3 bash scripts/run_ftdi.sh
```

### Key Parameters

| Category | Parameter | Value |
|----------|-----------|-------|
| LLM inference | Max context length | 4096 |
| LLM inference | Max new tokens | 768 |
| LLM inference | Temperature | 0 |
| LLM inference | Top-p | 0.95 |
| Multi-agent budget | Max rounds per task | 12 (Linear: 1) |
| Multi-agent budget | Token budget per task | 10000 |
| Multi-agent budget | Budget Gate threshold | 0.7 |
| Multi-agent budget | Max repair attempts | 5 |

Common experiment env vars:

| Variable | Typical value | Description |
|----------|---------------|-------------|
| `PM` | `0.8` | Message-level injection probability $P_m$ |
| `PE` | `0.3` | Line-level injection probability $P_e$ |
| `MAX_LINES` | `1` | Max mutated lines per message |
| `TOPOLOGY` | `hier` | Topology in scripts: `hier` / `flat` / `linear` |
| `REPAIR_ENABLED` | `on` | Enable or disable FTDI repair |
| `REPAIR_TRIGGERS` | `T0,T2` | Enabled repair stages |
| `REPAIR_KB_MODE` | `distilled` | Distilled / generic / none |
| `CONC` | `6` | Evaluation concurrency |

---

## Main Results

Under semantic fault injection ($P_m=0.8$, $P_e=0.3$, `max_lines=1`), FTDI achieves:

| Benchmark | Pass@1 (Clean) | Pass@1 (Faulty) | Pass@1 (FTDI) | Resilience | RecoveryRatio | Tok./1%Rec. |
|-----------|---------------|-----------------|---------------|------------|---------------|-------------|
| HumanEval | 64.02% | 50.00% | **63.41%** | **99.05%** | **95.65%** | **41.5** |
| HumanEval+ | 54.88% | 37.80% | **53.05%** | **96.67%** | **89.29%** | **83.9** |
| MBPP | 44.96% | 30.68% | **43.84%** | **97.51%** | **92.16%** | **64.8** |

---

## Sample Results

Representative experiment outputs are provided in [`sample_results/`](sample_results/). Each directory contains per-task `results.jsonl`, generated `samples.jsonl`, and token event logs.

| Directory | Condition | Benchmark | PM | PE |
|-----------|-----------|-----------|-----|-----|
| `humaneval_healthy` | Healthy baseline | HumanEval | 0 | 0 |
| `humaneval_attack` | Attack only | HumanEval | 0.8 | 0.3 |
| `humaneval_ftdi` | FTDI repair | HumanEval | 0.8 | 0.3 |
| `humaneval_plus_healthy` | Healthy baseline | HumanEval+ | 0 | 0 |
| `humaneval_plus_attack` | Attack only | HumanEval+ | 0.8 | 0.3 |
| `humaneval_plus_ftdi` | FTDI repair | HumanEval+ | 0.8 | 0.3 |
| `mbpp_healthy` | Healthy baseline | MBPP | 0 | 0 |
| `mbpp_attack` | Attack only | MBPP | 0.8 | 0.3 |
| `mbpp_ftdi` | FTDI repair | MBPP | 0.8 | 0.3 |

---

## Project Structure

```
FTDI/
├── fault_injection/
│   └── auto_inject.py          # Controlled fault injector
├── ftdi/
│   ├── auditor.py              # Online deviation scorer
│   ├── hook.py                 # Environment-level interceptor
│   ├── tiered_repair.py        # T0/T1/T2 tiered repair dispatcher
│   ├── repair_agent.py         # LLM-powered repair engine
│   ├── repair_strategy.py      # Error type → repair strategy mapping
│   └── defense_baseline.py     # Challenger & Inspector baselines
├── knowledge_base/
│   ├── cure_library.json       # Generic cure library
│   └── distilled_cures.json    # Distilled cure library (AutoInject patterns)
├── evaluation/
│   └── humaneval_eval.py       # HumanEval/HumanEval+/MBPP evaluation runner
├── scripts/
│   ├── run_healthy.sh          # Healthy baseline experiment
│   ├── run_attack.sh           # Attack-only experiment
│   └── run_ftdi.sh             # FTDI full defense experiment
├── sample_results/             # Representative experiment outputs
└── images/                     # README figures (PNG)
    ├── Introduction.png
    ├── FTDI_Framework_Overview.png
    └── Clean_Injected_Repaired.png
```

---

## :clap: Acknowledgements

The MAS backbone in this project is built upon [AutoAgents](https://github.com/Link-AGI/AutoAgents).  We appreciate their valuable contributions.

---

