#!/usr/bin/env bash
set -euo pipefail
# ===============================================
# ===============================================
# Purpose:
#   1. Source of Acc_attack and tokens_attack for RR/TPR.
#   2. Combined with healthy/attack+imm groups to form 3-way comparison for RR/TPR.
# Notes:
#   - PM/PE/MAX_LINES/SEED should match other scripts for comparability.
#
# Topology switch (TOPOLOGY):
#   - linear: Linear chain (single round, no iteration)
#   - flat: Flat mutual-view (multi-round iteration, mutual feedback)
#   - hier: Hierarchical (default, multi-round, Coder sees diagnostic summary only)

# ======== Step 1: Load credentials (don't override existing vars) ========
if [ -f .env.inference ]; then
  # Only load API keys etc., skip OPENAI_API_BASE
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$line" ]] && continue
    [[ "$line" =~ OPENAI_API_BASE ]] && continue
    [[ "$line" =~ OPENAI_BASE_URL ]] && continue
    if [[ "$line" =~ ^[[:space:]]*([A-Z_][A-Z0-9_]*)=(.+)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      value="${value%%#*}"
      value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^["\'"'"']//' -e 's/["\'"'"']$//')"
      export "$key=$value"
    fi
  done < .env.inference
fi

# ======== Step 2: Set API base based on USE ========
# USE=on: use probex API as base model; otherwise use local vLLM
USE_XIAOAI="${USE_XIAOAI:-on}"
if [ "$USE_XIAOAI" = "on" ]; then
  export OPENAI_API_BASE=""
  export OPENAI_API_KEY="${PROBEX_CHAT_API_KEY:-}"
  export OPENAI_API_MODEL="" 
  if [ "$OPENAI_API_KEY" = "EMPTY" ] || [ -z "$OPENAI_API_KEY" ]; then
    echo "[error] USE_XIAOAI=on but PROBEX_CHAT_API_KEY not provided" >&2
    exit 1
  fi
  echo "[env] Main flow uses probex API base model (${OPENAI_API_MODEL} @ ${OPENAI_API_BASE})"
else
  export OPENAI_API_BASE="http://localhost:8000/v1"
  export OPENAI_API_KEY="EMPTY"
  export OPENAI_API_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
  echo "[env] Main flow uses local vLLM (Llama3-8B @ ${OPENAI_API_BASE})"
fi

# Dataset with separated base_input and plus_input fields for split evaluation
DATA="/data/10T/Msx/project/LLMfix/DATA/humaneval_plus_with_inputs.jsonl"  # Evaluation dataset path
PM=0.8            # Message-level injection probability
PE=0.3             # Line-level injection probability
MAX_LINES=1        # Max lines modified per retry
SEED=42            # Injection random seed
CONC=5             # Concurrent workers
FAIL_PATIENCE=3    # Max consecutive failures (aligned with FTDI for fair comparison)
USE_SUBSET=off     # Subset sampling switch
SAMPLE_N=10        # Number of sampled problems
SAMPLE_SEED=20251117  # Sampling random seed
TS=$(date +%Y%m%d-%H%M%S)  # Result directory timestamp

# ======== Topology switch ========
# linear: Linear chain (Planner->Coder->Tester, single round)
# flat: Flat mutual-view (mutual feedback, multi-round iteration)
# hier: Hierarchical (Planner summarizes diagnostics, Coder sees summary only)
TOPOLOGY=${TOPOLOGY:-"hier"}  # Default: hierarchical

# ======== Test dataset mode switch ========
# base: Run HumanEval original test cases only (~7-10 per problem)
# plus: Run HumanEval+ enhanced test cases only (~80-1000 per problem)
# all: Run all test cases (default)
HUMANEVAL_TEST_MODE=${HUMANEVAL_TEST_MODE:-"plus"}  # Default: plus only

IMM_THRESHOLD_MODE=default  # Immunity threshold mode (placeholder)
IMM_THRESHOLD=0.7  # Immunity threshold (placeholder)
IMM_WINDOW=5  # Quantile window (placeholder)
IMM_QUANTILE=0.8  # Quantile percentile (placeholder)
IMM_TOPK=3  # Immunity prompt count (placeholder)
IMM_FORCE_RUNTIME=on  # Force immunity on runtime error (placeholder)
IMM_COOLDOWN=1  # Cooldown rounds (placeholder)

DATA_TO_USE="$DATA"
SUBTAG=""
if [ "$USE_SUBSET" = "on" ] && [ "$SAMPLE_N" -gt 0 ]; then
  mkdir -p workspace
  SUBSET="workspace/humaneval_plus_subset_${SAMPLE_N}_${SAMPLE_SEED}_${TS}.jsonl"
  python - "$DATA" "$SUBSET" "$SAMPLE_N" "$SAMPLE_SEED" <<PY
import json, random, sys
src, dst, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, 'r', encoding='utf-8') as f:
    lines = [ln for ln in f if ln.strip()]
n = min(n, len(lines))
random.seed(seed)
idx = sorted(random.sample(range(len(lines)), n))
with open(dst, 'w', encoding='utf-8') as g:
    for i in idx:
        g.write(lines[i])
PY
  DATA_TO_USE="$SUBSET"
  SUBTAG="-N${SAMPLE_N}-s${SAMPLE_SEED}"
fi

MODE=attack
GROUP_NAME="mas-env-${TOPOLOGY}-${HUMANEVAL_TEST_MODE}-PM${PM}-PE${PE}-ml${MAX_LINES}${SUBTAG}-${MODE}-${TS}"
RUN_GROUP_DIR="$(pwd)/traces/${GROUP_NAME}"
mkdir -p "$RUN_GROUP_DIR"
echo "[run] MODE=$MODE TOPOLOGY=$TOPOLOGY TEST_MODE=$HUMANEVAL_TEST_MODE AUTOINJECT=on"
(
  export RUN_GROUP_DIR  # Attack mode output directory
  export HUMANEVAL_TEST_MODE="$HUMANEVAL_TEST_MODE"  # Test dataset mode
  export AUTOINJECT_ENABLED=on  # Enable AutoInject
  #export AUTOINJECT_MANIFEST_OFF=1
  export AUTOINJECT_PM="$PM"  # Message-level injection probability
  export AUTOINJECT_PE="$PE"  # Line-level injection probability
  export AUTOINJECT_ERROR_TYPE=semantic  # Injection error type
  export AUTOINJECT_MAX_LINES="$MAX_LINES"  # Max injection lines
  export AUTOINJECT_SEED="$SEED"  # Injection random seed
  export AUTOINJECT_PROTECT_COMMENTS=on  # Preserve comments
  export AUTOINJECT_TASK_LEVEL=0  # Task-level budget (placeholder)
  export AUTOINJECT_TASK_PROB=1.0  # Task-level sampling probability (placeholder)
  export AUTOINJECT_MAX_MSG_PER_TASK=2  # Max injections per task (placeholder)
  # Generate attribution fragments for FAIL tasks only
  export ATTRIBUTION_HYBRID=off  # Attribution on failure only
  # Which attribution fragments to use in next round retrieval
  # Point to previous round group dir to enable feedback loop
  export ATTRIBUTION_DIR=""  # Can point to previous attribution dir
  export AUDITOR=on  # Enable auditor

  python examples/humaneval_official_eval.py \
    --problems-json "$DATA_TO_USE" \
    --concurrency "$CONC" \
    --use-mas-env \
    --topology "$TOPOLOGY" \
    --group-results \
    --write-samples \
    --bad-sig-action retry \
    --sig-retries 1 \
    --auditor on \
    --fail-patience "$FAIL_PATIENCE" \
    --attrib-hybrid on \
)

echo "[done] attack completed: $RUN_GROUP_DIR (TOPOLOGY=$TOPOLOGY)"