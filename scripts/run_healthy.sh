#!/usr/bin/env bash
set -euo pipefail

# =============================================================
# Supports oneshot / hierarchical MAS mode switch.
# Supports topology: linear / flat / hier
# =============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ======== Step 1: Load credentials (don't override existing vars) ========
if [ -f .env.inference ]; then
  # Only load API keys etc., skip OPENAI_API_BASE
  while IFS= read -r line; do
    # Skip comments, empty lines, OPENAI_API_BASE lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$line" ]] && continue
    [[ "$line" =~ OPENAI_API_BASE ]] && continue
    [[ "$line" =~ OPENAI_BASE_URL ]] && continue
    # Export other vars (strip quotes and comments)
    if [[ "$line" =~ ^[[:space:]]*([A-Z_][A-Z0-9_]*)=(.+)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      # Strip trailing comments
      value="${value%%#*}"
      # Strip leading/trailing whitespace and quotes
      value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^["\'"'"']//' -e 's/["\'"'"']$//')"
      export "$key=$value"
    fi
  done < .env.inference
fi

# ======== Step 2: Set API base based on USE_XIAOAI ========
# USE_XIAOAI=on: use probex API as base model; otherwise use local vLLM
USE_XIAOAI="${USE_XIAOAI:-off}"
if [ "$USE_XIAOAI" = "on" ]; then
  export OPENAI_API_BASE="https://api.probex.top/v1"
  export OPENAI_API_KEY="${PROBEX_CHAT_API_KEY:-sk-k6A4ulAEe2xe9XzBtvAsmkiR7ABfUUOzlEXBGNsqdvRFf22J}"
  export OPENAI_API_MODEL="qwen2.5-32b-instruct"  # probex base model
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
DATA="/data/10T/Msx/project/LLMfix/DATA/humaneval_plus_with_inputs.jsonl"  # Full evaluation dataset
SEED=42  # Base random seed
CONC=6  # Concurrent workers
FAIL_PATIENCE=5  # Max consecutive failure retries

# ============ MAS mode switch ============
# oneshot: Single-round generation (no hierarchical structure)
# hierarchical: Use hierarchical structure (Planner/Coder/Tester)
MAS_MODE=${MAS_MODE:-"hierarchical"}  # Default: hierarchical

# ============ Topology switch ============
# linear: Linear chain (Planner->Coder->Tester, single round)
# flat: Flat mutual-view (mutual feedback, multi-round iteration)
# hier: Hierarchical (Planner summarizes diagnostics, Coder sees summary only)
TOPOLOGY=${TOPOLOGY:-"hier"}  # Default: hierarchical

# ============ Test dataset mode switch ============
# base: Run HumanEval original test cases only (~7-10 per problem)
# plus: Run HumanEval+ enhanced test cases only (~80-1000 per problem)
# all: Run all test cases (default)
HUMANEVAL_TEST_MODE=${HUMANEVAL_TEST_MODE:-"base"}  # Default: base only

# Subset sampling
USE_SUBSET=${USE_SUBSET:-"off"}  # Enable subset sampling
SAMPLE_N=${SAMPLE_N:-5}  # Number of sampled problems
SAMPLE_SEED=${SAMPLE_SEED:-20251117}  # Sampling random seed

TS=${TS:-$(date +%Y%m%d-%H%M%S)}  # Result directory timestamp, can override externally

DATA_TO_USE="$DATA"
SUBTAG=""

prepare_subset() {
  if [[ "$USE_SUBSET" != "on" || "$SAMPLE_N" -le 0 ]]; then
    return
  fi
  mkdir -p workspace
  local subset="workspace/humaneval_plus_subset_${SAMPLE_N}_${SAMPLE_SEED}_${TS}.jsonl"
  python - "$DATA" "$subset" "$SAMPLE_N" "$SAMPLE_SEED" <<'PY'
import json, random, sys
src, dst, n, seed = sys.argv[1:5]
with open(src, "r", encoding="utf-8") as f:
    lines = [ln for ln in f if ln.strip()]
n = min(int(n), len(lines))
random.seed(int(seed))
idx = sorted(random.sample(range(len(lines)), n))
with open(dst, "w", encoding="utf-8") as g:
    for i in idx:
        g.write(lines[i])
PY
  DATA_TO_USE="$subset"
  SUBTAG="-N${SAMPLE_N}-s${SAMPLE_SEED}"
}

run_healthy() {
  local mas_flag=""
  local mode_tag=""
  
  case "$MAS_MODE" in
    oneshot)
      mas_flag=""  # Don't use --use-mas-env
      mode_tag="oneshot"
      ;;
    hierarchical|hier|mas)
      mas_flag="--use-mas-env"
      mode_tag="hierarchical"
      ;;
    *)
      echo "[warn] unknown MAS_MODE=$MAS_MODE, using hierarchical" >&2
      mas_flag="--use-mas-env"
      mode_tag="hierarchical"
      ;;
  esac

  echo "[run] MODE=healthy+repair MAS_MODE=$mode_tag TOPOLOGY=$TOPOLOGY TEST_MODE=$HUMANEVAL_TEST_MODE"
  local group_name="mas-env${SUBTAG}-${TOPOLOGY}-${HUMANEVAL_TEST_MODE}-healthy-repair-${mode_tag}-${TS}"
  local group_dir="$(pwd)/traces/${group_name}"
  mkdir -p "$group_dir"
  
  (
    export RUN_GROUP_DIR="$group_dir"  # Output directory for current mode
    export AUTOINJECT_ENABLED=on   # Enable repair infrastructure (PM=0 means no actual injection)
    export AUTOINJECT_PM=0         # Message-level injection prob = 0 (no attack)
    export AUTOINJECT_PE=0         # Line-level injection prob = 0 (no attack)
    export AUTOINJECT_ERROR_TYPE=semantic
    export AUTOINJECT_MAX_LINES=1
    export AUTOINJECT_SEED=42
    export AUTOINJECT_PROTECT_COMMENTS=on
    export ATTRIBUTION_HYBRID=off
    export ATTRIBUTION_DIR=""
    export AUDITOR=on              # Enable auditor
    export HUMANEVAL_TEST_MODE="$HUMANEVAL_TEST_MODE"

    python examples/humaneval_official_eval.py \
      --problems-json "$DATA_TO_USE" \
      --concurrency "$CONC" \
      $mas_flag \
      --topology "$TOPOLOGY" \
      --group-results \
      --write-samples \
      --bad-sig-action retry \
      --sig-retries 1 \
      --auditor on \
      --fail-patience "$FAIL_PATIENCE" \
      --attrib-hybrid on \
  )
}

prepare_subset
run_healthy

echo "[done] healthy+repair test completed (MAS_MODE=$MAS_MODE, TOPOLOGY=$TOPOLOGY, MODEL=Llama-3-8B)"
