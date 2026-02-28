#!/usr/bin/env bash
# ============================================================
# Attack + Repair Experiment Script
# Enable T0/T2 tiered repair strategy, test repair effectiveness
# ============================================================
set -euo pipefail

# ======== Activate correct conda environment ========
# Ensure autoagent env is used to avoid missing dependencies like httpx
if [ -f "/data/10T/Msx/miniconda3/etc/profile.d/conda.sh" ]; then
    source /data/10T/Msx/miniconda3/etc/profile.d/conda.sh
    conda activate autoagent
    echo "[env] Activated conda env: autoagent ($(python --version))"
fi

# ---------- Base config ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

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

# ======== Step 2: Set API base based on USE_XIAOAI ========
# USE_XIAOAI=on: use probex API as base model; otherwise use local vLLM
USE_XIAOAI="${USE_XIAOAI:-on}"
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

# ======== Repair system uses remote API (Probex deepseek-v3) ========
# T0/T1/T2 repair calls use REPAIR_* config, independent from OPENAI_*
# Repair API config is set in .env.inference:
#   REPAIR_BASE_URL="https://api.probex.top/v1"
#   REPAIR_API_KEY="${PROBEX_CHAT_API_KEY}"
#   REPAIR_MODEL="deepseek-v3"

# ---------- Experiment parameters ----------
# Dataset with separated base_input and plus_input fields for split evaluation
DATA="${DATA:-/data/10T/Msx/project/LLMfix/DATA/humaneval_plus_with_inputs.jsonl}"  # Evaluation dataset path
PM="${PM:-0.775}"                            # Message-level injection prob (0.0~1.0, set 0 to disable attack)
PE="${PE:-0.3}"                            # Line-level injection prob
MAX_LINES="${MAX_LINES:-1}"                # Max lines per injection
SEED="${SEED:-42}"                         # Injection random seed
CONC="${CONC:-6}"                          # Concurrent workers
FAIL_PATIENCE="${FAIL_PATIENCE:-3}"        # Consecutive failure tolerance
USE_SUBSET="${USE_SUBSET:-off}"            # Subset sampling: on=enable, off=disable
SAMPLE_N="${SAMPLE_N:-20}"                # Subset problem count
SAMPLE_SEED="${SAMPLE_SEED:-20251117}"     # Subset sampling random seed
TS="${TS:-$(date +%Y%m%d-%H%M%S)}"         # Result directory timestamp

# ---------- Repair config ----------
export REPAIR_ENABLED="${REPAIR_ENABLED:-on}"      # Repair master switch: on=enable, off=disable
export REPAIR_TRIGGERS="${REPAIR_TRIGGERS:-T0,T2}" # Enabled stages: T0=fast fix, T1=pre-test fix, T2=deep fix
export REPAIR_MODEL="${REPAIR_MODEL:-deepseek-v3}" # LLM model for repair
export REPAIR_BASE_URL="${REPAIR_BASE_URL:-https://xiaoai.plus/v1}"  # LLM API endpoint

CURE_MODE="${CURE_MODE:-distilled}"        # Cure prompt source: distilled|generic|hybrid
# Recommended: distilled_cures_v3.json (12 error types, includes regex_fix)
DISTILLED_CURE_PATH="${DISTILLED_CURE_PATH:-$REPO_ROOT/autoagents_ext/distilled_cures_v3.json}"
IMM_THRESHOLD="${IMM_THRESHOLD:-0.25}"     # Immunity trigger threshold
IMM_TOPK="${IMM_TOPK:-2}"                  # Immunity prompt count
IMM_FORCE_RUNTIME="${IMM_FORCE_RUNTIME:-off}"  # Force immunity on runtime error
IMM_COOLDOWN="${IMM_COOLDOWN:-1}"          # Immunity cooldown rounds

# ---------- Ablation switches (disabled by default) ----------
# AUDITOR_ENABLED: on=use diagnostic signals, off=disable (w/o Auditor ablation)
# REPAIR_KB_MODE: distilled=specialized distilled lib, generic=generic lib, none=disable (w/o Repair KB ablation)
export AUDITOR_ENABLED="${AUDITOR_ENABLED:-on}"
export REPAIR_KB_MODE="${REPAIR_KB_MODE:-distilled}"

# ---------- Test dataset mode switch ----------
# base: Run HumanEval original test cases only (~7-10 per problem)
# plus: Run HumanEval+ enhanced test cases only (~80-1000 per problem)
# all: Run all test cases (default)
HUMANEVAL_TEST_MODE="${HUMANEVAL_TEST_MODE:-plus}"  # Default: plus only

# ---------- Subset preparation ----------
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

prepare_subset

# ---------- Output directory ----------
MODE="attack+repair-${CURE_MODE}"
GROUP_NAME="mas-env-${HUMANEVAL_TEST_MODE}-PM${PM}-PE${PE}-ml${MAX_LINES}${SUBTAG}-${MODE}-${TS}"
RUN_GROUP_DIR="$(pwd)/traces/${GROUP_NAME}"
mkdir -p "$RUN_GROUP_DIR"

echo "============================================================"
echo "  Attack + Repair Experiment"
echo "============================================================"
echo "  Dataset:     ${DATA_TO_USE}"
echo "  Test mode:   ${HUMANEVAL_TEST_MODE} (base=HumanEval, plus=HumanEval+)"
echo "  Subset:      ${USE_SUBSET} (N=${SAMPLE_N})"
echo "  Attack prob: PM=${PM}, PE=${PE}"
echo "  Repair:      ${REPAIR_ENABLED}"
echo "  Repair stages: ${REPAIR_TRIGGERS}"
echo "  Repair model:  ${REPAIR_MODEL}"
echo "  Output dir:  ${RUN_GROUP_DIR}"
echo "============================================================"

# ---------- Run inference ----------
(
  export RUN_GROUP_DIR
  export HUMANEVAL_TEST_MODE="$HUMANEVAL_TEST_MODE"  # Test dataset mode
  export AUTOINJECT_ENABLED=on
  export AUTOINJECT_PM="$PM"
  export AUTOINJECT_PE="$PE"
  export AUTOINJECT_ERROR_TYPE=semantic
  export AUTOINJECT_MAX_LINES="$MAX_LINES"
  export AUTOINJECT_SEED="$SEED"
  export AUTOINJECT_PROTECT_COMMENTS=on
  export AUTOINJECT_TASK_LEVEL=0
  export AUTOINJECT_TASK_PROB=1.0
  export AUTOINJECT_MAX_MSG_PER_TASK=2
  export CURE_MODE="$CURE_MODE"
  export DISTILLED_CURE_PATH="$DISTILLED_CURE_PATH"
  export AUDITOR=on
  
  # ======== Repair system env vars (must export in subshell) ========
  export REPAIR_ENABLED="$REPAIR_ENABLED"
  export REPAIR_TRIGGERS="$REPAIR_TRIGGERS"
  export REPAIR_MODEL="$REPAIR_MODEL"
  export REPAIR_BASE_URL="$REPAIR_BASE_URL"
  export REPAIR_API_KEY="${REPAIR_API_KEY:-$PROBEX_CHAT_API_KEY}"

  python examples/humaneval_official_eval.py \
    --problems-json "$DATA_TO_USE" \
    --concurrency "$CONC" \
    --use-mas-env \
    --group-results \
    --write-samples \
    --bad-sig-action retry \
    --sig-retries 1 \
    --auditor on \
    --fail-patience "$FAIL_PATIENCE"
)

# ---------- Results statistics ----------
echo ""
echo "============================================================"
echo "  Experiment Results Summary"
echo "============================================================"

RESULTS_FILE="${RUN_GROUP_DIR}/results.jsonl"
if [ -f "$RESULTS_FILE" ]; then
    python -c "
import json
from pathlib import Path
from collections import Counter

results = [json.loads(line) for line in Path('$RESULTS_FILE').read_text().splitlines() if line.strip()]
total = len(results)
if total == 0:
    print('No results found')
    exit()

passed = sum(1 for r in results if r.get('pass', False))
pass_at_1 = sum(1 for r in results if r.get('pass@1', False))

# Count repair info
repair_triggered = 0
repair_stages = Counter()

traces_dir = Path('$RUN_GROUP_DIR')
for f in traces_dir.glob('*.jsonl'):
    if f.name == 'results.jsonl':
        continue
    for line in f.read_text().splitlines():
        try:
            rec = json.loads(line)
            if rec.get('repair'):
                repair_triggered += 1
                repair_stages[rec['repair'].get('repair_stage', 'unknown')] += 1
        except:
            pass

print(f'Total tasks:       {total}')
print(f'Passed (pass):     {passed} ({100*passed/total:.1f}%)')
print(f'Pass@1:            {pass_at_1} ({100*pass_at_1/total:.1f}%)')
print()
print(f'Repair triggered:  {repair_triggered}')
print(f'Repair stages:     {dict(repair_stages)}')
"
else
    echo "Results file not found: $RESULTS_FILE"
fi

echo ""
echo "[done] Attack+Repair completed: $RUN_GROUP_DIR"
echo "============================================================"
