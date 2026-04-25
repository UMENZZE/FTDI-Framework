from __future__ import annotations

import json
import os
import re
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from autoagents_ext.auditor import Auditor, AuditorCfg, extract_last_python_block

DEFAULT_WORKSPACE = os.environ.get("WORKSPACE_ROOT", "./workspace")
_PATCHED_INSTANCES: set[int] = set()

TIERS = ("T0", "T1", "T2")
SHALLOW_ROUTING = {"syntax_import", "name_attribute", "format_deviation"}
DEFAULT_PRIOR = {
    "syntax_import": {"T0": 0.90, "T1": 0.20, "T2": 0.10},
    "name_attribute": {"T0": 0.85, "T1": 0.35, "T2": 0.15},
    "format_deviation": {"T0": 0.80, "T1": 0.30, "T2": 0.10},
    "uncertainty": {"T0": 0.20, "T1": 0.55, "T2": 0.40},
    "plan_inconsistency": {"T0": 0.15, "T1": 0.60, "T2": 0.45},
    "type_error": {"T0": 0.20, "T1": 0.45, "T2": 0.60},
    "boundary_condition": {"T0": 0.10, "T1": 0.35, "T2": 0.55},
    "off_by_one": {"T0": 0.10, "T1": 0.40, "T2": 0.65},
    "semantic_logic": {"T0": 0.05, "T1": 0.30, "T2": 0.65},
}


@dataclass(frozen=True)
class BudgetState:
    r_rem: int
    t_rem: int
    used_tokens: int
    used_rounds: int
    r_max: int
    t_max: int


@dataclass(frozen=True)
class TierDecision:
    tier: str
    reason: str
    feasible: Tuple[str, ...]
    budget: BudgetState


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _safe_task_id(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id))


def _ensure_trace_path(task_id: str) -> Path:
    base = Path(DEFAULT_WORKSPACE) / "traces"
    group = os.environ.get("RUN_GROUP_DIR", "").strip()
    path = base / group if group else base
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{_safe_task_id(task_id)}.jsonl"


def _estimate_tokens(text: str) -> int:
    # Lightweight accounting used only for online gating when the runtime does
    # not expose tokenizer-level usage. Evaluation scripts can still use their
    # exact token logs for reporting.
    return max(1, len(str(text or "")) // 4)


def _tier_costs() -> Dict[str, int]:
    return {
        "T0": _env_int("FTDI_COST_T0", 128),
        "T1": _env_int("FTDI_COST_T1", 512),
        "T2": _env_int("FTDI_COST_T2", 1536),
    }


def _prior_table() -> Dict[str, Dict[str, float]]:
    raw = os.environ.get("FTDI_PRIOR_JSON", "").strip()
    if not raw:
        return DEFAULT_PRIOR
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return DEFAULT_PRIOR


def _inject_enabled(user_setting: Optional[bool]) -> bool:
    if user_setting is not None:
        return bool(user_setting)
    explicit = os.environ.get("AUTOINJECT_ENABLED", "").strip().lower()
    if explicit in {"0", "false", "no", "off"}:
        return False
    if explicit in {"1", "true", "yes", "on"}:
        return True
    try:
        return float(os.environ.get("AUTOINJECT_PM", "0")) > 0.0
    except Exception:
        return False


def _has_inject_bridge() -> bool:
    try:
        import autoagents_ext.inject_bridge  # noqa: F401
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Budget gate and tier routing
# -----------------------------------------------------------------------------

def _update_token_accounting(env: Any, task_id: str, msg_text: str) -> None:
    safe = _safe_task_id(task_id)
    key = f"FTDI_TOKENS_USED__{safe}"
    used = _env_int(key, 0)
    used += _estimate_tokens(msg_text)
    os.environ[key] = str(used)


def _budget_state(env: Any, task_id: str) -> BudgetState:
    safe = _safe_task_id(task_id)
    r_max = _env_int("FTDI_R_MAX", _env_int("MAX_ROUNDS", 12))
    t_max = _env_int("FTDI_T_MAX", _env_int("TOKEN_BUDGET", 10000))
    used_rounds = int(getattr(env, "current_round", 0) or 0)
    used_tokens = _env_int(f"FTDI_TOKENS_USED__{safe}", 0)
    return BudgetState(
        r_rem=max(0, r_max - used_rounds),
        t_rem=max(0, t_max - used_tokens),
        used_tokens=used_tokens,
        used_rounds=used_rounds,
        r_max=r_max,
        t_max=t_max,
    )


def _failure_count(task_id: str) -> int:
    return _env_int(f"FTDI_FAILURE_COUNT__{_safe_task_id(task_id)}", 0)


def _increment_failure_count(task_id: str) -> int:
    safe = _safe_task_id(task_id)
    count = _failure_count(task_id) + 1
    os.environ[f"FTDI_FAILURE_COUNT__{safe}"] = str(count)
    return count


def _select_repair_tier(diagnosis: Mapping[str, Any], env: Any, task_id: str) -> TierDecision:
    tau = _env_float("FTDI_TAU", 0.70)
    n_up = _env_int("FTDI_ESCALATE_AFTER", 2)
    costs = _tier_costs()
    budget = _budget_state(env, task_id)

    s = float(diagnosis.get("s", 0.0) or 0.0)
    fail_type = str(diagnosis.get("fail_type", "") or "")
    recommended = str(diagnosis.get("a", "None") or "None")

    if s <= tau:
        return TierDecision("Stop", "score_below_tau", (), budget)
    if budget.r_rem <= 0:
        return TierDecision("Stop", "round_budget_exhausted", (), budget)
    if budget.t_rem < costs["T0"]:
        return TierDecision("Stop", "token_budget_below_T0", (), budget)

    feasible = tuple(tier for tier in TIERS if costs[tier] <= budget.t_rem)
    if not feasible:
        return TierDecision("Stop", "no_affordable_tier", (), budget)

    if fail_type in SHALLOW_ROUTING and "T0" in feasible:
        return TierDecision("T0", "shallow_failure_type", feasible, budget)

    if _failure_count(task_id) >= n_up and "T2" in feasible:
        return TierDecision("T2", "escalation_threshold_reached", feasible, budget)

    if recommended in feasible:
        return TierDecision(recommended, "auditor_recommendation", feasible, budget)

    prior = _prior_table().get(fail_type, {})
    if prior:
        best = max(feasible, key=lambda k: float(prior.get(k, 0.0)) / max(1, costs[k]))
        return TierDecision(best, "cost_benefit_prior", feasible, budget)

    return TierDecision(feasible[0], "default_lowest_affordable", feasible, budget)


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------

def _latest_history(env: Any) -> Iterable[Any]:
    for attr in ("history", "messages", "dialogue", "trajectory"):
        value = getattr(env, attr, None)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            return value
    return []


def _is_tester_failure(env: Any, msg: Any) -> bool:
    if bool(getattr(env, "pass_flag", False)):
        return False
    fail_summary = str(getattr(env, "fail_summary", "") or "")
    content = str(getattr(msg, "content", "") or "")
    if fail_summary.strip():
        return True
    return bool(re.search(r"\b(FAIL|FAILED|Error|Exception|Traceback|AssertionError)\b", content, re.IGNORECASE))


def _replace_or_append_code_block(message_text: str, repaired_text: str) -> str:
    repaired_code = extract_last_python_block(repaired_text) or str(repaired_text or "").strip()
    if not repaired_code:
        return message_text
    current_code = extract_last_python_block(message_text)
    if current_code:
        return str(message_text).replace(current_code, repaired_code)
    return f"{message_text}\n\n```python\n{repaired_code}\n```"


def _store_json_env(prefix: str, task_id: str, payload: Mapping[str, Any]) -> None:
    os.environ[f"{prefix}__{_safe_task_id(task_id)}"] = json.dumps(payload, ensure_ascii=False)


def _append_trace(env: Any, task_id: str, trace_fp: Optional[Path], record: Mapping[str, Any]) -> None:
    if trace_fp is None:
        return
    trace_off = getattr(env, "_ftdi_trace_off", False)
    if trace_off:
        return
    try:
        with trace_fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[FTDI] trace write failed: {exc}")


def _call_repair_bridge(
    *,
    code_text: str,
    task_id: str,
    problem: Mapping[str, Any],
    tier: str,
    diagnosis: Mapping[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    from autoagents_ext.inject_bridge import repair_if_needed

    result = repair_if_needed(
        code_text,
        task_id=task_id,
        entry_point=str(problem.get("entry_point", "") or ""),
        task_description=str(problem.get("prompt", "") or "")[:1000],
        stage=tier,
        diagnosis=dict(diagnosis),
        suspect_span=str(diagnosis.get("suspect_span", "unknown")),
    )
    if not result:
        return None
    fixed_text, repair_meta = result
    if not isinstance(repair_meta, dict):
        repair_meta = {"raw_meta": repair_meta}
    repair_meta.update({"tier": tier})
    return str(fixed_text), repair_meta


# -----------------------------------------------------------------------------
# Patched publish method
# -----------------------------------------------------------------------------

async def _ftdi_publish(self: Any, msg: Any):
    task_id = getattr(self, "_ftdi_task_id", "unknown")
    problem = getattr(self, "_ftdi_problem", {})
    auditor: Auditor = getattr(self, "_ftdi_auditor")
    inject_enabled = bool(getattr(self, "_ftdi_inject_enabled", False))
    has_inject_bridge = bool(getattr(self, "_ftdi_has_inject_bridge", False))
    trace_fp: Optional[Path] = getattr(self, "_ftdi_trace_fp", None)
    original_publish = getattr(self, "_ftdi_original_publish")

    role = str(getattr(msg, "role", "") or "")
    content = str(getattr(msg, "content", "") or "")
    safe = _safe_task_id(task_id)

    repair_meta: Optional[Dict[str, Any]] = None
    diagnosis: Optional[Dict[str, Any]] = None
    tier_decision: Optional[TierDecision] = None

    # Apply repaired code from the previous failed Tester turn to the next Coder output.
    if role == "Coder":
        queued_repair = os.environ.pop(f"FTDI_REPAIRED_CODE__{safe}", "")
        if queued_repair:
            content = _replace_or_append_code_block(content, queued_repair)
            msg.content = content
            print(f"[FTDI] applied queued repair to Coder output: task={task_id}")

        # Evaluation-only fault injection: perturb Coder output before Tester sees it.
        if inject_enabled and has_inject_bridge:
            try:
                from autoagents_ext.inject_bridge import inject_if_needed
                os.environ["AUTOINJECT_ROUND"] = str(int(getattr(self, "current_round", 0) or 0))
                content = inject_if_needed(content, is_code=True, task_id=task_id)
                msg.content = content
            except Exception as exc:
                print(f"[FTDI] AutoInject failed: {exc}")

        os.environ[f"FTDI_LAST_CANDIDATE__{safe}"] = content[:20000]

    # Maintain lightweight token accounting for online gating.
    _update_token_accounting(self, task_id, str(getattr(msg, "content", "") or ""))

    # Online Auditor is invoked only after Tester confirms a failure.
    if role == "Tester" and _is_tester_failure(self, msg):
        failure_index = _increment_failure_count(task_id)
        candidate_text = os.environ.get(f"FTDI_LAST_CANDIDATE__{safe}", "")
        test_logs = "\n".join([
            str(getattr(self, "fail_summary", "") or ""),
            str(getattr(msg, "content", "") or ""),
        ]).strip()

        diagnosis = auditor.audit_failure(
            history=_latest_history(self),
            candidate_code=candidate_text,
            test_logs=test_logs,
            problem=problem,
        )
        tier_decision = _select_repair_tier(diagnosis, self, task_id)

        failure_record = {
            "H_t": "trace_stored_in_jsonl",
            "candidate_code": candidate_text,
            "test_logs": test_logs[:4000],
            "diagnosis": diagnosis,
            "failure_index": failure_index,
            "tier": tier_decision.tier,
            "tier_reason": tier_decision.reason,
        }
        _store_json_env("FTDI_DIAGNOSIS", task_id, diagnosis)
        _store_json_env("FTDI_FAILURE_RECORD", task_id, failure_record)
        _store_json_env("FTDI_TIER_DECISION", task_id, {
            "tier": tier_decision.tier,
            "reason": tier_decision.reason,
            "feasible": list(tier_decision.feasible),
            "budget": tier_decision.budget.__dict__,
        })

        if tier_decision.tier != "Stop" and has_inject_bridge:
            try:
                repair_result = _call_repair_bridge(
                    code_text=candidate_text,
                    task_id=task_id,
                    problem=problem,
                    tier=tier_decision.tier,
                    diagnosis=diagnosis,
                )
                if repair_result:
                    fixed_text, repair_meta = repair_result
                    os.environ[f"FTDI_REPAIRED_CODE__{safe}"] = fixed_text
                    print(
                        f"[FTDI] queued {tier_decision.tier} repair: "
                        f"task={task_id}, reason={tier_decision.reason}"
                    )
            except Exception as exc:
                print(f"[FTDI] repair bridge failed: {exc}")

    _append_trace(self, task_id, trace_fp, {
        "ts": int(time.time() * 1000),
        "task_id": task_id,
        "round": int(getattr(self, "current_round", 0) or 0),
        "role": role,
        "raw_text": str(getattr(msg, "content", "") or ""),
        "extracted_code": extract_last_python_block(str(getattr(msg, "content", "") or "")),
        "pass_flag": bool(getattr(self, "pass_flag", False)),
        "tester_summary": str(getattr(self, "fail_summary", "") or "")[:2000],
        "diagnosis": diagnosis,
        "tier_decision": None if tier_decision is None else {
            "tier": tier_decision.tier,
            "reason": tier_decision.reason,
            "feasible": list(tier_decision.feasible),
            "budget": tier_decision.budget.__dict__,
        },
        "repair": repair_meta,
        "inject_enabled": inject_enabled,
    })

    return await original_publish(self, msg)


# -----------------------------------------------------------------------------
# Public instrumentation entry
# -----------------------------------------------------------------------------

def instrument_env(
    env: Any,
    *,
    task_id: str,
    problem: Dict[str, Any],
    auditor_enabled: bool = True,
    inject_enabled: Optional[bool] = None,
    workspace_root: Optional[str] = None,
):
    """Patch one Environment instance with the FTDI loop."""
    env_id = id(env)
    if env_id in _PATCHED_INSTANCES:
        return env
    _PATCHED_INSTANCES.add(env_id)

    if workspace_root:
        global DEFAULT_WORKSPACE
        DEFAULT_WORKSPACE = workspace_root

    if auditor_enabled is True:
        auditor_enabled = _env_bool("AUDITOR", True)

    EnvClass = env.__class__
    if not hasattr(EnvClass, "_ftdi_original_publish_saved"):
        EnvClass._ftdi_original_publish_saved = EnvClass.publish_message

    original_publish = EnvClass._ftdi_original_publish_saved
    trace_off = _env_bool("TRACE_OFF", False)
    trace_fp = _ensure_trace_path(task_id)

    attrs = {
        "_ftdi_task_id": task_id,
        "_ftdi_problem": problem,
        "_ftdi_auditor_enabled": bool(auditor_enabled),
        "_ftdi_auditor": Auditor(AuditorCfg()) if auditor_enabled else Auditor(AuditorCfg()),
        "_ftdi_inject_enabled": _inject_enabled(inject_enabled),
        "_ftdi_has_inject_bridge": _has_inject_bridge(),
        "_ftdi_trace_off": trace_off,
        "_ftdi_trace_fp": trace_fp,
        "_ftdi_original_publish": original_publish,
    }
    for key, value in attrs.items():
        try:
            object.__setattr__(env, key, value)
        except Exception:
            env.__dict__[key] = value

    bound = types.MethodType(_ftdi_publish, env)
    try:
        object.__setattr__(env, "publish_message", bound)
    except Exception:
        env.__dict__["publish_message"] = bound

    print(
        f"[FTDI] task={task_id}, auditor={bool(auditor_enabled)}, "
        f"inject={attrs['_ftdi_inject_enabled']}, trace_off={trace_off}"
    )
    return env
