
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Mapping, Tuple

TIERS: Tuple[str, str, str] = ("T0", "T1", "T2")

# Failure types emitted by the new Online Auditor.
SHALLOW_TYPES = {"syntax_import", "name_attribute", "format_deviation"}
LOCAL_EDIT_TYPES = {"uncertainty", "plan_inconsistency", "type_error"}
DEEP_TYPES = {"boundary_condition", "off_by_one", "semantic_logic"}

# Aliases kept only to normalize older log labels into the new paper vocabulary.
_FAIL_TYPE_ALIASES = {
    "syntax_error": "syntax_import",
    "indentation_error": "syntax_import",
    "import_error": "syntax_import",
    "module_not_found": "syntax_import",
    "name_error": "name_attribute",
    "attribute_error": "name_attribute",
    "deviation": "format_deviation",
    "format": "format_deviation",
    "consistency": "plan_inconsistency",
    "logic_error": "semantic_logic",
    "algorithm_change": "semantic_logic",
    "complex_semantic": "semantic_logic",
    "empty_input_handling": "boundary_condition",
    "default": "semantic_logic",
}

REPAIR_STRATEGY_MAP: Dict[str, Dict[str, Any]] = {
    "syntax_import": {
        "preferred_tier": "T0",
        "method": "regex/ast",
        "expected_success_rate": 0.90,
        "description": "Syntax, indentation, and import-level failures.",
    },
    "name_attribute": {
        "preferred_tier": "T0",
        "method": "regex/ast",
        "expected_success_rate": 0.85,
        "description": "Undefined names, attribute typos, and shallow symbol errors.",
    },
    "format_deviation": {
        "preferred_tier": "T0",
        "method": "regex/format",
        "expected_success_rate": 0.80,
        "description": "Missing code block, wrong function signature, or output-format deviation.",
    },
    "uncertainty": {
        "preferred_tier": "T1",
        "method": "local_llm_edit",
        "expected_success_rate": 0.55,
        "description": "Ambiguous or abnormal candidate output requiring constrained local editing.",
    },
    "plan_inconsistency": {
        "preferred_tier": "T1",
        "method": "local_llm_edit",
        "expected_success_rate": 0.60,
        "description": "Mismatch between planner constraints and generated implementation.",
    },
    "type_error": {
        "preferred_tier": "T2",
        "method": "deep_llm_repair",
        "expected_success_rate": 0.60,
        "description": "Runtime type mismatch or incompatible operation.",
    },
    "boundary_condition": {
        "preferred_tier": "T2",
        "method": "deep_llm_repair+priors",
        "expected_success_rate": 0.55,
        "description": "Boundary, empty-input, or edge-case failure.",
    },
    "off_by_one": {
        "preferred_tier": "T2",
        "method": "deep_llm_repair+priors",
        "expected_success_rate": 0.50,
        "description": "Index, loop range, or +/-1 boundary failure.",
    },
    "semantic_logic": {
        "preferred_tier": "T2",
        "method": "deep_llm_repair+priors",
        "expected_success_rate": 0.45,
        "description": "Semantic logic failure detected by executable feedback.",
    },
}

DEFAULT_PRIOR: Dict[str, Dict[str, float]] = {
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

_LOG_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"ModuleNotFoundError|ImportError|SyntaxError|IndentationError", re.I), "syntax_import"),
    (re.compile(r"NameError|AttributeError", re.I), "name_attribute"),
    (re.compile(r"TypeError|ValueError", re.I), "type_error"),
    (re.compile(r"IndexError|KeyError|ZeroDivisionError|empty input|edge case", re.I), "boundary_condition"),
    (re.compile(r"off[\s_-]*by[\s_-]*one|range\(|index|len\(", re.I), "off_by_one"),
    (re.compile(r"AssertionError|expected\s+.+\s+got\s+.+|wrong answer|mismatch", re.I), "semantic_logic"),
)

_TOKEN_USAGE: Dict[str, Dict[str, int]] = {tier: {"calls": 0, "tokens": 0, "success": 0} for tier in TIERS}


def normalize_fail_type(fail_type: str | None) -> str:
    key = str(fail_type or "").strip()
    if not key:
        return "semantic_logic"
    key = key.lower().replace("-", "_").replace("/", "_")
    return key if key in REPAIR_STRATEGY_MAP else _FAIL_TYPE_ALIASES.get(key, "semantic_logic")


def detect_error_type(test_logs: str) -> str:
    text = str(test_logs or "")
    for pattern, fail_type in _LOG_PATTERNS:
        if pattern.search(text):
            return fail_type
    return "semantic_logic" if text else "format_deviation"


def get_repair_strategy(fail_type: str | None) -> Dict[str, Any]:
    key = normalize_fail_type(fail_type)
    return dict(REPAIR_STRATEGY_MAP[key])


def recommended_action(fail_type: str | None) -> str:
    return get_repair_strategy(fail_type).get("preferred_tier", "T2")


def get_empirical_prior(fail_type: str | None) -> Dict[str, float]:
    key = normalize_fail_type(fail_type)
    return dict(DEFAULT_PRIOR.get(key, DEFAULT_PRIOR["semantic_logic"]))


def cost_benefit_choice(feasible: Tuple[str, ...], fail_type: str | None, tier_costs: Mapping[str, int]) -> str:
    prior = get_empirical_prior(fail_type)
    return max(feasible, key=lambda tier: float(prior.get(tier, 0.0)) / max(1, int(tier_costs.get(tier, 1))))


def should_skip_repair(fail_type: str | None, stage: str | None = None) -> Tuple[bool, str]:
    """Return skip decision for explicit user ablation switches only.

    The paper-level Budget Gate lives in hook.py, so this function avoids hidden
    heuristic skipping.  It only respects environment switches useful for ablation.
    """
    if os.getenv("REPAIR_FORCE_SKIP", "off").strip().lower() in {"1", "true", "on", "yes"}:
        return True, "REPAIR_FORCE_SKIP"
    disabled = {x.strip().upper() for x in os.getenv("REPAIR_DISABLED_TIERS", "").split(",") if x.strip()}
    if stage and stage.upper() in disabled:
        return True, f"tier_disabled:{stage.upper()}"
    return False, ""


def recommend_strategy(test_logs: str, diagnosis: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    if diagnosis and diagnosis.get("fail_type"):
        fail_type = normalize_fail_type(str(diagnosis.get("fail_type")))
    else:
        fail_type = detect_error_type(test_logs)
    strategy = get_repair_strategy(fail_type)
    return {
        "fail_type": fail_type,
        "strategy": strategy,
        "recommended_action": strategy["preferred_tier"],
        "expected_success_rate": strategy["expected_success_rate"],
    }


def record_token_usage(stage: str, tokens: int, success: bool = False) -> None:
    tier = str(stage or "").upper()
    if tier not in _TOKEN_USAGE:
        _TOKEN_USAGE[tier] = {"calls": 0, "tokens": 0, "success": 0}
    _TOKEN_USAGE[tier]["calls"] += 1
    _TOKEN_USAGE[tier]["tokens"] += max(0, int(tokens or 0))
    if success:
        _TOKEN_USAGE[tier]["success"] += 1


def get_token_stats() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tier, data in _TOKEN_USAGE.items():
        calls = max(1, data["calls"])
        success = max(1, data["success"])
        out[tier] = {
            "calls": data["calls"],
            "tokens": data["tokens"],
            "success": data["success"],
            "avg_tokens_per_call": data["tokens"] / calls,
            "success_rate": data["success"] / calls,
            "tokens_per_success": data["tokens"] / success,
        }
    return out


def reset_token_stats() -> None:
    for tier in list(_TOKEN_USAGE):
        _TOKEN_USAGE[tier] = {"calls": 0, "tokens": 0, "success": 0}


def print_token_summary() -> None:
    stats = get_token_stats()
    print("\n" + "=" * 60)
    print("FTDI repair token usage")
    print("=" * 60)
    for tier in TIERS:
        s = stats.get(tier, {"calls": 0, "tokens": 0, "success": 0, "avg_tokens_per_call": 0, "success_rate": 0})
        print(f"{tier}: calls={s['calls']}, tokens={s['tokens']}, success={s['success']}, "
              f"avg={s['avg_tokens_per_call']:.1f}, rate={s['success_rate']:.1%}")
    print("=" * 60)
