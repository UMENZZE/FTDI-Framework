# ftdi/repair_strategy.py
# -*- coding: utf-8 -*-
"""
Repair Strategy Module - FTDI framework core component

Implementation:
1. Error type -> repair strategy mapping (REPAIR_STRATEGY_MAP)
2. Token cost tracking
4. Smart skip for logic errors

Based on FTDI framework design:
- Fault Tolerance: controlled injection via AutoInject
- Diagnosis: Who&When attribution
- Repair: strategy-based repair implemented in this module
"""

from __future__ import annotations

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# ============ Error Type -> Repair Strategy Map ============

REPAIR_STRATEGY_MAP: Dict[str, Dict[str, Any]] = {
    # =============== Syntax-level (T0 efficient) ===============
    "syntax_error": {
        "handler": "T0",
        "method": "regex/ast",
        "priority": 1,
        "expected_success_rate": 0.90,
        "description": "Syntax errors: unmatched brackets, indentation errors",
    },
    "name_error": {
        "handler": "T0",
        "method": "regex",
        "priority": 1,
        "expected_success_rate": 0.85,
        "description": "Undefined variable: typo e.g. resutl -> result",
    },
    "import_error": {
        "handler": "T0",
        "method": "regex",
        "priority": 1,
        "expected_success_rate": 0.95,
        "description": "Import error, e.g. from typing import lIST",
    },
    "type_annotation_error": {
        "handler": "T0",
        "method": "regex",
        "priority": 1,
        "expected_success_rate": 0.92,
        "description": "Type annotation error, e.g. wrong typing module usage",
    },
    "typo_error": {
        "handler": "T0",
        "method": "regex",
        "priority": 1,
        "expected_success_rate": 0.88,
        "description": "Spelling error, e.g. TruE -> True",
    },
    
    # =============== Semantic-level (requires T1/T2) ===============
    "type_error": {
        "handler": "T2",
        "method": "llm",
        "priority": 2,
        "expected_success_rate": 0.60,
        "description": "Type error, e.g. argument type mismatch",
    },
    "boundary_condition": {
        "handler": "T2",
        "method": "llm",
        "priority": 2,
        "expected_success_rate": 0.55,
        "description": "Boundary condition error, e.g. empty list handling",
    },
    "off_by_one": {
        "handler": "T2",
        "method": "llm",
        "priority": 2,
        "expected_success_rate": 0.50,
        "description": "Off-by-one error, e.g. range boundary",
    },
    "empty_input_handling": {
        "handler": "T2",
        "method": "llm",
        "priority": 2,
        "expected_success_rate": 0.55,
        "description": "Missing empty input handling",
    },
    "semantic_mutation": {
        "handler": "T2",
        "method": "llm",
        "priority": 2,
        "expected_success_rate": 0.45,
        "description": "Semantic mutation, e.g. operator substitution + -> -",
    },
    "control_flow_change": {
        "handler": "T2",
        "method": "llm+test",
        "priority": 2,
        "expected_success_rate": 0.40,
        "description": "Control flow change, e.g. condition inversion",
    },
    
    # =============== Logic-level (base model limitation) ===============
    "algorithm_change": {
        "handler": "skip",
        "method": None,
        "priority": 3,
        "expected_success_rate": 0.15,
        "flag": "base_model_limit",
        "description": "Algorithm logic error, requires base model comprehension",
    },
    "logic_error": {
        "handler": "conditional",  # decided by gain history
        "method": "llm",
        "priority": 3,
        "expected_success_rate": 0.20,
        "flag": "may_have_gain",
        "description": "Logic error, may have repair gain",
    },
    "complex_semantic": {
        "handler": "skip",
        "method": None,
        "priority": 3,
        "expected_success_rate": 0.10,
        "flag": "base_model_limit",
        "description": "Complex semantic error, beyond repair capability",
    },
    
    # =============== Default type ===============
    "default": {
        "handler": "T2",
        "method": "llm",
        "priority": 2,
        "expected_success_rate": 0.35,
        "description": "Unknown type, use default T2 strategy",
    },
}


# ============ Error Type Detector ============

# Error summary -> error type mapping rules
_ERROR_TYPE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Syntax/import errors (T0 can handle)
    (re.compile(r"ImportError.*cannot import name", re.I), "import_error"),
    (re.compile(r"from typing import\s+[a-z]", re.I), "import_error"),
    (re.compile(r"NameError.*is not defined", re.I), "name_error"),
    (re.compile(r"SyntaxError", re.I), "syntax_error"),
    (re.compile(r"IndentationError", re.I), "syntax_error"),
    (re.compile(r"unmatched|unexpected.*'[\}\)\]]'", re.I), "syntax_error"),
    
    # Type errors
    (re.compile(r"TypeError", re.I), "type_error"),
    
    # Boundary/index errors
    (re.compile(r"IndexError.*out of range", re.I), "boundary_condition"),
    (re.compile(r"list index out of range", re.I), "boundary_condition"),
    (re.compile(r"KeyError", re.I), "boundary_condition"),
    
    # Logic errors (output mismatch)
    (re.compile(r"FAIL@\d+.*expected.*got", re.I), "logic_error"),
    (re.compile(r"expected\s+.+\s+got\s+.+", re.I), "logic_error"),
    (re.compile(r"assertion.*failed", re.I), "logic_error"),
    
    # Empty input handling
    (re.compile(r"input=\(\[\],", re.I), "empty_input_handling"),
    (re.compile(r"input=\('',", re.I), "empty_input_handling"),
    
    # off-by-one
    (re.compile(r"off.by.one|off by one", re.I), "off_by_one"),
]


def detect_error_type(error_summary: str) -> str:
    """Detect error type from error summary"""
    if not error_summary:
        return "default"
    
    for pattern, error_type in _ERROR_TYPE_PATTERNS:
        if pattern.search(error_summary):
            return error_type
    
    return "default"


def get_repair_strategy(error_type: str) -> Dict[str, Any]:
    """Get repair strategy for given error type"""
    return REPAIR_STRATEGY_MAP.get(error_type, REPAIR_STRATEGY_MAP["default"])


def should_skip_repair(error_type: str, error_summary: str = "") -> Tuple[bool, str]:
    """
    Check whether repair should be skipped (to save token budget)
    
    Smart skip strategy:
    1. T0-level errors (syntax, name, import): always attempt repair
    2. handler="skip" types: skip directly (e.g. algorithm_change, complex_semantic)
    3. Logic error types: decided by env vars and historical success rate
    
    Returns:
        Tuple[bool, str]: (should_skip, skip_reason)
    """
    strategy = get_repair_strategy(error_type)
    handler = strategy.get("handler", "T2")
    expected_rate = strategy.get("expected_success_rate", 0.35)
    flag = strategy.get("flag", "")
    
    # 1. T0-level errors: always attempt repair
    t0_types = {"syntax_error", "name_error", "import_error", "type_annotation_error", "typo_error"}
    if error_type in t0_types or handler == "T0":
        return False, ""
    
    # 2. Explicitly marked as skip (base_model_limit)
    if handler == "skip":
        return True, f"base_model_limit: {strategy.get('description', error_type)}"
    
    # 3. Check if logic error (by error_type or error_summary)
    is_logic = error_type in ("logic_error", "algorithm_error")
    
    # Detect logic error indicators from error summary
    if error_summary and not is_logic:
        summary_lower = error_summary.lower()
        
        # Exclude T0-type error messages
        t0_indicators = ["syntaxerror", "nameerror", "importerror", "typeerror", "indentationerror"]
        if any(ind in summary_lower for ind in t0_indicators):
            return False, ""
        
        # If Traceback present, it's a runtime error, not pure logic error
        if "traceback" in summary_lower or "error" in summary_lower:
            return False, ""
        
        # Detect pure logic error (no Traceback, only I/O mismatch)
        logic_keywords = [
            "wrong answer", "wrong result", "incorrect output",
            "mismatch",
            "algorithm",
            "assertion failed",
        ]
        # Only classify as logic error when no traceback/error and logic keywords present
        if any(kw.lower() in summary_lower for kw in logic_keywords):
            is_logic = True
    
    # 4. If logic error, check whether repair is allowed
    if is_logic:
        allow_logic = os.getenv("REPAIR_ALLOW_LOGIC_ERROR", "on").lower() in ("1", "true", "on", "yes")
        if not allow_logic:
            return True, "logic_error_disabled: logic error repair disabled (save tokens)"
        
        # If expected success rate too low, suggest skip
        min_rate = float(os.getenv("REPAIR_MIN_SUCCESS_RATE", "0.15"))
        if expected_rate < min_rate:
            return True, f"low_success_rate: expected success rate {expected_rate:.0%} < threshold {min_rate:.0%}"
    
    return False, ""



# ============ Token cost tracking ============

_TOKEN_USAGE: Dict[str, Dict[str, int]] = {
    "T0": {"calls": 0, "tokens": 0, "success": 0},
    "T1": {"calls": 0, "tokens": 0, "success": 0},
    "T2": {"calls": 0, "tokens": 0, "success": 0},
}


def record_token_usage(stage: str, tokens: int, success: bool = False) -> None:
    """Record token usage"""
    if stage not in _TOKEN_USAGE:
        _TOKEN_USAGE[stage] = {"calls": 0, "tokens": 0, "success": 0}
    
    _TOKEN_USAGE[stage]["calls"] += 1
    _TOKEN_USAGE[stage]["tokens"] += tokens
    if success:
        _TOKEN_USAGE[stage]["success"] += 1


def get_token_stats() -> Dict[str, Dict[str, Any]]:
    """Get token usage stats"""
    stats = {}
    for stage, data in _TOKEN_USAGE.items():
        calls = data["calls"]
        tokens = data["tokens"]
        success = data["success"]
        
        stats[stage] = {
            "calls": calls,
            "tokens": tokens,
            "success": success,
            "avg_tokens_per_call": tokens / max(1, calls),
            "success_rate": success / max(1, calls),
            "tokens_per_success": tokens / max(1, success),
        }
    
    return stats


def print_token_summary() -> None:
    """Print token usage summary"""
    stats = get_token_stats()
    
    print("\n" + "=" * 60)
    print("            Token Cost Analysis")
    print("=" * 60)
    
    total_tokens = 0
    total_success = 0
    
    for stage in ["T0", "T1", "T2"]:
        if stage not in stats:
            continue
        s = stats[stage]
        total_tokens += s["tokens"]
        total_success += s["success"]
        
        print(f"\n  【{stage}】")
        print(f"    Calls: {s['calls']}")
        print(f"    Total tokens: {s['tokens']:,}")
        print(f"    Successes: {s['success']}")
        print(f"    Success rate: {s['success_rate']:.1%}")
        print(f"    Avg tokens/call: {s['avg_tokens_per_call']:.0f}")
        print(f"    Tokens/success: {s['tokens_per_success']:.0f}")
    
    print("\n" + "-" * 60)
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total successes: {total_success}")
    if total_success > 0:
        print(f"  Cost per success: {total_tokens / total_success:.0f} tokens/success")
    print("=" * 60)


def reset_token_stats() -> None:
    """Reset token stats"""
    global _TOKEN_USAGE
    _TOKEN_USAGE = {
        "T0": {"calls": 0, "tokens": 0, "success": 0},
        "T1": {"calls": 0, "tokens": 0, "success": 0},
        "T2": {"calls": 0, "tokens": 0, "success": 0},
    }



# ============ Strategy Recommendation ============

def recommend_strategy(error_summary: str, code: str = "") -> Dict[str, Any]:
    """
    Recommend best repair strategy from error message
    
    Returns:
        {
            "error_type": str,
            "strategy": Dict,
            "should_skip": bool,
            "skip_reason": str,
            "recommended_model": str,
        }
    """
    error_type = detect_error_type(error_summary)
    strategy = get_repair_strategy(error_type)
    should_skip, skip_reason = should_skip_repair(error_type, error_summary)
    
    # Recommend model based on strategy
    handler = strategy.get("handler", "T2")
    if handler == "T0":
        recommended_model = "regex"  # T0 prefers regex
    elif handler == "skip":
        recommended_model = None
    else:
        # T1/T2 use LLM, select model by complexity
        expected_rate = strategy.get("expected_success_rate", 0.35)
        if expected_rate >= 0.6:
            recommended_model = "deepseek-v3"  # standard model
        else:
            # Complex problems may need stronger model
            recommended_model = os.getenv("REPAIR_STRONG_MODEL", "qwen3-235b-a22b")
    
    return {
        "error_type": error_type,
        "strategy": strategy,
        "should_skip": should_skip,
        "skip_reason": skip_reason,
        "recommended_model": recommended_model,
        "handler": handler,
    }
