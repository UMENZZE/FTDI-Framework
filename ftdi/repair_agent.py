from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx

try:  # package import
    from .repair_strategy import (
        normalize_fail_type,
        get_repair_strategy,
        record_token_usage,
        should_skip_repair,
    )
except Exception:  # standalone fallback for local testing
    from repair_strategy import (  # type: ignore
        normalize_fail_type,
        get_repair_strategy,
        record_token_usage,
        should_skip_repair,
    )

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def _clean_env_value(value: str) -> str:
    value = (value or "").strip().strip('"').strip("'")
    if "#" in value:
        head = value.split("#", 1)[0].strip()
        if head:
            value = head
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "on", "yes"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_repair_config() -> Dict[str, Any]:
    api_key = _clean_env_value(os.getenv("REPAIR_API_KEY", "") or os.getenv("PROBEX_CHAT_API_KEY", ""))
    base_url = os.getenv("REPAIR_BASE_URL", "") or os.getenv("PROBEX_CHAT_BASE", "https://api.probex.top/v1")
    model = os.getenv("REPAIR_MODEL", "") or os.getenv("PROBEX_CHAT_MODEL", "deepseek-v3")
    return {
        "enabled": _env_bool("REPAIR_ENABLED", True),
        "triggers": {x.strip().upper() for x in os.getenv("REPAIR_TRIGGERS", "T0,T1,T2").split(",") if x.strip()},
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "t2_api_key": _clean_env_value(os.getenv("REPAIR_T2_API_KEY", "")) or api_key,
        "t2_base_url": os.getenv("REPAIR_T2_BASE_URL", "") or base_url,
        "t2_model": os.getenv("REPAIR_T2_MODEL", "") or os.getenv("REPAIR_STRONG_MODEL", "") or model,
        "max_tokens": _env_int("REPAIR_MAX_TOKENS", 1024),
        "timeout": _env_float("REPAIR_TIMEOUT", 45.0),
        "temperature": _env_float("REPAIR_TEMPERATURE", 0.0),
    }


def is_repair_enabled() -> bool:
    return bool(_get_repair_config()["enabled"])


def is_trigger_enabled(stage: str) -> bool:
    cfg = _get_repair_config()
    return bool(cfg["enabled"] and stage.upper() in cfg["triggers"])

# -----------------------------------------------------------------------------
# T0 deterministic patches
# -----------------------------------------------------------------------------

_PATCH_RULES: Tuple[Tuple[re.Pattern[str], str, str], ...] = tuple(
    (re.compile(pattern), replacement, label)
    for pattern, replacement, label in [
        (r"\bfrom\s+typing\s+import\s+(lst|lIST|lIst|lsit|Lst)\b", "from typing import List", "typing_import_List"),
        (r"\bfrom\s+typing\s+import\s+(int|str|float|bool|String|Integer|Float)\b\n?", "", "remove_builtin_typing_import"),
        (r"\bfrom\s+(tYping|tyPing|typign|typng)\s+import\b", "from typing import", "typing_module_typo"),
        (r"\bfrom\s+typing\s+(imoprt|improt)\b", "from typing import", "import_keyword_typo"),
        (r"\bimport\s+(mth|maht|mahth)\b", "import math", "math_import_typo"),
        (r"\bfrom\s+(colections|collecitons)\s+import\b", "from collections import", "collections_import_typo"),
        (r"\bresutl\b", "result", "name_typo"),
        (r"\bresullt\b", "result", "name_typo"),
        (r"\blenght\b", "length", "name_typo"),
        (r"\bindxe\b", "index", "name_typo"),
        (r"\bcoutner\b", "counter", "name_typo"),
        (r"\bTruE\b|\btruE\b|\bTRUE\b", "True", "bool_typo"),
        (r"\bFalsE\b|\bfalsE\b|\bFALSE\b", "False", "bool_typo"),
        (r"\bretrun\b|\bretunr\b|\bretrn\b", "return", "keyword_typo"),
        (r"\belses\s*:", "else:", "keyword_typo"),
        (r"\belse\s+if\b", "elif", "keyword_typo"),
    ]
)


def fast_patch(code: str) -> Tuple[str, List[Dict[str, str]]]:
    fixed = str(code or "")
    fixes: List[Dict[str, str]] = []
    for pattern, replacement, label in _PATCH_RULES:
        if pattern.search(fixed):
            before = fixed
            fixed = pattern.sub(replacement, fixed)
            if fixed != before:
                fixes.append({"type": label, "pattern": pattern.pattern, "replacement": replacement})
    fixed = re.sub(r"\n\s*\n\s*\n+", "\n\n", fixed).strip()
    return fixed, fixes


def detect_obvious_errors(code: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    text = str(code or "")
    if any(pattern.search(text) for pattern, _, _ in _PATCH_RULES):
        errors.append("format_deviation")
    try:
        import ast
        ast.parse(text)
    except SyntaxError:
        errors.append("syntax_import")
    return bool(errors), sorted(set(errors))

# Backward-compatible function name used by older scripts, but returns new T0 behavior.
def fast_typing_fix(code: str) -> Tuple[str, List[Dict[str, str]]]:
    return fast_patch(code)

# -----------------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------------

PROMPT_LOCAL_REPAIR = """You are the FTDI Tier-{stage} repair agent.

Repair goal:
- Fix the Python code with the smallest safe change.
- Preserve the function signature `{entry_point}(...)` and the return contract.
- Do not add print statements, demos, or explanations.
- Return only the complete fixed code inside one Python code block.

Failure type: {fail_type}
Auditor suspect span: {suspect_span}
Auditor diagnosis: {diagnosis_json}
Execution/test feedback:
{test_logs}

Task description:
{task_description}

Candidate code:
```python
{code}
```
"""

PROMPT_DEEP_REPAIR = """You are the FTDI Tier-2 deep repair agent.

The Tester has confirmed failure. Use executable feedback, the auditor diagnosis,
and typed repair priors to produce a corrected implementation.

Hard constraints:
1. Preserve the required function signature `{entry_point}(...)`.
2. Satisfy the task description and the failed tests.
3. Use the repair priors as hints, not as mandatory edits.
4. Return only the complete fixed code inside one Python code block.

Failure type: {fail_type}
Auditor suspect span: {suspect_span}
Auditor diagnosis: {diagnosis_json}

Task description:
{task_description}

Execution/test feedback:
{test_logs}

Typed repair priors:
{repair_priors}

Candidate code:
```python
{code}
```
"""

# -----------------------------------------------------------------------------
# LLM call and response parsing
# -----------------------------------------------------------------------------

def _extract_code_from_response(response_text: str) -> str:
    blocks = _CODE_BLOCK_RE.findall(str(response_text or ""))
    if blocks:
        return blocks[-1].strip()
    text = str(response_text or "").strip()
    # Accept raw code if it looks like Python code.
    if re.search(r"^\s*(from\s+|import\s+|def\s+|class\s+)", text, flags=re.MULTILINE):
        return text
    return ""


def call_repair_llm(prompt: str, *, stage: str, max_retries: int = 3) -> Tuple[Optional[str], int]:
    cfg = _get_repair_config()
    tier = stage.upper()
    api_key = cfg["t2_api_key"] if tier == "T2" else cfg["api_key"]
    base_url = cfg["t2_base_url"] if tier == "T2" else cfg["base_url"]
    model = cfg["t2_model"] if tier == "T2" else cfg["model"]
    if not api_key:
        print(f"[RepairAgent] no API key; skip {tier} LLM call")
        return None, 0

    max_tokens = cfg["max_tokens"]
    if tier == "T1":
        max_tokens = min(max_tokens, _env_int("REPAIR_T1_MAX_TOKENS", 768))
    if tier == "T2":
        max_tokens = max(max_tokens, _env_int("REPAIR_T2_MAX_TOKENS", 1200))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg["temperature"],
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{str(base_url).rstrip('/')}/chat/completions"

    last_tokens = 0
    for attempt in range(max_retries):
        if attempt:
            time.sleep(min(2 ** attempt, 8))
        try:
            with httpx.Client(timeout=cfg["timeout"]) as client:
                resp = client.post(url, headers=headers, content=json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                resp.raise_for_status()
                data = resp.json()
            usage = data.get("usage") or {}
            last_tokens = int(usage.get("total_tokens") or (len(prompt) // 4 + max_tokens // 2))
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            code = _extract_code_from_response(content)
            if code:
                record_token_usage(tier, last_tokens, success=True)
                return code, last_tokens
        except Exception as exc:
            print(f"[RepairAgent] {tier} LLM attempt {attempt + 1}/{max_retries} failed: {exc}")
    if last_tokens:
        record_token_usage(tier, last_tokens, success=False)
    return None, last_tokens

# -----------------------------------------------------------------------------
# Public repair entry
# -----------------------------------------------------------------------------

def repair_code(
    code: str,
    *,
    entry_point: str = "",
    fail_type: str | None = None,
    error_type: str | None = None,
    test_logs: str = "",
    error_message: str = "",
    task_description: str = "",
    stage: str = "T1",
    diagnosis: Optional[Mapping[str, Any]] = None,
    suspect_span: str = "unknown",
    repair_priors: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute the selected FTDI repair tier.

    ``fail_type`` is the new paper-level name. ``error_type`` is accepted only as
    an alias for callers that still use the old parameter name.
    """
    if not code:
        return None
    tier = str(stage or "T1").upper()
    if not is_trigger_enabled(tier):
        return None

    ft = normalize_fail_type(fail_type or error_type or (diagnosis or {}).get("fail_type"))
    skip, reason = should_skip_repair(ft, tier)
    if skip:
        return {"fixed_code": None, "stage": tier, "method": "skipped", "skip_reason": reason, "tokens_used": 0}

    logs = test_logs or error_message or ""
    diag_json = json.dumps(dict(diagnosis or {}), ensure_ascii=False, sort_keys=True)
    entry = entry_point or "solution"

    # T0 is intentionally deterministic by default.
    if tier == "T0":
        fixed, fixes = fast_patch(code)
        if fixes and fixed != code:
            return {
                "fixed_code": fixed,
                "stage": "T0",
                "method": "regex",
                "fixes_applied": fixes,
                "confidence": 0.90,
                "tokens_used": 0,
                "fail_type": ft,
            }
        if not _env_bool("FTDI_T0_ALLOW_LLM", False):
            return None
        prompt = PROMPT_LOCAL_REPAIR.format(
            stage="0", entry_point=entry, fail_type=ft, suspect_span=suspect_span,
            diagnosis_json=diag_json, test_logs=logs[:1200], task_description=task_description[:1000], code=code,
        )
        fixed_code, tokens = call_repair_llm(prompt, stage="T0")
        if fixed_code and fixed_code != code:
            return {"fixed_code": fixed_code, "stage": "T0", "method": "llm_micro_edit", "fixes_applied": [], "confidence": 0.75, "tokens_used": tokens, "fail_type": ft}
        return None

    if tier == "T1":
        # Start with deterministic cleanup, then ask for a constrained local edit.
        cleaned, fixes = fast_patch(code)
        base_code = cleaned if cleaned else code
        prompt = PROMPT_LOCAL_REPAIR.format(
            stage="1", entry_point=entry, fail_type=ft, suspect_span=suspect_span,
            diagnosis_json=diag_json, test_logs=logs[:1600], task_description=task_description[:1200], code=base_code,
        )
        fixed_code, tokens = call_repair_llm(prompt, stage="T1")
        if fixed_code and fixed_code != code:
            return {
                "fixed_code": fixed_code,
                "stage": "T1",
                "method": "llm_local_edit",
                "fixes_applied": fixes + [{"type": "local_edit", "fail_type": ft, "suspect_span": suspect_span}],
                "confidence": 0.80,
                "tokens_used": tokens,
                "fail_type": ft,
            }
        if fixes and base_code != code:
            return {"fixed_code": base_code, "stage": "T1", "method": "regex_preface", "fixes_applied": fixes, "confidence": 0.70, "tokens_used": 0, "fail_type": ft}
        return None

    if tier == "T2":
        prior_text = "\n".join(f"- {x}" for x in (repair_priors or [])[:3]) or "(no matched typed repair prior)"
        # T2 may still benefit from cheap cleanup before the deep call.
        cleaned, fixes = fast_patch(code)
        base_code = cleaned if fixes else code
        prompt = PROMPT_DEEP_REPAIR.format(
            entry_point=entry,
            fail_type=ft,
            suspect_span=suspect_span,
            diagnosis_json=diag_json,
            task_description=task_description[:1500] or "(no task description)",
            test_logs=logs[:1800] or "(no test logs)",
            repair_priors=prior_text,
            code=base_code,
        )
        fixed_code, tokens = call_repair_llm(prompt, stage="T2")
        if fixed_code and fixed_code != code:
            return {
                "fixed_code": fixed_code,
                "stage": "T2",
                "method": "llm_deep_repair",
                "fixes_applied": fixes + [{"type": "deep_repair", "fail_type": ft, "priors_used": len(repair_priors or [])}],
                "confidence": 0.75,
                "tokens_used": tokens,
                "fail_type": ft,
                "used_priors": list((repair_priors or [])[:3]),
            }
        if fixes and base_code != code:
            return {"fixed_code": base_code, "stage": "T2", "method": "regex_preface", "fixes_applied": fixes, "confidence": 0.65, "tokens_used": tokens, "fail_type": ft}
        return None

    return None

# -----------------------------------------------------------------------------
# Repair history
# -----------------------------------------------------------------------------

_REPAIR_HISTORY: Dict[str, List[Dict[str, Any]]] = {}


def record_repair(task_id: str, repair_result: Mapping[str, Any]) -> None:
    if not task_id:
        return
    entry = {"ts_ms": int(time.time() * 1000), **dict(repair_result)}
    _REPAIR_HISTORY.setdefault(str(task_id), []).append(entry)


def get_repair_history(task_id: str) -> List[Dict[str, Any]]:
    return list(_REPAIR_HISTORY.get(str(task_id), []))


def snapshot_repair_history(*, clear: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    snap = {task: [dict(x) for x in entries] for task, entries in _REPAIR_HISTORY.items()}
    if clear:
        _REPAIR_HISTORY.clear()
    return snap
