from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    from .repair_strategy import normalize_fail_type
except Exception:
    from repair_strategy import normalize_fail_type  # type: ignore

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
_REPAIR_HISTORY: Dict[str, List[Dict[str, Any]]] = {}
_INJECT_STATE: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "on", "yes"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _workspace_root() -> Path:
    return Path(os.getenv("WORKSPACE_ROOT", "./workspace")).expanduser().resolve()


def _extract_code_block(text: str) -> str:
    blocks = _CODE_BLOCK_RE.findall(str(text or ""))
    if blocks:
        candidates = [b for b in blocks if len(b.strip()) >= 16 or re.search(r"\b(def|return|for|while|if)\b", b)]
        return (candidates[-1] if candidates else blocks[-1]).strip()
    return ""


def _replace_code_block(text: str, old_code: str, new_code: str) -> str:
    if not new_code:
        return text
    if old_code and old_code in text:
        return str(text).replace(old_code, new_code)
    if _CODE_BLOCK_RE.search(str(text or "")):
        return _CODE_BLOCK_RE.sub(f"```python\n{new_code}\n```", str(text), count=1)
    # If the hook passed raw code rather than a markdown message, return raw code.
    if re.search(r"^\s*(from\s+|import\s+|def\s+|class\s+)", str(text or ""), flags=re.MULTILINE):
        return new_code
    return f"{text}\n\n```python\n{new_code}\n```"


def _load_json_env(prefix: str, task_id: str) -> Dict[str, Any]:
    raw = os.getenv(f"{prefix}__{_safe_name(task_id)}", "")
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

# -----------------------------------------------------------------------------
# Evaluation-time AutoInject
# -----------------------------------------------------------------------------

def _inject_should_run(task_id: str) -> bool:
    explicit = os.getenv("AUTOINJECT_ENABLED", "").strip().lower()
    if explicit in {"0", "false", "no", "off"}:
        return False
    pm = _env_float("AUTOINJECT_PM", 0.0)
    if explicit in {"1", "true", "yes", "on"} and pm <= 0:
        pm = 1.0
    if pm <= 0:
        return False
    state = _INJECT_STATE.setdefault(task_id, {})
    if "task_on" not in state:
        seed = int(os.getenv("AUTOINJECT_SEED", "13")) + abs(hash(task_id)) % 100000
        rng = random.Random(seed)
        state["task_on"] = rng.random() < pm
        state["rng"] = rng
        state["used"] = 0
    return bool(state["task_on"])


def _mutate_line(line: str) -> str:
    mutations = [
        (r"\+", "-"),
        (r"-", "+"),
        (r"<=", "<"),
        (r">=", ">"),
        (r"==", "!="),
        (r"\bTrue\b", "False"),
        (r"\bFalse\b", "True"),
        (r"return\s+(.+)$", r"return \1 + 1"),
    ]
    for pattern, repl in mutations:
        if re.search(pattern, line):
            return re.sub(pattern, repl, line, count=1)
    return line + "  # injected"


def inject_if_needed(text: str, *, is_code: bool = True, task_id: str = "unknown") -> str:
    """Controlled evaluation-only injection.

    This is deliberately minimal: it perturbs Coder output before Tester runs and
    records a manifest. FTDI repair itself does not depend on this mechanism.
    """
    if not is_code or not _inject_should_run(task_id):
        return text
    code = _extract_code_block(text) or str(text or "")
    if not code.strip():
        return text

    lines = code.splitlines()
    editable = [i for i, line in enumerate(lines) if line.strip() and not line.strip().startswith("#")]
    if not editable:
        return text

    state = _INJECT_STATE.setdefault(task_id, {})
    max_lines = max(1, _env_int("AUTOINJECT_MAX_LINES", _env_int("AUTOINJECT_MAX_MODIFIED_LINES", 1)))
    pe = _env_float("AUTOINJECT_PE", 0.3)
    rng = state.get("rng") or random.Random(13)
    n = min(max_lines, max(1, int(round(len(editable) * pe))))
    chosen = sorted(rng.sample(editable, k=min(n, len(editable))))

    before = list(lines)
    for idx in chosen:
        lines[idx] = _mutate_line(lines[idx])
    injected_code = "\n".join(lines)
    if injected_code == code:
        return text

    out = _replace_code_block(text, code, injected_code)
    state["used"] = int(state.get("used", 0)) + 1

    manifest_dir = _workspace_root() / "injections"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts_ms": int(time.time() * 1000),
        "task_id": task_id,
        "changed_lines": [i + 1 for i in chosen],
        "before": [before[i] for i in chosen],
        "after": [lines[i] for i in chosen],
    }
    try:
        with (manifest_dir / f"injection_manifest_{_safe_name(task_id)}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return out

# -----------------------------------------------------------------------------
# Typed repair prior retrieval
# -----------------------------------------------------------------------------

def _candidate_library_paths() -> List[Path]:
    here = Path(__file__).resolve().parent
    paths: List[Path] = []
    custom = os.getenv("FTDI_PRIOR_LIBRARY", "").strip()
    if custom:
        paths.append(Path(custom).expanduser())
    paths.extend([
        here / "typed_repair_prior_library.json",
        here / "distilled_cures_filtered.json",
        here / "distilled_cures.json",
        here / "cure_library.json",
    ])
    return paths


def _load_prior_library() -> Dict[str, Any]:
    for path in _candidate_library_paths():
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            continue
    return {}


def _entry_to_text(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, Mapping):
        parts = []
        for key in ("recipe", "text", "fix_description", "cause_text", "root_cause", "action"):
            val = entry.get(key)
            if val:
                parts.append(str(val).strip())
        tests = entry.get("tests") or entry.get("examples") or []
        if isinstance(tests, list) and tests:
            parts.append("Tests: " + "; ".join(str(x) for x in tests[:3]))
        return " | ".join(x for x in parts if x)
    return ""


def _rank_prior_texts(texts: List[str], context: str, suspect_span: str, k: int) -> List[str]:
    tokens = set(re.findall(r"[A-Za-z_]{3,}", (context + " " + suspect_span).lower()))
    scored: List[Tuple[int, str]] = []
    for text in texts:
        t = text.strip()
        if not t:
            continue
        hits = sum(1 for tok in set(re.findall(r"[A-Za-z_]{3,}", t.lower())) if tok in tokens)
        scored.append((hits, t))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in scored[:max(1, k)]]


def select_typed_repair_priors(
    *,
    fail_type: str,
    code: str,
    test_logs: str,
    suspect_span: str,
    topk: int = 3,
) -> List[str]:
    if os.getenv("REPAIR_KB_MODE", "distilled").strip().lower() == "none":
        return []
    library = _load_prior_library()
    if not library:
        return []
    ft = normalize_fail_type(fail_type)
    raw_entries: List[Any] = []
    for key in (ft, fail_type, "default"):
        value = library.get(key)
        if isinstance(value, list):
            raw_entries.extend(value)
        elif isinstance(value, dict):
            # Common formats: {"entries": [...]}, {"recipes": [...]}
            for subkey in ("entries", "recipes", "items"):
                if isinstance(value.get(subkey), list):
                    raw_entries.extend(value[subkey])
    texts = [_entry_to_text(x) for x in raw_entries]
    context = f"{test_logs}\n{code[:1200]}"
    return _rank_prior_texts(texts, context, suspect_span, topk)

# -----------------------------------------------------------------------------
# Repair bridge entry point expected by hook.py
# -----------------------------------------------------------------------------

def repair_if_needed(
    text: str,
    *,
    task_id: str,
    entry_point: str = "",
    task_description: str = "",
    stage: str = "T1",
    diagnosis: Optional[Mapping[str, Any]] = None,
    suspect_span: str = "unknown",
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not _env_bool("REPAIR_ENABLED", True):
        return None
    tier = str(stage or "T1").upper()
    triggers = {x.strip().upper() for x in os.getenv("REPAIR_TRIGGERS", "T0,T1,T2").split(",") if x.strip()}
    if tier not in triggers:
        return None

    code = _extract_code_block(text) or str(text or "")
    if not code.strip():
        return None

    diag = dict(diagnosis or {})
    if not diag:
        diag = _load_json_env("FTDI_DIAGNOSIS", task_id)
    failure_record = _load_json_env("FTDI_FAILURE_RECORD", task_id)
    test_logs = str(failure_record.get("test_logs") or "")
    fail_type = normalize_fail_type(str(diag.get("fail_type") or failure_record.get("diagnosis", {}).get("fail_type") or "semantic_logic"))
    span = str(suspect_span or diag.get("suspect_span") or "unknown")

    priors = select_typed_repair_priors(
        fail_type=fail_type,
        code=code,
        test_logs=test_logs,
        suspect_span=span,
        topk=_env_int("FTDI_REPAIR_PRIOR_TOPK", 3),
    )

    try:
        from autoagents_ext.repair_agent import repair_code, record_repair
    except Exception:
        try:
            from repair_agent import repair_code, record_repair  # type: ignore
        except Exception as exc:
            print(f"[RepairBridge] cannot import repair_agent: {exc}")
            return None

    result = repair_code(
        code,
        entry_point=entry_point,
        fail_type=fail_type,
        test_logs=test_logs,
        task_description=task_description,
        stage=tier,
        diagnosis=diag,
        suspect_span=span,
        repair_priors=priors,
    )
    if not result or not result.get("fixed_code"):
        return None

    fixed_code = str(result["fixed_code"])
    if fixed_code.strip() == code.strip():
        return None

    fixed_text = _replace_code_block(text, code, fixed_code)
    meta: Dict[str, Any] = {
        "repair_stage": tier,
        "fail_type": fail_type,
        "suspect_span": span,
        "auditor_s": diag.get("s"),
        "auditor_action": diag.get("a"),
        "repair_method": result.get("method"),
        "repair_confidence": result.get("confidence"),
        "fixes_applied": result.get("fixes_applied", []),
        "tokens_used": int(result.get("tokens_used", 0) or 0),
        "priors_used": priors,
        "ts_ms": int(time.time() * 1000),
    }
    _REPAIR_HISTORY.setdefault(str(task_id), []).append(meta)
    try:
        record_repair(task_id, meta)
    except Exception:
        pass
    os.environ[f"FTDI_REPAIR_METADATA__{_safe_name(task_id)}"] = json.dumps(meta, ensure_ascii=False)
    return fixed_text, meta


def consume_repair_metadata(task_id: str) -> Optional[Dict[str, Any]]:
    entries = _REPAIR_HISTORY.get(str(task_id), [])
    return dict(entries[-1]) if entries else None


def snapshot_repair_history(*, clear: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    snap = {task: [dict(x) for x in entries] for task, entries in _REPAIR_HISTORY.items()}
    if clear:
        _REPAIR_HISTORY.clear()
    return snap
