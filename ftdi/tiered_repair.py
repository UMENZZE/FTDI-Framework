# autoagents_ext/inject_bridge.py (drop-in replacement of file or append functions and wire call)

from __future__ import annotations
import os, hashlib, re, json, random, time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Import online immunity learning module
try:
    from autoagents_ext.immunity_strategy import (
        add_successful_repair_to_library,
        record_token_usage,
    )
    _HAS_ONLINE_LEARNING = True
except ImportError:
    _HAS_ONLINE_LEARNING = False

# Per-task injection state
_INJECT_STATE = {}  # task_id -> {"decided": bool, "task_on": bool, "used": int}
_IMMUNITY_FLAGS: dict[str, dict] = {}
_IMMUNITY_HISTORY: dict[str, list[dict]] = {}

# Distilled cure cache
_DISTILLED_CURE_CACHE: Dict[str, Any] = {"path": None, "data": None, "mtime": 0}


def consume_immunity_metadata(task_id: str) -> dict | None:
    """Return and clear the latest immunity metadata for a task."""
    return _IMMUNITY_FLAGS.pop(task_id, None)


def snapshot_immunity_flags(*, clear: bool = False) -> dict[str, list[dict]]:
    """Return a copy of all immunity metadata recorded during this run."""
    snap = {tid: [dict(entry) for entry in entries] for tid, entries in _IMMUNITY_HISTORY.items()}
    if clear:
        _IMMUNITY_HISTORY.clear()
    return snap


def _record_immunity_snapshot(task_id: str, payload: dict) -> None:
    if not task_id or not isinstance(payload, dict):
        return
    data = dict(payload)
    _IMMUNITY_HISTORY.setdefault(task_id, []).append(data)


def _current_round_index(fallback: int = 0) -> int:
    try:
        return int(os.getenv("AUTOINJECT_ROUND", fallback))
    except Exception:
        return fallback


# Cure library: fail_type -> multiple prompts
_CURE_LIBRARY_PATH = Path(__file__).parent / "cure_library.json"
# Use filtered distilled library (matches AutoInject attack patterns)
_DISTILLED_CURE_PATH = Path(__file__).parent / "distilled_cures_filtered.json"
_DISTILLED_CURE_FULL_PATH = Path(__file__).parent / "distilled_cures.json"

# ============ Ablation switches ============
# AUDITOR_ENABLED: on/off
# REPAIR_KB_MODE: distilled/generic/none

def _is_auditor_enabled() -> bool:
    """Check if Auditor diagnosis is enabled (ablation switch)."""
    return os.getenv("AUDITOR_ENABLED", "on").strip().lower() in ("1", "true", "on", "yes")

def _get_repair_kb_mode() -> str:
    """Get repair knowledge base mode (ablation switch).

    Returns:
        'distilled': use distilled library (specialized attribution)
        'generic': use generic library (cure_library.json)
        'none': do not use any attribution library
    """
    mode = os.getenv("REPAIR_KB_MODE", "distilled").strip().lower()
    if mode in ("distilled", "generic", "none"):
        return mode
    return "distilled"


def _load_distilled_cures() -> Dict[str, List[Dict]]:
    """Load distilled cure library (with cache and hot-reload support)"""
    global _DISTILLED_CURE_CACHE
    
    # Support custom path via env var
    custom_path = os.getenv("DISTILLED_CURE_PATH", "").strip()
    cure_path = Path(custom_path) if custom_path else _DISTILLED_CURE_PATH
    
    if not cure_path.exists():
        return {}
    
    try:
        mtime = cure_path.stat().st_mtime
        # Check cache validity
        if (_DISTILLED_CURE_CACHE["path"] == str(cure_path) and 
            _DISTILLED_CURE_CACHE["mtime"] == mtime and
            _DISTILLED_CURE_CACHE["data"] is not None):
            return _DISTILLED_CURE_CACHE["data"]
        
        data = json.loads(cure_path.read_text(encoding="utf-8"))
        _DISTILLED_CURE_CACHE = {"path": str(cure_path), "data": data, "mtime": mtime}
        return data
    except Exception:
        return {}


def _match_error_pattern(code: str, tester_summary: str, cure_entry: Dict) -> float:
    """Compute error pattern match score (0-1) for precise distilled cure matching"""
    score = 0.0
    patterns = cure_entry.get("patterns", [])
    tags = cure_entry.get("tags", [])
    attack_patterns = cure_entry.get("matched_attack_patterns", [])
    
    # Pattern matching: check if code or tester_summary contains specific patterns
    if patterns:
        text_to_check = f"{code} {tester_summary}".lower()
        for pattern in patterns:
            if pattern.lower() in text_to_check:
                score += 0.5
                break
    
    # Tag matching: check keywords in error message
    if tags:
        err_text = tester_summary.lower()
        matched_tags = sum(1 for tag in tags if tag.lower() in err_text)
        if matched_tags > 0:
            score += 0.3 * min(1.0, matched_tags / len(tags))
    
    # Attack pattern matching (for AutoInject attack types)
    # Match attack patterns based on code features
    attack_score = 0.0
    code_lower = code.lower()
    if attack_patterns:
        # Relational operators
        if "relational-operator" in attack_patterns and any(op in code for op in ["==", "!=", "<=", ">=", "<", ">"]):
            attack_score += 0.2
        # Boundary/index
        if "boundary-index" in attack_patterns and any(kw in code_lower for kw in ["range", "index", "len(", "[", "]"]):
            attack_score += 0.2
        # Boolean
        if "boolean-value" in attack_patterns and ("true" in code_lower or "false" in code_lower):
            attack_score += 0.2
        # Loop logic
        if "loop-logic" in attack_patterns and any(kw in code_lower for kw in ["for ", "while "]):
            attack_score += 0.15
        # Return value
        if "return-value" in attack_patterns and "return" in code_lower:
            attack_score += 0.15
        # Arithmetic operators
        if "arithmetic-operator" in attack_patterns and any(op in code for op in ["+", "-", "*", "/", "%"]):
            attack_score += 0.1
        score += attack_score
    
    # Confidence bonus
    confidence = cure_entry.get("confidence", 0.5)
    score *= (0.5 + 0.5 * confidence)
    
    return min(1.0, score)


def _select_distilled_cures(
    fail_type: str,
    code: str,
    tester_summary: str,
    *,
    topk: int = 2,
    min_score: float = 0.1
) -> List[str]:
    """Select best matching distilled cures.
    
    Prefers filtered version (matches AutoInject patterns),
    falls back to full version if filtered is empty.
    
    Respects REPAIR_KB_MODE ablation switch:
    - 'none': return empty list (w/o Repair KB ablation)
    - 'generic': skip distilled, caller uses cure_library.json
    - 'distilled': use distilled library normally
    """
    # ======== Ablation switch check ========
    kb_mode = _get_repair_kb_mode()
    if kb_mode == "none":
        # w/o Repair KB ablation: do not use any attribution library
        return []
    if kb_mode == "generic":
        # Generic library mode: skip distilled, caller uses cure_library.json
        return []
    
    distilled = _load_distilled_cures()
    
    # If filtered version empty, try loading full version
    if not distilled and _DISTILLED_CURE_FULL_PATH.exists():
        try:
            distilled = json.loads(_DISTILLED_CURE_FULL_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    if not distilled:
        return []
    
    candidates: List[Tuple[float, str]] = []
    
    # Check exact fail_type first
    for ft in [fail_type, "default"]:
        entries = distilled.get(ft, [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text", "")
            if not text:
                continue
            
            score = _match_error_pattern(code, tester_summary, entry)
            
            # Give extra weight to distilled sources
            source = entry.get("source", "")
            if "distilled" in source:
                score *= 1.3
            # Give higher weight to filtered version (v2)
            if source == "distilled_v2":
                score *= 1.2
            
            if score >= min_score:
                candidates.append((score, text))
    
    # Sort by score, return top-k
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in candidates[:topk]]


def _repo_root() -> Path:
    """Return the AutoAgents repo root regardless of WORKSPACE_ROOT overrides."""
    return Path(__file__).resolve().parents[1]


def _probex_manifest_path() -> Path:
    """Absolute path for the global probex manifest, overridable via env."""
    custom = os.environ.get("PROBEX_MANIFEST_PATH")
    if custom:
        return Path(custom).expanduser().resolve()
    return (_repo_root() / "workspace" / "injections" / "probex_manifest.jsonl").resolve()


def _workspace_root_path() -> Path:
    ws = os.environ.get("WORKSPACE_ROOT", "./workspace")
    try:
        return Path(ws).expanduser().resolve()
    except Exception:
        return Path("./workspace").resolve()


def _resolve_trace_group_dir(ws: Path | None = None, *, ensure: bool = True) -> Path:
    """Return the concrete traces directory honoring RUN_GROUP_DIR semantics."""
    ws_path = ws or _workspace_root_path()
    base = ws_path / "traces"
    raw = (os.environ.get("RUN_GROUP_DIR", "") or "").strip().strip("/\\")
    if not raw:
        target = base
    else:
        candidate = Path(raw)
        if candidate.is_absolute():
            target = candidate
        elif candidate.parts and candidate.parts[0] == "traces":
            target = ws_path / candidate
        else:
            target = base / candidate
    if ensure:
        target.mkdir(parents=True, exist_ok=True)
    return target

def _load_cure_library():
    try:
        data = json.loads(_CURE_LIBRARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {"default": ["General self-check: boundary conditions / function signatures / complexity."]}
    return data

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

def _trace_file(task_id: str) -> Path:
    """Resolve the current run's trace file path, honoring RUN_GROUP_DIR if present.

    Falls back to traces/<task>.jsonl when group folder is absent.
    """
    ws = _workspace_root_path()
    base = _resolve_trace_group_dir(ws)
    safe = _safe_name(task_id)
    return base / f"{safe}.jsonl"

def _injections_path(task_id: str) -> Path:
    """Return path for AutoInject manifest under a dedicated 'injections' folder.

    Structure mirrors traces: <WORKSPACE_ROOT>/injections/injection_manifest_<task_id>.jsonl
    """
    ws = os.environ.get("WORKSPACE_ROOT", "./workspace")
    p = Path(ws) / "injections"
    p.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(task_id)
    return p / f"injection_manifest_{safe}.jsonl"


def _count_role_msgs(task_id: str, role: str) -> int:
    fp = _trace_file(task_id)
    if not fp.exists():
        return 0
    cnt = 0
    try:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if obj.get("role") == role:
                    cnt += 1
    except Exception:
        return 0
    return cnt

_RUNTIME_ERR_PAT = re.compile(r"(TypeError|IndexError|KeyError|ValueError|ZeroDivisionError|NameError|AttributeError|AssertionError|RecursionError|TimeoutError|ImportError|SyntaxError|IndentationError)")

def _tester_has_runtime_error(task_id: str, summary: str | None = None) -> bool:
    s = summary if summary is not None else _latest_tester_summary(task_id)
    if not s:
        return False
    return bool(_RUNTIME_ERR_PAT.search(s))

def _ns_key(task_id: str) -> str:
    return f"AUDITOR_DIAGNOSIS__{_safe_name(task_id)}"

def _latest_auditor_score(task_id: str, default: float = 0.0) -> float:
    fp = _trace_file(task_id)
    if not fp.exists():
        return default
    try:
        with fp.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            back = min(65536, size)
            f.seek(size - back)
            chunk = f.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        for ln in reversed(lines):
            if '"auditor"' in ln:
                try:
                    obj = json.loads(ln)
                    sc = obj.get("auditor", {}) or {}
                    if isinstance(sc, dict) and "score" in sc:
                        return float(sc["score"])
                except Exception:
                    pass
        return default
    except Exception:
        return default

def _latest_tester_summary(task_id: str) -> str:
    """Tail the trace and return the most recent tester_summary (if any)."""
    fp = _trace_file(task_id)
    if not fp.exists():
        return ""
    try:
        with fp.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            back = min(65536, size)
            f.seek(size - back)
            chunk = f.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        for ln in reversed(lines):
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if obj.get("role") == "Tester":
                return str(obj.get("tester_summary") or obj.get("raw_text") or "")
        return ""
    except Exception:
        return ""

def _latest_planner_text(task_id: str) -> str:
    fp = _trace_file(task_id)
    if not fp.exists():
        return ""
    try:
        with fp.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            back = min(65536, size)
            f.seek(size - back)
            chunk = f.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        for ln in reversed(lines):
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if obj.get("role") == "Planner":
                return str(obj.get("raw_text") or "")
        return ""
    except Exception:
        return ""

_SUMMARY_FAILTYPE_HINTS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"list index out of range|index\s+out of range", re.IGNORECASE), ["off_by_one", "boundary_condition"]),
    (re.compile(r"expected\s+.+\s+got\s+.+", re.IGNORECASE), ["boundary_condition", "deviation"]),
    (re.compile(r"assert(?:ion)?error|assert\s+.+==", re.IGNORECASE), ["off_by_one", "boundary_condition"]),
    (re.compile(r"lengths?\s+do\s+not\s+match|different\s+lengths", re.IGNORECASE), ["boundary_condition"]),
    (re.compile(r"noneType", re.IGNORECASE), ["type_error", "boundary_condition"]),
]

def _summary_fail_type_hints(summary: str | None) -> list[str]:
    text = summary or ""
    if not text:
        return []
    hints: list[str] = []
    for pattern, types in _SUMMARY_FAILTYPE_HINTS:
        try:
            if pattern.search(text):
                hints.extend(types)
        except Exception:
            continue
    out: list[str] = []
    for t in hints:
        key = str(t or "").strip()
        if not key or key in out:
            continue
        out.append(key)
    return out

_ATTRIB_CACHE = {"dir": None, "map": {}}  # lazy-loaded task_id -> {cause_text, step_window}

def _load_attribution_map(dirpath: str) -> dict:
    """Load attribution_snippets.jsonl as a dict[task_id] -> snippet dict.

    Cached per dir. File format: one JSON per line with keys: task_id, cause_text, step_window.
    """
    if not dirpath:
        return {}
    try:
        p = Path(dirpath) / "attribution_snippets.jsonl"
        if not p.exists():
            return {}
        if _ATTRIB_CACHE["dir"] == str(p):
            return _ATTRIB_CACHE["map"] or {}
        m = {}
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                tid = str(obj.get("task_id", ""))
                if tid:
                    m[tid] = {"cause_text": obj.get("cause_text", ""), "step_window": obj.get("step_window", "")}
        _ATTRIB_CACHE["dir"] = str(p)
        _ATTRIB_CACHE["map"] = m
        return m
    except Exception:
        return {}

def _flatten_cures(v, *, include_tests: bool = True, include_patterns: bool = False) -> List[str]:
    """Accept list[str | {text:str, tests?:[str], patterns?:[str], ...}], return enriched texts.
    
    Args:
        v: list of cure entries
        include_tests: whether to append test cases
        include_patterns: whether to append error patterns (for debugging)
    """
    out: List[str] = []
    try:
        for item in (v or []):
            if isinstance(item, str):
                txt = item.strip()
                if txt:
                    out.append(txt)
                continue
            if isinstance(item, dict):
                base = str(item.get("text") or "").strip()
                if not base:
                    continue
                
                # Append test cases (help LLM understand boundaries)
                if include_tests:
                    tests = item.get("tests") or []
                    if tests:
                        tests_clean = [str(t).strip() for t in tests if str(t).strip()]
                        if tests_clean:
                            base = base + "\nKey boundary test cases (mentally run these):\n- " + "\n- ".join(tests_clean)
                
                # Append error patterns (for debugging)
                if include_patterns:
                    patterns = item.get("patterns") or []
                    if patterns:
                        base = base + f"\n[Matched patterns: {', '.join(patterns[:3])}]"
                
                out.append(base)
    except Exception:
        pass
    return out


def _rank_cures_enhanced(
    fail_type: str,
    context: str,
    code: str,
    cures: List[str],
    cures_raw: List[Dict],
    *,
    k: int = 3
) -> List[str]:
    """Enhanced cure ranking: keyword overlap + pattern matching"""
    if not cures:
        return []
    
    ctx_lower = context.lower()
    code_lower = code.lower()
    
    scores: List[Tuple[float, str]] = []
    
    for i, cure_text in enumerate(cures):
        # Base score: keyword overlap
        toks = set(re.findall(r"[A-Za-z_]{2,}", cure_text.lower()))
        keyword_hits = sum(1 for t in toks if t in ctx_lower or t in code_lower)
        base_score = keyword_hits * 0.1
        
        # If original entry available, use pattern matching
        if i < len(cures_raw) and isinstance(cures_raw[i], dict):
            entry = cures_raw[i]
            pattern_score = _match_error_pattern(code, context, entry)
            base_score += pattern_score
            
            # Distilled source bonus
            if entry.get("source") == "distilled":
                base_score *= 1.3
        
        scores.append((base_score, cure_text))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scores[:max(1, k)]]

def _rank_cures(fail_type: str, context: str, cures: List[str], k: int = 3) -> List[str]:
    """Very small keyword-overlap scorer as a TF-IDF/BM25 surrogate."""
    try:
        ctx = context.lower()
        import re as _re
        scores: List[Tuple[int, str]] = []
        for s in cures:
            toks = set(_re.findall(r"[A-Za-z_]{2,}", s.lower()))
            hit = sum(1 for t in toks if t in ctx)
            scores.append((hit, s))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scores[: max(1, k)]]
    except Exception:
        return cures[:k]






# ============ Repair Agent Integration (T0/T1/T2 Tiered Repair) ============

_CODE_BLOCK_RE_REPAIR = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)

def _extract_code_block(text: str) -> str:
    """Extract Python code block from message"""
    if not text:
        return ""
    blocks = _CODE_BLOCK_RE_REPAIR.findall(text)
    if blocks:
        # Return last longer code block
        candidates = [b for b in blocks if len(b.strip()) >= 16]
        if candidates:
            return candidates[-1].strip()
        return max(blocks, key=lambda s: len(s)).strip() if blocks else ""
    return ""


def repair_if_needed(
    text: str,
    *,
    task_id: str,
    entry_point: str = "",
    task_description: str = "",
    stage: str = "T1",
) -> Tuple[str, Dict[str, Any]] | None:
    """
    Attempt proactive code repair at given stage (T0/T1/T2).
    
    Repair strategy:
    - T0: Fast fix (typing/spelling/import errors) - regex + short LLM
    - T1: Pre-test fix (semantic repair before testing) - medium LLM call
    - T2: Deep fix (after test failure) - strong LLM + Who&When attribution
    
    Args:
        text: full message text containing code block
        task_id: task ID
        entry_point: Function name
        task_description: Task description
        stage: Repair stage (T0/T1/T2)
    
    Returns:
        Tuple[str, Dict]: (repaired message text, repair metadata) or None if not needed/possible
    """
    # Check if repair is enabled
    if os.getenv("REPAIR_ENABLED", "off").strip().lower() not in ("1", "true", "on", "yes"):
        return None
    
    # Check if this stage is enabled
    triggers = os.getenv("REPAIR_TRIGGERS", "T0,T2").strip().upper().split(",")
    if stage.upper() not in triggers:
        return None
    
    # Extract code block
    code = _extract_code_block(text)
    if not code:
        return None
    
    # Collect online signals
    safe_task = _safe_name(task_id)
    summary = os.getenv(f"TESTER_FAIL_SUMMARY__{safe_task}", "") or os.getenv(f"TESTER_RAW_OUTPUT__{safe_task}", "")
    if not summary:
        summary = _latest_tester_summary(task_id)
    
    # ======== Ablation: Auditor diagnosis ========
    auditor_enabled = _is_auditor_enabled()
    
    if auditor_enabled:
        aud_score = _latest_auditor_score(task_id, 0.0)
        hints = _summary_fail_type_hints(summary)
        fail_type = hints[0] if hints else "default"
    else:
        # w/o Auditor ablation: disable diagnosis
        aud_score = 1.0
        fail_type = "default"
        print(f"[RepairBridge] Auditor disabled (ablation), task={task_id}")
    
    # Detect runtime errors
    runtime_error = _tester_has_runtime_error(task_id, summary)
    
    try:
        from autoagents_ext.repair_agent import (
            repair_code,
            record_repair,
            fast_typing_fix,
            detect_obvious_errors,
            is_repair_enabled,
        )
    except ImportError as e:
        print(f"[RepairBridge] Cannot import repair_agent: {e}")
        return None
    
    # ======== T0: Fast fix (typing/spelling/import) ========
    if stage == "T0":
        has_errors, error_types = detect_obvious_errors(code)
        if not has_errors:
            return None
        
        print(f"[RepairBridge] T0 detected errors: {error_types}, task={task_id}")
        
        result = repair_code(
            code,
            entry_point=entry_point,
            error_type=", ".join(error_types) if error_types else fail_type,
            error_message=summary[:500] if summary else "",
            stage="T0",
        )
        
        if result and result.get("fixed_code") and result["fixed_code"] != code:
            fixed_text = text.replace(code, result["fixed_code"])
            meta = {
                "repair_stage": "T0",
                "repair_method": result.get("method", "regex"),
                "repair_confidence": result.get("confidence", 0.0),
                "fixes_applied": result.get("fixes_applied", []),
                "fail_type": fail_type,
                "error_types": error_types,
                "tokens_used": result.get("tokens_used", 0),
            }
            record_repair(task_id, meta)
            current_round = _current_round_index(1)
            
            # Online immunity learning: record T0 regex fix
            if _HAS_ONLINE_LEARNING and result.get("method") == "regex":
                try:
                    fix_desc = f"T0 regex fix: {', '.join(error_types)}"
                    added = add_successful_repair_to_library(
                        task_id=task_id,
                        fail_type=error_types[0] if error_types else "unknown",
                        fix_description=fix_desc,
                        code_before=code,
                        code_after=result["fixed_code"],
                        error_message=", ".join(error_types),
                        test_pass_rate=0.95,
                    )
                    if added:
                        print(f"[RepairBridge] Online immunity: T0 evidence added")
                except Exception as e:
                    print(f"[RepairBridge] Online immunity record failed: {e}")
            
            immunity_payload = {
                "immunity_injected": True,
                "immunity_round": current_round,
                "immunity_fail_type": fail_type,
                "immunity_evidence": [],
                "repair_applied": True,
                "repair_stage": "T0",
                "repair_method": result.get("method", "regex"),
                "tokens_used": result.get("tokens_used", 0),
                "ts_ms": int(time.time() * 1000),
            }
            _IMMUNITY_FLAGS[task_id] = immunity_payload
            _record_immunity_snapshot(task_id, immunity_payload)
            
            print(f"[RepairBridge] T0 repair succeeded: {len(result.get('fixes_applied', []))} fixes applied, round={current_round}")
            return fixed_text, meta
        
        return None
    
    # ======== T1: Pre-test fix (semantic repair) ========
    if stage == "T1":
        should_repair = aud_score > 0.7 or runtime_error
        if not should_repair:
            has_errors, _ = detect_obvious_errors(code)
            should_repair = has_errors
        
        if not should_repair:
            return None
        
        print(f"[RepairBridge] T1 triggered: score={aud_score:.2f}, runtime_error={runtime_error}, task={task_id}")
        
        evidence = _select_distilled_cures(fail_type, code, summary, topk=3)
        
        result = repair_code(
            code,
            entry_point=entry_point,
            error_type=fail_type,
            error_message=summary[:800] if summary else "",
            distilled_evidence=evidence,
            stage="T1",
        )
        
        if result and result.get("fixed_code") and result["fixed_code"] != code:
            fixed_text = text.replace(code, result["fixed_code"])
            meta = {
                "repair_stage": "T1",
                "repair_method": result.get("method", "llm"),
                "repair_confidence": result.get("confidence", 0.0),
                "fixes_applied": result.get("fixes_applied", []),
                "fail_type": fail_type,
                "auditor_score": aud_score,
                "evidence_used": evidence,
            }
            record_repair(task_id, meta)
            current_round = _current_round_index(1)
            
            immunity_payload = {
                "immunity_injected": True,
                "immunity_round": current_round,
                "immunity_fail_type": fail_type,
                "immunity_evidence": evidence,
                "repair_applied": True,
                "repair_stage": "T1",
                "repair_method": result.get("method", "llm"),
                "ts_ms": int(time.time() * 1000),
            }
            _IMMUNITY_FLAGS[task_id] = immunity_payload
            _record_immunity_snapshot(task_id, immunity_payload)
            
            print(f"[RepairBridge] T1 repair succeeded, round={current_round}")
            return fixed_text, meta
        
        return None
    
    # ======== T2: Deep fix (after test failure) ========
    if stage == "T2":
        summary_lower = (summary or "").lower()
        has_failure = runtime_error or (
            summary and (
                "FAIL" in summary or
                "fail" in summary_lower or
                "error" in summary_lower or
                "exception" in summary_lower or
                "traceback" in summary_lower
            )
        )
        if not has_failure:
            return None
        
        print(f"[RepairBridge] T2 triggered: summary={summary[:60] if summary else 'None'}, task={task_id}")
        
        evidence = _select_distilled_cures(fail_type, code, summary, topk=3)
        
        # Load Who&When attribution
        attrib_dir = os.getenv("ATTRIBUTION_DIR", "").strip() or os.getenv("RUN_GROUP_DIR", "").strip()
        who_when = {}
        if attrib_dir:
            attrib_map = _load_attribution_map(attrib_dir)
            who_when = attrib_map.get(_safe_name(task_id), {}) if attrib_map else {}
        
        # Extract failed test cases
        test_cases = []
        if summary:
            match = re.search(r"input=\((.*?)\)\s*expected=(.*?)\s+got=(.*?)(?:\s|$)", summary)
            if match:
                test_cases.append(f"input: {match.group(1)}, expected: {match.group(2)}, got: {match.group(3)}")
        
        result = repair_code(
            code,
            entry_point=entry_point,
            error_type=fail_type,
            error_message=summary[:1000] if summary else "",
            task_description=task_description,
            test_cases=test_cases,
            distilled_evidence=evidence,
            who_when=who_when,
            stage="T2",
        )
        
        if result and result.get("fixed_code") and result["fixed_code"] != code:
            fixed_text = text.replace(code, result["fixed_code"])
            meta = {
                "repair_stage": "T2",
                "repair_method": result.get("method", "llm"),
                "repair_confidence": result.get("confidence", 0.0),
                "fixes_applied": result.get("fixes_applied", []),
                "fail_type": fail_type,
                "evidence_used": evidence,
                "who_when_used": bool(who_when),
                "test_cases_used": test_cases,
                "tokens_used": result.get("tokens_used", 0),
            }
            record_repair(task_id, meta)
            current_round = _current_round_index(1)
            
            # Online immunity learning
            if _HAS_ONLINE_LEARNING:
                try:
                    fix_desc = f"Fix {fail_type} error: {summary[:200]}" if summary else f"Fix {fail_type} type error"
                    added = add_successful_repair_to_library(
                        task_id=task_id,
                        fail_type=fail_type,
                        fix_description=fix_desc,
                        code_before=code,
                        code_after=result["fixed_code"],
                        error_message=summary or "",
                        test_pass_rate=result.get("confidence", 0.5),
                    )
                    if added:
                        print(f"[RepairBridge] Online immunity: new evidence added ({fail_type})")
                except Exception as e:
                    print(f"[RepairBridge] Online immunity record failed: {e}")
            
            immunity_payload = {
                "immunity_injected": True,
                "immunity_round": current_round,
                "immunity_fail_type": fail_type,
                "immunity_evidence": evidence,
                "repair_applied": True,
                "repair_stage": "T2",
                "repair_method": result.get("method", "llm"),
                "tokens_used": result.get("tokens_used", 0),
                "ts_ms": int(time.time() * 1000),
            }
            _IMMUNITY_FLAGS[task_id] = immunity_payload
            _record_immunity_snapshot(task_id, immunity_payload)
            
            print(f"[RepairBridge] T2 repair succeeded, round={current_round}")
            return fixed_text, meta
        
        return None
    
    return None


def consume_repair_metadata(task_id: str) -> Dict[str, Any] | None:
    """Consume and return the latest repair metadata."""
    try:
        from autoagents_ext.repair_agent import get_repair_history
        history = get_repair_history(task_id)
        return history[-1] if history else None
    except Exception:
        return None
