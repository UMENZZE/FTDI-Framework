# autoagents_ext/hook.py
# -*- coding: utf-8 -*-
"""
Environment hook with toggles for AutoInject and Auditor.
(See inline docs for usage and CLI wiring.)

IMPORTANT: Instance-level monkey patch ensuring concurrent task isolation。
Each Environment instance has its own publish_message method with config in instance attributes。
"""
from __future__ import annotations

import os
import re
import json
import time
import types
from typing import Any, Dict, Optional
from pathlib import Path

# --------- helpers ---------
DEFAULT_WORKSPACE = os.environ.get("WORKSPACE_ROOT", "./workspace")


# Track patched instances to avoid double-patching
_PATCHED_INSTANCES: set = set()

# Try importing class-based Auditor (fall back to built-in _auditor_score at runtime)
try:
    from autoagents_ext.auditor import Auditor, AuditorCfg  # Prefer class-based auditor
    _aud_ins = Auditor(AuditorCfg())
    _HAS_AUD = True
except Exception:
    _aud_ins = None
    _HAS_AUD = False

def _ensure_traces_path(task_id: str) -> Path:
    base = Path(DEFAULT_WORKSPACE) / "traces"
    # Optional run grouping folder, e.g., mas-env-PM0.2-PE0.2-semantic-251025-23:46[:SS]
    group = os.environ.get("RUN_GROUP_DIR", "").strip()
    p = base / group if group else base
    p.mkdir(parents=True, exist_ok=True)
    # Sanitize task_id to avoid nested dirs like "HumanEval/1"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id))
    return p / f"{safe}.jsonl"

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)

def _extract_last_python_block(text: str) -> str:
    if not isinstance(text, str):
        return ""
    blocks = _CODE_BLOCK_RE.findall(text or "") or []
    if not blocks:
        return ""
    # Find blocks that look like code: length>=16 or contain def/return etc.
    import re as _re
    candidates = [b for b in blocks if len(b.strip()) >= 16 or _re.search(r"\b(def|return|for|while)\b", b)]
    if candidates:
        return candidates[-1].strip()
    # Otherwise fall back to longest block
    return max(blocks, key=lambda s: len(s)).strip()

HEDGE = r"\b(maybe|probably|perhaps|i think|not sure|guess)\b"

def _auditor_score(msg, env, problem, *, role_coder="Coder", role_planner="Planner") -> Dict[str, float]:
    try:
        role = getattr(msg, "role", "")
        text = (getattr(msg, "content", "") or "").strip()
        entry_point = str(problem.get("entry_point") or "").strip()

        # Deviation
        dev = 0.0
        if role == role_coder:
            code = _extract_last_python_block(text) or ""
            has_block = (code != "")
            dev += 0.5 if not has_block else 0.0
            if has_block and len(text.replace(code, "")) > 40:
                dev += 0.25
            if entry_point:
                ok_sig = bool(re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, flags=re.M))
                dev += 0.25 if not ok_sig else 0.0
        dev = min(1.0, dev)

        # Uncertainty
        unc = 0.0
        if re.search(HEDGE, text, flags=re.I):
            unc += 0.5
        if role == role_coder:
            n = len(text)
            if n < 40 or n > 6000:
                unc += 0.5
        unc = min(1.0, unc)

        # Consistency
        con = 0.0
        if role == role_coder:
            plan = ""
            try:
                plan = env._latest_role_message_text(role_planner) or ""
            except Exception:
                plan = ""
            if plan and ("test" in plan.lower()) and ("```" not in text):
                con += 0.5
            if "do not change the function signature" in plan.lower():
                if not re.search(r"```python", text, flags=re.I):
                    con += 0.5
        con = min(1.0, con)

        score = 0.5*dev + 0.25*unc + 0.25*con
        return {"dev": dev, "unc": unc, "con": con, "score": float(score)}
    except Exception:
        return {"dev": 0.0, "unc": 0.0, "con": 0.0, "score": 0.0}

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")

def _infer_inject_enabled(user_setting: Optional[bool]) -> bool:
    if user_setting is not None:
        return bool(user_setting)
    
    # Check AUTOINJECT_ENABLED first: if explicitly off, disable injection
    enabled_str = os.environ.get("AUTOINJECT_ENABLED", "").strip().lower()
    if enabled_str in ("0", "false", "no", "off"):
        return False  # explicitly disabled
    if enabled_str in ("1", "true", "yes", "on"):
        return True   # explicitly enabled
    
    # If AUTOINJECT_ENABLED not set, infer from PM
    try:
        return float(os.environ.get("AUTOINJECT_PM", "0")) > 0.0
    except Exception:
        return False


# ============ Instance-level patched publish method (module-level def) ============
async def _instance_patched_publish(self, msg):
    """
    Instance-level publish_message method.
    All config is read from self attributes, ensuring isolation between concurrent instances.
    """
    # Read config from instance attributes (avoid closure pollution)
    _task_id = getattr(self, '_hook_task_id', 'unknown')
    _problem = getattr(self, '_hook_problem', {})
    _auditor_enabled = getattr(self, '_hook_auditor_enabled', False)
    _inject_enabled = getattr(self, '_hook_inject_enabled', False)
    _trace_off = getattr(self, '_hook_trace_off', True)
    _trace_fp = getattr(self, '_hook_trace_fp', None)
    _has_autoinject = getattr(self, '_hook_has_autoinject', False)
    _orig_publish = getattr(self, '_hook_orig_publish', None)
    
    role_name = getattr(msg, "role", "")
    
    # ======== Defense: Challenger/Inspector (Coder only) ========
    defense_meta = None
    try:
        from autoagents_ext.defense import apply_defense, record_to_defense, get_defense_mode
        defense_mode = get_defense_mode()
        
        if defense_mode != "off" and role_name == "Coder":
            original_content = str(getattr(msg, "content", "") or "")
            defended_content, defense_meta = apply_defense(
                original_content, 
                role=role_name, 
                task_id=_task_id
            )
            
            if defended_content == "[UNSAFE_REGENERATE]":
                # Challenger detected unsafe, need regeneration
                # Simple handling: keep original (Inspector mode will rewrite)
                print(f"[Defense] Challenger detected unsafe content for task={_task_id}, proceeding with rewritten version")
                msg.content = original_content  # keep original, Inspector mode will rewrite
            else:
                msg.content = defended_content
            
            # Store defense metadata
            meta = getattr(msg, "meta", {}) or {}
            meta["defense"] = defense_meta
            msg.meta = meta
        
        # Record message to defense system history
        if defense_mode != "off":
            record_to_defense(role_name, str(getattr(msg, "content", "") or "")[:1000])
    except ImportError:
        pass  # ignore when defense.py doesn't exist
    except Exception as e:
        import traceback
        print(f"[Defense] Error: {e}")
        traceback.print_exc()

    # 1) Auditor
    diagnosis = {}
    if _auditor_enabled:
        try:
            if _HAS_AUD and _aud_ins:
                diagnosis = _aud_ins.score_message(msg, self, _problem)
            else:
                diagnosis = _auditor_score(msg, self, _problem)

            meta = getattr(msg, "meta", {}) or {}
            meta["auditor"] = diagnosis
            msg.meta = meta

            try:
                safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(_task_id))
                _role_name = getattr(msg, "role", "")
                if _role_name == "Tester":
                    os.environ[f"AUDITOR_TESTER_DIAGNOSIS__{safe_task}"] = json.dumps(diagnosis)
                    os.environ[f"AUDITOR_DIAGNOSIS__{safe_task}"] = json.dumps(diagnosis)
                    fail_summary = (getattr(self, "fail_summary", "") or "")[:2000]
                    tester_raw = (getattr(msg, "content", "") or "")[:2000]
                    os.environ[f"TESTER_FAIL_SUMMARY__{safe_task}"] = fail_summary
                    os.environ[f"TESTER_RAW_OUTPUT__{safe_task}"] = tester_raw
                elif _role_name == "Coder":
                    os.environ[f"AUDITOR_CODER_DIAGNOSIS__{safe_task}"] = json.dumps(diagnosis)
            except Exception:
                pass

            try:
                os.environ["AUDITOR_SCORE"] = str(float(diagnosis.get("score", 0.0)))
            except Exception:
                pass
        except Exception:
            diagnosis = {}

    # 2) AutoInject + T0 Repair（Coder only）
    repair_meta = None
    try:
        _enabled = _inject_enabled if _inject_enabled is not None else _infer_inject_enabled(None)
        if _enabled and _has_autoinject and getattr(msg, "role", "") == "Coder":
            from autoagents_ext.inject_bridge import inject_if_needed as _inject_if_needed
            try:
                os.environ["AUTOINJECT_ROUND"] = str(int(getattr(self, "current_round", 0) or 0))
            except Exception:
                pass
            msg.content = _inject_if_needed(str(getattr(msg, "content", "")), is_code=True, task_id=_task_id)
            # ======== T0: Fast fix before Coder message publish ========
            try:
                from autoagents_ext.inject_bridge import repair_if_needed as _repair_if_needed
                entry_point = str(_problem.get("entry_point", "") or "")
                result = _repair_if_needed(
                    str(getattr(msg, "content", "") or ""),
                    task_id=_task_id,
                    entry_point=entry_point,
                    task_description=str(_problem.get("prompt", "") or "")[:500],
                    stage="T0",
                )
                if result:
                    fixed_text, repair_meta = result
                    msg.content = fixed_text
                    meta = getattr(msg, "meta", {}) or {}
                    meta.setdefault("repair", {}).update(repair_meta)
                    msg.meta = meta
                    print(f"[Hook] ✅ T0 repair applied: task={_task_id}")
            except Exception as e:
                # T0 repair failure does not affect main flow
                pass
    except Exception:
        pass

    # 2.3) Apply previous round's T2 deep repair code to current Coder output
    # Key fix: T2 generated code was only stored but never consumed; replace Coder code block here
    try:
        if getattr(msg, "role", "") == "Coder":
            safe_task_t2 = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(_task_id))
            t2_repaired = os.environ.pop(f"T2_REPAIRED_CODE__{safe_task_t2}", "")
            if t2_repaired:
                # Extract pure code block from T2 repair result
                t2_block = _extract_last_python_block(t2_repaired)
                if t2_block:
                    current_content = str(getattr(msg, "content", "") or "")
                    current_block = _extract_last_python_block(current_content)
                    if current_block:
                        # Replace Coder output code block with T2 repair code (override injected mutation)
                        msg.content = current_content.replace(current_block, t2_block)
                        print(f"[Hook] ✅ T2 repair code applied to Coder output: task={_task_id}")
                    else:
                        # No code block in current output, append T2 code
                        msg.content = current_content + f"\n\n```python\n{t2_block}\n```"
                        print(f"[Hook] ✅ T2 repair code appended to Coder output: task={_task_id}")
    except Exception as e:
        print(f"[Hook] ⚠️ T2 code application failed: {e}")

    # 2.5) T2: Deep repair after Tester failure
    try:
        if getattr(msg, "role", "") == "Tester":
            fail_summary = getattr(self, "fail_summary", "") or ""
            pass_flag = getattr(self, "pass_flag", False)
            tester_content = str(getattr(msg, "content", "") or "")
            
            # Detect failure (has runtime error or assertion failure)
            has_failure = (
                not pass_flag and 
                (fail_summary or "FAIL" in tester_content or "Error" in tester_content)
            )
            
            if has_failure and _has_autoinject:
                from autoagents_ext.inject_bridge import repair_if_needed as _repair_if_needed
                from autoagents_ext.inject_bridge import consume_repair_metadata as _consume_repair
                
                # Set current round info
                try:
                    os.environ["AUTOINJECT_ROUND"] = str(int(getattr(self, "current_round", 0) or 0))
                except Exception:
                    pass
                
                # Get previous round Coder code (from env var or cache)
                safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(_task_id))
                last_coder_code = os.environ.get(f"LAST_CODER_CODE__{safe_task}", "")
                
                if last_coder_code:
                    entry_point = str(_problem.get("entry_point", "") or "")
                    
                    result = _repair_if_needed(
                        last_coder_code,
                        task_id=_task_id,
                        entry_point=entry_point,
                        task_description=str(_problem.get("prompt", "") or "")[:500],
                        stage="T2",
                    )
                    
                    if result:
                        fixed_text, repair_meta = result
                        # Store T2 repaired code in env var for next round Planner
                        os.environ[f"T2_REPAIRED_CODE__{safe_task}"] = fixed_text
                        
            # Save current Coder code for next round T2
            # (saved at Tester stage since next might be Coder)
    except Exception as e:
        pass
    
    # 2.6) Save Coder code for T2 use
    try:
        if getattr(msg, "role", "") == "Coder":
            safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(_task_id))
            os.environ[f"LAST_CODER_CODE__{safe_task}"] = str(getattr(msg, "content", "") or "")[:10000]
    except Exception:
        pass

    # 3) JSONL trace
    if not _trace_off and _trace_fp:
        try:
            rec = {
                "ts": int(time.time()*1000),
                "task_id": _task_id,
                "round": int(getattr(self, "current_round", 0) or 0),
                "role": getattr(msg, "role", ""),
                "raw_text": (getattr(msg, "content", "") or ""),
                "extracted_code": _extract_last_python_block(getattr(msg, "content", "") or ""),
                "tester_summary": (getattr(self, "fail_summary", "") or "")[:2000],
                "pass_flag": bool(getattr(self, "pass_flag", False)),
                "auditor": diagnosis if _auditor_enabled else None,
                "inject_enabled": bool(_inject_enabled if _inject_enabled is not None else _infer_inject_enabled(None)),
                "has_autoinject": bool(_has_autoinject),
            }
            if repair_meta:
                rec["repair"] = dict(repair_meta)
            if defense_meta:
                rec["defense"] = dict(defense_meta)
            with _trace_fp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # Call original method
    if _orig_publish is not None:
        return await _orig_publish(self, msg)
    else:
        # Fallback: get original method from class
        EnvClass = self.__class__
        if hasattr(EnvClass, '_original_publish_message_saved'):
            return await EnvClass._original_publish_message_saved(self, msg)
        return await EnvClass.publish_message(self, msg)


def instrument_env(
    env,
    *,
    task_id: str,
    problem: Dict[str, Any],
    auditor_enabled: bool = True,
    inject_enabled: Optional[bool] = None,
    workspace_root: Optional[str] = None,
):
    """
    Monkey patch env.publish_message to add AutoInject (optional), Auditor (optional), and JSONL tracing.
    
    IMPORTANT: Instance-level monkey patch ensuring concurrent task isolation。
    Each Environment instance has its own publish_message method with config in instance attributes。
    
    - auditor_enabled: wire to CLI --auditor on/off (or AUDITOR env var if not set via CLI)
    - inject_enabled:  if None, derive from env vars (AUTOINJECT_PM/AUTOINJECT_ENABLED)
    - workspace_root:  default ./workspace or WORKSPACE_ROOT env
    """
    # Check if already patched, avoid double-patching
    env_id = id(env)
    if env_id in _PATCHED_INSTANCES:
        return env
    _PATCHED_INSTANCES.add(env_id)
    
    if workspace_root:
        global DEFAULT_WORKSPACE
        DEFAULT_WORKSPACE = workspace_root

    if auditor_enabled is True:
        auditor_enabled = _env_bool("AUDITOR", True)

    inject_enabled = _infer_inject_enabled(inject_enabled)
    trace_off = _env_bool("TRACE_OFF", False)
    
    # Debug output
    print(f"[Hook] task_id={task_id}, inject_enabled={inject_enabled}, AUTOINJECT_ENABLED={os.environ.get('AUTOINJECT_ENABLED')}, AUTOINJECT_PM={os.environ.get('AUTOINJECT_PM')}")

    trace_fp = _ensure_traces_path(task_id)
    
    EnvClass = env.__class__
    
    # ============ Save original publish_message (once to class attr) ============
    if not hasattr(EnvClass, '_original_publish_message_saved'):
        EnvClass._original_publish_message_saved = EnvClass.publish_message
    orig_publish = EnvClass._original_publish_message_saved

    # ============ Save config to instance attributes (avoid closure pollution) ============
    # Use object.__setattr__ to bypass pydantic attribute restrictions
    try:
        object.__setattr__(env, '_hook_task_id', task_id)
        object.__setattr__(env, '_hook_problem', problem)
        object.__setattr__(env, '_hook_auditor_enabled', auditor_enabled)
        object.__setattr__(env, '_hook_inject_enabled', inject_enabled)
        object.__setattr__(env, '_hook_trace_off', trace_off)
        object.__setattr__(env, '_hook_trace_fp', trace_fp)
        object.__setattr__(env, '_hook_orig_publish', orig_publish)
    except Exception:
        # Fallback to __dict__ direct assignment
        env.__dict__['_hook_task_id'] = task_id
        env.__dict__['_hook_problem'] = problem
        env.__dict__['_hook_auditor_enabled'] = auditor_enabled
        env.__dict__['_hook_inject_enabled'] = inject_enabled
        env.__dict__['_hook_trace_off'] = trace_off
        env.__dict__['_hook_trace_fp'] = trace_fp
        env.__dict__['_hook_orig_publish'] = orig_publish

    # Detect if injection bridge is available
    try:
        _has_autoinject = True
    except Exception:
        _has_autoinject = False
    
    try:
        object.__setattr__(env, '_hook_has_autoinject', _has_autoinject)
    except Exception:
        env.__dict__['_hook_has_autoinject'] = _has_autoinject

    # ============ Create instance-level patched method ============
    bound_method = types.MethodType(_instance_patched_publish, env)
    
    try:
        object.__setattr__(env, 'publish_message', bound_method)
    except Exception:
        try:
            env.__dict__['publish_message'] = bound_method
        except Exception:
            # Last resort: class-level patch (not recommended, but ensures functionality)
            setattr(EnvClass, "publish_message", _instance_patched_publish)
    
    return env
