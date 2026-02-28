# ftdi/repair_agent.py
# -*- coding: utf-8 -*-
"""
Repair Agent - Code repair module (FTDI framework)

Three-tier repair strategy:
- T0: Fast fix (typing/spelling/import errors) - regex + short LLM micro-edit
- T1: Pre-test fix (semantic repair before testing) - medium LLM call
- T2: Deep fix (full repair after test failure) - strong LLM + Who&When attribution

Features (v2):
- Error type -> repair strategy mapping (REPAIR_STRATEGY_MAP)
- Smart skip for logic errors (save token budget)
- Token cost tracking

Environment variables:
- REPAIR_ENABLED: Enable repair agent (on/off)
- REPAIR_TRIGGERS: Enabled repair stages (T0,T1,T2)
- REPAIR_MODEL: Repair model name (default deepseek-v3)
- REPAIR_STRONG_MODEL: Strong model for complex problems (default qwen3-235b-a22b)
- REPAIR_BASE_URL: API base URL
- REPAIR_API_KEY: API key
- REPAIR_MAX_TOKENS: Max generation tokens (default 1024)
- REPAIR_TIMEOUT: Request timeout seconds (default 30)
- REPAIR_TEMPERATURE: Generation temperature (default 0.2)
- REPAIR_ALLOW_LOGIC_ERROR: Whether to attempt logic error repair (on/off, default on)
- REPAIR_MIN_SUCCESS_RATE: Min expected success rate threshold (default 0.15)
"""

from __future__ import annotations

import os
import re
import json
import time
import httpx
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import repair strategy module
try:
    from .repair_strategy import (
        detect_error_type,
        get_repair_strategy,
        should_skip_repair,
        recommend_strategy,
        record_token_usage,
        REPAIR_STRATEGY_MAP,
    )
    _HAS_STRATEGY = True
    print("[RepairAgent] ✅ repair_strategy module loaded")
except ImportError as e:
    _HAS_STRATEGY = False
    print(f"[RepairAgent] ⚠️ repair_strategy module not loaded: {e}")
    
    # Fallback stubs
    def should_skip_repair(*args, **kwargs):
        return False, ""
    def get_repair_strategy(*args, **kwargs):
        return {}
    def record_token_usage(*args, **kwargs):
        pass

# ============ Configuration ============

def _clean_env_value(value: str) -> str:
    """Clean env value: strip trailing comments and quotes"""
    if not value:
        return value
    # Strip surrounding quotes
    value = value.strip().strip('"').strip("'")
    # Strip trailing comments after #
    if '#' in value:
        # Truncate only when # is followed by non-ASCII or space
        parts = value.split('#')
        if len(parts) > 1:
            # Check if first part is a valid ASCII API key
            first_part = parts[0].strip()
            if first_part and all(ord(c) < 128 for c in first_part):
                value = first_part
    return value

def _get_repair_config() -> Dict[str, Any]:
    """Get repair agent configuration"""
    api_key = os.getenv("REPAIR_API_KEY", "") or os.getenv("PROBEX_CHAT_API_KEY", "")
    api_key = _clean_env_value(api_key)
    
    # T2-specific config (optional, falls back to defaults)
    t2_api_key = os.getenv("REPAIR_T2_API_KEY", "") or api_key
    t2_api_key = _clean_env_value(t2_api_key)
    t2_base_url = os.getenv("REPAIR_T2_BASE_URL", "") or os.getenv("REPAIR_BASE_URL", "") or "https://api.probex.top/v1"
    t2_model = os.getenv("REPAIR_T2_MODEL", "") or os.getenv("REPAIR_MODEL", "") or "deepseek-v3"
    
    return {
        "enabled": os.getenv("REPAIR_ENABLED", "off").strip().lower() in ("1", "true", "on", "yes"),
        "triggers": os.getenv("REPAIR_TRIGGERS", "T0,T2").strip().upper().split(","),
        "model": os.getenv("REPAIR_MODEL", "") or os.getenv("PROBEX_CHAT_MODEL", "deepseek-v3"),
        "strong_model": os.getenv("REPAIR_STRONG_MODEL", "qwen3-235b-a22b"),  # for complex problems
        "base_url": os.getenv("REPAIR_BASE_URL", "") or os.getenv("PROBEX_CHAT_BASE", "https://api.probex.top/v1"),
        "api_key": api_key,
        # T2-specific config
        "t2_model": t2_model,
        "t2_base_url": t2_base_url,
        "t2_api_key": t2_api_key,
        "max_tokens": int(os.getenv("REPAIR_MAX_TOKENS", "1024")),
        "timeout": float(os.getenv("REPAIR_TIMEOUT", "30")),
        "temperature": float(os.getenv("REPAIR_TEMPERATURE", "0.2")),
        "allow_logic_error": os.getenv("REPAIR_ALLOW_LOGIC_ERROR", "on").lower() in ("1", "true", "on", "yes"),
        "min_success_rate": float(os.getenv("REPAIR_MIN_SUCCESS_RATE", "0.15")),
    }


def is_repair_enabled() -> bool:
    """Check if repair agent is enabled"""
    return _get_repair_config()["enabled"]


def is_trigger_enabled(stage: str) -> bool:
    """Check if given stage is enabled"""
    cfg = _get_repair_config()
    return cfg["enabled"] and stage.upper() in cfg["triggers"]


# ============ Fast Fix Rules (T0 level, regex only) ============

# Common typing module error patterns
_TYPING_FIX_PATTERNS = [
    # lst/lIST/lIst/lsit -> List
    (r'\bfrom\s+typing\s+import\s+lst\b', 'from typing import List'),
    (r'\bfrom\s+typing\s+import\s+lIST\b', 'from typing import List'),
    (r'\bfrom\s+typing\s+import\s+lIst\b', 'from typing import List'),
    (r'\bfrom\s+typing\s+import\s+lsit\b', 'from typing import List'),
    (r'\bfrom\s+typing\s+import\s+Lst\b', 'from typing import List'),
    # Built-in types should not be imported from typing
    (r'\bfrom\s+typing\s+import\s+int\b', ''),  # remove
    (r'\bfrom\s+typing\s+import\s+str\b', ''),
    (r'\bfrom\s+typing\s+import\s+float\b', ''),
    (r'\bfrom\s+typing\s+import\s+Float\b', ''),  # Float does not exist in typing
    (r'\bfrom\s+typing\s+import\s+bool\b', ''),
    (r'\bfrom\s+typing\s+import\s+String\b', ''),  # String does not exist
    (r'\bfrom\s+typing\s+import\s+Integer\b', ''),  # Integer does not exist
    (r'\bfrom\s+typing\s+import\s+typing\b', ''),  # recursive import error
    # typing spelling errors in import statements
    (r'\bfrom\s+tYping\s+import\b', 'from typing import'),
    (r'\bfrom\s+tyPing\s+import\b', 'from typing import'),
    (r'\bfrom\s+typign\s+import\b', 'from typing import'),
    (r'\bfrom\s+typng\s+import\b', 'from typing import'),
    (r'\bimport\s+tYping\b', 'import typing'),
    (r'\bimport\s+tyPing\b', 'import typing'),
    # import statement syntax errors
    (r'\bfrom\s+typing\s+imoprt\b', 'from typing import'),
    (r'\bfrom\s+typing\s+improt\b', 'from typing import'),
    # common module name typos
    (r'\bimport\s+mth\b', 'import math'),
    (r'\bimport\s+maht\b', 'import math'),
    (r'\bimport\s+mahth\b', 'import math'),
    (r'\bimport\s+funcools\b', 'import functools'),
    (r'\bimport\s+funtools\b', 'import functools'),
    (r'\bimport\s+re\s+as\s+regex\b', 'import re'),
    (r'\bfrom\s+colections\s+import\b', 'from collections import'),
    (r'\bfrom\s+collecitons\s+import\b', 'from collections import'),
    # invalid import combinations
    (r'\bfrom\s+math\s+import\s+isclose\b', 'from math import isclose'),
]

# Variable name typo patterns
_TYPO_FIX_PATTERNS = [
    # common variable name typos
    (r'\bresutl\b', 'result'),
    (r'\bresullt\b', 'result'),
    (r'\bresutt\b', 'result'),
    (r'\bnumbres\b', 'numbers'),
    (r'\bnumers\b', 'numbers'),
    (r'\bnubmers\b', 'numbers'),
    (r'\btotaL\b', 'total'),
    (r'\btotla\b', 'total'),
    (r'\btoatl\b', 'total'),
    (r'\bwordss\b', 'words'),
    (r'\bwrods\b', 'words'),
    (r'\bworrds\b', 'words'),
    (r'\bdirectionss\b', 'directions'),
    (r'\bcharss\b', 'chars'),
    (r'\bstringss\b', 'strings'),
    (r'\blenght\b', 'length'),
    (r'\blegnth\b', 'length'),
    (r'\bindxe\b', 'index'),
    (r'\bcoutner\b', 'counter'),
    (r'\bcouter\b', 'counter'),
    # boolean spelling errors
    (r'\bTruE\b', 'True'),
    (r'\bFalsE\b', 'False'),
    (r'\btruE\b', 'True'),
    (r'\bfalsE\b', 'False'),
    (r'\bTRUE\b', 'True'),
    (r'\bFALSE\b', 'False'),
    # keyword spelling errors
    (r'\belses\b:', 'else:'),
    (r'\belifs\b', 'elif'),
    (r'\bretrun\b', 'return'),
    (r'\bretunr\b', 'return'),
    (r'\bretrn\b', 'return'),
    (r'\bdefint\b', 'def'),
    # common for-loop variable name errors (e.g. for word in word:)
    (r'\bfor\s+(\w+)\s+in\s+\1:', 'for item in \1:'),
]

# Pre-compiled regex
_COMPILED_TYPING_FIXES = [(re.compile(p), r) for p, r in _TYPING_FIX_PATTERNS]
_COMPILED_TYPO_FIXES = [(re.compile(p), r) for p, r in _TYPO_FIX_PATTERNS]


def fast_typing_fix(code: str) -> Tuple[str, List[Dict]]:
    """
    Fast fix typing/spelling errors (T0 level, regex only)
    
    Returns:
        Tuple[str, List[Dict]]: (fixed code, list of applied fixes)
    """
    if not code:
        return code, []
    
    fixes_applied = []
    fixed = code
    
    # Apply typing fixes
    for pattern, replacement in _COMPILED_TYPING_FIXES:
        if pattern.search(fixed):
            match = pattern.search(fixed)
            old_text = match.group(0) if match else ""
            fixed = pattern.sub(replacement, fixed)
            if old_text:
                fixes_applied.append({
                    "type": "typing_fix",
                    "old": old_text,
                    "new": replacement,
                    "pattern": pattern.pattern
                })
    
    # Apply typo fixes
    for pattern, replacement in _COMPILED_TYPO_FIXES:
        if pattern.search(fixed):
            match = pattern.search(fixed)
            old_text = match.group(0) if match else ""
            fixed = pattern.sub(replacement, fixed)
            if old_text:
                fixes_applied.append({
                    "type": "typo_fix",
                    "old": old_text,
                    "new": replacement,
                    "pattern": pattern.pattern
                })
    
    # Clean up extra blank lines left by removed imports
    fixed = re.sub(r'\n\s*\n\s*\n', '\n\n', fixed)
    
    return fixed, fixes_applied


def detect_obvious_errors(code: str) -> Tuple[bool, List[str]]:
    """
    Detect obvious repairable errors in code
    
    Detection methods:
    1. Regex pattern matching (typing/typo)
    2. AST analysis for undefined variables
    3. Syntax error detection
    
    Returns:
        Tuple[bool, List[str]]: (error detected, error type list)
    """
    if not code:
        return False, []
    
    detected = []
    
    # 1. Check typing errors (regex)
    for pattern, _ in _COMPILED_TYPING_FIXES:
        if pattern.search(code):
            detected.append("typing_error")
            break
    
    # 2. Check typo errors (regex)
    for pattern, _ in _COMPILED_TYPO_FIXES:
        if pattern.search(code):
            detected.append("typo_error")
            break
    
    # 3. Check syntax errors
    try:
        import ast
        ast.parse(code)
    except SyntaxError as e:
        detected.append("syntax_error")
    
    # 4. AST analysis for undefined variables (NameError pre-check)
    try:
        import ast
        tree = ast.parse(code)
        
        # Collect all defined variable names
        defined_names = set()
        # Built-in functions and common modules
        builtins = {'print', 'len', 'range', 'int', 'str', 'float', 'bool', 'list', 'dict', 
                    'set', 'tuple', 'abs', 'min', 'max', 'sum', 'sorted', 'enumerate', 
                    'zip', 'map', 'filter', 'any', 'all', 'open', 'type', 'isinstance',
                    'True', 'False', 'None', 'Exception', 'ValueError', 'TypeError',
                    'KeyError', 'IndexError', 'round', 'pow', 'ord', 'chr', 'hex', 'bin',
                    'divmod', 'reversed', 'slice', 'object', 'super', 'property',
                    'staticmethod', 'classmethod', 'input', 'format', 'repr', 'hash',
                    'id', 'dir', 'vars', 'globals', 'locals', 'eval', 'exec', 'compile',
                    '__name__', '__file__', '__doc__', 'math', 're', 'os', 'sys', 'json',
                    'collections', 'itertools', 'functools', 'typing', 'Counter'}
        defined_names.update(builtins)
        
        # Collect imported names
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != '*':
                        defined_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)
                for arg in node.args.args:
                    defined_names.add(arg.arg)
                for arg in node.args.kwonlyargs:
                    defined_names.add(arg.arg)
                if node.args.vararg:
                    defined_names.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined_names.add(node.args.kwarg.arg)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                defined_names.add(elt.id)
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    defined_names.add(node.target.id)
                elif isinstance(node.target, ast.Tuple):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            defined_names.add(elt.id)
            elif isinstance(node, ast.comprehension):
                if isinstance(node.target, ast.Name):
                    defined_names.add(node.target.id)
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    defined_names.add(node.name)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                        defined_names.add(item.optional_vars.id)
        
        # Check used but undefined variables
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
        
        undefined = used_names - defined_names
        if undefined:
            # Filter out attribute access (e.g. self.xxx)
            undefined = {n for n in undefined if not n.startswith('_')}
            if undefined:
                detected.append("name_error")
                # Log undefined variables for debugging
                print(f"[RepairAgent] detected undefined variable: {undefined}")
    except Exception:
        pass  # AST analysis failure does not affect other checks
    
    return bool(detected), detected


# ============ Prompt Templates ============

PROMPT_FAST_REPAIR = """You are a Python repair expert. Fix the code with minimal changes based on the given error hints.

Hard constraints:
1) Do not change function signature {entry_point}(...);
2) Do not change I/O;
3) Do not add extra print or I/O;
4) For typing errors: only import real types from typing (List, Dict, Tuple, Optional, Union, etc.), do not import int/str/float;
5) Keep indentation and style;
6) Return the complete fixed code, no explanation.

Error type: {error_type}
Error message (may be empty): {error_message}

Common error patterns (reference only):
- from typing import lIST/lst -> from typing import List
- from typing import int/str/float -> remove the line (these are builtins)
- resutl -> result; TruE -> True
- NameError: undefined variable -> change to nearest matching variable or parameter
- Syntax spelling: else if -> elif, elses -> else

Original code:
```python
{code}
```

Output the complete fixed code (enclosed in ```python and ```):"""


PROMPT_DEEP_REPAIR = """You are a "code debug-repair" agent.

Task: Fix errors in the Python code below based on task description, failed test cases, and failure cause.

Requirements:
1) Preserve function signature {entry_point}(...) and return contract;
2) Fix should cover failed cases without introducing side effects;
3) Output the complete fixed code, no explanation.

Task description (summary):
{task_description}

Failure summary (may be empty):
{error_message}

Failed test cases (optional):
{test_cases}

Offline attribution (may be empty):
{who_when_cause}

Repair evidence (top-k<=3):
{distilled_evidence}

Original code:
```python
{code}
```

Output the complete fixed code (enclosed in ```python and ```):"""


# ============ LLM Calls ============

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_code_from_response(response_text: str) -> str:
    """Extract code block from LLM response"""
    if not response_text:
        return ""
    
    blocks = _CODE_BLOCK_RE.findall(response_text)
    if blocks:
        # Return last code block (usually the final result)
        return blocks[-1].strip()
    
    # If no code block markers, try returning directly (might be pure code)
    lines = response_text.strip().split('\n')
    # Filter out obvious explanatory text
    code_lines = [l for l in lines if not l.strip().startswith(('#', '//', '/*', '*', '"""', "'''")) 
                  or l.strip().startswith('# ')]
    return '\n'.join(code_lines).strip()


def call_repair_llm(
    prompt: str,
    *,
    stage: str = "T1",
    max_retries: int = 5,  # 5 retries to handle 503 temporarily unavailable
    use_strong_model: bool = False,
) -> Tuple[Optional[str], int]:
    """
    Call LLM for code repair
    
    Args:
        prompt: Complete repair prompt
        stage: Repair stage (T0/T1/T2)
        max_retries: Max retries
        use_strong_model: Whether to use strong model (complex problems)
    
    Returns:
        Tuple[Optional[str], int]: (fixed code, tokens used) or (None, 0)
    """
    cfg = _get_repair_config()
    
    # T2 uses separate config if set
    if stage == "T2" and cfg.get("t2_api_key"):
        api_key = cfg["t2_api_key"]
        base_url = cfg.get("t2_base_url", cfg["base_url"])
        model_to_use = cfg.get("t2_model", cfg["model"])
    else:
        api_key = cfg["api_key"]
        base_url = cfg["base_url"]
        model_to_use = cfg["model"]
    
    if not api_key:
        print(f"[RepairAgent] ⚠️ REPAIR_API_KEY not set, skipping LLM call")
        return None, 0
    
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Adjust params by stage
    max_tokens = cfg["max_tokens"]
    timeout = cfg["timeout"]
    if stage == "T0":
        max_tokens = min(512, max_tokens)  # T0 uses fewer tokens
        timeout = min(20, timeout)  # T0 shorter timeout
    elif stage == "T2":
        max_tokens = max(1024, max_tokens)  # T2 allows more tokens
        timeout = max(45, timeout)  # T2 longer timeout
    
    # Select model
    # Note: T2-specific config takes priority over strong_model
    if stage == "T2" and cfg.get("t2_model"):
        model = cfg["t2_model"]
    elif use_strong_model:
        model = cfg["strong_model"]
    else:
        model = model_to_use
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg["temperature"],
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    total_tokens = 0
    last_error = None
    
    for attempt in range(max_retries):
        # Exponential backoff delay (no delay on first attempt)
        if attempt > 0:
            backoff = min(2 ** attempt, 10)  # 2, 4, 8, max 10 seconds
            print(f"[RepairAgent] ⏳ waiting {backoff}s s before retry...")
            time.sleep(backoff)
        
        try:
            with httpx.Client(timeout=timeout) as client:
                # Use content param with UTF-8 encoding
                import json as json_module
                body = json_module.dumps(payload, ensure_ascii=False).encode('utf-8')
                resp = client.post(url, headers=headers, content=body)
                resp.raise_for_status()
                data = resp.json()
                
                # Extract token usage
                usage = data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                if not total_tokens:
                    # Estimate tokens if API did not return usage
                    total_tokens = len(prompt) // 4 + max_tokens // 2
                
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    code = _extract_code_from_response(content)
                    if code:
                        print(f"[RepairAgent] ✅ {stage} LLM repair succeeded (model={model}, tokens={total_tokens}, attempt={attempt+1})")
                        
                        # Record token usage
                        if _HAS_STRATEGY:
                            record_token_usage(stage, total_tokens, success=True)
                        
                        return code, total_tokens
                
                print(f"[RepairAgent] ⚠️ {stage} LLM returned empty content (attempt={attempt+1})")
        
        except httpx.TimeoutException:
            last_error = "timeout"
            print(f"[RepairAgent] ⚠️ {stage} LLM request timed out (attempt={attempt+1}/{max_retries})")
        except httpx.HTTPStatusError as e:
            last_error = f"http_{e.response.status_code}"
            # 503 temporarily unavailable, worth retrying
            if e.response.status_code == 503:
                print(f"[RepairAgent] ⚠️ {stage} LLM service temporarily unavailable 503 (attempt={attempt+1}/{max_retries})")
            elif e.response.status_code == 429:
                print(f"[RepairAgent] ⚠️ {stage} LLM rate limited 429 (attempt={attempt+1}/{max_retries})")
            else:
                print(f"[RepairAgent] ⚠️ {stage} LLM HTTP error: {e.response.status_code} (attempt={attempt+1}/{max_retries})")
        except Exception as e:
            last_error = str(e)[:50]
            print(f"[RepairAgent] ⚠️ {stage} LLM call exception: {e} (attempt={attempt+1}/{max_retries})")
    
    # All retries exhausted
    print(f"[RepairAgent] ❌ {stage} LLM repair failed after {max_retries} retries (last_error={last_error})")
    
    # Record failed token usage
    if _HAS_STRATEGY and total_tokens > 0:
        record_token_usage(stage, total_tokens, success=False)
    
    return None, total_tokens


# ============ Repair Entry Point ============

def repair_code(
    code: str,
    *,
    entry_point: str = "",
    error_type: str = "default",
    error_message: str = "",
    task_description: str = "",
    test_cases: Optional[List[str]] = None,
    distilled_evidence: Optional[List[str]] = None,
    who_when: Optional[Dict[str, Any]] = None,
    stage: str = "T1",
) -> Optional[Dict[str, Any]]:
    """
    Main entry point for code repair
    
    Args:
        code: Code to repair
        entry_point: Function name
        error_type: Error type
        error_message: Error message
        task_description: Task description
        test_cases: Failed test cases
        distilled_evidence: Distilled cure evidence
        who_when: Who&When attribution
        stage: Repair stage (T0/T1/T2)
    
    Returns:
        {
            "fixed_code": str,
            "fixes_applied": List[Dict],
            "confidence": float,
            "stage": str,
            "method": str  # "regex" or "llm"
        }
        or None if repair failed
    """
    if not code:
        return None
    
    # T0: Try fast regex fix first
    if stage == "T0":
        fixed, fixes = fast_typing_fix(code)
        if fixes:
            return {
                "fixed_code": fixed,
                "fixes_applied": fixes,
                "confidence": 0.95,
                "stage": "T0",
                "method": "regex",
                "tokens_used": 0,
            }
        
        # If regex didn't fix but errors detected, try short LLM call
        # If T0_REGEX_ONLY=on, skip T0 LLM call, let T2 handle it
        t0_regex_only = os.getenv("T0_REGEX_ONLY", "off").strip().lower() in ("1", "true", "on", "yes")
        if t0_regex_only:
            return None  # Skip T0 LLM, let T2 handle
        
        has_errors, error_types = detect_obvious_errors(code)
        if has_errors:
            prompt = PROMPT_FAST_REPAIR.format(
                entry_point=entry_point or "solution",
                error_type=", ".join(error_types) if error_types else error_type,
                error_message=error_message[:500] if error_message else "",
                code=code,
            )
            fixed_code, tokens_used = call_repair_llm(prompt, stage="T0")
            if fixed_code and fixed_code != code:
                return {
                    "fixed_code": fixed_code,
                    "fixes_applied": [{"type": "llm_micro_edit", "error_types": error_types}],
                    "confidence": 0.85,
                    "stage": "T0",
                    "method": "llm",
                    "tokens_used": tokens_used,
                }
        
        return None
    
    # T1: Pre-test fix (medium LLM call)
    if stage == "T1":
        # Try fast fix first
        fixed, fixes = fast_typing_fix(code)
        if fixes:
            code = fixed  # Continue from fast fix result
        
        prompt = PROMPT_FAST_REPAIR.format(
            entry_point=entry_point or "solution",
            error_type=error_type,
            error_message=error_message[:800] if error_message else "",
            code=code,
        )
        fixed_code, tokens_used = call_repair_llm(prompt, stage="T1")
        if fixed_code and fixed_code != code:
            all_fixes = fixes + [{"type": "llm_semantic_fix", "error_type": error_type}]
            return {
                "fixed_code": fixed_code,
                "fixes_applied": all_fixes,
                "confidence": 0.80,
                "stage": "T1",
                "method": "llm",
                "tokens_used": tokens_used,
            }
        
        # If LLM added nothing but regex fixed something, return regex result
        if fixes:
            return {
                "fixed_code": fixed,
                "fixes_applied": fixes,
                "confidence": 0.90,
                "stage": "T1",
                "method": "regex",
                "tokens_used": 0,
            }
        
        return None
    
    # T2: Deep fix (strong LLM + full context)
    if stage == "T2":
        # Smart skip: logic errors may not be worth T2 repair
        if _HAS_STRATEGY:
            skip_repair, skip_reason = should_skip_repair(error_type, error_message)
            if skip_repair:
                print(f"[RepairAgent] ⏭️ T2 skipped: {skip_reason}")
                return {
                    "fixed_code": None,
                    "fixes_applied": [],
                    "confidence": 0.0,
                    "stage": "T2",
                    "method": "skipped",
                    "skip_reason": skip_reason,
                    "tokens_used": 0,
                }
        
        # Try fast fix first; if regex fixed and env var set, return regex result to avoid unnecessary deep rewrite
        fixed, fixes = fast_typing_fix(code)
        if fixes:
            regex_only = os.getenv("ALWAYS_T2_REGEX_ONLY_ON_FIX", "off").strip().lower() in ("1", "true", "on", "yes")
            if regex_only:
                return {
                    "fixed_code": fixed,
                    "fixes_applied": fixes,
                    "confidence": 0.85,
                    "stage": "T2",
                    "method": "regex",
                    "tokens_used": 0,
                }
            code = fixed
        
        # Build full context
        test_cases_str = ""
        if test_cases:
            test_cases_str = "\n".join(f"- {tc}" for tc in test_cases[:5])
        
        evidence_str = ""
        if distilled_evidence:
            evidence_str = "\n".join(f"- {ev}" for ev in distilled_evidence[:3])
        
        who_when_cause = ""
        if who_when:
            who_when_cause = who_when.get("cause_text", "")
            if who_when.get("mistake_agent"):
                who_when_cause = f"[{who_when.get('mistake_agent')} @ step {who_when.get('mistake_step', '?')}] {who_when_cause}"
        
        # Decide whether to use strong model based on error type
        use_strong = False
        if _HAS_STRATEGY:
            strategy = get_repair_strategy(error_type)
            if strategy.get("expected_success_rate", 0) < 0.5:
                use_strong = True  # Use strong model for hard problems
        
        prompt = PROMPT_DEEP_REPAIR.format(
            entry_point=entry_point or "solution",
            task_description=task_description[:1000] if task_description else "(no description)",
            error_message=error_message[:1000] if error_message else "(no error message)",
            test_cases=test_cases_str or "(no failed test cases)",
            who_when_cause=who_when_cause or "(no attribution info)",
            distilled_evidence=evidence_str or "(no repair evidence)",
            code=code,
        )
        
        fixed_code, tokens_used = call_repair_llm(prompt, stage="T2", use_strong_model=use_strong)
        if fixed_code and fixed_code != code:
            all_fixes = fixes + [{"type": "llm_deep_fix", "error_type": error_type, "has_who_when": bool(who_when_cause)}]
            return {
                "fixed_code": fixed_code,
                "fixes_applied": all_fixes,
                "confidence": 0.75,
                "stage": "T2",
                "method": "llm",
                "tokens_used": tokens_used,
                "used_strong_model": use_strong,
            }
        
        # If LLM added nothing but regex fixed something, return regex result
        if fixes:
            return {
                "fixed_code": fixed,
                "fixes_applied": fixes,
                "confidence": 0.85,
                "stage": "T2",
                "method": "regex",
                "tokens_used": tokens_used,
            }
        
        return None
    
    return None


# ============ Repair History ============

_REPAIR_HISTORY: Dict[str, List[Dict]] = {}


def record_repair(task_id: str, repair_result: Dict[str, Any]) -> None:
    """Record repair history"""
    if not task_id:
        return
    
    entry = {
        "ts_ms": int(time.time() * 1000),
        **repair_result
    }
    
    if task_id not in _REPAIR_HISTORY:
        _REPAIR_HISTORY[task_id] = []
    _REPAIR_HISTORY[task_id].append(entry)


def get_repair_history(task_id: str) -> List[Dict]:
    """Get repair history"""
    return _REPAIR_HISTORY.get(task_id, [])


def snapshot_repair_history(*, clear: bool = False) -> Dict[str, List[Dict]]:
    """Get snapshot of all repair history"""
    snap = {tid: list(entries) for tid, entries in _REPAIR_HISTORY.items()}
    if clear:
        _REPAIR_HISTORY.clear()
    return snap
