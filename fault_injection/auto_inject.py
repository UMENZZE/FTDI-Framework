from __future__ import annotations
import os
import json
import random
import re
import time
from dataclasses import dataclass, asdict
import ast
from typing import Dict, List, Literal, Optional, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# ----------------------------- Public configuration -----------------------------
ErrorType = Literal["syntax", "semantic", "text", "translate"]

@dataclass
class Config:
    # Prob. that *a message* will be injected
    Pm: float = float(os.getenv("AUTOINJECT_PM", "0.2"))
    # Prob. that *a selected line* in that message will be injected
    Pe: float = float(os.getenv("AUTOINJECT_PE", "0.2"))

    # Error type for this run
    error_type: ErrorType = os.getenv("AUTOINJECT_ERROR_TYPE", "semantic")  # type: ignore

    # Hard cap on how many lines to modify within a single message
    # Default 1, can be overridden by env (and CLI setting env)
    max_lines: int = int(os.getenv("AUTOINJECT_MAX_LINES", "1"))

    # OpenAI model + API key
    # Priority: AUTOINJECT_MODEL > OPENAI_API_MODEL > fallback default
    model: str = (
        os.getenv("AUTOINJECT_MODEL")
        or os.getenv("OPENAI_API_MODEL")
        or os.getenv("PROBEX_MODEL")
        or "Qwen3-Coder-30B-A3B-Instruct"
    )
    api_key: Optional[str] = (
        os.getenv("AUTOINJECT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("PROBEX_API_KEY")
    )
    # Base URL for OpenAI-compatible endpoints (vLLM/openai-proxy/Azure, etc.)
    # Prefer OPENAI_BASE_URL (OpenAI SDK standard), fallback to OPENAI_API_BASE used in this repo.
    base_url: Optional[str] = (
        os.getenv("AUTOINJECT_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("PROBEX_CHAT_BASE")
        or None
    )

    # RNG seed for reproducibility (None → non‑deterministic)
    seed: Optional[int] = int(os.getenv("AUTOINJECT_SEED", "42")) if os.getenv("AUTOINJECT_SEED", "") != "" else None

    # Optional JSONL file to append manifests
    manifest_path: Optional[str] = os.getenv("AUTOINJECT_MANIFEST", None)

CFG = Config()

def _is_protected_code_line(line: str) -> bool:
    st = line.strip()
    if not st:
        return True
    if st.startswith("#"):
        return True
    if st.startswith("def ") or st.startswith("class "):
        return True
    return False

# ----------------------------- Prompt templates -----------------------------
# We only ever ask the LLM to rewrite the *selected* lines.
# The model must return a JSON object: {"0": "new line for index 0", ...}

PROMPTS: Dict[ErrorType, str] = {
    "syntax": (
        "You will receive a JSON object whose keys are line indices and values are lines of Python code.\n"
        "Rewrite each provided line to introduce a *syntax* error while keeping the *indentation identical*.\n"
        "Do not add or remove lines; only modify tokens on that line. Do not touch comments.\n"
        "Examples: remove a colon, break parentheses/brackets, misspell a keyword.\n"
        "Return ONLY a JSON object mapping the same indices to the rewritten lines."),
    "semantic": (
        "You will receive a JSON object whose keys are line indices and values are lines of Python code.\n"
        "Rewrite each provided line to introduce a *subtle logic bug* but keep the code syntactically valid and indentation unchanged.\n"
        "Make a *single small* change per line (e.g., >→>= or <→<=, off‑by‑one, wrong variable, swapped operands).\n"
        "Do not change function signatures, imports, or add I/O.\n"
        "Return ONLY a JSON object mapping the same indices to the rewritten lines."),
    "text": (
        "You will receive a JSON object whose keys are line indices and values are sentences used in a *text evaluation* scenario.\n"
        "Rewrite each provided sentence to subtly change the meaning such that the evaluation would prefer the wrong choice.\n"
        "Keep grammar valid and style similar. Return ONLY a JSON object mapping the same indices to the rewritten sentences."),
    "translate": (
        "You will receive a JSON object whose keys are line indices and values are *translations*.\n"
        "Rewrite each provided translation to introduce subtle errors (literalism, wrong word sense, number mismatch), while keeping the language fluent.\n"
        "Return ONLY a JSON object mapping the same indices to the rewritten sentences.")
}

# ----------------------------- Injector core -----------------------------
class AutoInjector:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)
        # Construct OpenAI client with base_url for local vLLM/OpenAI-compatible gateways
        if OpenAI:
            try:
                # Some gateways ignore api_key; provide a default if missing
                api_key = cfg.api_key or "EMPTY"
                kwargs = {"api_key": api_key}
                if cfg.base_url:
                    kwargs["base_url"] = cfg.base_url
                self._client = OpenAI(**kwargs)
            except Exception:
                # As a last resort, disable client so we fallback gracefully
                self._client = None
        else:
            self._client = None

    # ---- public API (drop‑in): same signature as your old function ----
    def modify(self, message: str, *, iscode: bool = True, ismath: bool = False) -> str:
        """Maybe inject errors into a message.

        Args:
            message: original message text.
            iscode: if True, split by lines ("\n"); otherwise, sentence-like splitting.
            ismath: optional math branch (applies small numeric drift if no LLM return).
        Returns:
            Possibly modified message (string). If no injection is triggered, returns the original.
        """
        original_payload = message if isinstance(message, str) else ("" if message is None else str(message))
        candidate_count = 0
        message_skipped = False
        diagnostic_reason: Optional[str] = None
        ast_guard_triggered = False

        def _finalize(
            output_value: str,
            *,
            changes: Optional[List[Dict]] = None,
            selected: Optional[List[int]] = None,
            meta: Optional[Dict[str, object]] = None,
        ) -> str:
            local_changes = changes or []
            local_selected = selected or []
            meta_payload: Dict[str, object] = {
                "candidate_count": candidate_count,
                "selected_count": len(local_selected),
                "effective_changes": len(local_changes),
                "message_skipped": message_skipped,
                "ast_guard_triggered": ast_guard_triggered,
            }
            if diagnostic_reason:
                meta_payload.setdefault("diagnostic_reason", diagnostic_reason)
            if meta:
                meta_payload.update(meta)
            self._emit_manifest(
                original_payload,
                output_value if isinstance(output_value, str) else ("" if output_value is None else str(output_value)),
                local_changes,
                local_selected,
                meta=meta_payload,
            )
            return output_value

        if not isinstance(message, str):
            diagnostic_reason = "non_string_message"
            return _finalize(message)

        if not message:
            diagnostic_reason = "empty_message"
            return _finalize(message)

        original_is_valid_ast = False
        if iscode and self.cfg.error_type != "syntax":
            try:
                ast.parse(message)
                original_is_valid_ast = True
            except Exception:
                original_is_valid_ast = False

        sep = "\n" if iscode else "."
        parts = self._split_units(message, sep)

        def _is_fence(s: str) -> bool:
            return s.lstrip().startswith("```")

        candidate_idxs: List[int] = []
        for i, t in enumerate(parts):
            is_cand = bool(t.strip()) and not _is_fence(t) and (not iscode or not _is_protected_code_line(t))
            if is_cand:
                candidate_idxs.append(i)
        candidate_count = len(candidate_idxs)
        if not candidate_idxs:
            diagnostic_reason = "no_candidates"
            return _finalize(message)

        # Decide whether to inject this message (Pm gate after diagnostics are available)
        pm_roll = self._rng.random()
        if pm_roll >= max(0.0, min(1.0, self.cfg.Pm)):
            message_skipped = True
            diagnostic_reason = "Pm_gate_skip"
            return _finalize(message, meta={"pm_roll": pm_roll})

        # Sample which lines/sentences to modify
        selected = [i for i in candidate_idxs if self._rng.random() < max(0.0, min(1.0, self.cfg.Pe))]
        if not selected:
            selected = [self._rng.choice(candidate_idxs)]
        if len(selected) > self.cfg.max_lines:
            self._rng.shuffle(selected)
            selected = sorted(selected[: self.cfg.max_lines])

        def _run_micro_edit_fallback(reason: str, base_parts: List[str]) -> Optional[str]:
            if not iscode:
                return None
            nonlocal diagnostic_reason, ast_guard_triggered
            if not candidate_idxs:
                return None

            max_attempts = 3
            base_seed = self._rng.randint(0, 2**31 - 1)
            sample_size = max(1, min(len(selected), len(candidate_idxs)))

            for attempt in range(max_attempts):
                parts_work = base_parts.copy()
                manifest_local: List[Dict] = []
                parse_attempts = 0
                parse_passes = 0
                mutated = False

                if attempt == 0:
                    target_indices = sorted(list(selected))
                else:
                    if len(candidate_idxs) < sample_size:
                        target_indices = list(candidate_idxs)
                    else:
                        target_indices = sorted(self._rng.sample(candidate_idxs, sample_size))

                for idx in target_indices:
                    if idx < 0 or idx >= len(parts_work):
                        continue
                    old = parts_work[idx]
                    if _is_protected_code_line(old):
                        continue
                    seed_value = base_seed + idx
                    candidate = self._safe_micro_edit(old, seed_value)
                    candidate = self._force_same_indent(old, candidate)
                    if candidate == old:
                        continue
                    trial_parts = parts_work.copy()
                    trial_parts[idx] = candidate
                    if self.cfg.error_type != "syntax" and original_is_valid_ast:
                        parse_attempts += 1
                        try:
                            ast.parse(self._join_units(trial_parts, sep))
                        except Exception:
                            ast_guard_triggered = True
                            continue
                        parse_passes += 1
                    parts_work[idx] = candidate
                    manifest_local.append({
                        "index": idx,
                        "old": old,
                        "new": candidate,
                        "fallback": reason,
                    })
                    mutated = True

                if manifest_local:
                    out = self._join_units(parts_work, sep)
                    diagnostic_reason = reason
                    return _finalize(
                        out,
                        changes=manifest_local,
                        selected=target_indices,
                        meta={
                            "fallback": reason,
                            "parse_gate_attempts": parse_attempts,
                            "parse_gate_passes": parse_passes,
                            "parse_gate_ok": bool(parse_passes > 0),
                            "fallback_attempt": attempt + 1,
                        },
                    )

                if not mutated:
                    base_seed += 1

            return None

        # Build payload (index->original); also note if any selected line belongs to protected region (should be empty)
        payload = {str(k): parts[k] for k in selected}
        protected_selected = [k for k in selected if iscode and _is_protected_code_line(parts[k])]

        # Call LLM to get rewrites
        rewritten = self._rewrite_with_llm(payload, iscode=iscode)
        if not rewritten:  # Fallbacks
            fallback_out = _run_micro_edit_fallback("micro_edit_no_llm", parts)
            if fallback_out:
                return fallback_out
            if ismath:
                parts = self._fallback_math(parts, selected)
                diagnostic_reason = "math_drift_fallback"
                return _finalize(self._join_units(parts, sep), selected=selected, changes=[], meta={"fallback": "math_drift"})
            diagnostic_reason = "rewrite_failed_no_fallback"
            return _finalize(self._join_units(parts, sep), selected=selected, changes=[])

        # Apply rewrites with syntax gate; ensure indentation is preserved for code
        manifest: List[Dict] = []
        parse_gate_attempts = 0
        parse_gate_passes = 0
        parts_work = parts.copy()
        batch_semantic_gate = bool(iscode and self.cfg.error_type == "semantic" and len(rewritten) > 1)
        for k_str, new_line in rewritten.items():
            try:
                idx = int(k_str)
            except Exception:
                continue
            if idx < 0 or idx >= len(parts_work):
                continue
            old = parts_work[idx]
            if iscode:
                new_line = self._force_same_indent(old, new_line)
            trial_parts = parts_work.copy()
            trial_parts[idx] = new_line
            if iscode and self.cfg.error_type != "syntax" and original_is_valid_ast and not batch_semantic_gate:
                parse_gate_attempts += 1
                try:
                    ast.parse(self._join_units(trial_parts, sep))
                    parse_gate_passes += 1
                except Exception:
                    ast_guard_triggered = True
                    continue
            parts_work[idx] = new_line
            manifest.append({"index": idx, "old": old, "new": new_line})

        if batch_semantic_gate and manifest and original_is_valid_ast:
            parse_gate_attempts += 1
            try:
                ast.parse(self._join_units(parts_work, sep))
                parse_gate_passes += 1
            except Exception:
                ast_guard_triggered = True
                fallback_out = _run_micro_edit_fallback("micro_edit_batch_ast", parts)
                if fallback_out:
                    return fallback_out
                diagnostic_reason = "batch_semantic_gate_rejected"
                return _finalize(self._join_units(parts, sep), selected=selected, changes=[], meta={"fallback": "batch_semantic_gate"})

        if iscode and not manifest:
            fallback_out = _run_micro_edit_fallback("micro_edit_post_ast", parts_work)
            if fallback_out:
                return fallback_out
            diagnostic_reason = "post_ast_no_manifest"
            return _finalize(self._join_units(parts, sep), selected=selected, changes=[])

        out = self._join_units(parts_work, sep)
        if ast_guard_triggered and diagnostic_reason is None:
            diagnostic_reason = "ast_guard_triggered"
        return _finalize(
            out,
            changes=manifest,
            selected=selected,
            meta={
                "protected_selected": protected_selected,
                "touched_protected": bool(protected_selected),
                "parse_gate_attempts": parse_gate_attempts,
                "parse_gate_passes": parse_gate_passes,
                "parse_gate_ok": bool(parse_gate_passes > 0),
            },
        )

    # ---- helpers ----
    def _rewrite_with_llm(self, payload: Dict[str, str], *, iscode: bool) -> Optional[Dict[str, str]]:
        if not self._client:
            # Try raw HTTP fallback if SDK not available
            return self._rewrite_with_http(payload, iscode=iscode)
        prompt = PROMPTS[self.cfg.error_type]
        # For code, additionally remind about not changing indentation
        if iscode and self.cfg.error_type in ("syntax", "semantic"):
            prompt += "\nRemember: keep the indentation (leading spaces) *exactly* the same as input."
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a transformation engine. You MUST answer with a single valid JSON object only. "
                        "No prose, no code fences, no explanations."
                    ),
                },
                {"role": "user", "content": f"{prompt}\n\nINPUT JSON:\n{json.dumps(payload, ensure_ascii=False)}"},
            ]
            # First attempt: request structured JSON (some servers may not support response_format)
            try:
                completion = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                content = completion.choices[0].message.content
            except Exception:
                # Fallback attempt without response_format
                completion = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=0.2,
                )
                content = completion.choices[0].message.content

            # Try to parse JSON strictly; if fails, try to extract JSON substring
            data = None
            try:
                data = json.loads(content or "{}")
            except Exception:
                try:
                    s = self._extract_json_object(content or "")
                    data = json.loads(s) if s else None
                except Exception:
                    data = None
            if not isinstance(data, dict):
                return None
            # Coerce values to str and keep only provided keys
            out: Dict[str, str] = {}
            for k, v in data.items():
                if k in payload:
                    out[k] = str(v)
            return out
        except Exception:
            # As a last resort, try raw HTTP fallback
            return self._rewrite_with_http(payload, iscode=iscode)

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        """Extract the first top-level JSON object substring from text.
        Handles cases where the model wraps JSON with prose/code fences.
        """
        if not text:
            return None
        # Remove common code fences
        text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip())
        text = re.sub(r"```\s*$", "", text)
        # Scan for matching braces
        start = None
        depth = 0
        for i, ch in enumerate(text):
            if ch == '{':
                if start is None:
                    start = i
                depth += 1
            elif ch == '}':
                if start is not None:
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
        return None

    # ---- raw HTTP fallback (vLLM/OpenAI-compatible) ----
    def _rewrite_with_http(self, payload: Dict[str, str], *, iscode: bool) -> Optional[Dict[str, str]]:
        base = self.cfg.base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        if not base:
            return None
        try:
            import requests  # type: ignore
        except Exception:
            return None
        prompt = PROMPTS[self.cfg.error_type]
        if iscode and self.cfg.error_type in ("syntax", "semantic"):
            prompt += "\nRemember: keep the indentation (leading spaces) *exactly* the same as input."
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a transformation engine. You MUST answer with a single valid JSON object only. "
                    "No prose, no code fences, no explanations."
                ),
            },
            {"role": "user", "content": f"{prompt}\n\nINPUT JSON:\n{json.dumps(payload, ensure_ascii=False)}"},
        ]
        url = base.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key or os.getenv('PROBEX_API_KEY') or os.getenv('OPENAI_API_KEY','EMPTY')}"
        }
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": 0.2,
            # Many OpenAI-compatible gateways (incl. recent vLLM) support JSON mode; harmless to include if ignored
            "response_format": {"type": "json_object"},
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Parse as in SDK branch
            obj = None
            try:
                obj = json.loads(content or "{}")
            except Exception:
                try:
                    s = self._extract_json_object(content or "")
                    obj = json.loads(s) if s else None
                except Exception:
                    obj = None
            if not isinstance(obj, dict):
                return None
            out: Dict[str, str] = {}
            for k, v in obj.items():
                if k in payload:
                    out[k] = str(v)
            return out
        except Exception:
            return None

    @staticmethod
    def _split_units(text: str, sep: str) -> List[str]:
        if sep == "\n":
            return text.splitlines()
        # Sentence‑ish splitting for text; keep the separator during join
        parts: List[str] = []
        buf = []
        for ch in text:
            buf.append(ch)
            if ch == ".":
                parts.append("".join(buf).rstrip())
                buf = []
        if buf:
            parts.append("".join(buf).rstrip())
        return parts

    @staticmethod
    def _join_units(parts: List[str], sep: str) -> str:
        return "\n".join(parts) if sep == "\n" else ". ".join([p.strip() for p in parts if p.strip()])

    @staticmethod
    def _force_same_indent(old: str, new: str) -> str:
        # take leading spaces from old and apply to new
        old_ws = re.match(r"^\s*", old).group(0)
        body = new.lstrip("\r\n")
        body = re.sub(r"^\s*", "", body)
        return old_ws + body

    @staticmethod
    def _strip_triple_backticks(s: str) -> str:
        return re.sub(r"^\s*```.*$", "", s, flags=re.MULTILINE).strip()

    @staticmethod
    def _safe_micro_edit(line: str, seed: int) -> str:
        """Enhanced semantic mutation that targets loops, relations, and fallbacks."""
        st = line.lstrip()
        if not line.strip() or st.startswith('#'):
            return line

        rng = random.Random(seed)

        if "for " in line and " in " in line and ":" in line:
            if "range(" not in line and "enumerate(" not in line:
                return line.replace(":", "[:-1]:", 1)

        rel_ops = {
            "==": "!=",
            "!=": "==",
            "<=": ">",
            ">=": "<",
            "<": ">=",
            ">": "<=",
        }
        for op in sorted(rel_ops.keys(), key=len, reverse=True):
            if op in line:
                return line.replace(op, rel_ops[op], 1)

        if " and " in line:
            return line.replace(" and ", " or ", 1)
        if " or " in line:
            return line.replace(" or ", " and ", 1)
        if " not " in line:
            return line.replace(" not ", " ", 1)

        if "True" in line:
            return line.replace("True", "False", 1)
        if "False" in line:
            return line.replace("False", "True", 1)

        math_ops = {"+": "-", "-": "+", "*": "+", "/": "*", "%": "*"}
        for op, target in math_ops.items():
            token = f" {op} "
            if token in line:
                return line.replace(token, f" {target} ", 1)

        numbers = list(re.finditer(r"\b\d+\b", line))
        if numbers:
            match = rng.choice(numbers)
            val = int(match.group())
            new_val = str(val + rng.choice([-1, 1]))
            return line[:match.start()] + new_val + line[match.end():]

        if "range(" in line:
            return line.replace("range(", "range(1 + ", 1)

        stripped = line.strip()
        if stripped.startswith("return "):
            return line.rstrip() + " + 1"

        if " = " in line:
            parts = line.split(" = ", 1)
            return parts[0] + " = not " + parts[1]

        if stripped.startswith("if ") and ":" in line:
            return line.replace("if ", "if not ", 1)

        return line

    def _fallback_math(self, parts: List[str], selected: List[int]) -> List[str]:
        # Minimal numeric perturbation (+1 on the first integer per selected line)
        for idx in selected:
            line = parts[idx]
            def _bump(m: re.Match[str]) -> str:
                try:
                    return str(int(m.group(0)) + 1)
                except Exception:
                    return m.group(0)
            newline, n = re.subn(r"\b\d+\b", _bump, line, count=1)
            if n > 0:
                parts[idx] = newline
        return parts

    def _emit_manifest(self, original: str, modified: str, changes: List[Dict], selected: List[int], meta: Optional[Dict[str, object]] = None) -> None:
        manifest_off = (os.getenv("AUTOINJECT_MANIFEST_OFF", "").strip().lower() in {"1", "true", "yes", "on"})
        if manifest_off:
            return
        if not self.cfg.manifest_path and not os.getenv("AUTOINJECT_MANIFEST_AGGREGATE"):
            return
        if not changes:
            return
        record = {
            "type": "AutoInject",
            "error_type": self.cfg.error_type,
            "Pm": self.cfg.Pm,
            "Pe": self.cfg.Pe,
            "max_lines": self.cfg.max_lines,
            "seed": self.cfg.seed,
            "round": int(os.getenv("AUTOINJECT_ROUND", "0")),
            "selected_indices": selected,
            "changes": changes,
            "original_sha256": self._sha256(original),
            "modified_sha256": self._sha256(modified),
        }
        if meta:
            record.update({"metrics": meta})
        self._write_manifest_entry(self.cfg.manifest_path, record)
        agg_path = os.getenv("AUTOINJECT_MANIFEST_AGGREGATE")
        if agg_path:
            agg_record = dict(record)
            if self.cfg.manifest_path:
                agg_record.setdefault("per_task_manifest", self.cfg.manifest_path)
            self._write_manifest_entry(agg_path, agg_record)

    @staticmethod
    def _write_manifest_entry(path: Optional[str], payload: Dict) -> None:
        if not path:
            return
        try:
            from pathlib import Path
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _sha256(text: str) -> str:
        import hashlib
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _log_injection(path: str, entry: Dict[str, object]) -> None:
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def inject_fault(
    code_str: str,
    pm: float,
    pe: float,
    error_type: str = "semantic",
    seed: int = 42,
    max_lines: int = 1,
) -> str:
    """Standalone injector that retries across shuffled candidates until enough lines mutate."""
    lines = code_str.split("\n")
    rng = random.Random(seed + len(code_str) + int(time.time()))

    candidate_indices = [i for i, line in enumerate(lines) if not _is_protected_code_line(line)]
    message_skipped = False
    if rng.random() > pm:
        message_skipped = True

    selected_indices: List[int] = []
    changes: List[Dict[str, object]] = []
    mutated_lines = list(lines)

    if not message_skipped and candidate_indices:
        num_to_modify = max(1, int(len(candidate_indices) * pe))
        if max_lines > 0:
            num_to_modify = min(num_to_modify, max_lines)

        rng.shuffle(candidate_indices)
        success_count = 0
        for idx in candidate_indices:
            if success_count >= num_to_modify:
                break
            original_line = lines[idx]
            mutated_line = AutoInjector._safe_micro_edit(original_line, seed + idx)
            mutated_line = AutoInjector._force_same_indent(original_line, mutated_line)
            if mutated_line == original_line:
                continue
            mutated_lines[idx] = mutated_line
            selected_indices.append(idx)
            changes.append(
                {
                    "index": idx,
                    "old": original_line.strip(),
                    "new": mutated_line.strip(),
                }
            )
            success_count += 1

    manifest_path = os.environ.get("AUTOINJECT_MANIFEST")
    if manifest_path and changes:
        entry = {
            "timestamp": time.time(),
            "pm": pm,
            "pe": pe,
            "error_type": error_type,
            "seed": seed,
            "candidates_count": len(candidate_indices),
            "message_skipped": message_skipped,
            "selected_indices": selected_indices,
            "changes": changes,
            "code_snippet": code_str[:100].replace("\n", "\\n"),
        }
        _log_injection(manifest_path, entry)

    return "\n".join(mutated_lines)

# ----------------------------- FTDI module-level wrappers -----------------------------
_default_injector: Optional[AutoInjector] = None
_default_signature: Optional[Tuple[object, ...]] = None


def _env_enabled() -> bool:
    """Respect AUTOINJECT_ENABLED when this file is used directly."""
    explicit = os.getenv("AUTOINJECT_ENABLED", "").strip().lower()
    if explicit in {"0", "false", "no", "off"}:
        return False
    if explicit in {"1", "true", "yes", "on"}:
        return True
    try:
        return float(os.getenv("AUTOINJECT_PM", "0")) > 0.0
    except Exception:
        return False


def _fresh_config() -> Config:
    """Build config from current env values.

    The dataclass defaults are evaluated at import time, so this helper keeps
    repeated evaluation runs from accidentally reusing stale Pm/Pe/type/seed.
    """
    seed_raw = os.getenv("AUTOINJECT_SEED", "42")
    return Config(
        Pm=float(os.getenv("AUTOINJECT_PM", "0.2")),
        Pe=float(os.getenv("AUTOINJECT_PE", "0.2")),
        error_type=os.getenv("AUTOINJECT_ERROR_TYPE", "semantic"),  # type: ignore[arg-type]
        max_lines=int(os.getenv("AUTOINJECT_MAX_LINES", "1")),
        model=(
            os.getenv("AUTOINJECT_MODEL")
            or os.getenv("OPENAI_API_MODEL")
            or os.getenv("PROBEX_MODEL")
            or "Qwen3-Coder-30B-A3B-Instruct"
        ),
        api_key=(
            os.getenv("AUTOINJECT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("PROBEX_API_KEY")
        ),
        base_url=(
            os.getenv("AUTOINJECT_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("PROBEX_CHAT_BASE")
            or None
        ),
        seed=int(seed_raw) if seed_raw != "" else None,
        manifest_path=os.getenv("AUTOINJECT_MANIFEST", None),
    )


def _config_signature(cfg: Config) -> Tuple[object, ...]:
    return (cfg.Pm, cfg.Pe, cfg.error_type, cfg.max_lines, cfg.model, cfg.api_key, cfg.base_url, cfg.seed, cfg.manifest_path)


def _get_default_injector() -> AutoInjector:
    global _default_injector, _default_signature
    cfg = _fresh_config()
    sig = _config_signature(cfg)
    if _default_injector is None or _default_signature != sig:
        _default_injector = AutoInjector(cfg)
        _default_signature = sig
    return _default_injector


def _safe_task_id(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id))


def _ensure_task_manifest(task_id: str) -> None:
    """Provide a paper-style per-task manifest path when caller did not set one."""
    if os.getenv("AUTOINJECT_MANIFEST"):
        return
    workspace = Path(os.getenv("WORKSPACE_ROOT", "./workspace")).expanduser().resolve()
    manifest_dir = workspace / "injections"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AUTOINJECT_MANIFEST"] = str(manifest_dir / f"injection_manifest_{_safe_task_id(task_id)}.jsonl")


# Existing direct-use API: from auto_inject import modify

def modify(message: str, iscode: bool = True, ismath: bool = False) -> str:
    """FTDI evaluation-time injection wrapper.

    If AUTOINJECT_ENABLED is explicitly off, this returns the input unchanged.
    Otherwise Pm/Pe/max_lines/error_type are read from the current environment.
    """
    if not _env_enabled():
        return message
    inj = _get_default_injector()
    return inj.modify(message, iscode=iscode, ismath=ismath)


# New hook-friendly API, matching autoagents_ext.inject_bridge.inject_if_needed.

def inject_if_needed(text: str, *, is_code: bool = True, task_id: str = "unknown") -> str:
    _ensure_task_manifest(task_id)
    return modify(text, iscode=is_code, ismath=False)


# ----------------------------- Example usage (copy/paste into your pipeline) -----------------------------
EXAMPLE = r"""
# Controlled fault injection for FTDI evaluation:
#   AUTOINJECT_ENABLED=on AUTOINJECT_PM=0.8 AUTOINJECT_PE=0.3 \
#   AUTOINJECT_MAX_LINES=1 AUTOINJECT_ERROR_TYPE=semantic AUTOINJECT_SEED=42 \
#   python examples/humaneval_eval.py --use-mas-env ...
#
# Direct use:
#   from autoagents_ext.auto_inject import inject_if_needed
#   msg.content = inject_if_needed(msg.content, is_code=True, task_id=task_id)
"""
