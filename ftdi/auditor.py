from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
HEDGE_RE = re.compile(r"\b(maybe|probably|perhaps|i think|not sure|guess|seems like|might be)\b", re.IGNORECASE)

BOUNDARY_RE = re.compile(
    r"\b(if\s+(?:n\s*==\s*0|not\s+\w+|len\(.+?\)\s*==\s*0|n\s*<\s*1|\w+\s+is\s+None)|"
    r"return\s+\[\]|return\s+''|return\s+0)\b",
    re.IGNORECASE,
)
OBOE_RE = re.compile(
    r"(range\s*\(\s*len\(.+?\)\s*\)|range\s*\(.+?,\s*n\s*\)|\[\s*n\s*\]|\+\s*1\b|-\s*1\b)",
    re.IGNORECASE,
)

EXCEPTION_TO_FAIL_TYPE = {
    "ModuleNotFoundError": "syntax_import",
    "ImportError": "syntax_import",
    "SyntaxError": "syntax_import",
    "IndentationError": "syntax_import",
    "NameError": "name_attribute",
    "AttributeError": "name_attribute",
    "TypeError": "type_error",
    "IndexError": "boundary_condition",
    "KeyError": "boundary_condition",
    "ValueError": "type_error",
    "ZeroDivisionError": "boundary_condition",
    "AssertionError": "semantic_logic",
}
EXCEPTION_RE = re.compile(r"\b(" + "|".join(map(re.escape, EXCEPTION_TO_FAIL_TYPE)) + r")\b")
TRACEBACK_LINE_RE = re.compile(r"line\s+(\d+)", re.IGNORECASE)

SHALLOW_TYPES = {"syntax_import", "name_attribute", "format_deviation"}
LOCAL_EDIT_TYPES = {"uncertainty", "plan_inconsistency", "type_error"}
DEEP_TYPES = {"boundary_condition", "off_by_one", "semantic_logic"}


@dataclass(frozen=True)
class AuditorCfg:
    w_dev: float = 0.35
    w_unc: float = 0.20
    w_con: float = 0.25
    w_bnd: float = 0.20
    w_oboe: float = 0.20
    role_planner: str = "Planner"


class Auditor:
    """Deterministic FTDI auditor.

    This class intentionally exposes only the paper-level auditing method:
    ``audit_failure(history, candidate_code, test_logs, problem)``.
    It does not provide old per-message score fields such as ``score_message``.
    """

    def __init__(self, cfg: AuditorCfg = AuditorCfg()):
        self.cfg = cfg

    def audit_failure(
        self,
        *,
        history: Optional[Iterable[Any]],
        candidate_code: str,
        test_logs: str,
        problem: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return the four FTDI diagnostic outputs.

        Parameters
        ----------
        history:
            Recent multi-agent trajectory H_t. Items may be message objects or dicts.
        candidate_code:
            Candidate program y, either raw code or a markdown message with a Python block.
        test_logs:
            Tester logs ell_t.
        problem:
            Task dictionary. ``entry_point`` and ``prompt`` are used when available.
        """
        code = extract_last_python_block(candidate_code) or str(candidate_code or "")
        logs = str(test_logs or "")
        entry = str(problem.get("entry_point") or "").strip()
        planner_text = _latest_role_text(history, self.cfg.role_planner)

        # Runtime/tester path: concrete exceptions dominate and short-circuit s=1.0.
        runtime_type = self._runtime_fail_type(logs)
        if runtime_type:
            suspect_span = self._localize_span(code=code, logs=logs, entry=entry, fail_type=runtime_type)
            return {
                "s": 1.0,
                "fail_type": runtime_type,
                "suspect_span": suspect_span,
                "a": self._recommend_action(runtime_type),
            }

        # Static diagnostic dimensions.
        dev = self._score_deviation(candidate_text=candidate_code, code=code, entry=entry)
        unc = self._score_uncertainty(candidate_text=candidate_code)
        con = self._score_consistency(code=code, entry=entry, planner_text=planner_text)
        bnd = self._score_boundary(code=code)
        oboe = self._score_off_by_one(code=code)

        dims = {
            "format_deviation": dev,
            "uncertainty": unc,
            "plan_inconsistency": con,
            "boundary_condition": bnd,
            "off_by_one": oboe,
        }
        weights = {
            "format_deviation": self.cfg.w_dev,
            "uncertainty": self.cfg.w_unc,
            "plan_inconsistency": self.cfg.w_con,
            "boundary_condition": self.cfg.w_bnd,
            "off_by_one": self.cfg.w_oboe,
        }
        denom = sum(weights.values()) or 1.0
        s = sum(weights[k] * dims[k] for k in dims) / denom
        s = max(0.0, min(1.0, float(s)))

        fail_type = max(dims.items(), key=lambda kv: kv[1])[0]
        if dims[fail_type] <= 0.0:
            fail_type = "semantic_logic" if logs else "format_deviation"

        suspect_span = self._localize_span(code=code, logs=logs, entry=entry, fail_type=fail_type)
        return {
            "s": s,
            "fail_type": fail_type,
            "suspect_span": suspect_span,
            "a": self._recommend_action(fail_type),
        }

    def _runtime_fail_type(self, logs: str) -> Optional[str]:
        match = EXCEPTION_RE.search(logs or "")
        if not match:
            return None
        return EXCEPTION_TO_FAIL_TYPE.get(match.group(1))

    def _score_deviation(self, *, candidate_text: str, code: str, entry: str) -> float:
        text = str(candidate_text or "")
        score = 0.0
        has_block = bool(extract_last_python_block(text))
        if not has_block:
            score += 0.50
        if code and len(text.replace(code, "")) > 80:
            score += 0.20
        if entry and not re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", code, flags=re.MULTILINE):
            score += 0.30
        return min(1.0, score)

    def _score_uncertainty(self, *, candidate_text: str) -> float:
        text = str(candidate_text or "")
        score = 0.0
        if HEDGE_RE.search(text):
            score += 0.50
        n = len(text.strip())
        if n < 40 or n > 6000:
            score += 0.50
        return min(1.0, score)

    def _score_consistency(self, *, code: str, entry: str, planner_text: str) -> float:
        plan = str(planner_text or "").lower()
        score = 0.0
        if entry and entry.lower() not in code.lower():
            score += 0.50
        if "do not change the function signature" in plan and entry:
            if not re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", code, flags=re.MULTILINE):
                score += 0.50
        return min(1.0, score)

    def _score_boundary(self, *, code: str) -> float:
        if not code.strip():
            return 0.0
        return 0.0 if BOUNDARY_RE.search(code) else 0.80

    def _score_off_by_one(self, *, code: str) -> float:
        if not code.strip():
            return 0.0
        return 0.60 if OBOE_RE.search(code) else 0.0

    def _localize_span(self, *, code: str, logs: str, entry: str, fail_type: str) -> str:
        # Prefer concrete traceback line numbers when present.
        line_matches = TRACEBACK_LINE_RE.findall(logs or "")
        if line_matches:
            return f"L{line_matches[-1]}"

        lines = (code or "").splitlines()
        if entry:
            for i, line in enumerate(lines, start=1):
                if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", line):
                    if fail_type in {"format_deviation", "name_attribute", "syntax_import"}:
                        return f"L{i}:entry_point"

        if fail_type == "off_by_one":
            span = _first_matching_line(lines, OBOE_RE)
            if span:
                return span
        if fail_type == "boundary_condition":
            # Boundary bugs often concentrate around loops/returns/conditionals.
            span = _first_matching_line(lines, re.compile(r"\b(if|for|while|return)\b"))
            if span:
                return span
        if fail_type == "format_deviation" and entry:
            return "function_signature"
        return "unknown"

    def _recommend_action(self, fail_type: str) -> str:
        if fail_type in SHALLOW_TYPES:
            return "T0"
        if fail_type in LOCAL_EDIT_TYPES:
            return "T1"
        if fail_type in DEEP_TYPES:
            return "T2"
        return "None"


def extract_last_python_block(text: str) -> str:
    if not isinstance(text, str):
        return ""
    blocks = _CODE_BLOCK_RE.findall(text or "")
    if not blocks:
        return ""
    candidates = [
        b for b in blocks
        if len(b.strip()) >= 16 or re.search(r"\b(def|return|for|while|if|class)\b", b)
    ]
    return (candidates[-1] if candidates else max(blocks, key=len)).strip()


def _latest_role_text(history: Optional[Iterable[Any]], role_name: str) -> str:
    if history is None:
        return ""
    result = ""
    for item in history:
        if isinstance(item, Mapping):
            role = item.get("role", "")
            content = item.get("content") or item.get("message") or item.get("text") or ""
        else:
            role = getattr(item, "role", "")
            content = getattr(item, "content", "") or ""
        if role == role_name:
            result = str(content)
    return result


def _first_matching_line(lines: List[str], pattern: re.Pattern[str]) -> str:
    for i, line in enumerate(lines, start=1):
        if pattern.search(line):
            return f"L{i}"
    return ""
