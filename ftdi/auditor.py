from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, Any
from autoagents.environment import Environment as EnvUtil

HEDGE = r"\b(maybe|probably|perhaps|i think|not sure|guess)\b"

# Lightweight heuristics
BOUNDARY_RE = re.compile(r"\bif\s+(?:n\s*==\s*0|not\s+\w+|len\(.+?\)\s*==\s*0|n\s*<\s*1|n\s*is\s*None)\b", re.I)
OBOE_RE = re.compile(r"\bfor\s+\w+\s+in\s+range\s*\(\s*len\(.+?\)\s*\)|\[\s*n\s*\]|range\s*\(.+?,\s*n\s*\)", re.I)
# Runtime error keywords (from Tester messages / fail_summary)
RUNTIME_ERR_RE = re.compile(
    r"\b(ImportError|SyntaxError|IndentationError|NameError|AttributeError|"
    r"TypeError|IndexError|KeyError|ValueError|ZeroDivisionError)\b"
)

ERR_MAP = {
    "ImportError": "import_error",
    "SyntaxError": "syntax_error",
    "IndentationError": "indentation_error",
    "NameError": "name_error",
    "AttributeError": "attribute_error",
    "TypeError": "type_error",
    "IndexError": "type_error",
    "KeyError": "type_error",
    "ValueError": "type_error",
    "ZeroDivisionError": "type_error",
}


@dataclass
class AuditorCfg:
    # Lower heuristic weights to advisory level
    w_dev: float = 0.35
    w_unc: float = 0.20
    w_con: float = 0.25
    w_bnd: float = 0.20
    w_oboe: float = 0.20
    role_coder: str = "Coder"
    role_planner: str = "Planner"
    role_tester: str = "Tester"


class Auditor:
    def __init__(self, cfg: AuditorCfg = AuditorCfg()):
        self.cfg = cfg

    def score_message(self, msg, env, problem: Dict[str, Any]) -> Dict[str, Any]:
        role = getattr(msg, "role", "")
        text = (getattr(msg, "content", "") or "").strip()
        code = EnvUtil._extract_last_python_block(text) or ""
        entry = str(problem.get("entry_point") or "").strip()

        # --- 1) Deviation / Uncertainty / Consistency ---
        dev = 0.0
        if role == self.cfg.role_coder:
            has_block = (code != "")
            dev += 0.5 if not has_block else 0.0
            if has_block and len(text.replace(code, "")) > 40:  # excessive surrounding text
                dev += 0.25
            if entry:
                import re as _re
                ok_sig = bool(_re.search(rf"^\s*def\s+{entry}\s*\(", code, flags=_re.M))
                dev += 0.25 if not ok_sig else 0.0
        dev = min(1.0, dev)

        unc = 0.0
        import re as _re
        if _re.search(HEDGE, text, flags=_re.I):
            unc += 0.5
        if role == self.cfg.role_coder:
            n = len(text)
            if n < 40 or n > 6000:
                unc += 0.5
        unc = min(1.0, unc)

        con = 0.0
        if role == self.cfg.role_coder:
            plan = env._latest_role_message_text(self.cfg.role_planner) or ""
            if plan and ("test" in plan.lower()) and ("```" not in text):
                con += 0.5
            if "do not change the function signature" in plan.lower():
                if not _re.search(r"```python", text, flags=_re.I):
                    con += 0.5
        con = min(1.0, con)

        # --- 2) Code-level heuristics (early warning for Coder messages) ---
        bnd = 0.0
        oboe = 0.0
        if role == self.cfg.role_coder and code:
            if not BOUNDARY_RE.search(code):
                bnd += 0.8
            if OBOE_RE.search(code):
                oboe += 0.6
        bnd, oboe = min(1.0, bnd), min(1.0, oboe)

        # --- 3) Runtime errors (Tester messages map directly to fail_type) ---
        if role == self.cfg.role_tester and text:
            for exc, ftype in ERR_MAP.items():
                if exc in text:
                    return {"dev": 0, "unc": 0, "con": 0, "bnd": 0, "oboe": 0, "score": 1.0, "fail_type": ftype}

        # Aggregate (dual track: unweighted raw + weighted scores)
        raw_map = {
            "deviation": dev,
            "uncertainty": unc,
            "consistency": con,
            "boundary_condition": bnd,
            "off_by_one": oboe,
        }
        weighted = {
            "deviation": dev * self.cfg.w_dev,
            "uncertainty": unc * self.cfg.w_unc,
            "consistency": con * self.cfg.w_con,
            "boundary_condition": bnd * self.cfg.w_bnd,
            "off_by_one": oboe * self.cfg.w_oboe,
        }
        score = float(sum(weighted.values()))
        # fail_type based on highest unweighted item to avoid weight bias; top2 for reference
        try:
            sorted_raw = sorted(raw_map.items(), key=lambda kv: kv[1], reverse=True)
        except Exception:
            sorted_raw = [("default", 0.0)]
        fail_type = (sorted_raw[0][0] if sorted_raw and sorted_raw[0][1] > 0 else "default")
        top2 = [k for k, _ in sorted_raw[:2]] if sorted_raw else []
        max_v = sorted_raw[0][1] if sorted_raw else 0.0
        second_v = sorted_raw[1][1] if len(sorted_raw) > 1 else 0.0
        confidence = max(0.0, float(max_v - second_v))
        return {
            "dev": dev,
            "unc": unc,
            "con": con,
            "bnd": bnd,
            "oboe": oboe,
            "score": score,
            "fail_type": fail_type,
            "fail_type_top2": top2,
            "fail_confidence": confidence,
        }
