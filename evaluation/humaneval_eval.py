from __future__ import annotations
import argparse
import asyncio
import json
import time
import os
from pathlib import Path
import subprocess  # For calling Who&When inference/evaluate scripts at evaluation end
from typing import Any, Dict
import sys
import re

# Ensure project root on sys.path
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autoagents.environment import Environment
from autoagents.system.logs import logger
from autoagents.system.const import WORKSPACE_ROOT
from autoagents.actions.official_tester import OfficialTesterAction
from autoagents.roles import Role
from autoagents.system.schema import Message
from autoagents_ext.hook import instrument_env
from autoagents_ext.repair_agent import snapshot_repair_history
from autoagents.system.provider.llm_api import LLMAPI, get_token_events


def _resolve_run_group_dir_path(base_ws: Path | None = None) -> Path:
    """Resolve the concrete traces directory honoring RUN_GROUP_DIR semantics."""
    ws_candidate = base_ws or Path(os.environ.get("WORKSPACE_ROOT", str(WORKSPACE_ROOT)))
    try:
        ws_path = ws_candidate if isinstance(ws_candidate, Path) else Path(ws_candidate)
    except Exception:
        ws_path = Path(os.environ.get("WORKSPACE_ROOT", str(WORKSPACE_ROOT)))
    raw = (os.getenv("RUN_GROUP_DIR", "") or "").strip()
    if not raw:
        return ws_path / "traces"
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "traces":
        return ws_path / candidate
    return ws_path / "traces" / candidate

# -------------------- Who&When Minimal Integration: Offline Error Attribution (Optional Tail Flow) --------------------
def run_who_when_pipeline(
    dataset_path: str,
    method: str = "all_at_once",
    model: str = "gpt-4o",
    is_handcrafted: bool = False,
    azure_api_key: str | None = None,
    azure_endpoint: str | None = None,
    api_version: str = "2024-08-01-preview",
) -> Path:

    # Compute real script paths (one level up from examples/ is project root)
    wh_root = _ROOT / "autoagents_ext" / "Automated_FA"
    inf_py = wh_root / "inference.py"
    ev_py = wh_root / "evaluate.py"
    if not inf_py.exists() or not ev_py.exists():
        raise FileNotFoundError(f"Who&When scripts not found: {inf_py} or {ev_py}")

    # Pre-set env: let inference.py load project root's .env.inference
    from pathlib import Path as _P
    _root = _ROOT  # Project root (AutoAgents-main)
    env_vars = os.environ.copy()
    if not env_vars.get("WHO_WHEN_ENV_FILE"):
        for cand in ( _root/".env.inference", _root/".env-inference" ):
            if cand.exists():
                env_vars["WHO_WHEN_ENV_FILE"] = str(cand)
                break

    # 1) Run inference.py (auto-writes outputs/<method>_<model>_*.txt)
    handcrafted_flag = "True" if is_handcrafted else "False"
    inf_cmd = [
        sys.executable, str(inf_py),
        "--method", method,
        "--model", model,
        "--directory_path", dataset_path,
        "--is_handcrafted", handcrafted_flag,
        "--api_version", api_version,
        "--max_tokens", "1024",
    ]
    # For GPT models, pass auth/base URL (inference.py supports both OpenAI-compatible and Azure)
    if azure_api_key:
        inf_cmd += ["--api_key", azure_api_key]
    if azure_endpoint:
        # Also reuse this param name for OpenAI-compatible gateways (e.g. https://xiaoai.plus/v1)
        inf_cmd += ["--azure_endpoint", azure_endpoint]

    print(f"[Who&When] Running inference: {' '.join(inf_cmd)}")
    # Set working dir to script dir, ensure output writes to autoagents_ext/Automated_FA/outputs
    subprocess.run(inf_cmd, cwd=str(wh_root), check=True, env=env_vars)

    # Inference output filename fixed by inference.py: outputs/<method>_<model>_(alg|handcrafted).txt
    handcrafted_suffix = "_handcrafted" if is_handcrafted else "_alg_generated"
    eval_in_txt = wh_root / "outputs" / f"{method}_{model.replace('/', '_')}{handcrafted_suffix}.txt"
    if not eval_in_txt.exists():
        raise FileNotFoundError(f"Who&When inference output not found: {eval_in_txt}")

    # 2) Run evaluate.py (reads previous step's text file for parsing/scoring)
    ev_cmd = [
        sys.executable, str(ev_py),
        "--data_path", dataset_path,
        "--eval_file", str(eval_in_txt),
    ]
    print(f"[Who&When] Running evaluate: {' '.join(ev_cmd)}")
    subprocess.run(ev_cmd, cwd=str(wh_root), check=True, env=env_vars)

    print(f"[Who&When] Done. Eval log: {eval_in_txt}")
    return eval_in_txt
def _downgrade_generics(code: str) -> str:

    if not isinstance(code, str):
        return code
    try:
        import re as _re
        out = code
        repls = {
            "list[": "List[",
            "tuple[": "Tuple[",
            "dict[": "Dict[",
            "set[": "Set[",
        }
        for k, v in repls.items():
            out = _re.sub(rf"\\b{k}", v, out)

        needed_all = ["List", "Tuple", "Dict", "Set"]
        missing: list[str] = []
        for t in needed_all:
            if f"{t}[" in out:
                if not _re.search(rf"^from\\s+typing\\s+import\\s+.*\\b{t}\\b", out, _re.M):
                    missing.append(t)
        if missing:
            needed_line = ", ".join(sorted(set(missing)))
            out = f"from typing import {needed_line}\n\n" + out
        return out
    except Exception:
        return code



def _load_problems(path: str) -> Dict[str, Dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Problems file not found: {p}")
    problems: Dict[str, Dict[str, Any]] = {}
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    tid = str(obj.get("task_id") or obj.get("task") or obj.get("id") or "").strip()
                    if tid:
                        problems[tid] = obj
                except Exception:
                    continue
    else:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            problems = {str(k): v for k, v in data.items()}
        elif isinstance(data, list):
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                tid = str(obj.get("task_id") or obj.get("task") or obj.get("id") or "").strip()
                if tid:
                    problems[tid] = obj
        else:
            raise ValueError("Unsupported problems JSON structure")
    return problems


class Planner(Role):
    name: str = "Planner"
    profile: str = "Planner"
    goal: str = "Plan steps to solve the programming task"
    constraints: list[str] = []

    async def _act(self) -> str:
        # Minimal planner: just acknowledge planning one step
        return "## Execution Plan\n1. Implement the function as specified.\n##"

    async def run(self, message=None):
        from autoagents.system.schema import Message as _Msg
        text = await self._act()
        if getattr(self, "_rc", None) and getattr(self._rc, "env", None):
            msg = _Msg(content=text, role=self.profile)
            await self._rc.env.publish_message(msg)
            return msg
        return _Msg(content=text, role=self.profile)


class Coder(Role):
    name: str = "Coder"
    profile: str = "Coder"
    goal: str = "Write Python implementation for the given function signature"
    constraints: list[str] = []

    async def _act(self) -> str:
        # Pull latest problem prompt from env (assumption: problem stored in env.memory or env.new_roles_args not needed)
        # In this streamlined setup, we expect user to send the prompt in the message before calling roles.
        # Here we simply echo a reminder; actual message to model is sent via provider in examples/humaneval_generate.py.
        # For this official eval runner, we'll craft a direct one-shot call using Environment logic.
        return "```python\n# solution will be provided by LLM in generate step\n```"


class Tester(Role):
    name: str = "Tester"
    profile: str = "Tester"
    goal: str = "Run official assertions for the problem"
    constraints: list[str] = []

    async def _act(self) -> str:
        return "Running official tester..."

    async def run(self, message=None):
        from autoagents.system.schema import Message as _Msg
        text = await self._act()
        if getattr(self, "_rc", None) and getattr(self._rc, "env", None):
            msg = _Msg(content=text, role=self.profile)
            await self._rc.env.publish_message(msg)
            return msg
        return _Msg(content=text, role=self.profile)


class CoderLLM(Role):

    name: str = "Coder"
    profile: str = "Coder"
    goal: str = "Write Python implementation for the given function signature"
    constraints: list[str] = []

    def __init__(self, prompt: str, coder_params: dict | None = None, topology: str = "hier"):
        super().__init__()
        self._prompt = str(prompt or "")
        self._coder_params = dict(coder_params or {})
        self._topology = topology  # "linear" | "flat" | "hier"
        self._env_ref = None  # Environment reference, bound at run time

    def _build_user_content(self, env_obj) -> str:
 
        base = self._prompt
        if not env_obj:
            return base

        plan = (env_obj._latest_role_message_text("Planner") or "").strip()
        test = (env_obj._latest_role_message_text("Tester") or "").strip()
        fail = (getattr(env_obj, "fail_summary", "") or "").strip()

        if self._topology == "linear":
            # Linear: only upstream (Plan), no test feedback
            return base + (f"\n\n[Plan]\n{plan}" if plan else "")

        if self._topology == "flat":
            # Flat: mutual feedback, Plan + test summary (or fail_summary)
            buf = base
            if plan:
                buf += f"\n\n[Plan]\n{plan}"
            if fail:
                buf += f"\n\n[Tester Summary]\n{fail}"
            elif test and "PASS" not in test:
                buf += f"\n\n[Tester]\n{test}"
            return buf


        buf = base
        if plan:
            buf += f"\n\n[Manager Instruction]\n{plan}"
        if fail:
            buf += f"\n\n[Diagnosis Summary]\n{fail}"
        return buf

    async def _act(self) -> str:
        from autoagents.system.provider.llm_api import LLMAPI
        from autoagents.environment import Environment as EnvUtil
        llm = LLMAPI()
        coder_sys = "You are a Python coding expert. Only output ONE python fenced code block with the final solution and nothing else. For Python 3.8 compatibility, do NOT use list[...]/tuple[...]/dict[...]/set[...]. If you add generics, use typing.List/Tuple/Dict/Set and include from typing import .... Keep the exact function signature."
        
        # Track tokens/latency for this LLM call
        task_id = "UNKNOWN"; round_idx = None; env_obj = None
        try:
            if hasattr(self, "_rc") and getattr(self._rc, "env", None):
                env_obj = self._rc.env
                task_id = getattr(env_obj, "task_id", task_id)
                round_idx = getattr(env_obj, "current_round", None)
            elif self._env_ref:
                env_obj = self._env_ref
                task_id = getattr(env_obj, "task_id", task_id)
                round_idx = getattr(env_obj, "current_round", None)
        except Exception:
            pass
        
        # Build user content based on topology (with varying context levels)
        user_content = self._build_user_content(env_obj)
        
        messages = [
            {"role": "system", "content": coder_sys},
            {"role": "user", "content": user_content},
        ]
        async with llm.token_meter(task_id=task_id, round_idx=round_idx):
            text = await llm.acompletion_text(messages, stream=False, gen_params=self._coder_params)
        code = EnvUtil._extract_last_python_block(text) or text.strip()
        code = _downgrade_generics(code)
        return f"```python\n{code.strip()}\n```"

    async def run(self, message=None):
 
        from autoagents.system.schema import Message as _Msg
        text = await self._act()
        if getattr(self, "_rc", None) and getattr(self._rc, "env", None):
            msg = _Msg(content=text, role=self.profile)
            await self._rc.env.publish_message(msg)
            return msg
        return _Msg(content=text, role=self.profile)

async def _run_sequential_official_local(env: Environment, problem: Dict[str, Any], *, max_round: int, timeout_sec: int, coder_params: Dict[str, Any], fail_patience: int, topology: str = "hier"):

    # Roles extracted from env (eval_one_mas_env already called add_roles)
    planner = env.roles.get("Planner")
    coder = env.roles.get("Coder")
    tester_action = OfficialTesterAction(problem)
    
    # Linear topology: force single round (no iteration)
    effective_max_round = 1 if topology == "linear" else max_round
    # Bind env reference to coder so _build_user_content can access context
    if hasattr(coder, "_env_ref"):
        coder._env_ref = env

    # Guard: ensure required roles exist
    if planner is None or coder is None:
        raise RuntimeError("Planner/Coder role not initialized in environment")

    failure_hint: str | None = None
    consecutive_fail = 0
    for r in range(1, effective_max_round + 1):
        env.current_round = r

        # 1) Planner produces plan
        try:
            plan_msg = await planner._act()
            await env.publish_message(Message(content=plan_msg, role="Planner"))
        except Exception:
            pass

        # 2) CoderLLM produces code (internally calls provider + hook injection/auditing)
        try:
            await coder.run()
        except Exception:
            # Fallback: single LLM call (rarely triggered), track tokens
            from autoagents.environment import Environment as EnvUtil
            llm = LLMAPI()
            prompt = str(problem.get("prompt") or "")
            if failure_hint:
                prompt += ("\n\nPreviously failed details, fix them strictly: " + failure_hint)
            messages = [
                {"role": "system", "content": "You are a Python coding expert. Only output ONE python fenced code block with the final solution and nothing else. For Python 3.8 compatibility, do NOT use list[...]/tuple[...]/dict[...]/set[...]. If you add generics, use typing.List/Tuple/Dict/Set and include from typing import .... Keep the exact function signature."},
                {"role": "user", "content": prompt},
            ]
            async with llm.token_meter(task_id=getattr(env, "task_id", "UNKNOWN"), round_idx=r):
                text = await llm.acompletion_text(messages, stream=False, gen_params=coder_params)
            await env.publish_message(Message(content=text, role="Coder"))

        # 3) Get latest Coder message code and run official assertions (OfficialTesterAction has fixed reference/inputs)
        from autoagents.environment import Environment as EnvUtil
        coder_msg = env._latest_role_message_text("Coder") or ""
        code = EnvUtil._extract_last_python_block(coder_msg) or coder_msg.strip()
        code = _downgrade_generics(code)
        summary = await tester_action.run({"code": code, "round_idx": r, "timeout": timeout_sec})
        summary_text = str(summary)
        # Set pass/fail state before publishing Tester message so the FTDI hook
        # does not read stale failure state from the previous round.
        if summary_text.strip() == "PASS":
            env.pass_flag = True
            env.fail_summary = None
        else:
            env.pass_flag = False
            env.fail_summary = summary_text[:2000]
        await env.publish_message(Message(content=summary_text, role="Tester"))
        
        # Get separate base/plus test results
        split_results = tester_action.get_split_results()
        env.last_base_pass = split_results.get("base_pass", False)
        env.last_plus_pass = split_results.get("plus_pass", False)
        env.last_base_total = split_results.get("base_total", 0)
        env.last_plus_total = split_results.get("plus_total", 0)
        env.last_base_failed = split_results.get("base_failed", 0)
        env.last_plus_failed = split_results.get("plus_failed", 0)

        if summary_text.strip() == "PASS":
            break
        else:
            failure_hint = env.fail_summary
            consecutive_fail += 1
            if consecutive_fail >= int(fail_patience):
                break

async def eval_one(task_id: str, problem: Dict[str, Any], max_round: int, coder_params: Dict[str, Any], timeout_sec: int, fail_patience: int) -> Dict[str, Any]:
    env = Environment(task_id=task_id)
    # Reset per task: ensure clean dir
    tdir = WORKSPACE_ROOT / task_id
    if tdir.exists():
        import shutil
        shutil.rmtree(tdir, ignore_errors=True)

    # Minimal roles wiring
    planner = Planner()
    coder = Coder()
    tester = Tester()
    env.add_roles([planner, coder, tester])

    # One-shot code generation using the same provider as samples generator
    from autoagents.environment import Environment as EnvUtil
    llm = LLMAPI()
    prompt = problem.get("prompt") or ""
    messages = [
        {"role": "system", "content": "You are a Python coding expert. Only output ONE python fenced code block with the final solution and nothing else. For Python 3.8 compatibility, do NOT use list[...]/tuple[...]/dict[...]/set[...]. If you add generics, use typing.List/Tuple/Dict/Set and include from typing import .... Keep the exact function signature."},
        {"role": "user", "content": str(prompt)},
    ]
    async with llm.token_meter(task_id=task_id, round_idx=1):
        text = await llm.acompletion_text(messages, stream=False, gen_params=coder_params)
    code = EnvUtil._extract_last_python_block(text) or text.strip()
    code = _downgrade_generics(code)

    # Run official tester with per-round isolation and timeout
    action = OfficialTesterAction(problem)
    consecutive_fail = 0
    for r in range(1, max_round + 1):
        summary = await action.run({"code": code, "round_idx": r, "timeout": timeout_sec})
        if str(summary).strip() == "PASS":
            return {"task_id": task_id, "result": "PASS", "round": r}
        consecutive_fail += 1
        if consecutive_fail >= int(fail_patience):
            return {"task_id": task_id, "result": "FAIL", "round": r, "summary": summary[:4000]}
    return {"task_id": task_id, "result": "FAIL", "round": max_round, "summary": "Max rounds reached"}


async def eval_one_mas(task_id: str, problem: Dict[str, Any], max_round: int, coder_params: Dict[str, Any], timeout_sec: int, fail_patience: int) -> Dict[str, Any]:

    env = Environment(task_id=task_id)
    # Reset task dir
    tdir = WORKSPACE_ROOT / task_id
    if tdir.exists():
        import shutil
        shutil.rmtree(tdir, ignore_errors=True)

    # Roles (simple shells; generation is orchestrated here)
    planner = Planner()
    coder = Coder()
    tester = Tester()
    env.add_roles([planner, coder, tester])

    # Ensure hook is wired so injection/auditor/tracing happen at publish_message
    env = instrument_env(
        env,
        task_id=task_id,
        problem=problem,
        auditor_enabled=True,
        inject_enabled=None,
    )

    llm = LLMAPI()
    prompt = str(problem.get("prompt") or "")
    failure_hint: str | None = None
    last_code: str = ""
    consecutive_fail = 0

    for r in range(1, max_round + 1):
        env.current_round = r
        # Planner
        plan_msg = await planner._act()
        await env.publish_message(Message(content=plan_msg, role="Planner"))

        # Coder: build user message (include last failure hint if any)
        coder_sys = "You are a Python coding expert. Only output ONE python fenced code block with the final solution and nothing else. For Python 3.8 compatibility, do NOT use list[...]/tuple[...]/dict[...]/set[...]. If you add generics, use typing.List/Tuple/Dict/Set and include from typing import .... Keep the exact function signature."
        user_content = prompt
        if failure_hint:
            user_content += ("\n\nPreviously failed details, fix them strictly: " + failure_hint)
        messages = [
            {"role": "system", "content": coder_sys},
            {"role": "user", "content": user_content},
        ]
        async with llm.token_meter(task_id=task_id, round_idx=r):
            text = await llm.acompletion_text(messages, stream=False, gen_params=coder_params)
        # Log coder message (hook may inject here)
        await env.publish_message(Message(content=text, role="Coder"))
        # Extract code from the latest Coder message (post-injection)
        from autoagents.environment import Environment as EnvUtil
        coder_msg = env._latest_role_message_text("Coder") or text
        code = EnvUtil._extract_last_python_block(coder_msg) or coder_msg.strip()
        code = _downgrade_generics(code)
        last_code = _downgrade_generics(code or last_code)

        # Tester via official assertions
        action = OfficialTesterAction(problem)
        summary = await action.run({"code": last_code, "round_idx": r, "timeout": timeout_sec})
        summary_text = str(summary)
        if summary_text.strip() == "PASS":
            env.pass_flag = True
            env.fail_summary = None
        else:
            env.pass_flag = False
            env.fail_summary = summary_text[:2000]
        await env.publish_message(Message(content=summary_text, role="Tester"))
        if summary_text.strip() == "PASS":
            return {"task_id": task_id, "result": "PASS", "round": r, "solution": last_code}
        else:
            consecutive_fail += 1
            failure_hint = env.fail_summary
            if consecutive_fail >= int(fail_patience):
                return {"task_id": task_id, "result": "FAIL", "round": r, "solution": last_code, "summary": failure_hint}
    return {"task_id": task_id, "result": "FAIL", "round": max_round, "solution": last_code, "summary": "Max rounds reached"}


async def eval_one_mas_env(task_id: str, problem: Dict[str, Any], max_round: int, coder_params: Dict[str, Any], timeout_sec: int, *, auditor: str = "on", fail_patience: int, topology: str = "hier") -> Dict[str, Any]:
    """Hierarchical repair (env built-in loop): Planner -> CoderLLM -> OfficialTester (early stop/timeout/isolation)
    
    topology controls topology structure:
    - "linear": Linear (Planner -> Coder -> Tester), single round, no iteration
    - "flat": Flat (mutual feedback), multi-round iteration
    - "hier": Hierarchical (Planner summarizes diagnostics, Coder sees summary only), multi-round (default)
    """
    env = Environment(task_id=task_id)
    # Reset task dir
    tdir = WORKSPACE_ROOT / task_id
    if tdir.exists():
        import shutil
        shutil.rmtree(tdir, ignore_errors=True)

    prompt = str(problem.get("prompt") or "")
    # Roles: Planner (shell), CoderLLM (calls LLM directly, topology controls context visibility), Tester (mounts official assertion action)
    planner = Planner()
    coder = CoderLLM(prompt=prompt, coder_params=coder_params, topology=topology)
    tester = Tester()
    tester._init_actions([OfficialTesterAction(problem)])  # env's run_sequential_official will use tester._actions[0]
    env.add_roles([planner, coder, tester])

    # Hook into env: control auditing and injection via switches (injection controlled by AUTOINJECT_PM/AUTOINJECT_ENABLED env vars)
    env = instrument_env(
        env,
        task_id=task_id,
        problem=problem,
        auditor_enabled=(auditor == "on"),  # CLI controls auditor switch
        inject_enabled=None,                  # None = auto-infer from env vars
    )

    # Bootstrap a system message to avoid first-round _observe judging "no new messages" and skipping _act (critical when Planner/Coder have no actions)
    try:
        from autoagents.system.schema import Message as _Msg
        await env.publish_message(_Msg(content="BOOTSTRAP", role="System"))
    except Exception:
        pass

    # Run examples-layer overridden official eval loop, ensuring real test cases each round
    await _run_sequential_official_local(env, problem, max_round=max_round, timeout_sec=timeout_sec, coder_params=coder_params, fail_patience=fail_patience, topology=topology)
    # Get final code (last python code block from Coder's most recent message)
    from autoagents.environment import Environment as EnvUtil
    last_text = env._latest_role_message_text("Coder") or ""
    last_code = EnvUtil._extract_last_python_block(last_text) or last_text.strip()
    res = {
        "task_id": task_id,
        "result": "PASS" if env.pass_flag else "FAIL",
        "round": int(env.current_round or 0),
        "solution": last_code,
        "topology": topology,  # Record topology structure used
        # Separate HumanEval (base) and HumanEval+ (plus) pass statistics
        "base_pass": getattr(env, "last_base_pass", env.pass_flag),
        "plus_pass": getattr(env, "last_plus_pass", env.pass_flag),
        "base_total": getattr(env, "last_base_total", 0),
        "plus_total": getattr(env, "last_plus_total", 0),
        "base_failed": getattr(env, "last_base_failed", 0),
        "plus_failed": getattr(env, "last_plus_failed", 0),
    }
    if not env.pass_flag:
        res["summary"] = (env.fail_summary or "")[:2000]
    return res


async def main_async(args):
    problems = _load_problems(args.problems_json)
    items = list(problems.items())
    if args.limit is not None:
        items = items[: int(args.limit)]

    # Params: Llama 3 8B works better with temperature=0.3 for code generation
    # temperature=0.0 caused more errors like "from typing import float/int"
    coder_params = {"temperature": 0.3, "top_p": 0.95, "max_tokens": 768}
    timeout_sec = 20
    max_round = 12

    # Grouping label for outputs (honor RUN_GROUP_DIR if provided):
    # Final format: <base>-YYMMDD-HH:MM[:SS]
    route = "mas-env" if args.use_mas_env else ("mas" if args.use_mas else "one-shot")
    topology = args.topology  # linear | flat | hier
    pm_env = os.getenv("AUTOINJECT_PM", "")
    pe_env = os.getenv("AUTOINJECT_PE", "")
    et_env = os.getenv("AUTOINJECT_ERROR_TYPE", "")
    inj_on = False
    try:
        inj_on = float(pm_env) > 0.0
    except Exception:
        inj_on = os.getenv("AUTOINJECT_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
    def _truthy(v: str | None) -> bool:
        return str(v or "").strip().lower() in ("1", "true", "yes", "on")
    ts_label = time.strftime("%y%m%d-%H:%M:%S", time.localtime())

    # Helper: check if a label already ends with a timestamp suffix
    import re as _re_ts
    _ts_pat = _re_ts.compile(r"\d{6}-\d{2}:\d{2}(?::\d{2})?$")

    rgd_env_raw = os.getenv("RUN_GROUP_DIR", "")
    rgd_clean = (rgd_env_raw or "").strip()
    if rgd_clean:
        # If caller provides RUN_GROUP_DIR, honor it EXACTLY and derive a label from its basename.
        candidate = Path(rgd_clean)
        label = candidate.name if candidate.name else rgd_clean.strip("/\\")
        group_label = label or "run"
    else:
        # Include topology info in group_label
        if inj_on and pm_env and pe_env and et_env:
            base = f"{route}-{topology}-PM{pm_env}-PE{pe_env}-{et_env}"
        else:
            base = f"{route}-{topology}"
        group_label = f"{base}-{ts_label}"
        run_group_rel = Path("traces") / group_label
        os.environ["RUN_GROUP_DIR"] = str(run_group_rel)

    # Also redirect WORKSPACE_ROOT for extension traces (inject_bridge.py respects env WORKSPACE_ROOT)
    # to a group-specific folder so traces are namespaced per run.
    from autoagents.system.const import PROJECT_ROOT as _PROJ
    os.environ.setdefault("WORKSPACE_ROOT", str((_PROJ / "workspace" / group_label)))

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []
    samples: list[dict] = []

    def has_good_signature(src: str, entry_point: str | None) -> bool:
        if not entry_point:
            return True
        try:
            pat = rf"^\s*def\s+{re.escape(entry_point)}\s*\("
            return re.search(pat, src or "", re.M) is not None
        except Exception:
            return True

    async def worker(tid: str, prob: Dict[str, Any]):
        async with sem:
            try:
                if args.use_mas_env:
                    res = await eval_one_mas_env(tid, prob, max_round=max_round, coder_params=coder_params, timeout_sec=timeout_sec, auditor=args.auditor, fail_patience=args.fail_patience, topology=args.topology)
                elif args.use_mas:
                    res = await eval_one_mas(tid, prob, max_round=max_round, coder_params=coder_params, timeout_sec=timeout_sec, fail_patience=args.fail_patience)
                else:
                    res = await eval_one(tid, prob, max_round=max_round, coder_params=coder_params, timeout_sec=timeout_sec, fail_patience=args.fail_patience)
                results.append(res)
                # Optionally write final code to samples list for EvalPlus
                if args.write_samples:
                    sol = res.get("solution")
                    if not sol:
                        # If no solution returned (including MAS/MAS-ENV), regenerate quickly for samples
                        from autoagents.environment import Environment as EnvUtil
                        llm = LLMAPI()
                        prompt = prob.get("prompt") or ""
                        messages = [
                            {"role": "system", "content": "You are a Python coding expert. Only output ONE python fenced code block with the final solution and nothing else. For Python 3.8 compatibility, do NOT use list[...]/tuple[...]/dict[...]/set[...]. If you add generics, use typing.List/Tuple/Dict/Set and include from typing import .... Keep the exact function signature."},
                            {"role": "user", "content": str(prompt)},
                        ]
                        async with llm.token_meter(task_id=tid, round_idx=1):
                            text = await llm.acompletion_text(messages, stream=False, gen_params=coder_params)
                        code = EnvUtil._extract_last_python_block(text) or text.strip()
                        code = _downgrade_generics(code)
                        sol = code
                    if sol:
                        # Signature check + policy handling
                        entry_point = str(prob.get("entry_point") or "").strip()
                        action = (args.bad_sig_action or "retry").lower()
                        tries = 0
                        max_redraw = max(0, int(args.sig_retries)) if action == "retry" else 0
                        while sol and action == "retry" and not has_good_signature(sol, entry_point) and tries < max_redraw:
                            tries += 1
                            # quick redraw
                            llm = LLMAPI()
                            prompt = prob.get("prompt") or ""
                            messages = [
                                {"role": "system", "content": "You are a Python coding expert. Only output ONE python fenced code block with the final solution and nothing else. For Python 3.8 compatibility, do NOT use list[...]/tuple[...]/dict[...]/set[...]. If you add generics, use typing.List/Tuple/Dict/Set and include from typing import .... Keep the exact function signature."},
                                {"role": "user", "content": str(prompt)},
                            ]
                            async with llm.token_meter(task_id=tid, round_idx=0):
                                text = await llm.acompletion_text(messages, stream=False, gen_params=coder_params)
                            sol = EnvUtil._extract_last_python_block(text) or text.strip()
                            sol = _downgrade_generics(sol)
                        if sol and (has_good_signature(sol, entry_point) or action == "keep"):
                            # Tag source in gen_id (one-shot/mas-script/mas-env)
                            import time, hashlib
                            route = "mas-env" if args.use_mas_env else ("mas-script" if args.use_mas else "one-shot")
                            h8 = hashlib.sha256((sol or "").encode("utf-8")).hexdigest()[:8]
                            gen_id = f"{route}:{int(time.time()*1000)}:{h8}"
                            samples.append({"task_id": tid, "solution": sol, "gen_id": gen_id})
                        elif sol and action == "drop":
                            logger.warning(f"Dropping sample for {tid} due to bad signature (expect '{entry_point}')")
                        elif sol and action == "retry" and not has_good_signature(sol, entry_point):
                            logger.warning(f"No valid signature after retries for {tid}; skipping sample write (expect '{entry_point}')")
            except Exception as e:
                results.append({"task_id": tid, "result": "ERROR", "error": str(e)[:4000]})

    await asyncio.gather(*(worker(tid, prob) for tid, prob in items))

    def _safe_name(s: str) -> str:
        import re as _re
        return _re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

    def _trace_file(task_id: str):
        ws = Path(os.environ.get("WORKSPACE_ROOT", str(WORKSPACE_ROOT)))
        safe = _safe_name(task_id)
        primary = _resolve_run_group_dir_path(ws) / f"{safe}.jsonl"
        fallback = ws / "traces" / f"{safe}.jsonl"
        if primary.exists():
            return primary
        if fallback.exists():
            return fallback
        return None

    def _collect_defense_from_trace(task_id: str) -> dict:
        """Collect Defense metadata from trace file"""
        defense_stats = {
            "defense_tokens": 0,
            "defense_calls": 0,
            "challenger_unsafe": 0,
            "inspector_revised": 0,
        }
        fp = _trace_file(task_id)
        if not fp:
            return defense_stats
        try:
            with fp.open("r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    defense_info = obj.get("defense")
                    if not defense_info or not isinstance(defense_info, dict):
                        continue
                    # Extract defense_tokens
                    defense_tokens_info = defense_info.get("defense_tokens", {})
                    if isinstance(defense_tokens_info, dict):
                        tokens = defense_tokens_info.get("total_tokens", 0)
                        if tokens:
                            defense_stats["defense_tokens"] += tokens
                            defense_stats["defense_calls"] += defense_tokens_info.get("calls", 0)
                    # Count Challenger/Inspector usage
                    if defense_info.get("challenger_result") == "unsafe":
                        defense_stats["challenger_unsafe"] += 1
                    if defense_info.get("inspector_applied"):
                        defense_stats["inspector_revised"] += 1
        except Exception:
            pass
        return defense_stats

    def _collect_repair_from_trace(task_id: str) -> dict:
        """Collect repair metadata from trace file"""
        repair_stats = {
            "t0_applied": 0,
            "t1_applied": 0,
            "t2_applied": 0,
            "repair_tokens": 0,
            "repair_methods": [],
            "error_types": [],  # Use list instead of set to avoid JSON serialization issues
        }
        fp = _trace_file(task_id)
        if not fp:
            return repair_stats
        error_types_set = set()  # Temp set for deduplication
        try:
            with fp.open("r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    repair_info = obj.get("repair")
                    if not repair_info or not isinstance(repair_info, dict):
                        continue
                    stage = repair_info.get("repair_stage", "")
                    if stage == "T0":
                        repair_stats["t0_applied"] += 1
                    elif stage == "T1":
                        repair_stats["t1_applied"] += 1
                    elif stage == "T2":
                        repair_stats["t2_applied"] += 1
                    tokens = repair_info.get("tokens_used", 0)
                    if tokens:
                        repair_stats["repair_tokens"] += tokens
                    method = repair_info.get("repair_method")
                    if method and method not in repair_stats["repair_methods"]:
                        repair_stats["repair_methods"].append(method)
                    for et in repair_info.get("error_types", []):
                        error_types_set.add(et)
                    fail_type = repair_info.get("fail_type")
                    if fail_type:
                        error_types_set.add(fail_type)
        except Exception:
            pass
        repair_stats["error_types"] = sorted(list(error_types_set))
        return repair_stats

    # Get repair history snapshot
    try:
        repair_snapshot_cache = snapshot_repair_history(clear=True)
    except Exception:
        repair_snapshot_cache = {}

    def _collect_repair_from_snapshot(task_id: str) -> dict:
        """Collect repair metadata from memory snapshot (preferred over trace files)"""
        repair_stats = {
            "t0_applied": 0,
            "t1_applied": 0,
            "t2_applied": 0,
            "repair_tokens": 0,
            "repair_methods": [],
            "error_types": [],
        }
        entries = repair_snapshot_cache.get(task_id, [])
        if not entries:
            return repair_stats
        error_types_set = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            stage = entry.get("repair_stage", "")
            if stage == "T0":
                repair_stats["t0_applied"] += 1
            elif stage == "T1":
                repair_stats["t1_applied"] += 1
            elif stage == "T2":
                repair_stats["t2_applied"] += 1
            tokens = entry.get("tokens_used", 0)
            if tokens:
                repair_stats["repair_tokens"] += tokens
            method = entry.get("repair_method")
            if method and method not in repair_stats["repair_methods"]:
                repair_stats["repair_methods"].append(method)
            for et in entry.get("error_types", []):
                error_types_set.add(et)
            fail_type = entry.get("fail_type")
            if fail_type:
                error_types_set.add(fail_type)
        repair_stats["error_types"] = sorted(list(error_types_set))
        return repair_stats

    
    # Defense token stats
    total_defense_tokens = 0

    # Aggregate repair stats
    total_t0_applied = 0
    total_t1_applied = 0
    total_t2_applied = 0
    total_repair_tokens = 0

    traced_tasks: set[str] = set()
    for r in results:
        tid = str(r.get("task_id", ""))
        if not tid:
            continue
        if tid not in traced_tasks:
            traced_tasks.add(tid)
        r.setdefault("interventions", {})

        # Add repair metadata (prefer memory snapshot, fall back to trace file)
        repair_stats = _collect_repair_from_snapshot(tid)
        if repair_stats["t0_applied"] == 0 and repair_stats["t1_applied"] == 0 and repair_stats["t2_applied"] == 0:
            repair_stats = _collect_repair_from_trace(tid)
        r["interventions"]["t0_applied"] = repair_stats["t0_applied"]
        r["interventions"]["t1_applied"] = repair_stats["t1_applied"]
        r["interventions"]["t2_applied"] = repair_stats["t2_applied"]
        r["interventions"]["repair_tokens"] = repair_stats["repair_tokens"]
        r["interventions"]["repair_methods"] = repair_stats["repair_methods"]
        r["interventions"]["error_types"] = repair_stats["error_types"]
        
        # Add Defense metadata
        defense_stats = _collect_defense_from_trace(tid)
        r["interventions"]["defense_tokens"] = defense_stats["defense_tokens"]
        r["interventions"]["defense_calls"] = defense_stats["defense_calls"]
        r["interventions"]["challenger_unsafe"] = defense_stats["challenger_unsafe"]
        r["interventions"]["inspector_revised"] = defense_stats["inspector_revised"]
        
        # Determine if recovered through repair
        is_pass = r.get("result") == "PASS"
        round_num = int(r.get("round", 0) or 0)
        first_round_pass = r.get("first_round_pass", False)
        has_repair = (repair_stats["t0_applied"] > 0 or repair_stats["t1_applied"] > 0 or repair_stats["t2_applied"] > 0)
        # If finally passed, not first-round pass, and repair was applied -> recovered
        recovered = is_pass and round_num > 1 and has_repair
        r["interventions"]["recovered"] = recovered
        
        total_t0_applied += repair_stats["t0_applied"]
        total_t1_applied += repair_stats["t1_applied"]
        total_t2_applied += repair_stats["t2_applied"]
        total_repair_tokens += repair_stats["repair_tokens"]
        total_defense_tokens += defense_stats["defense_tokens"]

    # Always write results to grouped dir (naming consistent with humaneval_samples), also keep a top-level results.jsonl copy for compatibility
    grouped_dir = WORKSPACE_ROOT / "humaneval_official" / group_label
    grouped_dir.mkdir(parents=True, exist_ok=True)
    grouped_path = grouped_dir / "results.jsonl"
    # Mark whether first-round PASS for quick external aggregation
    for _r in results:
        try:
            _r.setdefault("first_round_pass", bool(_r.get("result") == "PASS" and int(_r.get("round", 0)) == 1))
        except Exception:
            _r["first_round_pass"] = False
    with grouped_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved results to: {grouped_path}")

    # Compat: also write latest results to top-level for legacy path (workspace/humaneval_official/results.jsonl)
    flat_dir = WORKSPACE_ROOT / "humaneval_official"
    flat_dir.mkdir(parents=True, exist_ok=True)
    flat_path = flat_dir / "results.jsonl"
    try:
        with flat_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Also wrote a convenience copy to: {flat_path}")
    except Exception as _e:
        print(f"[warn] failed to write flat results copy: {flat_path}: {_e}")
    if args.write_samples:
        smp_dir = WORKSPACE_ROOT / "humaneval_samples" / group_label
        smp_dir.mkdir(parents=True, exist_ok=True)
        smp_path = smp_dir / "samples.jsonl"
        with smp_path.open("w", encoding="utf-8") as sf:
            for s in samples:
                sf.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples)} samples to: {smp_path}")

    # ============ Copy injection manifests and samples to humaneval_official results dir ============
    # Only copy injection manifests when injection is enabled, avoid copying stale attack manifests during healthy baseline runs
    import shutil
    inject_enabled = os.environ.get("AUTOINJECT_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
    try:
        # Only copy probex_manifest.jsonl when injection is enabled (global injection manifest)
        if inject_enabled:
            manifest_src = WORKSPACE_ROOT / "injections" / "probex_manifest.jsonl"
            if manifest_src.exists():
                manifest_dst = grouped_dir / "probex_manifest.jsonl"
                shutil.copy2(manifest_src, manifest_dst)
                print(f"Copied injection manifest to: {manifest_dst}")
            
            # Copy per-task injection manifests for this group (if exists)
            injection_group_dir = WORKSPACE_ROOT / group_label / "injections"
            if injection_group_dir.exists() and injection_group_dir.is_dir():
                dst_injection_dir = grouped_dir / "injections"
                if dst_injection_dir.exists():
                    shutil.rmtree(dst_injection_dir)
                shutil.copytree(injection_group_dir, dst_injection_dir)
                print(f"Copied per-task injection manifests to: {dst_injection_dir}")
        
        # samples.jsonl always copied (regardless of injection setting)
        samples_src = WORKSPACE_ROOT / "humaneval_samples" / group_label / "samples.jsonl"
        if samples_src.exists():
            samples_dst = grouped_dir / "samples.jsonl"
            shutil.copy2(samples_src, samples_dst)
            print(f"Copied samples to: {samples_dst}")
    except Exception as _copy_err:
        print(f"[warn] failed to copy injection/samples artifacts: {_copy_err}")

    # Optional tail flow: trigger Who&When (or lightweight internal) attribution for enhancement when FAILs exist
    # Two-level switch: --who-when-dataset (provide datasource for official pipeline) and --attrib-hybrid=on (lightweight internal attribution even without dataset)
    try:
        any_fail = any(str(r.get("result", "")).upper() in ("FAIL", "ERROR") for r in results)
        if any_fail and (args.attrib_hybrid == "on" or args.who_when_dataset):
            # 1) Try official pipeline first (if dataset path provided)
            if args.who_when_dataset:
                try:
                    # all-at-once better at "agent-level"; step-by-step better at "step-level"
                    run_who_when_pipeline(
                        dataset_path=args.who_when_dataset,
                        method="all_at_once",
                        model=args.who_when_model,
                        is_handcrafted=bool(args.who_when_handcrafted),
                        azure_api_key=args.azure_key,
                        azure_endpoint=args.azure_endpoint,
                        api_version=args.azure_api_version,
                    )
                    run_who_when_pipeline(
                        dataset_path=args.who_when_dataset,
                        method="step_by_step",
                        model=args.who_when_model,
                        is_handcrafted=bool(args.who_when_handcrafted),
                        azure_api_key=args.azure_key,
                        azure_endpoint=args.azure_endpoint,
                        api_version=args.azure_api_version,
                    )
                except Exception as e:
                    print(f"[Who&When] hybrid pipeline failed (non-fatal): {e}")

            # 2) Generate lightweight "cause/step" snippets for failed tasks in this run, write to grouped dir,
            def _trace_file_for(task: str) -> Path:
                ws = Path(os.environ.get("WORKSPACE_ROOT", str(WORKSPACE_ROOT)))
                primary = _resolve_run_group_dir_path(ws) / f"{task}.jsonl"
                if primary.exists():
                    return primary
                return ws / "traces" / f"{task}.jsonl"

            def _collect_step_window(task: str, final_round: int) -> str:
                # Get text window of Planner/Coder/Tester around this round
                tf = _trace_file_for(task)
                if not tf.exists():
                    return ""
                texts = []
                try:
                    with tf.open("r", encoding="utf-8", errors="ignore") as f:
                        for ln in f:
                            try:
                                obj = json.loads(ln)
                            except Exception:
                                continue
                            rd = int(obj.get("round", 0) or 0)
                            if rd in (final_round - 1, final_round):
                                role = obj.get("role")
                                if role in ("Planner", "Coder", "Tester"):
                                    texts.append(str(obj.get("raw_text") or obj.get("tester_summary") or ""))
                except Exception:
                    return ""
                return "\n".join([t for t in texts if t])[:2000]

            attrib_out = grouped_dir / "attribution_snippets.jsonl"
            with attrib_out.open("w", encoding="utf-8") as af:
                for r in results:
                    if str(r.get("result", "")).upper() not in ("FAIL", "ERROR"):
                        continue
                    tid = str(r.get("task_id", ""))
                    try:
                        final_rd = int(r.get("round", 0) or 0)
                    except Exception:
                        final_rd = 0
                    # Cause: use latest Tester summary + auditor.fail_type (if available)
                    cause_parts = []
                    # Extract latest Tester summary from trace
                    tf = _trace_file_for(tid)
                    last_tester = ""
                    if tf.exists():
                        try:
                            with tf.open("rb") as f:
                                f.seek(0, os.SEEK_END)
                                size = f.tell(); back = min(65536, size); f.seek(max(0, size - back))
                                chunk = f.read().decode("utf-8", errors="ignore")
                            for ln in reversed([l for l in chunk.splitlines() if l.strip()]):
                                try:
                                    o = json.loads(ln)
                                except Exception:
                                    continue
                                if o.get("role") == "Tester":
                                    last_tester = str(o.get("tester_summary") or o.get("raw_text") or ""); break
                        except Exception:
                            pass
                    cause_parts.append(last_tester)
                    # Extract latest auditor.fail_type from trace (if any)
                    fail_type = None
                    if tf.exists():
                        try:
                            with tf.open("r", encoding="utf-8", errors="ignore") as f:
                                for ln in f:
                                    try:
                                        o = json.loads(ln)
                                    except Exception:
                                        continue
                                    diag = o.get("diagnosis") or o.get("auditor") or {}
                                    if isinstance(diag, dict) and diag.get("fail_type"):
                                        fail_type = diag.get("fail_type")
                        except Exception:
                            pass
                    if fail_type:
                        cause_parts.append(f"fail_type={fail_type}")
                    step_window = _collect_step_window(tid, final_rd)
                    rec = {"task_id": tid, "cause_text": ("\n".join([p for p in cause_parts if p]).strip()), "step_window": step_window}
                    af.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[Attribution] Wrote snippets for failed tasks: {attrib_out}")
        else:
            print("[Attribution] Skipped: no FAIL or attribution disabled.")
    except Exception as _e:
        print(f"[warn] attribution snippet generation failed: {_e}")


    try:
        evs_raw = get_token_events()
        events_out = grouped_dir / "intervention_events.jsonl"
        with events_out.open("w", encoding="utf-8") as ef:
            for e in evs_raw:
                rec = dict(e)
                tid = str(rec.get("task_id", ""))
                rd = rec.get("round")
                if rd is None:
                    rd = rec.get("round_idx")
                try:
                    rd_int = int(rd)
                except Exception:
                    rd_int = 0
                ef.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote intervention events to: {events_out}")
    except Exception as _e:
        print(f"[warn] failed to export intervention events: {_e}")

    try:
        evs = get_token_events()
        total_prompt = sum(e.get("prompt_tokens", 0) for e in evs)
        total_completion = sum(e.get("completion_tokens", 0) for e in evs)
        total_mas_tokens = total_prompt + total_completion  # MAS system tokens
        total_tokens = total_mas_tokens + total_repair_tokens + total_defense_tokens  # Total tokens = MAS + Repair API + Defense API
        tasks_total = len(results)
        passes_first = sum(1 for r in results if r.get("result") == "PASS" and int(r.get("round", 0)) == 1)
        passes_total = sum(1 for r in results if r.get("result") == "PASS")
        recovered = sum(1 for r in results if r.get("result") == "PASS" and int(r.get("round", 0)) > 1)
        
        # Count tasks recovered through repair
        recovered_with_repair = sum(1 for r in results if r.get("interventions", {}).get("recovered", False))
        
        failed_first_pool = tasks_total - passes_first
        recovery_ratio = (recovered / failed_first_pool) if failed_first_pool > 0 else 0.0
        repair_recovery_ratio = (recovered_with_repair / failed_first_pool) if failed_first_pool > 0 else 0.0
        

        pass_at_1 = (passes_total / tasks_total) if tasks_total else 0.0
        
        # First-round pass rate (Zero-shot baseline, for comparison)
        first_round_pass_rate = (passes_first / tasks_total) if tasks_total else 0.0
        
        avg_latency = (sum(e.get("latency_sec", 0.0) for e in evs) / len(evs)) if evs else 0.0
        tokens_per_1pct_recovery = None
        if recovered > 0 and tasks_total > 0:
            recovered_pct = (recovered / tasks_total) * 100.0
            if recovered_pct > 0:
                tokens_per_1pct_recovery = total_tokens / recovered_pct
        for r in results:
            try:
                if r.get("result") != "PASS":
                    continue
                rd = int(r.get("round", 0) or 0)
                if rd <= 1:
                    continue
                tid = str(r.get("task_id", ""))
            except Exception:
                continue
        agg_record = {
            "_meta": "aggregate_metrics",
            "group_label": group_label,
            "topology": args.topology,  # Record topology: linear | flat | hier
            "tasks_total": tasks_total,
            "passes_total": passes_total,
            "passes_first_round": passes_first,
            "recovered_tasks": recovered,
            "recovered_with_repair": recovered_with_repair,
            "pass@1": round(pass_at_1, 4),  # Final collaborative pass rate (paper definition)
            "first_round_pass_rate": round(first_round_pass_rate, 4),  # First-round pass rate (Zero-shot baseline)
            "recovery_ratio": round(recovery_ratio, 4),
            "repair_recovery_ratio": round(repair_recovery_ratio, 4),
            "tokens_per_1pct_recovery": (round(tokens_per_1pct_recovery, 2) if tokens_per_1pct_recovery is not None else None),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_mas_tokens": total_mas_tokens,
            "total_repair_tokens": total_repair_tokens,
            "total_defense_tokens": total_defense_tokens,
            "total_t0_applied": total_t0_applied,
            "total_t1_applied": total_t1_applied,
            "total_t2_applied": total_t2_applied,
            "total_tokens": total_tokens,
            "avg_latency_sec": round(avg_latency, 4),
            "token_events": len(evs),
            "tokens_per_task": (round(total_tokens / tasks_total, 2) if tasks_total else 0.0),
            "tokens_per_pass": (round(total_tokens / passes_total, 2) if passes_total else None),
            "timestamp": time.time(),
            "route": "mas-env" if args.use_mas_env else ("mas" if args.use_mas else "one-shot"),
        }
        try:
            with grouped_path.open("a", encoding="utf-8") as gf:
                gf.write(json.dumps(agg_record, ensure_ascii=False) + "\n")
        except Exception as _e:
            print(f"[warn] append aggregate grouped failed: {_e}")
        try:
            with flat_path.open("a", encoding="utf-8") as ff:
                ff.write(json.dumps(agg_record, ensure_ascii=False) + "\n")
        except Exception as _e:
            print(f"[warn] append aggregate flat failed: {_e}")
        print("[Aggregate] pass@1=%.3f recovery_ratio=%.3f total_tokens=%d (MAS=%d + Repair=%d + Defense=%d) recovered=%d" % (agg_record["pass@1"], agg_record["recovery_ratio"], total_tokens, total_mas_tokens, total_repair_tokens, total_defense_tokens, recovered))
    except Exception as e:
        print(f"[warn] aggregate metrics failed: {e}")

    # ---- Cross-run Recovery Ratio & Tokens per 1% Recovery (RR / TPR) ----

    try:
        eps = float(os.getenv("CROSS_RUN_EPS", "1e-9"))
        # Parse current group_label to infer base prefix and timestamp
        parts = group_label.split('-')
        if len(parts) >= 2:
            ts_candidate = parts[-1]
            # Rough timestamp check: contains digits and length >= 8
            if re.match(r"\d{8,}", ts_candidate):
                timestamp = ts_candidate
                # Mode could be attack or attack+imm or healthy (may not exist yet)
                mode_part = parts[-2]
                base_prefix = '-'.join(parts[:-2])  # Strip mode + timestamp
                modes_needed = ["healthy", "attack", "attack+imm"]
                sibling_metrics = {}
                run_root = WORKSPACE_ROOT / "humaneval_official"
                for m in modes_needed:
                    label = f"{base_prefix}-{m}-{timestamp}"
                    path = run_root / label / "results.jsonl"
                    if path.exists():
                        # Read last aggregate_metrics record
                        try:
                            with path.open("r", encoding="utf-8") as rf:
                                agg_line = None
                                for ln in rf:
                                    ln = ln.strip()
                                    if not ln:
                                        continue
                                    try:
                                        obj = json.loads(ln)
                                    except Exception:
                                        continue
                                    if obj.get("_meta") == "aggregate_metrics":
                                        agg_line = obj
                                if agg_line:
                                    sibling_metrics[m] = agg_line
                        except Exception as _e:
                            print(f"[warn] read sibling metrics failed: {path}: {_e}")
                if all(k in sibling_metrics for k in ("attack", "attack+imm")) and ("healthy" in sibling_metrics):
                    acc_attack = sibling_metrics["attack"].get("passes_total", 0) / max(1, sibling_metrics["attack"].get("tasks_total", 1))
                    acc_attack_imm = sibling_metrics["attack+imm"].get("passes_total", 0) / max(1, sibling_metrics["attack+imm"].get("tasks_total", 1))
                    acc_healthy = sibling_metrics["healthy"].get("passes_total", 0) / max(1, sibling_metrics["healthy"].get("tasks_total", 1))
                    rr_den = max(acc_healthy - acc_attack, eps)
                    rr = (acc_attack_imm - acc_attack) / rr_den * 100.0
                    t_attack = sibling_metrics["attack"].get("total_tokens", 0)
                    t_attack_imm = sibling_metrics["attack+imm"].get("total_tokens", 0)
                    tpr_den = max(rr, eps)
                    tpr = (t_attack_imm - t_attack) / tpr_den if tpr_den > 0 else None
                    cross_record = {
                        "_meta": "cross_run_recovery",
                        "base_prefix": base_prefix,
                        "timestamp": timestamp,
                        "healthy_group": f"{base_prefix}-healthy-{timestamp}",
                        "attack_group": f"{base_prefix}-attack-{timestamp}",
                        "attack_imm_group": f"{base_prefix}-attack+imm-{timestamp}",
                        "acc_healthy": round(acc_healthy, 4),
                        "acc_attack": round(acc_attack, 4),
                        "acc_attack_imm": round(acc_attack_imm, 4),
                        "RR_percent": round(rr, 4),
                        "TPR_tokens_per_pct": (round(tpr, 4) if tpr is not None else None),
                        "epsilon": eps,
                        "tokens_attack": t_attack,
                        "tokens_attack_imm": t_attack_imm,
                        # Copy current run's aggregate metrics for single-file analysis
                        "current_run_aggregate": agg_record if 'agg_record' in locals() else None,
                    }
                    # Write to current group dir's cross_run_metrics.jsonl (append)
                    cross_path = grouped_dir / "cross_run_metrics.jsonl"
                    with cross_path.open("a", encoding="utf-8") as cf:
                        cf.write(json.dumps(cross_record, ensure_ascii=False) + "\n")
                    _msg = (
                        f"[CrossRun] RR={cross_record['RR_percent']:.3f}% "
                        f"TPR={cross_record['TPR_tokens_per_pct']} written to {cross_path} "
                        f"(healthy={cross_record['acc_healthy']:.3f} "
                        f"attack={cross_record['acc_attack']:.3f} "
                        f"attack+imm={cross_record['acc_attack_imm']:.3f})"
                    )
                    print(_msg)
                else:
                    print("[CrossRun] Sibling groups for full RR/TPR not all present (need healthy, attack, attack+imm). Skipped.")
        else:
            print("[CrossRun] group_label format insufficient for RR/TPR inference. Skipped.")
    except Exception as e:
        print(f"[warn] cross-run RR/TPR computation failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="HumanEval(+) official evaluation via AutoAgents")
    ap.add_argument("--problems-json", type=str, required=True, help="Path to local HumanEval(+) problems JSON/JSONL file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--use-mas", action="store_true", help="Use hierarchical Planner->Coder->Tester loop with early stop")
    ap.add_argument("--use-mas-env", action="store_true", help="Use environment built-in Planner->Coder->OfficialTester loop (run_sequential_official)")
    ap.add_argument("--write-samples", action="store_true", help="Write final code into workspace/humaneval_samples/samples.jsonl for evalplus")
    ap.add_argument("--group-results", action="store_true", help="Write results.jsonl under grouped directory (humaneval_official/<RUN_GROUP_DIR>/results.jsonl)")
    ap.add_argument("--auditor", choices=["on", "off"], default="on", help="Enable/disable online Auditor scoring and write into trace meta.")
    ap.add_argument("--bad-sig-action", type=str, choices=["retry", "drop", "keep"], default="retry", help="When writing samples, how to handle wrong entry function name: retry (default), drop, or keep.")
    ap.add_argument("--sig-retries", type=int, default=1, help="Number of additional redraws when --bad-sig-action=retry.")
    ap.add_argument("--fail-patience", type=int, default=5, help="Consecutive FAIL threshold for early stop (default 5).")
    ap.add_argument("--topology", type=str, choices=["linear", "flat", "hier"], default="hier",
                    help="MAS topology structure: linear (single-round, no iteration), flat (mutual feedback), hier (hierarchical with diagnosis summary). Default: hier.")

    # Who&When offline attribution evaluation (minimal integration) args:
    ap.add_argument("--who-when-dataset", type=str, default=None,
                    help="Who&When GT dataset dir (with *.json GT files). If provided, auto-runs failure attribution evaluation after this run.")
    ap.add_argument("--who-when-method", type=str, default="all_at_once",
                    choices=["all_at_once", "step_by_step", "binary_search"],
                    help="Who&When method: all_at_once / step_by_step / binary_search.")
    ap.add_argument("--who-when-model", type=str, default="gpt-4o",
                    help="Who&When model alias, must align with autoagents_ext/Automated_FA/inference.py choices.")
    ap.add_argument("--who-when-handcrafted", action="store_true",
                    help="Enable if dataset is Hand-Crafted (affects output log filename suffix).")
    # Hybrid attribution and immunity retrieval augmentation control:
    ap.add_argument("--attrib-hybrid", choices=["on", "off"], default=os.getenv("ATTRIBUTION_HYBRID", "off"),
                    help="Generate lightweight Who&When-style attribution snippets after run (FAIL tasks only). Default off, controllable via ATTRIBUTION_HYBRID env var.")
    # Pass-through OpenAI-compatible / Azure auth: inference.py determines Azure vs OpenAI-compatible by endpoint, reuses chat.completions API
    ap.add_argument("--azure-key", type=str, default=os.environ.get("AZURE_OPENAI_API_KEY"))
    ap.add_argument("--azure-endpoint", type=str, default=(os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")))
    ap.add_argument("--azure-api-version", type=str, default="2024-08-01-preview")
    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
