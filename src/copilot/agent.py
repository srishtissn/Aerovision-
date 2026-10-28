"""
Tool-calling maintenance agent.

* With a Groq key: Llama 3.3 decides which tools to call (OpenAI-style function
  calling), we execute them, feed results back, and loop until it answers.
* Without a key: a small rule-based router calls the same tools and composes a
  deterministic answer, so the demo still works offline.

Guardrail: the final answer is verified against everything the tools returned
in this turn. Numbers, intervals, locations or defect types that no tool
returned trigger one repair round; if it is still unsupported the answer is
shown with an explicit warning rather than silently trusted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm import LLM
from .tools import CopilotTools
from .verifier import Verification, verify_answer_text

SYSTEM = """You are AeroVision Maintenance Copilot, an assistant for aircraft skin inspection.
You can call tools that run the real detector, severity scoring, scan comparison, RUL model, fusion, guidance retrieval and report generation.

RULES
- Never guess facts. Get them from tools. If a needed scan or engine is missing, say so.
- Resolve scans with list_scans when unsure; 'latest' and 'previous' are valid scan ids.
- For ANY question about safety, airworthiness, release or 'safe to fly', call assess_flight_readiness first (include previous_scan_id when the user mentions comparing or the last scan).
- You only advise. Never say an aircraft is certified safe; the release decision belongs to a certified person. State the rule-based recommendation and its reasons.
- Use only numbers, locations, defect types and intervals that appear in tool outputs. Express confidence as percentages.
- When you use guidance from retrieve_guidance, cite the chunk id in square brackets, e.g. [crack-high].
- Be concise: short paragraphs or a few bullets."""


@dataclass
class AgentResult:
    answer: str
    steps: List[Dict[str, Any]] = field(default_factory=list)   # tool trace
    verification: Optional[Verification] = None
    warning: Optional[str] = None
    mode: str = "llm"


def _short(obj: Any, limit: int = 7000) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + '..."truncated"'


class CopilotAgent:
    def __init__(self, tools: CopilotTools, llm: Optional[LLM] = None, max_steps: int = 6):
        self.tools = tools
        self.llm = llm
        self.max_steps = max_steps

    # ------------------------------------------------------------ public
    def run(self, user_message: str, history: Optional[List[Dict[str, str]]] = None) -> AgentResult:
        if self.llm is None:
            return self._run_rules(user_message)
        try:
            return self._run_llm(user_message, history or [])
        except Exception as e:  # network / rate limit / API error -> keep the demo alive
            res = self._run_rules(user_message)
            res.warning = f"LLM unavailable ({str(e)[:120]}); answered with the offline rule-based router."
            return res

    # --------------------------------------------------------------- LLM
    def _run_llm(self, user_message: str, history: List[Dict[str, str]]) -> AgentResult:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
        messages += [m for m in history[-8:] if m["role"] in ("user", "assistant")]
        messages.append({"role": "user", "content": user_message})
        steps: List[Dict[str, Any]] = []
        outputs: List[Any] = []
        answer = ""
        repaired = False
        schemas = self.tools.schemas()

        for _ in range(self.max_steps):
            resp = self.llm.chat(messages, tools=schemas)
            calls = resp.get("tool_calls") or []
            if calls:
                messages.append({
                    "role": "assistant", "content": resp.get("content") or "",
                    "tool_calls": [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                   for c in calls]})
                for c in calls:
                    result = self.tools.call(c["name"], c["arguments"])
                    steps.append({"tool": c["name"], "arguments": c["arguments"], "result": result})
                    outputs.append(result)
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": _short(result)})
                continue

            answer = (resp.get("content") or "").strip()
            v = verify_answer_text(answer, outputs) if outputs else Verification(ok=True)
            if v.ok or repaired:
                warning = None if v.ok else (
                    "Some statements could not be traced to tool outputs and should be double-checked:\n"
                    + "\n".join(f"- {i.detail}" for i in v.issues))
                return AgentResult(answer, steps, v, warning, "llm")
            repaired = True
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": (
                "Verifier rejected your answer because these claims are not in the tool outputs:\n"
                + v.feedback() + "\nRewrite the answer using only facts from the tool outputs.")})
        return AgentResult(answer or "I could not finish within the tool-call limit.", steps,
                           warning="Stopped at the tool-call limit.", mode="llm")

    # --------------------------------------------------------- rule-based
    def _run_rules(self, msg: str) -> AgentResult:
        low = msg.lower()
        steps: List[Dict[str, Any]] = []

        def call(name, **kw):
            r = self.tools.call(name, kw)
            steps.append({"tool": name, "arguments": kw, "result": r})
            return r

        scans = call("list_scans")
        ids = [s["scan_id"] for s in scans.get("scans", [])]
        if not ids:
            return AgentResult("No scans are loaded. Upload or select a scan first.", steps, mode="rules")

        mentioned = [i for i in ids if i.lower() in low]
        want_compare = any(w in low for w in ("compare", "previous", "last scan", "last one", "since", "grow", "progress", "change"))
        m = re.search(r"engine\s*(?:id\s*)?#?(\d+)", low)
        engine = int(m.group(1)) if m else None
        scan = mentioned[-1] if mentioned else "latest"
        prev = None
        if len(mentioned) >= 2:
            prev, scan = mentioned[-2], mentioned[-1]
        elif want_compare and len(ids) >= 2:
            prev = "previous"

        parts: List[str] = []
        if any(w in low for w in ("how to", "repair", "treat", "guidance", "procedure", "what should", "recommend")) \
                and not any(w in low for w in ("safe", "fly")):
            q = call("retrieve_guidance", query=msg, k=3)
            for c in q.get("chunks", []):
                parts.append(f"[{c['id']}] {c['action']}" + (f" ({c['interval']})" if c.get("interval") else ""))
            return AgentResult("Relevant maintenance guidance:\n" + "\n".join(f"- {p}" for p in parts), steps, mode="rules")

        r = call("assess_flight_readiness", scan_id=scan, previous_scan_id=prev, engine_id=engine) if prev or engine is not None \
            else call("assess_flight_readiness", scan_id=scan)
        if "error" in r:
            return AgentResult(f"Could not complete the request: {r['error']}", steps, mode="rules")

        c = r["counts"]
        head = f"Scan {r['scan_id']}: {c['defects_total']} defect(s) detected"
        if c["defects_total"]:
            head += f" ({c['high']} high, {c['medium']} medium, {c['low']} low severity)"
        parts.append(head + ".")
        if r.get("previous_scan_id"):
            parts.append(f"Compared with {r['previous_scan_id']}: {c['growing']} growing, {c['new']} new, {c['resolved']} resolved.")
        if "rul_cycles" in r:
            parts.append(f"Engine: RUL {r['rul_cycles']} cycles, health {r['health_score']}%, failure risk {r['failure_risk']}.")
        elif r.get("rul_unavailable_reason"):
            parts.append(f"Engine RUL unavailable: {r['rul_unavailable_reason']}")
        parts.append(f"Advisory risk: {r['overall_risk']}. {r['readiness_text']}.")
        parts.append("Reasons: " + "; ".join(r["reasons"]) + ".")

        # guidance for the worst defect, with citation
        worst = sorted(r["defects"], key=lambda d: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(d.get("severity_band"), 0), reverse=True)[:1]
        if worst:
            g = call("retrieve_guidance", query=f"{worst[0]['type']} {worst[0]['severity_band']} action",
                     defect_type=worst[0]["type"], severity=worst[0]["severity_band"], k=1)
            for ch in g.get("chunks", []):
                parts.append(f"Guidance for the most severe defect: {ch['action']} [{ch['id']}].")
        parts.append("This is an advisory output; a certified person must confirm it.")
        answer = "\n\n".join(parts)
        v = verify_answer_text(answer, [s["result"] for s in steps])
        return AgentResult(answer, steps, v, None if v.ok else "; ".join(i.detail for i in v.issues), "rules")
