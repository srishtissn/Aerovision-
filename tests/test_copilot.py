"""Run with:  python -m pytest tests -q     (no API keys needed; LLMs are scripted)"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.copilot.agent import CopilotAgent
from src.copilot.context import build_context
from src.copilot.decision import assess
from src.copilot.llm import ScriptedLLM
from src.copilot.rag import KnowledgeBase
from src.copilot.report_generator import build_template_report, generate_report, retrieve_for_context
from src.copilot.tools import CopilotTools, ScanStore
from src.copilot.verifier import verify_answer_text, verify_report

DETS = [
    dict(defect_type="crack", confidence=0.94, bbox=(100, 100, 60, 20), relative_area=0.004, length_cm=2.1),
    dict(defect_type="corrosion", confidence=0.8, bbox=(300, 300, 120, 100), relative_area=0.05),
]


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(backend="tfidf")


@pytest.fixture()
def ctx():
    return build_context(DETS, (480, 640), rul_cycles=18, health_score=63, failure_risk="medium")


def test_retrieval_matches_defect_and_severity(kb):
    hits = kb.for_defect("crack", "HIGH")
    assert hits[0].chunk.id == "crack-high"
    assert kb.search("rust on wing skin", k=1)[0].chunk.defect_type == "corrosion"


def test_template_passes_verifier(ctx, kb):
    rep = build_template_report(ctx)
    assert verify_report(rep, ctx).ok
    ret = retrieve_for_context(ctx, kb)
    rep = build_template_report(ctx, ret)
    assert verify_report(rep, ctx, ret, require_citations=True).ok


def test_verifier_rejects_hallucinations(ctx, kb):
    ret = retrieve_for_context(ctx, kb)
    good = build_template_report(ctx, ret)
    bad = good.model_copy(deep=True)
    bad.summary = "3 cracks and a dent found, RUL 25 cycles, inspect within 48 hours."
    bad.overall_risk = "LOW"
    bad.findings[0].cited_chunks = ["made-up-chunk"]
    codes = {i.code for i in verify_report(bad, ctx, ret, True).issues}
    assert {"unsupported_number", "unsupported_defect", "unsupported_interval",
            "invalid_citation", "risk_understated"} <= codes


def test_negated_defect_mentions_are_allowed(ctx, kb):
    ret = retrieve_for_context(ctx, kb)
    rep = build_template_report(ctx, ret)
    rep.summary += " No dent was detected; check for hidden cracks near the rivet line."
    assert verify_report(rep, ctx, ret, True).ok


def test_fallback_when_llm_hallucinates(ctx, kb):
    ret = retrieve_for_context(ctx, kb)
    bad = build_template_report(ctx, ret).model_copy(deep=True)
    bad.summary = "Found 9 dents."
    llm = ScriptedLLM([{"content": bad.model_dump_json()}])
    g = generate_report(ctx, "llm_rag", llm, kb)
    assert g.fallback_used and g.verification.ok and len(g.attempts) == 2


def test_repair_succeeds_on_second_try(ctx, kb):
    ret = retrieve_for_context(ctx, kb)
    good = build_template_report(ctx, ret)
    bad = good.model_copy(deep=True); bad.summary = "Found 9 dents."
    llm = ScriptedLLM([{"content": bad.model_dump_json()}, {"content": good.model_dump_json()}])
    g = generate_report(ctx, "llm_rag", llm, kb)
    assert not g.fallback_used and g.verification.ok and len(g.attempts) == 2


def test_invalid_json_falls_back(ctx, kb):
    g = generate_report(ctx, "llm_rag", ScriptedLLM(["not json"]), kb)
    assert g.fallback_used and g.attempts[0].report is None


def test_guardrails_off_keeps_raw_attempt(ctx, kb):
    bad = build_template_report(ctx).model_copy(deep=True); bad.summary = "Found 9 dents."
    g = generate_report(ctx, "llm", ScriptedLLM([{"content": bad.model_dump_json()}]), kb, guardrails=False)
    assert not g.first_attempt.verification.ok and len(g.attempts) == 1


def test_decision_rules():
    clean = build_context([], (480, 640))
    assert assess(clean)["flight_readiness"] == "NO_ACTION"
    healthy_crack = build_context([dict(defect_type="dent", confidence=0.5, bbox=(5, 5, 10, 10), relative_area=0.0005)], (480, 640))
    assert assess(healthy_crack)["overall_risk"] == "LOW"
    prog = [dict(defect_type="dent", trend="growing", change_pct=50.0, old_area=100, new_area=150)]
    grown = build_context([dict(defect_type="dent", confidence=0.5, bbox=(5, 5, 10, 10), relative_area=0.0005)], (480, 640), progression=prog)
    assert assess(grown)["overall_risk"] == "MEDIUM"


def test_chat_answer_verifier():
    tool_out = [{"overall_risk": "HIGH", "counts": {"defects_total": 4}, "defects": [{"type": "crack", "confidence": 0.9}]}]
    assert verify_answer_text("4 defects including a crack at 90% confidence; risk HIGH.", tool_out).ok
    assert not verify_answer_text("12 defects found, plus corrosion.", tool_out).ok


def test_agent_tool_loop_and_rules_fallback(kb):
    store = ScanStore(); store.load_samples()
    tools = CopilotTools(store, kb=kb)

    def script(messages, _tools):
        if not any(m["role"] == "tool" for m in messages):
            return {"tool_calls": [{"id": "1", "name": "assess_flight_readiness",
                                    "arguments": {"scan_id": "latest", "previous_scan_id": "previous"}}]}
        data = json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
        return {"content": f"Advisory risk is {data['overall_risk']}."}

    r = CopilotAgent(tools, ScriptedLLM(script)).run("is it safe to fly?")
    assert r.steps[0]["tool"] == "assess_flight_readiness" and r.warning is None

    r2 = CopilotAgent(tools, None).run("Compare this scan with the last one and tell me if it's safe to fly")
    assert r2.mode == "rules" and r2.verification.ok
    assert "Advisory risk" in r2.answer


def test_tool_errors_do_not_crash(kb):
    tools = CopilotTools(ScanStore(), kb=kb)
    assert "error" in tools.call("detect_defects", {"scan_id": "nope"})
    assert "error" in tools.call("no_such_tool", {})


def test_agent_falls_back_when_llm_fails(kb):
    class Boom(ScriptedLLM):
        def chat(self, *a, **k):
            raise RuntimeError("Connection error.")
    store = ScanStore(); store.load_samples()
    r = CopilotAgent(CopilotTools(store, kb=kb), Boom([])).run("is it safe to fly?")
    assert r.mode == "rules" and "LLM unavailable" in r.warning


def test_vision_second_opinion_with_fake_gemini(kb):
    class FakeGemini:
        name = "fake-gemini"
        def generate_with_image(self, prompt, jpeg_bytes, json_mode=True):
            assert jpeg_bytes[:2] == b"\xff\xd8"  # a real JPEG crop was sent
            return json.dumps({"defect_present": True, "defect_type": "corrosion", "confidence": 0.7, "rationale": "Brown staining."})
    store = ScanStore(); store.load_samples()
    tools = CopilotTools(store, kb=kb, vision_llm=FakeGemini())
    d = tools.detect_defects("panel_defects_scan2")["defects"][0]["id"]
    res = tools.vision_second_opinion("panel_defects_scan2", d)
    assert res["agreement"] in ("agree", "type_disagree") and "needs_human_review" in res
    assert "error" in CopilotTools(store, kb=kb).vision_second_opinion("latest", "D1")
