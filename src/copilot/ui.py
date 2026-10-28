"""Streamlit UI for the Maintenance Copilot (rendered from app.py)."""
from __future__ import annotations

import html
import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.report import draw_overlays

from . import config
from .agent import CopilotAgent
from .llm import get_gemini, get_groq
from .rag import get_kb
from .report_generator import MODES, generate_report, report_to_markdown
from .tools import CopilotTools, ScanStore

# ------------------------------------------------------------------ theme
NAVY, BLUE, TEAL = "#0B1F3A", "#2563EB", "#0EA5A4"
SEV = {"HIGH": ("#DC2626", "#FEE2E2"), "MEDIUM": ("#D97706", "#FEF3C7"), "LOW": ("#15803D", "#DCFCE7")}
READY = {
    "NO_GO": ("🛑", "Do not release", "Ground the aircraft for inspection"),
    "CONDITIONAL": ("⚠️", "Conditional", "Detailed inspection before release"),
    "MONITOR": ("👁️", "Monitor", "Log and verify at next check"),
    "NO_ACTION": ("✅", "No action", "Continue normal inspection program"),
}
TOOL_LABELS = {
    "list_scans": "📷 Checked loaded scans", "detect_defects": "🔎 Ran the defect detector",
    "score_severity": "📊 Scored severity", "compare_scans": "🔀 Compared two scans",
    "predict_rul": "⚙️ Predicted engine RUL", "fuse_inspection": "🧩 Built the fused record",
    "assess_flight_readiness": "🛫 Assessed flight readiness", "retrieve_guidance": "📚 Looked up maintenance guidance",
    "generate_report": "📄 Generated a report", "vision_second_opinion": "🔍 Asked Gemini for a second opinion",
}
QUICK_PROMPTS = [
    ("🛫 Safe to fly?", "Is the latest scan safe to fly?"),
    ("🔀 Compare scans", "Compare this scan with the last one and tell me if it's safe to fly"),
    ("🛠️ What should I fix?", "What should I do about the most severe defect?"),
    ("📊 Severity summary", "Give me a severity summary of the latest scan"),
]

CSS = f"""
<style>
.block-container {{padding-top: 1.2rem; max-width: 1180px;}}
#MainMenu, footer {{visibility: hidden;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #123C7A 55%, {BLUE} 100%); color: #fff;
  border-radius: 18px; padding: 26px 30px; margin-bottom: 14px; position: relative; overflow: hidden;}}
.hero:after {{content: "✈"; position: absolute; right: 26px; top: -8px; font-size: 120px; opacity: .08;}}
.hero h1 {{margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -.5px; color: #fff;}}
.hero p {{margin: 6px 0 14px; opacity: .85; font-size: 1.0rem; color: #fff;}}
.pill {{display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin: 0 6px 4px 0;
  background: rgba(255,255,255,.14); color: #fff; border: 1px solid rgba(255,255,255,.25);}}
.pill.on {{background: rgba(16,185,129,.25); border-color: rgba(16,185,129,.6);}}
.card {{background: #fff; border: 1px solid #E5EAF3; border-radius: 14px; padding: 16px 18px; box-shadow: 0 1px 2px rgba(15,23,42,.04);}}
.step {{background: #fff; border: 1px solid #E5EAF3; border-radius: 14px; padding: 18px; height: 100%;}}
.step .n {{display:inline-flex; width: 28px; height: 28px; border-radius: 50%; background: {BLUE}; color:#fff; font-weight:700;
  align-items:center; justify-content:center; margin-bottom: 8px;}}
.step h4 {{margin: 2px 0 4px; font-size: 1.0rem;}} .step p {{margin: 0; color:#475569; font-size:.9rem;}}
.tile {{background:#fff; border:1px solid #E5EAF3; border-radius:14px; padding:12px 16px;}}
.tile .l {{color:#64748B; font-size:.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em;}}
.tile .v {{font-size:1.65rem; font-weight:800; color:{NAVY}; line-height:1.2;}}
.tile .s {{color:#64748B; font-size:.78rem;}}
.banner {{border-radius:16px; padding: 18px 22px; display:flex; gap:16px; align-items:center; margin: 6px 0 14px;}}
.banner .ic {{font-size: 2.2rem;}} .banner .t {{font-size:1.35rem; font-weight:800;}} .banner .d {{font-size:.95rem; opacity:.9;}}
.sev {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem; font-weight:700;}}
.finding {{background:#fff; border:1px solid #E5EAF3; border-left-width:5px; border-radius:12px; padding:12px 16px; margin-bottom:10px;}}
.finding .h {{font-weight:700; color:{NAVY};}} .finding .o {{color:#334155; font-size:.92rem; margin:4px 0;}}
.finding .a {{color:#0F172A; font-size:.92rem;}}
.chip {{display:inline-block; background:#EEF2FF; color:#3730A3; border-radius:8px; padding:1px 8px; font-size:.75rem; font-weight:600; margin-right:5px;}}
.muted {{color:#64748B; font-size:.85rem;}}
.stButton > button {{border-radius: 10px; font-weight: 600;}}
button[data-baseweb="tab"] {{font-weight: 600;}}
.scanlabel {{font-size:.8rem; font-weight:600; color:{NAVY}; text-align:center;}}
</style>
"""


# ------------------------------------------------------------------ cached
@st.cache_resource(show_spinner="Loading knowledge base...")
def _kb():
    return get_kb()


@st.cache_resource
def _groq():
    return get_groq()


@st.cache_resource
def _gemini():
    return get_gemini()


def _state() -> ScanStore:
    if "copilot_store" not in st.session_state:
        st.session_state.copilot_store = ScanStore()
        st.session_state.copilot_chat = []
    return st.session_state.copilot_store


def _decode(file) -> np.ndarray:
    return cv2.imdecode(np.asarray(bytearray(file.read()), dtype=np.uint8), cv2.IMREAD_COLOR)


def _html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def _sev_pill(band: str | None) -> str:
    if not band:
        return ""
    fg, bg = SEV[band]
    return f'<span class="sev" style="background:{bg};color:{fg}">{band}</span>'


def _tile(label: str, value, sub: str = "") -> str:
    return f'<div class="tile"><div class="l">{html.escape(label)}</div><div class="v">{html.escape(str(value))}</div><div class="s">{html.escape(sub)}</div></div>'


# ------------------------------------------------------------------ pieces
def _hero(kb, groq, gemini):
    def pill(txt, on):
        return f'<span class="pill {"on" if on else ""}">{"●" if on else "○"} {txt}</span>'
    _html(f"""<div class="hero"><h1>Maintenance Copilot</h1>
      <p>Ask questions about your inspection scans, get verified reports, and see exactly which guidance backs every recommendation.</p>
      {pill("Llama 3.3 agent", bool(groq))}{pill("Gemini vision", bool(gemini))}{pill(f"RAG: {kb.backend}", True)}{pill("Claim verifier", True)}
    </div>""")


def _onboarding(store: ScanStore):
    st.markdown("#### Get started in three steps")
    c1, c2, c3 = st.columns(3)
    for c, n, t, d in [
        (c1, 1, "Add scans", "Load the sample panels or upload your own photos, oldest first."),
        (c2, 2, "Ask or generate", "Chat with the copilot, or build a verified inspection report."),
        (c3, 3, "Check the evidence", "Every number and recommendation traces back to the record or cited guidance."),
    ]:
        c.markdown(f'<div class="step"><div class="n">{n}</div><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1, 1.4])
    with left:
        if st.button("✨ Try with sample scans", type="primary", use_container_width=True):
            store.load_samples()
            st.rerun()
        st.caption("Loads 4 demo panels (clean, rivet line, and a before/after pair).")
    with right:
        files = st.file_uploader("…or upload your own scans (oldest first)", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key="copilot_up_first")
        if files and st.button("Add uploaded scans", use_container_width=True):
            for f in files:
                store.add(os.path.splitext(f.name)[0], _decode(f))
            st.rerun()


def _scan_strip(store: ScanStore):
    ids = list(store.scans)
    cols = st.columns(min(len(ids), 6) + 1)
    for i, sid in enumerate(ids[-6:]):
        tag = "latest" if sid == ids[-1] else ("previous" if len(ids) > 1 and sid == ids[-2] else "")
        with cols[i]:
            st.image(cv2.cvtColor(store.scans[sid].image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f'<div class="scanlabel">{html.escape(sid)}</div>' + (f'<div style="text-align:center"><span class="chip">{tag}</span></div>' if tag else ""),
                        unsafe_allow_html=True)
    with cols[-1]:
        with st.popover("⚙️ Manage"):
            files = st.file_uploader("Add more scans", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="copilot_up_more")
            if files and st.button("Add", key="add_more"):
                for f in files:
                    store.add(os.path.splitext(f.name)[0], _decode(f))
                st.rerun()
            if st.button("Load samples", key="load_more"):
                store.load_samples(); st.rerun()
            if st.button("🗑️ Clear all", key="clear_all"):
                st.session_state.copilot_store = ScanStore()
                st.session_state.copilot_chat = []
                st.session_state.pop("last_gen", None)
                st.rerun()


def _risk_banner(report):
    fg, bg = SEV[report.overall_risk]
    ic, title, sub = READY[report.flight_recommendation]
    _html(f'<div class="banner" style="background:{bg};border:1px solid {fg}33;color:{fg}"><div class="ic">{ic}</div>'
          f'<div><div class="t">{title} · {report.overall_risk} risk</div><div class="d" style="color:#1E293B">{html.escape(report.summary)}</div></div></div>')


def _finding_card(f, band: str | None):
    fg = SEV.get(band, ("#64748B", ""))[0]
    cites = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in f.cited_chunks)
    _html(f'<div class="finding" style="border-left-color:{fg}"><div class="h">{f.defect_id} · {f.type.replace("_", " ").title()} &nbsp;{_sev_pill(band)}</div>'
          f'<div class="o">{html.escape(f.observation)}</div><div class="a"><b>Action:</b> {html.escape(f.recommended_action)}</div>'
          f'<div style="margin-top:6px">{cites}</div></div>')


# -------------------------------------------------------------------- page
def render_copilot_page():
    _html(CSS)
    store = _state()
    kb, groq, gemini = _kb(), _groq(), _gemini()
    tools = CopilotTools(store, kb=kb, report_llm=groq, vision_llm=gemini)

    with st.sidebar:
        st.markdown("**Copilot status**")
        st.markdown(f"{'🟢' if groq else '⚪'} **Agent & reports**  \n<span class='muted'>{'Groq · ' + groq.model if groq else 'No key: offline rule-based mode'}</span>", unsafe_allow_html=True)
        st.markdown(f"{'🟢' if gemini else '⚪'} **Vision second opinion**  \n<span class='muted'>{'Gemini · ' + gemini.model if gemini else 'No key: disabled'}</span>", unsafe_allow_html=True)
        st.markdown(f"🟢 **Knowledge base**  \n<span class='muted'>{len(kb.chunks)} chunks · {kb.backend}</span>", unsafe_allow_html=True)
        with st.expander("How the guardrails work"):
            st.markdown("1. The LLM only sees the **inspection record** and **retrieved guidance**.\n"
                        "2. Its output must match a strict JSON schema.\n"
                        "3. A verifier rejects any number, defect, location, interval or citation not in those sources.\n"
                        "4. Rejected? One retry, then a safe template report.\n"
                        "5. Risk can never be *lower* than the rule-based decision.")
        st.caption("Keys are read from `.env` (see `.env.example`).")

    _hero(kb, groq, gemini)

    if not store.scans:
        _onboarding(store)
        return

    _scan_strip(store)
    st.write("")
    t_chat, t_report, t_kb, t_vis, t_eval = st.tabs(
        ["💬 Ask the copilot", "📄 Inspection report", "📚 Guidance library", "🔍 Second opinion", "📊 Evaluation"])

    # ---------------------------------------------------------------- chat
    with t_chat:
        agent = CopilotAgent(tools, groq)
        hist = st.session_state.copilot_chat
        if not hist:
            _html('<div class="card">👋 <b>Hi, I\'m your maintenance copilot.</b> I run the detector, compare scans, '
                  'check guidance and tell you what the data supports. Pick a question below or type your own.</div>')
        clicked = None
        cols = st.columns(len(QUICK_PROMPTS))
        for c, (label, text) in zip(cols, QUICK_PROMPTS):
            if c.button(label, use_container_width=True, key=f"qp_{label}"):
                clicked = text
        for m in hist:
            with st.chat_message(m["role"], avatar="🧑‍🔧" if m["role"] == "user" else "✈️"):
                st.markdown(m["content"])
                if m.get("warning"):
                    st.warning(m["warning"])
                if m.get("steps"):
                    with st.expander(f"How I got this · {len(m['steps'])} step(s)"):
                        for s in m["steps"]:
                            st.markdown(f"**{TOOL_LABELS.get(s['tool'], s['tool'])}**")
                            st.json(s["result"], expanded=False)
        prompt = st.chat_input("Ask about your scans…") or clicked
        if prompt:
            hist.append({"role": "user", "content": prompt})
            with st.spinner("Working on it: running tools and checking the evidence…"):
                try:
                    res = agent.run(prompt, [{"role": m["role"], "content": m["content"]} for m in hist[:-1]])
                    hist.append({"role": "assistant", "content": res.answer, "steps": res.steps, "warning": res.warning})
                except Exception as e:
                    hist.append({"role": "assistant", "content": f"Sorry, something went wrong: {e}"})
            st.rerun()
        st.caption(("Answering with " + groq.name + " + tools") if groq else "Offline mode: rule-based router (add a Groq key for full LLM chat).")

    # -------------------------------------------------------------- report
    with t_report:
        ids = list(store.scans)
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            scan = c1.selectbox("Scan to report on", ids, index=len(ids) - 1)
            prev = c2.selectbox("Compare with (optional)", ["None"] + [i for i in ids if i != scan],
                                index=0, help="Adds growth / new / resolved defects to the report.")
            engine = c3.number_input("Engine ID for RUL (0 = skip)", 0, 100, 0,
                                     help="NASA C-MAPSS test engine. Needs the trained RUL model.")
            modes = [m for m in MODES if groq or m == "template"]
            labels = {"llm_rag": "🧠 LLM + guidance (recommended)", "llm": "💬 LLM only", "template": "📐 Template (no LLM)"}
            c4, c5 = st.columns([2, 1.2])
            mode = c4.radio("Report style", modes, format_func=lambda m: labels[m],
                            index=modes.index("llm_rag") if "llm_rag" in modes else 0, horizontal=True)
            guard = c5.toggle("Verify & auto-fix", value=True, help="Reject unsupported claims, retry once, then fall back to the template.")
            if not groq:
                st.caption("Add GROQ_API_KEY to `.env` to unlock the LLM styles.")
            go = st.button("Generate report", type="primary", use_container_width=True)
        if go:
            with st.spinner("Detecting, retrieving guidance, writing and verifying…"):
                ctx = tools._ctx(scan, None if prev == "None" else prev, int(engine) or None)
                gen = generate_report(ctx, mode, groq, kb, guardrails=guard)
            st.session_state.last_gen = (gen, ctx)

        if "last_gen" in st.session_state:
            gen, ctx = st.session_state.last_gen
            v, rep = gen.verification, gen.report
            _risk_banner(rep)
            bands = {d["id"]: d.get("severity_band") for d in ctx["defects"]}
            n = ctx["counts"]
            tiles = [("Defects", n["defects_total"], f'{n["high"]} high · {n["medium"]} medium · {n["low"]} low')]
            if "rul_cycles" in ctx:
                tiles.append(("Engine RUL", f'{ctx["rul_cycles"]}', f'health {ctx["health_score"]}% · {ctx["failure_risk"]}'))
            if ctx.get("progression"):
                tiles.append(("Progression", f'{n["growing"]} growing', f'{n["new"]} new · {n["resolved"]} resolved'))
            tiles.append(("Verification", "Passed" if v.ok else "Failed", f"{v.n_claims} claims · {v.n_unsupported} unsupported"))
            for c, t in zip(st.columns(len(tiles)), tiles):
                c.markdown(_tile(*t), unsafe_allow_html=True)
            if "rul_unavailable_reason" in ctx:
                st.warning(f"Engine RUL unavailable: {ctx['rul_unavailable_reason']}")
            if gen.fallback_used:
                st.info("The model's draft failed verification, so a safe template report is shown instead.")
            st.write("")
            left, right = st.columns([1, 1.25])
            with left:
                sc = store.resolve(ctx["scan_id"])
                st.image(cv2.cvtColor(draw_overlays(sc.image, store.detections(sc)), cv2.COLOR_BGR2RGB),
                         caption=f"{sc.id}: detected regions", use_container_width=True)
            with right:
                st.markdown("##### Findings")
                if not rep.findings:
                    st.success("No defects detected in this scan.")
                for f in rep.findings:
                    _finding_card(f, bands.get(f.defect_id))
            if rep.engine_note:
                st.markdown(f"**⚙️ Engine:** {rep.engine_note}")
            if rep.progression_note:
                st.markdown(f"**🔀 Progression:** {rep.progression_note}")
            if rep.inspection_interval:
                st.markdown(f"**⏱️ Inspection interval:** {rep.inspection_interval}")
            if rep.limitations:
                st.caption(" ".join(rep.limitations))

            with st.expander("🛡️ Verification details"):
                if not v.issues:
                    st.success("Every claim traced to the record or retrieved guidance.")
                else:
                    st.dataframe(pd.DataFrame([i.to_dict() for i in v.issues]), hide_index=True, use_container_width=True)
                for k, a in enumerate(gen.attempts, 1):
                    if a.error:
                        st.error(f"Attempt {k}: {a.error}")
                    elif a.verification and not a.verification.ok:
                        st.warning(f"Attempt {k} was rejected:\n\n{a.verification.feedback()}")
            with st.expander("📚 Guidance used"):
                if not gen.retrieved:
                    st.caption("No guidance retrieved in this mode.")
                for r in gen.retrieved:
                    st.markdown(f"<span class='chip'>{r['id']}</span> **{r['title']}** <span class='muted'>· match {r['score']}</span>  \n{r['text']}  \n"
                                f"<span class='muted'>Source: {r['source']}</span>", unsafe_allow_html=True)
            with st.expander("🧾 Inspection record given to the model"):
                st.json(ctx, expanded=False)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download JSON", json.dumps(rep.model_dump(), indent=2), "aerovision_report.json",
                               "application/json", use_container_width=True)
            d2.download_button("⬇️ Download Markdown", report_to_markdown(rep), "aerovision_report.md",
                               "text/markdown", use_container_width=True)

    # ------------------------------------------------------------------ KB
    with t_kb:
        q = st.text_input("🔎 Search guidance", placeholder="e.g. how to treat corrosion near fasteners")
        if q:
            for h in kb.search(q, k=4):
                _html(f'<div class="finding"><div class="h"><span class="chip">{h.chunk.id}</span>{html.escape(h.chunk.title)} '
                      f'{_sev_pill(h.chunk.severity if h.chunk.severity in SEV else None)}</div>'
                      f'<div class="o">{html.escape(h.chunk.text)}</div><div class="a"><b>Action:</b> {html.escape(h.chunk.action)}</div>'
                      f'<div class="muted">{html.escape(h.chunk.source)}</div></div>')
        with st.expander(f"Browse all {len(kb.chunks)} chunks", expanded=not q):
            st.dataframe(pd.DataFrame([{"id": c.id, "defect": c.defect_type, "severity": c.severity, "action": c.action,
                                        "interval": c.interval} for c in kb.chunks]), hide_index=True, use_container_width=True)
        st.caption("Add your own documents as `.md` chunks in `src/copilot/knowledge/` (format in the README), then restart.")

    # -------------------------------------------------------------- vision
    with t_vis:
        st.markdown("A second AI looks at the cropped defect **without being told** what the detector found. "
                    "If they disagree, the finding is flagged for human review. Severity is never lowered.")
        if not gemini:
            st.info("Add GEMINI_API_KEY to `.env` to enable this.")
        else:
            sid = st.selectbox("Scan", ids, index=len(ids) - 1, key="vis_scan")
            det = tools.detect_defects(sid)
            if not det["defects"]:
                st.success("No defects flagged in this scan.")
            else:
                cA, cB = st.columns([1.2, 1])
                with cA:
                    st.dataframe(pd.DataFrame(det["defects"])[["id", "type", "confidence", "location", "severity_band"]],
                                 hide_index=True, use_container_width=True)
                with cB:
                    did = st.selectbox("Defect", [d["id"] for d in det["defects"]])
                    if st.button("Ask Gemini", type="primary", use_container_width=True):
                        with st.spinner("Asking Gemini…"):
                            st.session_state.vis_res = tools.vision_second_opinion(sid, did)
                res = st.session_state.get("vis_res")
                if res and "error" in res:
                    st.error(res["error"])
                elif res:
                    icon = {"agree": "✅", "type_disagree": "⚠️", "disagree": "❌"}[res["agreement"]]
                    st.markdown(f"### {icon} {res['agreement'].replace('_', ' ').title()}")
                    st.markdown(f"Detector: `{res['detector_type']}` · Gemini: `{res['vision_type']}` ({int(res['vision_confidence'] * 100)}% sure)")
                    st.write(res["vision_rationale"])
                    if res["needs_human_review"]:
                        st.warning("Flag for human review.")

    # ---------------------------------------------------------------- eval
    with t_eval:
        p = os.path.join(config.EVAL_OUT_DIR, "summary.csv")
        if not os.path.exists(p):
            st.info("No results yet. Run this in your terminal, then refresh:")
            st.code("python -m src.copilot.evaluation.run_eval --n 30 --judge gemini", language="bash")
        else:
            df = pd.read_csv(p)
            num = lambda s: float(str(s).split()[0]) if str(s).split()[0].replace(".", "", 1).isdigit() else np.nan
            chart = pd.DataFrame({
                "Faithfulness ↑": df["faithfulness"].map(num), "Completeness ↑": df["completeness"].map(num),
                "Hallucination rate ↓": df["hallucinated"].map(num)}, index=df["config"])
            st.bar_chart(chart)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption("Mean [95% bootstrap CI]. Produced by `src.copilot.evaluation.run_eval`.")

    _html('<div class="muted" style="text-align:center;margin-top:18px">Advisory prototype · a certified person makes every airworthiness decision.</div>')
