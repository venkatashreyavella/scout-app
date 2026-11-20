from __future__ import annotations

import json
import os
import re
from urllib.parse import urlsplit

import streamlit as st
from scout_agent.core import run_agent
from scout_agent.demo import DEMO_QUERIES
from scout_agent.models import RunReport, Settings, TraceEvent

st.set_page_config(page_title="Scout · Research, with receipts", page_icon="✦", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container { max-width: 1450px; padding-top: 2.2rem; padding-bottom: 2rem; }
h1 { font-family: Georgia, serif !important; font-weight: 400 !important; letter-spacing: -1.5px; }
h2,h3 { letter-spacing: -.35px; }
[data-testid="stSidebar"] { border-right: 1px solid #dfe6d8; }
[data-testid="stText"] { font-family: Arial, sans-serif; white-space: pre-wrap; line-height: 1.7; }
[data-testid="stTextArea"] textarea { line-height: 1.8; border-radius: 8px; }
[data-testid="stForm"] { background: #ffffff; border: 1px solid #dfe6d8; border-radius: 12px; padding: 1.35rem; }
[data-testid="stExpander"] { background: #fff; border-radius: 8px; }
.scout-kicker { color: #607657; font-size: .7rem; letter-spacing: .15rem; font-weight: 600; }
.scout-subtitle { color: #64715d; font-size: 1rem; line-height: 1.7; margin-bottom: 1.5rem; }
.scout-brand { font-size: 2rem; color: #315e43; font-weight: 700; letter-spacing: -.08rem; }
.scout-foot { color: #66735f; font-size: .8rem; margin-top: 2rem; border-top: 1px solid #dfe6d8; padding-top: 1rem; }
@media(max-width:700px) { .block-container { padding-top: 1rem; } h1 { font-size: 2.3rem !important; } }
</style>
""", unsafe_allow_html=True)

def escaped(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+.!<>])", r"\\\1", value)

def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, os.environ.get(name, default))
        return str(value) if value is not None else default
    except Exception:
        return os.environ.get(name, default)

def settings() -> Settings:
    return Settings(api_key=secret("OPENAI_API_KEY"), tavily_key=secret("TAVILY_API_KEY"),
                    access_token=secret("AGENT_ACCESS_TOKEN"),
                    model=secret("OPENAI_MODEL", "gpt-4o-mini"))

def choose_example(query: str):
    st.session_state["question"] = query

def reset():
    st.session_state["question"] = ""
    st.session_state["report"] = None
    st.session_state["events"] = []
    st.session_state["live_password"] = ""

st.session_state.setdefault("question", DEMO_QUERIES[0])
st.session_state.setdefault("events", [])
st.session_state.setdefault("report", None)

with st.sidebar:
    st.markdown('<div class="scout-brand">✦ scout.</div>', unsafe_allow_html=True)
    st.caption("RESEARCH LAB")
    st.divider()
    st.button("New research", on_click=reset, use_container_width=True)
    mode_label = st.radio("Research mode", ["Demo", "Live"], horizontal=True)
    mode = mode_label.lower()
    if mode == "live":
        password = st.text_input("Live access password", type="password", key="live_password",
                                 help="This is the server-configured access password, not an API key.")
        st.caption("Uses server-side OpenAI and Tavily keys. Provider usage may incur charges.")
    else:
        password = ""
        st.caption("Curated examples. No API keys or provider charges.")
    st.divider()
    with st.expander("How Scout works"):
        st.write("Search technical sources, calculate numbers, and check the final answer.")
        st.write("The trace shows tool inputs, bounded outputs, latency, and reported token usage.")
        st.write("Private model reasoning is never shown.")
        st.write("Citation checks verify source IDs, not whether a source proves every claim.")
    with st.expander("Privacy"):
        st.write("Live questions go to OpenAI; search queries go to Tavily. Do not submit secrets.")
        st.write("This app does not save your research to a database. Each session has its own result and trace.")
        st.write("Demo mode uses a scripted model and curated source excerpts.")

st.markdown('<div class="scout-kicker">YOUR RESEARCH, WITH RECEIPTS</div>', unsafe_allow_html=True)
st.title("Good research.\nBetter decisions.")
st.markdown('<div class="scout-subtitle">Go from a question to a grounded answer. Scout searches, calculates, and checks its work along the way.</div>', unsafe_allow_html=True)

left, right = st.columns([1.65, 1], gap="large")
with left:
    with st.form("research_form"):
        st.subheader("What are we looking into?")
        query = st.text_area("Technical research question", key="question", height=150, max_chars=2000,
                             placeholder="Compare tools, explore documentation, or estimate a cost.")
        submitted = st.form_submit_button("Run research", type="primary", use_container_width=True)
    st.caption("Demo uses a scripted model and fixture search." if mode == "demo" else "Live keys stay on the server.")
    st.caption("A FEW PLACES TO START")
    example_columns = st.columns(3)
    for column, label, example in zip(example_columns,
            ["Postgres hosting", "Next.js performance", "Missing evidence"], DEMO_QUERIES):
        column.button(label, on_click=choose_example, args=(example,), use_container_width=True)

with right:
    st.subheader("Execution trace")
    trace_status = st.empty()
    trace_area = st.empty()
    st.caption("Execution events, not private model reasoning.")

def draw_trace(events: list[TraceEvent], running: bool = False):
    trace_status.caption("● Running" if running else ("✓ Finished" if events else "Ready to follow the evidence"))
    with trace_area.container():
        if not events:
            st.info("Run a question to inspect tool calls and verification as they happen.")
            st.write("01 · Gather evidence")
            st.write("02 · Work the numbers")
            st.write("03 · Check the answer")
        for index, event in enumerate(events, 1):
            elapsed = f" · {event.duration_ms} ms" if event.duration_ms is not None else ""
            with st.expander(f"{index:02d} · {escaped(event.label)}{elapsed}"):
                st.json(event.model_dump(mode="json"), expanded=False)
        if events:
            input_tokens = sum(e.tokens.input for e in events if e.tokens)
            output_tokens = sum(e.tokens.output for e in events if e.tokens)
            st.caption(f"{len(events)} events · {input_tokens:,} input / {output_tokens:,} output tokens")

draw_trace(st.session_state["events"])

if submitted:
    st.session_state["report"] = None
    st.session_state["events"] = []

    def on_trace(event: TraceEvent):
        st.session_state["events"].append(event)
        draw_trace(st.session_state["events"], running=True)

    with left:
        with st.spinner("Following the evidence…"):
            report = run_agent(query, mode, settings=settings(), access_password=password, on_trace=on_trace)
    st.session_state["report"] = report
    draw_trace(report.trace)

report: RunReport | None = st.session_state["report"]
with left:
    if report:
        result = report.result
        st.divider()
        st.caption("RESEARCH BRIEF · " + result.status.replace("_", " ").upper())
        st.subheader(escaped(result.title))
        st.text(result.summary)
        if report.mode == "demo":
            st.info("Demo brief: scripted model and curated evidence, not live research.")
        if result.findings:
            st.markdown("#### What the evidence says")
            for finding in result.findings:
                st.text("✓ " + finding.claim + " [" + ", ".join(finding.source_ids) + "]")
        for calculation in result.calculations:
            with st.container(border=True):
                st.caption("CHECKED CALCULATION")
                st.code(f"{calculation.expression} = {calculation.value:g}", language=None)
                st.text(calculation.explanation)
        if result.actions:
            st.markdown("#### Suggested next steps")
            for index, action in enumerate(result.actions, 1):
                st.text(f"{index}. {action}")
        if report.sources:
            st.markdown("#### Sources")
            for source in report.sources:
                st.link_button(f"{source.id} · {escaped(source.title)}", source.url, use_container_width=True)
                st.caption(urlsplit(source.url).hostname)
        if result.limitations:
            with st.expander("Assumptions & limitations"):
                for limitation in result.limitations:
                    st.text("• " + limitation)
        st.caption(f"{result.confidence.capitalize()} confidence · {report.duration_ms / 1000:.2f}s · Run {report.run_id[:8]}")
        st.download_button("Download brief & trace", data=report.model_dump_json(indent=2),
                           file_name=f"scout-{report.run_id}.json", mime="application/json",
                           use_container_width=True)
    else:
        st.divider()
        st.subheader("Built to show its work.")
        st.write("Cited findings. Checked numbers. A trace you can inspect.")
        st.caption("Choose an example and run it. The missing-evidence example demonstrates safe abstention.")

st.markdown('<div class="scout-foot">Scout · Technical research and decision support · Structured outputs with Pydantic</div>', unsafe_allow_html=True)
