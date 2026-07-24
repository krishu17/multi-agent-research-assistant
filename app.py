"""Streamlit UI for the multi-agent research assistant.

Runs the same LangGraph / CrewAI orchestration paths defined in
src/graph.py and src/crew.py -- this file adds no new agent logic, it
just gives a human a form instead of a CLI. Provider selection and API
keys are session-only: keys typed here are written to os.environ for
this process only, never to disk.

    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

st.set_page_config(page_title="Research Assistant", page_icon="\U0001F578️", layout="centered")

_CSS = """
<style>
:root {
    --graph: #1c7f72;
    --graph-soft: #e2f0ec;
    --crew: #a8631a;
    --crew-soft: #f4e9d8;
}
@media (prefers-color-scheme: dark) {
    :root { --graph: #5fc9b8; --graph-soft: #132824; --crew: #e0a458; --crew-soft: #2a2015; }
}
.stApp [data-testid="stHeader"] { background: transparent; }
.raa-hero { padding: 4px 0 18px; border-bottom: 1px solid rgba(128,128,128,.25); margin-bottom: 22px; }
.raa-hero h1 { font-size: 1.9rem; margin: 0 0 4px; letter-spacing: -.01em; }
.raa-hero p { opacity: .72; margin: 0; font-size: .95rem; }
.raa-chip { display:inline-flex; align-items:center; gap:6px; font-family: ui-monospace, Consolas, monospace;
    font-size: .72rem; font-weight: 600; padding: 2px 10px; border-radius: 999px; margin-bottom: 10px; }
.raa-chip.graph { background: var(--graph-soft); color: var(--graph); }
.raa-chip.crew { background: var(--crew-soft); color: var(--crew); }
.raa-final { border-left: 3px solid var(--graph); background: var(--graph-soft); padding: 16px 18px;
    border-radius: 0 10px 10px 0; font-size: 1.02rem; line-height: 1.5; }
.raa-final.crew { border-left-color: var(--crew); background: var(--crew-soft); }
.raa-trace { font-family: ui-monospace, Consolas, monospace; font-size: .82rem; opacity: .85; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="raa-hero"><h1>Multi-Agent Research Assistant</h1>'
    "<p>Break a request into subtasks, research each with tools, synthesize an answer "
    "&mdash; via a LangGraph state machine or a CrewAI role-based crew.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Orchestrator")
    mode = st.radio(
        "Path", ["LangGraph", "CrewAI"], label_visibility="collapsed",
        captions=["Explicit planner → fan-out → aggregator", "Researcher → Writer → Reviewer"],
    )

    st.subheader("Model")
    if mode == "LangGraph":
        provider = st.selectbox("LLM_PROVIDER", ["mock", "openai", "anthropic", "ollama"], index=0)
        os.environ["LLM_PROVIDER"] = provider
        if provider == "mock":
            st.caption("Offline deterministic stand-in — no key needed, exercises the real control flow.")
        elif provider == "openai":
            key = st.text_input("OPENAI_API_KEY", type="password")
            if key:
                os.environ["OPENAI_API_KEY"] = key
            model = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4.1"))
            os.environ["OPENAI_MODEL"] = model
        elif provider == "anthropic":
            key = st.text_input("ANTHROPIC_API_KEY", type="password")
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key
            model = st.text_input("ANTHROPIC_MODEL", value=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"))
            os.environ["ANTHROPIC_MODEL"] = model
        else:
            model = st.text_input("OLLAMA_MODEL", value=os.getenv("OLLAMA_MODEL", "llama3.1"))
            os.environ["OLLAMA_MODEL"] = model
            st.caption("Needs a local Ollama server running.")
    else:
        st.caption("CrewAI drives its own LLM calls via litellm — always needs a real key, no mock mode.")
        key_provider = st.selectbox("Key for", ["openai", "anthropic"], index=0)
        crewai_model_default = os.getenv("CREWAI_MODEL", "openai/gpt-4.1")
        crewai_model = st.text_input("CREWAI_MODEL", value=crewai_model_default, help='litellm format, e.g. "openai/gpt-4.1" or "anthropic/claude-sonnet-4-5-20250929"')
        os.environ["CREWAI_MODEL"] = crewai_model
        if key_provider == "openai":
            key = st.text_input("OPENAI_API_KEY", type="password")
            if key:
                os.environ["OPENAI_API_KEY"] = key
        else:
            key = st.text_input("ANTHROPIC_API_KEY", type="password")
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key

    st.caption("Keys are kept in memory for this session only — never written to disk.")

question = st.text_area(
    "Your request",
    placeholder="e.g. What is 12 * (4 + 3) and what is LangGraph used for?",
    height=90,
)
run_clicked = st.button("Run", type="primary", use_container_width=True)

if run_clicked:
    if not question.strip():
        st.warning("Enter a request first.")
    elif mode == "CrewAI" and not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        st.error("CrewAI has no mock mode — enter an API key in the sidebar first.")
    else:
        chip_cls = "graph" if mode == "LangGraph" else "crew"
        st.markdown(f'<span class="raa-chip {chip_cls}">{mode}</span>', unsafe_allow_html=True)
        with st.spinner(f"Running the {mode} path…"):
            try:
                if mode == "LangGraph":
                    from src.graph import run as run_graph

                    result = run_graph(question)
                    final_answer = result.get("final_answer", "")
                else:
                    from src.crew import run as run_crew

                    final_answer = run_crew(question)
                    result = None
            except Exception as exc:  # noqa: BLE001
                st.error(f"Run failed: {exc}")
                final_answer = None
                result = None

        if final_answer:
            st.markdown(f'<div class="raa-final {chip_cls}">{final_answer}</div>', unsafe_allow_html=True)

            if mode == "LangGraph" and result and result.get("subtask_results"):
                st.divider()
                st.subheader("How it got there")
                for r in result["subtask_results"]:
                    with st.expander(r["subtask"]):
                        st.write(r["answer"])
                        if r.get("tool_trace"):
                            st.markdown('<div class="raa-trace">' + "<br>".join(r["tool_trace"]) + "</div>", unsafe_allow_html=True)
                        else:
                            st.caption("No tool calls — answered directly.")
