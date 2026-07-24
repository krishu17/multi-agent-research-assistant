"""Deterministic offline stand-in for a real chat model.

This is NOT a language model. It pattern-matches on the structured prompts
that graph.py sends (each one starts with a `### TASK: <NAME>` marker) and
returns a canned-but-valid response in the format the calling node expects.

Why this exists: it lets the LangGraph control flow, the ReAct tool-calling
loop, and the pytest suite run end-to-end with no API key, no network
access, and no cost, so the *architecture* can be verified independently of
which model eventually powers it. Point LLM_PROVIDER at openai / anthropic /
ollama to use a real model instead -- the graph code does not change.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_MATH_RE = re.compile(r"[-+*/^]|\d+\s*(percent|squared|cubed)", re.IGNORECASE)
_SEARCH_WORDS = ("search", "find", "look up", "latest", "current", "news", "who is", "what is")


class MockChatModel(BaseChatModel):
    """Offline stand-in LLM used for tests and local development."""

    @property
    def _llm_type(self) -> str:  # pragma: no cover - trivial
        return "mock-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = messages[-1].content
        reply = self._route(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    # -- routing -----------------------------------------------------
    def _route(self, prompt: str) -> str:
        if "### TASK: PLAN" in prompt:
            return self._plan(prompt)
        if "### TASK: REACT_STEP" in prompt:
            return self._react_step(prompt)
        if "### TASK: SYNTHESIZE" in prompt:
            return self._synthesize(prompt)
        return "I don't have enough context to respond to that prompt format."

    def _plan(self, prompt: str) -> str:
        request = _after_marker(prompt, "REQUEST:")
        parts = [p.strip() for p in re.split(r"\band\b|,", request) if p.strip()]
        if not parts:
            parts = [request.strip() or "Answer the user's request"]
        if len(parts) == 1:
            parts = [f"Research: {parts[0]}", f"Summarize findings for: {parts[0]}"]
        subtasks = parts[:3]
        quoted = ", ".join(f'"{s}"' for s in subtasks)
        return f"[{quoted}]"

    def _react_step(self, prompt: str) -> str:
        subtask = _after_marker(prompt, "SUBTASK:", until="SCRATCHPAD:").strip()
        scratchpad = _after_marker(prompt, "SCRATCHPAD:").strip()

        # If we've already made one tool call (scratchpad has an OBSERVATION),
        # wrap up instead of looping forever.
        if "OBSERVATION" in scratchpad:
            last_obs_line = [l for l in scratchpad.splitlines() if l.startswith("OBSERVATION:")][-1]
            last_obs = last_obs_line.split(":", 1)[1].strip()
            return f"FINAL: {subtask.rstrip('.')} -> {last_obs}"

        if _MATH_RE.search(subtask):
            expr = re.sub(r"[^0-9.+\-*/() ]", "", subtask) or "0"
            return f"ACTION: calculator | {expr.strip()}"

        if any(w in subtask.lower() for w in _SEARCH_WORDS):
            return f"ACTION: web_search | {subtask.strip()}"

        return f"ACTION: file_lookup | {subtask.strip()}"

    def _synthesize(self, prompt: str) -> str:
        results = _after_marker(prompt, "RESULTS:")
        lines = [l.strip("- ").strip() for l in results.strip().splitlines() if l.strip()]
        if not lines:
            return "No subtask results were available to synthesize."
        bullet_summary = "; ".join(lines)
        return f"Summary: {bullet_summary}"


def _after_marker(prompt: str, marker: str, until: Optional[str] = None) -> str:
    idx = prompt.find(marker)
    if idx == -1:
        return ""
    rest = prompt[idx + len(marker):]
    stops = [s for s in (rest.find("###"), rest.find(until) if until else -1) if s != -1]
    end = min(stops) if stops else -1
    return rest if end == -1 else rest[:end]
