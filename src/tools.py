"""Tools the worker agent can call.

Each tool is a plain Python function with a docstring and a small,
predictable string-in/string-out contract, wrapped with LangChain's
`@tool` decorator so the same functions can be handed to a real
tool-calling model (`.bind_tools(...)`) or invoked directly by the
hand-rolled ReAct loop in graph.py.
"""
from __future__ import annotations

import ast
import operator
import os

from langchain_core.tools import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (4 + 3)'.

    Supports + - * / ** and parentheses. Does not use eval() -- the
    expression is parsed into an AST and walked so it cannot execute
    arbitrary code.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"error: could not evaluate '{expression}' ({exc})"


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the public web for `query` and return short result snippets.

    Backed by DuckDuckGo (no API key required). Returns an explanatory
    string instead of raising if the network call fails, so a single
    failed search does not crash the agent loop.
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001
        return f"error: web search unavailable ({exc})"

    if not results:
        return "No results found."

    lines = [f"- {r.get('title', '')}: {r.get('body', '')[:200]}" for r in results]
    return "\n".join(lines)


@tool
def file_lookup(query: str) -> str:
    """Search local text files under knowledge/ for lines matching `query`.

    Used as a stand-in for an internal document/database lookup when no
    external network call is appropriate.
    """
    if not os.path.isdir(_KNOWLEDGE_DIR):
        return "No local knowledge base configured."

    query_terms = [t.lower() for t in query.split() if len(t) > 2]
    hits = []
    for fname in sorted(os.listdir(_KNOWLEDGE_DIR)):
        path = os.path.join(_KNOWLEDGE_DIR, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if any(term in line.lower() for term in query_terms):
                    hits.append(f"{fname}: {line.strip()}")
    if not hits:
        return "No matching local knowledge found."
    return "\n".join(hits[:5])


ALL_TOOLS = [calculator, web_search, file_lookup]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
