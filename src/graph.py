"""LangGraph implementation: planner -> parallel worker(s) -> aggregator.

    START -> planner -> [Send(run_subtask) for each subtask] -> aggregator -> END

`planner` breaks the user's request into subtasks. Each subtask is fanned
out to its own `run_subtask` node (LangGraph's Send/map-reduce pattern),
where a small hand-rolled ReAct loop lets the model decide whether to call
a tool (calculator / web_search / file_lookup) before answering. Results
from every branch are reduced back into a single list and `aggregator`
synthesizes the final answer.
"""
from __future__ import annotations

import json
import operator
import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .llm import get_llm
from .tools import TOOLS_BY_NAME

MAX_REACT_STEPS = 3


class SubtaskResult(TypedDict):
    subtask: str
    answer: str
    tool_trace: List[str]


class AgentState(TypedDict):
    request: str
    subtasks: List[str]
    subtask_results: Annotated[List[SubtaskResult], operator.add]
    final_answer: str


class SubtaskState(TypedDict):
    request: str
    subtask: str


def _ask(prompt: str) -> str:
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def planner_node(state: AgentState) -> dict:
    prompt = (
        "### TASK: PLAN\n"
        "Break the user's request into 1-3 concrete, independent subtasks.\n"
        "Respond with a JSON array of short strings, nothing else.\n"
        f"REQUEST:\n{state['request']}\n"
    )
    raw = _ask(prompt)
    subtasks = _parse_json_list(raw) or [state["request"]]
    return {"subtasks": subtasks}


def _parse_json_list(raw: str) -> List[str]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
        return [str(i) for i in items if str(i).strip()]
    except json.JSONDecodeError:
        return []


def fan_out(state: AgentState) -> List[Send]:
    return [
        Send("run_subtask", {"request": state["request"], "subtask": s})
        for s in state["subtasks"]
    ]


def run_subtask_node(state: SubtaskState) -> dict:
    subtask = state["subtask"]
    scratchpad_lines: List[str] = []
    tool_trace: List[str] = []
    final_answer = None

    for _ in range(MAX_REACT_STEPS):
        prompt = (
            "### TASK: REACT_STEP\n"
            "You are a research worker agent. You may call one tool per step:\n"
            "  calculator(expression) | web_search(query) | file_lookup(query)\n"
            "Respond with exactly one line, either:\n"
            "  ACTION: <tool_name> | <tool_input>\n"
            "  FINAL: <answer>\n"
            f"SUBTASK:\n{subtask}\n"
            f"SCRATCHPAD:\n{chr(10).join(scratchpad_lines) or '(empty)'}\n"
        )
        raw = _ask(prompt).strip()

        if raw.upper().startswith("FINAL:"):
            final_answer = raw.split(":", 1)[1].strip()
            break

        action_match = re.match(r"ACTION:\s*(\w+)\s*\|\s*(.+)", raw, re.IGNORECASE)
        if not action_match:
            final_answer = raw
            break

        tool_name, tool_input = action_match.group(1).strip(), action_match.group(2).strip()
        tool_fn = TOOLS_BY_NAME.get(tool_name)
        if tool_fn is None:
            observation = f"error: unknown tool '{tool_name}'"
        else:
            observation = tool_fn.invoke(tool_input)

        tool_trace.append(f"{tool_name}({tool_input}) -> {observation}")
        scratchpad_lines.append(f"ACTION: {tool_name} | {tool_input}")
        scratchpad_lines.append(f"OBSERVATION: {observation}")

    if final_answer is None:
        final_answer = "Could not resolve subtask within step limit."

    result: SubtaskResult = {"subtask": subtask, "answer": final_answer, "tool_trace": tool_trace}
    return {"subtask_results": [result]}


def aggregator_node(state: AgentState) -> dict:
    results_block = "\n".join(f"- {r['subtask']}: {r['answer']}" for r in state["subtask_results"])
    prompt = (
        "### TASK: SYNTHESIZE\n"
        "Combine the subtask results below into one concise final answer for the user.\n"
        f"RESULTS:\n{results_block}\n"
    )
    final_answer = _ask(prompt).strip()
    return {"final_answer": final_answer}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("run_subtask", run_subtask_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", fan_out, ["run_subtask"])
    graph.add_edge("run_subtask", "aggregator")
    graph.add_edge("aggregator", END)
    return graph.compile()


def run(request: str) -> AgentState:
    app = build_graph()
    return app.invoke({"request": request, "subtasks": [], "subtask_results": [], "final_answer": ""})


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is 12 * (4 + 3) and summarize what LangGraph is used for?"
    final_state = run(query)
    print(json.dumps(final_state, indent=2))
