"""CrewAI implementation: role-based agents (researcher, writer, reviewer).

This is the second orchestration pattern for the same problem graph.py
solves with LangGraph -- built to compare CrewAI's sequential, role-based
delegation style against LangGraph's explicit planner/fan-out/aggregate
state machine.

Running this for real requires an LLM key: set OPENAI_API_KEY or
ANTHROPIC_API_KEY and CREWAI_MODEL (e.g. "openai/gpt-4.1" or
"anthropic/claude-sonnet-4-5-20250929"). CrewAI drives its own LLM calls
through litellm rather than through this project's llm.py abstraction, so
the offline mock backend used elsewhere does not apply here -- that's a
real architectural difference worth calling out versus the LangGraph path.
"""
from __future__ import annotations

import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool as CrewBaseTool

from . import tools as tool_module


class CalculatorTool(CrewBaseTool):
    name: str = "calculator"
    description: str = "Evaluate a basic arithmetic expression, e.g. '12 * (4 + 3)'."

    def _run(self, expression: str) -> str:
        return tool_module.calculator.func(expression)


class WebSearchTool(CrewBaseTool):
    name: str = "web_search"
    description: str = "Search the public web for a query and return short result snippets."

    def _run(self, query: str) -> str:
        return tool_module.web_search.func(query)


class FileLookupTool(CrewBaseTool):
    name: str = "file_lookup"
    description: str = "Search local text files under knowledge/ for lines matching a query."

    def _run(self, query: str) -> str:
        return tool_module.file_lookup.func(query)


def _crew_tools():
    return [CalculatorTool(), WebSearchTool(), FileLookupTool()]


def build_crew(request: str) -> Crew:
    model = os.getenv("CREWAI_MODEL", "openai/gpt-4.1")
    tools = _crew_tools()

    researcher = Agent(
        role="Researcher",
        goal=f"Gather accurate, well-sourced information needed to answer: {request}",
        backstory="A meticulous research analyst who always checks facts before reporting them.",
        tools=tools,
        llm=model,
        verbose=False,
    )

    writer = Agent(
        role="Writer",
        goal="Turn raw research notes into a clear, well-structured answer for the user.",
        backstory="A technical writer who values clarity and concision over jargon.",
        llm=model,
        verbose=False,
    )

    reviewer = Agent(
        role="Reviewer",
        goal="Check the draft answer for factual consistency with the research notes and flag unsupported claims.",
        backstory="A skeptical editor who double-checks every claim against the source research.",
        llm=model,
        verbose=False,
    )

    research_task = Task(
        description=f"Research the following request and collect the key facts, using tools where useful: {request}",
        expected_output="A bullet list of researched facts with brief source notes.",
        agent=researcher,
    )

    writing_task = Task(
        description="Write a concise final answer to the user's request based on the research notes.",
        expected_output="A short, well-structured answer (3-6 sentences).",
        agent=writer,
        context=[research_task],
    )

    review_task = Task(
        description="Review the draft answer against the research notes. Fix any unsupported claims and return the final, corrected version.",
        expected_output="The final, fact-checked answer.",
        agent=reviewer,
        context=[research_task, writing_task],
    )

    return Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=False,
    )


def run(request: str) -> str:
    crew = build_crew(request)
    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Summarize what LangGraph is used for."
    print(run(query))
