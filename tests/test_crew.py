"""Structural tests for the CrewAI path.

crew.kickoff() needs a real LLM key and network access, so it is not
exercised in the automated suite -- these tests confirm the agents,
tasks, and tool wiring construct correctly, which is what tends to break
first (bad tool schemas, missing context wiring between tasks, etc.).
"""
from src.crew import CalculatorTool, FileLookupTool, WebSearchTool, build_crew


def test_tool_wrappers_delegate_to_shared_tools():
    assert CalculatorTool()._run("3 + 4") == "7"


def test_build_crew_wires_three_agents_and_tasks():
    crew = build_crew("Summarize what LangGraph is used for")
    roles = [a.role for a in crew.agents]
    assert roles == ["Researcher", "Writer", "Reviewer"]
    assert len(crew.tasks) == 3
    # writing/review tasks should carry context from earlier tasks
    assert crew.tasks[1].context == [crew.tasks[0]]
    assert crew.tasks[2].context == [crew.tasks[0], crew.tasks[1]]
