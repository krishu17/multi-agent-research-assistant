import os

os.environ["LLM_PROVIDER"] = "mock"

from src.graph import build_graph, run, run_subtask_node  # noqa: E402


def test_run_subtask_calculator_path():
    out = run_subtask_node({"request": "x", "subtask": "What is 3 + 4"})
    result = out["subtask_results"][0]
    assert "7" in result["answer"]
    assert any(t.startswith("calculator") for t in result["tool_trace"])


def test_run_subtask_file_lookup_path(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools._KNOWLEDGE_DIR", str(tmp_path))
    (tmp_path / "notes.txt").write_text("CrewAI coordinates role-based agents.\n")
    out = run_subtask_node({"request": "x", "subtask": "summarize CrewAI"})
    result = out["subtask_results"][0]
    assert "CrewAI" in result["answer"]


def test_full_graph_end_to_end():
    final_state = run("What is 5 * 6")
    assert final_state["subtasks"]
    assert final_state["subtask_results"]
    assert final_state["final_answer"]
    assert "30" in str(final_state["subtask_results"])


def test_graph_compiles():
    app = build_graph()
    assert app is not None
