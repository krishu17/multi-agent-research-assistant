import os

from src.tools import calculator, file_lookup, web_search


def test_calculator_basic():
    assert calculator.invoke("12 * (4 + 3)") == "84"


def test_calculator_rejects_bad_input():
    result = calculator.invoke("__import__('os').system('echo hi')")
    assert result.startswith("error:")


def test_file_lookup_finds_match(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools._KNOWLEDGE_DIR", str(tmp_path))
    (tmp_path / "notes.txt").write_text("LangGraph powers the planner agent.\n")
    result = file_lookup.invoke("planner agent")
    assert "notes.txt" in result
    assert "planner agent" in result


def test_file_lookup_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools._KNOWLEDGE_DIR", str(tmp_path))
    (tmp_path / "notes.txt").write_text("Nothing relevant here.\n")
    result = file_lookup.invoke("quantum entanglement")
    assert "No matching" in result


def test_web_search_handles_network_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("network disabled in test")

    monkeypatch.setattr("ddgs.DDGS.text", _boom)
    result = web_search.invoke("anything")
    assert result.startswith("error:")
