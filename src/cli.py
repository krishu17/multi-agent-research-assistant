"""Command-line entry point.

    python -m src.cli langgraph "your question"
    python -m src.cli crewai "your question"
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python -m src.cli [langgraph|crewai] <question>")
        raise SystemExit(1)

    mode, question = sys.argv[1], " ".join(sys.argv[2:])

    if mode == "langgraph":
        from .graph import run

        result = run(question)
        print(json.dumps(result, indent=2))
    elif mode == "crewai":
        from .crew import run

        print(run(question))
    else:
        print(f"unknown mode: {mode!r} (expected 'langgraph' or 'crewai')")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
