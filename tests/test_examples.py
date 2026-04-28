"""Check that example files run without crashing."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_examples_run() -> None:
    """Run every example file."""
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"

    example_files = sorted(examples_dir.glob("*.py"))

    assert example_files, "No example files found."

    for example_file in example_files:
        if example_file.name.startswith("_"):
            continue

        runpy.run_path(str(example_file), run_name="__main__")