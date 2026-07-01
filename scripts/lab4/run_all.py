#!/usr/bin/env python3
"""Run all Lab 4 clustering scripts."""

from __future__ import annotations

import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in sorted(HERE.glob("[0-9][0-9]_*.py")):
    print("\n" + "=" * 72)
    print(f"Running {script.name}")
    print("=" * 72)
    runpy.run_path(str(script), run_name="__main__")
