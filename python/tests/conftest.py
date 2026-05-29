"""Pytest configuration for the FAA-classifier test suite.

Adds the `python/` directory (this file's parent's parent) to sys.path so
tests can `from pipeline.X import Y` exactly as the production scripts do
(production sets PYTHONPATH=<repo>/python in classify.sh). Keeping the path
fix here means `pytest` works without callers having to export PYTHONPATH.
"""
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parent.parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

# Fixtures directory, importable by tests via `from conftest import FIXTURES`.
FIXTURES = Path(__file__).resolve().parent / "fixtures"
