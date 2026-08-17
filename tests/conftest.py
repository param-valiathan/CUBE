"""Shared pytest fixtures for the CUBE test suite.

Fixture generators are added incrementally as later phases need them
(synthetic DLC-format DataFrames, tmp_path file trees, small .h5/.csv
files). Nothing here touches real user data or CUBE_logs/.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
