from __future__ import annotations

from pathlib import Path


def default_db_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root
        / "handoff_database_completed_20260729"
        / "dietrecommendation_no_empty_enhanced.sqlite",
        project_root.parent
        / "handoff_database_completed_20260729"
        / "dietrecommendation_no_empty_enhanced.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DB_PATH = default_db_path()
