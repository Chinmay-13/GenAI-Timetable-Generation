"""
loader.py — Semester-aware data loader for the timetable system.

Public API
----------
load_all(sem_id=None)
    Load courses, faculty, assignments, lab_allotment, rooms from the
    correct directory.  Returns the same 5-tuple as before so all
    existing callers continue to work with zero changes.

load_elective_slots(sem_id=None)
    Load elective_slots.csv for semesters that have electives.
    Returns None when the file doesn't exist (e.g. cse_sem3).

get_sem_metadata(sem_id=None)
    Return a small summary dict describing a semester's contents.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── Resolve project root so we can import config ──────────────────────────────
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]          # …/timetable_system/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_sem_paths, DATA_DIR, SECTIONS   # noqa: E402

# Legacy flat-file directory (kept for backward compatibility)
_LEGACY_DATA_DIR = DATA_DIR               # same as DATA_DIR in config


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_data_dir(sem_id: str | None) -> Path:
    """Return the data directory for *sem_id*, or the legacy dir if None."""
    if sem_id is not None:
        return get_sem_paths(sem_id).data_dir
    return _LEGACY_DATA_DIR


def _as_bool_series(series: pd.Series) -> pd.Series:
    """Convert a string/mixed column to proper booleans."""
    return series.map(
        lambda v: str(v).strip().lower() in {"true", "1", "yes", "y"}
        if not isinstance(v, bool)
        else v
    )


def _enrich_courses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure courses DataFrame has is_elective and elective_group columns.
    Safe to call on legacy CSVs that don't have these columns yet.
    """
    if "is_elective" not in df.columns:
        df["is_elective"] = False
    else:
        df["is_elective"] = _as_bool_series(df["is_elective"])

    if "elective_group" not in df.columns:
        df["elective_group"] = "none"
    else:
        df["elective_group"] = df["elective_group"].fillna("none").astype(str).str.strip()

    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_all(sem_id: str | None = None):
    """
    Load all core CSV files for a semester.

    Parameters
    ----------
    sem_id : str or None
        Semester slug (e.g. "cse_sem5").  If None, loads from the legacy
        flat ``data/`` directory — existing callers are unaffected.

    Returns
    -------
    tuple: (courses, faculty, assignments, lab_allotment, rooms)
        Same 5-tuple as before; all are pandas DataFrames.
    """
    base = _resolve_data_dir(sem_id)
    label = sem_id or "legacy"

    courses       = _enrich_courses(pd.read_csv(base / "courses.csv"))
    faculty       = pd.read_csv(base / "faculty.csv")
    assignments   = pd.read_csv(base / "assignments.csv")
    lab_allotment = pd.read_csv(base / "lab_allotment.csv")
    rooms         = pd.read_csv(base / "rooms.csv")

    print(f"=== DATA LOAD SUMMARY [{label}] ===")
    print(f"  Courses loaded       : {len(courses)}")
    print(f"  Faculty loaded       : {len(faculty)}")
    print(f"  Assignments loaded   : {len(assignments)}")
    print(f"  Lab allotments loaded: {len(lab_allotment)}")
    print(f"  Rooms loaded         : {len(rooms)}")
    print(f"  Elective courses     : {df_count(courses, 'is_elective', True)}")
    print("=" * 36)

    return courses, faculty, assignments, lab_allotment, rooms


def df_count(df: pd.DataFrame, col: str, value) -> int:
    """Utility: count rows where df[col] == value."""
    if col not in df.columns:
        return 0
    return int((df[col] == value).sum())


# Expected columns for elective_slots.csv
_ELECTIVE_SLOTS_COLS = {
    "elective_group", "course_code", "day",
    "period_start", "period_end", "room",
    "faculty_id", "enrolled_sections",
}


def load_elective_slots(sem_id: str | None = None) -> pd.DataFrame | None:
    """
    Load elective_slots.csv for the given semester.

    Parameters
    ----------
    sem_id : str or None
        Semester slug.  If None (legacy mode), returns None immediately.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with 8 validated columns, or None if the file doesn't
        exist (e.g. cse_sem3 has no electives).

    Raises
    ------
    ValueError
        If the file exists but is missing expected columns.
    """
    if sem_id is None:
        return None

    base = _resolve_data_dir(sem_id)
    fpath = base / "elective_slots.csv"

    if not fpath.exists():
        return None          # semester has no electives — silent, no error

    df = pd.read_csv(fpath)

    # Validate columns
    missing = _ELECTIVE_SLOTS_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"elective_slots.csv for '{sem_id}' is missing columns: "
            + ", ".join(sorted(missing))
        )

    # Type coercions
    df["period_start"] = df["period_start"].astype(int)
    df["period_end"]   = df["period_end"].astype(int)
    df["enrolled_sections"] = df["enrolled_sections"].astype(str).str.strip()

    return df


def get_sem_metadata(sem_id: str | None = None) -> dict:
    """
    Return a summary dict describing what a semester contains.

    Parameters
    ----------
    sem_id : str or None
        Semester slug, or None for legacy mode.

    Returns
    -------
    dict with keys:
        sem_id, has_electives, elective_groups, sections,
        num_courses, num_faculty
    """
    base  = _resolve_data_dir(sem_id)
    label = sem_id or "legacy"

    courses = _enrich_courses(pd.read_csv(base / "courses.csv"))
    faculty = pd.read_csv(base / "faculty.csv")

    elective_df   = load_elective_slots(sem_id)
    has_electives = elective_df is not None and not elective_df.empty

    elective_groups: list[str] = []
    if has_electives:
        elective_groups = sorted(elective_df["elective_group"].unique().tolist())

    return {
        "sem_id":          label,
        "has_electives":   has_electives,
        "elective_groups": elective_groups,
        "sections":        list(SECTIONS),
        "num_courses":     len(courses),
        "num_faculty":     len(faculty),
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("\n" + "─" * 50)
    print("SMOKE TEST: load_all()")
    load_all()

    print("\n" + "─" * 50)
    print("SMOKE TEST: load_all('cse_sem3')")
    load_all("cse_sem3")

    print("\n" + "─" * 50)
    print("SMOKE TEST: load_all('cse_sem5')")
    load_all("cse_sem5")

    print("\n" + "─" * 50)
    print("SMOKE TEST: load_elective_slots('cse_sem3')  → expect None")
    result = load_elective_slots("cse_sem3")
    print(f"  Result: {result}")

    print("\n" + "─" * 50)
    print("SMOKE TEST: load_elective_slots('cse_sem5')  → expect 12 rows")
    df = load_elective_slots("cse_sem5")
    if df is not None:
        print(f"  Rows: {len(df)}")
        print(df.to_string(index=False))
    else:
        print("  Result: None")

    print("\n" + "─" * 50)
    print("SMOKE TEST: get_sem_metadata('cse_sem3')")
    meta3 = get_sem_metadata("cse_sem3")
    print(json.dumps(meta3, indent=2))

    print("\n" + "─" * 50)
    print("SMOKE TEST: get_sem_metadata('cse_sem5')")
    meta5 = get_sem_metadata("cse_sem5")
    print(json.dumps(meta5, indent=2))
