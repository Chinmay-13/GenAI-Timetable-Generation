"""
Dynamic system prompt builder.
Reads structural facts from CSVs and config at call time.
Kept compact (<400 tokens) for Groq free tier compatibility.
Full data is accessed via tools, not injected into prompt.
"""
import csv
from pathlib import Path
from typing import Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def _read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_system_prompt(
    outputs_dir: Path = None,
    data_dir: Path = None,
    summary_text: Optional[str] = None,
    sem_id: Optional[str] = None,
) -> str:
    """
    Builds a compact system prompt under 800 tokens.
    Full data is accessed via tools, not injected into prompt.

    Parameters
    ----------
    sem_id : str or None
        Semester slug (e.g. "cse_sem5").  If given, data is loaded from
        the correct semester directory.  If None, falls back to the
        explicit data_dir/outputs_dir arguments (legacy behaviour).
    """
    # ── Resolve data_dir from sem_id if provided ──────────────────────────────
    if sem_id is not None:
        from config import get_sem_paths
        paths = get_sem_paths(sem_id)
        data_dir   = paths.data_dir
        outputs_dir = paths.output_dir
        sem_label  = sem_id.replace("_", " ").title()
    else:
        # Legacy: use whatever was passed in, fall back to config defaults
        if data_dir is None:
            data_dir = config.DATA_DIR
        if outputs_dir is None:
            outputs_dir = config.OUTPUT_DIR
        sem_label = "CSE · 3rd Semester (legacy)"

    courses = _read_csv(data_dir / "courses.csv")
    faculty = _read_csv(data_dir / "faculty.csv")
    labs    = _read_csv(data_dir / "lab_allotment.csv")

    # Compact course list — just code + name (skip electives to save tokens)
    course_lines = [
        f"{c['course_code']}: {c['course_name']}"
        for c in courses
        if str(c.get("is_elective", "False")).strip().lower() not in ("true", "1", "yes")
    ]

    # Compact faculty list — just ID + name + designation initial
    desig_short = {
        "Professor": "Prof",
        "Associate Professor": "Asso",
        "Assistant Professor": "Asst",
        "Prof": "Prof",
        "Asso Prof": "Asso",
        "Asst Prof": "Asst",
    }
    faculty_lines = [
        f"{f['faculty_id']}: {f['name']} "
        f"({desig_short.get(f['designation'], f['designation'])})"
        for f in faculty
    ]

    # Lab days only — compact
    lab_lines = [
        f"{l['course_code']} {l['section_pair']} {l['day']}"
        for l in labs
    ]

    return f"""University timetable system — {sem_label}.
Sections: A-L (12 total). Periods: P1-P4 theory, P5-P6 labs.
Days: Monday-Friday.

Courses: {', '.join(course_lines)}

Faculty: {', '.join(faculty_lines)}

Lab schedule: {', '.join(lab_lines)}

Weekly caps: Professor=12h, Associate=16h, Assistant=20h.
Labs are 2-period blocks (P5-P6). Professors cannot take labs.
"""
