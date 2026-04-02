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
    outputs_dir: Path,
    data_dir: Path,
    summary_text: Optional[str] = None,
) -> str:
    """
    Builds a compact system prompt under 800 tokens.
    Full data is accessed via tools, not injected into prompt.
    """
    courses = _read_csv(data_dir / "courses.csv")
    faculty = _read_csv(data_dir / "faculty.csv")
    labs = _read_csv(data_dir / "lab_allotment.csv")

    # Compact course list — just code + name
    course_lines = [
        f"{c['course_code']}: {c['course_name']}"
        for c in courses
    ]

    # Compact faculty list — just ID + name + designation initial
    desig_short = {
        "Professor": "Prof",
        "Associate Professor": "Asso",
        "Assistant Professor": "Asst",
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

    return f"""University timetable system — CSE Dept, 3rd Semester.
Sections: A-L (12 total). Periods: P1-P4 theory, P5-P6 labs.
Days: Monday-Friday.

Courses: {', '.join(course_lines)}

Faculty: {', '.join(faculty_lines)}

Lab schedule: {', '.join(lab_lines)}

Weekly caps: Professor=12h, Associate=16h, Assistant=20h.
Labs are 2-period blocks (P5-P6). Professors cannot take labs.
"""
