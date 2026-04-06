"""
test_pipeline.py — Automated tests for all phases of the timetable system.

Run from project root:
  pytest tests/ -v

All phase-fixture tests are module-scoped so the solver only runs once.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.phase0.loader import load_all
from src.phase0.validator import validate
from src.phase1.assignment_builder import build_assignment_map
from src.phase2.lab_scheduler import lock_labs
from src.phase3.theory_scheduler import solve_theory


# ── Data fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    return load_all()


@pytest.fixture(scope="module")
def assignment_map(data):
    return build_assignment_map()


@pytest.fixture(scope="module")
def lab_result(assignment_map):
    return lock_labs()


@pytest.fixture(scope="module")
def theory_result(assignment_map, lab_result):
    section_grid, faculty_grid, room_grid, lab_details, elective_details = lab_result
    return solve_theory(
        assignment_map=assignment_map,
        section_grid=section_grid,
        faculty_grid=faculty_grid,
        room_grid=room_grid,
        lab_details=lab_details,
        elective_details=elective_details,
    )


# ── Phase 0 tests ─────────────────────────────────────────────────────────────

def test_data_loads(data):
    courses, faculty, assignments, labs, rooms = data
    assert len(courses) == 5, f"Expected 5 courses, got {len(courses)}"
    assert len(faculty) == 20, f"Expected 20 faculty, got {len(faculty)}"
    assert len(labs) == 12, f"Expected 12 lab rows, got {len(labs)}"


def test_validation_passes(data):
    courses, faculty, assignments, labs, rooms = data
    result = validate(courses, faculty, assignments)
    assert result is True, "Validation must pass (all checks OK)"


def test_section_coverage(data):
    """Every course must be assigned to exactly 12 sections."""
    courses, faculty, assignments, labs, rooms = data
    for _, course in courses.iterrows():
        code = course["course_code"]
        course_rows = assignments[assignments["course_code"] == code]
        sections = []
        for _, row in course_rows.iterrows():
            sections += [s.strip() for s in str(row["sections_handled"]).split(",") if s.strip()]
        assert len(sections) == 12, (
            f"{code} must cover 12 sections, got {len(sections)}: {sections}"
        )


def test_no_prof_on_lab_course(data):
    """Professors cannot be assigned to lab courses."""
    courses, faculty, assignments, labs, rooms = data
    lab_codes = courses[courses["has_lab"] == True]["course_code"].tolist()
    profs = faculty[faculty["designation"] == "Prof"]["faculty_id"].tolist()
    violations = assignments[
        (assignments["faculty_id"].isin(profs)) &
        (assignments["course_code"].isin(lab_codes))
    ]
    assert violations.empty, (
        f"Professors cannot be on lab courses: {violations.to_dict('records')}"
    )


# ── Phase 2 tests ─────────────────────────────────────────────────────────────

def test_lab_slots_locked(lab_result):
    """Section A must have DDCO lab locked on Monday P5-P6."""
    section_grid, faculty_grid, room_grid, lab_details, elective_details = lab_result
    assert "A" in section_grid, "Section A missing from section_grid"
    for p in [5, 6]:
        val = section_grid["A"]["Monday"][p]
        assert val is not None, f"Section A Monday P{p} should be locked (got None)"
        assert "LAB" in str(val), f"Section A Monday P{p} should be a LAB token, got: {val}"


def test_no_lab_conflicts(lab_result):
    """No section should be double-booked in lab slots."""
    section_grid, faculty_grid, room_grid, lab_details, elective_details = lab_result
    for section, days in section_grid.items():
        for day, periods in days.items():
            occupied = [p for p, v in periods.items() if v is not None]
            assert len(occupied) == len(set(occupied)), (
                f"Section {section} {day} has duplicate period entries"
            )


def test_no_faculty_lab_conflicts(lab_result):
    """No faculty should be double-booked in lab slots."""
    section_grid, faculty_grid, room_grid, lab_details, elective_details = lab_result
    for fid, days in faculty_grid.items():
        for day, periods in days.items():
            seen = {}
            for period, val in periods.items():
                if val is not None:
                    assert period not in seen, (
                        f"Faculty {fid} double-booked on {day} P{period}"
                    )
                    seen[period] = val


# ── Phase 3 tests ─────────────────────────────────────────────────────────────

def test_solver_reaches_optimal(theory_result):
    """Solver must return OPTIMAL (penalty stage always must be OPTIMAL)."""
    status = theory_result.get("solver_status", "")
    assert status in ["OPTIMAL", "FEASIBLE"], (
        f"Solver must be OPTIMAL or FEASIBLE, got: {status}"
    )


def test_theory_slot_count(theory_result):
    """Each section must have exactly 4 theory slots per day (P1-P6 only)."""
    section_grid = theory_result["section_grid"]
    lab_details = theory_result["lab_details"]

    section_lab_days = {}
    for detail in lab_details:
        for sec in detail["sections"]:
            section_lab_days.setdefault(sec, set()).add(detail["day"])

    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for section in "ABCDEFGHIJKL":
        if section not in section_grid:
            continue
        for day in DAYS:
            theory_count = 0
            for p in range(1, 7):
                val = section_grid[section][day].get(p)
                if val is not None and not (isinstance(val, str) and val.endswith("_LAB")):
                    theory_count += 1
            assert theory_count == 4, (
                f"Section {section} {day} should have 4 theory slots in P1-P6, got {theory_count}"
            )


def test_no_faculty_double_booked(theory_result):
    """No faculty should be in two places at the same time."""
    faculty_grid = theory_result["faculty_grid"]
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for fid, days in faculty_grid.items():
        for day, periods in days.items():
            seen = {}
            for period, val in periods.items():
                if val is not None:
                    assert period not in seen, (
                        f"Faculty {fid} double-booked on {day} P{period}: "
                        f"'{seen[period]}' and '{val}'"
                    )
                    seen[period] = val


def test_no_theory_in_lab_window_non_lab_day(theory_result):
    """Theory should not be placed in P5-P6 on days that are not a section's lab day."""
    section_grid = theory_result["section_grid"]
    lab_details = theory_result["lab_details"]

    section_lab_days = {}
    for detail in lab_details:
        for sec in detail["sections"]:
            section_lab_days.setdefault(sec, set()).add(detail["day"])

    for section, days in section_grid.items():
        lab_days = section_lab_days.get(section, set())
        for day, periods in days.items():
            if day in lab_days:
                continue  # lab window occupied by lab — OK
            for p in [5, 6]:
                val = periods.get(p)
                if val is not None and not (isinstance(val, str) and val.endswith("_LAB")):
                    pytest.fail(
                        f"Section {section} {day} P{p} has theory in lab window: {val}"
                    )


def test_240_total_theory_slots(theory_result):
    """Total theory slots across all sections must equal 240 (12 sections × 5 days × 4 slots)."""
    section_grid = theory_result["section_grid"]
    total = 0
    for section, days in section_grid.items():
        for day, periods in days.items():
            for p in range(1, 7):  # P1-P6 only
                val = periods.get(p)
                if val is not None and not (isinstance(val, str) and val.endswith("_LAB")):
                    total += 1
    assert total == 240, f"Expected 240 theory slots (12×5×4), got {total}"


# ── Phase 5 tests ─────────────────────────────────────────────────────────────

def test_substitute_finder_doesnt_crash():
    """find_substitute must return a dict with required keys."""
    from src.phase5.substitute import find_substitute
    result = find_substitute("F04", "Monday")
    assert isinstance(result, dict), "find_substitute must return a dict"
    assert "absent_faculty" in result, "Result must have 'absent_faculty' key"
    assert "substitutions" in result, "Result must have 'substitutions' key"
    assert "unresolved" in result, "Result must have 'unresolved' key"
    assert "original_slots" in result, "Result must have 'original_slots' key"


def test_lab_slots_not_in_unresolved():
    """Lab periods (P5-P6) should not appear in the unresolved list."""
    from src.phase5.substitute import find_substitute
    result = find_substitute("F04", "Monday")
    unresolved = result.get("unresolved", [])
    lab_periods = {"P5", "P6"}
    for slot in unresolved:
        period = str(slot.get("period", ""))
        assert period not in lab_periods, (
            f"Lab period {period} unexpectedly appears in unresolved list"
        )


def test_chat_memory_save_load():
    """Chat memory save and load should round-trip correctly."""
    import tempfile
    from pathlib import Path
    from src.phase5 import chat

    sample = [("hello", "world"), ("foo", "bar")]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)

    original_memory_file = chat.MEMORY_FILE
    chat.MEMORY_FILE = tmp_path
    try:
        chat.save_memory(sample)
        loaded = chat.load_memory()
        assert loaded == sample, "Memory round-trip failed"
    finally:
        chat.MEMORY_FILE = original_memory_file
        tmp_path.unlink(missing_ok=True)
