"""
room_allocator.py — Phase 3.5: Assign theory classes to classrooms.

Post-processing step after the CP-SAT solver (Phase 3).  For every
(day, period) slot that has at least one theory class scheduled, this
module uses a greedy bipartite match to assign classroom rooms.

Key facts about the data:
  - rooms.csv has 6 LAB rooms  (type="LAB")  and 6 CLASSROOM rooms
    (type="CLASSROOM", all capacity=70).
  - Lab rooms are NEVER eligible for theory classes.
  - Lab slots are P5-P6 and are already captured in room_grid from Phase 2.
  - 12 sections may be teaching simultaneously; only 6 classrooms exist,
    so up to 6 sections per slot can receive a classroom assignment.
    Unassigned slots are logged as ROOM_UNASSIGNED (non-fatal).

Output
------
  room_grid_theory : dict  room_grid_theory[section][day][period] = room_name
                           or "ROOM_UNASSIGNED"
  rows             : list  of dicts for the output CSV
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

_RA_ROOT = Path(__file__).resolve().parents[2]
if str(_RA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RA_ROOT))

import pandas as pd
from config import DAYS, PERIODS, SECTIONS, SHORT_NAMES, OUTPUT_DIR, DATA_DIR

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _is_lab_token(value) -> bool:
    return isinstance(value, str) and value.endswith("_LAB")


def _load_rooms(rooms_df: pd.DataFrame) -> List[dict]:
    """Return only CLASSROOM-type rooms, sorted by capacity descending."""
    classrooms = rooms_df[
        rooms_df["room_type"].str.upper().isin(["CLASSROOM", "LECTURE_HALL"])
    ].copy()
    classrooms = classrooms.sort_values("capacity", ascending=False)
    return classrooms.to_dict(orient="records")


def _get_lab_occupied(room_grid: dict) -> Dict[str, set]:
    """
    Build a mapping: (day, period) -> set of room_names already used for labs.
    room_grid is from Phase 2: room_grid[room_name][day][period] = token or None.
    """
    occupied: Dict[tuple, set] = {}
    for room_name, day_map in room_grid.items():
        for day, period_map in day_map.items():
            for period, token in period_map.items():
                if token is not None:
                    key = (day, period)
                    occupied.setdefault(key, set()).add(room_name)
    return occupied


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def assign_theory_rooms(
    section_grid: dict,
    room_grid: dict,
    assignment_map: dict,
    rooms_df: pd.DataFrame,
) -> tuple[dict, list]:
    """
    Assign classroom rooms to all theory slots via greedy bipartite matching.

    Parameters
    ----------
    section_grid   : section_grid[section][day][period] = course_code|None|"_LAB"
    room_grid      : room_grid[room_name][day][period]  = token|None  (Phase 2 output)
    assignment_map : assignment_map[course][section] = faculty_id
    rooms_df       : DataFrame loaded from data/rooms.csv

    Returns
    -------
    theory_room_grid : dict  theory_room_grid[section][day][period] = room_name
                             or "ROOM_UNASSIGNED" or None (lab/free)
    rows             : list  of CSV-ready dicts with keys:
                             Section, Day, Period, Course, Faculty, Room
    """
    classrooms = _load_rooms(rooms_df)
    if not classrooms:
        logger.warning("No CLASSROOM-type rooms found in rooms_df. "
                       "All theory slots will be ROOM_UNASSIGNED.")

    lab_occupied = _get_lab_occupied(room_grid)

    # Initialise output grid
    theory_room_grid: dict = {
        section: {day: {period: None for period in PERIODS} for day in DAYS}
        for section in SECTIONS
    }

    rows: list = []
    total_assigned = 0
    total_unassigned = 0

    for day in DAYS:
        for period in PERIODS:

            # ── Collect sections with a theory class this slot ────────────────
            theory_sections: list[str] = []
            for section in SECTIONS:
                value = section_grid[section][day][period]
                if value is None:
                    continue
                if _is_lab_token(value):
                    continue
                theory_sections.append(section)

            if not theory_sections:
                continue

            # ── Determine available classrooms ────────────────────────────────
            used_rooms: set = lab_occupied.get((day, period), set())
            available = [
                r for r in classrooms
                if r["room_name"] not in used_rooms
            ]

            # ── Greedy match: sections (alphabetical) → rooms (capacity desc) ─
            # Alphabetical section order is a deterministic proxy for "priority".
            # All 12 sections are the same size in this system (no enrollment data),
            # so alphabetical gives a stable, reproducible assignment.
            sorted_sections = sorted(theory_sections)

            for i, section in enumerate(sorted_sections):
                course = section_grid[section][day][period]
                faculty = assignment_map.get(course, {}).get(section, "?")
                course_short = SHORT_NAMES.get(course, course)

                if i < len(available):
                    room_name = available[i]["room_name"]
                    theory_room_grid[section][day][period] = room_name
                    total_assigned += 1
                    logger.debug(
                        "Assigned %s P%s Section %s (%s) → %s",
                        day, period, section, course_short, room_name
                    )
                else:
                    room_name = "ROOM_UNASSIGNED"
                    theory_room_grid[section][day][period] = room_name
                    total_unassigned += 1
                    logger.warning(
                        "No room available: %s P%s Section %s (%s)",
                        day, period, section, course_short
                    )

                rows.append({
                    "Section":  section,
                    "Day":      day,
                    "Period":   f"P{period}",
                    "Course":   course_short,
                    "Faculty":  faculty,
                    "Room":     room_name,
                })

    logger.info(
        "Phase 3.5 complete — assigned: %d, unassigned: %d",
        total_assigned, total_unassigned
    )
    return theory_room_grid, rows


def save_room_assignment(rows: list, output_dir: str | Path = OUTPUT_DIR) -> Path:
    """Write room_assignment.csv to output_dir and return its path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "room_assignment.csv"
    df = pd.DataFrame(
        rows,
        columns=["Section", "Day", "Period", "Course", "Faculty", "Room"]
    )
    df.to_csv(path, index=False)
    return path


def run_phase35(
    section_grid: dict,
    room_grid: dict,
    assignment_map: dict,
    data_dir: str | Path = DATA_DIR,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict:
    """
    Convenience wrapper called from run_all.py.

    Returns
    -------
    dict with keys:
      theory_room_grid, rows, csv_path,
      total_assigned, total_unassigned
    """
    rooms_df = pd.read_csv(Path(data_dir) / "rooms.csv")
    theory_room_grid, rows = assign_theory_rooms(
        section_grid=section_grid,
        room_grid=room_grid,
        assignment_map=assignment_map,
        rooms_df=rooms_df,
    )
    csv_path = save_room_assignment(rows, output_dir)

    total_assigned   = sum(1 for r in rows if r["Room"] != "ROOM_UNASSIGNED")
    total_unassigned = sum(1 for r in rows if r["Room"] == "ROOM_UNASSIGNED")

    print(f"\nPHASE 3.5 COMPLETE — Room Assignment")
    print(f"  CSV             : {csv_path}")
    print(f"  Slots assigned  : {total_assigned}")
    print(f"  Slots unassigned: {total_unassigned}")
    if total_unassigned:
        print(
            f"  ⚠  {total_unassigned} slot(s) could not be assigned a classroom "
            "(not enough CLASSROOM-type rooms for simultaneous classes)."
        )

    return {
        "theory_room_grid":  theory_room_grid,
        "rows":              rows,
        "csv_path":          str(csv_path),
        "total_assigned":    total_assigned,
        "total_unassigned":  total_unassigned,
    }


if __name__ == "__main__":
    # Standalone test: requires run_all.py to have been run first
    # (loads grids from Phase 2 + Phase 3 via full pipeline)
    from src.phase2.lab_scheduler import lock_labs
    from src.phase3.theory_scheduler import solve_theory
    from src.phase1.assignment_builder import build_assignment_map

    assignment_map = build_assignment_map()
    sec_grid, fac_grid, rm_grid, lab_details = lock_labs()
    result = solve_theory(
        assignment_map=assignment_map,
        section_grid=sec_grid,
        faculty_grid=fac_grid,
        room_grid=rm_grid,
        lab_details=lab_details,
    )
    run_phase35(
        section_grid=result["section_grid"],
        room_grid=result["room_grid"],
        assignment_map=result["assignment_map"],
    )
