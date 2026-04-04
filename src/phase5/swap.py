from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.substitute import (
    DAYS,
    MAX_HOURS,
    faculty_lookup,
    get_faculty_day_row,
    get_faculty_load,
    normalize_course,
    normalize_day,
    parse_timetable_cell,
)

PERIOD_COLUMNS = [f"P{i}" for i in range(1, 7)]


def find_swap_slot(
    faculty_a_id: str,
    faculty_b_id: str,
    return_day: str,
    course_code: str,
) -> Dict[str, object]:
    faculty_a = str(faculty_a_id).strip().upper()
    faculty_b = str(faculty_b_id).strip().upper()
    day = normalize_day(return_day)
    course_info = normalize_course(course_code)
    lookup = faculty_lookup()

    if faculty_a not in lookup:
        raise ValueError(f"Unknown faculty_id: {faculty_a}")
    if faculty_b not in lookup:
        raise ValueError(f"Unknown faculty_id: {faculty_b}")
    if course_info is None:
        raise ValueError(f"Unknown course: {course_code}")

    day_index = DAYS.index(day)
    search_days = DAYS[day_index + 1 :]

    original_load_a = get_faculty_load(faculty_a)
    original_load_b = get_faculty_load(faculty_b)
    load_before_swap_a = max(0, original_load_a - 1)
    load_before_swap_b = original_load_b + 1
    max_hours_a = MAX_HOURS.get(lookup[faculty_a]["designation"], 16)
    max_hours_b = MAX_HOURS.get(lookup[faculty_b]["designation"], 16)

    for search_day in search_days:
        row_a = get_faculty_day_row(faculty_a, search_day)
        row_b = get_faculty_day_row(faculty_b, search_day)

        for period in PERIOD_COLUMNS:
            slot_b = parse_timetable_cell(row_b[period])
            if not slot_b:
                continue
            if slot_b.get("course_code") != course_info["course_code"]:
                continue
            if str(row_a[period]).strip() != "----":
                continue

            return {
                "swap_found": True,
                "swap_day": search_day,
                "swap_period": period,
                "course": course_info["course_short"],
                "course_code": course_info["course_code"],
                "section": slot_b.get("section", ""),
                "faculty_a": faculty_a,
                "faculty_a_name": lookup[faculty_a]["name"],
                "faculty_b": faculty_b,
                "faculty_b_name": lookup[faculty_b]["name"],
                "faculty_a_load_before": load_before_swap_a,
                "faculty_b_load_before": load_before_swap_b,
                "faculty_a_load_after": original_load_a,
                "faculty_b_load_after": original_load_b,
                "faculty_a_max": max_hours_a,
                "faculty_b_max": max_hours_b,
                "result": (
                    f"{faculty_a} takes back {period} {search_day} from {faculty_b}. "
                    "Weekly loads restored."
                ),
            }

    return {
        "swap_found": False,
        "swap_day": None,
        "swap_period": None,
        "course": course_info["course_short"],
        "course_code": course_info["course_code"],
        "section": None,
        "faculty_a": faculty_a,
        "faculty_a_name": lookup[faculty_a]["name"],
        "faculty_b": faculty_b,
        "faculty_b_name": lookup[faculty_b]["name"],
        "faculty_a_load_before": load_before_swap_a,
        "faculty_b_load_before": load_before_swap_b,
        "faculty_a_load_after": load_before_swap_a,
        "faculty_b_load_after": load_before_swap_b,
        "faculty_a_max": max_hours_a,
        "faculty_b_max": max_hours_b,
        "result": "No swap possible this week. Substitute keeps the extra class.",
    }


def print_swap_plan(result: Dict[str, object]) -> None:
    print("═" * 50)
    print("SWAP PLAN - Load Restoration")
    print("═" * 50)
    print()
    print(f"Returning faculty : {result['faculty_a_name']} ({result['faculty_a']})")
    print(f"Substitute        : {result['faculty_b_name']} ({result['faculty_b']})")
    print()

    if result["swap_found"]:
        print("SWAP SLOT FOUND:")
        print(
            f"{result['swap_day']} {result['swap_period']} - {result['course']} "
            f"(Section {result['section']})"
        )
        print(f"{result['faculty_a']} takes this class back from {result['faculty_b']}.")
        print(f"{result['faculty_b']} is now free at {result['swap_day']} {result['swap_period']}.")
        print()
        print("Weekly load after swap:")
        print(
            f"  {result['faculty_a']}: {result['faculty_a_load_after']}h/{result['faculty_a_max']}h max"
        )
        print(
            f"  {result['faculty_b']}: {result['faculty_b_load_after']}h/{result['faculty_b_max']}h max"
        )
    else:
        print("NO SWAP POSSIBLE THIS WEEK")
        print(f"{result['faculty_b']} carries the extra class for this week only.")
        print("No action needed - load returns to normal next week.")

    print()
    print("═" * 50)


def commit_swap(swap_result: Dict[str, object], reason: str = "", sem_id: str = None) -> dict:
    """
    Commit a confirmed load-swap to the timetable.

    Accepts the dict returned by ``find_swap_slot()`` when ``swap_found`` is True
    and routes the write through ``sync_manager.commit_schedule_change()`` so that
    the section CSV, faculty CSVs, summary_report.txt, and the RAG index are all
    updated atomically.

    Parameters
    ----------
    swap_result : dict  (from find_swap_slot, must have swap_found=True)
    reason      : str   optional free-text reason for the log
    sem_id      : str or None  — semester slug; passed to commit_schedule_change
                  so writes land in the correct semester output directory.

    Returns
    -------
    dict  as returned by sync_manager.commit_schedule_change()

    Raises
    ------
    ValueError   if swap_result has swap_found=False or is missing required keys.
    RuntimeError on write failure (all changes already rolled back by sync_manager).
    """
    if not swap_result.get("swap_found"):
        raise ValueError("commit_swap: nothing to commit — swap_found is False.")

    required = {"swap_day", "swap_period", "section", "faculty_a", "faculty_b"}
    missing = required - swap_result.keys()
    if missing:
        raise ValueError(f"commit_swap: swap_result missing keys {missing}")

    # Parse period string "P3" → int 3
    period_str: str = str(swap_result["swap_period"])  # e.g. "P3"
    if period_str.upper().startswith("P"):
        period_int = int(period_str[1:])
    else:
        period_int = int(period_str)

    from src.phase5.sync_manager import commit_schedule_change
    return commit_schedule_change(
        {
            "section":          str(swap_result["section"]),
            "day":              str(swap_result["swap_day"]),
            "period_start":     period_int,
            "period_end":       period_int,        # swaps are single-period
            # faculty_b was covering the slot; faculty_a reclaims it
            "original_faculty": str(swap_result["faculty_b"]),
            "new_faculty":      str(swap_result["faculty_a"]),
            "change_type":      "swap",
            "reason":           reason or swap_result.get("result", ""),
        },
        sem_id=sem_id,
    )


def main() -> None:
    try:
        faculty_a = input("Enter returning faculty: ").strip()
        faculty_b = input("Enter substitute faculty: ").strip()
        return_day = input("Enter return day: ").strip()
        course_code = input("Enter course: ").strip()
        result = find_swap_slot(faculty_a, faculty_b, return_day, course_code)
        print_swap_plan(result)
    except Exception as exc:
        print(f"Swap planning failed: {exc}")


if __name__ == "__main__":
    main()
