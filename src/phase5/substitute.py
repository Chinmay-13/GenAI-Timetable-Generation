from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MAX_HOURS = {"Prof": 12, "Asso Prof": 16, "Asst Prof": 20}
SHORT_NAMES = {
    "UE24CS251A": "DDCO",
    "UE24CS252A": "DSA",
    "UE24MA242A": "MATH",
    "UE24CS242A": "WT",
    "UE24CS243A": "AFLL",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
PERIOD_COLUMNS = [f"P{i}" for i in range(1, 10)]
DAY_ALIASES = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
}
DESIGNATION_RANK = {"Asst Prof": 1, "Asso Prof": 2, "Prof": 3}
SHORT_NAME_TO_CODE = {short_name: code for code, short_name in SHORT_NAMES.items()}
CELL_PATTERN = re.compile(
    r"^(?P<course>[A-Z0-9]+)(?:\s+LAB)?\s*\((?P<section>[A-Z](?:\+[A-Z])*)\)$"
)


def normalize_day(day: str) -> str:
    normalized = DAY_ALIASES.get(str(day).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported day: {day}")
    return normalized


def normalize_course(course_value: str) -> Optional[Dict[str, str]]:
    value = str(course_value).strip().upper()
    if not value:
        return None
    if value in SHORT_NAME_TO_CODE:
        return {"course_code": SHORT_NAME_TO_CODE[value], "course_short": value}
    if value in SHORT_NAMES:
        return {"course_code": value, "course_short": SHORT_NAMES[value]}
    return None


def parse_timetable_cell(value: object) -> Optional[Dict[str, object]]:
    text = str(value).strip()
    if text in {"", "----", "nan", "None"}:
        return None

    match = CELL_PATTERN.match(text)
    if not match:
        return {
            "raw": text,
            "course": text,
            "course_short": text,
            "course_code": None,
            "section": "",
            "is_lab": "LAB" in text,
        }

    course_short = match.group("course")
    is_lab = " LAB " in f" {text} "
    course_info = normalize_course(course_short)
    return {
        "raw": text,
        "course": course_short,
        "course_short": course_short,
        "course_code": course_info["course_code"] if course_info else None,
        "section": match.group("section"),
        "is_lab": is_lab,
    }


@lru_cache(maxsize=1)
def load_faculty_metadata() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "faculty.csv")
    for column in ["faculty_id", "name", "designation"]:
        df[column] = df[column].astype(str).str.strip()
    return df


@lru_cache(maxsize=1)
def load_assignment_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "assignments.csv")
    for column in ["faculty_id", "course_code", "sections_handled"]:
        df[column] = df[column].astype(str).str.strip()
    return df


@lru_cache(maxsize=1)
def load_course_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "courses.csv")
    df["course_code"] = df["course_code"].astype(str).str.strip()
    return df


@lru_cache(maxsize=1)
def faculty_lookup() -> Dict[str, Dict[str, str]]:
    return load_faculty_metadata().set_index("faculty_id").to_dict("index")


@lru_cache(maxsize=1)
def assignments_by_faculty() -> Dict[str, set]:
    grouped: Dict[str, set] = {}
    for _, row in load_assignment_data().iterrows():
        grouped.setdefault(row["faculty_id"], set()).add(row["course_code"])
    return grouped


@lru_cache(maxsize=1)
def all_faculty_ids() -> List[str]:
    return load_faculty_metadata()["faculty_id"].tolist()


@lru_cache(maxsize=None)
def load_faculty_timetable(faculty_id: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"faculty_{faculty_id}_timetable.csv"
    if not path.exists():
        raise FileNotFoundError(f"Faculty timetable not found for {faculty_id}: {path}")
    df = pd.read_csv(path)
    df["Day"] = df["Day"].astype(str).str.strip()
    return df


@lru_cache(maxsize=None)
def load_section_timetable(section: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"section_{section}_timetable.csv"
    if not path.exists():
        raise FileNotFoundError(f"Section timetable not found for {section}: {path}")
    df = pd.read_csv(path)
    df["Day"] = df["Day"].astype(str).str.strip()
    return df


def get_faculty_day_row(faculty_id: str, day: str) -> pd.Series:
    df = load_faculty_timetable(faculty_id)
    day_df = df[df["Day"].str.lower() == day.lower()]
    if day_df.empty:
        raise ValueError(f"{day} not found in faculty timetable for {faculty_id}")
    return day_df.iloc[0]


def get_faculty_load(faculty_id: str) -> int:
    df = load_faculty_timetable(faculty_id)
    return int((df[PERIOD_COLUMNS] != "----").sum().sum())


def build_load_snapshot() -> Dict[str, Dict[str, object]]:
    lookup = faculty_lookup()
    snapshot: Dict[str, Dict[str, object]] = {}
    for faculty_id in all_faculty_ids():
        designation = lookup[faculty_id]["designation"]
        max_hours = MAX_HOURS.get(designation, 16)
        total_hours = get_faculty_load(faculty_id)
        snapshot[faculty_id] = {
            "name": lookup[faculty_id]["name"],
            "designation": designation,
            "total_hours": total_hours,
            "max_hours": max_hours,
            "status": "OK" if total_hours <= max_hours else "OVERLOAD",
        }
    return snapshot


def _candidate_priority(absent_designation: str, candidate_designation: str) -> int:
    if candidate_designation == absent_designation:
        return 2
    return 3


def _candidate_reason(day: str, period: str, match_type: str, course_short: str) -> str:
    if match_type == "same_course":
        return f"free at {period} {day}, teaches {course_short}"
    if match_type == "same_designation":
        return f"free at {period} {day}, same designation"
    if match_type == "available":
        return f"free at {period} {day}, available within load limit"
    return f"free at {period} {day}"


def _rank_candidates(
    absent_faculty_id: str,
    absent_designation: str,
    day: str,
    period: str,
    slot_info: Dict[str, object],
) -> List[Dict[str, object]]:
    loads = build_load_snapshot()
    faculty_info = faculty_lookup()
    teaching_map = assignments_by_faculty()
    required_course_code = slot_info.get("course_code")
    is_lab = bool(slot_info.get("is_lab"))
    candidates: List[Dict[str, object]] = []

    for faculty_id in all_faculty_ids():
        if faculty_id == absent_faculty_id:
            continue

        info = faculty_info[faculty_id]
        candidate_designation = info["designation"]
        if is_lab and candidate_designation == "Prof":
            continue

        projected_load = loads[faculty_id]["total_hours"] + 1
        max_hours = loads[faculty_id]["max_hours"]
        if projected_load > max_hours:
            continue

        day_row = get_faculty_day_row(faculty_id, day)
        if str(day_row[period]).strip() != "----":
            continue

        if (
            DESIGNATION_RANK.get(candidate_designation, 0)
            < DESIGNATION_RANK.get(absent_designation, 0)
            and required_course_code not in teaching_map.get(faculty_id, set())
        ):
            continue

        if required_course_code and required_course_code in teaching_map.get(faculty_id, set()):
            match_type = "same_course"
            priority = 1
        elif candidate_designation == absent_designation:
            match_type = "same_designation"
            priority = 2
        else:
            match_type = "available"
            priority = _candidate_priority(absent_designation, candidate_designation)

        candidates.append(
            {
                "faculty_id": faculty_id,
                "faculty_name": info["name"],
                "designation": candidate_designation,
                "projected_load": projected_load,
                "current_load": loads[faculty_id]["total_hours"],
                "max_hours": max_hours,
                "match_type": match_type,
                "priority": priority,
                "reason": _candidate_reason(day, period, match_type, str(slot_info["course"])),
            }
        )

    return sorted(
        candidates,
        key=lambda item: (
            item["priority"],
            item["projected_load"],
            -DESIGNATION_RANK.get(str(item["designation"]), 0),
            item["faculty_id"],
        ),
    )


def _collect_absent_slots(absent_faculty_id: str, day: str) -> List[Dict[str, object]]:
    row = get_faculty_day_row(absent_faculty_id, day)
    slots: List[Dict[str, object]] = []
    for period in PERIOD_COLUMNS:
        parsed = parse_timetable_cell(row[period])
        if not parsed:
            continue
        slots.append(
            {
                "period": period,
                "course": parsed["course_short"],
                "course_code": parsed["course_code"],
                "section": parsed["section"],
                "is_lab": parsed["is_lab"],
                "raw": parsed["raw"],
            }
        )
    return slots


def find_substitute(absent_faculty_id: str, absent_day: str) -> Dict[str, object]:
    faculty_id = str(absent_faculty_id).strip().upper()
    day = normalize_day(absent_day)
    lookup = faculty_lookup()
    if faculty_id not in lookup:
        raise ValueError(f"Unknown faculty_id: {faculty_id}")

    absent_info = lookup[faculty_id]
    original_slots = _collect_absent_slots(faculty_id, day)
    substitutions: List[Dict[str, object]] = []
    unresolved: List[Dict[str, object]] = []

    for slot in original_slots:
        ranked = _rank_candidates(
            absent_faculty_id=faculty_id,
            absent_designation=absent_info["designation"],
            day=day,
            period=str(slot["period"]),
            slot_info=slot,
        )
        if not ranked:
            unresolved.append(
                {
                    "period": slot["period"],
                    "course": slot["course"],
                    "section": slot["section"],
                    "reason": "No substitute available",
                }
            )
            continue

        chosen = ranked[0]
        substitutions.append(
            {
                "period": slot["period"],
                "course": slot["course"],
                "course_code": slot["course_code"],
                "section": slot["section"],
                "substitute_id": chosen["faculty_id"],
                "substitute_name": chosen["faculty_name"],
                "match_type": chosen["match_type"],
                "reason": chosen["reason"],
                "projected_load": f"{chosen['projected_load']}/{chosen['max_hours']}",
            }
        )

    return {
        "absent_faculty": faculty_id,
        "absent_faculty_name": absent_info["name"],
        "absent_day": day,
        "original_slots": [
            {"period": slot["period"], "course": slot["course"], "section": slot["section"]}
            for slot in original_slots
        ],
        "substitutions": substitutions,
        "unresolved": unresolved,
        "loads": build_load_snapshot(),
    }


def print_substitute_plan(result: Dict[str, object]) -> None:
    print("═" * 50)
    print("SUBSTITUTE PLAN")
    print(f"Absent: {result['absent_faculty_name']} ({result['absent_faculty']}) - {result['absent_day']}")
    print("═" * 50)
    print()

    substitutions = result.get("substitutions", [])
    unresolved = result.get("unresolved", [])
    substitution_by_period = {item["period"]: item for item in substitutions}

    for slot in result.get("original_slots", []):
        print(f"{slot['period']} | {slot['course']} | Section {slot['section']}")
        match = substitution_by_period.get(slot["period"])
        if match:
            print(f"   -> Substitute: {match['substitute_name']} ({match['substitute_id']})")
            print(f"      Reason: {match['reason']}")
        else:
            print("   -> Substitute: No substitute available")
        print()

    if unresolved:
        unresolved_text = ", ".join(
            f"{item['period']} ({item['course']} - Section {item['section']})" for item in unresolved
        )
        print(f"UNRESOLVED SLOTS: {unresolved_text}")
    else:
        print("UNRESOLVED SLOTS: None")

    print()
    print("NOTE: This is a temporary arrangement for the selected day only.")
    print("      Original timetable remains unchanged.")
    print("═" * 50)


def main() -> None:
    try:
        faculty_id = input("Enter faculty_id: ").strip()
        day = input("Enter day: ").strip()
        result = find_substitute(faculty_id, day)
        print_substitute_plan(result)
    except Exception as exc:
        print(f"Substitute planning failed: {exc}")


if __name__ == "__main__":
    main()

