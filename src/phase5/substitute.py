from __future__ import annotations

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

from config import DAYS, LAB_PERIODS, MAX_HOURS, SHORT_NAMES, resolve_output_path, get_sem_paths


OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR   = PROJECT_ROOT / "data"         # legacy fallback only
PERIOD_COLUMNS = [f"P{i}" for i in range(1, 7)]
DAY_ALIASES = {
    "mon":       "Monday",
    "monday":    "Monday",
    "tue":       "Tuesday",
    "tues":      "Tuesday",
    "tuesday":   "Tuesday",
    "wed":       "Wednesday",
    "wednesday": "Wednesday",
    "thu":       "Thursday",
    "thur":      "Thursday",
    "thurs":     "Thursday",
    "thursday":  "Thursday",
    "fri":       "Friday",
    "friday":    "Friday",
}
DESIGNATION_RANK = {"Asst Prof": 1, "Asso Prof": 2, "Prof": 3}
SHORT_NAME_TO_CODE = {short_name: code for code, short_name in SHORT_NAMES.items()}
CELL_PATTERN = re.compile(
    r"^(?P<course>[A-Z0-9]+)(?:\s+LAB)?\s*\((?P<section>[A-Z](?:\+[A-Z])*)\)$"
)

LAB_PERIOD_COLUMNS = {f"P{p}" for p in LAB_PERIODS}


# ── Pure helpers (no I/O) ─────────────────────────────────────────────────────

def get_lab_block_periods(period: str) -> list:
    """
    If the given period is in the lab window (P5 or P6), return the full
    lab block ["P5", "P6"] so substitute search treats them atomically.
    Otherwise return [period] unchanged.
    """
    if period in LAB_PERIOD_COLUMNS:
        return [f"P{p}" for p in LAB_PERIODS]
    return [period]


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


# ── Semester-aware data loaders (no lru_cache on metadata — must be per-sem) ──
# NOTE: lru_cache is NOT used on metadata loaders because they are keyed on
# data_dir, which changes per semester.  Caching is left to the caller (app.py
# uses @st.cache_data keyed on sem_id).  Output-file loaders keep
# lru_cache(maxsize=None) because they ARE keyed on (id, out_dir).

def load_faculty_metadata(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load faculty.csv from data_dir (semester-specific) or legacy DATA_DIR."""
    path = (data_dir or DATA_DIR) / "faculty.csv"
    df = pd.read_csv(path)
    for column in ["faculty_id", "name", "designation"]:
        df[column] = df[column].astype(str).str.strip()
    return df


def load_assignment_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load assignments.csv from data_dir or legacy DATA_DIR."""
    path = (data_dir or DATA_DIR) / "assignments.csv"
    df = pd.read_csv(path)
    for column in ["faculty_id", "course_code", "sections_handled"]:
        df[column] = df[column].astype(str).str.strip()
    return df


def load_course_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load courses.csv from data_dir or legacy DATA_DIR."""
    path = (data_dir or DATA_DIR) / "courses.csv"
    df = pd.read_csv(path)
    df["course_code"] = df["course_code"].astype(str).str.strip()
    return df


def faculty_lookup(data_dir: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    return load_faculty_metadata(data_dir).set_index("faculty_id").to_dict("index")


def assignments_by_faculty(data_dir: Optional[Path] = None) -> Dict[str, set]:
    grouped: Dict[str, set] = {}
    for _, row in load_assignment_data(data_dir).iterrows():
        grouped.setdefault(row["faculty_id"], set()).add(row["course_code"])
    return grouped


def all_faculty_ids(data_dir: Optional[Path] = None) -> List[str]:
    return load_faculty_metadata(data_dir)["faculty_id"].tolist()


def build_short_name_to_code(data_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Build short_name -> course_code map from courses.csv for the active semester.
    Falls back to the global SHORT_NAMES in config if courses.csv is unavailable.
    This is the FIX for P1 matching: config.SHORT_NAMES only covers sem3 courses,
    so sem5 course shorts like 'ML', 'DBMS', 'CA' were never resolved to a code.
    """
    try:
        df = load_course_data(data_dir)
        mapping: Dict[str, str] = {}
        for _, row in df.iterrows():
            code  = str(row["course_code"]).strip()
            short = str(row.get("short_name", "")).strip().upper()
            if code and short:
                mapping[short] = code
        if mapping:
            return mapping
    except Exception:
        pass
    # Fallback: global SHORT_NAMES
    return {s: c for c, s in SHORT_NAMES.items()}


# ── Output-file loaders (keyed on (id, out_dir) — safe to cache) ─────────────

from functools import lru_cache

@lru_cache(maxsize=None)
def load_faculty_timetable(faculty_id: str, out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a faculty timetable CSV, using out_dir when semester-specific."""
    if out_dir is not None:
        path = out_dir / f"faculty_{faculty_id}_timetable.csv"
    else:
        path = resolve_output_path(f"faculty_{faculty_id}_timetable.csv")
    if not path.exists():
        raise FileNotFoundError(f"Faculty timetable not found for {faculty_id}: {path}")
    df = pd.read_csv(path)
    df["Day"] = df["Day"].astype(str).str.strip()
    return df


@lru_cache(maxsize=None)
def load_section_timetable(section: str, out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a section timetable CSV, using out_dir when semester-specific."""
    if out_dir is not None:
        path = out_dir / f"section_{section}_timetable.csv"
    else:
        path = resolve_output_path(f"section_{section}_timetable.csv")
    if not path.exists():
        raise FileNotFoundError(f"Section timetable not found for {section}: {path}")
    df = pd.read_csv(path)
    df["Day"] = df["Day"].astype(str).str.strip()
    return df


# ── Core helpers that now carry both out_dir and data_dir ─────────────────────

def get_faculty_day_row(
    faculty_id: str, day: str, out_dir: Optional[Path] = None
) -> pd.Series:
    df = load_faculty_timetable(faculty_id, out_dir)
    day_df = df[df["Day"].str.lower() == day.lower()]
    if day_df.empty:
        raise ValueError(f"{day} not found in faculty timetable for {faculty_id}")
    return day_df.iloc[0]


def get_faculty_load(faculty_id: str, out_dir: Optional[Path] = None) -> int:
    df = load_faculty_timetable(faculty_id, out_dir)
    return int((df[PERIOD_COLUMNS] != "----").sum().sum())


def build_load_snapshot(
    out_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    lookup = faculty_lookup(data_dir)
    snapshot: Dict[str, Dict[str, object]] = {}
    for faculty_id in all_faculty_ids(data_dir):
        designation = lookup[faculty_id]["designation"]
        max_hours   = MAX_HOURS.get(designation, 16)
        total_hours = get_faculty_load(faculty_id, out_dir)
        snapshot[faculty_id] = {
            "name":        lookup[faculty_id]["name"],
            "designation": designation,
            "total_hours": total_hours,
            "max_hours":   max_hours,
            "status":      "OK" if total_hours <= max_hours else "OVERLOAD",
        }
    return snapshot


def _sections_for_faculty(faculty_id: str, data_dir: Optional[Path] = None) -> set:
    """Return the set of section letters this faculty teaches (from assignments.csv)."""
    sections: set = set()
    for _, row in load_assignment_data(data_dir).iterrows():
        if row["faculty_id"] == faculty_id:
            for sec in str(row["sections_handled"]).split(","):
                sec = sec.strip()
                if sec:
                    sections.add(sec)
    return sections


def _faculty_teaching_sections(data_dir: Optional[Path] = None) -> Dict[str, set]:
    """Map faculty_id -> set of section letters they teach."""
    result: Dict[str, set] = {}
    for _, row in load_assignment_data(data_dir).iterrows():
        fid = row["faculty_id"]
        for sec in str(row["sections_handled"]).split(","):
            sec = sec.strip()
            if sec:
                result.setdefault(fid, set()).add(sec)
    return result


def _rank_candidates(
    absent_faculty_id: str,
    day: str,
    period: str,
    slot_info: Dict[str, object],
    absent_sections: set,
    priority: str = "subject_first",
    out_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Returns (ranked_candidates, at_cap_notes).

    3-tier priority (designation is info only — never a filter):
      P1 — teaches the exact same course_code, free at period, under cap
      P2 — teaches any course to the SAME SECTION as this slot, free at period, under cap
      P3 — any free faculty, under cap

    priority='subject_first'  → rank 1=P1, 2=P2, 3=P3
    priority='section_first'  → rank 1=P2, 2=P1, 3=P3
    """
    loads        = build_load_snapshot(out_dir, data_dir)
    faculty_info = faculty_lookup(data_dir)
    teaching_map = assignments_by_faculty(data_dir)      # faculty_id → {course_codes}
    section_map  = _faculty_teaching_sections(data_dir)  # faculty_id → {section letters}

    # ── Resolve required_code from the semester's courses.csv ─────────────────
    # slot_info["course_code"] is None when SHORT_NAMES in config doesn't cover
    # the semester's courses (e.g. sem5 courses ML, DBMS, CA are not in config.SHORT_NAMES).
    required_code: Optional[str] = slot_info.get("course_code")  # type: ignore
    if not required_code:
        short_to_code = build_short_name_to_code(data_dir)
        course_short_key = str(slot_info.get("course", "")).strip().upper()
        required_code = short_to_code.get(course_short_key)  # None if truly unknown

    # The specific section that needs coverage for THIS slot
    slot_section = str(slot_info.get("section", "")).strip()

    is_lab       = bool(slot_info.get("is_lab"))
    course_short = str(slot_info.get("course", ""))

    candidates:   List[Dict[str, object]] = []
    at_cap_notes: List[Dict[str, object]] = []

    for faculty_id in all_faculty_ids(data_dir):
        if faculty_id == absent_faculty_id:
            continue

        info           = faculty_info.get(faculty_id, {})
        designation    = info.get("designation", "")
        total_hours    = loads[faculty_id]["total_hours"]
        max_hours      = loads[faculty_id]["max_hours"]
        projected_load = total_hours + 1
        at_cap         = projected_load > max_hours

        # ── Determine match tier ─────────────────────────────────────────────
        # P1: teaches the exact same course to any section
        teaches_same = bool(
            required_code and required_code in teaching_map.get(faculty_id, set())
        )
        # P2: teaches ANY course to the SAME SECTION as this slot
        #     Use slot_section (specific section needing cover) not all absent sections.
        cand_sections  = section_map.get(faculty_id, set())
        teaches_section = bool(slot_section and slot_section in cand_sections)

        if teaches_same:
            match_type = "same_course"
        elif teaches_section:
            match_type = "same_section"
        else:
            match_type = "available"

        # ── At-cap note for high-priority (P1/P2) candidates ─────────────────
        if at_cap and (teaches_same or teaches_section):
            at_cap_notes.append({
                "faculty_id":   faculty_id,
                "faculty_name": info.get("name", faculty_id),
                "designation":  designation,
                "match_type":   match_type,
                "total_hours":  total_hours,
                "max_hours":    max_hours,
                "course":       course_short,
            })
            continue

        if at_cap:
            continue  # P3 at-cap: silently skip

        # ── Must be free at the requested period ─────────────────────────────
        try:
            day_row = get_faculty_day_row(faculty_id, day, out_dir)
        except Exception:
            continue
        if str(day_row.get(period, "----")).strip() != "----":
            continue

        # ── Compute numeric rank ─────────────────────────────────────────────
        if priority == "section_first":
            if teaches_section:
                rank = 1
            elif teaches_same:
                rank = 2
            else:
                rank = 3
        else:  # subject_first (default)
            if teaches_same:
                rank = 1
            elif teaches_section:
                rank = 2
            else:
                rank = 3

        reason_parts = [f"free on {day} {period}"]
        if teaches_same:
            reason_parts.append(f"teaches {course_short} ({required_code})")
        if teaches_section:
            reason_parts.append(f"co-teaches section {slot_section}")

        candidates.append({
            "faculty_id":     faculty_id,
            "faculty_name":   info.get("name", faculty_id),
            "designation":    designation,
            "total_hours":    total_hours,
            "projected_load": projected_load,
            "max_hours":      max_hours,
            "match_type":     match_type,
            "priority":       rank,
            "reason":         "; ".join(reason_parts),
        })

    ranked = sorted(
        candidates,
        key=lambda c: (c["priority"], c["projected_load"], c["faculty_id"]),
    )
    return ranked, at_cap_notes


def _collect_absent_slots(
    absent_faculty_id: str, day: str, out_dir: Optional[Path] = None
) -> List[Dict[str, object]]:
    row   = get_faculty_day_row(absent_faculty_id, day, out_dir)
    slots: List[Dict[str, object]] = []
    for period in PERIOD_COLUMNS:
        parsed = parse_timetable_cell(row[period])
        if not parsed:
            continue
        slots.append(
            {
                "period":      period,
                "course":      parsed["course_short"],
                "course_code": parsed["course_code"],
                "section":     parsed["section"],
                "is_lab":      parsed["is_lab"],
                "raw":         parsed["raw"],
            }
        )
    return slots


# ── Public entry point ────────────────────────────────────────────────────────

def find_substitute(
    absent_faculty_id: str,
    absent_day: str,
    sem_id: str = None,
    priority: str = "subject_first",   # "subject_first" | "section_first"
) -> Dict[str, object]:
    """
    Find substitutes for an absent faculty member on a given day.

    Priority tiers (designation is info-only, never a filter):
      subject_first  → same-subject → same-section → any free
      section_first  → same-section → same-subject → any free

    sem_id → resolves both out_dir (output CSVs) and data_dir
    (faculty/assignment metadata).  Without sem_id falls back to legacy.
    """
    if sem_id:
        sem_paths = get_sem_paths(sem_id)
        out_dir:  Optional[Path] = sem_paths.output_dir
        data_dir: Optional[Path] = sem_paths.data_dir
    else:
        out_dir  = None
        data_dir = None

    faculty_id = str(absent_faculty_id).strip().upper()
    day        = normalize_day(absent_day)
    lookup     = faculty_lookup(data_dir)
    if faculty_id not in lookup:
        raise ValueError(f"Unknown faculty_id: {faculty_id}")

    absent_info     = lookup[faculty_id]
    absent_sections = _sections_for_faculty(faculty_id, data_dir)
    original_slots  = _collect_absent_slots(faculty_id, day, out_dir)
    substitutions:  List[Dict[str, object]] = []
    unresolved:     List[Dict[str, object]] = []
    all_at_cap_notes: List[Dict[str, object]] = []

    handled_periods: set = set()

    for slot in original_slots:
        period_str = str(slot["period"])
        if period_str in handled_periods:
            continue

        block_periods = get_lab_block_periods(period_str)
        is_lab_block  = len(block_periods) > 1

        ranked, at_cap_notes = _rank_candidates(
            absent_faculty_id=faculty_id,
            day=day,
            period=period_str,
            slot_info=slot,
            absent_sections=absent_sections,
            priority=priority,
            out_dir=out_dir,
            data_dir=data_dir,
        )
        all_at_cap_notes.extend(at_cap_notes)

        if is_lab_block:
            filtered = []
            for candidate in ranked:
                all_free = True
                for bp in block_periods:
                    try:
                        bp_row = get_faculty_day_row(candidate["faculty_id"], day, out_dir)
                        if str(bp_row.get(bp, "----")).strip() != "----":
                            all_free = False
                            break
                    except Exception:
                        all_free = False
                        break
                if all_free:
                    filtered.append(candidate)
            ranked = filtered

        if not ranked:
            for bp in block_periods:
                unresolved.append({
                    "period":  bp,
                    "course":  slot["course"],
                    "section": slot["section"],
                    "reason":  "No substitute available" + (
                        " (lab block requires full P5-P6 coverage)"
                        if is_lab_block else ""
                    ),
                })
            handled_periods.update(block_periods)
            continue

        chosen = ranked[0]
        for bp in block_periods:
            substitutions.append({
                "period":          bp,
                "course":          slot["course"],
                "course_code":     slot["course_code"],
                "section":         slot["section"],
                "substitute_id":   chosen["faculty_id"],
                "substitute_name": chosen["faculty_name"],
                "designation":     chosen["designation"],
                "match_type":      chosen["match_type"],
                "priority":        chosen["priority"],
                "reason":          chosen["reason"] + (
                    " (lab block: P5-P6 atomic)" if is_lab_block else ""
                ),
                "projected_load":  f"{chosen['projected_load']}/{chosen['max_hours']}",
            })
        handled_periods.update(block_periods)

    # Deduplicate at-cap notes
    seen_at_cap: set = set()
    unique_at_cap: List[Dict[str, object]] = []
    for note in all_at_cap_notes:
        key = (note["faculty_id"], note["match_type"])
        if key not in seen_at_cap:
            seen_at_cap.add(key)
            unique_at_cap.append(note)

    return {
        "absent_faculty":      faculty_id,
        "absent_faculty_name": absent_info["name"],
        "absent_day":          day,
        "absent_sections":     sorted(absent_sections),
        "priority_mode":       priority,
        "original_slots": [
            {"period": s["period"], "course": s["course"], "section": s["section"]}
            for s in original_slots
        ],
        "substitutions":  substitutions,
        "unresolved":     unresolved,
        "at_cap_notes":   unique_at_cap,
        "loads":          build_load_snapshot(out_dir, data_dir),
    }


def print_substitute_plan(result: Dict[str, object]) -> None:
    print("═" * 50)
    print("SUBSTITUTE PLAN")
    print(f"Absent: {result['absent_faculty_name']} ({result['absent_faculty']}) - {result['absent_day']}")
    print("═" * 50)
    print()

    substitutions = result.get("substitutions", [])
    unresolved    = result.get("unresolved", [])
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
        day        = input("Enter day: ").strip()
        result     = find_substitute(faculty_id, day)
        print_substitute_plan(result)
    except Exception as exc:
        print(f"Substitute planning failed: {exc}")


if __name__ == "__main__":
    main()
