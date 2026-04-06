import pandas as pd
from pathlib import Path
import sys

_LS_ROOT = Path(__file__).resolve().parents[2]
if str(_LS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LS_ROOT))

from config import DAYS, PERIODS, LAB_PERIODS, SECTIONS

DATA_DIR = str(_LS_ROOT / "data")


def _build_empty_grid(keys):
    return {
        key: {
            day: {period: None for period in PERIODS}
            for day in DAYS
        }
        for key in keys
    }


def _split_sections(raw_value: object) -> list[str]:
    return [s.strip() for s in str(raw_value).split(",") if s and s.strip()]


def _optional_elective_slots(data_dir):
    path = Path(data_dir) / "elective_slots.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["day"] = df["day"].astype(str).str.strip()
    df["course_code"] = df["course_code"].astype(str).str.strip()
    df["room"] = df["room"].astype(str).str.strip()
    df["faculty_id"] = df["faculty_id"].astype(str).str.strip()
    df["elective_group"] = df["elective_group"].astype(str).str.strip()
    df["enrolled_sections"] = df["enrolled_sections"].astype(str).str.strip()
    df["period_start"] = df["period_start"].astype(int)
    df["period_end"] = df["period_end"].astype(int)
    return df


def lock_labs(data_dir=DATA_DIR):
    lab_allotment = pd.read_csv(f"{data_dir}/lab_allotment.csv")
    faculty = pd.read_csv(f"{data_dir}/faculty.csv")
    rooms = pd.read_csv(f"{data_dir}/rooms.csv")
    elective_slots = _optional_elective_slots(data_dir)

    faculty_ids = faculty["faculty_id"].astype(str).str.strip().tolist()
    room_names = rooms["room_name"].astype(str).str.strip().tolist()
    if elective_slots is not None and not elective_slots.empty:
        for room_name in elective_slots["room"].tolist():
            if room_name not in room_names:
                room_names.append(room_name)

    faculty_designation = dict(
        zip(
            faculty["faculty_id"].astype(str).str.strip(),
            faculty["designation"].astype(str).str.strip(),
        )
    )

    section_grid = _build_empty_grid(SECTIONS)
    faculty_grid = _build_empty_grid(faculty_ids)
    room_grid = _build_empty_grid(room_names)

    all_conflicts = []
    lab_details = []
    elective_details = []

    print("\n=== PHASE 2: FIXED SLOT LOCKING ===\n")

    for _, row in lab_allotment.iterrows():
        day = str(row["day"]).strip()
        course_code = str(row["course_code"]).strip()
        room = str(row["room"]).strip()
        faculty_id = str(row["faculty_id"]).strip()
        sections = _split_sections(row["section_pair"])

        row_conflicts = []

        if day not in DAYS:
            row_conflicts.append(f"Invalid day in lab allotment: {day}")

        if len(sections) != 2:
            row_conflicts.append(
                f"Invalid section pair '{row['section_pair']}' for {course_code} on {day}"
            )

        if faculty_designation.get(faculty_id) == "Prof":
            row_conflicts.append(f"Prof {faculty_id} assigned to lab {course_code} on {day}")

        for section in sections:
            if section not in SECTIONS:
                row_conflicts.append(
                    f"Invalid section {section} in pair for {course_code} on {day}"
                )

        if not row_conflicts:
            for period in LAB_PERIODS:
                for section in sections:
                    if section_grid[section][day][period] is not None:
                        row_conflicts.append(
                            f"Section conflict: {section} already has "
                            f"{section_grid[section][day][period]} at {day} P{period}"
                        )
                if faculty_grid[faculty_id][day][period] is not None:
                    row_conflicts.append(
                        f"Faculty conflict: {faculty_id} already has "
                        f"{faculty_grid[faculty_id][day][period]} at {day} P{period}"
                    )
                if room_grid[room][day][period] is not None:
                    row_conflicts.append(
                        f"Room conflict: {room} already has "
                        f"{room_grid[room][day][period]} at {day} P{period}"
                    )

        if row_conflicts:
            all_conflicts.extend(row_conflicts)
            continue

        token = f"{course_code}_LAB"
        for period in LAB_PERIODS:
            for section in sections:
                section_grid[section][day][period] = token
            faculty_grid[faculty_id][day][period] = token
            room_grid[room][day][period] = token

        lab_details.append(
            {
                "day": day,
                "course_code": course_code,
                "sections": sections,
                "room": room,
                "faculty_id": faculty_id,
            }
        )

        print(
            f"LOCKED LAB: {course_code} | {day} | P5-P6 | "
            f"Sections {'+'.join(sections)} | Room: {room} | Faculty: {faculty_id}"
        )

    if elective_slots is not None and not elective_slots.empty:
        for _, row in elective_slots.iterrows():
            day = row["day"]
            course_code = row["course_code"]
            room = row["room"]
            faculty_id = row["faculty_id"]
            elective_group = row["elective_group"]
            sections = _split_sections(row["enrolled_sections"])
            period_start = int(row["period_start"])
            period_end = int(row["period_end"])

            row_conflicts = []

            if day not in DAYS:
                row_conflicts.append(f"Invalid day in elective_slots: {day}")
            if period_start < 1 or period_end > max(PERIODS) or period_start > period_end:
                row_conflicts.append(
                    f"Invalid elective period range P{period_start}-P{period_end} "
                    f"for {course_code} on {day}"
                )
            if faculty_id not in faculty_grid:
                row_conflicts.append(f"Unknown faculty {faculty_id} for elective {course_code}")
            if not sections:
                row_conflicts.append(f"Elective {course_code} has no enrolled sections")

            for section in sections:
                if section not in SECTIONS:
                    row_conflicts.append(
                        f"Invalid section {section} in elective {course_code} on {day}"
                    )

            if not row_conflicts:
                for period in range(period_start, period_end + 1):
                    for section in sections:
                        if section_grid[section][day][period] is not None:
                            row_conflicts.append(
                                f"Section conflict: {section} already has "
                                f"{section_grid[section][day][period]} at {day} P{period}"
                            )
                    if faculty_grid[faculty_id][day][period] is not None:
                        row_conflicts.append(
                            f"Faculty conflict: {faculty_id} already has "
                            f"{faculty_grid[faculty_id][day][period]} at {day} P{period}"
                        )
                    if room_grid[room][day][period] is not None:
                        row_conflicts.append(
                            f"Room conflict: {room} already has "
                            f"{room_grid[room][day][period]} at {day} P{period}"
                        )

            if row_conflicts:
                all_conflicts.extend(row_conflicts)
                continue

            for period in range(period_start, period_end + 1):
                # Use a human-readable label in the section grid so it
                # appears as "Elective 1" / "Elective 2" in every CSV.
                elective_label = (
                    "Elective 1" if elective_group == "E1" else "Elective 2"
                )
                for section in sections:
                    section_grid[section][day][period] = elective_label
                # Faculty and room grids keep the real course code (for
                # conflict detection and faculty timetable output).
                faculty_grid[faculty_id][day][period] = course_code
                room_grid[room][day][period] = course_code


            elective_details.append(
                {
                    "day": day,
                    "course_code": course_code,
                    "sections": sections,
                    "room": room,
                    "faculty_id": faculty_id,
                    "period_start": period_start,
                    "period_end": period_end,
                    "elective_group": elective_group,
                }
            )

            print(
                f"LOCKED ELECTIVE: {course_code} | {day} | P{period_start}-P{period_end} | "
                f"Sections {'+'.join(sections)} | Room: {room} | Faculty: {faculty_id}"
            )

    if all_conflicts:
        print("\nConflicts detected in fixed slots:")
        for conflict in all_conflicts:
            print(f"  CONFLICT: {conflict}")
        raise ValueError("Phase 2 failed due to fixed-slot conflicts.")

    print("\nNo conflicts detected in fixed slots")
    print("PHASE 2 COMPLETE - labs/electives locked")

    return section_grid, faculty_grid, room_grid, lab_details, elective_details


if __name__ == "__main__":
    lock_labs()
