import pandas as pd
from pathlib import Path
import sys

_AB_ROOT = Path(__file__).resolve().parents[2]
if str(_AB_ROOT) not in sys.path:
    sys.path.insert(0, str(_AB_ROOT))

from config import SECTIONS

DATA_DIR = str(_AB_ROOT / "data")
LAB_COURSE_SHORT = {
    "UE24CS251A": "DDCO",
    "UE24CS252A": "DSA",
}


def build_assignment_map(data_dir=DATA_DIR):
    courses = pd.read_csv(f"{data_dir}/courses.csv")
    faculty = pd.read_csv(f"{data_dir}/faculty.csv")
    assignments = pd.read_csv(f"{data_dir}/assignments.csv")

    assignment_map = {code: {} for code in courses["course_code"].tolist()}
    errors = []

    faculty_designation = dict(zip(faculty["faculty_id"], faculty["designation"]))
    lab_courses = set(courses[courses["has_lab"] == True]["course_code"].tolist())

    for _, row in assignments.iterrows():
        faculty_id = str(row["faculty_id"]).strip()
        course_code = str(row["course_code"]).strip()
        sections = [s.strip() for s in str(row["sections_handled"]).split(",") if s.strip()]

        if course_code not in assignment_map:
            errors.append(f"Unknown course_code in assignments: {course_code}")
            continue

        if course_code in lab_courses and faculty_designation.get(faculty_id) == "Prof":
            errors.append(
                f"Designation violation: Prof faculty {faculty_id} assigned to lab course {course_code}"
            )

        for section in sections:
            if section not in SECTIONS:
                errors.append(f"Invalid section {section} for {course_code} by {faculty_id}")
                continue
            if section in assignment_map[course_code]:
                prev = assignment_map[course_code][section]
                errors.append(
                    f"Duplicate section mapping for {course_code}-{section}: {prev} and {faculty_id}"
                )
                continue
            assignment_map[course_code][section] = faculty_id

    print("\n=== PHASE 1: FACULTY-SECTION ASSIGNMENT MAP ===\n")
    print(f"{'Course':<12} {'Section':<8} {'Faculty':<8} {'Designation':<10}")
    print("-" * 42)

    row_count = 0
    for course_code in courses["course_code"].tolist():
        for section in SECTIONS:
            faculty_id = assignment_map.get(course_code, {}).get(section, "MISSING")
            designation = faculty_designation.get(faculty_id, "-")
            print(f"{course_code:<12} {section:<8} {faculty_id:<8} {designation:<10}")
            row_count += 1

    for course_code in courses["course_code"].tolist():
        missing = [s for s in SECTIONS if s not in assignment_map[course_code]]
        if missing:
            errors.append(f"Coverage missing for {course_code}: {','.join(missing)}")

    print("\nValidation summary:")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError("Phase 1 validation failed.")

    if row_count != 60:
        raise ValueError(f"Expected 60 rows in printed table, got {row_count}")

    print("  All sections A-L covered for all courses")
    print("  No duplicate section entries per course")
    print("  Designation rules respected")
    print("PHASE 1 COMPLETE - assignment map ready")

    return assignment_map


if __name__ == "__main__":
    build_assignment_map()
