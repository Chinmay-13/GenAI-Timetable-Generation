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
    courses     = pd.read_csv(f"{data_dir}/courses.csv")
    faculty     = pd.read_csv(f"{data_dir}/faculty.csv")
    assignments = pd.read_csv(f"{data_dir}/assignments.csv")

    # Enrich courses with is_elective flag (safe for legacy CSVs without the column)
    if "is_elective" not in courses.columns:
        courses["is_elective"] = False
    else:
        courses["is_elective"] = courses["is_elective"].map(
            lambda v: str(v).strip().lower() in {"true", "1", "yes", "y"}
            if not isinstance(v, bool) else v
        )

    elective_codes: set = set(
        courses[courses["is_elective"] == True]["course_code"].tolist()
    )

    assignment_map = {code: {} for code in courses["course_code"].tolist()}
    errors = []

    faculty_designation = dict(zip(faculty["faculty_id"], faculty["designation"]))
    lab_courses = set(courses[courses["has_lab"] == True]["course_code"].tolist())

    for _, row in assignments.iterrows():
        faculty_id  = str(row["faculty_id"]).strip()
        course_code = str(row["course_code"]).strip()
        sections    = [s.strip() for s in str(row["sections_handled"]).split(",") if s.strip()]

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

    # ── Print core (non-elective) courses ────────────────────────────────────
    core_codes = [c for c in courses["course_code"].tolist() if c not in elective_codes]
    if core_codes:
        print(f"{'Course':<12} {'Section':<8} {'Faculty':<8} {'Designation':<10}")
        print("-" * 42)
        for course_code in core_codes:
            for section in SECTIONS:
                faculty_id  = assignment_map.get(course_code, {}).get(section, "MISSING")
                designation = faculty_designation.get(faculty_id, "-")
                print(f"  {course_code:<12} {section:<8} {faculty_id:<8} {designation:<10}")

    # ── Print elective courses (partial coverage is expected) ────────────────
    if elective_codes:
        print(f"\n[Electives — partial section coverage is expected]")
        print(f"{'Course':<12} {'Section':<8} {'Faculty':<8} {'Designation':<10}")
        print("-" * 42)
        for course_code in courses["course_code"].tolist():
            if course_code not in elective_codes:
                continue
            for section, fid in sorted(assignment_map[course_code].items()):
                designation = faculty_designation.get(fid, "-")
                print(f"  {course_code:<12} {section:<8} {fid:<8} {designation:<10}")

    # ── Coverage validation ───────────────────────────────────────────────────
    for course_code in courses["course_code"].tolist():
        if course_code in elective_codes:
            # Electives only need ≥1 section assigned — partial coverage is by design
            if not assignment_map[course_code]:
                errors.append(f"Elective {course_code} has no sections assigned at all")
        else:
            missing = [s for s in SECTIONS if s not in assignment_map[course_code]]
            if missing:
                errors.append(f"Coverage missing for {course_code}: {','.join(missing)}")

    print("\nValidation summary:")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError("Phase 1 validation failed.")

    print(f"  Core courses: all sections A-L covered")
    if elective_codes:
        print(f"  Elective courses ({len(elective_codes)}): partial coverage accepted")
    print("  No duplicate section entries per course")
    print("  Designation rules respected")
    print("PHASE 1 COMPLETE - assignment map ready")

    return assignment_map


if __name__ == "__main__":
    build_assignment_map()

