import pandas as pd
from pathlib import Path
import sys

_VALIDATOR_ROOT = Path(__file__).resolve().parents[2]
if str(_VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VALIDATOR_ROOT))

from config import DATA_DIR as CONFIG_DATA_DIR, TOTAL_SECTIONS, MAX_HOURS, LAB_BLOCK_LENGTH, get_theory_periods, get_lab_sessions

DATA_DIR = str(CONFIG_DATA_DIR)


def validate(courses, faculty, assignments, data_dir: str = None):
    """
    Validate timetable input data.

    Parameters
    ----------
    courses, faculty, assignments : pd.DataFrame
        Data already loaded from the correct semester directory.
    data_dir : str or None
        Path to the semester data directory used to locate lab_allotment.csv.
        Falls back to the legacy DATA_DIR constant when None.
    """
    print("\n=== PHASE 0: CAPACITY VALIDATION ===\n")
    all_ok = True

    # ── Check 1: Each course has enough sections assigned ──────────────────
    print("[ Check 1 ] Section coverage per course")
    for _, course in courses.iterrows():
        code = course["course_code"]
        name = course["course_name"][:40]
        # Elective courses are by design assigned to fewer than TOTAL_SECTIONS
        # (only the sections that chose that elective).  Require >= 1 instead.
        is_elective = bool(course.get("is_elective", False))
        course_assignments = assignments[assignments["course_code"] == code]

        assigned_sections = []
        for _, row in course_assignments.iterrows():
            assigned_sections += [s.strip() for s in row["sections_handled"].split(",")]

        count = len(assigned_sections)
        if is_elective:
            status = "OK" if count >= 1 else "MISMATCH"
            tag = f"elective (assigned={count} sections)"
        else:
            status = "OK" if count == TOTAL_SECTIONS else "MISMATCH"
            tag = f"assigned={count} | expected={TOTAL_SECTIONS}"
        if status != "OK":
            all_ok = False
        print(f"  {status} | {code} | {tag} | {name}")

    # ── Check 2: No faculty assigned to more than 2 courses ───────────────
    print("\n[ Check 2 ] Faculty assigned to max 2 courses")
    faculty_course_count = assignments.groupby("faculty_id")["course_code"].nunique()
    for fid, count in faculty_course_count.items():
        fname = faculty[faculty["faculty_id"] == fid]["name"].values[0]
        status = "OK" if count <= 2 else "OVERLOADED"
        if status != "OK":
            all_ok = False
            print(f"  {status} | {fid} | {fname} | courses={count}")
    if all(c <= 2 for c in faculty_course_count):
        print("  All faculty within 2-course limit")

    # ── Check 3: Professor not assigned to lab courses ────────────────────
    print("\n[ Check 3 ] Professors not assigned to lab courses")
    lab_courses = courses[courses["has_lab"] == True]["course_code"].tolist()
    profs = faculty[faculty["designation"] == "Prof"]["faculty_id"].tolist()
    violations = assignments[
        (assignments["faculty_id"].isin(profs)) &
        (assignments["course_code"].isin(lab_courses))
    ]
    if violations.empty:
        print("  No Professors assigned to lab courses — OK")
    else:
        all_ok = False
        for _, v in violations.iterrows():
            fname = faculty[faculty["faculty_id"] == v["faculty_id"]]["name"].values[0]
            print(f"  VIOLATION | {fname} (Prof) assigned to lab course {v['course_code']}")

    # ── Check 4: Weekly hour load per faculty ─────────────────────────────
    print("\n[ Check 4 ] Weekly hour load per faculty")
    course_lookup = courses.set_index("course_code").to_dict("index")
    # Use the semester-specific data_dir when provided; fall back to legacy DATA_DIR
    _lab_dir = data_dir if data_dir is not None else DATA_DIR
    try:
        lab_allotment = pd.read_csv(f"{_lab_dir}/lab_allotment.csv")
    except FileNotFoundError:
        lab_allotment = pd.DataFrame(
            columns=["day", "course_code", "section_pair", "room", "faculty_id"]
        )

    for _, frow in faculty.iterrows():
        fid  = frow["faculty_id"]
        name = frow["name"]
        desig = frow["designation"]
        max_h = MAX_HOURS.get(desig, 16)

        fassign = assignments[assignments["faculty_id"] == fid]
        total_hours = 0
        for _, arow in fassign.iterrows():
            course = course_lookup[arow["course_code"]]
            n_sections = len([s.strip() for s in arow["sections_handled"].split(",")])
            total_hours += get_theory_periods(course["credits"], course["has_lab"]) * n_sections

        faculty_labs = lab_allotment[lab_allotment["faculty_id"] == fid]
        for _, lab_row in faculty_labs.iterrows():
            course = course_lookup.get(lab_row["course_code"])
            if course is None:
                continue
            total_hours += get_lab_sessions(course["credits"], course["has_lab"]) * LAB_BLOCK_LENGTH

        status = "OK" if total_hours <= max_h else "OVERLOADED"
        if status != "OK":
            all_ok = False
            print(f"  {status} | {fid} | {name} | {desig} | load={total_hours}h | max={max_h}h")

    if all_ok:
        print("\n  All faculty within hour limits")

    # ── Final result ──────────────────────────────────────────────────────
    print("\n=====================================")
    if all_ok:
        print("RESULT: All checks passed. Safe to proceed to Phase 1.")
    else:
        print("RESULT: Some checks FAILED. Fix the issues above before proceeding.")
    print("=====================================\n")

    return all_ok


if __name__ == "__main__":
    courses       = pd.read_csv(f"{DATA_DIR}/courses.csv")
    faculty       = pd.read_csv(f"{DATA_DIR}/faculty.csv")
    assignments   = pd.read_csv(f"{DATA_DIR}/assignments.csv")
    validate(courses, faculty, assignments)
