import pandas as pd

DATA_DIR = "data"

# How many sections exist in this semester
TOTAL_SECTIONS = 12

# Max weekly teaching hours by designation
MAX_HOURS = {
    "Prof":       12,
    "Asso Prof":  16,
    "Asst Prof":  20,
}

def validate(courses, faculty, assignments):
    print("\n=== PHASE 0: CAPACITY VALIDATION ===\n")
    all_ok = True

    # ── Check 1: Each course has enough sections assigned ──────────────────
    print("[ Check 1 ] Section coverage per course")
    for _, course in courses.iterrows():
        code = course["course_code"]
        name = course["course_name"][:40]
        course_assignments = assignments[assignments["course_code"] == code]

        assigned_sections = []
        for _, row in course_assignments.iterrows():
            assigned_sections += [s.strip() for s in row["sections_handled"].split(",")]

        count = len(assigned_sections)
        status = "OK" if count == TOTAL_SECTIONS else "MISMATCH"
        if status != "OK":
            all_ok = False
        print(f"  {status} | {code} | assigned={count} | expected={TOTAL_SECTIONS} | {name}")

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
    for _, frow in faculty.iterrows():
        fid  = frow["faculty_id"]
        name = frow["name"]
        desig = frow["designation"]
        max_h = MAX_HOURS.get(desig, 16)

        fassign = assignments[assignments["faculty_id"] == fid]
        total_hours = 0
        for _, arow in fassign.iterrows():
            course = courses[courses["course_code"] == arow["course_code"]].iloc[0]
            n_sections = len([s.strip() for s in arow["sections_handled"].split(",")])
            total_hours += (course["theory_hours"] + course["lab_hours"]) * n_sections

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