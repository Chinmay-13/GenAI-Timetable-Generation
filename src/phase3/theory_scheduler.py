from pathlib import Path
import sys

from ortools.sat.python import cp_model
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.assignment_builder import build_assignment_map
from src.phase2.lab_scheduler import lock_labs

DATA_DIR = "data"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PERIODS = list(range(1, 10))
SECTIONS = [chr(ord("A") + i) for i in range(12)]
SHORT_NAMES = {
    "UE24CS251A": "DDCO",
    "UE24CS252A": "DSA",
    "UE24MA242A": "MATH",
    "UE24CS242A": "WT",
    "UE24CS243A": "AFLL",
}
MAX_HOURS = {"Prof": 12, "Asso Prof": 16, "Asst Prof": 20}


def _is_lab_token(value):
    return isinstance(value, str) and value.endswith("_LAB")


def _print_section_grid(section_grid, section="A"):
    print(f"\nSection {section} timetable:\n")
    header = ["Day", "P1", "P2", "P3", "P4", "P5", "P6", "LUNCH", "P7", "P8", "P9"]
    print(" | ".join(f"{h:<6}" for h in header))
    print("-" * 96)

    for day in DAYS:
        row = [f"{day:<6}"]
        for p in PERIODS:
            value = section_grid[section][day][p]
            if value is None:
                cell = "----"
            elif _is_lab_token(value):
                course_code = value.replace("_LAB", "")
                cell = f"{SHORT_NAMES.get(course_code, course_code)} LAB"
            else:
                cell = SHORT_NAMES.get(value, value)
            row.append(f"{cell:<6}")
            if p == 6:
                row.append(f"{'':<6}")
        print(" | ".join(row))


def solve_theory(
    assignment_map=None,
    section_grid=None,
    faculty_grid=None,
    room_grid=None,
    lab_details=None,
    data_dir=DATA_DIR,
):
    courses_df = pd.read_csv(f"{data_dir}/courses.csv")
    faculty_df = pd.read_csv(f"{data_dir}/faculty.csv")

    if assignment_map is None:
        assignment_map = build_assignment_map(data_dir=data_dir)
    if section_grid is None or faculty_grid is None or room_grid is None or lab_details is None:
        section_grid, faculty_grid, room_grid, lab_details = lock_labs(data_dir=data_dir)

    course_codes = courses_df["course_code"].tolist()
    # Override weekly theory load in-code so the scheduler can run with
    # 6 theory slots per course without requiring changes to the CSV generator.
    theory_hours = {course_code: 6 for course_code in courses_df["course_code"].tolist()}
    faculty_ids = faculty_df["faculty_id"].tolist()

    section_lab_days = {section: set() for section in SECTIONS}
    for detail in lab_details:
        for section in detail["sections"]:
            section_lab_days[section].add(detail["day"])

    model = cp_model.CpModel()
    x = {}

    for section in SECTIONS:
        for course in course_codes:
            for day in DAYS:
                for period in PERIODS:
                    if section_grid[section][day][period] is not None:
                        continue
                    if day in section_lab_days[section] and period in (7, 8, 9):
                        continue
                    x[(section, course, day, period)] = model.NewBoolVar(
                        f"x_{section}_{course}_{day}_{period}"
                    )

    for section in SECTIONS:
        for course in course_codes:
            vars_for_course = [
                x[(section, course, day, period)]
                for day in DAYS
                for period in PERIODS
                if (section, course, day, period) in x
            ]
            model.Add(sum(vars_for_course) == int(theory_hours[course]))

    for section in SECTIONS:
        for day in DAYS:
            for period in PERIODS:
                vars_in_slot = [
                    x[(section, course, day, period)]
                    for course in course_codes
                    if (section, course, day, period) in x
                ]
                if vars_in_slot:
                    model.Add(sum(vars_in_slot) <= 1)

    for faculty_id in faculty_ids:
        for day in DAYS:
            for period in PERIODS:
                vars_for_faculty = []
                for course in course_codes:
                    for section, assigned_faculty in assignment_map[course].items():
                        if assigned_faculty != faculty_id:
                            continue
                        key = (section, course, day, period)
                        if key in x:
                            vars_for_faculty.append(x[key])

                if faculty_grid[faculty_id][day][period] is not None:
                    if vars_for_faculty:
                        model.Add(sum(vars_for_faculty) == 0)
                else:
                    if vars_for_faculty:
                        model.Add(sum(vars_for_faculty) <= 1)

    penalty_terms = []
    soft_counts = {
        "same_subject_same_day": 0,
        "back_to_back_same_subject": 0,
        "p7_p9_theory_non_lab_day": 0,
        "intra_day_gaps_p1_p6": 0,
        "daily_load_imbalance": 0,
    }

    # is_used[(section, day, period)] for P1-P6 only: whether section has any theory class in that slot.
    is_used = {}
    has_before = {}
    has_after = {}
    gap_var = {}
    daily_count = {}

    for section in SECTIONS:
        for day in DAYS:
            for period in range(1, 7):
                used = model.NewBoolVar(f"is_used_{section}_{day}_{period}")
                is_used[(section, day, period)] = used

                slot_vars = [
                    x[(section, course, day, period)]
                    for course in course_codes
                    if (section, course, day, period) in x
                ]
                if slot_vars:
                    model.Add(sum(slot_vars) >= used)
                    model.Add(sum(slot_vars) <= len(course_codes) * used)
                else:
                    model.Add(used == 0)

            # Enforce prefix usage in P1-P6: if a later period is used, all earlier periods are used.
            # This removes both leading gaps and intra-day holes in the morning block.
            for period in range(1, 6):
                model.Add(
                    is_used[(section, day, period)]
                    >= is_used[(section, day, period + 1)]
                )

            dcount = model.NewIntVar(0, 6, f"daily_count_{section}_{day}")
            model.Add(
                dcount
                == sum(is_used[(section, day, period)] for period in range(1, 7))
            )
            # Hard daily distribution: exactly 6 theory slots per day in P1-P6.
            model.Add(dcount == 6)
            daily_count[(section, day)] = dcount

            # Gap penalties in P1-P6:
            # gap at P2..P5 if no class at P but there is at least one before and one after.
            for p2 in range(2, 6):
                hb = model.NewBoolVar(f"has_before_{section}_{day}_{p2}")
                ha = model.NewBoolVar(f"has_after_{section}_{day}_{p2}")
                gv = model.NewBoolVar(f"gap_{section}_{day}_{p2}")
                has_before[(section, day, p2)] = hb
                has_after[(section, day, p2)] = ha
                gap_var[(section, day, p2)] = gv

                before_vars = [is_used[(section, day, p)] for p in range(1, p2)]
                after_vars = [is_used[(section, day, p)] for p in range(p2 + 1, 7)]

                model.Add(sum(before_vars) >= hb)
                model.Add(sum(before_vars) <= len(before_vars) * hb)
                model.Add(sum(after_vars) >= ha)
                model.Add(sum(after_vars) <= len(after_vars) * ha)

                # gv = hb AND (not is_used[p2]) AND ha
                model.Add(gv <= hb)
                model.Add(gv <= ha)
                model.Add(gv <= 1 - is_used[(section, day, p2)])
                model.Add(gv >= hb + ha + (1 - is_used[(section, day, p2)]) - 2)

                penalty_terms.append(gv * 100)

    for section in SECTIONS:
        for course in course_codes:
            for day in DAYS:
                day_vars = [
                    x[(section, course, day, period)]
                    for period in PERIODS
                    if (section, course, day, period) in x
                ]
                if day_vars:
                    over = model.NewIntVar(0, len(PERIODS), f"over_{section}_{course}_{day}")
                    model.Add(over >= sum(day_vars) - 1)
                    penalty_terms.append(over * 3)

                for period in range(1, 9):
                    k1 = (section, course, day, period)
                    k2 = (section, course, day, period + 1)
                    if k1 in x and k2 in x:
                        adj = model.NewBoolVar(f"adj_{section}_{course}_{day}_{period}")
                        model.Add(adj <= x[k1])
                        model.Add(adj <= x[k2])
                        model.Add(adj >= x[k1] + x[k2] - 1)
                        penalty_terms.append(adj * 5)

    # Morning preference: on non-lab days, strongly discourage theory in P7-P9.
    for section in SECTIONS:
        for day in DAYS:
            if day in section_lab_days[section]:
                continue
            for course in course_codes:
                for period in (7, 8, 9):
                    key = (section, course, day, period)
                    if key in x:
                        penalty_terms.append(x[key] * 50)

    model.Minimize(sum(penalty_terms) if penalty_terms else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("PHASE 3 FAILED - solver returned INFEASIBLE")
        print("Likely causes:")
        print("  1) Faculty overlap constraints too tight for available free slots")
        print("  2) Section theory-hour requirements exceed free capacity after lab locks")
        print("  3) Assignment distribution creates unavoidable conflicts on some days")
        raise ValueError("Phase 3 infeasible")

    for (section, course, day, period), var in x.items():
        if solver.Value(var) == 1:
            section_grid[section][day][period] = course
            faculty_id = assignment_map[course][section]
            if faculty_grid[faculty_id][day][period] is not None:
                raise ValueError(
                    f"Faculty grid collision for {faculty_id} at {day} P{period}"
                )
            faculty_grid[faculty_id][day][period] = f"{course} ({section})"

    for section in SECTIONS:
        for course in course_codes:
            day_counts = []
            for day in DAYS:
                c = sum(
                    1
                    for period in PERIODS
                    if section_grid[section][day][period] == course
                )
                day_counts.append(c)
                if c > 1:
                    soft_counts["same_subject_same_day"] += c - 1
            for day in DAYS:
                for period in range(1, 9):
                    if (
                        section_grid[section][day][period] == course
                        and section_grid[section][day][period + 1] == course
                    ):
                        soft_counts["back_to_back_same_subject"] += 1

            if day not in section_lab_days[section]:
                for period in (7, 8, 9):
                    if section_grid[section][day][period] == course:
                        soft_counts["p7_p9_theory_non_lab_day"] += 1

    for section in SECTIONS:
        for day in DAYS:
            soft_counts["daily_load_imbalance"] += abs(
                solver.Value(daily_count[(section, day)]) - 6
            )
            for p2 in range(2, 6):
                if solver.Value(gap_var[(section, day, p2)]) == 1:
                    soft_counts["intra_day_gaps_p1_p6"] += 1

    _print_section_grid(section_grid, section="A")
    print(
        f"\nPHASE 3 COMPLETE - theory slots filled | solver status: {status_name}"
    )

    return {
        "section_grid": section_grid,
        "faculty_grid": faculty_grid,
        "room_grid": room_grid,
        "assignment_map": assignment_map,
        "lab_details": lab_details,
        "solver_status": status_name,
        "soft_violations": soft_counts,
    }


if __name__ == "__main__":
    solve_theory()
