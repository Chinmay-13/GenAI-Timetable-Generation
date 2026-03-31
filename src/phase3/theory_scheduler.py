from itertools import combinations
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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


def _initials(name):
    tokens = [t.strip(".") for t in str(name).split() if t and t.lower() != "prof."]
    return "".join(t[0].upper() for t in tokens[:3]) if tokens else "NA"


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


def _faculty_course_sections(assignment_map):
    grouped = {}
    for course, section_map in assignment_map.items():
        by_faculty = {}
        for section, faculty_id in section_map.items():
            by_faculty.setdefault(faculty_id, []).append(section)
        for faculty_id, sections in by_faculty.items():
            if len(sections) < 2:
                continue
            grouped.setdefault(faculty_id, {})[course] = sorted(sections)
    return grouped


def _print_same_day_consecutive_analysis(section_grid, assignment_map, faculty_df):
    faculty_name = dict(zip(faculty_df["faculty_id"], faculty_df["name"]))
    faculty_initials = {
        faculty_id: _initials(name) for faculty_id, name in faculty_name.items()
    }

    grouped = _faculty_course_sections(assignment_map)
    entries_by_day = {day: [] for day in DAYS}
    total_pairs = 0
    consecutive_pairs = 0

    for faculty_id in sorted(grouped):
        for course in sorted(grouped[faculty_id]):
            sections = grouped[faculty_id][course]
            course_short = SHORT_NAMES.get(course, course)
            faculty_tag = faculty_initials.get(faculty_id, faculty_id)

            for section_a, section_b in combinations(sections, 2):
                for day in DAYS:
                    periods_a = [
                        period
                        for period in PERIODS
                        if section_grid[section_a][day][period] == course
                    ]
                    periods_b = [
                        period
                        for period in PERIODS
                        if section_grid[section_b][day][period] == course
                    ]
                    if not periods_a or not periods_b:
                        continue

                    period_a, period_b = min(
                        ((pa, pb) for pa in periods_a for pb in periods_b),
                        key=lambda pair: (abs(pair[0] - pair[1]), min(pair), max(pair)),
                    )
                    is_consecutive = abs(period_a - period_b) == 1
                    total_pairs += 1
                    if is_consecutive:
                        consecutive_pairs += 1
                    entries_by_day[day].append(
                        {
                            "course": course_short,
                            "faculty": faculty_tag,
                            "section_a": section_a,
                            "period_a": period_a,
                            "section_b": section_b,
                            "period_b": period_b,
                            "is_consecutive": is_consecutive,
                        }
                    )

    print("\n" + "═" * 50)
    print("SAME-DAY CONSECUTIVE ANALYSIS")
    print("═" * 50)
    print("Checking all days where same faculty teaches")
    print("same course twice on same day...")

    for day in DAYS:
        day_entries = entries_by_day[day]
        if not day_entries:
            continue

        day_entries.sort(
            key=lambda item: (
                item["course"],
                item["faculty"],
                min(item["period_a"], item["period_b"]),
                max(item["period_a"], item["period_b"]),
                item["section_a"],
                item["section_b"],
            )
        )

        print(f"\nDay: {day}")
        current_group = None
        for item in day_entries:
            group = (item["course"], item["faculty"])
            if group != current_group:
                print(f"  {item['course']} - Faculty {item['faculty']}:")
                current_group = group
            label = "CONSECUTIVE ✓" if item["is_consecutive"] else "GAP (not consecutive)"
            print(
                f"    Section {item['section_a']}: P{item['period_a']}  "
                f"Section {item['section_b']}: P{item['period_b']}  -> {label}"
            )

    non_consecutive_pairs = total_pairs - consecutive_pairs
    percentage = (100.0 * consecutive_pairs / total_pairs) if total_pairs else 0.0
    print("\nSummary:")
    print(f"  Total same-faculty same-day pairs : {total_pairs}")
    print(f"  Consecutive                       : {consecutive_pairs} ({percentage:.1f}%)")
    print(f"  Non-consecutive (gap)             : {non_consecutive_pairs}")
    print("═" * 50)

    return {
        "total_pairs": total_pairs,
        "consecutive_pairs": consecutive_pairs,
        "non_consecutive_pairs": non_consecutive_pairs,
        "percentage": percentage,
    }


def _build_solver(max_time_seconds, workers):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = workers
    return solver


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
    theory_hours = {course_code: 6 for course_code in courses_df["course_code"].tolist()}
    faculty_ids = faculty_df["faculty_id"].tolist()
    faculty_course_sections = _faculty_course_sections(assignment_map)

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
    reward_terms = []
    soft_counts = {
        "same_subject_same_day": 0,
        "back_to_back_same_subject": 0,
        "p7_p9_theory_non_lab_day": 0,
        "intra_day_gaps_p1_p6": 0,
        "daily_load_imbalance": 0,
    }

    is_used = {}
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
            model.Add(dcount == 6)
            daily_count[(section, day)] = dcount

            for p2 in range(2, 6):
                hb = model.NewBoolVar(f"has_before_{section}_{day}_{p2}")
                ha = model.NewBoolVar(f"has_after_{section}_{day}_{p2}")
                gv = model.NewBoolVar(f"gap_{section}_{day}_{p2}")
                gap_var[(section, day, p2)] = gv

                before_vars = [is_used[(section, day, p)] for p in range(1, p2)]
                after_vars = [is_used[(section, day, p)] for p in range(p2 + 1, 7)]

                model.Add(sum(before_vars) >= hb)
                model.Add(sum(before_vars) <= len(before_vars) * hb)
                model.Add(sum(after_vars) >= ha)
                model.Add(sum(after_vars) <= len(after_vars) * ha)

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

    for faculty_id, course_map in faculty_course_sections.items():
        for course, sections in course_map.items():
            for section_i, section_j in combinations(sections, 2):
                for day in DAYS:
                    day_reward_vars = []
                    for period in range(1, 6):
                        key_i = (section_i, course, day, period)
                        key_j_next = (section_j, course, day, period + 1)
                        if key_i in x and key_j_next in x:
                            consec = model.NewBoolVar(
                                f"consec_{faculty_id}_{course}_{section_i}_{section_j}_{day}_{period}"
                            )
                            model.Add(consec <= x[key_i])
                            model.Add(consec <= x[key_j_next])
                            model.Add(consec >= x[key_i] + x[key_j_next] - 1)
                            reward_terms.append(consec)
                            day_reward_vars.append(consec)

                        key_j = (section_j, course, day, period)
                        key_i_next = (section_i, course, day, period + 1)
                        if key_j in x and key_i_next in x:
                            consec_rev = model.NewBoolVar(
                                f"consec_rev_{faculty_id}_{course}_{section_i}_{section_j}_{day}_{period}"
                            )
                            model.Add(consec_rev <= x[key_j])
                            model.Add(consec_rev <= x[key_i_next])
                            model.Add(consec_rev >= x[key_j] + x[key_i_next] - 1)
                            reward_terms.append(consec_rev)
                            day_reward_vars.append(consec_rev)

                    if day_reward_vars:
                        model.Add(sum(day_reward_vars) <= 1)

    for section in SECTIONS:
        for day in DAYS:
            if day in section_lab_days[section]:
                continue
            for course in course_codes:
                for period in (7, 8, 9):
                    key = (section, course, day, period)
                    if key in x:
                        penalty_terms.append(x[key] * 50)

    penalty_expr = sum(penalty_terms) if penalty_terms else 0
    reward_expr = sum(reward_terms) if reward_terms else 0

    model.Minimize(penalty_expr)
    penalty_solver = _build_solver(max_time_seconds=120, workers=8)
    penalty_status = penalty_solver.Solve(model)
    penalty_status_name = penalty_solver.StatusName(penalty_status)

    if penalty_status != cp_model.OPTIMAL:
        print("PHASE 3 FAILED - penalty stage did not reach OPTIMAL")
        print(f"Penalty stage status: {penalty_status_name}")
        raise ValueError("Phase 3 penalty optimization not optimal")

    best_penalty = int(round(penalty_solver.ObjectiveValue()))
    model.Add(penalty_expr == best_penalty)
    for key in x:
        model.AddHint(x[key], penalty_solver.Value(x[key]))
    model.Maximize(reward_expr)

    reward_solver = _build_solver(max_time_seconds=60, workers=8)
    reward_status = reward_solver.Solve(model)
    reward_status_name = reward_solver.StatusName(reward_status)

    if reward_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        final_solver = reward_solver
    else:
        final_solver = penalty_solver

    status_name = penalty_status_name

    for (section, course, day, period), var in x.items():
        if final_solver.Value(var) == 1:
            section_grid[section][day][period] = course
            faculty_id = assignment_map[course][section]
            if faculty_grid[faculty_id][day][period] is not None:
                raise ValueError(
                    f"Faculty grid collision for {faculty_id} at {day} P{period}"
                )
            faculty_grid[faculty_id][day][period] = f"{course} ({section})"

    for section in SECTIONS:
        for course in course_codes:
            for day in DAYS:
                count_for_day = sum(
                    1
                    for period in PERIODS
                    if section_grid[section][day][period] == course
                )
                if count_for_day > 1:
                    soft_counts["same_subject_same_day"] += count_for_day - 1
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
                final_solver.Value(daily_count[(section, day)]) - 6
            )
            for p2 in range(2, 6):
                if final_solver.Value(gap_var[(section, day, p2)]) == 1:
                    soft_counts["intra_day_gaps_p1_p6"] += 1

    _print_section_grid(section_grid, section="A")
    consecutive_stats = _print_same_day_consecutive_analysis(
        section_grid=section_grid,
        assignment_map=assignment_map,
        faculty_df=faculty_df,
    )
    print(
        f"Consecutive pairs achieved : {consecutive_stats['consecutive_pairs']} / "
        f"{consecutive_stats['total_pairs']} total ({consecutive_stats['percentage']:.1f}%)"
    )
    print(f"Solver status              : {status_name}")
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
        "consecutive_analysis": consecutive_stats,
        "penalty_status": penalty_status_name,
        "optimal_penalty": best_penalty,
        "reward_stage_status": reward_status_name,
    }


if __name__ == "__main__":
    solve_theory()






