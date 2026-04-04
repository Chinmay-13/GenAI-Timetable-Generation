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

from config import (
    DAYS,
    PERIODS,
    THEORY_PERIODS,
    LAB_PERIODS,
    SECTIONS,
    SHORT_NAMES,
    MAX_HOURS,
    PENALTY_STAGE_TIME,
    REWARD_STAGE_TIME,
    NUM_WORKERS,
    PENALTY_GAP,
    PENALTY_LAB_WINDOW,
    PENALTY_BACK_TO_BACK,
    PENALTY_SAME_DAY,
    REWARD_CONSECUTIVENESS,
    CONSECUTIVENESS_TIME_LIMIT,
    get_theory_periods,
    PENALTY_PREF_TIME,
    PENALTY_PREF_NO_BTB,
    PENALTY_PREF_FREE_DAY,
    PENALTY_ROOM_OVERCAP,
)
from src.phase1.assignment_builder import build_assignment_map
from src.phase2.lab_scheduler import lock_labs

DATA_DIR = str(PROJECT_ROOT / "data")


def _is_lab_token(value):
    return isinstance(value, str) and value.endswith("_LAB")


def _initials(name):
    tokens = [t.strip(".") for t in str(name).split() if t and t.lower() != "prof."]
    return "".join(t[0].upper() for t in tokens[:3]) if tokens else "NA"


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _print_section_grid(section_grid, section="A"):
    print(f"\nSection {section} timetable:\n")
    header = ["Day", "P1", "P2", "P3", "P4", "P5", "P6"]
    print(" | ".join(f"{h:<6}" for h in header))
    print("-" * 60)

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


def _extract_room_assignment_map(
    room_assigned: dict,
    solver,
    room_id_to_name: dict,
) -> dict:
    """
    Extract room assignments from the CP-SAT solution.

    Returns
    -------
    dict : (section, day, period) -> room_name
    """
    result = {}
    for (section, day, period, rid), var in room_assigned.items():
        if solver.Value(var) == 1:
            result[(section, day, period)] = room_id_to_name.get(rid, rid)
    return result


def solve_theory(
    assignment_map=None,
    section_grid=None,
    faculty_grid=None,
    room_grid=None,
    lab_details=None,
    elective_details=None,
    elective_slots=None,
    data_dir=DATA_DIR,
    rooms_df=None,
):
    courses_df = pd.read_csv(f"{data_dir}/courses.csv")
    faculty_df = pd.read_csv(f"{data_dir}/faculty.csv")

    # ── Load faculty preferences (graceful: missing columns → defaults) ───────
    faculty_prefs = {}
    for _, frow in faculty_df.iterrows():
        fid = str(frow["faculty_id"]).strip()
        faculty_prefs[fid] = {
            "pref_time":          str(frow.get("pref_time", "none") or "none").strip().lower()
                                  if "pref_time" in faculty_df.columns else "none",
            "pref_no_backtoback": _as_bool(frow.get("pref_no_backtoback", False))
                                  if "pref_no_backtoback" in faculty_df.columns else False,
            "pref_no_teaching_day": str(frow.get("pref_no_teaching_day", "none") or "none").strip()
                                    if "pref_no_teaching_day" in faculty_df.columns else "none",
        }

    if "is_elective" not in courses_df.columns:
        courses_df["is_elective"] = False
    else:
        courses_df["is_elective"] = courses_df["is_elective"].map(_as_bool)

    if assignment_map is None:
        assignment_map = build_assignment_map(data_dir=data_dir)
    if (
        section_grid is None
        or faculty_grid is None
        or room_grid is None
        or lab_details is None
        or elective_details is None
    ):
        section_grid, faculty_grid, room_grid, lab_details, elective_details = lock_labs(
            data_dir=data_dir
        )

    # ── Pre-lock elective slots (defense-in-depth — Phase 2 fills the grid) ──
    if elective_slots is not None and not elective_slots.empty:
        for _, erow in elective_slots.iterrows():
            day = str(erow["day"]).strip()
            sections = [s.strip() for s in str(erow["enrolled_sections"]).split(",")]
            p_start = int(erow["period_start"])
            p_end = int(erow["period_end"])
            course_code = str(erow["course_code"]).strip()
            faculty_id = str(erow["faculty_id"]).strip()
            for section in sections:
                if section not in SECTIONS:
                    continue
                for p in range(p_start, p_end + 1):
                    if section_grid[section][day][p] is None:
                        section_grid[section][day][p] = course_code
                    if faculty_id in faculty_grid and faculty_grid[faculty_id][day][p] is None:
                        faculty_grid[faculty_id][day][p] = course_code

    course_codes = courses_df["course_code"].tolist()
    elective_course_codes = set(
        courses_df[courses_df["is_elective"] == True]["course_code"].tolist()
    )
    scheduled_course_codes = [
        course for course in course_codes if course not in elective_course_codes
    ]
    theory_periods_map = {}
    for _, course_row in courses_df.iterrows():
        course_code = course_row["course_code"]
        credits = int(course_row["credits"])
        has_lab = course_row["has_lab"]
        theory_periods_map[course_code] = get_theory_periods(credits, has_lab)

    total_theory = sum(theory_periods_map[course] for course in scheduled_course_codes)
    base_daily_target = total_theory // len(DAYS)
    extra_days = total_theory % len(DAYS)
    daily_targets = {
        day: base_daily_target + (1 if day_idx < extra_days else 0)
        for day_idx, day in enumerate(DAYS)
    }
    faculty_ids = faculty_df["faculty_id"].tolist()
    faculty_course_sections = _faculty_course_sections(
        {course: assignment_map.get(course, {}) for course in scheduled_course_codes}
    )

    section_lab_days = {section: set() for section in SECTIONS}
    for detail in lab_details:
        for section in detail["sections"]:
            section_lab_days[section].add(detail["day"])

    model = cp_model.CpModel()
    x = {}

    for course in scheduled_course_codes:
        for section in sorted(assignment_map.get(course, {})):
            for day in DAYS:
                for period in PERIODS:
                    if section_grid[section][day][period] is not None:
                        continue
                    if day in section_lab_days[section] and period in LAB_PERIODS:
                        continue
                    x[(section, course, day, period)] = model.NewBoolVar(
                        f"x_{section}_{course}_{day}_{period}"
                    )

    for section in SECTIONS:
        for course in scheduled_course_codes:
            vars_for_course = [
                x[(section, course, day, period)]
                for day in DAYS
                for period in PERIODS
                if (section, course, day, period) in x
            ]
            model.Add(sum(vars_for_course) == int(theory_periods_map[course]))

    for section in SECTIONS:
        for day in DAYS:
            for period in PERIODS:
                vars_in_slot = [
                    x[(section, course, day, period)]
                    for course in scheduled_course_codes
                    if (section, course, day, period) in x
                ]
                if vars_in_slot:
                    model.Add(sum(vars_in_slot) <= 1)

    for faculty_id in faculty_ids:
        for day in DAYS:
            for period in PERIODS:
                vars_for_faculty = []
                for course in scheduled_course_codes:
                    for section, assigned_faculty in assignment_map.get(course, {}).items():
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

    # ── Hard faculty total-slot cap per designation ──────────────────────────
    faculty_designation = dict(zip(faculty_df["faculty_id"], faculty_df["designation"]))
    for faculty_id in faculty_ids:
        all_faculty_vars = []
        for course in scheduled_course_codes:
            for section, assigned_faculty in assignment_map.get(course, {}).items():
                if assigned_faculty != faculty_id:
                    continue
                for day in DAYS:
                    for period in PERIODS:
                        key = (section, course, day, period)
                        if key in x:
                            all_faculty_vars.append(x[key])
        if all_faculty_vars:
            desig = faculty_designation.get(faculty_id, "Asst Prof")
            cap = MAX_HOURS.get(desig, 20)
            model.Add(sum(all_faculty_vars) <= cap)


    penalty_terms = []
    reward_terms = []
    soft_counts = {
        "same_subject_same_day": 0,
        "back_to_back_same_subject": 0,
        "lab_window_theory_non_lab_day": 0,
        "intra_day_gaps": 0,
        "daily_load_imbalance": 0,
    }

    is_used = {}
    gap_var = {}
    daily_count = {}

    for section in SECTIONS:
        for day in DAYS:
            for period in THEORY_PERIODS:
                used = model.NewBoolVar(f"is_used_{section}_{day}_{period}")
                is_used[(section, day, period)] = used

                slot_vars = [
                    x[(section, course, day, period)]
                    for course in scheduled_course_codes
                    if (section, course, day, period) in x
                ]
                if slot_vars:
                    model.Add(sum(slot_vars) >= used)
                    model.Add(sum(slot_vars) <= len(scheduled_course_codes) * used)
                else:
                    model.Add(used == 0)

            # ── Hard compactness: occupied slots form a prefix from P1 ────
            # occupied(p) = 1 if pre-locked OR theory-assigned.
            # Enforce occupied(p) >= occupied(p+1) for adjacent theory periods.
            for idx in range(len(THEORY_PERIODS) - 1):
                p_curr = THEORY_PERIODS[idx]
                p_next = THEORY_PERIODS[idx + 1]
                curr_locked = section_grid[section][day][p_curr] is not None
                next_locked = section_grid[section][day][p_next] is not None
                if curr_locked and next_locked:
                    pass  # both occupied — trivially compact
                elif next_locked and not curr_locked:
                    # next always occupied, so curr must be too
                    model.Add(is_used[(section, day, p_curr)] == 1)
                elif curr_locked and not next_locked:
                    pass  # curr always occupied, next unconstrained
                else:
                    # neither locked — monotonic
                    model.Add(
                        is_used[(section, day, p_curr)]
                        >= is_used[(section, day, p_next)]
                    )

            dcount = model.NewIntVar(0, len(THEORY_PERIODS), f"daily_count_{section}_{day}")
            model.Add(
                dcount
                == sum(is_used[(section, day, period)] for period in THEORY_PERIODS)
            )
            model.Add(dcount == daily_targets[day])
            daily_count[(section, day)] = dcount

            for p2 in range(2, len(THEORY_PERIODS) + 1):  # P2 to P4
                p_start = THEORY_PERIODS[0]
                p_end = THEORY_PERIODS[-1]
                hb = model.NewBoolVar(f"has_before_{section}_{day}_{p2}")
                ha = model.NewBoolVar(f"has_after_{section}_{day}_{p2}")
                gv = model.NewBoolVar(f"gap_{section}_{day}_{p2}")
                gap_var[(section, day, p2)] = gv

                before_vars = [is_used[(section, day, p)] for p in range(p_start, p2)]
                after_vars = [is_used[(section, day, p)] for p in range(p2 + 1, p_end + 1)]

                if before_vars:
                    model.Add(sum(before_vars) >= hb)
                    model.Add(sum(before_vars) <= len(before_vars) * hb)
                else:
                    model.Add(hb == 0)
                if after_vars:
                    model.Add(sum(after_vars) >= ha)
                    model.Add(sum(after_vars) <= len(after_vars) * ha)
                else:
                    model.Add(ha == 0)

                model.Add(gv <= hb)
                model.Add(gv <= ha)
                model.Add(gv <= 1 - is_used[(section, day, p2)])
                model.Add(gv >= hb + ha + (1 - is_used[(section, day, p2)]) - 2)

                penalty_terms.append(gv * PENALTY_GAP)

    for section in SECTIONS:
        for course in scheduled_course_codes:
            for day in DAYS:
                day_vars = [
                    x[(section, course, day, period)]
                    for period in PERIODS
                    if (section, course, day, period) in x
                ]
                if day_vars:
                    over = model.NewIntVar(0, len(PERIODS), f"over_{section}_{course}_{day}")
                    model.Add(over >= sum(day_vars) - 1)
                    penalty_terms.append(over * PENALTY_SAME_DAY)

                for period in range(1, len(PERIODS)):  # adjacent pairs within P1-P6
                    k1 = (section, course, day, period)
                    k2 = (section, course, day, period + 1)
                    if k1 in x and k2 in x:
                        adj = model.NewBoolVar(f"adj_{section}_{course}_{day}_{period}")
                        model.Add(adj <= x[k1])
                        model.Add(adj <= x[k2])
                        model.Add(adj >= x[k1] + x[k2] - 1)
                        penalty_terms.append(adj * PENALTY_BACK_TO_BACK)

    for faculty_id, course_map in faculty_course_sections.items():
        for course, sections in course_map.items():
            for section_i, section_j in combinations(sections, 2):
                for day in DAYS:
                    day_reward_vars = []
                    for period in range(1, len(THEORY_PERIODS)):  # adjacent in P1-P4
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

    # ── Penalize theory in lab window (P5-P6) on non-lab days ────────────
    for section in SECTIONS:
        for day in DAYS:
            if day in section_lab_days[section]:
                continue
            for course in scheduled_course_codes:
                for period in LAB_PERIODS:
                    key = (section, course, day, period)
                    if key in x:
                        penalty_terms.append(x[key] * PENALTY_LAB_WINDOW)

    # ── A. Faculty time preference (morning=P1-P2, afternoon=P3-P4) ───────────
    # PENALTY_PREF_TIME=8 — below all structural penalties; feasibility first.
    _MORNING_PERIODS    = {1, 2}   # P1, P2
    _AFTERNOON_PERIODS  = {3, 4}   # P3, P4
    for faculty_id, prefs in faculty_prefs.items():
        pt = prefs["pref_time"]
        if pt not in ("morning", "afternoon"):
            continue
        bad_periods = _AFTERNOON_PERIODS if pt == "morning" else _MORNING_PERIODS
        for course in scheduled_course_codes:
            for section, assigned_fac in assignment_map.get(course, {}).items():
                if assigned_fac != faculty_id:
                    continue
                for day in DAYS:
                    for period in bad_periods:
                        key = (section, course, day, period)
                        if key in x:
                            penalty_terms.append(x[key] * PENALTY_PREF_TIME)

    # ── B. Faculty no-back-to-back preference ─────────────────────────────────
    # PENALTY_PREF_NO_BTB=6 — penalise consecutive faculty slots on same day.
    for faculty_id, prefs in faculty_prefs.items():
        if not prefs["pref_no_backtoback"]:
            continue
        # Collect all x vars for this faculty across (section, course)
        # keyed by (day, period)
        fac_slot_vars: dict[tuple, list] = {}  # (day, period) -> list[BoolVar]
        for course in scheduled_course_codes:
            for section, assigned_fac in assignment_map.get(course, {}).items():
                if assigned_fac != faculty_id:
                    continue
                for day in DAYS:
                    for period in THEORY_PERIODS:
                        key = (section, course, day, period)
                        if key in x:
                            dp_key = (day, period)
                            fac_slot_vars.setdefault(dp_key, []).append(x[key])
        # For each consecutive period pair on each day, penalise co-occupancy
        for day in DAYS:
            for p in range(THEORY_PERIODS[0], THEORY_PERIODS[-1]):  # P1-P3
                vars_p  = fac_slot_vars.get((day, p), [])
                vars_p1 = fac_slot_vars.get((day, p + 1), [])
                if not vars_p or not vars_p1:
                    continue
                # used_p = 1 iff faculty has a slot at (day, p)
                used_p  = model.NewBoolVar(f"fbtb_{faculty_id}_{day}_{p}")
                used_p1 = model.NewBoolVar(f"fbtb_{faculty_id}_{day}_{p+1}")
                model.Add(sum(vars_p)  >= used_p)
                model.Add(sum(vars_p)  <= len(vars_p)  * used_p)
                model.Add(sum(vars_p1) >= used_p1)
                model.Add(sum(vars_p1) <= len(vars_p1) * used_p1)
                btb_pair = model.NewBoolVar(f"fbtb_pair_{faculty_id}_{day}_{p}")
                model.Add(btb_pair <= used_p)
                model.Add(btb_pair <= used_p1)
                model.Add(btb_pair >= used_p + used_p1 - 1)
                penalty_terms.append(btb_pair * PENALTY_PREF_NO_BTB)

    # ── C. Faculty preferred free-day penalty ─────────────────────────────────
    # PENALTY_PREF_FREE_DAY=10 — penalise each slot taught on preferred free day.
    for faculty_id, prefs in faculty_prefs.items():
        free_day = prefs["pref_no_teaching_day"]
        if free_day.lower() == "none" or free_day not in DAYS:
            continue
        for course in scheduled_course_codes:
            for section, assigned_fac in assignment_map.get(course, {}).items():
                if assigned_fac != faculty_id:
                    continue
                for period in THEORY_PERIODS:
                    key = (section, course, free_day, period)
                    if key in x:
                        penalty_terms.append(x[key] * PENALTY_PREF_FREE_DAY)

    # ── Room assignment inside CP-SAT ───────────────────────────────────────
    # Only active when rooms_df is provided; if None → skip (legacy path).
    room_assigned = {}          # (section, day, period, room_id) -> BoolVar
    theory_room_ids = []        # room_id strings eligible for theory
    room_cap = {}               # room_id -> capacity
    room_id_to_name = {}        # room_id -> room_name (for output)
    _room_vars_active = False

    if rooms_df is not None:
        # Filter to theory-eligible rooms only (CLASSROOM / LECTURE_HALL)
        eligible = rooms_df[
            rooms_df["room_type"].str.upper().isin(["CLASSROOM", "LECTURE_HALL"])
        ].copy()
        theory_room_ids = eligible["room_id"].astype(str).tolist()
        room_cap        = dict(zip(eligible["room_id"].astype(str),
                                   eligible["capacity"].astype(int)))
        room_id_to_name = dict(zip(eligible["room_id"].astype(str),
                                   eligible["room_name"].astype(str)))

        if theory_room_ids:
            _room_vars_active = True

            # Build set of (room_name, day, period) pre-occupied by Phase 2
            preoccupied_room_names: set[tuple] = set()
            if room_grid:
                for rname, day_map in room_grid.items():
                    for day, period_map in day_map.items():
                        for period, token in period_map.items():
                            if token is not None:
                                preoccupied_room_names.add((rname, day, period))

            # Build reverse map room_name -> room_id for pre-occupation check
            room_name_to_id = {v: k for k, v in room_id_to_name.items()}

            # ASSUMPTION: section size = 60 (default; no enrollment data)
            SECTION_SIZE = 60

            for section in SECTIONS:
                for day in DAYS:
                    for period in THEORY_PERIODS:
                        if (section, day, period) not in is_used:
                            continue  # period not in theory window
                        for rid in theory_room_ids:
                            rname = room_id_to_name[rid]
                            # Pre-lock: if room pre-occupied by Phase 2 fix it to 0
                            if (rname, day, period) in preoccupied_room_names:
                                # Room unavailable — don't create a decision var
                                continue
                            room_assigned[(section, day, period, rid)] = \
                                model.NewBoolVar(f"ra_{section}_{day}_{period}_{rid}")

            # Hard A: each occupied theory slot gets exactly one room
            for section in SECTIONS:
                for day in DAYS:
                    for period in THEORY_PERIODS:
                        if (section, day, period) not in is_used:
                            continue
                        slot_room_vars = [
                            room_assigned[(section, day, period, rid)]
                            for rid in theory_room_ids
                            if (section, day, period, rid) in room_assigned
                        ]
                        used_var = is_used[(section, day, period)]
                        if slot_room_vars:
                            # occupied → exactly one room
                            model.Add(sum(slot_room_vars) == 1).OnlyEnforceIf(used_var)
                            # free → no room
                            model.Add(sum(slot_room_vars) == 0).OnlyEnforceIf(used_var.Not())
                        else:
                            # no available rooms for this slot (all pre-occupied)
                            # force slot to be free (shouldn't happen with 12 classrooms)
                            model.Add(used_var == 0)

            # Hard B: no two sections share a room in the same slot
            for day in DAYS:
                for period in THEORY_PERIODS:
                    for rid in theory_room_ids:
                        competing = [
                            room_assigned[(s, day, period, rid)]
                            for s in SECTIONS
                            if (s, day, period, rid) in room_assigned
                        ]
                        if len(competing) > 1:
                            model.Add(sum(competing) <= 1)

            # Soft: penalise under-capacity room assignments
            for (section, day, period, rid), var in room_assigned.items():
                if room_cap.get(rid, SECTION_SIZE) < SECTION_SIZE:
                    penalty_terms.append(var * PENALTY_ROOM_OVERCAP)

            print(f"[Phase 3] Room vars added: "
                  f"{len(theory_room_ids)} theory rooms ×"
                  f" {len(SECTIONS)} sections × {len(THEORY_PERIODS)} periods")

    # ── Solve penalty stage ──────────────────────────────────────────────────
    penalty_expr = sum(penalty_terms) if penalty_terms else 0
    reward_expr  = sum(reward_terms)  if reward_terms  else 0

    model.Minimize(penalty_expr)
    penalty_solver = _build_solver(max_time_seconds=PENALTY_STAGE_TIME, workers=NUM_WORKERS)
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

    reward_solver = _build_solver(max_time_seconds=REWARD_STAGE_TIME, workers=NUM_WORKERS)
    reward_status = reward_solver.Solve(model)
    reward_status_name = reward_solver.StatusName(reward_status)

    if reward_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        final_solver = reward_solver
    else:
        final_solver = penalty_solver

    status_name = penalty_status_name

    # ── Consecutiveness for same-faculty same-section same-day ───────────
    # Phase A enforces hard constraint; Phase B falls back to soft reward
    # if Phase A is infeasible. This ensures consecutive placement
    # whenever the problem structure allows it.
    consecutiveness_phase = "N/A"

    # Collect (faculty, section, course, day) combinations where faculty
    # teaches 2+ theory slots for the SAME section on the SAME day.
    # We build this from the current solution to see where it might apply.
    same_sec_tuples = []
    for course in scheduled_course_codes:
        for section in SECTIONS:
            faculty_id = assignment_map.get(course, {}).get(section)
            if faculty_id is None:
                continue
            for day in DAYS:
                day_vars_list = [
                    (period, x[(section, course, day, period)])
                    for period in THEORY_PERIODS
                    if (section, course, day, period) in x
                ]
                if len(day_vars_list) >= 2:
                    count_val = sum(
                        final_solver.Value(v) for _, v in day_vars_list
                    )
                    if count_val >= 2:
                        same_sec_tuples.append((faculty_id, section, course, day, day_vars_list))

    if same_sec_tuples:
        print(f"\nConsecutiveness: {len(same_sec_tuples)} (faculty, section, course, day) groups with 2+ slots detected")

        # Phase A: Try hard constraint
        model_a = model.Clone()
        consec_hard_constraints = []
        for faculty_id, section, course, day, day_vars_list in same_sec_tuples:
            # For periods where var exists, require at least one adjacent used pair
            adj_vars = []
            for i in range(len(day_vars_list) - 1):
                p1, v1 = day_vars_list[i]
                p2, v2 = day_vars_list[i + 1]
                if p2 == p1 + 1:  # adjacent periods
                    adj_pair = model_a.NewBoolVar(
                        f"consec_hard_{faculty_id}_{section}_{course}_{day}_{p1}"
                    )
                    model_a.Add(adj_pair <= v1)
                    model_a.Add(adj_pair <= v2)
                    model_a.Add(adj_pair >= v1 + v2 - 1)
                    adj_vars.append(adj_pair)
            if adj_vars:
                model_a.Add(sum(adj_vars) >= 1)
                consec_hard_constraints.append((faculty_id, section, course, day))

        if consec_hard_constraints:
            solver_a = _build_solver(max_time_seconds=CONSECUTIVENESS_TIME_LIMIT, workers=NUM_WORKERS)
            status_a = solver_a.Solve(model_a)

            if status_a in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                print("  Phase A (hard consecutiveness) FEASIBLE — using this solution")
                consecutiveness_phase = "Phase A (hard)"
                final_solver = solver_a
            else:
                print("  Phase A INFEASIBLE — falling back to Phase B (soft reward)")
                # Phase B: Add soft reward for consecutiveness
                consec_reward_terms = []
                for faculty_id, section, course, day, day_vars_list in same_sec_tuples:
                    for i in range(len(day_vars_list) - 1):
                        p1, v1 = day_vars_list[i]
                        p2, v2 = day_vars_list[i + 1]
                        if p2 == p1 + 1:
                            soft_adj = model.NewBoolVar(
                                f"consec_soft_{faculty_id}_{section}_{course}_{day}_{p1}"
                            )
                            model.Add(soft_adj <= v1)
                            model.Add(soft_adj <= v2)
                            model.Add(soft_adj >= v1 + v2 - 1)
                            consec_reward_terms.append(soft_adj * REWARD_CONSECUTIVENESS)

                if consec_reward_terms:
                    combined_reward = reward_expr + sum(consec_reward_terms)
                    model.Maximize(combined_reward)
                    solver_b = _build_solver(max_time_seconds=REWARD_STAGE_TIME, workers=NUM_WORKERS)
                    status_b = solver_b.Solve(model)
                    if status_b in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                        final_solver = solver_b
                    consecutiveness_phase = "Phase B (soft)"
        else:
            consecutiveness_phase = "N/A (no adjacent period pairs to constrain)"
    else:
        print("\nConsecutiveness: No same-faculty same-section same-day 2+ slot groups found")
        consecutiveness_phase = "N/A (no qualifying groups)"

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
        for course in scheduled_course_codes:
            for day in DAYS:
                count_for_day = sum(
                    1
                    for period in PERIODS
                    if section_grid[section][day][period] == course
                )
                if count_for_day > 1:
                    soft_counts["same_subject_same_day"] += count_for_day - 1
            for day in DAYS:
                for period in range(1, len(PERIODS)):
                    if (
                        section_grid[section][day][period] == course
                        and section_grid[section][day][period + 1] == course
                    ):
                        soft_counts["back_to_back_same_subject"] += 1

            for day in DAYS:
                if day not in section_lab_days[section]:
                    for period in LAB_PERIODS:
                        if section_grid[section][day][period] == course:
                            soft_counts["lab_window_theory_non_lab_day"] += 1

    for section in SECTIONS:
        for day in DAYS:
            soft_counts["daily_load_imbalance"] += abs(
                final_solver.Value(daily_count[(section, day)]) - daily_targets[day]
            )
            for p2 in range(2, len(THEORY_PERIODS) + 1):
                if final_solver.Value(gap_var[(section, day, p2)]) == 1:
                    soft_counts["intra_day_gaps"] += 1

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
    print(f"Consecutiveness phase      : {consecutiveness_phase}")
    print(
        f"\nPHASE 3 COMPLETE - theory slots filled | solver status: {status_name}"
    )

    return {
        "section_grid": section_grid,
        "faculty_grid": faculty_grid,
        "room_grid": room_grid,
        "assignment_map": assignment_map,
        "lab_details": lab_details,
        "elective_details": elective_details,
        "solver_status": status_name,
        "soft_violations": soft_counts,
        "consecutive_analysis": consecutive_stats,
        "penalty_status": penalty_status_name,
        "optimal_penalty": best_penalty,
        "reward_stage_status": reward_status_name,
        "consecutiveness_phase": consecutiveness_phase,
        "room_assignment_map": _extract_room_assignment_map(
            room_assigned, final_solver, room_id_to_name
        ) if _room_vars_active else {},
    }


if __name__ == "__main__":
    solve_theory()
