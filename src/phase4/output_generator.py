from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.theory_scheduler import solve_theory, DAYS, PERIODS, SHORT_NAMES, MAX_HOURS

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
SECTIONS = [chr(ord("A") + i) for i in range(12)]


def _initials(name):
    tokens = [t.strip(".") for t in str(name).split() if t and t.lower() != "prof."]
    return "".join(t[0].upper() for t in tokens[:3]) if tokens else "NA"


def generate_outputs(result=None, data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    if result is None:
        result = solve_theory(data_dir=data_dir)

    section_grid = result["section_grid"]
    faculty_grid = result["faculty_grid"]
    assignment_map = result["assignment_map"]
    lab_details = result["lab_details"]
    soft_violations = result.get("soft_violations", {})

    faculty_df = pd.read_csv(f"{data_dir}/faculty.csv")
    faculty_name = dict(zip(faculty_df["faculty_id"], faculty_df["name"]))
    faculty_designation = dict(zip(faculty_df["faculty_id"], faculty_df["designation"]))
    faculty_initials = {fid: _initials(name) for fid, name in faculty_name.items()}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    period_cols = [f"P{p}" for p in PERIODS]

    lab_lookup = {}
    for item in lab_details:
        course = item["course_code"]
        day = item["day"]
        sections = item["sections"]
        faculty_id = item["faculty_id"]
        for sec in sections:
            for p in [7, 8, 9]:
                lab_lookup[(sec, day, p)] = {
                    "course": course,
                    "faculty_id": faculty_id,
                    "pair": sections,
                }

    for section in SECTIONS:
        rows = []
        for day in DAYS:
            row = {"Day": day}
            for p in PERIODS:
                value = section_grid[section][day][p]
                if value is None:
                    cell = "----"
                elif isinstance(value, str) and value.endswith("_LAB"):
                    course = value.replace("_LAB", "")
                    cell = f"{SHORT_NAMES.get(course, course)} LAB"
                else:
                    faculty_id = assignment_map[value][section]
                    cell = f"{SHORT_NAMES.get(value, value)} ({faculty_initials[faculty_id]})"
                row[f"P{p}"] = cell
            rows.append(row)

        df = pd.DataFrame(rows, columns=["Day"] + period_cols)
        df.to_csv(out_path / f"section_{section}_timetable.csv", index=False)

    faculty_ids = faculty_df["faculty_id"].tolist()
    faculty_table = {
        fid: {day: {p: "----" for p in PERIODS} for day in DAYS}
        for fid in faculty_ids
    }

    for section in SECTIONS:
        for day in DAYS:
            for p in PERIODS:
                value = section_grid[section][day][p]
                if value is None:
                    continue
                if isinstance(value, str) and value.endswith("_LAB"):
                    info = lab_lookup.get((section, day, p))
                    if not info:
                        continue
                    fid = info["faculty_id"]
                    pair = "+".join(info["pair"])
                    cell = f"{SHORT_NAMES.get(info['course'], info['course'])} LAB ({pair})"
                    faculty_table[fid][day][p] = cell
                else:
                    fid = assignment_map[value][section]
                    faculty_table[fid][day][p] = f"{SHORT_NAMES.get(value, value)} ({section})"

    for fid in faculty_ids:
        rows = []
        for day in DAYS:
            row = {"Day": day}
            for p in PERIODS:
                row[f"P{p}"] = faculty_table[fid][day][p]
            rows.append(row)
        df = pd.DataFrame(rows, columns=["Day"] + period_cols)
        df.to_csv(out_path / f"faculty_{fid}_timetable.csv", index=False)

    total_theory_slots = 0
    total_lab_slots = 0
    for section in SECTIONS:
        for day in DAYS:
            for p in PERIODS:
                value = section_grid[section][day][p]
                if value is None:
                    continue
                if isinstance(value, str) and value.endswith("_LAB"):
                    total_lab_slots += 1
                else:
                    total_theory_slots += 1

    report_lines = []
    report_lines.append("TIMETABLE SUMMARY REPORT")
    report_lines.append("========================")
    report_lines.append(f"Total sections scheduled: {len(SECTIONS)}")
    report_lines.append(f"Total theory slots placed: {total_theory_slots}")
    report_lines.append(
        f"Total lab slots placed: {total_lab_slots} (12 pairs x 3 periods x 2 sections = 72)"
    )
    report_lines.append("Soft constraint violations:")
    report_lines.append(
        f"  same_subject_same_day: {soft_violations.get('same_subject_same_day', 0)}"
    )
    report_lines.append(
        f"  back_to_back_same_subject: {soft_violations.get('back_to_back_same_subject', 0)}"
    )
    report_lines.append("")
    report_lines.append("Faculty Load Table")
    report_lines.append("faculty_id | name | total_hours | max_hours | status")

    for fid in faculty_ids:
        total = sum(
            1
            for day in DAYS
            for p in PERIODS
            if faculty_table[fid][day][p] != "----"
        )
        max_h = MAX_HOURS.get(faculty_designation[fid], 16)
        status = "OK" if total <= max_h else "OVERLOAD"
        report_lines.append(
            f"{fid} | {faculty_name[fid]} | {total} | {max_h} | {status}"
        )

    (out_path / "summary_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("PHASE 4 COMPLETE - all outputs generated in outputs/")

    return {
        "total_theory_slots": total_theory_slots,
        "total_lab_slots": total_lab_slots,
        "output_dir": str(out_path),
    }


if __name__ == "__main__":
    generate_outputs()
