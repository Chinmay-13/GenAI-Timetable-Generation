from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DAYS, PERIODS, LAB_PERIODS, SHORT_NAMES, MAX_HOURS, SECTIONS
from src.phase3.theory_scheduler import solve_theory

DATA_DIR = str(PROJECT_ROOT / "data")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs")


def _write_csv_with_fallback(df, path: Path):
    try:
        df.to_csv(path, index=False)
        return str(path)
    except PermissionError:
        fallback_path = path.with_name(f"{path.stem}.latest{path.suffix}")
        df.to_csv(fallback_path, index=False)
        print(
            f"Warning: could not overwrite {path.name}; "
            f"wrote latest data to {fallback_path.name} instead."
        )
        return str(fallback_path)


def _write_text_with_fallback(content: str, path: Path):
    try:
        path.write_text(content, encoding="utf-8")
        return str(path)
    except PermissionError:
        fallback_path = path.with_name(f"{path.stem}.latest{path.suffix}")
        fallback_path.write_text(content, encoding="utf-8")
        print(
            f"Warning: could not overwrite {path.name}; "
            f"wrote latest data to {fallback_path.name} instead."
        )
        return str(fallback_path)


def _initials(name):
    tokens = [t.strip(".") for t in str(name).split() if t and t.lower() != "prof."]
    return "".join(t[0].upper() for t in tokens[:3]) if tokens else "NA"


def _build_short_name_map(data_dir: str) -> dict:
    """
    Build course_code -> display_name lookup.
    Priority:
      1. SHORT_NAMES from config.py
      2. short_name column in courses.csv (if present)
      3. Fallback: course_code itself
    """
    name_map: dict = {}
    try:
        df = pd.read_csv(f"{data_dir}/courses.csv")
        if "short_name" in df.columns:
            for _, row in df.iterrows():
                code = str(row["course_code"]).strip()
                sn   = str(row["short_name"]).strip()
                if sn and sn.lower() not in {"nan", "none", ""}:
                    name_map[code] = sn
    except Exception:
        pass  # courses.csv unreadable — fall through to SHORT_NAMES / code
    # SHORT_NAMES overrides the CSV (highest priority)
    name_map.update(SHORT_NAMES)
    return name_map


def _set_faculty_cell(faculty_table, faculty_id, day, period, cell):
    current = faculty_table[faculty_id][day][period]
    if current not in {"----", cell}:
        raise ValueError(
            f"Faculty output collision for {faculty_id} at {day} P{period}: "
            f"'{current}' vs '{cell}'"
        )
    faculty_table[faculty_id][day][period] = cell


def generate_outputs(result=None, data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    if result is None:
        result = solve_theory(data_dir=data_dir)

    section_grid = result["section_grid"]
    faculty_grid = result["faculty_grid"]
    assignment_map = result["assignment_map"]
    lab_details = result["lab_details"]
    elective_details = result.get("elective_details", [])
    soft_violations = result.get("soft_violations", {})

    faculty_df = pd.read_csv(f"{data_dir}/faculty.csv")
    faculty_name = dict(zip(faculty_df["faculty_id"], faculty_df["name"]))
    faculty_designation = dict(zip(faculty_df["faculty_id"], faculty_df["designation"]))
    faculty_initials = {fid: _initials(name) for fid, name in faculty_name.items()}

    # Short-name lookup: SHORT_NAMES > courses.csv short_name > raw code
    sn = _build_short_name_map(data_dir)

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
            for p in LAB_PERIODS:
                lab_lookup[(sec, day, p)] = {
                    "course": course,
                    "faculty_id": faculty_id,
                    "pair": sections,
                }

    elective_lookup = {}
    for item in elective_details:
        course = item["course_code"]
        day = item["day"]
        sections = item["sections"]
        faculty_id = item["faculty_id"]
        for p in range(int(item["period_start"]), int(item["period_end"]) + 1):
            for sec in sections:
                elective_lookup[(sec, day, p)] = {
                    "course": course,
                    "faculty_id": faculty_id,
                    "sections": sections,
                    "elective_group": item.get("elective_group"),
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
                    cell = f"{sn.get(course, course)} LAB"
                else:
                    elective_info = elective_lookup.get((section, day, p))
                    if elective_info is not None:
                        faculty_id = elective_info["faculty_id"]
                    else:
                        faculty_id = assignment_map.get(value, {}).get(section)
                    if faculty_id is None:
                        cell = sn.get(value, value)
                    else:
                        cell = f"{sn.get(value, value)} ({faculty_initials[faculty_id]})"
                row[f"P{p}"] = cell
            rows.append(row)

        df = pd.DataFrame(rows, columns=["Day"] + period_cols)
        _write_csv_with_fallback(df, out_path / f"section_{section}_timetable.csv")

    faculty_ids = faculty_df["faculty_id"].tolist()
    faculty_table = {
        fid: {day: {p: "----" for p in PERIODS} for day in DAYS}
        for fid in faculty_ids
    }

    for item in lab_details:
        fid = item["faculty_id"]
        pair = "+".join(item["sections"])
        cell = f"{sn.get(item['course_code'], item['course_code'])} LAB ({pair})"
        for p in LAB_PERIODS:
            _set_faculty_cell(faculty_table, fid, item["day"], p, cell)

    for item in elective_details:
        fid = item["faculty_id"]
        sections_str = "+".join(item["sections"])
        cell = f"{sn.get(item['course_code'], item['course_code'])} ({sections_str})"
        for p in range(int(item["period_start"]), int(item["period_end"]) + 1):
            _set_faculty_cell(faculty_table, fid, item["day"], p, cell)

    for section in SECTIONS:
        for day in DAYS:
            for p in PERIODS:
                value = section_grid[section][day][p]
                if value is None:
                    continue
                if isinstance(value, str) and value.endswith("_LAB"):
                    continue
                if (section, day, p) in elective_lookup:
                    continue
                fid = assignment_map.get(value, {}).get(section)
                if fid is None:
                    continue
                _set_faculty_cell(
                    faculty_table,
                    fid,
                    day,
                    p,
                    f"{sn.get(value, value)} ({section})",
                )

    for fid in faculty_ids:
        rows = []
        for day in DAYS:
            row = {"Day": day}
            for p in PERIODS:
                row[f"P{p}"] = faculty_table[fid][day][p]
            rows.append(row)
        df = pd.DataFrame(rows, columns=["Day"] + period_cols)
        _write_csv_with_fallback(df, out_path / f"faculty_{fid}_timetable.csv")

    total_theory_slots = 0
    total_lab_slots = 0
    total_elective_slots = 0
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
                    if (section, day, p) in elective_lookup:
                        total_elective_slots += 1

    report_lines = []
    report_lines.append("TIMETABLE SUMMARY REPORT")
    report_lines.append("========================")
    report_lines.append(f"Total sections scheduled: {len(SECTIONS)}")
    report_lines.append(f"Total theory/elective slots placed: {total_theory_slots}")
    report_lines.append(f"Total fixed elective slots placed: {total_elective_slots}")
    report_lines.append(
        f"Total lab slots placed: {total_lab_slots} (12 pairs x 2 periods x 2 sections = 48)"
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

    _write_text_with_fallback("\n".join(report_lines), out_path / "summary_report.txt")

    print(f"PHASE 4 COMPLETE - all outputs generated in {out_path}/")

    return {
        "total_theory_slots": total_theory_slots,
        "total_elective_slots": total_elective_slots,
        "total_lab_slots": total_lab_slots,
        "output_dir": str(out_path),
    }


if __name__ == "__main__":
    generate_outputs()
