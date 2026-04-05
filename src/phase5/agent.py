"""
agent.py — LangChain tool-use agent for timetable queries and operations.

The agent has READ and WRITE capabilities:
  READ tools:
    - get_section_timetable   : full weekly timetable for a section
    - get_faculty_schedule    : full weekly schedule for a faculty member
    - find_free_slots         : free periods for a faculty on a given day
    - get_summary_stats       : summary statistics from summary_report.txt
    - find_substitute         : substitute suggestions for absent faculty

  WRITE tools:
    - commit_substitute       : commit a substitute assignment to timetable
    - list_agent_ops          : list recent agent operations
    - rollback_last_operation : rollback a committed substitution
    - generate_session_summary: generate session summary report

Falls back gracefully if GROQ_API_KEY is missing.

Usage:
  python src/phase5/agent.py
"""

from pathlib import Path
import sys
import os
import tempfile

_AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from dotenv import load_dotenv
load_dotenv(_AGENT_ROOT / ".env")

import pandas as pd
import config
from config import OUTPUT_DIR, SECTIONS, LAB_PERIODS
from config import resolve_output_path
from src.phase5.agent_ops import (
    backup_timetable, log_operation, list_operations, rollback_operation
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_section_csv(section: str, output_dir=None) -> str:
    if output_dir is not None:
        path = Path(output_dir) / f"section_{section.strip().upper()}_timetable.csv"
    else:
        path = resolve_output_path(f"section_{section.strip().upper()}_timetable.csv")
    try:
        df = pd.read_csv(path)
        return df.to_string(index=False)
    except FileNotFoundError:
        return f"No timetable found for section {section}. Run run_all.py first."


def _read_faculty_csv(faculty_id: str, output_dir=None) -> str:
    if output_dir is not None:
        path = Path(output_dir) / f"faculty_{faculty_id.strip().upper()}_timetable.csv"
    else:
        path = resolve_output_path(f"faculty_{faculty_id.strip().upper()}_timetable.csv")
    try:
        df = pd.read_csv(path)
        return df.to_string(index=False)
    except FileNotFoundError:
        return f"No timetable found for faculty {faculty_id}. Run run_all.py first."


def _is_faculty_free(faculty_id: str, day: str,
                     period_start: int, period_end: int,
                     output_dir=None) -> bool:
    """Check if faculty has no bookings for the given period range on day."""
    if output_dir is not None:
        path = Path(output_dir) / f"faculty_{faculty_id}_timetable.csv"
    else:
        path = resolve_output_path(f"faculty_{faculty_id}_timetable.csv")
    if not path.exists():
        return False
    df = pd.read_csv(path)
    row = df[df["Day"] == day]
    if row.empty:
        return False
    for p in range(period_start, period_end + 1):
        col = f"P{p}"
        if col in row.columns and str(row.iloc[0][col]).strip() not in \
           ["", "----", "nan"]:
            return False
    return True


def _atomic_write_csv(path: Path, df: pd.DataFrame):
    """Write DataFrame to CSV atomically via temp file + rename."""
    dir_ = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w", dir=str(dir_), suffix=".tmp",
        delete=False, encoding="utf-8"
    ) as tmp:
        df.to_csv(tmp, index=False)
        tmp_path = tmp.name
    os.replace(tmp_path, str(path))


_DAY_ALIASES = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
}


def _normalize_day_input(day_raw: str) -> str | None:
    return _DAY_ALIASES.get(str(day_raw).strip().lower())


def _normalize_period_input(period_raw, *, max_period: int = 4) -> str | None:
    period_clean = str(period_raw).strip().upper().lstrip("P")
    try:
        period_int = int(period_clean)
    except ValueError:
        return None
    if period_int < 1 or period_int > max_period:
        return None
    return f"P{period_int}"


def _load_rooms_inventory() -> pd.DataFrame:
    return pd.read_csv(_AGENT_ROOT / "data" / "rooms.csv")


# ── LangChain Tools ──────────────────────────────────────────────────────────

def _make_tools(agent_output_dir=None, sem_id: str = None):
    """
    Build all LangChain tools.  If agent_output_dir is given all output reads
    (section CSVs, faculty CSVs, room_assignment.csv, summary_report.txt) use
    that directory instead of the legacy resolve_output_path().
    sem_id is forwarded to write-path functions (commit_schedule_change, find_substitute)
    so they operate on the correct semester's output tree.
    """
    _out = Path(agent_output_dir) if agent_output_dir else None

    def _rop(filename: str) -> Path:
        """Resolve an output file path, honouring the active semester dir."""
        if _out is not None:
            return _out / filename
        return resolve_output_path(filename)

    try:
        from langchain.tools import tool
    except ImportError:
        return None

    @tool
    def get_section_timetable(section: str) -> str:
        """
        Get the full weekly timetable for a section.
        Input: section letter (A through L)
        Returns: timetable as formatted text
        """
        return _read_section_csv(section, output_dir=_out)

    @tool
    def get_faculty_schedule(faculty_id: str) -> str:
        """
        Get the full weekly schedule for a faculty member.
        Input: faculty_id — the faculty ID string such as F01, F02, F03, F04, etc.
               This is ALWAYS a faculty ID like F01, NEVER a section letter like A or B.
               Example: faculty_id="F04"
        Returns: full weekly schedule as formatted text
        """
        return _read_faculty_csv(faculty_id, output_dir=_out)

    @tool
    def find_free_slots(faculty_id_and_day: str) -> str:
        """
        Find free periods for a faculty on a given day.
        Input: "faculty_id,day" e.g. "F04,Monday"
        Returns: list of free periods
        """
        parts = [p.strip() for p in faculty_id_and_day.split(",", 1)]
        if len(parts) != 2:
            return "Input must be 'faculty_id,day' e.g. 'F04,Monday'"
        faculty_id, day = parts
        path = _rop(f"faculty_{faculty_id.upper()}_timetable.csv")
        try:
            df = pd.read_csv(path)
            row = df[df["Day"].str.lower() == day.lower()]
            if row.empty:
                return f"No data for {faculty_id} on {day}"
            free = [
                f"P{p}" for p in range(1, 7)
                if str(row[f"P{p}"].values[0]).strip() == "----"
            ]
            if free:
                return f"Free slots for {faculty_id} on {day}: {', '.join(free)}"
            return f"{faculty_id} has no free slots on {day}"
        except FileNotFoundError:
            return f"No schedule found for {faculty_id}. Run run_all.py first."

    @tool
    def get_summary_stats(query: str) -> str:
        """
        Get summary statistics about the timetable.
        Input: any string (e.g., "faculty load", "violations", "theory slots")
        Returns: relevant stats from the summary report
        """
        report_path = _rop("summary_report.txt")
        try:
            return report_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "Summary report not found. Run run_all.py first."

    @tool
    def find_substitute(absent_faculty_and_day: str) -> str:
        """
        Find a substitute for an absent faculty member.
        Input: "faculty_id,day" e.g. "F04,Monday"
        Returns: substitute suggestions.
        For lab periods (P5-P6) the full 2-period block is always treated
        atomically — only candidates free for both P5 and P6 are shown.
        """
        parts = [p.strip() for p in absent_faculty_and_day.split(",", 1)]
        if len(parts) != 2:
            return "Input must be 'faculty_id,day' e.g. 'F04,Monday'"
        faculty_id, day = parts
        try:
            from src.phase5.substitute import find_substitute as _find_sub
            result = _find_sub(faculty_id, day, sem_id=sem_id)
            lines = [f"Absent: {result.get('absent_faculty', faculty_id)} on {day}"]
            for sub in result.get("substitutions", []):
                lines.append(
                    f"  P{sub['period']}: "
                    f"{sub.get('course_short', sub.get('course', '?'))} "
                    f"-> {sub.get('substitute_name', sub.get('substitute', '?'))} "
                    f"({sub.get('match_type', '?')})"
                )
            unresolved = result.get("unresolved", [])
            if unresolved:
                lines.append(f"  Unresolved: {unresolved}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error finding substitute: {e}"

    @tool
    def commit_substitute(input_str: str) -> str:
        """
        Commit a substitute teacher assignment to the timetable.
        Input format: "section,day,period_start,period_end,absent_faculty,
                       substitute_faculty,reason"
        Example: "A,Monday,5,6,F03,F07,Lab coverage for DDCO"
        Validates substitute availability before writing.
        Updates section CSV, faculty CSVs, summary report, and RAG index.
        Creates backup and logs the operation.
        """
        parts = [p.strip() for p in input_str.split(",", 6)]
        if len(parts) < 7:
            return "Error: expected 7 comma-separated fields."
        section, day, p_start, p_end, absent, substitute, reason = parts
        p_start, p_end = int(p_start), int(p_end)

        # Validate substitute availability
        if not _is_faculty_free(substitute, day, p_start, p_end, output_dir=_out):
            return (
                f"Cannot commit: {substitute} is not free on {day} "
                f"P{p_start}-P{p_end}."
            )

        # Delegate ALL writes to sync_manager
        try:
            from src.phase5.sync_manager import commit_schedule_change
            result = commit_schedule_change({
                "section":          section,
                "day":              day,
                "period_start":     p_start,
                "period_end":       p_end,
                "original_faculty": absent,
                "new_faculty":      substitute,
                "change_type":      "substitute",
                "reason":           reason,
            }, sem_id=sem_id)
            return result["message"]
        except Exception as exc:
            return f"Commit failed (rolled back): {exc}"

    @tool
    def list_agent_ops(input_str: str) -> str:
        """
        List recent autonomous agent operations (substitutions/swaps).
        Input: number of recent ops to show (e.g. "10"). Default: 10.
        """
        try:
            limit = int(input_str.strip()) if input_str.strip() else 10
        except ValueError:
            limit = 10
        ops = list_operations(limit)
        if not ops:
            return "No agent operations logged yet."
        lines = []
        for op in ops:
            lines.append(
                f"[{op['operation_id']}] {op['action'].upper()} — "
                f"Section {op['section_id']}, {op['day']}, "
                f"P{op['period_range'][0]}-P{op['period_range'][1]} | "
                f"Absent: {op['absent_faculty']} → "
                f"Sub: {op['substitute_faculty']} | "
                f"Result: {op['commit_result']}"
            )
        return "\n".join(lines)

    @tool
    def rollback_last_operation(input_str: str) -> str:
        """
        Rollback a committed substitution using its operation ID.
        Input: operation_id (e.g. "a3f2b1c4")
        Restores the timetable to its pre-substitution state.
        """
        op_id = input_str.strip()
        if not op_id:
            return "Error: provide an operation_id."
        return rollback_operation(op_id)

    @tool
    def generate_session_summary(input_str: str) -> str:
        """
        Generate a human-readable summary report of all agent actions
        taken in the current session. Writes to outputs/agent_ops/
        Input: optional session label (e.g. "Monday absence session")
        """
        from datetime import datetime as _dt
        label = input_str.strip() or "Agent Session"
        ops = list_operations(50)
        if not ops:
            return "No operations to summarise."

        lines = [
            f"# {label} — Agent Operations Summary",
            f"Generated: {_dt.now().isoformat()}",
            f"Total operations: {len(ops)}",
            "",
        ]
        for op in ops:
            lines += [
                f"## [{op['operation_id']}] {op['action'].upper()}",
                f"- Section: {op['section_id']} | Day: {op['day']} | "
                f"Periods: P{op['period_range'][0]}-P{op['period_range'][1]}",
                f"- Absent faculty: {op['absent_faculty']}",
                f"- Substitute: {op['substitute_faculty']}",
                f"- Result: {op['commit_result']}",
                f"- Reasoning: {'; '.join(op['reasoning_chain'])}",
                "",
            ]

        summary_text = "\n".join(lines)
        ops_dir = _AGENT_ROOT / "outputs" / "agent_ops"
        ops_dir.mkdir(parents=True, exist_ok=True)
        out_path = ops_dir / f"summary_{_dt.now().strftime('%Y%m%dT%H%M%S')}.txt"
        out_path.write_text(summary_text, encoding="utf-8")
        return summary_text

    @tool
    def get_absent_periods(input_str: str) -> str:
        """
        Get the actual periods a faculty member teaches on a given day.
        Input format: "faculty_id,day"
        Example: "F01,Monday"
        Returns the specific periods they are scheduled to teach.
        Use this BEFORE commit_substitute to know which periods to cover.
        """
        parts = [p.strip() for p in input_str.split(",", 1)]
        if len(parts) < 2:
            return "Error: provide faculty_id,day"
        fid, day = parts

        path = _rop(f"faculty_{fid}_timetable.csv")
        if not path or not Path(path).exists():
            return f"No timetable found for {fid}"

        import pandas as pd
        df = pd.read_csv(path)
        row = df[df["Day"].str.strip() == day.strip()]
        if row.empty:
            return f"No schedule found for {fid} on {day}"

        row = row.iloc[0]
        periods = []
        for p in range(1, 7):  # P1-P6
            col = f"P{p}"
            val = str(row.get(col, "")).strip()
            if val and val not in ["", "----", "nan"]:
                periods.append(f"P{p}: {val}")

        if not periods:
            return f"{fid} has no classes on {day}"

        return f"{fid} on {day}:\n" + "\n".join(periods)

    @tool
    def find_free_rooms(input_str: str) -> str:
        """
        Find available (unoccupied) classrooms for a given day and period.
        Input format: "day,period[,capacity_needed]"
        Examples:
          "Monday,3"        → all free rooms on Monday P3
          "Tuesday,2,60"    → free rooms on Tuesday P2 with capacity >= 60
        Returns a list of free rooms with their capacities.
        """
        parts = [p.strip() for p in input_str.split(",")]
        if len(parts) < 2:
            return "Input must be 'day,period' or 'day,period,capacity_needed'"

        day = parts[0]
        # Accept "P3" or "3"
        period_raw = parts[1].upper().lstrip("P")
        try:
            period = int(period_raw)
        except ValueError:
            return f"Invalid period '{parts[1]}'. Use a number or 'P<n>'."

        capacity_needed = 0
        if len(parts) >= 3:
            try:
                capacity_needed = int(parts[2])
            except ValueError:
                return f"Invalid capacity '{parts[2]}'. Must be an integer."

        # Load room assignment CSV
        csv_path = _rop("room_assignment.csv")
        if not csv_path.exists():
            return (
                "room_assignment.csv not found. "
                "Run python run_all.py to generate room assignments first."
            )

        try:
            import pandas as _pd
            df = _pd.read_csv(csv_path)
        except Exception as exc:
            return f"Error reading room_assignment.csv: {exc}"

        # Rooms occupied in this slot
        slot_rows = df[
            (df["Day"].str.lower() == day.lower()) &
            (df["Period"] == f"P{period}")
        ]
        occupied = set(slot_rows["Room"].tolist())
        occupied.discard("ROOM_UNASSIGNED")

        # All classrooms from rooms.csv
        rooms_path = _AGENT_ROOT / "data" / "rooms.csv"
        if not rooms_path.exists():
            rooms_path = resolve_output_path("..") / "data" / "rooms.csv"
        try:
            rooms_df = _pd.read_csv(
                Path(__file__).resolve().parents[2] / "data" / "rooms.csv"
            )
        except Exception as exc:
            return f"Error reading rooms.csv: {exc}"

        classrooms = rooms_df[
            rooms_df["room_type"].str.upper().isin(["CLASSROOM", "LECTURE_HALL"])
        ]
        if capacity_needed > 0:
            classrooms = classrooms[classrooms["capacity"] >= capacity_needed]

        free = classrooms[~classrooms["room_name"].isin(occupied)]

        if free.empty:
            return (
                f"No free classrooms on {day} P{period}"
                + (f" with capacity >= {capacity_needed}" if capacity_needed else "")
                + "."
            )

        lines = [f"Free classrooms on {day} P{period}:"]
        for _, row in free.iterrows():
            lines.append(
                f"  {row['room_name']} (capacity: {row['capacity']}, "
                f"floor: {row['floor']})"
            )
        return "\n".join(lines)

    # ── TOOL 7 — get_faculty_workload ────────────────────────────────────────
    @tool
    def get_faculty_workload(input_str: str) -> str:
        """
        Get total workload for a faculty member this week.
        Input: faculty_id (e.g. "F04") or part of their name (e.g. "Sharma")
        Returns: total hours, day-by-day breakdown, courses taught, and
                 whether they are over their weekly cap.
        """
        query = input_str.strip().upper()
        import pandas as _pd
        from config import DATA_DIR as _DATA_DIR, MAX_HOURS as _MAX_HOURS
        from config import resolve_output_path as _rop

        # Resolve faculty_id from id or partial name
        try:
            fac_df = _pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "faculty.csv")
        except Exception as exc:
            return f"Cannot load faculty.csv: {exc}"

        match = fac_df[fac_df["faculty_id"].str.upper() == query]
        if match.empty:
            match = fac_df[fac_df["name"].str.upper().str.contains(query, na=False)]
        if match.empty:
            return f"No faculty found matching '{input_str}'."

        row = match.iloc[0]
        fid        = str(row["faculty_id"]).strip()
        name       = str(row["name"]).strip()
        designation = str(row["designation"]).strip()
        max_h      = _MAX_HOURS.get(designation, 16)

        fac_path = _rop(f"faculty_{fid}_timetable.csv")
        if not fac_path.exists():
            return f"Timetable not found for {fid}. Run run_all.py first."

        df = _pd.read_csv(fac_path)
        period_cols = [f"P{p}" for p in range(1, 7)]
        DAYS_ORDER  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        total = 0
        day_breakdown = []
        courses_seen: set = set()

        for _, day_row in df.iterrows():
            day = str(day_row["Day"]).strip()
            day_slots = []
            for col in period_cols:
                cell = str(day_row.get(col, "----")).strip()
                if cell not in ("----", "", "nan"):
                    total += 1
                    day_slots.append(f"{col}: {cell}")
                    # Extract course short name before " ("
                    course_part = cell.split(" (")[0].strip()
                    courses_seen.add(course_part)
            if day_slots:
                day_breakdown.append(f"  {day}: {', '.join(day_slots)}")

        status = "⚠ OVERLOAD" if total > max_h else "✓ Within limit"
        lines = [
            f"Faculty: {name} ({fid}) — {designation}",
            f"Weekly hours: {total} / {max_h} max  [{status}]",
            f"Courses taught: {', '.join(sorted(courses_seen)) or 'None'}",
            "Schedule:",
        ] + (day_breakdown if day_breakdown else ["  (no classes scheduled)"])
        return "\n".join(lines)

    # ── TOOL 8 — get_free_faculty ────────────────────────────────────────────
    @tool
    def get_free_faculty(input_str: str) -> str:
        """
        Find all faculty with no class in a given day + period slot.
        Input format: "day,period"  e.g. "Monday,3" or "Tuesday,P2"
        Returns: list of free faculty with their designation.
        """
        parts = [p.strip() for p in input_str.split(",", 1)]
        if len(parts) != 2:
            return "Input must be 'day,period' e.g. 'Monday,3'"
        day_raw, period_raw = parts
        period_str = period_raw.upper().lstrip("P")
        try:
            period_int = int(period_str)
        except ValueError:
            return f"Invalid period '{period_raw}'."
        col = f"P{period_int}"

        import pandas as _pd
        fac_path = Path(__file__).resolve().parents[2] / "data" / "faculty.csv"
        try:
            fac_df = _pd.read_csv(fac_path)
        except Exception as exc:
            return f"Cannot load faculty.csv: {exc}"

        free_list = []
        for _, row in fac_df.iterrows():
            fid  = str(row["faculty_id"]).strip()
            name = str(row["name"]).strip()
            desig = str(row["designation"]).strip()
            tpath = _rop(f"faculty_{fid}_timetable.csv")
            if not tpath.exists():
                continue
            try:
                tdf = _pd.read_csv(tpath)
                day_row = tdf[tdf["Day"].str.lower() == day_raw.lower()]
                if day_row.empty:
                    continue
                cell = str(day_row.iloc[0].get(col, "----")).strip()
                if cell in ("----", "", "nan"):
                    free_list.append(f"  {fid} — {name} ({desig})")
            except Exception:
                continue

        if not free_list:
            return f"All faculty are busy on {day_raw} P{period_int}."
        header = f"Free faculty on {day_raw} P{period_int} ({len(free_list)} total):"
        return "\n".join([header] + free_list)

    # ── TOOL 9 — get_room_availability ──────────────────────────────────────
    @tool
    def get_room_availability(day: str, period: str) -> str:
        """
        Find which rooms are free and which are occupied in a given theory slot.
        Inputs:
          day: accepts Tuesday, Tue, TUE, tue, tuesday
          period: accepts P2, p2, 2, or "2"

        Reads outputs/room_assignment.csv and data/rooms.csv.
        Returns free rooms with type and capacity, plus occupied rooms by section.
        """
        import pandas as _pd

        normalized_day = _normalize_day_input(day)
        if normalized_day is None:
            return (
                f"Unrecognized day '{day}'. "
                "Use Monday, Tuesday, Wednesday, Thursday, or Friday "
                "(full name or common abbreviation)."
            )

        period_label = _normalize_period_input(period, max_period=4)
        if period_label is None:
            return (
                f"Unrecognized period '{period}'. "
                "Use P1, P2, P3, P4 or just 1, 2, 3, 4. "
                "Room assignment data is only generated for theory periods."
            )

        ra_path = _rop("room_assignment.csv")
        if not ra_path.exists():
            return "Room assignment data not found. Please run the pipeline first with: python run_all.py"

        try:
            ra_df = _pd.read_csv(ra_path)
        except Exception as exc:
            return f"Error reading outputs/room_assignment.csv: {exc}"

        slot_df = ra_df[
            (ra_df["Day"].astype(str).str.strip().str.lower() == normalized_day.lower()) &
            (ra_df["Period"].astype(str).str.strip().str.upper() == period_label)
        ].copy()

        if slot_df.empty:
            return (
                f"No room assignment rows found for {normalized_day} {period_label}. "
                "Please regenerate outputs with: python run_all.py"
            )

        try:
            rooms_df = _load_rooms_inventory()
        except Exception as exc:
            return f"Error reading data/rooms.csv: {exc}"

        rooms_df = rooms_df.copy()
        rooms_df["room_name"] = rooms_df["room_name"].astype(str).str.strip()
        rooms_df["room_type"] = rooms_df["room_type"].astype(str).str.strip()

        occupied_rows = slot_df[
            slot_df["Room"].astype(str).str.strip().ne("ROOM_UNASSIGNED")
        ].copy()
        occupied_rows["Room"] = occupied_rows["Room"].astype(str).str.strip()
        occupied_room_names = set(occupied_rows["Room"].tolist())

        free_rooms = rooms_df[
            ~rooms_df["room_name"].isin(occupied_room_names)
        ].sort_values(["room_type", "capacity", "room_name"], ascending=[True, False, True])

        occupied_sections = sorted(slot_df["Section"].astype(str).str.strip().unique().tolist())
        unassigned_rows = slot_df[
            slot_df["Room"].astype(str).str.strip().eq("ROOM_UNASSIGNED")
        ]

        lines = [
            f"Room availability for {normalized_day} {period_label}:",
            f"Total rooms in inventory: {len(rooms_df)}",
            f"Sections with classes: {len(occupied_sections)} ({', '.join(occupied_sections)})",
            f"Occupied rooms: {len(occupied_room_names)}",
            f"Free rooms: {len(free_rooms)}",
        ]

        if not unassigned_rows.empty:
            lines.append(
                "Unassigned sections: "
                + ", ".join(unassigned_rows["Section"].astype(str).str.strip().tolist())
            )

        lines.append("")
        lines.append(f"FREE ROOMS ({len(free_rooms)}):")
        for _, room_row in free_rooms.iterrows():
            lines.append(
                f"  - {room_row['room_name']} | type={room_row['room_type']} | "
                f"capacity={room_row['capacity']}"
            )

        lines.append("")
        lines.append(f"OCCUPIED ROOMS ({len(occupied_room_names)}):")
        if occupied_rows.empty:
            lines.append("  - None")
        else:
            for room_name, room_group in occupied_rows.sort_values(["Room", "Section"]).groupby("Room"):
                sections_using_room = ", ".join(
                    f"Section {row['Section']} ({row['Course']}, {row['Faculty']})"
                    for _, row in room_group.iterrows()
                )
                lines.append(f"  - {room_name}: {sections_using_room}")

        return "\n".join(lines)

    # ── TOOL 10 — detect_schedule_conflicts ──────────────────────────────────
    @tool
    def detect_schedule_conflicts(input_str: str) -> str:
        """
        Scan all section and faculty CSVs for scheduling conflicts.
        Input: any string (ignored) — just pass an empty string or "check".
        Detects:
          • Same faculty assigned to two sections in the same slot
          • Same room assigned to two sections in the same slot
        Returns a structured conflict report.
        """
        import pandas as _pd
        from config import SECTIONS as _SECTIONS, DAYS as _DAYS

        period_cols = [f"P{p}" for p in range(1, 7)]
        # map (day, period, faculty_initials) -> [section, ...]
        faculty_slot: dict = {}
        # map (day, period, room_name) -> [section, ...]
        room_slot: dict = {}

        for section in _SECTIONS:
            sp = _rop(f"section_{section}_timetable.csv")
            if not sp.exists():
                continue
            try:
                df = _pd.read_csv(sp)
            except Exception:
                continue
            for _, row in df.iterrows():
                day = str(row["Day"]).strip()
                for col in period_cols:
                    cell = str(row.get(col, "----")).strip()
                    if cell in ("----", "", "nan") or "LAB" in cell.upper():
                        continue
                    # Extract initials "DDCO (RP)" → "RP"
                    import re as _re
                    m = _re.search(r"\(([^)]+)\)", cell)
                    if m:
                        key = (day, col, m.group(1).strip())
                        faculty_slot.setdefault(key, []).append(section)

        # Check room_assignment.csv for room conflicts
        ra_path = _rop("room_assignment.csv")
        if ra_path.exists():
            ra_df = _pd.read_csv(ra_path)
            for _, row in ra_df.iterrows():
                room = str(row.get("Room", "")).strip()
                if room in ("ROOM_UNASSIGNED", "", "nan"):
                    continue
                key = (str(row["Day"]).strip(), str(row["Period"]).strip(), room)
                room_slot.setdefault(key, []).append(str(row["Section"]).strip())

        conflicts = []
        for (day, period, fac_initials), sections in faculty_slot.items():
            if len(sections) > 1:
                conflicts.append(
                    f"FACULTY CONFLICT: {fac_initials} teaching "
                    f"sections {'+'.join(sections)} simultaneously "
                    f"on {day} {period}"
                )
        for (day, period, room), sections in room_slot.items():
            if len(sections) > 1:
                conflicts.append(
                    f"ROOM CONFLICT: {room} assigned to "
                    f"sections {'+'.join(sections)} simultaneously "
                    f"on {day} {period}"
                )

        if not conflicts:
            return "✓ No conflicts detected across all section and faculty timetables."
        header = f"⚠ {len(conflicts)} conflict(s) found:"
        return "\n".join([header] + [f"  • {c}" for c in conflicts])

    # ── TOOL 11 — get_weekly_stats ───────────────────────────────────────────
    @tool
    def get_weekly_stats(input_str: str) -> str:
        """
        Return key statistics from the timetable summary report.
        Input: any string (ignored).
        Returns: sections count, theory/lab slot totals, soft-constraint
                 violations, and a faculty load summary.
        """
        report_path = _rop("summary_report.txt")
        if not report_path.exists():
            return "summary_report.txt not found. Run run_all.py first."
        text = report_path.read_text(encoding="utf-8")
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        def _grab(prefix):
            for l in lines:
                if l.startswith(prefix):
                    return l
            return None

        highlights = []
        for prefix in [
            "Total sections", "Total theory", "Total lab",
            "same_subject_same_day", "back_to_back_same_subject",
        ]:
            found = _grab(prefix)
            if found:
                highlights.append(f"  {found}")

        # Faculty load summary
        overloaded = [l for l in lines if "OVERLOAD" in l]
        ok_count   = sum(1 for l in lines if "| OK" in l)
        load_summary = [
            f"  Faculty OK: {ok_count}",
            f"  Faculty OVERLOAD: {len(overloaded)}",
        ]
        if overloaded:
            load_summary.append("  Overloaded faculty:")
            for ol in overloaded:
                parts = ol.split(" | ")
                if len(parts) >= 2:
                    load_summary.append(f"    → {parts[0].strip()} — {parts[1].strip()}")

        return "\n".join(
            ["=== Weekly Timetable Stats ==="]
            + highlights
            + ["Faculty load:"]
            + load_summary
        )

    # ── TOOL 12 — simulate_substitute ───────────────────────────────────────
    @tool
    def simulate_substitute(input_str: str) -> str:
        """
        Preview the top substitute candidates for a faculty on a given day/period.
        DOES NOT commit any changes — simulation only.
        Input format: "faculty_id,day,period"  e.g. "F04,Monday,3"
        Returns: top 3 candidates with match reasons and load info.
        Clearly states this is a preview — use commit_substitute to apply.
        """
        parts = [p.strip() for p in input_str.split(",", 2)]
        if len(parts) < 3:
            return "Input must be 'faculty_id,day,period' e.g. 'F04,Monday,3'"
        fid, day_raw, period_raw = parts
        period_str = period_raw.upper().lstrip("P")
        try:
            period_int = int(period_str)
        except ValueError:
            return f"Invalid period '{period_raw}'."
        period_col = f"P{period_int}"

        try:
            from src.phase5.substitute import (
                find_substitute as _find_sub,
                _rank_candidates, _collect_absent_slots,
                get_faculty_day_row, faculty_lookup, normalize_day,
                get_lab_block_periods, parse_timetable_cell,
            )
        except ImportError as exc:
            return f"Cannot import substitute module: {exc}"

        try:
            day = normalize_day(day_raw)
            lookup = faculty_lookup()
            fid_upper = fid.strip().upper()
            if fid_upper not in lookup:
                return f"Unknown faculty_id: {fid_upper}"

            absent_info = lookup[fid_upper]
            absent_designation = absent_info["designation"]

            # Get the slot info for the requested period
            day_row = get_faculty_day_row(fid_upper, day)
            cell = str(day_row.get(period_col, "----")).strip()
            if cell in ("----", "", "nan"):
                return (
                    f"{fid_upper} has no class on {day} {period_col}. "
                    "Nothing to simulate."
                )

            slot_info = parse_timetable_cell(cell)
            if slot_info is None:
                return f"Could not parse slot '{cell}' for {fid_upper} on {day} {period_col}."

            # Expand to lab block if needed
            block = get_lab_block_periods(period_col)
            ranked = _rank_candidates(
                absent_faculty_id=fid_upper,
                absent_designation=absent_designation,
                day=day,
                period=period_col,
                slot_info=slot_info,
            )

            if len(block) > 1:
                # Lab block: filter to candidates free for all periods
                filtered = []
                for cand in ranked:
                    all_free = all(
                        str(get_faculty_day_row(cand["faculty_id"], day).get(bp, "----")).strip()
                        in ("----", "", "nan")
                        for bp in block
                    )
                    if all_free:
                        filtered.append(cand)
                ranked = filtered

            top3 = ranked[:3]
            if not top3:
                return (
                    f"No substitute candidates found for {fid_upper} on "
                    f"{day} {period_col} ({slot_info['course']}).\n"
                    "All faculty are either busy or at load cap."
                )

            lines = [
                f"[SIMULATION] Top substitute candidates for {absent_info['name']} "
                f"({fid_upper}) on {day} {period_col} — {slot_info['course']}",
                "",
            ]
            for i, c in enumerate(top3, 1):
                lines.append(
                    f"  #{i} {c['faculty_id']} — {c['faculty_name']} "
                    f"({c['designation']})"
                )
                lines.append(
                    f"     Match: {c['match_type']}  |  "
                    f"Load after: {c['projected_load']}/{c['max_hours']}h"
                )
                lines.append(f"     Reason: {c['reason']}")
                lines.append("")

            lines.append(
                "⚠ This is a SIMULATION only — no changes have been made.\n"
                "  To apply, use: commit_substitute("
                f"{slot_info.get('section','?')},{day},{period_int},{period_int},"
                f"{fid_upper},<chosen_id>,<reason>)"
            )
            return "\n".join(lines)

        except Exception as exc:
            return f"Simulation error: {exc}"

    return [
        get_section_timetable,
        get_faculty_schedule,
        find_free_slots,
        get_summary_stats,
        find_substitute,
        commit_substitute,
        list_agent_ops,
        rollback_last_operation,
        generate_session_summary,
        get_absent_periods,
        find_free_rooms,
        get_faculty_workload,
        get_free_faculty,
        get_room_availability,
        detect_schedule_conflicts,
        get_weekly_stats,
        simulate_substitute,
    ]


# ── Agent creation ────────────────────────────────────────────────────────────

def _get_agent_system_prompt(sem_id: str = None) -> str:
    """Build system prompt for the agent."""
    try:
        from src.phase5.prompt_builder import build_system_prompt
        base = build_system_prompt(sem_id=sem_id)
    except Exception:
        base = "University timetable assistant. CSE dept, 12 sections A-L."

    return base + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMETABLE MANAGEMENT ASSISTANT — SYSTEM RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are a professional timetable management assistant for CSE department
college admin staff. You are accurate, concise, and helpful.

WHAT YOU CAN READ:
  • Section timetables (A–L)  — get_section_timetable
  • Faculty schedules          — get_faculty_schedule, get_absent_periods
  • Faculty workload & caps    — get_faculty_workload, get_weekly_stats
  • Free slots per faculty     — find_free_slots
  • Free faculty in a slot     — get_free_faculty
  • Room availability          — get_room_availability, find_free_rooms
  • Substitute candidates      — find_substitute, simulate_substitute
  • Schedule conflicts         — detect_schedule_conflicts
  • Committed operations log   — list_agent_ops

WHAT YOU CAN WRITE (only with user confirmation):
  • Commit a substitute assignment  — commit_substitute
  • Undo a committed change         — rollback_last_operation
  • Save a session summary          — generate_session_summary

WHAT YOU CANNOT DO:
  • Modify CP-SAT solver constraints or re-run the solver
  • Add new sections, courses, or faculty to the system
  • Edit raw CSV files directly
  • Access data outside the outputs/ and data/ directories

ABSENCE HANDLING — follow this exact order:
  1. get_absent_periods(faculty_id, day)         → confirm actual periods
  2. simulate_substitute(faculty_id, day, Pn)    → show top 3 candidates
  3. Wait for user to select a candidate
  4. commit_substitute(...)                      → apply after confirmation
  5. generate_session_summary()                  → log the session

RULES:
  • NEVER guess period numbers — always call get_absent_periods first
  • Lab periods P5–P6 must always be substituted as an atomic block
  • Always state when you are simulating vs committing
  • If data is unavailable, say so clearly and suggest running run_all.py
  • When answering questions about room availability, always read from
    outputs/room_assignment.csv using the get_room_availability tool.
    Never try to infer room assignments from section timetables — rooms
    are only known after room allocation is run.
"""


def create_timetable_agent(sem_id: str = None):
    """Create a tool-calling agent using bind_tools + manual ReAct loop."""
    if not config.GROQ_API_KEY:
        return None

    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from src.phase5.llm_wrapper import get_llm, safe_llm_call

    # ── Resolve output directory once ─────────────────────────────────────────
    if sem_id is not None:
        from config import get_sem_paths as _gsp
        agent_output_dir = _gsp(sem_id).output_dir
    else:
        agent_output_dir = None  # tools fall back to resolve_output_path()

    tools = _make_tools(agent_output_dir=agent_output_dir, sem_id=sem_id)
    if tools is None:
        return None

    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def run_agent(input_dict: dict) -> dict:
        query = input_dict.get("input", "")
        messages = [
            SystemMessage(content=_get_agent_system_prompt(sem_id=sem_id)),
            HumanMessage(content=query),
        ]
        reasoning = []

        for step in range(8):
            response = safe_llm_call(llm_with_tools, messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", []) or []
            if not tool_calls:
                # Agent is done — no more tool calls
                return {
                    "output": getattr(response, "content", str(response)),
                    "reasoning": reasoning,
                }

            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("args", {})
                tool_input = args if args else ""
                reasoning.append(f"Step {step + 1}: {name}({tool_input})")

                if name in tool_map:
                    try:
                        result = tool_map[name].invoke(tool_input)
                    except Exception as e:
                        if args and len(args) == 1:
                            try:
                                result = tool_map[name].invoke(next(iter(args.values())))
                            except Exception as inner_exc:
                                result = f"Tool error: {inner_exc}"
                        else:
                            result = f"Tool error: {e}"
                else:
                    result = f"Unknown tool: {name}"

                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

        return {
            "output": "Agent reached max steps.",
            "reasoning": reasoning,
        }

    return run_agent


# ── Interactive CLI ───────────────────────────────────────────────────────────

def agent_chat():
    """Interactive agent chat loop."""
    print("=" * 50)
    print("TIMETABLE AI AGENT (Tool-enabled)")
    print("Uses real timetable data via function calls")
    print("Type questions. Type 'quit' to exit.")
    print("=" * 50)

    agent = create_timetable_agent()
    if agent is None:
        print("\nAgent unavailable (missing GROQ_API_KEY or LangChain packages).")
        print("Falling back to direct tool calls for demo:\n")
        tools = _make_tools()
        if tools:
            print(f"Demo – Section A timetable:\n{_read_section_csv('A')}\n")
        return

    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() == "quit":
            print("Exiting.")
            break
        try:
            result = agent({"input": q})
            print(f"\nAgent: {result['output']}")
            if result.get("reasoning"):
                print(f"  Steps: {' → '.join(result['reasoning'])}")
        except Exception as e:
            print(f"\nAgent error: {e}")


if __name__ == "__main__":
    agent_chat()
