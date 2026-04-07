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
    - commit_substitute       : create a substitute preview for UI confirmation
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
import json
import re
from datetime import datetime, timedelta, timezone

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


def _normalize_period_input(period_raw, *, max_period: int = 6) -> str | None:
    period_clean = str(period_raw).strip().upper().lstrip("P")
    try:
        period_int = int(period_clean)
    except ValueError:
        return None
    if period_int < 1 or period_int > max_period:
        return None
    return f"P{period_int}"


def _load_rooms_inventory(data_dir: Path | None = None) -> pd.DataFrame:
    base_dir = data_dir if data_dir is not None else (_AGENT_ROOT / "data")
    return pd.read_csv(base_dir / "rooms.csv")


def _safe_json_dumps(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return str(value)


def _summarize_text(value, max_len: int = 100) -> str:
    text = value if isinstance(value, str) else _safe_json_dumps(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def _parse_absent_periods_result(result: str) -> list[str]:
    periods = re.findall(r"^\s*(P[1-6])\s*:", str(result), flags=re.MULTILINE)
    return list(dict.fromkeys(periods))


MAX_TOOL_RESULT_CHARS = 300


def _truncate_tool_history(text: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars - 3] + "..."


def _compact_substitute_candidates_for_history(result: str) -> str:
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    kept: list[str] = []
    for line in lines:
        if line.startswith("[SIMULATION]"):
            kept.append(line)
        elif line.startswith("#1"):
            kept.append(line)
            break
    if not kept:
        kept = lines[:2]
    return "\n".join(kept)


def _compact_find_substitute_for_history(result: str) -> str:
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    kept = [
        line for line in lines
        if line.startswith("Absent:") or "->" in line or "→" in line
    ]
    return "\n".join(kept or lines[:4])


def _compact_tool_result_for_history(tool_name: str, result) -> str:
    text = str(result)
    if tool_name == "get_substitute_candidates":
        text = _compact_substitute_candidates_for_history(text)
    elif tool_name == "find_substitute":
        text = _compact_find_substitute_for_history(text)
    elif tool_name == "get_faculty_schedule":
        text = text[:MAX_TOOL_RESULT_CHARS]
    return _truncate_tool_history(text)


def _periods_from_commit_args(args) -> list[str]:
    if not isinstance(args, dict):
        return []
    try:
        p_start = int(args.get("period_start", args.get("p_start", 0)))
        p_end = int(args.get("period_end", args.get("p_end", p_start)))
    except (TypeError, ValueError):
        return []
    return [f"P{p}" for p in range(p_start, p_end + 1) if 1 <= p <= 6]


def _extract_sections_from_faculty_cell(cell: object) -> list[str]:
    text = str(cell).strip()
    if not text or text in {"----", "", "nan"}:
        return []

    groups = re.findall(r"\(([^()]+)\)", text)
    for token in reversed(groups):
        sections = [part.strip().upper() for part in token.split("+") if part.strip()]
        if sections and all(section in SECTIONS for section in sections):
            return sections
    return []


def _derive_target_section_for_absence(
    faculty_id: str,
    day: str,
    period_start: int,
    period_end: int,
    *,
    requested_section: str = "",
    output_dir=None,
) -> tuple[str | None, str | None]:
    fid = faculty_id.strip().upper()
    day_norm = _normalize_day_input(day) or str(day).strip()
    if output_dir is not None:
        path = Path(output_dir) / f"faculty_{fid}_timetable.csv"
    else:
        path = resolve_output_path(f"faculty_{fid}_timetable.csv")
    if not path.exists():
        return None, f"No timetable found for {fid}."

    df = pd.read_csv(path)
    row = df[df["Day"].astype(str).str.strip().str.lower() == day_norm.lower()]
    if row.empty:
        return None, f"No schedule found for {fid} on {day_norm}."
    row = row.iloc[0]

    per_period_sections: dict[str, list[str]] = {}
    missing_periods: list[str] = []
    multi_section_periods: list[str] = []

    for p in range(period_start, period_end + 1):
        col = f"P{p}"
        sections = _extract_sections_from_faculty_cell(row.get(col, "----"))
        if not sections:
            missing_periods.append(col)
            continue
        per_period_sections[col] = sections
        if len(sections) > 1:
            multi_section_periods.append(f"{col} ({'+'.join(sections)})")

    if missing_periods:
        return None, (
            f"Cannot create preview: {fid} is not scheduled on {day_norm} "
            f"{', '.join(missing_periods)}."
        )

    if multi_section_periods:
        return None, (
            "Cannot create preview: combined-section slots are not supported by "
            f"commit_substitute yet ({'; '.join(multi_section_periods)})."
        )

    resolved_sections = sorted({sections[0] for sections in per_period_sections.values()})
    if len(resolved_sections) != 1:
        return None, (
            "Cannot create preview: the requested range spans multiple sections "
            f"for {fid} on {day_norm}: {', '.join(resolved_sections)}."
        )

    resolved_section = resolved_sections[0]
    requested = str(requested_section or "").strip().upper()
    if requested and requested != resolved_section:
        return resolved_section, (
            f"Resolved section {resolved_section} from {fid}'s timetable; "
            f"ignored requested section {requested}."
        )

    return resolved_section, None


def _parse_timestamp(value) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


# ── LangChain Tools ──────────────────────────────────────────────────────────

def _make_tools(agent_output_dir=None, sem_id: str = None,
                session_started_at: str | None = None):
    """
    Build all LangChain tools.  If agent_output_dir is given all output reads
    (section CSVs, faculty CSVs, room_assignment.csv, summary_report.txt) use
    that directory instead of the legacy resolve_output_path().
    sem_id is forwarded to write-path functions (commit_schedule_change, find_substitute)
    so they operate on the correct semester's output tree.
    """
    _out = Path(agent_output_dir) if agent_output_dir else None

    # Semester-aware data directory (faculty.csv, rooms.csv, etc.)
    if sem_id is not None:
        from config import get_sem_paths as _gsp_inner
        _dat = _gsp_inner(sem_id).data_dir
    else:
        _dat = _AGENT_ROOT / "data"

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
        faculty_id: Faculty ID to look up, e.g. 'F01', 'F03'.
                    This is ALWAYS a faculty ID like F01, NEVER a section letter like A or B.
        Returns: full weekly schedule as formatted text.
        """
        return _read_faculty_csv(faculty_id, output_dir=_out)

    @tool
    def find_free_slots(faculty_id: str, day: str) -> str:
        """
        Find free periods for a faculty on a given day.
        faculty_id: Faculty ID to look up, e.g. 'F01', 'F03'.
        day: Day of the week, e.g. 'Monday', 'Tuesday'.
        Returns: list of free periods.
        """
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
            text = report_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "Summary report not found. Run run_all.py first."

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        q = query.lower()

        def _find_line(prefix: str) -> str | None:
            return next((line for line in lines if line.startswith(prefix)), None)

        theory_line = _find_line("Total theory/elective slots placed")
        fixed_line = _find_line("Total fixed elective slots placed")
        sections_line = _find_line("Total sections scheduled")
        lab_line = _find_line("Total lab slots placed")
        same_day_line = _find_line("same_subject_same_day")
        back_to_back_line = _find_line("back_to_back_same_subject")
        quality_line = next(
            (line for line in lines if "quality" in line.lower() and "score" in line.lower()),
            None,
        )
        overload_lines = [line for line in lines if "| OVERLOAD" in line]
        faculty_load_lines = [line for line in lines if line.startswith("F") and "|" in line]

        if "overload" in q:
            if overload_lines:
                return "Overloaded faculty:\n" + "\n".join(overload_lines)
            return "No overloaded faculty found."

        if "quality" in q:
            return quality_line or "Quality score not available in summary report."

        if "violation" in q or "constraint" in q:
            selected = [line for line in [same_day_line, back_to_back_line] if line]
            return "\n".join(selected) if selected else "No violation lines found."

        if "faculty load" in q or "load table" in q:
            if faculty_load_lines:
                return "\n".join(faculty_load_lines)
            return "Faculty load table not found."

        if "lab" in q:
            return lab_line or "Lab slot summary not found."

        if "theory" in q or "slot" in q or "elective" in q:
            selected = [line for line in [theory_line, fixed_line, lab_line] if line]
            return "\n".join(selected) if selected else "Slot statistics not found."

        headline_lines = [
            line for line in [
                sections_line,
                theory_line,
                fixed_line,
                lab_line,
                (
                    f"Soft constraint violations: "
                    f"{same_day_line or 'same_subject_same_day: n/a'}; "
                    f"{back_to_back_line or 'back_to_back_same_subject: n/a'}"
                ),
            ]
            if line
        ]
        return "\n".join(headline_lines[:5])

    @tool
    def find_substitute(faculty_id: str, day: str) -> str:
        """
        Find a substitute for an absent faculty member.
        faculty_id: Faculty ID to look up, e.g. 'F01', 'F03'.
        day: Day of the week, e.g. 'Monday', 'Tuesday'.
        Returns: substitute suggestions.
        For lab periods (P5-P6) the full 2-period block is always treated
        atomically — only candidates free for both P5 and P6 are shown.
        """
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
    def commit_substitute(
        section: str,
        day: str,
        period_start: int,
        period_end: int,
        absent_faculty: str,
        substitute_faculty: str,
        reason: str,
    ) -> str:
        """
        Create a substitute assignment preview.
        section: Section letter, e.g. 'A', 'B', 'C'.
        day: Day of the week, e.g. 'Monday', 'Tuesday'.
        period_start: Starting period number (integer), e.g. 1, 5.
        period_end: Ending period number (integer), e.g. 1, 6.
        absent_faculty: Faculty ID who is absent, e.g. 'F03'.
        substitute_faculty: Faculty ID who will substitute, e.g. 'F07'.
        reason: Short reason text, e.g. 'Lab coverage for DDCO'.
        On the FIRST call this writes a preview to a temp folder and returns
        a diff summary - no canonical timetable files are changed.
        During a full absence flow, the agent loop finalizes each preview into
        outputs/<sem_id>/substitutes/<day>/substitute_notice.txt before moving
        to the next uncovered period.
        If a preview already exists, returns immediately with a message to
        confirm it in the UI or discard it before creating a new one.
        Validates substitute availability before writing.
        """
        from src.phase5.sync_manager import get_active_preview, preview_schedule_change

        # Guard: if a preview already exists, don't re-write
        existing = get_active_preview(sem_id)
        if existing:
            return (
                f"A preview already exists (op_id={existing['op_id']}). "
                "Ask the user to confirm it in the UI, "
                "or discard_pending_preview to cancel before creating a new one."
            )

        p_start  = int(period_start)
        p_end    = int(period_end)
        absent   = absent_faculty.strip().upper()
        substitute = substitute_faculty.strip().upper()
        resolved_section, section_note = _derive_target_section_for_absence(
            absent,
            day,
            p_start,
            p_end,
            requested_section=section,
            output_dir=_out,
        )
        if resolved_section is None:
            return section_note or (
                f"Cannot create preview for {absent} on {day} P{p_start}-P{p_end}."
            )

        # Validate substitute availability
        if not _is_faculty_free(substitute, day, p_start, p_end, output_dir=_out):
            return (
                f"Cannot commit: {substitute} is not free on {day} "
                f"P{p_start}-P{p_end}."
            )

        try:
            result = preview_schedule_change({
                "section":          resolved_section,
                "day":              day,
                "period_start":     p_start,
                "period_end":       p_end,
                "original_faculty": absent,
                "new_faculty":      substitute,
                "change_type":      "substitute",
                "reason":           reason,
            }, sem_id=sem_id)
            prefix = f"{section_note}\n" if section_note else ""
            return (
                f"{prefix}{result['diff_summary']}\n\n"
                f"{result['message']}\n"
                "Present this preview to the user and ask them to confirm "
                "in the UI. Do not auto-commit."
            )
        except Exception as exc:
            return f"Preview generation failed: {exc}"

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
        ops = list_operations(limit, sem_id=sem_id)
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
        return rollback_operation(op_id, sem_id=sem_id)

    @tool
    def generate_session_summary(input_str: str) -> str:
        """
        Generate a human-readable summary report of all agent actions
        taken in the current session. Writes to outputs/agent_ops/
        Input: optional session label (e.g. "Monday absence session")
        """
        label = input_str.strip() or "Agent Session"
        from src.phase5.agent_ops import _ops_dir as _get_ops_dir

        cutoff = _parse_timestamp(session_started_at)
        if cutoff is None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=2)

        ops = []
        ops_dir = _get_ops_dir(sem_id)
        for op_path in sorted(ops_dir.glob("*.json")):
            try:
                op = json.loads(op_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            op_ts = _parse_timestamp(op.get("timestamp_utc")) or _parse_timestamp(op.get("timestamp_local"))
            if op_ts is None or op_ts < cutoff:
                continue
            ops.append(op)

        ops.sort(
            key=lambda op: _parse_timestamp(op.get("timestamp_utc"))
            or _parse_timestamp(op.get("timestamp_local"))
            or cutoff
        )
        if not ops:
            return "No operations found for the current session."

        lines = [
            f"# {label} — Agent Operations Summary",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Session window start: {cutoff.isoformat()}",
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
        ops_dir.mkdir(parents=True, exist_ok=True)
        out_path = ops_dir / f"summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.txt"
        out_path.write_text(summary_text, encoding="utf-8")
        return summary_text

    @tool
    def apply_pending_preview(input_str: str) -> str:
        """
        Finalise a pending substitute preview by writing substitute overlay files.
        Call this ONLY after the user has explicitly confirmed the proposed change.
        Input: any string (ignored) - the active preview is detected automatically.
        Writes outputs/<sem_id>/substitutes/<day>/ and leaves canonical
        timetable CSVs unchanged.
        """
        from src.phase5.sync_manager import get_active_preview, commit_from_preview
        meta = get_active_preview(sem_id)
        if not meta:
            return "No pending preview found. Nothing to apply."
        try:
            result = commit_from_preview(meta["op_id"], sem_id=sem_id)
            return result["message"]
        except Exception as exc:
            return f"Apply failed (rolled back): {exc}"

    @tool
    def discard_pending_preview(input_str: str) -> str:
        """
        Discard a pending substitute preview without making any live changes.
        Input: any string (ignored) — the active preview is detected automatically.
        """
        from src.phase5.sync_manager import get_active_preview, discard_preview
        meta = get_active_preview(sem_id)
        if not meta:
            return "No pending preview to discard."
        discard_preview(meta["op_id"], sem_id=sem_id)
        return (
            f"Preview discarded (op_id={meta['op_id']}). "
            "No live files were changed."
        )

    @tool
    def get_absent_periods(faculty_id: str, day: str) -> str:
        """
        Get the actual periods a faculty member teaches on a given day.
        faculty_id: Faculty ID to look up, e.g. 'F01', 'F03'.
        day: Day of the week, e.g. 'Monday', 'Tuesday'.
        Returns the specific periods they are scheduled to teach.
        Use this BEFORE commit_substitute to know which periods to cover.
        """
        fid = faculty_id.strip().upper()
        path = _rop(f"faculty_{fid}_timetable.csv")
        if not path or not Path(path).exists():
            return f"No timetable found for {fid}"

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

        # All classrooms from rooms.csv — Bug 3 fix: use per-semester data dir
        rooms_csv_path = _dat / "rooms.csv"
        try:
            rooms_df = _pd.read_csv(rooms_csv_path)
        except Exception as exc:
            return f"Error reading rooms.csv: {exc}"

        candidate_rooms = rooms_df.copy()
        if capacity_needed > 0:
            candidate_rooms = candidate_rooms[candidate_rooms["capacity"] >= capacity_needed]

        free = candidate_rooms[~candidate_rooms["room_name"].isin(occupied)]

        if free.empty:
            return (
                f"No free rooms on {day} P{period}"
                + (f" with capacity >= {capacity_needed}" if capacity_needed else "")
                + "."
            )

        lines = [f"Free rooms on {day} P{period}:"]
        for _, row in free.iterrows():
            lines.append(
                f"  {row['room_name']} (type: {row['room_type']}, "
                f"capacity: {row['capacity']}, floor: {row['floor']})"
            )
        return "\n".join(lines)

    # ── TOOL 7 — get_faculty_workload ────────────────────────────────────────
    @tool
    def get_faculty_workload(faculty_id: str) -> str:
        """
        Get total workload for a faculty member this week.
        faculty_id: Faculty ID to look up, e.g. 'F01', 'F03'. Can also be a partial name like 'Sharma'.
        Returns: total hours, day-by-day breakdown, courses taught, and
                 whether they are over their weekly cap.
        """
        query = faculty_id.strip().upper()
        import pandas as _pd
        # Bug 1 fix: use closure _rop (semester-aware) and _dat — do NOT
        # shadow-import resolve_output_path as _rop here.
        from config import MAX_HOURS as _MAX_HOURS

        # Resolve faculty_id from id or partial name — use sem-aware data dir
        try:
            fac_df = _pd.read_csv(_dat / "faculty.csv")
        except Exception as exc:
            return f"Cannot load faculty.csv: {exc}"

        match = fac_df[fac_df["faculty_id"].str.upper() == query]
        if match.empty:
            match = fac_df[fac_df["name"].str.upper().str.contains(query, na=False)]
        if match.empty:
            return f"No faculty found matching '{faculty_id}'."

        row = match.iloc[0]
        fid        = str(row["faculty_id"]).strip()
        name       = str(row["name"]).strip()
        designation = str(row["designation"]).strip()
        max_h      = _MAX_HOURS.get(designation, 16)

        # Use closure _rop — resolves to active semester output dir
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
    def get_free_faculty(day: str, period: str) -> str:
        """
        Find all faculty with no class in a given day + period slot.
        day: Day of the week, e.g. 'Monday', 'Tuesday'.
        period: Period number, e.g. '3' or 'P3'.
        Returns: list of free faculty with their designation.
        """
        day_raw = day
        period_raw = period
        period_str = period_raw.upper().lstrip("P")
        try:
            period_int = int(period_str)
        except ValueError:
            return f"Invalid period '{period_raw}'."
        col = f"P{period_int}"

        import pandas as _pd
        # Bug 2 fix: use per-semester data dir via closure _dat
        try:
            fac_df = _pd.read_csv(_dat / "faculty.csv")
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
    def get_room_availability(day: str, period: str, room_name: str = "") -> str:
        """
        Find whether a specific room is free or occupied in a given slot,
        or return a full slot-level availability report when room_name is blank.
        Inputs:
          day: accepts Tuesday, Tue, TUE, tue, tuesday
          period: accepts P1-P6, p1-p6, 1-6, or "1"-"6"
          room_name: optional room identifier, e.g. "Room_G12"

        Reads outputs/room_assignment.csv and semester-aware rooms.csv.
        Returns room status or a free/occupied slot report.
        """
        import pandas as _pd

        normalized_day = _normalize_day_input(day)
        if normalized_day is None:
            return (
                f"Unrecognized day '{day}'. "
                "Use Monday, Tuesday, Wednesday, Thursday, or Friday "
                "(full name or common abbreviation)."
            )

        period_label = _normalize_period_input(period, max_period=6)
        if period_label is None:
            return (
                f"Unrecognized period '{period}'. "
                "Use P1, P2, P3, P4, P5, P6 or just 1-6."
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
            rooms_df = _load_rooms_inventory(_dat)
        except Exception as exc:
            return f"Error reading semester rooms.csv: {exc}"

        rooms_df = rooms_df.copy()
        rooms_df["room_name"] = rooms_df["room_name"].astype(str).str.strip()
        rooms_df["room_type"] = rooms_df["room_type"].astype(str).str.strip()

        occupied_rows = slot_df[
            slot_df["Room"].astype(str).str.strip().ne("ROOM_UNASSIGNED")
        ].copy()
        occupied_rows["Room"] = occupied_rows["Room"].astype(str).str.strip()
        occupied_room_names = set(occupied_rows["Room"].tolist())

        target_room = room_name.strip()
        if target_room:
            room_match = rooms_df[
                rooms_df["room_name"].astype(str).str.strip().str.lower() == target_room.lower()
            ]
            if room_match.empty:
                return f"Room '{room_name}' not found in the semester inventory."

            canonical_room = str(room_match.iloc[0]["room_name"]).strip()
            room_rows = occupied_rows[
                occupied_rows["Room"].astype(str).str.strip().str.lower() == canonical_room.lower()
            ]
            if room_rows.empty:
                return f"{canonical_room} is free on {normalized_day} {period_label}."

            users = ", ".join(
                f"Section {row['Section']} ({row['Course']}, {row['Faculty']})"
                for _, row in room_rows.sort_values(["Section", "Course"]).iterrows()
            )
            return f"{canonical_room} is occupied on {normalized_day} {period_label} by {users}."

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
        Scan room_assignment.csv for scheduling conflicts.
        Input: optional free-text filter such as "check" or "Section B".
        Detects:
          • Same faculty assigned to two sections in the same slot
          • Same room assigned to two sections in the same slot
        Returns a structured conflict report.
        """
        import pandas as _pd
        ra_path = _rop("room_assignment.csv")
        if not ra_path.exists():
            return "room_assignment.csv not found. Run run_all.py first."

        try:
            ra_df = _pd.read_csv(ra_path)
        except Exception as exc:
            return f"Error reading room_assignment.csv: {exc}"

        section_match = re.search(r"\bsection\s+([A-L])\b", input_str, re.IGNORECASE)
        section_filter = section_match.group(1).upper() if section_match else None

        faculty_slot: dict[tuple[str, str, str], set[str]] = {}
        room_slot: dict[tuple[str, str, str], set[str]] = {}

        for _, row in ra_df.iterrows():
            day = str(row.get("Day", "")).strip()
            period = str(row.get("Period", "")).strip().upper()
            section = str(row.get("Section", "")).strip().upper()
            faculty = str(row.get("Faculty", "")).strip().upper()
            room = str(row.get("Room", "")).strip()

            if day and period and faculty and faculty not in {"", "NAN", "NONE"}:
                faculty_slot.setdefault((day, period, faculty), set()).add(section)
            if day and period and room and room not in {"ROOM_UNASSIGNED", "", "nan"}:
                room_slot.setdefault((day, period, room), set()).add(section)

        conflicts = []
        for (day, period, faculty_id), sections in sorted(faculty_slot.items()):
            if len(sections) > 1:
                if section_filter and section_filter not in sections:
                    continue
                section_text = ", ".join(sorted(sections))
                conflicts.append(
                    f"Conflict found: {faculty_id} double-booked on {day} {period} "
                    f"(sections {section_text})"
                )

        for (day, period, room_name), sections in sorted(room_slot.items()):
            if len(sections) > 1:
                if section_filter and section_filter not in sections:
                    continue
                section_text = ", ".join(sorted(sections))
                conflicts.append(
                    f"Conflict found: {room_name} double-booked on {day} {period} "
                    f"(sections {section_text})"
                )

        if not conflicts:
            if section_filter:
                return f"No conflicts found for Section {section_filter}."
            return "No conflicts found."
        return "\n".join(conflicts)

    # ── TOOL 11 — get_system_status ──────────────────────────────────────────
    @tool
    def get_system_status(input_str: str) -> str:
        """
        Return system-health information for the current semester.
        Input: optional free text such as "status", "missing files", "env".
        Returns: file presence, RAG availability, environment, and package health.
        """
        from utils.health_check import check_system_health

        report = check_system_health(sem_id=sem_id)
        summary = report["summary"]
        q = input_str.lower()

        if "missing" in q or "failed" in q or "error" in q:
            failed = summary.get("failed_items", [])
            if not failed:
                return "System status: all required checks are passing."
            return "Failed checks:\n" + "\n".join(f"  - {item}" for item in failed)

        lines = [
            "=== System Status ===",
            f"Overall OK: {'Yes' if report['overall_ok'] else 'No'}",
            f"Checks passed: {summary['passed']} / {summary['total']}",
            f"Checks failed: {summary['failed']}",
            f"GROQ_API_KEY: {report['environment']['GROQ_API_KEY']['message']}",
            f"Output files: {sum(1 for r in report['output_files'].values() if r['ok'])} / {len(report['output_files'])} present",
            f"RAG index files: {sum(1 for r in report['rag_index'].values() if r['ok'])} / {len(report['rag_index'])} present",
        ]
        if summary["failed_items"]:
            lines.append("Failed items:")
            for item in summary["failed_items"][:10]:
                lines.append(f"  - {item}")
        return "\n".join(lines)

    # ── TOOL 12 — get_substitute_candidates ─────────────────────────────────
    @tool
    def get_substitute_candidates(faculty_id: str, day: str, period: str) -> str:
        """
        Preview the top substitute candidates for a faculty on a given day/period.
        DOES NOT commit any changes — simulation only.
        faculty_id: Faculty ID to look up, e.g. 'F01', 'F03'.
        day: Day of the week, e.g. 'Monday', 'Tuesday'.
        period: Period number, e.g. '3' or 'P3'.
        Returns: top 3 candidates with match reasons and load info.
        Clearly states this is a preview — use commit_substitute to apply.
        """
        fid, day_raw, period_raw = faculty_id, day, period
        period_str = period_raw.upper().lstrip("P")
        try:
            period_int = int(period_str)
        except ValueError:
            return f"Invalid period '{period_raw}'."
        period_col = f"P{period_int}"

        try:
            from src.phase5.substitute import (
                _rank_candidates, _collect_absent_slots, _sections_for_faculty,
                faculty_lookup, normalize_day,
                get_lab_block_periods, parse_timetable_cell,
            )
        except ImportError as exc:
            return f"Cannot import substitute module: {exc}"

        # Resolve semester-specific dirs from the closure sem_id
        from config import get_sem_paths as _gsp
        if sem_id:
            _sp    = _gsp(sem_id)
            _out_d = _sp.output_dir
            _dat_d = _sp.data_dir
        else:
            _out_d = _dat_d = None

        try:
            day_norm = normalize_day(day_raw)
            lookup   = faculty_lookup(_dat_d)
            fid_upper = fid.strip().upper()
            if fid_upper not in lookup:
                return f"Unknown faculty_id: {fid_upper}"

            absent_info     = lookup[fid_upper]
            absent_sections = _sections_for_faculty(fid_upper, _dat_d)

            # Trust the period already confirmed by get_absent_periods.
            # Read the live faculty CSV directly first so this stays consistent
            # with get_absent_periods rather than relying only on cached helpers.
            slot_info = None
            fac_path = _rop(f"faculty_{fid_upper}_timetable.csv")
            if fac_path and Path(fac_path).exists():
                df = pd.read_csv(fac_path)
                row = df[
                    df["Day"].astype(str).str.strip().str.lower() == day_norm.lower()
                ]
                if not row.empty:
                    cell = str(row.iloc[0].get(period_col, "----")).strip()
                    parsed = parse_timetable_cell(cell)
                    if parsed is not None:
                        slot_info = parsed

            if slot_info is None:
                absent_slots = _collect_absent_slots(fid_upper, day_norm, _out_d)
                slot_info = next(
                    (
                        slot for slot in absent_slots
                        if str(slot.get("period", "")).strip().upper() == period_col
                    ),
                    None,
                )

            if slot_info is None:
                return (
                    f"No class data available for {fid_upper} on {day_norm} {period_col}. "
                    "Use find_substitute(faculty_id, day) as fallback."
                )

            # Expand to lab block if needed
            block  = get_lab_block_periods(period_col)
            ranked, at_cap = _rank_candidates(
                absent_faculty_id=fid_upper,
                day=day_norm,
                period=period_col,
                slot_info=slot_info,
                absent_sections=absent_sections,
                out_dir=_out_d,
                data_dir=_dat_d,
            )

            if len(block) > 1:
                # Lab block: filter to candidates free for all periods
                filtered = []
                for cand in ranked:
                    all_free = all(
                        str(get_faculty_day_row(cand["faculty_id"], day_norm, _out_d)
                            .get(bp, "----")).strip() in ("----", "", "nan")
                        for bp in block
                    )
                    if all_free:
                        filtered.append(cand)
                ranked = filtered

            top3 = ranked[:3]
            if not top3:
                return (
                    f"No substitute candidates found for {fid_upper} on "
                    f"{day_norm} {period_col} ({slot_info['course']}).\n"
                    "All faculty are either busy or at load cap."
                )

            lines = [
                f"[SIMULATION] Top substitute candidates for {absent_info['name']} "
                f"({fid_upper}) on {day_norm} {period_col} — {slot_info['course']}",
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
                "\u26a0 This is a SIMULATION only \u2014 no changes have been made.\n"
                "  Proceed with commit_substitute using the top candidate."
            )
            return "\n".join(lines)

        except Exception as exc:
            import traceback
            return f"Simulation error: {exc}\n{traceback.format_exc()}"

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
        discard_pending_preview,
        get_absent_periods,
        find_free_rooms,
        get_faculty_workload,
        get_free_faculty,
        get_room_availability,
        detect_schedule_conflicts,
        get_system_status,
        get_substitute_candidates,
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
  • Section timetables (A-L)  - get_section_timetable
  • Faculty schedules          - get_faculty_schedule for general schedule questions;
                                 get_absent_periods for absence handling
  • Schedule quality / loads   — get_summary_stats, get_faculty_workload
  • System health / files      — get_system_status
  • Free slots per faculty     — find_free_slots
  • Free faculty in a slot     — get_free_faculty
  • Free rooms in a slot       — find_free_rooms
  • Specific room status       — get_room_availability
  • Substitute candidates      — get_substitute_candidates
  • Full-day substitute plan   — find_substitute
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
  Step 1: Call get_absent_periods(faculty_id, day) — this returns ALL
          periods the faculty teaches that day.
          Never call get_faculty_schedule before get_absent_periods —
          get_absent_periods is always the first step.
  Step 2: For EACH period returned, call get_substitute_candidates to get
          the best available substitute.
  Step 3: Call commit_substitute for EACH period separately, one by one,
          using the top candidate for that period.
  Step 4: Only stop when every period from Step 1 has been covered.
          Never stop after the first period. Never skip a period.
  Step 5: After every period is covered, call generate_session_summary()
          to log the session.

RULES:
  • NEVER guess period numbers — always call get_absent_periods first
  • After calling get_absent_periods, store the exact list of occupied periods
    returned. Call get_substitute_candidates ONLY for those periods. Never call
    it for any other period.
  • Never call get_faculty_schedule during absence handling. It returns the full
    week and bloats context; get_absent_periods gives the exact day data needed.
  • NEVER guess section IDs for commit_substitute. The target section must come
    from the absent faculty's actual timetable slot, not from LLM inference.
  • After get_substitute_candidates, proceed to commit_substitute only when you
    have enough information to create a safe substitute notice entry
  • If get_substitute_candidates returns "no class" or empty for a period that
    get_absent_periods already confirmed, do NOT retry the same call. Call
    find_substitute(faculty_id, day) directly as fallback and use those results.
  • Use find_substitute only when the user explicitly wants a full-day substitute plan
    across all absent periods rather than a per-slot candidate ranking
  • Use get_summary_stats for schedule quality, overload, violation, and slot-count questions
  • Use get_system_status for environment, file existence, and system-health checks
  • Use find_free_rooms when the user asks "which rooms are free at slot X"
  • Use get_room_availability when the user asks whether a specific room is free or occupied
  • Use detect_schedule_conflicts first for any conflict-checking question
  • Do NOT call apply_pending_preview yourself under any circumstance.
    The agent runtime finalizes each substitute notice before moving to the next
    uncovered period.
  • Do not call discard_pending_preview unless the user explicitly asks to cancel the preview
  • Lab periods P5–P6 must always be substituted as an atomic block
  • Always state when you are simulating vs writing a preview vs committing
  • If data is unavailable, say so clearly and suggest running run_all.py
  • Never try to infer room assignments from section timetables — rooms are only
    known after room allocation is run.
"""


def create_timetable_agent(sem_id: str = None,
                           session_started_at: str | None = None):
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

    tools = _make_tools(
        agent_output_dir=agent_output_dir,
        sem_id=sem_id,
        session_started_at=session_started_at,
    )
    if tools is None:
        return None

    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def run_agent(input_dict: dict) -> dict:
        query = input_dict.get("input", "")
        step_callback = input_dict.get("step_callback")
        messages = [
            SystemMessage(content=_get_agent_system_prompt(sem_id=sem_id)),
            HumanMessage(content=query),
        ]
        reasoning = []
        steps: list[dict] = []
        absence_state = {
            "faculty_id": None,
            "day": None,
            "periods": [],
            "covered": set(),
            "commit_messages": [],
        }

        def _remaining_absence_periods() -> list[str]:
            return [
                period for period in absence_state["periods"]
                if period not in absence_state["covered"]
            ]

        def _record_runtime_step(tool_name: str, tool_input, tool_result, status: str = "ok") -> None:
            step_record = {
                "step_num": len(steps) + 1,
                "tool_name": tool_name,
                "tool_input": _safe_json_dumps(tool_input),
                "tool_result": str(tool_result),
                "status": status,
            }
            steps.append(step_record)
            reasoning.append(
                f"Step {step_record['step_num']}: {tool_name}("
                f"{_summarize_text(step_record['tool_input'])})"
            )
            if callable(step_callback):
                try:
                    step_callback(steps)
                except Exception:
                    pass

        def _append_tool_message(tool_call_id: str, tool_name: str, tool_result) -> None:
            messages.append(ToolMessage(
                content=_compact_tool_result_for_history(tool_name, tool_result),
                tool_call_id=tool_call_id,
            ))

        def _is_absence_query() -> bool:
            q = query.lower()
            explicit_period = bool(re.search(r"\bp\s*[1-6]\b|\bperiod\s*[1-6]\b", q))
            return (
                "absent" in q
                or "absence" in q
                or "cover every period" in q
                or "cover all" in q
                or ("find substitute" in q and not explicit_period)
            )

        for step in range(16):
            try:
                response = safe_llm_call(llm_with_tools, messages)
            except Exception as exc:
                return {
                    "output": f"Agent error: {exc}",
                    "reasoning": reasoning,
                    "steps": steps,
                }
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", []) or []
            if not tool_calls:
                remaining = _remaining_absence_periods()
                if remaining:
                    messages.append(HumanMessage(content=(
                        "You stopped before covering every absence period. "
                        f"Remaining periods for {absence_state['faculty_id']} on "
                        f"{absence_state['day']}: {', '.join(remaining)}. "
                        "Continue by calling get_substitute_candidates for the next "
                        "remaining period, then commit_substitute for that period."
                    )))
                    continue
                # Agent is done - no more tool calls
                return {
                    "output": getattr(response, "content", str(response)),
                    "reasoning": reasoning,
                    "steps": steps,
                }

            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("args", {})
                tool_input = args if args else ""
                step_index = len(steps) + 1
                call_signature = (name, _safe_json_dumps(tool_input))

                if len(steps) >= 2:
                    recent_signatures = [
                        (prev["tool_name"], prev["tool_input"])
                        for prev in steps[-2:]
                    ]
                    if recent_signatures[0] == recent_signatures[1] == call_signature:
                        return {
                            "output": (
                                "Agent stopped after repeating the same tool call "
                                "three times. Review the collected results and use "
                                "find_substitute(faculty_id, day) as fallback if needed."
                            ),
                            "reasoning": reasoning,
                            "steps": steps,
                        }

                if name == "get_faculty_schedule" and _is_absence_query():
                    result = (
                        "Skipped get_faculty_schedule during absence handling. "
                        "Use get_absent_periods(faculty_id, day) first; it returns "
                        "the exact occupied periods for the absence day."
                    )
                    _append_tool_message(tc["id"], name, result)
                    continue

                if (
                    name == "get_substitute_candidates"
                    and _is_absence_query()
                    and not absence_state["periods"]
                ):
                    result = (
                        "Skipped get_substitute_candidates before get_absent_periods. "
                        "In an absence flow, first call get_absent_periods(faculty_id, day) "
                        "and then request candidates only for the returned occupied periods."
                    )
                    _append_tool_message(tc["id"], name, result)
                    continue

                if name == "get_substitute_candidates" and absence_state["periods"]:
                    requested_period = None
                    if isinstance(tool_input, dict):
                        requested_period = _normalize_period_input(tool_input.get("period", ""))
                    remaining = _remaining_absence_periods()
                    if requested_period not in absence_state["periods"]:
                        result = (
                            "Skipped get_substitute_candidates for a non-occupied "
                            f"period: {requested_period or tool_input}. Occupied "
                            f"periods from get_absent_periods: "
                            f"{', '.join(absence_state['periods'])}. "
                            f"Remaining periods: {', '.join(remaining)}."
                        )
                        _append_tool_message(tc["id"], name, result)
                        continue
                    if requested_period in absence_state["covered"]:
                        result = (
                            "Skipped get_substitute_candidates for already-covered "
                            f"period {requested_period}. Remaining periods: "
                            f"{', '.join(remaining)}."
                        )
                        _append_tool_message(tc["id"], name, result)
                        continue

                commit_periods_before_call = _periods_from_commit_args(tool_input)
                if (
                    name == "commit_substitute"
                    and absence_state["periods"]
                    and commit_periods_before_call
                    and all(
                        period in absence_state["covered"]
                        for period in commit_periods_before_call
                    )
                ):
                    remaining = _remaining_absence_periods()
                    result = (
                        f"Skipped duplicate commit_substitute for already-covered "
                        f"periods: {', '.join(commit_periods_before_call)}."
                    )
                    if remaining:
                        result += (
                            f" Remaining periods still uncovered: "
                            f"{', '.join(remaining)}. Continue to the next period."
                        )
                    step_status = "ok"
                elif name in tool_map:
                    try:
                        result = tool_map[name].invoke(tool_input)
                        step_status = "ok"
                    except Exception as e:
                        if args and len(args) == 1:
                            try:
                                result = tool_map[name].invoke(next(iter(args.values())))
                                step_status = "ok"
                            except Exception as inner_exc:
                                result = f"Tool error: {inner_exc}"
                                step_status = "error"
                        else:
                            result = f"Tool error: {e}"
                            step_status = "error"
                else:
                    result = f"Unknown tool: {name}"
                    step_status = "error"

                step_record = {
                    "step_num": step_index,
                    "tool_name": name,
                    "tool_input": _safe_json_dumps(tool_input),
                    "tool_result": str(result),
                    "status": step_status,
                }
                steps.append(step_record)
                reasoning.append(
                    f"Step {step_index}: {name}("
                    f"{_summarize_text(step_record['tool_input'])})"
                )
                if callable(step_callback):
                    try:
                        step_callback(steps)
                    except Exception:
                        pass

                if name == "get_absent_periods" and step_status == "ok":
                    periods = _parse_absent_periods_result(str(result))
                    if periods and isinstance(tool_input, dict):
                        absence_state["faculty_id"] = str(
                            tool_input.get("faculty_id", "")
                        ).strip().upper()
                        absence_state["day"] = str(tool_input.get("day", "")).strip()
                        absence_state["periods"] = periods
                        absence_state["covered"] = set()
                        absence_state["commit_messages"] = []

                if (
                    name == "commit_substitute"
                    and step_status == "ok"
                    and "Preview generated" in str(result)
                ):
                    if not absence_state["periods"]:
                        return {
                            "output": str(result),
                            "reasoning": reasoning,
                            "steps": steps,
                        }

                    commit_periods = _periods_from_commit_args(tool_input)
                    try:
                        from src.phase5.sync_manager import (
                            commit_from_preview,
                            get_active_preview,
                        )
                        meta = get_active_preview(sem_id)
                        if not meta:
                            raise RuntimeError("No active preview found to finalize.")
                        commit_result = commit_from_preview(meta["op_id"], sem_id=sem_id)
                        commit_message = commit_result.get("message", str(commit_result))
                        _record_runtime_step(
                            "commit_from_preview",
                            {
                                "op_id": meta["op_id"],
                                "periods": commit_periods,
                                "sem_id": sem_id,
                            },
                            commit_message,
                            "ok",
                        )
                        absence_state["covered"].update(commit_periods)
                        absence_state["commit_messages"].append(commit_message)
                    except Exception as exc:
                        _record_runtime_step(
                            "commit_from_preview",
                            {"periods": commit_periods, "sem_id": sem_id},
                            f"Commit failed: {exc}",
                            "error",
                        )
                        return {
                            "output": (
                                f"{result}\n\nCommit failed before all periods were "
                                f"covered: {exc}"
                            ),
                            "reasoning": reasoning,
                            "steps": steps,
                        }

                    remaining = _remaining_absence_periods()
                    if not remaining:
                        summary_label = (
                            f"{absence_state['faculty_id']} absence on "
                            f"{absence_state['day']}"
                        )
                        try:
                            summary_result = tool_map["generate_session_summary"].invoke(
                                summary_label
                            )
                            _record_runtime_step(
                                "generate_session_summary",
                                summary_label,
                                summary_result,
                                "ok",
                            )
                        except Exception as exc:
                            summary_result = f"Session summary failed: {exc}"
                            _record_runtime_step(
                                "generate_session_summary",
                                summary_label,
                                summary_result,
                                "error",
                            )

                        covered = ", ".join(absence_state["periods"])
                        commit_messages = "\n".join(absence_state["commit_messages"])
                        return {
                            "output": (
                                f"Covered every period for "
                                f"{absence_state['faculty_id']} on "
                                f"{absence_state['day']}: {covered}.\n\n"
                                f"{commit_messages}\n\n{summary_result}"
                            ),
                            "reasoning": reasoning,
                            "steps": steps,
                        }

                    result = (
                        f"{result}\n\n{commit_message}\n"
                        f"Covered periods: {', '.join(sorted(absence_state['covered']))}. "
                        f"Remaining periods: {', '.join(remaining)}. "
                        "Continue to the next remaining period."
                    )
                    _append_tool_message(tc["id"], name, result)
                    continue

                if (
                    name == "commit_substitute"
                    and step_status == "ok"
                    and absence_state["periods"]
                ):
                    remaining = _remaining_absence_periods()
                    if remaining:
                        result = (
                            f"{result}\n\nRemaining periods still uncovered: "
                            f"{', '.join(remaining)}. Continue to the next period."
                        )

                _append_tool_message(tc["id"], name, result)

        return {
            "output": "Agent reached max steps.",
            "reasoning": reasoning,
            "steps": steps,
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
