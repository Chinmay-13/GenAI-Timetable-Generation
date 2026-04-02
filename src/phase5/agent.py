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

Falls back gracefully if GEMINI_API_KEY is missing.

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

def _read_section_csv(section: str) -> str:
    path = resolve_output_path(f"section_{section.strip().upper()}_timetable.csv")
    try:
        df = pd.read_csv(path)
        return df.to_string(index=False)
    except FileNotFoundError:
        return f"No timetable found for section {section}. Run run_all.py first."


def _read_faculty_csv(faculty_id: str) -> str:
    path = resolve_output_path(f"faculty_{faculty_id.strip().upper()}_timetable.csv")
    try:
        df = pd.read_csv(path)
        return df.to_string(index=False)
    except FileNotFoundError:
        return f"No timetable found for faculty {faculty_id}. Run run_all.py first."


def _is_faculty_free(faculty_id: str, day: str,
                     period_start: int, period_end: int) -> bool:
    """Check if faculty has no bookings for the given period range on day."""
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


# ── LangChain Tools ──────────────────────────────────────────────────────────

def _make_tools():
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
        return _read_section_csv(section)

    @tool
    def get_faculty_schedule(faculty_id: str) -> str:
        """
        Get the full weekly schedule for a faculty member.
        Input: faculty_id (e.g., F04)
        Returns: schedule as formatted text
        """
        return _read_faculty_csv(faculty_id)

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
        path = resolve_output_path(
            f"faculty_{faculty_id.upper()}_timetable.csv"
        )
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
        report_path = resolve_output_path("summary_report.txt")
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
            result = _find_sub(faculty_id, day)
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
        Creates backup and logs the operation.
        """
        parts = [p.strip() for p in input_str.split(",", 6)]
        if len(parts) < 7:
            return "Error: expected 7 comma-separated fields."
        section, day, p_start, p_end, absent, substitute, reason = parts
        p_start, p_end = int(p_start), int(p_end)

        # Validate
        if not _is_faculty_free(substitute, day, p_start, p_end):
            return (
                f"Cannot commit: {substitute} is not free on {day} "
                f"P{p_start}-P{p_end}."
            )

        # Backup
        backup = backup_timetable(section)

        # Read current section timetable
        path = resolve_output_path(
            f"section_{section}_timetable.csv"
        )
        df = pd.read_csv(path)
        row_mask = df["Day"] == day
        if not row_mask.any():
            return f"Day {day} not found in section {section} timetable."

        pre_state = df[row_mask].to_csv(index=False)

        # Apply substitution
        for p in range(p_start, p_end + 1):
            col = f"P{p}"
            if col in df.columns:
                current = str(df.loc[row_mask, col].values[0])
                df.loc[row_mask, col] = f"{current}→{substitute}"

        post_state = df[row_mask].to_csv(index=False)

        # Write atomically
        _atomic_write_csv(path, df)

        # Log
        log_path = log_operation(
            action="substitute",
            absent_faculty=absent,
            section_id=section,
            day=day,
            period_range=(p_start, p_end),
            substitute_faculty=substitute,
            reasoning_chain=[reason],
            pre_state=pre_state,
            post_state=post_state,
            commit_result="SUCCESS",
            backup_path=str(backup),
        )

        return (
            f"Committed: {substitute} covers section {section} on {day} "
            f"P{p_start}-P{p_end}. Log: {log_path.name}"
        )

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

        path = resolve_output_path(f"faculty_{fid}_timetable.csv")
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
    ]


# ── Agent creation ────────────────────────────────────────────────────────────

def _get_agent_system_prompt() -> str:
    """Build system prompt for the agent."""
    try:
        from src.phase5.prompt_builder import build_system_prompt
        base = build_system_prompt(
            outputs_dir=config.OUTPUT_DIR,
            data_dir=config.DATA_DIR,
        )
    except Exception:
        base = "University timetable assistant. CSE dept, 12 sections A-L."

    return base + """
You are an autonomous timetable agent with read and write tools.
When a teacher is absent, follow these steps IN ORDER:
1. Call get_absent_periods(faculty_id,day) to find their actual schedule.
2. Call find_substitute(faculty_id,day) to find available substitutes.
3. Tell the user the proposed assignment with exact period numbers.
4. Call commit_substitute only when confirmed.
5. Call generate_session_summary at end of session.

CRITICAL: Never guess period numbers. Always use get_absent_periods first.
For lab blocks P5-P6 substitute the full block atomically.
"""


def create_timetable_agent():
    """Create a tool-calling agent using bind_tools + manual ReAct loop."""
    if not config.GROQ_API_KEY:
        return None

    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from src.phase5.llm_wrapper import get_llm, safe_llm_call

    tools = _make_tools()
    if tools is None:
        return None

    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def run_agent(input_dict: dict) -> dict:
        query = input_dict.get("input", "")
        messages = [
            SystemMessage(content=_get_agent_system_prompt()),
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
                arg_val = list(args.values())[0] if args else ""
                reasoning.append(f"Step {step + 1}: {name}({arg_val})")

                if name in tool_map:
                    try:
                        result = tool_map[name].invoke(arg_val)
                    except Exception as e:
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

