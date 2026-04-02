"""
demo_ai.py — Manual test script for the autonomous agent system.

Tests:
1. commit_substitute via direct tool call (no API key needed)
2. list_agent_ops
3. generate_session_summary
4. rollback_last_operation
5. Verify restored timetable
"""
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import resolve_output_path
from src.phase5.agent_ops import list_operations
import pandas as pd


def main():
    print("=" * 60)
    print("DEMO: Autonomous Agent System — Direct Tool Calls")
    print("=" * 60)

    # Import tools by calling _make_tools()
    from src.phase5.agent import _make_tools
    tools = _make_tools()
    if tools is None:
        print("ERROR: Could not create tools (langchain not installed?)")
        return

    # Build tool lookup
    tool_map = {t.name: t for t in tools}

    # ── Step 1: find_substitute ──────────────────────────────────────────
    print("\n--- Step 1: Find substitute for F03 on Monday ---")
    find_sub = tool_map["find_substitute"]
    sub_result = find_sub.invoke("F03,Monday")
    print(sub_result)

    # ── Step 2: Read section A timetable before commit ──────────────────
    print("\n--- Step 2: Section A timetable BEFORE commit ---")
    section_a_path = resolve_output_path("section_A_timetable.csv")
    before_df = pd.read_csv(section_a_path)
    monday_before = before_df[before_df["Day"] == "Monday"]
    print(monday_before.to_string(index=False))

    # ── Step 3: commit_substitute ────────────────────────────────────────
    print("\n--- Step 3: Commit substitute ---")
    commit = tool_map["commit_substitute"]
    # F03 teaches WT to sections A,B,C — commit a substitute for section A
    # on Monday at the period where F03 teaches
    # First find which period F03 teaches section A on Monday
    faculty_path = resolve_output_path("faculty_F03_timetable.csv")
    f03_df = pd.read_csv(faculty_path)
    f03_monday = f03_df[f03_df["Day"] == "Monday"]
    print(f"F03's Monday schedule: {f03_monday.to_string(index=False)}")

    # Find a period where F03 is busy on Monday
    busy_period = None
    busy_section = None
    for p in range(1, 7):
        col = f"P{p}"
        val = str(f03_monday[col].values[0]).strip()
        if val != "----":
            busy_period = p
            # Extract section from cell like "WT (A)"
            if "(" in val and ")" in val:
                busy_section = val.split("(")[1].split(")")[0]
            break

    if busy_period is None:
        print("F03 has no classes on Monday — picking F04 instead")
        faculty_path = resolve_output_path("faculty_F04_timetable.csv")
        f04_df = pd.read_csv(faculty_path)
        f04_monday = f04_df[f04_df["Day"] == "Monday"]
        for p in range(1, 5):
            col = f"P{p}"
            val = str(f04_monday[col].values[0]).strip()
            if val != "----":
                busy_period = p
                if "(" in val and ")" in val:
                    busy_section = val.split("(")[1].split(")")[0]
                break
        absent_faculty = "F04"
    else:
        absent_faculty = "F03"

    if busy_period and busy_section:
        # Find a free substitute for that period
        free_slots = tool_map["find_free_slots"]

        # Try F16 as substitute
        sub_faculty = "F16"
        free_result = free_slots.invoke(f"{sub_faculty},Monday")
        print(f"\n{sub_faculty} Monday availability: {free_result}")

        if f"P{busy_period}" in free_result:
            commit_input = (
                f"{busy_section},Monday,{busy_period},{busy_period},"
                f"{absent_faculty},{sub_faculty},"
                f"Demo: covering for absent {absent_faculty}"
            )
            print(f"\nCommitting: {commit_input}")
            commit_result = commit.invoke(commit_input)
            print(f"Result: {commit_result}")
        else:
            # Try other faculty
            for fid_num in range(16, 21):
                sub_faculty = f"F{fid_num:02d}"
                free_result = free_slots.invoke(f"{sub_faculty},Monday")
                if f"P{busy_period}" in free_result:
                    commit_input = (
                        f"{busy_section},Monday,{busy_period},{busy_period},"
                        f"{absent_faculty},{sub_faculty},"
                        f"Demo: covering for absent {absent_faculty}"
                    )
                    print(f"\nCommitting: {commit_input}")
                    commit_result = commit.invoke(commit_input)
                    print(f"Result: {commit_result}")
                    break
    else:
        print("Could not find a busy period to demo with.")
        return

    # ── Step 4: list_agent_ops ───────────────────────────────────────────
    print("\n--- Step 4: List agent operations ---")
    list_ops = tool_map["list_agent_ops"]
    ops_result = list_ops.invoke("5")
    print(ops_result)

    # ── Step 5: Show the JSON log file ───────────────────────────────────
    print("\n--- Step 5: Agent ops JSON log ---")
    ops = list_operations(1)
    if ops:
        print(json.dumps(ops[0], indent=2))
        op_id = ops[0]["operation_id"]
    else:
        print("No operations found!")
        return

    # ── Step 6: generate_session_summary ─────────────────────────────────
    print("\n--- Step 6: Session summary ---")
    summary = tool_map["generate_session_summary"]
    summary_result = summary.invoke("Monday absence demo")
    print(summary_result)

    # ── Step 7: Section A timetable AFTER commit ─────────────────────────
    print("\n--- Step 7: Section timetable AFTER commit ---")
    after_df = pd.read_csv(section_a_path)
    monday_after = after_df[after_df["Day"] == "Monday"]
    print(monday_after.to_string(index=False))

    # ── Step 8: Rollback ─────────────────────────────────────────────────
    print(f"\n--- Step 8: Rolling back operation {op_id} ---")
    rollback = tool_map["rollback_last_operation"]
    rollback_result = rollback.invoke(op_id)
    print(f"Result: {rollback_result}")

    # ── Step 9: Verify restored ──────────────────────────────────────────
    print("\n--- Step 9: Section timetable AFTER rollback ---")
    restored_df = pd.read_csv(section_a_path)
    monday_restored = restored_df[restored_df["Day"] == "Monday"]
    print(monday_restored.to_string(index=False))

    # Verify restoration
    if monday_before.to_csv(index=False) == monday_restored.to_csv(index=False):
        print("\n✓ ROLLBACK VERIFIED: Timetable restored to original state.")
    else:
        print("\n✗ ROLLBACK MISMATCH: Timetable differs from original.")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
