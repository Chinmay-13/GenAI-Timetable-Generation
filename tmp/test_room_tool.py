"""
Direct test of the get_room_availability tool logic —
bypasses the agent/LLM and calls the function directly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import resolve_output_path
import pandas as _pd

# ── inline copy of the fixed tool logic (must match agent.py) ───────────────

_DAY_ALIASES = {
    "mon": "Monday",    "monday": "Monday",
    "tue": "Tuesday",   "tuesday": "Tuesday",   "tues": "Tuesday",
    "wed": "Wednesday", "wednesday": "Wednesday",
    "thu": "Thursday",  "thursday": "Thursday", "thur": "Thursday",
                                                "thurs": "Thursday",
    "fri": "Friday",    "friday": "Friday",
}


def get_room_availability(day_arg: str, period_arg: str) -> str:
    input_str = f"{day_arg},{period_arg}"
    parts = [p.strip() for p in input_str.split(",", 1)]
    if len(parts) != 2:
        return "bad input"
    day_raw, period_raw = parts

    day = _DAY_ALIASES.get(day_raw.strip().lower())
    if day is None:
        return f"Unknown day: {day_raw}"

    period_clean = str(period_raw).strip().upper().lstrip("P")
    try:
        period_int = int(period_clean)
    except ValueError:
        return f"Unknown period: {period_raw}"
    if period_int not in range(1, 5):
        return f"Period {period_int} is out of range (must be 1-4)"
    period_label = f"P{period_int}"

    ra_path = resolve_output_path("room_assignment.csv")
    if not ra_path.exists():
        return "room_assignment.csv not found"

    ra_df = _pd.read_csv(ra_path)
    slot_df = ra_df[
        (ra_df["Day"].str.strip().str.lower() == day.lower()) &
        (ra_df["Period"].str.strip().str.upper() == period_label)
    ]
    if slot_df.empty:
        return f"No data for {day} {period_label}"

    occupied_rows = slot_df[slot_df["Room"] != "ROOM_UNASSIGNED"]
    occupied_rooms: dict = {}
    for _, r in occupied_rows.iterrows():
        rname = str(r["Room"]).strip()
        label = f"Sec {r['Section']} ({r['Course']})"
        occupied_rooms.setdefault(rname, []).append(label)

    rooms_path = Path(__file__).resolve().parents[1] / "data" / "rooms.csv"
    rooms_df = _pd.read_csv(rooms_path)
    all_classrooms = rooms_df[
        rooms_df["room_type"].str.upper().isin(["CLASSROOM", "LECTURE_HALL"])
    ]
    free_rooms = all_classrooms[
        ~all_classrooms["room_name"].isin(occupied_rooms.keys())
    ]

    unassigned_count = (slot_df["Room"] == "ROOM_UNASSIGNED").sum()
    sections_in_slot = slot_df["Section"].nunique()

    lines = [
        f"Room status for {day} {period_label}:",
        f"  Sections with classes : {sections_in_slot}",
        f"  Classrooms occupied   : {len(occupied_rooms)}",
        f"  Classrooms free       : {len(free_rooms)}",
    ]
    if unassigned_count:
        lines.append(f"  ⚠ {unassigned_count} ROOM_UNASSIGNED")

    if free_rooms.empty:
        lines.append(
            f"\n✗ No free classrooms on {day} {period_label} "
            f"— all {len(occupied_rooms)} classrooms are in use."
        )
    else:
        lines.append(f"\n✓ FREE classrooms ({len(free_rooms)}):")
        for _, r in free_rooms.iterrows():
            lines.append(
                f"  • {r['room_name']:<8}  capacity={r['capacity']}  floor={r['floor']}"
            )

    lines.append(f"\n● OCCUPIED classrooms ({len(occupied_rooms)}):")
    for room_name, users in sorted(occupied_rooms.items()):
        lines.append(f"  • {room_name:<8}  → {', '.join(users)}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: get_room_availability(day='Tuesday', period='P2')")
    print("=" * 60)
    print(get_room_availability("Tuesday", "P2"))

    print("\n" + "=" * 60)
    print("TEST 2: aliases — get_room_availability('tue', '2')")
    print("=" * 60)
    print(get_room_availability("tue", "2"))

    print("\n" + "=" * 60)
    print("TEST 3: Monday P1")
    print("=" * 60)
    print(get_room_availability("Monday", "P1"))

    print("\n" + "=" * 60)
    print("TEST 4: bad day")
    print("=" * 60)
    print(get_room_availability("Toosdae", "P2"))

    print("\n" + "=" * 60)
    print("TEST 5: period out of range (P6 = lab)")
    print("=" * 60)
    print(get_room_availability("Tuesday", "P6"))
