import pandas as pd

DATA_DIR = "data"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PERIODS = list(range(1, 10))
LAB_PERIODS = [7, 8, 9]
SECTIONS = [chr(ord("A") + i) for i in range(12)]


def _build_empty_grid(keys):
    return {
        key: {
            day: {period: None for period in PERIODS}
            for day in DAYS
        }
        for key in keys
    }


def lock_labs(data_dir=DATA_DIR):
    lab_allotment = pd.read_csv(f"{data_dir}/lab_allotment.csv")
    faculty = pd.read_csv(f"{data_dir}/faculty.csv")
    rooms = pd.read_csv(f"{data_dir}/rooms.csv")

    faculty_ids = faculty["faculty_id"].tolist()
    room_names = rooms["room_name"].tolist()
    faculty_designation = dict(zip(faculty["faculty_id"], faculty["designation"]))

    section_grid = _build_empty_grid(SECTIONS)
    faculty_grid = _build_empty_grid(faculty_ids)
    room_grid = _build_empty_grid(room_names)

    conflicts = []
    lab_details = []

    print("\n=== PHASE 2: LAB SLOT LOCKING ===\n")

    for _, row in lab_allotment.iterrows():
        day = str(row["day"]).strip()
        course_code = str(row["course_code"]).strip()
        room = str(row["room"]).strip()
        faculty_id = str(row["faculty_id"]).strip()
        sections = [s.strip() for s in str(row["section_pair"]).split(",") if s.strip()]

        if day not in DAYS:
            conflicts.append(f"Invalid day in lab allotment: {day}")
            continue

        if len(sections) != 2:
            conflicts.append(f"Invalid section pair '{row['section_pair']}' for {course_code} on {day}")
            continue

        if faculty_designation.get(faculty_id) == "Prof":
            conflicts.append(f"Prof {faculty_id} assigned to lab {course_code} on {day}")

        for section in sections:
            if section not in SECTIONS:
                conflicts.append(f"Invalid section {section} in pair for {course_code} on {day}")

        for period in LAB_PERIODS:
            for section in sections:
                if section_grid[section][day][period] is not None:
                    conflicts.append(
                        f"Section conflict: {section} already has {section_grid[section][day][period]} at {day} P{period}"
                    )
            if faculty_grid[faculty_id][day][period] is not None:
                conflicts.append(
                    f"Faculty conflict: {faculty_id} already has {faculty_grid[faculty_id][day][period]} at {day} P{period}"
                )
            if room_grid[room][day][period] is not None:
                conflicts.append(
                    f"Room conflict: {room} already has {room_grid[room][day][period]} at {day} P{period}"
                )

        if conflicts:
            continue

        token = f"{course_code}_LAB"
        for period in LAB_PERIODS:
            for section in sections:
                section_grid[section][day][period] = token
            faculty_grid[faculty_id][day][period] = token
            room_grid[room][day][period] = token

        lab_details.append(
            {
                "day": day,
                "course_code": course_code,
                "sections": sections,
                "room": room,
                "faculty_id": faculty_id,
            }
        )

        print(
            f"LOCKED: {course_code.replace('UE24CS251A', 'DDCO').replace('UE24CS252A', 'DSA')}"
            f" | {day} | P7-P9 | Sections {sections[0]}+{sections[1]} | Room: {room} | Faculty: {faculty_id}"
        )

    if conflicts:
        print("\nConflicts detected in lab allotment:")
        for c in conflicts:
            print(f"  CONFLICT: {c}")
        raise ValueError("Phase 2 failed due lab conflicts.")

    print("\nNo conflicts detected in lab allotment")
    print("PHASE 2 COMPLETE - lab slots locked")

    return section_grid, faculty_grid, room_grid, lab_details


if __name__ == "__main__":
    lock_labs()
