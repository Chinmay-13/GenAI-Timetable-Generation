import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_all():
    courses       = pd.read_csv(DATA_DIR / "courses.csv")
    faculty       = pd.read_csv(DATA_DIR / "faculty.csv")
    assignments   = pd.read_csv(DATA_DIR / "assignments.csv")
    lab_allotment = pd.read_csv(DATA_DIR / "lab_allotment.csv")
    rooms         = pd.read_csv(DATA_DIR / "rooms.csv")

    print("=== DATA LOAD SUMMARY ===")
    print(f"Courses loaded       : {len(courses)}")
    print(f"Faculty loaded       : {len(faculty)}")
    print(f"Assignments loaded   : {len(assignments)}")
    print(f"Lab allotments loaded: {len(lab_allotment)}")
    print(f"Rooms loaded         : {len(rooms)}")
    print("=========================")

    return courses, faculty, assignments, lab_allotment, rooms

if __name__ == "__main__":
    load_all()
