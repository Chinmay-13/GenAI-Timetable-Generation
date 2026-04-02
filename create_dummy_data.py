import pandas as pd
import os
from pathlib import Path

data_dir = str(Path(__file__).resolve().parent / "data")
os.makedirs(data_dir, exist_ok=True)

# ── 1. COURSES ──────────────────────────────────────────────────────────────
courses = pd.DataFrame([
    {"course_code": "UE24CS251A", "course_name": "Digital Design and Computer Organization",      "credits": 5, "theory_hours": 4, "lab_hours": 2, "has_lab": True,  "semester": 3, "department": "CSE"},
    {"course_code": "UE24CS252A", "course_name": "Data Structures and its Applications",          "credits": 5, "theory_hours": 4, "lab_hours": 2, "has_lab": True,  "semester": 3, "department": "CSE"},
    {"course_code": "UE24MA242A", "course_name": "Mathematics for Computer Science and Engg",     "credits": 4, "theory_hours": 4, "lab_hours": 0, "has_lab": False, "semester": 3, "department": "CSE"},
    {"course_code": "UE24CS242A", "course_name": "Web Technologies",                              "credits": 4, "theory_hours": 4, "lab_hours": 0, "has_lab": False, "semester": 3, "department": "CSE"},
    {"course_code": "UE24CS243A", "course_name": "Automata Formal Languages and Logic",           "credits": 4, "theory_hours": 4, "lab_hours": 0, "has_lab": False, "semester": 3, "department": "CSE"},
])

# ── 2. FACULTY ───────────────────────────────────────────────────────────────
faculty = pd.DataFrame([
    {"faculty_id": "F01",  "name": "Prof. Aryan Sharma",     "designation": "Prof"},
    {"faculty_id": "F02",  "name": "Prof. Meera Iyer",       "designation": "Prof"},
    {"faculty_id": "F03",  "name": "Prof. Suresh Nair",      "designation": "Prof"},
    {"faculty_id": "F04",  "name": "Prof. Kavya Reddy",      "designation": "Asso Prof"},
    {"faculty_id": "F05",  "name": "Prof. Rajan Pillai",     "designation": "Asso Prof"},
    {"faculty_id": "F06",  "name": "Prof. Divya Menon",      "designation": "Asso Prof"},
    {"faculty_id": "F07",  "name": "Prof. Kiran Bhat",       "designation": "Asso Prof"},
    {"faculty_id": "F08",  "name": "Prof. Ananya Das",       "designation": "Asso Prof"},
    {"faculty_id": "F09",  "name": "Prof. Siddharth Rao",    "designation": "Asso Prof"},
    {"faculty_id": "F10",  "name": "Prof. Preethi Joshi",    "designation": "Asso Prof"},
    {"faculty_id": "F11",  "name": "Prof. Vikram Hegde",     "designation": "Asst Prof"},
    {"faculty_id": "F12",  "name": "Prof. Sneha Kulkarni",   "designation": "Asst Prof"},
    {"faculty_id": "F13",  "name": "Prof. Arun Krishnan",    "designation": "Asst Prof"},
    {"faculty_id": "F14",  "name": "Prof. Deepa Verma",      "designation": "Asst Prof"},
    {"faculty_id": "F15",  "name": "Prof. Rohit Shenoy",     "designation": "Asst Prof"},
    {"faculty_id": "F16",  "name": "Prof. Lakshmi Prasad",   "designation": "Asst Prof"},
    {"faculty_id": "F17",  "name": "Prof. Nandini Gowda",    "designation": "Asst Prof"},
    {"faculty_id": "F18",  "name": "Prof. Tejas Murthy",     "designation": "Asst Prof"},
    {"faculty_id": "F19",  "name": "Prof. Pooja Srinivas",   "designation": "Asst Prof"},
    {"faculty_id": "F20",  "name": "Prof. Harsha Rao",       "designation": "Asst Prof"},
])

# ── 3. FACULTY-SECTION ASSIGNMENT ────────────────────────────────────────────
# Each row = one faculty handles N sections of a course
# sections_handled = comma-separated section letters
assignments = pd.DataFrame([
    # DDCO (12 sections split across 6 Asso/Asst Profs — no Prof for lab courses)
    {"faculty_id": "F04",  "course_code": "UE24CS251A", "sections_handled": "A,B"},
    {"faculty_id": "F05",  "course_code": "UE24CS251A", "sections_handled": "C,D"},
    {"faculty_id": "F06",  "course_code": "UE24CS251A", "sections_handled": "E,F"},
    {"faculty_id": "F07",  "course_code": "UE24CS251A", "sections_handled": "G,H"},
    {"faculty_id": "F08",  "course_code": "UE24CS251A", "sections_handled": "I,J"},
    {"faculty_id": "F09",  "course_code": "UE24CS251A", "sections_handled": "K,L"},
    # DSA (12 sections split across 6 Asso/Asst Profs)
    {"faculty_id": "F10",  "course_code": "UE24CS252A", "sections_handled": "A,B"},
    {"faculty_id": "F11",  "course_code": "UE24CS252A", "sections_handled": "C,D"},
    {"faculty_id": "F12",  "course_code": "UE24CS252A", "sections_handled": "E,F"},
    {"faculty_id": "F13",  "course_code": "UE24CS252A", "sections_handled": "G,H"},
    {"faculty_id": "F14",  "course_code": "UE24CS252A", "sections_handled": "I,J"},
    {"faculty_id": "F15",  "course_code": "UE24CS252A", "sections_handled": "K,L"},
    # MATH (12 sections — Profs can take this, theory only)
    {"faculty_id": "F01",  "course_code": "UE24MA242A", "sections_handled": "A,B,C"},
    {"faculty_id": "F02",  "course_code": "UE24MA242A", "sections_handled": "D,E,F"},
    {"faculty_id": "F16",  "course_code": "UE24MA242A", "sections_handled": "G,H,I"},
    {"faculty_id": "F17",  "course_code": "UE24MA242A", "sections_handled": "J,K,L"},
    # WT (12 sections — Profs can take this)
    {"faculty_id": "F03",  "course_code": "UE24CS242A", "sections_handled": "A,B,C"},
    {"faculty_id": "F18",  "course_code": "UE24CS242A", "sections_handled": "D,E,F"},
    {"faculty_id": "F19",  "course_code": "UE24CS242A", "sections_handled": "G,H,I"},
    {"faculty_id": "F20",  "course_code": "UE24CS242A", "sections_handled": "J,K,L"},
    # AFLL (12 sections — matches data/assignments.csv exactly)
    # 8 single-section rows + 2 paired rows
    {"faculty_id": "F04",  "course_code": "UE24CS243A", "sections_handled": "A"},
    {"faculty_id": "F16",  "course_code": "UE24CS243A", "sections_handled": "B"},
    {"faculty_id": "F06",  "course_code": "UE24CS243A", "sections_handled": "C"},
    {"faculty_id": "F17",  "course_code": "UE24CS243A", "sections_handled": "D"},
    {"faculty_id": "F08",  "course_code": "UE24CS243A", "sections_handled": "E"},
    {"faculty_id": "F18",  "course_code": "UE24CS243A", "sections_handled": "F"},
    {"faculty_id": "F10",  "course_code": "UE24CS243A", "sections_handled": "G"},
    {"faculty_id": "F19",  "course_code": "UE24CS243A", "sections_handled": "H"},
    {"faculty_id": "F12",  "course_code": "UE24CS243A", "sections_handled": "I,J"},
    {"faculty_id": "F14",  "course_code": "UE24CS243A", "sections_handled": "K,L"},
])

# ── 4. LAB ALLOTMENT ─────────────────────────────────────────────────────────
# This is the OUTPUT we want to eventually generate — using real data as reference
lab_allotment = pd.DataFrame([
    # DDCO Lab — Monday
    {"day": "Monday",   "course_code": "UE24CS251A", "section_pair": "A,D", "room": "2nd floor seminar hall",    "faculty_id": "F04"},
    {"day": "Monday",   "course_code": "UE24CS251A", "section_pair": "B,J", "room": "6th floor seminar hall",    "faculty_id": "F05"},
    {"day": "Monday",   "course_code": "UE24CS251A", "section_pair": "C,L", "room": "7th floor seminar hall",    "faculty_id": "F06"},
    {"day": "Monday",   "course_code": "UE24CS251A", "section_pair": "E,I", "room": "9th floor seminar hall",    "faculty_id": "F07"},
    {"day": "Monday",   "course_code": "UE24CS251A", "section_pair": "F,H", "room": "11th floor seminar hall",   "faculty_id": "F08"},
    {"day": "Monday",   "course_code": "UE24CS251A", "section_pair": "G,K", "room": "Ground floor seminar hall", "faculty_id": "F09"},
    # DSA Lab — Thursday
    {"day": "Thursday", "course_code": "UE24CS252A", "section_pair": "A,B", "room": "6th floor seminar hall",    "faculty_id": "F10"},
    {"day": "Thursday", "course_code": "UE24CS252A", "section_pair": "C,J", "room": "7th floor seminar hall",    "faculty_id": "F11"},
    {"day": "Thursday", "course_code": "UE24CS252A", "section_pair": "H,K", "room": "2nd floor seminar hall",    "faculty_id": "F12"},
    {"day": "Thursday", "course_code": "UE24CS252A", "section_pair": "E,D", "room": "Ground floor seminar hall", "faculty_id": "F13"},
    {"day": "Thursday", "course_code": "UE24CS252A", "section_pair": "F,I", "room": "9th floor seminar hall",    "faculty_id": "F14"},
    {"day": "Thursday", "course_code": "UE24CS252A", "section_pair": "G,L", "room": "11th floor seminar hall",   "faculty_id": "F15"},
])

# ── 5. ROOMS ─────────────────────────────────────────────────────────────────
rooms = pd.DataFrame([
    {"room_id": "R01", "room_name": "Ground floor seminar hall", "floor": 0,  "room_type": "LAB", "capacity": 60},
    {"room_id": "R02", "room_name": "2nd floor seminar hall",    "floor": 2,  "room_type": "LAB", "capacity": 60},
    {"room_id": "R03", "room_name": "6th floor seminar hall",    "floor": 6,  "room_type": "LAB", "capacity": 60},
    {"room_id": "R04", "room_name": "7th floor seminar hall",    "floor": 7,  "room_type": "LAB", "capacity": 60},
    {"room_id": "R05", "room_name": "9th floor seminar hall",    "floor": 9,  "room_type": "LAB", "capacity": 60},
    {"room_id": "R06", "room_name": "11th floor seminar hall",   "floor": 11, "room_type": "LAB", "capacity": 60},
    {"room_id": "R07", "room_name": "G06",  "floor": 0, "room_type": "CLASSROOM", "capacity": 70},
    {"room_id": "R08", "room_name": "G07",  "floor": 0, "room_type": "CLASSROOM", "capacity": 70},
    {"room_id": "R09", "room_name": "G08",  "floor": 0, "room_type": "CLASSROOM", "capacity": 70},
    {"room_id": "R10", "room_name": "G09",  "floor": 0, "room_type": "CLASSROOM", "capacity": 70},
    {"room_id": "R11", "room_name": "101",  "floor": 1, "room_type": "CLASSROOM", "capacity": 70},
    {"room_id": "R12", "room_name": "102",  "floor": 1, "room_type": "CLASSROOM", "capacity": 70},
])

# ── SAVE ALL ──────────────────────────────────────────────────────────────────
courses.to_csv(      f"{data_dir}/courses.csv",      index=False)
faculty.to_csv(      f"{data_dir}/faculty.csv",       index=False)
assignments.to_csv(  f"{data_dir}/assignments.csv",   index=False)
lab_allotment.to_csv(f"{data_dir}/lab_allotment.csv", index=False)
rooms.to_csv(        f"{data_dir}/rooms.csv",         index=False)

print("All dummy data files created successfully in data/")
print(f"  courses.csv       → {len(courses)} rows")
print(f"  faculty.csv       → {len(faculty)} rows")
print(f"  assignments.csv   → {len(assignments)} rows")
print(f"  lab_allotment.csv → {len(lab_allotment)} rows")
print(f"  rooms.csv         → {len(rooms)} rows")