from src.phase0.validator import validate
from src.phase1.assignment_builder import build_assignment_map
from src.phase2.lab_scheduler import lock_labs
from src.phase3.theory_scheduler import solve_theory
from src.phase4.output_generator import generate_outputs
import pandas as pd


def run_all(data_dir="data"):
    try:
        print("\nRunning Phase 0...")
        courses = pd.read_csv(f"{data_dir}/courses.csv")
        faculty = pd.read_csv(f"{data_dir}/faculty.csv")
        assignments = pd.read_csv(f"{data_dir}/assignments.csv")
        ok = validate(courses, faculty, assignments)
        if not ok:
            print("Phase 0 failed")
            return False

        print("\nRunning Phase 1...")
        assignment_map = build_assignment_map(data_dir=data_dir)

        print("\nRunning Phase 2...")
        section_grid, faculty_grid, room_grid, lab_details = lock_labs(data_dir=data_dir)

        print("\nRunning Phase 3...")
        result = solve_theory(
            assignment_map=assignment_map,
            section_grid=section_grid,
            faculty_grid=faculty_grid,
            room_grid=room_grid,
            lab_details=lab_details,
            data_dir=data_dir,
        )

        print("\nRunning Phase 4...")
        generate_outputs(result=result, data_dir=data_dir, output_dir="outputs")

        print("\nTIMETABLE GENERATION COMPLETE")
        print("Section timetables: outputs/section_*.csv")
        print("Faculty timetables: outputs/faculty_*.csv")
        print("Report: outputs/summary_report.txt")
        print("\nStarting AI Assistant...")
        print("Run: python src/phase5/chat.py")
        return True

    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        return False


if __name__ == "__main__":
    run_all()
