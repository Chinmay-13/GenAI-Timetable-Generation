import pandas as pd

from src.phase0.validator import validate
from src.phase1.assignment_builder import build_assignment_map
from src.phase2.lab_scheduler import lock_labs
from src.phase3.theory_scheduler import solve_theory
from src.phase4.output_generator import generate_outputs


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
        print("Section timetables : outputs/section_*.csv")
        print("Faculty timetables : outputs/faculty_*.csv")
        print("Report             : outputs/summary_report.txt")
        print("RAG index          : outputs/rag_index.faiss")

        try:
            from src.phase5.rag_indexer import build_index

            print("\nBuilding RAG index...")
            index_result = build_index()
            if index_result and index_result[0] is not None:
                print("RAG index built successfully")
            else:
                print("RAG index not built")
        except ImportError:
            print("RAG indexer not available — skipping")
        except Exception as e:
            print(f"RAG index failed: {e}")

        print("\nTo use the system:")
        print("  streamlit run app.py")
        print("  python src/phase5/chat.py")
        print("  python src/phase5/agent.py")
        print("  pytest tests/ -v")
        return True

    except Exception as exc:
        import traceback
        print(f"Pipeline failed: {exc}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all()
