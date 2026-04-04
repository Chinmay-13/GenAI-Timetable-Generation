"""
run_all.py — Full timetable generation pipeline.

Phases:
  0 → Validate input CSVs
  1 → Build faculty-section assignment map
  2 → Lock lab slots (P5-P6)
  3 → CP-SAT theory scheduling
  3.5 → Greedy room allocation
  4 → Export section/faculty CSVs + summary report
  RAG → Build FAISS index for AI assistant
"""

import sys
import time
from pathlib import Path

import pandas as pd

from src.phase0.validator import validate
from src.phase1.assignment_builder import build_assignment_map
from src.phase2.lab_scheduler import lock_labs
from src.phase3.theory_scheduler import solve_theory
from src.phase3_5.room_allocator import run_phase35
from src.phase4.output_generator import generate_outputs

# ── Helpers ───────────────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _ok(msg):   print(f"  {_GREEN}✓{_RESET} {msg}")
def _warn(msg): print(f"  {_YELLOW}⚠{_RESET} {msg}")
def _err(msg):  print(f"  {_RED}✗{_RESET} {msg}")

def _section(title: str) -> None:
    print(f"\n{_BOLD}{_CYAN}{'─' * 55}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─' * 55}{_RESET}")

def _phase_header(label: str) -> float:
    print(f"\n{_BOLD}▶ {label}{_RESET}")
    return time.perf_counter()

def _phase_done(label: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    _ok(f"{label} completed in {elapsed:.2f}s")
    return elapsed


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_all(data_dir: str = "data") -> bool:
    pipeline_start = time.perf_counter()
    timings: dict[str, float] = {}

    _section("TIMETABLE GENERATION PIPELINE")
    print(f"  Data directory : {Path(data_dir).resolve()}")
    print(f"  Output dir     : {Path('outputs').resolve()}")

    try:
        # ── Phase 0 — Validate ────────────────────────────────────────────────
        t0 = _phase_header("Phase 0 — Input Validation")
        courses     = pd.read_csv(f"{data_dir}/courses.csv")
        faculty     = pd.read_csv(f"{data_dir}/faculty.csv")
        assignments = pd.read_csv(f"{data_dir}/assignments.csv")
        ok = validate(courses, faculty, assignments)
        if not ok:
            _err("Phase 0 validation FAILED — aborting")
            return False
        timings["Phase 0 · Validation"] = _phase_done("Validation", t0)

        # ── Phase 1 — Assignment Map ──────────────────────────────────────────
        t0 = _phase_header("Phase 1 — Assignment Map")
        assignment_map = build_assignment_map(data_dir=data_dir)
        timings["Phase 1 · Assignments"] = _phase_done("Assignment map built", t0)

        # ── Phase 2 — Lab Locking ─────────────────────────────────────────────
        t0 = _phase_header("Phase 2 — Lab Slot Locking")
        section_grid, faculty_grid, room_grid, lab_details = lock_labs(data_dir=data_dir)
        timings["Phase 2 · Lab locking"] = _phase_done("Lab slots locked", t0)

        # ── Phase 3 — Theory Scheduling ───────────────────────────────────────
        t0 = _phase_header("Phase 3 — CP-SAT Theory Scheduler")
        result = solve_theory(
            assignment_map=assignment_map,
            section_grid=section_grid,
            faculty_grid=faculty_grid,
            room_grid=room_grid,
            lab_details=lab_details,
            data_dir=data_dir,
        )
        timings["Phase 3 · CP-SAT solver"] = _phase_done("Theory schedule solved", t0)

        # ── Phase 3.5 — Room Allocation ───────────────────────────────────────
        t0 = _phase_header("Phase 3.5 — Greedy Room Allocation")
        room_result = run_phase35(
            section_grid=result["section_grid"],
            room_grid=result["room_grid"],
            assignment_map=result["assignment_map"],
            data_dir=data_dir,
            output_dir="outputs",
        )
        timings["Phase 3.5 · Room allocation"] = _phase_done("Room assignments written", t0)

        # ── Phase 4 — Output Generation ───────────────────────────────────────
        t0 = _phase_header("Phase 4 — Export CSVs & Summary Report")
        generate_outputs(result=result, data_dir=data_dir, output_dir="outputs")
        timings["Phase 4 · Export"] = _phase_done("Outputs generated", t0)

        # ── RAG Index ─────────────────────────────────────────────────────────
        t0 = _phase_header("RAG — FAISS Index")
        try:
            from src.phase5.rag_indexer import build_index
            index_result = build_index()
            if index_result and index_result[0] is not None:
                _ok("FAISS index built")
                timings["RAG · Index build"] = _phase_done("RAG index ready", t0)
            else:
                _warn("RAG index not built (sentence-transformers/faiss-cpu missing?)")
                timings["RAG · Index build"] = 0.0
        except ImportError:
            _warn("rag_indexer not importable — skipping")
            timings["RAG · Index build"] = 0.0
        except Exception as exc:
            _warn(f"RAG index failed: {exc}")
            timings["RAG · Index build"] = 0.0

        # ── Final Summary ─────────────────────────────────────────────────────
        total_elapsed = time.perf_counter() - pipeline_start
        _section("PIPELINE COMPLETE")

        col_w = max(len(k) for k in timings) + 2
        for phase, secs in timings.items():
            bar_len  = int((secs / max(total_elapsed, 0.01)) * 20)
            bar      = "█" * bar_len + "░" * (20 - bar_len)
            marker   = f"{_GREEN}✓{_RESET}"
            print(f"  {marker} {phase:<{col_w}} {bar}  {secs:>6.2f}s")

        print(f"\n  {_BOLD}Total wall time: {total_elapsed:.2f}s{_RESET}")

        _section("GENERATED OUTPUTS")
        outputs_dir = Path("outputs")
        artifacts = [
            ("Section timetables",  f"section_{{A-L}}_timetable.csv",   "12 files"),
            ("Faculty timetables",  f"faculty_{{F01-F20}}_timetable.csv","per faculty"),
            ("Room assignments",    "room_assignment.csv",               "Phase 3.5"),
            ("Summary report",      "summary_report.txt",                "quality stats"),
            ("RAG index",          "rag_index.faiss",                   "AI assistant"),
        ]
        for label, filename, note in artifacts:
            print(f"  • {label:<22} outputs/{filename:<38} [{note}]")

        print(f"\n  {_BOLD}Next steps:{_RESET}")
        print("    streamlit run app.py          # Launch web dashboard")
        print("    python src/phase5/agent.py    # Interactive AI agent CLI")
        print("    pytest tests/ -v              # Run test suite")

        return True

    except Exception as exc:
        import traceback
        _err(f"Pipeline failed: {exc}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
