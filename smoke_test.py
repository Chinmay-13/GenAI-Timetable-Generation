"""
smoke_test.py - Multi-semester end-to-end smoke test.

Run from project root (inside venv):
    python smoke_test.py

Tests:
  T1  Legacy pipeline        python run_all.py
  T2  sem3 pipeline          python run_all.py --sem cse_sem3
  T3  sem5 pipeline          python run_all.py --sem cse_sem5
  T4  RAG isolation          sem3 and sem5 both return non-empty, different docs
  T5  Metadata check         sem3 has_electives=False, sem5=True
  T6  list_available_sems    ['cse_sem3', 'cse_sem5'] present
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_GREEN = "\033[92m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []


def _pass(label: str, detail: str) -> None:
    results.append((label, True, detail))
    print(f"  {_GREEN}[PASS]{_RESET} {label}: {detail}")


def _fail(label: str, detail: str) -> None:
    results.append((label, False, detail))
    print(f"  {_RED}[FAIL]{_RESET} {label}: {detail}")


def _run_pipeline(label: str, args: list[str]) -> bool:
    """Run run_all.py with the given CLI args using the current interpreter."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, str(PROJECT_ROOT / "run_all.py")] + args
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PROJECT_ROOT),
            timeout=300,
            env=env,
        )
        if proc.returncode == 0:
            _pass(label, "exit 0 - pipeline succeeded")
            return True

        out_tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-10:])
        _fail(label, f"exit {proc.returncode}\n{out_tail}")
        return False
    except subprocess.TimeoutExpired:
        _fail(label, "timed out after 300 s")
        return False
    except Exception as exc:
        _fail(label, f"exception: {exc}")
        return False


print(f"\n{_BOLD}T1 - Legacy pipeline (run_all.py){_RESET}")
_run_pipeline("T1 legacy", [])

print(f"\n{_BOLD}T2 - sem3 pipeline (run_all.py --sem cse_sem3){_RESET}")
_run_pipeline("T2 cse_sem3", ["--sem", "cse_sem3"])

print(f"\n{_BOLD}T3 - sem5 pipeline (run_all.py --sem cse_sem5){_RESET}")
_run_pipeline("T3 cse_sem5", ["--sem", "cse_sem5"])

print(f"\n{_BOLD}T4 - RAG isolation check{_RESET}")
try:
    from src.phase5.rag_indexer import retrieve

    query = "What is the timetable for section A?"
    sem3_results = retrieve(query, k=3, sem_id="cse_sem3")
    sem5_results = retrieve(query, k=3, sem_id="cse_sem5")

    if not sem3_results:
        _fail("T4 RAG isolation", "sem3 retrieval returned no documents")
    elif not sem5_results:
        _fail("T4 RAG isolation", "sem5 retrieval returned no documents")
    elif sem3_results != sem5_results:
        _pass("T4 RAG isolation", "sem3 and sem5 indexes return different documents")
    else:
        _fail(
            "T4 RAG isolation",
            "sem3 and sem5 returned identical results - indexes may be shared",
        )
except ImportError as exc:
    _fail("T4 RAG isolation", f"ImportError (sentence-transformers/faiss missing?): {exc}")
except FileNotFoundError as exc:
    _fail("T4 RAG isolation", f"Index file not found - did T2/T3 build it? {exc}")
except Exception as exc:
    _fail("T4 RAG isolation", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")

print(f"\n{_BOLD}T5 - Semester metadata check{_RESET}")
try:
    from src.phase0.loader import get_sem_metadata

    sem3_meta = get_sem_metadata("cse_sem3")
    sem5_meta = get_sem_metadata("cse_sem5")
    print(f"     sem3: {sem3_meta}")
    print(f"     sem5: {sem5_meta}")

    sem3_ok = sem3_meta.get("has_electives") is False
    sem5_ok = sem5_meta.get("has_electives") is True
    if sem3_ok and sem5_ok:
        _pass("T5 metadata", "sem3 has_electives=False and sem5 has_electives=True")
    else:
        messages = []
        if not sem3_ok:
            messages.append(
                f"sem3 has_electives={sem3_meta.get('has_electives')} (expected False)"
            )
        if not sem5_ok:
            messages.append(
                f"sem5 has_electives={sem5_meta.get('has_electives')} (expected True)"
            )
        _fail("T5 metadata", "; ".join(messages))
except Exception as exc:
    _fail("T5 metadata", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")

print(f"\n{_BOLD}T6 - list_available_semesters(){_RESET}")
try:
    from config import list_available_semesters

    sems = list_available_semesters()
    print(f"     returned: {sems}")
    has_sem3 = "cse_sem3" in sems
    has_sem5 = "cse_sem5" in sems
    if has_sem3 and has_sem5:
        _pass("T6 list_semesters", f"Both cse_sem3 and cse_sem5 present: {sems}")
    else:
        missing = [s for s, ok in [("cse_sem3", has_sem3), ("cse_sem5", has_sem5)] if not ok]
        _fail("T6 list_semesters", f"Missing from result: {missing} (got {sems})")
except Exception as exc:
    _fail("T6 list_semesters", f"{type(exc).__name__}: {exc}")

print(f"\n{_BOLD}{'-' * 60}{_RESET}")
print(f"{_BOLD}  SMOKE TEST RESULTS{_RESET}")
print(f"{_BOLD}{'-' * 60}{_RESET}")
for label, ok, detail in results:
    icon = f"{_GREEN}[PASS]{_RESET}" if ok else f"{_RED}[FAIL]{_RESET}"
    short = detail.split("\n")[0][:80]
    print(f"  {icon}  {label:<20s}  {short}")

passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"\n  {_BOLD}{passed}/{total} passed{_RESET}")
if passed == total:
    print(f"  {_GREEN}{_BOLD}All checks passed{_RESET}")
else:
    print(f"  {_RED}{_BOLD}{total - passed} failure(s) - see details above{_RESET}")
    sys.exit(1)
