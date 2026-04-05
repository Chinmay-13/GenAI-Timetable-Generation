"""
utils/health_check.py
Verifies every dependency of the timetable system in one call.
Accepts an optional sem_id so the UI can check per-semester paths.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is on sys.path when this file is imported standalone
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DATA_DIR, OUTPUT_DIR, SECTIONS, get_sem_paths

# ─────────────────────────────────────────────────────────────────────────────
# Expected files
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_INPUT_CSVS = [
    "courses.csv",
    "faculty.csv",
    "assignments.csv",
    "rooms.csv",
    "lab_allotment.csv",
]

REQUIRED_OUTPUT_FILES = (
    [f"section_{s}_timetable.csv" for s in SECTIONS]
    + ["summary_report.txt", "room_assignment.csv"]
)

RAG_INDEX_FILES = ["rag_index.faiss", "rag_docs.json"]

REQUIRED_PACKAGES = {
    "langchain_groq":          "langchain-groq",
    "langchain_core":          "langchain-core",
    "sentence_transformers":   "sentence-transformers",
    "faiss":                   "faiss-cpu",
    "ortools":                 "ortools",
    "streamlit":               "streamlit",
    "altair":                  "altair",
}


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def _check(ok: bool, msg_ok: str, msg_fail: str) -> dict:
    return {"ok": ok, "message": msg_ok if ok else msg_fail}


def check_input_csvs(data_dir: Path) -> Dict[str, dict]:
    results = {}
    for fname in REQUIRED_INPUT_CSVS:
        path = data_dir / fname
        results[fname] = _check(
            path.exists(),
            f"Found ({path})",
            f"MISSING — expected at {path}",
        )
    return results


def check_output_files(output_dir: Path, sem_id: Optional[str] = None) -> Dict[str, dict]:
    results = {}
    hint = f"python run_all.py --sem {sem_id}" if sem_id else "python run_all.py"
    for fname in REQUIRED_OUTPUT_FILES:
        path = output_dir / fname
        results[fname] = _check(
            path.exists(),
            "Found",
            f"MISSING — run {hint}",
        )
    return results


def check_rag_index(output_dir: Path, sem_id: Optional[str] = None) -> Dict[str, dict]:
    results = {}
    hint = f"python src/phase5/rag_indexer.py --sem {sem_id}" if sem_id else "python src/phase5/rag_indexer.py"
    for fname in RAG_INDEX_FILES:
        path = output_dir / fname
        results[fname] = _check(
            path.exists(),
            "Found",
            f"MISSING — run Rebuild RAG Index or {hint}",
        )
    return results


def check_env() -> Dict[str, dict]:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    return {
        "GROQ_API_KEY": _check(
            bool(groq_key),
            f"Set ({groq_key[:6]}…)" if groq_key else "Set",
            "NOT SET — add to .env: GROQ_API_KEY=your_key",
        )
    }


def check_packages() -> Dict[str, dict]:
    results = {}
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
            ok = True
            msg_ok = "Importable"
            msg_fail = ""
        except ImportError:
            ok = False
            msg_ok = ""
            msg_fail = f"NOT INSTALLED — pip install {pip_name}"
        results[pip_name] = _check(ok, msg_ok, msg_fail)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Master check
# ─────────────────────────────────────────────────────────────────────────────

def check_system_health(sem_id: Optional[str] = None) -> Dict[str, object]:
    """
    Run all health checks and return a structured report.

    Parameters
    ----------
    sem_id : str or None
        If given, resolve input/output paths from the semester-specific dirs.
        If None, fall back to the legacy flat DATA_DIR / OUTPUT_DIR.

    Return schema:
    {
        "overall_ok": bool,
        "input_csvs":     { filename: {"ok": bool, "message": str}, ... },
        "output_files":   { filename: {"ok": bool, "message": str}, ... },
        "rag_index":      { filename: {"ok": bool, "message": str}, ... },
        "environment":    { "GROQ_API_KEY": {"ok": bool, "message": str} },
        "packages":       { pip_name: {"ok": bool, "message": str}, ... },
        "summary": {
            "total": int, "passed": int, "failed": int,
            "failed_items": [str, ...]
        }
    }
    """
    if sem_id:
        sem_paths = get_sem_paths(sem_id)
        data_dir   = sem_paths.data_dir
        output_dir = sem_paths.output_dir
    else:
        data_dir   = DATA_DIR
        output_dir = OUTPUT_DIR

    report: Dict[str, object] = {
        "input_csvs":   check_input_csvs(data_dir),
        "output_files": check_output_files(output_dir, sem_id),
        "rag_index":    check_rag_index(output_dir, sem_id),
        "environment":  check_env(),
        "packages":     check_packages(),
    }

    all_checks = {}
    for section, items in report.items():
        for key, result in items.items():
            all_checks[f"{section}/{key}"] = result

    total  = len(all_checks)
    passed = sum(1 for r in all_checks.values() if r["ok"])
    failed = total - passed
    failed_items = [k for k, r in all_checks.items() if not r["ok"]]

    report["overall_ok"] = failed == 0
    report["summary"] = {
        "total":        total,
        "passed":       passed,
        "failed":       failed,
        "failed_items": failed_items,
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem", default=None, help="Semester slug, e.g. cse_sem3")
    args = parser.parse_args()

    report = check_system_health(sem_id=args.sem)
    summary = report["summary"]

    print("\n═══════════════════════════════════════")
    print("   TIMETABLE SYSTEM — HEALTH CHECK")
    if args.sem:
        print(f"   Semester: {args.sem}")
    print("═══════════════════════════════════════")

    SECTIONS_ORDER = [
        ("input_csvs",   "Input CSVs"),
        ("output_files", "Output Files"),
        ("rag_index",    "RAG Index"),
        ("environment",  "Environment"),
        ("packages",     "Python Packages"),
    ]
    for key, label in SECTIONS_ORDER:
        print(f"\n▌ {label}")
        for item, result in report[key].items():
            icon = "✓" if result["ok"] else "✗"
            print(f"  {icon} {item:<40} {result['message']}")

    print(f"\n{'─' * 55}")
    status = "ALL CLEAR" if report["overall_ok"] else "ISSUES FOUND"
    print(f"  {status}: {summary['passed']}/{summary['total']} checks passed")
    if summary["failed_items"]:
        print("  Failed:")
        for item in summary["failed_items"]:
            print(f"    • {item}")
    print("═══════════════════════════════════════\n")
