"""Central configuration for the timetable system."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent

# legacy: will be removed after full migration
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ── Multi-semester path system ──────────────────────────────────────────
SEMESTERS_DIR = PROJECT_ROOT / "data"  # parent of all semester folders


@dataclass(frozen=True)
class SemesterPaths:
    """All resolved paths for a single semester."""
    sem_id: str
    data_dir: Path
    output_dir: Path
    rag_index_path: Path
    rag_docs_path: Path
    chat_memory_path: Path
    agent_ops_dir: Path


def get_sem_paths(sem_id: str) -> SemesterPaths:
    """
    Build a SemesterPaths object for the given semester slug.

    Layout:
        data/<sem_id>/          – input CSVs
        outputs/<sem_id>/       – generated outputs, RAG artefacts, chat memory
    """
    data = PROJECT_ROOT / "data" / sem_id
    out = PROJECT_ROOT / "outputs" / sem_id
    return SemesterPaths(
        sem_id=sem_id,
        data_dir=data,
        output_dir=out,
        rag_index_path=out / "rag_index.faiss",
        rag_docs_path=out / "rag_docs.json",
        chat_memory_path=out / "chat_memory.json",
        agent_ops_dir=out / "agent_ops",
    )


def list_available_semesters() -> list[str]:
    """
    Scan SEMESTERS_DIR for subdirectories and return their names as valid
    sem_id slugs.  Only directories are returned (files are ignored).
    """
    if not SEMESTERS_DIR.is_dir():
        return []
    return sorted(
        entry.name
        for entry in SEMESTERS_DIR.iterdir()
        if entry.is_dir()
    )


# ── Semester-agnostic constants ─────────────────────────────────────────

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PERIODS = list(range(1, 7))          # P1-P6 (6 periods per day)
THEORY_PERIODS = list(range(1, 5))   # P1-P4 (theory slots)
LAB_PERIODS = [5, 6]                 # P5-P6 (lab window)
LAB_PERIOD_START = 5
LAB_PERIOD_END = 6
LAB_BLOCK_LENGTH = 2
PERIODS_PER_DAY = 6
MAX_THEORY_PERIODS_PER_DAY = 4

SECTIONS = list("ABCDEFGHIJKL")
TOTAL_SECTIONS = 12

MAX_HOURS = {
    "Prof": 12,
    "Asso Prof": 16,
    "Asst Prof": 20,
}

WEEKLY_CAPS = {
    "Professor": 12,
    "Associate Professor": 16,
    "Assistant Professor": 20,
}

PENALTY_STAGE_TIME = 180   # was 120
REWARD_STAGE_TIME = 90     # was 60
NUM_WORKERS = 8

PENALTY_GAP = 100
PENALTY_LAB_WINDOW = 50              # penalty for theory in P5-P6 on non-lab days
PENALTY_BACK_TO_BACK = 5
PENALTY_SAME_DAY = 3
REWARD_CONSECUTIVE = 40

# faculty preference penalties (soft — all below PENALTY_LAB_WINDOW=50)
PENALTY_PREF_TIME = 8        # faculty prefers morning/afternoon; wrong half penalised
PENALTY_PREF_NO_BTB = 6     # faculty prefers no consecutive periods; each pair penalised
PENALTY_PREF_FREE_DAY = 10  # faculty wants one free day; each slot on that day penalised
PENALTY_ROOM_OVERCAP = 4    # room capacity < section size; prefer larger rooms (lowest priority)

# Consecutiveness for same-faculty same-section same-day (Change 2)
REWARD_CONSECUTIVENESS = 200         # soft reward weight for Phase B fallback
CONSECUTIVENESS_TIME_LIMIT = 30      # seconds for Phase A hard constraint attempt

SHORT_NAMES = {
    "UE24CS251A": "DDCO",
    "UE24CS252A": "DSA",
    "UE24MA242A": "MATH",
    "UE24CS242A": "WT",
    "UE24CS243A": "AFLL",
}

GEMINI_MODEL = "gemini-2.0-flash-lite"  # kept for future use
MAX_CHAT_HISTORY = 4

# --- Groq (primary LLM provider) ---
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL     = "llama-3.3-70b-versatile"    # primary
GROQ_MODEL_ALT = "llama-3.1-8b-instant"       # fast fallback

# --- Gemini (kept for future use, empty for now) ---
GEMINI_API_KEYS: list = []
GEMINI_API_KEY: str = ""

CREDIT_MAP = {
    5: {"theory_periods": 4, "lab_sessions": 1},
    4: {"theory_periods": 4, "lab_sessions": 0},
    3: {"theory_periods": 3, "lab_sessions": 0},
    2: {"theory_periods": 2, "lab_sessions": 0},
}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def get_theory_periods(credits: int, has_lab: bool) -> int:
    """
    Get number of theory periods per week from credits.
    For lab courses, lab hours are separate (Phase 2).
    """
    credits = int(credits)
    has_lab = _as_bool(has_lab)
    if credits == 5 and not has_lab:
        return 5
    if credits in CREDIT_MAP:
        return CREDIT_MAP[credits]["theory_periods"]
    return min(credits, 6)


def get_lab_sessions(credits: int, has_lab: bool) -> int:
    """
    Get number of lab sessions per week.
    Always 0 or 1 for now (one 2-period block).
    """
    credits = int(credits)
    if not _as_bool(has_lab):
        return 0
    if credits in CREDIT_MAP:
        return CREDIT_MAP[credits]["lab_sessions"]
    return 0


def resolve_output_path(filename: str, sem_id: str = None) -> Path:
    """
    Resolve the latest-or-canonical output path for *filename*.

    If *sem_id* is given, resolve inside that semester's output dir.
    Otherwise fall back to the legacy flat OUTPUT_DIR.
    """
    base = get_sem_paths(sem_id).output_dir if sem_id else OUTPUT_DIR
    canonical = base / filename
    latest = canonical.with_name(f"{canonical.stem}.latest{canonical.suffix}")
    return latest if latest.exists() else canonical


if __name__ == "__main__":
    print("config.py loaded successfully")
    print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"  DATA_DIR     : {DATA_DIR}  (legacy)")
    print(f"  OUTPUT_DIR   : {OUTPUT_DIR}  (legacy)")
    print(f"  SEMESTERS_DIR: {SEMESTERS_DIR}")
    print(f"  SECTIONS     : {SECTIONS}")
    print(f"  DAYS         : {DAYS}")
    print(f"  PERIODS      : {PERIODS}")
    print(f"  THEORY_PERIODS: {THEORY_PERIODS}")
    print(f"  LAB_PERIODS  : {LAB_PERIODS}")
    print(f"  MAX_HOURS    : {MAX_HOURS}")
    print(f"  CREDIT_MAP   : {CREDIT_MAP}")

    print("\n── Semester path demo ──")
    demo = get_sem_paths("cse_sem3")
    for field in ("sem_id", "data_dir", "output_dir", "rag_index_path",
                  "rag_docs_path", "chat_memory_path", "agent_ops_dir"):
        print(f"  {field:20s}: {getattr(demo, field)}")

    print(f"\n  Available semesters: {list_available_semesters()}")
