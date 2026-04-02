"""Central configuration for the timetable system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

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

PENALTY_STAGE_TIME = 120
REWARD_STAGE_TIME = 60
NUM_WORKERS = 8

PENALTY_GAP = 100
PENALTY_LAB_WINDOW = 50              # penalty for theory in P5-P6 on non-lab days
PENALTY_BACK_TO_BACK = 5
PENALTY_SAME_DAY = 3
REWARD_CONSECUTIVE = 40

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


def resolve_output_path(filename: str) -> Path:
    canonical = OUTPUT_DIR / filename
    latest = canonical.with_name(f"{canonical.stem}.latest{canonical.suffix}")
    return latest if latest.exists() else canonical


if __name__ == "__main__":
    print("config.py loaded successfully")
    print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"  DATA_DIR     : {DATA_DIR}")
    print(f"  OUTPUT_DIR   : {OUTPUT_DIR}")
    print(f"  SECTIONS     : {SECTIONS}")
    print(f"  DAYS         : {DAYS}")
    print(f"  PERIODS      : {PERIODS}")
    print(f"  THEORY_PERIODS: {THEORY_PERIODS}")
    print(f"  LAB_PERIODS  : {LAB_PERIODS}")
    print(f"  MAX_HOURS    : {MAX_HOURS}")
    print(f"  CREDIT_MAP   : {CREDIT_MAP}")
