"""
ai_explainer.py — Timetable AI explainer with RAG support.

Uses dynamic prompt builder (no hardcoded facts),
Groq LLM via safe wrapper with retry, and data-driven fallback.
"""
from pathlib import Path
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import resolve_output_path, SECTIONS
from src.phase5.prompt_builder import build_system_prompt
from src.phase5.llm_wrapper import get_llm, safe_llm_call

_CONTEXT_CACHE: Optional[Dict] = None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _extract_faculty_load_table(summary_text: str) -> str:
    lines = summary_text.splitlines()
    start = next(
        (i for i, line in enumerate(lines)
         if line.strip().lower().startswith("faculty load table")),
        None,
    )
    if start is None:
        return "Faculty load table not found in summary report."
    return "\n".join(lines[start:]).strip()


def load_actual_stats() -> dict:
    """Read real values from outputs/summary_report.txt and data/courses.csv."""
    import re
    stats = {
        "theory_slots": "unknown",
        "lab_slots": "unknown",
        "same_day_violations": "unknown",
        "back_to_back_violations": "unknown",
        "theory_hours_per_course": "unknown",
    }
    try:
        summary_path = resolve_output_path("summary_report.txt")
        content = summary_path.read_text(encoding="utf-8")
        m = re.search(r"Total theory slots placed:\s*(\d+)", content)
        if m:
            stats["theory_slots"] = m.group(1)
        m = re.search(r"Total lab slots placed:\s*(\d+)", content)
        if m:
            stats["lab_slots"] = m.group(1)
        m = re.search(r"same_subject_same_day:\s*(\d+)", content)
        if m:
            stats["same_day_violations"] = m.group(1)
        m = re.search(r"back_to_back_same_subject:\s*(\d+)", content)
        if m:
            stats["back_to_back_violations"] = m.group(1)
    except FileNotFoundError:
        pass
    try:
        courses_df = pd.read_csv(config.DATA_DIR / "courses.csv")
        hours = courses_df["theory_hours"].iloc[0] if not courses_df.empty else "unknown"
        stats["theory_hours_per_course"] = str(int(hours))
    except Exception:
        pass
    return stats


def setup_context(force_reload: bool = False) -> Dict:
    """Load all context needed for the AI layer."""
    global _CONTEXT_CACHE
    if _CONTEXT_CACHE is not None and not force_reload:
        return _CONTEXT_CACHE

    summary_path = resolve_output_path("summary_report.txt")
    section_a_path = resolve_output_path("section_A_timetable.csv")

    summary_report = _read_text(summary_path)
    section_a_csv = _read_text(section_a_path)
    section_a_df = pd.read_csv(section_a_path)

    context = {
        "summary_report": summary_report,
        "section_a_csv": section_a_csv,
        "section_a_csv_for_prompt": section_a_csv,
        "faculty_csv": _read_text(config.DATA_DIR / "faculty.csv"),
        "assignments_csv": _read_text(config.DATA_DIR / "assignments.csv"),
        "courses_csv": _read_text(config.DATA_DIR / "courses.csv"),
        "faculty_load_table": _extract_faculty_load_table(summary_report),
        "section_a_df": section_a_df,
        "lab_df": pd.read_csv(config.DATA_DIR / "lab_allotment.csv"),
    }

    # Use dynamic prompt builder
    context["system_prompt"] = build_system_prompt(
        outputs_dir=config.OUTPUT_DIR,
        data_dir=config.DATA_DIR,
        summary_text=summary_report,
    )
    _CONTEXT_CACHE = context
    return context


def _get_llm(context: Dict):
    """Create Groq LLM. Returns None if GROQ_API_KEY not set."""
    if not config.GROQ_API_KEY:
        return None
    try:
        return get_llm()
    except Exception:
        return None


def _history_block(history: Optional[List[Tuple[str, str]]]) -> str:
    if not history:
        return ""
    lines = ["Recent conversation context (last 4 exchanges):"]
    for user_text, ai_text in history[-4:]:
        lines.append(f"User: {user_text}")
        lines.append(f"AI: {ai_text}")
    return "\n".join(lines)


def _try_direct_lookup(user_question: str) -> Optional[str]:
    """
    If query mentions a faculty name or ID, read their CSV directly
    and return it as context. Returns None if no faculty detected.
    """
    q = user_question.lower()

    try:
        faculty_df = pd.read_csv(config.DATA_DIR / "faculty.csv")
    except Exception:
        return None

    matched_fid = None
    for _, row in faculty_df.iterrows():
        name_lower = str(row["name"]).lower()
        fid = str(row["faculty_id"]).strip()
        # Match by name words (>3 chars) or faculty ID
        if any(word in q for word in name_lower.split() if len(word) > 3):
            matched_fid = fid
            break
        if fid.lower() in q:
            matched_fid = fid
            break

    if not matched_fid:
        return None

    path = resolve_output_path(f"faculty_{matched_fid}_timetable.csv")
    if not path or not path.exists():
        return None

    try:
        df = pd.read_csv(path)
        day_counts = {}
        for _, row in df.iterrows():
            day = str(row.get("Day", ""))
            count = sum(
                1 for p in ["P1", "P2", "P3", "P4"]
                if str(row.get(p, "")).strip() not in ["", "----", "nan"]
            )
            day_counts[day] = count
        schedule_text = df.to_csv(index=False)
        counts_text = ", ".join(
            f"{d}: {c} classes" for d, c in day_counts.items()
        )
        return (
            f"Faculty {matched_fid} timetable:\n{schedule_text}\n"
            f"Class counts per day: {counts_text}"
        )
    except Exception:
        return None


def _fallback_answer(user_question: str, context: Dict) -> str:
    """
    Data-driven fallback when LLM is unavailable.
    Reads from output CSVs directly instead of hardcoded strings.
    """
    q = user_question.lower()

    try:
        # Section timetable query
        for section in SECTIONS:
            if f"section {section.lower()}" in q or \
               f"section {section}" in q:
                path = resolve_output_path(
                    f"section_{section}_timetable.csv"
                )
                if path.exists():
                    return path.read_text(encoding="utf-8")

        # Faculty query by ID or name
        faculty_df = pd.read_csv(config.DATA_DIR / "faculty.csv")
        for _, row in faculty_df.iterrows():
            fid = str(row["faculty_id"]).strip()
            fname = str(row["name"]).strip()
            if fid.lower() in q or fname.lower() in q:
                path = resolve_output_path(
                    f"faculty_{fid}_timetable.csv"
                )
                if path.exists():
                    return path.read_text(encoding="utf-8")

        # Default: return summary report
        path = resolve_output_path("summary_report.txt")
        if path.exists():
            return path.read_text(encoding="utf-8")

    except Exception as e:
        return f"Timetable data temporarily unavailable. Error: {e}"

    return "Could not find relevant timetable information for your query."


def explain(user_question: str,
            history: Optional[List[Tuple[str, str]]] = None) -> str:
    """Answer a timetable question using Groq LLM with safe retry."""
    context = setup_context()
    llm = _get_llm(context)
    if llm is None:
        return _fallback_answer(user_question, context)

    # Direct data lookup — inject actual CSV into prompt
    direct_context = _try_direct_lookup(user_question)

    # Build system prompt with optional direct data
    system_prompt = context["system_prompt"]
    if direct_context:
        system_prompt += f"\n\nDIRECT TIMETABLE DATA:\n{direct_context}"

    history_text = _history_block(history)
    full_prompt = (
        f"{system_prompt}\n\n"
        f"{history_text}\n\n"
        f"User question: {user_question}"
    ).strip()

    try:
        response = safe_llm_call(llm, full_prompt)
        text = getattr(response, "content", None) or str(response)
        if text and text.strip():
            return text.strip()
    except RuntimeError as e:
        if "exhausted" in str(e).lower() or "rate" in str(e).lower():
            return (
                "⚠️ AI temporarily unavailable (rate limit).\n\n"
                + _fallback_answer(user_question, context)
            )
        raise
    except Exception:
        pass

    fallback = _fallback_answer(user_question, context)
    return fallback if fallback.strip() else "Could not generate response. Please try again."


def explain_with_rag(user_question: str,
                     history: Optional[List[Tuple[str, str]]] = None) -> str:
    """Use RAG to retrieve relevant timetable context, then call explain()."""
    try:
        from src.phase5.rag_indexer import retrieve
        relevant = retrieve(user_question, k=5)
        if relevant:
            context_lines = [r["text"] for r in relevant]
            citations = list({r["source"] for r in relevant})
            augmented_question = (
                f"Context from timetable data:\n"
                + "\n".join(context_lines)
                + f"\n\nQuestion: {user_question}"
                + f"\n\nSources: {', '.join(citations)}"
            )
            return explain(augmented_question, history=history)
    except Exception:
        pass
    return explain(user_question, history=history)


def detect_issues() -> str:
    question = (
        "Based on the timetable data provided, identify any potential issues, "
        "imbalances, or improvements. Focus on: faculty load fairness, "
        "subject distribution across the week for Section A, and fatigue patterns."
    )
    return explain(question)


if __name__ == "__main__":
    setup_context(force_reload=True)
    print(detect_issues())
