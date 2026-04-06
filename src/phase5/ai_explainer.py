"""
ai_explainer.py — Timetable AI explainer with RAG support.

Uses dynamic prompt builder (no hardcoded facts),
Groq LLM via safe wrapper with retry, and data-driven fallback.

RAG Improvement #2: query decomposition for multi-hop questions.
  _is_multihop()     — heuristic gate (no LLM cost for simple queries)
  _decompose_query() — splits complex question into 2-4 sub-queries via LLM
  explain_with_rag() — routes through decomposition when warranted
"""
from pathlib import Path
import sys
import json
import re
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

_CONTEXT_CACHE: dict = {}   # keyed by sem_id ("legacy" when sem_id is None)


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


def setup_context(force_reload: bool = False, sem_id: str = None) -> Dict:
    """Load all context needed for the AI layer."""
    global _CONTEXT_CACHE
    cache_key = sem_id or "legacy"
    if cache_key in _CONTEXT_CACHE and not force_reload:
        return _CONTEXT_CACHE[cache_key]

    if sem_id is not None:
        from config import get_sem_paths
        _paths = get_sem_paths(sem_id)
        _data_dir   = _paths.data_dir
        _output_dir = _paths.output_dir
        from config import resolve_output_path as _rop
        summary_path   = _output_dir / "summary_report.txt"
        section_a_path = _output_dir / "section_A_timetable.csv"
    else:
        _data_dir   = config.DATA_DIR
        _output_dir = config.OUTPUT_DIR
        summary_path   = resolve_output_path("summary_report.txt")
        section_a_path = resolve_output_path("section_A_timetable.csv")

    summary_report = _read_text(summary_path)
    section_a_csv = _read_text(section_a_path)
    section_a_df = pd.read_csv(section_a_path)

    context = {
        "summary_report": summary_report,
        "section_a_csv": section_a_csv,
        "section_a_csv_for_prompt": section_a_csv,
        "faculty_csv":      _read_text(_data_dir / "faculty.csv"),
        "assignments_csv":  _read_text(_data_dir / "assignments.csv"),
        "courses_csv":      _read_text(_data_dir / "courses.csv"),
        "faculty_load_table": _extract_faculty_load_table(summary_report),
        "section_a_df": section_a_df,
        "lab_df": pd.read_csv(_data_dir / "lab_allotment.csv"),
    }

    # Use dynamic prompt builder
    context["system_prompt"] = build_system_prompt(
        outputs_dir=_output_dir,
        data_dir=_data_dir,
        summary_text=summary_report,
        sem_id=sem_id,
    )
    _CONTEXT_CACHE[cache_key] = context
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


def _try_direct_lookup(user_question: str, sem_id: str = None) -> Optional[str]:
    """
    If query mentions a faculty name or ID, read their CSV directly
    and return it as context. Returns None if no faculty detected.
    sem_id is used to resolve the correct semester's data and output dirs.
    """
    q = user_question.lower()

    if sem_id is not None:
        from config import get_sem_paths as _gsp
        _data_dir = _gsp(sem_id).data_dir
        _out_dir  = _gsp(sem_id).output_dir
    else:
        _data_dir = config.DATA_DIR
        _out_dir  = config.OUTPUT_DIR

    try:
        faculty_df = pd.read_csv(_data_dir / "faculty.csv")
    except Exception:
        return None

    matched_fid = None
    for _, row in faculty_df.iterrows():
        name_lower = str(row["name"]).lower()
        fid = str(row["faculty_id"]).strip()
        if any(word in q for word in name_lower.split() if len(word) > 3):
            matched_fid = fid
            break
        if fid.lower() in q:
            matched_fid = fid
            break

    if not matched_fid:
        return None

    path = _out_dir / f"faculty_{matched_fid}_timetable.csv"
    if not path.exists():
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


def _fallback_answer(user_question: str, context: Dict, sem_id: str = None) -> str:
    """
    Data-driven fallback when LLM is unavailable.
    Reads from sem_id-aware output CSVs directly instead of hardcoded paths.
    """
    q = user_question.lower()

    if sem_id is not None:
        from config import get_sem_paths as _gsp
        _out_dir  = _gsp(sem_id).output_dir
        _data_dir = _gsp(sem_id).data_dir
    else:
        _out_dir  = config.OUTPUT_DIR
        _data_dir = config.DATA_DIR

    try:
        # Section timetable query
        for section in SECTIONS:
            if f"section {section.lower()}" in q or f"section {section}" in q:
                path = _out_dir / f"section_{section}_timetable.csv"
                if path.exists():
                    return path.read_text(encoding="utf-8")

        # Faculty query by ID or name
        faculty_df = pd.read_csv(_data_dir / "faculty.csv")
        for _, row in faculty_df.iterrows():
            fid   = str(row["faculty_id"]).strip()
            fname = str(row["name"]).strip()
            if fid.lower() in q or fname.lower() in q:
                path = _out_dir / f"faculty_{fid}_timetable.csv"
                if path.exists():
                    return path.read_text(encoding="utf-8")

        # Default: return summary report
        path = _out_dir / "summary_report.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")

    except Exception as e:
        return f"Timetable data temporarily unavailable. Error: {e}"

    return "Could not find relevant timetable information for your query."


def clear_context_cache(sem_id: str = None) -> None:
    """Clear the module-level context cache. sem_id=None clears everything."""
    if sem_id:
        _CONTEXT_CACHE.pop(sem_id, None)
    else:
        _CONTEXT_CACHE.clear()


def explain(user_question: str,
            history: Optional[List[Tuple[str, str]]] = None,
            sem_id: str = None) -> str:
    """Answer a timetable question using Groq LLM with safe retry."""
    context = setup_context(sem_id=sem_id)
    llm = _get_llm(context)
    if llm is None:
        return _fallback_answer(user_question, context, sem_id=sem_id)

    # Direct data lookup — inject actual CSV into prompt
    direct_context = _try_direct_lookup(user_question, sem_id=sem_id)

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
                + _fallback_answer(user_question, context, sem_id=sem_id)
            )
        raise
    except Exception:
        pass

    fallback = _fallback_answer(user_question, context, sem_id=sem_id)
    return fallback if fallback.strip() else "Could not generate response. Please try again."


# ── Multi-hop query decomposition ────────────────────────────────────────────

# Patterns that strongly suggest a multi-hop question
_MULTIHOP_WORDS = re.compile(
    r"\b(when|where|which days|compare|both|all sections|all faculty|\b)"
    r"(and|but)\b",
    re.IGNORECASE,
)
_FACULTY_PAT = re.compile(r"\bF\d{2}\b")
_SECTION_PAT = re.compile(r"\bsection\s+[A-La-l]\b", re.IGNORECASE)

_DECOMPOSE_SYSTEM = """\
You are a query decomposer for a university timetable retrieval system.
Given a question, split it into simple sub-queries (max 4).
Each sub-query must be answerable from a single timetable lookup.
Return ONLY a JSON array of strings. No explanation. No markdown.
Example input: "Which sections does F03 teach on days section A has labs?"
Example output: ["What days does section A have labs?", "What does F03 teach and on which days?"]
"""


def _is_multihop(question: str) -> bool:
    """
    Heuristic gate — returns True only when the question is genuinely
    multi-hop so we don't burn an LLM call on simple queries.

    Rules (ANY one sufficient):
    1. Contains both a section reference AND a faculty ID.
    2. Contains a multi-hop keyword (when/where/which days/compare/both/
       all sections/all faculty) AND a conjunction (and/but).
    3. Contains two or more distinct section references.
    4. Contains a faculty ID AND words like "which sections" or "all sections".
    """
    has_faculty  = bool(_FACULTY_PAT.search(question))
    sections     = _SECTION_PAT.findall(question)
    has_section  = len(sections) >= 1
    multi_section = len(set(s.upper() for s in sections)) >= 2

    ql = question.lower()

    # Rule 1: both section + faculty → almost always cross-entity
    if has_faculty and has_section:
        return True

    # Rule 2: multi-hop keywords + conjunction
    multihop_kw = any(kw in ql for kw in (
        "which days", "compare", "both", "all sections",
        "all faculty", "when does", "where does",
    ))
    has_conjunction = bool(re.search(r"\b(and|but)\b", ql))
    if multihop_kw and has_conjunction:
        return True

    # Rule 3: two distinct sections
    if multi_section:
        return True

    # Rule 4: faculty + "sections" (e.g. "which sections does F03 teach")
    if has_faculty and "section" in ql:
        return True

    return False


def _decompose_query(question: str, sem_id: str = None) -> list[str]:
    """
    Ask the LLM to split `question` into 2-4 simple sub-queries.

    Returns a list of sub-query strings, or raises on failure so the
    caller can fall back to single-query retrieval.
    """
    if not config.GROQ_API_KEY:
        raise RuntimeError("No GROQ_API_KEY — cannot decompose")

    llm = get_llm()   # temp llm instance; no context needed for decomposition
    prompt = (
        f"{_DECOMPOSE_SYSTEM}\n"
        f"Question: {question}\n"
        f"JSON array:"
    )
    response = safe_llm_call(llm, prompt)
    raw = getattr(response, "content", None) or str(response)

    # Strip markdown fences if the LLM wraps the JSON
    raw = re.sub(r"```[\w]*", "", raw).strip()

    sub_queries = json.loads(raw)          # raises json.JSONDecodeError on bad output
    if not isinstance(sub_queries, list) or not sub_queries:
        raise ValueError(f"Unexpected decomposition output: {raw!r}")

    # Sanitise: keep only non-empty strings, cap at 4
    sub_queries = [str(q).strip() for q in sub_queries if str(q).strip()][:4]
    return sub_queries


def explain_with_rag(user_question: str,
                     history: Optional[List[Tuple[str, str]]] = None,
                     sem_id: str = None) -> str:
    """
    Use RAG to retrieve relevant timetable context, then call explain().

    Routing logic
    -------------
    Simple query  → retrieve(k=5) → explain()          [unchanged path]
    Multi-hop     → _decompose_query() → retrieve(k=3)
                    per sub-query → merge + dedup → explain()
    Any failure   → fall back to single-query path, then bare explain()
    """
    import logging
    _log = logging.getLogger(__name__)

    try:
        from src.phase5.rag_indexer import retrieve

        if _is_multihop(user_question):
            # ── Multi-hop path ────────────────────────────────────────────────
            try:
                sub_queries = _decompose_query(user_question, sem_id=sem_id)
                _log.info("[RAG] Decomposed into %d sub-queries: %s",
                          len(sub_queries), sub_queries)

                seen_texts: set[str] = set()
                merged: list[dict] = []

                for sq in sub_queries:
                    for doc in retrieve(sq, k=3, sem_id=sem_id):
                        if doc["text"] not in seen_texts:
                            seen_texts.add(doc["text"])
                            merged.append(doc)

                # Cap at 10 docs to stay within prompt token budget
                merged = merged[:10]

                if merged:
                    context_lines = [r["text"] for r in merged]
                    citations = list({r["source"] for r in merged})
                    augmented_question = (
                        f"Context from timetable data:\n"
                        + "\n".join(context_lines)
                        + f"\n\nQuestion: {user_question}"
                        + f"\n\nSources: {', '.join(citations)}"
                    )
                    return explain(augmented_question, history=history,
                                  sem_id=sem_id)

            except Exception as decomp_err:
                # Graceful fallback — log and continue to single-query path
                _log.warning(
                    "[RAG] Decomposition failed (%s). "
                    "Falling back to single-query retrieval.",
                    decomp_err,
                )
                # fall through ↓

        # ── Simple (or fallback) path — identical to original behaviour ───────
        relevant = retrieve(user_question, k=5, sem_id=sem_id)
        if relevant:
            context_lines = [r["text"] for r in relevant]
            citations = list({r["source"] for r in relevant})
            augmented_question = (
                f"Context from timetable data:\n"
                + "\n".join(context_lines)
                + f"\n\nQuestion: {user_question}"
                + f"\n\nSources: {', '.join(citations)}"
            )
            return explain(augmented_question, history=history, sem_id=sem_id)

    except Exception:
        pass

    return explain(user_question, history=history, sem_id=sem_id)


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
