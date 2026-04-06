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
import logging
import re
import time
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

LOGGER = logging.getLogger(__name__)
_CONTEXT_CACHE: dict = {}   # keyed by sem_id ("legacy" when sem_id is None)

SYSTEM_PROMPT = """You are a timetable query assistant for a CSE department.
Answer ONLY using the context provided. Never infer, guess, or fill gaps.

RULES (non-negotiable):
1. Max 4 sentences. Never repeat a fact already stated.
2. If context is incomplete, say: "The retrieved data is incomplete for this query."
   Do NOT guess the missing values.
3. For list queries (e.g. "who is free"), output a bullet list, nothing else.
4. For comparisons, use this format:
     - F01: X periods (Mon, Wed, Fri)
     - F02: Y periods (Tue, Thu)
     - Conclusion: [one sentence]
5. Never start a sentence with "However," or "To compare" or "We can see that".
6. For superlative queries (highest, lowest, most, least), first list all values briefly, then end with:
     "Highest: [name]" or "Lowest: [name]" on its own line.

Context:
{context}
"""
RAG_STOP_SEQUENCES = [
    "However,",
    "To compare",
    "We can see that",
    "Unfortunately",
]
_QUERY_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
_QUERY_PERIOD_PATTERN = re.compile(r"\bp([1-6])\b", re.IGNORECASE)
_FACULTY_ID_PATTERN = re.compile(r"\bF\d{2}\b", re.IGNORECASE)
_SUBSTITUTE_KEYWORDS = {
    "substitute", "replacement", "replace", "cover", "covering", "absent", "who can",
}
_FREE_LINE_PATTERN = re.compile(r"Free \(\d+\):\s*(.+?)\.$")
_DAY_LINE_PATTERN = re.compile(
    r"^\s*(Monday|Tuesday|Wednesday|Thursday|Friday):\s*(.*)$",
    re.IGNORECASE,
)


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


def _extract_query_day(query: str) -> Optional[str]:
    q = query.lower()
    for day in _QUERY_DAYS:
        if day in q:
            return day.title()
    return None


def _extract_query_faculty_id(query: str) -> Optional[str]:
    match = _FACULTY_ID_PATTERN.search(query)
    return match.group(0).upper() if match else None


def detect_query_intent(query: str) -> dict:
    q = query.lower()
    faculty_id = _extract_query_faculty_id(query)
    day = _extract_query_day(query)

    explicit_substitute = any(
        keyword in q for keyword in {"substitute", "replacement", "replace", "absent"}
    )
    cover_style_substitute = (
        any(keyword in q for keyword in {"cover", "covering", "who can"})
        and any(trigger in q for trigger in {"absent", "instead", "replacement", "substitute", "for f"})
    )

    if explicit_substitute or cover_style_substitute:
        return {
            "route": "substitute_finder",
            "faculty_id": faculty_id,
            "day": day,
        }

    if faculty_id and ("name of" in q or "designation of" in q or "who is" in q):
        return {"source_type": "faculty_profile"}

    has_day = any(day in q for day in _QUERY_DAYS)
    has_period = bool(_QUERY_PERIOD_PATTERN.search(q))

    if (
        ("room" in q or "classroom" in q or "lab" in q)
        and ("free" in q or "available" in q or "empty" in q)
        and has_day
    ):
        return {"source_type": "room_roster"}

    if "highest" in q or "most" in q or "lowest" in q or "least" in q:
        return {"source_type": "faculty_week_summary", "superlative": True}

    if ("free" in q or "available" in q or "not teaching" in q) and has_day:
        return {"source_type": "slot_roster" if has_period else "day_roster"}

    if any(word in q for word in ["workload", "total", "how many", "compare", "periods"]):
        return {"source_type": "faculty_week_summary"}

    return {}


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


def _invoke_rag_llm(prompt: str) -> str:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    from langchain_groq import ChatGroq

    models_to_try = [config.GROQ_MODEL, config.GROQ_MODEL_ALT]
    last_error = None

    for model_idx, model_name in enumerate(models_to_try):
        llm = ChatGroq(
            model=model_name,
            api_key=config.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=250,
        ).bind(stop=RAG_STOP_SEQUENCES)
        label = f"Groq/{model_name}"

        for attempt in range(3):
            try:
                response = llm.invoke(prompt)
                text = getattr(response, "content", None) or str(response)
                if text and text.strip():
                    return text.strip()
                raise RuntimeError(f"{label} returned an empty response.")
            except Exception as exc:  # noqa: BLE001
                err = str(exc).lower()
                is_rate = "429" in err or "rate" in err or "too many" in err
                is_overload = "503" in err or "overload" in err or "unavailable" in err
                is_invalid = ("401" in err or "unauthorized" in err) and "decommission" not in err
                is_not_found = (
                    "404" in err or "not found" in err or "does not exist" in err
                    or "decommissioned" in err or "model_decommissioned" in err
                )

                if is_invalid:
                    raise RuntimeError(
                        "GROQ_API_KEY is invalid or unauthorized. Check your key in .env"
                    ) from exc

                if is_not_found and model_idx < len(models_to_try) - 1:
                    LOGGER.warning("%s model not found; trying fallback model.", label)
                    last_error = exc
                    break

                if is_rate or is_overload:
                    if attempt < 2:
                        wait = 2.0 * (2 ** attempt)
                        LOGGER.warning(
                            "%s temporarily unavailable (attempt %d/3). Waiting %.1fs.",
                            label,
                            attempt + 1,
                            wait,
                        )
                        time.sleep(wait)
                        last_error = exc
                        continue
                    if model_idx < len(models_to_try) - 1:
                        LOGGER.warning("%s exhausted; trying fallback model.", label)
                        last_error = exc
                        break
                    raise RuntimeError(
                        f"All Groq models rate limited or unavailable. Error: {exc}"
                    ) from exc

                LOGGER.error("%s error: %s", label, exc)
                raise

    raise RuntimeError(f"All LLM options failed. Last error: {last_error}")


def _extract_query_faculty_ids(question: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\bF\d{2}\b", question.upper())))


def _load_faculty_name_map(sem_id: str = None) -> dict[str, str]:
    try:
        if sem_id is not None:
            from config import get_sem_paths
            faculty_path = get_sem_paths(sem_id).data_dir / "faculty.csv"
        else:
            faculty_path = config.DATA_DIR / "faculty.csv"
        faculty_df = pd.read_csv(faculty_path)
        return {
            str(row["faculty_id"]).strip(): str(row["name"]).strip()
            for _, row in faculty_df.iterrows()
        }
    except Exception:
        return {}


def _format_substitute_answer(result: dict) -> str:
    absent_name = result.get("absent_faculty_name", result.get("absent_faculty", "Unknown faculty"))
    absent_id = result.get("absent_faculty", "")
    absent_day = result.get("absent_day", "")

    lines = [f"Substitute plan for {absent_name} ({absent_id}) on {absent_day}:"]

    substitutions = result.get("substitutions", [])
    substitution_by_period = {item["period"]: item for item in substitutions}

    for slot in result.get("original_slots", []):
        period = slot.get("period", "?")
        course = slot.get("course", "?")
        section = slot.get("section", "?")
        chosen = substitution_by_period.get(period)
        if chosen:
            projected = chosen.get("projected_load")
            load_suffix = f", load {projected}" if projected else ""
            lines.append(
                f"- {period} Section {section} {course}: "
                f"{chosen.get('substitute_name', '?')} ({chosen.get('substitute_id', '?')}) "
                f"[{chosen.get('match_type', 'candidate')}{load_suffix}]"
            )
        else:
            lines.append(f"- {period} Section {section} {course}: No substitute available")

    unresolved = result.get("unresolved", [])
    if unresolved:
        unresolved_periods = ", ".join(item.get("period", "?") for item in unresolved)
        lines.append(f"Unresolved periods: {unresolved_periods}")

    return "\n".join(lines)


def _normalize_substitute_result(result: dict) -> list[dict]:
    docs = []
    substitutions = result.get("substitutions", [])
    substitution_by_period = {item["period"]: item for item in substitutions}

    for slot in result.get("original_slots", []):
        period = slot.get("period", "?")
        chosen = substitution_by_period.get(period)
        if chosen:
            text = (
                f"Substitute suggestion for {result.get('absent_faculty')} on {result.get('absent_day')} "
                f"{period}: Section {slot.get('section', '?')} {slot.get('course', '?')} -> "
                f"{chosen.get('substitute_id', '?')} ({chosen.get('substitute_name', '?')}); "
                f"match_type={chosen.get('match_type', '?')}; projected_load={chosen.get('projected_load', '?')}."
            )
        else:
            text = (
                f"Substitute suggestion for {result.get('absent_faculty')} on {result.get('absent_day')} "
                f"{period}: Section {slot.get('section', '?')} {slot.get('course', '?')} -> no substitute available."
            )

        docs.append({
            "source_type": "substitute",
            "text": text,
            "score": 0.0,
            "source": "substitute_finder",
            "metadata": {
                "faculty": result.get("absent_faculty"),
                "day": result.get("absent_day"),
                "period": period,
                "section": slot.get("section"),
            },
        })

    return docs


def _handle_substitute_query(intent: dict, sem_id: str = None) -> tuple[str, list[dict]]:
    faculty_id = intent.get("faculty_id")
    day = intent.get("day")

    if not faculty_id or not day:
        return (
            "Please specify the absent faculty ID and day, for example: "
            "\"Find a substitute for F01 on Tuesday.\"",
            [],
        )

    try:
        from src.phase5.substitute import find_substitute
        result = find_substitute(faculty_id, day, sem_id=sem_id)
        return _format_substitute_answer(result), _normalize_substitute_result(result)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[Substitute] Direct routing failed (%s).", exc)
        return f"Substitute finder error: {exc}", []


def _extract_free_faculty_ids(doc_text: str) -> list[str]:
    faculty_ids: list[str] = []
    for line in doc_text.splitlines():
        match = _FREE_LINE_PATTERN.search(line.strip())
        if not match:
            continue
        values = match.group(1).strip()
        if not values or values.lower() == "none":
            continue
        faculty_ids.extend(
            item.strip()
            for item in values.split(",")
            if item.strip()
        )
    return list(dict.fromkeys(faculty_ids))


def _extract_free_room_ids(doc_text: str) -> list[str]:
    rooms: list[str] = []
    for line in doc_text.splitlines():
        match = _FREE_LINE_PATTERN.search(line.strip())
        if not match:
            continue
        values = match.group(1).strip()
        if not values or values.lower() == "none":
            continue
        rooms.extend(
            item.strip()
            for item in values.split(",")
            if item.strip()
        )
    return list(dict.fromkeys(rooms))


def _extract_profile_field(doc_text: str, field_name: str) -> Optional[str]:
    match = re.search(rf"{re.escape(field_name)}:\s*([^.]*)\.", doc_text)
    return match.group(1).strip() if match else None


def _summarize_faculty_week(doc: dict) -> Optional[tuple[str, int, list[str]]]:
    faculty_id = doc.get("faculty")
    if not faculty_id:
        match = re.search(r"\bF\d{2}\b", doc.get("text", ""))
        faculty_id = match.group(0) if match else None
    if not faculty_id:
        return None

    total_periods = 0
    active_days: list[str] = []
    for line in doc.get("text", "").splitlines():
        match = _DAY_LINE_PATTERN.match(line)
        if not match:
            continue
        day_name, detail = match.groups()
        periods = re.findall(r"\bP\d\b", detail)
        if periods:
            total_periods += len(periods)
            active_days.append(day_name[:3].title())

    return faculty_id, total_periods, active_days


def _build_workload_comparison_answer(user_question: str,
                                      retrieved_docs: list[dict]) -> Optional[str]:
    summaries = {}
    for doc in retrieved_docs:
        summary = _summarize_faculty_week(doc)
        if summary:
            faculty_id, total_periods, active_days = summary
            summaries[faculty_id] = (total_periods, active_days)

    ordered_ids = _extract_query_faculty_ids(user_question)
    ordered_ids = [fid for fid in ordered_ids if fid in summaries]
    if len(ordered_ids) < 2:
        ordered_ids = sorted(summaries)[:2]
    if len(ordered_ids) < 2:
        return None

    first, second = ordered_ids[:2]
    first_total, first_days = summaries[first]
    second_total, second_days = summaries[second]

    if first_total == second_total:
        conclusion = f"{first} and {second} have the same workload."
    elif first_total > second_total:
        conclusion = f"{first} has {first_total - second_total} more periods than {second}."
    else:
        conclusion = f"{second} has {second_total - first_total} more periods than {first}."

    return "\n".join([
        f"- {first}: {first_total} periods ({', '.join(first_days) if first_days else 'none'})",
        f"- {second}: {second_total} periods ({', '.join(second_days) if second_days else 'none'})",
        f"- Conclusion: {conclusion}",
    ])


def _build_faculty_profile_answer(user_question: str,
                                  retrieved_docs: list[dict]) -> Optional[str]:
    if not retrieved_docs:
        return None

    doc = retrieved_docs[0]
    text = doc.get("text", "")
    faculty_id = doc.get("faculty") or doc.get("faculty_id") or _extract_query_faculty_id(user_question)
    name = _extract_profile_field(text, "Name")
    designation = _extract_profile_field(text, "Designation")

    q = user_question.lower()
    if "designation of" in q:
        if designation:
            return f"{faculty_id} is {designation}."
        return "The retrieved data is incomplete for this query."

    if "name of" in q:
        if name:
            return f"{faculty_id} is {name}."
        return "The retrieved data is incomplete for this query."

    if "who is" in q:
        if name and designation:
            return f"{faculty_id} is {name}, {designation}."
        if name:
            return f"{faculty_id} is {name}."
        return "The retrieved data is incomplete for this query."

    return None


def _build_workload_superlative_answer(user_question: str,
                                       retrieved_docs: list[dict],
                                       sem_id: str = None) -> Optional[str]:
    summaries = {}
    for doc in retrieved_docs:
        summary = _summarize_faculty_week(doc)
        if summary:
            faculty_id, total_periods, _active_days = summary
            summaries[faculty_id] = total_periods

    if not summaries:
        return None

    q = user_question.lower()
    is_lowest = "lowest" in q or "least" in q
    label = "Lowest" if is_lowest else "Highest"
    target_value = min(summaries.values()) if is_lowest else max(summaries.values())
    faculty_name_map = _load_faculty_name_map(sem_id=sem_id)

    sorted_items = sorted(
        summaries.items(),
        key=lambda item: (item[1], item[0]) if is_lowest else (-item[1], item[0]),
    )
    lines = []
    for faculty_id, total_periods in sorted_items:
        name = faculty_name_map.get(faculty_id, faculty_id)
        lines.append(f"- {name} ({faculty_id}): {total_periods} periods")

    winners = [
        f"{faculty_name_map.get(faculty_id, faculty_id)} ({faculty_id})"
        for faculty_id, total_periods in sorted_items
        if total_periods == target_value
    ]
    lines.append(f"{label}: {', '.join(winners)}")
    return "\n".join(lines)


def _best_retrieved_snippet(user_question: str, retrieved_docs: list[dict]) -> str:
    q = user_question.lower()
    sec_match = re.search(r"\bsection\s+([A-Za-z])\b", user_question, re.IGNORECASE)
    target_section = sec_match.group(1).upper() if sec_match else None
    target_day = next((day.title() for day in _QUERY_DAYS if day in q), None)

    if target_section and target_day:
        for doc in retrieved_docs:
            if doc.get("section") == target_section and doc.get("day") == target_day:
                if doc.get("period") in (None, "", "all"):
                    return doc.get("text", "")

    for doc in retrieved_docs:
        text = doc.get("text", "")
        if text:
            return text
    return ""


def _fallback_rag_answer(user_question: str,
                         retrieved_docs: list[dict],
                         sem_id: str = None) -> str:
    if not retrieved_docs:
        return "The retrieved data is incomplete for this query."

    structured_answer = _build_structured_rag_answer(
        user_question,
        retrieved_docs,
        sem_id=sem_id,
    )
    if structured_answer:
        return structured_answer

    intent = detect_query_intent(user_question)
    q = user_question.lower()

    if intent.get("source_type") in {"slot_roster", "day_roster"}:
        free_ids: list[str] = []
        for doc in retrieved_docs:
            free_ids.extend(_extract_free_faculty_ids(doc.get("text", "")))
        free_ids = list(dict.fromkeys(free_ids))
        if free_ids:
            return "\n".join(f"- {fid}" for fid in free_ids)
        return "The retrieved data is incomplete for this query."

    if intent.get("source_type") == "faculty_week_summary" and "compare" in q:
        comparison = _build_workload_comparison_answer(user_question, retrieved_docs)
        if comparison:
            return comparison
        return "The retrieved data is incomplete for this query."

    snippet = _best_retrieved_snippet(user_question, retrieved_docs)
    return snippet if snippet else "The retrieved data is incomplete for this query."


def _build_structured_rag_answer(user_question: str,
                                 retrieved_docs: list[dict],
                                 sem_id: str = None) -> Optional[str]:
    intent = detect_query_intent(user_question)
    q = user_question.lower()

    if intent.get("source_type") in {"slot_roster", "day_roster"}:
        free_ids: list[str] = []
        for doc in retrieved_docs:
            free_ids.extend(_extract_free_faculty_ids(doc.get("text", "")))
        free_ids = list(dict.fromkeys(free_ids))
        if free_ids:
            return "\n".join(f"- {fid}" for fid in free_ids)
        return "The retrieved data is incomplete for this query."

    if intent.get("source_type") == "room_roster":
        free_rooms: list[str] = []
        for doc in retrieved_docs:
            free_rooms.extend(_extract_free_room_ids(doc.get("text", "")))
        free_rooms = list(dict.fromkeys(free_rooms))
        if free_rooms:
            return "\n".join(f"- {room}" for room in free_rooms)
        return "The retrieved data is incomplete for this query."

    if intent.get("source_type") == "faculty_profile":
        profile_answer = _build_faculty_profile_answer(user_question, retrieved_docs)
        if profile_answer:
            return profile_answer
        return "The retrieved data is incomplete for this query."

    if intent.get("source_type") == "faculty_week_summary":
        if intent.get("superlative"):
            superlative = _build_workload_superlative_answer(
                user_question,
                retrieved_docs,
                sem_id=sem_id,
            )
            if superlative:
                return superlative
            return "The retrieved data is incomplete for this query."

        if "compare" in q or len(_extract_query_faculty_ids(user_question)) >= 2:
            comparison = _build_workload_comparison_answer(user_question, retrieved_docs)
            if comparison:
                return comparison
            return "The retrieved data is incomplete for this query."

    return None


def _answer_rag_query(user_question: str,
                      retrieved_docs: list[dict],
                      history: Optional[List[Tuple[str, str]]] = None,
                      sem_id: str = None) -> str:
    structured_answer = _build_structured_rag_answer(
        user_question,
        retrieved_docs,
        sem_id=sem_id,
    )
    if structured_answer:
        return structured_answer

    context_text = "\n".join(
        doc.get("text", "").strip()
        for doc in retrieved_docs
        if doc.get("text", "").strip()
    )
    if not context_text:
        return "The retrieved data is incomplete for this query."

    history_text = _history_block(history)
    prompt = SYSTEM_PROMPT.format(context=context_text)
    if history_text:
        prompt = f"{prompt}\n\n{history_text}"
    prompt = f"{prompt}\n\nQuestion: {user_question}".strip()

    try:
        return _invoke_rag_llm(prompt)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[RAG] LLM answer generation failed (%s). Using fallback.", exc)
        return _fallback_rag_answer(user_question, retrieved_docs, sem_id=sem_id)


def _normalize_doc(doc: dict) -> dict:
    """
    Normalise a raw retrieved doc into the shape expected by the UI:
      source_type : coarse category (section / faculty / room / slot / other)
      text        : the chunk text
      score       : cross-encoder score if available, else 0.0
      metadata    : dict with section / day / faculty if present
    """
    src = doc.get("source", "")
    explicit_type = doc.get("source_type")
    if explicit_type in {"section", "section_slot", "section_day", "section_week_summary"}:
        source_type = "section"
    elif explicit_type in {"faculty", "faculty_slot", "faculty_day", "faculty_week_summary", "faculty_profile"}:
        source_type = "faculty"
    elif explicit_type in {"room", "room_availability", "room_roster"}:
        source_type = "room"
    elif explicit_type in {"slot_summary", "slot_roster", "day_roster"}:
        source_type = "slot"
    elif explicit_type == "substitute":
        source_type = "substitute"
    elif "section" in src:
        source_type = "section"
    elif "faculty" in src:
        source_type = "faculty"
    elif "room" in src:
        source_type = "room"
    elif src == "slot_summary":
        source_type = "slot"
    else:
        source_type = "other"

    score = float(doc.get("ce_score", 0.0))

    metadata = {}
    for key in ("section", "day", "faculty", "faculty_id", "period"):
        if key in doc:
            metadata[key] = doc[key]

    return {
        "source_type": source_type,
        "text":        doc.get("text", ""),
        "score":       score,
        "source":      src,
        "metadata":    metadata,
    }


def explain_with_rag(user_question: str,
                     history: Optional[List[Tuple[str, str]]] = None,
                     sem_id: str = None) -> tuple:
    """
    Use RAG to retrieve relevant timetable context, then call explain().

    Returns
    -------
    (answer_text: str, retrieved_docs: list[dict])
        retrieved_docs contains normalised doc dicts (source_type, text, score, metadata).
        Empty list when no RAG docs were used (fallback path).

    Routing logic
    -------------
    Simple query  → retrieve(k=5) → explain()          [unchanged path]
    Multi-hop     → _decompose_query() → retrieve(k=3)
                    per sub-query → merge + dedup → explain()
    Any failure   → fall back to single-query path, then bare explain()
    """
    try:
        from src.phase5.rag_indexer import retrieve
        intent_filter = detect_query_intent(user_question)

        if intent_filter.get("route") == "substitute_finder":
            return _handle_substitute_query(intent_filter, sem_id=sem_id)

        if intent_filter:
            relevant = retrieve(
                user_question,
                k=5,
                sem_id=sem_id,
                metadata_filter=intent_filter,
            )
            if relevant:
                answer = _answer_rag_query(
                    user_question,
                    relevant,
                    history=history,
                    sem_id=sem_id,
                )
                return answer, [_normalize_doc(d) for d in relevant]
            return "The retrieved data is incomplete for this query.", []

        if _is_multihop(user_question):
            # ── Multi-hop path ────────────────────────────────────────────────
            try:
                sub_queries = _decompose_query(user_question, sem_id=sem_id)
                LOGGER.info("[RAG] Decomposed into %d sub-queries: %s",
                            len(sub_queries), sub_queries)

                seen_texts: set[str] = set()
                merged: list[dict] = []

                for sq in sub_queries:
                    subquery_intent = detect_query_intent(sq)
                    for doc in retrieve(
                        sq,
                        k=3,
                        sem_id=sem_id,
                        metadata_filter=subquery_intent or None,
                    ):
                        if doc["text"] not in seen_texts:
                            seen_texts.add(doc["text"])
                            merged.append(doc)

                # Cap at 10 docs to stay within prompt token budget
                merged = merged[:10]

                if merged:
                    answer = _answer_rag_query(
                        user_question,
                        merged,
                        history=history,
                        sem_id=sem_id,
                    )
                    return answer, [_normalize_doc(d) for d in merged]

            except Exception as decomp_err:
                # Graceful fallback — log and continue to single-query path
                LOGGER.warning(
                    "[RAG] Decomposition failed (%s). "
                    "Falling back to single-query retrieval.",
                    decomp_err,
                )
                # fall through ↓

        # ── Simple (or fallback) path — identical to original behaviour ───────
        relevant = retrieve(user_question, k=5, sem_id=sem_id)
        if relevant:
            answer = _answer_rag_query(
                user_question,
                relevant,
                history=history,
                sem_id=sem_id,
            )
            return answer, [_normalize_doc(d) for d in relevant]

    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[RAG] Retrieval pipeline failed (%s). Falling back.", exc)

    # Bare fallback — no RAG docs available
    return explain(user_question, history=history, sem_id=sem_id), []


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
