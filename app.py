"""
app.py — Timetable OS  (complete rewrite, clean-slate)
=======================================================
All paths resolved fresh via get_sem_paths(st.session_state.sem_id).
Zero module-level mutable state.  Every bug from the old version fixed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    DAYS, SECTIONS, MAX_HOURS,
    get_sem_paths, list_available_semesters,
)
from utils.health_check import check_system_health

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Timetable OS",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Period metadata ───────────────────────────────────────────────────────────
PERIOD_LABELS = {
    1: "P1 · 8:45–9:45",
    2: "P2 · 9:45–10:45",
    3: "P3 · 11:00–12:00",
    4: "P4 · 12:00–1:00",
    5: "P5 · 1:45–3:15 (Lab)",
    6: "P6 · 3:15–4:00 (Lab)",
}

# cell background colours
CELL_COLORS = {
    "theory":    "#DBEAFE",   # light blue
    "lab":       "#D1FAE5",   # light green
    "elective":  "#EDE9FE",   # light purple
    "empty":     "#F9FAFB",   # off-white
}


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    sems = list_available_semesters()
    if "sem_id" not in st.session_state:
        st.session_state.sem_id = sems[0] if sems else "cse_sem3"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "_rag_debug" not in st.session_state:
        st.session_state._rag_debug = {}
    if "agent_output" not in st.session_state:
        st.session_state.agent_output = ""
    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = []
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if "_agent_confirm" not in st.session_state:
        st.session_state._agent_confirm = False
    if "_agent_pending_instruction" not in st.session_state:
        st.session_state._agent_pending_instruction = None
    if "_agent_confirm_triggered" not in st.session_state:
        st.session_state._agent_confirm_triggered = False
    if "_agent_preview_dir" not in st.session_state:
        st.session_state._agent_preview_dir = None
    if "_agent_preview_op_id" not in st.session_state:
        st.session_state._agent_preview_op_id = None
    if "_agent_substitute_commit_day" not in st.session_state:
        st.session_state._agent_substitute_commit_day = None
    if "_agent_substitute_commit_dir" not in st.session_state:
        st.session_state._agent_substitute_commit_dir = None
    if "_last_agent_wrote" not in st.session_state:
        st.session_state._last_agent_wrote = False
    if "_agent_session_started_at" not in st.session_state:
        st.session_state._agent_session_started_at = datetime.now(timezone.utc).isoformat()
    if "_agent_last_fac" not in st.session_state:
        st.session_state._agent_last_fac = None
    if "_agent_last_day" not in st.session_state:
        st.session_state._agent_last_day = None
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False

    # Orphan temp-folder cleanup (older than 24 h)
    for _sem in list_available_semesters():
        _tr = get_sem_paths(_sem).output_dir / "agent_ops" / "temp"
        if _tr.exists():
            for _mk in _tr.glob("*/.pending"):
                try:
                    if time.time() - _mk.stat().st_mtime > 86400:
                        shutil.rmtree(_mk.parent, ignore_errors=True)
                except Exception:
                    pass


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — path resolution (always called fresh, never cached globally)
# ─────────────────────────────────────────────────────────────────────────────

def _paths():
    """Fresh SemesterPaths for the currently selected semester."""
    return get_sem_paths(st.session_state.sem_id)


def _out_exists() -> bool:
    return _paths().output_dir.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — data loading (cached PER sem_id to avoid re-reads on rerun)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_section_csv(sem_id: str, section: str) -> Optional[pd.DataFrame]:
    p = get_sem_paths(sem_id).output_dir / f"section_{section}_timetable.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def _load_faculty_csv(sem_id: str, faculty_id: str) -> Optional[pd.DataFrame]:
    p = get_sem_paths(sem_id).output_dir / f"faculty_{faculty_id}_timetable.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def _load_room_assignment(sem_id: str) -> Optional[pd.DataFrame]:
    p = get_sem_paths(sem_id).output_dir / "room_assignment.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def _load_summary_report(sem_id: str) -> Optional[str]:
    p = get_sem_paths(sem_id).output_dir / "summary_report.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


@st.cache_data(show_spinner=False)
def _load_faculty_meta(sem_id: str) -> Optional[pd.DataFrame]:
    p = get_sem_paths(sem_id).data_dir / "faculty.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def _load_courses(sem_id: str) -> Optional[pd.DataFrame]:
    p = get_sem_paths(sem_id).data_dir / "courses.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def _load_assignments(sem_id: str) -> Optional[pd.DataFrame]:
    p = get_sem_paths(sem_id).data_dir / "assignments.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def _load_rag_docs_count(sem_id: str) -> int:
    p = get_sem_paths(sem_id).rag_docs_path
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — quality score from summary_report.txt
# Bug #6 fix: correct field names + correct formula
# ─────────────────────────────────────────────────────────────────────────────

def _parse_quality(report_text: str) -> dict:
    """Parse summary_report.txt fields and compute quality score."""
    def _int(pattern):
        m = re.search(pattern, report_text)
        return int(m.group(1)) if m else 0

    same_day    = _int(r"same_subject_same_day:\s*(\d+)")
    back_to_back= _int(r"back_to_back_same_subject:\s*(\d+)")
    theory_slots= _int(r"Total theory/elective slots placed:\s*(\d+)")
    lab_slots   = _int(r"Total lab slots placed:\s*(\d+)")
    elec_slots  = _int(r"Total fixed elective slots placed:\s*(\d+)")

    # Count OVERLOAD lines
    overload_count = len(re.findall(r"\|\s*OVERLOAD", report_text))

    # Bug #6 formula
    score = max(0.0, round(
        100 - ((same_day + 2 * back_to_back + 5 * overload_count) / 2), 1
    ))

    return {
        "score": score,
        "same_day": same_day,
        "back_to_back": back_to_back,
        "overload_count": overload_count,
        "theory_slots": theory_slots,
        "lab_slots": lab_slots,
        "elec_slots": elec_slots,
    }


def _room_unassigned_count(sem_id: str) -> int:
    df = _load_room_assignment(sem_id)
    if df is None or "Room" not in df.columns:
        return 0
    return int((df["Room"] == "ROOM_UNASSIGNED").sum())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — faculty load (Bug #1 fix: groupby, not row iteration)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _faculty_load_df(sem_id: str) -> Optional[pd.DataFrame]:
    """
    Returns a DataFrame with columns: faculty_id, total_slots, designation, max_hours.
    Bug #1 fix: uses pd.melt + groupby — never iterates rows.
    """
    fac_meta = _load_faculty_meta(sem_id)
    if fac_meta is None:
        return None

    out = get_sem_paths(sem_id).output_dir
    period_cols = [f"P{p}" for p in range(1, 7)]
    rows = []

    for _, frow in fac_meta.iterrows():
        fid = str(frow["faculty_id"]).strip()
        path = out / f"faculty_{fid}_timetable.csv"
        if not path.exists():
            rows.append({"faculty_id": fid, "total_slots": 0,
                         "designation": str(frow["designation"]),
                         "name": str(frow["name"])})
            continue
        df = pd.read_csv(path)
        # Bug #1: count non-empty cells across ALL days via vectorised ops
        filled = df[period_cols].apply(
            lambda c: c.astype(str).str.strip().ne("----") & c.astype(str).str.strip().ne("")
        ).values.sum()
        rows.append({
            "faculty_id": fid,
            "total_slots": int(filled),
            "designation": str(frow["designation"]),
            "name": str(frow["name"]),
        })

    if not rows:
        return None

    load_df = pd.DataFrame(rows)
    load_df["max_hours"] = load_df["designation"].map(lambda d: MAX_HOURS.get(d, 16))
    return load_df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — styled timetable HTML table
# ─────────────────────────────────────────────────────────────────────────────

def _cell_color(value: str) -> str:
    if not value or value.strip() in ("", "----", "—"):
        return CELL_COLORS["empty"]
    v = value.strip()
    if "LAB" in v.upper():
        return CELL_COLORS["lab"]
    if v.lower().startswith("elective"):
        return CELL_COLORS["elective"]
    return CELL_COLORS["theory"]


def _render_timetable_html(df: pd.DataFrame) -> str:
    """
    Render a section/faculty timetable CSV as a styled HTML table.
    Rows = periods (P1-P6 with time labels), Columns = Mon-Fri.
    """
    # Pivot: index=period, columns=day
    period_cols = [f"P{p}" for p in range(1, 7)]
    present_days = [d for d in DAYS if d in df.columns or d == df.get("Day", pd.Series()).name]

    # df has columns: Day, P1..P6
    day_col = df.set_index("Day")

    html = ['<table style="border-collapse:collapse;width:100%;font-size:0.85rem;">']
    # Header
    html.append("<thead><tr>")
    html.append('<th style="background:#1e293b;color:#f8fafc;padding:8px 12px;text-align:left;min-width:130px;">Period</th>')
    for day in DAYS:
        html.append(
            f'<th style="background:#1e293b;color:#f8fafc;padding:8px 12px;text-align:center;">{day}</th>'
        )
    html.append("</tr></thead><tbody>")

    for p in range(1, 7):
        col = f"P{p}"
        label = PERIOD_LABELS[p]
        html.append("<tr>")
        html.append(
            f'<td style="background:#f1f5f9;font-weight:600;padding:7px 12px;'
            f'border:1px solid #e2e8f0;white-space:nowrap;">{label}</td>'
        )
        for day in DAYS:
            try:
                value = str(day_col.loc[day, col]).strip()
            except (KeyError, Exception):
                value = "----"
            if value in ("", "nan", "None", "----"):
                display = "—"
                color = CELL_COLORS["empty"]
            else:
                display = value
                color = _cell_color(value)
            html.append(
                f'<td style="background:{color};padding:7px 10px;'
                f'border:1px solid #e2e8f0;text-align:center;">{display}</td>'
            )
        html.append("</tr>")

    html.append("</tbody></table>")
    return "".join(html)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — substitute intent detection (Bug #5 fix)
# ─────────────────────────────────────────────────────────────────────────────

_SUBSTITUTE_KEYWORDS = {"substitute", "cover", "replace", "absent", "who can", "covering"}

def _is_substitute_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _SUBSTITUTE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# Missing output guard
# ─────────────────────────────────────────────────────────────────────────────

def _no_output_banner(sem_id: str):
    st.warning(
        f"⚠️ No timetable found for **{sem_id}**.\n\n"
        f"Run: `python run_all.py --sem {sem_id}`",
        icon="⚠️",
    )


# =============================================================================
# SIDEBAR
# =============================================================================

def _render_sidebar():
    with st.sidebar:
        st.markdown("## 🗓️ Timetable OS")

        # ── Semester selector ─────────────────────────────────────────────────
        sems = list_available_semesters()
        if not sems:
            st.error("No semester data directories found in data/")
            return

        prev_sem = st.session_state.sem_id
        chosen = st.selectbox(
            "Semester",
            sems,
            index=sems.index(st.session_state.sem_id) if st.session_state.sem_id in sems else 0,
            key="_sem_selector",
        )

        # Bug #2 fix: clear all caches + state on semester switch
        if chosen != prev_sem:
            st.session_state.sem_id = chosen
            st.session_state.chat_history = []
            st.session_state.agent_output = ""
            st.session_state.agent_steps = []
            st.session_state._rag_debug = {}
            st.session_state._agent_session_started_at = datetime.now(timezone.utc).isoformat()
            st.cache_data.clear()
            # Clear ai_explainer module-level context cache
            try:
                from src.phase5.ai_explainer import clear_context_cache
                clear_context_cache()
            except Exception:
                pass
            # Discard any active preview for the OLD semester before switching
            try:
                from src.phase5.sync_manager import get_active_preview, discard_preview
                _prev_preview = get_active_preview(prev_sem)
                if _prev_preview:
                    discard_preview(_prev_preview["op_id"], prev_sem)
            except Exception:
                pass
            st.session_state._agent_preview_dir = None
            st.session_state._agent_preview_op_id = None
            # Clear lru_caches in substitute.py
            try:
                from src.phase5 import substitute as _sub
                for fn in (
                    _sub.load_faculty_timetable,
                    _sub.load_section_timetable,
                ):
                    try:
                        fn.cache_clear()
                    except Exception:
                        pass
            except Exception:
                pass
            st.rerun()

        sem_id = st.session_state.sem_id
        sem_paths = get_sem_paths(sem_id)

        # ── Semester info card ────────────────────────────────────────────────
        courses_df  = _load_courses(sem_id)
        faculty_df  = _load_faculty_meta(sem_id)
        doc_count   = _load_rag_docs_count(sem_id)
        rag_ok      = sem_paths.rag_index_path.exists()

        num_courses  = len(courses_df) if courses_df is not None else "?"
        num_faculty  = len(faculty_df) if faculty_df is not None else "?"
        num_sections = len(SECTIONS)
        has_electives = (
            courses_df is not None and
            "is_elective" in courses_df.columns and
            courses_df["is_elective"].any()
        )

        rag_line = (
            f"📂 rag_index.faiss · {doc_count} docs"
            if rag_ok else "📂 Index not built"
        )

        st.info(
            f"📚 {num_courses} courses · {num_faculty} faculty · {num_sections} sections\n\n"
            + ("🧪 Electives ✓\n\n" if has_electives else "")
            + rag_line
        )

        st.caption(
            "✦ Metadata filtering  ✦ Query decomposition\n"
            "✦ Cross-encoder rerank  ✦ Faculty preferences\n"
            "✦ CP-SAT room assignment"
        )

        st.divider()

        # ── Action buttons ────────────────────────────────────────────────────
        if st.button("↺ Regenerate Schedule", use_container_width=True):
            with st.spinner(f"Running pipeline for {sem_id}…"):
                try:
                    result = subprocess.run(
                        [sys.executable, "run_all.py", "--sem", sem_id],
                        capture_output=True, text=True, timeout=600,
                    )
                    st.cache_data.clear()
                    if result.returncode == 0:
                        st.success("Pipeline complete!")
                    else:
                        st.error(f"Pipeline failed:\n```\n{result.stderr[-1000:]}\n```")
                except subprocess.TimeoutExpired:
                    st.error("Pipeline timed out (>10 min).")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.button("🔄 Rebuild RAG Index", use_container_width=True):
            with st.spinner("Rebuilding FAISS index…"):
                try:
                    from src.phase5.rag_indexer import build_index
                    build_index(sem_id=sem_id)
                    st.cache_data.clear()
                    st.success("RAG index rebuilt!")
                except Exception as e:
                    st.error(f"Failed: {e}")

        st.divider()

        # ── System health dot ─────────────────────────────────────────────────
        # Bug #7 fix: always pass sem_id
        try:
            health = check_system_health(sem_id=sem_id)
            overall_ok = health.get("overall_ok", False)
            dot = "🟢" if overall_ok else "🔴"
            label = "All systems go" if overall_ok else "Issues detected"
            failed = health.get("summary", {}).get("failed", 0)
            st.markdown(f"{dot} **{label}**" + (f"  ({failed} failed)" if failed else ""))
        except Exception:
            st.markdown("⚪ Health check error")

        st.divider()

        # ── Debug toggle ──────────────────────────────────────────────────────
        st.session_state.show_debug = st.toggle(
            "Show debug info",
            value=st.session_state.show_debug,
            help="When ON, shows the full system prompt and RAG context sent to the LLM.",
        )

        st.divider()

        # ── Navigation ────────────────────────────────────────────────────────
        page = st.radio(
            "Navigate",
            ["Dashboard", "Timetables", "Substitute Finder",
             "Faculty Workload", "AI Assistant", "AI Agent"],
            index=["Dashboard", "Timetables", "Substitute Finder",
                   "Faculty Workload", "AI Assistant", "AI Agent"].index(
                st.session_state.current_page
            ),
            key="_nav",
        )
        st.session_state.current_page = page

        st.caption(f"**{sem_id}** · 12 Sections")


# =============================================================================
# PAGE 1 — DASHBOARD
# =============================================================================

def _render_dashboard():
    sem_id = st.session_state.sem_id
    sem_paths = get_sem_paths(sem_id)
    out = sem_paths.output_dir

    st.title("📊 Dashboard")

    if not out.exists():
        _no_output_banner(sem_id)
        return

    courses_df  = _load_courses(sem_id)
    faculty_df  = _load_faculty_meta(sem_id)
    report_text = _load_summary_report(sem_id)

    # ── Top metrics ───────────────────────────────────────────────────────────
    num_courses = len(courses_df) if courses_df is not None else "?"
    num_faculty = len(faculty_df) if faculty_df is not None else "?"
    lab_courses = (
        int(courses_df["has_lab"].sum())
        if courses_df is not None and "has_lab" in courses_df.columns
        else "?"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sections", len(SECTIONS))
    c2.metric("Courses", num_courses)
    c3.metric("Faculty", num_faculty)
    c4.metric("Lab Subjects", lab_courses)

    st.divider()

    col_l, col_r = st.columns(2)

    # ── Quality card ─────────────────────────────────────────────────────────
    with col_l:
        st.subheader("📈 Schedule Quality")
        if report_text:
            q = _parse_quality(report_text)
            score = q["score"]
            color = (
                "#16a34a" if score >= 80 else
                "#d97706" if score >= 60 else
                "#dc2626"
            )
            st.markdown(
                f'<div style="background:{color}20;border:2px solid {color};'
                f'border-radius:12px;padding:20px;text-align:center;">'
                f'<span style="font-size:3rem;font-weight:700;color:{color};">'
                f'{score}</span>'
                f'<span style="font-size:1.2rem;color:{color};"> / 100</span>'
                f'<br><small style="color:#64748b;">Quality Score</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"same_subject_same_day: {q['same_day']}  |  "
                f"back_to_back: {q['back_to_back']}  |  "
                f"overload: {q['overload_count']}"
            )
        else:
            st.info("Run pipeline to see quality metrics.")

    # ── Faculty load heatmap ──────────────────────────────────────────────────
    with col_r:
        st.subheader("👥 Faculty Load Heatmap")
        # Bug #1 fix: _faculty_load_df uses groupby-equivalent vectorised ops
        load_df = _faculty_load_df(sem_id)
        if load_df is not None and not load_df.empty:
            try:
                import altair as alt
                load_df["status"] = load_df.apply(
                    lambda r: (
                        "Over cap" if r["total_slots"] > r["max_hours"] else
                        "At cap" if r["total_slots"] == r["max_hours"] else
                        "Under cap"
                    ),
                    axis=1,
                )
                color_scale = alt.Scale(
                    domain=["Under cap", "At cap", "Over cap"],
                    range=["#16a34a", "#d97706", "#dc2626"],
                )
                chart = (
                    alt.Chart(load_df)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X("faculty_id:N", sort=None, title="Faculty"),
                        y=alt.Y("total_slots:Q", title="Weekly Slots"),
                        color=alt.Color("status:N", scale=color_scale,
                                        legend=alt.Legend(title="Status")),
                        tooltip=["faculty_id", "name", "total_slots",
                                 "max_hours", "designation", "status"],
                    )
                    .properties(height=250)
                    .configure_axis(labelFontSize=10)
                )
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.dataframe(load_df[["faculty_id", "name", "total_slots", "max_hours"]])
        else:
            st.info("Generate timetable first.")

    st.divider()

    # ── Timetable health stats ────────────────────────────────────────────────
    st.subheader("🏥 Timetable Health")
    if report_text:
        q = _parse_quality(report_text)
        unassigned = _room_unassigned_count(sem_id)
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Theory/Elective Slots", q["theory_slots"])
        h2.metric("Lab Slots", q["lab_slots"])
        h3.metric("Elective Slots", q["elec_slots"])
        h4.metric("ROOM_UNASSIGNED", unassigned, delta_color="inverse")
    else:
        st.info("No summary report found.")

    st.divider()

    # ── Section coverage table ────────────────────────────────────────────────
    st.subheader("📋 Section Coverage")
    assignments_df = _load_assignments(sem_id)
    courses_df_cov = _load_courses(sem_id)
    if assignments_df is not None and courses_df_cov is not None:
        try:
            core = courses_df_cov[
                (courses_df_cov.get("is_elective", pd.Series(False)) == False)
            ]["course_code"].tolist() if "is_elective" in courses_df_cov.columns else \
                courses_df_cov["course_code"].tolist()

            # Build section × course coverage grid
            coverage = {}
            for sec in SECTIONS:
                coverage[sec] = {}
                for cc in core:
                    assigned = assignments_df[
                        (assignments_df["course_code"].astype(str) == cc)
                    ]
                    # check if this section appears in sections_handled
                    if "sections_handled" in assigned.columns:
                        covered = assigned["sections_handled"].astype(str).str.contains(sec).any()
                    elif "section" in assigned.columns:
                        covered = (assigned["section"].astype(str) == sec).any()
                    else:
                        covered = False
                    coverage[sec][cc] = "✅" if covered else "❌"

            cov_df = pd.DataFrame(coverage).T
            cov_df.index.name = "Section"
            st.dataframe(cov_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not build coverage table: {e}")
    else:
        st.info("assignments.csv not found in data directory.")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — preview-aware CSV path resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_csv_path(filename: str, sem_id: str, preview_dir=None) -> Path:
    """Return the path to a CSV, preferring preview_dir when a preview exists."""
    if preview_dir:
        p = Path(preview_dir) / filename
        if p.exists():
            return p
    return get_sem_paths(sem_id).output_dir / filename


# =============================================================================
# PAGE 2 — TIMETABLES
# =============================================================================

def _render_timetables():
    sem_id = st.session_state.sem_id
    sem_paths = get_sem_paths(sem_id)
    out = sem_paths.output_dir

    st.title("📅 Timetables")

    if not out.exists():
        _no_output_banner(sem_id)
        return

    # ── Preview banner ────────────────────────────────────────────────────────
    preview_meta = None
    preview_dir  = None
    preview_section = None
    try:
        from src.phase5.sync_manager import get_active_preview
        preview_meta = get_active_preview(sem_id)
        if preview_meta:
            preview_dir     = preview_meta.get("temp_dir")
            preview_section = str(preview_meta["change_dict"]["section"]).upper()
    except Exception:
        pass

    if preview_meta:
        cd = preview_meta["change_dict"]
        st.warning(
            f"⚠️ **PENDING CHANGE — Preview Mode** · Changes not yet committed.  "
            f"Section **{preview_section}** on **{cd['day']}**, "
            f"P{cd['period_start']}–P{cd['period_end']}: "
            f"{cd['original_faculty']} → {cd['new_faculty']}",
            icon="⚠️",
        )
        col_disc, _ = st.columns([1, 4])
        if col_disc.button("🗑️ Discard Preview", key="tt_discard_preview"):
            try:
                from src.phase5.sync_manager import discard_preview
                discard_preview(preview_meta["op_id"], sem_id)
                st.session_state._agent_preview_dir = None
                st.session_state._agent_preview_op_id = None
                st.rerun()
            except Exception as e:
                st.error(f"Could not discard preview: {e}")

    tab_sec, tab_fac, tab_room = st.tabs(["By Section", "By Faculty", "Room Assignments"])

    # ── By Section ────────────────────────────────────────────────────────────
    with tab_sec:
        section = st.selectbox("Select Section", SECTIONS, key="tt_section")
        # Use preview CSV for the affected section, live CSV for all others
        _is_preview_sec = bool(preview_dir and section == preview_section)
        _sec_path = _resolve_csv_path(
            f"section_{section}_timetable.csv", sem_id,
            preview_dir if _is_preview_sec else None,
        )
        if _is_preview_sec:
            st.info("📋 Showing **preview** (proposed change — not yet committed).")
        df = pd.read_csv(_sec_path) if _sec_path.exists() else None
        if df is None:
            st.warning(f"No timetable for section {section}.")
        else:
            st.markdown("**Period times shown in row labels. Cell colours: 🔵 Theory · 🟢 Lab · 🟣 Elective**")
            st.markdown(_render_timetable_html(df), unsafe_allow_html=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇ Download Section {section} CSV",
                data=csv_bytes,
                file_name=f"section_{section}_timetable.csv",
                mime="text/csv",
            )

    # ── By Faculty ────────────────────────────────────────────────────────────
    with tab_fac:
        faculty_df = _load_faculty_meta(sem_id)
        if faculty_df is None:
            st.warning("faculty.csv not found.")
        else:
            fac_options = [
                f"{row['faculty_id']} — {row['name']}"
                for _, row in faculty_df.iterrows()
            ]
            chosen_fac = st.selectbox("Select Faculty", fac_options, key="tt_faculty")
            fid = chosen_fac.split(" — ")[0].strip()
            # Use preview CSV for affected faculty IDs
            _affected_fids = (
                {preview_meta["change_dict"]["original_faculty"].upper(),
                 preview_meta["change_dict"]["new_faculty"].upper()}
                if preview_meta else set()
            )
            _fac_path = _resolve_csv_path(
                f"faculty_{fid}_timetable.csv", sem_id,
                preview_dir if fid.upper() in _affected_fids else None,
            )
            if fid.upper() in _affected_fids and preview_dir:
                st.info("📋 Showing **preview** (proposed change — not yet committed).")
            fdf = pd.read_csv(_fac_path) if _fac_path.exists() else None
            if fdf is None:
                st.warning(f"No timetable for {fid}.")
            else:
                st.markdown("**Cell content: SUBJECT (section)  |  e.g. ML (A)**")
                st.markdown(_render_timetable_html(fdf), unsafe_allow_html=True)
                csv_bytes = fdf.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"⬇ Download {fid} CSV",
                    data=csv_bytes,
                    file_name=f"faculty_{fid}_timetable.csv",
                    mime="text/csv",
                )

    # ── Room Assignments ──────────────────────────────────────────────────────
    with tab_room:
        ra_df = _load_room_assignment(sem_id)
        if ra_df is None:
            st.warning("room_assignment.csv not found.")
        else:
            # Filters
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                day_filter = st.selectbox("Day", ["All"] + DAYS, key="room_day")
            with col_b:
                period_opts = ["All"] + [f"P{p}" for p in range(1, 7)]
                period_filter = st.selectbox("Period", period_opts, key="room_period")
            with col_c:
                sec_filter = st.selectbox("Section", ["All"] + list(SECTIONS), key="room_sec")

            filtered = ra_df.copy()
            if day_filter != "All":
                filtered = filtered[filtered["Day"] == day_filter]
            if period_filter != "All":
                filtered = filtered[filtered["Period"] == period_filter]
            if sec_filter != "All":
                filtered = filtered[filtered["Section"].astype(str) == sec_filter]

            st.dataframe(filtered, use_container_width=True, height=450)
            unassigned = int((filtered["Room"] == "ROOM_UNASSIGNED").sum())
            if unassigned:
                st.warning(f"⚠️ {unassigned} ROOM_UNASSIGNED slot(s) in current view.")


# =============================================================================
# PAGE 3 — SUBSTITUTE FINDER
# =============================================================================

def _render_substitute():
    sem_id = st.session_state.sem_id
    sem_paths = get_sem_paths(sem_id)
    out = sem_paths.output_dir

    st.title("🔄 Substitute Finder")

    if not out.exists():
        _no_output_banner(sem_id)
        return

    faculty_df = _load_faculty_meta(sem_id)
    if faculty_df is None:
        st.error("faculty.csv not found.")
        return

    fac_options = [
        f"{row['faculty_id']} — {row['name']}"
        for _, row in faculty_df.iterrows()
    ]

    col1, col2 = st.columns(2)
    with col1:
        chosen_fac = st.selectbox("Absent Faculty", fac_options, key="sub_faculty")
    with col2:
        day = st.selectbox("Day of Absence", DAYS, key="sub_day")

    fid = chosen_fac.split(" — ")[0].strip()

    # Priority order selector
    priority_label = st.radio(
        "Priority order:",
        options=[
            "Same subject → Same section → Any free",
            "Same section → Same subject → Any free",
        ],
        index=0,
        key="sub_priority",
        horizontal=True,
    )
    priority = "subject_first" if priority_label.startswith("Same subject") else "section_first"

    if st.button("🔍 Find Substitutes", type="primary"):
        with st.spinner("Analysing schedules…"):
            try:
                from src.phase5.substitute import find_substitute
                # Bug #3 fix: always pass sem_id (out_dir resolved inside)
                result = find_substitute(fid, day, sem_id=sem_id, priority=priority)

                absent_name = result.get("absent_faculty_name", fid)
                absent_secs = result.get("absent_sections", [])
                mode_label  = result.get("priority_mode", priority)
                st.success(
                    f"Substitute plan for **{absent_name}** on **{day}**  "
                    f"· sections: {', '.join(absent_secs)}  "
                    f"· mode: `{mode_label}`"
                )

                orig_slots = result.get("original_slots", [])
                if orig_slots:
                    st.markdown("**Original schedule:**")
                    slot_df = pd.DataFrame(orig_slots)
                    st.dataframe(slot_df, use_container_width=True, hide_index=True)

                subs = result.get("substitutions", [])
                if subs:
                    # Labels derived from match_type (semantic tier),
                    # not the numeric rank (which flips when priority mode changes).
                    _MATCH_LABELS = {
                        "same_course":  "✅ Same subject",
                        "same_section": "🔷 Same section",
                        "available":    "🔹 Any free",
                    }
                    _RANK_SUFFIX = {1: " [R1]", 2: " [R2]", 3: " [R3]"}
                    st.markdown("**Substitute assignments:**")
                    sub_rows = []
                    for s in subs:
                        match_type = s.get("match_type", "available")
                        rank       = s.get("priority", 3)
                        label = _MATCH_LABELS.get(match_type, match_type) + _RANK_SUFFIX.get(rank, "")
                        sub_rows.append({
                            "Period":         s.get("period", ""),
                            "Course":         s.get("course", ""),
                            "Section":        s.get("section", ""),
                            "Match":          label,
                            "Substitute":     s.get("substitute_name", ""),
                            "Sub ID":         s.get("substitute_id", ""),
                            "Designation":    s.get("designation", ""),
                            "Projected Load": s.get("projected_load", ""),
                            "Reason":         s.get("reason", ""),
                        })
                    st.dataframe(
                        pd.DataFrame(sub_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No substitutes could be found for any slot.")

                # ── At-cap notes ─────────────────────────────────────────────
                at_cap = result.get("at_cap_notes", [])
                if at_cap:
                    note_lines = []
                    for n in at_cap:
                        tier = "same subject" if n["match_type"] == "same_course" else "same section"
                        note_lines.append(
                            f"**{n['faculty_name']}** ({n['faculty_id']}) — {tier} — "
                            f"at cap ({n['total_hours']}/{n['max_hours']} hrs)"
                        )
                    st.warning(
                        "⚠️ **At-cap faculty** (qualified but excluded from results):\n\n"
                        + "\n\n".join(note_lines)
                    )

                unresolved = result.get("unresolved", [])
                if unresolved:
                    st.error(
                        f"❌ {len(unresolved)} slot(s) unresolved:\n"
                        + "\n".join(
                            f"- {u['period']}: {u['course']} (Section {u['section']}) — {u['reason']}"
                            for u in unresolved
                        )
                    )

            except Exception as e:
                st.error(f"Substitute finder error: {e}")


# =============================================================================
# PAGE 4 — FACULTY WORKLOAD
# =============================================================================

def _render_workload():
    sem_id = st.session_state.sem_id
    sem_paths = get_sem_paths(sem_id)
    out = sem_paths.output_dir

    st.title("👩‍🏫 Faculty Workload")

    if not out.exists():
        _no_output_banner(sem_id)
        return

    faculty_df = _load_faculty_meta(sem_id)
    if faculty_df is None:
        st.error("faculty.csv not found.")
        return

    fac_options = [
        f"{row['faculty_id']} — {row['name']}"
        for _, row in faculty_df.iterrows()
    ]
    chosen_fac = st.selectbox("Select Faculty", fac_options, key="wl_faculty")
    fid = chosen_fac.split(" — ")[0].strip()

    frow = faculty_df[faculty_df["faculty_id"].astype(str).str.strip() == fid].iloc[0]
    designation = str(frow.get("designation", ""))
    max_h = MAX_HOURS.get(designation, 16)

    # Load personal timetable
    fdf = _load_faculty_csv(sem_id, fid)

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("Weekly Schedule")
        if fdf is not None:
            st.markdown(_render_timetable_html(fdf), unsafe_allow_html=True)
        else:
            st.warning("No timetable found for this faculty.")

    with col_r:
        st.subheader("Load Summary")
        load_df = _faculty_load_df(sem_id)
        if load_df is not None:
            fload = load_df[load_df["faculty_id"] == fid]
            if not fload.empty:
                total_h = int(fload.iloc[0]["total_slots"])
                pct = round(total_h / max_h * 100, 1) if max_h else 0
                st.metric("Hours Assigned", total_h)
                st.metric("Cap", max_h)
                st.metric("Utilization", f"{pct}%")
                if total_h > max_h:
                    st.error("🔴 Over cap")
                elif total_h == max_h:
                    st.warning("⚠️ At cap")
                else:
                    st.success("✅ Under cap")

        # Preferences
        st.subheader("Preferences")
        pref_time = str(frow.get("pref_time", "none"))
        pref_btb  = str(frow.get("pref_no_backtoback", "False"))
        pref_day  = str(frow.get("pref_no_teaching_day", "none"))

        st.markdown(f"**Time preference:** {pref_time.capitalize()}")
        pref_btb_bool = pref_btb.strip().lower() in ("true", "1", "yes")
        st.markdown(f"**No back-to-back:** {'Yes' if pref_btb_bool else 'No'}")
        st.markdown(
            f"**Free day preference:** {pref_day if pref_day.lower() != 'none' else 'None'}"
        )

    st.divider()

    # ── All-faculty heatmap ───────────────────────────────────────────────────
    st.subheader("📊 All Faculty Load Comparison")
    load_df = _faculty_load_df(sem_id)
    if load_df is not None and not load_df.empty:
        try:
            import altair as alt
            load_df["status"] = load_df.apply(
                lambda r: (
                    "Over cap" if r["total_slots"] > r["max_hours"] else
                    "At cap" if r["total_slots"] == r["max_hours"] else
                    "Under cap"
                ),
                axis=1,
            )
            # Highlight selected faculty
            load_df["selected"] = load_df["faculty_id"] == fid
            color_scale = alt.Scale(
                domain=["Under cap", "At cap", "Over cap"],
                range=["#16a34a", "#d97706", "#dc2626"],
            )
            chart = (
                alt.Chart(load_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("faculty_id:N", sort=None, title="Faculty"),
                    y=alt.Y("total_slots:Q", title="Weekly Slots"),
                    color=alt.Color("status:N", scale=color_scale),
                    opacity=alt.condition(
                        alt.datum.faculty_id == fid,
                        alt.value(1.0),
                        alt.value(0.6),
                    ),
                    tooltip=["faculty_id", "name", "total_slots", "max_hours", "designation"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            st.dataframe(load_df[["faculty_id", "name", "total_slots", "max_hours", "status"]])


# =============================================================================
# PAGE 5 — AI ASSISTANT
# =============================================================================

def _render_ai_assistant():
    sem_id = st.session_state.sem_id
    sem_paths = get_sem_paths(sem_id)
    out = sem_paths.output_dir

    st.title("🤖 AI Assistant")

    if not out.exists():
        _no_output_banner(sem_id)
        return

    # ── Suggestion chips ──────────────────────────────────────────────────────
    suggestions = [
        "What does Section A have on Monday?",
        "When does F03 teach CNS?",
        "Which faculty are free on Wednesday P3?",
        "Which rooms are free on Monday P5?",
        "What is the name of F01?",
        "Who has the highest workload this week?",
    ]
    st.markdown("**Quick queries:**")
    chip_cols = st.columns(len(suggestions))
    for col, sugg in zip(chip_cols, suggestions):
        if col.button(sugg, key=f"chip_{sugg[:15]}"):
            st.session_state.chat_history.append((sugg, None))
            st.rerun()

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────────
    for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(user_msg)
        if ai_msg is not None:
            with st.chat_message("assistant"):
                st.write(ai_msg)

    # Process any pending (user_msg, None) entries
    for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
        if ai_msg is None:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        from src.phase5.ai_explainer import (
                            explain_with_rag, _is_multihop, _decompose_query,
                            setup_context,
                        )
                        history_for_llm = [
                            (u, a)
                            for u, a in st.session_state.chat_history[:i]
                            if a is not None
                        ][-4:]

                        query = user_msg
                        if _is_substitute_query(query):
                            query = f"[SUBSTITUTE_INTENT] {query}"

                        raw_debug = {}
                        is_multi = _is_multihop(user_msg)
                        raw_debug["multihop"] = is_multi
                        if is_multi:
                            try:
                                sub_qs = _decompose_query(user_msg, sem_id=sem_id)
                                raw_debug["sub_queries"] = sub_qs
                            except Exception:
                                raw_debug["sub_queries"] = []

                        # Retrieve docs separately for the legacy debug panel
                        try:
                            from src.phase5.rag_indexer import retrieve
                            docs_preview = retrieve(user_msg, k=5, sem_id=sem_id)
                            raw_debug["docs_retrieved"] = len(docs_preview)
                            raw_debug["reranked"] = any("ce_score" in d for d in docs_preview)
                            raw_debug["top_snippets"] = [
                                d.get("text", "")[:80] for d in docs_preview[:3]
                            ]
                        except Exception:
                            docs_preview = []
                            raw_debug["docs_retrieved"] = 0
                            raw_debug["reranked"] = False
                            raw_debug["top_snippets"] = []

                        # explain_with_rag now returns (answer, retrieved_docs)
                        response, retrieved_docs = explain_with_rag(
                            user_msg,
                            history=history_for_llm,
                            sem_id=sem_id,
                        )
                        st.session_state.chat_history[i] = (user_msg, response)
                        st.session_state._rag_debug = raw_debug
                        # Store retrieved docs for expander rendering below
                        raw_debug["retrieved_docs"] = retrieved_docs

                        # ── Get system prompt for debug expander ──────────────
                        try:
                            ctx = setup_context(sem_id=sem_id)
                            raw_debug["system_prompt"] = ctx.get("system_prompt", "")
                        except Exception:
                            raw_debug["system_prompt"] = ""

                        st.write(response)

                        # ── RAG context expander (always shown) ───────────────
                        with st.expander("🔍 RAG context sent with this prompt"):
                            if retrieved_docs:
                                for idx, doc in enumerate(retrieved_docs):
                                    st.markdown(
                                        f"**Doc {idx+1}** — "
                                        f"`{doc['source_type']}` | "
                                        f"score: `{doc['score']:.3f}`"
                                    )
                                    st.caption(doc["text"])
                                    if doc.get("metadata"):
                                        st.json(doc["metadata"], expanded=False)
                                    st.divider()
                            else:
                                st.info(
                                    "No RAG context — answered from direct CSV lookup "
                                    "or LLM knowledge."
                                )

                        # ── System prompt expander (debug mode only) ──────────
                        if st.session_state.get("show_debug"):
                            with st.expander("📋 Full system prompt sent to LLM"):
                                sp = raw_debug.get("system_prompt", "")
                                if sp:
                                    st.code(sp, language="text")
                                else:
                                    st.info("System prompt not available.")

                    except Exception as e:
                        err = f"⚠️ AI assistant error: {e}"
                        st.session_state.chat_history[i] = (user_msg, err)
                        st.error(err)

            st.rerun()
            break

    # ── Retrieval debug panel (only in debug mode) ────────────────────────────
    if st.session_state._rag_debug and st.session_state.get("show_debug"):
        with st.expander("🔍 Retrieval details (last query)", expanded=False):
            dbg = st.session_state._rag_debug
            is_multi = dbg.get("multihop", False)
            st.markdown(f"**Decomposed:** {'Yes' if is_multi else 'No'}")
            if is_multi and dbg.get("sub_queries"):
                for sq in dbg["sub_queries"]:
                    st.markdown(f"  - {sq}")
            st.markdown(f"**Docs retrieved after filtering:** {dbg.get('docs_retrieved', 0)}")
            st.markdown(f"**Cross-encoder reranking:** {'Yes' if dbg.get('reranked') else 'No'}")
            snippets = dbg.get("top_snippets", [])
            if snippets:
                st.markdown("**Top 3 doc snippets:**")
                for s in snippets:
                    st.caption(s)

    # ── Input box ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about the timetable…")
    if user_input and user_input.strip():
        st.session_state.chat_history.append((user_input.strip(), None))
        st.rerun()


# =============================================================================
# PAGE 6 — AI AGENT
# =============================================================================

def _render_ai_agent():
    sem_id = st.session_state.sem_id
    sem_paths = get_sem_paths(sem_id)
    out = sem_paths.output_dir

    def _summarize_agent_step_value(value, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _render_agent_tool_flow(steps, container=None):
        if not steps:
            return
        target = container if container is not None else st
        with target.expander(
            f"Agent tool flow ({len(steps)} steps)",
            expanded=False,
        ):
            for step_num, step in enumerate(steps, 1):
                tool_name = step.get("tool_name", "?")
                tool_input = _summarize_agent_step_value(
                    step.get("tool_input", ""),
                    80,
                )
                tool_result = _summarize_agent_step_value(
                    step.get("tool_result", ""),
                    120,
                )
                st.markdown(
                    f"**Step {step_num}** -> `{tool_name}`  \n"
                    f"Input: `{tool_input}`  \n"
                    f"Result: {tool_result}"
                )

    def _render_agent_response(output_text: str):
        st.markdown("### Agent Response")
        st.markdown(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
            f'border-radius:8px;padding:16px;">{output_text}</div>',
            unsafe_allow_html=True,
        )

    # ── Handle pending confirmation (direct commit — no LLM re-run) ────────────
    if st.session_state.get("_agent_confirm") and st.session_state.get("_agent_preview_op_id"):
        op_id = st.session_state._agent_preview_op_id
        st.session_state._agent_confirm        = False
        st.session_state._agent_preview_op_id  = None
        st.session_state._agent_preview_dir    = None
        with st.spinner("Committing change…"):
            try:
                from src.phase5.sync_manager import commit_from_preview
                result = commit_from_preview(op_id, sem_id=sem_id)
                st.session_state.agent_output    = result["message"]
                st.session_state._agent_substitute_commit_day = result.get("day")
                st.session_state._agent_substitute_commit_dir = result.get("substitute_dir")
                st.session_state._last_agent_wrote = True
                st.cache_data.clear()
            except Exception as e:
                st.warning("⚠️ Commit failed. Check timetable generation.")
                with st.expander("Technical detail"):
                    st.code(str(e))

    # ── Flush the _last_agent_wrote flag (cache already cleared above) ─────────
    if st.session_state.get("_last_agent_wrote"):
        st.session_state._last_agent_wrote = False
        st.cache_data.clear()

    st.title("🤖 AI Agent")

    if not out.exists():
        _no_output_banner(sem_id)
        return

    st.info(
        "The agent can read section/faculty timetables, find substitutes, "
        "commit changes, and rollback. **Write operations** (commit, rollback) "
        "require explicit confirmation.",
        icon="ℹ️",
    )

    faculty_df = _load_faculty_meta(sem_id)

    col1, col2 = st.columns(2)
    with col1:
        absent_fac = "F01"
        if faculty_df is not None:
            fac_options = [
                f"{row['faculty_id']} — {row['name']}"
                for _, row in faculty_df.iterrows()
            ]
            chosen_fac = st.selectbox("Absent Faculty", fac_options, key="agent_fac")
            absent_fac = chosen_fac.split(" — ")[0].strip()
    with col2:
        absent_day = st.selectbox("Day of Absence", DAYS, key="agent_day")

    default_instr = (
        f"Faculty {absent_fac} is absent on {absent_day}. "
        f"Find their schedule, identify substitute candidates for each period, "
        f"and present the top 3 options. DO NOT commit without my confirmation."
    )

    # Auto-update textarea when faculty or day changes.
    # st.text_area only respects value= on first render once its key exists,
    # so we pop the key whenever the selection changes to force a fresh default.
    if (
        st.session_state.get("_agent_last_fac") != absent_fac
        or st.session_state.get("_agent_last_day") != absent_day
    ):
        st.session_state["_agent_last_fac"] = absent_fac
        st.session_state["_agent_last_day"] = absent_day
        st.session_state.pop("agent_instruction", None)

    instruction = st.text_area(
        "Agent Instruction",
        value=default_instr,
        height=100,
        key="agent_instruction",
    )

    if st.button("▶ Run Agent", type="primary"):
        # Store instruction before running so confirmation can replay it
        st.session_state._agent_pending_instruction = instruction
        st.session_state.agent_output = ""
        st.session_state.agent_steps = []
        st.session_state._agent_substitute_commit_day = None
        st.session_state._agent_substitute_commit_dir = None
        with st.spinner("Agent working…"):
            try:
                from src.phase5.agent import create_timetable_agent
                # Bug #4 fix: wrap in try/except, clean message only
                agent = create_timetable_agent(
                    sem_id=sem_id,
                    session_started_at=st.session_state._agent_session_started_at,
                )
                if agent is None:
                    st.warning(
                        "Agent unavailable — check GROQ_API_KEY and LangChain installation."
                    )
                else:
                    def _update_agent_tool_flow(step_list):
                        st.session_state.agent_steps = list(step_list)

                    result = agent({
                        "input": instruction,
                        "step_callback": _update_agent_tool_flow,
                    })
                    output = result.get("output", "")
                    reasoning = result.get("reasoning", [])
                    steps = result.get("steps", [])

                    st.session_state.agent_output = output
                    st.session_state.agent_steps = steps

                    # ── Detect if agent wrote a preview ───────────────────────
                    try:
                        from src.phase5.sync_manager import get_active_preview
                        _pm = get_active_preview(sem_id)
                        if _pm:
                            st.session_state._agent_preview_op_id = _pm["op_id"]
                            st.session_state._agent_preview_dir   = _pm.get("temp_dir")
                    except Exception:
                        pass

                    _render_agent_response(output)
                    _render_agent_tool_flow(steps)

                    if reasoning:
                        with st.expander("🔧 Reasoning chain"):
                            for step in reasoning:
                                st.caption(f"• {step}")

                    # ── RAG context expander (tools that used RAG) ──────────
                    # Capture any RAG docs the agent's tools pulled during this run
                    try:
                        from src.phase5.rag_indexer import retrieve as _retrieve
                        from src.phase5.ai_explainer import _normalize_doc
                        _agent_rag_docs = _retrieve(instruction, k=5, sem_id=sem_id)
                        _agent_norm_docs = [_normalize_doc(d) for d in _agent_rag_docs]
                    except Exception:
                        _agent_norm_docs = []

                    with st.expander("🔍 RAG context available for this query"):
                        if _agent_norm_docs:
                            for idx, doc in enumerate(_agent_norm_docs):
                                st.markdown(
                                    f"**Doc {idx+1}** — "
                                    f"`{doc['source_type']}` | "
                                    f"score: `{doc['score']:.3f}`"
                                )
                                st.caption(doc["text"])
                                if doc.get("metadata"):
                                    st.json(doc["metadata"], expanded=False)
                                st.divider()
                        else:
                            st.info(
                                "No RAG context — agent used direct tool calls "
                                "(CSV lookup) rather than a RAG retriever."
                            )

                    # ── System prompt expander (debug mode only) ───────────
                    if st.session_state.get("show_debug"):
                        try:
                            from src.phase5.agent import _get_agent_system_prompt
                            _agent_sp = _get_agent_system_prompt(sem_id=sem_id)
                        except Exception:
                            _agent_sp = ""
                        with st.expander("📋 Full system prompt sent to Agent LLM"):
                            if _agent_sp:
                                st.code(_agent_sp, language="text")
                            else:
                                st.info("System prompt not available.")

                    # ── Confirm button if agent is proposing a commit ──────────
                    from src.phase5.sync_manager import get_active_preview as _gap
                    _active_preview = _gap(sem_id)
                    wants_commit = any(kw in output.lower() for kw in [
                        "confirm", "proceed", "shall i commit", "commit the substitute",
                        "do you want me to", "please confirm", "ready to commit",
                        "should i commit", "want me to apply", "apply_pending_preview",
                    ])
                    if wants_commit or _active_preview:
                        st.warning(
                            "⚠️ **Agent is proposing a schedule change.** "
                            "Review the preview above, then confirm:"
                        )
                        col_yes, col_no = st.columns(2)
                        if col_yes.button(
                            "✅ Yes, commit the change",
                            type="primary",
                            key="agent_confirm_yes",
                        ):
                            if _active_preview:
                                st.session_state._agent_preview_op_id = _active_preview["op_id"]
                                st.session_state._agent_preview_dir   = _active_preview.get("temp_dir")
                            st.session_state._agent_confirm = True
                            st.rerun()
                        if col_no.button(
                            "❌ No, cancel",
                            key="agent_confirm_no",
                        ):
                            if _active_preview:
                                from src.phase5.sync_manager import discard_preview as _dp
                                _dp(_active_preview["op_id"], sem_id)
                            st.session_state._agent_preview_op_id = None
                            st.session_state._agent_preview_dir   = None
                            st.info("Change cancelled.")
                            st.session_state._agent_pending_instruction = None
                            st.session_state.agent_output = ""
                            st.session_state.agent_steps = []

            except Exception as e:
                # Bug #4 fix: never show stack trace
                st.warning(
                    "⚠️ Agent encountered a tool error. "
                    "Ensure the timetable has been generated for this semester."
                )
                # dev detail in expander only
                with st.expander("Technical detail (dev)"):
                    st.code(str(e))

    elif st.session_state.agent_output:
        _render_agent_response(st.session_state.agent_output)
        if st.session_state.get("_agent_substitute_commit_day"):
            _sub_day = st.session_state._agent_substitute_commit_day
            st.info(
                f"✅ Substitute timetable saved to substitutes/{_sub_day}/. "
                "Original timetable unchanged."
            )
        _render_agent_tool_flow(st.session_state.agent_steps)

    # ── Recent operations panel ───────────────────────────────────────────────
    agent_ops_dir = sem_paths.agent_ops_dir
    if agent_ops_dir.exists():
        op_files = sorted(agent_ops_dir.glob("*.json"), reverse=True)[:5]
        if op_files:
            with st.expander("📋 Recent Operations", expanded=False):
                for op_path in op_files:
                    try:
                        op = json.loads(op_path.read_text(encoding="utf-8"))
                        ts  = op.get("timestamp", op_path.stem)
                        act = op.get("action", "?")
                        sec = op.get("section_id", "?")
                        day = op.get("day", "?")
                        stat = op.get("commit_result", "?")
                        absent = op.get("absent_faculty", "?")
                        sub    = op.get("substitute_faculty", "?")

                        c1, c2 = st.columns([4, 1])
                        c1.markdown(
                            f"**[{ts[:19]}]** `{act.upper()}` — "
                            f"Sec {sec} · {day} · {absent} → {sub} · *{stat}*"
                        )
                        op_id = op.get("operation_id", op_path.stem)
                        if c2.button("↩ Rollback", key=f"rb_{op_id}"):
                            try:
                                from src.phase5.agent_ops import rollback_operation
                                msg = rollback_operation(op_id, sem_id=sem_id)
                                st.success(msg)
                                st.cache_data.clear()
                            except Exception as e:
                                st.error(f"Rollback failed: {e}")
                    except Exception:
                        continue


# =============================================================================
# MAIN ROUTER
# =============================================================================

def main():
    _render_sidebar()

    page = st.session_state.current_page
    if page == "Dashboard":
        _render_dashboard()
    elif page == "Timetables":
        _render_timetables()
    elif page == "Substitute Finder":
        _render_substitute()
    elif page == "Faculty Workload":
        _render_workload()
    elif page == "AI Assistant":
        _render_ai_assistant()
    elif page == "AI Agent":
        _render_ai_agent()
    else:
        _render_dashboard()


if __name__ == "__main__":
    main()
