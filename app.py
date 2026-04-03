"""
app.py — Timetable OS: Modern academic dashboard.
Streamlit UI for the university timetable generation system.

Pages: Dashboard, Timetables, Substitute Finder, AI Assistant, AI Agent
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
import subprocess
import sys
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import resolve_output_path, SECTIONS, DATA_DIR, OUTPUT_DIR
from src.phase5.substitute import (
    build_load_snapshot, find_substitute, load_course_data, faculty_lookup
)
from src.phase5.swap import find_swap_slot

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ═══════════════════════════════════════════════════════════════════════════════
# CSS INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Cards */
    .card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }

    .card-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #1F2937;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #F3F4F6;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #1F2937;
        line-height: 1;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #9CA3AF;
        margin-top: 0.25rem;
    }

    /* Status badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-green  { background: #D1FAE5; color: #065F46; }
    .badge-blue   { background: #DBEAFE; color: #1E40AF; }
    .badge-amber  { background: #FEF3C7; color: #92400E; }
    .badge-red    { background: #FEE2E2; color: #991B1B; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #F9FAFB;
        border-right: 1px solid #E5E7EB;
        position: sticky;
        top: 0;
        height: 100vh;
        align-self: flex-start;
        z-index: 20;
    }
    [data-testid="stSidebar"] > div:first-child {
        height: 100vh;
        overflow-y: auto;
        overflow-x: hidden;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #374151;
    }

    /* Hide Streamlit chrome — keep the sidebar toggle visible */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* Hide header toolbar buttons but NOT the sidebar collapse control */
    header [data-testid="stToolbar"]        { visibility: hidden; }
    header [data-testid="stDecoration"]     { display: none; }
    header [data-testid="stStatusWidget"]   { visibility: hidden; }

    /* Always show the sidebar open/close arrow — override any inherited hide */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
    }
    /* Style the arrow button so it stands out on a plain background */
    [data-testid="collapsedControl"] button {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 50% !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
        color: #374151 !important;
        width: 2rem !important;
        height: 2rem !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.875rem;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        color: #374151;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #2563EB;
        color: #2563EB;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
    }

    /* Chat messages */
    .chat-user {
        background: #EFF6FF;
        border-radius: 12px 12px 4px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #1E40AF;
        font-size: 0.9rem;
    }
    .chat-ai {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 12px 12px 12px 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #1F2937;
        font-size: 0.9rem;
        white-space: pre-wrap;
    }

    /* Agent ops log */
    .ops-entry {
        background: #F9FAFB;
        border-left: 3px solid #2563EB;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
        color: #374151;
    }
    .ops-entry.rollback { border-left-color: #F59E0B; }

    /* Sub result rows */
    .sub-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid #F3F4F6;
    }
    .sub-row:last-child { border-bottom: none; }
    /* Candidate cards */
    .cand-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 0.5rem;
    }
    .cand-card:hover {
        border-color: #2563EB;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.08);
    }
    .match-chip {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-left: 0.4rem;
    }
    .chip-course  { background: #D1FAE5; color: #065F46; }
    .chip-desig   { background: #DBEAFE; color: #1E40AF; }
    .chip-avail   { background: #FEF3C7; color: #92400E; }
    /* Example question chips */
    .eq-btn > button {
        background: #EFF6FF !important;
        color: #1E40AF !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 999px !important;
        font-size: 0.78rem !important;
        padding: 0.25rem 0.9rem !important;
        font-weight: 500 !important;
    }
    .eq-btn > button:hover {
        background: #DBEAFE !important;
        border-color: #2563EB !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 1.5rem 0;">
            <div style="font-family:'DM Serif Display',serif;
                        font-size:1.3rem; color:#1F2937;">
                📅 Timetable OS
            </div>
            <div style="font-size:0.7rem; color:#9CA3AF;
                        margin-top:0.2rem;">
                CSE · 3rd Semester · 12 Sections
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["Dashboard", "Timetables", "Substitute Finder",
             "Faculty Workload", "AI Assistant", "AI Agent"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        outputs_exist = resolve_output_path("summary_report.txt").exists()

        # Pipeline status
        if outputs_exist:
            try:
                import os as _os
                mtime = _os.path.getmtime(str(resolve_output_path("summary_report.txt")))
                from datetime import datetime as _dt
                last_run = _dt.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M")
            except Exception:
                last_run = "unknown"
            st.markdown(
                f'<span class="badge badge-green">● Schedule Active</span>'
                f'<div style="font-size:0.7rem;color:#6B7280;margin-top:0.3rem;">'
                f'Last run: {last_run}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="badge badge-amber">● No Schedule Yet</span>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # Regenerate button
        if st.button("↺ Regenerate Schedule", use_container_width=True,
                     key="sb_regen"):
            with st.spinner("Running pipeline..."):
                res = subprocess.run(
                    [sys.executable, "run_all.py"],
                    capture_output=True, text=True, cwd=str(PROJECT_ROOT),
                )
            if res.returncode == 0:
                st.cache_data.clear()
                st.success("Done! Refresh to see new data.")
                st.rerun()
            else:
                st.error(res.stderr[:400] or "Pipeline failed.")

        # Rebuild RAG button
        if st.button("🔍 Rebuild RAG Index", use_container_width=True,
                     key="sb_rag"):
            with st.spinner("Rebuilding RAG index..."):
                try:
                    from src.phase5.rag_indexer import build_index
                    result_rag = build_index()
                    if result_rag and result_rag[0] is not None:
                        st.success("RAG index rebuilt.")
                    else:
                        st.warning("RAG index not built (deps missing?).")
                except Exception as _e:
                    st.error(f"RAG error: {_e}")

        # Health check expander
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        with st.expander("🩺 System Health", expanded=False):
            try:
                from utils.health_check import check_system_health
                health = check_system_health()
                summary = health["summary"]
                total, passed, failed = (
                    summary["total"], summary["passed"], summary["failed"]
                )
                if failed == 0:
                    st.markdown(
                        f'<span class="badge badge-green">✓ {passed}/{total} OK</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span class="badge badge-red">✗ {failed} issue(s)</span>',
                        unsafe_allow_html=True,
                    )

                SECTIONS_MAP = [
                    ("input_csvs",   "Input CSVs"),
                    ("output_files", "Outputs"),
                    ("rag_index",    "RAG Index"),
                    ("environment",  "Environment"),
                    ("packages",     "Packages"),
                ]
                for key, label in SECTIONS_MAP:
                    items = health.get(key, {})
                    n_fail = sum(1 for r in items.values() if not r["ok"])
                    icon   = "🟢" if n_fail == 0 else "🔴"
                    detail = f"{icon} {label}"
                    if n_fail:
                        detail += f" ({n_fail} missing)"
                    st.caption(detail)
                    if n_fail:
                        for item_name, result in items.items():
                            if not result["ok"]:
                                st.caption(
                                    f"&nbsp;&nbsp;↳ `{item_name}` — {result['message']}",
                                )
            except Exception as _he:
                st.caption(f"Health check unavailable: {_he}")

    return page


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_summary_data() -> Dict[str, object]:
    """Load summary report and compute quality metrics."""
    summary_path = resolve_output_path("summary_report.txt")
    if not summary_path.exists():
        return {
            "summary_text": "",
            "total_sections": 0,
            "total_theory_slots": 0,
            "total_lab_slots": 0,
            "same_subject": 0,
            "back_to_back": 0,
            "quality_score": 0,
            "faculty_load_df": pd.DataFrame(),
        }

    summary_text = summary_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in summary_text.splitlines() if line.strip()]

    def _extract(prefix: str) -> int:
        for line in lines:
            if line.startswith(prefix):
                val = line.split(":")[-1].strip()
                # Handle "48 (12 pairs × 2 periods..." style
                return int(val.split()[0])
        return 0

    total_sections = _extract("Total sections scheduled:")
    total_theory_slots = _extract("Total theory slots placed:")
    total_lab_slots = _extract("Total lab slots placed:")
    same_subject = _extract("same_subject_same_day:")
    back_to_back = _extract("back_to_back_same_subject:")

    load_df = (
        pd.DataFrame(build_load_snapshot())
        .T.reset_index()
        .rename(columns={"index": "faculty_id"})
    )
    load_df["total_hours"] = load_df["total_hours"].astype(int)
    load_df["max_hours"] = load_df["max_hours"].astype(int)
    overload_count = int((load_df["status"] == "OVERLOAD").sum())
    quality_score = max(
        0.0,
        round(100 - ((same_subject + (2 * back_to_back) + (5 * overload_count)) / 2), 1),
    )

    return {
        "summary_text": summary_text,
        "total_sections": total_sections,
        "total_theory_slots": total_theory_slots,
        "total_lab_slots": total_lab_slots,
        "same_subject": same_subject,
        "back_to_back": back_to_back,
        "quality_score": quality_score,
        "faculty_load_df": load_df,
    }


@st.cache_data(show_spinner=False)
def load_faculty_list() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "faculty.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TIMETABLE STYLING
# ═══════════════════════════════════════════════════════════════════════════════

def _style_timetable_cell(val: object) -> str:
    """Color timetable cells: blue=theory, amber=lab, grey=empty."""
    v = str(val).strip()
    if "LAB" in v.upper():
        return "background-color: #FEF3C7; color: #92400E; font-weight: 500;"
    if v in ("", "----", "nan", "None"):
        return "background-color: #FAFAFA; color: #D1D5DB;"
    return "background-color: #EFF6FF; color: #1E40AF;"


def _style_room_cell(val: object) -> str:
    """Color room assignment cells: green=assigned, red=unassigned, grey=empty."""
    v = str(val).strip()
    if v == "ROOM_UNASSIGNED":
        return "background-color: #FEE2E2; color: #991B1B; font-weight: 500;"
    if v in ("", "nan", "None"):
        return "background-color: #FAFAFA; color: #D1D5DB;"
    return "background-color: #D1FAE5; color: #065F46; font-weight: 500;"


@st.cache_data(show_spinner=False)
def _load_room_assignment() -> pd.DataFrame:
    path = resolve_output_path("room_assignment.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _render_room_assignments() -> None:
    """Render the Room Assignments sub-page inside the Timetables tab."""
    df = _load_room_assignment()

    if df.empty:
        st.warning(
            "⚠ Room assignment data not found. "
            "Run the pipeline (`python run_all.py`) to generate room assignments."
        )
        return

    # Filter controls
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sections_all = ["All"] + sorted(df["Section"].unique().tolist())
        sel_section  = st.selectbox("Filter by Section", sections_all,
                                    key="ra_section")
    with fc2:
        days_all = ["All"] + list(DAYS)
        sel_day  = st.selectbox("Filter by Day", days_all, key="ra_day")
    with fc3:
        rooms_all = ["All"] + sorted(
            r for r in df["Room"].unique() if r != "ROOM_UNASSIGNED"
        ) + (["ROOM_UNASSIGNED"] if "ROOM_UNASSIGNED" in df["Room"].values else [])
        sel_room  = st.selectbox("Filter by Room", rooms_all, key="ra_room")

    filtered = df.copy()
    if sel_section != "All":
        filtered = filtered[filtered["Section"] == sel_section]
    if sel_day != "All":
        filtered = filtered[filtered["Day"] == sel_day]
    if sel_room != "All":
        filtered = filtered[filtered["Room"] == sel_room]

    # Summary badges
    total      = len(filtered)
    assigned   = int((filtered["Room"] != "ROOM_UNASSIGNED").sum())
    unassigned = total - assigned

    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        st.markdown(
            f'<span class="badge badge-blue">Total slots: {total}</span>',
            unsafe_allow_html=True,
        )
    with bcol2:
        st.markdown(
            f'<span class="badge badge-green">Assigned: {assigned}</span>',
            unsafe_allow_html=True,
        )
    with bcol3:
        badge_cls = "badge-red" if unassigned else "badge-green"
        st.markdown(
            f'<span class="badge {badge_cls}">Unassigned: {unassigned}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><div class="card-header">Room Assignments</div>',
        unsafe_allow_html=True,
    )
    styled = filtered.style.applymap(
        _style_room_cell, subset=["Room"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def render_dashboard() -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Serif Display\',serif; '
        'font-size:1.8rem; color:#1F2937; margin-bottom:0.25rem;">'
        'Department Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6B7280; font-size:0.9rem; margin-bottom:1.5rem;">'
        'CSE · 3rd Semester · Academic Timetable Overview</p>',
        unsafe_allow_html=True,
    )

    data = load_summary_data()

    # --- Last-updated timestamp ---
    summary_path = resolve_output_path("summary_report.txt")
    if summary_path.exists():
        import os as _os
        from datetime import datetime as _dt
        mtime = _os.path.getmtime(str(summary_path))
        st.caption(f"🕐 Schedule last generated: {_dt.fromtimestamp(mtime).strftime('%A, %d %b %Y at %H:%M')}")

    # --- st.metric quick-stats row ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sections",     str(data.get("total_sections",    12)),  "A through L")
    m2.metric("Theory Slots", str(data.get("total_theory_slots", 0)), "All sections")
    m3.metric("Lab Slots",    str(data.get("total_lab_slots",    0)), "Paired blocks")
    violations = data.get("same_subject", 0) + data.get("back_to_back", 0)
    m4.metric("Violations",   str(violations), "Lower is better",
              delta_color="inverse")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # --- Quality score + faculty load side by side ---
    left, right = st.columns([1, 2])

    with left:
        score = data.get("quality_score", 0)
        color = (
            "#10B981" if score >= 80
            else "#F59E0B" if score >= 60
            else "#EF4444"
        )
        violations_html = (
            f'<div style="font-size:0.75rem; color:#6B7280; margin-top:0.75rem;">'
            f'Same-day violations: <strong>{data.get("same_subject", 0)}</strong>'
            f'<br>Back-to-back: <strong>{data.get("back_to_back", 0)}</strong></div>'
        )
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="card-header">Schedule Quality</div>
            <div style="font-family:'DM Serif Display',serif;
                        font-size:3.5rem; color:{color};
                        line-height:1; margin: 1rem 0;">
                {score}
            </div>
            <div style="font-size:0.75rem; color:#6B7280;">out of 100</div>
            {violations_html}
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="card"><div class="card-header">'
            'Faculty Load Heatmap</div>',
            unsafe_allow_html=True,
        )
        load_df = data.get("faculty_load_df")
        if load_df is not None and not load_df.empty:
            # Build day-level heatmap from faculty CSVs
            period_cols = [f"P{p}" for p in range(1, 7)]
            heat_rows = []
            for _, frow in load_df.iterrows():
                fid = str(frow["faculty_id"])
                fname = str(frow.get("name", fid))
                fp = resolve_output_path(f"faculty_{fid}_timetable.csv")
                if not fp.exists():
                    continue
                try:
                    tdf = pd.read_csv(fp)
                except Exception:
                    continue
                for _, drow in tdf.iterrows():
                    day = str(drow["Day"]).strip()
                    cnt = sum(
                        1 for c in period_cols
                        if str(drow.get(c, "----")).strip()
                        not in ("----", "", "nan")
                    )
                    heat_rows.append({"Faculty": fname, "Day": day, "Periods": cnt})

            if heat_rows:
                heat_df = pd.DataFrame(heat_rows)
                heat_df = heat_df[heat_df["Day"].isin(DAYS)].copy()
                heat_df["Day"] = pd.Categorical(
                    heat_df["Day"], categories=DAYS, ordered=True
                )
                heatmap = (
                    alt.Chart(heat_df)
                    .mark_rect(cornerRadius=3)
                    .encode(
                        x=alt.X("Day:N", sort=DAYS, title="Day"),
                        y=alt.Y("Faculty:N", sort=None, title=""),
                        color=alt.Color(
                            "Periods:Q",
                            scale=alt.Scale(
                                domain=[0, 6],
                                scheme="blues",
                            ),
                            title="Periods",
                        ),
                        tooltip=["Faculty", "Day", "Periods"],
                    )
                    .properties(height=max(180, len(heat_df["Faculty"].unique()) * 24))
                )
                st.altair_chart(heatmap, use_container_width=True)
            else:
                st.info("No faculty timetable data available.")
        else:
            st.info("No faculty load data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Regen button hidden from dashboard now that sidebar has one
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TIMETABLES
# ═══════════════════════════════════════════════════════════════════════════════

def render_timetables() -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Serif Display\',serif; '
        'font-size:1.8rem; color:#1F2937; margin-bottom:1.5rem;">'
        'View Timetables</h1>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📋 By Section", "👤 By Faculty", "🏫 Room Assignments"])

    with tab1:
        section = st.selectbox(
            "Select section", SECTIONS,
            label_visibility="collapsed",
        )
        path = resolve_output_path(f"section_{section}_timetable.csv")
        if path.exists():
            df = pd.read_csv(path)
            st.markdown(
                '<div class="card">'
                f'<div class="card-header">Section {section} — Weekly Timetable</div>',
                unsafe_allow_html=True,
            )
            # Pivot: rows = Periods P1-P6, columns = Days Mon-Fri
            period_cols = [f"P{p}" for p in range(1, 7)]
            pivot: dict = {col: {} for col in period_cols}
            for _, row in df.iterrows():
                day = str(row["Day"]).strip()
                for col in period_cols:
                    cell = str(row.get(col, "----")).strip()
                    pivot[col][day] = "" if cell in ("----", "nan") else cell
            piv_df = pd.DataFrame(pivot).T
            ordered = [d for d in DAYS if d in piv_df.columns]
            piv_df = piv_df.reindex(columns=ordered)
            piv_df.index.name = "Period"
            piv_df = piv_df.reset_index()

            def _sec_cell(val):
                v = str(val).strip()
                if "LAB" in v.upper():
                    return "background-color:#DBEAFE;color:#1E40AF;font-weight:500;"
                if v in ("", "----", "nan"):
                    return "background-color:#F9FAFB;color:#D1D5DB;"
                return "background-color:#D1FAE5;color:#065F46;"

            st.dataframe(
                piv_df.style.applymap(_sec_cell, subset=ordered),
                use_container_width=True, hide_index=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No timetable found. Generate one from the Dashboard.")

    with tab2:
        faculty_df = load_faculty_list()
        options = [
            f"{r['faculty_id']} — {r['name']} ({r['designation']})"
            for _, r in faculty_df.iterrows()
        ]
        choice = st.selectbox(
            "Select faculty", options,
            label_visibility="collapsed",
        )
        fid = choice.split(" — ")[0]
        path = resolve_output_path(f"faculty_{fid}_timetable.csv")
        if path.exists():
            df = pd.read_csv(path)
            st.markdown(
                '<div class="card">'
                f'<div class="card-header">{choice}</div>',
                unsafe_allow_html=True,
            )
            # Pivot: rows = Periods, columns = Days
            period_cols = [f"P{p}" for p in range(1, 7)]
            pivot: dict = {col: {} for col in period_cols}
            for _, row in df.iterrows():
                day = str(row["Day"]).strip()
                for col in period_cols:
                    cell = str(row.get(col, "----")).strip()
                    pivot[col][day] = "" if cell in ("----", "nan") else cell
            piv_df = pd.DataFrame(pivot).T
            ordered = [d for d in DAYS if d in piv_df.columns]
            piv_df = piv_df.reindex(columns=ordered)
            piv_df.index.name = "Period"
            piv_df = piv_df.reset_index()

            st.dataframe(
                piv_df.style.applymap(_sec_cell, subset=ordered),
                use_container_width=True, hide_index=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No faculty timetable found.")

    with tab3:
        _render_room_assignments()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SUBSTITUTE FINDER
# ═══════════════════════════════════════════════════════════════════════════════

def _substitute_row_style(row: pd.Series) -> List[str]:
    if row.get("Status") == "Unresolved":
        return ["background-color: #FEE2E2; color: #991B1B;" for _ in row]
    return ["background-color: #F0FDF4; color: #065F46;" for _ in row]


def render_substitute() -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Serif Display\',serif; '
        'font-size:1.8rem; color:#1F2937; margin-bottom:1.5rem;">'
        'Substitute Finder</h1>',
        unsafe_allow_html=True,
    )

    faculty_df = load_faculty_list()
    options = [
        f"{r['faculty_id']} — {r['name']}"
        for _, r in faculty_df.iterrows()
    ]

    # Input card
    st.markdown(
        '<div class="card"><div class="card-header">Report Absence</div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        absent = st.selectbox("Absent Faculty", options)
    with col2:
        day = st.selectbox("Day", DAYS)
    with col3:
        st.markdown(
            "<div style='height:1.9rem'></div>", unsafe_allow_html=True
        )
        find = st.button(
            "🔍 Find Substitutes", type="primary",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if find:
        fid = absent.split(" — ")[0]
        with st.spinner("Finding available substitutes..."):
            try:
                result = find_substitute(fid, day)
            except Exception as exc:
                st.error(f"Error: {exc}")
                return

        # Build results table (reusing existing logic)
        rows: List[Dict[str, str]] = []
        subs_map = {
            (s["period"], s["course"], s["section"]): s
            for s in result.get("substitutions", [])
        }
        for slot in result.get("original_slots", []):
            key = (slot["period"], slot["course"], slot["section"])
            s = subs_map.get(key)
            if s:
                rows.append({
                    "Period": slot["period"],
                    "Course": slot["course"],
                    "Section": slot["section"],
                    "Substitute": f"{s['substitute_name']} ({s['substitute_id']})",
                    "Match": s.get("match_type", ""),
                    "Reason": s.get("reason", ""),
                    "Status": "Assigned",
                })
            else:
                rows.append({
                    "Period": slot["period"],
                    "Course": slot["course"],
                    "Section": slot["section"],
                    "Substitute": "—",
                    "Match": "",
                    "Reason": "No substitute available",
                    "Status": "Unresolved",
                })

        if rows:
            st.markdown(
                '<div class="card">'
                '<div class="card-header">Substitute Plan</div>',
                unsafe_allow_html=True,
            )
            unresolved = result.get("unresolved", [])
            if unresolved:
                st.markdown(
                    f'<span class="badge badge-amber">'
                    f'{len(unresolved)} unresolved slot(s)</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="badge badge-green">All slots covered</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

            # Group substitutions by period for card rendering
            subs_by_period = {}
            for s in result.get("substitutions", []):
                subs_by_period.setdefault(s["period"], []).append(s)

            for slot in result.get("original_slots", []):
                period  = slot["period"]
                course  = slot["course"]
                section = slot["section"]
                cands   = subs_by_period.get(period, [])

                st.markdown(
                    f'<div style="font-size:0.8rem;color:#6B7280;'
                    f'font-weight:600;margin:0.9rem 0 0.4rem 0;">'
                    f'{period} · {course} · Section {section}</div>',
                    unsafe_allow_html=True,
                )

                if not cands:
                    st.warning(f"No substitute available for {period} ({course}).")
                    continue

                # Show each candidate as a card in columns (max 3 per row)
                for i in range(0, len(cands), 3):
                    cols = st.columns(min(3, len(cands) - i))
                    for col, sub in zip(cols, cands[i:i+3]):
                        with col:
                            match_type = sub.get("match_type", "available")
                            chip_cls = (
                                "chip-course" if match_type == "same_course"
                                else "chip-desig" if match_type == "same_designation"
                                else "chip-avail"
                            )
                            chip_label = (
                                "Same course" if match_type == "same_course"
                                else "Same rank" if match_type == "same_designation"
                                else "Available"
                            )
                            load_str = sub.get("projected_load", "?")
                            st.markdown(f"""
                            <div class="cand-card">
                              <div style="font-weight:600;font-size:0.9rem;color:#1F2937;">
                                {sub['substitute_name']}
                                <span class="match-chip {chip_cls}">{chip_label}</span>
                              </div>
                              <div style="font-size:0.78rem;color:#6B7280;margin-top:0.2rem;">
                                {sub.get('designation','')}
                              </div>
                              <div style="font-size:0.78rem;color:#374151;margin-top:0.35rem;">
                                Load after: <strong>{load_str}</strong>
                              </div>
                              <div style="font-size:0.75rem;color:#9CA3AF;margin-top:0.2rem;">
                                {sub.get('reason','')}
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Confirm & Commit button
                            btn_key = f"commit_{period}_{sub['substitute_id']}"
                            if st.button(
                                f"✓ Commit {sub['substitute_id']}",
                                key=btn_key, use_container_width=True,
                            ):
                                try:
                                    from src.phase5.sync_manager import commit_schedule_change
                                    p_int = int(period.lstrip("P"))
                                    commit_result = commit_schedule_change({
                                        "section":          section,
                                        "day":              day,
                                        "period_start":     p_int,
                                        "period_end":       p_int,
                                        "original_faculty": fid,
                                        "new_faculty":      sub["substitute_id"],
                                        "change_type":      "substitute",
                                        "reason":           f"UI commit from Substitute Finder",
                                    })
                                    st.success(commit_result["message"])
                                    st.cache_data.clear()
                                except Exception as _ce:
                                    st.error(f"Commit failed: {_ce}")

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Faculty has no classes on this day.")

    # --- Swap section ---
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="card">'
        '<div class="card-header">Plan Load Swap on Return</div>',
        unsafe_allow_html=True,
    )
    courses_df = load_course_data()
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        ret_faculty = st.selectbox(
            "Returning Faculty", options, key="swap_ret"
        )
    with sc2:
        sub_faculty = st.selectbox(
            "Substitute Faculty", options, key="swap_sub"
        )
    with sc3:
        ret_day = st.selectbox("Return Day", DAYS, key="swap_day")
    with sc4:
        course_code = st.selectbox(
            "Course", courses_df["course_code"].tolist(), key="swap_course"
        )

    if st.button("Suggest Swap Slots"):
        rfid = ret_faculty.split(" — ")[0]
        sfid = sub_faculty.split(" — ")[0]
        try:
            swap = find_swap_slot(rfid, sfid, ret_day, course_code)
            if swap.get("swap_found"):
                st.success(
                    f"Swap found: {swap['swap_day']} {swap['swap_period']} "
                    f"— {swap['course']} (Section {swap['section']})"
                )
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric(
                        f"{swap['faculty_a']} load",
                        f"{swap['faculty_a_load_after']}h / {swap['faculty_a_max']}h",
                        delta=f"was {swap['faculty_a_load_before']}h",
                    )
                with mc2:
                    st.metric(
                        f"{swap['faculty_b']} load",
                        f"{swap['faculty_b_load_after']}h / {swap['faculty_b_max']}h",
                        delta=f"was {swap['faculty_b_load_before']}h",
                    )
            else:
                st.info("No swap possible this week. Substitute keeps the extra class.")
        except Exception as exc:
            st.error(f"Swap error: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════

def render_ai_assistant() -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Serif Display\',serif; '
        'font-size:1.8rem; color:#1F2937; margin-bottom:0.5rem;">'
        'AI Assistant</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6B7280; font-size:0.875rem; '
        'margin-bottom:1.5rem;">Ask anything about the timetable. '
        'Powered by Groq + RAG retrieval.</p>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_prefill" not in st.session_state:
        st.session_state.chat_prefill = ""

    # --- Example question chips ---
    example_qs = [
        "Who can substitute for F03 on Monday?",
        "Which rooms are free on Tuesday P2?",
        "Show faculty workload summary",
        "List all sections on Friday",
        "Who teaches DSA for Section A?",
        "Are there any schedule conflicts?",
    ]
    st.markdown(
        '<div style="font-size:0.8rem;color:#6B7280;'
        'margin-bottom:0.5rem;">💡 Try asking:</div>',
        unsafe_allow_html=True,
    )
    eq_cols = st.columns(len(example_qs))
    for ec, q in zip(eq_cols, example_qs):
        with ec:
            st.markdown('<div class="eq-btn">', unsafe_allow_html=True)
            if st.button(q, key=f"eq_{q[:20]}", use_container_width=True):
                st.session_state.chat_prefill = q
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Chat history display
    if st.session_state.chat_history:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-ai">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # Input form — pre-filled if a chip was clicked
    prefill = st.session_state.pop("chat_prefill", "") if "chat_prefill" in st.session_state else ""
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Message",
                value=prefill,
                placeholder="e.g. Who teaches DDCO for Section A?",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input.strip()}
        )
        with st.spinner("Thinking..."):
            from src.phase5.ai_explainer import explain_with_rag
            history_pairs = []
            for i in range(0, len(st.session_state.chat_history) - 1, 2):
                msgs = st.session_state.chat_history
                if i + 1 < len(msgs):
                    history_pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
            response = explain_with_rag(
                user_input.strip(), history=history_pairs[-5:]
            )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )
        st.rerun()

    # Clear button
    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def render_ai_agent() -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Serif Display\',serif; '
        'font-size:1.8rem; color:#1F2937; margin-bottom:0.5rem;">'
        'Autonomous Agent</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6B7280; font-size:0.875rem; '
        'margin-bottom:1.5rem;">'
        'The agent can find substitutes, commit assignments, '
        'log operations, and generate reports.</p>',
        unsafe_allow_html=True,
    )

    col_main, col_log = st.columns([3, 2])

    with col_main:
        # Absence input card
        st.markdown(
            '<div class="card">'
            '<div class="card-header">Report Absence to Agent</div>',
            unsafe_allow_html=True,
        )
        faculty_df = load_faculty_list()
        options = [
            f"{r['faculty_id']} — {r['name']}"
            for _, r in faculty_df.iterrows()
        ]
        absent = st.selectbox("Absent Faculty", options, key="ag_faculty")
        day = st.selectbox("Day", DAYS, key="ag_day")
        auto = st.checkbox(
            "Auto-commit (no confirmation required)", value=False
        )

        col_a, col_b = st.columns(2)
        with col_a:
            run = st.button(
                "▶ Run Agent", type="primary", use_container_width=True
            )
        with col_b:
            summarise = st.button(
                "📄 Generate Summary", use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Agent output
        if "agent_output" not in st.session_state:
            st.session_state.agent_output = None

        if run:
            fid = absent.split(" — ")[0]
            action = (
                "commit the assignment automatically."
                if auto
                else "show me the proposed assignment."
            )
            query = (
                f"Faculty {fid} is absent on {day}. "
                f"Find the best substitute and {action}"
            )
            with st.spinner("Agent is working..."):
                try:
                    from src.phase5.agent import create_timetable_agent
                    agent = create_timetable_agent()
                    if agent is None:
                        # Fallback to direct substitute finder
                        from src.phase5.ai_explainer import explain_with_rag
                        result = explain_with_rag(query)
                        st.session_state.agent_output = f"[Fallback] {result}"
                    else:
                        result = agent({"input": query})
                        st.session_state.agent_output = result.get(
                            "output", "No response from agent."
                        )
                except Exception as exc:
                    st.session_state.agent_output = f"Agent error: {exc}"

        if st.session_state.agent_output:
            st.markdown(
                '<div class="card">'
                '<div class="card-header">Agent Response</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="chat-ai">{st.session_state.agent_output}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if summarise:
            with st.spinner("Generating summary..."):
                from src.phase5.agent_ops import list_operations
                ops = list_operations(50)
            if ops:
                st.markdown(
                    '<div class="card">'
                    '<div class="card-header">Session Summary</div>',
                    unsafe_allow_html=True,
                )
                for op in ops:
                    badge = (
                        "badge-green"
                        if op["commit_result"] == "SUCCESS"
                        else "badge-red"
                    )
                    st.markdown(f"""
                    <div class="ops-entry">
                        <span class="badge {badge}">
                            {op['commit_result']}
                        </span>
                        <strong style="margin-left:0.5rem;">
                            {op['action'].upper()}
                        </strong>
                        — Section {op['section_id']}, {op['day']},
                        P{op['period_range'][0]}-P{op['period_range'][1]}
                        <br>
                        <span style="color:#6B7280;">
                            Absent: {op['absent_faculty']} →
                            Sub: {op['substitute_faculty']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No operations logged yet.")

    with col_log:
        # Live agent ops log
        st.markdown(
            '<div class="card">'
            '<div class="card-header">Agent Operations Log</div>',
            unsafe_allow_html=True,
        )
        from src.phase5.agent_ops import list_operations, rollback_operation
        ops = list_operations(10)
        if not ops:
            st.markdown(
                '<p style="color:#9CA3AF; font-size:0.85rem;">'
                'No operations yet.</p>',
                unsafe_allow_html=True,
            )
        else:
            for op in ops:
                badge = (
                    "badge-green"
                    if op["commit_result"] == "SUCCESS"
                    else "badge-red"
                )
                st.markdown(f"""
                <div class="ops-entry">
                    <div style="display:flex; justify-content:space-between;
                                margin-bottom:0.25rem;">
                        <span class="badge {badge}">
                            {op['action'].upper()}
                        </span>
                        <span style="font-size:0.7rem; color:#9CA3AF;">
                            {op['operation_id']}
                        </span>
                    </div>
                    <div style="font-size:0.8rem; color:#374151;">
                        Section <strong>{op['section_id']}</strong>
                        · {op['day']}
                        · P{op['period_range'][0]}-P{op['period_range'][1]}
                    </div>
                    <div style="font-size:0.75rem; color:#6B7280;
                                margin-top:0.2rem;">
                        {op['absent_faculty']} →
                        {op['substitute_faculty']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Rollback UI
        st.markdown(
            '<div style="margin-top:1rem; padding-top:1rem; '
            'border-top:1px solid #F3F4F6;">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.8rem; font-weight:600; '
            'color:#374151; margin-bottom:0.5rem;">Rollback</div>',
            unsafe_allow_html=True,
        )
        op_id = st.text_input(
            "Operation ID", placeholder="e.g. a3f2b1c4",
            label_visibility="collapsed",
        )
        if st.button("⏪ Rollback", use_container_width=True):
            if op_id.strip():
                msg = rollback_operation(op_id.strip())
                if "Rolled back" in msg:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning("Enter an operation ID first.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — FACULTY WORKLOAD
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_all_faculty_workloads() -> pd.DataFrame:
    """
    Read every faculty_*_timetable.csv and compute workload metrics.
    Returns a DataFrame with one row per faculty.
    """
    from config import MAX_HOURS as _MH, DATA_DIR as _DD
    period_cols = [f"P{p}" for p in range(1, 7)]
    lab_cols    = {"P5", "P6"}

    try:
        fac_meta = pd.read_csv(_DD / "faculty.csv")
    except Exception:
        return pd.DataFrame()

    rows = []
    for _, frow in fac_meta.iterrows():
        fid   = str(frow["faculty_id"]).strip()
        name  = str(frow["name"]).strip()
        desig = str(frow["designation"]).strip()
        max_h = _MH.get(desig, 16)

        path = resolve_output_path(f"faculty_{fid}_timetable.csv")
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        theory = 0
        lab    = 0
        day_counts: dict = {}

        for _, drow in df.iterrows():
            day = str(drow["Day"]).strip()
            cnt = 0
            for col in period_cols:
                cell = str(drow.get(col, "----")).strip()
                if cell in ("----", "", "nan"):
                    continue
                cnt += 1
                if col in lab_cols:
                    lab += 1
                else:
                    theory += 1
            if cnt > 0:
                day_counts[day] = cnt

        total    = theory + lab
        busiest  = max(day_counts, key=day_counts.get) if day_counts else "—"
        n_days   = len(day_counts)

        if total > max_h:
            status = "OVERLOAD"
        elif total >= max_h - 1:
            status = "NEAR"
        else:
            status = "OK"

        rows.append({
            "faculty_id":        fid,
            "name":              name,
            "designation":       desig,
            "max_hours":         max_h,
            "theory":            theory,
            "lab":               lab,
            "total":             total,
            "days_with_classes": n_days,
            "busiest_day":       busiest,
            "status":            status,
        })

    return pd.DataFrame(rows)


def _stylize_workload(row: pd.Series) -> list:
    s = row.get("status", "OK")
    if s == "OVERLOAD":
        bg = "background-color:#FEE2E2; color:#991B1B;"
    elif s == "NEAR":
        bg = "background-color:#FEF3C7; color:#92400E;"
    else:
        bg = "background-color:#D1FAE5; color:#065F46;"
    return [bg] * len(row)


def render_faculty_workload() -> None:
    st.markdown(
        '<h1 style="font-family:\'DM Serif Display\',serif; '
        'font-size:1.8rem; color:#1F2937; margin-bottom:0.25rem;">'
        'Faculty Workload</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6B7280; font-size:0.9rem; margin-bottom:1.5rem;">'
        'Weekly load analysis, overload alerts, and free-slot finder.</p>',
        unsafe_allow_html=True,
    )

    wl_df = _load_all_faculty_workloads()

    if wl_df.empty:
        st.info(
            "No faculty timetables found. "
            "Run `python run_all.py` first to generate the schedule."
        )
        return

    # ── SECTION 4 (top) — Overload Alerts ────────────────────────────────────
    overloads = wl_df[wl_df["status"] == "OVERLOAD"]
    if not overloads.empty:
        for _, r in overloads.iterrows():
            over_by = int(r["total"]) - int(r["max_hours"])
            st.warning(
                f"⚠ **{r['name']}** ({r['faculty_id']}) is **{over_by} period(s)** "
                f"over their cap ({r['max_hours']}h max for {r['designation']})."
            )
    else:
        st.success("✓ All faculty are within their weekly teaching caps.")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── SECTION 1 — Workload Summary Table ────────────────────────────────────
    st.markdown(
        '<div class="card"><div class="card-header">'
        '📊 Workload Summary</div>',
        unsafe_allow_html=True,
    )
    display_cols = [
        "faculty_id", "name", "designation",
        "theory", "lab", "total", "max_hours",
        "days_with_classes", "busiest_day", "status",
    ]
    st.dataframe(
        wl_df[display_cols].style.apply(_stylize_workload, axis=1),
        use_container_width=True, hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── SECTION 2 — Individual Faculty Deep Dive ──────────────────────────────
    st.markdown(
        '<div class="card"><div class="card-header">'
        '🔍 Individual Faculty Deep Dive</div>',
        unsafe_allow_html=True,
    )

    fac_opts = [
        f"{r['faculty_id']} — {r['name']} ({r['designation']})"
        for _, r in wl_df.iterrows()
    ]
    chosen = st.selectbox(
        "Select faculty", fac_opts,
        key="fw_faculty_pick", label_visibility="collapsed",
    )
    chosen_id  = chosen.split(" — ")[0]
    chosen_row = wl_df[wl_df["faculty_id"] == chosen_id].iloc[0]

    # Progress bar
    total_h = int(chosen_row["total"])
    max_h   = int(chosen_row["max_hours"])
    pct     = min(total_h / max_h, 1.0) if max_h else 0.0
    bar_clr = (
        "#EF4444" if chosen_row["status"] == "OVERLOAD"
        else "#F59E0B" if chosen_row["status"] == "NEAR"
        else "#10B981"
    )
    st.markdown(
        f'<div style="font-size:0.85rem; color:#6B7280; margin-bottom:0.3rem;">'
        f'Teaching load: <strong style="color:{bar_clr};">{total_h}h</strong>'
        f' of {max_h}h weekly max</div>',
        unsafe_allow_html=True,
    )
    st.progress(pct)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Weekly timetable grid: rows=periods, columns=days
    DAYS_ORDER  = list(DAYS)  # from config
    period_cols = [f"P{p}" for p in range(1, 7)]

    fac_path = resolve_output_path(f"faculty_{chosen_id}_timetable.csv")
    if fac_path.exists():
        raw_df = pd.read_csv(fac_path)

        # Build period × day pivot
        grid: dict = {col: {} for col in period_cols}
        for _, drow in raw_df.iterrows():
            day = str(drow["Day"]).strip()
            for col in period_cols:
                cell = str(drow.get(col, "----")).strip()
                grid[col][day] = (
                    "Free" if cell in ("----", "", "nan") else cell
                )

        grid_df = pd.DataFrame(grid).T
        ordered_days = [d for d in DAYS_ORDER if d in grid_df.columns]
        grid_df = grid_df.reindex(columns=ordered_days)
        grid_df.index.name = "Period"
        grid_df = grid_df.reset_index()

        def _grid_cell(val):
            v = str(val)
            if v == "Free":
                return "background-color:#FAFAFA; color:#D1D5DB;"
            if "LAB" in v.upper():
                return "background-color:#FEF3C7; color:#92400E; font-weight:500;"
            return "background-color:#EFF6FF; color:#1E40AF;"

        st.dataframe(
            grid_df.style.applymap(_grid_cell, subset=ordered_days),
            use_container_width=True, hide_index=True,
        )

        # Altair bar chart — periods per day
        chart_rows = []
        for _, drow in raw_df.iterrows():
            day = str(drow["Day"]).strip()
            cnt = sum(
                1 for col in period_cols
                if str(drow.get(col, "----")).strip() not in ("----", "", "nan")
            )
            chart_rows.append({"Day": day, "Periods": cnt})

        chart_df = pd.DataFrame(chart_rows)
        chart_df = chart_df[chart_df["Day"].isin(DAYS_ORDER)].copy()
        chart_df["Day"] = pd.Categorical(
            chart_df["Day"], categories=DAYS_ORDER, ordered=True
        )
        chart_df = chart_df.sort_values("Day")

        bar = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Day:N", sort=DAYS_ORDER, title="Day"),
                y=alt.Y(
                    "Periods:Q", title="Periods taught",
                    scale=alt.Scale(domain=[0, 6]),
                ),
                color=alt.condition(
                    alt.datum.Periods >= max_h // len(DAYS_ORDER) + 1,
                    alt.value("#EF4444"),
                    alt.value("#2563EB"),
                ),
                tooltip=["Day", "Periods"],
            )
            .properties(height=200, title=f"{chosen_id} — Periods per Day")
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("Timetable file not found for this faculty. Run the pipeline first.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── SECTION 3 — Free Slot Finder ─────────────────────────────────────────
    st.markdown(
        '<div class="card"><div class="card-header">'
        '🔎 Free Slot Finder</div>',
        unsafe_allow_html=True,
    )

    fs_c1, fs_c2, fs_c3 = st.columns([2, 1, 1])
    with fs_c1:
        sel_day = st.selectbox(
            "Day", DAYS, key="fw_day", label_visibility="collapsed"
        )
    with fs_c2:
        sel_period = st.selectbox(
            "Period", [f"P{p}" for p in range(1, 7)],
            key="fw_period", label_visibility="collapsed",
        )
    with fs_c3:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        find_clicked = st.button(
            "🔍 Find Free Faculty", type="primary",
            use_container_width=True, key="fw_find",
        )

    if find_clicked:
        free_rows = []
        for _, frow in wl_df.iterrows():
            fid  = str(frow["faculty_id"]).strip()
            fp   = resolve_output_path(f"faculty_{fid}_timetable.csv")
            if not fp.exists():
                continue
            try:
                tdf  = pd.read_csv(fp)
                drow = tdf[tdf["Day"].str.lower() == sel_day.lower()]
                if drow.empty:
                    continue
                cell = str(drow.iloc[0].get(sel_period, "----")).strip()
                if cell in ("----", "", "nan"):
                    free_rows.append({
                        "Faculty ID":     fid,
                        "Name":           frow["name"],
                        "Designation":    frow["designation"],
                        "Weekly Load":    f"{int(frow['total'])} / {int(frow['max_hours'])}h",
                        "Status":         frow["status"],
                    })
            except Exception:
                continue

        if free_rows:
            free_df = pd.DataFrame(free_rows)

            def _free_row_style(row):
                s = row.get("Status", "OK")
                bg = (
                    "#FEE2E2" if s == "OVERLOAD"
                    else "#FEF3C7" if s == "NEAR"
                    else "#D1FAE5"
                )
                return [
                    f"background-color:{bg};" if c == "Status" else ""
                    for c in row.index
                ]

            st.caption(
                f"{len(free_rows)} faculty free on "
                f"**{sel_day} {sel_period}**:"
            )
            st.dataframe(
                free_df.style.apply(_free_row_style, axis=1),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info(f"All faculty are teaching on {sel_day} {sel_period}.")

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Timetable OS",
        page_icon="📅",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Timetables":
        render_timetables()
    elif page == "Substitute Finder":
        render_substitute()
    elif page == "Faculty Workload":
        render_faculty_workload()
    elif page == "AI Assistant":
        render_ai_assistant()
    elif page == "AI Agent":
        render_ai_agent()


if __name__ == "__main__":
    main()
