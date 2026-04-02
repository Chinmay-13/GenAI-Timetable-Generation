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
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #374151;
    }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }

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
             "AI Assistant", "AI Agent"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        outputs_exist = resolve_output_path("summary_report.txt").exists()
        if outputs_exist:
            st.markdown(
                '<span class="badge badge-green">● Schedule Active</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="badge badge-amber">● No Schedule Yet</span>',
                unsafe_allow_html=True
            )

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

    # --- Metric cards row ---
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Sections", str(data.get("total_sections", 12)),
         "A through L"),
        ("Courses", "5", "Theory + Lab"),
        ("Theory Slots", str(data.get("total_theory_slots", 240)),
         "Across all sections"),
        ("Lab Slots", str(data.get("total_lab_slots", 48)),
         "12 pairs × 2 periods"),
    ]
    for col, (label, value, sub) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

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
            'Faculty Load Distribution</div>',
            unsafe_allow_html=True,
        )
        load_df = data.get("faculty_load_df")
        if load_df is not None and not load_df.empty:
            chart = (
                alt.Chart(load_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("name:N", sort=None, title="Faculty"),
                    y=alt.Y("total_hours:Q", title="Hours"),
                    color=alt.Color(
                        "status:N",
                        scale=alt.Scale(
                            domain=["OK", "OVERLOAD"],
                            range=["#2563EB", "#EF4444"],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        "faculty_id", "name", "designation",
                        "total_hours", "max_hours", "status",
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No faculty load data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Regenerate button ---
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    regen_col, _, _ = st.columns([1, 2, 2])
    with regen_col:
        if st.button("↺  Regenerate Timetable"):
            with st.spinner("Running scheduling pipeline..."):
                result = subprocess.run(
                    [sys.executable, "run_all.py"],
                    capture_output=True, text=True,
                    cwd=str(PROJECT_ROOT),
                )
            if result.returncode == 0:
                st.cache_data.clear()
                st.success("Timetable regenerated successfully.")
                st.rerun()
            else:
                st.error(f"Pipeline error:\n{result.stderr[:500]}")


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

    tab1, tab2 = st.tabs(["📋 By Section", "👤 By Faculty"])

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
            styled = df.style.applymap(
                _style_timetable_cell,
                subset=df.columns[1:],
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
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
            styled = df.style.applymap(
                _style_timetable_cell,
                subset=df.columns[1:],
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No faculty timetable found.")


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
            table_df = pd.DataFrame(rows)
            st.markdown(
                '<div class="card">'
                '<div class="card-header">Substitute Plan</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                table_df.style.apply(_substitute_row_style, axis=1),
                use_container_width=True,
                hide_index=True,
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
        'Powered by Gemini + RAG retrieval.</p>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Message",
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
    elif page == "AI Assistant":
        render_ai_assistant()
    elif page == "AI Agent":
        render_ai_agent()


if __name__ == "__main__":
    main()
