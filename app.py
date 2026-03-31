from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.ai_explainer import explain
from src.phase5.substitute import build_load_snapshot, find_substitute, load_course_data
from src.phase5.swap import find_swap_slot

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SECTIONS = [chr(ord("A") + i) for i in range(12)]
FACULTY_IDS = [f"F{i:02d}" for i in range(1, 21)]
DAY_OPTIONS = {
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday",
    "Fri": "Friday",
}


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_summary_data() -> Dict[str, object]:
    summary_text = (OUTPUT_DIR / "summary_report.txt").read_text(encoding="utf-8")
    lines = [line.strip() for line in summary_text.splitlines() if line.strip()]

    total_sections = next(
        int(line.split(":")[-1].strip())
        for line in lines
        if line.startswith("Total sections scheduled:")
    )
    total_theory_slots = next(
        int(line.split(":")[-1].strip())
        for line in lines
        if line.startswith("Total theory slots placed:")
    )
    same_subject = next(
        int(line.split(":")[-1].strip())
        for line in lines
        if line.startswith("same_subject_same_day:")
    )
    back_to_back = next(
        int(line.split(":")[-1].strip())
        for line in lines
        if line.startswith("back_to_back_same_subject:")
    )

    load_df = pd.DataFrame(build_load_snapshot()).T.reset_index().rename(columns={"index": "faculty_id"})
    load_df["total_hours"] = load_df["total_hours"].astype(int)
    load_df["max_hours"] = load_df["max_hours"].astype(int)
    overload_count = int((load_df["status"] == "OVERLOAD").sum())
    quality_score = max(0.0, round(100 - ((same_subject + (2 * back_to_back) + (5 * overload_count)) / 2), 1))

    return {
        "summary_text": summary_text,
        "total_sections": total_sections,
        "total_theory_slots": total_theory_slots,
        "same_subject": same_subject,
        "back_to_back": back_to_back,
        "quality_score": quality_score,
        "faculty_load_df": load_df,
    }


def slot_style(value: object) -> str:
    text = str(value).strip()
    if text in {"", "----", "nan", "None"}:
        return "background-color: white; color: black;"
    if "LAB" in text:
        return "background-color: #f9c58b; color: black;"
    return "background-color: #d8ebff; color: black;"


def render_card(title: str, value: object) -> None:
    st.markdown(
        f"""
        <div style="padding:1rem;border:1px solid #d9e2ec;border-radius:0.8rem;background:#f7fafc;">
            <div style="font-size:0.9rem;color:#4a5568;">{title}</div>
            <div style="font-size:1.8rem;font-weight:700;color:#1a202c;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    summary = load_summary_data()
    courses_df = load_course_data()
    load_df = summary["faculty_load_df"].copy()

    st.title("Dashboard")
    cols = st.columns(4)
    with cols[0]:
        render_card("Sections", summary["total_sections"])
    with cols[1]:
        render_card("Courses", len(courses_df))
    with cols[2]:
        render_card("Theory Slots", summary["total_theory_slots"])
    with cols[3]:
        render_card("Quality Score", summary["quality_score"])

    st.subheader("Faculty Load")
    chart = (
        alt.Chart(load_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("name:N", sort=None, title="Faculty"),
            y=alt.Y("total_hours:Q", title="Hours"),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=["OK", "OVERLOAD"], range=["#2f855a", "#c53030"]),
                legend=None,
            ),
            tooltip=["faculty_id", "name", "designation", "total_hours", "max_hours", "status"],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)


def render_timetables() -> None:
    st.title("View Timetables")
    left, right = st.columns(2)
    with left:
        selected_section = st.selectbox("View Section", SECTIONS, key="view_section")
    with right:
        selected_faculty = st.selectbox("View Faculty", FACULTY_IDS, key="view_faculty")

    section_df = load_csv(str(OUTPUT_DIR / f"section_{selected_section}_timetable.csv"))
    faculty_df = load_csv(str(OUTPUT_DIR / f"faculty_{selected_faculty}_timetable.csv"))

    st.subheader(f"Section {selected_section}")
    st.dataframe(
        section_df.style.applymap(slot_style, subset=section_df.columns[1:]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(f"Faculty {selected_faculty}")
    st.dataframe(
        faculty_df.style.applymap(slot_style, subset=faculty_df.columns[1:]),
        use_container_width=True,
        hide_index=True,
    )


def build_substitute_table(result: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    substitutions = {
        (item["period"], item["course"], item["section"]): item for item in result["substitutions"]
    }
    unresolved = {
        (item["period"], item["course"], item["section"]): item for item in result["unresolved"]
    }

    for slot in result["original_slots"]:
        key = (slot["period"], slot["course"], slot["section"])
        if key in substitutions:
            item = substitutions[key]
            rows.append(
                {
                    "Period": slot["period"],
                    "Course": slot["course"],
                    "Section": slot["section"],
                    "Substitute": f"{item['substitute_name']} ({item['substitute_id']})",
                    "Reason": item["reason"],
                    "Status": "Assigned",
                }
            )
        else:
            rows.append(
                {
                    "Period": slot["period"],
                    "Course": slot["course"],
                    "Section": slot["section"],
                    "Substitute": "No substitute available",
                    "Reason": unresolved.get(key, {}).get("reason", "No substitute available"),
                    "Status": "Unresolved",
                }
            )
    return pd.DataFrame(rows)


def substitute_row_style(row: pd.Series) -> List[str]:
    if row["Status"] == "Unresolved":
        return ["background-color: #ffe3e3;" for _ in row]
    return ["" for _ in row]


def render_substitute_page() -> None:
    st.title("Faculty Absence Manager")

    faculty_id = st.text_input("Faculty ID", value="F04").strip().upper()
    absent_day_label = st.selectbox("Absent Day", list(DAY_OPTIONS.keys()), key="absent_day")
    if st.button("Find Substitute", use_container_width=True):
        try:
            result = find_substitute(faculty_id, DAY_OPTIONS[absent_day_label])
            table_df = build_substitute_table(result)
            st.dataframe(
                table_df.style.apply(substitute_row_style, axis=1),
                use_container_width=True,
                hide_index=True,
            )
            if result["unresolved"]:
                st.warning("Some slots are unresolved")
            else:
                st.success("All slots covered")
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Load Swap on Return")

    courses_df = load_course_data()
    return_faculty = st.text_input("Returning Faculty ID", value="F04").strip().upper()
    substitute_faculty = st.text_input("Substitute Faculty ID", value="F11").strip().upper()
    return_day_label = st.selectbox("Return Day", list(DAY_OPTIONS.keys()), key="return_day")
    course_choice = st.selectbox(
        "Course Code",
        courses_df["course_code"].tolist(),
        format_func=lambda code: f"{code}",
    )

    if st.button("Find Swap Slot", use_container_width=True):
        try:
            result = find_swap_slot(
                return_faculty,
                substitute_faculty,
                DAY_OPTIONS[return_day_label],
                course_choice,
            )
            if result["swap_found"]:
                st.success(
                    f"{result['swap_day']} {result['swap_period']} - {result['course']} (Section {result['section']})"
                )
            else:
                st.warning("No swap possible this week")

            metrics = st.columns(2)
            with metrics[0]:
                st.metric(
                    f"{result['faculty_a']} load",
                    f"{result['faculty_a_load_after']}h/{result['faculty_a_max']}h",
                    delta=f"before: {result['faculty_a_load_before']}h",
                )
            with metrics[1]:
                st.metric(
                    f"{result['faculty_b']} load",
                    f"{result['faculty_b_load_after']}h/{result['faculty_b_max']}h",
                    delta=f"before: {result['faculty_b_load_before']}h",
                )
            st.write(result["result"])
        except Exception as exc:
            st.error(str(exc))


def render_ai_page() -> None:
    st.title("AI Assistant")
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    question = st.text_input("Ask a question about the timetable")
    if st.button("Ask", use_container_width=True) and question.strip():
        answer = explain(question.strip(), history=st.session_state.qa_history[-5:])
        st.session_state.qa_history.append((question.strip(), answer))

    for question_text, answer_text in st.session_state.qa_history[-5:][::-1]:
        st.markdown(f"**Q:** {question_text}")
        st.write(answer_text)


def main() -> None:
    st.set_page_config(page_title="Timetable Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Pages",
        ["Dashboard", "View Timetables", "Substitute Finder", "AI Assistant"],
    )

    if page == "Dashboard":
        render_dashboard()
    elif page == "View Timetables":
        render_timetables()
    elif page == "Substitute Finder":
        render_substitute_page()
    else:
        render_ai_page()


if __name__ == "__main__":
    main()
