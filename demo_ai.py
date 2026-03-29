import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.phase5.ai_explainer import setup_context, explain

QUESTIONS = [
    "Summarize the generated timetable in 3 sentences.",
    "Which faculty members have the most balanced workload?",
    "Is Section A's timetable good for students? Any issues?",
    "What are the key constraints this timetable satisfies?",
    "If a faculty member is absent on Monday, which sections are most affected and why?",
]


def run_demo():
    setup_context(force_reload=True)

    for i, question in enumerate(QUESTIONS, start=1):
        print("══════════════════════════════════")
        print(f"DEMO QUESTION {i}/5")
        print(f"Q: {question}")
        print("══════════════════════════════════")
        answer = explain(question)
        if not answer.strip():
            answer = "Could not generate response. Please try again."
        print(f"A: {answer}\n")

    print("══════════════════════════════════")
    print("DEMO COMPLETE")
    print("══════════════════════════════════")


if __name__ == "__main__":
    run_demo()
