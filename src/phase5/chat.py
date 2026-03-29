from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.ai_explainer import setup_context, explain, detect_issues


BANNER = """═══════════════════════════════════════
TIMETABLE AI ASSISTANT
CSE Department — 3rd Semester (2024 Batch)
Type your question. Type 'quit' to exit.
═══════════════════════════════════════"""


def run_chat():
    setup_context(force_reload=True)
    print(BANNER)
    print("\nAI INITIAL ANALYSIS:")
    print(detect_issues())

    history = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Exiting chat.")
            break

        response = explain(user_input, history=history[-4:])
        if not response.strip():
            response = "Could not generate response. Please try again."
        print(f"AI: {response}")
        history.append((user_input, response))


if __name__ == "__main__":
    run_chat()
