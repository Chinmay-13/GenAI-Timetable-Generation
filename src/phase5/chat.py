"""
chat.py — CLI chat interface with persistent session memory.

History is saved to outputs/chat_memory.json between runs.
Commands:
  quit     — exit the chat
  clear    — delete saved memory and start fresh
  history  — print last 5 exchanges from saved memory
"""

from pathlib import Path
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR
from src.phase5.ai_explainer import setup_context, explain_with_rag, detect_issues

MEMORY_FILE = OUTPUT_DIR / "chat_memory.json"
MAX_MEMORY  = 10   # keep last 10 exchanges on disk
MAX_CONTEXT = 4    # pass last 4 to the model per call

BANNER = """═══════════════════════════════════════
TIMETABLE AI ASSISTANT
CSE Department — 3rd Semester (2024 Batch)
Type your question. Type 'quit' to exit.
Commands: 'clear' (wipe memory) | 'history' (last 5)
═══════════════════════════════════════"""


def load_memory() -> list:
    """Load conversation history from disk."""
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            normalized = []
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    normalized.append((item[0], item[1]))
            return normalized
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def save_memory(history: list):
    """Persist last MAX_MEMORY exchanges to disk."""
    trimmed = history[-MAX_MEMORY:]
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2, ensure_ascii=False)


def format_history_for_prompt(history: list) -> list:
    """Return last MAX_CONTEXT exchanges as list of tuples for explain()."""
    return [(q, a) for q, a in history[-MAX_CONTEXT:]]


def run_chat():
    setup_context(force_reload=True)

    # Load persisted memory
    history = load_memory()
    if history:
        print(f"\nLoaded {len(history)} previous exchange(s) from memory.")
    else:
        print("\nNo previous chat history found. Starting fresh.")

    print(BANNER)
    print("\nAI INITIAL ANALYSIS:")
    print(detect_issues())

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            save_memory(history)
            break

        if not user_input:
            continue

        # ── Special commands ──────────────────────────────────────────────
        if user_input.lower() == "quit":
            print("Exiting chat.")
            save_memory(history)
            break

        if user_input.lower() == "clear":
            history = []
            try:
                MEMORY_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            print("Memory cleared.")
            continue

        if user_input.lower() == "history":
            recent = history[-5:]
            if not recent:
                print("No history yet.")
            else:
                print(f"\n--- Last {len(recent)} exchange(s) ---")
                for i, (q, a) in enumerate(recent, 1):
                    print(f"\n[{i}] You: {q}")
                    print(f"    AI : {a[:200]}{'...' if len(a) > 200 else ''}")
            continue

        # ── Normal question ───────────────────────────────────────────────
        context_history = format_history_for_prompt(history)
        response = explain_with_rag(user_input, history=context_history)
        if not response.strip():
            response = "Could not generate response. Please try again."

        print(f"AI: {response}")
        history.append((user_input, response))
        save_memory(history)


if __name__ == "__main__":
    run_chat()
