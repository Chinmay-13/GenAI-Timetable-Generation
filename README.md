# Timetable Generation System

AI-powered academic timetable generator for university CSE departments. Schedules theory and lab sessions for 12 sections across a 5-day week using constraint programming, then layers an AI assistant on top for natural-language Q&A, substitute finding, and autonomous change management.

---

## What It Does

- **Generates conflict-free timetables** for 12 sections (A–L) across 5 theory subjects and 2 lab subjects using Google OR-Tools CP-SAT
- **Assigns classrooms** inside the solver — hard constraints prevent double-booking; soft constraints prefer larger rooms
- **Answers questions** via a Groq LLM + FAISS RAG assistant with metadata filtering and cross-encoder reranking
- **Manages absences** through an LangChain agent with 16 tools, atomic commits, and rollback support

---

## Architecture

```
data/{sem}/                       Input CSVs
├── courses.csv
├── faculty.csv  (+ preferences)
├── assignments.csv
├── rooms.csv
└── lab_allotment.csv
        │
        ▼
┌─────────────────────────────────────────────┐
│              GENERATION PIPELINE            │
│                                             │
│  Phase 0 → Phase 1 → Phase 2 → Phase 3     │
│  Validate   Assign   Lock labs  CP-SAT      │
│                                 + Rooms     │
│                  ↓                          │
│  Phase 3.5 → Phase 4                        │
│  Format      Export CSVs                    │
│  rooms       + summary_report.txt           │
└─────────────────────┬───────────────────────┘
                      │
        outputs/{sem}/         Generated files
              │
              ▼
┌─────────────────────────────────────────────┐
│           AI LAYER  (Phase 5)               │
│                                             │
│  rag_indexer.py   FAISS + cross-encoder     │
│  ai_explainer.py  RAG Q&A + decomposition   │
│  agent.py         LangChain 16-tool agent   │
│  substitute.py    Absence management        │
│  sync_manager.py  Atomic write-back         │
└─────────────────────┬───────────────────────┘
                      │
              Streamlit app.py
              6-page dashboard
```

---

## Phases

| Phase | What it does | Key file |
|---|---|---|
| **0** | Validate all CSVs — missing cols, unknown IDs, capacity limits | `src/phase0/validator.py` |
| **1** | Build faculty ↔ section ↔ course assignment map | `src/phase1/assignment_builder.py` |
| **2** | Lock P5–P6 lab blocks from `lab_allotment.csv`; mark rooms occupied | `src/phase2/lab_scheduler.py` |
| **3** | OR-Tools CP-SAT — assign theory periods + **room variables inside solver** | `src/phase3/theory_scheduler.py` |
| **3.5** | Format CP-SAT room assignments into CSV (greedy fallback if solver skipped rooms) | `src/phase3_5/room_allocator.py` |
| **4** | Write all section/faculty CSVs + `summary_report.txt` | `src/phase4/output_generator.py` |
| **5** | AI layer — RAG, agent, substitutes, sync | `src/phase5/` |

---

## AI Features

### RAG with Metadata Filtering + Cross-Encoder Reranking
`rag_indexer.py` builds a FAISS `IndexFlatL2` over 732 timetable documents. At query time:
1. **Metadata filters** (`section`, `day`, `faculty`, `source_type`) narrow the search to a filtered sub-index
2. **k×3 candidate fetch** — retrieves 15 candidates instead of 5 when reranking is on
3. **Cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores (query, doc) pairs jointly and re-ranks to top 5
4. Graceful fallback to FAISS order if the cross-encoder model is unavailable

### Query Decomposition for Multi-Hop Questions
`ai_explainer.py` detects multi-hop queries using a heuristic gate (cross-entity + cross-day patterns) and decomposes them via the Groq LLM into 2–4 simple sub-queries, runs RAG on each, merges results, then answers the original question.

### LLM Agent with 16 Tools, Atomic Commits, Rollback
`agent.py` wraps a LangChain ReAct agent. All write operations go through `sync_manager.py` which:
- Backs up affected CSVs before any change
- Writes an operation JSON to `agent_ops/`
- Supports full rollback to the pre-change state

### Faculty Preference Soft Constraints
`faculty.csv` includes `pref_time` (morning/afternoon/none), `pref_no_backtoback` (bool), and `pref_no_teaching_day` (day/none). These become soft penalty terms in the CP-SAT objective:

| Constant | Value | Meaning |
|---|---|---|
| `PENALTY_PREF_TIME` | 8 | Penalty per slot in wrong half of day |
| `PENALTY_PREF_NO_BTB` | 6 | Penalty per consecutive pair |
| `PENALTY_PREF_FREE_DAY` | 10 | Penalty per slot on preferred free day |

All weights are strictly below structural constraints (`PENALTY_GAP=100`, `PENALTY_LAB_WINDOW=50`).

### CP-SAT Room Assignment
Classrooms are assigned inside the CP-SAT model (not post-hoc):
- **Hard A**: every occupied theory slot gets exactly one room
- **Hard B**: no two sections share a room in the same slot
- **Lab exclusion**: lab/computer rooms are excluded at variable-creation time
- **Phase 2 lock**: rooms pre-occupied by lab sessions are never offered as decision variables
- **Soft**: `PENALTY_ROOM_OVERCAP=4` nudges toward larger rooms

Result: 240/240 theory slots assigned, 0 `ROOM_UNASSIGNED`.

---

## Multi-Semester Support

The system supports isolated semester environments. Each semester has its own data directory and output directory.

**To add a new semester (`cse_sem7`):**

1. **Create data directory** with the standard CSVs:
   ```
   data/cse_sem7/
   ├── courses.csv, faculty.csv, assignments.csv
   ├── rooms.csv, lab_allotment.csv
   └── (optional) elective_slots.csv
   ```

2. **Register it in `config.py`**:
   ```python
   # config.py → SEMESTER_CONFIGS dict
   "cse_sem7": SemesterConfig(
       data_dir=DATA_ROOT / "cse_sem7",
       output_dir=OUTPUT_ROOT / "cse_sem7",
   )
   ```

3. **Run the pipeline:**
   ```bash
   python run_all.py --sem cse_sem7
   ```
   The UI auto-detects new semesters from `list_available_semesters()`.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set Groq API key (only needed for AI Assistant / Agent)
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/Mac
# Edit .env → GROQ_API_KEY=gsk_...

# 3. Generate timetable
python run_all.py --sem cse_sem3

# 4. Launch dashboard
streamlit run app.py

# 5. (Optional) AI agent CLI
python src/phase5/agent.py
```

---

## Configuration — `config.py`

| Constant | Value | Purpose |
|---|---|---|
| `DAYS` | Mon–Fri | Week structure |
| `PERIODS` | P1–P6 | 6 periods per day |
| `THEORY_PERIODS` | P1–P4 | Theory window |
| `LAB_PERIODS` | P5–P6 | Lab window (2-period block) |
| `SECTIONS` | A–L | 12 sections per semester |
| `MAX_HOURS` | Prof=12, Asso=16, Asst=20 | Weekly teaching caps |
| `PENALTY_GAP` | 100 | Hard-constraint proxy — no intra-day gaps |
| `PENALTY_LAB_WINDOW` | 50 | Prevent theory in P5–P6 on non-lab days |
| `PENALTY_ROOM_OVERCAP` | 4 | Prefer larger classrooms (lowest-priority soft) |
| `GROQ_MODEL` | llama-3.3-70b-versatile | Primary LLM |
| `GROQ_MODEL_ALT` | llama-3.1-8b-instant | Fast fallback |

---

## Streamlit Dashboard — 6 Pages

| # | Page | What it shows |
|---|---|---|
| 1 | **Dashboard** | Quality score, `st.metric` quick stats, faculty load heatmap (Altair) |
| 2 | **Timetables** | Period×Day pivot grid by section or faculty; Room Assignments tab; CSV download |
| 3 | **Substitute Finder** | Report absence → ranked candidates with Confirm & Commit; load-swap planning |
| 4 | **Faculty Workload** | Sortable workload table, individual deep-dive with progress bar & bar chart, free-slot finder |
| 5 | **AI Assistant** | Groq + RAG chat with retrieval debug expander (decomposition status, doc count, CE rerank, top snippets) |
| 6 | **AI Agent** | LangChain agent with 16 tools; recent operations table; rollback UI |

### Sidebar
- **Semester selector** with info card (courses · faculty · sections · RAG doc count)
- **AI Features Active badge** (static — lists all 5 improvements)
- **Pipeline status** — last-run timestamp
- **↺ Regenerate Schedule** — runs `run_all.py` subprocess
- **🔍 Rebuild RAG Index** — rebuilds FAISS in-process
- **🩺 System Health** — expandable checker

---

## Known Limitations

1. **Enrollment data** — section size is fixed at 60 (no student enrollment CSV); room capacity soft penalty uses this constant
2. **Single-building assumption** — travel time between rooms is not modelled; back-to-back classes across buildings are not penalised
3. **Groq rate limits** — the AI assistant degrades gracefully to a data-driven fallback, but rate-limited sessions see delayed responses
4. **Solve time scales with sections** — adding more sections beyond 12 will increase CP-SAT solve time super-linearly; the 120s time budget may need increasing
5. **Cross-encoder first load** — `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB) is downloaded from HuggingFace on first use; subsequent uses are instant from cache

---

## Generated Outputs

```
outputs/{sem}/
  section_A_timetable.csv  ...  section_L_timetable.csv   (12 files)
  faculty_F01_timetable.csv ... faculty_F20_timetable.csv  (per faculty)
  room_assignment.csv        Phase 3.5 — classroom per theory slot
  summary_report.txt         Quality metrics, violation counts, faculty load
  rag_index.faiss            FAISS vector index  (gitignored)
  rag_docs.json              Document store      (gitignored)
  agent_ops/                 Committed operation logs (gitignored)
```

---

## Project Structure

```
timetable_system/
├── app.py                   Streamlit dashboard (6 pages)
├── run_all.py               Full pipeline runner with timing
├── config.py                All system constants + semester registry
├── requirements.txt
├── .env.example
├── data/
│   ├── cse_sem3/            Semester 3 input CSVs
│   └── cse_sem5/            Semester 5 input CSVs
├── outputs/
│   ├── cse_sem3/            Generated files for sem 3
│   └── cse_sem5/            Generated files for sem 5
├── utils/
│   └── health_check.py      System health verifier
├── src/
│   ├── phase0/              CSV validation
│   ├── phase1/              Assignment map builder
│   ├── phase2/              Lab slot locking
│   ├── phase3/              CP-SAT theory scheduler + room assignment
│   ├── phase3_5/            Room output formatter
│   ├── phase4/              Output generation
│   └── phase5/              AI layer
│       ├── agent.py         LangChain tool-use agent (16 tools)
│       ├── agent_ops.py     Operation log & rollback
│       ├── ai_explainer.py  RAG Q&A + query decomposition
│       ├── chat.py          Interactive CLI chat
│       ├── llm_wrapper.py   Groq LLM factory + retry
│       ├── prompt_builder.py System prompt assembly
│       ├── rag_indexer.py   FAISS index + cross-encoder reranker
│       ├── substitute.py    Substitute candidate ranking
│       ├── swap.py          Load-swap planning
│       └── sync_manager.py  Atomic write-back on changes
└── tests/                   pytest test suite
```
