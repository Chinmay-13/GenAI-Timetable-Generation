# Timetable OS

AI-powered academic timetable generator for CSE Department — 3rd Semester, 12 sections (A–L), 5 courses, powered by Google OR-Tools CP-SAT and Groq LLaMA.

---

## Quick Start

```bash
# 1. Clone & install
pip install -r requirements.txt

# 2. Set Groq API key
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/Mac
# Edit .env  →  GROQ_API_KEY=gsk_...

# 3. Generate timetable
python run_all.py

# 4. Launch dashboard
streamlit run app.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIMETABLE OS — PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

  data/                     Inputs
  ├── courses.csv            5 courses (code, credits, has_lab)
  ├── faculty.csv            Faculty roster (id, name, designation)
  ├── assignments.csv        Who teaches which course to which sections
  ├── rooms.csv              20 rooms (6 labs + 14 classrooms)
  └── lab_allotment.csv      Which section has which lab on which day

       │
       ▼
┌─────────────┐
│  Phase 0    │  Validate all CSVs — missing cols, unknown IDs
│  validator  │
└──────┬──────┘
       │
┌──────▼──────┐
│  Phase 1    │  Build faculty↔section↔course assignment map
│  assignment │
└──────┬──────┘
       │
┌──────▼──────┐
│  Phase 2    │  Lock lab slots (P5–P6) from lab_allotment.csv
│  lab_sched  │  Generates: section_grid, faculty_grid, room_grid
└──────┬──────┘
       │
┌──────▼──────┐
│  Phase 3    │  CP-SAT solver (OR-Tools) — assign theory periods
│  CP-SAT     │  Constraints: no double-booking, subject spread,
│  theory     │  load caps, soft: reduce back-to-back same subject
└──────┬──────┘
       │
┌──────▼──────┐
│ Phase 3.5   │  Greedy bipartite room allocation
│ room_alloc  │  Assigns classrooms to theory slots post-solve
│             │  → outputs/room_assignment.csv
└──────┬──────┘
       │
┌──────▼──────┐
│  Phase 4    │  Export CSVs + summary report
│  output_gen │  → outputs/section_*.csv  faculty_*.csv
└──────┬──────┘
       │
┌──────▼──────┐
│  Phase 5    │  AI Layer (optional, requires GROQ_API_KEY)
│  AI / RAG   │  ├── rag_indexer.py  → FAISS index over outputs/
│  / Agent    │  ├── ai_explainer.py → RAG-augmented Q&A (Groq)
│             │  ├── agent.py        → LangChain tool-use agent
│             │  └── sync_manager.py → Atomic write-back on changes
└─────────────┘

       │
       ▼
  Streamlit app.py  (6 pages — see below)
```

---

## Phase Details

| Phase | Module | Description |
|---|---|---|
| **0** | `src/phase0/validator.py` | Load & validate CSVs; fail fast on bad data |
| **1** | `src/phase1/assignment_builder.py` | Faculty → section → course map |
| **2** | `src/phase2/lab_scheduler.py` | Lock P5–P6 lab blocks; mark rooms occupied |
| **3** | `src/phase3/theory_scheduler.py` | OR-Tools CP-SAT; penalty/reward weights |
| **3.5** | `src/phase3_5/room_allocator.py` | Greedy classroom assignment; avoids lab rooms |
| **4** | `src/phase4/output_generator.py` | Write all CSVs + `summary_report.txt` |
| **5** | `src/phase5/` | AI explainer, agent, substitute engine, RAG |

---

## Streamlit Dashboard — 6 Pages

| # | Page | What it shows |
|---|---|---|
| 1 | **Dashboard** | `st.metric` quick stats, last-generation timestamp, faculty load heatmap (Altair), quality score |
| 2 | **Timetables** | Period×Day pivot grid for each section or faculty (🟢 theory · 🔵 lab · ⬜ free); Room Assignments tab |
| 3 | **Substitute Finder** | Report absence → candidate cards with Confirm & Commit button → load-swap planning |
| 4 | **Faculty Workload** | Sortable workload table (green/yellow/red by cap), individual deep-dive with progress bar & bar chart, free-slot finder |
| 5 | **AI Assistant** | Groq + RAG chat with example question chips; persistent session history |
| 6 | **AI Agent** | LangChain tool-use agent (17 tools); shows reasoning steps; commit/rollback support |

### Sidebar
- **Pipeline status** — last-run timestamp
- **↺ Regenerate Schedule** — runs `run_all.py` subprocess with spinner
- **🔍 Rebuild RAG Index** — rebuilds FAISS index in-process
- **🩺 System Health** — expandable checker: inputs, outputs, RAG, API key, packages

---

## AI Agent Tools (17 total)

| Tool | Category | What it does |
|---|---|---|
| `get_section_timetable` | Read | Full week grid for a section |
| `get_faculty_schedule` | Read | Full week grid for a faculty |
| `get_absent_periods` | Read | Exact periods a faculty teaches on a day |
| `find_free_slots` | Read | Free periods for one faculty on a day |
| `get_faculty_workload` | Read | Hours, courses, overload status |
| `get_free_faculty` | Read | All faculty free in a given slot |
| `get_room_availability` | Read | Free classrooms with capacity for a slot |
| `find_free_rooms` | Read | Free rooms matching optional capacity filter |
| `get_summary_stats` | Read | Raw summary report text |
| `get_weekly_stats` | Read | Parsed key metrics |
| `detect_schedule_conflicts` | Read | Faculty/room double-booking scan |
| `find_substitute` | Read | Full substitute plan for a day |
| `simulate_substitute` | Read | Top 3 candidates preview — no commit |
| `list_agent_ops` | Read | Recent committed operations |
| `commit_substitute` | **Write** | Apply substitute + sync all artifacts |
| `rollback_last_operation` | **Write** | Undo committed change |
| `generate_session_summary` | **Write** | Write session report |

---

## Generated Outputs

```
outputs/
  section_A_timetable.csv   ... section_L_timetable.csv   (12 files)
  faculty_F01_timetable.csv ... faculty_F20_timetable.csv  (per faculty)
  room_assignment.csv        (Phase 3.5 — classroom per theory slot)
  summary_report.txt         (quality metrics, violation counts)
  rag_index.faiss            (FAISS vector index — gitignored)
  rag_docs.json              (document store — gitignored)
  agent_ops/                 (committed operation logs — gitignored)
```

---

## Configuration — config.py

| Constant | Value | Purpose |
|---|---|---|
| `DAYS` | Mon–Fri | Week structure |
| `PERIODS` | P1–P6 | 6 periods per day |
| `LAB_PERIODS` | P5–P6 | Lab window (2-period block) |
| `SECTIONS` | A–L | 12 sections |
| `MAX_HOURS` | Prof=12, Asso=16, Asst=20 | Weekly teaching caps |
| `GROQ_MODEL` | llama-3.3-70b-versatile | Primary LLM |
| `GROQ_MODEL_ALT` | llama-3.1-8b-instant | Fast fallback |

---

## Course Structure

| Code | Short | Theory/week | Lab |
|---|---|---|---|
| UE24CS251A | DDCO | 4 | P5–P6 (from lab_allotment.csv) |
| UE24CS252A | DSA  | 4 | P5–P6 (from lab_allotment.csv) |
| UE24MA242A | MATH | 4 | — |
| UE24CS242A | WT   | 4 | — |
| UE24CS243A | AFLL | 4 | — |

---

## CLI Tools

```bash
# Interactive RAG chat
python src/phase5/chat.py

# AI agent CLI
python src/phase5/agent.py

# Rebuild RAG index only
python src/phase5/rag_indexer.py

# System health check
python utils/health_check.py

# Run test suite
pytest tests/ -v
```

---

## Troubleshooting

### `ModuleNotFoundError: langchain_groq`
```bash
pip install langchain-groq
```

### `GROQ_API_KEY not set` / agent returns None
```bash
# Create .env in project root:
echo GROQ_API_KEY=gsk_your_key_here > .env
```

### `No timetable found` in the UI
The pipeline hasn't run yet, or ran with errors.
```bash
python run_all.py   # Check for ✗ errors in output
```

### `room_assignment.csv not found`
Phase 3.5 failed. Check that `data/rooms.csv` exists and has `CLASSROOM` type rooms.
```bash
python utils/health_check.py   # See which files are missing
```

### `RAG index not built`
```bash
pip install sentence-transformers faiss-cpu
python src/phase5/rag_indexer.py
```

### Solver timeout (Phase 3 takes too long)
Reduce time limits in `config.py`:
```python
PENALTY_STAGE_TIME = 60   # was 120
REWARD_STAGE_TIME  = 30   # was 60
```

### `Phase 0 validation FAILED`
Check your input CSVs match the expected columns:
- `courses.csv` → `course_code, course_name, credits, has_lab`
- `faculty.csv` → `faculty_id, name, designation`
- `assignments.csv` → `faculty_id, course_code, sections_handled`

### Commit fails in Substitute Finder
`sync_manager.py` cannot find the section CSV. Ensure `outputs/` exists and the pipeline ran successfully.

---

## Project Structure

```
timetable_system/
├── app.py                   Streamlit dashboard (6 pages)
├── run_all.py               Full pipeline runner with timing
├── config.py                All system constants
├── requirements.txt
├── .env.example
├── data/                    Input CSVs
├── outputs/                 Generated files (gitignored except .gitkeep)
├── utils/
│   └── health_check.py      System health verifier
├── src/
│   ├── phase0/              CSV validation
│   ├── phase1/              Assignment map builder
│   ├── phase2/              Lab slot locking
│   ├── phase3/              CP-SAT theory scheduler
│   ├── phase3_5/            Room allocator
│   ├── phase4/              Output generation
│   └── phase5/              AI layer
│       ├── agent.py         LangChain tool-use agent (17 tools)
│       ├── agent_ops.py     Operation log & rollback
│       ├── ai_explainer.py  RAG-augmented Q&A
│       ├── chat.py          Interactive CLI chat
│       ├── llm_wrapper.py   Groq LLM factory + retry
│       ├── prompt_builder.py System prompt assembly
│       ├── rag_indexer.py   FAISS index builder
│       ├── substitute.py    Substitute candidate ranking
│       ├── swap.py          Load-swap planning
│       └── sync_manager.py  Atomic write-back on changes
└── tests/                   pytest test suite
```
