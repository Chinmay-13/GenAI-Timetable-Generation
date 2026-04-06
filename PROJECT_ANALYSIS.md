# Project Analysis: AI-Powered Academic Timetable Generation System

> **Document Type**: Comprehensive Technical Analysis
> **Generated**: April 6, 2026
> **Project Root**: `c:\admin\3rd year\sem6\E3_genai\timetable_system`
> **Lines of Code Analyzed**: ~8,000+

---

## 1. Executive Overview

This is a **sophisticated AI-powered academic timetable generation system** for university CSE departments. The system combines **Google OR-Tools CP-SAT** constraint programming with **modern AI capabilities** (LLM-powered RAG, LangChain agents) to generate, manage, and query academic timetables.

### 1.1 Core Capabilities
- **Automated Timetable Generation**: Schedules theory and lab sessions for 12 sections (A-L) across 5 days
- **Conflict-Free Scheduling**: Uses CP-SAT solver with hard/soft constraints
- **Integrated Room Assignment**: Assigns classrooms inside the solver
- **AI Assistant**: Natural language Q&A via Groq LLM + FAISS RAG
- **Autonomous Change Management**: LangChain agent with 16 tools
- **Multi-Semester Support**: Isolated environments (cse_sem3, cse_sem5)

---

## 2. Project Architecture

### 2.1 Directory Structure

```
timetable_system/
├── app.py                    # Streamlit dashboard (1,530 lines)
├── run_all.py                # Pipeline runner (297 lines)
├── config.py                 # System constants (212 lines)
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
├── README.md                 # Documentation
├── PROJECT_ANALYSIS.md       # This document
│
├── data/                     # Input data (semester-isolated)
│   ├── cse_sem3/             # Semester 3
│   │   ├── courses.csv       # 5 courses
│   │   ├── faculty.csv       # 20 faculty
│   │   ├── assignments.csv   # Faculty mappings
│   │   ├── lab_allotment.csv # Lab assignments
│   │   └── rooms.csv         # 18 rooms
│   └── cse_sem5/             # Semester 5 (with electives)
│       ├── courses.csv       # 13 courses
│       ├── faculty.csv
│       ├── assignments.csv
│       ├── lab_allotment.csv
│       ├── rooms.csv
│       └── elective_slots.csv
│
├── outputs/                  # Generated outputs
│   ├── cse_sem3/
│   │   ├── section_{A-L}_timetable.csv
│   │   ├── faculty_{F01-F20}_timetable.csv
│   │   ├── room_assignment.csv
│   │   ├── summary_report.txt
│   │   ├── rag_index.faiss
│   │   └── agent_ops/
│   └── cse_sem5/
│
├── src/
│   ├── phase0/               # Validation
│   │   ├── loader.py         # Data loading (252 lines)
│   │   └── validator.py      # CSV validation (138 lines)
│   ├── phase1/               # Assignment building
│   │   └── assignment_builder.py (123 lines)
│   ├── phase2/               # Lab locking
│   │   └── lab_scheduler.py  (245 lines)
│   ├── phase3/               # Theory scheduling
│   │   └── theory_scheduler.py (932 lines)
│   ├── phase3_5/             # Room allocation
│   │   └── room_allocator.py (359 lines)
│   ├── phase4/               # Output generation
│   │   └── output_generator.py (281 lines)
│   └── phase5/               # AI layer
│       ├── agent.py          # 16-tool agent (50KB)
│       ├── rag_indexer.py    # FAISS + cross-encoder (32KB)
│       ├── sync_manager.py   # Atomic write-back (35KB)
│       ├── substitute.py     # Substitute ranking (24KB)
│       └── ai_explainer.py   # RAG Q&A (20KB)
│
├── utils/
│   └── health_check.py       # System verifier (227 lines)
└── tests/
    └── test_pipeline.py      # Test suite
```

### 2.2 Pipeline Flow

```
Input CSVs → Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 3.5 → Phase 4 → RAG
             Validate   Build     Lock      CP-SAT    Room       Export    AI
                       Map       Labs      Theory    Assign              Ready
```

---

## 3. Configuration (`config.py`)

### 3.1 Core Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DAYS` | Mon-Fri | Operating week |
| `PERIODS` | 1-6 | 6 periods per day |
| `THEORY_PERIODS` | 1-4 | Theory window |
| `LAB_PERIODS` | 5-6 | Lab window (2-period block) |
| `SECTIONS` | A-L (12) | Total sections |
| `SECTION_SIZE` | 60 | Students per section |

### 3.2 Faculty Caps

| Designation | Max Hours |
|-------------|-----------|
| Prof | 12 |
| Asso Prof | 16 |
| Asst Prof | 20 |

### 3.3 CP-SAT Penalty Weights

| Penalty | Value | Meaning |
|---------|-------|---------|
| PENALTY_GAP | 100 | Intra-day gaps |
| PENALTY_LAB_WINDOW | 50 | Theory in P5-P6 |
| PENALTY_BACK_TO_BACK | 5 | Same subject consecutive |
| PENALTY_SAME_DAY | 3 | Same subject multiple times |
| PENALTY_PREF_TIME | 8 | Wrong time preference |
| PENALTY_PREF_NO_BTB | 6 | Consecutive when unwanted |
| PENALTY_PREF_FREE_DAY | 10 | Teaching on free day |
| REWARD_CONSECUTIVENESS | 200 | Reward for consecutive |

### 3.4 Solver Time Limits

| Stage | Time (seconds) |
|-------|----------------|
| Penalty stage | 180 |
| Reward stage | 90 |
| Consecutiveness hard constraint | 30 |
| Parallel workers | 8 |

### 3.5 Course Credits to Periods

| Credits | Theory | Lab |
|---------|--------|-----|
| 5 (with lab) | 4 | 1 |
| 4 | 4 | 0 |

### 3.6 Multi-Semester Path System

```python
@dataclass(frozen=True)
class SemesterPaths:
    sem_id: str
    data_dir: Path
    output_dir: Path
    rag_index_path: Path
    rag_docs_path: Path
    chat_memory_path: Path
    agent_ops_dir: Path

# Usage: get_sem_paths("cse_sem3") → SemesterPaths
```

---

## 4. Data Layer

### 4.1 cse_sem3 Data

**courses.csv**: 5 core courses
- UE24CS351A: CNS (5 credits, has_lab)
- UE24CS352A: OS (5 credits, has_lab)
- UE24MA342A: PROB (4 credits)
- UE24CS342A: SE (4 credits)
- UE24CS343A: DMGT (4 credits)

**faculty.csv**: 20 faculty with preferences
- Fields: faculty_id, name, designation, pref_time, pref_no_backtoback, pref_no_teaching_day
- Designations: Prof, Asso Prof, Asst Prof
- pref_time: morning/afternoon/none
- pref_no_backtoback: True/False
- pref_no_teaching_day: day name or "none"

**assignments.csv**: Faculty-section mappings
- Fields: faculty_id, course_code, sections_handled
- sections_handled: comma-separated (e.g., "A,B,C")
- Core courses: 2-3 sections per faculty
- Lab courses: distributed among non-Prof faculty

**lab_allotment.csv**: Lab slot assignments
- Fields: day, course_code, section_pair, room, faculty_id
- section_pair: JSON array format (e.g., "[\"A\",\"D\"]")
- 12 total lab sessions
- Monday: CNS labs, Thursday: OS labs
- Each session: 2 periods (P5-P6), 2 sections per lab

**rooms.csv**: 18 rooms
- 6 LAB rooms (capacity 120)
- 12 CLASSROOM rooms (capacity 60)
- Fields: room_id, room_name, floor, room_type, capacity

### 4.2 cse_sem5 Data

**courses.csv**: 13 courses
- Core: ML, DBMS, CA (3 courses)
- Elective Group E1: CC, CYS, NLP, CV, IoT, BCT (6 courses)
- Elective Group E2: DM, PC, IR, ES, CD, GTA (6 courses)

**elective_slots.csv**: 24 fixed elective slots
- Fields: elective_group, course_code, day, period_start, period_end, room, faculty_id, enrolled_sections
- E1: Monday/Friday P3-P4
- E2: Tuesday/Thursday P1-P2

---

## 5. Phase Details

### 5.1 Phase 0: Validation (`validator.py`)

**Function**: `validate(courses, faculty, assignments, data_dir) → bool`

**Validations**:
1. Section coverage: Core courses need all 12 sections
2. Faculty course count: Max 2 courses per faculty
3. Professor lab rule: Profs not assigned to lab courses
4. Weekly hour load: Must not exceed MAX_HOURS[designation]

**Calculation**: Theory periods = credits × sections; Lab sessions = 1 × 2 periods

### 5.2 Phase 1: Assignment Builder (`assignment_builder.py`)

**Function**: `build_assignment_map(data_dir) → dict`

**Data Structure**:
```python
assignment_map[course_code][section] = faculty_id
```

**Process**:
1. Load and enrich data
2. Build assignment_map nested dictionary
3. Validate coverage (12 sections for core)
4. Raise ValueError on any validation failure

### 5.3 Phase 2: Lab Scheduler (`lab_scheduler.py`)

**Function**: `lock_labs(data_dir) → tuple`

**Grids**:
```python
section_grid[section][day][period] = token or None
faculty_grid[faculty_id][day][period] = token or None
room_grid[room_name][day][period] = token or None
```

**Process**:
1. Read lab_allotment.csv
2. Validate section pairs (must be exactly 2)
3. Check for conflicts (section/faculty/room double-booking)
4. Lock P5-P6 slots for both sections
5. Process elective_slots.csv if present

**Tokens**:
- Lab: `{course_code}_LAB`
- Elective: `Elective 1` or `Elective 2` (sections), `course_code` (faculty/room)

### 5.4 Phase 3: Theory Scheduler (`theory_scheduler.py`)

**Function**: `solve_theory(...) → dict`

This is the **core CP-SAT solver** (932 lines).

**Variables**:
- `x[(section, course, day, period)]`: BoolVar - assign course to slot
- `room_assigned[(section, day, period, room_id)]`: BoolVar - room assignment
- `is_used[(section, day, period)]`: BoolVar - slot occupied
- `daily_count[(section, day)]`: IntVar - theory slots per day
- `gap_var[(section, day, period)]`: BoolVar - gap indicator

**Hard Constraints**:
1. Course period requirement: Exact periods from credits
2. No section double-booking: Sum of courses in slot ≤ 1
3. No faculty double-booking: Sum for faculty in slot ≤ 1
4. Faculty cap: Total slots ≤ MAX_HOURS[designation]
5. Hard compactness: Monotonicity (no gaps)
6. Daily target exact: Must fill all free theory periods

**Soft Constraints (Penalty Terms)**:
1. Intra-day gaps: gap_var × PENALTY_GAP (100)
2. Lab window theory: × PENALTY_LAB_WINDOW (50)
3. Same subject same day: × PENALTY_SAME_DAY (3)
4. Back-to-back same subject: × PENALTY_BACK_TO_BACK (5)
5. Faculty time preference: × PENALTY_PREF_TIME (8)
6. Faculty no-BTB preference: × PENALTY_PREF_NO_BTB (6)
7. Faculty free day preference: × PENALTY_PREF_FREE_DAY (10)

**Room Assignment Inside CP-SAT**:
1. Exactly one room per occupied slot
2. No room double-booking
3. Soft: Prefer larger rooms (penalty for overcapacity)

**Two-Stage Solving**:
```python
# Stage 1: Minimize penalties (180s)
model.Minimize(sum(penalty_terms))
solver.Solve(model)

# Stage 2: Maximize rewards with frozen penalty (90s)
model.Add(sum(penalty_terms) == best_penalty)
model.Maximize(sum(reward_terms))
solver2.Solve(model)
```

**Consecutiveness Enforcement**:
- Phase A: Hard constraint requiring adjacent pairs (30s timeout)
- Phase B: Soft reward if Phase A infeasible

**Output Dictionary**:
```python
{
    "section_grid": ..., "faculty_grid": ..., "room_grid": ...,
    "assignment_map": ..., "lab_details": ..., "elective_details": ...,
    "solver_status": str,
    "soft_violations": dict,
    "consecutive_analysis": dict,
    "room_assignment_map": dict
}
```

### 5.5 Phase 3.5: Room Allocator (`room_allocator.py`)

**Functions**:
- `assign_theory_rooms(...) → tuple[dict, list]`
- `run_phase35(...) → dict`

**Dual Path**:
1. **CP-SAT Path**: If `room_assignment_map` non-empty, use solver assignments
2. **Greedy Path**: Bipartite matching (sorted by capacity/alphabet)

**Output**: `theory_room_grid[section][day][period] = room_name`

### 5.6 Phase 4: Output Generator (`output_generator.py`)

**Function**: `generate_outputs(result, data_dir, output_dir) → dict`

**Generates**:
1. **Section CSVs** (12 files): `section_{A-L}_timetable.csv`
   - Columns: Day, P1-P6
   - Cell format: "COURSE (FACULTY)" or "COURSE LAB" or "----"

2. **Faculty CSVs** (per faculty): `faculty_{FID}_timetable.csv`
   - Cell format: "COURSE (section)" or "COURSE LAB (pair)"

3. **Room Assignment CSV**: `room_assignment.csv`
   - Columns: Section, Day, Period, Course, Faculty, Room

4. **Summary Report**: `summary_report.txt`
   - Total slots, violation counts, faculty load table

**Quality Score Calculation**:
```python
score = max(0.0, round(100 - ((same_day + 2*back_to_back + 5*overload_count) / 2), 1))
```

---

## 6. AI Layer (Phase 5)

### 6.1 RAG Indexer (`rag_indexer.py`)

**Purpose**: FAISS vector index for semantic search

**Document Types** (732+ per semester):
1. Per-slot: "Section A on Monday P1: CNS (AG)"
2. Per-day: "Section A on Monday: P1=CNS, P2=OS..."
3. Full-week: "Section A full weekly timetable: ..."
4. Room availability summaries

**Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

**FAISS**: `IndexFlatL2` (exact L2 search)

**Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Retrieves 15 candidates, reranks to top 5

**Metadata Filtering**:
- Filters: section, day, faculty, source_type
- Applied before embedding search

### 6.2 AI Explainer (`ai_explainer.py`)

**Query Decomposition**:
- Detect multi-hop queries (multiple entities/days)
- Decompose via LLM into 2-4 simple sub-queries
- Run RAG on each, merge results

**Answer Generation**:
- Groq model: `llama-3.3-70b-versatile`
- Temperature: 0.3, max_tokens: 512

### 6.3 Agent (`agent.py`)

**Framework**: LangChain ReAct with 16 tools

**Query Tools** (8):
1. get_section_timetable
2. get_faculty_schedule
3. find_free_slots
4. get_summary_stats
5. find_substitute
6. get_faculty_workload
7. get_room_availability
8. detect_schedule_conflicts

**Action Tools** (8):
9. commit_substitute
10. swap_faculty
11. reassign_section_faculty
12. mark_absence
13. assign_substitute
14. undo_last_change
15. preview_change
16. get_system_status

**Atomic Operations via sync_manager**:
- Preview → Validate → Backup → Commit → Log
- Full rollback capability

### 6.4 Substitute Finder (`substitute.py`)

**Scoring Algorithm**:
```python
# Positive
+20 same course teaching
+15 same designation
+10 under hours
+10 free at slot

# Negative
-30 overloaded
-20 teaching 2+ courses
-15 busy at slot
-10 different designation
```

**Lab Block Handling**: P5-P6 treated as atomic 2-period block

### 6.5 Sync Manager (`sync_manager.py`)

**Atomic Commit Flow**:
1. Write to temp directory (preview)
2. Validate CSV integrity
3. Backup originals to `agent_ops/backups/`
4. Atomic move temp → canonical
5. Log operation JSON

**Operation Log**:
```json
{
  "op_id": "20260401T203501Z-8b8e600f",
  "timestamp": "2026-04-01T20:35:01Z",
  "type": "substitute_assignment",
  "change_dict": {...},
  "affected_files": [...],
  "backup_paths": {...},
  "committed": true
}
```

---

## 7. Streamlit Dashboard (`app.py`)

### 7.1 Session State
- sem_id, chat_history, _rag_debug, agent_output
- current_page, _agent_confirm, _agent_pending_instruction
- _agent_preview_dir, _agent_preview_op_id, show_debug

### 7.2 Six Pages
1. **Dashboard**: Quality score, metrics, faculty heatmap
2. **Timetables**: Period×Day grids, room assignments
3. **Substitute Finder**: Absence form, ranked candidates
4. **Faculty Workload**: Sortable table, deep-dive
5. **AI Assistant**: RAG chat, retrieval debug
6. **AI Agent**: LangChain interface, operation history

### 7.3 Sidebar
- Semester selector (auto-detect)
- Info card (courses, faculty, sections, RAG docs)
- Regenerate/Rebuild buttons
- Health status dot

---

## 8. Testing & QA

### 8.1 Test Suite (`tests/test_pipeline.py`)
Coverage: Phase 0-4 validation, solver feasibility, output generation

### 8.2 Health Check (`utils/health_check.py`)
Checks:
- Input CSVs (5 required)
- Output files (12 section + 12 faculty + summary + room)
- RAG index (faiss + json)
- Environment (GROQ_API_KEY)
- Packages (7 critical)

---

## 9. Technology Stack

| Component | Technology |
|-----------|------------|
| Solver | Google OR-Tools CP-SAT |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS (IndexFlatL2) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq API (Llama 3.3 70B) |
| LLM Framework | LangChain |
| UI | Streamlit |
| Viz | Altair |
| Data | pandas |
| Testing | pytest |

---

## 10. Entry Points

```bash
# Generate timetable
python run_all.py --sem cse_sem3

# Launch dashboard
streamlit run app.py

# AI Agent CLI
python src/phase5/agent.py

# Interactive chat
python src/phase5/chat.py

# Health check
python utils/health_check.py --sem cse_sem3

# Run tests
pytest tests/ -v
```

---

## 11. File Statistics

| Component | Files | Lines |
|-----------|-------|-------|
| Root | 3 | ~2,000 |
| Phase 0 | 2 | 390 |
| Phase 1 | 1 | 123 |
| Phase 2 | 1 | 245 |
| Phase 3 | 1 | 932 |
| Phase 3.5 | 1 | 359 |
| Phase 4 | 1 | 281 |
| Phase 5 | 10 | ~175KB |
| Utils | 1 | 227 |
| Tests | 1 | 267 |
| **Total** | **~22** | **~8,000+** |

---

## 12. Known Limitations

1. Section size fixed at 60 (no enrollment CSV)
2. Single building (no travel time)
3. Groq rate limits under heavy load
4. Solve time scales super-linearly with section count
5. Cross-encoder ~90MB download on first use

---

*Generated: April 6, 2026*
*System: GenAI Timetable Generation*
