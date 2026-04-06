# Timetable Generation System - Comprehensive Analysis Report

## Executive Summary

This is an **AI-powered academic timetable generator** for university CSE (Computer Science & Engineering) departments. The system generates conflict-free timetables for 12 sections (A-L) using Google OR-Tools CP-SAT solver, then layers an AI assistant on top for natural-language Q&A, substitute finding, and autonomous change management.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TIMETABLE OS ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌──────────────────────────────────────────────────────┐
│   INPUT DATA    │     │              GENERATION PIPELINE                      │
│  data/{sem}/    │     │                                                      │
│  ├── courses.csv│────▶│  Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3          │
│  ├── faculty.csv│     │  Validate    Build       Lock      CP-SAT Solver      │
│  ├── assignments│     │  Input       Faculty     Labs      + Room Assignment │
│  ├── rooms.csv  │     │  Data        Map                                    │
│  └── lab_allot  │     │                         │                            │
└─────────────────┘     │                         ▼                            │
                        │              Phase 3.5 ──▶ Phase 4 ──▶ RAG Index      │
                        │              Room       Output      FAISS Vector     │
                        │              Allocator  Generator     Store            │
                        └────────────────────┬────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI LAYER (Phase 5)                                    │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ rag_indexer │  │ ai_explainer│  │    agent    │  │ substitute  │          │
│  │  FAISS +    │──│   RAG Q&A   │──│ LangChain   │──│   Finder    │          │
│  │Cross-Encoder│  │+ Query Decomp│  │ 16 Tools    │  │  Ranking    │          │
│  └─────────────┘  └─────────────┘  └──────┬──────┘  └─────────────┘          │
│                                             │                                │
│                              ┌──────────────┴──────────────┐                │
│                              ▼                              ▼                │
│                       ┌─────────────┐               ┌─────────────┐           │
│                       │ agent_ops   │               │sync_manager │           │
│                       │ Logging &   │               │Atomic Write │           │
│                       │ Rollback    │               │+ Rollback  │           │
│                       └─────────────┘               └─────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                        ┌─────────────────────────────────────────────────────┐
                        │              STREAMLIT UI (app.py)                    │
                        │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
                        │  │Dashboard│ │Timetable│ │Substitute│ │  AI    │     │
                        │  │  Page   │ │  Page   │ │ Finder  │ │Assistant│     │
                        │  └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
                        └─────────────────────────────────────────────────────┘
```

---

## Project Structure Analysis

### Root Directory Files

| File | Purpose | Lines | Key Responsibilities |
|------|---------|-------|---------------------|
| `app.py` | Streamlit Dashboard | 1,281 | Main UI with 6 pages: Dashboard, Timetables, Substitute Finder, Faculty Workload, AI Assistant, AI Agent |
| `run_all.py` | Pipeline Runner | 257 | Executes all phases 0→4 + RAG indexing with timing reports |
| `config.py` | Central Configuration | 212 | Constants, semester paths, MAX_HOURS, penalties, LLM config |
| `README.md` | Documentation | 268 | Usage guide, architecture diagram, known limitations |
| `requirements.txt` | Dependencies | ~15 | ortools, streamlit, pandas, sentence-transformers, faiss-cpu, langchain-groq |
| `.env.example` | Environment Template | - | GROQ_API_KEY placeholder |
| `create_dummy_data.py` | Data Generator | ~400 | Generates sample CSVs for testing |
| `demo_ai.py` | AI Demo | ~300 | Standalone AI demonstrations |
| `smoke_test.py` | Quick Tests | ~200 | Basic functionality verification |
| `verify_rerank.py` | RAG Verification | ~200 | Tests cross-encoder reranking |

---

## Phase-by-Phase Deep Dive

### Phase 0: Input Validation (`src/phase0/`)

**Files:**
- `validator.py` (138 lines) - Comprehensive data validation
- `loader.py` (252 lines) - Semester-aware data loading

**Key Validations:**
```python
# Check 1: Section coverage per course (12 sections required)
# Check 2: Faculty max 2 courses
# Check 3: Professors NOT assigned to lab courses  
# Check 4: Weekly hour load per faculty (respects MAX_HOURS)
```

**Features:**
- Multi-semester support via `sem_id` parameter
- Elective course handling (partial coverage allowed)
- Automatic `is_elective` column enrichment

---

### Phase 1: Assignment Builder (`src/phase1/`)

**File:** `assignment_builder.py` (123 lines)

**Purpose:** Builds `faculty_id → course → [sections]` mapping

**Process:**
```python
assignment_map = {
    "UE24CS251A": {  # DDCO
        "A": "F01", "B": "F02", ...  # Each section gets one faculty
    },
    "UE24CS252A": {  # DSA
        "A": "F03", "B": "F04", ...
    }
}
```

**Validation:**
- Detects duplicate section assignments
- Respects designation rules (no Prof on labs)
- Core courses: all 12 sections required
- Electives: ≥1 section required

---

### Phase 2: Lab & Elective Locking (`src/phase2/`)

**File:** `lab_scheduler.py` (245 lines)

**Purpose:** Locks fixed slots (labs + electives) before theory scheduling

**Grid Structure:**
```python
section_grid[section][day][period] = course_code or None
faculty_grid[faculty_id][day][period] = course_code or None  
room_grid[room_name][day][period] = course_code or None
```

**Conflict Detection:**
- Section double-booking
- Faculty double-booking
- Room double-booking
- Professor-on-lab violations

**Output:** 3 populated grids ready for Phase 3

---

### Phase 3: Theory Scheduling (`src/phase3/`)

**File:** `theory_scheduler.py` (932 lines)

**Engine:** Google OR-Tools CP-SAT Solver

**Decision Variables:**
```python
# Binary variable: assign[course][section][day][period] ∈ {0, 1}
# Room variable: room[course][section][day][period] ∈ available_rooms
```

**Hard Constraints:**
| Constraint | Implementation |
|------------|----------------|
| No intra-day gaps | `PENALTY_GAP = 100` |
| Theory only in P1-P4 | `PENALTY_LAB_WINDOW = 50` for P5-P6 |
| No faculty conflicts | All-different per (day, period, faculty) |
| No room conflicts | All-different per (day, period, room) |
| Room type match | LAB rooms excluded from theory |

**Soft Constraints (Faculty Preferences):**
| Preference | Penalty | Description |
|------------|---------|-------------|
| `pref_time` | 8 | Morning/afternoon preference |
| `pref_no_backtoback` | 6 | Avoid consecutive periods |
| `pref_free_day` | 10 | One free day preference |
| `PENALTY_ROOM_OVERCAP` | 4 | Prefer larger rooms |

**Solve Strategy (Two-Stage):**
```python
# Stage 1: REWARD stage (60s time limit)
#   - Maximize consecutive same-course same-day pairings
#   - High tolerance for soft violations

# Stage 2: PENALTY stage (180s time limit)  
#   - Minimize all penalty terms
#   - Must reach OPTIMAL status
```

---

### Phase 3.5: Room Allocation (`src/phase3_5/`)

**File:** `room_allocator.py` (359 lines)

**Two Modes:**
1. **CP-SAT Mode:** Uses solver's room assignments (preferred)
2. **Greedy Fallback:** Bipartite matching if solver skipped rooms

**Algorithm (Greedy):**
```python
# For each (day, period) slot:
#   1. Get all theory classes scheduled
#   2. Get available classrooms (sorted by capacity desc)
#   3. Assign largest room to largest class
#   4. Mark rooms occupied
```

**Output:** `room_assignment.csv` with columns:
- Section, Day, Period, Course, Faculty, Room

---

### Phase 4: Output Generation (`src/phase4/`)

**File:** `output_generator.py` (281 lines)

**Generates:**
1. **Section CSVs** (12 files): `section_A_timetable.csv` → `section_L_timetable.csv`
2. **Faculty CSVs** (per faculty): `faculty_F01_timetable.csv` etc.
3. **Summary Report**: `summary_report.txt`

**Cell Format:**
```
DDCO (ABC)        # Theory: CourseShort (FacultyInitials)
DDCO LAB (ABC)    # Lab: CourseShort LAB (FacultyInitials)
Elective 1        # Electives: Generic label
----              # Free period
```

**Summary Report Contains:**
- Total theory/elective/lab slots placed
- Soft constraint violation counts
- Faculty load table with OVERLOAD indicators

---

## Phase 5: AI Layer (`src/phase5/`)

### 5.1 RAG Indexer (`rag_indexer.py` - 729 lines)

**Purpose:** FAISS vector index for semantic search over timetable data

**Architecture:**
```python
# Document Types:
#   1. Per-slot docs: "Section A on Monday P1: DDCO (ABC)"
#   2. Per-day docs: "Section A on Monday: P1=DDCO, P2=DSA..."
#   3. Full-week docs: "Section A full weekly timetable: ..."
#   4. Slot summaries: "On Wednesday P3: DDCO Sec A, DSA Sec B..."
#   5. Room docs: "Room availability on Monday P1: 4 free..."

# Embedding: all-MiniLM-L6-v2 (384-dim)
# Index: FAISS IndexFlatL2
# Cross-Encoder: ms-marco-MiniLM-L-6-v2 (for reranking)
```

**Query Flow:**
```
1. Extract filters (section/day/faculty/source_type) from query
2. Build filtered sub-index from matching docs
3. FAISS search → k×3 candidates
4. Cross-encoder rerank → top 5
5. Return with citations
```

---

### 5.2 AI Explainer (`ai_explainer.py` - 477 lines)

**Purpose:** Natural language Q&A over timetable data

**Key Functions:**
- `setup_context()` - Loads all CSVs into memory
- `explain()` - Direct LLM call with data context
- `explain_with_rag()` - RAG-enhanced version
- `detect_issues()` - Automated problem detection

**Multi-Hop Query Decomposition:**
```python
# Complex query: "Which sections does F03 teach on days section A has labs?"
#   ↓
# Decomposed:
#   1. "What days does section A have labs?"
#   2. "What does F03 teach and on which days?"
#   ↓
# Retrieve(1) + Retrieve(2) → Merge → Answer
```

**Fallback Strategy:**
- If LLM unavailable → Direct CSV lookup
- If rate limited → Return raw data
- Section/faculty queries → Return matching CSV

---

### 5.3 LLM Wrapper (`llm_wrapper.py` - 105 lines)

**Purpose:** Groq API abstraction with retry logic

**Features:**
```python
# Primary model: llama-3.3-70b-versatile
# Fallback model: llama-3.1-8b-instant

# Retry strategy:
#   - Rate limit (429): Exponential backoff (2s, 4s, 8s)
#   - Model not found (404): Switch to fallback
#   - Overload (503): Wait and retry
#   - Invalid key (401): Fatal error
```

---

### 5.4 Prompt Builder (`prompt_builder.py` - 101 lines)

**Purpose:** Dynamic system prompt generation

**Strategy:**
- Read live data from CSVs (never hardcoded)
- Compact format (<800 tokens)
- Include: courses, faculty, lab schedules, weekly caps
- Multi-semester aware

---

### 5.5 Agent (`agent.py` - 1,154 lines)

**Purpose:** LangChain ReAct agent with tool-use capabilities

**16 Tools:**

| # | Tool | Type | Description |
|---|------|------|-------------|
| 1 | `get_section_timetable` | READ | Full weekly timetable for section |
| 2 | `get_faculty_schedule` | READ | Full weekly schedule for faculty |
| 3 | `find_free_slots` | READ | Free periods for faculty on day |
| 4 | `get_summary_stats` | READ | Quality metrics from summary |
| 5 | `find_substitute` | READ | Substitute suggestions |
| 6 | `get_faculty_workload` | READ | Total hours + day breakdown |
| 7 | `get_free_faculty` | READ | All free faculty at slot |
| 8 | `get_room_availability` | READ | Free/occupied rooms |
| 9 | `find_free_rooms` | READ | Available classrooms |
| 10 | `get_absent_periods` | READ | Periods faculty teaches |
| 11 | `detect_schedule_conflicts` | READ | Scan for double-bookings |
| 12 | `get_weekly_stats` | READ | Key statistics |
| 13 | `commit_substitute` | WRITE | Apply substitute to timetable |
| 14 | `list_agent_ops` | WRITE | View operation history |
| 15 | `rollback_last_operation` | WRITE | Undo committed change |
| 16 | `generate_session_summary` | WRITE | Export session report |

---

### 5.6 Agent Operations (`agent_ops.py` - 136 lines)

**Purpose:** Audit logging and rollback

**Log Entry Structure:**
```json
{
  "operation_id": "a3f2b1c4",
  "timestamp_utc": "2024-01-15T09:30:00Z",
  "action": "substitute",
  "absent_faculty": "F03",
  "section_id": "A",
  "day": "Monday",
  "period_range": [5, 6],
  "substitute_faculty": "F07",
  "pre_state": "...CSV before...",
  "post_state": "...CSV after...",
  "backup_path": "outputs/agent_ops/backups/..."
}
```

---

### 5.7 Substitute Finder (`substitute.py` - 613 lines)

**Purpose:** Find replacement faculty for absent teachers

**3-Tier Priority Ranking:**
```python
# P1: Teaches SAME COURSE (any section) + free + under cap
# P2: Teaches SAME SECTION (any course) + free + under cap  
# P3: Any free faculty + under cap
```

**Lab Block Handling:**
```python
# P5-P6 treated as atomic 2-period block
# Substitute must be free for BOTH periods
```

---

### 5.8 Swap Planner (`swap.py` - 219 lines)

**Purpose:** Load restoration - find slots where original faculty can reclaim

**Algorithm:**
```python
# For returning faculty A, substitute B:
#   Search days after return_day:
#     Find slot where:
#       - B teaches A's course
#       - A is free at that slot
#     → Swap possible
```

---

### 5.9 Sync Manager (`sync_manager.py` - 562 lines)

**Purpose:** Atomic write-back for all changes

**Atomic Commit Flow:**
```
1. Create timestamped backup directory
2. Backup: section CSV, both faculty CSVs, summary_report.txt
3. Patch section CSV (add →F07 annotation)
4. Rebuild faculty CSVs (scan all sections)
5. Rebuild summary_report.txt
6. Rebuild RAG index (best-effort)
7. Log operation

ON ANY FAILURE:
  → Restore all files from backup
  → Raise RuntimeError with details
```

---

### 5.10 Chat Interface (`chat.py` - 130 lines)

**Purpose:** CLI chatbot with persistent memory

**Features:**
- Persistent history: `outputs/chat_memory.json`
- Keep last 10 exchanges on disk
- Commands: `quit`, `clear`, `history`
- Auto-initial analysis on startup

---

## Support Files

### Utils (`utils/`)

**File:** `health_check.py` (227 lines)

**Checks:**
```python
✓ Required input CSVs (courses, faculty, assignments, rooms, lab_allotment)
✓ Required output files (12 section CSVs, summary, room_assignment)
✓ RAG index files (rag_index.faiss, rag_docs.json)
✓ Environment (GROQ_API_KEY set)
✓ Python packages (langchain-groq, sentence-transformers, faiss-cpu, ortools, streamlit, altair)
```

---

### Tests (`tests/`)

**File:** `test_pipeline.py` (267 lines)

**Test Coverage:**
- Phase 0: Data loads, validation passes, section coverage, no prof-on-lab
- Phase 2: Lab slots locked, no conflicts
- Phase 3: Solver optimal, theory slot counts, room assignments

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

INPUTS (data/{sem}/)                          OUTPUTS (outputs/{sem}/)
┌──────────────┐                              ┌──────────────────────────┐
│ courses.csv  │──┐                           │ section_A_timetable.csv  │
│ (5 courses)  │  │    ┌─────────────┐          │ ... section_L_timetable  │
├──────────────┤  ├───▶│   Phase 0   │──┐       ├──────────────────────────┤
│ faculty.csv  │  │    │ Validation  │  │       │ faculty_F01_timetable    │
│ (20 faculty) │──┘    └─────────────┘  │       │ ... faculty_F20          │
├──────────────┤                        │       ├──────────────────────────┤
│assignments   │──┐    ┌─────────────┐  │       │ room_assignment.csv      │
│.csv          │  ├───▶│   Phase 1   │──┤       │ (all theory slots)       │
├──────────────┤  │    │Assignment   │  │       ├──────────────────────────┤
│lab_allotment │──┤    │   Map       │  │       │ summary_report.txt       │
│.csv          │  │    └─────────────┘  │       │ (quality metrics)        │
├──────────────┤  │         │           │       ├──────────────────────────┤
│rooms.csv     │──┘         ▼           │       │ rag_index.faiss          │
│(12 rooms)    │       ┌─────────────┐ │       │ rag_docs.json            │
└──────────────┘       │   Phase 2   │ │       │ (vector embeddings)      │
                       │ Lock Labs   │─┘       ├──────────────────────────┤
                       └─────────────┘         │ agent_ops/               │
                            │                  │ (operation logs)         │
                            ▼                  └──────────────────────────┘
                       ┌─────────────┐
                       │   Phase 3   │
                       │ CP-SAT      │
                       │ Theory      │
                       │ Scheduler   │
                       └─────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ section  │  │ faculty  │  │  room    │
        │  _grid   │  │  _grid   │  │  _grid   │
        └──────────┘  └──────────┘  └──────────┘
```

---

## Key Algorithms

### 1. CP-SAT Theory Scheduling

```python
# Variables
x[course, section, day, period] ∈ {0, 1}
room[course, section, day, period] ∈ Rooms

# Hard Constraints
∀s,d,p: sum(x[c,s,d,p] for c in courses) ≤ 1           # One course per slot
∀f,d,p: sum(x[c,s,d,p] for c,s where faculty[c,s]=f) ≤ 1 # No faculty conflicts
∀r,d,p: sum(x[c,s,d,p] where room[c,s,d,p]=r) ≤ 1       # No room conflicts

# Soft Constraints (objective weights)
minimize: PENALTY_GAP * gaps 
        + PENALTY_LAB_WINDOW * theory_in_p5_p6
        + PENALTY_BACK_TO_BACK * back_to_back_same_course
        + PENALTY_SAME_DAY * same_course_same_day
        - REWARD_CONSECUTIVENESS * consecutive_pairs
```

### 2. Substitute Ranking Algorithm

```python
def rank_candidates(absent_faculty, day, period):
    candidates = []
    
    for faculty in all_faculty:
        if faculty == absent_faculty:
            continue
            
        # Check availability
        if not is_free(faculty, day, period):
            continue
            
        # Check load cap
        if projected_load(faculty) > max_hours(faculty):
            continue
            
        # Determine match tier
        if teaches_same_course(faculty, course):
            tier = 1  # Same course
        elif teaches_same_section(faculty, section):
            tier = 2  # Same section
        else:
            tier = 3  # Available only
            
        candidates.append((faculty, tier, projected_load))
    
    # Sort by tier, then by load (lower = better)
    return sorted(candidates, key=lambda x: (x[1], x[2]))
```

### 3. RAG Retrieval with Metadata Filtering

```python
def retrieve(query, k=5):
    # 1. Extract filters
    filters = {
        "section": extract_section(query),    # e.g., "A"
        "day": extract_day(query),            # e.g., "Monday"
        "faculty": extract_faculty(query),    # e.g., "F03"
        "source_type": extract_source(query)  # "section" | "faculty" | "room"
    }
    
    # 2. Build candidate set
    candidates = [doc for doc in docs if matches_filters(doc, filters)]
    
    # 3. Embed query
    q_emb = model.encode(query)
    
    # 4. Search (sub-index if filters found, else full index)
    if candidates and len(candidates) >= k*3:
        results = search_subindex(q_emb, candidates, k*3)
    else:
        results = search_full_index(q_emb, k*3)
    
    # 5. Rerank with cross-encoder
    pairs = [(query, doc.text) for doc in results]
    scores = cross_encoder.predict(pairs)
    
    # 6. Return top k
    return sorted(zip(scores, results), reverse=True)[:k]
```

---

## Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Solver** | Google OR-Tools CP-SAT | Constraint programming |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Vector encoding |
| **Vector DB** | FAISS (IndexFlatL2) | Semantic search |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Result ranking |
| **LLM** | Groq API (Llama 3.3 70B) | Natural language |
| **LLM Framework** | LangChain | Agent/tools |
| **UI** | Streamlit | Web dashboard |
| **Viz** | Altair | Charts/heatmaps |
| **Data** | pandas | CSV processing |
| **Testing** | pytest | Automated tests |

---

## Multi-Semester Architecture

```
data/
├── cse_sem3/
│   ├── courses.csv       (5 courses, no electives)
│   ├── faculty.csv       (20 faculty)
│   ├── assignments.csv
│   ├── rooms.csv
│   └── lab_allotment.csv
│
└── cse_sem5/
    ├── courses.csv       (6 courses, 2 electives)
    ├── faculty.csv       (25 faculty)
    ├── assignments.csv
│   ├── rooms.csv
    ├── lab_allotment.csv
    └── elective_slots.csv  (extra file)

outputs/
├── cse_sem3/
│   ├── section_A_timetable.csv ... section_L_timetable.csv
│   ├── faculty_F01_timetable.csv ...
│   ├── room_assignment.csv
│   ├── summary_report.txt
│   ├── rag_index.faiss
│   └── rag_docs.json
│
└── cse_sem5/
    └── ... (same structure)
```

---

## Configuration Constants (config.py)

| Constant | Value | Description |
|----------|-------|-------------|
| `SECTIONS` | A-L (12) | Total sections |
| `DAYS` | Mon-Fri | Week days |
| `PERIODS` | 1-6 | Periods per day |
| `THEORY_PERIODS` | 1-4 | Theory window |
| `LAB_PERIODS` | 5-6 | Lab window |
| `MAX_HOURS` | Prof=12, Asso=16, Asst=20 | Weekly caps |
| `PENALTY_GAP` | 100 | No gaps penalty |
| `PENALTY_LAB_WINDOW` | 50 | No theory in P5-P6 |
| `PENALTY_BACK_TO_BACK` | 5 | Same course back-to-back |
| `PENALTY_SAME_DAY` | 3 | Same course twice/day |
| `REWARD_CONSECUTIVENESS` | 200 | Preferred: consecutive |
| `NUM_WORKERS` | 8 | CP-SAT parallel workers |
| `PENALTY_STAGE_TIME` | 180 | Stage 2 time limit (sec) |
| `REWARD_STAGE_TIME` | 90 | Stage 1 time limit (sec) |

---

## Entry Points & Usage

### 1. Generate Timetable
```bash
python run_all.py --sem cse_sem3
```

### 2. Launch Dashboard
```bash
streamlit run app.py
```

### 3. AI Agent CLI
```bash
python src/phase5/agent.py
```

### 4. Interactive Chat
```bash
python src/phase5/chat.py
```

### 5. Health Check
```bash
python utils/health_check.py --sem cse_sem3
```

### 6. Run Tests
```bash
pytest tests/ -v
```

---

## Quality Metrics

The system computes a quality score based on:

```
score = 100 - ((same_day + 2*back_to_back + 5*overload_count) / 2)
```

Where:
- `same_day`: Same course taught twice on same day (bad)
- `back_to_back`: Same course in consecutive periods (bad)
- `overload_count`: Faculty exceeding weekly cap (very bad)

**Target:** Score ≥ 90 indicates good timetable quality

---

## File Dependencies Graph

```
run_all.py
    ├── phase0/validator.py ◄── phase0/loader.py
    ├── phase1/assignment_builder.py
    ├── phase2/lab_scheduler.py
    ├── phase3/theory_scheduler.py ◄── ortools
    ├── phase3_5/room_allocator.py
    ├── phase4/output_generator.py
    └── phase5/rag_indexer.py ◄── sentence-transformers, faiss

app.py
    ├── config.py
    ├── utils/health_check.py
    └── phase5/
        ├── ai_explainer.py
        ├── agent.py
        ├── substitute.py
        └── rag_indexer.py

agent.py
    ├── agent_ops.py
    ├── substitute.py
    ├── swap.py
    └── sync_manager.py ◄── atomic operations
```

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | ~25 |
| Total Lines of Code | ~8,000+ |
| Test Coverage | Phase 0-3 |
| AI Tools | 16 |
| RAG Documents | ~732 per semester |
| CP-SAT Variables | ~10,000+ |
| Constraint Types | 6 hard + 6 soft |
| Supported Semesters | Unlimited (isolated) |
| Max Sections Tested | 12 |

---

## Known Limitations (from README)

1. **Enrollment data** - Section size fixed at 60 (no enrollment CSV)
2. **Single-building** - Travel time not modeled
3. **Groq rate limits** - Graceful fallback to data-driven responses
4. **Solve time** - Scales super-linearly with section count
5. **Cross-encoder download** - ~90MB model on first use

---

*Generated: April 6, 2026*
*System: GenAI Timetable Generation v2.0*
