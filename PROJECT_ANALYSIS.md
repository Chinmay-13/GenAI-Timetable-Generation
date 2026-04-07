# Project Analysis: AI-Powered Academic Timetable Generation System

> **Document Type**: Comprehensive Technical Analysis  
> **Generated**: April 7, 2026  
> **Project Root**: `c:\admin\3rd year\sem6\E3_genai\timetable_system`  
> **Lines of Code Analyzed**: ~8,000+  
> **Purpose**: Detailed explanation of all phases for presentation

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [Configuration (config.py)](#2-configuration-configpy)
3. [Phase 0: Input Validation](#3-phase-0-input-validation)
4. [Phase 1: Assignment Builder](#4-phase-1-assignment-builder)
5. [Phase 2: Lab Scheduler](#5-phase-2-lab-scheduler)
6. [Phase 3: Theory Scheduler (CP-SAT)](#6-phase-3-theory-scheduler-cp-sat)
7. [Phase 3.5: Room Allocator](#7-phase-35-room-allocator)
8. [Phase 4: Output Generator](#8-phase-4-output-generator)
9. [Phase 5: AI Layer](#9-phase-5-ai-layer)
   - 9.1 [RAG Indexer](#91-rag-indexer)
   - 9.2 [AI Explainer](#92-ai-explainer)
   - 9.3 [Substitute Finder](#93-substitute-finder)
   - 9.4 [Agent with 16 Tools](#94-agent-with-16-tools)
   - 9.5 [Sync Manager](#95-sync-manager)

---

## 1. Executive Overview

This is an **AI-powered academic timetable generation system** for university CSE departments. It combines **Google OR-Tools CP-SAT** constraint programming with **modern AI capabilities** (LLM-powered RAG, LangChain agents).

### Core Capabilities
- **Automated Timetable Generation**: Schedules theory and lab sessions for 12 sections (A-L) across 5 days
- **Conflict-Free Scheduling**: Uses CP-SAT solver with hard/soft constraints
- **Integrated Room Assignment**: Assigns classrooms inside the solver
- **AI Assistant**: Natural language Q&A via Groq LLM + FAISS RAG
- **Autonomous Agent**: 16 tools for schedule modifications with atomic rollback

### Pipeline Flow
```
Input CSVs → Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 3.5 → Phase 4 → Phase 5
             Validate   Build     Lock      CP-SAT    Room       Export    AI
                       Map       Labs      Theory    Assign              Ready
```

---

## 2. Configuration (config.py)

This file defines **ALL constants** that every phase uses. It's the single source of truth.

### 2.1 Core Schedule Structure
```python
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PERIODS = [1, 2, 3, 4, 5, 6]                    # P1-P6 daily
THEORY_PERIODS = [1, 2, 3, 4]                   # P1-P4 = theory
LAB_PERIODS = [5, 6]                            # P5-P6 = labs (2-period block)
SECTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
SECTION_SIZE = 60                               # Assumed students per section
```

### 2.2 Faculty Workload Caps
```python
MAX_HOURS = {
    "Prof": 12,        # Professor
    "Asso Prof": 16,   # Associate Professor
    "Asst Prof": 20    # Assistant Professor
}
```

**Used by:**
- Phase 0 validator to check if faculty are overloaded
- Phase 3 CP-SAT as hard constraints

### 2.3 CP-SAT Penalty Weights (Soft Constraints)

| Penalty | Value | Meaning |
|---------|-------|---------|
| `PENALTY_GAP` | 100 | Intra-day gaps (highest priority soft) |
| `PENALTY_LAB_WINDOW` | 50 | Theory scheduled in P5-P6 on non-lab days |
| `PENALTY_BACK_TO_BACK` | 5 | Same subject consecutive periods |
| `PENALTY_SAME_DAY` | 3 | Same subject multiple times same day |
| `PENALTY_PREF_TIME` | 8 | Wrong morning/afternoon preference |
| `PENALTY_PREF_NO_BTB` | 6 | Consecutive when faculty wants gaps |
| `PENALTY_PREF_FREE_DAY` | 10 | Teaching on preferred free day |
| `REWARD_CONSECUTIVENESS` | 200 | Reward for consecutive same-subject |

**What These Mean:**

**`PENALTY_GAP = 100`** (Intra-day gaps)
- **What it means:** The solver tries to avoid "free periods in the middle of the day"
- **Example BAD:** P1=Math, P2=FREE, P3=CSE, P4=FREE → Gap at P2
- **Example GOOD:** P1=Math, P2=CSE, P3=FREE, P4=FREE → Free slots at END
- **Why 100?** Highest penalty. The solver loses 100 points for every gap, so it really tries to avoid them.

**`REWARD_CONSECUTIVENESS = 200`** (Consecutive same-subject)
- **What it means:** The solver REWARDS when same course happens in consecutive periods
- **Example GOOD (gets +200 reward):** P1=CNS, P2=CNS (same course back-to-back)
- **Example NORMAL (no reward):** P1=CNS, P2=OS (different courses)
- **Why?** 2-hour blocks for the same subject are better for learning continuity
- **Note:** Even though `PENALTY_BACK_TO_BACK = 5` penalizes same course consecutive, `REWARD_CONSECUTIVENESS = 200` is much stronger - it actually WANTS them consecutive.

### 2.4 Solver Time Limits
```python
PENALTY_STAGE_TIME = 180    # 3 minutes - minimize all penalties
REWARD_STAGE_TIME = 90      # 1.5 minutes - maximize rewards
CONSECUTIVENESS_TIME_LIMIT = 30  # Try hard constraint for consecutiveness
NUM_WORKERS = 8             # CP-SAT parallel workers
```

**Two-Stage Solving:**
1. **Stage 1 (180s):** Minimize all penalties (gaps, conflicts, preferences). Must find "optimal" or "close to optimal".
2. **Stage 2 (90s):** Maximize rewards with frozen penalty score. Fix the penalty score, then improve rewards.

**Why time limits?** Without them, the solver could run forever trying to find "perfect". We stop it at "good enough" for real-world use.

### 2.5 Multi-Semester Path System
```python
@dataclass(frozen=True)
class SemesterPaths:
    sem_id: str
    data_dir: Path          # data/{sem_id}/
    output_dir: Path        # outputs/{sem_id}/
    rag_index_path: Path     # outputs/{sem_id}/rag_index.faiss
    rag_docs_path: Path      # outputs/{sem_id}/rag_docs.json
    chat_memory_path: Path   # outputs/{sem_id}/chat_memory.json
    agent_ops_dir: Path      # outputs/{sem_id}/agent_ops/

def get_sem_paths(sem_id: str) -> SemesterPaths:
    # Returns all resolved paths for a semester
    
def list_available_semesters() -> list[str]:
    # Auto-discovers semesters from data/ folder
```

### 2.6 LLM Configuration
```python
GROQ_MODEL = "llama-3.3-70b-versatile"      # Primary LLM
GROQ_MODEL_ALT = "llama-3.1-8b-instant"    # Fallback (faster)
MAX_CHAT_HISTORY = 4                        # Memory window (last 4 exchanges)
```

**What is `MAX_CHAT_HISTORY = 4`?**
- The AI assistant only remembers the last 4 exchanges (8 messages: 4 user + 4 AI)
- Allows follow-up questions: "When is F04 free?" → "What about Tuesday?" (understands "F04" from context)

### 2.7 Course Credits to Periods
```python
CREDIT_MAP = {
    5: {"theory_periods": 4, "lab_sessions": 1},  # if has_lab=True
    4: {"theory_periods": 4, "lab_sessions": 0},
}
```

---

## 3. Phase 0: Input Validation

**Files:** `loader.py` (252 lines) + `validator.py` (138 lines)

**Purpose:** Check the 5 input CSVs for errors BEFORE spending time generating timetables. Garbage in = garbage out.

### 3.1 Loader (`loader.py`)

**Key Functions:**
- `load_semester(sem_id)` → Loads all CSVs for a semester
- `load_elective_slots(sem_id)` → Optional elective data
- `get_sem_metadata(sem_id)` → Summary for UI display

**What it does:**
1. Reads `courses.csv`, `faculty.csv`, `assignments.csv`, `rooms.csv`, `lab_allotment.csv`
2. Enriches course data with `is_elective` boolean flag
3. Returns DataFrames ready for validation

### 3.2 Validator (`validator.py`)

**Function:** `validate(courses, faculty, assignments, data_dir) → bool`

**The 4 Validation Checks:**

| Check | What It Validates | Why It Matters |
|-------|-------------------|----------------|
| **1. Section Coverage** | Core courses must have exactly 12 sections assigned | Ensures every student gets every core course |
| **2. Faculty Course Count** | No faculty assigned to more than 2 courses | Prevents overload, respects human limits |
| **3. Professor Lab Rule** | Professors (Prof) cannot teach lab courses | Lab courses need hands-on supervision, typically handled by junior faculty |
| **4. Weekly Hour Load** | Faculty hours ≤ MAX_HOURS[designation] | Prof=12, Asso=16, Asst=20. Calculates: theory(credits × sections) + labs(1 × 2 periods) |

**Example Calculation for Check 4:**
```
Faculty F04 (Asso Prof, cap=16):
- Teaches CNS to sections A,B (2 sections × 4 credits = 8 theory hours)
- Teaches DMGT to section A (1 section × 4 credits = 4 theory hours)
- Teaches CNS lab to A,D pair (1 lab × 2 periods = 2 lab hours)
Total = 8 + 4 + 2 = 14 hours ≤ 16 ✓ PASS
```

**Output:** Returns `True` if all pass, `False` if any fail + prints exact errors.

**For Presentation:** "Phase 0 is our safety net. It catches data entry errors before we waste 3 minutes running the CP-SAT solver on bad data."

---

## 4. Phase 1: Assignment Builder

**File:** `assignment_builder.py` (123 lines)

**Purpose:** Build the master mapping of **who teaches what to whom**. This is the foundation everything else builds on.

### 4.1 Core Data Structure

```python
assignment_map[course_code][section] = faculty_id
```

**Example:**
```python
{
    "UE24CS351A": {  # CNS - Computer Networks
        "A": "F04",   # F04 teaches CNS to section A
        "B": "F04",   # F04 also teaches CNS to section B
        "C": "F05",   # F05 teaches CNS to section C
        "D": "F05",
        "E": "F06",
        ...
        "L": "F09"
    },
    "UE24CS352A": {  # OS - Operating Systems
        "A": "F10",
        "B": "F10",
        ...
    }
}
```

### 4.2 What It Does

1. **Load CSVs** from `data/{sem}/`: courses, faculty, assignments
2. **Enrich course data**: Convert `is_elective` string to boolean
3. **Build the map**: Parse `assignments.csv` where `sections_handled` = "A,B,C"
4. **Validate coverage**:
   - Core courses: Must have ALL 12 sections (A-L)
   - Elective courses: Can have partial coverage (1+ sections)
5. **Error collection**: Gather all errors, raise `ValueError` if any found

### 4.3 Validation Errors It Catches

| Error | Example |
|-------|---------|
| Duplicate section assignment | Section A assigned to both F04 and F05 for same course |
| Invalid section code | Section "Z" doesn't exist (only A-L) |
| Core course under-covered | CNS only has 10 sections, missing K and L |
| Empty sections_handled | Faculty assigned to course but no sections listed |

### 4.4 Why This Matters

**This map is the SINGLE SOURCE OF TRUTH for:**
- Phase 2: Knowing which faculty to lock for labs
- Phase 3: Knowing who teaches what for conflict detection
- Phase 4: Writing faculty names in output CSVs
- Phase 5: Finding substitutes who already know the course

**For Presentation:** "Phase 1 creates a lookup table: for any course and section, we instantly know which faculty is responsible. This prevents double-booking faculty and ensures every section has a teacher assigned."

---

## 5. Phase 2: Lab Scheduler

**File:** `lab_scheduler.py` (245 lines)

**Purpose:** Lock the **fixed slots** (labs and electives) BEFORE theory scheduling. Labs happen at specific times and can't move.

### 5.1 Input Files Used

| File | What It Provides |
|------|------------------|
| `lab_allotment.csv` | Lab schedule: day, course, section pairs, room, faculty |
| `elective_slots.csv` (sem5 only) | Elective schedule: day, periods, room, faculty, enrolled sections |

### 5.2 Core Concept: 3 Grids (3D Dictionaries)

**Think of it as 3 different perspectives of the SAME timetable:**

| Grid | Perspective | Answers |
|------|-------------|---------|
| `section_grid` | Student view | "What is Section A doing on Monday P3?" |
| `faculty_grid` | Teacher view | "Where is Faculty F04 teaching on Monday P3?" |
| `room_grid` | Room view | "Is Room_G03 occupied on Monday P3?" |

**Example after locking a lab:**

Input from `lab_allotment.csv`:
```
Monday, UE24CS351A, [A,D], Lab_Room_1, F04
```

**After locking:**
```python
# SECTION view: What are students doing?
section_grid["A"]["Monday"][5] = "UE24CS351A_LAB"
section_grid["A"]["Monday"][6] = "UE24CS351A_LAB"
section_grid["D"]["Monday"][5] = "UE24CS351A_LAB"
section_grid["D"]["Monday"][6] = "UE24CS351A_LAB"

# FACULTY view: Where is F04?
faculty_grid["F04"]["Monday"][5] = "UE24CS351A"
faculty_grid["F04"]["Monday"][6] = "UE24CS351A"

# ROOM view: Is Lab_Room_1 free?
room_grid["Lab_Room_1"]["Monday"][5] = "UE24CS351A"
room_grid["Lab_Room_1"]["Monday"][6] = "UE24CS351A"
```

### 5.3 Conflict Detection

**Scenario: Try to book another lab at same time:**
```
Monday, UE24CS352A, [A,B], Lab_Room_2, F04   # ← F04 already busy!
```

**Check happens in Phase 2:**
```python
if faculty_grid["F04"]["Monday"][5] is not None:
    raise ValueError("CONFLICT: F04 already assigned at Monday P5")

if room_grid["Lab_Room_2"]["Monday"][5] is not None:
    raise ValueError("Room already occupied")

if section_grid["A"]["Monday"][5] is not None:
    raise ValueError("Section A already has a class")
```

### 5.4 How Grids Flow Through Pipeline

```
Phase 2 → Phase 3 → Phase 4
   ↓         ↓         ↓
Creates   Fills      Writes
empty     theory     to CSVs
grids     slots      (from grids)
```

**Phase 3 (CP-SAT)** fills empty slots:
```python
# Before Phase 3:
section_grid["A"]["Monday"][1] = None  # Empty

# After Phase 3:
section_grid["A"]["Monday"][1] = "UE24CS342A"  # SE class assigned
```

**Phase 4** reads grids to generate CSVs.

### 5.5 Output (5 items)

```python
return (
    section_grid,      # What's in each section's slots
    faculty_grid,      # What's each faculty teaching
    room_grid,         # What's each room hosting
    lab_details,       # List of locked lab info
    elective_details   # List of locked elective info
)
```

**For Presentation:** "Phase 2 is like 'penciling in' the immovable commitments. Labs have fixed rooms and faculty, so we lock them first. The 3 grids act as a shared calendar that Phase 3 will fill the remaining empty slots into."

---

## 6. Phase 3: Theory Scheduler (CP-SAT)

**File:** `theory_scheduler.py` (932 lines)

**Purpose:** The **core scheduling module** using Google OR-Tools CP-SAT solver.

### 6.1 What is CP-SAT?

**CP-SAT** = **C**onstraint **P**rogramming - **SAT**isfiability

- **SAT part:** Determines if there's ANY valid solution (satisfiable = possible)
- **CP part:** Among all valid solutions, finds the BEST one (optimization)

**Google OR-Tools:** Google's open-source library for optimization problems (C++ with Python bindings).

**The Method: Constraint Satisfaction Problem (CSP)**

| Component | Our Timetable Example |
|-----------|----------------------|
| **Variables** | `x[(A, CNS, Monday, 1)]` = 0 or 1 |
| **Domains** | {0, 1} (binary - scheduled or not) |
| **Constraints** | Sum ≤ 1 (no conflicts), Sum = target (exact periods needed) |
| **Objective** | Minimize penalties + Maximize rewards |

### 6.2 Why This Method for Timetables?

| Challenge | How CSP Solves It |
|-----------|-----------------|
| Thousands of possibilities | Solver tries millions/second |
| Hard rules (no conflicts) | Constraints are **ABSOLUTE** |
| Soft preferences (faculty likes) | Penalties guide toward better solutions |
| Multiple goals | Two-stage: first feasible, then optimal |

### 6.3 Simple Conflict Example

**Scenario:**
- Section A needs CNS (4 periods/week) and OS (4 periods/week)
- F04 teaches CNS to section A
- F10 teaches OS to section A

**Problem:** Both CNS and OS want Monday P1.

**Without Phase 3 (BAD):**
```
Section A on Monday:
P1: CNS (F04) ← Booked
P1: OS (F10)  ← ALSO BOOKED! ❌ CONFLICT!
```

**With Phase 3 CP-SAT (GOOD):**

**Decision Variables Created:**
```python
x[("A", "CNS", "Monday", 1)] = BoolVar  # 0 or 1
x[("A", "OS", "Monday", 1)] = BoolVar   # 0 or 1
```

**Hard Constraint Applied:**
```python
# For Section A on Monday P1:
model.Add(
    x[("A", "CNS", "Monday", 1)] + 
    x[("A", "OS", "Monday", 1)] 
    <= 1
)
# Translation: "At most ONE of these can be 1 (scheduled)"
```

**What Solver Does:**
```
Try: CNS at P1=1, OS at P1=1 → Sum=2 → VIOLATES constraint ❌
Try: CNS at P1=1, OS at P1=0 → Sum=1 → OK ✓
Try: CNS at P1=0, OS at P1=1 → Sum=1 → OK ✓
Try: Both at P1=0 → Sum=0 → OK (but then where do they go?)
```

**Result:** Solver picks different periods:
```
Section A on Monday:
P1: CNS (F04)  ← Scheduled
P2: OS (F10)   ← Scheduled
P3-P4: FREE
P5-P6: CNS LAB (locked from Phase 2)
```

### 6.4 Same Check for Faculty

**What if F04 teaches CNS to both Section A AND Section B?**
```python
# F04 can't teach 2 sections at same time
model.Add(
    x[("A", "CNS", "Monday", 1)] + 
    x[("B", "CNS", "Monday", 1)] 
    <= 1
)
```

### 6.5 Summary of Constraints

| Conflict Type | Constraint |
|---------------|------------|
| Section double-booking | Sum of courses in slot ≤ 1 |
| Faculty double-booking | Sum of faculty's courses in slot ≤ 1 |
| Room double-booking | Sum of room assignments ≤ 1 |

### 6.6 Faculty Hour Cap Constraint

```python
# From lines 386-399 in theory_scheduler.py:
for faculty_id in faculty_ids:
    all_faculty_vars = []
    for course in scheduled_course_codes:
        for section, assigned_faculty in assignment_map.get(course, {}).items():
            if assigned_faculty != faculty_id:
                continue
            for day in DAYS:
                for period in PERIODS:
                    key = (section, course, day, period)
                    if key in x:
                        all_faculty_vars.append(x[key])
    if all_faculty_vars:
        desig = faculty_designation.get(faculty_id, "Asst Prof")
        cap = MAX_HOURS.get(desig, 20)
        model.Add(sum(all_faculty_vars) <= cap)
```

**What this does:**
- Collect ALL variables (slots) for each faculty
- Get their designation (Prof/Asso/Asst)
- Apply hard constraint: Total assigned slots ≤ MAX_HOURS

### 6.7 Pre-defined Constraints + Our Values

**Google OR-Tools provides the constraint types:**
- `model.Add()` → Linear constraints
- `model.NewBoolVar()` → Binary variables (0 or 1)
- `model.Minimize()` → Optimization objective

**We provide:**
- The **variables** (which slots to consider)
- The **coefficients** (penalty weights from config.py)
- The **rules** (sum ≤ 1, sum ≥ target, etc.)

**Example:**
```python
# OR-TOOLS provides: "Add a linear constraint"
model.Add(                          # ← Library function
    sum(all_faculty_vars) <= cap    # ← Our rule
)
```

### 6.8 Two-Stage Solving

```python
# STAGE 1: Penalty Minimization (180 seconds)
model.Minimize(sum(penalty_terms))  # Minimize gaps, conflicts, preferences
solver.Solve(model)                  # OR-Tools finds "good enough"

# STAGE 2: Reward Maximization (90 seconds)
best_penalty = solver.ObjectiveValue()
model.Add(sum(penalty_terms) == best_penalty)  # Freeze penalty score
model.Maximize(sum(reward_terms))               # Maximize consecutiveness
solver2.Solve(model)                            # Improve rewards
```

**Think of it like:**
1. **Stage 1:** "Find a schedule with NO hard conflicts and minimal soft violations"
2. **Stage 2:** "Keep that quality, but also make subjects consecutive when possible"

### 6.9 Why CP-SAT vs Other Methods?

| Method | Why We Didn't Use It |
|--------|----------------------|
| Genetic Algorithm | Slow, might still create conflicts |
| Simulated Annealing | Random, no guarantee of optimal |
| Simple Greedy | Would definitely create double-bookings |
| Manual scheduling | Takes weeks, humans make mistakes |

**CP-SAT was chosen because:** It mathematically PROVES no conflicts exist while optimizing for preferences.

### 6.10 Key Technical Terms

| Term | Meaning |
|------|---------|
| **Constraint** | A rule that must be followed |
| **Variable** | A decision (0 or 1 for each slot) |
| **Domain** | Possible values (binary = {0,1}) |
| **Objective Function** | What to minimize/maximize |
| **Feasible Solution** | Any schedule with no conflicts |
| **Optimal Solution** | Best schedule (lowest penalties) |

**For Presentation:** "We use a mathematical technique called Constraint Programming with SAT solving, implemented via Google's OR-Tools library. Instead of trial-and-error, it treats timetabling as a system of equations with 10,000+ variables and solves them simultaneously to guarantee zero conflicts while optimizing faculty preferences."

---

## 7. Phase 3.5: Room Allocator

**File:** `room_allocator.py` (359 lines)

**Purpose:** Assign actual **classroom names** to every theory slot after CP-SAT scheduling is done.

### 7.1 Why Separate Phase?

- **Phase 3 (CP-SAT):** Decided **WHEN** classes happen
- **Phase 3.5:** Decides **WHERE** they happen

### 7.2 Two Paths

| Path | When Used | How It Works |
|------|-----------|--------------|
| **CP-SAT Path** | `room_assignment_map` from Phase 3 exists | Use solver's room choices directly |
| **Greedy Path** | Fallback | Sort sections alphabetically, assign first available room |

### 7.3 The Greedy Algorithm (Fallback)

```python
for each section (A, B, C, ... L):
    for each day (Mon-Fri):
        for each theory period (P1-P4):
            if slot has a class:
                find first CLASSROOM (not LAB) that is:
                    - Free at this time
                    - Capacity >= SECTION_SIZE (60)
                assign room to theory_room_grid[section][day][period]
```

**Sorting priority:** Rooms by capacity (largest first), then alphabetically.

### 7.4 Output

```python
theory_room_grid[section][day][period] = room_name
# Example:
theory_room_grid["A"]["Monday"][1] = "Room_G03"
```

**Also generates:** `room_assignment.csv` with columns: Section, Day, Period, Course, Faculty, Room

### 7.5 Why Different from Labs?

| Aspect | Labs | Theory Classes |
|--------|------|----------------|
| Room type | LAB only (fixed) | CLASSROOM (12 options) |
| Assignment | Hard-coded in CSV | Solver or algorithm decides |
| Periods | P5-P6 only | P1-P4 flexible |

**For Presentation:** "Phase 3.5 fills in the 'WHERE.' Labs were locked to specific rooms in Phase 2, but theory classes need dynamic assignment. We either use the CP-SAT solver's suggestions or fall back to a greedy algorithm that assigns the first available classroom to each section."

---

## 8. Phase 4: Output Generator

**File:** `output_generator.py` (281 lines)

**Purpose:** Convert the solved grids into **human-readable CSVs** and a **summary report**.

### 8.1 Inputs Received

| Input | What It Contains |
|-------|------------------|
| `section_grid` | All 12 sections × 5 days × 6 periods |
| `faculty_grid` | All faculty × 5 days × 6 periods |
| `theory_room_grid` | Room assignments from Phase 3.5 |
| `assignment_map` | Who teaches what |
| `courses` | Course metadata (names, short names) |
| `lab_details` | Lab schedule info |
| `elective_details` | Elective schedule info |

### 8.2 Outputs Generated

| File | Count | Example Content |
|------|-------|-----------------|
| `section_{A-L}_timetable.csv` | 12 files | `section_A_timetable.csv`: Day, P1-P6 with cells like "CNS (F04)" |
| `faculty_{F01-F20}_timetable.csv` | 20 files | `faculty_F04_timetable.csv`: Shows where F04 teaches each period |
| `room_assignment.csv` | 1 file | Maps every slot to a specific room |
| `summary_report.txt` | 1 file | Quality metrics, violation counts, faculty load table |

### 8.3 Cell Format Examples

**Section CSV:**
```
Day    | P1      | P2     | P3     | P4     | P5          | P6
Monday | CNS(F04)| OS(F10)| SE(F06)| DMGT(F12)| CNS LAB    | CNS LAB
```

**Faculty CSV:**
```
Day    | P1           | P2     | P3     | P4     | P5          | P6
Monday | CNS (A,B)    | CNS (C,D)| ---- | ----   | CNS LAB (A,D)| CNS LAB (A,D)
```

### 8.4 Quality Score Calculation (User-Defined)

```python
score = 100 - ((same_day_violations + 
                2*back_to_back_violations + 
                5*overloaded_faculty) / 2)
```

- **100** = Perfect timetable
- **Lower** = More soft constraint violations

**Note:** This formula is completely user-defined in our code. We invented it to give a simple 0-100 rating.

### 8.5 Fallback Writing

```python
def _write_csv_with_fallback(df, path: Path):
    try:
        df.to_csv(path, index=False)
        return str(path)
    except PermissionError:
        fallback_path = path.with_name(f"{path.stem}.latest{path.suffix}")
        df.to_csv(fallback_path, index=False)
        print(f"Warning: could not overwrite {path.name}; "
              f"wrote latest data to {fallback_path.name} instead.")
        return str(fallback_path)
```

**For Presentation:** "Phase 4 is the 'translator.' It takes the raw grid data from CP-SAT and converts it into timetables that students and faculty can actually read. It also calculates a quality score so we know how 'good' the schedule is."

---

## 9. Phase 5: AI Layer

This is where your project gets "AI-powered." Phase 5 adds **natural language queries** and **autonomous scheduling changes**.

### 5 Components:

| Component | File | Purpose |
|-----------|------|---------|
| **RAG Indexer** | `rag_indexer.py` | Creates searchable vector index from timetables |
| **AI Explainer** | `ai_explainer.py` | Answers questions like "When is F04 free?" |
| **Agent** | `agent.py` | 16-tool autonomous agent that can MODIFY schedules |
| **Substitute Finder** | `substitute.py` | Ranks best substitutes when faculty absent |
| **Sync Manager** | `sync_manager.py` | Atomic commit/rollback for safe changes |

### Big Picture Flow

```
User Question → RAG Retriever → Cross-Encoder Reranker → LLM (Groq) → Answer
                    ↑
            FAISS Index (732 docs)
            Created from CSVs
```

---

## 9.1 RAG Indexer

**File:** `rag_indexer.py` (~32KB)

**Purpose:** Convert timetable CSVs into a **searchable knowledge base** using embeddings and vector search.

### What is RAG?

**R**etrieval-**A**ugmented **G**eneration = Search first, then answer.

```
User asks: "When is section A's CNS class?"
     ↓
RAG retrieves relevant timetable snippets
     ↓
LLM generates answer using retrieved context
```

Without RAG: LLM hallucinates.  
With RAG: LLM answers from actual data.

### The 3-Step Pipeline

| Step | Component | What It Does |
|------|-----------|--------------|
| **1. Document Generation** | Python code | Creates ~732 text documents from CSVs |
| **2. Embedding** | `sentence-transformers/all-MiniLM-L6-v2` | Converts text to 384-dimension vectors |
| **3. Vector Index** | FAISS (`IndexFlatL2`) | Stores vectors for fast similarity search |

### Document Types Created (732 total per semester)

| Type | Count | Example Text |
|------|-------|--------------|
| **Per-slot** | ~360 | "Section A on Monday Period 1: CNS (F04) in Room_G03" |
| **Per-day** | ~60 | "Section A on Monday: P1=CNS, P2=OS, P3=SE, P4=FREE, P5=CNS LAB, P6=CNS LAB" |
| **Per-week** | ~12 | "Section A full weekly timetable: Monday has..." |
| **Faculty schedule** | ~60 | "Faculty F04 schedule: Monday P1=Section A CNS, P5-P6=Lab..." |
| **Room availability** | ~240 | "Room_G03 on Monday: P1=Section A CNS, P2=FREE..." |

### Two-Stage Retrieval

**Stage 1: FAISS (Fast, Approximate)**
- Converts query to embedding vector
- Finds 15 most similar documents (by L2 distance)
- Fast but not always precise

**Stage 2: Cross-Encoder (Slow, Accurate)**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranks the 15 candidates with actual neural scoring
- Returns top 5 most relevant

### Files Created

| File | Purpose |
|------|---------|
| `rag_index.faiss` | Binary vector index (searchable) |
| `rag_docs.json` | Document text + metadata (retrievable) |

### Metadata Filtering

Before embedding search, can filter by:
- `section`: A, B, C...
- `day`: Monday, Tuesday...
- `faculty`: F04, F10...
- `source_type`: "section_slot", "faculty_schedule"...

**Example:** Query about "F04" → only search docs with `faculty: F04`

**For Presentation:** "The RAG Indexer transforms static CSVs into a searchable knowledge base. It creates 732+ text documents, converts them to mathematical vectors using sentence-transformers, and stores them in FAISS. When users ask questions, we retrieve the most relevant timetable snippets using two-stage search: fast FAISS followed by precise cross-encoder reranking."

---

## 9.2 AI Explainer

**File:** `ai_explainer.py` (~20KB)

**Purpose:** Answer natural language questions about the timetable using RAG + LLM.

### How It Works

```
User: "When does section A have CNS on Monday?"
    ↓
AI Explainer detects entities: section=A, course=CNS, day=Monday
    ↓
Filters RAG docs by metadata (section=A, day=Monday)
    ↓
Retrieves top 5 relevant snippets via FAISS + Cross-Encoder
    ↓
Builds prompt with context + question
    ↓
Sends to Groq LLM (llama-3.3-70b-versatile)
    ↓
Returns natural language answer
```

### Query Decomposition (For Complex Questions)

**Simple query:** "When is F04 free?" → Single RAG search

**Complex query:** "What classes do sections A and B have on Monday and Tuesday?" → Multi-hop

```
Decomposition:
1. "Section A classes on Monday"
2. "Section A classes on Tuesday"  
3. "Section B classes on Monday"
4. "Section B classes on Tuesday"
```

Each sub-query → RAG → Combine results → LLM synthesizes final answer

### LLM Configuration

```python
Model: llama-3.3-70b-versatile (via Groq API)
Temperature: 0.3  (low = factual, not creative)
Max tokens: 512
System prompt: "You are a timetable assistant. Answer ONLY from provided context."
```

### Example Q&A

| User Question | Retrieved Context | LLM Answer |
|---------------|-----------------|------------|
| "When is F04 free?" | "Faculty F04: Monday P1=Section A CNS, P2=FREE, P3=Section B CNS..." | "F04 is free on Monday Period 2, Wednesday Period 4..." |
| "What room for section A CNS?" | "Section A on Monday P1: CNS (F04) in Room_G03" | "Section A's CNS class is in Room_G03." |
| "Who teaches OS to section C?" | "OS course assignments: A=F10, B=F10, C=F11..." | "F11 teaches OS to section C." |

### Memory Window

```python
MAX_CHAT_HISTORY = 4  # Last 4 exchanges kept
```

Follow-up questions work:
```
User: "When is F04 free?"
AI: "F04 is free Monday P2, Wednesday P4..."
User: "What about Tuesday?"  ← Understands "F04" from context
AI: "On Tuesday, F04 is busy all day with CNS labs..."
```

**For Presentation:** "AI Explainer is the chat interface. It takes natural language questions, uses RAG to find relevant timetable snippets, and feeds them to a Groq LLM to generate human-friendly answers. For complex multi-part questions, it breaks them down, searches each part separately, then combines the results."

---

## 9.3 Substitute Finder

**File:** `substitute.py` (~24KB)

**Purpose:** Ranks best substitutes when a faculty is absent.

### Scoring Algorithm

```python
# Positive scores (good things)
+20 same course teaching      # Already knows the subject
+15 same designation          # Similar experience level
+10 under hours             # Has capacity
+10 free at slot            # Available at that time

# Negative scores (bad things)
-30 overloaded              # Already at max hours
-20 teaching 2+ courses       # Too much variety
-15 busy at slot            # Already teaching then
-10 different designation   # Experience mismatch
```

### Lab Block Handling

P5-P6 treated as atomic 2-period block. Can't substitute just one period of a lab.

### Example Ranking

```
Faculty F04 is absent Monday P1 (CNS class for Section A)

Candidates ranked:
1. F06 (+45): Same course (CNS), free at P1, under hours
2. F09 (+25): Same course, but busy at P1
3. F12 (+10): Free at P1, but different course
4. F03 (-15): Overloaded, teaching 3 courses
```

**For Presentation:** "When a faculty is absent, the Substitute Finder scores all other faculty based on whether they already teach that course, have free time, and aren't overloaded. It returns a ranked list so administrators can pick the best replacement."

---

## 9.4 Agent with 16 Tools

**File:** `agent.py` (~50KB)

**Purpose:** Autonomous LangChain agent that can actually MODIFY schedules safely.

### Architecture: LangChain ReAct

**ReAct** = **Re**asoning + **Act**ing

The agent thinks step-by-step, uses tools to gather info, then acts.

### 16 Tools (8 Query + 8 Action)

**Query Tools (Read-Only):**
| # | Tool | What It Does |
|---|------|--------------|
| 1 | `get_section_timetable` | Returns a section's full week |
| 2 | `get_faculty_schedule` | Returns when faculty teaches |
| 3 | `find_free_slots` | Finds empty periods for meetings |
| 4 | `get_summary_stats` | Returns quality score, violations |
| 5 | `find_substitute` | Runs substitute finder scoring |
| 6 | `get_faculty_workload` | Returns hours per faculty |
| 7 | `get_room_availability` | Shows which rooms are free |
| 8 | `detect_schedule_conflicts` | Checks for double-bookings |

**Action Tools (Can Modify):**
| # | Tool | What It Does |
|---|------|--------------|
| 9 | `commit_substitute` | Assigns substitute officially |
| 10 | `swap_faculty` | Swaps two faculty's assignments |
| 11 | `reassign_section_faculty` | Changes who teaches a section |
| 12 | `mark_absence` | Records faculty as absent |
| 13 | `assign_substitute` | One-click substitute assignment |
| 14 | `undo_last_change` | Rollback via sync_manager |
| 15 | `preview_change` | Shows what would happen |
| 16 | `get_system_status` | Checks all systems |

### Example Agent Flow

```
User: "F04 is absent tomorrow, assign a substitute"

Agent reasoning:
1. mark_absence(F04, tomorrow)
2. find_substitute(course="CNS", slot="tomorrow P1")
3. preview_change(assign F06 as substitute)
4. commit_substitute(F06, "CNS", "Section A", tomorrow P1)
```

**For Presentation:** "The Agent is the 'autonomous brain.' Unlike the chat interface which just answers questions, the Agent can actually make changes. It has 16 tools - 8 for reading info and 8 for modifying schedules. It uses the ReAct pattern: think, gather info, then act. All changes go through the Sync Manager for safety."

---

## 9.5 Sync Manager

**File:** `sync_manager.py` (~35KB)

**Purpose:** **Atomic commit/rollback** for safe schedule modifications.

### The Problem

What if an agent crashes mid-change?  
What if we want to undo yesterday's change?

### Solution: Atomic Operations

**Flow:**
```
1. PREVIEW → Write to temp directory, show user
2. VALIDATE → Check CSV integrity
3. BACKUP → Copy originals to agent_ops/backups/
4. COMMIT → Atomic move temp → canonical
5. LOG → Write operation JSON
```

### Operation Log Format

```json
{
  "op_id": "20260401T203501Z-8b8e600f",
  "timestamp": "2026-04-01T20:35:01Z",
  "type": "substitute_assignment",
  "change_dict": {
    "faculty": "F06",
    "course": "UE24CS351A",
    "section": "A",
    "day": "Monday",
    "period": 1
  },
  "affected_files": [
    "outputs/cse_sem3/faculty_F04_timetable.csv",
    "outputs/cse_sem3/faculty_F06_timetable.csv",
    "outputs/cse_sem3/section_A_timetable.csv"
  ],
  "backup_paths": {
    "faculty_F04": "outputs/cse_sem3/agent_ops/backups/faculty_F04_timetable.20260401T203501Z.csv"
  },
  "committed": true
}
```

### Rollback

```python
undo_last_change():
    1. Find latest operation log
    2. Restore all files from backups
    3. Mark operation as "rolled_back"
    4. Restore in-memory grids
```

**For Presentation:** "The Sync Manager is our safety net. Every change made by the agent is atomic - it either completes fully or not at all. We backup files before changing them, log every operation, and can rollback to any previous state. This prevents data corruption and enables 'undo' functionality."

---

## Summary: All Phases

| Phase | File(s) | Lines | Purpose |
|-------|---------|-------|---------|
| 0 | loader.py, validator.py | 390 | Validate input CSVs |
| 1 | assignment_builder.py | 123 | Build faculty-section mapping |
| 2 | lab_scheduler.py | 245 | Lock labs/electives in 3 grids |
| 3 | theory_scheduler.py | 932 | CP-SAT solver for theory scheduling |
| 3.5 | room_allocator.py | 359 | Assign classrooms |
| 4 | output_generator.py | 281 | Generate CSVs and summary |
| 5.1 | rag_indexer.py | ~32KB | Create searchable vector index |
| 5.2 | ai_explainer.py | ~20KB | Natural language Q&A |
| 5.3 | substitute.py | ~24KB | Rank substitute candidates |
| 5.4 | agent.py | ~50KB | 16-tool autonomous agent |
| 5.5 | sync_manager.py | ~35KB | Atomic commit/rollback |

**Total: ~22 files, ~8,000+ lines of code**

---

*Generated for presentation: April 7, 2026*  
*System: GenAI Timetable Generation*  
*Prepared by: AI Assistant for project defense*
