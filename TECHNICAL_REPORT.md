# Technical Report — Timetable Scheduling & AI Methods

## 1. Constraint Model (Phase 3 — theory_scheduler.py)

### Solver and Model Type
- **Solver**: Google OR-Tools CP-SAT (`ortools.sat.python.cp_model`)
- **Model Type**: Binary Integer Programming (0-1 boolean decision variables)
- **Solver Configuration**: 8 workers (`NUM_WORKERS = 8`), penalty stage 120s, reward stage 60s

### Hard Constraints
1. **Course-hour satisfaction**: Each (section, course) combination receives exactly its required theory periods based on credits
2. **Single-class-per-slot**: A section can have at most one course per (day, period) slot
3. **Faculty conflict prevention**: A faculty member cannot teach two sections simultaneously
4. **Faculty load cap**: Per-designation weekly hour limits (Prof=12h, Asso Prof=16h, Asst Prof=20h)
5. **Lab window blocking**: No theory scheduling in P5-P6 on days when a section has lab
6. **Compactness (no-gap)**: Occupied theory slots must form a prefix from P1 (P1 ≥ P2 ≥ P3 ≥ P4 monotonicity)
7. **Daily load balancing**: Each section must have exactly `daily_targets[day]` theory slots per day
8. **Phase 2 pre-locking**: Respects all lab/elective slots already placed by Phase 2

### Soft Constraints / Penalties
| Penalty/Reward | Weight | Description |
|----------------|--------|-------------|
| `PENALTY_GAP` | 100 | Intra-day gaps (unused slots between used slots) |
| `PENALTY_LAB_WINDOW` | 50 | Theory scheduled in P5-P6 on non-lab days |
| `PENALTY_BACK_TO_BACK` | 5 | Same subject in consecutive periods same day |
| `PENALTY_SAME_DAY` | 3 | Same subject appearing >1 time same day (beyond 1) |
| `REWARD_CONSECUTIVENESS` | 200 | Reward for consecutive placement when faculty teaches same course to same section on same day |

### Variable Definition
- **Primary variable**: `x[(section, course, day, period)]` — binary decision variable indicating whether a specific section-course combo is scheduled at that (day, period) slot
- **Auxiliary variables**: `is_used[(section, day, period)]` — whether any theory class occupies that slot; `gap_var[(section, day, p2)]` — whether a gap exists at period p2; `daily_count[(section, day)]` — total slots per day; adjacency tracking variables

### Objective Function
Two-phase optimization:
1. **Penalty phase**: Minimize `sum(penalty_terms)` — drives toward feasibility with minimal soft violations
2. **Reward phase**: Fix the achieved penalty value as hard constraint, then maximize `sum(reward_terms)` for consecutiveness rewards

### Compactness / No-Gap Implementation
Uses monotonicity constraint on `is_used` variables across theory periods P1-P4: `occupied(p) >= occupied(p+1)`. This enforces that if P2 is occupied, P1 must be too; if P3 is occupied, P2 must be too, etc. Combined with daily load balancing, this creates compact block scheduling without intra-day gaps.

### Two-Phase (Penalty/Reward) Structure
**Phase A (Penalty)**: Runs first to find a feasible solution minimizing penalties. If not OPTIMAL, the solve fails. Best penalty value is captured.

**Phase B (Reward)**: Locks the penalty at the best value (hard constraint `penalty_expr == best_penalty`), then maximizes reward terms. Uses hints from Phase A solution to warm-start.

**Consecutiveness sub-phase**: After main solve, tries Phase A (hard consecutiveness constraint) for same-faculty same-section same-day slots. If infeasible, falls back to Phase B (soft reward weight 200).

---

## 2. Lab Scheduling (Phase 2 — lab_scheduler.py)

### How Lab Slots Are Pre-Locked
1. Reads `lab_allotment.csv` containing: day, course_code, room, faculty_id, section_pair (e.g., "A,B")
2. For each valid row, marks P5-P6 slots in all three grids:
   - `section_grid[section][day][period] = "{course_code}_LAB"` for both sections
   - `faculty_grid[faculty_id][day][period] = "{course_code}_LAB"`
   - `room_grid[room][day][period] = "{course_code}_LAB"`
3. Also processes `elective_slots.csv` for pre-locked electives with custom period ranges

### Lab-Specific Constraints
1. **Professor prohibition**: Prof designation cannot be assigned to lab courses (checked during validation)
2. **Section pair validation**: Lab entries must specify exactly 2 sections
3. **Conflict detection**: Pre-locked slots cannot overlap with existing entries in any grid
4. **Atomic P5-P6 blocking**: Labs always consume the full 2-period block
5. **Validation**: Invalid days, unknown sections, faculty conflicts, room conflicts all raise errors

### Phase 3 Interaction
Returns `section_grid`, `faculty_grid`, `room_grid`, `lab_details`, `elective_details` which Phase 3:
- Uses as immutable locked cells (skips creating variables for occupied slots)
- Uses `lab_details` to build `section_lab_days` dict that blocks P5-P6 theory on lab days
- Re-applies elective locking as defense-in-depth (though Phase 2 already did it)

---

## 3. Room Allocation (Phase 3.5 — room_allocator.py)

### Algorithm
**Greedy bipartite matching** — deterministic alphabetical priority:
1. For each (day, period), collect all sections with theory classes (excluding lab tokens)
2. Determine available classrooms: all CLASSROOM/LECTURE_HALL type rooms not occupied in `room_grid` from Phase 2
3. Sort sections alphabetically; sort rooms by capacity descending
4. Assign rooms in order until either sections or rooms exhausted
5. Unassigned sections get "ROOM_UNASSIGNED" (non-fatal, logged as warning)

### Conflict Detection
- Uses `preoccupied_rooms` mapping from Phase 2's `room_grid` — tracks which rooms are already used by labs/electives at each (day, period)
- Only considers rooms not in the preoccupied set as available

### Capacity Matching
- All classrooms sorted by capacity descending (largest first)
- Hardcoded assumption: all sections have same enrollment (no enrollment data in system)
- No capacity constraints enforced — assignment is purely availability-based
- 6 classrooms available, 12 sections maximum, so up to 6 sections per slot can be assigned

---

## 4. Assignment Building (Phase 1 — assignment_builder.py)

### Assignment Map Data Structure
```python
assignment_map = {
    "UE24CS251A": {  # course_code
        "A": "F01",   # section -> faculty_id
        "B": "F02",
        ...
    },
    "UE24CS252A": {
        "A": "F03",
        ...
    }
}
```
- Nested dict: course_code → section → faculty_id
- Built from `assignments.csv` containing: faculty_id, course_code, sections_handled (comma-separated)

### Faculty-Section-Course Triple Building
1. Loads `courses.csv`, `faculty.csv`, `assignments.csv`
2. For each assignment row, splits `sections_handled` by comma
3. Validates each section against `SECTIONS` list (A-L)
4. Checks for duplicate section mappings per course (error if conflict)
5. Populates `assignment_map[course_code][section] = faculty_id`

### Elective Handling
- Identifies electives via `is_elective` column in courses.csv (boolean, case-insensitive parsing)
- **Core courses**: Must have all 12 sections (A-L) assigned — full coverage enforced
- **Electives**: Only need ≥1 section assigned — partial coverage is by design
- Validation skips missing sections for electives but ensures at least one section has assignment

### Designation Rules Validation
- Prohibits Prof faculty from lab courses (checked during assignment validation)
- Enforced via `faculty_designation` lookup against `LAB_COURSE_SHORT` set

---

## 5. RAG Pipeline (Phase 5 — rag_indexer.py)

### Embedding Model and Vector Store
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimension embeddings)
- **Vector Store**: FAISS (`faiss.IndexFlatL2`) — exact L2 distance search
- **Offline mode**: Forces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for local-only model loading

### Document Construction
Each indexed record is a dictionary with:
- **Per-slot documents**: `"Section {X} on {day} {period}: {course}"` — individual slot entries
- **Per-day summaries**: `"Section {X} on {day}: P1=..., P2=..., ..."` — aggregated day view
- **Faculty equivalents**: Same structure for faculty timetables
- **Room availability**: `"Room availability on {day} {period}: {N} free classrooms (...)"`

### Metadata Attached
```python
{
    "text": "human-readable description",
    "section": "A",           # for section docs
    "faculty": "F01",        # for faculty docs
    "day": "Monday",
    "period": "P3",
    "source": "section_A_timetable.csv"  # origin file
}
```

### Retrieval Strategy
- **k**: Default 5 results (`k=5`)
- **Similarity metric**: L2 (Euclidean) distance via FAISS IndexFlatL2
- **Query encoding**: Same MiniLM model encodes the natural language query
- **Result ranking**: Sorted by FAISS distance (lower = more similar)

### Current Weaknesses
1. **No semantic chunking**: Documents are rigid slot/day-level, not conceptually grouped
2. **Flat index only**: No HNSW or IVF for scalability — O(N) search time
3. **No cross-encoder reranking**: Initial retrieval only, no second-stage precision
4. **Offline dependency**: Requires model pre-cached; fails silently if not available
5. **Static index**: Must be rebuilt manually after any schedule changes (not automatic)
6. **No query expansion**: Raw user query embedded directly without expansion/rewriting
7. **Limited context**: 5 documents may not capture multi-day or multi-section patterns

---

## 6. LLM Agent (Phase 5 — agent.py)

### Framework and LLM
- **Framework**: LangChain with custom tool definitions (`@tool` decorator)
- **Primary LLM**: Groq `llama-3.3-70b-versatile` (via `GROQ_API_KEY` env var)
- **Fallback LLM**: Groq `llama-3.1-8b-instant` (fast fallback, configured in config)
- **Alternative (unused)**: Gemini 2.0 Flash Lite (empty API key in config)

### Tools (12 total)
| Tool | Type | Description |
|------|------|-------------|
| `get_section_timetable` | READ | Full weekly CSV for a section (A-L) |
| `get_faculty_schedule` | READ | Full weekly CSV for a faculty (F01-Fnn) |
| `find_free_slots` | READ | Free periods for faculty on specific day |
| `get_summary_stats` | READ | Raw text from summary_report.txt |
| `find_substitute` | READ | Substitute candidates for absent faculty |
| `commit_substitute` | WRITE | Commit substitute to timetable (via sync_manager) |
| `list_agent_ops` | READ | List recent autonomous operations |
| `rollback_last_operation` | WRITE | Rollback by operation_id |
| `generate_session_summary` | READ | Human-readable summary of agent actions |
| `get_absent_periods` | READ | Specific periods a faculty teaches on a day |
| `find_free_rooms` | READ | Available classrooms by day/period/capacity |
| `get_faculty_workload` | READ | Weekly hours, day breakdown, courses, status |
| `get_free_faculty` | READ | All faculty free in a given slot |
| `get_room_availability` | READ | Free vs occupied rooms with details |
| `detect_schedule_conflicts` | READ | Scan for faculty/room double-bookings |
| `get_weekly_stats` | READ | Key stats from summary report |

### Tool Loop Structure
- **Max steps**: Controlled by LangChain's `AgentExecutor` with `max_iterations=15`
- **Termination**: Agent returns final answer when satisfied, or hits iteration limit
- **No explicit early stopping**: Relies on LLM's own judgment via `AgentFinish`
- **Intermediate steps**: Agent may chain multiple tool calls (e.g., find_substitute → get_faculty_schedule → commit_substitute)

### Write Safety Enforcement
1. **Pre-validation**: `commit_substitute` calls `_is_faculty_free()` to verify substitute availability before any write
2. **Atomic delegation**: All writes routed through `sync_manager.commit_schedule_change()` — never direct file writes
3. **Backup first**: sync_manager creates timestamped backup before any mutation
4. **Rollback on failure**: Any exception during write triggers automatic restore from backup
5. **Annotation format**: Substitutions use "COURSE (INITIALS)→NEW_FACULTY" format to preserve history

### Rollback Implementation
- **Method**: `rollback_last_operation(input_str)` where input_str is operation_id
- **Mechanism**: Reads JSON log from `agent_ops/`, finds backup directory, restores all files
- **Scope**: Restores section CSV, both faculty CSVs, summary_report.txt
- **RAG refresh**: Best-effort re-index after rollback
- **Two formats supported**: New (directory backup from sync_manager) and legacy (single file backup)

---

## 7. AI Explainer / Assistant (Phase 5 — ai_explainer.py, prompt_builder.py)

### System Prompt Construction (prompt_builder.py)
```python
build_system_prompt(outputs_dir, data_dir, summary_text, sem_id) -> str
```
- **Dynamic loading**: Reads courses.csv, faculty.csv, lab_allotment.csv at call time
- **Compact format**: Target <800 tokens, <400 tokens ideal for Groq free tier
- **Structure**:
  - Header: University, sections (A-L), periods (P1-P4 theory, P5-P6 labs), days (Mon-Fri)
  - Course list: code + name (core only, electives excluded to save tokens)
  - Faculty list: ID + name + designation (abbreviated: Prof/Asso/Asst)
  - Lab schedule: course + section_pair + day
  - Weekly caps hardcoded: Prof=12h, Associate=16h, Assistant=20h
  - Rule reminder: Labs are 2-period blocks, Professors cannot take labs

### Context Injection (ai_explainer.py)
**setup_context()** loads:
- `summary_report.txt` — full pipeline output statistics
- `section_A_timetable.csv` — representative sample section
- `faculty.csv`, `assignments.csv`, `courses.csv`, `lab_allotment.csv` — raw data
- Faculty load table extracted from summary report
- Pre-built system prompt from prompt_builder

### Chat History Management
- **History parameter**: `Optional[List[Tuple[str, str]]]` passed to `explain()`
- **Truncation**: Only last 4 exchanges included (`history[-4:]`)
- **Format**: "Recent conversation context (last 4 exchanges):" prefix
- **No persistence**: History not saved to disk (session-only)

### Fallback (No API Key)
**`_fallback_answer()`** triggers when `GROQ_API_KEY` missing or LLM unavailable:
1. **Section queries**: Returns raw CSV from `section_{X}_timetable.csv`
2. **Faculty queries**: Matches by ID or name, returns `faculty_{ID}_timetable.csv`
3. **Default**: Returns `summary_report.txt` content
- No hardcoded strings — always reads from actual output files
- Data-driven responses ensure accuracy even without LLM

---

## 8. Substitute & Swap Logic (substitute.py, swap.py)

### find_substitute() Algorithm
1. **Collect absent slots**: Parse faculty day row from CSV, extract all non-empty periods
2. **Lab block handling**: If slot is P5/P6, expand to full block ["P5", "P6"] — atomic requirement
3. **Candidate ranking** via `_rank_candidates()`:
   - Filter: Same faculty excluded
   - Filter: Prof excluded for lab slots
   - Filter: Load check (projected_load ≤ max_hours)
   - Filter: Availability check (cell == "----")
   - Filter: Designation rank check (candidates cannot be lower designation without teaching the course)
   - Priority 1: Same course teachers (`match_type="same_course"`)
   - Priority 2: Same designation (`match_type="same_designation"`)
   - Priority 3: Available with lowest projected load, highest designation rank
4. **Lab block filtering**: For lab blocks, only keep candidates free for ALL periods in block
5. **Selection**: Top-ranked candidate per slot

### Candidate Ranking Formula
```python
sorted(candidates, key=lambda item: (
    item["priority"],           # 1=same_course, 2=same_designation, 3=available
    item["projected_load"],    # lower is better
    -DESIGNATION_RANK.get(item["designation"], 0),  # higher rank better
    item["faculty_id"],        # tie-breaker: alphabetical
))
```

### Conflict Checks Before Swap Commit
**swap.py:find_swap_slot()**:
1. Validates both faculty IDs exist
2. Validates course code exists
3. Searches days after `return_day` for when faculty B teaches the same course
4. Verifies faculty A is free at that (day, period)
5. Calculates load before/after to ensure caps not exceeded

### commit_swap() Interaction with sync_manager
- Receives swap_result dict from `find_swap_slot()` (must have `swap_found=True`)
- Extracts: swap_day, swap_period, section, faculty_a, faculty_b
- Converts period string "P3" → integer 3
- Calls `sync_manager.commit_schedule_change()` with:
  - `change_type="swap"`
  - `original_faculty=faculty_b` (substitute)
  - `new_faculty=faculty_a` (returning faculty)
- Benefits from same atomic backup/restore, RAG refresh, and audit logging as substitutes

---

## 9. Sync & Audit (sync_manager.py, agent_ops.py)

### commit_schedule_change() — Step by Step
1. **Validate**: Check required keys exist; resolve semester-aware output dir
2. **Backup**: Create timestamped backup dir; copy section CSV, both faculty CSVs, summary_report.txt
3. **Patch section CSV**: Read CSV, find day row, annotate cells with "→NEW_FACULTY", write atomically
4. **Rebuild faculty CSVs**: Scan all section CSVs to rebuild both original and new faculty timetables
5. **Rebuild summary_report.txt**: Recount theory/lab slots, rebuild faculty load table
6. **RAG re-index**: Best-effort call to `rag_indexer.build_index()`
7. **Log operation**: Write structured JSON to `agent_ops/{ts}-{op_id}.json`
8. **Rollback on any failure**: Restore all backed-up files if any step raises exception

### Backup Structure
```
outputs/{sem_id}/agent_ops/backups/{timestamp}/
├── section_{X}_timetable.csv    # pre-substitution state
├── faculty_{F01}_timetable.csv  # original faculty
├── faculty_{F02}_timetable.csv  # substitute faculty
└── summary_report.txt           # pre-change stats
```

### Audit Log Contents (agent_ops.py)
Each JSON log entry contains:
```python
{
    "operation_id": "a3f2b1c4",           # UUID[:8]
    "timestamp_utc": "2026-01-15T09:30:00Z",
    "timestamp_local": "2026-01-15T15:00:00",
    "action": "substitute" or "swap",
    "absent_faculty": "F03",
    "section_id": "A",
    "day": "Monday",
    "period_range": [5, 6],
    "substitute_faculty": "F07",
    "reasoning_chain": ["Lab coverage for DDCO"],
    "pre_state": "CSV snippet before change",
    "post_state": "CSV snippet after change",
    "commit_result": "SUCCESS",
    "backup_path": ".../backups/20260115T093000Z"
}
```

---

## Known Gaps and TODOs

### Constraint Model Gaps
1. **No room capacity constraints in solver**: Room allocation happens post-solve (Phase 3.5), can over-assign
2. **Fixed daily targets**: All sections forced to identical daily load distribution
3. **No faculty preference respect**: No way to express "Prof X prefers mornings"
4. **Single consecutive pair reward**: Multiple same-day slots for same course not fully optimized
5. **Hard compactness may over-constrain**: Some valid compact schedules rejected by prefix rule

### Lab & Room Gaps
6. **Static lab allotment**: No rescheduling of labs — Phase 2 is strictly pre-lock
7. **Room unassignment accepted**: When 12 sections need rooms but only 6 classrooms exist, 6 get "ROOM_UNASSIGNED"
8. **No room preference**: No way to prefer specific rooms for specific courses
9. **No travel time modeling**: Consecutive periods in different rooms not penalized

### RAG & AI Gaps
10. **No online learning**: Index must be manually rebuilt; not triggered automatically on every change
11. **No conversational memory persistence**: Chat history lost between sessions
12. **Single embedding model**: No multi-modal or task-specific embeddings
13. **No query decomposition**: Complex multi-part questions not broken down
14. **LLM rate limit exposure**: Fallback works but is slow; no caching layer

### Agent & Sync Gaps
15. **No batch operations**: Each substitute requires separate commit/backup cycle
16. **No conflict prediction**: Agent detects conflicts only after they occur
17. **No optimization suggestions**: Agent reports problems but doesn't suggest schedule improvements
18. **Limited swap window**: Swaps only search forward from return_day; no backward search
19. **No faculty preference input**: No way to capture "I'd prefer to swap my Tuesday slot"

### Data & Integration Gaps
20. **No enrollment data**: All sections treated identically; no room capacity planning based on actual students
21. **No holiday/exception handling**: System assumes fixed Monday-Friday week
22. **No external calendar sync**: No integration with university holiday schedules
23. **Hardcoded section count**: SECTIONS = A-L (12) — not dynamically determined from data
24. **CSV-only data layer**: No database backend; concurrent access risks corruption

### Performance Gaps
25. **CP-SAT time limits hardcoded**: 120s penalty + 60s reward may be insufficient for larger instances
26. **FAISS flat index**: O(N) search; will degrade with semester data growth
27. **Full CSV reads on every tool call**: No in-memory caching for repeated queries
