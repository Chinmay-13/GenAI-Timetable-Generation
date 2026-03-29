# GenAI-Timetable-Generation

This project generates a weekly timetable for the CSE department, 3rd semester, across 12 sections using a phased pipeline. It combines rule validation, fixed lab allocation, OR-Tools based theory scheduling, CSV export, and an optional Gemini-powered explainer for generated results.

## What it does

- Validates course, faculty, and assignment input data.
- Builds faculty-to-section course assignments.
- Locks fixed lab sessions before theory scheduling.
- Solves theory-slot allocation with OR-Tools CP-SAT.
- Exports section-wise and faculty-wise timetable CSVs plus a summary report.
- Provides an optional AI assistant over the generated timetable outputs.

## Project structure

```text
.
|-- data/                     # Input CSV files
|-- src/
|   |-- phase0/              # Validation and loading
|   |-- phase1/              # Faculty-course assignment mapping
|   |-- phase2/              # Fixed lab scheduling
|   |-- phase3/              # OR-Tools theory scheduler
|   |-- phase4/              # Output generation
|   `-- phase5/              # Optional AI explainer/chat
|-- create_dummy_data.py     # Generates sample CSV data
|-- run_all.py               # Runs the full pipeline
`-- demo_ai.py               # Sample AI explainer prompts
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the env template and set your Gemini key only if you want AI responses:

```bash
copy .env.example .env
```

Then update `GEMINI_API_KEY` in `.env`.

## Run the timetable pipeline

```bash
python run_all.py
```

The generated files are written to `outputs/`:

- `section_<SECTION>_timetable.csv`
- `faculty_<FACULTY_ID>_timetable.csv`
- `summary_report.txt`

## Run the optional AI assistant

Generate outputs first, then run either:

```bash
python src/phase5/chat.py
```

or

```bash
python demo_ai.py
```

If no valid Gemini API key is configured, the assistant falls back to built-in rule-based answers for common timetable questions.

## Input data

The project expects these CSV files inside `data/`:

- `courses.csv`
- `faculty.csv`
- `assignments.csv`
- `lab_allotment.csv`
- `rooms.csv`

To regenerate the sample dataset:

```bash
python create_dummy_data.py
```

## Notes

- `.env`, virtual environments, logs, caches, and generated outputs are excluded from git.
- The committed `.env.example` is safe to share and contains no private key.
