"""
rag_indexer.py — Builds a FAISS vector index over timetable output CSVs
so the AI can retrieve relevant rows instead of reading entire files.

Usage:
  python src/phase5/rag_indexer.py              # legacy: reads/writes outputs/
  python src/phase5/rag_indexer.py --sem cse_sem3
  from src.phase5.rag_indexer import retrieve   # query
"""

from pathlib import Path
import sys
import json
import os
import re

_RAG_ROOT = Path(__file__).resolve().parents[2]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from config import SECTIONS, OUTPUT_DIR, get_sem_paths  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Lazy-loaded model cache ───────────────────────────────────────────────────

_CROSS_ENCODER = None   # cached CrossEncoder instance


def _get_cross_encoder():
    """
    Lazily load and cache the cross-encoder model.
    Returns None (with a warning) if loading fails for any reason.
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        _CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_MODEL)
        return _CROSS_ENCODER
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(
            f"[RAG] Cross-encoder load failed ({exc}). "
            "Falling back to FAISS ranking only."
        )
        return None

# ── Day / section constants used by filter extraction ─────────────────────────

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
         "Saturday", "Sunday"]

_SECTIONS_LIST = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_output_dir(sem_id: str | None) -> Path:
    """Return the output directory for sem_id, or the legacy OUTPUT_DIR."""
    if sem_id is not None:
        return get_sem_paths(sem_id).output_dir
    return Path(OUTPUT_DIR)


def _resolve_data_dir(sem_id: str | None) -> Path:
    """Return the data directory for sem_id, or the legacy data/ dir."""
    if sem_id is not None:
        return get_sem_paths(sem_id).data_dir
    return _RAG_ROOT / "data"


def _iter_timetable_files(output_dir: Path, pattern: str):
    return sorted(output_dir.glob(pattern))


def _period_columns(df) -> list[str]:
    return [f"P{i}" for i in range(1, 7) if f"P{i}" in df.columns]


# ── Metadata filter helpers ───────────────────────────────────────────────────

def _extract_filters(query: str) -> dict:
    """
    Extract metadata filters from a natural-language query using keyword matching.

    Returns a dict with zero or more of the following keys:
      - "section"     : str, e.g. "A"
      - "day"         : str, e.g. "Monday"
      - "faculty"     : str, e.g. "F03"
      - "source_type" : str, one of "section", "faculty", "room"

    Only keys whose values are positively detected are included.
    An empty dict means no filters were detected → fall back to full search.
    """
    filters = {}
    q = query  # keep original case for regex; use lower for keyword checks
    ql = query.lower()

    # ── Section filter ────────────────────────────────────────────────────────
    # Match "section A", "section B", ..., "section L" etc. (case-insensitive)
    sec_match = re.search(r"\bsection\s+([A-Za-z])\b", q, re.IGNORECASE)
    if sec_match:
        filters["section"] = sec_match.group(1).upper()

    # ── Day filter ────────────────────────────────────────────────────────────
    for day in _DAYS:
        if re.search(rf"\b{day}\b", q, re.IGNORECASE):
            filters["day"] = day  # store canonical capitalisation
            break

    # ── Faculty filter ────────────────────────────────────────────────────────
    fac_match = re.search(r"\bF(\d{2})\b", q)
    if fac_match:
        filters["faculty"] = f"F{fac_match.group(1)}"

    # ── Source-type filter ────────────────────────────────────────────────────
    # Check most-specific first; "faculty" beats "section" if both appear.
    if "room" in ql or "classroom" in ql:
        filters["source_type"] = "room"
    elif "faculty" in ql:
        filters["source_type"] = "faculty"
    elif "section" in ql:
        filters["source_type"] = "section"

    return filters


def _apply_filters(docs: list[dict], filters: dict) -> list[int]:
    """
    Return indices of docs that match ALL filters in `filters`.

    Matching rules
    --------------
    section     : doc["section"] == filters["section"]
    day         : doc["day"]     == filters["day"]
                  EXCEPTION: day="all" (full-week summary docs) always pass
                  the day filter so they are always included in section/faculty
                  filtered searches.
    faculty     : doc["faculty"] == filters["faculty"]
    source_type : filters["source_type"] string is contained in doc["source"]
                  (e.g. "section" ⊆ "section_A_timetable.csv")

    If `filters` is empty, all indices are returned.
    """
    if not filters:
        return list(range(len(docs)))

    matching = []
    for i, doc in enumerate(docs):
        match = True

        if "section" in filters:
            # docs with section="all" (cross-section summaries) always pass
            if doc.get("section") != "all" and doc.get("section") != filters["section"]:
                match = False

        if match and "day" in filters:
            # full-week summary docs (day="all") always pass the day filter
            if doc.get("day") != "all" and doc.get("day") != filters["day"]:
                match = False

        if match and "faculty" in filters:
            if doc.get("faculty") != filters["faculty"]:
                match = False

        if match and "source_type" in filters:
            src = doc.get("source", "")
            if filters["source_type"] not in src:
                match = False

        if match:
            matching.append(i)

    return matching


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank(query: str, docs: list[dict], top_n: int = 5) -> list[dict]:
    """
    Rerank a list of retrieved docs using a cross-encoder model.

    The cross-encoder scores each (query, document-text) pair using full
    token-level interaction — far more precise than embedding cosine/L2
    distance, at the cost of O(n) forward passes.

    Parameters
    ----------
    query  : str            Natural-language question.
    docs   : list[dict]     Candidate docs (each must have a "text" field).
    top_n  : int            Number of top results to return.

    Returns
    -------
    list[dict]  Top-n docs sorted by cross-encoder score (descending),
                each with a "ce_score" field added.
                If the model fails to load, the input docs are returned
                unchanged (no "ce_score" field).
    """
    if len(docs) <= top_n:
        # Nothing to rerank — already within budget
        return docs

    ce = _get_cross_encoder()
    if ce is None:
        # Model unavailable — return first top_n without reranking
        return docs[:top_n]

    try:
        pairs = [(query, doc["text"]) for doc in docs]
        scores = ce.predict(pairs)
        ranked = sorted(
            zip(scores, docs), key=lambda t: t[0], reverse=True
        )
        out = []
        for score, doc in ranked[:top_n]:
            d = dict(doc)
            d["ce_score"] = float(score)
            out.append(d)
        return out
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"[RAG] Reranking failed ({exc}). Returning FAISS order.")
        return docs[:top_n]


# ── Public API ────────────────────────────────────────────────────────────────

def build_index(sem_id: str = None):
    """
    Build FAISS index from all section and faculty timetable CSVs.
    Embeddings are stored inside rag_docs.json (field: "embedding") so that
    retrieve() can perform filtered sub-index searches without re-encoding.

    Embedding storage decision
    --------------------------
    all-MiniLM-L6-v2 produces 384-dim float32 vectors.
    Even for 5 000 docs: 5000 × 384 × 4 B ≈ 7.7 MB — well below the 50 MB
    threshold.  Storing them inline avoids a model forward-pass at query time
    for every filtered search, which is the cheaper option.

    Parameters
    ----------
    sem_id : str or None
        Semester slug (e.g. "cse_sem3").  If None, uses legacy OUTPUT_DIR.
    """
    # ── Resolve paths ─────────────────────────────────────────────────────────
    output_dir = _resolve_output_dir(sem_id)
    data_dir   = _resolve_data_dir(sem_id)
    label      = sem_id or "legacy"

    index_path = str(output_dir / "rag_index.faiss")
    docs_path  = str(output_dir / "rag_docs.json")

    print(f"[RAG] Building index for [{label}]")
    print(f"  Reading from : {output_dir}")
    print(f"  Writing to   : {output_dir}")

    try:
        import pandas as pd
    except ImportError as e:
        print(f"RAG dependency missing: {e}")
        return None, None, None

    docs = []

    # ── Section timetable files ───────────────────────────────────────────────
    section_files = _iter_timetable_files(output_dir, "section_*_timetable.csv")
    faculty_files = _iter_timetable_files(output_dir, "faculty_*_timetable.csv")

    discovered_sections = []
    for csv_path in section_files:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            continue

        section = csv_path.stem.removeprefix("section_").removesuffix("_timetable")
        discovered_sections.append(section)
        period_cols = _period_columns(df)

        # ── Per-slot and per-day docs ──────────────────────────────────────────
        for _, row in df.iterrows():
            day = str(row["Day"]).strip()
            occupied = []
            for period in period_cols:
                cell = str(row.get(period, "----")).strip()
                if cell in ("", "----", "nan"):
                    continue
                occupied.append((period, cell))
                docs.append({
                    "text": f"Section {section} on {day} {period}: {cell}",
                    "section": section,
                    "day": day,
                    "period": period,
                    "source": csv_path.name,
                })

            docs.append({
                "text": (
                    f"Section {section} on {day}: "
                    + (", ".join(f"{period}={cell}" for period, cell in occupied) if occupied else "no classes")
                ),
                "section": section,
                "day": day,
                "source": csv_path.name,
            })

        # ── Full-week summary doc (one per section) ────────────────────────────
        # This single document answers "What is Section X's full timetable?"
        # Cross-encoder ranks it #1 for broad schedule queries.
        week_lines = [f"Section {section} full weekly timetable:"]
        for day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            day_rows = df[df["Day"].str.strip() == day_name]
            if day_rows.empty:
                week_lines.append(f"  {day_name}: no data")
                continue
            row = day_rows.iloc[0]
            slots = []
            for p in period_cols:
                val = str(row.get(p, "----")).strip()
                if val and val not in ("", "----", "nan"):
                    slots.append(f"{p}={val}")
            week_lines.append(f"  {day_name}: " + (", ".join(slots) if slots else "no classes"))
        docs.append({
            "text":    "\n".join(week_lines),
            "section": section,
            "day":     "all",
            "period":  "all",
            "source":  "section_full_week",
        })

    # ── Faculty timetable files ───────────────────────────────────────────────
    for csv_path in faculty_files:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            continue

        faculty_id = csv_path.stem.removeprefix("faculty_").removesuffix("_timetable")
        period_cols = _period_columns(df)

        # ── Per-slot and per-day docs ──────────────────────────────────────────
        for _, row in df.iterrows():
            day = str(row["Day"]).strip()
            occupied = []
            for period in period_cols:
                cell = str(row.get(period, "----")).strip()
                if cell in ("", "----", "nan"):
                    continue
                occupied.append((period, cell))
                docs.append({
                    "text": f"Faculty {faculty_id} on {day} {period}: {cell}",
                    "faculty": faculty_id,
                    "day": day,
                    "period": period,
                    "source": csv_path.name,
                })

            docs.append({
                "text": (
                    f"Faculty {faculty_id} on {day}: "
                    + (", ".join(f"{period}={cell}" for period, cell in occupied) if occupied else "no classes")
                ),
                "faculty": faculty_id,
                "day": day,
                "source": csv_path.name,
            })

        # ── Full-week summary doc (one per faculty) ────────────────────────────
        # Answers "What is F05's full schedule / workload?"
        week_lines = [f"Faculty {faculty_id} full weekly schedule:"]
        for day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            day_rows = df[df["Day"].str.strip() == day_name]
            if day_rows.empty:
                week_lines.append(f"  {day_name}: no classes")
                continue
            row = day_rows.iloc[0]
            slots = []
            for p in period_cols:
                val = str(row.get(p, "----")).strip()
                if val and val not in ("", "----", "nan"):
                    slots.append(f"{p}={val}")
            week_lines.append(f"  {day_name}: " + (", ".join(slots) if slots else "no classes"))
        docs.append({
            "text":    "\n".join(week_lines),
            "faculty": faculty_id,
            "day":     "all",
            "period":  "all",
            "source":  "faculty_full_week",
        })

    # ── Per-slot cross-section summary documents ─────────────────────────────
    # One doc per (day, period) listing ALL faculty+sections teaching that slot.
    # This makes "Who teaches on Wednesday P3?" hit a single document with every
    # faculty member in one shot, instead of requiring 12 separate section docs.
    THEORY_PERIODS = [f"P{p}" for p in range(1, 5)]  # P1-P4 theory only
    DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # Build a map: section -> {day -> {period -> cell}} from already-read files
    section_day_period: dict = {}
    for csv_path2 in section_files:
        try:
            df2 = pd.read_csv(csv_path2)
        except Exception:
            continue
        sec2 = csv_path2.stem.removeprefix("section_").removesuffix("_timetable")
        section_day_period[sec2] = {}
        for _, row2 in df2.iterrows():
            d2 = str(row2["Day"]).strip()
            section_day_period[sec2][d2] = {}
            for p2 in THEORY_PERIODS:
                if p2 in df2.columns:
                    section_day_period[sec2][d2][p2] = str(row2.get(p2, "----")).strip()

    for day_name in DAY_ORDER:
        for col in THEORY_PERIODS:
            entries = []
            for sec in sorted(section_day_period.keys()):
                val = section_day_period[sec].get(day_name, {}).get(col, "----")
                if val and val not in ("", "----", "nan"):
                    entries.append(f"{val} Section {sec}")
            if entries:
                text = f"On {day_name} {col}: " + ", ".join(entries)
                docs.append({
                    "text":    text,
                    "day":     day_name,
                    "period":  col,
                    "source":  "slot_summary",
                    "section": "all",
                })

    slot_summary_count = sum(1 for d in docs if d.get("source") == "slot_summary")
    print(f"  Slot-summary docs: {slot_summary_count} (one per day×period)")

    # ── Room assignment CSV ───────────────────────────────────────────────────
    room_csv = output_dir / "room_assignment.csv"
    if room_csv.exists():
        try:
            rdf = pd.read_csv(room_csv)
            for (day, period), grp in rdf.groupby(["Day", "Period"]):
                occupied = grp[grp["Room"] != "ROOM_UNASSIGNED"]
                free_names = set()
                try:
                    rooms_df = pd.read_csv(data_dir / "rooms.csv")
                    all_cr = set(
                        rooms_df[
                            rooms_df["room_type"].str.upper().isin(
                                ["CLASSROOM", "LECTURE_HALL"]
                            )
                        ]["room_name"].tolist()
                    )
                    free_names = all_cr - set(occupied["Room"].tolist())
                except Exception:
                    pass

                occ_parts = [
                    f"{r['Room']} (Sec {r['Section']} {r['Course']})"
                    for _, r in occupied.iterrows()
                ]
                unassigned = int((grp["Room"] == "ROOM_UNASSIGNED").sum())

                text = (
                    f"Room availability on {day} {period}: "
                    f"{len(free_names)} free classrooms"
                    + (f" ({', '.join(sorted(free_names))})" if free_names else "")
                    + f"; {len(occ_parts)} occupied"
                    + (f" — {'; '.join(occ_parts[:6])}" if occ_parts else "")
                    + (f"; {unassigned} ROOM_UNASSIGNED" if unassigned else "")
                )
                docs.append({
                    "text":   text,
                    "day":    day,
                    "period": period,
                    "source": "room_assignment.csv",
                })
        except Exception as exc:
            print(f"Warning: could not index room_assignment.csv: {exc}")

    if not docs:
        print(f"No timetable CSVs found in {output_dir}. Run run_all.py --sem {label} first.")
        return None, None, None

    discovered_section_set = sorted(set(discovered_sections))
    print(
        "Section files indexed: "
        f"{len(discovered_section_set)} ({', '.join(discovered_section_set)})"
    )
    if len(discovered_section_set) != len(SECTIONS):
        print(
            "Warning: expected "
            f"{len(SECTIONS)} sections but found {len(discovered_section_set)} section timetable files."
        )

    # ── Embed and build FAISS index ───────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        print(f"RAG embeddings skipped: {e}")
        print("Install sentence-transformers and faiss-cpu to rebuild rag_index.faiss.")
        # Write docs without embeddings (no filtered search possible later)
        Path(docs_path).parent.mkdir(parents=True, exist_ok=True)
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2)
        return None, docs, None

    previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception as exc:
        print(
            f"RAG embeddings skipped: model '{MODEL_NAME}' is not available in the local cache. "
            f"({exc})"
        )
        Path(docs_path).parent.mkdir(parents=True, exist_ok=True)
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2)
        return None, docs, None
    finally:
        if previous_hf_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_hf_offline

        if previous_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers_offline

    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # ── Store embeddings inside each doc (enables filtered sub-index search) ──
    # Strategy: inline in rag_docs.json as plain Python lists.
    # Cost: 984 docs × 384 dims × 4 B ≈ 1.5 MB — well below 50 MB threshold.
    for i, doc in enumerate(docs):
        doc["embedding"] = embeddings[i].tolist()

    # ── Write docs JSON (now includes embeddings) ─────────────────────────────
    Path(docs_path).parent.mkdir(parents=True, exist_ok=True)
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    section_docs = sum(1 for d in docs if "section" in d)
    faculty_docs = sum(1 for d in docs if "faculty" in d)
    room_docs    = sum(1 for d in docs if d.get("source") == "room_assignment.csv")
    print(f"RAG docs built: {len(docs)} documents")
    print(f"  Section docs : {section_docs}")
    print(f"  Faculty docs : {faculty_docs}")
    print(f"  Room docs    : {room_docs}")
    print(f"  Docs         : {docs_path}")

    # ── Build full FAISS index (unchanged — used for unfiltered fallback) ──────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"  Index        : {index_path}")
    return index, docs, model


def _is_full_schedule_query(query: str) -> bool:
    """
    Returns True if the query needs the FULL weekly schedule for a faculty.
    For these queries, capping results at k=3 is wrong — we need all 5-6 day-docs.
    """
    q = query.lower()
    return any(kw in q for kw in [
        "workload", "schedule", "timetable", "how many", "total hours",
        "how much", "weekly", "all classes", "full schedule", "compare",
        "vs", "versus",
    ])


def retrieve(query: str, k: int = 5, sem_id: str = None, use_rerank: bool = True) -> list:
    """
    Retrieve top-k relevant timetable records for a query string.

    Behaviour
    ---------
    1. Extract metadata filters from the query (section / day / faculty /
       source_type) using keyword matching — no LLM call needed.
    2. Find doc indices that satisfy ALL detected filters.
    3a. If filtered set ≥ fetch_k  → build a temporary numpy sub-index from
        those embeddings and run FAISS search within it.
    3b. If 0 < filtered set < fetch_k → return those docs directly.
    3c. If filtered set is empty OR no filters → full FAISS index search.
    4. If use_rerank=True and more than k candidates were fetched, run the
       cross-encoder to rerank and return the top k by joint score.
       If use_rerank=False → return FAISS top-k exactly as before.

    Parameters
    ----------
    query  : str   Natural-language query.
    k      : int   Number of results to return.
    sem_id : str or None
        Semester slug.  If None, reads from legacy OUTPUT_DIR.
    use_rerank : bool
        If True (default) — fetch k*3 candidates and rerank with cross-encoder.
        If False — fetch exactly k candidates from FAISS (legacy behaviour).
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except ImportError:
        return []

    output_dir = _resolve_output_dir(sem_id)
    index_path = str(output_dir / "rag_index.faiss")
    docs_path  = str(output_dir / "rag_docs.json")

    if not Path(index_path).exists() or not Path(docs_path).exists():
        label = sem_id or "legacy"
        print(f"RAG index not found for [{label}]. Run: python src/phase5/rag_indexer.py --sem {label}")
        return []

    model = SentenceTransformer(MODEL_NAME)
    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)

    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")

    # When reranking is on, fetch a wider net for the reranker to work with
    fetch_k = k * 3 if use_rerank else k

    # ── Step 1: extract metadata filters ─────────────────────────────────────
    filters = _extract_filters(query)

    # ── Step 2: find matching doc indices ─────────────────────────────────────
    if filters:
        candidate_indices = _apply_filters(docs, filters)
    else:
        candidate_indices = []  # signal: use full index

    # ── Step 3: search strategy ───────────────────────────────────────────────
    candidates = []

    # Full-schedule shortcut: if a faculty filter is active and the query needs
    # a complete weekly view, bypass vector search and return ALL matching docs.
    # A faculty's full week is 5-6 day-docs — small enough to return entirely.
    faculty_filter_active = bool(filters.get("faculty"))
    is_full_schedule      = _is_full_schedule_query(query)

    if faculty_filter_active and is_full_schedule and candidate_indices:
        for i in candidate_indices:
            result = {ke: v for ke, v in docs[i].items() if ke != "embedding"}
            result["distance"] = 0.0
            candidates.append(result)
        # Skip reranking for full-schedule returns — order is already meaningful
        return candidates

    if filters and len(candidate_indices) >= fetch_k:
        # 3a — filtered sub-index search
        if "embedding" not in docs[candidate_indices[0]]:
            candidate_indices = []  # fall through to full search
        else:
            sub_embs = np.array(
                [docs[i]["embedding"] for i in candidate_indices], dtype="float32"
            )
            sub_index = faiss.IndexFlatL2(sub_embs.shape[1])
            sub_index.add(sub_embs)
            actual_k = min(fetch_k, len(candidate_indices))
            D, I = sub_index.search(q_emb, actual_k)
            for sub_pos, dist in zip(I[0], D[0]):
                if 0 <= sub_pos < len(candidate_indices):
                    orig_idx = candidate_indices[sub_pos]
                    result = {ke: v for ke, v in docs[orig_idx].items() if ke != "embedding"}
                    result["distance"] = float(dist)
                    candidates.append(result)

    if not candidates and filters and 0 < len(candidate_indices) < fetch_k:
        # 3b — too few docs to search; return them all directly
        for i in candidate_indices:
            result = {ke: v for ke, v in docs[i].items() if ke != "embedding"}
            result["distance"] = 0.0
            candidates.append(result)

    if not candidates:
        # 3c — no filters / zero matches: full FAISS index search
        index = faiss.read_index(index_path)
        D, I = index.search(q_emb, fetch_k)
        for idx, dist in zip(I[0], D[0]):
            if 0 <= idx < len(docs):
                result = {ke: v for ke, v in docs[idx].items() if ke != "embedding"}
                result["distance"] = float(dist)
                candidates.append(result)

    # ── Step 4: rerank or truncate ────────────────────────────────────────────
    if use_rerank and len(candidates) > k:
        return rerank(query, candidates, top_n=k)
    return candidates[:k]


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build RAG FAISS index from timetable outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/phase5/rag_indexer.py                  # legacy outputs/\n"
            "  python src/phase5/rag_indexer.py --sem cse_sem3\n"
            "  python src/phase5/rag_indexer.py --sem cse_sem5\n"
        ),
    )
    parser.add_argument(
        "--sem",
        type=str,
        default=None,
        metavar="SEM_ID",
        help="Semester slug (e.g. cse_sem3). Omit for legacy mode.",
    )
    args = parser.parse_args()
    build_index(sem_id=args.sem)
