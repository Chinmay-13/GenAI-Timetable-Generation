"""
rag_indexer.py — Builds a FAISS vector index over timetable output CSVs
so the AI can retrieve relevant rows instead of reading entire files.

Usage:
  python src/phase5/rag_indexer.py       # build index
  from src.phase5.rag_indexer import retrieve  # query
"""

from pathlib import Path
import sys
import json
import os

_RAG_ROOT = Path(__file__).resolve().parents[2]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from config import SECTIONS, OUTPUT_DIR, resolve_output_path

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = str(OUTPUT_DIR / "rag_index.faiss")
DOCS_PATH  = str(OUTPUT_DIR / "rag_docs.json")


def _iter_timetable_files(pattern: str):
    return sorted(Path(OUTPUT_DIR).glob(pattern))


def _period_columns(df) -> list[str]:
    return [f"P{i}" for i in range(1, 7) if f"P{i}" in df.columns]


def build_index():
    """Build FAISS index from all section and faculty timetable CSVs."""
    try:
        import pandas as pd
    except ImportError as e:
        print(f"RAG dependency missing: {e}")
        return None, None, None
    docs = []

    section_files = _iter_timetable_files("section_*_timetable.csv")
    faculty_files = _iter_timetable_files("faculty_*_timetable.csv")

    discovered_sections = []
    for csv_path in section_files:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            continue

        section = csv_path.stem.removeprefix("section_").removesuffix("_timetable")
        discovered_sections.append(section)
        period_cols = _period_columns(df)

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

    for csv_path in faculty_files:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            continue

        faculty_id = csv_path.stem.removeprefix("faculty_").removesuffix("_timetable")
        period_cols = _period_columns(df)

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

    # One document per (day, period) slot from room_assignment.csv
    room_csv = resolve_output_path("room_assignment.csv")
    if room_csv.exists():
        try:
            rdf = pd.read_csv(room_csv)
            # Group by Day + Period
            for (day, period), grp in rdf.groupby(["Day", "Period"]):
                occupied = grp[grp["Room"] != "ROOM_UNASSIGNED"]
                free_names = set()
                try:
                    rooms_df = pd.read_csv(_RAG_ROOT / "data" / "rooms.csv")
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
        print("No timetable CSVs found in outputs/. Run run_all.py first.")
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

    Path(DOCS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    section_docs = sum(1 for d in docs if "section" in d)
    faculty_docs = sum(1 for d in docs if "faculty" in d)
    room_docs    = sum(1 for d in docs if d.get("source") == "room_assignment.csv")
    print(f"RAG docs built: {len(docs)} documents")
    print(f"  Section docs : {section_docs}")
    print(f"  Faculty docs : {faculty_docs}")
    print(f"  Room docs    : {room_docs}")
    print(f"  Docs : {DOCS_PATH}")

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        print(f"RAG embeddings skipped: {e}")
        print("Install sentence-transformers and faiss-cpu to rebuild rag_index.faiss.")
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

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"  Index: {INDEX_PATH}")
    return index, docs, model


def retrieve(query: str, k: int = 5) -> list:
    """Retrieve top-k relevant timetable records for a query string."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        return []

    if not Path(INDEX_PATH).exists() or not Path(DOCS_PATH).exists():
        print("RAG index not found. Run: python src/phase5/rag_indexer.py")
        return []

    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        if 0 <= idx < len(docs):
            result = docs[idx].copy()
            result["distance"] = float(dist)
            results.append(result)
    return results


if __name__ == "__main__":
    build_index()
