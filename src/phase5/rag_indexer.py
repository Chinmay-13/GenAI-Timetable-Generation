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

_RAG_ROOT = Path(__file__).resolve().parents[2]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from config import SECTIONS, DAYS, OUTPUT_DIR, resolve_output_path

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = str(OUTPUT_DIR / "rag_index.faiss")
DOCS_PATH  = str(OUTPUT_DIR / "rag_docs.json")


def build_index():
    """Build FAISS index from all section and faculty timetable CSVs."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"RAG dependencies not installed: {e}")
        print("Run: pip install sentence-transformers faiss-cpu")
        return None, None, None

    model = SentenceTransformer(MODEL_NAME)
    docs = []

    # One document per section per day
    for section in SECTIONS:
        csv_path = resolve_output_path(f"section_{section}_timetable.csv")
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                day = row["Day"]
                slots = {f"P{i}": str(row.get(f"P{i}", "----")) for i in range(1, 7)}
                occupied = {p: v for p, v in slots.items() if v != "----"}
                doc = {
                    "text": (
                        f"Section {section} on {day}: "
                        + (", ".join(f"{p}={v}" for p, v in occupied.items())
                           if occupied else "no classes")
                    ),
                    "section": section,
                    "day": day,
                    "source": f"section_{section}_timetable.csv",
                }
                docs.append(doc)
        except FileNotFoundError:
            continue

    # One document per faculty per day (F01-F20)
    for i in range(1, 21):
        fid = f"F{i:02d}"
        csv_path = resolve_output_path(f"faculty_{fid}_timetable.csv")
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                day = row["Day"]
                slots = {f"P{j}": str(row.get(f"P{j}", "----")) for j in range(1, 7)}
                occupied = {p: v for p, v in slots.items() if v != "----"}
                doc = {
                    "text": (
                        f"Faculty {fid} on {day}: "
                        + (", ".join(f"{p}={v}" for p, v in occupied.items())
                           if occupied else "no classes")
                    ),
                    "faculty": fid,
                    "day": day,
                    "source": f"faculty_{fid}_timetable.csv",
                }
                docs.append(doc)
        except FileNotFoundError:
            continue

    if not docs:
        print("No timetable CSVs found in outputs/. Run run_all.py first.")
        return None, None, None

    # Encode all documents
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # Build FAISS index
    import faiss
    import numpy as np
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    print(f"RAG index built: {len(docs)} documents")
    print(f"  Index: {INDEX_PATH}")
    print(f"  Docs : {DOCS_PATH}")
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
    print("\nSample retrieval: 'Section A Monday timetable'")
    results = retrieve("Section A Monday timetable")
    for r in results:
        print(f"  [{r.get('distance', 0):.2f}] {r['text']}")
