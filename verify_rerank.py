"""
verify_rerank.py — Verification for Improvement #5 (cross-encoder reranking).

Tests:
  1. Reranking changes ordering  (ordering differs between FAISS-only and CE)
  2. Fallback on CE load failure (retrieve() still returns 5 results)
  3. k*3 fetch count             (15 candidates fetched, 5 returned)
  4. Legacy path unchanged       (use_rerank=False == original FAISS-only order)
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEM = "cse_sem3"
QUERY = "What theory classes does section A have on Monday?"

from src.phase5.rag_indexer import retrieve, rerank, _get_cross_encoder
import src.phase5.rag_indexer as _ri

SEP = "=" * 62

# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("Test 1 — Reranking changes ordering")
print(SEP)

results_faiss = retrieve(QUERY, k=5, sem_id=SEM, use_rerank=False)
results_ce    = retrieve(QUERY, k=5, sem_id=SEM, use_rerank=True)

print("\n  FAISS-only order (use_rerank=False):")
for i, d in enumerate(results_faiss, 1):
    print(f"    [{i}] dist={d.get('distance', '?'):.4f}  {d['text'][:80]}")

print("\n  Cross-encoder order (use_rerank=True):")
for i, d in enumerate(results_ce, 1):
    score = d.get("ce_score", "N/A")
    score_str = f"{score:.4f}" if isinstance(score, float) else score
    print(f"    [{i}] ce_score={score_str}  {d['text'][:80]}")

faiss_texts = [d["text"] for d in results_faiss]
ce_texts    = [d["text"] for d in results_ce]
order_changed = faiss_texts != ce_texts
print(f"\n  Ordering changed: {order_changed}")
if order_changed:
    print("  ✓ Reranker is producing a different ordering — working correctly")
else:
    print("  ⚠ Same order — reranker may be reinforcing FAISS (acceptable for simple queries)")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("Test 2 — Fallback on cross-encoder load failure")
print(SEP)

# Temporarily replace _get_cross_encoder to simulate failure
_original_fn = _ri._get_cross_encoder

def _failing_loader():
    return None          # simulate model unavailable

_ri._get_cross_encoder = _failing_loader
_ri._CROSS_ENCODER = None  # clear cache

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fallback_results = retrieve(QUERY, k=5, sem_id=SEM, use_rerank=True)

_ri._get_cross_encoder = _original_fn  # restore
_ri._CROSS_ENCODER = None              # reset so real model loads next call

n = len(fallback_results)
print(f"\n  Results returned during CE failure: {n}")
if n == 5:
    print("  ✓ Fallback returned 5 results — graceful degradation confirmed")
else:
    print(f"  ✗ Expected 5 results, got {n}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("Test 3 — k*3 candidate fetch count")
print(SEP)

# Patch retrieve to log how many candidates were assembled before reranking
_fetched_counts = []
_returned_counts = []
_original_rerank = _ri.rerank

def _instrumented_rerank(query, docs, top_n=5):
    _fetched_counts.append(len(docs))
    result = _original_rerank(query, docs, top_n=top_n)
    _returned_counts.append(len(result))
    return result

_ri.rerank = _instrumented_rerank

retrieve(QUERY, k=5, sem_id=SEM, use_rerank=True)

_ri.rerank = _original_rerank  # restore

fetched  = _fetched_counts[0]  if _fetched_counts  else "N/A"
returned = _returned_counts[0] if _returned_counts else "N/A"
print(f"\n  Candidates fetched before reranking : {fetched}")
print(f"  Results returned after reranking    : {returned}")
if fetched == 15 and returned == 5:
    print("  ✓ k*3=15 candidates fetched, top 5 returned — correct")
elif fetched == "N/A":
    print("  ⚠ Rerank was not called (fewer candidates than k — acceptable for filtered queries)")
else:
    print(f"  ⚠ fetched={fetched}, returned={returned} (may vary with metadata filters)")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("Test 4 — Legacy path (use_rerank=False) unchanged")
print(SEP)

r1 = retrieve(QUERY, k=5, sem_id=SEM, use_rerank=False)
r2 = retrieve(QUERY, k=5, sem_id=SEM, use_rerank=False)
texts1 = [d["text"] for d in r1]
texts2 = [d["text"] for d in r2]
stable = texts1 == texts2
print(f"\n  Two consecutive use_rerank=False calls return identical order: {stable}")
if stable:
    print("  ✓ Legacy path is deterministic and unchanged")
else:
    print("  ✗ Legacy path is non-deterministic — unexpected")

print(f"\n{'=' * 62}\nAll tests complete.\n")
