from collections import defaultdict
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import BM25_POOL, VECTOR_POOL, FINAL_TOP_K, RRF_K
from database import qdrant, chunk_store, bm25_store
from ingestion import embed, tokenize

def bm25_retrieve(query: str, tenant_id: str, user_id: str) -> list[str]:
    """
    Score all tenant chunks with BM25, filter by user ACL, return top chunk IDs.
    """
    if tenant_id not in bm25_store:
        return []

    store  = bm25_store[tenant_id]
    scores = store["index"].get_scores(tokenize(query))

    ranked = sorted(
        zip(scores, store["chunks"]),
        key=lambda x: x[0],
        reverse=True,
    )
    return [
        c.id for sc, c in ranked
        if sc > 0 and (not c.allowed_users or user_id == "admin" or user_id in c.allowed_users)
    ][:BM25_POOL]


def vector_retrieve(query: str, tenant_id: str, user_id: str) -> list[str]:
    """
    Dense vector search in Qdrant, filtered by tenant_id and post-filtered by user ACL.
    """
    query_emb = embed(query)
    response = qdrant.query_points(
        collection_name="kb",
        query=query_emb,
        limit=VECTOR_POOL * 3,   # fetch extra to compensate for post-filtering loss
        query_filter=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        ),
    )
    
    # Log scores to console
    print(f"\n--- [Vector Search] Query: '{query}' ---")
    for r in response.points:
        print(f"  - Document: '{r.payload.get('source')}' | Similarity Score: {r.score:.4f}")

    accessible = [
        r for r in response.points
        if r.score >= 0.50 and (not r.payload.get("allowed_users") or user_id == "admin" or user_id in r.payload["allowed_users"])
    ]
    return [r.payload["chunk_id"] for r in accessible[:VECTOR_POOL]]


def rrf_merge(bm25_ids: list[str], vector_ids: list[str]) -> list[str]:
    """
    Reciprocal Rank Fusion (RRF) to merge candidate ranked lists.
    """
    scores: dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(bm25_ids):
        scores[cid] += 1.0 / (rank + RRF_K)
    for rank, cid in enumerate(vector_ids):
        scores[cid] += 1.0 / (rank + RRF_K)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def hybrid_search(query: str, tenant_id: str, user_id: str):
    """Orchestrate the full hybrid retrieval pipeline."""
    bm25_ids  = bm25_retrieve(query, tenant_id, user_id)
    vec_ids   = vector_retrieve(query, tenant_id, user_id)
    fused_ids = rrf_merge(bm25_ids, vec_ids)[:FINAL_TOP_K]
    chunks    = [chunk_store[cid] for cid in fused_ids if cid in chunk_store]

    def srcs(ids):
        return list({chunk_store[c].source for c in ids if c in chunk_store})

    debug = {
        "bm25_hits":    len(bm25_ids),
        "vector_hits":  len(vec_ids),
        "bm25_sources": srcs(bm25_ids),
        "vec_sources":  srcs(vec_ids),
        "final":        [c.source for c in chunks],
    }
    return chunks, debug
