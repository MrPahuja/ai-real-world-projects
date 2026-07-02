import database

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """Retrieve top-K chunks using ChromaDB's local semantic search."""
    results = database.collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    return [
        {"text": doc, "source": meta["source"], "chunk": meta["chunk"], "score": 1 - dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
