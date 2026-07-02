import database

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """Return top-K chunks ranked by BM25 score."""
    if database.bm25 is None or not database.corpus:
        return []
    tokens = database.tokenise(question)
    scores = database.bm25.get_scores(tokens)
    ranked = sorted(
        zip(scores, database.corpus), key=lambda x: x[0], reverse=True
    )
    return [
        {**doc, "score": float(score)}
        for score, doc in ranked[:top_k]
        if score > 0
    ]
