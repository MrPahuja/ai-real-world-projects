import string
from rank_bm25 import BM25Okapi

corpus: list[dict] = []
bm25: BM25Okapi | None = None

def tokenise(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace — classic BM25 tokens."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()

def rebuild_index() -> None:
    """Rebuild the BM25 index from the current corpus."""
    global bm25
    if corpus:
        bm25 = BM25Okapi([tokenise(doc["text"]) for doc in corpus])
    else:
        bm25 = None

def clear_index() -> None:
    """Wipe the in-memory corpus and index."""
    global bm25
    corpus.clear()
    bm25 = None
