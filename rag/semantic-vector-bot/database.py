import chromadb

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="documents_local",
    metadata={"hnsw:space": "cosine"},
)

def reset_collection():
    global collection
    try:
        chroma.delete_collection("documents_local")
    except Exception:
        pass
    collection = chroma.get_or_create_collection(
        name="documents_local",
        metadata={"hnsw:space": "cosine"},
    )
    return collection
