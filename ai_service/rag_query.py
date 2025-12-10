import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "../text_extractor/faiss.index")

_db = None
_idx = None
_model = None


async def init_rag(mongo_uri: str, db_name: str):
    """
    Initialize Mongo connection, SentenceTransformer model and FAISS index.
    Call this once at application startup.
    """
    global _db, _idx, _model
    client = AsyncIOMotorClient(mongo_uri)
    _db = client[db_name]

    # load model (this may download weights on first run)
    _model = SentenceTransformer(MODEL_NAME)

    # load or create FAISS IndexIDMap (maps int ids -> vectors)
    if os.path.exists(INDEX_PATH):
        try:
            _idx = faiss.read_index(INDEX_PATH)
        except Exception as e:
            # if index file corrupted or incompatible, recreate empty IndexIDMap
            print(f"Warning: failed reading faiss index at {INDEX_PATH}: {e}")
            base = faiss.IndexFlatL2(EMBED_DIM)
            _idx = faiss.IndexIDMap(base)
    else:
        base = faiss.IndexFlatL2(EMBED_DIM)
        _idx = faiss.IndexIDMap(base)


def save_index():
    """Save FAISS index to disk (call after modifying the index)."""
    global _idx
    if _idx is None:
        return
    # ensure directory exists
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(_idx, INDEX_PATH)


async def retrieve_with_rag(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Given a query string, return top_k most similar chunk docs from Mongo.
    Each returned item is the chunk document (without _id).
    """
    global _idx, _model, _db
    if _idx is None or _model is None or _db is None:
        raise RuntimeError("RAG not initialized; call init_rag first")

    q_emb = _model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = _idx.search(q_emb, top_k)
    ids = I[0].tolist()
    results: List[Dict[str, Any]] = []
    for idx_id in ids:
        if idx_id < 0:
            continue
        doc = await _db.chunks.find_one({"index_id": int(idx_id)}, {"_id": 0})
        if doc:
            results.append(doc)
    return results


async def get_chunks_for_file(file_id: str) -> List[Dict[str, Any]]:
    """
    Return stored chunks for `file_id`, ordered by index_id (ascending).
    Each chunk is a dict like {"chunk_id", "file_id", "text", "index_id"}.
    """
    global _db
    if _db is None:
        raise RuntimeError("DB not initialized. Call init_rag first.")
    cursor = _db.chunks.find({"file_id": file_id}).sort("index_id", 1)
    docs: List[Dict[str, Any]] = []
    async for doc in cursor:
        doc.pop("_id", None)
        docs.append(doc)
    return docs
