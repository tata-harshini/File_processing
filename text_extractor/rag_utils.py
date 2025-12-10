import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from motor.motor_asyncio import AsyncIOMotorClient
import uuid

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384  # embedding dim for all-MiniLM-L6-v2
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss.index")

_db = None
_model = None
_idx = None

async def init_db(mongo_uri="mongodb://localhost:27017", db_name="ai_mini_project"):
    global _db
    client = AsyncIOMotorClient(mongo_uri)
    _db = client[db_name]
    # ensure counters doc exists
    await _db.counters.update_one({"_id": "faiss_id"}, {"$setOnInsert": {"seq": 1}}, upsert=True)
    return _db

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def load_faiss_index():
    global _idx
    if _idx is not None:
        return _idx
    if os.path.exists(INDEX_PATH):
        _idx = faiss.read_index(INDEX_PATH)
    else:
        base = faiss.IndexFlatL2(EMBED_DIM)
        _idx = faiss.IndexIDMap(base)
    return _idx

def save_faiss_index():
    global _idx
    if _idx is None:
        return
    faiss.write_index(_idx, INDEX_PATH)

async def _get_next_ids(n: int):
    # atomically increment the counter in Mongo and return starting id
    res = await _db.counters.find_one_and_update(
        {"_id": "faiss_id"},
        {"$inc": {"seq": n}},
        return_document=True
    )
    # seq after increment; compute start = seq - n
    seq_after = res["seq"]
    start = seq_after - n
    ids = np.arange(start, start + n).astype(np.int64)
    return ids

async def store_chunks_and_embeddings(file_id: str, chunks: list, model):
    """
    chunks: list[str]
    returns: list of dicts inserted into DB (with index_id)
    """
    idx = load_faiss_index()
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    # ensure embeddings dtype float32
    vecs = np.array(embeddings).astype("float32")
    n = vecs.shape[0]
    ids = await _get_next_ids(n)  # numpy array of ints

    # add vectors with explicit ids
    idx.add_with_ids(vecs, ids)
    save_faiss_index()

    # store metadata for each chunk with its index id
    docs = []
    for i, chunk_text in enumerate(chunks):
        docs.append({
            "chunk_id": str(uuid.uuid4()),
            "file_id": file_id,
            "text": chunk_text,
            "index_id": int(ids[i])
        })
    if docs:
        await _db.chunks.insert_many(docs)
    return docs

async def retrieve_by_query(query: str, top_k=5):
    idx = load_faiss_index()
    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = idx.search(q_emb, top_k)
    ids = I[0].tolist()
    # query mongo for index_id in ids
    results = []
    for idx_id in ids:
        if idx_id < 0:
            continue
        doc = await _db.chunks.find_one({"index_id": int(idx_id)}, {"_id":0})
        if doc:
            results.append(doc)
    return results
