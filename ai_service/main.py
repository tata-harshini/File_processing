import os
import re
import math
from math import log
from collections import Counter
from typing import Any, List, Optional, Dict
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# package-relative import (ai_service must be a package and uvicorn run from project root)
from . import rag_query

# load env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "file_service_db")

app = FastAPI(title="AI Service")

# -------------------------
# Text helpers / summarizer
# -------------------------
def split_into_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ").strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

def build_term_freq(tokens: List[str]) -> Counter:
    return Counter(tokens)

def compute_idf(doc_tokens: List[List[str]]) -> Dict[str, float]:
    N = len(doc_tokens)
    df = Counter()
    for tokens in doc_tokens:
        unique = set(tokens)
        for tok in unique:
            df[tok] += 1
    idf = {}
    for tok, freq in df.items():
        idf[tok] = log((N + 1) / (freq + 1)) + 1.0
    return idf

def score_sentences(sentences: List[str], idf: Dict[str, float], top_k: int = 3) -> List[str]:
    sentence_scores = []
    for s in sentences:
        tokens = tokenize(s)
        if not tokens:
            sentence_scores.append((s, 0.0))
            continue
        tf = build_term_freq(tokens)
        score = 0.0
        for w, cnt in tf.items():
            score += cnt * idf.get(w, 0.0)
        sentence_scores.append((s, score))
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [s for s, _ in sentence_scores[:top_k]]
    selected_in_order = [s for s in sentences if s in selected]
    return selected_in_order

def simple_extractive_summary(text: str, num_sentences: int = 3) -> str:
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    sent_tokens = [tokenize(s) for s in sentences]
    idf = compute_idf(sent_tokens)
    selected = score_sentences(sentences, idf, top_k=num_sentences)
    summary = " ".join(selected)
    if not summary.strip():
        summary = text[:500].strip() + ("..." if len(text) > 500 else "")
    return summary

# -------------------------
# API models
# -------------------------
class SummarizePayload(BaseModel):
    file_id: str

class AnalyzePayload(BaseModel):
    file_id: str

# -------------------------
# Startup - initialize RAG
# -------------------------
@app.on_event("startup")
async def startup_event():
    # initialize rag (loads model + faiss + mongo connection)
    try:
        await rag_query.init_rag(MONGO_URI, DB_NAME)
        print("RAG initialized.")
    except Exception as e:
        print("Error initializing RAG:", repr(e))
    # print registered routes for debugging
    print("Registered routes:")
    for route in app.routes:
        print(f" - {route.path}  [{getattr(route, 'methods', None)}]")

# -------------------------
# /summarize endpoint
# -------------------------
@app.post("/summarize")
async def summarize(payload: SummarizePayload):

    file_id = payload.file_id

    # fetch stored chunks for that file_id
    try:
        chunks = await rag_query.get_chunks_for_file(file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chunks: {e}")

    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for this file_id")

    # reconstruct full text
    text = "\n\n".join([c.get("text", "") for c in chunks])
    if not text.strip():
        raise HTTPException(status_code=400, detail="Extracted text is empty")

    # summarise (extractive)
    try:
        summary = simple_extractive_summary(text, num_sentences=3)
    except Exception as e:
        print("summarizer error:", repr(e))
        summary = text[:500].strip() + ("..." if len(text) > 500 else "")


    # normal minimal response
    return {"file_id": file_id, "summary": summary}

# -------------------------
# /analyze endpoint (insights, topics, sentiment)
# -------------------------
# small English stoplist for topic extraction
_ANALYZE_STOPWORDS = {
    "the","and","is","in","to","of","a","for","with","on","that","this","as","are",
    "by","an","it","be","or","from","we","which","at","these","have","has","was","will",
    "such","their","they","but","not","can","may","also","other","using","used","use"
}

# small sentiment lexicon (extend as needed)
_POS_WORDS = {
    "good","great","excellent","positive","benefit","improve","improved","success","successful",
    "easy","useful","helpful","recommend","love","like","best","effective","gain","advantage"
}
_NEG_WORDS = {
    "bad","poor","fail","failure","problem","error","hard","difficult","worse","worst",
    "issue","issues","negative","hate","risk","risks","limitation","limitations","loss"
}

def _top_keywords(text: str, top_n: int = 8):
    toks = tokenize(text)  # re-uses tokenize()
    toks = [t for t in toks if t not in _ANALYZE_STOPWORDS]
    c = Counter(toks)
    return [{"term": term, "count": count} for term, count in c.most_common(top_n)]

def _sentiment_simple(text: str):
    toks = tokenize(text)
    if not toks:
        return {"score": 0.0, "label": "neutral", "pos_count": 0, "neg_count": 0}
    pos = sum(1 for t in toks if t in _POS_WORDS)
    neg = sum(1 for t in toks if t in _NEG_WORDS)
    score = (pos - neg) / max(1, len(toks))
    if score > 0.01:
        label = "positive"
    elif score < -0.01:
        label = "negative"
    else:
        label = "neutral"
    return {"score": round(score, 4), "label": label, "pos_count": pos, "neg_count": neg}

@app.post("/analyze")
async def analyze(payload: AnalyzePayload):

    file_id = payload.file_id

    # 1) load chunks
    try:
        chunks = await rag_query.get_chunks_for_file(file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chunks: {e}")

    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for this file_id")

    # 2) reconstruct full text
    text = "\n\n".join([c.get("text", "") for c in chunks]).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Extracted text is empty")

    # 3) compute insights
    chars = len(text)
    words = len(tokenize(text))
    sentences = len(split_into_sentences(text))
    avg_sentence_length = (words / sentences) if sentences else (words if words else 0)
    num_chunks = len(chunks)
    avg_chunk_chars = int(chars / num_chunks) if num_chunks else chars
    estimated_read_time_minutes = math.ceil(words / 200) if words else 0

    insights = {
        "file_id": file_id,
        "num_chunks": num_chunks,
        "chars": chars,
        "words": words,
        "sentences": sentences,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_chunk_chars": avg_chunk_chars,
        "estimated_read_time_minutes": estimated_read_time_minutes
    }

    # 4) topics (top keywords)
    topics = _top_keywords(text, top_n=8)

    # 5) sentiment (simple lexicon-based)
    sentiment = _sentiment_simple(text)

    # 6) return only insights, topics and sentiment (no chunk text)
    return {
        "insights": insights,
        "topics": topics,
        "sentiment": sentiment
    }
