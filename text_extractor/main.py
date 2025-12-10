import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from .text_utils import (
    extract_text_from_pdf_bytes,
    extract_text_from_docx_bytes,
    clean_text,
)
from .rag_utils import get_model, init_db, store_chunks_and_embeddings
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "file_service_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))

app = FastAPI(title="Text Extractor")

@app.on_event("startup")
async def startup_event():
    app.state.model = get_model()
    app.state.db = await init_db(MONGO_URI, DB_NAME)

def chunk_text_by_chars(text: str, size=800):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+size]
        chunks.append(chunk)
        i += size
    return chunks

FILE_STORAGE = os.getenv(
    "FILE_STORAGE_PATH",
    "C:/Users/welcome/OneDrive/Documents/AGENTIC AI POC's/File Processing Microservices/file_service/storage"
)

def chunk_text(text: str, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

@app.post("/extract-text/{file_id}")
async def extract_text_by_id(file_id: str):
    # locate the actual file from storage
    for ext in [".pdf", ".docx", ".txt"]:
        file_path = os.path.join(FILE_STORAGE, f"{file_id}{ext}")
        if os.path.exists(file_path):
            break
    else:
        raise HTTPException(404, f"File with ID {file_id} not found in storage")

    # read file bytes
    with open(file_path, "rb") as f:
        data = f.read()

    ext = os.path.splitext(file_path)[1].lower()

    # extract text
    if ext == ".pdf":
        text = extract_text_from_pdf_bytes(data)
    elif ext == ".docx":
        text = extract_text_from_docx_bytes(data)
    else:
        try:
            text = data.decode("utf-8")
        except:
            text = ""

    text = clean_text(text)
    if not text:
        raise HTTPException(400, "No text extracted")

    chunks = chunk_text(text, CHUNK_SIZE)

    # save chunks + FAISS
    await store_chunks_and_embeddings(
        file_id=file_id,
        chunks=chunks,
        model=app.state.model
    )

    return JSONResponse({
        "file_id": file_id,
        "num_chunks": len(chunks),
        "sample_chunks": chunks[:2],
        "text": text
    })