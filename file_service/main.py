import os
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import motor.motor_asyncio
import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "file_service_db")
STORAGE_DIR = os.getenv("STORAGE_DIR", "C:/Users/welcome/OneDrive/Documents/AGENTIC AI POC's/File Processing Microservices/file_service/storage")
TEXT_EXTRACTOR_URL = os.getenv("TEXT_EXTRACTOR_URL", "http://localhost:8001/extract-text")
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8002/summarize")

os.makedirs(STORAGE_DIR, exist_ok=True)

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
files_coll = db.files
results_coll = db.results

app = FastAPI(title="File Service")

class UploadResp(BaseModel):
    file_id: str

@app.post("/upload", response_model=UploadResp)
async def upload(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    filename = file.filename
    ext = os.path.splitext(filename)[1]
    stored_name = f"{uid}{ext}"
    path = os.path.join(STORAGE_DIR, stored_name)

    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)

    meta = {
        "file_id": uid,
        "filename": filename,
        "stored_name": stored_name,
        "path": os.path.abspath(path),
        "upload_time": datetime.utcnow().isoformat(),
    }
    await files_coll.insert_one(meta)
    return UploadResp(file_id=uid)

@app.get("/file/{file_id}")
async def get_metadata(file_id: str):
    meta = await files_coll.find_one({"file_id": file_id}, {"_id":0})
    if not meta:
        raise HTTPException(status_code=404, detail="File not found")
    return meta

@app.get("/download/{file_id}")
async def download(file_id: str):
    meta = await files_coll.find_one({"file_id": file_id})
    if not meta:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=meta["path"], filename=meta["filename"])


@app.post("/process/{file_id}")
async def process_file(file_id: str):
    # 1) fetch metadata
    meta = await files_coll.find_one({"file_id": file_id})
    if not meta:
        raise HTTPException(status_code=404, detail="File not found")

    TEXT_EXTRACTOR_URL = os.getenv("TEXT_EXTRACTOR_URL", "http://localhost:8001/extract-text")
    AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8002/summarize")

    async with httpx.AsyncClient(timeout=120) as client:
        # Call text extractor by file_id (no file upload)
        extractor_url = f"{TEXT_EXTRACTOR_URL.rstrip('/')}/{file_id}"
        r = await client.post(extractor_url)
        if r.status_code != 200:
            # include the extractor response body for debugging
            raise HTTPException(status_code=500, detail=f"Text extraction failed: {r.status_code} {r.text}")

        # Now call AI service with just the file_id (AI will fetch chunks)
        payload = {"file_id": file_id}
        r2 = await client.post(AI_SERVICE_URL, json=payload, timeout=120)
        if r2.status_code != 200:
            raise HTTPException(status_code=500, detail=f"AI service failed: {r2.status_code} {r2.text}")
        ai_resp = r2.json()

    # store final output in DB (minimal fields)
    out = {
        "file_id": file_id,
        "summary": ai_resp.get("summary"),
        "processed_at": datetime.utcnow().isoformat()
    }
    await results_coll.replace_one({"file_id": file_id}, out, upsert=True)

    return {"status": "processed", "result": out}

@app.get("/results/{file_id}")
async def get_results(file_id: str):
    res = await results_coll.find_one({"file_id": file_id}, {"_id":0})
    if not res:
        raise HTTPException(status_code=404, detail="Results not found")
    return res

