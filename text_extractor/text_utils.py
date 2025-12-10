from pypdf import PdfReader
from docx import Document
import io

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text_parts)

def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    doc = Document(io.BytesIO(docx_bytes))
    parts = []
    for p in doc.paragraphs:
        parts.append(p.text)
    return "\n".join(parts)

def clean_text(text: str) -> str:
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)
