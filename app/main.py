from fastapi import FastAPI, UploadFile, File, HTTPException
from app.pdf_utils import extract_text_from_pdf
from app.summarizer import summarize_text
from app.schemas import SummaryResponse

app = FastAPI(title="LLM-Powered PDF Summarizer API")


@app.post("/summarize-pdf", response_model=SummaryResponse)
async def summarize_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    summary = summarize_text(text)

    return summary
