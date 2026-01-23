# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.summarizer import summarize_text

# Load environment variables from .env (API keys, etc.)
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered PDF Summarizer API",
    description="Upload PDFs and get summaries with key points using OpenAI GPT",
    version="1.0.0",
)

# Root route to verify server is running
@app.get("/")
def root():
    return {"message": "PDF Summarizer API is running"}

# Endpoint to summarize a PDF
@app.post("/summarize-pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    # 1. Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        # 2. Read uploaded file into raw bytes
        pdf_bytes = await file.read()

        # 3. Delegate ALL processing to the summarizer layer
        result = summarize_text(pdf_bytes)

        # 4. Return structured JSON response
        return JSONResponse(content=result)

    except ValueError as e:
        # Expected, user-related errors (e.g., unreadable PDF)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected internal errors
        raise HTTPException(status_code=500, detail=str(e))
