# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.summarizer import summarize_text
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered PDF Summarizer API",
    description="Upload PDFs and get summaries with key points using OpenAI GPT",
    version="1.0.0"
)

# Simple root route to verify server is running
@app.get("/")
def root():
    return {"message": "PDF Summarizer API is running"}

# Endpoint to summarize a PDF
@app.post("/summarize-pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    # Only allow PDF files
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        # Read the uploaded file
        pdf_bytes = await file.read()
        # Use your summarizer function
        result = summarize_text(pdf_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
