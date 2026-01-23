# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
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

# Setup templates folder for HTML frontend
templates = Jinja2Templates(directory="templates")


# --------------------
# Root route
# --------------------
@app.get("/")
def root():
    return {"message": "PDF Summarizer API is running"}


# --------------------
# Existing API: JSON-based PDF summarization
# --------------------
@app.post("/summarize-pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        pdf_bytes = await file.read()
        result = summarize_text(pdf_bytes)
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------
# New frontend route: serve HTML page
# --------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    """
    Serves a minimal HTML page with a file upload form.
    """
    return templates.TemplateResponse("index.html", {"request": request})


# --------------------
# New frontend route: handle form submission
# --------------------
@app.post("/summarize", response_class=HTMLResponse)
async def summarize_form(request: Request, file: UploadFile = File(...)):
    """
    Accepts PDF uploaded via the HTML form, processes it,
    and returns the same page with the summary rendered.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "summary": "Error: Only PDF files are accepted", "key_points": []},
        )

    # Read PDF bytes
    pdf_bytes = await file.read()

    try:
        # Use your existing summarizer
        result = summarize_text(pdf_bytes)
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "summary": f"Error: {str(e)}", "key_points": []},
        )

    # Render HTML page with summary and key points
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "summary": result["document_summary"],
            "key_points": result["key_points"],
        },
    )
