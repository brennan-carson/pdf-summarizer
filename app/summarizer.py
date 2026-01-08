import os
from typing import List
from openai import OpenAI
from PyPDF2 import PdfReader
from io import BytesIO

def get_client() -> OpenAI:
    """
    Lazily creates and returns an OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")
    return OpenAI(api_key=api_key)


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """
    Extracts text from PDF bytes.

    Why this exists:
    - PDF files are binary; LLMs cannot read PDF natively
    - Converts document into a string suitable for chunking
    """
    pdf_file = BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1200) -> List[str]:
    """
    Splits large text into fixed-size chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks


def summarize_text(pdf_bytes: bytes) -> dict:
    """
    Summarizes a PDF document.

    Steps:
    1. Extract text from PDF
    2. Chunk the text
    3. Summarize each chunk independently
    4. Aggregate partial summaries into a final summary
    """
    client = get_client()

    # Step 1: PDF → text
    text = pdf_bytes_to_text(pdf_bytes)
    if not text:
        raise ValueError("No text could be extracted from the PDF")

    # Step 2: Chunk text
    chunks = chunk_text(text)
    partial_summaries = []

    # Step 3: Summarize each chunk
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical document summarizer."},
                {"role": "user", "content": chunk}
            ],
        )
        partial_summaries.append(response.choices[0].message.content)

    # Step 4: Aggregate partial summaries
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Combine the following summaries into a concise final summary with clear key points."},
            {"role": "user", "content": "\n".join(partial_summaries)}
        ],
    )

    return {
        "document_summary": final_response.choices[0].message.content,
        "key_points": partial_summaries[:5],
        "chunk_count": len(chunks),
    }
