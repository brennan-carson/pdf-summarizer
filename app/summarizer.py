import os
from typing import List
from openai import OpenAI
from app.pdf_utils import extract_text_from_pdf


def get_client() -> OpenAI:
    """
    Lazily creates and returns an OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")
    return OpenAI(api_key=api_key)


def chunk_text(text: str, chunk_size: int = 1200) -> List[str]:
    """
    Splits large text into fixed-size chunks to stay within model limits.
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

    Flow:
    1. Convert PDF bytes → plain text
    2. Split text into chunks
    3. Summarize each chunk
    4. Combine summaries into a final result
    """
    client = get_client()

    # Step 1: PDF → text
    text = extract_text_from_pdf(pdf_bytes)
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
                {
                    "role": "system",
                    "content": "You are a technical document summarizer."
                },
                {
                    "role": "user",
                    "content": chunk
                }
            ],
        )

        partial_summaries.append(response.choices[0].message.content)

    # Step 4: Aggregate summaries
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Combine the following summaries into a concise final summary "
                    "with clear key points."
                )
            },
            {
                "role": "user",
                "content": "\n".join(partial_summaries)
            }
        ],
    )

    return {
        "document_summary": final_response.choices[0].message.content,
        "key_points": partial_summaries[:5],
        "chunk_count": len(chunks),
    }
