import os
from typing import List
from openai import OpenAI


def get_client() -> OpenAI:
    """
    Lazily creates and returns an OpenAI client.

    Why this exists:
    - Prevents environment variable access at import time
    - Ensures OPENAI_API_KEY is available when the client is created
    - Avoids crashes during uvicorn reloads and multiprocessing
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")

    return OpenAI(api_key=api_key)


def chunk_text(text: str, chunk_size: int = 1200) -> List[str]:
    """
    Splits large text into fixed-size chunks.

    Why this exists:
    - LLMs have token limits
    - Chunking enables processing arbitrarily large documents
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end

    return chunks


def summarize_text(text: str) -> dict:
    """
    Summarizes a large document by:
    1. Chunking the text
    2. Summarizing each chunk independently
    3. Aggregating partial summaries into a final summary
    """
    client = get_client()

    chunks = chunk_text(text)
    partial_summaries = []

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
