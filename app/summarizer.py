import os
from typing import List
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chunk_text(text: str, chunk_size: int = 1200) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end

    return chunks


def summarize_text(text: str) -> dict:
    chunks = chunk_text(text)
    partial_summaries = []

    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical document summarizer."},
                {"role": "user", "content": chunk}
            ],
        )

        partial_summaries.append(response.choices[0].message.content)

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Combine the following summaries into a concise final summary with key points."},
            {"role": "user", "content": "\n".join(partial_summaries)}
        ],
    )

    return {
        "document_summary": final_response.choices[0].message.content,
        "key_points": partial_summaries[:5],
        "chunk_count": len(chunks),
    }
