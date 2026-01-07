from pydantic import BaseModel
from typing import List


class SummaryResponse(BaseModel):
    document_summary: str
    key_points: List[str]
    chunk_count: int
