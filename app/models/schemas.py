from pydantic import BaseModel, Field
from typing import List
import uuid


class InvokeRequest(BaseModel):
    query: str
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class InvokeResponse(BaseModel):
    response: str
    sources: List[str]
    token_usage: TokenUsage
