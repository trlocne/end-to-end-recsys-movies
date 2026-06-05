from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime

class RegisterRequest(BaseModel):
    user_id: int
    metadata: Optional[dict] = None

class RegisterResponse(BaseModel):
    user_id: int
    created_at: datetime
    message: str

class LoginRequest(BaseModel):
    user_id: int

class LoginResponse(BaseModel):
    user_id: int
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Any] = None

class InteractionEvent(BaseModel):
    user_id: int
    item_id: int
    event_type: str = "click"
    rating: Optional[float] = None

class RecommendationItem(BaseModel):
    item_id: int
    ann_score: float
    source: str
    rerank_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    user_id: int
    items: List[dict]
    scenario: str
    latency_ms: float
    query: Optional[str] = None
    source_item_id: Optional[int] = None
