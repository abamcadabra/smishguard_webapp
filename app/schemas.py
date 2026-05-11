from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    message_text: str = Field(..., min_length=1, description="Incoming SMS or message text")
    received_at: Optional[str] = Field(default=None, description="Original receive timestamp")


class PredictResponse(BaseModel):
    clean_text: str
    extracted_url: Optional[str]
    normalized_url: Optional[str]

    text_prob: Optional[float]
    text_pred: int

    url_prob: Optional[float]
    url_pred: int

    final_pred: int
    fusion_rule: str
    reason: str

    latency_ms: float
    model_version_text: str
    model_version_url: str
    processed_at: str