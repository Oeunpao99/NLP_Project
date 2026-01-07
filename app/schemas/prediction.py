"""
Pydantic schemas for NER predictions
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    database: bool
    model_loaded: bool
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "database": True,
                "model_loaded": True,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

class NERRequest(BaseModel):
    """NER prediction request"""
    text: str = Field(..., min_length=1, max_length=5000, description="Khmer text to analyze")
    format: str = Field("json", description="Output format: json, html, or text")
    request_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "ឧត្តមសេនីយ៍ឯក ជួន ណារិន្ទ ដឹកនាំ បើក កិច្ចប្រជុំ",
                "format": "json",
                "request_id": "req_123"
            }
        }

class TokenResult(BaseModel):
    """Token-level NER result"""
    token: str
    label: str
    entity_type: str
    confidence: Optional[float] = None

class NERResponse(BaseModel):
    """NER prediction response"""
    prediction_id: str
    text: str
    results: List[TokenResult]
    entities: Dict[str, List[str]]
    formatted_output: Any
    inference_time_ms: float
    created_at: str
    model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "ឧត្តមសេនីយ៍ឯក ជួន ណារិន្ទ ដឹកនាំ បើក កិច្ចប្រជុំ",
                "results": [
                    {"token": "ឧត្តមសេនីយ៍ឯក", "label": "B-PER", "entity_type": "PER", "confidence": 0.95},
                    {"token": "ជួន", "label": "I-PER", "entity_type": "PER", "confidence": 0.93},
                    {"token": "ណារិន្ទ", "label": "I-PER", "entity_type": "PER", "confidence": 0.94}
                ],
                "entities": {"PER": ["ឧត្តមសេនីយ៍ឯក ជួន ណារិន្ទ"]},
                "inference_time_ms": 25.5,
                "created_at": "2024-01-15T10:30:00Z",
                "model_version": "1.0.0"
            }
        }

class BatchNERRequest(BaseModel):
    """Batch NER prediction request"""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    format: str = Field("json", description="Output format")

class BatchNERResponse(BaseModel):
    """Batch NER prediction response"""
    results: List[Dict[str, Any]]

class PredictionHistory(BaseModel):
    """Prediction history response"""
    predictions: List[dict]
    total: int
    page: int
    page_size: int
    total_pages: int

class EntityStatsResponse(BaseModel):
    """Entity statistics response"""
    total_entities: int
    entity_counts: Dict[str, int]
    period_days: int
    total_predictions: int

class LabelMappingRequest(BaseModel):
    """Request body for updating label mapping"""
    idx2label: Dict[str, str]


class LabelMappingResponse(BaseModel):
    """Return the current label mapping and metadata"""
    idx2label: Dict[str, str]
    tagset_size: int
    is_bio: bool


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    code: Optional[int] = None