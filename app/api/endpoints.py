"""
FastAPI endpoints for NER
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional
from datetime import datetime
import logging

from app.models.database import get_db
from app.services.ner_service import NERService
from app.schemas.prediction import *
from app.api.auth import get_current_user_optional
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    from app.services.ner_service import model_loader
    
    try:
        # Check database connection
        db.execute(text('SELECT 1'))
        database_ok = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        database_ok = False
    
    # Check model (ensure predictor exists and internal model is loaded)
    try:
        pred = model_loader.predictor
        model_loaded = getattr(pred, 'model', None) is not None
        if not model_loaded:
            logger.warning("Model health check: predictor present but internal model is not loaded (CRF missing).")
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        model_loaded = False
    
    status_text = "healthy" if database_ok and model_loaded else "unhealthy"
    
    return HealthResponse(
        status=status_text,
        database=database_ok,
        model_loaded=model_loaded,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

@router.post("/predict", response_model=NERResponse)
async def predict(
    request: NERRequest = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """Predict NER tags for Khmer text (POST with JSON body)"""
    try:
        service = NERService(db)
        user_id = getattr(current_user, 'id', None)
        result = service.predict(request, user_id=user_id)
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/predict", response_model=NERResponse)
async def predict_get(
    text: str = Query(..., min_length=1, description="Khmer text to analyze"),
    format: str = Query("json", description="Output format: json, html, or text"),
    request_id: Optional[str] = Query(None, description="Optional request id to track this request"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """Predict NER tags for Khmer text (GET with query params)"""
    try:
        service = NERService(db)
        req = NERRequest(text=text, format=format, request_id=request_id)
        user_id = getattr(current_user, 'id', None)
        result = service.predict(req, user_id=user_id)
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/predictions", response_model=PredictionHistory)
async def get_predictions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_optional)
):
    """Get prediction history (global or user-specific if authenticated)"""
    try:
        service = NERService(db)
        user_id = getattr(current_user, 'id', None)
        result = service.get_prediction_history(
            page=page,
            page_size=page_size,
            user_id=user_id
        )
        
        return PredictionHistory(**result)
        
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# --- Model label mapping endpoints ---
@router.get("/model/labels", response_model=LabelMappingResponse)
async def get_label_mapping():
    """Get current model label mapping"""
    try:
        from app.services.ner_service import model_loader
        pred = model_loader.predictor
        idx2 = {str(k): v for k, v in pred.idx2label.items()}
        # check if BIO-like
        is_bio = any(v.startswith(("B-","I-")) for v in idx2.values())
        return LabelMappingResponse(idx2label=idx2, tagset_size=pred.tagset_size, is_bio=is_bio)
    except Exception as e:
        logger.error(f"Error fetching label mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/labels", response_model=LabelMappingResponse)
async def set_label_mapping(mapping: LabelMappingRequest):
    """Set a new label mapping and persist it to model dir"""
    try:
        from app.services.ner_service import model_loader
        idx2 = mapping.idx2label
        new_idx2 = model_loader.update_label_mapping(idx2, persist=True)
        pred = model_loader.predictor
        is_bio = any(v.startswith(("B-","I-")) for v in new_idx2.values())
        return LabelMappingResponse(idx2label={str(k): v for k, v in new_idx2.items()}, tagset_size=pred.tagset_size, is_bio=is_bio)
    except ValueError as ve:
        logger.error(f"Invalid mapping: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error updating label mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/reload")
async def reload_model():
    """Reload model from disk (weights and label mapping)"""
    try:
        from app.services.ner_service import model_loader
        model_loader.reload_model()
        return JSONResponse({"status": "reloaded"})
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))