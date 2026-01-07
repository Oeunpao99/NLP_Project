"""
NER Service for business logic
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import time
import uuid
import logging

from app.models.ner_prediction import NERPrediction
from app.schemas.prediction import NERRequest, BatchNERRequest
from ml.ner_predictor import KhmerNERPredictor
from ml.utils import extract_entities, format_ner_output
import json

logger = logging.getLogger(__name__)

class ModelLoader:
    """Singleton model loader"""
    
    _instance = None
    _predictor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Load the NER model"""
        if self._predictor is None:
            try:
                logger.info("Loading Khmer NER model...")
                self._predictor = KhmerNERPredictor("ml/model")
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        return self._predictor

    def reload_model(self):
        """Force reload the predictor from disk"""
        logger.info("Reloading NER predictor from disk...")
        self._predictor = None
        return self.load_model()

    def update_label_mapping(self, idx2label: dict, persist: bool = True):
        """Update the predictor's idx2label mapping in memory and optionally persist to disk.
        Validates that the mapping length equals the predictor's tagset size."""
        pred = self.predictor
        tagset = pred.tagset_size
        if len(idx2label) != tagset:
            raise ValueError(f"Provided mapping size {len(idx2label)} does not match tagset size {tagset}")
        # Convert keys to int keys if necessary, but store as str keys for JSON
        pred.idx2label = {int(k): v for k, v in idx2label.items()}
        # Also update label2idx for reverse lookup
        pred.label2idx = {v: int(k) for k, v in idx2label.items()}

        if persist:
            import json
            path = "ml/model/label_mappings.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"label2idx": pred.label2idx, "idx2label": {str(k): v for k, v in pred.idx2label.items()}}, f, ensure_ascii=False, indent=2)
            logger.info(f"Persisted new label mapping to {path}")
        return pred.idx2label
    
    @property
    def predictor(self):
        """Get the predictor instance"""
        if self._predictor is None:
            self.load_model()
        return self._predictor

model_loader = ModelLoader()

class NERService:
    """Service for NER operations"""
    
    def __init__(self, db: Session):
        self.db = db
        self.predictor = model_loader.predictor
        # Ensure the underlying CRF model is available before handling requests
        if getattr(self.predictor, 'model', None) is None:
            raise RuntimeError("NER model is not available. Install 'pytorch-crf' (pip install pytorch-crf) and restart the server.")
    
    def predict(self, request: NERRequest, user_id: str = None) -> Dict[str, Any]:
        """Make a NER prediction and save to database (optionally associated to a user)"""
        start_time = time.time()
        
        try:
            # Make prediction
            ner_results = self.predictor.predict_sentence(request.text)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000
            
            # Extract entities
            entities = extract_entities(ner_results)
            
            # Format output
            formatted_output = format_ner_output(ner_results, request.format)
            
            # Create database record
            prediction_record = NERPrediction(
                text=request.text,
                tokens=[r["token"] for r in ner_results],
                labels=[r["label"] for r in ner_results],
                entities=entities,
                formatted_output=formatted_output if isinstance(formatted_output, str) else json.dumps(formatted_output),
                format=request.format,
                inference_time_ms=inference_time,
                user_id=user_id,
                extra_metadata={
                    "language": "khmer",
                    "model_version": "1.0.0",
                    "request_id": request.request_id
                }
            )
            
            # Save to database
            self.db.add(prediction_record)
            self.db.commit()
            self.db.refresh(prediction_record)
            
            # Prepare response
            response = {
                "prediction_id": str(prediction_record.id),
                "text": request.text,
                "results": ner_results,
                "entities": entities,
                "formatted_output": formatted_output,
                "inference_time_ms": inference_time,
                "created_at": prediction_record.created_at.isoformat() + "Z",
                "model_version": "1.0.0"
            }
            
            logger.info(f"NER prediction made: {prediction_record.id}")
            
            return response
            
        except Exception as e:
            self.db.rollback()
            logger.exception("Prediction failed")
            raise
    def get_prediction_history(self, page: int = 1, page_size: int = 10, user_id: Optional[str] = None):
            """Get prediction history with pagination"""
            try:
                # Calculate offset
                offset = (page - 1) * page_size
                
                # Build query
                query = self.db.query(Prediction).order_by(Prediction.created_at.desc())
                
                # Filter by user if provided
                if user_id:
                    query = query.filter(Prediction.user_id == user_id)
                else:
                    # For anonymous users, return only their recent predictions
                    # This could be handled differently based on your requirements
                    query = query.filter(Prediction.user_id.is_(None))
                
                # Get total count
                total = query.count()
                
                # Get paginated results
                predictions = query.offset(offset).limit(page_size).all()
                
                # Calculate total pages
                total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
                
                return {
                    "predictions": predictions,
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1
                }
                
            except Exception as e:
                logger.error(f"Error fetching prediction history: {e}")
                raise
    def predict_batch(self, request: BatchNERRequest) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        results = []
        
        for text in request.texts:
            try:
                ner_request = NERRequest(text=text, format=request.format)
                result = self.predict(ner_request)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                results.append({
                    "text": text,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def get_prediction_history(
        self, 
        page: int = 1, 
        page_size: int = 10,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Get prediction history with pagination. If user_id is provided, return only that user's predictions."""
        
        offset = (page - 1) * page_size
        
        query = self.db.query(NERPrediction)
        if user_id:
            query = query.filter(NERPrediction.user_id == user_id)
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        predictions = query.order_by(NERPrediction.created_at.desc())\
            .offset(offset)\
            .limit(page_size)\
            .all()
        
        # Convert to dictionaries
        predictions_list = [pred.to_dict() for pred in predictions]
        
        return {
            "predictions": predictions_list,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
    
    def get_entity_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get entity statistics"""
        import json
        from datetime import datetime, timedelta
        from sqlalchemy import func
        
        # Calculate date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Query for entities in date range
        predictions = self.db.query(NERPrediction)\
            .filter(NERPrediction.created_at >= cutoff_date)\
            .all()
        
        # Count entities
        entity_counts = {}
        total_entities = 0
        
        for pred in predictions:
            try:
                if not pred.entities:
                    entities = {}
                else:
                    entities = json.loads(pred.entities) if isinstance(pred.entities, str) else pred.entities
                for entity_type, entity_list in entities.items():
                    count = len(entity_list)
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + count
                    total_entities += count
            except Exception:
                continue
        
        return {
            "total_entities": total_entities,
            "entity_counts": entity_counts,
            "period_days": days,
            "total_predictions": len(predictions)
        }