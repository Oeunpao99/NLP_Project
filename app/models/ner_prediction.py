"""
Database models for NER predictions
"""

from sqlalchemy import Column, String, Text, DateTime, Float, JSON, ForeignKey
from sqlalchemy.sql import func
# REMOVE THIS: from sqlalchemy.orm import relationship
from app.models.database import Base
import uuid

class NERPrediction(Base):
    """NER prediction record"""
    
    __tablename__ = "predictions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    text = Column(Text, nullable=False)
    
    tokens = Column(JSON, nullable=True)
    labels = Column(JSON, nullable=True)
    entities = Column(JSON, nullable=True)
    
    formatted_output = Column(Text, nullable=True)
    format = Column(String(20), default="json")
    
    inference_time_ms = Column(Float, default=0.0)
    language = Column(String(10), default="khmer")
    model_version = Column(String(20), default="1.0.0")
    
    # Map to 'metadata' column in database but use different attribute name
    extra_metadata = Column("metadata", JSON, nullable=True)
    
    # Foreign key to user - BUT NO RELATIONSHIP
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)
    # REMOVE THIS ENTIRE LINE: user = relationship("User", backref="predictions")

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "text": self.text,
            "tokens": self.tokens,
            "labels": self.labels,
            "entities": self.entities,
            "formatted_output": self.formatted_output,
            "format": self.format,
            "inference_time_ms": self.inference_time_ms,
            "language": self.language,
            "model_version": self.model_version,
            "metadata": self.extra_metadata,
            "user_id": self.user_id
        }