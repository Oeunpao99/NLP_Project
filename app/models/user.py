"""
User database model
"""
from typing import Optional
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import VARCHAR
from app.models.database import Base
import uuid
from pydantic import BaseModel, EmailStr

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class DeleteAccountRequest(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    is_active: bool
    created_at: str
    last_login: Optional[str] = None
    
    class Config:
        from_attributes = True
class User(Base):
    __tablename__ = "users"

    id = Column(VARCHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
