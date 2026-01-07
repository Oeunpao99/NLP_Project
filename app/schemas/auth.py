"""Pydantic schemas for authentication"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class UserCreate(BaseModel):
    email: str
    password: str = Field(..., min_length=6)

    @field_validator('email')
    def validate_email(cls, v: str) -> str:
        """Validate email using email_validator but tolerate different return shapes."""
        # defer import to runtime to avoid hard dependency issues during startup
        try:
            from email_validator import validate_email
        except Exception:
            # if email-validator missing, perform a minimal sanity check
            if '@' not in v or len(v) < 5:
                raise ValueError('Invalid email')
            return v.lower()

        try:
            res = validate_email(v)
            # Try common attributes on ValidatedEmail objects across versions
            normalized = getattr(res, 'normalized', None)
            if not normalized:
                normalized = getattr(res, 'email', None)
            if not normalized:
                # fallback to string conversion
                normalized = str(res)
            return normalized.lower()
        except Exception as e:
            raise ValueError('Invalid email') from e

class UserOut(BaseModel):
    id: str
    email: str
    is_active: bool

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: Optional[str] = None
    exp: Optional[int] = None
