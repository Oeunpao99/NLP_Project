"""Authentication endpoints"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer

from app.schemas.auth import UserCreate, UserOut, Token
from app.models.database import get_db
from app.services.user_service import create_user, authenticate_user, get_user
from app.core.security import create_access_token, decode_access_token
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

@router.post("/auth/register", response_model=UserOut)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Check existing
        from app.services.user_service import get_user_by_email
        if get_user_by_email(db, user_in.email):
            raise HTTPException(status_code=400, detail="Email already registered")
        user = create_user(db, user_in.email, user_in.password)
        return UserOut(**user.to_dict())
    except ValueError as ve:
        # Validation error (e.g., email validation)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail="Failed to register user")

@router.post("/auth/login", response_model=Token)
def login(form_data: UserCreate, db: Session = Depends(get_db)):
    # Accepts JSON body with email & password
    user = authenticate_user(db, form_data.email, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(subject=user.id)
    return Token(access_token=token)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        user = get_user(db, user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_current_user_optional(request: Request, db: Session = Depends(get_db)):
    """Return current user if Authorization header present, otherwise None"""
    auth = request.headers.get("Authorization")
    if not auth:
        return None
    try:
        # Expect form: 'Bearer <token>'
        parts = auth.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        token = parts[1]
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = get_user(db, user_id)
        return user
    except Exception:
        return None

@router.get("/users/me", response_model=UserOut)
def read_users_me(current_user = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return UserOut(**current_user.to_dict())
