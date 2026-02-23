from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.db.models.user import User
from app.schemas.user import UserCreate, UserOut
from app.services import user_service

router = APIRouter()

@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user_signup(
    *,
    db: AsyncSession = Depends(deps.get_database),
    user_in: UserCreate
) -> Any:
    """
    Create new user without the need to be logged in.
    """
    user = await user_service.get_user_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this username already exists in the system",
        )
    user = await user_service.create_user(db, user_in=user_in)
    return user

@router.get("/me", response_model=UserOut)
async def read_user_me(
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Get current user.
    """
    return current_user
