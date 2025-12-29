from fastapi import APIRouter, Response, HTTPException, Depends
from app.schemas.user_schema import UserCreate, UserResponse
from app.services.auth_service import create_user, authenticate_user, get_user_by_email
from app.core.security import create_access_token
from app.models.user_model import user_entity
from app.utils.auth_dependency import get_current_user
from pydantic import BaseModel, EmailStr

router = APIRouter(
    prefix="/api",
    tags=["Auth"]
)

# --- нова модель для логіну ---


class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Реєстрація


@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, response: Response):
    if get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = create_user(user.name, user.email, user.password)

    token = create_access_token({"sub": str(new_user["_id"])})

    response.set_cookie(
        key="token",
        value=token,
        httponly=False,
        secure=False,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
        path="/"
    )

    return user_entity(new_user)

# --- Логін з новою моделлю ---


@router.post("/login", response_model=UserResponse)
def login(user: UserLogin, response: Response):
    db_user = authenticate_user(user.email, user.password)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": str(db_user["_id"])})

    response.set_cookie(
        key="token",
        value=token,
        httponly=False,
        secure=False,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
        path="/"
    )

    return user_entity(db_user)

# Отримання поточного користувача


@router.get("/user", response_model=UserResponse)
def get_user(current_user: dict = Depends(get_current_user)):
    return user_entity(current_user)

# Вихід


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="token", path="/")
    return {"message": "Logged out successfully"}
