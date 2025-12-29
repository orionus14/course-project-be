from fastapi import Cookie, HTTPException
from jose import jwt, JWTError
from bson import ObjectId
from app.core.config import SECRET_KEY, ALGORITHM
from app.db.mongo import users_collection


def get_current_user(token: str | None = Cookie(default=None)):
    """Отримує поточного користувача з токену в cookie"""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
