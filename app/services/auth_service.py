from app.db.mongo import users_collection
from app.core.security import hash_password, verify_password


def get_user_by_email(email: str):
    return users_collection.find_one({"email": email})


def create_user(name: str, email: str, password: str):
    user = {
        "name": name,
        "email": email,
        "password": hash_password(password)
    }
    result = users_collection.insert_one(user)
    user["_id"] = result.inserted_id
    return user


def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user["password"]):
        return None
    return user
