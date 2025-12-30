from datetime import datetime
from bson import ObjectId
from app.db.mongo import voice_records_collection


def create_voice_record(user_id: str, text: str, language: str, engine: str, audio_filename: str = None):
    """Створює новий голосовий запис"""
    record = {
        "user_id": ObjectId(user_id),
        "text": text,
        "language": language,
        "engine": engine,
        "audio_filename": audio_filename,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    result = voice_records_collection.insert_one(record)
    record["_id"] = result.inserted_id
    return record


def get_user_voice_records(user_id: str, limit: int = 50):
    """Отримує всі записи користувача"""
    records = voice_records_collection.find(
        {"user_id": ObjectId(user_id)}
    ).sort("created_at", -1).limit(limit)
    return list(records)


def get_voice_record_by_id(record_id: str, user_id: str):
    """Отримує конкретний запис"""
    return voice_records_collection.find_one({
        "_id": ObjectId(record_id),
        "user_id": ObjectId(user_id)
    })


def update_voice_record_text(record_id: str, user_id: str, new_text: str):
    """Оновлює текст запису"""
    result = voice_records_collection.update_one(
        {
            "_id": ObjectId(record_id),
            "user_id": ObjectId(user_id)
        },
        {
            "$set": {
                "text": new_text,
                "updated_at": datetime.utcnow()
            }
        }
    )
    return result.modified_count > 0


def delete_voice_record(record_id: str, user_id: str):
    """Видаляє запис"""
    result = voice_records_collection.delete_one({
        "_id": ObjectId(record_id),
        "user_id": ObjectId(user_id)
    })
    return result.deleted_count > 0
