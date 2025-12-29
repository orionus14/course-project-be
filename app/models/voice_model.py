def voice_record_entity(record: dict) -> dict:
    """Перетворює MongoDB документ уdict"""
    return {
        "id": str(record["_id"]),
        "user_id": str(record["user_id"]),
        "text": record["text"],
        "language": record["language"],
        "engine": record["engine"],
        "audio_filename": record.get("audio_filename"),
        "created_at": record["created_at"]
    }


def voice_records_entity(records: list) -> list:
    """Перетворює список записів"""
    return [voice_record_entity(record) for record in records]
