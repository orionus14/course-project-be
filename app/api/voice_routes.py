from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Depends
from fastapi.responses import FileResponse
import os
import uuid
from datetime import datetime
from asr.whisper_asr import transcribe_whisper
from asr.custom_asr import transcribe_custom
from app.utils.auth_dependency import get_current_user
from app.services.voice_service import (
    create_voice_record,
    get_user_voice_records,
    get_voice_record_by_id,
    delete_voice_record
)
from app.models.voice_model import voice_record_entity, voice_records_entity
from app.schemas.voice_schema import VoiceRecordResponse

router = APIRouter(prefix="/api/voice", tags=["Voice"])

# Папка для збереження аудіо
AUDIO_STORAGE = "uploads/audio"
os.makedirs(AUDIO_STORAGE, exist_ok=True)


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    engine: str = Query("whisper", regex="^(whisper|custom)$"),
    save_audio: bool = Query(True),
    current_user: dict = Depends(get_current_user)
):
    """
    Розпізнає голос з аудіо файлу

    - **file**: аудіо файл (webm, wav, mp3)
    - **engine**: whisper (EN) або custom (UA)
    - **save_audio**: зберігати чи ні аудіо файл
    """

    # Генеруємо унікальне ім'я файлу
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] or ".webm"
    temp_filename = f"{file_id}{file_extension}"
    temp_path = os.path.join(AUDIO_STORAGE, temp_filename)

    try:
        # Зберігаємо тимчасовий файл
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Розпізнаємо
        if engine == "whisper":
            result = transcribe_whisper(temp_path)
        else:
            result = transcribe_custom(temp_path)

        # Перевіряємо на помилки
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Зберігаємо в MongoDB
        audio_filename = temp_filename if save_audio else None
        voice_record = create_voice_record(
            user_id=str(current_user["_id"]),
            text=result["text"],
            language=result["language"],
            engine=engine,
            audio_filename=audio_filename
        )

        # Видаляємо файл якщо не треба зберігати
        if not save_audio and os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "id": str(voice_record["_id"]),
            "text": result["text"],
            "language": result["language"],
            "engine": engine,
            "audio_filename": audio_filename,
            "created_at": voice_record["created_at"]
        }

    except Exception as e:
        # Видаляємо файл у випадку помилки
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/records", response_model=list[VoiceRecordResponse])
async def get_records(
    limit: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Отримує всі голосові записи користувача"""
    records = get_user_voice_records(str(current_user["_id"]), limit)
    return voice_records_entity(records)


@router.get("/records/{record_id}", response_model=VoiceRecordResponse)
async def get_record(
    record_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Отримує конкретний запис"""
    record = get_voice_record_by_id(record_id, str(current_user["_id"]))
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return voice_record_entity(record)


@router.delete("/records/{record_id}")
async def delete_record(
    record_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Видаляє запис"""
    success = delete_voice_record(record_id, str(current_user["_id"]))
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"message": "Record deleted successfully"}


@router.get("/audio/{filename}")
async def get_audio(
    filename: str,
    current_user: dict = Depends(get_current_user)
):
    """Завантажує аудіо файл"""
    file_path = os.path.join(AUDIO_STORAGE, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        file_path,
        media_type="audio/webm",
        filename=filename
    )
