from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class VoiceRecordCreate(BaseModel):
    text: str
    language: str
    engine: str


class VoiceRecordResponse(BaseModel):
    id: str
    user_id: str
    text: str
    language: str
    engine: str
    audio_filename: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
