from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import soundfile as sf

from asr.model import ASRModel
from asr.features import load_audio, extract_log_mel_spectrogram
from asr.decoder import ctc_greedy_decode
from asr.vocab import VOCAB

app = FastAPI()

# CORS для Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Ініціалізація моделі
num_classes = len(VOCAB)
model = ASRModel(num_classes)
model.eval()  # тестовий режим


@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    # Збереження тимчасового файлу
    audio_path = f"temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Обробка аудіо
    audio = load_audio(audio_path)
    features = extract_log_mel_spectrogram(audio)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(
        0)  # batch x time x mel

    # Прогон через модель
    with torch.no_grad():
        logits = model(features)
        logits = logits[0]  # беремо перший (і єдиний) batch

    # Декодування
    text = ctc_greedy_decode(logits)

    return {"text": text}
