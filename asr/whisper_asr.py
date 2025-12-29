import whisper
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)


def transcribe_whisper(audio_path: str):
    result = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        fp16=False
    )
    return {
        "text": result["text"].strip(),
        "language": "en",
        "engine": "whisper"
    }
