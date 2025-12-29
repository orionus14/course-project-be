from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.auth_routes import router as auth_router
from app.api.voice_routes import router as voice_router

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

os.environ["PATH"] += os.pathsep + r"D:\Python\ffmpeg\bin"

app.include_router(auth_router)
app.include_router(voice_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
