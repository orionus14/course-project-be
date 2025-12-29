import librosa
import numpy as np


def load_audio(file_path, sr=16000):
    """
    Завантаження аудіо та конвертація в моно
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio


def extract_log_mel(audio, sr=16000, n_mels=80, n_fft=400, hop_length=160):
    """
    Обчислення log-mel спектрограми
    """
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    log_mel = librosa.power_to_db(spectrogram)
    # нормалізація
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
    return log_mel.T  # Транспонування: time x mel
