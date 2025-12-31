import torch
import librosa
import subprocess
import os

# Перевірка доступності transformers
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"  Transformers недоступний: {e}")
    print(f"Встановіть: pip install transformers torch torchvision")
    TRANSFORMERS_AVAILABLE = False

# Налаштування
# Використовуємо перевірену модель для української ASR
MODEL_NAME = "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm"  # Найкраща для української
# Альтернативи:
# MODEL_NAME = "ukrainian-nlp/wav2vec2-xls-r-300m-uk"
# MODEL_NAME = "aramakus/wav2vec2-large-xlsr-ukrainian"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ініціалізація моделі
processor = None
model = None

if TRANSFORMERS_AVAILABLE:
    try:
        print(f"Завантаження Wav2Vec2 моделі...")
        print(f"Device: {DEVICE}")

        # Завантаження з HuggingFace
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

        print(f" Модель завантажена: {MODEL_NAME}")
    except Exception as e:
        print(f" Помилка завантаження моделі: {e}")
        print(f"\nПереконайтесь:")
        print(f"1. Є інтернет (модель завантажується з HuggingFace)")
        print(f"2. Встановлені: pip install transformers torch librosa")
        model = None
        processor = None


def transcribe_custom(file_path: str) -> dict:
    """
    Розпізнавання української мови через Wav2Vec2

    Args:
        file_path: шлях до WebM/WAV аудіо

    Returns:
        dict: {"text": str, "language": "uk"}
    """
    # Перевірка доступності моделі
    if not TRANSFORMERS_AVAILABLE:
        return {
            "text": "",
            "language": "uk",
            "error": "Transformers не встановлено. Запустіть: pip install transformers torch torchvision"
        }

    if model is None or processor is None:
        return {
            "text": "",
            "language": "uk",
            "error": "Модель не завантажена. Перевірте інтернет-з'єднання."
        }

    # Конвертація WebM в WAV
    wav_path = file_path.replace(".webm", "_custom.wav")

    try:
        # Використовуємо subprocess замість os.system
        result = subprocess.run([
            "ffmpeg", "-y", "-i", file_path,
            "-ar", "16000", "-ac", "1",
            wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    except subprocess.CalledProcessError as e:
        return {
            "text": "",
            "language": "uk",
            "error": f"FFmpeg помилка: {e.stderr.decode() if e.stderr else str(e)}"
        }
    except FileNotFoundError:
        return {
            "text": "",
            "language": "uk",
            "error": "FFmpeg не знайдено. Додайте його до PATH."
        }

    # Завантаження аудіо через librosa
    try:
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)

        if len(audio) == 0:
            raise ValueError("Пусте аудіо")

    except Exception as e:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return {
            "text": "",
            "language": "uk",
            "error": f"Помилка читання аудіо: {e}"
        }

    # Розпізнавання
    try:
        # Підготовка входу
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Декодування
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # Очищення тексту
        transcription = transcription.strip()

    except Exception as e:
        transcription = ""
        print(f"  Помилка розпізнавання: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Видалення тимчасового файлу
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except:
                pass

    return {
        "text": transcription,
        "language": "uk"
    }


# Тестування
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ТЕСТ CUSTOM ASR (WAV2VEC2)")
    print("="*70)

    if not TRANSFORMERS_AVAILABLE:
        print(" Transformers недоступний")
        print("\nВстановіть залежності:")
        print("  pip install transformers torch torchvision librosa")
        exit(1)

    if model is None:
        print(" Модель не завантажена")
        exit(1)

    print(" Модель готова")
    print(f"\nДля тестування:")
    print(f"  result = transcribe_custom('test.webm')")
    print(f"  print(result)")
