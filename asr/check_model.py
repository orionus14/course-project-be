import os
import torch
from asr.model import ASRModel
from asr.vocab import VOCAB

print("="*60)
print("ПЕРЕВІРКА МОДЕЛІ ASR")
print("="*60)

# 1. Перевірка файлу моделі
MODEL_PATH = "models/asr_uk_model.pth"
print(f"\n1. Перевірка файлу моделі:")
print(f"   Шлях: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"    Файл існує")
    print(f"   Розмір: {size_mb:.2f} MB")

    if size_mb < 0.5:
        print(f"     УВАГА: Файл дуже маленький!")
        print(f"   Це може бути порожня або не натренована модель")
    else:
        print(f"    Розмір нормальний")
else:
    print(f"    ФАЙЛ НЕ ЗНАЙДЕНО!")
    print(f"   Потрібно спочатку натренувати модель через train_asr.py")
    exit(1)

# 2. Завантаження моделі
print(f"\n2. Завантаження моделі:")
try:
    num_classes = len(VOCAB)
    print(f"   Кількість класів (літер): {num_classes}")

    model = ASRModel(num_classes=num_classes)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    # Перевірка структури checkpoint
    if isinstance(checkpoint, dict):
        print(f"    Checkpoint - це словник")
        print(f"   Ключі: {list(checkpoint.keys())}")

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(
                f"    Модель завантажена з epoch {checkpoint.get('epoch', '?')}")
            print(f"   Validation loss: {checkpoint.get('val_loss', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"    Модель завантажена (старий формат)")
    else:
        model.load_state_dict(checkpoint)
        print(f"    Модель завантажена")

    model.eval()

except Exception as e:
    print(f"    ПОМИЛКА: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. Тест моделі на випадкових даних
print(f"\n3. Тест моделі:")
print(f"   Створюємо випадкові дані (імітація звуку)...")

# Створюємо "фейкове" аудіо: 100 таймстепів x 80 мел-частот
fake_mel = torch.randn(1, 100, 80)  # batch=1, time=100, mels=80
print(f"   Розмір входу: {fake_mel.shape}")

try:
    with torch.no_grad():
        output = model(fake_mel)

    print(f"    Модель працює!")
    print(f"   Розмір виходу: {output.shape}")
    print(f"   Очікували: (1, 100, {num_classes})")

    # Перевірка виходів
    logits = output[0]  # Беремо перший батч
    probs = torch.softmax(logits, dim=-1)
    predicted_classes = torch.argmax(probs, dim=-1)

    # Статистика
    unique_classes = torch.unique(predicted_classes)
    blank_percent = (predicted_classes == 0).float().mean() * 100

    print(f"\n    Аналіз виходів:")
    print(f"   Унікальних класів: {len(unique_classes)} з {num_classes}")
    print(f"   Blank (клас 0): {blank_percent:.1f}%")

    if blank_percent > 95:
        print(f"     ПРОБЛЕМА: Модель видає майже тільки blank!")
        print(f"   Це означає, що модель НЕ НАТРЕНОВАНА")
    elif len(unique_classes) < 10:
        print(f"     ПРОБЛЕМА: Модель використовує мало класів")
        print(f"   Потрібно більше тренування")
    else:
        print(f"    Модель виглядає нормально")

except Exception as e:
    print(f"    ПОМИЛКА ПРИ РОБОТІ МОДЕЛІ: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. Висновок
print(f"\n{'='*60}")
print(f"ВИСНОВОК:")
print(f"{'='*60}")

if blank_percent > 95:
    print(f" Модель НЕ ГОТОВА до використання")
    print(f"\nЩо робити:")
    print(f"1. Запустіть: python train_asr.py")
    print(f"2. Почекайте завершення тренування (30-60 хв)")
    print(f"3. Перезапустіть backend: python app.py")
    print(f"\nАБО використовуйте Whisper замість custom моделі:")
    print(f"   У frontend виберіть engine='whisper'")
elif os.path.getsize(MODEL_PATH) < 1024 * 1024:  # < 1MB
    print(f"  Модель існує, але занадто маленька")
    print(f"Рекомендується перетренувати")
else:
    print(f" Модель виглядає робочою")
    print(f"Якщо все одно не працює - проблема може бути в:")
    print(f"  - Якості аудіо з мікрофону")
    print(f"  - Конвертації WebM → WAV")
    print(f"  - Несумісності даних тренування та інференсу")

print(f"{'='*60}")
