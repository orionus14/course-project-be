import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
from tqdm import tqdm
import traceback
import sys

from asr.model import ASRModel
from asr.vocab import VOCAB, CHAR2IDX, IDX2CHAR

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "dataset/uk"
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
TRAIN_TSV = os.path.join(DATA_DIR, "train.tsv")
VAL_TSV = os.path.join(DATA_DIR, "dev.tsv")
MODEL_OUT = "models/asr_uk_model.pth"
CHECKPOINT_DIR = "checkpoints"

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160

BATCH_SIZE = 16
EPOCHS = 10
MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Параметри для відновлення
RESUME_TRAINING = True  # Автоматично продовжувати з останнього checkpoint
SAVE_EVERY_N_EPOCHS = 2  # Зберігати checkpoint кожні 2 епохи

# -----------------------------
# DATASET
# -----------------------------


class CommonVoiceDataset(Dataset):
    def __init__(self, tsv_path, max_samples=None):
        print(f"Завантаження датасету: {tsv_path}")
        try:
            df = pd.read_csv(tsv_path, sep="\t")
            print(f"  Завантажено рядків: {len(df)}")
        except Exception as e:
            print(f" Помилка читання TSV: {e}")
            raise

        df = df[df["sentence"].notna() & df["path"].notna()]
        print(f"  Після фільтрації NaN: {len(df)}")

        # Фільтрація занадто довгих/коротких
        df = df[df["sentence"].str.len() > 3]
        df = df[df["sentence"].str.len() < 200]
        print(f"  Після фільтрації довжини: {len(df)}")

        if max_samples:
            df = df.iloc[:max_samples]
            print(f"  Обмежено до: {len(df)}")

        self.df = df.reset_index(drop=True)
        self.failed_loads = 0
        print(f" Датасет готовий: {len(self.df)} семплів\n")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(CLIPS_DIR, row["path"])

        try:
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            self.failed_loads += 1
            if self.failed_loads < 10:  # Логуємо перші 10 помилок
                print(f"  Помилка завантаження {audio_path}: {e}")
            # Повертаємо мінімальний семпл
            return torch.zeros(50, N_MELS), torch.tensor([CHAR2IDX[" "]], dtype=torch.long)

        # Mel-спектрограма
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel = librosa.power_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        mel = torch.tensor(mel.T, dtype=torch.float32)

        # Текст
        text = row["sentence"].lower()
        text = [c for c in text if c in CHAR2IDX]

        if len(text) == 0:
            text = [" "]

        target = torch.tensor([CHAR2IDX[c] for c in text], dtype=torch.long)
        return mel, target


# -----------------------------
# COLLATE FUNCTION
# -----------------------------

def collate_fn(batch):
    mels, targets = zip(*batch)

    mel_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0]
                                  for t in targets], dtype=torch.long)

    mels_padded = nn.utils.rnn.pad_sequence(mels, batch_first=True)
    targets_cat = torch.cat(targets)

    return mels_padded, targets_cat, mel_lengths, target_lengths


# -----------------------------
# ДЕКОДУВАННЯ
# -----------------------------

def ctc_greedy_decode(logits):
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    prev_idx = -1
    output = []
    for idx in preds:
        if idx != prev_idx and idx != 0:
            output.append(IDX2CHAR.get(idx, "?"))
        prev_idx = idx
    return "".join(output)


# -----------------------------
# ВАЛІДАЦІЯ
# -----------------------------

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for mels, targets, mel_lens, tgt_lens in val_loader:
            try:
                mels, targets = mels.to(DEVICE), targets.to(DEVICE)

                logits = model(mels)
                log_probs = logits.log_softmax(dim=-1)
                log_probs = log_probs.permute(1, 0, 2)

                loss = criterion(log_probs, targets, mel_lens, tgt_lens)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    num_batches += 1
            except Exception as e:
                print(f"  Помилка валідації батчу: {e}")
                continue

    if num_batches == 0:
        return float('inf')

    avg_loss = total_loss / num_batches
    return avg_loss


# -----------------------------
# ЗБЕРЕЖЕННЯ ТА ЗАВАНТАЖЕННЯ
# -----------------------------

def save_checkpoint(epoch, model, optimizer, val_loss, is_best=False):
    """Зберігає checkpoint"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }

    # Звичайний checkpoint
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)

    # Останній checkpoint
    last_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
    torch.save(checkpoint, last_path)

    # Найкращий checkpoint
    if is_best:
        torch.save(checkpoint, MODEL_OUT)
        print(f"  ✓ Найкраща модель збережена: {MODEL_OUT}")

    return checkpoint_path


def load_checkpoint():
    """Завантажує останній checkpoint, якщо існує"""
    last_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

    if not RESUME_TRAINING or not os.path.exists(last_path):
        return None

    try:
        checkpoint = torch.load(last_path, map_location=DEVICE)
        print(f"\n Знайдено checkpoint з епохи {checkpoint['epoch']}")
        print(f"  Val loss: {checkpoint['val_loss']:.4f}")
        return checkpoint
    except Exception as e:
        print(f"⚠️  Помилка завантаження checkpoint: {e}")
        return None


# -----------------------------
# TRAINING
# -----------------------------

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ПОЧАТОК ТРЕНУВАННЯ ASR МОДЕЛІ")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Vocab size: {len(VOCAB)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Resume training: {RESUME_TRAINING}")
    print(f"{'='*70}\n")

    # Датасети
    try:
        train_ds = CommonVoiceDataset(TRAIN_TSV, max_samples=MAX_TRAIN_SAMPLES)
        val_ds = CommonVoiceDataset(VAL_TSV, max_samples=MAX_VAL_SAMPLES)
    except Exception as e:
        print(f" Помилка завантаження датасетів: {e}")
        traceback.print_exc()
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    # Модель
    model = ASRModel(num_classes=len(VOCAB)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Завантаження checkpoint
    start_epoch = 1
    best_val_loss = float('inf')

    checkpoint = load_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"→ Продовження з епохи {start_epoch}\n")

    print(f"Починаємо тренування з епохи {start_epoch}...\n")

    # Основний цикл
    for epoch in range(start_epoch, EPOCHS + 1):
        try:
            print(f"\n{'─'*70}")
            print(f"EPOCH {epoch}/{EPOCHS}")
            print(f"{'─'*70}")

            # TRAINING
            model.train()
            train_loss = 0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Training")
            for batch_idx, (mels, targets, mel_lens, tgt_lens) in enumerate(pbar):
                try:
                    mels, targets = mels.to(DEVICE), targets.to(DEVICE)

                    # Forward
                    logits = model(mels)
                    log_probs = logits.log_softmax(dim=-1)
                    log_probs = log_probs.permute(1, 0, 2)

                    # Loss
                    loss = criterion(log_probs, targets, mel_lens, tgt_lens)

                    # Перевірка на NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(
                            f"\n⚠️  NaN/Inf loss в батчі {batch_idx}, пропускаємо...")
                        continue

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=5.0)
                    optimizer.step()

                    train_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n OUT OF MEMORY! Зменшіть BATCH_SIZE")
                        torch.cuda.empty_cache()
                        sys.exit(1)
                    else:
                        print(f"\n  Помилка в батчі {batch_idx}: {e}")
                        continue

            avg_train_loss = train_loss / \
                num_batches if num_batches > 0 else float('inf')

            # VALIDATION
            print(f"\nВалідація...")
            val_loss = validate(model, val_loader, criterion)

            # Learning rate
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            # Виведення результатів
            print(f"\n{'─'*70}")
            print(f"РЕЗУЛЬТАТИ EPOCH {epoch}")
            print(f"{'─'*70}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Learning Rate: {new_lr:.6f}")
            if old_lr != new_lr:
                print(f"  → LR змінено: {old_lr:.6f} → {new_lr:.6f}")

            # Приклад декодування
            if epoch % 2 == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        sample_mel, sample_target = train_ds[0]
                        sample_mel = sample_mel.unsqueeze(0).to(DEVICE)
                        logits = model(sample_mel)[0]
                        decoded = ctc_greedy_decode(logits)
                        target_text = "".join([IDX2CHAR[i]
                                              for i in sample_target.numpy()])
                        print(f"\n Приклад розпізнавання:")
                        print(f"   Target: '{target_text}'")
                        print(f"   Pred:   '{decoded}'")
                    except Exception as e:
                        print(f"  Помилка декодування: {e}")

            # Збереження checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if epoch % SAVE_EVERY_N_EPOCHS == 0 or is_best:
                checkpoint_path = save_checkpoint(
                    epoch, model, optimizer, val_loss, is_best)
                print(f"  Checkpoint збережено: {checkpoint_path}")

        except KeyboardInterrupt:
            print(f"\n\n  Тренування перервано користувачем!")
            print(f"Зберігаємо checkpoint...")
            save_checkpoint(epoch, model, optimizer, val_loss, is_best=False)
            print(f"✓ Checkpoint збережено. Можна продовжити пізніше.")
            sys.exit(0)

        except Exception as e:
            print(f"\n КРИТИЧНА ПОМИЛКА в епосі {epoch}:")
            traceback.print_exc()
            print(f"\nЗберігаємо emergency checkpoint...")
            save_checkpoint(epoch, model, optimizer, val_loss, is_best=False)
            print(f"Продовжуємо з наступної епохи...\n")
            continue

    # Завершення
    print(f"\n{'='*70}")
    print(f"ТРЕНУВАННЯ ЗАВЕРШЕНО!")
    print(f"{'='*70}")
    print(f"Найкраща validation loss: {best_val_loss:.4f}")
    print(f"Модель збережена: {MODEL_OUT}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n ФАТАЛЬНА ПОМИЛКА:")
        traceback.print_exc()
        sys.exit(1)
