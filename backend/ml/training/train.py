"""
backend/ml/training/train.py

Training pipeline for the MedGuard action recognition model.
Architecture: Temporal pose feature extraction → LSTM → MLP classifier.

Trains on skeleton keypoint sequences to classify:
  - fall, seizure, cardiac, unconscious, normal

Usage:
  python -m backend.ml.training.train \
      --data_dir ./datasets/processed \
      --output_dir ./models \
      --epochs 50 \
      --batch_size 32
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from backend.utils.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

CLASSES = ["normal", "fall", "seizure", "cardiac", "unconscious", "facial_distress"]
NUM_KEYPOINTS = 17
FEATURES_PER_KP = 3  # x, y, confidence
SEQUENCE_LEN = 30    # frames (2s @ 15fps)


# ── Dataset ──────────────────────────────────────────────────

class PoseSequenceDataset(Dataset):
    """
    Loads .npy files with shape (N, SEQUENCE_LEN, 17*3).
    Labels stored in accompanying _labels.json.
    """

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(CLASSES)

        for cls in CLASSES:
            cls_dir = data_dir / cls
            if not cls_dir.exists():
                continue
            for npy_file in cls_dir.glob("*.npy"):
                seq = np.load(npy_file)
                if seq.shape != (SEQUENCE_LEN, NUM_KEYPOINTS * FEATURES_PER_KP):
                    seq = self._pad_or_trim(seq)
                label = self.label_encoder.transform([cls])[0]
                self.samples.append((seq.astype(np.float32), label))

        logger.info(f"dataset_loaded samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _pad_or_trim(seq: np.ndarray) -> np.ndarray:
        target = (SEQUENCE_LEN, NUM_KEYPOINTS * FEATURES_PER_KP)
        if seq.shape[0] < SEQUENCE_LEN:
            pad = np.zeros((SEQUENCE_LEN - seq.shape[0], seq.shape[1]))
            seq = np.vstack([seq, pad])
        return seq[:SEQUENCE_LEN, :NUM_KEYPOINTS * FEATURES_PER_KP]


# ── Model ─────────────────────────────────────────────────────

class MedicalActionModel(nn.Module):
    """
    Bi-LSTM + attention + MLP classifier for pose sequence classification.
    Input: (B, T, 51)  → Output: (B, n_classes)
    """

    def __init__(self, input_size: int = 51, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = len(CLASSES), dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        lstm_out, _ = self.lstm(x)          # (B, T, 2H)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        context = (attn_w * lstm_out).sum(dim=1)                  # (B, 2H)
        return self.classifier(context)


# ── Training ──────────────────────────────────────────────────

def train(args):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("medguard_action_recognition")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"training_device={device}")

    dataset = PoseSequenceDataset(args.data_dir)
    if len(dataset) == 0:
        logger.error("no_data_found check --data_dir")
        return

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MedicalActionModel(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
        })

        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            train_loss, correct, total = 0.0, 0, 0
            for seqs, labels in train_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(seqs)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)

            train_acc = correct / total
            train_loss /= total

            # Validate
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for seqs, labels in val_loader:
                    seqs, labels = seqs.to(device), labels.to(device)
                    logits = model(seqs)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += len(labels)

            val_acc = val_correct / val_total
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }, step=epoch)

            logger.info(f"epoch={epoch}/{args.epochs} "
                        f"loss={train_loss:.4f} "
                        f"train_acc={train_acc:.3f} "
                        f"val_acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(args.output_dir, exist_ok=True)
                save_path = os.path.join(args.output_dir, "action_classifier.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "epoch": epoch,
                    "val_acc": val_acc,
                }, save_path)
                mlflow.log_artifact(save_path)
                logger.info(f"model_saved val_acc={val_acc:.3f} path={save_path}")

    logger.info(f"training_complete best_val_acc={best_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./datasets/processed")
    parser.add_argument("--output_dir", default="./models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
