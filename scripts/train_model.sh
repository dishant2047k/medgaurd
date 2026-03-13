#!/usr/bin/env bash
# ============================================================
# MedGuard AI — Model Training Script
# ============================================================
set -e

echo "🧠 MedGuard AI — Training Pipeline"
echo "===================================="

# Activate venv
source venv/bin/activate || true

# 1. Prepare dataset
echo "📦 Step 1: Preparing dataset..."
python datasets/prepare_data.py \
  --input_dir ./datasets/raw \
  --output_dir ./datasets/processed \
  --sequence_len 30 \
  --fps 15 || echo "⚠️  No raw data found — skipping preparation"

# 2. Train model
echo "🏋️  Step 2: Training action classifier..."
python -m backend.ml.training.train \
  --data_dir ./datasets/processed \
  --output_dir ./models \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001

echo ""
echo "✅ Training complete!"
echo "📁 Model saved to: ./models/action_classifier.pt"
echo "📊 View experiments: http://localhost:5000"
