# ============================================================
# MedGuard AI — Windows Setup Script (PowerShell)
# Run this from inside C:\medguard-ai\
# Usage: .\scripts\setup-windows.ps1
# ============================================================

Write-Host ""
Write-Host "  ██╗  ██╗███████╗██████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ " -ForegroundColor Cyan
Write-Host "  ███╗███║██╔════╝██╔══██╗██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗" -ForegroundColor Cyan
Write-Host "  ████╔████║█████╗  ██║  ██║██║  ███╗██║   ██║███████║██████╔╝██║  ██║" -ForegroundColor Cyan
Write-Host "  MedGuard AI — Windows Setup" -ForegroundColor Cyan
Write-Host ""

# ── Check Python version ─────────────────────────────────────
Write-Host "Step 1/7: Checking Python..." -ForegroundColor Yellow
$pyver = python --version 2>&1
Write-Host "  Found: $pyver"

# Warn about Python 3.13
if ($pyver -match "3\.13") {
    Write-Host ""
    Write-Host "  ⚠  Python 3.13 detected." -ForegroundColor Yellow
    Write-Host "  Some packages (mediapipe, torch) work best on Python 3.11 or 3.12." -ForegroundColor Yellow
    Write-Host "  The install will still work — we use compatible version ranges." -ForegroundColor Yellow
    Write-Host ""
}

# ── Create venv ──────────────────────────────────────────────
Write-Host "Step 2/7: Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  venv already exists, skipping creation."
} else {
    python -m venv venv --without-pip
    Write-Host "  Downloading pip..."
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py" -ErrorAction SilentlyContinue
    if (Test-Path "get-pip.py") {
        .\venv\Scripts\python.exe get-pip.py
        Remove-Item "get-pip.py"
    } else {
        python -m venv venv  # fallback with bundled pip
    }
}

# ── Activate venv ────────────────────────────────────────────
Write-Host "Step 3/7: Activating venv..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip first
python -m pip install --upgrade pip

# ── Install Stage 1: Core ────────────────────────────────────
Write-Host ""
Write-Host "Step 4/7: Installing core packages (API + DB)..." -ForegroundColor Yellow
pip install -r requirements-core.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Core install failed. Check internet connection." -ForegroundColor Red
    exit 1
}
Write-Host "  Core packages installed." -ForegroundColor Green

# ── Install Stage 2: ML/CV ───────────────────────────────────
Write-Host ""
Write-Host "Step 5/7: Installing ML/CV packages (this takes ~5 min)..." -ForegroundColor Yellow
pip install -r requirements-ml.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ML install had errors. Trying individual packages..." -ForegroundColor Yellow
    pip install opencv-python
    pip install mediapipe
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install ultralytics scikit-learn
}
Write-Host "  ML packages installed." -ForegroundColor Green

# ── Install Stage 3: AI ──────────────────────────────────────
Write-Host ""
Write-Host "Step 6/7: Installing AI/LLM packages..." -ForegroundColor Yellow
pip install -r requirements-ai.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  AI install had errors. Trying one-by-one..." -ForegroundColor Yellow
    pip install langchain langchain-community langchain-core
    pip install langgraph langchain-openai langchain-ollama
    pip install chromadb sentence-transformers pypdf
    pip install faiss-cpu --extra-index-url https://download.pytorch.org/whl/cpu
}
Write-Host "  AI packages installed." -ForegroundColor Green

# ── Download YOLO model ──────────────────────────────────────
Write-Host ""
Write-Host "Step 7/7: Downloading YOLOv8-Pose model..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "models" | Out-Null
python -c "
from ultralytics import YOLO
import shutil, os
model = YOLO('yolov8n-pose.pt')
if os.path.exists('yolov8n-pose.pt'):
    shutil.move('yolov8n-pose.pt', 'models/yolov8n-pose.pt')
print('Model saved to models/yolov8n-pose.pt')
"

# ── Create .env ──────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "  Created .env from template." -ForegroundColor Green
    Write-Host "  Edit .env if you want to change camera source (default: 0 = webcam)" -ForegroundColor Cyan
}

# ── Initialise memory DB ─────────────────────────────────────
New-Item -ItemType Directory -Force -Path "data" | Out-Null
python -c "
import sys; sys.path.insert(0, '.')
from backend.utils.memory_db import init_memory_db, get_db_info
path = init_memory_db()
info = get_db_info()
print(f'Memory DB ready: {path}')
print(f'Tables: {list(info[\"row_counts\"].keys())}')
" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Memory database initialised." -ForegroundColor Green
}

# ── Done ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To START the system, run:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "    python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload" -ForegroundColor White
Write-Host ""
Write-Host "  Then open in browser:" -ForegroundColor Cyan
Write-Host "    http://localhost:8000/docs   (API)" -ForegroundColor White
Write-Host "    frontend\dashboard\index.html  (Dashboard)" -ForegroundColor White
Write-Host ""
