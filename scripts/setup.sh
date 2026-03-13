#!/usr/bin/env bash
# ============================================================
# MedGuard AI — Complete Setup & Run Script
# ============================================================
set -e

echo ""
echo "  ███╗   ███╗███████╗██████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ "
echo "  ████╗ ████║██╔════╝██╔══██╗██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗"
echo "  ██╔████╔██║█████╗  ██║  ██║██║  ███╗██║   ██║███████║██████╔╝██║  ██║"
echo "  ██║╚██╔╝██║██╔══╝  ██║  ██║██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║"
echo "  ██║ ╚═╝ ██║███████╗██████╔╝╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝"
echo "  ╚═╝     ╚═╝╚══════╝╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ "
echo ""
echo "  Real-Time Medical Emergency Detection System"
echo ""

# ── Prerequisites check ──────────────────────────────────────
echo "📋 Checking prerequisites..."

command -v docker >/dev/null 2>&1 || { echo "❌ Docker not installed. Visit https://docs.docker.com/get-docker/"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || command -v docker compose >/dev/null 2>&1 || { echo "❌ Docker Compose not installed"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 not installed"; exit 1; }

echo "✅ All prerequisites found"

# ── Environment setup ─────────────────────────────────────────
if [ ! -f ".env" ]; then
  echo "📄 Creating .env from template..."
  cp .env.example .env
  echo "⚠️  Please edit .env with your API keys before production use"
fi

# ── Python virtual environment ────────────────────────────────
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Python environment ready"

# ── Download YOLO model ───────────────────────────────────────
echo "🤖 Downloading YOLOv8 pose model..."
mkdir -p models
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
import shutil, os
if os.path.exists('yolov8n-pose.pt'):
    shutil.move('yolov8n-pose.pt', 'models/yolov8n-pose.pt')
print('✅ YOLOv8-Pose ready')
" 2>/dev/null || echo "⚠️  YOLO model will be downloaded on first run"

# ── Docker services ───────────────────────────────────────────
echo "🐳 Starting Docker services..."
cd deployment
docker-compose up -d postgres redis kafka zookeeper
echo "⏳ Waiting for services to be healthy..."
sleep 10
docker-compose up -d mlflow prometheus grafana ollama
echo "✅ Infrastructure services started"
cd ..

# ── Pull Ollama model ─────────────────────────────────────────
echo "🦙 Pulling Ollama LLM (this may take a few minutes)..."
docker exec $(docker ps -q --filter name=ollama) ollama pull llama3 2>/dev/null || \
  echo "⚠️  Pull Ollama model manually: docker exec <ollama-container> ollama pull llama3"

# ── Database migration ────────────────────────────────────────
echo "🗄️  Initialising database..."
source venv/bin/activate
python3 -c "
import asyncio
from backend.utils.database import init_db
asyncio.run(init_db())
print('✅ Database initialised')
" || echo "⚠️  Database will be initialised on first API start"

# ── Start API ─────────────────────────────────────────────────
echo ""
echo "🚀 Starting MedGuard AI API..."
echo ""
echo "  📊 Dashboard:     http://localhost:3000"
echo "  🔌 API Docs:      http://localhost:8000/docs"
echo "  📈 MLflow:        http://localhost:5000"
echo "  📉 Grafana:       http://localhost:3001  (admin/medguard)"
echo "  📡 Prometheus:    http://localhost:9090"
echo ""
echo "  To test camera detection:"
echo "  python scripts/test_camera.py --source 0"
echo ""
echo "  To train the model:"
echo "  bash scripts/train_model.sh"
echo ""

source venv/bin/activate
uvicorn backend.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info
