# MedGuard AI — Complete Implementation Guide

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MEDGUARD AI SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────┐   │
│  │  CCTV/RTSP   │   │  Webcam/USB  │   │  Dashcam  │   │  Mobile  │   │
│  └──────┬───────┘   └──────┬───────┘   └─────┬─────┘   └────┬─────┘   │
│         └──────────────────┴─────────────────┴──────────────┘          │
│                              │ VideoProcessor                           │
│                              ▼ (multi-threaded, async queue)            │
│              ┌───────────────────────────────────────┐                 │
│              │         INFERENCE ENGINE               │                 │
│              │  ┌──────────────┐  ┌───────────────┐  │                 │
│              │  │ YOLOv8-Pose  │  │  MediaPipe    │  │                 │
│              │  │ (primary)    │  │  (fallback)   │  │                 │
│              │  └──────┬───────┘  └──────┬────────┘  │                 │
│              │         └──────────────────┘           │                 │
│              │  ┌──────────────────────────────────┐  │                 │
│              │  │        DETECTOR ENSEMBLE          │  │                 │
│              │  │  FallDetector (physics+ML)        │  │                 │
│              │  │  ActionClassifier (Bi-LSTM+Attn)  │  │                 │
│              │  │  FacialAnalyzer (DeepFace+MP)     │  │                 │
│              │  └──────────────┬───────────────────┘  │                 │
│              └─────────────────┼───────────────────────┘                │
│                                │ MedicalEvent                           │
│              ┌─────────────────▼───────────────────────┐               │
│              │           FastAPI Backend                │               │
│              │  WebSocket (real-time push to frontend)  │               │
│              │  REST API  (CRUD for patients/events)    │               │
│              └─────────┬──────────────────┬────────────┘               │
│                        │                  │                             │
│              ┌─────────▼────────┐  ┌──────▼────────────────────┐       │
│              │  LangGraph       │  │  RAG Chat Assistant        │       │
│              │  Emergency Agent │  │  (LangChain + ChromaDB)    │       │
│              │                  │  │  Patient history + LLM     │       │
│              │  Tools:          │  └───────────────────────────┘        │
│              │  • find_hospital  │                                       │
│              │  • send_sms       │  ┌────────────────────────┐          │
│              │  • call_emergency │  │  PostgreSQL             │          │
│              │  • push_notify    │  │  (patients, events,     │          │
│              │  • log_event      │  │   alerts, chat history) │          │
│              └──────────────────┘  └────────────────────────┘          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    FRONTEND LAYER                                │   │
│  │  React Dashboard  │  Chrome Extension  │  Mobile PWA            │   │
│  │  (WebSocket live  │  (Background WS    │  (Responsive UI)       │   │
│  │   video + alerts) │   + notifications) │                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 OBSERVABILITY STACK                              │   │
│  │  MLflow (model versioning) │ Prometheus │ Grafana │ structlog   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Step-by-Step Setup

### Prerequisites
- Python 3.10+
- Docker + Docker Compose
- 8GB RAM (16GB recommended for GPU)
- Optional: NVIDIA GPU (greatly accelerates inference)

### Step 1: Clone & Configure
```bash
git clone https://github.com/your-org/medguard-ai
cd medguard-ai
cp .env.example .env
# Edit .env with your keys (Twilio, SendGrid etc.)
```

### Step 2: Run Setup Script
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This will:
- Create Python virtualenv and install all dependencies
- Download YOLOv8-Pose model weights (~6MB)
- Start Docker services (PostgreSQL, Redis, Kafka, Ollama, MLflow)
- Pull Llama3 local LLM via Ollama (~4GB first time)
- Start the FastAPI server on :8000

### Step 3: Open Dashboard
Open `frontend/dashboard/index.html` in your browser, or serve it:
```bash
cd frontend/dashboard
python3 -m http.server 3000
# Visit http://localhost:3000
```

### Step 4: Install Chrome Extension
1. Open Chrome → `chrome://extensions`
2. Enable Developer Mode
3. Click "Load unpacked" → select `frontend/extension/`
4. Extension will auto-connect to your API

### Step 5: Test Camera Detection
```bash
source venv/bin/activate
python scripts/test_camera.py --source 0     # webcam
```

---

## 🧠 ML Model Details

### Detection Pipeline

| Component | Technology | Notes |
|-----------|-----------|-------|
| Object Detection + Pose | YOLOv8n-Pose | 6MB, 60fps on CPU |
| Pose Fallback | MediaPipe Pose | Mobile-optimised |
| Fall Detection | Physics rules + Bi-LSTM | No training needed |
| Seizure/Cardiac | FFT + temporal analysis | Runs on pose features |
| Facial Distress | DeepFace + MP FaceMesh | AffectNet backbone |
| Action Classifier | Bi-LSTM + Attention | Trained on NTU/UCF101 |

### Training Your Own Model
```bash
# Download datasets (see datasets/README.md for links)
# Annotate using CVAT: https://cvat.ai

# Extract pose sequences from videos
python datasets/prepare_data.py \
  --input_dir ./datasets/raw \
  --output_dir ./datasets/processed

# Train
bash scripts/train_model.sh

# Monitor training
open http://localhost:5000   # MLflow dashboard
```

---

## 🤖 Emergency Agent Workflow

```
Medical Event Detected
        │
        ▼
LangGraph Agent Invoked
        │
        ├──► find_nearest_hospitals(lat, lon)
        │         └── OSM Nominatim API (free)
        │
        ├──► send_emergency_sms(contacts, message)
        │         └── Twilio SMS API
        │
        ├──► call_emergency_services(number, twiml)
        │         └── Twilio Voice API (if HIGH/CRITICAL)
        │
        ├──► send_push_notification(title, body)
        │         └── Firebase Cloud Messaging
        │
        └──► log_emergency_event(type, severity, actions)
                  └── PostgreSQL + structlog
```

**Local LLM**: Uses Ollama + Llama3 (free, runs locally)  
**API LLM**: Swap to GPT-4o by setting `LLM_PROVIDER=openai` in `.env`

---

## 💬 RAG Chat System

- **Vector DB**: ChromaDB (persistent) or FAISS (in-memory)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (free, local)
- **LLM**: Ollama/Llama3 or OpenAI GPT-4o
- **Memory**: 10-turn sliding window per session

### Ingest Patient Documents
```bash
curl -X POST http://localhost:8000/api/patients/{id}/documents \
  -F "file=@patient_history.pdf"
```

---

## 🌐 Deployment Options

### Option A: Home / Laptop
```bash
./scripts/setup.sh   # Single command setup
```

### Option B: Edge AI (Raspberry Pi / Jetson)
```bash
# Use docker-compose.edge.yml with lighter models
docker-compose -f deployment/docker-compose.edge.yml up
# Uses YOLOv8n (nano) + MediaPipe only, no Kafka
```

### Option C: Cloud (AWS / GCP / Azure)
```bash
# Scale with Kubernetes
kubectl apply -f deployment/k8s/
# Kafka for multi-camera stream ingestion
# Redis for alert deduplication at scale
```

### Option D: Car Dashcam (Offline)
```bash
# Set camera source to dashcam USB device
CAMERA_SOURCES=/dev/video0 ./scripts/setup.sh
# All inference runs locally — no internet needed
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not opening | Check `CAMERA_SOURCES=0` in .env; try `python scripts/test_camera.py` |
| Ollama not responding | `docker exec <ollama> ollama pull llama3` |
| No detections | Lower `DETECTION_CONFIDENCE=0.5` in .env |
| WebSocket disconnecting | Check API is running: `curl http://localhost:8000/health` |
| GPU not used | Install `torch` with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |

---

## 📊 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health + camera status |
| `/ws/events` | WS | Real-time detection events |
| `/api/patients` | GET/POST | Patient CRUD |
| `/api/patients/{id}/documents` | POST | Upload medical docs for RAG |
| `/api/detections` | GET | Detection event history |
| `/api/detections/{id}/resolve` | PATCH | Mark event as resolved |
| `/api/chat` | POST | Medical chat assistant |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Interactive API documentation |

---

## 🔐 Security Notes

- Change `APP_SECRET_KEY` in production
- Use HTTPS + WSS in production (nginx reverse proxy)
- Restrict CORS origins in production
- Store credentials in proper secrets manager (Vault, AWS Secrets Manager)
- Enable PostgreSQL SSL for remote deployments

---

## 📦 Cost: 100% Free & Open Source

| Service | Free Option Used |
|---------|-----------------|
| LLM | Ollama + Llama3 (local) |
| Embeddings | sentence-transformers (local) |
| Vector DB | ChromaDB (local) |
| ML Models | YOLOv8 (AGPL), MediaPipe (Apache 2) |
| Database | PostgreSQL (open source) |
| Message Queue | Apache Kafka (open source) |
| Monitoring | Prometheus + Grafana (open source) |
| SMS/Calls | Twilio (free trial available) |

All components run locally. No cloud API costs required.
