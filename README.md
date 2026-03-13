# 🏥 MedGuard AI — Real-Time Medical Emergency Detection System

> Production-ready, scalable, end-to-end AI system for detecting medical emergencies via live video feeds.

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-org/medguard-ai
cd medguard-ai
cp .env.example .env
docker-compose up --build
```

Access:
- Dashboard: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Grafana: http://localhost:3001

---

## 🧠 System Overview

MedGuard AI is a multi-modal, real-time medical emergency detection platform that:

1. **Detects** abnormal medical events (seizures, falls, cardiac events, unconsciousness) via live video
2. **Alerts** emergency services, hospitals, and family members autonomously via AI Agent
3. **Manages** patient profiles, medical history, and RAG-powered chat assistant
4. **Scales** from a single laptop webcam to a city-wide CCTV network

---

## 📁 Project Structure

```
medguard-ai/
├── backend/
│   ├── api/                    # FastAPI REST + WebSocket server
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── detection.py
│   │   │   ├── patients.py
│   │   │   ├── alerts.py
│   │   │   └── chat.py
│   │   ├── websocket_manager.py
│   │   └── dependencies.py
│   │
│   ├── ml/                     # ML/AI Vision Engine
│   │   ├── pipeline/
│   │   │   ├── video_processor.py
│   │   │   ├── frame_analyzer.py
│   │   │   └── inference_engine.py
│   │   ├── models/
│   │   │   ├── pose_estimator.py
│   │   │   ├── action_classifier.py
│   │   │   ├── fall_detector.py
│   │   │   ├── facial_analyzer.py
│   │   │   └── anomaly_detector.py
│   │   ├── training/
│   │   │   ├── train.py
│   │   │   ├── dataset.py
│   │   │   ├── augmentation.py
│   │   │   └── evaluate.py
│   │   └── utils/
│   │       ├── model_manager.py
│   │       └── metrics.py
│   │
│   ├── agents/                 # LangGraph Emergency Agent
│   │   ├── emergency_agent.py
│   │   ├── tools/
│   │   │   ├── hospital_caller.py
│   │   │   ├── sms_notifier.py
│   │   │   ├── gps_tracker.py
│   │   │   └── snapshot_sender.py
│   │   └── graph.py
│   │
│   ├── rag/                    # Medical RAG Chat System
│   │   ├── chat_assistant.py
│   │   ├── vector_store.py
│   │   ├── document_loader.py
│   │   └── retriever.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       ├── database.py
│       └── redis_client.py
│
├── frontend/
│   ├── dashboard/              # React/Next.js Dashboard
│   │   ├── pages/
│   │   ├── components/
│   │   └── hooks/
│   └── extension/              # Chrome Extension
│       ├── manifest.json
│       ├── background.js
│       └── popup/
│
├── datasets/
│   ├── download_datasets.py
│   ├── prepare_data.py
│   └── README.md
│
├── deployment/
│   ├── docker-compose.yml
│   ├── docker-compose.edge.yml
│   ├── Dockerfile.api
│   ├── Dockerfile.ml
│   └── k8s/
│
├── scripts/
│   ├── setup.sh
│   ├── train_model.sh
│   └── test_camera.py
│
├── .env.example
├── requirements.txt
└── README.md
```
