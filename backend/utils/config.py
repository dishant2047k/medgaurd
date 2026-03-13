"""
backend/utils/config.py
Centralised settings loaded from environment / .env file.
"""
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "development"
    app_secret_key: str = "dev-secret"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://medguard:medguard@localhost:5432/medguard"
    redis_url: str = "redis://localhost:6379/0"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_detection_topic: str = "medical_detections"
    kafka_alert_topic: str = "emergency_alerts"

    # ML
    model_dir: str = "./models"
    yolo_pose_model: str = "yolov8n-pose.pt"
    action_model_path: str = "./models/action_classifier.pt"
    anomaly_model_path: str = "./models/anomaly_detector.pt"
    detection_confidence: float = 0.65
    alert_cooldown_seconds: int = 30

    # Camera
    camera_sources: str = "0"

    @property
    def camera_source_list(self) -> List[str]:
        sources = []
        for s in self.camera_sources.split(","):
            s = s.strip()
            sources.append(int(s) if s.isdigit() else s)
        return sources

    # LLM
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_db: str = "chroma"
    chroma_persist_dir: str = "./chroma_db"

    # Alerting
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    sendgrid_api_key: str = ""
    alert_email_from: str = "alerts@medguard.ai"
    firebase_credentials_path: str = ""

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # GPS
    default_latitude: float = 28.6139
    default_longitude: float = 77.2090


@lru_cache
def get_settings() -> Settings:
    return Settings()
