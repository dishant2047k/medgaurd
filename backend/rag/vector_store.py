"""
backend/rag/vector_store.py

Vector store abstraction supporting Chroma (persistent) and FAISS (in-memory).
Handles per-patient document namespacing via metadata filtering.
"""
from __future__ import annotations
from typing import List, Optional
import os

from langchain_core.documents import Document
from backend.utils.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class MedicalVectorStore:

    def __init__(self):
        self._store = None
        self._embeddings = None

    def init(self, embeddings):
        self._embeddings = embeddings

        if settings.vector_db == "chroma":
            self._init_chroma()
        else:
            self._init_faiss()

    def _init_chroma(self):
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            logger.warning("chroma_unavailable_falling_back_to_faiss", error=str(exc))
            self._init_faiss()
            return

        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self._store = Chroma(
            collection_name="medical_knowledge",
            embedding_function=self._embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info("chroma_vector_store_ready", dir=settings.chroma_persist_dir)

    def _init_faiss(self):
        from langchain_community.vectorstores import FAISS
        # Seed with a placeholder document so the store is non-empty
        self._store = FAISS.from_texts(
            ["MedGuard AI medical knowledge base initialised."],
            self._embeddings,
        )
        logger.info("faiss_vector_store_ready")

    def get_retriever(self, patient_id: Optional[str] = None, k: int = 5):
        if patient_id and settings.vector_db == "chroma":
            return self._store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "filter": {"patient_id": patient_id},
                },
            )
        return self._store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k},
        )

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        self._store.add_texts(texts, metadatas=metadatas)

    def add_documents(self, documents: List[Document]):
        self._store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5, patient_id: Optional[str] = None):
        if patient_id and settings.vector_db == "chroma":
            return self._store.similarity_search(
                query, k=k,
                filter={"patient_id": patient_id},
            )
        return self._store.similarity_search(query, k=k)
