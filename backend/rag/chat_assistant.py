"""
backend/rag/chat_assistant.py

Medical RAG Chat Assistant using LangChain + ChromaDB / FAISS.
Provides:
  - Patient-specific Q&A from medical history
  - General medical knowledge retrieval
  - Conversation memory
  - Streaming responses
"""
from __future__ import annotations
import asyncio
from typing import AsyncIterator, List, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from backend.rag.vector_store import MedicalVectorStore
from backend.utils.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

MEDICAL_SYSTEM_PROMPT = """You are MedGuard AI — an intelligent medical assistant.
You have access to the patient's medical history, conditions, medications, and past events.

Guidelines:
- Be concise, accurate, and empathetic
- Always remind users that you are NOT a replacement for professional medical advice
- Flag any urgent symptoms immediately
- Reference the patient's specific conditions when relevant
- Use simple language unless the user demonstrates medical expertise

Patient Context: {patient_context}

Retrieved Medical Information:
{context}
"""


class MedicalChatAssistant:
    """
    RAG-based medical chat assistant with per-session memory.
    """

    def __init__(self):
        self.vector_store = MedicalVectorStore()
        self._sessions: dict = {}   # session_id → memory
        self._llm = None
        self._embeddings = None
        self._init()

    def _init(self):
        # Load embeddings
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
        )
        self.vector_store.init(self._embeddings)

        # Load LLM
        if settings.llm_provider == "groq" and settings.groq_api_key:
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=0.3,
                streaming=True,
            )
        elif settings.llm_provider == "openai" and settings.openai_api_key:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.3,
                streaming=True,
            )
        else:
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0.3,
            )

        logger.info("medical_chat_assistant_ready", llm=settings.llm_provider)

    def _get_memory(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
        return self._sessions[session_id]

    async def chat(
        self,
        message: str,
        session_id: str,
        patient_id: Optional[str] = None,
        patient_context: Optional[dict] = None,
    ) -> str:
        """
        Non-streaming chat. Returns full response string.
        """
        memory = self._get_memory(session_id)

        # Retrieve relevant documents
        retriever = self.vector_store.get_retriever(patient_id=patient_id)
        docs = await asyncio.to_thread(retriever.invoke, message)
        context = "\n\n".join(d.page_content for d in docs[:4])

        # Build patient context string
        ctx_str = ""
        if patient_context:
            ctx_str = (
                f"Name: {patient_context.get('name', 'Unknown')}\n"
                f"Age: {patient_context.get('age', 'N/A')}\n"
                f"Conditions: {', '.join(patient_context.get('conditions', []))}\n"
                f"Medications: {', '.join(patient_context.get('medications', []))}\n"
                f"Allergies: {', '.join(patient_context.get('allergies', []))}"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        chain = prompt | self._llm | StrOutputParser()

        chat_history = memory.messages[-20:]  # last 10 turns
        response = await chain.ainvoke({
            "question": message,
            "context": context,
            "patient_context": ctx_str or "No patient context available.",
            "chat_history": chat_history,
        })

        # Save to memory
        memory.add_user_message(message)
        memory.add_ai_message(response)

        return response

    async def stream_chat(
        self,
        message: str,
        session_id: str,
        patient_id: Optional[str] = None,
        patient_context: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        """
        Streaming chat — yields tokens as they arrive.
        """
        memory = self._get_memory(session_id)
        retriever = self.vector_store.get_retriever(patient_id=patient_id)
        docs = await asyncio.to_thread(retriever.invoke, message)
        context = "\n\n".join(d.page_content for d in docs[:4])

        ctx_str = ""
        if patient_context:
            ctx_str = str(patient_context)

        prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        chain = prompt | self._llm | StrOutputParser()
        chat_history = memory.messages[-20:]

        full_response = ""
        async for chunk in chain.astream({
            "question": message,
            "context": context,
            "patient_context": ctx_str or "No patient context available.",
            "chat_history": chat_history,
        }):
            full_response += chunk
            yield chunk

        memory.add_user_message(message)
        memory.add_ai_message(full_response)

    def clear_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    async def ingest_patient_document(
        self,
        patient_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ):
        """Add a medical document to the patient's vector store."""
        await asyncio.to_thread(
            self.vector_store.add_texts,
            texts=[text],
            metadatas=[{"patient_id": patient_id, **(metadata or {})}],
        )
        logger.info("patient_document_ingested", patient_id=patient_id)
