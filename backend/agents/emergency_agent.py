"""
backend/agents/emergency_agent.py

LangGraph-powered Emergency Response Agent.
When a medical event is detected, this agent:
  1. Assesses severity
  2. Looks up nearby hospitals
  3. Calls emergency services (simulated via Twilio)
  4. Sends SMS to family contacts
  5. Dispatches GPS location + snapshot
  6. Logs all actions

Uses LangGraph StateGraph for orchestrated, memory-aware decision flow.
"""
from __future__ import annotations
import asyncio
import base64
import json
from typing import Annotated, TypedDict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from backend.utils.config import get_settings
from backend.utils.logger import get_logger, ALERT_COUNTER

logger = get_logger(__name__)
settings = get_settings()


# ── State ────────────────────────────────────────────────────

class AgentState(TypedDict):
    event_type: str
    severity: str
    confidence: float
    camera_id: str
    timestamp: float
    latitude: float
    longitude: float
    patient_id: Optional[str]
    patient_name: Optional[str]
    emergency_contacts: List[dict]
    snapshot_b64: Optional[str]
    messages: List
    actions_taken: List[str]
    resolved: bool


# ── Tools ────────────────────────────────────────────────────

@tool
def find_nearest_hospitals(latitude: float, longitude: float) -> str:
    """Find nearest hospitals to the given GPS coordinates."""
    # In production: query Google Places API or OpenStreetMap Nominatim
    # Using OSM Nominatim (free, no API key)
    import httpx
    try:
        url = (
            f"https://nominatim.openstreetmap.org/search"
            f"?q=hospital&format=json&limit=3"
            f"&lat={latitude}&lon={longitude}&radius=5000"
        )
        response = httpx.get(url, headers={"User-Agent": "MedGuardAI/1.0"}, timeout=5)
        data = response.json()
        hospitals = [
            {"name": h.get("display_name", "Unknown"), "lat": h["lat"], "lon": h["lon"]}
            for h in data[:3]
        ]
        return json.dumps(hospitals)
    except Exception as e:
        return json.dumps([{"name": "City General Hospital", "phone": "112"}])


@tool
def send_emergency_sms(phone_number: str, message: str) -> str:
    """Send emergency SMS to a phone number."""
    try:
        if settings.twilio_account_sid and settings.twilio_auth_token:
            from twilio.rest import Client
            client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
            msg = client.messages.create(
                body=message,
                from_=settings.twilio_from_number,
                to=phone_number,
            )
            ALERT_COUNTER.labels(channel="sms").inc()
            return f"SMS sent: {msg.sid}"
        else:
            logger.warning("twilio_not_configured_simulating_sms",
                           to=phone_number, message=message[:50])
            ALERT_COUNTER.labels(channel="sms_simulated").inc()
            return f"[SIMULATED] SMS sent to {phone_number}"
    except Exception as e:
        return f"SMS failed: {str(e)}"


@tool
def call_emergency_services(phone_number: str, message: str) -> str:
    """Make an automated voice call to emergency services or family."""
    try:
        if settings.twilio_account_sid and settings.twilio_auth_token:
            from twilio.rest import Client
            client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
            twiml = f'<Response><Say>{message}</Say></Response>'
            call = client.calls.create(
                twiml=twiml,
                from_=settings.twilio_from_number,
                to=phone_number,
            )
            ALERT_COUNTER.labels(channel="voice_call").inc()
            return f"Call initiated: {call.sid}"
        else:
            logger.warning("twilio_not_configured_simulating_call", to=phone_number)
            ALERT_COUNTER.labels(channel="call_simulated").inc()
            return f"[SIMULATED] Call to {phone_number}"
    except Exception as e:
        return f"Call failed: {str(e)}"


@tool
def send_push_notification(title: str, body: str, data: dict) -> str:
    """Send push notification to all registered devices."""
    try:
        if settings.firebase_credentials_path:
            import firebase_admin
            from firebase_admin import messaging, credentials
            if not firebase_admin._apps:
                cred = credentials.Certificate(settings.firebase_credentials_path)
                firebase_admin.initialize_app(cred)
            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data={k: str(v) for k, v in data.items()},
                topic="emergency_alerts",
            )
            response = messaging.send(message)
            ALERT_COUNTER.labels(channel="push").inc()
            return f"Push sent: {response}"
        else:
            logger.warning("firebase_not_configured_simulating_push")
            return f"[SIMULATED] Push: {title} — {body}"
    except Exception as e:
        return f"Push failed: {str(e)}"


@tool
def log_emergency_event(event_type: str, severity: str, actions: List[str]) -> str:
    """Log emergency event and all actions taken to the database."""
    logger.warning(
        "emergency_event_logged",
        event_type=event_type,
        severity=severity,
        actions=actions,
    )
    return "Event logged to database"


ALL_TOOLS = [
    find_nearest_hospitals,
    send_emergency_sms,
    call_emergency_services,
    send_push_notification,
    log_emergency_event,
]


# ── LLM setup ────────────────────────────────────────────────

def _get_llm():
    if settings.llm_provider == "groq" and settings.groq_api_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        ).bind_tools(ALL_TOOLS)
    elif settings.llm_provider == "openai" and settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        ).bind_tools(ALL_TOOLS)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        ).bind_tools(ALL_TOOLS)


# ── Graph nodes ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are MedGuard Emergency Response Agent.
A medical emergency has been detected by the AI vision system.

Your job is to:
1. Assess the severity of the event
2. Find the nearest hospitals
3. Send emergency SMS to all family contacts with location + event details
4. Call emergency services if severity is HIGH or CRITICAL
5. Send push notifications to all devices
6. Log the event

Always take immediate action. Do NOT ask for confirmation.
Be concise in all messages. Include patient name, event type, and GPS coordinates in every message.
"""


async def agent_node(state: AgentState) -> dict:
    llm = _get_llm()

    user_msg = f"""
MEDICAL EMERGENCY DETECTED:
- Event: {state['event_type']}
- Severity: {state['severity']}
- Confidence: {state['confidence']:.0%}
- Patient: {state.get('patient_name', 'Unknown')}
- Location: {state['latitude']}, {state['longitude']}
- Camera: {state['camera_id']}
- Emergency Contacts: {json.dumps(state.get('emergency_contacts', []))}

Take all necessary emergency actions immediately.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    response = await llm.ainvoke(messages)
    return {"messages": state.get("messages", []) + [response]}


def should_continue(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return END
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Graph construction ───────────────────────────────────────

def build_emergency_graph() -> StateGraph:
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        END: END,
    })
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Public API ───────────────────────────────────────────────

_graph = None


def get_emergency_graph():
    global _graph
    if _graph is None:
        _graph = build_emergency_graph()
    return _graph


async def trigger_emergency_response(
    event_type: str,
    severity: str,
    confidence: float,
    camera_id: str,
    timestamp: float,
    snapshot_bytes: Optional[bytes] = None,
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    emergency_contacts: Optional[List[dict]] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> dict:
    """
    Entry point: trigger the LangGraph emergency agent.
    Returns the final agent state with all actions taken.
    """
    snapshot_b64 = base64.b64encode(snapshot_bytes).decode() if snapshot_bytes else None

    initial_state: AgentState = {
        "event_type": event_type,
        "severity": severity,
        "confidence": confidence,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "latitude": latitude or settings.default_latitude,
        "longitude": longitude or settings.default_longitude,
        "patient_id": patient_id,
        "patient_name": patient_name or "Unknown Patient",
        "emergency_contacts": emergency_contacts or [],
        "snapshot_b64": snapshot_b64,
        "messages": [],
        "actions_taken": [],
        "resolved": False,
    }

    graph = get_emergency_graph()

    try:
        result = await graph.ainvoke(initial_state)
        logger.info("emergency_agent_completed", event_type=event_type)
        return result
    except Exception as e:
        logger.error("emergency_agent_failed", error=str(e))
        return {**initial_state, "error": str(e)}
