"""
backend/api/websocket_manager.py

Manages WebSocket connections for real-time event broadcasting.
Supports rooms (one per camera feed) and global broadcast.
"""
from __future__ import annotations
import asyncio
import json
from typing import Dict, List, Set

from fastapi import WebSocket
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    def __init__(self):
        # All active connections
        self._connections: Set[WebSocket] = set()
        # Camera-specific rooms
        self._rooms: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, room: str = "global"):
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
            self._rooms.setdefault(room, set()).add(websocket)
        logger.info("ws_connected", room=room,
                    total=len(self._connections))

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self._connections.discard(websocket)
            for room_members in self._rooms.values():
                room_members.discard(websocket)

    async def broadcast(self, data: dict, room: str = "global"):
        """Send JSON message to all clients in a room."""
        payload = json.dumps(data)
        targets = list(self._rooms.get(room, set()))
        if not targets:
            targets = list(self._connections)

        dead = []
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        # Cleanup dead connections
        for ws in dead:
            await self.disconnect(ws)

    async def broadcast_all(self, data: dict):
        """Broadcast to ALL connected clients regardless of room."""
        payload = json.dumps(data)
        dead = []
        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Singleton
manager = ConnectionManager()
