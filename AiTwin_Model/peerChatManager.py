import asyncio
import json
import os
from typing import Dict, List, Any
from fastapi import WebSocket
import aiofiles
from collections import deque
import hashlib
import uuid


class PeerChatManager:
    """
    Manages WebSocket connections and per-user message queues for P2P chat.
    It handles two-way communication, file-based history, and concurrency issues.
    Every message is persisted immediately to JSON files in a Windows-safe way.
    """

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.history_cache: Dict[str, List[dict]] = {}
        self.history_locks: Dict[str, asyncio.Lock] = {}
        self.processed_messages: Dict[str, set] = {}
        self.file_lock = asyncio.Lock()  # global lock for file operations

    async def connect(self, websocket: WebSocket, user_id: str):
        """Prepares a new WebSocket connection for use."""
        self.active_connections[user_id] = websocket
        self.message_queues[user_id] = asyncio.Queue()

    def disconnect(self, user_id: str):
        """Disconnects a user by removing their WebSocket connection and message queue."""
        if user_id in self.active_connections:
            if user_id in self.message_queues:
                try:
                    self.message_queues[user_id].put_nowait(None)
                except asyncio.QueueFull:
                    pass
                del self.message_queues[user_id]
            del self.active_connections[user_id]

    def is_connected(self, user_id: str) -> bool:
        """Checks if a user is currently connected."""
        return user_id in self.active_connections

    async def get_message(self, user_id: str):
        """Retrieves the next message from a user's queue."""
        if user_id in self.message_queues:
            return await self.message_queues[user_id].get()
        return None

    async def send_message(self, user_id: str, message: Any):
        """Sends a JSON message to a specific user's WebSocket."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except Exception as e:
                print(f"Failed to send message to {user_id}: {e}")
                self.disconnect(user_id)

    async def send_message_to_peer(self, peer_id: str, message: Any):
        """Sends a JSON message to the peer's WebSocket, if they are connected."""
        await self.send_message(peer_id, message)

    async def save_history_atomically(self, user_id: str, peer_id: str, history: list):
        """Saves the conversation history to a JSON file using an atomic replace to prevent corruption."""
        sorted_ids = tuple(sorted((user_id, peer_id)))
        history_file_path = f"./history_files/p2p_history_{sorted_ids[0]}_{sorted_ids[1]}.json"
        temp_file = history_file_path + '.tmp'

        async with self.file_lock:
            try:
                # Write to a temporary file first
                async with aiofiles.open(temp_file, 'w') as f:
                    await f.write(json.dumps(history, indent=2))
                
                # Atomically replace the original file (Windows-safe)
                os.replace(temp_file, history_file_path)
                
            except Exception as e:
                print(f"Error saving history atomically for {user_id}-{peer_id}: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    async def load_history_from_cache(self, user_id: str, peer_id: str) -> list:
        """Loads the history from the in-memory cache, or from the file if not present."""
        sorted_ids = tuple(sorted((user_id, peer_id)))
        cache_key = f"{sorted_ids[0]}_{sorted_ids[1]}"
        
        if cache_key not in self.history_cache:
            history = await self.load_history_from_json(user_id, peer_id)
            self.history_cache[cache_key] = history
            return history
        
        return self.history_cache[cache_key]

    async def save_history_to_json(self, user_id: str, peer_id: str, history: list):
        """Saves history using the atomic file writing method."""
        await self.save_history_atomically(user_id, peer_id, history)

    async def load_history_from_json(self, user_id: str, peer_id: str) -> list:
        """Loads the conversation history from a JSON file."""
        sorted_ids = tuple(sorted((user_id, peer_id)))
        history_file_path = f"./history_files/p2p_history_{sorted_ids[0]}_{sorted_ids[1]}.json"
        
        if not os.path.exists(history_file_path):
            return []
        
        try:
            async with self.file_lock:
                async with aiofiles.open(history_file_path, "r") as file:
                    content = await file.read()
                    return json.loads(content) if content.strip() else []
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading history for {user_id}-{peer_id}: {e}")
            return []

    async def add_message_to_history(self, user_id: str, peer_id: str, message: dict):
        """
        Adds a message to the in-memory cache and immediately persists it to file.
        Every message is logged without waiting for batching.
        """
        sorted_ids = tuple(sorted((user_id, peer_id)))
        cache_key = f"{sorted_ids[0]}_{sorted_ids[1]}"
        
        if cache_key not in self.history_locks:
            self.history_locks[cache_key] = asyncio.Lock()
        
        async with self.history_locks[cache_key]:
            if cache_key not in self.history_cache:
                self.history_cache[cache_key] = await self.load_history_from_json(user_id, peer_id)
            
            self.history_cache[cache_key].append(message)
            
            # Save immediately for every message
            await self.save_history_atomically(user_id, peer_id, self.history_cache[cache_key])

    async def is_duplicate_message(self, user_id: str, peer_id: str, message_id: str) -> bool:
        """Checks for duplicate message IDs to prevent double processing of a message."""
        key = f"{user_id}_{peer_id}"
        if key not in self.processed_messages:
            self.processed_messages[key] = set()
        
        if message_id in self.processed_messages[key]:
            return True
        
        self.processed_messages[key].add(message_id)
        if len(self.processed_messages[key]) > 1000:
            self.processed_messages[key] = set(list(self.processed_messages[key])[-500:])
        
        return False
        
    @property
    def connection_count(self) -> int:
        """Returns the number of currently active WebSocket connections."""
        return len(self.active_connections)
