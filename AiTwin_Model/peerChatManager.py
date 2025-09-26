# peerChatManager.py - Database Operations Abstraction (MongoDB version)
import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import WebSocket

# MongoDB imports
from mongodb import get_mongo_db
import mongo_crud as crud


# Helper function to get an async DB session
async def get_db_session():
    async for db in get_mongo_db():
        if db is None:   # ✅ explicit check
            return None
        return db
    return None



# Convert DB message model to dict
def db_message_to_dict(message: Any) -> dict:
    timestamp = (
        message.timestamp.timestamp()
        if isinstance(message.timestamp, datetime)
        else time.time()
    )

    result_dict = {
        "id": getattr(message, "id", getattr(message, "_id", None)),
        "source": getattr(message, "sender_id", None),
        "content": getattr(message, "text", ""),
        "timestamp": timestamp,
        "status": getattr(message.status, "value", message.status),
        "to": getattr(message, "receiver_id", None),
        "type": getattr(message.message_type, "value", message.message_type),
        "embedding_id": getattr(message, "embedding_id", None),
    }

    # Add optional file-related fields if they exist
    filename = getattr(message, "filename", None)
    if filename is not None:
        result_dict["filename"] = filename
    
    file_url = getattr(message, "file_url", None)
    if file_url is not None:
        result_dict["file_url"] = file_url

    return result_dict


class PeerChatManager:
    """
    Manages WebSocket connections and persists message history in MongoDB.
    """

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.conversation_locks: Dict[str, asyncio.Lock] = {}
        self.processed_messages: Dict[str, set] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        self.active_connections[user_id] = websocket
        self.message_queues[user_id] = asyncio.Queue()

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            if user_id in self.message_queues:
                try:
                    self.message_queues[user_id].put_nowait(None)
                except asyncio.QueueFull:
                    pass
                del self.message_queues[user_id]
            del self.active_connections[user_id]

    def is_connected(self, user_id: str) -> bool:
        return user_id in self.active_connections

    async def get_message(self, user_id: str):
        if user_id in self.message_queues:
            return await self.message_queues[user_id].get()
        return None

    async def send_message(self, user_id: str, message: Any):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except Exception as e:
                print(f"Failed to send message to {user_id}: {e}")
                self.disconnect(user_id)

    async def send_message_to_peer(self, peer_id: str, message: Any):
        await self.send_message(peer_id, message)

    def _get_conv_lock(self, conversation_id: str) -> asyncio.Lock:
        if conversation_id not in self.conversation_locks:
            self.conversation_locks[conversation_id] = asyncio.Lock()
        return self.conversation_locks[conversation_id]

    async def load_history_from_cache(self, user_id: str, peer_id: str) -> List[dict]:
        """Loads history from MongoDB for P2P."""
        db = await get_db_session()
        if db is None:
            return []

        try:
            await crud.ensure_users_exist(db, [user_id, peer_id])
            conversation = await crud.get_or_create_conversation(
                db, [user_id, peer_id], "peer_to_peer"
            )
            messages = await crud.get_messages_for_conversation(
                db, conversation.conversation_id, limit=50
            )
            return [db_message_to_dict(msg) for msg in reversed(messages)]
        except Exception as e:
            print(f"Error loading history for {user_id}-{peer_id}: {e}")
            return []

    async def get_model_chat_history(self, user_id: str) -> List[dict]:
        """Loads model chat history from MongoDB."""
        db = await get_db_session()
        if db is None:
            return []

        try:
            model_id = "model"
            await crud.ensure_users_exist(db, [user_id, model_id])
            conversation = await crud.get_or_create_conversation(
                db, [user_id, model_id], "model_chat"
            )
            messages = await crud.get_messages_for_conversation(
                db, conversation.conversation_id, limit=50
            )
            return [db_message_to_dict(msg) for msg in reversed(messages)]
        except Exception as e:
            print(f"Error loading model chat history for {user_id}: {e}")
            return []

    async def add_message_to_history(
        self, user_id: str, peer_id: str, message: dict, embedding_id: Optional[str] = None
    ):
        """Adds a message to MongoDB."""
        db = await get_db_session()
        if db is None:
            return

        try:
            is_model_chat = peer_id == "model"
            conversation_type = "model_chat" if is_model_chat else "peer_to_peer"

            conversation = await crud.get_or_create_conversation(
                db, [user_id, peer_id], conversation_type
            )
            
            
            conversation_id = conversation.conversation_id
            conv_lock = self._get_conv_lock(conversation_id)

            async with conv_lock:
                message_id = message.get("id", str(uuid.uuid4()))

                receiver_id = (
                    peer_id
                    if not is_model_chat
                    else (user_id if message.get("source") == "assistant" else "model")
                )
                
                message_type = message.get("type", "text")
                filename = message.get("filename")
                file_url = message.get("file_url")

                await crud.create_message(
                    db=db,
                    message_id=message_id,
                    conversation_id=conversation_id,
                    sender_id=message.get("source", user_id),
                    text=message.get("content", ""),
                    receiver_id=receiver_id,
                    embedding_id=embedding_id or message_id,
                    message_type=message_type,
                    filename=filename,
                    file_url=file_url,
                )
        except Exception as e:
            print(f"Error adding message to Mongo: {e}")

    async def add_model_message_to_history(
        self, user_id: str, message: dict, embedding_id: Optional[str] = None
    ):
        """Adds a message to MongoDB for model chat."""
        await self.add_message_to_history(user_id, "model", message, embedding_id)

    async def is_duplicate_message(self, user_id: str, peer_id: str, message_id: str) -> bool:
        """Checks for duplicate message IDs."""
        key = f"{user_id}_{peer_id}"
        if key not in self.processed_messages:
            self.processed_messages[key] = set()

        if message_id in self.processed_messages[key]:
            return True

        self.processed_messages[key].add(message_id)

        if len(self.processed_messages[key]) > 1000:
            # Keep last 500 only
            self.processed_messages[key] = set(list(self.processed_messages[key])[-500:])

        return False

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)
