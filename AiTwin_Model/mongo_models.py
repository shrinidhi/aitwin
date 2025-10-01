# mongo_models.py - MongoDB Document Definitions
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List, Literal
from datetime import datetime
import enum
import uuid

# --- ENUMS ---
class MessageType(str, enum.Enum):
    TEXT = "text"
    TEXT_MESSAGE = "TextMessage"
    DOCUMENT = "document"
    DOCUMENT_SHARED = "document_shared"
    SYSTEM = "system"
    ERROR = "error"
    IMAGE = "image"
    FILE = "filename"
    FILE_URL = "file_url"

class MessageStatus(str, enum.Enum):
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"

class ConversationType(str, enum.Enum):
    PEER_TO_PEER = "peer_to_peer"
    GROUP = "group"
    MODEL_CHAT = "model_chat"
    SYSTEM = "system"

# --- BASE MODEL ---
class MongoBaseModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id") 

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            uuid.UUID: str
        }

# --- USER MODEL ---
class User(MongoBaseModel):
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# --- MESSAGE MODEL ---
class Message(MongoBaseModel):
    conversation_id: str
    sender_id: str
    receiver_id: Optional[str] = None
    text: str
    # Accept either "text" or "TextMessage" for backward compatibility
    message_type: Literal["text", "TextMessage", "document", "document_shared", 
                          "system", "error", "image", "filename", "file_url"] = "text"
    filename: Optional[str] = None
    file_url: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    embedding_id: str  # MUST be compulsory, set to message_id/id
    status: MessageStatus = MessageStatus.SENT
    extra_info: Optional[Dict[str, Any]] = None

# --- CONVERSATION MODEL ---
class Conversation(MongoBaseModel):
    conversation_id: str
    participants: List[str] = Field(..., min_length=1)
    conversation_type: ConversationType = ConversationType.PEER_TO_PEER
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_message_at: Optional[datetime] = None
    extra_info: Optional[Dict[str, Any]] = None
