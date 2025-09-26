# mongo_crud.py - MongoDB Asynchronous Operations
import motor.motor_asyncio
from typing import List, Optional
from datetime import datetime
import uuid
from mongo_models import User, Conversation, Message, MessageType, ConversationType
from pydantic import ValidationError

# Define Collection Names
USER_COLLECTION = "users"
CONV_COLLECTION = "conversations"
MSG_COLLECTION = "messages"

# --- USER OPERATIONS ---
async def create_user(db: motor.motor_asyncio.AsyncIOMotorDatabase, user_id: str, name: str, email: Optional[str] = None) -> User:
    """Create new user in MongoDB."""
    new_user = User(user_id=user_id)
    await db[USER_COLLECTION].insert_one(new_user.model_dump(by_alias=True))
    return new_user

async def get_user(db: motor.motor_asyncio.AsyncIOMotorDatabase, user_id: str) -> Optional[User]:
    """Get user by ID."""
    user_doc = await db[USER_COLLECTION].find_one({"user_id": user_id})
    return User(**user_doc) if user_doc else None

async def ensure_users_exist(db: motor.motor_asyncio.AsyncIOMotorDatabase, user_ids: List[str]) -> List[User]:
    """Creates users if they don't exist in MongoDB."""
    users = []
    for user_id in user_ids:
        user_doc = await db[USER_COLLECTION].find_one({"user_id": user_id})
        
        if user_doc:
            users.append(User(**user_doc))
        else:
            new_user = await create_user(db, user_id, f"User {user_id}")
            users.append(new_user)
    return users

# --- CONVERSATION OPERATIONS ---
async def create_conversation(db: motor.motor_asyncio.AsyncIOMotorDatabase, participants: List[str], conversation_type: str = "peer_to_peer") -> Conversation:
    """Create new conversation."""
    sorted_participants = sorted(participants)
    
    if conversation_type == "peer_to_peer" and len(participants) == 2:
        conversation_id = f"conv_{'_'.join(sorted_participants)}"
    elif conversation_type == "model_chat" and len(participants) == 2: # <--- NEW: Consistent ID for Model Chat
        # Use a stable, sorted ID like conv_model_userX
        # Note: Ensure "model" is treated consistently (e.g., lowercase)
        conversation_id = f"conv_{'_'.join(sorted_participants)}"
    else:
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
    
    new_conversation = Conversation(
        conversation_id=conversation_id,
        participants=sorted_participants,
        conversation_type=ConversationType(conversation_type)
    )
    await db[CONV_COLLECTION].insert_one(new_conversation.model_dump(by_alias=True))
    return new_conversation

async def get_conversation(db: motor.motor_asyncio.AsyncIOMotorDatabase, conversation_id: str) -> Optional[Conversation]:
    """Get conversation by ID."""
    conv_doc = await db[CONV_COLLECTION].find_one({"conversation_id": conversation_id})
    return Conversation(**conv_doc) if conv_doc else None

async def get_or_create_conversation(db: motor.motor_asyncio.AsyncIOMotorDatabase, participants: List[str], conversation_type: str = "peer_to_peer") -> Conversation:
    """Get existing conversation or create new one."""
    sorted_participants = sorted(participants)
    if (conversation_type == "peer_to_peer" or conversation_type == "model_chat") and len(participants) == 2:
        conversation_id = f"conv_{'_'.join(sorted_participants)}"
        conversation = await get_conversation(db, conversation_id)
        if conversation:
            return conversation
    return await create_conversation(db, participants, conversation_type)

# --- MESSAGE OPERATIONS ---
async def create_message(db: motor.motor_asyncio.AsyncIOMotorDatabase, message_id: str, conversation_id: str, sender_id: str, 
                  text: str, receiver_id: Optional[str] = None, embedding_id: Optional[str] = None,filename: Optional[str] = None, file_url: Optional[str] = None, 
                  message_type: str = "text") -> Message:
    """Create new message, setting embedding_id compulsively."""
    
    # CRITICAL CHANGE: Ensure embedding_id is set to message_id if None
    final_embedding_id = embedding_id if embedding_id is not None else message_id
    
        # Prepare extra fields for file messages
    extra_fields = {}
    if message_type == "document_shared":
        if filename is not None:
            extra_fields["filename"] = filename
        if file_url is not None:
            extra_fields["file_url"] = file_url
            
        # For file messages, we typically use a default text describing the file
        if not text:
            text = f"Shared document: {filename or 'file'}"
        
    new_message = Message(
        id=message_id, 
        conversation_id=conversation_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        text=text,
        embedding_id=final_embedding_id, # Compulsory link for RAG
        message_type=MessageType(message_type),
        timestamp=datetime.utcnow(),
        **extra_fields  # Add file details here
    )
    

    
    # 1. Insert message
    await db[MSG_COLLECTION].insert_one(new_message.model_dump(by_alias=True))
    
    # 2. Update conversation timestamp and status
    await db[CONV_COLLECTION].update_one(
        {"conversation_id": conversation_id},
        {"$set": {
            "last_message_at": new_message.timestamp,
            "updated_at": datetime.utcnow()
        }}
    )
    
    return new_message

async def get_messages_for_conversation(db: motor.motor_asyncio.AsyncIOMotorDatabase, conversation_id: str, limit: int = 50) -> List[Message]:
    """Get messages in a conversation, newest first."""
    cursor = db[MSG_COLLECTION].find(
        {"conversation_id": conversation_id}
    ).sort("timestamp", -1).limit(limit) 
    
    messages = []
    async for doc in cursor:
        try:
            messages.append(Message(**doc))
        except ValidationError as e:
            print(f"Error validating message document: {e}")
            # Skip invalid documents
            continue
            
    return messages

async def get_message_by_embedding_id(db: motor.motor_asyncio.AsyncIOMotorDatabase, embedding_id: str) -> Optional[Message]:
    """Get message by embedding ID (ChromaDB link)."""
    message_doc = await db[MSG_COLLECTION].find_one({"embedding_id": embedding_id})
    return Message(**message_doc) if message_doc else None