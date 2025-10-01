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
    new_user = User(user_id=user_id)
    await db[USER_COLLECTION].insert_one(new_user.model_dump(by_alias=True))
    return new_user

async def get_user(db: motor.motor_asyncio.AsyncIOMotorDatabase, user_id: str) -> Optional[User]:
    user_doc = await db[USER_COLLECTION].find_one({"user_id": user_id})
    return User(**user_doc) if user_doc else None

async def ensure_users_exist(db: motor.motor_asyncio.AsyncIOMotorDatabase, user_ids: List[str]) -> List[User]:
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
    sorted_participants = sorted(participants)
    
    if conversation_type == "peer_to_peer" and len(participants) == 2:
        conversation_id = f"conv_{'_'.join(sorted_participants)}"
    elif conversation_type == "model_chat" and len(participants) == 2:
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
    conv_doc = await db[CONV_COLLECTION].find_one({"conversation_id": conversation_id})
    return Conversation(**conv_doc) if conv_doc else None

async def get_or_create_conversation(db: motor.motor_asyncio.AsyncIOMotorDatabase, participants: List[str], conversation_type: str = "peer_to_peer") -> Conversation:
    sorted_participants = sorted(participants)
    if (conversation_type in ["peer_to_peer", "model_chat"]) and len(participants) == 2:
        conversation_id = f"conv_{'_'.join(sorted_participants)}"
        conversation = await get_conversation(db, conversation_id)
        if conversation:
            return conversation
    return await create_conversation(db, participants, conversation_type)

# --- MESSAGE OPERATIONS ---
async def create_message(
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    message_id: str,
    conversation_id: str,
    sender_id: str,
    text: str,
    receiver_id: Optional[str] = None,
    embedding_id: Optional[str] = None,
    filename: Optional[str] = None,
    file_url: Optional[str] = None,
    message_type: str = "text"
) -> Message:
    """Create new message, ensuring embedding_id and correct MessageType."""

    # Ensure embedding_id is set
    final_embedding_id = embedding_id or message_id

    # Extra fields for file messages
    extra_fields = {}
    if message_type == "document_shared":
        if filename:
            extra_fields["filename"] = filename
        if file_url:
            extra_fields["file_url"] = file_url
        if not text:
            text = f"Shared document: {filename or 'file'}"

    # Convert "text" to "TextMessage" for literal validation
    if message_type.lower() == "text":
        message_type = "TextMessage"

    # Convert to MessageType enum safely
    if isinstance(message_type, MessageType):
        mt = message_type
    else:
        mt = MessageType(message_type)

    # Create the message
    new_message = Message(
        id=message_id,
        conversation_id=conversation_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        text=text,
        embedding_id=final_embedding_id,
        message_type=mt,
        timestamp=datetime.utcnow(),
        **extra_fields
    )

    # Insert into MongoDB
    await db[MSG_COLLECTION].insert_one(new_message.model_dump(by_alias=True))

    # Update conversation timestamp
    await db[CONV_COLLECTION].update_one(
        {"conversation_id": conversation_id},
        {"$set": {
            "last_message_at": new_message.timestamp,
            "updated_at": datetime.utcnow()
        }}
    )

    return new_message

async def get_messages_for_conversation(db: motor.motor_asyncio.AsyncIOMotorDatabase, conversation_id: str, limit: int = 50) -> List[Message]:
    cursor = db[MSG_COLLECTION].find({"conversation_id": conversation_id}).sort("timestamp", -1).limit(limit)
    messages = []
    async for doc in cursor:
        try:
            messages.append(Message(**doc))
        except ValidationError as e:
            print(f"Error validating message document: {e}")
            continue
    return messages

async def get_message_by_embedding_id(db: motor.motor_asyncio.AsyncIOMotorDatabase, embedding_id: str) -> Optional[Message]:
    message_doc = await db[MSG_COLLECTION].find_one({"embedding_id": embedding_id})
    return Message(**message_doc) if message_doc else None
