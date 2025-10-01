import json
import os
import uuid
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from debugLogger import DebugLogger
from peerChatManager import PeerChatManager 
from rag_system import RAGSystem, get_agent
from utils import create_text_message, safe_model_dump, history_to_textmessages


debugger = DebugLogger(service_name="chat-processor")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./tmp/chromadb") 

chat_manager = PeerChatManager()


# ----------------------------------------------------------------------
# --- AGENT AND CHAT PROCESSORS ---
# ----------------------------------------------------------------------


async def model_chat_task_processor(user_id: str):
    # Initial setup
    history_list = await chat_manager.get_model_chat_history(user_id)
    if not isinstance(history_list, list):
        history_list = []
    await chat_manager.send_message(user_id, {"type": "history", "content": history_list})

    try: # Outer try block for the entire loop
        while True:
            data = await chat_manager.get_message(user_id)
            
            # CRITICAL FIX: Check for None and break immediately.
            if data is None:
                debugger.log(f"No message received for user {user_id}, ending chat loop.", level="info")
                break

            try: # Inner try block for message processing
                with debugger.start_span("chat_message"):
                    debugger.log("Received message", user_id=user_id, user_input=json.dumps(data))

                    # 1. Convert raw dict to Pydantic object
                    # Note: data is the dict created by create_text_message in the listener
                    request = TextMessage(**data) 

                    # 2. Persist user message to DB
                    message_dict = request.model_dump()
                    persisted_user_message = await chat_manager.add_model_message_to_history(user_id, message_dict)

                    # 3. RAG/Generation Logic
                    history_list = await chat_manager.get_model_chat_history(user_id)
                    if not history_list: break # Exit if history fails
                    
                    # Use the new helper function
                    messages_for_agent = history_to_textmessages(history_list)

                    # 4. Ingest history to ChromaDB (for subsequent retrievals)
                    RAGSystem.ingest_messages_for_rag(messages=history_list, user_id=user_id, peer_id="model")

                    agent = await get_agent(history=messages_for_agent, rag_system=RAGSystem)

                    # 5. Generate RAG response
                    response, metrics = await agent.on_messages(
                        messages=messages_for_agent,
                        cancellation_token=CancellationToken(),
                        user_id=user_id,
                        peer_id="model"
                    )

                    # 6. Persist assistant's response to DB
                    response_dict = response.model_dump()
                    persisted_rag_message = await chat_manager.add_model_message_to_history(user_id, response_dict)

                    # 7. Ingest response to ChromaDB (for future history retrieval)
                    RAGSystem.ingest_messages_for_rag(messages=[persisted_rag_message], user_id=user_id, peer_id="model")

                    # 8. Send final message back to client
                    await chat_manager.send_message(user_id, safe_model_dump(response.model_dump()))

            except Exception as e:
                debugger.log(f"An unexpected error occurred in the model chat processor for user '{user_id}': {e}", level="error", user_id=user_id)
                
                # Use the utility to create a clean error message
                error_message = create_text_message(content="An internal server error occurred.", source="system", type="error")
                
                # Send the error message (must use safe_model_dump if the utility creates complex objects)
                await chat_manager.send_message(user_id, safe_model_dump(error_message)) 
                break
            
    except Exception as e: # Catch any errors from outside the inner loop (e.g., initial setup fail)
        debugger.log(f"Fatal error in model_chat_task_processor for {user_id}: {e}", level="error")
        
    finally:
        # Disconnect should always be outside the processing logic try/except.
        chat_manager.disconnect(user_id)

async def peer_chat_task_processor(user_id: str, peer_id: str, chat_manager: PeerChatManager, rag_system: RAGSystem):
    # Initial setup
    history = await chat_manager.load_history_from_cache(user_id, peer_id)
    await chat_manager.send_message(user_id, {"type": "history", "content": history})

    try:
        while True:
            message_data = await chat_manager.get_message(user_id)
            
            # 🛑 THE FIX 1: The check for None must be the first line using message_data
            if message_data is None:
                debugger.log(f"No message received for user {user_id}, ending chat loop.", level="info")
                break
            
            # FIX 2: Added defensive log for incoming message (optional, but safe now)
            debugger.log(f"Processing received message: {message_data}", user_id=user_id)
            
            # Defensive check for expected dictionary type and content
            if not isinstance(message_data, dict):
                 debugger.log(f"Received non-dict message data: {message_data}", level="error")
                 continue
                 
            message_content = message_data.get("content")
            message_type = message_data.get("type")

            # Handle document messages specifically
            if message_type == "document_shared":
                
                filename = message_data.get("filename")
                file_url = message_data.get("file_url")

                if not filename:
                    debugger.log("Document message missing filename.", level="warning")
                    continue
                    
                file_message = create_text_message(
                    content=f"Document shared: {filename}",
                    source=user_id,
                    to=peer_id,
                    type="document_shared",
                    filename=filename,
                    file_url=file_url,
                    id=message_data.get("id", str(uuid.uuid4()))
                )

                persisted_message = await chat_manager.add_message_to_history(user_id, peer_id, file_message)
                
                if chat_manager.is_connected(peer_id):
                    await chat_manager.send_message_to_peer(peer_id, persisted_message)
                continue

            # Existing logic for text messages
            if message_content is None:
                debugger.log("Text message missing 'content'. Skipping.", level="warning")
                continue
                
            message = create_text_message(
                content=message_content,
                source=user_id,
                to=peer_id
            )
            
            persisted_message = await chat_manager.add_message_to_history(user_id, peer_id, message)
            rag_system.ingest_messages_for_rag(messages=[persisted_message], user_id=user_id, peer_id=peer_id)

            if chat_manager.is_connected(peer_id):
                await chat_manager.send_message_to_peer(peer_id, {
                    "type": "chat",
                    "message_id": persisted_message["id"],
                    "source": persisted_message["source"],
                    "content": persisted_message["content"],
                    "timestamp": persisted_message["timestamp"],
                    "requires_ack": True
                })

    except Exception as e:
        debugger.log(f"Error in peer chat processor for {user_id}: {e}", level="error")
        error_message = create_text_message(content="An internal server error occurred.", source="system", type="error")
        try:
            await chat_manager.send_message(user_id, error_message)
        except Exception:
            pass
            
    finally:
        chat_manager.disconnect(user_id)


async def rag_peer_chat_processor(user_id: str, peer_id: str, chat_manager: PeerChatManager, rag_system: RAGSystem):
    # Initial setup
    history_list = await chat_manager.load_history_from_cache(user_id, peer_id)
    if not isinstance(history_list, list):
        history_list = []
            
    await chat_manager.send_message(user_id, {"type": "history", "content": history_list})
    debugger.log(f"Initial history loaded and sent for RAG user '{user_id}'.")
    
    try:
        while True:
            message_data = await chat_manager.get_message(user_id)
            
            if message_data is None:
                break

            # 1. User message creation and persistence to DB
            message_content = message_data.get("content")
            if message_content is None:
                debugger.log("RAG user message missing content.", level="warning")
                continue
                
            user_message = create_text_message(
                content=message_content,
                source=user_id,
                to=peer_id
            )
            await chat_manager.add_message_to_history(user_id, peer_id, user_message)

            # 2. RAG generation preparation
            history_list = await chat_manager.load_history_from_cache(user_id, peer_id)
            rag_system.ingest_messages_for_rag(messages=history_list, user_id=user_id, peer_id=peer_id)
            messages_for_agent = history_to_textmessages(history_list)
            
            agent = await get_agent(history=messages_for_agent, rag_system=rag_system)
            response, _ = await agent.on_messages(
                messages=messages_for_agent,
                cancellation_token=CancellationToken(),
                user_id=user_id,
                peer_id=peer_id
            )
            
            # 4. RAG message creation, persistence, and sending
            rag_message = create_text_message(
                content=response.content,
                source="assistant",
                to=user_id
            )
            await chat_manager.add_message_to_history(user_id, peer_id, rag_message)

            await chat_manager.send_message(user_id, {
                "type": "chat",
                "source": "assistant",
                "content": rag_message["content"]
            })

    except Exception as e:
        debugger.log(f"Error in RAG peer chat processor for {user_id}: {e}", level="error")
        error_message = create_text_message(content="An internal server error occurred in RAG processor.", source="system", type="error")
        await chat_manager.send_message(user_id, error_message)
        
    finally:
        chat_manager.disconnect(user_id)