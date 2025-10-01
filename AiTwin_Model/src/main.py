# main.py

import json
import os
import time
import asyncio
from typing import Any, List, Tuple, Dict, Optional
import uuid

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from debugLogger import DebugLogger
from peerChatManager import PeerChatManager
from mongodb import connect_to_mongo, close_mongo_connection

# --- NEW MODULE IMPORTS ---
from utils import create_text_message, safe_model_dump, load_llm_config, METRICS_PATH, _record_metrics
from rag_system import RAGSystem
from agents import get_agent

# --- INITIALIZATION ---
debugger = DebugLogger(service_name="rag-chatbot")

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./tmp/chromadb")
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "./downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

try:
    model_config = load_llm_config("config/model_config.yaml")
    config_list = model_config.get("config_list", [])
except FileNotFoundError:
    debugger.log("Error: model_config.yaml not found. Please create it.", level="error")
    exit()
except Exception as e:
    debugger.log(f"Error loading model_config.yaml: {e}", level="error")
    exit()

rag_system = RAGSystem(
    config_list=config_list,
    prompt_file="config/prompts.yaml",
    collection_name="generalized_collection",
    history_collection_name="chat_history_collection",
    upload_dir=UPLOAD_DIR,
    doc_extension="*.*",
    batch_size=100,
    debugger=debugger # Pass the debugger instance
)

chat_manager = PeerChatManager()

# --- FASTAPI APP SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="."), name="static")

# --- FASTAPI LIFECYCLE HOOKS ---
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# --- API ENDPOINTS ---

@app.get("/")
async def root():
    return FileResponse("app_agent.html")

@app.post("/ingest")
async def ingest_documents():
    return rag_system.ingest_docs()

@app.get("/metrics")
async def get_metrics():
    try:
        if not os.path.exists(METRICS_PATH):
            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "No metrics data available yet."})
        async with aiofiles.open(METRICS_PATH, "r") as file:
            contents = await file.read()
            return json.loads(contents)
    except FileNotFoundError:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "No metrics data available yet."})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {str(e)}")

@app.get("/connections")
async def get_current_connections():
    return JSONResponse(content={"active_connections": chat_manager.connection_count})

# --- DOCUMENT ENDPOINTS ---

@app.post("/upload_doc/{user_id}/{peer_id}")
async def upload_document(user_id: str, peer_id: str, file: UploadFile = File(...)):
    """Uploads a document and adds it to the peer-to-peer RAG index."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        async with aiofiles.open(file_path, 'wb') as f:
            while chunk := await file.read(8192):
                await f.write(chunk)

        success, message = await rag_system.ingest_peer_doc(file_path, user_id, peer_id)

        if success:
            notification = {
                "type": "document_shared",
                "filename": file.filename,
                "file_url": f"/download_doc/{file.filename}",
                "source": user_id,
                "content": f"A new document '{file.filename}' has been shared and is ready for RAG.",
                "id": str(uuid.uuid4())
            }
            # Persist document message to DB
            await chat_manager.add_message_to_history(user_id,peer_id,message=notification)

            # Notify peers
            await chat_manager.send_message(user_id, notification)
            if chat_manager.is_connected(peer_id):
                await chat_manager.send_message_to_peer(peer_id, notification)

            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": message})
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message)

    except Exception as e:
        debugger.log(f"Error during document upload: {e}", level="error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during document upload: {str(e)}")

@app.get("/download_doc/{filename}")
async def download_document(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

# --- CHAT PROCESSORS ---

async def model_chat_task_processor(user_id: str):
    history_list = await chat_manager.get_model_chat_history(user_id)
    if not isinstance(history_list, list):
        history_list = []

    await chat_manager.send_message(user_id, {"type": "history", "content": history_list})

    while True:
        try:
            data = await chat_manager.get_message(user_id)
            if data is None:
                break

            with debugger.start_span("chat_message"):
                debugger.log("Received message", user_id=user_id, user_input=json.dumps(data))

                # Note: The original code assumes data is already a TextMessage dict, but the listener creates it.
                request_message = create_text_message(
                    content=data.get('content', ''),
                    source=data.get('source', user_id)
                )

                # 1. Persist user message to DB
                message_dict = request_message
                await chat_manager.add_model_message_to_history(user_id, message_dict)

                # 2. Re-load the list to get the full DB message (including its now-set embedding_id)
                history_list = await chat_manager.get_model_chat_history(user_id)

                if not history_list:
                    debugger.log("CRITICAL ERROR: History list is empty after persisting user message.", level="error")
                    await chat_manager.send_message(user_id, {"type": "error", "content": "Failed to retrieve history for processing the request.", "source": "system"})
                    break

                messages_for_agent = [TextMessage(**msg) for msg in history_list if 'source' in msg and 'content' in msg]

                # 3. Ingest all history up to this point to ChromaDB (for retrieval)
                rag_system.ingest_messages_for_rag(messages=history_list, user_id=user_id, peer_id="model")

                agent = await get_agent(history=messages_for_agent, rag_system=rag_system)

                # 4. Generate RAG response
                response, metrics = await agent.on_messages(
                    messages=messages_for_agent,
                    cancellation_token=CancellationToken(),
                    user_id=user_id,
                    peer_id="model" 
                )

                debugger.log("Assistant response", response_content=response.content)

                # 5. Persist assistant's response to DB
                response_dict = response.model_dump()
                await chat_manager.add_model_message_to_history(user_id, response_dict)

                # 6. Ingest response to ChromaDB (only the response, as the user message was handled)
                rag_system.ingest_messages_for_rag(messages=[response_dict], user_id=user_id, peer_id="model")

                assert isinstance(response, TextMessage)

                await chat_manager.send_message(user_id, safe_model_dump(response.model_dump()))

        except Exception as e:
            debugger.log(f"An unexpected error occurred in the processor for user '{user_id}': {e}", level="error", user_id=user_id)
            error_message = {
                "type": "error",
                "content": "An internal server error occurred.",
                "source": "system"
            }
            await chat_manager.send_message(user_id, error_message)
            break
        
        
async def peer_chat_task_processor(user_id: str, peer_id: str):
    """Processes peer-to-peer chat messages (no RAG response, only forwarding and persistence)."""
    try:
        history = await chat_manager.load_history_from_cache(user_id, peer_id)
        await chat_manager.send_message(user_id, {"type": "history", "content": history})

        while True:
            message_data = await chat_manager.get_message(user_id)
            if message_data is None:
                break

            # If it's a document_shared type message, it was handled in the endpoint. Skip here.
            if message_data.get("type") == "document_shared":
                 # Load the most recent message from DB to get full ID/metadata for ingestion
                 latest_message = await chat_manager.get_latest_message(user_id, peer_id)
                 if latest_message and latest_message.get("type") == "document_shared":
                     # Ingest the document notification message (which contains the RAG status)
                     rag_system.ingest_messages_for_rag(messages=[latest_message], user_id=user_id, peer_id=peer_id)
                 
                 # Document forwarding is handled by the upload endpoint.
                 # This processor only needs to handle subsequent text messages.
                 continue 

            # Existing logic for text messages
            message = create_text_message(
                content=message_data["content"],
                source=user_id,
                to=peer_id
            )
            
            # Persist to DB
            await chat_manager.add_message_to_history(user_id, peer_id, message)

            # RAG ingestion (of the message itself)
            rag_system.ingest_messages_for_rag(messages=[message], user_id=user_id, peer_id=peer_id)

            # Forward to peer
            if chat_manager.is_connected(peer_id):
                await chat_manager.send_message_to_peer(peer_id, {
                    "type": "chat",
                    "message_id": message["id"],
                    "source": message["source"],
                    "content": message["content"],
                    "timestamp": message["timestamp"],
                    "requires_ack": True
                })

    except Exception as e:
        debugger.log(f"Error in peer chat processor for {user_id}: {e}", level="error")
        try:
            await chat_manager.send_message(user_id, {
                "type": "error",
                "content": "An internal server error occurred.",
                "source": "system"
            })
        except Exception:
            pass
        finally:
            chat_manager.disconnect(user_id)

async def rag_peer_chat_processor(user_id: str, peer_id: str):
    """Processes messages for RAG-enabled peer chat (peer is offline or designated RAG chat)."""
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
                
            if message_data.get("type") == "document_shared":
                # Document handling is done via the upload endpoint, just skip processing here.
                continue

            # 1. Persist user message to DB
            user_message = create_text_message(
                content=message_data["content"],
                source=user_id,
                to=peer_id
            )
            await chat_manager.add_message_to_history(user_id, peer_id, user_message)

            # 2. Re-load the history to get the latest DB-assigned ID
            history_list = await chat_manager.load_history_from_cache(user_id, peer_id)

            # 3. Ingest all history up to this point to ChromaDB (for retrieval)
            rag_system.ingest_messages_for_rag(messages=history_list, user_id=user_id, peer_id=peer_id)

            # 4. Prepare messages for agent
            messages_for_agent = []
            for msg in history_list:
                if 'source' in msg and 'content' in msg:
                    msg_with_type = msg.copy()
                    msg_with_type['type'] = 'TextMessage'
                    messages_for_agent.append(TextMessage(**msg_with_type))
            
            # 5. Generate RAG response
            agent = await get_agent(history=messages_for_agent, rag_system=rag_system)
            response, _ = await agent.on_messages(
                messages=messages_for_agent,
                cancellation_token=CancellationToken(),
                user_id=user_id,
                peer_id=peer_id 
            )
            
            # 6. Persist RAG response
            rag_message = create_text_message(
                content=response.content,
                source="assistant",
                to=user_id
            )
            await chat_manager.add_message_to_history(user_id, peer_id, rag_message)

            # 7. Send RAG response to user
            await chat_manager.send_message(user_id, {
                "type": "chat",
                "source": "assistant",
                "content": rag_message["content"]
            })

    except Exception as e:
        debugger.log(f"Error in RAG peer chat processor for {user_id}: {e}", level="error")
        await chat_manager.send_message(user_id, {"type": "error", "content": "An internal server error occurred in RAG processor.", "source": "system"})
    finally:
        chat_manager.disconnect(user_id)


# --- WEBSOCKET ENDPOINTS ---

@app.websocket("/ws/model/{user_id}")
async def websocket_model_chat_endpoint(websocket: WebSocket, user_id: str):
    await chat_manager.connect(websocket, user_id)
    await websocket.accept()

    debugger.log(f"WebSocket connected for model chat user: {user_id}", user_id=user_id)

    async def websocket_listener():
        try:
            while True:
                data = await websocket.receive_text()
                if data is not None:
                    message_data = json.loads(data)
                    message = create_text_message(
                        content=message_data.get('content', ''),
                        source=message_data.get('source', user_id),
                        status=message_data.get('status', 'sent')
                    )
                    await chat_manager.message_queues[user_id].put(message)
        except WebSocketDisconnect:
            debugger.log(f"WebSocket listener disconnected for model chat user '{user_id}'.", user_id=user_id)
        except Exception as e:
            debugger.log(f"Error in WebSocket listener for user '{user_id}': {e}", level="error")
        finally:
            if user_id in chat_manager.message_queues:
                await chat_manager.message_queues[user_id].put(None)

    listener_task = asyncio.create_task(websocket_listener())
    processing_task = asyncio.create_task(model_chat_task_processor(user_id))

    try:
        await asyncio.gather(listener_task, processing_task)
    except asyncio.CancelledError:
        debugger.log(f"Tasks for model user '{user_id}' were cancelled.", user_id=user_id)
    except Exception as e:
        debugger.log(f"An unexpected error occurred in model endpoint for user '{user_id}': {e}", level="error")
    finally:
        debugger.log(f"Cleaning up tasks for model user '{user_id}'.", user_id=user_id)
        if 'listener_task' in locals() and not listener_task.done():
            listener_task.cancel()
        if not processing_task.done():
            processing_task.cancel()
        chat_manager.disconnect(user_id)


@app.websocket("/ws/{user_id}/{peer_id}")
async def websocket_peer_chat_endpoint(websocket: WebSocket, user_id: str, peer_id: str):
    await chat_manager.connect(websocket, user_id)
    await websocket.accept()

    debugger.log(f"WebSocket connected for user: {user_id} in a chat with peer: {peer_id}", user_id=user_id, peer_id=peer_id)

    async def websocket_listener():
        try:
            while True:
                data = await websocket.receive_text()
                if data is not None:
                    message_data = json.loads(data)
                    message = create_text_message(
                        content=message_data.get('content', ''),
                        source=message_data.get('source', user_id),
                        to=peer_id
                    )
                    await chat_manager.message_queues[user_id].put(message)
        except WebSocketDisconnect:
            debugger.log(f"WebSocket listener disconnected for user '{user_id}'.", user_id=user_id)
        except Exception as e:
            debugger.log(f"Error in WebSocket listener for user '{user_id}': {e}", level="error")
        finally:
            if user_id in chat_manager.message_queues:
                await chat_manager.message_queues[user_id].put(None)

    listener_task = asyncio.create_task(websocket_listener())
    
    # debugger.log(f"Peer '{peer_id}' is online. Starting normal chat processor.", user_id=user_id)
    # processing_task = asyncio.create_task(peer_chat_task_processor(user_id, peer_id))
    
    # # Check if peer is online, if not, activate RAG mode
    if chat_manager.is_connected(peer_id):
        debugger.log(f"Peer '{peer_id}' is online. Starting normal chat processor.", user_id=user_id)
        processing_task = asyncio.create_task(peer_chat_task_processor(user_id, peer_id))
    else:
        debugger.log(f"Peer '{peer_id}' is offline. Starting RAG chat processor.", user_id=user_id)
        processing_task = asyncio.create_task(rag_peer_chat_processor(user_id, peer_id))


    try:
        await asyncio.gather(listener_task, processing_task)
    except asyncio.CancelledError:
        debugger.log(f"Tasks for user '{user_id}' were cancelled.", user_id=user_id)
    except Exception as e:
        debugger.log(f"An unexpected error occurred in endpoint for user '{user_id}': {e}", level="error")
    finally:
        debugger.log(f"Cleaning up tasks for user '{user_id}'.", user_id=user_id)
        if 'listener_task' in locals() and not listener_task.done():
            listener_task.cancel()
        if not processing_task.done():
            processing_task.cancel()
        chat_manager.disconnect(user_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)