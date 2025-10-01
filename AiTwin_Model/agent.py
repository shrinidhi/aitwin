import json
import os
import time
import glob
import chromadb
import hashlib
import fitz
import re
import asyncio
from transformers import AutoTokenizer, AutoModel
import torch
from openai import OpenAI
from typing import Any, List, Tuple, Dict, Optional
import yaml
import uuid

import aiofiles
from autogen import ConversableAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from collections import deque

from debugLogger import DebugLogger
from peerChatManager import PeerChatManager
# CHANGE: Import MongoDB connection functions
from mongodb import connect_to_mongo, close_mongo_connection
# Note: mongo_models.py and mongo_crud.py are imported indirectly via peerChatManager.py

debugger = DebugLogger(service_name="rag-chatbot")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="."), name="static")

# REMOVED: HISTORY_DIR logic
METRICS_PATH = "metrics.json"
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./tmp/chromadb")
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "./downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

chat_manager = PeerChatManager()

# --- FASTAPI LIFECYCLE HOOKS (NEW) ---
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# --- UTILITY FUNCTIONS ---
# ... (create_text_message, safe_model_dump, _record_metrics, load_llm_config remain the same) ...

def create_text_message(content: str, source: str, **kwargs) -> dict:
    """Creates a dictionary for a TextMessage with all required fields."""
    message = {
        "id": str(uuid.uuid4()),
        "source": source,
        "content": content,
        "timestamp": time.time(),
        "status": "sent",
        **kwargs
    }
    return message

def safe_model_dump(obj) -> dict:
    """Safely converts a Pydantic model to a JSON-serializable dictionary."""
    return json.loads(json.dumps(obj, default=str))

async def _record_metrics(data: dict):
    if not os.path.exists(METRICS_PATH):
        async with aiofiles.open(METRICS_PATH, "w") as file:
            await file.write(json.dumps([]))

    async with aiofiles.open(METRICS_PATH, "r+") as file:
        contents = await file.read()
        metrics_list = json.loads(contents)
        metrics_list.append(data)
        await file.seek(0)
        await file.truncate()
        await file.write(json.dumps(metrics_list, indent=2, default=str))

def load_llm_config(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

# --- RAG SYSTEM CLASS ---
class RAGSystem:
    def __init__(self, config_list: List[dict], prompt_file: str, collection_name: str, history_collection_name: str, upload_dir: str, doc_extension: str = "*.{txt,pdf}", model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 50):
        self.collection_name = collection_name
        self.history_collection_name = history_collection_name
        self.upload_dir = upload_dir
        self.doc_extension = doc_extension
        self.batch_size = batch_size
        self.hf_model = model_name
        self.primary_llm_config = next(cfg for cfg in config_list if cfg.get("type") == "primary_response")
        self.rephrase_llm_config = next(cfg for cfg in config_list if cfg.get("type") == "rephrase_response")
        self.prompts = self._load_prompts(prompt_file)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
            self.model = AutoModel.from_pretrained(self.hf_model)
        except Exception as e:
            debugger.log(f"Failed to load HuggingFace models: {e}", level="error")
            raise RuntimeError("Model loading failed.")

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collections = {} # New dictionary to hold peer-specific collections

        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            debugger.log(f"Global collection '{self.collection_name}' found.", level="info")
        except Exception:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            debugger.log(f"Global collection '{self.collection_name}' created.", level="info")

        try:
            self.history_collection = self.chroma_client.get_or_create_collection(name=self.history_collection_name)
            debugger.log(f"History collection '{self.history_collection_name}' found or created.", level="info")
        except Exception as e:
            debugger.log(f"Failed to get/create history collection: {e}", level="error")

    def get_peer_collection(self, user_id: str, peer_id: str):
        # Create a unique, consistent name for the collection based on the two user IDs
        sorted_ids = sorted([user_id, peer_id])
        collection_name = hashlib.md5(f"{sorted_ids[0]}_{sorted_ids[1]}".encode('utf-8')).hexdigest()
        if collection_name not in self.collections:
            try:
                self.collections[collection_name] = self.chroma_client.get_or_create_collection(name=collection_name)
                debugger.log(f"Peer-to-peer collection for {user_id} and {peer_id} found or created.", level="info")
            except Exception as e:
                debugger.log(f"Failed to get/create peer collection: {e}", level="error")
                return None
        return self.collections[collection_name]

    def _load_prompts(self, filepath: str) -> dict:
        try:
            with open(filepath, 'r') as file:
                return yaml.safe_load(file).get('config/prompts', {})
        except FileNotFoundError:
            debugger.log(f"Warning: Prompt file '{filepath}' not found. Using default prompts.", level="warning")
            return {}
        except Exception as e:
            debugger.log(f"Error loading prompt file '{filepath}': {e}. Using default prompts.", level="error")
            return {}

    def embed_texts(self, texts: list[str]):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy().tolist()

    def _chunk_text(self, text, chunk_size=600, overlap=120):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            if end == len(text):
                break
            start = end - overlap
        return [c.strip() for c in chunks if c.strip()]

    def _make_id(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _normalize_roman_numerals(self, text):
        roman_to_arabic = {
            'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
            'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10',
            'XI': '11', 'XII': '12'
        }
        for roman, arabic in roman_to_arabic.items():
            text = re.sub(r'\b' + re.escape(roman) + r'[\.\s]', arabic + '. ', text, flags=re.IGNORECASE)
        return text


    def ingest_docs(self) -> JSONResponse:
        """
        Ingests all documents from the upload directory into the Chroma DB collection.
        """
        files = glob.glob(os.path.join(self.upload_dir, f"**/{self.doc_extension}"), recursive=True)
        if not files:
            message = f"No documents found in '{self.upload_dir}' with extension '{self.doc_extension}'."
            debugger.log(message, level="warning")
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": message})

        if self.collection.count() > 0:
            debugger.log("Clearing existing documents...")
            self.collection.delete(ids=self.collection.get(include=[])['ids'])

        start = time.time()
        total_chunks = 0

        for doc_path in files:
            debugger.log(f"Ingesting knowledge base file: {doc_path}")

            file_extension = os.path.splitext(doc_path)[1].lower()
            full_text = ""
            try:
                if file_extension == ".txt":
                    with open(doc_path, "r", encoding="utf-8") as f:
                        full_text = f.read()
                elif file_extension == ".pdf":
                    doc = fitz.open(doc_path)
                    for page in doc:
                        full_text += page.get_text()
                    doc.close()
                else:
                    debugger.log(f" Skipping unsupported file type: {file_extension} for {doc_path}", level="warning")
                    continue

                full_text = self._normalize_roman_numerals(full_text)
                chunks = self._chunk_text(full_text)

                all_chunks = []
                for chunk in chunks:
                    all_chunks.append({
                        "document": chunk,
                        "id": self._make_id(chunk),
                        "metadata": {"source_file": doc_path}
                    })

                documents_to_add = [c["document"] for c in all_chunks]
                ids_to_add = [c["id"] for c in all_chunks]
                metadatas_to_add = [c["metadata"] for c in all_chunks]

                self.collection.upsert(
                    documents=documents_to_add,
                    ids=ids_to_add,
                    metadatas=metadatas_to_add
                )
                debugger.log(f"Upserted {len(documents_to_add)} documents from {os.path.basename(doc_path)}.", num_docs=len(documents_to_add), source=os.path.basename(doc_path))
                total_chunks += len(documents_to_add)

            except Exception as e:
                debugger.log(f"Failed to process document {doc_path}: {e}", level="error", doc_path=doc_path)

        end = time.time()

        message = f" Ingestion & indexing of {total_chunks} docs completed in {end-start:.2f} seconds."
        debugger.log(message, total_chunks=total_chunks, duration=end-start)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": message})

    async def ingest_peer_doc(self, file_path: str, user_id: str, peer_id: str):
        collection = self.get_peer_collection(user_id, peer_id)
        if collection is None:
            return False, "Failed to get peer collection."

        file_extension = os.path.splitext(file_path)[1].lower()
        full_text = ""
        try:
            if file_extension == ".txt":
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    full_text = await f.read()
            elif file_extension == ".pdf":
                async with aiofiles.open(file_path, "rb") as f:
                    doc = fitz.open(stream=await f.read(), filetype="pdf")
                    for page in doc:
                        full_text += page.get_text()
                    doc.close()
            else:
                debugger.log(f" Skipping unsupported file type for ingestion: {file_extension}", level="warning")
                return False, "Unsupported file type."

            full_text = self._normalize_roman_numerals(full_text)
            chunks = self._chunk_text(full_text)

            documents_to_add = [c for c in chunks]
            ids_to_add = [self._make_id(c) for c in chunks]
            metadatas_to_add = [{"source_file": os.path.basename(file_path), "sender_id": user_id, "receiver_id": peer_id} for _ in chunks]

            collection.upsert(
                documents=documents_to_add,
                ids=ids_to_add,
                metadatas=metadatas_to_add
            )
            debugger.log(f"Upserted {len(documents_to_add)} documents for P2P chat.", user_id=user_id, peer_id=peer_id)
            return True, f"Ingested {len(documents_to_add)} chunks from {os.path.basename(file_path)}."

        except Exception as e:
            debugger.log(f"Failed to process document {file_path}: {e}", level="error", file_path=file_path)
            return False, f"Failed to ingest document: {str(e)}"

    def retrieve_and_answer(self, messages: List[TextMessage], user_id: str, peer_id: Optional[str] = None) -> Tuple[str, dict]:
        start_total = time.time()
        debugger.log(f"--- RAG pipeline executing for user '{user_id}' with conversation history. ---", user_id=user_id)

        latest_query = messages[-1].content
        # Get the latest K messages for immediate context
        # K = 5 is a common setting for short-term memory
        K_RECENT = 5
        recent_messages = messages[-K_RECENT:]
        
        # Format the full history for the rephrase prompt and the recent history for the primary prompt context
        full_history_text = "\n".join([f"{msg.source}: {msg.content}" for msg in messages])
        recent_history_context_texts = [f"{msg.source}: {msg.content}" for msg in recent_messages]

        metrics = {
            "rephrase_latency": 0,
            "rephrase_tokens": 0,
            "retrieval_latency": 0,
            "retrieved_docs": 0,
            "generation_latency": 0,
            "generation_tokens": 0,
            "total_latency": 0
        }

        try:
            client = OpenAI(
                base_url=self.rephrase_llm_config["base_url"],
                api_key=self.rephrase_llm_config["api_key"]
            )

            # --- 1. Rephrase the query ---
            start_rephrase = time.time()
            rephrase_prompt_template = self.prompts.get("rephrase_prompt_v1", "Default rephrase prompt not found.")
            rephrase_prompt = rephrase_prompt_template.format(
                history=full_history_text,
                latest_query=latest_query
            )
            rephrased_response = client.chat.completions.create(
                model=self.rephrase_llm_config["model"],
                messages=[{"role": "user", "content": rephrase_prompt}],
                temperature=self.rephrase_llm_config["temperature"],
                max_tokens=self.rephrase_llm_config["max_tokens"],
            )
            end_rephrase = time.time()
            standalone_query = rephrased_response.choices[0].message.content.strip()
            metrics["rephrase_latency"] = end_rephrase - start_rephrase
            metrics["rephrase_tokens"] = rephrased_response.usage.total_tokens
            debugger.log(f"Rephrased standalone query: '{standalone_query}'", standalone_query=standalone_query)

            # --- 2. Retrieve Context (Semantic Similarity + Recent History) ---
            start_retrieval = time.time()
            query_emb = self.embed_texts([standalone_query])[0]
            
            N_SIMILAR = 3 # Number of semantically similar documents to retrieve
            
            # Documents for the final prompt context
            context_documents = [] 
            
            # --- Retrieval from Peer/KB/History ---
            if peer_id and peer_id != "model":
                # Peer-to-peer chat with RAG mode (using peer-specific document collection)
                peer_collection = self.get_peer_collection(user_id, peer_id)
                if peer_collection:
                    # Query peer-specific document collection for relevant external docs
                    peer_results = peer_collection.query(
                        query_embeddings=[query_emb],
                        n_results=N_SIMILAR,
                        where={"$or": [{"sender_id": user_id}, {"receiver_id": user_id}]}
                    )
                    context_documents.extend(peer_results.get("documents", [[]])[0])
                    debugger.log("Retrieved documents from peer-specific knowledge base.", source="peer_docs")
                    
            elif peer_id == "model":
                # Chat with the RAG Model (user_id and peer_id="model")
                # Retrieve semantically similar chat history from the user's conversation with the model
                history_results = self.history_collection.query(
                    query_embeddings=[query_emb],
                    n_results=N_SIMILAR,
                    # FIX: Explicitly wrap multiple conditions in '$and'
                    where={"$and": [
                    {"user_id": user_id},
                    {"peer_user_id": "model"}
        ]})
                
                similar_history_docs = history_results.get("documents", [[]])[0]
                
                # Filter out history messages that are already in the recent_history_context_texts to avoid duplication
                # A simple filter: if a similar doc's content is the same as any recent message's content
                similar_history_docs_filtered = []
                recent_contents = [msg.content for msg in recent_messages]
                for doc in similar_history_docs:
                    if doc not in recent_contents:
                        similar_history_docs_filtered.append(f"PAST CHAT: {doc}")
                
                context_documents.extend(similar_history_docs_filtered)
                
                debugger.log(f"Retrieved {len(similar_history_docs_filtered)} semantically similar chat history documents.", source="chat_history")
                
                # Fallback: If not enough context, also query the global KB
                if len(context_documents) < N_SIMILAR:
                    kb_results = self.collection.query(query_embeddings=[query_emb], n_results=N_SIMILAR)
                    context_documents.extend(kb_results.get("documents", [[]])[0])
                    debugger.log("Augmented context with global knowledge base documents.", source="knowledge_base")
            
            # --- Combine Contexts ---
            # 1. Start with recent chat history
            final_context_texts = recent_history_context_texts
            
            # 2. Add semantically retrieved documents/history
            final_context_texts.extend(context_documents)
            
            # Join all context pieces for the LLM prompt
            context = "\n".join(final_context_texts)

            end_retrieval = time.time()

            metrics["retrieval_latency"] = end_retrieval - start_retrieval
            metrics["retrieved_docs"] = len(context_documents) # Count only the RAG-retrieved docs

            if not context.strip():
                metrics["total_latency"] = time.time() - start_total
                return "Sorry, I could not find any relevant information on that topic.", metrics
            
            


            # --- 3. Generate Answer ---
            start_generation = time.time()
            primary_prompt_template = self.prompts.get("primary_prompt_v1","Default rephrase prompt not found.")
            prompt = primary_prompt_template.format(
                context=context,
                history=full_history_text, # Pass full history for conversational tone
                latest_query=latest_query
            )

            debugger.log("Primary LLM prompt generated")

            response = client.chat.completions.create(
                model=self.primary_llm_config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.primary_llm_config["temperature"],
                max_tokens=self.primary_llm_config["max_tokens"],
            )
            end_generation = time.time()

            answer = response.choices[0].message.content
            metrics["generation_latency"] = end_generation - start_generation
            metrics["generation_tokens"] = response.usage.total_tokens
            metrics["total_latency"] = end_generation - start_total

            debugger.log("RAG system generated a contextual answer.", answer=answer)
            return answer, metrics
        except Exception as e:
            error_message = f"An error occurred during the RAG process: {str(e)}"
            debugger.log(f"An unexpected error occurred in the processor: {e}", level="error", error_message=error_message)
            metrics["total_latency"] = time.time() - start_total
            return "Sorry, an internal error occurred while processing your request.", metrics

    def ingest_messages_for_rag(self, messages: List[dict], user_id: str, peer_id: str):
        """
        Ingests messages into the ChromaDB history collection, using the DB-stored
        embedding_id (which is guaranteed to exist) as the ChromaDB document ID.
        """
        documents_to_add = []
        ids_to_add = []
        metadatas_to_add = []

        for message in messages:
            content = message.get('content')
            source = message.get('source')
            # CRITICAL: Use the 'embedding_id' which is guaranteed to be set by the DB layer
            message_id = message.get('id')
            debugger.log(message_id)
            
            if not content or not message_id:
                # Skip messages without content or a valid embedding_id
                continue

            metadata = {
                "source": source,
                "timestamp": message.get('timestamp', time.time()),
                "user_id": user_id,
                "peer_user_id": peer_id,
                "message_id": message_id
            }

            documents_to_add.append(content)
            ids_to_add.append(message_id)
            metadatas_to_add.append(metadata)

        if documents_to_add:
            try:
                self.history_collection.upsert(
                    documents=documents_to_add,
                    ids=ids_to_add,
                    metadatas=metadatas_to_add
                )
                debugger.log(f"Ingested {len(documents_to_add)} messages into history collection for conversation between '{user_id}' and '{peer_id}'.")
            except Exception as e:
                debugger.log(f"Failed to ingest messages into history collection for '{user_id}' and '{peer_id}': {e}", level="error")

# --- AGENT AND CHAT PROCESSORS ---

class RAGAssistantAgent(ConversableAgent):
    def __init__(self, rag_system: RAGSystem, **kwargs):
        super().__init__(**kwargs)
        self.rag_system = rag_system

    async def on_messages(self, messages: List[TextMessage], cancellation_token: CancellationToken, user_id: str, peer_id: Optional[str] = None) -> Tuple[TextMessage, dict]:
        rag_response, metrics = self.rag_system.retrieve_and_answer(messages, user_id, peer_id)
        return TextMessage(content=rag_response, source=self.name), metrics

async def get_agent(history: list[dict[str, Any]], rag_system: RAGSystem) -> RAGAssistantAgent:
    agent = RAGAssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        rag_system=rag_system
    )
    # Safely convert history dictionaries to TextMessage objects
    agent._history = [TextMessage(**msg) for msg in history if 'source' in msg and 'content' in msg]
    return agent

# --- CONFIGURATION AND INSTANTIATION (MOVED TO THE TOP) ---
try:
    model_config = load_llm_config("model_config.yaml")
    config_list = model_config.get("config_list", [])
except FileNotFoundError:
    debugger.log("Error: model.yaml not found. Please create it.", level="error")
    exit()
except Exception as e:
    debugger.log(f"Error loading model.yaml: {e}", level="error")
    exit()

rag_system = RAGSystem(
    config_list=config_list,
    prompt_file="prompts.yaml",
    collection_name="generalized_collection",
    history_collection_name="chat_history_collection",
    upload_dir=UPLOAD_DIR,
    doc_extension="*.*",
    batch_size=100
)

# --- FASTAPI ENDPOINTS ---

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

# REMOVED: Deleted the old file-based get_history_for_user function

async def model_chat_task_processor(user_id: str):
    # CHANGE: Use chat_manager.get_model_chat_history to load history from DB
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

                request = TextMessage(**data)

                # 1. Persist user message to DB (embedding_id is set inside crud.py)
                message_dict = request.model_dump()
                await chat_manager.add_model_message_to_history(user_id, message_dict)

                # 2. Re-load the list to get the full DB message (including its now-set embedding_id)
                history_list = await chat_manager.get_model_chat_history(user_id)

                # --- VITAL FIX: Prevent 'list index out of range' error in RAGSystem.retrieve_and_answer ---
                if not history_list:
                    debugger.log("CRITICAL ERROR: History list is empty after persisting user message. Cannot proceed with RAG.", level="error")
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
                    peer_id="model" # Pass peer_id="model"
                )

                debugger.log("Assistant response", response_content=response.content)

                # 5. Persist assistant's response to DB
                response_dict = response.model_dump()
                await chat_manager.add_model_message_to_history(user_id, response_dict)

                # 6. Ingest response to ChromaDB (Only the response is needed, as the message above was already handled)
                # Note: We use the already-prepared response_dict which will have its DB-assigned ID set after persistence.
                # Re-loading the history list again is redundant if we trust the persistence call.
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
    try:
        # CHANGE: load_history_from_cache reads from DB
        history = await chat_manager.load_history_from_cache(user_id, peer_id)
        await chat_manager.send_message(user_id, {"type": "history", "content": history})

        while True:
            message_data = await chat_manager.get_message(user_id)
            if message_data is None:
                break

            # Handle document messages specifically
            if message_data.get("type") == "document_shared":
                filename = message_data.get("filename")
                file_url = message_data.get("file_url")

                file_message = {
                    "type": "document_shared",
                    "filename": filename,
                    "file_url": file_url,
                    "source": user_id,
                    "timestamp": time.time(),
                    "id": str(uuid.uuid4()) # Ensure document message has an ID
                }

                success, doc_message = await rag_system.ingest_peer_doc(os.path.join(UPLOAD_DIR, filename), user_id, peer_id)

                # CHANGE: add_message_to_history persists to DB
                await chat_manager.add_message_to_history(user_id, peer_id, file_message)

                if chat_manager.is_connected(peer_id):
                    await chat_manager.send_message_to_peer(peer_id, file_message)
                continue

            # Existing logic for text messages
            message = create_text_message(
                content=message_data["content"],
                source=user_id,
                to=peer_id
            )
            
            debugger.log(message)

            # CHANGE: add_message_to_history persists to DB
            await chat_manager.add_message_to_history(user_id, peer_id, message)

            # RAG ingestion uses the stored message details
            rag_system.ingest_messages_for_rag(messages=[message], user_id=user_id, peer_id=peer_id)

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
    
            # FIX START: Load and send history immediately upon connection to User 1
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

            # ... (user_message creation and persistence to DB) ...
            user_message = create_text_message(
                content=message_data["content"],
                source=user_id,
                to=peer_id
            )
            await chat_manager.add_message_to_history(user_id, peer_id, user_message)

            history_list = await chat_manager.load_history_from_cache(user_id, peer_id)
            print(history_list)

            rag_system.ingest_messages_for_rag(messages=history_list, user_id=user_id, peer_id=peer_id)

            # FIX: Explicitly add 'type': 'TextMessage' to each message dict before Pydantic parsing
            messages_for_agent = []
            for msg in history_list:
                if 'source' in msg and 'content' in msg:
                    # Create a copy and insert the required 'type' field
                    msg_with_type = msg.copy()
                    msg_with_type['type'] = 'TextMessage'
                    messages_for_agent.append(TextMessage(**msg_with_type))
            
            # --- OLD LINE THAT WAS FAILING ---
            # messages_for_agent = [TextMessage(**msg) for msg in history_list if 'source' in msg and 'content' in msg]

            agent = await get_agent(history=messages_for_agent, rag_system=rag_system)
            response, _ = await agent.on_messages(
                messages=messages_for_agent,
                cancellation_token=CancellationToken(),
                user_id=user_id,
                peer_id=peer_id # Pass peer_id here
            )
            
            # ... (rag_message creation, persistence, and sending) ...
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
        await chat_manager.send_message(user_id, {"type": "error", "content": "An internal server error occurred in RAG processor.", "source": "system"})
    finally:
        chat_manager.disconnect(user_id)

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
            # Send a WebSocket message to both peers to notify them of the new document
            notification = {
                "type": "document_shared",
                "filename": file.filename,
                "file_url": f"/download_doc/{file.filename}",
                "source": user_id,
                "content": f"A new document '{file.filename}' has been shared and is ready for RAG.",
                "id": str(uuid.uuid4())
            }
            await chat_manager.add_message_to_history(user_id,peer_id,message=notification)

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
            await chat_manager.message_queues[user_id].put(None)

    listener_task = asyncio.create_task(websocket_listener())
    
    debugger.log(f"Peer '{peer_id}' is online. Starting normal chat processor.", user_id=user_id)
    processing_task = asyncio.create_task(peer_chat_task_processor(user_id, peer_id))

    # Check if peer is online, if not, activate RAG mode
    # if chat_manager.is_connected(peer_id):
    #   debugger.log(f"Peer '{peer_id}' is online. Starting normal chat processor.", user_id=user_id)
    #   processing_task = asyncio.create_task(peer_chat_task_processor(user_id, peer_id))
    # else:
    #     debugger.log(f"Peer '{peer_id}' is offline. Starting RAG chat processor.", user_id=user_id)
    #     processing_task = asyncio.create_task(rag_peer_chat_processor(user_id, peer_id))

    try:
        await asyncio.gather(listener_task, processing_task)
    except asyncio.CancelledError:
        debugger.log(f"Tasks for user '{user_id}' were cancelled.", user_id=user_id)
    except Exception as e:
        debugger.log(f"An unexpected error occurred in endpoint for user '{user_id}': {e}", level="error")
    finally:
        debugger.log(f"Cleaning up tasks for user '{user_id}'.", user_id=user_id)
        listener_task.cancel()
        processing_task.cancel()
        chat_manager.disconnect(user_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)