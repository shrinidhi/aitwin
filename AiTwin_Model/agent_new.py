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
from typing import Any, List, Tuple, Dict, Optional
import yaml
import uuid
from functools import partial

import aiofiles
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent 

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from collections import deque

# Assuming these modules exist and are imported correctly
from debugLogger import DebugLogger
from peerChatManager import PeerChatManager
from mongodb import connect_to_mongo, close_mongo_connection

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

METRICS_PATH = "metrics.json"
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./tmp/chromadb")
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "./downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

chat_manager = PeerChatManager()

# --- FASTAPI LIFECYCLE HOOKS (UNCHANGED) ---
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()
    
@app.get("/")
async def root():
    return FileResponse("app_agent.html")

@app.post("/ingest")
async def ingest_documents():
    return rag_system.ingest_docs()     

# --- UTILITY FUNCTIONS (UNCHANGED) ---
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

# --- RAG SYSTEM CLASS (UNCHANGED) ---
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
        self.collections = {}

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

    def ingest_messages_for_rag(self, messages: List[dict], user_id: str, peer_id: str):
        documents_to_add = []
        ids_to_add = []
        metadatas_to_add = []

        for message in messages:
            content = message.get('content')
            source = message.get('source')
            message_id = message.get('id')
            debugger.log(message_id)
            
            if not content or not message_id:
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

    def rag_function_call(self, query: str, user_id: str, peer_id: Optional[str] = None, history_text: str = "") -> str:
        """
        Performs the RAG process (retrieve) as a synchronous function call.
        Returns the retrieved context as a string for the LLM to use.
        """
        start_total = time.time()
        debugger.log(f"--- RAG function call executing for user '{user_id}' with query: '{query}'. ---", user_id=user_id)
        
        latest_query = query
        full_history_text = history_text

        metrics = {
            "retrieval_latency": 0,
            "retrieved_docs": 0,
            "total_latency": 0
        }
        
        try:
            # 1. Retrieve Context (Semantic Similarity)
            start_retrieval = time.time()
            # NOTE: For production scaling, embed_texts should be wrapped in asyncio.to_thread()
            query_emb = self.embed_texts([latest_query])[0]
            
            N_SIMILAR = 3 
            context_documents = [] 
            
            # --- Retrieval from Peer/KB/History ---
            if peer_id and peer_id != "model":
                # A. Retrieve from Peer-Specific Knowledge Base
                peer_collection = self.get_peer_collection(user_id, peer_id)
                if peer_collection:
                    peer_results = peer_collection.query(
                        query_embeddings=[query_emb],
                        n_results=N_SIMILAR,
                        where={"$or": [{"sender_id": user_id}, {"receiver_id": user_id}]}
                    )
                    context_documents.extend([f"PEER DOCS: {doc}" for doc in peer_results.get("documents", [[]])[0]])
                    debugger.log("Retrieved documents from peer-specific knowledge base.", source="peer_docs")
                
                # B. Retrieve from Global Chat History (P2P conversation history)
                history_results = self.history_collection.query(
                    query_embeddings=[query_emb],
                    n_results=N_SIMILAR,
                    # Retrieve history chunks that involve EITHER user_id OR peer_id
                    where={"$or": [
                        {"$and": [{"user_id": user_id}, {"peer_user_id": peer_id}]},
                        {"$and": [{"user_id": peer_id}, {"peer_user_id": user_id}]}
                    ]}
                )
                context_documents.extend([f"PAST CHAT: {doc}" for doc in history_results.get("documents", [[]])[0]])
                debugger.log(f"Retrieved {len(context_documents)} semantically similar P2P chat history documents.", source="chat_history")
                                            
            elif peer_id == "model":
                # --- Standard Model Retrieval (Unchanged) ---
                history_results = self.history_collection.query(
                    query_embeddings=[query_emb],
                    n_results=N_SIMILAR,
                    where={"$and": [
                        {"user_id": user_id},
                        {"peer_user_id": "model"}
                    ]}
                )
                context_documents.extend([f"PAST CHAT: {doc}" for doc in history_results.get("documents", [[]])[0]])
                
                # Fallback: Also query the global KB
                kb_results = self.collection.query(query_embeddings=[query_emb], n_results=N_SIMILAR)
                context_documents.extend([f"KNOWLEDGE BASE: {doc}" for doc in kb_results.get("documents", [[]])[0]])
                debugger.log("Augmented context with global knowledge base documents.", source="knowledge_base")
            
            # Combine Contexts
            final_context_texts = [f"USER QUERY: {latest_query}"]
            final_context_texts.extend(context_documents)
            if history_text:
                final_context_texts.append(f"CONVERSATION HISTORY (Full Text): {history_text}")

            context = "\n---\n".join(final_context_texts)
            
            end_retrieval = time.time()
            metrics["retrieval_latency"] = end_retrieval - start_retrieval
            metrics["retrieved_docs"] = len(context_documents) 
            metrics["total_latency"] = time.time() - start_total
            
            debugger.log(f"RAG function call finished. Returning context of size {len(context)}.", metrics=metrics)
            
            if not context.strip():
                return "No relevant information found in the knowledge bases."
            
            return context

        except Exception as e:
            error_message = f"An error occurred during the RAG context retrieval: {str(e)}"
            debugger.log(f"An unexpected error occurred in the RAG function: {e}", level="error", error_message=error_message)
            metrics["total_latency"] = time.time() - start_total
            return "An internal error occurred while fetching RAG context."


# --- AGENT AND CHAT PROCESSORS (UNCHANGED) ---

class RAGAssistantAgent(AssistantAgent):
    def __init__(self, rag_system: RAGSystem, user_id: str, peer_id: str, is_p2p_proxy: bool = False, **kwargs):
        
        self.rag_system = rag_system
        self.user_id = user_id
        self.peer_id = peer_id
        self.is_p2p_proxy = is_p2p_proxy # New flag to adjust system prompt
        
        # 1. Define the base system message and tool instruction
        base_system_message = kwargs.pop("system_message", "You are a helpful assistant.")
        
        if self.is_p2p_proxy:
            # CUSTOM PROMPT for P2P Proxy Mode - STRONGLY FORCED TOOL CALL
            tool_instruction = f"""
            Your **FIRST and ONLY** response MUST be a call to the `retrieve_rag_context` tool. 
            You MUST NOT output any text, explanation, or reasoning before the tool call.

            You are acting as an **Auto-Responder for the user '{peer_id}'**. 
            User '{user_id}' is waiting for a response from '{peer_id}', who is currently offline.
            
            Your goal is to reply to '{user_id}'s last message by **impersonating '{peer_id}'** and ensuring the conversation continues naturally.
            
            You must use the retrieved RAG context (PAST CHAT, PEER DOCS, etc.) 
            to understand the ongoing topic, but your tone and voice MUST match '{peer_id}' 
            based on the CONVERSATION HISTORY.

            Before generating a response, you MUST call the `retrieve_rag_context` tool.

            Once you receive the RAG context from the tool, use it to formulate 
            your final, concise answer in the persona of '{peer_id}'. The final message 
            MUST contain the word 'TERMINATE'.
            """
        else:
            # Standard RAG Assistant Prompt (for /ws/model endpoint) - STRONGLY FORCED TOOL CALL
            tool_instruction = """
            Your **FIRST and ONLY** response MUST be a call to the `retrieve_rag_context` tool. 
            You MUST NOT output any text, explanation, or reasoning before the tool call.
            
            You are an intelligent RAG assistant. Your primary function is to answer questions 
            based on your knowledge bases (global and peer-specific documents/history).
            
            Before generating a response, you MUST call the `retrieve_rag_context` tool.
            
            Once you receive the RAG context from the tool, use ONLY that context to formulate 
            your final, concise answer. The final message MUST contain the word 'TERMINATE'.
            If the context is empty, state that no relevant information was found, and still output 'TERMINATE'.
            """
        
        # Append the tool instructions and termination format
        kwargs["system_message"] = base_system_message + "\n\n" + tool_instruction + """
        
        Example final message:
        FINAL ANSWER: The capital of France is Paris. TERMINATE
        
        You MUST ensure your final response contains ONLY the answer text, followed by the word 'TERMINATE'.
        DO NOT INCLUDE the Example final message or Example tool call in your actual response.
        """

        # 3. Call the parent constructor with the final system message
        super().__init__(**kwargs)
        
        # 4. Register the function AFTER super().__init__() is called
        rag_tool_func = partial(
            self.rag_system.rag_function_call, 
            user_id=self.user_id, 
            peer_id=self.peer_id
        )
        
        # Register the tool for LLM calling and execution
        self.register_function(
            function_map={
                "retrieve_rag_context": rag_tool_func
            }
        )
        
    # FIX APPLIED HERE: Robust parsing logic to avoid 'raw_content' error
    async def on_messages(self, messages: List[dict], user_id: str, peer_id: Optional[str] = None) -> dict:
        user_query_message = messages[-1].get("content", "")
        # The full history text for the tool call prompt
        full_history_text = "\n".join([f"{msg.get('source', 'unknown')}: {msg.get('content', '')}" for msg in messages])
        debugger.log(f"Full history for chat init: {full_history_text}")

        # We must set a specific termination check for the UserProxyAgent to stop the loop
        def termination_check(message):
            return isinstance(message, dict) and "TERMINATE" in message.get("content", "").upper()

        user_proxy = UserProxyAgent(
            name=user_id,
            # Max replies raised to ensure full RAG/Tool-Call cycle can complete.
            max_consecutive_auto_reply=7, 
            human_input_mode="NEVER",
            code_execution_config=False,
            is_termination_msg=termination_check, # Use the custom termination check
        )
        
        rag_agent = self
        
        try:
            # initiate_chat accepts a string for the initial message
            chat_result = await user_proxy.a_initiate_chat( # Using a_initiate_chat for async
                rag_agent,
                message=user_query_message,
                silent=True,
                **{"history_text": full_history_text}
            )
            
            # Default fallback message if no TERMINATE message is found
            final_answer = "Sorry, the conversation failed to produce a result." 
            
            # The chat_history contains the conversation
            if chat_result.chat_history:
                # Find the last message that contains "TERMINATE"
                for msg in reversed(chat_result.chat_history):
                    
                    # 1. Only consider messages with content
                    if isinstance(msg, dict) and msg.get("content") is not None:
                        
                        raw_content = msg["content"]
                        debugger.log(f"Parsing content: {raw_content[:100]}") # DEBUG LOG
                        
                        # Check for termination keyword
                        if "TERMINATE" in raw_content.upper():
                            
                            # Use regex for robust cleaning:
                            
                            # 1. Strip the 'FINAL ANSWER:' tag robustly (case-insensitive and whitespace-tolerant)
                            # This targets the start of the required format.
                            cleaned_answer = re.sub(r'^\s*FINAL ANSWER:\s*', '', raw_content, flags=re.IGNORECASE).strip()

                            # 2. Aggressively strip everything from the point of 'TERMINATE' onwards 
                            # (or until the first occurrence of TERMINATE if the model put it mid-sentence)
                            answer_part = cleaned_answer
                            if 'TERMINATE' in answer_part.upper():
                                final_answer = answer_part[:answer_part.upper().find('TERMINATE')].strip()
                            else:
                                final_answer = answer_part.strip()

                            # 3. Aggressively remove any lingering example text/instructions from the end
                            final_answer = re.sub(r'Example tool call:.*$', '', final_answer, flags=re.IGNORECASE).strip()
                            
                            debugger.log(f"SUCCESS: Final Answer Extracted: {final_answer[:50]}") # DEBUG LOG
                            break # Found the answer, exit the loop
                            
            return create_text_message(content=final_answer, source=self.name)
            
        except Exception as e:
            # This catch block is for true Autogen/AIO/API failures (not parsing errors)
            debugger.log(f"AutoGen chat failure: {e}", level="error")
            return create_text_message(content="An unexpected error occurred in the RAG agent process.", source=self.name)


# REFACTORED: get_agent now filters config
async def get_agent(history: list[dict[str, Any]], rag_system: RAGSystem, user_id: str, peer_id: str, is_p2p_proxy: bool = False) -> RAGAssistantAgent:
    
    raw_config = rag_system.primary_llm_config
    
    # FILTERING FIX: These keys are internal to your app's config loader and must be removed
    INTERNAL_KEYS = ["type", "provider", "rephrase_response"] 
    
    filtered_config = {
        k: v for k, v in raw_config.items() 
        if k not in INTERNAL_KEYS
    }

    # Pass only the clean configuration to the Autogen agent
    llm_config = {
        "config_list": [filtered_config],
        "temperature": raw_config.get("temperature", 0.7) 
    }
    
    # NOTE: Agent name is "assistant" by default, which is used as the source in create_text_message
    agent = RAGAssistantAgent(
        name="assistant",
        system_message="You are a helpful RAG assistant.",
        rag_system=rag_system,
        user_id=user_id,
        peer_id=peer_id,
        llm_config=llm_config,
        is_p2p_proxy=is_p2p_proxy # Pass the new flag
    )
    agent._history = history
    return agent

# --- CONFIGURATION AND INSTANTIATION (UNCHANGED) ---
try:
    model_config = load_llm_config("model_config.yaml")
    config_list = model_config.get("config_list", [])
except FileNotFoundError:
    debugger.log("Error: model_config.yaml not found. Please create it.", level="error")
    exit()
except Exception as e:
    debugger.log(f"Error loading model_config.yaml: {e}", level="error")
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

# --- FASTAPI PROCESSOR FUNCTIONS ---

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
                request_dict = data

                await chat_manager.add_model_message_to_history(user_id, request_dict)

                history_list = await chat_manager.get_model_chat_history(user_id)
                if not history_list:
                    debugger.log("CRITICAL ERROR: History list is empty.", level="error")
                    await chat_manager.send_message(user_id, {"type": "error", "content": "Failed to retrieve history for processing the request.", "source": "system"})
                    break

                messages_for_agent = [msg for msg in history_list if 'source' in msg and 'content' in msg]

                rag_system.ingest_messages_for_rag(messages=history_list, user_id=user_id, peer_id="model")

                # Note: is_p2p_proxy=False (default) for model chat
                agent = await get_agent(history=messages_for_agent, rag_system=rag_system, user_id=user_id, peer_id="model")
                
                response_dict = await agent.on_messages(
                    messages=messages_for_agent,
                    user_id=user_id,
                    peer_id="model"
                )
                
                # Check for and log the extracted response for debugging
                debugger.log(response_dict)
                debugger.log("Assistant response", response_content=response_dict.get("content"))

                await chat_manager.add_model_message_to_history(user_id, response_dict)

                rag_system.ingest_messages_for_rag(messages=[response_dict], user_id=user_id, peer_id="model")

                await chat_manager.send_message(user_id, response_dict)

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
        history = await chat_manager.load_history_from_cache(user_id, peer_id)
        await chat_manager.send_message(user_id, {"type": "history", "content": history})

        while True:
            # ⭐ FIX 1: Peer Offline Check - Allows switching from Live Chat to RAG Proxy
            if not chat_manager.is_connected(peer_id):
                debugger.log(f"Peer '{peer_id}' went offline. Terminating live chat task for switching to RAG proxy.", user_id=user_id)
                await chat_manager.send_message(user_id, {"type": "status_update", "content": "Peer went offline. AI proxy mode started.", "peer_online": False})
                break # Exit the live chat loop to trigger the endpoint's outer switch loop 
            
            message_data = await chat_manager.get_message(user_id)
            if message_data is None:
                break

            if message_data.get("type") == "document_shared":
                filename = message_data.get("filename")
                file_url = message_data.get("file_url")

                file_message = {
                    "type": "document_shared",
                    "filename": filename,
                    "file_url": file_url,
                    "source": user_id,
                    "timestamp": time.time(),
                    "id": str(uuid.uuid4()) 
                }

                success, doc_message = await rag_system.ingest_peer_doc(os.path.join(UPLOAD_DIR, filename), user_id, peer_id)
                await chat_manager.add_message_to_history(user_id, peer_id, file_message)

                if chat_manager.is_connected(peer_id):
                    await chat_manager.send_message_to_peer(peer_id, file_message)
                continue
            
            message = create_text_message(
                content=message_data["content"],
                source=user_id,
                to=peer_id
            )
            
            debugger.log(message)

            await chat_manager.add_message_to_history(user_id, peer_id, message)

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
            # Do not disconnect user_id, let the endpoint's finally block handle that.
            await chat_manager.send_message(user_id, {
                "type": "error",
                "content": "An internal server error occurred.",
                "source": "system"
            })
        except Exception:
            pass
    # Removing the finally block here allows the outer endpoint to manage disconnect/cleanup.
    # The break/exception handles termination of this task.

async def rag_peer_chat_processor(user_id: str, peer_id: str):
    """
    Handles chat when peer_id is offline. The RAG agent acts as a proxy for the peer.
    """
    history_list = await chat_manager.load_history_from_cache(user_id, peer_id)
    if not isinstance(history_list, list):
        history_list = []
            
    await chat_manager.send_message(user_id, {"type": "history", "content": history_list})
    debugger.log(f"Initial history loaded and sent for RAG PROXY user '{user_id}'.")
    
    try:
        while True:
            # 1. Peer Online Check (must be first, allows switching from RAG Proxy to Live Chat)
            if chat_manager.is_connected(peer_id):
                debugger.log(f"Peer '{peer_id}' came online! Terminating RAG proxy task.", user_id=user_id)
                await chat_manager.send_message(user_id, {"type": "status_update", "content": "Peer is now online. Please refresh or reconnect.", "peer_online": True})
                break # Exit the RAG proxy loop

            # 2. Fetch message with timeout (to allow for periodic Peer Online Check)
            message_data = None
            try:
                # Use a timeout to periodically check is_connected status
                message_data = await asyncio.wait_for(
                    chat_manager.get_message(user_id), 
                    timeout=2.0 # Check for peer status every 2 seconds
                )
            except asyncio.TimeoutError:
                continue # Go back to the top of the while loop to re-check status
                
            if message_data is None:
                # User disconnected (listener put None in the queue)
              break
                    
            # ⭐ FIX 2 APPLIED HERE: The message is already in 'message_data'
            
            # The message is sent by the active user (user_id)
            user_message = create_text_message(
                content=message_data["content"], # Use the consumed message_data
                source=user_id,
                to=peer_id
            )
            await chat_manager.add_message_to_history(user_id, peer_id, user_message)

            history_list = await chat_manager.load_history_from_cache(user_id, peer_id)
            
            # Ingest all history up to this point for RAG context building
            rag_system.ingest_messages_for_rag(messages=history_list, user_id=user_id, peer_id=peer_id)

            messages_for_agent = [msg for msg in history_list if 'source' in msg and 'content' in msg]
            
            # Agent is created to act as a proxy (is_p2p_proxy=True)
            agent = await get_agent(
                history=messages_for_agent, 
                rag_system=rag_system, 
                user_id=user_id, 
                peer_id=peer_id,
                is_p2p_proxy=True
            )
            
            # This is where the model is called to reply to the message now stored in history
            rag_response_dict = await agent.on_messages(
                messages=messages_for_agent,
                user_id=user_id,
                peer_id=peer_id
            )
            
            # The AI response is sourced as the *offline peer* (peer_id), not 'assistant'
            ai_proxy_message = create_text_message(
                content=rag_response_dict["content"],
                source=peer_id, # Source is the offline peer
                to=user_id,
                is_proxy=True
            )
            
            await chat_manager.add_message_to_history(user_id, peer_id, ai_proxy_message)

            await chat_manager.send_message(user_id, {
                "type": "chat",
                "source": ai_proxy_message["source"],
                "content": ai_proxy_message["content"],
                "is_proxy": True
            })

    except Exception as e:
        debugger.log(f"Error in RAG peer chat processor for {user_id}: {e}", level="error")
        await chat_manager.send_message(user_id, {"type": "error", "content": "An internal server error occurred in RAG processor.", "source": "system"})
    finally:
        # We rely on the outer endpoint loop for chat_manager.disconnect(user_id)
        pass


@app.websocket("/ws/model/{user_id}")
async def websocket_model_chat_endpoint(websocket: WebSocket, user_id: str):
    # ... (model websocket endpoint is UNCHANGED) ...
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
    """
    WebSocket endpoint for peer-to-peer chat, dynamically switching to RAG proxy if the peer is offline.
    The outer loop manages the switching between the RAG proxy task and the live chat task.
    """
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
            # Signal the processing task to shut down gracefully
            if user_id in chat_manager.message_queues:
                await chat_manager.message_queues[user_id].put(None)

    listener_task = asyncio.create_task(websocket_listener())
    processing_task = None # Initialize processing task

    try:
        # Outer loop to handle processor switching (RAG -> Live Chat or vice versa)
        while True:
            # 1. Determine which processor to run
            if chat_manager.is_connected(peer_id):
                debugger.log(f"Peer '{peer_id}' is ONLINE. Starting normal chat processor.", user_id=user_id)
                processor_func = peer_chat_task_processor
            else:
                debugger.log(f"Peer '{peer_id}' is OFFLINE. Starting RAG chat processor (PROXY MODE).", user_id=user_id)
                processor_func = rag_peer_chat_processor
            
            # 2. Start the chosen processor
            processing_task = asyncio.create_task(processor_func(user_id, peer_id))
            
            # 3. Wait for EITHER the listener (user disconnect) OR the processor (peer status change)
            done, pending = await asyncio.wait(
                [listener_task, processing_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 4. Handle Completion/Switching
            if listener_task in done:
                # User disconnected. Break the outer loop and proceed to cleanup.
                debugger.log(f"Listener terminated for '{user_id}', stopping processing task.", user_id=user_id)
                break
                
            if processing_task in done:
                # The processor finished. This is the desired switch point (e.g., RAG proxy stopped because peer is online, or live chat stopped because peer is offline).
                if not processing_task.done():
                    processing_task.cancel()

                # Continue the 'while True' loop, which immediately re-runs the status check
                # and starts the new, correct processor.
                debugger.log(f"Processing task terminated. Re-checking peer status and restarting processor.", user_id=user_id)
                await asyncio.sleep(0.1) # Small pause to prevent tight CPU loop on error
                continue
            
    except asyncio.CancelledError:
        debugger.log(f"Tasks for user '{user_id}' were cancelled.", user_id=user_id)
    except Exception as e:
        debugger.log(f"An unexpected error occurred in endpoint for user '{user_id}': {e}", level="error")
    finally:
        debugger.log(f"Cleaning up tasks for user '{user_id}'.", user_id=user_id)
        # Ensure all tasks are cancelled before the endpoint exits
        if listener_task and not listener_task.done():
            listener_task.cancel()
        if processing_task and not processing_task.done():
            processing_task.cancel()
        
        chat_manager.disconnect(user_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)