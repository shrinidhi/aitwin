"""
This module provides a FastAPI application for a RAG (Retrieval-Augmented Generation) chatbot.

The application serves a web interface, manages WebSocket connections for real-time
chat, and implements a RAG pipeline to answer user questions based on a knowledge
base and chat history. It uses ChromaDB for vector storage and HuggingFace models
for text embedding.

Key components:
- `QueueManager`: Manages WebSocket connections and per-user message queues.
- `RAGSystem`: Handles the core RAG logic, including document ingestion, text chunking,
  embedding, retrieval from ChromaDB, and response generation using an LLM.
- `RAGAssistantAgent`: An Autogen agent that integrates with the `RAGSystem` to
  process chat messages and generate responses.
- FastAPI Endpoints:
  - `/`: Serves the main HTML file.
  - `/ingest`: Triggers the ingestion of documents into the knowledge base.
  - `/metrics`: Provides performance metrics for the RAG pipeline.
  - `/connections`: Returns the number of active WebSocket connections.
  - `/ws/{user_id}`: The WebSocket endpoint for chat communication.
"""

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
from typing import Any, List, Tuple, Dict
import yaml

import aiofiles
from autogen import ConversableAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# NEW: Import the DebugLogger
from debugLogger import DebugLogger

# OLD: Remove the manual OpenTelemetry setup. The DebugLogger handles this.
# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# trace.set_tracer_provider(TracerProvider())
# tracer = trace.get_tracer(__name__)
# otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
# trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))

# NEW: Initialize the centralized debugger
debugger = DebugLogger(service_name="rag-chatbot")

# --- FASTAPI SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="."), name="static")

# Use environment variables for sensitive or frequently changed configs
HISTORY_DIR = os.environ.get("HISTORY_DIR", "./history_files")
os.makedirs(HISTORY_DIR, exist_ok=True)
METRICS_PATH = "metrics.json"
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./tmp/chromadb")


# --- QUEUE AND CONNECTION MANAGEMENT ---
class QueueManager:
    """Manages WebSocket connections and per-user message queues."""
    def __init__(self):
        """Initializes the QueueManager with empty dictionaries for connections and queues."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Accepts a new WebSocket connection and initializes a message queue for the user.

        Args:
            websocket (WebSocket): The new WebSocket connection.
            user_id (str): A unique identifier for the user.
        """
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.message_queues[user_id] = asyncio.Queue()
        debugger.log("Client connected", user_id=user_id, total_connections=len(self.active_connections))

    def disconnect(self, user_id: str):
        """
        Disconnects a user by removing their WebSocket connection and message queue.

        Args:
            user_id (str): The unique identifier of the user to disconnect.
        """
        if user_id in self.active_connections:
            # Signal the processor to shut down
            if user_id in self.message_queues:
                self.message_queues[user_id].put_nowait(None)
                del self.message_queues[user_id]
            del self.active_connections[user_id]
        debugger.log("Client disconnected", user_id=user_id, total_connections=len(self.active_connections))

    async def get_message(self, user_id: str):
        """
        Retrieves the next message from a user's queue.

        Args:
            user_id (str): The unique identifier for the user.

        Returns:
            Any: The message from the queue, or None if the queue doesn't exist.
        """
        if user_id in self.message_queues:
            return await self.message_queues[user_id].get()
        return None
    
    async def send_message(self, user_id: str, message: Any):
        """
        Sends a JSON message to a specific user's WebSocket.

        Args:
            user_id (str): The unique identifier for the user.
            message (Any): The JSON-serializable message to send.
        """
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)

    @property
    def connection_count(self) -> int:
        """
        Returns the number of currently active WebSocket connections.
        
        Returns:
            int: The number of active connections.
        """
        return len(self.active_connections)

queue_manager = QueueManager()


# --- UTILITY FUNCTIONS FOR METRICS ---
async def _record_metrics(data: dict):
    """
    Safely appends a new metric entry to the metrics JSON file.

    Args:
        data (dict): A dictionary containing metric data to be recorded.
    """
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
    """
    Loads and parses a YAML configuration file for LLMs.

    Args:
        filepath (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(filepath, 'r') as file:
        return yaml.safe_load(file) 
            

# --- RAG PIPELINE CONFIGURATION ---
class RAGSystem:
    """
    Manages the Retrieval-Augmented Generation pipeline.

    This class handles document ingestion, text embedding, retrieval from a vector database,
    and generating responses using a Large Language Model (LLM).
    """
    def __init__(self, config_list: List[dict], prompt_file: str, collection_name: str, history_collection_name: str, upload_dir: str, doc_extension: str = "*.{txt,pdf}", model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 50):
        """
        Initializes the RAGSystem with configurations and database clients.

        Args:
            config_list (List[dict]): A list of LLM configurations.
            prompt_file (str): The file path to the YAML file containing prompt templates.
            collection_name (str): The name of the ChromaDB collection for the knowledge base.
            history_collection_name (str): The name of the ChromaDB collection for chat history.
            upload_dir (str): The directory where documents for ingestion are stored.
            doc_extension (str): The file extension pattern for documents to ingest.
            model_name (str): The name of the HuggingFace model for embedding.
            batch_size (int): The batch size for processing documents.
        """
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
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            debugger.log(f"Collection '{self.collection_name}' found.", level="info")
        except Exception:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            debugger.log(f"Collection '{self.collection_name}' created.", level="info")
            
        try:
            self.history_collection = self.chroma_client.get_or_create_collection(name=self.history_collection_name)
            debugger.log(f"History collection '{self.history_collection_name}' found or created.", level="info")
        except Exception as e:
            debugger.log(f"Failed to get/create history collection: {e}", level="error")
            
    def _load_prompts(self, filepath: str) -> dict:
        """
        Loads and parses a YAML configuration file for prompts.
        
        Args:
            filepath (str): The path to the YAML configuration file.
        
        Returns:
            dict: The loaded configuration as a dictionary, or an empty dict on error.
        """
        try:
            with open(filepath, 'r') as file:
                return yaml.safe_load(file).get('prompts', {})
        except FileNotFoundError:
            debugger.log(f"Warning: Prompt file '{filepath}' not found. Using default prompts.", level="warning")
            return {}
        except Exception as e:
            debugger.log(f"Error loading prompt file '{filepath}': {e}. Using default prompts.", level="error")
            return {} 
            
    def embed_texts(self, texts: list[str]):
        """
        Generates embeddings for a list of text documents using the HuggingFace model.

        Args:
            texts (list[str]): A list of strings to embed.

        Returns:
            list: A list of embedding vectors.
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy().tolist()

    def _chunk_text(self, text, chunk_size=600, overlap=120):
        """
        Splits a long text into smaller, overlapping chunks.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The desired size of each chunk.
            overlap (int): The number of characters to overlap between chunks.

        Returns:
            list[str]: A list of text chunks.
        """
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
        """
        Generates a unique ID for a text chunk using a hash function.

        Args:
            text (str): The text chunk to hash.

        Returns:
            str: The MD5 hash of the text.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _normalize_roman_numerals(self, text):
        """
        Converts Roman numerals in text to Arabic numerals.

        Args:
            text (str): The text to normalize.

        Returns:
            str: The text with Roman numerals converted.
        """
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
        Ingests all documents from the upload directory into the ChromaDB collection.

        The function clears existing documents, reads new ones (txt or pdf),
        chunks them, and upserts them into the database.

        Returns:
            JSONResponse: A response indicating the status of the ingestion process.
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

    def ingest_message(self, message: TextMessage, user_id: str):
        """
        Ingests a single chat message into the chat history collection with user metadata.

        Args:
            message (TextMessage): The message object to ingest.
            user_id (str): The unique identifier of the user who sent the message.
        """
        if not message.content:
            return
        
        message_id = str(hashlib.sha256(f"{user_id}_{message.content}".encode('utf-8')).hexdigest())
        metadata = {
            "source": message.source,
            "timestamp": time.time(),
            "user_id": user_id
        }
        
        try:
            self.history_collection.upsert(
                documents=[message.content],
                ids=[message_id],
                metadatas=[metadata]
            )
            debugger.log(f"Ingested message from '{message.source}' for user '{user_id}' into history collection.", source=message.source, user_id=user_id)
        except Exception as e:
            debugger.log(f"Failed to ingest message into history collection for user '{user_id}': {e}", level="error", user_id=user_id)

    def retrieve_and_answer(self, messages: List[TextMessage], user_id: str) -> Tuple[str, dict]:
        """
        Executes the full RAG pipeline for a given conversation.

        The process includes:
        1. Rephrasing the latest query into a standalone question using a LLM.
        2. Retrieving relevant documents from the knowledge base and chat history.
        3. Generating a final answer based on the retrieved context using a primary LLM.

        Args:
            messages (List[TextMessage]): The list of all messages in the current conversation.
            user_id (str): The unique identifier of the user.

        Returns:
            Tuple[str, dict]: A tuple containing the generated answer and a dictionary
                              of performance metrics.
        """
        start_total = time.time()
        debugger.log(f"--- RAG pipeline executing for user '{user_id}' with conversation history. ---", user_id=user_id)
        
        latest_query = messages[-1].content
        
        if latest_query.lower().strip() in ["hi", "hello", "hey"]:
            return "Hello! I am a helpful RAG chatbot. How can I assist you today?", {}

        history = "\n".join([f"{msg.source}: {msg.content}" for msg in messages])
        
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
            
            start_rephrase = time.time()
            rephrase_prompt_template = self.prompts.get("rephrase_prompt_v1", "Default rephrase prompt not found.")
            rephrase_prompt = rephrase_prompt_template.format(
                history=history,
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
            
            start_retrieval = time.time()
            query_emb = self.embed_texts([standalone_query])[0]
            
            kb_results = self.collection.query(query_embeddings=[query_emb], n_results=5)
            history_results = self.history_collection.query(
                query_embeddings=[query_emb],
                n_results=5,
                where={"user_id": user_id}
            )
            
            end_retrieval = time.time()
            
            kb_docs = kb_results.get("documents", [[]])[0]
            history_docs = history_results.get("documents", [[]])[0]
            combined_docs = kb_docs + history_docs
            context = "\n".join(combined_docs)
            
            metrics["retrieval_latency"] = end_retrieval - start_retrieval
            metrics["retrieved_docs"] = len(combined_docs)
            
            if not context.strip():
                metrics["total_latency"] = time.time() - start_total
                return "Sorry, I could not find any relevant information on that topic in the knowledge base or your chat history.", metrics
            
            start_generation = time.time()
            primary_prompt_template = self.prompts.get("primary_prompt_v1", "Default primary prompt not found.")
            prompt = primary_prompt_template.format(
                context=context,
                history=history,
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

# --- AGENT SETUP ---
class RAGAssistantAgent(ConversableAgent):
    """
    An Autogen ConversableAgent that uses the RAGSystem to respond to messages.
    """
    def __init__(self, rag_system: RAGSystem, **kwargs):
        """
        Initializes the agent with a reference to the RAGSystem.

        Args:
            rag_system (RAGSystem): The RAG system instance to use for generating responses.
            **kwargs: Additional arguments for the ConversableAgent base class.
        """
        super().__init__(**kwargs)
        self.rag_system = rag_system

    async def on_messages(self, messages: List[TextMessage], cancellation_token: CancellationToken, user_id: str) -> Tuple[TextMessage, dict]:
        """
        Handles incoming messages and generates a response using the RAG pipeline.

        Args:
            messages (List[TextMessage]): The list of messages in the conversation.
            cancellation_token (CancellationToken): A token to check for cancellation.
            user_id (str): The unique identifier of the user.

        Returns:
            Tuple[TextMessage, dict]: A tuple containing the generated response message
                                      and the performance metrics.
        """
        rag_response, metrics = self.rag_system.retrieve_and_answer(messages, user_id)
        return TextMessage(content=rag_response, source=self.name), metrics

async def get_agent(history: list[dict[str, Any]], rag_system: RAGSystem) -> RAGAssistantAgent:
    """
    Creates and initializes a RAGAssistantAgent with the given history.

    Args:
        history (list[dict[str, Any]]): The conversation history to load into the agent.
        rag_system (RAGSystem): The RAG system instance for the agent to use.

    Returns:
        RAGAssistantAgent: The initialized agent instance.
    """
    agent = RAGAssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        rag_system=rag_system
    )
    agent._history = history
    return agent


#  Initialize RAGSystem globally
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
# # Initialize RAGSystem globally
# rag_system = RAGSystem(collection_name="generalized_collection", history_collection_name="chat_history_collection", upload_dir=UPLOAD_DIR, doc_extension="*.*", batch_size=100)

# --- FASTAPI ENDPOINTS ---
@app.get("/")
async def root():
    """Serves the main HTML page for the chatbot interface."""
    return FileResponse("app_agent.html")

@app.post("/ingest")
async def ingest_documents():
    """Triggers the ingestion of documents into the RAG knowledge base."""
    return rag_system.ingest_docs()

async def get_history_for_user(user_id: str) -> list[dict[str, Any]]:
    """
    Loads chat history from a user-specific file.

    Args:
        user_id (str): The unique identifier of the user.

    Returns:
        list[dict[str, Any]]: The loaded chat history as a list of dictionaries.
    """
    history_file_path = os.path.join(HISTORY_DIR, f"agent_history_{user_id}.json")
    try:
        if not os.path.exists(history_file_path):
            return []
        async with aiofiles.open(history_file_path, "r") as file:
            contents = await file.read()
            if not contents.strip():
                return []
            return json.loads(contents)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        debugger.log(f"Failed to decode JSON from {history_file_path}. File might be corrupted.", level="error", file_path=history_file_path)
        return []
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while loading history: {str(e)}") from e

@app.get("/metrics")
async def get_metrics():
    """
    Returns the performance metrics for each chat response.

    Returns:
        JSONResponse: A JSON object containing the recorded metrics.
    """
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
    """
    Returns the number of currently active WebSocket connections.

    Returns:
        JSONResponse: A JSON object with the active connection count.
    """
    return JSONResponse(content={"active_connections": queue_manager.connection_count})

async def task_processor(user_id: str):
    """
    Background task to process messages from a user's queue.

    This task continuously listens for new messages in the queue, processes them
    using the RAG system, and sends the response back to the client.

    Args:
        user_id (str): The unique identifier of the user.
    """
    history_list = await get_history_for_user(user_id)
    if not isinstance(history_list, list):
        history_list = []
    
    # Send history to the client upon connection
    await queue_manager.send_message(user_id, {"type": "history", "content": history_list})

    while True:
        try:
            data = await queue_manager.get_message(user_id)
            if data is None:
                # Disconnect signal received
                break
            
            # REPLACED: Old OpenTelemetry span and attribute calls
            with debugger.start_span("chat_message"):
                debugger.log("Received message", user_id=user_id, user_input=data)

                request_data = json.loads(data)
                request = TextMessage(**request_data)

                rag_system.ingest_message(request, user_id)
                history_list.append(request.model_dump())
                
                agent = await get_agent(history=history_list, rag_system=rag_system)
                
                response, metrics = await agent.on_messages(
                    messages=[TextMessage(**msg) for msg in history_list], 
                    cancellation_token=CancellationToken(),
                    user_id=user_id
                )

                debugger.log("Assistant response", response_content=response.content)
            
            def safe_model_dump(obj):
                return json.loads(json.dumps(obj, default=str))
            
            rag_system.ingest_message(response, user_id)
            history_list.append(response.model_dump())
            
            # await _record_metrics(metrics)
            
            history_file_path = os.path.join(HISTORY_DIR, f"agent_history_{user_id}.json")
            async with aiofiles.open(history_file_path, "w") as file:
                await file.write(json.dumps(history_list, indent=2, default=str))
                
            assert isinstance(response, TextMessage)
            
            await queue_manager.send_message(user_id, safe_model_dump(response.model_dump()))

        except Exception as e:
            debugger.log(f"An unexpected error occurred in the processor for user '{user_id}': {e}", level="error", user_id=user_id)
            error_message = {
                "type": "error",
                "content": "An internal server error occurred.",
                "source": "system"
            }
            await queue_manager.send_message(user_id, error_message)
            # Break the loop to stop processing for this user
            break


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    Manages the WebSocket connection for a specific user.

    Args:
        websocket (WebSocket): The WebSocket connection object.
        user_id (str): The unique identifier for the user.
    """
    await queue_manager.connect(websocket, user_id)
    debugger.log(f"WebSocket connected for user: {user_id}", user_id=user_id)
    
    # Start the background task to process messages from the queue
    processing_task = asyncio.create_task(task_processor(user_id))
    
    try:
        # Loop to receive messages and place them in the queue
        while True:
            data = await websocket.receive_text()
            if data is not None:
                await queue_manager.message_queues[user_id].put(data)
    except WebSocketDisconnect:
        debugger.log(f"WebSocket disconnected for user '{user_id}'. Signaling processor to stop.", user_id=user_id)
    finally:
        queue_manager.disconnect(user_id)
        if not processing_task.done():
            processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)