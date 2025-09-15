# create folder temp/chromadb/ add .txt files inside this folder




import json
import os
import time
import glob
import chromadb
import hashlib
import fitz
import re
from transformers import AutoTokenizer, AutoModel
import torch
from openai import OpenAI
from typing import Any, List, Tuple

import aiofiles
from autogen import ConversableAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
HISTORY_PATH = os.environ.get("HISTORY_PATH", "agent_history.json")
METRICS_PATH = "metrics.json"
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./tmp/chromadb")

# --- UTILITY FUNCTIONS FOR METRICS ---
async def _record_metrics(data: dict):
    """Safely appends a new metric entry to the metrics JSON file."""
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

# --- RAG PIPELINE CONFIGURATION ---
class RAGSystem:
    def __init__(self, collection_name: str, upload_dir: str, doc_extension: str = "*.{txt,pdf}", model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 50):
        self.collection_name = collection_name
        self.upload_dir = upload_dir
        self.doc_extension = doc_extension
        self.batch_size = batch_size
        self.hf_model = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
            self.model = AutoModel.from_pretrained(self.hf_model)
        except Exception as e:
            print(f"Failed to load HuggingFace models: {e}")
            raise RuntimeError("Model loading failed.")
        
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' found.")
        except Exception:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' created.")

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
            print(message)
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": message})
        
        if self.collection.count() > 0:
            print("Clearing existing documents...")
            self.collection.delete(ids=self.collection.get(include=[])['ids'])
        
        start = time.time()
        total_chunks = 0
        
        for doc_path in files:
            print(f"Ingesting knowledge base file: {doc_path}")
            
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
                    print(f" Skipping unsupported file type: {file_extension} for {doc_path}")
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
                print(f" Upserted {len(documents_to_add)} documents from {os.path.basename(doc_path)}.")
                total_chunks += len(documents_to_add)

            except Exception as e:
                print(f" Failed to process document {doc_path}: {e}")
        
        end = time.time()
        
        message = f" Ingestion & indexing of {total_chunks} docs completed in {end-start:.2f} seconds."
        print(message)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": message})

    def retrieve_and_answer(self, messages: List[TextMessage]) -> Tuple[str, dict]:
        start_total = time.time()
        print(f"\n---  RAG pipeline executing with conversation history. ---")
        
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
            client = OpenAI(base_url="http://127.0.0.1:11433/v1", api_key="not-needed")
            
            # Step 1: Condense conversation to standalone query
            start_rephrase = time.time()
            rephrase_prompt = f"""You are a STRICT rephrasing agent.

Your ONLY task is to take the given conversation history and the follow-up question,  
and restate the follow-up question as a clear standalone question.  

Rules:
- Use ONLY the information from the CONVERSATION HISTORY and the FOLLOW-UP QUESTION.  
- Do NOT add, assume, or invent any new details.  
- If the follow-up question is already standalone, return it unchanged.  

CONVERSATION HISTORY:
{history}

FOLLOW-UP QUESTION:
{latest_query}

Standalone Question:"""

            
            rephrased_response = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct-v0.3",
                messages=[{"role": "user", "content": rephrase_prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            end_rephrase = time.time()
            standalone_query = rephrased_response.choices[0].message.content.strip()
            metrics["rephrase_latency"] = end_rephrase - start_rephrase
            metrics["rephrase_tokens"] = rephrased_response.usage.total_tokens
            print(f" Rephrased standalone query: '{standalone_query}'")
            
            # Step 2: Retrieval using the standalone query
            start_retrieval = time.time()
            query_emb = self.embed_texts([standalone_query])[0]
            results = self.collection.query(query_embeddings=[query_emb], n_results=10)
            end_retrieval = time.time()
            
            docs = results.get("documents", [[]])[0]
            context = "\n".join(docs)
            metrics["retrieval_latency"] = end_retrieval - start_retrieval
            metrics["retrieved_docs"] = len(docs)
            
            if not context.strip():
                metrics["total_latency"] = time.time() - start_total
                return "Sorry, I could not find any relevant information on that topic in the knowledge base.", metrics
            
            # Step 3: Generation using the retrieved context
            start_generation = time.time()
            prompt = f"""You are a STRICT knowledge agent.  

            Your job is ONLY to answer questions using the provided CONTEXT (retrieved documents) and the CONVERSATION HISTORY.  
            - Do NOT use your own brain, prior knowledge, or outside assumptions.  
            - Do NOT generate facts, explanations, or background information unless it is explicitly found in the CONTEXT or CONVERSATION HISTORY.  
            - If the answer cannot be found directly in the CONTEXT or CONVERSATION HISTORY, you MUST reply with exactly:  
            "Sorry, I could not find relevant information in the knowledge base."

CONTEXT (retrieved documents):
{context}

CONVERSATION HISTORY:
{history}

QUESTION:
{latest_query}

Answer strictly from CONTEXT and CONVERSATION HISTORY only.
"""

            response = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct-v0.3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800,
            )
            end_generation = time.time()
            
            answer = response.choices[0].message.content
            metrics["generation_latency"] = end_generation - start_generation
            metrics["generation_tokens"] = response.usage.total_tokens
            metrics["total_latency"] = end_generation - start_total
            
            print(f"---  RAG system generated a contextual answer. ---")
            return answer, metrics
        except Exception as e:
            error_message = f"An error occurred during the RAG process: {str(e)}"
            print(f"❗ {error_message}")
            metrics["total_latency"] = time.time() - start_total
            return "Sorry, an internal error occurred while processing your request.", metrics

# --- AGENT SETUP ---
class RAGAssistantAgent(ConversableAgent):
    def __init__(self, rag_system: RAGSystem, **kwargs):
        super().__init__(**kwargs)
        self.rag_system = rag_system

    async def on_messages(self, messages: List[TextMessage], cancellation_token: CancellationToken) -> Tuple[TextMessage, dict]:
        rag_response, metrics = self.rag_system.retrieve_and_answer(messages)
        return TextMessage(content=rag_response, source=self.name), metrics

async def get_agent(history: list[dict[str, Any]], rag_system: RAGSystem) -> RAGAssistantAgent:
    agent = RAGAssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        rag_system=rag_system
    )
    agent._history = history
    return agent

# Initialize RAGSystem globally
rag_system = RAGSystem(collection_name="generalized_collection", upload_dir=UPLOAD_DIR, doc_extension="*.*", batch_size=100)

# --- FASTAPI ENDPOINTS ---
@app.get("/")
async def root():
    return FileResponse("app_agent.html")

@app.post("/ingest")
async def ingest_documents():
    return rag_system.ingest_docs()

@app.get("/history")
async def get_history() -> list[dict[str, Any]]:
    try:
        if not os.path.exists(HISTORY_PATH):
            return []
        async with aiofiles.open(HISTORY_PATH, "r") as file:
            contents = await file.read()
            if not contents.strip():
                return []
            return json.loads(contents)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print(f"❗ Error: Failed to decode JSON from {HISTORY_PATH}. File might be corrupted.")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "Corrupted history file."})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while loading history: {str(e)}") from e

@app.get("/metrics")
async def get_metrics():
    """Returns the performance metrics for each chat response."""
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

@app.post("/chat", response_model=TextMessage)
async def chat(request: TextMessage) -> TextMessage:
    try:
        history_list = await get_history()
        if not isinstance(history_list, list):
            history_list = []
        
        history_list.append(request.model_dump())
        
        agent = await get_agent(history=history_list, rag_system=rag_system)
        
        response, metrics = await agent.on_messages(
            messages=[TextMessage(**msg) for msg in history_list], 
            cancellation_token=CancellationToken()
        )
        
        history_list.append(response.model_dump())
        
        await _record_metrics(metrics)
        
        async with aiofiles.open(HISTORY_PATH, "w") as file:
            await file.write(json.dumps(history_list, indent=2, default=str))
            
        assert isinstance(response, TextMessage)
        return response
    except Exception as e:
        print(f"An unexpected error occurred in the chat endpoint: {e}")
        error_message = {
            "type": "error",
            "content": "An internal server error occurred.",
            "source": "system"
        }
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message)