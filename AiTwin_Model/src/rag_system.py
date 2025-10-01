# rag_system.py

import os
import time
import glob
import chromadb
import hashlib
import fitz
import re
import aiofiles
import torch
import json
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from typing import List, Tuple, Optional, Dict, Any
from fastapi.responses import JSONResponse
from fastapi import status
from debugLogger import DebugLogger
from autogen_agentchat.messages import TextMessage
import yaml

# Constants loaded from main.py's environment/config
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")


class RAGSystem:
    def __init__(self, config_list: List[dict], prompt_file: str, collection_name: str, history_collection_name: str, upload_dir: str, debugger: DebugLogger, doc_extension: str = "*.{txt,pdf}", model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 50):
        self.collection_name = collection_name
        self.history_collection_name = history_collection_name
        self.upload_dir = upload_dir
        self.doc_extension = doc_extension
        self.batch_size = batch_size
        self.hf_model = model_name
        self.debugger = debugger
        
        self.primary_llm_config = next(cfg for cfg in config_list if cfg.get("type") == "primary_response")
        self.rephrase_llm_config = next(cfg for cfg in config_list if cfg.get("type") == "rephrase_response")
        self.prompts = self._load_prompts(prompt_file)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
            self.model = AutoModel.from_pretrained(self.hf_model)
        except Exception as e:
            self.debugger.log(f"Failed to load HuggingFace models: {e}", level="error")
            raise RuntimeError("Model loading failed.")

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collections = {} # New dictionary to hold peer-specific collections

        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            self.debugger.log(f"Global collection '{self.collection_name}' found.", level="info")
        except Exception:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            self.debugger.log(f"Global collection '{self.collection_name}' created.", level="info")

        try:
            self.history_collection = self.chroma_client.get_or_create_collection(name=self.history_collection_name)
            self.debugger.log(f"History collection '{self.history_collection_name}' found or created.", level="info")
        except Exception as e:
            self.debugger.log(f"Failed to get/create history collection: {e}", level="error")

    def get_peer_collection(self, user_id: str, peer_id: str):
        # Create a unique, consistent name for the collection based on the two user IDs
        sorted_ids = sorted([user_id, peer_id])
        collection_name = hashlib.md5(f"{sorted_ids[0]}_{sorted_ids[1]}".encode('utf-8')).hexdigest()
        if collection_name not in self.collections:
            try:
                self.collections[collection_name] = self.chroma_client.get_or_create_collection(name=collection_name)
                self.debugger.log(f"Peer-to-peer collection for {user_id} and {peer_id} found or created.", level="info")
            except Exception as e:
                self.debugger.log(f"Failed to get/create peer collection: {e}", level="error")
                return None
        return self.collections[collection_name]

    def _load_prompts(self, filepath: str) -> dict:
        try:
            with open(filepath, 'r') as file:
                return json.loads(json.dumps(yaml.safe_load(file))).get('config/prompts', {}) # Ensure serializable
        except FileNotFoundError:
            self.debugger.log(f"Warning: Prompt file '{filepath}' not found. Using default prompts.", level="warning")
            return {}
        except Exception as e:
            self.debugger.log(f"Error loading prompt file '{filepath}': {e}. Using default prompts.", level="error")
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
        """Ingests all documents from the upload directory into the Global Chroma DB collection."""
        files = glob.glob(os.path.join(self.upload_dir, f"**/{self.doc_extension}"), recursive=True)
        if not files:
            message = f"No documents found in '{self.upload_dir}' with extension '{self.doc_extension}'."
            self.debugger.log(message, level="warning")
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": message})

        if self.collection.count() > 0:
            self.debugger.log("Clearing existing documents...")
            self.collection.delete(ids=self.collection.get(include=[])['ids'])

        start = time.time()
        total_chunks = 0

        for doc_path in files:
            self.debugger.log(f"Ingesting knowledge base file: {doc_path}")

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
                    self.debugger.log(f" Skipping unsupported file type: {file_extension} for {doc_path}", level="warning")
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
                self.debugger.log(f"Upserted {len(documents_to_add)} documents from {os.path.basename(doc_path)}.", num_docs=len(documents_to_add), source=os.path.basename(doc_path))
                total_chunks += len(documents_to_add)

            except Exception as e:
                self.debugger.log(f"Failed to process document {doc_path}: {e}", level="error", doc_path=doc_path)

        end = time.time()

        message = f" Ingestion & indexing of {total_chunks} docs completed in {end-start:.2f} seconds."
        self.debugger.log(message, total_chunks=total_chunks, duration=end-start)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": message})

    async def ingest_peer_doc(self, file_path: str, user_id: str, peer_id: str):
        """Asynchronously ingests a document into a peer-specific Chroma DB collection."""
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
                self.debugger.log(f" Skipping unsupported file type for ingestion: {file_extension}", level="warning")
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
            self.debugger.log(f"Upserted {len(documents_to_add)} documents for P2P chat.", user_id=user_id, peer_id=peer_id)
            return True, f"Ingested {len(documents_to_add)} chunks from {os.path.basename(file_path)}."

        except Exception as e:
            self.debugger.log(f"Failed to process document {file_path}: {e}", level="error", file_path=file_path)
            return False, f"Failed to ingest document: {str(e)}"

    def retrieve_and_answer(self, messages: List[TextMessage], user_id: str, peer_id: Optional[str] = None) -> Tuple[str, dict]:
        """Runs the complete RAG pipeline: Rephrase -> Retrieve Context -> Generate Answer."""
        start_total = time.time()
        self.debugger.log(f"--- RAG pipeline executing for user '{user_id}' with conversation history. ---", user_id=user_id)

        latest_query = messages[-1].content
        K_RECENT = 5
        recent_messages = messages[-K_RECENT:]
        
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
            rephrase_prompt_template = self.prompts.get("rephrase_prompt_v1", "Default rephrase prompt not found. Rephrase the following latest query: '{latest_query}' considering the history: '{history}' to be a standalone search query.")
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
            self.debugger.log(f"Rephrased standalone query: '{standalone_query}'", standalone_query=standalone_query)

            # --- 2. Retrieve Context (Semantic Similarity + Recent History) ---
            start_retrieval = time.time()
            query_emb = self.embed_texts([standalone_query])[0]
            
            N_SIMILAR = 3 
            context_documents = [] 
            
            if peer_id and peer_id != "model":
                # Peer-to-peer chat RAG (using peer-specific document collection)
                peer_collection = self.get_peer_collection(user_id, peer_id)
                if peer_collection:
                    peer_results = peer_collection.query(
                        query_embeddings=[query_emb],
                        n_results=N_SIMILAR,
                        where={"$or": [{"sender_id": user_id}, {"receiver_id": user_id}]}
                    )
                    context_documents.extend(peer_results.get("documents", [[]])[0])
                    self.debugger.log("Retrieved documents from peer-specific knowledge base.", source="peer_docs")
                    
            elif peer_id == "model":
                # Chat with the RAG Model
                history_results = self.history_collection.query(
                    query_embeddings=[query_emb],
                    n_results=N_SIMILAR,
                    where={"$and": [
                    {"user_id": user_id},
                    {"peer_user_id": "model"}
                ]})
                
                similar_history_docs = history_results.get("documents", [[]])[0]
                
                similar_history_docs_filtered = []
                recent_contents = [msg.content for msg in recent_messages]
                for doc in similar_history_docs:
                    if doc not in recent_contents:
                        similar_history_docs_filtered.append(f"PAST CHAT: {doc}")
                
                context_documents.extend(similar_history_docs_filtered)
                self.debugger.log(f"Retrieved {len(similar_history_docs_filtered)} semantically similar chat history documents.", source="chat_history")
                
                # Fallback: If not enough context, also query the global KB
                if len(context_documents) < N_SIMILAR:
                    kb_results = self.collection.query(query_embeddings=[query_emb], n_results=N_SIMILAR)
                    context_documents.extend(kb_results.get("documents", [[]])[0])
                    self.debugger.log("Augmented context with global knowledge base documents.", source="knowledge_base")
            
            # --- Combine Contexts ---
            final_context_texts = recent_history_context_texts
            final_context_texts.extend(context_documents)
            context = "\n".join(final_context_texts)

            end_retrieval = time.time()

            metrics["retrieval_latency"] = end_retrieval - start_retrieval
            metrics["retrieved_docs"] = len(context_documents) 

            if not context.strip():
                metrics["total_latency"] = time.time() - start_total
                return "Sorry, I could not find any relevant information on that topic.", metrics
            
            
            # --- 3. Generate Answer ---
            start_generation = time.time()
            primary_prompt_template = self.prompts.get("primary_prompt_v1","Default primary prompt not found. Answer the query: '{latest_query}' based on the context: '{context}'")
            prompt = primary_prompt_template.format(
                context=context,
                history=full_history_text, 
                latest_query=latest_query
            )

            self.debugger.log("Primary LLM prompt generated")

            response = client.chat.completions.create(
                model=self.primary_llm_config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.primary_llm_config["temperature"],
                max_tokens=self.primary_llm_config["max_tokens"],
                stop=["REPLY:", "In response to your question,"],
            )
            end_generation = time.time()

            answer = response.choices[0].message.content
            metrics["generation_latency"] = end_generation - start_generation
            metrics["generation_tokens"] = response.usage.total_tokens
            metrics["total_latency"] = end_generation - start_total

            self.debugger.log("RAG system generated a contextual answer.", answer=answer)
            return answer, metrics
        except Exception as e:
            error_message = f"An error occurred during the RAG process: {str(e)}"
            self.debugger.log(f"An unexpected error occurred in the processor: {e}", level="error", error_message=error_message)
            metrics["total_latency"] = time.time() - start_total
            return "Sorry, an internal error occurred while processing your request.", metrics

    def ingest_messages_for_rag(self, messages: List[dict], user_id: str, peer_id: str):
        """
        Ingests messages into the ChromaDB history collection for retrieval.
        """
        documents_to_add = []
        ids_to_add = []
        metadatas_to_add = []

        for message in messages:
            content = message.get('content')
            source = message.get('source')
            # Use the DB-assigned ID (which should be guaranteed to exist post-persistence)
            message_id = message.get('id')
            self.debugger.log(f"Attempting ingestion for message ID: {message_id}")
            
            if not content or not message_id or message.get('type') == 'document_shared':
                # Skip document messages and messages without content/ID
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
                self.debugger.log(f"Ingested {len(documents_to_add)} messages into history collection for conversation between '{user_id}' and '{peer_id}'.")
            except Exception as e:
                self.debugger.log(f"Failed to ingest messages into history collection for '{user_id}' and '{peer_id}': {e}", level="error")