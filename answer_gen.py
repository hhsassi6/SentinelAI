import os
import gc
import torch
import numpy as np
import faiss
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List, Dict, Any, Optional

class BGEEmbeddings(Embeddings):
    """Embeddings implementation using BAAI BGE base model"""
    
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0]

class AnswerGenerator:
    def __init__(self, documents_folder: str, api_key: Optional[str] = None):
        self.documents_folder = documents_folder
        self.api_key = api_key
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup memory optimization
        self._optimize_memory()
        
        # Load model and tokenizer
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Initialize embedding model
        self.embedding = BGEEmbeddings()
        
        # Initialize or load vectorstore
        self.vectorstore = self._initialize_vectorstore()
    
    def _optimize_memory(self):
        """Configure memory optimizations"""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        if torch.cuda.is_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    def _load_model_and_tokenizer(self):
        """Load the LLM model and tokenizer"""
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "saved_model")
        
        # Check if the model is already downloaded locally
        if os.path.exists(model_path) and os.path.isdir(model_path):
            print(f"Loading model and tokenizer from local path: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            # Fallback to loading from HuggingFace Hub
            print(f"Loading model and tokenizer from HuggingFace: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Save model locally for future use
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"Model and tokenizer saved to {model_path}")
        
        return model, tokenizer
    
    def _initialize_vectorstore(self):
        """Initialize the FAISS vector store or load it if it exists"""
        faiss_index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faiss_index")
        
        try:
            # Try to load existing index
            if os.path.exists(faiss_index_path) and os.path.isdir(faiss_index_path):
                print(f"Loading FAISS index from {faiss_index_path}")
                return FAISS.load_local(faiss_index_path, self.embedding, allow_dangerous_deserialization=True)
            else:
                print("Creating new FAISS index...")
                # Initialize with empty index if none exists
                empty_texts = ["Placeholder text for initialization"]
                vectorstore = FAISS.from_texts(empty_texts, self.embedding)
                # Save the empty index
                os.makedirs(faiss_index_path, exist_ok=True)
                vectorstore.save_local(faiss_index_path)
                return vectorstore
        except Exception as e:
            print(f"Error initializing vectorstore: {e}")
            # Fallback to memory-only vectorstore
            empty_texts = ["Placeholder text for initialization"]
            return FAISS.from_texts(empty_texts, self.embedding)
    
    def _build_prompt(self, user_question: str, context: str) -> str:
        """Build the prompt with system context and user question"""
        return f"""
<|system|>
You are a senior compliance officer, highly experienced in the Due Diligence process for crypto asset funds.
Use ONLY the provided documentation context below to answer the user's question.
Keep the answer short, structured, professional, and comprehensive.
If the question is unclear or irrelevant, politely ask for clarification.

Context:
{context}
</|system|>

<|user|>
{user_question}
</|user|>

<|assistant|>
"""
    
    def generate_answer(self, question: str) -> str:
        """Generate an answer to the user's question using RAG"""
        try:
            # Retrieve relevant documents
            print(f"Searching relevant context for: '{question}'")
            docs = self.vectorstore.similarity_search(question, k=3)
            
            # Build context from retrieved documents
            context = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
            
            # Build the full prompt
            full_prompt = self._build_prompt(question, context)
            
            # Generate response
            print("Generating response...")
            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096)
            
            # Move inputs to the correct device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "<|assistant|>" in generated_text:
                response = generated_text.split("<|assistant|>")[-1].strip()
            else:
                response = generated_text.strip()
            
            print("Response generated successfully")
            return response
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I encountered an error while trying to generate an answer. Please try again later. Technical details: {str(e)}"
    
    def update_vectorstore_with_document(self, document_path: str) -> bool:
        """Process a new document and add it to the vectorstore"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.document_loaders import PyPDFLoader
            
            # Load the document
            loader = PyPDFLoader(document_path)
            documents = loader.load()
            
            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add chunks to vectorstore
            self.vectorstore.add_documents(chunks)
            
            # Save updated vectorstore
            faiss_index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faiss_index")
            os.makedirs(faiss_index_path, exist_ok=True)
            self.vectorstore.save_local(faiss_index_path)
            
            print(f"Document {document_path} added to vectorstore")
            return True
            
        except Exception as e:
            print(f"Error updating vectorstore with document {document_path}: {e}")
            return False 