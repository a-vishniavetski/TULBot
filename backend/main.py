import os
import dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from prompts import PROMPT_BASE
from language_model import (
    setup_local_language_model, 
    get_answer_from_local, 
    get_asnwer_from_groq
)

# ============================== APP SETTINGS =====================================
app = FastAPI(title="RAG Backend")
dotenv.load_dotenv()  # .env variables
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================== LANGUAGE MODEL SETUP ==============================
is_local_language_model = os.getenv("local_model", "false").lower() == "true"
if is_local_language_model:
    tokenizer, model, generation_config = setup_local_language_model()
else:
    tokenizer, model, generation_config = None, None, None

# ============================== POSSIBLE QDRANT SETUP =============================
# Initialize Qdrant client
# Replace with your actual Qdrant cloud instance details or local instance
# qdrant_client = QdrantClient(
#     url="https://your-qdrant-cluster-url.qdrant.io",  # Replace with your actual URL
#     api_key="your-api-key"  # Replace with your actual API key
# )
# COLLECTION_NAME = "your_collection_name"  # Replace with your collection name

# Initialize Llama model
# Replace with your preferred Llama model from HuggingFace

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5  # Number of results to retrieve from Qdrant

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[Any, Any]] = []

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
#         # Step 1: Vectorize the query
#         query_vector = embedder.encode(request.query).tolist()
        
#         # Step 2: Search Qdrant
#         search_results = qdrant_client.search(
#             collection_name=COLLECTION_NAME,
#             query_vector=query_vector,
#             limit=request.top_k
#         )
        
#         # Extract relevant information from search results
#         retrieved_documents = []
#         for result in search_results:
#             payload = result.payload
#             score = result.score
#             retrieved_documents.append({
#                 "content": payload.get("content", ""),
#                 "metadata": {k: v for k, v in payload.items() if k != "content"},
#                 "score": score
#             })
        
        # context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" 
        #                      for i, doc in enumerate(retrieved_documents)])
        prompt = PROMPT_BASE.format(context="No documents retrieved from RAG.", query=request.query)

        if is_local_language_model and tokenizer and model:
            answer = get_answer_from_local(prompt, request.query, tokenizer, model)
        else:
            answer = get_asnwer_from_groq(prompt, request.query)
        retrieved_documents = [
            {
                "content": "Sample document content",
                "metadata": {"source": "sample_source"},
                "score": 0.95
            }
        ]

        print(f"=== DEBUG ===")
        print(f"Query: {request.query}\n===")
        print(f"RAG data: {retrieved_documents}\n===")
        print(f"Prompt: {prompt}===")
        print(f"Model answer: {answer}===")
        return QueryResponse(
            answer=answer,
            sources=retrieved_documents
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)