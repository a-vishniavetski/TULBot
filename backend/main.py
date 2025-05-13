from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import numpy as np
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Filter, FieldCondition, MatchValue
# from huggingface_hub import hf_hub_download
# from llama_cpp import Llama

app = FastAPI(title="RAG Backend")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Sentence Transformer for vectorization
# You can replace with other embedding models as needed
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
# Replace with your actual Qdrant cloud instance details or local instance
# qdrant_client = QdrantClient(
#     url="https://your-qdrant-cluster-url.qdrant.io",  # Replace with your actual URL
#     api_key="your-api-key"  # Replace with your actual API key
# )
# COLLECTION_NAME = "your_collection_name"  # Replace with your collection name

# # Initialize Llama model
# # Replace with your preferred Llama model from HuggingFace
# MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGUF"  # Replace with your preferred model
# MODEL_BASENAME = "llama-2-7b-chat.q4_K_M.gguf"  # Replace with the specific weights file
# MODEL_PATH = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_BASENAME)

# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=4096,  # Context window size
#     n_gpu_layers=-1  # Use all available GPU layers, set to 0 for CPU only
# )

# Request and response models
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
        
#         # Step 3: Construct prompt for Llama
#         context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" 
#                              for i, doc in enumerate(retrieved_documents)])
        
#         prompt = f"""You are a helpful assistant that answers questions based on the provided context.
# Context information is below:
# {context}

# Given the context information and not prior knowledge, answer the following question:
# {request.query}

# Answer:"""
        
#         # Step 4: Get response from Llama
#         llama_response = llm(
#             prompt,
#             max_tokens=1024,
#             stop=["</s>", "Human:"],
#             echo=False
#         )
        
#         answer = llama_response["choices"][0]["text"].strip()
        
#         # Step 5: Return response to user
        answer = "Up and running"
        retrieved_documents = [
            {
                "content": "Sample document content",
                "metadata": {"source": "sample_source"},
                "score": 0.95
            }
        ]
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