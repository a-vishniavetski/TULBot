import os
import getpass
import sys
import dotenv
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import httpx
import numpy as np
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from huggingface_hub import hf_hub_download, login
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel
# from llama_cpp import Llama
from qdrant_client.grpc import SearchParams
import torch
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel

sys.path.append(
    os.path.dirname(  # Contains backend and qdrant
        os.path.dirname(  # /backend
            os.path.abspath(__file__))))  # backend/main.py

# from qdrant.importing_data_to_qdrant import get_embeddings
from prompts import PROMPT_BASE

import language_model


# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Backend")
dotenv.load_dotenv()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize Sentence Transformer for vectorization
# You can replace with other embedding models as needed
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
# Replace with your actual Qdrant cloud instance details or local instance
# client = QdrantClient(
#     url="https://bc99e4de-626a-4844-bd3c-ddadabe39525.europe-west3-0.gcp.cloud.qdrant.io:6333",  # Replace with your actual URL
#     api_key = os.getenv('qdrant_api_key')
# )

COLLECTINO_NAME = "subjects"

local_model = os.getenv('local_model', 'false').lower() == 'true'
# if not local_model:
# # Initialize Llama model
#     hf_token = os.getenv('hf_token')
#     if hf_token is None:
#         raise ValueError("Hugging Face token not found in environment variables.")
#     login(token=hf_token)

#     MODEL_ID = "TLLMDH/1b_test"  # Replace with your preferred model
#     MODEL_BASENAME = "llama-2-7b-chat.q4_K_M.gguf"  # Replace with the specific weights file
#     # MODEL_PATH = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_BASENAME)

#     # Aleksiej's tokenizers
#     tokenizer = AutoTokenizer.from_pretrained("eryk-mazus/polka-1.1b-chat")
#     model = AutoModelForCausalLM.from_pretrained("eryk-mazus/polka-1.1b-chat")

#Agata's tokenizers
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# model = AutoModel.from_pretrained("bert-base-multilingual-cased")

# Set proper generation configuration
# generation_config = GenerationConfig(
#     max_new_tokens=128,  # Controls the maximum length of the generation
#     do_sample=True,      # Enable sampling
#     temperature=0.65,     # Control randomness
#     top_p=0.9,           # Nucleus sampling
#     top_k=10,            # Top-k sampling
#     repetition_penalty=1.4,  # Penalize repetition
#     pad_token_id=tokenizer.eos_token_id  # Properly set pad token
# )

# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=4096,  # Context window size
#     n_gpu_layers=-1  # Use all available GPU layers, set to 0 for CPU only
# )

# Request and response models
class QueryRequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = 5  # Number of results to retrieve from Qdrant

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[Any, Any]] = []


@app.post("/query_deprecated", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Step 1: Encode the query using your tokenizer
        query_text = request.query

        # Get the [CLS] token vector (query vector)
        # query_vector = await get_embeddings(query_text)  # embedding function, returns e.g. 768-d vector

        # Step 2: Search Qdrant using the encoded query vector
        # search_result = client.search(
        #     collection_name=request.collection_name,
        #     query_vector=query_vector,
        #     limit=request.top_k,  # Use the value from the request
        #     search_params=SearchParams(hnsw_ef=128, exact=False)
        # )

        # Step 3: Extract and format the retrieved documents
        retrieved_documents = []

        # if request.collection_name == "majors":
        #     for i, result in enumerate(search_result, 1):
        #         payload = result.payload
        #         retrieved_documents.append({
        #             "content": payload.get("description", ""),
        #             "metadata": {
        #                 "major_name": payload.get("major"),
        #             },
        #             "score": result.score
        #         })

        # elif request.collection_name == "subjects":
        #     for i, result in enumerate(search_result, 1):
        #         payload = result.payload
        #         retrieved_documents.append({
        #             "content": payload.get("subject_content", ""),
        #             "metadata": {
        #                 "subject_name": payload.get("subject_name"),
        #                 "subject_id": payload.get("subject_id"),
        #                 "specialisation": payload.get("specialisation")
        #             },
        #             "score": result.score
        #         })

        # else:
        #     raise HTTPException(status_code=400, detail=f"Unsupported collection: {request.collection_name}")

        context = ""
        # context = "\n\n".join([
        #     f"Document {i + 1}:\n{doc['content']}" for i, doc in enumerate(retrieved_documents)
        # ])

        # prompt = PROMPT_BASE.format(context=context, query=request.query)
        # inputs = tokenizer(prompt, return_tensors="pt")

        # Generate response from the model
        logging.info(f"Generating response for prompt...")
        # outputs = model.generate(**inputs, generation_config=generation_config)
        # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not local_model:
            answer = language_model.get_asnwer_from_groq(PROMPT_BASE, request.query)

        # Clean up the answer to avoid the prompt being included
        # if prompt in answer:
        #     answer = answer.split(prompt)[-1].strip()

        # Step 5: Return the response to the user
        print(f"=== DEBUG ===")
        print(f"Query: {request.query}\n===")
        print(f"RAG data: {retrieved_documents}\n===")
        print(f"Prompt: {PROMPT_BASE}===")
        print(f"Model answer: {answer}===")
        return QueryResponse(
            answer=answer,
            sources=retrieved_documents
        )


    except Exception as e:
        # raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
        logging.error(f"Error processing query: {str(e)}")



@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    from _langchain import get_langchain_service, LangChainService

    _langchain_service: LangChainService = get_langchain_service()

    input_message = request.query
    messages = [{"role": "user", "content": input_message}]

    try:
        response = await _langchain_service.process_query(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    return QueryResponse(
        answer=response["messages"][-1].content,
        sources=[{"content": "Dummy source content", "metadata": {"source": "dummy_source"}}]
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)