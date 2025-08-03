import os
import sys
import dotenv
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np

sys.path.append(
    os.path.dirname(  # Contains backend and qdrant
        os.path.dirname(  # /backend
            os.path.abspath(__file__))))  # backend/main.py

# from qdrant.importing_data_to_qdrant import get_embeddings
from prompts import PROMPT_BASE

from _langchain import get_langchain_service, LangChainService
from _embeddings import get_embedding


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

# Request and response models
class QueryRequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = 5  # Number of results to retrieve from Qdrant

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[Any, Any]] = []

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    input_message = request.query

    _langchain_service: LangChainService = get_langchain_service()

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


# @app.post("/query_deprecated", response_model=QueryResponse)
# async def process_query(request: QueryRequest):
#     try:
#         # Step 1: Encode the query using your tokenizer
#         query_text = request.query

#         # Get the [CLS] token vector (query vector)
#         # query_vector = await get_embeddings(query_text)  # embedding function, returns e.g. 768-d vector

#         # Step 2: Search Qdrant using the encoded query vector
#         # search_result = client.search(
#         #     collection_name=request.collection_name,
#         #     query_vector=query_vector,
#         #     limit=request.top_k,  # Use the value from the request
#         #     search_params=SearchParams(hnsw_ef=128, exact=False)
#         # )

#         # Step 3: Extract and format the retrieved documents
#         retrieved_documents = []

#         # if request.collection_name == "majors":
#         #     for i, result in enumerate(search_result, 1):
#         #         payload = result.payload
#         #         retrieved_documents.append({
#         #             "content": payload.get("description", ""),
#         #             "metadata": {
#         #                 "major_name": payload.get("major"),
#         #             },
#         #             "score": result.score
#         #         })

#         # elif request.collection_name == "subjects":
#         #     for i, result in enumerate(search_result, 1):
#         #         payload = result.payload
#         #         retrieved_documents.append({
#         #             "content": payload.get("subject_content", ""),
#         #             "metadata": {
#         #                 "subject_name": payload.get("subject_name"),
#         #                 "subject_id": payload.get("subject_id"),
#         #                 "specialisation": payload.get("specialisation")
#         #             },
#         #             "score": result.score
#         #         })

#         # else:
#         #     raise HTTPException(status_code=400, detail=f"Unsupported collection: {request.collection_name}")

#         context = ""
#         # context = "\n\n".join([
#         #     f"Document {i + 1}:\n{doc['content']}" for i, doc in enumerate(retrieved_documents)
#         # ])

#         # prompt = PROMPT_BASE.format(context=context, query=request.query)
#         # inputs = tokenizer(prompt, return_tensors="pt")

#         # Generate response from the model
#         logging.info(f"Generating response for prompt...")
#         # outputs = model.generate(**inputs, generation_config=generation_config)
#         # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         if not local_model:
#             answer = language_model.get_asnwer_from_groq(PROMPT_BASE, request.query)

#         # Clean up the answer to avoid the prompt being included
#         # if prompt in answer:
#         #     answer = answer.split(prompt)[-1].strip()

#         # Step 5: Return the response to the user
#         print(f"=== DEBUG ===")
#         print(f"Query: {request.query}\n===")
#         print(f"RAG data: {retrieved_documents}\n===")
#         print(f"Prompt: {PROMPT_BASE}===")
#         print(f"Model answer: {answer}===")
#         return QueryResponse(
#             answer=answer,
#             sources=retrieved_documents
#         )


#     except Exception as e:
#         # raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
#         logging.error(f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)