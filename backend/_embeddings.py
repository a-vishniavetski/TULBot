from fastapi import HTTPException
import httpx
import os
from typing import List, Union

async def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text"""
    if not os.environ.get("OLLAMA_BASE_URL"):
        os.environ["OLLAMA_BASE_URL"] = "http://ollama:11434"
    if not os.environ.get("EMBEDDINGS_MODEL"):
        os.environ["EMBEDDINGS_MODEL"] = "bge-m3"

    OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
    MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "bge-m3")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "model": MODEL_NAME,
            "prompt": text
        }
        
        response = await client.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("embedding", [])
        else:
            raise HTTPException(status_code=500, detail="Failed to get embedding")