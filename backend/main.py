from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


class CompleteRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    temperature: float = 0.3


@app.post("/api/ai/complete")
async def complete(req: CompleteRequest):
    if not req.user_prompt.strip():
        raise HTTPException(status_code=400, detail="user_prompt cannot be empty")

    payload = {
        "model": MODEL,
        "system": req.system_prompt,
        "prompt": req.user_prompt,
        "stream": False,
        "options": {"temperature": req.temperature},
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    return {"text": resp.json()["response"]}


@app.get("/health")
async def health():
    return {"status": "ok"}
