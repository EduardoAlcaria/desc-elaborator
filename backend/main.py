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
MODEL = os.getenv("OLLAMA_MODEL", "mistral")

SYSTEM_PROMPT = """You are an expert writer. When given a short or rough description,
you elaborate it into a clear, detailed, and well-structured version.
Keep the same intent and tone but make it richer, more precise, and more compelling.
Return only the elaborated description, nothing else."""


class DescRequest(BaseModel):
    description: str


@app.post("/elaborate")
async def elaborate(req: DescRequest):
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    payload = {
        "model": MODEL,
        "prompt": req.description,
        "system": SYSTEM_PROMPT,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    return {"result": resp.json()["response"]}


@app.get("/health")
async def health():
    return {"status": "ok"}
