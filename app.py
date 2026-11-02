"""FastAPI backend for the chatbot prototype.

Run:  uvicorn app:app --reload   then open http://localhost:8000
"""
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from providers import Provider, get_provider  # noqa: E402  (load .env first)
from store import ConversationStore  # noqa: E402

BASE_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or (BASE_DIR / "system_prompt.txt").read_text(encoding="utf-8")
store = ConversationStore(max_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "20")))

app = FastAPI(title="AI Chatbot Prototype")

_provider: Provider | None = None


def provider() -> Provider:
    """Create the LLM client on first use, so the app starts even before a key is set."""
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be empty")
        return value.strip()


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)


def sse(event: dict) -> str:
    """Format one Server-Sent Event."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = store.context_for(req.session_id, req.message)

    async def generate():
        parts: list[str] = []
        try:
            async for chunk in provider().stream(SYSTEM_PROMPT, messages):
                parts.append(chunk)
                yield sse({"type": "token", "text": chunk})
        except Exception:
            logger.exception("LLM request failed")
            # The turn is not saved, so the user can simply send the message again.
            yield sse({"type": "error", "message": "Sorry, I couldn't reach the AI service. Please try again."})
            return

        store.save_turn(req.session_id, req.message, "".join(parts))
        yield sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/reset")
async def reset(req: ResetRequest):
    store.reset(req.session_id)
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
