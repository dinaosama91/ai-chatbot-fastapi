"""Tests run against a fake LLM provider, so no API key or network is needed."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as chatbot  # noqa: E402
from store import ConversationStore  # noqa: E402


class FakeProvider:
    """Streams a fixed reply and records what it was sent."""

    def __init__(self, reply="Hello there!", fail=False):
        self.reply = reply
        self.fail = fail
        self.calls: list[list[dict]] = []

    async def stream(self, system, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("simulated API outage")
        for word in self.reply.split(" "):
            yield word + " "


def parse_events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.split("\n\n") if line.startswith("data: ")]


@pytest.fixture
def fake(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(chatbot, "_provider", provider)
    monkeypatch.setattr(chatbot, "store", ConversationStore(max_messages=20))
    return provider


@pytest.fixture
def client():
    return TestClient(chatbot.app)


def chat(client, message, session="s1"):
    response = client.post("/api/chat", json={"session_id": session, "message": message})
    return response, parse_events(response.text)


def test_streams_reply_then_done(client, fake):
    response, events = chat(client, "Hi")
    assert response.status_code == 200
    assert "".join(e["text"] for e in events if e["type"] == "token").strip() == "Hello there!"
    assert events[-1] == {"type": "done"}


def test_second_message_includes_conversation_history(client, fake):
    chat(client, "My name is Dina")
    chat(client, "What is my name?")
    sent = fake.calls[-1]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[0]["content"] == "My name is Dina"
    assert sent[-1]["content"] == "What is my name?"


def test_sessions_are_isolated(client, fake):
    chat(client, "Secret for session one", session="one")
    chat(client, "Hello", session="two")
    assert fake.calls[-1] == [{"role": "user", "content": "Hello"}]


def test_reset_clears_history(client, fake):
    chat(client, "Remember this")
    assert client.post("/api/reset", json={"session_id": "s1"}).json() == {"status": "ok"}
    chat(client, "Fresh start")
    assert fake.calls[-1] == [{"role": "user", "content": "Fresh start"}]


def test_provider_failure_returns_error_and_does_not_save_turn(client, fake):
    fake.fail = True
    _, events = chat(client, "This will fail")
    assert events[-1]["type"] == "error"

    fake.fail = False
    chat(client, "Try again")
    assert fake.calls[-1] == [{"role": "user", "content": "Try again"}]


@pytest.mark.parametrize("message", ["", "   "])
def test_empty_message_is_rejected(client, fake, message):
    response = client.post("/api/chat", json={"session_id": "s1", "message": message})
    assert response.status_code == 422
    assert fake.calls == []


def test_too_long_message_is_rejected(client, fake):
    response = client.post("/api/chat", json={"session_id": "s1", "message": "x" * 4001})
    assert response.status_code == 422


def test_history_is_trimmed_and_starts_with_user():
    store = ConversationStore(max_messages=4)
    for i in range(5):
        store.save_turn("s", f"question {i}", f"answer {i}")
    history = store.history("s")
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "question 3"}

    context = store.context_for("s", "new question")
    assert len(context) <= 4
    assert context[0]["role"] == "user"
    assert context[-1] == {"role": "user", "content": "new question"}


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Chatbot" in response.text
