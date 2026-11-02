"""In-memory conversation history, one list of messages per session."""


class ConversationStore:
    def __init__(self, max_messages: int = 20):
        # Keep an even number so trimming removes whole user/assistant turns.
        self.max_messages = max(2, max_messages - max_messages % 2)
        self._conversations: dict[str, list[dict]] = {}

    def history(self, session_id: str) -> list[dict]:
        return list(self._conversations.get(session_id, []))

    def context_for(self, session_id: str, user_text: str) -> list[dict]:
        """The messages to send to the model: recent history plus the new user message."""
        messages = self.history(session_id) + [{"role": "user", "content": user_text}]
        return self._trim(messages)

    def save_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Store a completed exchange. Called only after the reply finished successfully."""
        messages = self.history(session_id) + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        self._conversations[session_id] = self._trim(messages)

    def reset(self, session_id: str) -> None:
        self._conversations.pop(session_id, None)

    def _trim(self, messages: list[dict]) -> list[dict]:
        """Keep the most recent messages, always starting with a user message."""
        trimmed = messages[-self.max_messages:]
        while trimmed and trimmed[0]["role"] != "user":
            trimmed = trimmed[1:]
        return trimmed
