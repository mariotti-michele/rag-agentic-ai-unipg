from collections import deque

class ConversationMemory:
    def __init__(self, max_turns: int = 4):
        self.history = deque(maxlen=max_turns)

    def add_turn(self, user: str, assistant: str):
        self.history.append({"user": user, "assistant": assistant})

    def get_context(self) -> str:
        if not self.history:
            return ""
        out = "[Conversazione precedente]:\n"
        for t in self.history:
            out += f"Utente: {t['user']}\nAssistente: {t['assistant']}\n"
        out += "[Fine conversazione precedente]\n"
        return out
        
