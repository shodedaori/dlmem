from src.core.types import DialogueTurn
from src.memory.base_memory import BaseMemory


class FullContextMemory(BaseMemory):
    """
    Concatenates the entire dialogue history and returns it as context.
    This is the simplest possible memory baseline.
    """

    name = "full_context"

    def __init__(self, max_turns: int | None = None):
        self.max_turns = max_turns
        self._turns: list[DialogueTurn] = []

    def reset(self) -> None:
        self._turns = []

    def update(self, turn: DialogueTurn) -> None:
        self._turns.append(turn)

    def build_context(self, query: str) -> str:
        turns = self._turns
        if self.max_turns is not None:
            turns = turns[-self.max_turns:]

        lines = []
        for turn in turns:
            prefix = "User" if turn.role == "user" else "Assistant"
            date_str = f" [{turn.date}]" if turn.date else ""
            lines.append(f"{prefix}{date_str}: {turn.content}")

        return "\n".join(lines)
