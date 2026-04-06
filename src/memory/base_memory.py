from abc import ABC, abstractmethod

from src.core.types import DialogueTurn


class BaseMemory(ABC):
    name: str

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, turn: DialogueTurn) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_context(self, query: str) -> str:
        """Return a string context to prepend (or append) to the query prompt."""
        raise NotImplementedError
