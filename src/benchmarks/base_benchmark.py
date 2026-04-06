from abc import ABC, abstractmethod

from src.core.types import DialogueSample


class BaseBenchmark(ABC):
    name: str

    @abstractmethod
    def load_samples(self) -> list[DialogueSample]:
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, sample: DialogueSample, context: str) -> str:
        """
        Build the final string prompt for the model.

        Args:
            sample: the dialogue sample
            context: the string context produced by the memory module

        Returns:
            A single string to pass to model.generate()
        """
        raise NotImplementedError
