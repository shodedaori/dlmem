from abc import ABC, abstractmethod


class BaseModel(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> tuple[str, dict]:
        """
        Generate a response for the given prompt.

        Returns:
            (prediction, meta) where meta contains optional keys:
              - prompt_tokens: int
              - generation_tokens: int
        """
        raise NotImplementedError
