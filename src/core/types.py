from dataclasses import dataclass, field
from typing import Any


@dataclass
class DialogueTurn:
    role: str   # "user" or "assistant"
    content: str
    date: str | None = None


@dataclass
class DialogueSample:
    sample_id: str
    task_type: str          # e.g. "single-session-user", "multi-session", "temporal-reasoning", etc.
    history: list[list[DialogueTurn]]   # list of sessions, each session is a list of turns
    query: str
    target: str | list[str] | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    sample_id: str
    prediction: str
    reference: str | None
    used_context: str | None        # the context that was fed to the model
    prompt_tokens: int | None
    generation_tokens: int | None
    latency_sec: float | None
    model_name: str
    memory_name: str
    benchmark_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
