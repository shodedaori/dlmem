"""
LongMemEval benchmark adapter.

Dataset format (each item in the JSON file):
  question_id       : str
  question_type     : str  (single-session-user | multi-session | temporal-reasoning |
                             knowledge-update | abstention)
  question          : str
  answer            : str
  question_date     : str
  haystack_session_ids : list[str]
  haystack_dates    : list[str]
  haystack_sessions : list[list[dict]]  # each session = list of {role, content} turns
  answer_session_ids: list[str]

Reference: https://github.com/xiaowu0162/LongMemEval
"""

import json
from pathlib import Path

from src.benchmarks.base_benchmark import BaseBenchmark
from src.core.types import DialogueSample, DialogueTurn


_PROMPT_TEMPLATE = """\
The following is a conversation history between a user and an assistant.
{context}

Based on the conversation history above, answer the following question concisely.
Question: {query}
Answer:"""


class LongMemEvalBenchmark(BaseBenchmark):
    name = "longmemeval"

    def __init__(self, data_path: str, split: str = "oracle"):
        """
        Args:
            data_path: path to the JSON file, e.g. data/longmemeval_oracle.json
            split: informational label (oracle | s | m), not used for loading
        """
        self.data_path = Path(data_path)
        self.split = split

    def load_samples(self) -> list[DialogueSample]:
        with open(self.data_path, encoding="utf-8") as f:
            raw = json.load(f)

        samples = []
        for item in raw:
            sessions: list[list[DialogueTurn]] = []
            for session_turns, date in zip(
                item["haystack_sessions"], item["haystack_dates"]
            ):
                turns = [
                    DialogueTurn(
                        role=t["role"],
                        content=t["content"],
                        date=date,
                    )
                    for t in session_turns
                ]
                sessions.append(turns)

            samples.append(
                DialogueSample(
                    sample_id=item["question_id"],
                    task_type=item["question_type"],
                    history=sessions,
                    query=item["question"],
                    target=item.get("answer"),
                    metadata={
                        "question_date": item.get("question_date"),
                        "answer_session_ids": item.get("answer_session_ids", []),
                    },
                )
            )
        return samples

    def build_prompt(self, sample: DialogueSample, context: str) -> str:
        return _PROMPT_TEMPLATE.format(
            context=context if context else "(no history available)",
            query=sample.query,
        )
