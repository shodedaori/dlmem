import json
import time
from pathlib import Path

from src.core.types import PredictionRecord
from src.models.base_model import BaseModel
from src.memory.base_memory import BaseMemory
from src.benchmarks.base_benchmark import BaseBenchmark


class Runner:
    def __init__(
        self,
        model: BaseModel,
        memory: BaseMemory,
        benchmark: BaseBenchmark,
        output_dir: str,
        save_traces: bool = True,
    ):
        self.model = model
        self.memory = memory
        self.benchmark = benchmark
        self.output_dir = Path(output_dir)
        self.save_traces = save_traces

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        samples = self.benchmark.load_samples()
        pred_path = self.output_dir / "predictions.jsonl"

        with open(pred_path, "w", encoding="utf-8") as f:
            for sample in samples:
                self.memory.reset()
                for session in sample.history:
                    for turn in session:
                        self.memory.update(turn)

                context = self.memory.build_context(sample.query)
                prompt = self.benchmark.build_prompt(sample, context)

                t0 = time.time()
                prediction, gen_meta = self.model.generate(prompt)
                latency = time.time() - t0

                record = PredictionRecord(
                    sample_id=sample.sample_id,
                    prediction=prediction,
                    reference=sample.target if isinstance(sample.target, str) else None,
                    used_context=context if self.save_traces else None,
                    prompt_tokens=gen_meta.get("prompt_tokens"),
                    generation_tokens=gen_meta.get("generation_tokens"),
                    latency_sec=round(latency, 3),
                    model_name=self.model.name,
                    memory_name=self.memory.name,
                    benchmark_name=self.benchmark.name,
                    metadata=sample.metadata,
                )
                f.write(json.dumps(record.__dict__) + "\n")
                f.flush()

        print(f"Predictions saved to {pred_path}")
        return pred_path
