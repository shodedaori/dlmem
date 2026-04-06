"""
Run generation for a single experiment.

Usage:
  python -m src.cli.run_eval --config configs/experiments/exp_longmemeval_dream_full.yaml
"""

import argparse
import random
import yaml
import numpy as np
import torch

from src.models.dream_model import DreamModel
from src.memory.full_context import FullContextMemory
from src.benchmarks.longmemeval.adapter import LongMemEvalBenchmark
from src.core.runner import Runner


_MODEL_REGISTRY = {
    "dream": DreamModel,
}

_MEMORY_REGISTRY = {
    "full_context": FullContextMemory,
}

_BENCHMARK_REGISTRY = {
    "longmemeval": LongMemEvalBenchmark,
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _set_seed(cfg.get("runner", {}).get("seed", 42))

    # --- model ---
    model_cfg = cfg["model"]
    model_cls = _MODEL_REGISTRY[model_cfg["name"]]
    model_params = {k: v for k, v in model_cfg.items() if k != "name"}
    model = model_cls(**model_params)

    # --- memory ---
    mem_cfg = cfg["memory"]
    mem_cls = _MEMORY_REGISTRY[mem_cfg["name"]]
    mem_params = {k: v for k, v in mem_cfg.items() if k != "name"}
    memory = mem_cls(**mem_params)

    # --- benchmark ---
    bench_cfg = cfg["benchmark"]
    bench_cls = _BENCHMARK_REGISTRY[bench_cfg["name"]]
    bench_params = {k: v for k, v in bench_cfg.items() if k != "name"}
    benchmark = bench_cls(**bench_params)

    # --- runner ---
    runner_cfg = cfg.get("runner", {})
    output_dir = runner_cfg.get("output_dir", f"outputs/{cfg.get('experiment_name', 'run')}")

    runner = Runner(
        model=model,
        memory=memory,
        benchmark=benchmark,
        output_dir=output_dir,
        save_traces=runner_cfg.get("save_traces", True),
    )
    runner.run()


if __name__ == "__main__":
    main()
