"""
Convert runner predictions to LongMemEval hypothesis format and optionally
run the official evaluator.

Usage:
  python -m src.cli.score_eval \\
      --predictions outputs/exp_longmemeval_dream_full/predictions.jsonl \\
      --benchmark longmemeval \\
      --data data/longmemeval_oracle.json \\
      [--run-official] [--judge-model gpt-4o]
"""

import argparse
from pathlib import Path

from src.benchmarks.longmemeval.evaluator import (
    convert_to_hypothesis_file,
    run_official_evaluator,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--benchmark", required=True, choices=["longmemeval"])
    parser.add_argument("--data", required=True, help="Path to benchmark data file")
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Call the official evaluate_qa.py script after converting",
    )
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument(
        "--evaluator-script",
        default="external/LongMemEval/src/qa_eval/evaluate_qa.py",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    hyp_path = pred_path.parent / "hypothesis.jsonl"

    if args.benchmark == "longmemeval":
        convert_to_hypothesis_file(str(pred_path), str(hyp_path))
        if args.run_official:
            run_official_evaluator(
                hypothesis_path=str(hyp_path),
                data_path=args.data,
                judge_model=args.judge_model,
                evaluator_script=args.evaluator_script,
            )
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
