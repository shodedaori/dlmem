"""
Converts runner prediction output to the format expected by the official
LongMemEval evaluator, then calls evaluate_qa.py.

Official evaluator usage:
  python3 evaluate_qa.py gpt-4o <hypothesis_file> <data_file>
  python3 print_qa_metrics.py gpt-4o <hypothesis_file>.log

Reference: https://github.com/xiaowu0162/LongMemEval
"""

import json
import subprocess
from pathlib import Path


def convert_to_hypothesis_file(predictions_path: str, output_path: str) -> Path:
    """
    Convert predictions.jsonl (runner format) to LongMemEval hypothesis JSONL.

    Output format per line:
      {"question_id": "...", "hypothesis": "..."}
    """
    pred_path = Path(predictions_path)
    out_path = Path(output_path)

    with open(pred_path, encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            f_out.write(
                json.dumps({
                    "question_id": record["sample_id"],
                    "hypothesis": record["prediction"],
                }) + "\n"
            )

    print(f"Hypothesis file written to {out_path}")
    return out_path


def run_official_evaluator(
    hypothesis_path: str,
    data_path: str,
    judge_model: str = "gpt-4o",
    evaluator_script: str = "external/LongMemEval/src/qa_eval/evaluate_qa.py",
) -> None:
    """
    Call the official LongMemEval evaluate_qa.py script.

    Requires the official repo to be available at evaluator_script location.
    """
    cmd = [
        "python3", evaluator_script,
        judge_model,
        hypothesis_path,
        data_path,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
