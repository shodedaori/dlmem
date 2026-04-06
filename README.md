# dlmem

Unified experimental framework for multi-turn dialogue memory / state compression research. Supports multiple models (DLMs and AR baselines), multiple memory strategies, and multiple dialogue benchmarks under a shared runner.

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.5.1, and a GPU with at least 20 GB VRAM for Dream.

### 2. Download benchmark data

```bash
mkdir -p data
# LongMemEval (oracle split — evidence sessions only, fastest to run)
wget -P data/ https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json

# Full splits (optional)
# wget -P data/ .../longmemeval_s.json
# wget -P data/ .../longmemeval_m.json
```

### 3. Run generation

```bash
python -m src.cli.run_eval \
    --config configs/experiments/exp_longmemeval_dream_full.yaml
```

Predictions are saved to `outputs/exp_longmemeval_dream_full/predictions.jsonl`.

### 4. Score predictions

```bash
# Convert to LongMemEval hypothesis format
python -m src.cli.score_eval \
    --predictions outputs/exp_longmemeval_dream_full/predictions.jsonl \
    --benchmark longmemeval \
    --data data/longmemeval_oracle.json

# Also call the official GPT-4o judge (requires OPENAI_API_KEY)
python -m src.cli.score_eval \
    --predictions outputs/exp_longmemeval_dream_full/predictions.jsonl \
    --benchmark longmemeval \
    --data data/longmemeval_oracle.json \
    --run-official \
    --judge-model gpt-4o
```

The official scorer lives in the LongMemEval repo. Clone it into `external/`:

```bash
git clone https://github.com/xiaowu0162/LongMemEval external/LongMemEval
```

---

## Repository structure

```
dlmem/
├── configs/
│   ├── benchmarks/        # per-benchmark data configs
│   ├── experiments/       # full experiment configs (model + memory + benchmark)
│   ├── memory/            # memory strategy configs
│   └── models/            # model configs
├── src/
│   ├── benchmarks/
│   │   └── longmemeval/
│   │       ├── adapter.py    # loads JSON → DialogueSample
│   │       └── evaluator.py  # converts predictions → hypothesis.jsonl
│   ├── cli/
│   │   ├── run_eval.py    # generation entrypoint
│   │   └── score_eval.py  # scoring entrypoint
│   ├── core/
│   │   ├── runner.py      # main experiment loop
│   │   └── types.py       # DialogueSample, PredictionRecord, …
│   ├── memory/
│   │   └── full_context.py   # FullContextMemory baseline
│   ├── models/
│   │   └── dream_model.py    # Dream-v0-Instruct-7B wrapper
│   └── utils/
│       └── io.py
├── data/                  # benchmark data (gitignored)
└── outputs/               # prediction and metric files (gitignored)
```

---

## Experiment configuration

Each experiment is a single YAML file that specifies model, memory strategy, and benchmark:

```yaml
# configs/experiments/exp_longmemeval_dream_full.yaml
experiment_name: exp_longmemeval_dream_full

model:
  name: dream
  model_name_or_path: Dream-org/Dream-v0-Instruct-7B
  max_new_tokens: 512
  steps: 512
  temperature: 0.2
  top_p: 0.95
  alg: entropy          # remasking strategy: entropy | confidence | random

memory:
  name: full_context    # concatenate all history sessions as context

benchmark:
  name: longmemeval
  data_path: data/longmemeval_oracle.json
  split: oracle

runner:
  seed: 42
  save_traces: true     # saves used_context per sample for later analysis
  output_dir: outputs/exp_longmemeval_dream_full
```

---

## Supported components

### Models

| Key | Description |
|---|---|
| `dream` | [Dream-v0-Instruct-7B](https://github.com/DreamLM/Dream) — diffusion language model |

### Memory strategies

| Key | Description |
|---|---|
| `full_context` | Concatenate all history turns as plain text context |

### Benchmarks

| Key | Description |
|---|---|
| `longmemeval` | [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — long-term memory QA across multi-session dialogues |

---

## Output format

`predictions.jsonl` — one JSON object per sample:

```json
{
  "sample_id": "...",
  "prediction": "...",
  "reference": "...",
  "used_context": "...",
  "prompt_tokens": 4096,
  "generation_tokens": 64,
  "latency_sec": 12.3,
  "model_name": "dream",
  "memory_name": "full_context",
  "benchmark_name": "longmemeval"
}
```

`hypothesis.jsonl` — converted for the official LongMemEval scorer:

```json
{"question_id": "...", "hypothesis": "..."}
```

---

## Roadmap

Following the macro-plan in [plan/macro-plan.md](plan/macro-plan.md):

- [x] Phase 1 — minimal viable framework (Dream + FullContextMemory + LongMemEval)
- [ ] Phase 2 — add LoCoMo benchmark; add AR baseline model
- [ ] Phase 3 — add MultiChallenge and ES-MemEval
- [ ] Memory strategies — NoMemory, RetrievalMemory, StateCompression
