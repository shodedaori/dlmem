# plan.md

## Project Goal

Build a unified experimental framework for **multi-turn dialogue memory / state compression research** that supports:

- multiple models, including multiple DLMs and AR baselines
- a pluggable generation interface for each model
- multiple memory/state strategies
- multiple dialogue benchmarks under a shared runner
- unified prediction, scoring, logging, and trace analysis

The core principle is:

> Use **one main project** with **benchmark adapters**, instead of building one fully separate project per benchmark.

This keeps model wrapping, memory logic, logging, configs, and experiment outputs consistent across tasks.

---

## High-Level Design Choice

### Recommended
Use a **monorepo + plugin-style benchmark adapters**.

That means:

- one shared `core` for model interfaces, memory interfaces, runners, configs, logging, and outputs
- one `benchmarks/` submodule per benchmark
- one unified experiment entrypoint
- one shared prediction format
- one shared result summary pipeline

### Not Recommended
Do **not** build a completely separate full project for each benchmark unless:

1. the benchmark has very heavy and incompatible dependencies
2. the benchmark requires a very special official evaluation environment
3. the task is fundamentally different from the rest

For the current target benchmarks such as:

- LongMemEval
- LoCoMo
- MultiChallenge
- ES-MemEval

a unified project is the better design.

---

## Why a Unified Project is Better

If each benchmark becomes its own standalone project, the following problems appear quickly:

- repeated implementation of model loading and generation
- repeated implementation of memory/state logic
- inconsistent prompt formatting
- inconsistent output logging
- inconsistent metric files
- poor comparability across experiments
- more glue code, less real research progress

A shared framework is more appropriate because the real research object is:

> comparing **methods** across **benchmarks**

rather than merely reproducing each benchmark independently.

---

## Core Research Axes

The framework should make it easy to compare along three axes:

### 1. Model axis
Examples:

- AR baseline models
- DLM model A
- DLM model B
- future models

### 2. Memory / state axis
Examples:

- no memory
- full history
- retrieval memory
- compressed state
- oracle memory

### 3. Benchmark axis
Examples:

- LongMemEval
- LoCoMo
- MultiChallenge
- ES-MemEval

The project should support experiments that are essentially the Cartesian product:

\[
(\text{model}, \text{memory}, \text{benchmark})
\]

---

## Directory Structure

```text
memory_dialogue_project/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── models/
│   │   ├── llada.yaml
│   │   ├── mdlm.yaml
│   │   ├── baseline_ar.yaml
│   │   └── qwen_instruct.yaml
│   ├── memory/
│   │   ├── no_memory.yaml
│   │   ├── full_context.yaml
│   │   ├── retrieval_memory.yaml
│   │   ├── state_compression.yaml
│   │   └── oracle_memory.yaml
│   ├── benchmarks/
│   │   ├── longmemeval.yaml
│   │   ├── locomo.yaml
│   │   ├── multichallenge.yaml
│   │   └── es_memeval.yaml
│   └── experiments/
│       ├── exp_longmemeval_llada_state.yaml
│       └── ...
├── src/
│   ├── core/
│   │   ├── types.py
│   │   ├── registry.py
│   │   ├── runner.py
│   │   ├── generation.py
│   │   └── metrics.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── ar_model.py
│   │   ├── dlm_model.py
│   │   ├── llada_model.py
│   │   └── mdlm_model.py
│   ├── memory/
│   │   ├── base_memory.py
│   │   ├── no_memory.py
│   │   ├── full_context.py
│   │   ├── retrieval_memory.py
│   │   ├── state_memory.py
│   │   └── oracle_memory.py
│   ├── prompts/
│   │   ├── qa_prompts.py
│   │   ├── summary_prompts.py
│   │   └── dialogue_prompts.py
│   ├── benchmarks/
│   │   ├── base_benchmark.py
│   │   ├── longmemeval/
│   │   │   ├── adapter.py
│   │   │   ├── evaluator.py
│   │   │   └── parser.py
│   │   ├── locomo/
│   │   │   ├── adapter.py
│   │   │   ├── evaluator.py
│   │   │   └── parser.py
│   │   ├── multichallenge/
│   │   │   ├── adapter.py
│   │   │   ├── evaluator.py
│   │   │   └── parser.py
│   │   └── es_memeval/
│   │       ├── adapter.py
│   │       ├── evaluator.py
│   │       └── parser.py
│   ├── utils/
│   │   ├── io.py
│   │   ├── jsonl.py
│   │   ├── seed.py
│   │   └── logging.py
│   └── cli/
│       ├── run_eval.py
│       ├── score_eval.py
│       └── summarize_results.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   └── traces/
└── scripts/
    ├── run_all.sh
    ├── run_ablation.sh
    └── collect_tables.py
```

---

## Three Critical Unified Interfaces

## 1. Unified Model Interface

Every model should expose the same generation API, regardless of whether it is AR or DLM.

Example:

```python
class BaseModel:
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
```

### Key rule
A benchmark should **never** know the internal sampling details of a model.

So even if a DLM internally uses:

- iterative denoising
- remasking
- multiple refinement passes
- custom decoding schedule

the benchmark should still only call something like:

```python
response = model.generate(compiled_input, generation_config)
```

That way, all model-specific complexity stays inside the model wrapper.

---

## 2. Unified Memory / State Interface

This is central to the research.

Suggested API:

```python
class BaseMemory:
    def reset(self): ...
    def update(self, turn): ...
    def build_context(self, query): ...
```

### Example implementations

#### `NoMemory`
- ignores history
- only uses current query

#### `FullContextMemory`
- directly returns all prior turns

#### `RetrievalMemory`
- retrieves top-k relevant items from history or external store

#### `StateMemory`
- maintains a compressed latent / symbolic / textual state across turns

#### `OracleMemory`
- uses benchmark annotations or gold information to estimate an upper bound

### Why this matters
The project is not just comparing models. It is comparing:

- different ways to represent dialogue history
- different memory write/read policies
- different compression strategies

---

## 3. Unified Benchmark Interface

Each benchmark should convert its own raw format into a common schema.

Suggested schema:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class DialogueSample:
    sample_id: str
    task_type: str          # qa / summary / generation
    history: list[dict]     # [{"role": "user", "text": ...}, ...]
    query: str | None
    target: str | list[str] | None
    metadata: dict[str, Any]
```

### Benefit
The runner should not care whether the sample came from:

- LongMemEval
- LoCoMo
- MultiChallenge
- ES-MemEval

It should always receive the same schema.

---

## Standard Execution Pipeline

The recommended execution pipeline is:

```python
sample
-> memory.build_context(sample)
-> prompt_builder(...)
-> model.generate(...)
-> evaluator.score(...)
```

### Responsibility split

#### Benchmark adapter
- loads raw data
- converts raw sample to unified schema

#### Memory module
- manages history, retrieval, compression, or state update
- builds model-facing context

#### Prompt builder
- turns sample + memory output into prompt or model input format

#### Model wrapper
- runs generation using model-specific logic

#### Evaluator
- scores prediction for the target benchmark

This separation is important for keeping the project clean and extensible.

---

## Prediction and Scoring Should Be Separated

This is one of the most important engineering decisions.

### Stage 1: Generate predictions
Run the model and save outputs in a standard format.

Example JSONL:

```json
{"sample_id": "xxx", "prediction": "...", "trace": {...}}
```

### Stage 2: Score predictions
Read saved prediction files and compute benchmark metrics.

### Advantages

- can rescore without regenerating
- can update judge prompts without rerunning the model
- easier checkpointing and resume
- easier integration with official benchmark evaluators
- better reproducibility

---

## What to Log for Each Sample

Since this project studies memory/state mechanisms, traces are extremely important.

Recommended per-sample record:

```json
{
  "sample_id": "...",
  "prediction": "...",
  "reference": "...",
  "used_memory": [...],
  "memory_state_summary": "...",
  "prompt_tokens": 1234,
  "generation_tokens": 87,
  "latency_sec": 3.21,
  "model_name": "llada",
  "memory_name": "state_compression",
  "benchmark_name": "longmemeval"
}
```

### Why traces matter

They help answer questions such as:

- what memory was actually retrieved?
- what information was kept inside the compressed state?
- why did the model fail?
- how many tokens were saved?
- how much latency was reduced?

Without detailed traces, later analysis and paper writing become much harder.

---

## Benchmark Integration Strategy

Do not fully merge every official benchmark repository into the main codebase.

### Recommended strategy
Use the official repositories as references for:

- data format
- scoring rules
- evaluation protocol

But maintain your own:

- adapter
- parser
- evaluator wrapper
- unified sample conversion

### Exception
If a benchmark has a highly specialized official evaluator, add a lightweight `external/` integration layer that consumes your standard prediction files.

### Good compromise
- main project handles generation and storage
- optional external scripts handle special official scoring if needed

---

## Suggested Initial Scope

Do not build everything at once.

### Phase 1: Minimal viable framework
Implement:

- unified model API
- unified memory API
- unified benchmark sample API
- shared runner
- shared prediction format
- shared scoring entrypoint

### Phase 2: First benchmark pair
Integrate:

- LongMemEval
- LoCoMo

These are enough to validate:
- long-term memory correctness
- long-session event reasoning
- multi-session memory behavior

### Phase 3: Response-quality expansion
Add:

- MultiChallenge
- ES-MemEval

These are more useful for:
- final response quality
- self-coherence
- instruction retention
- user modeling
- personalization

### Phase 4: Official-evaluator compatibility
Only if needed:
- build small wrappers for official benchmark scripts
- keep them outside the core abstraction layer

---

## Suggested Initial Experiment Matrix

Start with a small but meaningful experiment matrix.

### Models
- 1 AR baseline
- 1 or 2 DLMs
- your own model variant if available

### Memory settings
- no memory
- full context
- retrieval memory
- state compression

### Benchmarks
- LongMemEval
- LoCoMo
- MultiChallenge

This already creates a strong first paper-style experiment setup.

---

## Recommended File Responsibilities

## `src/models/base_model.py`
Defines the model abstraction.

Should include:
- model loading API
- generation API
- optional batch generation API
- generation config handling

## `src/memory/base_memory.py`
Defines the memory abstraction.

Should include:
- reset
- update
- build_context
- optional serialize_state / deserialize_state

## `src/benchmarks/base_benchmark.py`
Defines benchmark abstraction.

Should include:
- dataset loading
- sample conversion
- evaluation entrypoint

## `src/core/runner.py`
Main experiment controller.

Should:
- load config
- instantiate model, memory, benchmark
- loop through samples
- save predictions
- save traces
- trigger evaluator

## `src/cli/run_eval.py`
Command-line entry for generation.

Example:
```bash
python -m src.cli.run_eval --config configs/experiments/exp_longmemeval_llada_state.yaml
```

## `src/cli/score_eval.py`
Command-line entry for metric computation.

Example:
```bash
python -m src.cli.score_eval --pred outputs/predictions/...
```

## `src/cli/summarize_results.py`
Collects metrics from multiple runs and generates tables for analysis.

---

## Prompt Design Guidance

Because benchmarks contain different task types, prompt construction should not be mixed into the benchmark core.

Instead, keep prompts modular by task type:

- QA prompts
- summarization prompts
- final-response generation prompts

### Important rule
A benchmark defines the **task**.
A prompt builder defines the **rendering** of that task for a specific model family.

This is especially important when some DLMs may need a different input structure from AR models.

---

## Configuration Design

Use YAML-based configuration for reproducibility.

A single experiment config should specify:

- model
- tokenizer
- generation hyperparameters
- memory type
- memory hyperparameters
- benchmark
- split
- batch size
- output directory
- seed

### Example experiment config
```yaml
experiment_name: exp_longmemeval_llada_state

model:
  name: llada
  config_path: configs/models/llada.yaml

memory:
  name: state_compression
  config_path: configs/memory/state_compression.yaml

benchmark:
  name: longmemeval
  config_path: configs/benchmarks/longmemeval.yaml

runner:
  batch_size: 1
  save_traces: true
  output_dir: outputs/exp_longmemeval_llada_state
  seed: 42
```

---

## Practical Development Order

A good build order is:

1. implement `BaseModel`
2. implement `BaseMemory`
3. implement `DialogueSample`
4. implement `Runner`
5. implement one simple AR baseline
6. implement one simple memory baseline (`FullContextMemory`)
7. integrate one benchmark (`LongMemEval`)
8. verify end-to-end prediction + scoring
9. add one DLM wrapper
10. add `StateMemory`
11. add second benchmark (`LoCoMo`)
12. add result summarization scripts
13. expand to `MultiChallenge`
14. expand to `ES-MemEval`

This order minimizes engineering risk.

---

## When a Separate Benchmark Subproject Is Reasonable

A benchmark can be partially separated if:

- it requires its own environment
- it requires a special judge model stack
- it has strict official scripts that are hard to port
- it depends on unique external services

In that case, do this:

- keep the main framework for generation
- export predictions in a standard format
- let the benchmark-specific subproject consume those files for official scoring

So even in the split case, the project should still preserve a single unified upstream generation framework.

---

## Recommended Minimal Milestone

The first milestone should be:

### Goal
Run one AR model and one DLM on:

- LongMemEval
- LoCoMo

under three memory conditions:

- no memory
- full context
- state compression

and produce:

- prediction files
- metric files
- trace files
- a summary table

If this works, the framework is already strong enough for meaningful ablation studies.

---

## Final Recommendation

The best structure for this research project is:

> **one main experimental framework + benchmark adapters + optional external evaluators**

This is better than maintaining one full standalone project per benchmark because it preserves:

- consistent abstractions
- consistent logging
- consistent prompt handling
- consistent experiment configs
- clean cross-benchmark comparability

For your current research direction, this is both feasible and the right engineering choice.
