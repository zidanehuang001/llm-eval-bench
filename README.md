# llm-eval-bench

Run LLM and VLM benchmarks against any OpenAI-compatible inference endpoint (vLLM, NIM, etc.) with multi-server parallel execution and automatic result aggregation.

## Requirements

```bash
pip install tqdm           # progress bar (optional but recommended)

# Install at least one evaluation tool:
pip install evalscope      # default — supports all LLM + VLM benchmarks
pip install lm-eval        # EleutherAI harness — LLM only
pip install opencompass    # strong Chinese + VLM support
```

## Quick Start

```bash
# LLM benchmarks against a remote vLLM server
python llm_bench.py --host 10.7.2.33 --model qwen2.5-7b

# VLM/multimodal benchmarks
python llm_bench.py --host 10.7.2.33 --model qwen2.5-vl-7b --vlm

# All benchmarks
python llm_bench.py --host 10.7.2.33 --model qwen2.5-vl-7b --all
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--model MODEL` | `step3p5-flash` | Model name passed to the API |
| `--host HOST` | — | vLLM server hostname or IP; constructs `http://HOST:PORT/v1` |
| `--port PORT` | `8000` | vLLM server port (used with `--host`) |
| `--hosts H1,H2` | — | Multiple hosts (`host` or `host:port`); distributes benchmarks across servers |
| `--url URL` | `http://10.7.2.33:8000/v1` | Full API base URL; overridden by `--host`/`--hosts` |
| `--api-key KEY` | `EMPTY` | API key |
| `--tool TOOL` | `evalscope` | Eval framework: `evalscope`, `lm-eval`, `opencompass` |
| `--batch N` | `16` | Default eval batch size (per-benchmark overrides apply) |
| `--timeout N` | `120` | Default per-request timeout in seconds (per-benchmark overrides apply) |
| `--workers N` | `1` | Concurrent benchmarks per host (total = hosts × workers) |
| `--vlm` | — | Run VLM/multimodal benchmarks only |
| `--all` | — | Run all LLM + VLM benchmarks |
| `--benches a,b,c` | — | Override benchmark list (comma-separated) |
| `--resume` | — | Skip benchmarks that already have a `.done` marker |
| `--report` | — | Print result summary table and exit (no benchmarks run) |

All flags can also be set via environment variables: `MODEL`, `BASE_URL`, `API_KEY`, `EVAL_TOOL`, `BATCH_SIZE`, `TIMEOUT`, `VLLM_HOST`, `VLLM_PORT`.

## Startup Sequence

Before any benchmarks run, the script performs these steps automatically:

**1. Host warmup** — each host receives a `/v1/models` health check followed by a minimal generation request (`max_tokens=1`). This catches connectivity issues early and, critically, forces vLLM to load model weights into GPU memory so the first real benchmark request doesn't time out. All hosts are warmed up in parallel.

```
Warming up hosts...
  [WARMUP] 10.6.142.6:8000   ready
  [WARMUP] 10.6.142.22:8000  ready
```

If any host fails warmup the script aborts before running any benchmarks.

**2. Pre-run breakdown** — shows exactly which benchmarks each host will run and which will be skipped (when `--resume` is used), before any work starts:

```
============================================================
  Tool:    evalscope
  Model:   step3p6
  Workers: 1 per host  (2 concurrent max)

  Host 1: 10.6.142.6:8000
    pending (7): aime25, gsm8k, humaneval, mmlu_pro, cmmlu, arc, truthful_qa
  Host 2: 10.6.142.22:8000
    pending (7): hmmt25, math500, mmlu, ceval, hellaswag, simple_qa, ifeval
============================================================
```

**3. Progress bar** — a single `tqdm` bar tracks total completion across all hosts. Log lines (`[START]`, `[PASS]`, `[FAIL]`) print above the bar so it stays pinned at the bottom:

```
  [PASS]  gsm8k      [10.6.142.6:8000]
  [PASS]  mmlu       [10.6.142.22:8000]
 28%|███       | 4/14 [12:03<30:11, gsm8k pass]
```

## Benchmarks

Per-benchmark timeout and batch size are set automatically. The `--timeout` and `--batch` flags act as the fallback for benchmarks not listed below.

### LLM

| Key | Benchmark | What it measures | Timeout | Batch |
|---|---|---|---|---|
| `aime25` | AIME 2025 | Competition math | 600s | 4 |
| `hmmt25` | HMMT 2025 | Competition math | 600s | 4 |
| `math500` | MATH-500 | Hard math problems | 300s | 8 |
| `humaneval` | HumanEval | Python code generation | 300s | 8 |
| `ifeval` | IFEval | Instruction-following | 240s | 16 |
| `gsm8k` | GSM8K | Grade-school math reasoning | 120s | 16 |
| `mmlu` | MMLU | General knowledge (57 subjects) | 120s | 16 |
| `mmlu_pro` | MMLU-Pro | Harder knowledge, 10-choice | 120s | 16 |
| `ceval` | C-Eval | Chinese general knowledge | 120s | 16 |
| `cmmlu` | CMMLU | Chinese language & knowledge | 120s | 16 |
| `hellaswag` | HellaSwag | Commonsense reasoning | 120s | 16 |
| `arc` | ARC-Challenge | Science QA | 120s | 16 |
| `simple_qa` | SimpleQA | Factual short-answer | 120s | 16 |
| `truthful_qa` | TruthfulQA | Factuality / avoiding misconceptions | 120s | 16 |

### VLM / Multimodal

All VLM benchmarks use reduced batch sizes because image tokens consume significantly more GPU memory than text. High-resolution benchmarks (`chartqa`, `docvqa`) and those requiring visual reasoning (`mathvista`, `hallusionbench`) are further constrained.

| Key | Benchmark | What it measures | Timeout | Batch |
|---|---|---|---|---|
| `mathvista` | MathVista | Math with visual context | 360s | 4 |
| `chartqa` | ChartQA | Chart and figure understanding | 300s | 4 |
| `docvqa` | DocVQA | Document VQA (high-res images) | 300s | 4 |
| `hallusionbench` | HallusionBench | Visual hallucination | 300s | 4 |
| `scienceqa` | ScienceQA | Science QA with diagrams | 240s | 8 |
| `mmbench` | MMBench | General multimodal understanding | 180s | 8 |
| `mme` | MME | Perception + cognition | 180s | 8 |
| `mmstar` | MMStar | Multi-modal reasoning | 180s | 8 |
| `seed_bench` | SEED-Bench | Spatial, temporal, egocentric understanding | 180s | 8 |
| `ocrbench` | OCRBench | Text recognition in images | 180s | 8 |
| `textvqa` | TextVQA | Scene-text VQA | 180s | 8 |
| `vqav2` | VQAv2 | General visual QA | 180s | 8 |
| `pope` | POPE | Object hallucination | 180s | 8 |

## Multi-server Distribution

`--hosts` distributes benchmarks across servers using round-robin assignment and runs them concurrently via `ThreadPoolExecutor`. All hosts are warmed up before benchmarks start, output is serialised so lines never interleave.

```bash
# Two servers, default port 8000
python llm_bench.py --hosts 10.7.2.33,10.7.2.34 --model qwen2.5-7b --all

# Mixed ports
python llm_bench.py --hosts 10.7.2.33:8000,10.7.2.34:9000 --model qwen2.5-7b --all

# Two servers, two concurrent benchmarks each (4 total in flight)
python llm_bench.py --hosts 10.7.2.33,10.7.2.34 --model qwen2.5-7b --all --workers 2
```

With 2 servers and `--workers 1` (default): 2 benchmarks run at any time, one per server. When a benchmark finishes the next one on that server starts immediately.

For request-level load balancing (every individual API call distributed across servers) use an nginx upstream instead:

```nginx
upstream vllm {
    server 10.7.2.33:8000;
    server 10.7.2.34:8000;
}
```

Then point `--url http://localhost/v1` at the nginx proxy.

## Examples

```bash
# Custom port
python llm_bench.py --host gpu-node-01 --port 9000 --model llama-3-8b

# Switch eval tool
python llm_bench.py --host 10.7.2.33 --model qwen2.5-7b --tool lm-eval

# Run a specific subset
python llm_bench.py --host 10.7.2.33 --model qwen2.5-7b --benches gsm8k,mmlu,humaneval

# Resume an interrupted run
python llm_bench.py --hosts 10.7.2.33,10.7.2.34 --model qwen2.5-7b --all --resume

# Read results from a previous run (from a second terminal while a run is in progress)
python llm_bench.py --report

# Use environment variables
export VLLM_HOST=10.7.2.33
export MODEL=qwen2.5-vl-7b
export EVAL_TOOL=opencompass
python llm_bench.py --vlm
```

## Output & Results

```
logs/      per-benchmark log files  (<bench>_YYYYMMDD_HHMMSS.log)
outputs/   tool output directories + <bench>.done markers
```

`.done` marker files are written on success. `--resume` skips any benchmark that already has one, so interrupted runs (crash, timeout, Ctrl-C) can be continued without re-running completed benchmarks.

### Reading results

A summary table prints automatically at the end of every run. To read results at any time without running benchmarks — including while a run is still in progress in another terminal:

```bash
python llm_bench.py --report
```

Example output:

```
────────────────────────────────────────────────────────────
Benchmark                Score   Tool         Completed
────────────────────────────────────────────────────────────
aime25                   42.0%   evalscope    2026-04-29 14:32
gsm8k                    91.3%   evalscope    2026-04-29 12:10
humaneval                78.5%   evalscope    2026-04-29 13:05
mathvista                58.2%   evalscope    2026-04-29 15:47
mmlu                     83.6%   evalscope    2026-04-29 11:48
mmbench                  75.1%   evalscope    2026-04-29 16:20
pope                       —     —            —
────────────────────────────────────────────────────────────
  6/7 benchmarks have results  (1 not yet parseable)
```

The report scans `outputs/` and auto-detects result files from all three tools (evalscope `report.json`, lm-eval `results_*.json`, opencompass `summary/*.csv`). Benchmarks still running or that failed show `—`.

### Dataset caching

Datasets are downloaded once and cached automatically — no re-download on subsequent runs.

| Tool | Cache location |
|---|---|
| evalscope | `~/.cache/modelscope/` |
| lm-eval | `~/.cache/huggingface/datasets/` |
| opencompass | `~/.cache/huggingface/datasets/` |

To reuse the cache on a different machine: `rsync -a ~/.cache/modelscope/ user@host2:~/.cache/modelscope/`

## Tool Notes

**EvalScope** natively supports all LLM and VLM benchmarks listed above. Use it by default unless you have a specific reason to switch.

**lm-eval** (EleutherAI) is the most widely cited harness for LLM leaderboards. Does not natively support VLM benchmarks — use EvalScope or OpenCompass for those. Benchmark names are mapped automatically.

**OpenCompass** has the strongest support for Chinese benchmarks and VLM evaluation. Benchmark names follow the `<name>_gen` convention for generative tasks; mapped automatically.
