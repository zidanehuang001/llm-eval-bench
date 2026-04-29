#!/usr/bin/env python3
"""LLM/VLM benchmark runner with concurrent multi-server support."""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ─── Benchmark lists ──────────────────────────────────────────────────────────
LLM_BENCHES = [
    # Math / reasoning
    "aime25", "hmmt25", "gsm8k", "math500",
    # Coding
    "humaneval",
    # General knowledge
    "mmlu", "mmlu_pro", "ceval", "cmmlu",
    # Commonsense / QA
    "hellaswag", "arc", "simple_qa", "truthful_qa",
    # Instruction following
    "ifeval",
]

VLM_BENCHES = [
    # General multimodal
    "mmbench", "mme", "mmstar", "seed_bench",
    # Math + science
    "mathvista", "scienceqa",
    # Document / OCR
    "ocrbench", "chartqa", "docvqa", "textvqa",
    # VQA
    "vqav2",
    # Hallucination
    "pope", "hallusionbench",
]

# ─── Per-benchmark overrides ──────────────────────────────────────────────────
BENCH_TIMEOUT = {
    "aime25": 600, "hmmt25": 600,
    "math500": 300, "humaneval": 300, "ifeval": 240,
    "mmbench": 180, "mme": 180, "mmstar": 180, "seed_bench": 180,
    "pope": 180, "vqav2": 180, "textvqa": 180, "ocrbench": 180, "scienceqa": 240,
    "chartqa": 300, "docvqa": 300, "mathvista": 360, "hallusionbench": 300,
}

BENCH_BATCH = {
    "aime25": 4, "hmmt25": 4, "math500": 8, "humaneval": 8,
    "mmbench": 8, "mme": 8, "mmstar": 8, "seed_bench": 8,
    "pope": 8, "vqav2": 8, "textvqa": 8, "ocrbench": 8, "scienceqa": 8,
    "chartqa": 4, "docvqa": 4, "mathvista": 4, "hallusionbench": 4,
}

# ─── Tool-specific name mappings ──────────────────────────────────────────────
LM_EVAL_MAP = {
    "mmlu": "mmlu", "mmlu_pro": "mmlu_pro", "gsm8k": "gsm8k",
    "math500": "minerva_math", "humaneval": "humaneval",
    "hellaswag": "hellaswag", "arc": "arc_challenge",
    "truthful_qa": "truthfulqa_mc2", "ifeval": "ifeval",
    "ceval": "ceval-valid", "cmmlu": "cmmlu", "simple_qa": "simpleqa",
    "aime25": "aime24", "hmmt25": "hmmt_nov2024",
}

OC_MAP = {
    "mmlu": "mmlu_gen", "mmlu_pro": "mmlu_pro_gen", "gsm8k": "gsm8k_gen",
    "math500": "math_gen", "humaneval": "humaneval_gen",
    "hellaswag": "hellaswag_ppl", "arc": "ARC_e_gen",
    "ceval": "ceval_gen", "cmmlu": "cmmlu_gen",
    "truthful_qa": "truthful_qa_gen", "ifeval": "ifeval_gen",
    "simple_qa": "simpleqa_gen",
    "mmbench": "MMBench_DEV_EN_V11", "mme": "MME", "mmstar": "MMStar",
    "seed_bench": "SEEDBench_IMG", "mathvista": "MathVista_MINI",
    "scienceqa": "ScienceQA_VAL", "ocrbench": "OCRBench",
    "chartqa": "ChartQA_TEST", "docvqa": "DocVQA_VAL",
    "textvqa": "TextVQA_VAL", "vqav2": "VQAv2_VAL",
    "pope": "POPE", "hallusionbench": "HallusionBench",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
_print_lock = Lock()

def log(msg: str) -> None:
    with _print_lock:
        if HAS_TQDM:
            tqdm.write(msg)
        else:
            print(msg, flush=True)

def resolve_url(host_entry: str, default_port: str) -> str:
    if "://" in host_entry:
        return host_entry.rstrip("/")
    if ":" in host_entry:
        return f"http://{host_entry}/v1"
    return f"http://{host_entry}:{default_port}/v1"

def host_label(url: str) -> str:
    return url.replace("http://", "").replace("https://", "").replace("/v1", "")

def warmup_host(url: str, model: str, api_key: str) -> tuple[bool, str]:
    """Health-check + one minimal generation to wake the model into GPU memory."""
    label = host_label(url)

    # 1. /v1/models — fast connectivity check
    try:
        req = urllib.request.Request(
            f"{url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        return False, f"  [WARMUP] {label}  health check failed: {e}"

    # 2. Minimal chat completion — triggers model weight loading on first call
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }).encode()
    try:
        req = urllib.request.Request(
            f"{url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        urllib.request.urlopen(req, timeout=120)
    except Exception as e:
        return False, f"  [WARMUP] {label}  generation ping failed: {e}"

    return True, f"  [WARMUP] {label}  ready"

def build_command(bench, url, model, api_key, tool, batch, timeout, out_dir):
    if tool == "evalscope":
        return [
            "evalscope", "eval",
            "--model",           model,
            "--api-url",         f"{url}/chat/completions",
            "--api-key",         api_key,
            "--datasets",        bench,
            "--eval-batch-size", str(batch),
            "--repeats",         "1",
            "--timeout",         str(timeout),
        ]
    if tool == "lm-eval":
        task = LM_EVAL_MAP.get(bench, bench)
        return [
            "lm_eval",
            "--model",       "openai-chat-completions",
            "--model-args",  f"model={model},base_url={url},api_key={api_key}",
            "--tasks",       task,
            "--num_fewshot", "0",
            "--batch_size",  str(batch),
            "--output_path", os.path.join(out_dir, bench),
        ]
    if tool == "opencompass":
        task = OC_MAP.get(bench, f"{bench}_gen")
        return [
            "python", "-m", "opencompass.cli.main",
            "--models",       "openai_api",
            "--model-kwargs", f"path={model} api_base={url} key={api_key}",
            "--datasets",     task,
            "--work-dir",     os.path.join(out_dir, bench),
        ]
    raise ValueError(f"Unknown tool: {tool}")

# ─── Core runner ──────────────────────────────────────────────────────────────
def run_one(bench, url, args, out_dir, log_dir, pbar=None):
    label     = host_label(url)
    done_file = os.path.join(out_dir, f"{bench}.done")

    def _done(status, msg):
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f"{bench} {status}")
        log(msg)
        return bench, status

    if args.resume and os.path.exists(done_file):
        return _done("skip", f"  [SKIP]  {bench:<22} [{label}]")

    timeout  = BENCH_TIMEOUT.get(bench, args.timeout)
    batch    = BENCH_BATCH.get(bench, args.batch)
    log_path = os.path.join(log_dir, f"{bench}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    log(f"  [START] {bench:<22} [{label}]  batch={batch}  timeout={timeout}s")

    cmd = build_command(bench, url, args.model, args.api_key, args.tool, batch, timeout, out_dir)
    try:
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout + 60)
        if proc.returncode == 0:
            open(done_file, "w").close()
            return _done("pass", f"  [PASS]  {bench:<22} [{label}]")
        return _done("fail", f"  [FAIL]  {bench:<22} [{label}] exit={proc.returncode} log={log_path}")
    except subprocess.TimeoutExpired:
        return _done("fail", f"  [FAIL]  {bench:<22} [{label}] process timeout — {log_path}")
    except Exception as exc:
        return _done("fail", f"  [FAIL]  {bench:<22} [{label}] {exc}")

# ─── Report ───────────────────────────────────────────────────────────────────
def generate_report(out_dir):
    done_files = glob.glob(os.path.join(out_dir, "*.done"))
    if not done_files:
        print(f"No completed benchmarks (.done files) in: {out_dir}")
        return

    benches = sorted(os.path.splitext(os.path.basename(f))[0] for f in done_files)

    def _evalscope(bench):
        for f in sorted(glob.glob(os.path.join(out_dir, "**", "report.json"), recursive=True)):
            try:
                data = json.load(open(f))
                hint = str(data.get("dataset_name", data.get("dataset", ""))).lower()
                if hint and bench.lower() not in hint and hint not in bench.lower():
                    continue
                score = None
                for key in ("score", "accuracy", "pass@1", "exact_match"):
                    if key in data:
                        score = data[key]; break
                if score is None and isinstance(data.get("metrics"), dict):
                    score = next(iter(data["metrics"].values()), None)
                if score is None and isinstance(data.get("report"), list) and data["report"]:
                    score = data["report"][0].get("score") or data["report"][0].get("accuracy")
                if score is None and isinstance(data.get("results"), dict):
                    first = next(iter(data["results"].values()), {})
                    for key in ("accuracy", "score", "pass@1"):
                        if key in first:
                            score = first[key]; break
                if score is not None:
                    ts = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
                    return score, ts
            except Exception:
                pass
        return None, None

    def _lm_eval(bench):
        for f in sorted(glob.glob(os.path.join(out_dir, bench, "results_*.json")), reverse=True):
            try:
                data = json.load(open(f))
                results = data.get("results", {})
                if results:
                    first = next(iter(results.values()))
                    score = None
                    for key in ("acc,none", "acc_norm,none", "exact_match,none", "pass@1,none"):
                        if key in first:
                            score = first[key]; break
                    if score is None:
                        score = next(
                            (v for k, v in first.items()
                             if isinstance(v, float) and "stderr" not in k), None
                        )
                    if score is not None:
                        ts = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
                        return score, ts
            except Exception:
                pass
        return None, None

    def _opencompass(bench):
        pattern = os.path.join(out_dir, bench, "**", "summary", "*.csv")
        for f in sorted(glob.glob(pattern, recursive=True), reverse=True):
            try:
                reader = csv.DictReader(open(f))
                for row in reader:
                    raw = row.get("score") or row.get("accuracy") or row.get("Accuracy")
                    if raw and raw.strip() not in ("-", ""):
                        ts = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
                        return float(raw), ts
            except Exception:
                pass
        return None, None

    rows = []
    for bench in benches:
        score = ts = tool = None
        for extractor, name in [(_evalscope, "evalscope"), (_lm_eval, "lm-eval"), (_opencompass, "opencompass")]:
            s, t = extractor(bench)
            if s is not None:
                score, ts, tool = s, t, name
                break
        rows.append((bench, score, tool, ts))

    def fmt(s):
        if s is None: return "—"
        return f"{s * 100:.1f}%" if s <= 1.0 else f"{s:.1f}%"

    sep = "─" * 60
    print()
    print(sep)
    print(f"{'Benchmark':<22} {'Score':>8}  {'Tool':<12} Completed")
    print(sep)
    for bench, score, tool, ts in rows:
        print(f"{bench:<22} {fmt(score):>8}  {tool or '—':<12} {ts or '—'}")
    print(sep)
    found = sum(1 for _, s, _, _ in rows if s is not None)
    print(f"  {found}/{len(rows)} benchmarks have results  ({len(rows) - found} not yet parseable)")
    print()

# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="LLM/VLM benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --host 10.7.2.33 --model qwen2.5-7b
  %(prog)s --hosts 10.7.2.33:8000,10.7.2.34:8000 --model qwen2.5-7b --all
  %(prog)s --hosts 10.7.2.33,10.7.2.34 --model qwen2.5-vl-7b --vlm --workers 2
  %(prog)s --report
        """,
    )
    p.add_argument("--model",    default=os.environ.get("MODEL", "step3p5-flash"), help="Model name")
    p.add_argument("--host",     default=os.environ.get("VLLM_HOST", ""),          help="Single vLLM host")
    p.add_argument("--port",     default=os.environ.get("VLLM_PORT", "8000"),      help="Default port")
    p.add_argument("--hosts",    default="",   help="Comma-separated host[:port] list for multi-server")
    p.add_argument("--url",      default=os.environ.get("BASE_URL", "http://10.7.2.33:8000/v1"), help="Full API base URL")
    p.add_argument("--api-key",  default=os.environ.get("API_KEY", "EMPTY"),       help="API key")
    p.add_argument("--tool",     default=os.environ.get("EVAL_TOOL", "evalscope"),
                   choices=["evalscope", "lm-eval", "opencompass"])
    p.add_argument("--batch",    type=int, default=int(os.environ.get("BATCH_SIZE", "16")), help="Default batch size")
    p.add_argument("--timeout",  type=int, default=int(os.environ.get("TIMEOUT", "120")),  help="Default timeout (s)")
    p.add_argument("--workers",  type=int, default=1, help="Concurrent benchmarks per host (default: 1)")
    p.add_argument("--vlm",      action="store_true", help="Run VLM benchmarks only")
    p.add_argument("--all",      action="store_true", help="Run all LLM + VLM benchmarks")
    p.add_argument("--benches",  default="", help="Comma-separated benchmark override list")
    p.add_argument("--resume",   action="store_true", help="Skip already-completed benchmarks")
    p.add_argument("--report",   action="store_true", help="Print results table and exit")
    return p.parse_args()

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir, log_dir = "outputs", "logs"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.report:
        generate_report(out_dir)
        return

    # Resolve host list
    if args.hosts:
        hosts = [resolve_url(h.strip(), args.port) for h in args.hosts.split(",") if h.strip()]
    elif args.host:
        hosts = [resolve_url(args.host, args.port)]
    else:
        hosts = [args.url.rstrip("/")]

    # Build benchmark list
    if args.benches:
        benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    elif args.all:
        benches = LLM_BENCHES + VLM_BENCHES
    elif args.vlm:
        benches = VLM_BENCHES
    else:
        benches = LLM_BENCHES

    # Warm up all hosts in parallel before starting benchmarks
    print("Warming up hosts...")
    with ThreadPoolExecutor(max_workers=len(hosts)) as ex:
        warmup_futures = {ex.submit(warmup_host, h, args.model, args.api_key): h for h in hosts}
        failed_hosts = set()
        for f in as_completed(warmup_futures):
            ok, msg = f.result()
            print(msg)
            if not ok:
                failed_hosts.add(warmup_futures[f])
    if failed_hosts:
        print(f"\nAborting: {len(failed_hosts)} host(s) failed warmup.")
        sys.exit(1)
    print()

    # Round-robin assignment: bench i → hosts[i % len(hosts)]
    assignments = [(bench, hosts[i % len(hosts)]) for i, bench in enumerate(benches)]

    # Pre-run breakdown: show pending vs skip per host before anything starts
    host_pending = {h: [] for h in hosts}
    host_skip    = {h: [] for h in hosts}
    for bench, url in assignments:
        done_file = os.path.join(out_dir, f"{bench}.done")
        if args.resume and os.path.exists(done_file):
            host_skip[url].append(bench)
        else:
            host_pending[url].append(bench)

    print("=" * 60)
    print(f"  Tool:    {args.tool}")
    print(f"  Model:   {args.model}")
    print(f"  Workers: {args.workers} per host  ({len(hosts) * args.workers} concurrent max)")
    print()
    for i, h in enumerate(hosts):
        lbl = host_label(h)
        pending = host_pending[h]
        skipped = host_skip[h]
        print(f"  Host {i+1}: {lbl}")
        print(f"    pending ({len(pending)}): {', '.join(pending) or '—'}")
        if skipped:
            print(f"    skip   ({len(skipped)}): {', '.join(skipped)}")
    print("=" * 60)
    print()

    totals = {"pass": 0, "fail": 0, "skip": 0}
    max_workers = len(hosts) * args.workers
    pbar = tqdm(total=len(assignments), unit="bench", disable=not HAS_TQDM) if HAS_TQDM else None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one, bench, url, args, out_dir, log_dir, pbar): bench
            for bench, url in assignments
        }
        for future in as_completed(futures):
            try:
                _, status = future.result()
                totals[status] = totals.get(status, 0) + 1
            except Exception as exc:
                log(f"  Unexpected error: {exc}")
                totals["fail"] += 1

    if pbar:
        pbar.close()

    print()
    print("=" * 60)
    print(f"  Summary: {totals['pass']} passed, {totals['fail']} failed, {totals['skip']} skipped")
    print("=" * 60)

    generate_report(out_dir)

if __name__ == "__main__":
    main()
