"""Drive a full OpenEvolve "evolution" BubbleLab workflow over the :8000 REST API.

This script performs, over HTTP, exactly what the BubbleLab GUI does when an
operator creates an evolutionary workflow and presses Run:

    1. POST /api/workflows            -> create the workflow (type=evolution)
    2. POST /api/workflows/{id}/start -> execute it (real openevolve engine)
    3. GET  /api/workflows/{id}       -> poll status + attached openevolve result

The workflow's ``parameters`` carry the LZ77 ``initial_program`` + ``evaluator``
source and the live-LLM configuration, so the run optimizes a real algorithm.

Usage:
    python run_workflow.py --llm mock --iterations 2 --population 4
    python run_workflow.py --llm kimi   --iterations 8 --population 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent

NV_API_BASE = "https://integrate.api.nvidia.com/v1"
KIMI_MODEL = "moonshotai/kimi-k3"


def _nvidia_key() -> str:
    """Read the configured NVIDIA NIM key from the BubbleLab API .env."""
    env_file = (
        REPO_ROOT
        / "core-projects"
        / "BubbleLab"
        / "apps"
        / "bubblelab-api"
        / ".env"
    )
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("NVIDIA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("NVIDIA_API_KEY not found in bubblelab-api/.env")


def _http(method: str, path: str, data: dict | None = None, timeout: float = 30.0):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode("utf-8") if data is not None else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body) if body else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["mock", "kimi"], default="mock")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = ap.parse_args()

    initial_program = (BENCH_DIR / "initial_program.py").read_text(encoding="utf-8")
    evaluator = (BENCH_DIR / "evaluator.py").read_text(encoding="utf-8")

    llm = {}
    if args.llm == "kimi":
        llm = {
            "name": KIMI_MODEL,
            "provider": "nvidia",
            "api_key": _nvidia_key(),
            "api_base": NV_API_BASE,
            "temperature": 0.7,
            "max_tokens": 4096,
        }

    parameters = {
        "max_iterations": args.iterations,
        "population_size": args.population,
        "seed": args.seed,
        "initial_program": initial_program,
        "evaluator": evaluator,
        "llm": llm,
    }

    create_payload = {
        "name": f"LZ77 Compression Optimizer ({args.llm})",
        "description": (
            "Evolve a naive LZ77 compressor toward higher compression ratio and "
            "throughput while preserving exact round-trip correctness."
        ),
        "workflow_type": "evolution",
        "problem_statement": (
            "Optimize the LZ77 compressor in initial_program: compress(data) and "
            "decompress(data). Improve compressed size (ratio) and throughput "
            "(speed) without ever breaking decompress(compress(x)) == x."
        ),
        "content_type": "code",
        "metadata": {"evolution_params": parameters},
        "parameters": parameters,
    }

    status, wf = _http("POST", "/api/workflows", create_payload)
    wid = wf["id"]
    print(f"[create] workflow_id={wid} status={status} name={wf['name']}")

    status, wf = _http("POST", f"/api/workflows/{wid}/start", {"context": "algorithm optimization"})
    print(f"[start]  status=HTTP{status} workflow_status={wf.get('status')}")

    # Poll until the engine result is attached.
    deadline = time.time() + args.timeout_seconds
    result = None
    while time.time() < deadline:
        _, wf = _http("GET", f"/api/workflows/{wid}")
        params = wf.get("parameters") or {}
        result = params.get("openevolve")
        if result:
            break
        time.sleep(5.0)

    if not result:
        print("[result] FAIL: no openevolve result attached in time")
        return 1

    print("[result] engine        =", result.get("engine"))
    print("[result] llm_mode      =", result.get("llm_mode"))
    print("[result] best_score    =", result.get("best_score"))
    print("[result] generations   =", result.get("generations"))
    print("[result] iterations    =", result.get("iterations"))
    print("[result] population    =", result.get("population_size"))
    print("[result] duration_s    =", result.get("duration_seconds"))
    print("[result] started_at    =", result.get("started_at"))
    print("[result] completed_at  =", result.get("completed_at"))
    print("[result] metrics       =", json.dumps(result.get("metrics"), indent=2))

    best_code = result.get("best_code") or ""
    out_path = BENCH_DIR / "evolved_program.py"
    if best_code:
        out_path.write_text(best_code, encoding="utf-8")
        print(f"[result] best_code saved to {out_path}")
    else:
        print("[result] FAIL: best_code is empty")
        return 1

    # Persist the full result JSON for the report.
    (BENCH_DIR / "result.json").write_text(
        json.dumps({"workflow_id": wid, "workflow": wf, "result": result}, indent=2),
        encoding="utf-8",
    )
    print(f"[result] full result saved to {BENCH_DIR / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())