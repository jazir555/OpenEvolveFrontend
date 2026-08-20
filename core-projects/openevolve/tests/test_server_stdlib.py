"""
Verification for the OpenEvolve standard-library HTTP server.

Starts the server in a background thread, then exercises:
  - GET  /api/v1/health
  - POST /api/v1/evolve   (real offline mock-LLM run, iterations=2)
  - GET  /api/v1/runs/{id}  (poll until completed)

Run directly:  python core-projects/openevolve/tests/test_server_stdlib.py
"""

import json
import threading
import time
import urllib.request

from openevolve.server_stdlib import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    OpenEvolveServer,
)


def _wait_for_run(run_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        req = urllib.request.Request(f"http://127.0.0.1:{DEFAULT_PORT}/api/v1/runs/{run_id}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.5)
    raise TimeoutError(f"Run {run_id} did not finish within {timeout}s")


def main():
    server = OpenEvolveServer(DEFAULT_HOST, DEFAULT_PORT)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{DEFAULT_PORT}"

        # 1) Health
        with urllib.request.urlopen(f"{base}/api/v1/health", timeout=10) as resp:
            assert resp.status == 200, f"health status {resp.status}"
            health = json.loads(resp.read().decode("utf-8"))
        assert health.get("status") == "healthy", health
        assert "version" in health, health
        print("PASS: health ->", health)

        # 2) Evolve (tiny offline program + evaluator)
        program = "def add(a, b):\n    return a + b\n"
        evaluator = (
            "def evaluate(program_path):\n"
            "    import importlib.util\n"
            "    spec = importlib.util.spec_from_file_location('p', program_path)\n"
            "    m = importlib.util.module_from_spec(spec)\n"
            "    spec.loader.exec_module(m)\n"
            "    ok = m.add(2, 3) == 5 and m.add(-1, 1) == 0\n"
            "    return {'score': 1.0 if ok else 0.0}\n"
        )
        body = json.dumps(
            {
                "initial_program": program,
                "evaluator": evaluator,
                "iterations": 2,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/v1/evolve",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 202, f"evolve status {resp.status}"
            ev = json.loads(resp.read().decode("utf-8"))
        assert ev.get("status") == "running", ev
        run_id = ev["run_id"]
        print("PASS: evolve accepted ->", run_id)

        # 3) Poll run
        result = _wait_for_run(run_id)
        assert result["status"] == "completed", result
        assert result["result"] is not None, result
        assert "best_code" in result["result"], result
        print("PASS: run completed ->", json.dumps(result["result"])[:200])

        print("ALL TESTS PASSED")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
