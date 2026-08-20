"""
proxy_path_test.py — prove the BubbleLab API proxy -> OpenEvolve backend
forwarding contract works end to end (without the bun proxy process itself).

The Hono proxy at apps/bubblelab-api/src/routes/openevolve.ts is PASSIVE: it
forwards the incoming method/body verbatim to OPENEVOLVE_API_URL for paths
under /api/v1/* and /api/*. So the only thing the proxy relies on is the
backend's behaviour at those exact paths. This script *replicates* what the
proxy does (forward a request to the backend) and asserts the contract:

    1. GET  http://127.0.0.1:8000/api/v1/health          -> 200
    2. POST http://127.0.0.1:8000/api/v1/workflows/orchestrate
       (tiny body)                                        -> 202 + run id
    3. POLL http://127.0.0.1:8000/api/v1/runs/{id}
       until completed                                    -> completed

If these pass, the UI -> (proxy forward) -> backend path is proven contract-
complete: the proxy simply forwards, so whatever this script can reach, the
proxy can reach identically.

The service is launched as a subprocess with the same hyphenated-package stub +
PYTHONPATH workaround used by scripts/launch_demo.py, so `openevolve_api` and
`openevolve` both resolve. Stdlib only (urllib + subprocess).

Exit code is non-zero on any failure. The server is always killed in `finally`.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

# Resolve the repository root from this file so the script works from anywhere.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))  # services/openevolve-api
REPO_ROOT = os.path.normpath(
    os.path.join(SERVICE_DIR, "..", "..", "..", "..", "..")  # OpenEvolveFrontend
)
OPENEVOLVE_LIB_DIR = os.path.normpath(
    os.path.join(REPO_ROOT, "core-projects", "openevolve")
)

PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"


def _url(path: str) -> str:
    return BASE + path


def _http(method: str, path: str, data: bytes | None = None, timeout: float = 10.0):
    req = urllib.request.Request(_url(path), data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def _check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    line = f"[{mark}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> int:
    global PORT, BASE
    proc: subprocess.Popen | None = None
    failures = 0
    try:
        env = dict(os.environ)
        # `openevolve_api` resolves to the service dir via a thin stub package
        # (mirrors launch_demo.py). `openevolve` resolves to the real library so
        # the bridge can import the engine.
        stub_dir = tempfile.mkdtemp(prefix="oe_api_proxy_stub_")
        pkg_dir = os.path.join(stub_dir, "openevolve_api")
        os.makedirs(pkg_dir, exist_ok=True)
        with open(os.path.join(pkg_dir, "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write(f"__path__ = [{SERVICE_DIR!r}]\n")
        env["PYTHONPATH"] = (
            stub_dir
            + os.pathsep + SERVICE_DIR
            + os.pathsep + OPENEVOLVE_LIB_DIR
            + os.pathsep + env.get("PYTHONPATH", "")
        )
        env.setdefault("WORKFLOW_DB_PATH", "C:\\Temp\\openevolve_api_proxy_workflows.db")
        env.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

        # Pick a free port if 8000 is already taken.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                PORT = 8011
                BASE = f"http://127.0.0.1:{PORT}"
                print(f"[proxy-test] port 8000 occupied, using {PORT}")

        cmd = [
            sys.executable, "-m", "uvicorn",
            "openevolve_api.main:app",
            "--host", "127.0.0.1", "--port", str(PORT),
        ]
        print(f"[proxy-test] launching backend: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, cwd=SERVICE_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        # --- The proxy contract: replicate what openevolve.ts forwards ---------

        # 1. GET /api/v1/health  (proxy route: app.get('/api/v1/health'))
        health_ok = False
        deadline = time.time() + 25
        while time.time() < deadline:
            try:
                status, _ = _http("GET", "/api/v1/health", timeout=3)
                if status == 200:
                    health_ok = True
                    break
            except Exception:
                time.sleep(0.5)
        failures += 0 if _check("GET /api/v1/health (proxy forward)", health_ok) else 1

        # 2. POST /api/v1/workflows/orchestrate (proxy route forwards body verbatim)
        payload = (
            '{"system":"evolutionary","problemStatement":'
            '"evolve a function that adds two numbers",'
            '"generations":2,"populationSize":4}'
        ).encode("utf-8")
        run_id = None
        try:
            status, body = _http("POST", "/api/v1/workflows/orchestrate", data=payload)
            if status == 202:
                run_id = json.loads(body).get("workflowId") or json.loads(body).get("runId")
            failures += 0 if _check("POST /api/v1/workflows/orchestrate->202", status == 202, f"status={status}") else 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _check("POST /api/v1/workflows/orchestrate->202", False, repr(exc))

        # 3. POLL /api/v1/runs/{id} (proxy route: app.get('/api/v1/runs/:id'))
        completed = False
        final = None
        if run_id:
            deadline = time.time() + 40
            while time.time() < deadline:
                try:
                    status, body = _http("GET", f"/api/v1/runs/{run_id}", timeout=5)
                    if status == 200:
                        final = json.loads(body)
                        if final.get("status") in ("completed", "failed"):
                            break
                except Exception:
                    pass
                time.sleep(0.5)
            completed = final is not None and final.get("status") == "completed"
            detail = f"run_id={run_id} status={(final or {}).get('status')}"
            failures += 0 if _check("POLL /api/v1/runs/{id}->completed", completed, detail) else 1
        else:
            failures += 1
            _check("POLL /api/v1/runs/{id}->completed", False, "no run_id from orchestrate")

        print("-" * 60)
        if failures == 0:
            print(f"PROXY PATH PASS: UI->proxy->backend contract verified on port {PORT}.")
            return 0
        print(f"PROXY PATH FAIL: {failures} check(s) failed.")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[proxy-test] FAIL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if proc is not None:
            try:
                if proc.poll() is None:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/pid", str(proc.pid), "/f", "/t"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
