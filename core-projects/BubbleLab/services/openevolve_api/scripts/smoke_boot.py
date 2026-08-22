"""
Boot smoke test: prove the PRIMARY openevolve-api service actually BOOTS as a
real uvicorn server (not just TestClient) and runs a REAL evolution over HTTP.

Steps:
    1. Launch ``uvicorn openevolve_api.main:app`` as a subprocess.
    2. Poll ``/health`` until it returns 200 (up to ~20s).
    3. POST ``/api/v1/workflows/orchestrate`` to start a real (offline mock) run.
    4. Poll ``/api/v1/runs/{workflowId}`` until ``completed`` (up to ~30s).
    5. Assert ``result.best_code`` is non-empty.

Prints PASS/FAIL and exits non-zero on failure. Kills the subprocess in finally.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

SERVICE_DIR = (
    "C:\\Users\\mmeadow\\Documents\\OpenEvolve\\OpenEvolveFrontend"
    "\\core-projects\\BubbleLab\\services\\openevolve-api"
)

# Prefer 8000; fall back to 8011 if occupied.
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


def main() -> int:
    proc: subprocess.Popen | None = None
    try:
        env = dict(os.environ)
        # The service directory is named "openevolve-api" (hyphen) and has no
        # top-level __init__.py, so `import openevolve_api` does not resolve as a
        # package. Generate a thin package stub on PYTHONPATH that points __path__
        # at the real service directory, mirroring the mechanism the TestClient
        # test uses (sys.modules["openevolve_api"]). This lets
        # `uvicorn openevolve_api.main:app` import correctly.
        stub_dir = tempfile.mkdtemp(prefix="oe_api_stub_")
        pkg_dir = os.path.join(stub_dir, "openevolve_api")
        os.makedirs(pkg_dir, exist_ok=True)
        with open(os.path.join(pkg_dir, "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write(f"__path__ = [{SERVICE_DIR!r}]\n")
        env["PYTHONPATH"] = (
            stub_dir + os.pathsep + SERVICE_DIR + os.pathsep + env.get("PYTHONPATH", "")
        )
        env.setdefault("WORKFLOW_DB_PATH", "C:\\Temp\\openevolve_api_smoke_workflows.db")
        env.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

        # Find a free port if 8000 is taken.
        global PORT, BASE

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                PORT = 8011
                BASE = f"http://127.0.0.1:{PORT}"
                print(f"[smoke] port 8000 occupied, using {PORT}")

        cmd = [
            sys.executable, "-m", "uvicorn",
            "openevolve_api.main:app",
            "--host", "127.0.0.1", "--port", str(PORT),
        ]
        print(f"[smoke] launching: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, cwd=SERVICE_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        # 1. Wait for /health.
        deadline = time.time() + 20
        health_ok = False
        while time.time() < deadline:
            try:
                status, _ = _http("GET", "/health", timeout=3)
                if status == 200:
                    health_ok = True
                    break
            except Exception:
                time.sleep(0.5)
        if not health_ok:
            print("[smoke] FAIL: /health never returned 200")
            return 1
        print("[smoke] OK: service is up (/health=200)")

        # 2. Orchestrate a real evolution run.
        payload = (
            '{"system":"evolutionary","problemStatement":'
            '"evolve a function that adds two numbers",'
            '"generations":2,"populationSize":4}'
        ).encode("utf-8")
        status, body = _http("POST", "/api/v1/workflows/orchestrate", data=payload)
        if status != 202:
            print(f"[smoke] FAIL: orchestrate returned {status}: {body}")
            return 1
        import json
        workflow_id = json.loads(body)["workflowId"]
        print(f"[smoke] OK: orchestrated run {workflow_id}")

        # 3. Poll until completed.
        deadline = time.time() + 30
        final = None
        while time.time() < deadline:
            status, body = _http("GET", f"/api/v1/runs/{workflow_id}", timeout=5)
            if status == 200:
                final = json.loads(body)
                if final["status"] in ("completed", "failed"):
                    break
            time.sleep(0.5)
        if final is None or final["status"] != "completed":
            print(f"[smoke] FAIL: run did not complete: {final}")
            return 1

        best_code = (final.get("result") or {}).get("best_code", "")
        if not best_code or not best_code.strip():
            print("[smoke] FAIL: best_code is empty")
            return 1

        print("[smoke] PASS: primary service booted and ran a REAL evolution over HTTP")
        print(f"[smoke]   workflowId={workflow_id} engine={final['result'].get('engine')} "
              f"llm_mode={final['result'].get('llm_mode')} "
              f"best_score={final['result'].get('best_score')}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] FAIL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if proc is not None:
            try:
                if proc.poll() is None:
                    if sys.platform == "win32":
                        # Kill the process tree on Windows.
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
