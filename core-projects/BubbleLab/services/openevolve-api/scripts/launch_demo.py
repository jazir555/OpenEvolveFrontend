"""
launch_demo.py — end-to-end happy-path demo for the OpenEvolve + BubbleLab
integration WITHOUT a browser.

What it proves:
    1. `uvicorn openevolve_api.main:app` boots as a real server subprocess.
    2. GET  /health                                   -> 200
    3. GET  /api/parameters/schema                    -> 200
    4. GET  /api/monitoring/dashboard                 -> 200
    5. POST /api/v1/workflows/orchestrate             -> 202, returns a run id
    6. POLL /api/v1/runs/{run_id} until "completed"   -> completed
       and best_code is non-empty (a REAL offline mock-LLM evolution ran).

The OpenEvolve engine is driven through the REAL bridge
(core/openevolve_bridge.py) with an offline mock LLM — no API keys needed.

Cross-platform: uses the same package-stub + PYTHONPATH trick that
scripts/smoke_boot.py uses so `openevolve_api` and `openevolve` resolve.

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
SERVICE_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..")  # services/openevolve-api
)
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
    proc: subprocess.Popen | None = None
    failures = 0
    try:
        env = dict(os.environ)
        # `openevolve_api` resolves to the service dir via a thin stub package
        # (mirrors smoke_boot.py). `openevolve` resolves to the real library so
        # the bridge can import the engine.
        stub_dir = tempfile.mkdtemp(prefix="oe_api_demo_stub_")
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
        env.setdefault("WORKFLOW_DB_PATH", "C:\\Temp\\openevolve_api_demo_workflows.db")
        env.setdefault("OPENEVOLVE_BRIDGE_ENABLED", "1")

        # Pick a free port if 8000 is already taken.
        global PORT, BASE
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                PORT = 8011
                BASE = f"http://127.0.0.1:{PORT}"
                print(f"[demo] port 8000 occupied, using {PORT}")

        cmd = [
            sys.executable, "-m", "uvicorn",
            "openevolve_api.main:app",
            "--host", "127.0.0.1", "--port", str(PORT),
        ]
        print(f"[demo] launching: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, cwd=SERVICE_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        # 1. /health
        health_ok = False
        deadline = time.time() + 25
        while time.time() < deadline:
            try:
                status, _ = _http("GET", "/health", timeout=3)
                if status == 200:
                    health_ok = True
                    break
            except Exception:
                time.sleep(0.5)
        failures += 0 if _check("service boots (/health=200)", health_ok) else 1

        # 2. /api/parameters/schema
        try:
            status, body = _http("GET", "/api/parameters/schema", timeout=5)
            ok = status == 200 and bool(json.loads(body))
            failures += 0 if _check("GET /api/parameters/schema", ok, f"status={status}") else 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _check("GET /api/parameters/schema", False, repr(exc))

        # 3. /api/monitoring/dashboard
        try:
            status, body = _http("GET", "/api/monitoring/dashboard", timeout=5)
            ok = status == 200 and bool(json.loads(body))
            failures += 0 if _check("GET /api/monitoring/dashboard", ok, f"status={status}") else 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _check("GET /api/monitoring/dashboard", False, repr(exc))

        # 4. POST /api/v1/workflows/orchestrate -> 202
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
            failures += 0 if _check("POST /api/v1/workflows/orchestrate", status == 202, f"status={status}") else 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _check("POST /api/v1/workflows/orchestrate", False, repr(exc))

        # 5. Poll /api/v1/runs/{run_id} until completed.
        completed = False
        best_code = ""
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
            best_code = (final or {}).get("result", {}) or {}
            best_code = best_code.get("best_code", "") if isinstance(best_code, dict) else ""
            detail = f"run_id={run_id} status={(final or {}).get('status')}"
            if completed and best_code and best_code.strip():
                detail += " (best_code present, real evolution ran)"
            failures += 0 if _check("evolution run completed with best_code", completed and bool(best_code.strip()), detail) else 1
        else:
            failures += 1
            _check("evolution run completed with best_code", False, "no run_id from orchestrate")

        print("-" * 60)
        if failures == 0:
            print("DEMO PASS: OpenEvolve + BubbleLab happy path verified (offline mock LLM).")
            return 0
        print(f"DEMO FAIL: {failures} check(s) failed.")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] FAIL: {type(exc).__name__}: {exc}")
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
