"""
Lightweight HTTP server for OpenEvolve (standard library only).

Exposes evolution endpoints over HTTP so the TypeScript integration can drive
offline evolution runs without any third-party web framework. Built on
``http.server`` + ``threading`` so it runs with zero extra dependencies.

Endpoints
---------
GET  /api/v1/health                      -> {"status":"healthy","version": ...}
POST /api/v1/evolve                      -> spawn offline evolution run (202)
GET  /api/v1/runs/{run_id}               -> poll run status/result
POST /api/v1/workflows/orchestrate       -> spawn a workflow-style evolve run (202)
"""

import json
import logging
import signal as _signal_module
import threading
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import openevolve
from openevolve.api import run_evolution
from openevolve.config import Config, LLMModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "RUNS",
    "build_offline_config",
    "start_run",
    "OpenEvolveHandler",
    "OpenEvolveServer",
    "main",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
]

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# Module-level registry of runs: run_id -> {"status", "result", "error", "meta"}
RUNS: Dict[str, Dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def build_offline_config(
    population_size: int = 6,
    max_iterations: int = 4,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Build a Config that runs fully offline using the deterministic mock LLM.

    The mock backend is selected by giving a model whose name starts with
    ``"mock"`` (or ``provider="mock"``). Small population/max_iteration defaults
    keep a run completion fast enough for interactive use.
    """
    config = Config()
    config.llm.models = [LLMModelConfig(name="mock", provider="mock")]
    config.llm.evaluator_models = [LLMModelConfig(name="mock", provider="mock")]
    config.database.population_size = population_size
    config.max_iterations = max_iterations
    config.database.random_seed = 42
    config.random_seed = 42
    config.log_level = "WARNING"

    if config_overrides:
        _apply_overrides(config, config_overrides)

    return config


def _apply_overrides(config: Config, overrides: Dict[str, Any]) -> None:
    """Apply a flat dict of overrides onto the Config object safely."""
    for key, value in overrides.items():
        if key == "llm":
            continue  # never override the mock models from user input
        if hasattr(config, key):
            try:
                setattr(config, key, value)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Ignoring config override %s=%r: %s", key, value, exc)


def _safe_json(value: Any) -> Any:
    """Recursively convert values to JSON-serializable forms."""
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Extract a JSON-safe summary from an EvolutionResult."""
    try:
        return _safe_json(
            {
                "best_score": getattr(result, "best_score", None),
                "best_code": getattr(result, "best_code", ""),
                "metrics": getattr(result, "metrics", {}) or {},
                "output_dir": getattr(result, "output_dir", None),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"failed to serialize result: {exc}"}


# --------------------------------------------------------------------------- #
# Run lifecycle
# --------------------------------------------------------------------------- #
@contextmanager
def _silence_signal_registration():
    """Allow run_evolution() to execute from a background thread.

    ``openevolve.controller.run`` installs SIGINT/SIGTERM handlers via
    ``signal.signal()``. That call raises ``ValueError`` outside the main
    thread, which would abort the whole run. For a headless HTTP server the
    worker thread is not interactive, so we temporarily make signal
    registration a no-op and restore it afterwards.
    """
    original = _signal_module.signal
    _signal_module.signal = lambda signum, handler: None  # type: ignore[assignment]
    try:
        yield
    finally:
        _signal_module.signal = original


def start_run(
    initial_program: str,
    evaluator: str,
    config: Optional[Config] = None,
    iterations: Optional[int] = None,
) -> str:
    """Spawn a background evolution run and return its run_id (status: running)."""
    run_id = uuid.uuid4().hex

    with _RUNS_LOCK:
        RUNS[run_id] = {
            "run_id": run_id,
            "status": "running",
            "result": None,
            "error": None,
            "meta": {
                "initial_program": initial_program,
                "iterations": iterations,
            },
        }

    def _worker() -> None:
        try:
            cfg = config or build_offline_config()
            with _silence_signal_registration():
                result = run_evolution(
                initial_program=initial_program,
                evaluator=evaluator,
                config=cfg,
                iterations=iterations,
                cleanup=True,
            )
            with _RUNS_LOCK:
                RUNS[run_id]["status"] = "completed"
                RUNS[run_id]["result"] = _result_to_dict(result)
        except Exception as exc:  # Never crash the server thread.
            logger.exception("Evolution run %s failed", run_id)
            with _RUNS_LOCK:
                RUNS[run_id]["status"] = "failed"
                RUNS[run_id]["error"] = str(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return run_id


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _RUNS_LOCK:
        run = RUNS.get(run_id)
        return dict(run) if run else None


# --------------------------------------------------------------------------- #
# Workflow translation
# --------------------------------------------------------------------------- #
def translate_workflow(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a workflow orchestration request into evolve inputs.

    Recognized fields (all optional except problemStatement):
        system            - "evolutionary" | "decomposition" | "mdap_maker" |
                            "adversarial" | "integrated"
        problemStatement  - str describing the task
        generations       - int (-> iterations)
        populationSize    - int (-> population_size)
        config            - optional dict of config overrides
    """
    system = payload.get("system", "evolutionary")
    problem = payload.get("problemStatement", "")
    generations = payload.get("generations")
    population_size = payload.get("populationSize")

    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("'problemStatement' is required and must be a non-empty string")

    # Synthesize a tiny offline-runnable program + evaluator from the problem.
    initial_program = (
        f'"""{system} workflow target.\n\n{problem}\n"""\n\n'
        "# EVOLVE-BLOCK-START\n"
        "def solve(x):\n"
        "    # TODO: implement the workflow target\n"
        "    return x\n"
        "# EVOLVE-BLOCK-END\n"
    )

    evaluator = (
        "def evaluate(program_path):\n"
        "    import importlib.util\n"
        "    spec = importlib.util.spec_from_file_location('prog', program_path)\n"
        "    if spec is None or spec.loader is None:\n"
        "        return {'score': 0.0}\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    try:\n"
        "        spec.loader.exec_module(module)\n"
        "    except Exception:\n"
        "        return {'score': 0.0}\n"
        "    if not hasattr(module, 'solve'):\n"
        "        return {'score': 0.0}\n"
        "    try:\n"
        "        ok = module.solve(2) == 2\n"
        "    except Exception:\n"
        "        ok = False\n"
        "    return {'score': 1.0 if ok else 0.0}\n"
    )

    overrides: Dict[str, Any] = {}
    config_in = payload.get("config")
    if isinstance(config_in, dict):
        overrides.update(config_in)
    if isinstance(population_size, int):
        overrides["population_size"] = population_size

    iterations = generations if isinstance(generations, int) else None

    config = build_offline_config(config_overrides=overrides or None)

    return {
        "initial_program": initial_program,
        "evaluator": evaluator,
        "config": config,
        "iterations": iterations,
    }


# --------------------------------------------------------------------------- #
# HTTP handler / server
# --------------------------------------------------------------------------- #
class OpenEvolveHandler(BaseHTTPRequestHandler):
    server_version = f"OpenEvolveServer/{openevolve.__version__}"

    # Quieter default logging (still shows errors).
    def log_message(self, fmt, *args):  # noqa: A003
        logger.info("%s - %s", self.address_string(), fmt % args)

    # -- helpers ----------------------------------------------------------- #
    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid JSON body: {exc}")
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        return data

    def _handle_error(self, exc: Exception, status: int = 400) -> None:
        self._send_json(status, {"error": str(exc)})

    # -- routing ----------------------------------------------------------- #
    def do_GET(self):  # noqa: N802
        try:
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "" or path == "/api/v1/health":
                return self._health()
            if path.startswith("/api/v1/runs/"):
                run_id = path[len("/api/v1/runs/"):]
                return self._get_run(run_id)
            return self._send_json(404, {"error": f"Not found: {path}"})
        except Exception as exc:  # Never crash the server thread.
            logger.exception("Unhandled GET error")
            self._handle_error(exc, status=500)

    def do_POST(self):  # noqa: N802
        try:
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/api/v1/evolve":
                return self._evolve()
            if path == "/api/v1/workflows/orchestrate":
                return self._orchestrate()
            return self._send_json(404, {"error": f"Not found: {path}"})
        except ValueError as exc:
            self._handle_error(exc, status=400)
        except Exception as exc:  # Never crash the server thread.
            logger.exception("Unhandled POST error")
            self._handle_error(exc, status=500)

    # -- endpoints --------------------------------------------------------- #
    def _health(self):
        self._send_json(
            200,
            {"status": "healthy", "version": openevolve.__version__},
        )

    def _evolve(self):
        data = self._read_json_body()
        initial_program = data.get("initial_program")
        evaluator = data.get("evaluator")
        if not isinstance(initial_program, str) or not initial_program.strip():
            raise ValueError("'initial_program' (str) is required")
        if not isinstance(evaluator, str) or not evaluator.strip():
            raise ValueError("'evaluator' (str) is required")
        if "def evaluate" not in evaluator:
            raise ValueError("'evaluator' must contain an 'evaluate(program_path)' function")

        iterations = data.get("iterations")
        if iterations is not None and not isinstance(iterations, int):
            raise ValueError("'iterations' must be an int")

        config = None
        config_in = data.get("config")
        if config_in is not None:
            if not isinstance(config_in, dict):
                raise ValueError("'config' must be an object")
            config = build_offline_config(config_overrides=config_in)

        run_id = start_run(
            initial_program=initial_program,
            evaluator=evaluator,
            config=config,
            iterations=iterations,
        )
        self._send_json(202, {"run_id": run_id, "status": "running"})

    def _get_run(self, run_id: str):
        run = get_run(run_id)
        if run is None:
            return self._send_json(404, {"error": f"Run not found: {run_id}"})
        self._send_json(
            200,
            {
                "run_id": run["run_id"],
                "status": run["status"],
                "result": run["result"],
                "error": run["error"],
            },
        )

    def _orchestrate(self):
        data = self._read_json_body()
        try:
            translated = translate_workflow(data)
        except ValueError as exc:
            raise exc
        run_id = start_run(
            initial_program=translated["initial_program"],
            evaluator=translated["evaluator"],
            config=translated["config"],
            iterations=translated["iterations"],
        )
        # Mirror the workflowId convention used by the integration.
        self._send_json(202, {"workflowId": run_id, "status": "running"})


class OpenEvolveServer(ThreadingHTTPServer):
    """Threading HTTP server exposing the OpenEvolve evolution endpoints."""

    daemon_threads = True

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        super().__init__((host, port), OpenEvolveHandler)


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    logging.basicConfig(level=logging.INFO)
    server = OpenEvolveServer(host, port)
    print(f"OpenEvolve stdlib server listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
