"""
OpenEvolve Bridge for the OpenEvolve API service.

This module is the single seam between this FastAPI service (a standalone
reimplementation) and the REAL ``openevolve`` library that lives in
``core-projects/openevolve``.

Design notes
------------
* The library entry point is ``openevolve.api.run_evolution``, which is
  *synchronous* and internally calls ``asyncio.run(...)``. It therefore cannot be
  invoked from inside a running event loop, and ``openevolve.controller`` also
  installs SIGINT/SIGTERM handlers via ``signal.signal()`` which raises
  ``ValueError`` outside the main thread. Both constraints are handled here by
  running the evolution inside a dedicated worker thread with signal
  registration temporarily neutered, so ``run_openevolve_workflow`` is safe to
  call from an async endpoint, from a thread-pool worker, or from plain
  synchronous code.
* By default the run is fully OFFLINE and deterministic: a model configured with
  ``name="mock", provider="mock"`` routes the library to its built-in
  ``MockLLM`` backend, so no API keys or network access are required. If the
  request supplies real LLM configuration (model name + api key) that is used
  instead.
* Failures are never silently swallowed into fake data: any problem raises
  ``OpenEvolveBridgeError``.

Follows CLAUDE.md principles: structured logging, explicit configuration,
UTC timestamps.
"""

from __future__ import annotations

import importlib
import os
import signal as _signal_module
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import structlog

logger = structlog.get_logger()


__all__ = [
    "OpenEvolveBridgeError",
    "run_openevolve_workflow",
    "build_initial_program",
    "build_evaluator",
    "LIBRARY_PATH_ENV",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_POPULATION_SIZE",
]


# Small defaults so an offline mock run finishes in seconds rather than minutes.
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_POPULATION_SIZE = 6
DEFAULT_RANDOM_SEED = 42

# Hard ceiling for a synchronous (request-scoped) evolution run.
DEFAULT_TIMEOUT_SECONDS = 300.0


class OpenEvolveBridgeError(RuntimeError):
    """Raised when the real OpenEvolve engine could not be driven to completion."""


# --------------------------------------------------------------------------- #
# Real-library resolution
# --------------------------------------------------------------------------- #
# Several routers in this service do ``sys.path.append(<core-projects>)`` so they
# can import sibling projects. That makes the *directory*
# ``core-projects/openevolve`` (which has no ``__init__.py``) importable as a
# namespace package named ``openevolve``, shadowing the real editable-installed
# library at ``core-projects/openevolve/openevolve``. Because setuptools'
# editable finder is appended to ``sys.meta_path`` (after the default
# ``PathFinder``), ``sys.path`` wins and ``import openevolve`` silently resolves
# to the wrong thing. The helpers below detect that and repair it.

LIBRARY_PATH_ENV = "OPENEVOLVE_LIBRARY_PATH"


def _is_real_openevolve(module: Any) -> bool:
    """True when ``module`` is the real library package (regular pkg with api.py)."""
    module_file = getattr(module, "__file__", None)
    if not module_file:
        # Namespace package (no __init__.py) -> not the real library.
        return False
    try:
        return (Path(module_file).parent / "api.py").is_file()
    except Exception:  # pragma: no cover - defensive
        return False


def _candidate_library_roots() -> List[Path]:
    """Directories that may contain the real ``openevolve`` package."""
    roots: List[Path] = []

    env_root = os.getenv(LIBRARY_PATH_ENV)
    if env_root:
        roots.append(Path(env_root))

    here = Path(__file__).resolve()
    # .../core-projects/BubbleLab/services/openevolve-api/core/openevolve_bridge.py
    # parents[4] == core-projects  ->  core-projects/openevolve
    for depth in (4, 3, 5, 6):
        try:
            roots.append(here.parents[depth] / "openevolve")
        except IndexError:
            continue

    unique: List[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _ensure_real_openevolve_importable() -> None:
    """Make ``import openevolve`` resolve to the real library, repairing if needed."""
    module = sys.modules.get("openevolve")
    if module is None:
        try:
            module = importlib.import_module("openevolve")
        except ImportError:
            module = None

    if module is not None and _is_real_openevolve(module):
        return

    shadowed_path = list(getattr(module, "__path__", [])) if module is not None else []

    for root in _candidate_library_roots():
        if not (root / "openevolve" / "__init__.py").is_file():
            continue
        if not (root / "openevolve" / "api.py").is_file():
            continue

        # Drop the shadowed modules so the import machinery re-resolves them.
        for name in [
            n for n in list(sys.modules) if n == "openevolve" or n.startswith("openevolve.")
        ]:
            del sys.modules[name]

        root_str = str(root)
        while root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        importlib.invalidate_caches()

        try:
            repaired = importlib.import_module("openevolve")
        except ImportError:
            continue

        if _is_real_openevolve(repaired):
            logger.warning(
                "openevolve_import_path_repaired",
                library_root=root_str,
                shadowed_by=shadowed_path,
            )
            return

    # Leave things as they are; the caller raises a clear OpenEvolveBridgeError.


# --------------------------------------------------------------------------- #
# Program / evaluator synthesis
# --------------------------------------------------------------------------- #

# The library requires an ``# EVOLVE-BLOCK-START`` / ``# EVOLVE-BLOCK-END``
# region marking the mutable part of the program.
_PROGRAM_TEMPLATE = '''"""Evolved candidate for: {problem}

Synthesized by the openevolve-api bridge at {timestamp}.
{context_block}"""

# EVOLVE-BLOCK-START
TARGET = 42


def solve(x):
    """Candidate solution under evolution."""
    return x * 2


def score_hint():
    """Auxiliary knob the optimizer may tune."""
    return 1
# EVOLVE-BLOCK-END


def run():
    """Stable entry point used by the evaluator."""
    return solve(21)
'''


# The evaluator must expose ``evaluate(program_path) -> dict`` of numeric metrics.
# It is written to a temp file and imported by the library, so it must be fully
# self-contained (no closures over this module).
_EVALUATOR_TEMPLATE = '''"""Evaluator synthesized by the openevolve-api bridge."""

import importlib.util


def _load(program_path):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not build import spec for candidate program")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(program_path):
    """Score a candidate program. Always returns a dict of numeric metrics."""
    try:
        module = _load(program_path)
    except Exception as exc:  # unparsable / crashing candidate
        return {{
            "combined_score": 0.0,
            "correctness": 0.0,
            "runs": 0.0,
            "error": str(exc)[:200],
        }}

    target = getattr(module, "TARGET", {target})

    try:
        value = module.run()
    except Exception as exc:
        return {{
            "combined_score": 0.0,
            "correctness": 0.0,
            "runs": 0.0,
            "error": str(exc)[:200],
        }}

    if not isinstance(value, (int, float)):
        return {{"combined_score": 0.0, "correctness": 0.0, "runs": 1.0}}

    # Closeness to the target, in [0, 1].
    distance = abs(float(value) - float(target))
    correctness = 1.0 / (1.0 + distance)

    # Mild bonus for keeping the program compact.
    try:
        with open(program_path, "r", encoding="utf-8") as handle:
            line_count = len(handle.readlines())
    except Exception:
        line_count = 0
    brevity = 1.0 / (1.0 + max(0, line_count - 20) / 50.0)

    combined = 0.9 * correctness + 0.1 * brevity

    return {{
        "combined_score": round(combined, 6),
        "correctness": round(correctness, 6),
        "brevity": round(brevity, 6),
        "value": float(value),
        "runs": 1.0,
    }}
'''


def build_initial_program(request: Dict[str, Any]) -> str:
    """Build the initial program source for an evolution run.

    An explicit ``initial_program`` in the request always wins; otherwise a
    minimal placeholder program with a valid EVOLVE-BLOCK region is synthesized
    from the problem statement.
    """
    explicit = request.get("initial_program") or request.get("program")
    if isinstance(explicit, str) and explicit.strip():
        return explicit

    problem = _problem_statement(request)
    context = request.get("context") or ""
    context_block = f"\nContext: {context}\n" if str(context).strip() else ""

    return _PROGRAM_TEMPLATE.format(
        problem=str(problem).replace('"""', "'''")[:500],
        timestamp=datetime.now(timezone.utc).isoformat(),
        context_block=str(context_block).replace('"""', "'''")[:1000],
    )


def build_evaluator(request: Dict[str, Any]) -> str:
    """Build the evaluator source (must define ``evaluate(program_path)``)."""
    explicit = request.get("evaluator") or request.get("evaluator_code")
    if isinstance(explicit, str) and explicit.strip():
        if "def evaluate" not in explicit:
            raise OpenEvolveBridgeError(
                "Supplied evaluator code must define an 'evaluate(program_path)' function"
            )
        return explicit

    target = _coerce_number(request.get("target"), default=42)
    return _EVALUATOR_TEMPLATE.format(target=target)


def _problem_statement(request: Dict[str, Any]) -> str:
    for key in ("problem_statement", "problem", "prompt", "description"):
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Evolve a trivial function toward its target value"


def _coerce_number(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int, minimum: int = 1, maximum: int = 10_000) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def _extract_parameters(request: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the evolution parameters from the various request shapes."""
    params: Dict[str, Any] = {}
    for key in ("parameters", "evolution_params", "params"):
        candidate = request.get(key)
        if isinstance(candidate, dict):
            params.update(candidate)
    metadata = request.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("evolution_params"), dict):
        for k, v in metadata["evolution_params"].items():
            params.setdefault(k, v)
    # Top-level shorthands.
    for key in ("max_iterations", "iterations", "population_size", "temperature", "seed"):
        if key in request and key not in params:
            params[key] = request[key]
    return params


def _build_llm_models(request: Dict[str, Any], params: Dict[str, Any]) -> tuple[List[Any], str]:
    """Return ``(models, mode)`` where mode is ``"mock"`` or ``"live"``.

    A live model is only used when the request explicitly supplies a real model
    name AND an API key; otherwise the deterministic offline mock backend is
    selected so the service never needs credentials or network access.
    """
    from openevolve.config import LLMModelConfig

    llm_request = request.get("llm") or request.get("llm_config") or {}
    if not isinstance(llm_request, dict):
        llm_request = {}

    # Explicit multi-model ensemble.
    raw_models = llm_request.get("models") or request.get("models")
    model_dicts: List[Dict[str, Any]] = []
    if isinstance(raw_models, list):
        model_dicts = [m for m in raw_models if isinstance(m, dict)]
    elif llm_request:
        model_dicts = [llm_request]

    live_models: List[Any] = []
    for entry in model_dicts:
        name = entry.get("name") or entry.get("model")
        api_key = entry.get("api_key")
        provider = entry.get("provider")
        if not name:
            continue
        # A mock request stays mock.
        if str(name).lower().startswith("mock") or str(provider or "").lower() == "mock":
            continue
        if not api_key:
            # No credentials -> cannot run live.
            continue
        live_models.append(
            LLMModelConfig(
                name=str(name),
                api_key=str(api_key),
                api_base=entry.get("api_base") or "https://api.openai.com/v1",
                provider=provider,
                temperature=_coerce_number(
                    entry.get("temperature", params.get("temperature")), 0.7
                ),
                max_tokens=_coerce_int(
                    entry.get("max_tokens", params.get("max_tokens")), 4096, 1, 200_000
                ),
                weight=_coerce_number(entry.get("weight"), 1.0),
            )
        )

    if live_models:
        return live_models, "live"

    return [LLMModelConfig(name="mock", provider="mock")], "mock"


def _build_config(request: Dict[str, Any], params: Dict[str, Any]) -> tuple[Any, int, str]:
    """Build the library ``Config``. Returns ``(config, iterations, llm_mode)``."""
    from openevolve.config import Config

    iterations = _coerce_int(
        params.get("max_iterations", params.get("iterations")),
        DEFAULT_MAX_ITERATIONS,
        minimum=1,
        maximum=1000,
    )
    population_size = _coerce_int(
        params.get("population_size"), DEFAULT_POPULATION_SIZE, minimum=2, maximum=1000
    )

    config = Config()
    models, llm_mode = _build_llm_models(request, params)
    config.llm.models = list(models)
    config.llm.evaluator_models = list(models)

    config.max_iterations = iterations
    config.database.population_size = population_size

    seed = params.get("seed")
    seed_value = DEFAULT_RANDOM_SEED if seed in (None, -1) else _coerce_int(seed, DEFAULT_RANDOM_SEED, 0, 2**31 - 1)
    config.random_seed = seed_value
    config.database.random_seed = seed_value

    # Keep the library quiet inside a request/worker thread.
    config.log_level = str(params.get("log_level") or "WARNING")

    # The synthesized evaluator is single-stage; cascade would just warn.
    try:
        config.evaluator.cascade_evaluation = False
    except Exception:  # pragma: no cover - defensive
        pass

    # Islands must not outnumber the population.
    try:
        if config.database.num_islands > max(1, population_size // 2):
            config.database.num_islands = max(1, population_size // 2)
    except Exception:  # pragma: no cover - defensive
        pass

    return config, iterations, llm_mode


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #

@contextmanager
def _silence_signal_registration() -> Iterator[None]:
    """Allow ``run_evolution()`` to execute outside the main thread.

    ``openevolve.controller.run`` registers SIGINT/SIGTERM handlers, and
    ``signal.signal()`` raises ``ValueError`` when called off the main thread.
    A headless worker thread is not interactive, so registration is temporarily
    turned into a no-op and restored afterwards.
    """
    original = _signal_module.signal
    _signal_module.signal = lambda signum, handler: None  # type: ignore[assignment]
    try:
        yield
    finally:
        _signal_module.signal = original


def _run_isolated(fn, timeout: float):
    """Run ``fn`` in a dedicated thread (fresh event loop, no signal handlers)."""
    box: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            with _silence_signal_registration():
                box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - propagated to caller below
            box["error"] = exc

    thread = threading.Thread(target=_worker, name="openevolve-bridge", daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise OpenEvolveBridgeError(
            f"OpenEvolve run exceeded timeout of {timeout:.0f}s"
        )
    if "error" in box:
        raise OpenEvolveBridgeError(
            f"OpenEvolve run failed: {type(box['error']).__name__}: {box['error']}"
        ) from box["error"]
    if "result" not in box:
        raise OpenEvolveBridgeError("OpenEvolve run produced no result")
    return box["result"]


def _json_safe(value: Any) -> Any:
    """Recursively coerce values into JSON-serializable forms."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _mock_call_count() -> Optional[int]:
    """Number of deterministic mock LLM completions produced so far (if available)."""
    try:
        from openevolve.llm.mock import total_calls

        return int(total_calls())
    except Exception:  # pragma: no cover - library without the counter
        return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_openevolve_workflow(request: Dict[str, Any]) -> Dict[str, Any]:
    """Drive the REAL OpenEvolve engine for an evolutionary workflow request.

    Args:
        request: Workflow request dict. All keys are optional:
            - ``system`` / ``workflow_type``: informational (e.g. ``"evolutionary"``).
            - ``problem_statement`` / ``problem`` / ``prompt``: what to evolve.
            - ``context``: extra constraints, embedded in the program docstring.
            - ``initial_program`` / ``program``: explicit program source. Must
              contain an ``# EVOLVE-BLOCK-START`` / ``# EVOLVE-BLOCK-END`` region
              (the library wraps the whole file if absent).
            - ``evaluator`` / ``evaluator_code``: explicit evaluator source; must
              define ``evaluate(program_path) -> dict``.
            - ``parameters`` (or ``evolution_params`` / ``metadata.evolution_params``):
              ``max_iterations``, ``population_size``, ``temperature``,
              ``max_tokens``, ``seed``, ``log_level``.
            - ``llm``: ``{"name"/"model", "api_key", "api_base", "provider", ...}``
              or ``{"models": [...]}``. Used only when a name AND api_key are
              present; otherwise the offline deterministic mock LLM is used.
            - ``timeout_seconds``: wall-clock ceiling for the run.

    Returns:
        Dict with the bridge contract::

            {
                "best_score": float,
                "best_code": str,
                "metrics": dict,
                "generations": int,
                # additive diagnostics
                "engine": "openevolve",
                "engine_entrypoint": "openevolve.api.run_evolution",
                "llm_mode": "mock" | "live",
                "iterations": int,
                "population_size": int,
                "mock_llm_calls": int | None,
                "duration_seconds": float,
                "started_at": str, "completed_at": str,
            }

    Raises:
        OpenEvolveBridgeError: if the library is missing or the run fails. The
            bridge never fabricates results.
    """
    if not isinstance(request, dict):
        raise OpenEvolveBridgeError(f"request must be a dict, got {type(request).__name__}")

    started_at = datetime.now(timezone.utc)
    wall_start = time.time()

    # Import the REAL library. Kept inside the function so importing this module
    # never hard-fails the service.
    _ensure_real_openevolve_importable()
    try:
        from openevolve.api import run_evolution
    except ImportError as exc:
        raise OpenEvolveBridgeError(
            "The 'openevolve' library is not importable. Install it with: "
            "pip install -e ../../../openevolve "
            f"(or set {LIBRARY_PATH_ENV} to the library repo root). "
            f"Underlying error: {exc}"
        ) from exc

    params = _extract_parameters(request)

    try:
        initial_program = build_initial_program(request)
        evaluator = build_evaluator(request)
        config, iterations, llm_mode = _build_config(request, params)
    except OpenEvolveBridgeError:
        raise
    except Exception as exc:
        raise OpenEvolveBridgeError(f"Failed to prepare OpenEvolve inputs: {exc}") from exc

    timeout = _coerce_number(request.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    calls_before = _mock_call_count()

    logger.info(
        "openevolve_bridge_run_started",
        system=request.get("system") or request.get("workflow_type"),
        llm_mode=llm_mode,
        iterations=iterations,
        population_size=config.database.population_size,
        program_chars=len(initial_program),
    )

    result = _run_isolated(
        lambda: run_evolution(
            initial_program=initial_program,
            evaluator=evaluator,
            config=config,
            iterations=iterations,
            cleanup=True,
        ),
        timeout=timeout,
    )

    best_code = getattr(result, "best_code", "") or ""
    metrics = _json_safe(getattr(result, "metrics", {}) or {})
    best_score = _coerce_number(getattr(result, "best_score", 0.0), 0.0)

    if not best_code:
        raise OpenEvolveBridgeError(
            "OpenEvolve returned an empty best program; the run did not produce a result"
        )

    best_program = getattr(result, "best_program", None)
    generations = _coerce_int(getattr(best_program, "generation", None), iterations, 0, 10**6)

    calls_after = _mock_call_count()
    mock_calls = None
    if calls_before is not None and calls_after is not None:
        mock_calls = calls_after - calls_before

    completed_at = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "best_score": best_score,
        "best_code": best_code,
        "metrics": metrics,
        "generations": generations,
        # Additive diagnostics proving which engine actually ran.
        "engine": "openevolve",
        "engine_entrypoint": "openevolve.api.run_evolution",
        "llm_mode": llm_mode,
        "iterations": iterations,
        "population_size": config.database.population_size,
        "mock_llm_calls": mock_calls,
        "duration_seconds": round(time.time() - wall_start, 3),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
    }

    logger.info(
        "openevolve_bridge_run_completed",
        best_score=best_score,
        generations=generations,
        llm_mode=llm_mode,
        mock_llm_calls=mock_calls,
        duration_seconds=payload["duration_seconds"],
    )

    return payload
