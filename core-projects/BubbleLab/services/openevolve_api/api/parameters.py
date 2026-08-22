"""
OpenEvolve parameter catalog router (mounted at ``/api/parameters``).

Serves the well-known OpenEvolve evolution parameters so the BubbleLab UI can
render real, representative configuration controls instead of 404ing.

Endpoints (paths relative to the ``/api`` prefix in ``main.py``):
    GET  /parameters/schema       -> { parameters: ParameterDefinition[] }
    GET  /parameters/defaults       -> { name: value, ... }
    GET  /parameters/categories     -> { categories: string[] }
    POST /parameters/validate       -> { valid, errors, warnings }

Data source: a curated, representative catalog of the real openevolve evolution
parameters (``max_iterations``, ``population_size``, ``temperature``,
``elite_ratio``, ...). Where the live library is importable, defaults are
enriched from ``openevolve.config.Config``; otherwise the well-known defaults
below are used. No random/fake values are generated.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover - structlog always present in this service
    import logging

    logger = logging.getLogger("openevolve_api.parameters")

router = APIRouter()


# --------------------------------------------------------------------------- #
# Catalog (real/representative openevolve evolution parameters)
# --------------------------------------------------------------------------- #
_CATEGORIES = ["evolution", "population", "llm", "database", "logging"]

_PARAMETER_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "max_iterations",
        "type": "int",
        "default": 100,
        "description": "Maximum number of evolution iterations (generations) to run.",
        "category": "evolution",
        "min_value": 1,
        "max_value": 10000,
        "required": False,
    },
    {
        "name": "population_size",
        "type": "int",
        "default": 50,
        "description": "Number of candidate programs maintained in the population.",
        "category": "population",
        "min_value": 2,
        "max_value": 1000,
        "required": False,
    },
    {
        "name": "temperature",
        "type": "float",
        "default": 0.7,
        "description": "Sampling temperature for the LLM mutation proposals.",
        "category": "llm",
        "min_value": 0.0,
        "max_value": 2.0,
        "required": False,
    },
    {
        "name": "elite_ratio",
        "type": "float",
        "default": 0.1,
        "description": "Fraction of the population carried over unchanged each iteration.",
        "category": "population",
        "min_value": 0.0,
        "max_value": 1.0,
        "required": False,
    },
    {
        "name": "exploration_rate",
        "type": "float",
        "default": 0.3,
        "description": "Probability of exploratory (random) mutations over exploitative ones.",
        "category": "evolution",
        "min_value": 0.0,
        "max_value": 1.0,
        "required": False,
    },
    {
        "name": "mutation_rate",
        "type": "float",
        "default": 0.2,
        "description": "Rate at which the mutation operator perturbs candidate programs.",
        "category": "evolution",
        "min_value": 0.0,
        "max_value": 1.0,
        "required": False,
    },
    {
        "name": "crossover_rate",
        "type": "float",
        "default": 0.5,
        "description": "Rate at which crossover combines two parent candidates.",
        "category": "evolution",
        "min_value": 0.0,
        "max_value": 1.0,
        "required": False,
    },
    {
        "name": "num_islands",
        "type": "int",
        "default": 1,
        "description": "Number of parallel island populations for diversity.",
        "category": "population",
        "min_value": 1,
        "max_value": 64,
        "required": False,
    },
    {
        "name": "seed",
        "type": "int",
        "default": 42,
        "description": "Global random seed for reproducible evolution runs.",
        "category": "database",
        "min_value": 0,
        "max_value": 2_147_483_647,
        "required": False,
    },
    {
        "name": "random_seed",
        "type": "int",
        "default": 42,
        "description": "Seed used for the evaluation/database sampling.",
        "category": "database",
        "min_value": 0,
        "max_value": 2_147_483_647,
        "required": False,
    },
    {
        "name": "max_tokens",
        "type": "int",
        "default": 4096,
        "description": "Maximum tokens the LLM may generate per mutation proposal.",
        "category": "llm",
        "min_value": 1,
        "max_value": 200_000,
        "required": False,
    },
    {
        "name": "log_level",
        "type": "str",
        "default": "WARNING",
        "description": "Logging verbosity for the evolution engine.",
        "category": "logging",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "required": False,
    },
]


def _enrich_defaults_from_library() -> None:
    """Best-effort enrichment of defaults from the real openevolve config."""
    try:
        from openevolve.config import Config

        cfg = Config()
        _overrides = {
            "max_iterations": getattr(cfg, "max_iterations", None),
            "population_size": cfg.database.population_size,
            "num_islands": cfg.database.num_islands,
            "log_level": str(cfg.log_level or "WARNING"),
        }
        for entry in _PARAMETER_CATALOG:
            value = _overrides.get(entry["name"])
            if value is not None:
                entry["default"] = value
    except Exception as exc:  # pragma: no cover - library optional at import time
        logger.debug("parameters_enrich_skipped", error=str(exc))


_enrich_defaults_from_library()


def _catalog() -> List[Dict[str, Any]]:
    return [dict(entry) for entry in _PARAMETER_CATALOG]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/parameters/schema")
async def get_parameter_schema() -> JSONResponse:
    return JSONResponse(
        {"parameters": _catalog()},
        headers={"Content-Type": "application/json"},
    )


@router.get("/parameters/defaults")
async def get_parameter_defaults() -> JSONResponse:
    defaults = {entry["name"]: entry["default"] for entry in _PARAMETER_CATALOG}
    return JSONResponse(defaults, headers={"Content-Type": "application/json"})


@router.get("/parameters/categories")
async def get_parameter_categories() -> JSONResponse:
    return JSONResponse(
        {"categories": list(_CATEGORIES)},
        headers={"Content-Type": "application/json"},
    )


def _coerce(value: Any, type_name: str) -> Any:
    if type_name == "int":
        return int(value)
    if type_name == "float":
        return float(value)
    return value


@router.post("/parameters/validate")
async def validate_parameters(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"valid": False, "errors": ["Body must be a JSON object"], "warnings": []},
            headers={"Content-Type": "application/json"},
        )

    by_name = {entry["name"]: entry for entry in _PARAMETER_CATALOG}
    errors: List[str] = []
    warnings: List[str] = []

    for name, value in body.items():
        spec = by_name.get(name)
        if spec is None:
            warnings.append(f"Unknown parameter '{name}' (not in catalog)")
            continue
        expected = spec["type"]
        try:
            coerced = _coerce(value, expected)
        except (TypeError, ValueError):
            errors.append(f"Parameter '{name}' must be of type {expected}")
            continue
        if spec.get("options") is not None and coerced not in spec["options"]:
            errors.append(
                f"Parameter '{name}' must be one of {spec['options']}, got {coerced!r}"
            )
        min_value = spec.get("min_value")
        max_value = spec.get("max_value")
        if min_value is not None and coerced < min_value:
            errors.append(f"Parameter '{name}' must be >= {min_value}")
        if max_value is not None and coerced > max_value:
            errors.append(f"Parameter '{name}' must be <= {max_value}")

    return JSONResponse(
        {"valid": len(errors) == 0, "errors": errors, "warnings": warnings},
        headers={"Content-Type": "application/json"},
    )
