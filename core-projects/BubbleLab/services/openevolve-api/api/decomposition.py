"""
Decomposition API Routes for OpenEvolve

Provides problem analysis + decomposition planning.
"""

from pathlib import Path
import sys
from typing import Optional, Dict, Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..database import get_setting
from ..models import DecompositionDefaults

logger = structlog.get_logger()
router = APIRouter()

# Ensure repo root is on sys.path so decomposition modules can be imported
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

try:
    from problem_analyzer import ProblemAnalyzer
    from decomposition_engine import DecompositionEngine
except Exception as exc:
    logger.warning("decomposition_modules_unavailable", error=str(exc))
    ProblemAnalyzer = None
    DecompositionEngine = None


class DecompositionPlanRequest(BaseModel):
    problem_statement: str
    title: Optional[str] = None
    strategy: Optional[str] = None
    enable_adaptive_selection: Optional[bool] = None
    maker_config: Optional[Dict[str, Any]] = None
    openevolve_client_config: Optional[Dict[str, Any]] = None


_DECOMPOSITION_DEFAULTS_KEY = "decomposition_defaults"


def _get_decomposition_defaults() -> DecompositionDefaults:
    config_data = get_setting(_DECOMPOSITION_DEFAULTS_KEY)
    if config_data:
        try:
            return DecompositionDefaults(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_decomposition_defaults")
    return DecompositionDefaults()


def _ensure_decomposition_available() -> None:
    if ProblemAnalyzer is None or DecompositionEngine is None:
        raise HTTPException(status_code=503, detail="Decomposition engine is not available")


@router.post("/plan")
async def create_decomposition_plan(request: DecompositionPlanRequest):
    _ensure_decomposition_available()

    defaults = _get_decomposition_defaults()
    strategy = request.strategy or defaults.strategy
    enable_adaptive_selection = (
        request.enable_adaptive_selection
        if request.enable_adaptive_selection is not None
        else defaults.enable_adaptive_selection
    )
    maker_config = {**(defaults.maker_config or {}), **(request.maker_config or {})}
    client_config = {
        **(defaults.openevolve_client_config or {}),
        **(request.openevolve_client_config or {}),
    }

    analyzer = ProblemAnalyzer(openevolve_client_config=client_config)
    problem = analyzer.analyze_problem(request.problem_statement, title=request.title or "")
    if problem is None:
        raise HTTPException(status_code=500, detail="Problem analysis failed")

    engine = DecompositionEngine(
        problem_analyzer=analyzer,
        enable_adaptive_selection=enable_adaptive_selection,
        maker_config=maker_config,
    )

    try:
        plan = engine.decompose(problem, strategy=strategy)
    except Exception as exc:
        logger.error("decomposition_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Decomposition failed")

    plan.metadata = {
        **plan.metadata,
        "problem_statement": request.problem_statement,
    }

    return {
        "problem": problem.to_dict(),
        "plan": plan.to_dict(),
    }
