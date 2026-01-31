"""
Complete Mathematical Knowledge API

Provides endpoints for:
- Z3 solving
- Lean proving
- Unified solving
- Knowledge management

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Configure logging
logger = logging.getLogger(__name__)

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Dummy classes
    class BaseModel:
        pass
    class HTTPException(Exception):
        pass

# Import connectors
try:
    from z3_solver_connector import get_z3_connector, Z3SolverConfig, Z3ResultStatus
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    from leanaide_real_connector import get_leanaide_connector
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False

try:
    from unified_math_bridge_complete import get_unified_bridge_complete, SolverSystem
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False

try:
    from z3_knowledge_complete import get_z3_knowledge_manager
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False


# =============================================================================
# Pydantic Models
# =============================================================================

class SolveZ3Request(BaseModel):
    """Request to solve with Z3."""
    content: str = Field(..., description="SMT-LIB content")
    timeout_ms: int = Field(30000, ge=1000, le=300000, description="Timeout in milliseconds")
    get_model: bool = Field(True, description="Return model if satisfiable")
    get_proof: bool = Field(True, description="Return proof if unsatisfiable")


class SolveZ3Response(BaseModel):
    """Response from Z3 solver."""
    status: str
    model: Optional[Dict[str, Any]]
    proof: Optional[str]
    solving_time_ms: float
    error: Optional[str]


class ProveLeanRequest(BaseModel):
    """Request to prove with Lean."""
    theorem: str = Field(..., description="Theorem statement")
    timeout_seconds: int = Field(300, ge=10, le=3600)
    auto_tactics: Optional[List[str]] = None


class ProveLeanResponse(BaseModel):
    """Response from Lean prover."""
    success: bool
    proof: Optional[str]
    error: Optional[str]
    execution_time_ms: float


class SolveUnifiedRequest(BaseModel):
    """Request for unified solving."""
    problem: str = Field(..., description="Problem statement")
    preferred_solver: str = Field("auto", description="auto, z3, lean, or hybrid")
    timeout_seconds: int = Field(300, ge=10, le=3600)
    require_consensus: bool = Field(False)


class SolveUnifiedResponse(BaseModel):
    """Response from unified solver."""
    result_status: str
    primary_solver: str
    result: Optional[Any]
    verified: bool
    consensus_status: Optional[str]
    solving_time_ms: float


class LearnRequest(BaseModel):
    """Request to learn from solution."""
    problem_statement: str
    constraints: List[str]
    result: str
    proof: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LearnResponse(BaseModel):
    """Response from learning."""
    success: bool
    items_learned: int
    features: Dict[str, Any]


class SearchRequest(BaseModel):
    """Request to search knowledge base."""
    query: str
    top_k: int = Field(5, ge=1, le=50)
    pattern_type: Optional[str] = None


class SearchResponse(BaseModel):
    """Response from search."""
    results: List[Dict[str, Any]]
    total_found: int


class StrategyRequest(BaseModel):
    """Request strategy recommendation."""
    problem_statement: str
    constraints: List[str]


class StrategyResponse(BaseModel):
    """Response with strategy recommendation."""
    strategy: Optional[str]
    confidence: float
    expected_time_ms: Optional[float]


# =============================================================================
# API Factory
# =============================================================================

def create_math_api() -> Optional["FastAPI"]:
    """Create complete mathematical knowledge API."""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available")
        return None
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        logger.info("Starting Mathematical Knowledge API")
        yield
        logger.info("Mathematical Knowledge API stopped")
    
    app = FastAPI(
        title="Mathematical Knowledge API",
        description="Complete API for Z3, LeanAIDE, and knowledge management",
        version="1.1.0",
        lifespan=lifespan
    )
    
    # ==================================================================
    # Health & Info
    # ==================================================================
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "components": {
                "z3": Z3_AVAILABLE,
                "leanaide": LEANAIDE_AVAILABLE,
                "bridge": BRIDGE_AVAILABLE,
                "knowledge": KNOWLEDGE_AVAILABLE
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @app.get("/")
    async def api_info():
        """API information."""
        return {
            "name": "Mathematical Knowledge API",
            "version": "1.1.0",
            "endpoints": [
                "/health",
                "/solve/z3",
                "/solve/lean",
                "/solve/unified",
                "/knowledge/learn",
                "/knowledge/search",
                "/knowledge/strategy"
            ]
        }
    
    # ==================================================================
    # Solver Endpoints
    # ==================================================================
    
    @app.post("/solve/z3", response_model=SolveZ3Response)
    async def solve_z3(request: SolveZ3Request):
        """Solve problem using Z3 SMT solver."""
        if not Z3_AVAILABLE:
            raise HTTPException(status_code=503, detail="Z3 not available")
        
        try:
            z3 = get_z3_connector()
            config = Z3SolverConfig(
                timeout_ms=request.timeout_ms,
                model_generation=request.get_model,
                proof_generation=request.get_proof
            )
            
            result = await z3.solve_smtlib(request.content, config)
            
            return SolveZ3Response(
                status=result.status.value if hasattr(result.status, 'value') else str(result.status),
                model=result.model if request.get_model else None,
                proof=result.proof if request.get_proof else None,
                solving_time_ms=result.solving_time_ms,
                error=result.error_message
            )
        except Exception as e:
            logger.error(f"Z3 solving failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/solve/lean", response_model=ProveLeanResponse)
    async def solve_lean(request: ProveLeanRequest):
        """Prove theorem using LeanAIDE."""
        if not LEANAIDE_AVAILABLE:
            raise HTTPException(status_code=503, detail="LeanAIDE not available")
        
        try:
            connector = await get_leanaide_connector()
            
            result = await connector.prove_theorem(
                theorem=request.theorem,
                timeout=request.timeout_seconds,
                auto_tactics=request.auto_tactics or ["simp", "rfl", "tauto"]
            )
            
            return ProveLeanResponse(
                success=result.get("success", False),
                proof=result.get("proof"),
                error=result.get("error"),
                execution_time_ms=result.get("execution_time_ms", 0.0)
            )
        except Exception as e:
            logger.error(f"Lean proving failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/solve/unified", response_model=SolveUnifiedResponse)
    async def solve_unified(request: SolveUnifiedRequest):
        """Solve using unified bridge with intelligent solver selection."""
        if not BRIDGE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Unified bridge not available")
        
        try:
            bridge = await get_unified_bridge_complete()
            
            solver_map = {
                "auto": SolverSystem.AUTO,
                "z3": SolverSystem.Z3,
                "lean": SolverSystem.LEANAIDE,
                "hybrid": SolverSystem.HYBRID
            }
            
            result = await bridge.solve(
                problem=request.problem,
                preferred_solver=solver_map.get(request.preferred_solver, SolverSystem.AUTO),
                timeout=request.timeout_seconds
            )
            
            return SolveUnifiedResponse(
                result_status=result.get("result_status", "unknown"),
                primary_solver=result.get("primary_solver", "unknown"),
                result=result.get("result"),
                verified=result.get("verified", False),
                consensus_status=result.get("consensus_status"),
                solving_time_ms=result.get("metadata", {}).get("solving_time_ms", 0.0)
            )
        except Exception as e:
            logger.error(f"Unified solving failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==================================================================
    # Knowledge Endpoints
    # ==================================================================
    
    @app.post("/knowledge/learn", response_model=LearnResponse)
    async def learn_from_solution(request: LearnRequest):
        """Learn from a solved problem."""
        if not KNOWLEDGE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Knowledge manager not available")
        
        try:
            manager = await get_z3_knowledge_manager()
            
            result = await manager.learn_from_solution(
                problem_statement=request.problem_statement,
                constraints=request.constraints,
                result=request.result,
                proof=request.proof,
                metadata=request.metadata
            )
            
            return LearnResponse(
                success=True,
                items_learned=result.get("items_learned", 0),
                features=result.get("features", {})
            )
        except Exception as e:
            logger.error(f"Learning failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/knowledge/search", response_model=SearchResponse)
    async def search_knowledge(request: SearchRequest):
        """Search knowledge base for similar solutions."""
        if not KNOWLEDGE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Knowledge manager not available")
        
        try:
            manager = await get_z3_knowledge_manager()
            
            results = await manager.find_similar_solutions(
                problem_statement=request.query,
                constraints=[],
                top_k=request.top_k
            )
            
            return SearchResponse(
                results=results,
                total_found=len(results)
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/knowledge/strategy")
    async def get_strategy(request: StrategyRequest):
        """Get recommended strategy for a problem."""
        if not KNOWLEDGE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Knowledge manager not available")
        
        try:
            manager = await get_z3_knowledge_manager()
            
            strategy = await manager.get_recommended_strategy(
                problem_statement=request.problem_statement,
                constraints=request.constraints
            )
            
            return StrategyResponse(
                strategy=strategy.get("strategy"),
                confidence=strategy.get("confidence", 0.0),
                expected_time_ms=strategy.get("expected_time_ms")
            )
        except Exception as e:
            logger.error(f"Strategy recommendation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/knowledge/stats")
    async def get_knowledge_stats():
        """Get knowledge base statistics."""
        if not KNOWLEDGE_AVAILABLE:
            return {"error": "Knowledge manager not available"}
        
        try:
            manager = await get_z3_knowledge_manager()
            stats = manager.get_statistics()
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Global instance
math_api = create_math_api() if FASTAPI_AVAILABLE else None


# Example usage
if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        exit(1)
    
    import uvicorn
    
    app = create_math_api()
    if app:
        uvicorn.run(app, host="0.0.0.0", port=8765)
    else:
        print("Failed to create API")
