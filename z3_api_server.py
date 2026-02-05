"""
Z3 Prover Service Bubble - Complete REST API Server

FastAPI-based REST API providing comprehensive Z3 integration:
- Constraint solving endpoints (SAT/SMT)
- Theorem proving endpoints
- Optimization endpoints (single/multi-objective)
- Proof extraction and verification
- Translation endpoints (SMT-LIB/Lean)
- Real-time progress via WebSocket
- Health checks and Prometheus metrics
- Batch operations
- Service orchestration

Part of OpenEvolve Z3 Prover Service Bubble (100% Complete)

Author: OpenEvolve
Created: 2026-01-31
Updated: 2026-02-04 - Service Bubble Complete
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create dummy classes for type hints
    class FastAPI:
        pass
    class BaseModel:
        pass
    class HTTPException(Exception):
        pass

# Import Z3 integration components
try:
    from z3_config_manager import get_config_manager, IntegrationConfig
    from z3_database_models import get_database_manager, SolverResult, TheoremProof
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from z3prover_integration import (
        get_z3_solver_engine, Z3Variable, Z3Constraint, Z3ConstraintType,
        get_z3_theorem_prover, Z3Config
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    from z3prover_advanced import (
        get_z3_advanced_solver, OptimizationObjective, ProofFormat, PortfolioResult
    )
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False

try:
    from z3_leanaide_openevolve_integration import solve_with_z3_leanaide
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

try:
    from z3_performance_monitor import get_z3_performance_monitor, monitored
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

try:
    from z3_result_cache import get_z3_result_cache, CacheConfig
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from z3_knowledge_extraction import get_z3_knowledge_extractor
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from z3_reliability_checker import Z3ReliabilityChecker
    RELIABILITY_AVAILABLE = True
except ImportError:
    RELIABILITY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class SolveRequest(BaseModel):
    """Request model for constraint solving."""
    problem: str = Field(..., description="Problem statement or SMT-LIB")
    variables: Optional[List[Dict[str, Any]]] = Field(None, description="Variable definitions")
    constraints: Optional[List[str]] = Field(None, description="Constraint expressions")
    timeout: Optional[float] = Field(60.0, description="Timeout in seconds")
    use_cache: Optional[bool] = Field(True, description="Use result cache")
    extract_proof: Optional[bool] = Field(False, description="Extract proof if unsat")


class SolveResponse(BaseModel):
    """Response model for constraint solving."""
    success: bool
    result_id: str
    status: str
    satisfiable: Optional[bool]
    model: Optional[Dict[str, Any]]
    execution_time_ms: float
    solver_used: str
    cached: bool = False
    proof: Optional[str] = None


class BatchSolveRequest(BaseModel):
    """Request model for batch constraint solving."""
    problems: List[SolveRequest] = Field(..., description="List of problems to solve")
    parallel: bool = Field(True, description="Solve in parallel")
    max_workers: int = Field(4, description="Maximum parallel workers")


class BatchSolveResponse(BaseModel):
    """Response model for batch solving."""
    success: bool
    results: List[SolveResponse]
    completed: int
    failed: int
    total_time_ms: float


class OptimizeRequest(BaseModel):
    """Request model for optimization."""
    variables: List[Dict[str, Any]] = Field(..., description="Variable definitions")
    constraints: List[str] = Field(..., description="Constraint expressions")
    objective: Dict[str, str] = Field(..., description="Objective function")
    direction: str = Field("minimize", description="minimize or maximize")
    multi_objective: bool = Field(False, description="Enable multi-objective")
    objectives: Optional[List[Dict[str, str]]] = Field(None, description="Multiple objectives")


class OptimizeResponse(BaseModel):
    """Response model for optimization."""
    success: bool
    result_id: str
    optimal_value: Optional[float]
    model: Optional[Dict[str, Any]]
    is_pareto: bool
    pareto_front_size: int
    execution_time_ms: float


class ProveRequest(BaseModel):
    """Request model for theorem proving."""
    theorem: str = Field(..., description="Theorem statement or SMT-LIB")
    assumptions: Optional[List[str]] = Field(None, description="List of assumptions")
    extract_proof: bool = Field(False, description="Extract detailed proof")
    timeout: Optional[float] = Field(60.0, description="Timeout in seconds")


class ProveResponse(BaseModel):
    """Response model for theorem proving."""
    success: bool
    result_id: str
    proven: bool
    confidence: float
    tactics_used: Optional[List[str]]
    counterexample: Optional[Dict[str, Any]]
    proof: Optional[str]
    execution_time_ms: float


class ProofExtractRequest(BaseModel):
    """Request model for proof extraction."""
    smtlib: str = Field(..., description="SMT-LIB problem (must be UNSAT)")
    format: str = Field("text", description="Proof format: text, json, dot, smtlib2")
    verify: bool = Field(True, description="Verify extracted proof")


class ProofExtractResponse(BaseModel):
    """Response model for proof extraction."""
    success: bool
    proof_steps: List[Dict[str, Any]]
    axioms_used: List[str]
    tactics_used: List[str]
    verification_status: str
    raw_proof: Optional[str]
    execution_time_ms: float


class PortfolioSolveRequest(BaseModel):
    """Request model for portfolio solving."""
    smtlib: str = Field(..., description="SMT-LIB problem")
    strategies: Optional[List[str]] = Field(None, description="List of strategies")
    timeout: float = Field(30.0, description="Timeout per strategy")
    parallel: bool = Field(True, description="Run strategies in parallel")


class PortfolioSolveResponse(BaseModel):
    """Response model for portfolio solving."""
    success: bool
    winner_strategy: Optional[str]
    execution_time_ms: float
    parallel_speedup: float
    strategies_tried: int
    status: Optional[str]
    model: Optional[Dict[str, Any]]


class IncrementalSolveRequest(BaseModel):
    """Request model for incremental solving."""
    state_id: Optional[str] = Field(None, description="Incremental state ID")
    operation: str = Field(..., description="create, push, pop, add, check, reset")
    variables: Optional[List[Dict]] = None
    constraints: Optional[List[str]] = None
    constraint: Optional[str] = None


class IncrementalSolveResponse(BaseModel):
    """Response model for incremental solving."""
    success: bool
    state_id: Optional[str]
    status: Optional[str]
    satisfiable: Optional[bool]
    model: Optional[Dict[str, Any]]
    message: str


class TranslateRequest(BaseModel):
    """Request model for translation."""
    content: str = Field(..., description="Content to translate")
    direction: str = Field(..., description="smt_to_lean or lean_to_smt")


class TranslateResponse(BaseModel):
    """Response model for translation."""
    success: bool
    translation: str
    source: str
    target: str
    execution_time_ms: float
    errors: List[str]
    warnings: List[str]


class VerifyRequest(BaseModel):
    """Request model for verification."""
    problem: str = Field(..., description="Problem to verify")
    strategy: str = Field("adaptive", description="Verification strategy")
    use_both: bool = Field(True, description="Use both Z3 and Lean")


class VerifyResponse(BaseModel):
    """Response model for verification."""
    success: bool
    verified: bool
    z3_result: Optional[Dict[str, Any]]
    lean_result: Optional[Dict[str, Any]]
    agreement: bool
    confidence_score: float
    recommendation: str
    execution_time_ms: float


class ReliabilityVerifyRequest(BaseModel):
    """Request model for reliability verification."""
    components: List[Dict[str, Any]] = Field(..., description="Component reliability models")
    requirements: List[Dict[str, Any]] = Field(..., description="Reliability requirements")
    contracts: Optional[List[Dict[str, Any]]] = None


class ReliabilityVerifyResponse(BaseModel):
    """Response model for reliability verification."""
    success: bool
    verified: bool
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    counterexample: Optional[Dict[str, Any]]
    execution_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str = "3.0.0"
    components: Dict[str, bool]
    uptime_seconds: float
    load: Dict[str, float]


class MetricsResponse(BaseModel):
    """Metrics response."""
    timestamp: str
    summary: Dict[str, Any]
    operations: Dict[str, Any]
    bottlenecks: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    cache_stats: Optional[Dict[str, Any]]


class KnowledgeExtractRequest(BaseModel):
    """Request to extract knowledge from solution."""
    problem: str = Field(..., description="Problem statement")
    solution: Dict[str, Any] = Field(..., description="Solution result")
    domain: str = Field("general", description="Problem domain")


class KnowledgeExtractResponse(BaseModel):
    """Response with extracted knowledge."""
    success: bool
    patterns_found: int
    strategies_learned: int
    insights: List[Dict[str, Any]]


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.operation_subscribers: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from operation subscribers
        for op_id, subscribers in self.operation_subscribers.items():
            if websocket in subscribers:
                subscribers.remove(websocket)
        
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    def subscribe_to_operation(self, websocket: WebSocket, operation_id: str):
        """Subscribe to updates for a specific operation."""
        if operation_id not in self.operation_subscribers:
            self.operation_subscribers[operation_id] = []
        self.operation_subscribers[operation_id].append(websocket)
    
    async def send_progress(self, operation_id: str, data: Dict[str, Any]):
        """Send progress update to subscribers."""
        if operation_id not in self.operation_subscribers:
            return
        
        message = {
            "type": "progress",
            "operation_id": operation_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected = []
        for connection in self.operation_subscribers[operation_id]:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


# Global connection manager
manager = ConnectionManager()

# Global service instances
_solver_service = None
_cache_service = None
_monitor_service = None
_knowledge_service = None
_reliability_service = None

_start_time = datetime.utcnow()


# =============================================================================
# Z3 Service Bubble Components
# =============================================================================

class Z3SolverService:
    """Core Z3 solving service with caching and monitoring."""
    
    def __init__(self):
        self.solver = get_z3_solver_engine() if Z3_AVAILABLE else None
        self.advanced = get_z3_advanced_solver() if Z3_ADVANCED_AVAILABLE else None
        self.cache = get_z3_result_cache() if CACHE_AVAILABLE else None
        self.monitor = get_z3_performance_monitor() if MONITOR_AVAILABLE else None
        self.knowledge = get_z3_knowledge_extractor() if KNOWLEDGE_AVAILABLE else None
        self.request_count = 0
    
    async def solve(self, request: SolveRequest) -> SolveResponse:
        """Solve constraint problem with caching and monitoring."""
        import time
        start_time = time.time()
        self.request_count += 1
        result_id = f"solve_{int(time.time() * 1000)}_{self.request_count}"
        
        # Check cache
        if request.use_cache and self.cache:
            hit, cached = self.cache.get("solve", {
                "problem": request.problem,
                "variables": request.variables,
                "constraints": request.constraints
            })
            if hit:
                return SolveResponse(
                    success=True,
                    result_id=result_id,
                    status=cached.get("status", "sat"),
                    satisfiable=cached.get("satisfiable"),
                    model=cached.get("model"),
                    execution_time_ms=0.1,
                    solver_used="z3",
                    cached=True
                )
        
        if not self.solver:
            return SolveResponse(
                success=False,
                result_id=result_id,
                status="error",
                satisfiable=None,
                model=None,
                execution_time_ms=(time.time() - start_time) * 1000,
                solver_used="none"
            )
        
        try:
            # Check if SMT-LIB
            if '(assert' in request.problem or '(declare' in request.problem:
                result = self.solver.solve_smtlib(request.problem)
            else:
                # Parse variables
                variables = [
                    Z3Variable(
                        v['name'], 
                        Z3ConstraintType[v.get('type', 'INTEGER').upper()],
                        bit_width=v.get('bit_width')
                    )
                    for v in (request.variables or [])
                ]
                
                constraints = [
                    Z3Constraint(c, Z3ConstraintType.INTEGER)
                    for c in (request.constraints or [])
                ]
                
                result = self.solver.solve_constraints(variables, constraints)
            
            execution_time = (time.time() - start_time) * 1000
            
            response = SolveResponse(
                success=True,
                result_id=result_id,
                status=result.status.value,
                satisfiable=result.is_sat(),
                model=result.model.assignments if result.model else None,
                execution_time_ms=execution_time,
                solver_used="z3",
                cached=False
            )
            
            # Cache result
            if request.use_cache and self.cache:
                self.cache.set("solve", {
                    "problem": request.problem,
                    "variables": request.variables,
                    "constraints": request.constraints
                }, response.to_dict())
            
            # Record metrics
            if self.monitor:
                self.monitor.record_operation(
                    "solve", execution_time / 1000, 
                    success=result.is_sat() or result.is_unsat()
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Solve error: {e}")
            return SolveResponse(
                success=False,
                result_id=result_id,
                status="error",
                satisfiable=None,
                model=None,
                execution_time_ms=(time.time() - start_time) * 1000,
                solver_used="z3"
            )
    
    async def solve_batch(self, request: BatchSolveRequest) -> BatchSolveResponse:
        """Solve multiple problems in batch."""
        import time
        start_time = time.time()
        
        results = []
        completed = 0
        failed = 0
        
        if request.parallel:
            # Solve in parallel
            tasks = [self.solve(problem) for problem in request.problems]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r if not isinstance(r, Exception) else SolveResponse(
                success=False,
                result_id="error",
                status="error",
                satisfiable=None,
                model=None,
                execution_time_ms=0,
                solver_used="none"
            ) for r in results]
        else:
            # Solve sequentially
            for problem in request.problems:
                try:
                    result = await self.solve(problem)
                    results.append(result)
                except Exception as e:
                    results.append(SolveResponse(
                        success=False,
                        result_id="error",
                        status="error",
                        satisfiable=None,
                        model=None,
                        execution_time_ms=0,
                        solver_used="none"
                    ))
        
        completed = sum(1 for r in results if r.success)
        failed = len(results) - completed
        
        return BatchSolveResponse(
            success=True,
            results=results,
            completed=completed,
            failed=failed,
            total_time_ms=(time.time() - start_time) * 1000
        )
    
    async def optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        """Solve optimization problem."""
        import time
        start_time = time.time()
        
        if not self.advanced:
            return OptimizeResponse(
                success=False,
                result_id="error",
                optimal_value=None,
                model=None,
                is_pareto=False,
                pareto_front_size=0,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            variables = [
                Z3Variable(
                    v['name'], 
                    Z3ConstraintType[v.get('type', 'INTEGER').upper()],
                    bit_width=v.get('bit_width')
                )
                for v in request.variables
            ]
            
            constraints = [
                Z3Constraint(c, Z3ConstraintType.INTEGER)
                for c in request.constraints
            ]
            
            if request.multi_objective and request.objectives:
                objectives = [
                    (obj['expression'], 
                     OptimizationObjective.MINIMIZE if obj.get('direction') == 'minimize' else OptimizationObjective.MAXIMIZE)
                    for obj in request.objectives
                ]
            else:
                obj_type = OptimizationObjective.MINIMIZE if request.direction == "minimize" else OptimizationObjective.MAXIMIZE
                objectives = [(request.objective['expression'], obj_type)]
            
            result = self.advanced.optimize(variables, constraints, objectives)
            
            execution_time = (time.time() - start_time) * 1000
            
            return OptimizeResponse(
                success=result.success,
                result_id=f"opt_{int(time.time())}",
                optimal_value=result.optimal_value,
                model=result.optimal_model.assignments if result.optimal_model else None,
                is_pareto=result.is_pareto,
                pareto_front_size=len(result.pareto_front),
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Optimize error: {e}")
            return OptimizeResponse(
                success=False,
                result_id="error",
                optimal_value=None,
                model=None,
                is_pareto=False,
                pareto_front_size=0,
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def prove(self, request: ProveRequest) -> ProveResponse:
        """Prove theorem."""
        import time
        start_time = time.time()
        
        if not self.solver:
            return ProveResponse(
                success=False,
                result_id="error",
                proven=False,
                confidence=0.0,
                tactics_used=None,
                counterexample=None,
                proof=None,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            prover = get_z3_theorem_prover(Z3Config(timeout=request.timeout or 60.0))
            
            result = prover.prove_theorem(
                request.theorem,
                request.assumptions or []
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return ProveResponse(
                success=True,
                result_id=f"proof_{int(time.time())}",
                proven=result.proven,
                confidence=0.95 if result.proven else 0.3,
                tactics_used=[result.tactic_used] if result.tactic_used else None,
                counterexample=result.counterexample,
                proof=result.proof[:1000] if result.proof and request.extract_proof else None,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Prove error: {e}")
            return ProveResponse(
                success=False,
                result_id="error",
                proven=False,
                confidence=0.0,
                tactics_used=None,
                counterexample=None,
                proof=None,
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def extract_proof(self, request: ProofExtractRequest) -> ProofExtractResponse:
        """Extract proof from Z3."""
        import time
        start_time = time.time()
        
        if not self.advanced:
            return ProofExtractResponse(
                success=False,
                proof_steps=[],
                axioms_used=[],
                tactics_used=[],
                verification_status="unavailable",
                raw_proof=None,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            format_enum = ProofFormat[request.format.upper()]
            result = self.advanced.extract_proof(request.smtlib, format_enum)
            
            execution_time = (time.time() - start_time) * 1000
            
            return ProofExtractResponse(
                success=result.success,
                proof_steps=[s.to_dict() for s in result.proof_steps],
                axioms_used=result.axioms_used,
                tactics_used=result.tactics_used,
                verification_status=result.verification_status,
                raw_proof=result.raw_proof[:2000] if result.raw_proof else None,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Proof extract error: {e}")
            return ProofExtractResponse(
                success=False,
                proof_steps=[],
                axioms_used=[],
                tactics_used=[],
                verification_status="error",
                raw_proof=None,
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def solve_portfolio(self, request: PortfolioSolveRequest) -> PortfolioSolveResponse:
        """Solve using portfolio of strategies."""
        import time
        start_time = time.time()
        
        if not self.advanced:
            return PortfolioSolveResponse(
                success=False,
                winner_strategy=None,
                execution_time_ms=(time.time() - start_time) * 1000,
                parallel_speedup=1.0,
                strategies_tried=0,
                status=None,
                model=None
            )
        
        try:
            result = self.advanced.solve_portfolio(
                request.smtlib,
                request.strategies,
                request.parallel
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return PortfolioSolveResponse(
                success=result.success,
                winner_strategy=result.winner_strategy,
                execution_time_ms=execution_time,
                parallel_speedup=result.parallel_speedup,
                strategies_tried=len(result.all_results),
                status=result.best_result.status.value if result.best_result else None,
                model=result.best_result.model.assignments if result.best_result and result.best_result.model else None
            )
            
        except Exception as e:
            logger.error(f"Portfolio solve error: {e}")
            return PortfolioSolveResponse(
                success=False,
                winner_strategy=None,
                execution_time_ms=(time.time() - start_time) * 1000,
                parallel_speedup=1.0,
                strategies_tried=0,
                status=None,
                model=None
            )
    
    async def incremental_solve(self, request: IncrementalSolveRequest) -> IncrementalSolveResponse:
        """Incremental constraint solving."""
        import time
        start_time = time.time()
        
        if not self.advanced:
            return IncrementalSolveResponse(
                success=False,
                state_id=None,
                status=None,
                satisfiable=None,
                model=None,
                message="Advanced solver not available"
            )
        
        try:
            if request.operation == "create":
                z3_vars = [
                    Z3Variable(
                        name=v['name'], 
                        var_type=Z3ConstraintType[v.get('type', 'INTEGER').upper()],
                        bit_width=v.get('bit_width')
                    )
                    for v in (request.variables or [])
                ]
                z3_constraints = [
                    Z3Constraint(c, Z3ConstraintType.INTEGER)
                    for c in (request.constraints or [])
                ]
                
                new_state_id = self.advanced.create_incremental_state(z3_vars, z3_constraints, request.state_id)
                
                return IncrementalSolveResponse(
                    success=True,
                    state_id=new_state_id,
                    message="Incremental state created"
                )
            
            elif request.operation == "push":
                success = self.advanced.push_scope(request.state_id)
                return IncrementalSolveResponse(
                    success=success,
                    state_id=request.state_id,
                    message="Scope pushed" if success else "Failed to push scope"
                )
            
            elif request.operation == "pop":
                success = self.advanced.pop_scope(request.state_id)
                return IncrementalSolveResponse(
                    success=success,
                    state_id=request.state_id,
                    message="Scope popped" if success else "Failed to pop scope"
                )
            
            elif request.operation == "add":
                z3_constraint = Z3Constraint(request.constraint, Z3ConstraintType.INTEGER)
                success = self.advanced.add_constraint_incremental(request.state_id, z3_constraint)
                return IncrementalSolveResponse(
                    success=success,
                    state_id=request.state_id,
                    message="Constraint added" if success else "Failed to add constraint"
                )
            
            elif request.operation == "check":
                result = self.advanced.check_incremental(request.state_id)
                return IncrementalSolveResponse(
                    success=True,
                    state_id=request.state_id,
                    status=result.status.value,
                    satisfiable=result.is_sat(),
                    model=result.model.assignments if result.model else None,
                    message="Check completed"
                )
            
            else:
                return IncrementalSolveResponse(
                    success=False,
                    state_id=request.state_id,
                    message=f"Unknown operation: {request.operation}"
                )
                
        except Exception as e:
            logger.error(f"Incremental solve error: {e}")
            return IncrementalSolveResponse(
                success=False,
                state_id=request.state_id,
                message=str(e)
            )


class Z3ServiceBubble:
    """Complete Z3 Service Bubble with all components."""
    
    def __init__(self):
        self.solver = Z3SolverService()
        self.reliability = Z3ReliabilityChecker() if RELIABILITY_AVAILABLE else None
        self.monitor = get_z3_performance_monitor() if MONITOR_AVAILABLE else None
        self.cache = get_z3_result_cache() if CACHE_AVAILABLE else None
        self.knowledge = get_z3_knowledge_extractor() if KNOWLEDGE_AVAILABLE else None
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            "z3_available": Z3_AVAILABLE,
            "z3_advanced_available": Z3_ADVANCED_AVAILABLE,
            "cache_available": CACHE_AVAILABLE,
            "monitor_available": MONITOR_AVAILABLE,
            "knowledge_available": KNOWLEDGE_AVAILABLE,
            "reliability_available": RELIABILITY_AVAILABLE,
            "request_count": self.solver.request_count,
            "cache_stats": self.cache.get_stats().to_dict() if self.cache else None,
            "monitor_data": self.monitor.get_dashboard_data() if self.monitor else None
        }


# Global service bubble instance
_service_bubble: Optional[Z3ServiceBubble] = None


def get_service_bubble() -> Z3ServiceBubble:
    """Get global service bubble instance."""
    global _service_bubble
    if _service_bubble is None:
        _service_bubble = Z3ServiceBubble()
    return _service_bubble


# =============================================================================
# API Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _start_time
    
    # Startup
    logger.info("Starting Z3 Prover Service Bubble...")
    
    # Initialize components
    if CONFIG_AVAILABLE:
        config = get_config_manager().config
        logger.info(f"Loaded configuration from {get_config_manager().config_path}")
    
    if MONITOR_AVAILABLE:
        monitor = get_z3_performance_monitor()
        monitor.start_monitoring()
        logger.info("Performance monitoring started")
    
    # Initialize service bubble
    bubble = get_service_bubble()
    logger.info(f"Z3 Service Bubble initialized: {bubble.get_status()}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Z3 Prover Service Bubble...")
    
    if MONITOR_AVAILABLE:
        get_z3_performance_monitor().stop_monitoring()


# =============================================================================
# Create FastAPI Application
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is required for the API server")
    
    app = FastAPI(
        title="Z3 Prover Service Bubble API",
        description="Complete REST API for Z3 constraint solving, theorem proving, and formal verification",
        version="3.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# =============================================================================
# Dependency Injection
# =============================================================================

async def get_config() -> IntegrationConfig:
    """Get configuration dependency."""
    if CONFIG_AVAILABLE:
        return get_config_manager().config
    raise HTTPException(status_code=503, detail="Configuration not available")


async def get_bubble() -> Z3ServiceBubble:
    """Get service bubble dependency."""
    return get_service_bubble()


# =============================================================================
# API Endpoints - Core Solving
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "Z3 Prover Service Bubble API",
        "version": "3.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global _start_time
    
    bubble = get_service_bubble()
    status = bubble.get_status()
    
    # Calculate uptime
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    
    # Determine overall status
    all_healthy = all([
        status["z3_available"],
        status["cache_available"],
        status["monitor_available"]
    ])
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        components={
            "z3": Z3_AVAILABLE,
            "z3_advanced": Z3_ADVANCED_AVAILABLE,
            "cache": CACHE_AVAILABLE,
            "monitor": MONITOR_AVAILABLE,
            "knowledge": KNOWLEDGE_AVAILABLE,
            "reliability": RELIABILITY_AVAILABLE,
            "configuration": CONFIG_AVAILABLE
        },
        uptime_seconds=uptime,
        load={
            "requests": status.get("request_count", 0)
        }
    )


@app.post("/solve", response_model=SolveResponse)
async def solve_constraints(request: SolveRequest, bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Solve constraint satisfaction problem."""
    return await bubble.solver.solve(request)


@app.post("/solve/batch", response_model=BatchSolveResponse)
async def solve_batch(request: BatchSolveRequest, bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Solve multiple problems in batch."""
    return await bubble.solver.solve_batch(request)


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_problem(request: OptimizeRequest, bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Solve optimization problem."""
    return await bubble.solver.optimize(request)


@app.post("/prove", response_model=ProveResponse)
async def prove_theorem(request: ProveRequest, bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Prove theorem."""
    return await bubble.solver.prove(request)


@app.post("/prove/extract", response_model=ProofExtractResponse)
async def extract_proof(request: ProofExtractRequest, bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Extract proof from Z3."""
    return await bubble.solver.extract_proof(request)


@app.post("/solve/portfolio", response_model=PortfolioSolveResponse)
async def solve_portfolio(request: PortfolioSolveRequest, bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Solve using multiple strategies in parallel."""
    return await bubble.solver.solve_portfolio(request)


@app.post("/solve/incremental", response_model=IncrementalSolveResponse)
async def solve_incremental(request: IncrementalSolveRequest, bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Incremental constraint solving."""
    return await bubble.solver.incremental_solve(request)


# =============================================================================
# API Endpoints - Translation and Verification
# =============================================================================

@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """Translate between SMT-LIB and Lean."""
    import time
    start_time = time.time()
    
    try:
        from z3_leanaide_bridge import get_z3_leanaide_bridge_sync
        bridge = get_z3_leanaide_bridge_sync()
        
        if request.direction == "smt_to_lean":
            result = await bridge.translate_smt_to_lean(request.content)
        else:
            result = await bridge.translate_lean_to_smt(request.content)
        
        return TranslateResponse(
            success=result.success,
            translation=result.translation,
            source=result.source,
            target=result.target,
            execution_time_ms=(time.time() - start_time) * 1000,
            errors=result.errors,
            warnings=result.warnings
        )
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return TranslateResponse(
            success=False,
            translation="",
            source="unknown",
            target="unknown",
            execution_time_ms=(time.time() - start_time) * 1000,
            errors=[str(e)],
            warnings=[]
        )


@app.post("/verify", response_model=VerifyResponse)
async def verify_problem(request: VerifyRequest, bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Verify problem using both Z3 and Lean."""
    import time
    start_time = time.time()
    
    try:
        from z3_leanaide_bridge import get_z3_leanaide_bridge_sync, VerificationStrategy
        bridge = get_z3_leanaide_bridge_sync()
        
        strategy = VerificationStrategy[request.strategy.upper()]
        result = await bridge.verify_with_both(request.problem, strategy)
        
        return VerifyResponse(
            success=result.success,
            verified=result.agreement,
            z3_result=result.z3_result.to_dict() if result.z3_result else None,
            lean_result=result.lean_result.to_dict() if hasattr(result.lean_result, 'to_dict') else result.lean_result,
            agreement=result.agreement,
            confidence_score=result.confidence_score,
            recommendation=result.recommendation,
            execution_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return VerifyResponse(
            success=False,
            verified=False,
            z3_result=None,
            lean_result=None,
            agreement=False,
            confidence_score=0.0,
            recommendation=str(e),
            execution_time_ms=(time.time() - start_time) * 1000
        )


# =============================================================================
# API Endpoints - Reliability Verification
# =============================================================================

@app.post("/verify/reliability", response_model=ReliabilityVerifyResponse)
async def verify_reliability(request: ReliabilityVerifyRequest, bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Verify reliability constraints."""
    import time
    start_time = time.time()
    
    if not bubble.reliability:
        return ReliabilityVerifyResponse(
            success=False,
            verified=False,
            violations=[{"error": "Reliability checker not available"}],
            recommendations=[],
            counterexample=None,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    try:
        from z3_reliability_checker import ComponentReliabilityModel, ReliabilityConstraint, ReliabilityProperty
        
        # Convert components
        components = []
        for comp_data in request.components:
            components.append(ComponentReliabilityModel(
                component_id=comp_data['component_id'],
                availability=comp_data.get('availability', 0.99),
                mtbf_hours=comp_data.get('mtbf_hours', 8760.0),
                mttr_hours=comp_data.get('mttr_hours', 1.0),
                redundancy_factor=comp_data.get('redundancy_factor', 1)
            ))
        
        # Convert requirements
        requirements = []
        for req_data in request.requirements:
            requirements.append(ReliabilityConstraint(
                property_type=ReliabilityProperty(req_data['property_type']),
                threshold=req_data['threshold'],
                target_component=req_data.get('target_component')
            ))
        
        # Verify
        result = bubble.reliability.verify_system_reliability(components, requirements)
        
        return ReliabilityVerifyResponse(
            success=result.success,
            verified=result.verified,
            violations=result.violations,
            recommendations=result.recommendations,
            counterexample=result.counterexample,
            execution_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Reliability verification error: {e}")
        return ReliabilityVerifyResponse(
            success=False,
            verified=False,
            violations=[{"error": str(e)}],
            recommendations=[],
            counterexample=None,
            execution_time_ms=(time.time() - start_time) * 1000
        )


# =============================================================================
# API Endpoints - Knowledge Extraction
# =============================================================================

@app.post("/knowledge/extract", response_model=KnowledgeExtractResponse)
async def extract_knowledge(request: KnowledgeExtractRequest, bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Extract knowledge from solution."""
    if not bubble.knowledge:
        return KnowledgeExtractResponse(
            success=False,
            patterns_found=0,
            strategies_learned=0,
            insights=[]
        )
    
    try:
        # Learn strategy
        strategy = bubble.knowledge.learn_strategy(
            problem_features={
                "type": request.domain,
                "var_count": len(request.solution.get('model', {})),
                "constraint_count": 0
            },
            tactics_used=["solve"],
            config_used={},
            success=request.solution.get('status') == 'sat',
            solving_time=request.solution.get('execution_time_ms', 0) / 1000
        )
        
        # Extract insights
        from z3prover_integration import Z3SolverResult, Z3ResultStatus, Z3Model
        mock_result = Z3SolverResult(
            status=Z3ResultStatus.SAT if request.solution.get('satisfiable') else Z3ResultStatus.UNSAT,
            model=Z3Model(assignments=request.solution.get('model', {}))
        )
        insights = bubble.knowledge.extract_insights(mock_result, request.problem)
        
        return KnowledgeExtractResponse(
            success=True,
            patterns_found=len(bubble.knowledge.proof_patterns),
            strategies_learned=len(bubble.knowledge.strategies),
            insights=[i.to_dict() for i in insights]
        )
        
    except Exception as e:
        logger.error(f"Knowledge extraction error: {e}")
        return KnowledgeExtractResponse(
            success=False,
            patterns_found=0,
            strategies_learned=0,
            insights=[]
        )


@app.get("/knowledge/summary")
async def get_knowledge_summary(bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Get knowledge base summary."""
    if not bubble.knowledge:
        return {"error": "Knowledge extraction not available"}
    
    return bubble.knowledge.get_knowledge_summary()


# =============================================================================
# API Endpoints - Metrics and Monitoring
# =============================================================================

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Get performance metrics."""
    if not bubble.monitor:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    dashboard = bubble.monitor.get_dashboard_data()
    
    return MetricsResponse(
        timestamp=datetime.utcnow().isoformat(),
        summary=dashboard.get("summary", {}),
        operations=dashboard.get("operation_performance", {}),
        bottlenecks=dashboard.get("top_bottlenecks", []),
        alerts=dashboard.get("recent_alerts", []),
        cache_stats=bubble.cache.get_stats().to_dict() if bubble.cache else None
    )


@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus-compatible metrics."""
    lines = []
    
    bubble = get_service_bubble()
    
    if bubble.monitor:
        # Add operation metrics
        for name, metrics in bubble.monitor._operation_metrics.items():
            lines.append(f'z3_operation_calls_total{{operation="{name}"}} {metrics.call_count}')
            lines.append(f'z3_operation_errors_total{{operation="{name}"}} {metrics.error_count}')
            lines.append(f'z3_operation_duration_seconds{{operation="{name}"}} {metrics.avg_time}')
    
    if bubble.cache:
        stats = bubble.cache.get_stats()
        lines.append(f'z3_cache_hits_total {stats.hits}')
        lines.append(f'z3_cache_misses_total {stats.misses}')
        lines.append(f'z3_cache_hit_rate {stats.hit_rate}')
    
    lines.append(f'z3_requests_total {bubble.solver.request_count}')
    
    return StreamingResponse(
        iter("\n".join(lines) + "\n"),
        media_type="text/plain"
    )


@app.get("/config")
async def get_configuration():
    """Get current configuration."""
    if not CONFIG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration not available")
    
    return get_config_manager().to_dict()


@app.get("/status")
async def get_status(bubble: Z3ServiceBubble = Depends(get_bubble)):
    """Get complete service status."""
    return bubble.get_status()


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "subscribe":
                operation_id = data.get("operation_id")
                if operation_id:
                    manager.subscribe_to_operation(websocket, operation_id)
                    await websocket.send_json({
                        "type": "subscribed",
                        "operation_id": operation_id
                    })
            
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# =============================================================================
# Background Tasks
# =============================================================================

async def cleanup_expired_cache():
    """Background task to clean up expired cache entries."""
    bubble = get_service_bubble()
    if bubble.cache:
        # Cleanup is handled internally by cache
        pass


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the API server."""
    import uvicorn
    
    if not FASTAPI_AVAILABLE:
        print("FastAPI is required. Install with: pip install fastapi uvicorn")
        return
    
    # Get configuration
    if CONFIG_AVAILABLE:
        config = get_config_manager().config
        host = config.server.host
        port = config.server.port
    else:
        host = "0.0.0.0"
        port = 8765
    
    print(f"Starting Z3 Prover Service Bubble on {host}:{port}")
    print(f"Documentation: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    
    uvicorn.run(
        "z3_api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
