"""
FastAPI Endpoints for Z3 Knowledge Integration

Provides REST API for:
- Knowledge extraction from Z3 results
- Strategy recommendations
- Pattern search
- Knowledge graph queries

Author: OpenEvolve
Created: 2026-01-31
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Dummy classes for when FastAPI is not available
    class BaseModel:
        pass
    class FastAPI:
        pass
    def HTTPException(*args, **kwargs):
        pass
    def Field(*args, **kwargs):
        return None
    class JSONResponse:
        pass

# Z3 integration imports
try:
    from z3_knowledge_integration import (
        Z3KnowledgeIntegration,
        get_z3_knowledge_integration
    )
    Z3_INTEGRATION_AVAILABLE = True
except ImportError:
    Z3_INTEGRATION_AVAILABLE = False
    Z3KnowledgeIntegration = None
    get_z3_knowledge_integration = None

# CAV-NLP integration imports
try:
    from openevolve.unified_math_service import UnifiedMathService
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    UnifiedMathService = None
    EnhancedZ3Solver = None


logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class SolverResultInput(BaseModel):
    """Input model for Z3 solver result."""
    result_data: Dict[str, Any] = Field(..., description="Solver result data")
    problem_statement: str = Field(..., description="Original problem statement")
    problem_type: str = Field("general", description="Problem classification")
    problem_id: Optional[str] = Field(None, description="Optional problem identifier")


class ExtractKnowledgeResponse(BaseModel):
    """Response model for knowledge extraction."""
    problem_id: str
    extracted: Dict[str, Any]
    stored_artifacts: List[str]
    success: bool
    processing_time_ms: float


class StrategyRequest(BaseModel):
    """Request model for strategy recommendation."""
    problem_features: Dict[str, Any] = Field(..., description="Problem characteristics")


class StrategyResponse(BaseModel):
    """Response model for strategy recommendation."""
    strategy: Optional[Dict[str, Any]]
    confidence: float
    alternatives: List[Dict[str, Any]]


class PatternSearchRequest(BaseModel):
    """Request model for pattern search."""
    query: str = Field(..., description="Search query")
    pattern_type: Optional[str] = Field(None, description="Pattern type filter")
    top_k: int = Field(5, ge=1, le=50, description="Number of results")


class PatternSearchResponse(BaseModel):
    """Response model for pattern search."""
    results: List[Dict[str, Any]]
    total_found: int
    query: str


class KnowledgeSummaryResponse(BaseModel):
    """Response model for knowledge summary."""
    z3_extractor_stats: Optional[Dict[str, Any]]
    extraction_stats: Dict[str, int]
    storage_available: bool
    timestamp: str


class InsightFilterRequest(BaseModel):
    """Request model for insight filtering."""
    category: Optional[str] = Field(None, description="Insight category")
    min_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence")
    limit: int = Field(10, ge=1, le=100)


# =============================================================================
# API Router Setup
# =============================================================================

if FASTAPI_AVAILABLE:
    from fastapi import APIRouter
    router = APIRouter(prefix="/z3-knowledge", tags=["Z3 Knowledge"])
else:
    router = None


# Global integration instance
_z3_integration: Optional[Z3KnowledgeIntegration] = None


class CAVNLPEnhancedAPI:
    """CAV-NLP enhanced API operations."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.math_service = UnifiedMathService()
            self.enhanced_solver = EnhancedZ3Solver()
            logger.info("CAV-NLP enhanced API initialized")
    
    async def formalize_text(self, text: str) -> Dict[str, Any]:
        """Formalize natural language text using CAV-NLP."""
        if not self.use_cav_nlp:
            return {"error": "CAV-NLP not available"}
        try:
            formalized = self.math_service.formalize(text)
            return {
                "success": True,
                "original": text,
                "formalized_code": getattr(formalized, 'code', str(formalized)),
                "language": getattr(formalized, 'language', 'unknown')
            }
        except Exception as e:
            logger.error(f"CAV-NLP formalization failed: {e}")
            return {"error": str(e)}
    
    async def extract_with_cav_nlp(self, text: str) -> Dict[str, Any]:
        """Extract knowledge using CAV-NLP."""
        if not self.use_cav_nlp:
            return {"error": "CAV-NLP not available"}
        try:
            formalized = self.math_service.formalize(text)
            return {
                "success": True,
                "formalized": getattr(formalized, 'code', str(formalized)),
                "patterns": []
            }
        except Exception as e:
            logger.error(f"CAV-NLP extraction failed: {e}")
            return {"error": str(e)}


async def get_integration() -> Z3KnowledgeIntegration:
    """Get or initialize Z3 knowledge integration."""
    global _z3_integration
    if _z3_integration is None:
        if not Z3_INTEGRATION_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Z3 knowledge integration not available"
            )
        _z3_integration = await get_z3_knowledge_integration()
    return _z3_integration


# =============================================================================
# API Endpoints
# =============================================================================

if router:
    
    @router.post("/extract", response_model=ExtractKnowledgeResponse)
    async def extract_knowledge(
        input_data: SolverResultInput,
        background_tasks: BackgroundTasks
    ):
        """
        Extract and store knowledge from a Z3 solver result.
        
        This endpoint processes solver results, extracts patterns, strategies,
        and insights, and stores them in the knowledge base.
        """
        import time
        start_time = time.time()
        
        integration = await get_integration()
        
        try:
            # Create mock result object from input data
            class MockResult:
                pass
            
            result = MockResult()
            for key, value in input_data.result_data.items():
                setattr(result, key, value)
            
            # Process the result
            processing = await integration.process_solver_result(
                result=result,
                problem_statement=input_data.problem_statement,
                problem_id=input_data.problem_id,
                problem_type=input_data.problem_type
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return ExtractKnowledgeResponse(
                problem_id=processing["problem_id"],
                extracted=processing["extracted"],
                stored_artifacts=processing["stored_artifacts"],
                success=processing["success"],
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Extraction failed: {str(e)}"
            )
    
    
    @router.post("/recommend-strategy", response_model=StrategyResponse)
    async def recommend_strategy(request: StrategyRequest):
        """
        Recommend a solving strategy based on problem features.
        
        Analyzes problem characteristics and returns the best matching
        strategy from the knowledge base.
        """
        integration = await get_integration()
        
        try:
            strategy = await integration.get_recommended_strategy(
                request.problem_features
            )
            
            if strategy:
                return StrategyResponse(
                    strategy=strategy,
                    confidence=float(strategy.get("success_rate", "0%").rstrip("%")) / 100,
                    alternatives=[]
                )
            else:
                return StrategyResponse(
                    strategy=None,
                    confidence=0.0,
                    alternatives=[]
                )
                
        except Exception as e:
            logger.error(f"Strategy recommendation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Recommendation failed: {str(e)}"
            )
    
    
    @router.post("/search-patterns", response_model=PatternSearchResponse)
    async def search_patterns(request: PatternSearchRequest):
        """
        Search for patterns in the Z3 knowledge base.
        
        Searches stored patterns, strategies, and insights matching the query.
        """
        integration = await get_integration()
        
        try:
            results = await integration.search_similar_patterns(
                query=request.query,
                pattern_type=request.pattern_type,
                top_k=request.top_k
            )
            
            return PatternSearchResponse(
                results=results,
                total_found=len(results),
                query=request.query
            )
            
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )
    
    
    @router.get("/summary", response_model=KnowledgeSummaryResponse)
    async def get_knowledge_summary():
        """Get summary of Z3 knowledge in the system."""
        integration = await get_integration()
        
        try:
            summary = await integration.get_knowledge_summary()
            
            return KnowledgeSummaryResponse(
                z3_extractor_stats=summary.get("z3_extractor"),
                extraction_stats=summary.get("extraction_stats", {}),
                storage_available=summary.get("storage_available", False),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get summary: {str(e)}"
            )
    
    
    @router.get("/patterns/{pattern_type}")
    async def get_patterns_by_type(
        pattern_type: str,
        limit: int = Query(10, ge=1, le=100)
    ):
        """
        Get patterns of a specific type.
        
        Args:
            pattern_type: Type of pattern ('proof', 'constraint', 'strategy', 'insight')
            limit: Maximum number of results
        """
        integration = await get_integration()
        
        try:
            # Map to storage artifact types
            artifact_type_map = {
                "proof": "z3_proof_pattern",
                "constraint": "z3_constraint_pattern",
                "strategy": "z3_strategy",
                "insight": "z3_insight"
            }
            
            artifact_type = artifact_type_map.get(pattern_type)
            if not artifact_type:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid pattern type. Valid types: {list(artifact_type_map.keys())}"
                )
            
            # Search for all patterns of this type
            results = await integration.search_similar_patterns(
                query="",
                pattern_type=artifact_type.replace("z3_", "").replace("_pattern", ""),
                top_k=limit
            )
            
            return {
                "pattern_type": pattern_type,
                "count": len(results),
                "patterns": results
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get patterns: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get patterns: {str(e)}"
            )
    
    
    @router.post("/insights/filter")
    async def filter_insights(request: InsightFilterRequest):
        """Filter insights by criteria."""
        # This would integrate with Z3KnowledgeExtractor.find_related_insights
        # For now, return placeholder
        return {
            "category": request.category,
            "min_confidence": request.min_confidence,
            "insights": [],
            "total": 0
        }


# =============================================================================
# Application Factory
# =============================================================================

def create_z3_knowledge_app() -> Optional[FastAPI]:
    """Create FastAPI application with Z3 knowledge endpoints."""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available")
        return None
    
    if not Z3_INTEGRATION_AVAILABLE:
        logger.error("Z3 knowledge integration not available")
        return None
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        # Startup
        logger.info("Starting Z3 Knowledge API")
        global _z3_integration
        _z3_integration = await get_z3_knowledge_integration()
        yield
        # Shutdown
        if _z3_integration:
            await _z3_integration.close()
            logger.info("Z3 Knowledge API stopped")
    
    app = FastAPI(
        title="Z3 Knowledge Engine API",
        description="API for Z3 knowledge extraction and management",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Include router
    app.include_router(router)
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        integration = await get_integration()
        summary = await integration.get_knowledge_summary()
        
        return {
            "status": "healthy",
            "z3_integration": Z3_INTEGRATION_AVAILABLE,
            "storage_available": summary.get("storage_available", False),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    return app


# =============================================================================
# Global App Instance
# =============================================================================

# Create app instance for import (when FastAPI is available)
app = create_z3_knowledge_app() if FASTAPI_AVAILABLE else None

# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        exit(1)
    
    import uvicorn
    
    app = create_z3_knowledge_app()
    if app:
        uvicorn.run(app, host="0.0.0.0", port=8766)
    else:
        print("Failed to create Z3 knowledge API")
