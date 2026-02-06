"""
LeanAide API Routes for OpenEvolve API Server

This module provides REST API endpoints for LeanAide operations:
- /api/leanaide/verify - Quick verification endpoint
- /api/leanaide/prove - Proof generation endpoint
- /api/leanaide/translate - Theorem translation endpoint
- /api/leanaide/status - Server status check
- /api/leanaide/quality-gate - Quality gate verification
- /api/leanaide/cav-nlp/formalize - CAV-NLP formalization endpoint

Author: OpenEvolve
Created: 2026-02-02
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional


logger = logging.getLogger(__name__)

# Add CAV-NLP imports with graceful fallback
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.warning("CAV-NLP integration not available for API routes")


# =============================================================================
# Pydantic Models
# =============================================================================

class LeanAideVerifyRequest(BaseModel):
    """Request for LeanAide verification."""
    code: str = Field(..., description="Lean code to verify")
    timeout: int = Field(default=300, description="Timeout in seconds")


class LeanAideVerifyResponse(BaseModel):
    """Response for LeanAide verification."""
    success: bool
    verified: bool
    confidence: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class LeanAideProveRequest(BaseModel):
    """Request for LeanAide proof generation."""
    theorem_text: str = Field(..., description="Theorem statement to prove")
    theorem_name: Optional[str] = Field(default=None, description="Optional theorem name")
    timeout: int = Field(default=300, description="Timeout in seconds")


class LeanAideProveResponse(BaseModel):
    """Response for LeanAide proof generation."""
    success: bool
    theorem_name: str
    formal_code: Optional[str] = None
    proof: Optional[str] = None
    confidence: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class LeanAideTranslateRequest(BaseModel):
    """Request for LeanAide theorem translation."""
    theorem_text: str = Field(..., description="Theorem text to translate")
    name: Optional[str] = Field(default=None, description="Optional theorem name")
    timeout: int = Field(default=300, description="Timeout in seconds")


class LeanAideTranslateResponse(BaseModel):
    """Response for LeanAide theorem translation."""
    success: bool
    name: str
    code: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class LeanAideQualityGateRequest(BaseModel):
    """Request for LeanAide quality gate verification."""
    solution_content: str = Field(..., description="Solution content to verify")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence threshold")
    verification_level: str = Field(default="verification", description="Verification level")


class LeanAideQualityGateResponse(BaseModel):
    """Response for LeanAide quality gate verification."""
    decision: str
    overall_score: float
    is_mathematical: bool
    verification_passed: bool
    confidence_score: float
    critical_issues: List[str] = []
    recommendations: List[str] = []
    error: Optional[str] = None


class LeanAideStatusResponse(BaseModel):
    """Response for LeanAide status check."""
    available: bool
    server_status: str
    version: Optional[str] = None
    components: Dict[str, bool] = {}
    config: Dict[str, Any] = {}


# =============================================================================
# CAV-NLP Models
# =============================================================================

class CAVNLPFormalizeRequest(BaseModel):
    """Request for CAV-NLP formalization."""
    natural_language: str = Field(..., description="Natural language mathematical statement")
    statement_type: str = Field(default="theorem", description="Type of statement (theorem, lemma, definition)")
    name: Optional[str] = Field(default=None, description="Optional name for the formalized statement")
    domain: str = Field(default="general", description="Mathematical domain (algebra, analysis, etc.)")
    use_constraints: bool = Field(default=True, description="Whether to use constraint-based formalization")
    timeout: int = Field(default=300, description="Timeout in seconds")


class CAVNLPFormalizeResponse(BaseModel):
    """Response for CAV-NLP formalization."""
    success: bool
    lean_code: Optional[str] = None
    formalized_name: str
    confidence: float
    constraints_used: List[str] = []
    verification_status: str = "pending"
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class CAVNLPVerifyRequest(BaseModel):
    """Request for CAV-NLP enhanced verification."""
    lean_code: str = Field(..., description="Lean code to verify")
    use_semantic_analysis: bool = Field(default=True, description="Use semantic analysis")
    use_constraint_checking: bool = Field(default=True, description="Use constraint-based checking")
    timeout: int = Field(default=300, description="Timeout in seconds")


class CAVNLPVerifyResponse(BaseModel):
    """Response for CAV-NLP enhanced verification."""
    success: bool
    verified: bool
    semantic_score: float
    constraint_satisfied: bool
    issues_found: List[str] = []
    suggestions: List[str] = []
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


# =============================================================================
# LeanAide Router
# =============================================================================

router = APIRouter(
    prefix="/api/leanaide",
    tags=["LeanAide"],
    dependencies=[]  # Add authentication if needed
)


# =============================================================================
# Import LeanAide Components
# =============================================================================

# LeanAide client
try:
    from leanaide_client import LeanAideClient, LeanAideConfig
    from leanaide_mcp_tools import (
        leanaide_translate_theorem_async,
        leanaide_verify_solution_async,
        leanaide_generate_proof_async,
        leanaide_elaborate_code_async,
        get_leanaide_status as get_mcp_status
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    LeanAideClient = None
    LeanAideConfig = None
    logger.warning("LeanAide not available for API routes")


# Quality Gate
try:
    from quality_gate_leanaide_verifier import (
        LeanAideQualityGateVerifier,
        get_leanaide_quality_gate_verifier,
        MathematicalCorrectnessLevel
    )
    QUALITY_GATE_AVAILABLE = True
except ImportError:
    QUALITY_GATE_AVAILABLE = False
    LeanAideQualityGateVerifier = None
    get_leanaide_quality_gate_verifier = None
    MathematicalCorrectnessLevel = None
    logger.warning("LeanAide quality gate not available")


# Ragbits integration
try:
    from knowledge_engine.integrations.leanaide_ragbits_integration import (
        LeanAideRagbitsIntegration,
        get_leanaide_ragbits_integration
    )
    RAGBITS_INTEGRATION_AVAILABLE = True
except ImportError:
    RAGBITS_INTEGRATION_AVAILABLE = False
    LeanAideRagbitsIntegration = None
    get_leanaide_ragbits_integration = None
    logger.warning("LeanAide-Ragbits integration not available")


# =============================================================================
# Lazy Initialization
# =============================================================================

_leanaide_client: Optional[LeanAideClient] = None
_quality_gate_verifier: Optional[LeanAideQualityGateVerifier] = None
_ragbits_integration: Optional[LeanAideRagbitsIntegration] = None
_enhanced_solver: Optional[Any] = None
_math_service: Optional[Any] = None


def get_enhanced_solver() -> Any:
    """Get or create CAV-NLP enhanced solver."""
    global _enhanced_solver
    if _enhanced_solver is None and CAV_NLP_AVAILABLE:
        try:
            from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
            _enhanced_solver = EnhancedZ3Solver()
        except Exception as e:
            logger.error(f"Failed to create enhanced solver: {e}")
    return _enhanced_solver


def get_math_service() -> Any:
    """Get or create unified math service."""
    global _math_service
    if _math_service is None and CAV_NLP_AVAILABLE:
        try:
            from openevolve.unified_math_service import UnifiedMathService
            _math_service = UnifiedMathService()
        except Exception as e:
            logger.error(f"Failed to create math service: {e}")
    return _math_service


def get_leanaide_client() -> LeanAideClient:
    """Get or create LeanAide client."""
    global _leanaide_client
    if _leanaide_client is None and LEANAIDE_AVAILABLE:
        try:
            config = LeanAideConfig(host="localhost", port=7654, timeout=300.0)
            _leanaide_client = LeanAideClient(config)
        except Exception as e:
            logger.error(f"Failed to create LeanAide client: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LeanAide client not available: {str(e)}"
            )
    if _leanaide_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LeanAide not available"
        )
    return _leanaide_client


def get_quality_gate_verifier() -> LeanAideQualityGateVerifier:
    """Get or create quality gate verifier."""
    global _quality_gate_verifier
    if _quality_gate_verifier is None and QUALITY_GATE_AVAILABLE:
        try:
            _quality_gate_verifier = get_leanaide_quality_gate_verifier()
        except Exception as e:
            logger.error(f"Failed to create quality gate verifier: {e}")
    if _quality_gate_verifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LeanAide quality gate not available"
        )
    return _quality_gate_verifier


def get_ragbits_integration() -> LeanAideRagbitsIntegration:
    """Get or create Ragbits integration."""
    global _ragbits_integration
    if _ragbits_integration is None and RAGBITS_INTEGRATION_AVAILABLE:
        try:
            _ragbits_integration = get_leanaide_ragbits_integration()
        except Exception as e:
            logger.error(f"Failed to create Ragbits integration: {e}")
    if _ragbits_integration is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LeanAide-Ragbits integration not available"
        )
    return _ragbits_integration


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/status", response_model=LeanAideStatusResponse)
async def leanaide_status():
    """
    Get LeanAide server status.
    
    Returns:
        LeanAideStatusResponse with server status
    """
    if not LEANAIDE_AVAILABLE:
        return LeanAideStatusResponse(
            available=False,
            server_status="unavailable",
            components={}
        )
    
    try:
        # Get status from MCP tools
        mcp_status = get_mcp_status() if hasattr(get_mcp_status, '__call__') else {}
        
        return LeanAideStatusResponse(
            available=True,
            server_status="running",
            version=mcp_status.get("version"),
            components={
                "client": True,
                "mcp_tools": True,
                "quality_gate": QUALITY_GATE_AVAILABLE,
                "ragbits": RAGBITS_INTEGRATION_AVAILABLE,
                "cav_nlp": CAV_NLP_AVAILABLE
            },
            config={
                "host": "localhost",
                "port": 7654,
                "timeout": 300
            }
        )
    except Exception as e:
        return LeanAideStatusResponse(
            available=False,
            server_status="error",
            error=str(e)
        )


@router.post("/verify", response_model=LeanAideVerifyResponse)
async def leanaide_verify(request: LeanAideVerifyRequest):
    """
    Verify Lean code.
    
    Args:
        request: LeanAideVerifyRequest with code to verify
        
    Returns:
        LeanAideVerifyResponse with verification result
    """
    if not LEANAIDE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LeanAide not available"
        )
    
    try:
        result = await leanaide_verify_solution_async(
            request.code,
            timeout=request.timeout
        )
        
        return LeanAideVerifyResponse(
            success=result.get("success", False),
            verified=result.get("success", False),
            confidence=result.get("confidence", 0.0),
            error=result.get("error"),
            metadata=result.get("metadata", {})
        )
    except Exception as e:
        logger.error(f"LeanAide verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )


@router.post("/prove", response_model=LeanAideProveResponse)
async def leanaide_prove(request: LeanAideProveRequest):
    """
    Generate a proof for a theorem.
    
    Args:
        request: LeanAideProveRequest with theorem to prove
        
    Returns:
        LeanAideProveResponse with proof result
    """
    if not LEANAIDE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LeanAide not available"
        )
    
    try:
        # Translate theorem first
        translate_result = await leanaide_translate_theorem_async(
            request.theorem_text,
            theorem_name=request.theorem_name,
            timeout=request.timeout
        )
        
        if not translate_result.get("success"):
            return LeanAideProveResponse(
                success=False,
                theorem_name=request.theorem_name or "unknown",
                error=translate_result.get("error", "Translation failed")
            )
        
        # Generate proof
        formal_code = translate_result.get("lean_code", "")
        proof_result = await leanaide_generate_proof_async(
            formal_code,
            timeout=request.timeout
        )
        
        return LeanAideProveResponse(
            success=proof_result.get("success", False),
            theorem_name=translate_result.get("name", request.theorem_name or "unknown"),
            formal_code=formal_code,
            proof=proof_result.get("proof"),
            confidence=proof_result.get("confidence", 0.0),
            error=proof_result.get("error"),
            metadata={
                "translate_result": translate_result,
                "proof_result": proof_result
            }
        )
    except Exception as e:
        logger.error(f"LeanAide proof generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Proof generation failed: {str(e)}"
        )


@router.post("/translate", response_model=LeanAideTranslateResponse)
async def leanaide_translate(request: LeanAideTranslateRequest):
    """
    Translate theorem text to Lean code.
    
    Args:
        request: LeanAideTranslateRequest with theorem to translate
        
    Returns:
        LeanAideTranslateResponse with translated code
    """
    if not LEANAIDE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LeanAide not available"
        )
    
    try:
        result = await leanaide_translate_theorem_async(
            request.theorem_text,
            theorem_name=request.name,
            timeout=request.timeout
        )
        
        return LeanAideTranslateResponse(
            success=result.get("success", False),
            name=result.get("name", request.name or "unknown"),
            code=result.get("lean_code"),
            error=result.get("error"),
            metadata=result.get("metadata", {})
        )
    except Exception as e:
        logger.error(f"LeanAide translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@router.post("/quality-gate", response_model=LeanAideQualityGateResponse)
async def leanaide_quality_gate(request: LeanAideQualityGateRequest):
    """
    Run quality gate verification on solution content.
    
    Args:
        request: LeanAideQualityGateRequest with content to verify
        
    Returns:
        LeanAideQualityGateResponse with quality gate result
    """
    if not QUALITY_GATE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LeanAide quality gate not available"
        )
    
    try:
        verifier = get_quality_gate_verifier()
        
        # Map verification level
        verification_level = request.verification_level
        if MathematicalCorrectnessLevel and hasattr(MathematicalCorrectnessLevel, verification_level):
            config_level = getattr(MathematicalCorrectnessLevel, verification_level)
        else:
            config_level = MathematicalCorrectnessLevel.VERIFICATION if MathematicalCorrectnessLevel else "verification"
        
        # Run verification
        from quality_gate_leanaide_verifier import MathematicalVerificationResult
        import asyncio
        
        result: MathematicalVerificationResult = await verifier.verify_mathematical_correctness(
            request.solution_content
        )
        
        # Determine decision
        if result.verification_passed and result.confidence_score >= request.confidence_threshold:
            decision = "pass"
        elif result.is_mathematical and not result.verification_passed:
            decision = "fail"
        elif result.confidence_score >= request.confidence_threshold * 0.8:
            decision = "conditional_pass"
        else:
            decision = "deferred"
        
        return LeanAideQualityGateResponse(
            decision=decision,
            overall_score=result.confidence_score * 100,
            is_mathematical=result.is_mathematical,
            verification_passed=result.verification_passed,
            confidence_score=result.confidence_score,
            critical_issues=result.errors,
            recommendations=[],
            error=None
        )
    except Exception as e:
        logger.error(f"LeanAide quality gate failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality gate failed: {str(e)}"
        )


@router.post("/rag-retrieve")
async def leanaide_rag_retrieve(
    query: str,
    top_k: int = 5
):
    """
    Retrieve similar proofs using RAG.
    
    Args:
        query: Theorem or proof to search for
        top_k: Number of results to return
        
    Returns:
        List of retrieved proofs
    """
    if not RAGBITS_INTEGRATION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LeanAide-Ragbits integration not available"
        )
    
    try:
        integration = get_ragbits_integration()
        
        proofs = await integration.retrieve_similar_proofs(query, top_k=top_k)
        
        return {
            "success": True,
            "query": query,
            "retrieved_count": len(proofs),
            "proofs": [p.to_dict() for p in proofs]
        }
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG retrieval failed: {str(e)}"
        )


@router.post("/rag-prove")
async def leanaide_rag_prove(
    theorem_text: str,
    theorem_name: Optional[str] = None
):
    """
    Generate proof using RAG augmentation.
    
    Args:
        theorem_text: Theorem to prove
        theorem_name: Optional theorem name
        
    Returns:
        RAG-augmented proof result
    """
    if not RAGBITS_INTEGRATION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LeanAide-Ragbits integration not available"
        )
    
    try:
        integration = get_ragbits_integration()
        
        result = await integration.generate_proof_with_retrieval(
            theorem_text=theorem_text,
            theorem_name=theorem_name
        )
        
        return result.to_dict()
    except Exception as e:
        logger.error(f"RAG proof generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG proof generation failed: {str(e)}"
        )


@router.post("/cav-nlp/formalize", response_model=CAVNLPFormalizeResponse)
async def cav_nlp_formalize(request: CAVNLPFormalizeRequest):
    """
    Formalize natural language to Lean code using CAV-NLP.
    
    Args:
        request: CAVNLPFormalizeRequest with natural language statement
        
    Returns:
        CAVNLPFormalizeResponse with formalized Lean code
    """
    if not CAV_NLP_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CAV-NLP integration not available"
        )
    
    try:
        math_service = get_math_service()
        if not math_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CAV-NLP math service not available"
            )
        
        # Use CAV-NLP for formalization
        result = await math_service.formalize_async(
            natural_language=request.natural_language,
            statement_type=request.statement_type,
            name=request.name or f"formalized_{hash(request.natural_language) % 10000}",
            domain=request.domain,
            use_constraints=request.use_constraints,
            timeout=request.timeout
        )
        
        return CAVNLPFormalizeResponse(
            success=result.get("success", False),
            lean_code=result.get("lean_code"),
            formalized_name=result.get("name", request.name or "unknown"),
            confidence=result.get("confidence", 0.0),
            constraints_used=result.get("constraints_used", []),
            verification_status=result.get("verification_status", "pending"),
            error=result.get("error"),
            metadata=result.get("metadata", {})
        )
    except Exception as e:
        logger.error(f"CAV-NLP formalization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CAV-NLP formalization failed: {str(e)}"
        )


@router.post("/cav-nlp/verify", response_model=CAVNLPVerifyResponse)
async def cav_nlp_verify(request: CAVNLPVerifyRequest):
    """
    Verify Lean code using CAV-NLP enhanced verification.
    
    Args:
        request: CAVNLPVerifyRequest with Lean code to verify
        
    Returns:
        CAVNLPVerifyResponse with verification results
    """
    if not CAV_NLP_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CAV-NLP integration not available"
        )
    
    try:
        enhanced_solver = get_enhanced_solver()
        math_service = get_math_service()
        
        if not enhanced_solver or not math_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CAV-NLP components not available"
            )
        
        # Perform semantic analysis if requested
        semantic_score = 0.0
        if request.use_semantic_analysis:
            semantic_result = await math_service.analyze_semantics_async(
                lean_code=request.lean_code,
                timeout=request.timeout
            )
            semantic_score = semantic_result.get("semantic_score", 0.0)
        
        # Perform constraint checking if requested
        constraint_satisfied = True
        if request.use_constraint_checking:
            constraint_result = await enhanced_solver.check_constraints_async(
                lean_code=request.lean_code,
                timeout=request.timeout
            )
            constraint_satisfied = constraint_result.get("satisfiable", True)
        
        return CAVNLPVerifyResponse(
            success=True,
            verified=semantic_score > 0.7 and constraint_satisfied,
            semantic_score=semantic_score,
            constraint_satisfied=constraint_satisfied,
            issues_found=[],
            suggestions=[],
            metadata={
                "semantic_analysis_used": request.use_semantic_analysis,
                "constraint_checking_used": request.use_constraint_checking
            }
        )
    except Exception as e:
        logger.error(f"CAV-NLP verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CAV-NLP verification failed: {str(e)}"
        )


# =============================================================================
# Include in Main API Server
# =============================================================================

# To include these routes in your main api_server.py, add:
#
# from leanaide_api_routes import router as leanaide_router
# app.include_router(leanaide_router)
#
# This will add all LeanAide endpoints under /api/leanaide/

if __name__ == "__main__":
    # Test the routes
    print("LeanAide API Routes module loaded")
    print("Available endpoints:")
    print("  GET  /api/leanaide/status")
    print("  POST /api/leanaide/verify")
    print("  POST /api/leanaide/prove")
    print("  POST /api/leanaide/translate")
    print("  POST /api/leanaide/quality-gate")
    print("  POST /api/leanaide/rag-retrieve")
    print("  POST /api/leanaide/rag-prove")
    print("  POST /api/leanaide/cav-nlp/formalize")
    print("  POST /api/leanaide/cav-nlp/verify")
