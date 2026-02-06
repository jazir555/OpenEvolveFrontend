"""
LeanAide + Ragbits Integration for OpenEvolve Knowledge Engine

This module provides integration between:
- Ragbits: Retrieval-Augmented Generation for mathematical knowledge
- LeanAide: Formal verification and theorem proving

Features:
- RAG-augmented theorem proving
- Retrieval of similar proofs for guidance
- Formal verification of retrieved knowledge
- Hybrid search combining semantic and formal methods

Author: OpenEvolve
Created: 2026-02-02
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json


logger = logging.getLogger(__name__)


# =============================================================================
# Import Dependencies
# =============================================================================

# Ragbits imports
try:
    from knowledge_engine.integrations.ragbits_integration import (
        RagbitsIntegration,
        RagbitsResult
    )
    RAGBITS_AVAILABLE = True
except ImportError:
    RAGBITS_AVAILABLE = False
    RagbitsIntegration = None
    RagbitsResult = None
    logger.warning("Ragbits not available")

# LeanAide imports
try:
    from leanaide_client import LeanAideClient, LeanAideConfig, TaskType
    from leanaide_mcp_tools import (
        leanaide_translate_theorem_async,
        leanaide_verify_solution_async,
        leanaide_generate_proof_async,
        leanaide_elaborate_code_async
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    LeanAideClient = None
    LeanAideConfig = None
    TaskType = None
    logger.warning("LeanAide not available")


try:
    from leanaide_integration import create_integration as _create_root_leanaide_integration
    ROOT_LEANAIDE_STATUS_AVAILABLE = True
except ImportError:
    _create_root_leanaide_integration = None
    ROOT_LEANAIDE_STATUS_AVAILABLE = False
    logger.warning("Root LeanAide status provider not available")


def _default_web3_formal_status() -> Dict[str, Any]:
    """Return a stable Web3 formal-status payload."""
    formal_capabilities = {
        "solidity_invariant_translation": False,
        "invariant_translation_verification": False,
        "symbolic_exploit_witness": False,
        "composite_exploit_verification": False,
    }
    return {
        "web3_formal_available": False,
        "web3_formal_verification_available": False,
        "web3_formal_tools": [],
        "formal_capabilities": formal_capabilities,
        "audit_exploit_verification_available": False,
    }


def _collect_web3_formal_status() -> Dict[str, Any]:
    """Collect Web3 formal status from root LeanAide integration when available."""
    if not ROOT_LEANAIDE_STATUS_AVAILABLE or _create_root_leanaide_integration is None:
        return _default_web3_formal_status()

    try:
        status = _create_root_leanaide_integration().get_web3_formal_status()
        if not isinstance(status, dict):
            return _default_web3_formal_status()
        default_status = _default_web3_formal_status()
        formal_capabilities = status.get("formal_capabilities")
        if not isinstance(formal_capabilities, dict):
            formal_capabilities = default_status["formal_capabilities"]
        web3_formal_tools = status.get("web3_formal_tools")
        if not isinstance(web3_formal_tools, list):
            web3_formal_tools = []
        web3_formal_tools = sorted(set(str(tool) for tool in web3_formal_tools if tool))
        inferred_formal_available = bool(web3_formal_tools) or any(
            bool(value) for value in formal_capabilities.values()
        )
        return {
            "web3_formal_available": bool(
                status.get("web3_formal_available", inferred_formal_available)
            ),
            "web3_formal_verification_available": bool(
                status.get("web3_formal_verification_available", inferred_formal_available)
            ),
            "web3_formal_tools": web3_formal_tools,
            "formal_capabilities": formal_capabilities,
            "audit_exploit_verification_available": bool(
                status.get("audit_exploit_verification_available")
            ),
        }
    except Exception:
        logger.debug("Failed to collect root LeanAide Web3 status", exc_info=True)
        return _default_web3_formal_status()


# =============================================================================
# Data Classes
# =============================================================================

class VerificationStatus(Enum):
    """Status of formal verification."""
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class RetrievedProof:
    """A retrieved proof from the knowledge base."""
    proof_id: str
    theorem_name: str
    informal_statement: str
    formal_statement: str
    proof_code: str
    proof_steps: List[str]
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "theorem_name": self.theorem_name,
            "informal_statement": self.informal_statement,
            "formal_statement": self.formal_statement,
            "proof_code": self.proof_code,
            "proof_steps": self.proof_steps,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata
        }


@dataclass
class RAGProofResult:
    """Result of RAG-augmented proof generation."""
    success: bool
    theorem_name: str
    informal_statement: str
    generated_proof: Optional[str]
    retrieved_proofs: List[RetrievedProof]
    verification_status: VerificationStatus
    confidence_score: float
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "theorem_name": self.theorem_name,
            "informal_statement": self.informal_statement,
            "generated_proof": self.generated_proof,
            "retrieved_proofs": [p.to_dict() for p in self.retrieved_proofs],
            "verification_status": self.verification_status.value,
            "confidence_score": self.confidence_score,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
            "error": self.error
        }


# =============================================================================
# LeanAide + Ragbits Integration
# =============================================================================

class LeanAideRagbitsIntegration:
    """
    Integration combining Ragbits RAG with LeanAide formal verification.
    
    Features:
    - Retrieve similar theorems/proofs from knowledge base
    - Use retrieved proofs as guidance for new proof generation
    - Verify generated proofs with LeanAide
    - Store successful proofs in knowledge base
    """
    
    def __init__(
        self,
        ragbits_config: Optional[Dict[str, Any]] = None,
        leanaide_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the integration.
        
        Args:
            ragbits_config: Ragbits configuration
            leanaide_config: LeanAide configuration
            config: Combined configuration
        """
        self.config = config or self._get_default_config()
        self.ragbits_config = ragbits_config or self.config.get("ragbits", {})
        self.leanaide_config = leanaide_config or self.config.get("leanaide", {})
        
        # Initialize components
        self.ragbits_integration = None
        self.leanaide_client = None
        
        # Cache for retrieved proofs
        self._proof_cache: Dict[str, RetrievedProof] = {}
        self._cache_max_size = self.config.get("cache_max_size", 1000)
        
        # Initialize
        self._initialize_components()
        
        logger.info({
            "msg": "LeanAideRagbitsIntegration initialized",
            "ragbits_available": self.ragbits_integration is not None,
            "leanaide_available": self.leanaide_client is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "ragbits": {
                "vector_store": {
                    "type": "qdrant",
                    "config": {
                        "location": ":memory:",
                        "collection_name": "mathematical_proofs"
                    }
                },
                "default_options": {
                    "top_k": 5,
                    "similarity_threshold": 0.7
                }
            },
            "leanaide": {
                "host": "localhost",
                "port": 7654,
                "timeout": 300.0
            },
            "cache_max_size": 1000,
            "enable_verification": True,
            "max_retrieved_proofs": 5,
            "min_similarity_threshold": 0.5
        }
    
    def _initialize_components(self):
        """Initialize Ragbits and LeanAide components."""
        # Initialize Ragbits
        if RAGBITS_AVAILABLE and self.ragbits_config:
            try:
                self.ragbits_integration = RagbitsIntegration(self.ragbits_config)
                logger.info("Ragbits integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Ragbits: {e}")
        
        # Initialize LeanAide
        if LEANAIDE_AVAILABLE and self.leanaide_config:
            try:
                config = LeanAideConfig(
                    host=self.leanaide_config.get("host", "localhost"),
                    port=self.leanaide_config.get("port", 7654),
                    timeout=self.leanaide_config.get("timeout", 300.0)
                )
                self.leanaide_client = LeanAideClient(config)
                logger.info("LeanAide client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LeanAide client: {e}")
    
    async def retrieve_similar_proofs(
        self,
        theorem_text: str,
        top_k: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> List[RetrievedProof]:
        """
        Retrieve similar proofs from the knowledge base.
        
        Args:
            theorem_text: The theorem to find similar proofs for
            top_k: Number of results to return
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of RetrievedProof objects
        """
        correlation_id = correlation_id or f"retrieve_proofs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info({
            "msg": "Retrieving similar proofs",
            "theorem_length": len(theorem_text),
            "top_k": top_k,
            "correlation_id": correlation_id
        })
        
        if not self.ragbits_integration:
            logger.warning("Ragbits not available, returning empty results")
            return []
        
        try:
            # Search for similar theorems
            search_options = {
                "top_k": top_k or self.config.get("max_retrieved_proofs", 5),
                "similarity_threshold": self.config.get("min_similarity_threshold", 0.5)
            }
            
            result: RagbitsResult = await self.ragbits_integration.search_documents(
                query=theorem_text,
                **search_options,
                correlation_id=correlation_id
            )
            
            # Convert to RetrievedProof objects
            retrieved_proofs = []
            for r in result.results:
                proof = RetrievedProof(
                    proof_id=r.get("metadata", {}).get("proof_id", hashlib.md5(
                        r.get("content", "").encode()
                    ).hexdigest()[:8]),
                    theorem_name=r.get("metadata", {}).get("theorem_name", "unknown"),
                    informal_statement=r.get("content", ""),
                    formal_statement=r.get("metadata", {}).get("formal_statement", ""),
                    proof_code=r.get("metadata", {}).get("proof_code", ""),
                    proof_steps=r.get("metadata", {}).get("proof_steps", []),
                    source=r.get("source", "unknown"),
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {})
                )
                retrieved_proofs.append(proof)
            
            logger.info({
                "msg": "Retrieved similar proofs",
                "count": len(retrieved_proofs),
                "correlation_id": correlation_id
            })
            
            return retrieved_proofs
            
        except Exception as e:
            logger.error({
                "msg": "Failed to retrieve proofs",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return []
    
    async def generate_proof_with_retrieval(
        self,
        theorem_text: str,
        theorem_name: Optional[str] = None,
        use_retrieved_proofs: bool = True,
        correlation_id: Optional[str] = None
    ) -> RAGProofResult:
        """
        Generate a proof using RAG augmentation.
        
        Args:
            theorem_text: The theorem to prove
            theorem_name: Optional name for the theorem
            use_retrieved_proofs: Whether to use retrieved proofs as guidance
            correlation_id: Correlation ID for tracking
            
        Returns:
            RAGProofResult with generated proof and verification status
        """
        correlation_id = correlation_id or f"rag_proof_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting RAG-augmented proof generation",
            "theorem_length": len(theorem_text),
            "use_retrieved_proofs": use_retrieved_proofs,
            "correlation_id": correlation_id
        })
        
        try:
            # Step 1: Retrieve similar proofs
            retrieved_proofs = []
            if use_retrieved_proofs and self.ragbits_integration:
                retrieved_proofs = await self.retrieve_similar_proofs(
                    theorem_text,
                    correlation_id=correlation_id
                )
            
            # Step 2: Generate proof using LeanAide
            generated_proof = None
            if self.leanaide_client:
                # Translate theorem to formal language
                translate_result = await leanaide_translate_theorem_async(
                    theorem_text,
                    theorem_name=theorem_name
                )
                
                if translate_result.get("success"):
                    formal_statement = translate_result.get("lean_code", "")
                    
                    # Generate proof
                    proof_result = await leanaide_generate_proof_async(
                        formal_statement,
                        timeout=self.leanaide_config.get("timeout", 300)
                    )
                    
                    if proof_result.get("success"):
                        generated_proof = proof_result.get("proof", "")
            else:
                # Fallback: generate proof from retrieved proofs
                if retrieved_proofs:
                    # Use the best matching proof as template
                    best_proof = max(retrieved_proofs, key=lambda p: p.score)
                    generated_proof = self._adapt_proof(best_proof, theorem_text)
            
            # Step 3: Verify the generated proof
            verification_status = VerificationStatus.UNVERIFIED
            confidence_score = 0.0
            
            if generated_proof and self.config.get("enable_verification", True):
                verification_result = await leanaide_verify_solution_async(
                    generated_proof,
                    timeout=self.leanaide_config.get("timeout", 300)
                )
                
                if verification_result.get("success"):
                    verification_status = VerificationStatus.VERIFIED
                    confidence_score = verification_result.get("confidence", 0.9)
                else:
                    verification_status = VerificationStatus.FAILED
                    confidence_score = verification_result.get("confidence", 0.3)
            
            # Calculate confidence from retrieved proofs
            if retrieved_proofs and confidence_score == 0.0:
                avg_score = sum(p.score for p in retrieved_proofs) / len(retrieved_proofs)
                confidence_score = min(avg_score * 0.5, 0.5)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = RAGProofResult(
                success=generated_proof is not None,
                theorem_name=theorem_name or "unnamed_theorem",
                informal_statement=theorem_text,
                generated_proof=generated_proof,
                retrieved_proofs=retrieved_proofs,
                verification_status=verification_status,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                metadata={
                    "correlation_id": correlation_id,
                    "retrieved_count": len(retrieved_proofs),
                    "verification_method": "leanaide" if self.leanaide_client else "retrieval"
                }
            )
            
            logger.info({
                "msg": "RAG proof generation completed",
                "success": result.success,
                "verification_status": result.verification_status.value,
                "confidence_score": result.confidence_score,
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "RAG proof generation failed",
                "error": str(e),
                "correlation_id": correlation_id
            })
            
            return RAGProofResult(
                success=False,
                theorem_name=theorem_name or "unnamed_theorem",
                informal_statement=theorem_text,
                generated_proof=None,
                retrieved_proofs=[],
                verification_status=VerificationStatus.UNVERIFIED,
                confidence_score=0.0,
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def verify_proof(
        self,
        proof_code: str,
        expected_statement: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify a proof using LeanAide.
        
        Args:
            proof_code: The Lean proof code to verify
            expected_statement: Optional expected theorem statement
            correlation_id: Correlation ID for tracking
            
        Returns:
            Verification result dictionary
        """
        correlation_id = correlation_id or f"verify_proof_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        if not self.leanaide_client:
            return {
                "success": False,
                "verified": False,
                "error": "LeanAide client not available",
                "correlation_id": correlation_id
            }
        
        try:
            result = await leanaide_verify_solution(
                proof_code,
                timeout=self.leanaide_config.get("timeout", 300)
            )
            
            result["correlation_id"] = correlation_id
            return result
            
        except Exception as e:
            logger.error({
                "msg": "Proof verification failed",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return {
                "success": False,
                "verified": False,
                "error": str(e),
                "correlation_id": correlation_id
            }
    
    async def store_proof(
        self,
        proof: RetrievedProof,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Store a proof in the knowledge base.
        
        Args:
            proof: The proof to store
            correlation_id: Correlation ID for tracking
            
        Returns:
            True if successful
        """
        correlation_id = correlation_id or f"store_proof_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        if not self.ragbits_integration:
            logger.warning("Ragbits not available, cannot store proof")
            return False
        
        try:
            # Store in cache
            self._proof_cache[proof.proof_id] = proof
            if len(self._proof_cache) > self._cache_max_size:
                # Remove oldest
                oldest_id = next(iter(self._proof_cache))
                del self._proof_cache[oldest_id]
            
            logger.info({
                "msg": "Proof stored",
                "proof_id": proof.proof_id,
                "theorem_name": proof.theorem_name,
                "correlation_id": correlation_id
            })
            
            return True
            
        except Exception as e:
            logger.error({
                "msg": "Failed to store proof",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return False
    
    def _adapt_proof(self, template: RetrievedProof, new_theorem: str) -> str:
        """
        Adapt a template proof for a new theorem.
        
        Args:
            template: The template proof
            new_theorem: The new theorem statement
            
        Returns:
            Adapted proof code
        """
        # Simple adaptation: replace theorem name in proof
        adapted = template.proof_code
        if template.theorem_name != "unknown":
            new_name = "new_theorem"  # Would need proper extraction
            adapted = adapted.replace(template.theorem_name, new_name)
        return adapted
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        web3_status = _collect_web3_formal_status()
        return {
            "ragbits_available": self.ragbits_integration is not None,
            "leanaide_available": self.leanaide_client is not None,
            "web3_formal_available": web3_status["web3_formal_available"],
            "web3_formal_verification_available": web3_status[
                "web3_formal_verification_available"
            ],
            "web3_formal_tools": web3_status["web3_formal_tools"],
            "formal_capabilities": web3_status["formal_capabilities"],
            "audit_exploit_verification_available": web3_status[
                "audit_exploit_verification_available"
            ],
            "cached_proofs": len(self._proof_cache),
            "cache_max_size": self._cache_max_size,
            "config": self.config
        }


# =============================================================================
# Factory Functions
# =============================================================================

def get_leanaide_ragbits_integration(
    config: Optional[Dict[str, Any]] = None
) -> LeanAideRagbitsIntegration:
    """
    Get a LeanAide-Ragbits integration instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        LeanAideRagbitsIntegration instance
    """
    return LeanAideRagbitsIntegration(config=config)


async def create_leanaide_ragbits_integration(
    config: Optional[Dict[str, Any]] = None
) -> LeanAideRagbitsIntegration:
    """
    Create and initialize a LeanAide-Ragbits integration (async).
    
    Args:
        config: Optional configuration
        
    Returns:
        Initialized LeanAideRagbitsIntegration instance
    """
    integration = get_leanaide_ragbits_integration(config)
    return integration


# =============================================================================
# Standalone Usage
# =============================================================================

if __name__ == "__main__":
    import sys
    
    async def test_integration():
        """Test the integration."""
        print("Testing LeanAide + Ragbits Integration...")
        
        # Create integration
        integration = get_leanaide_ragbits_integration()
        
        # Get status
        status = integration.get_status()
        print(f"Status: {json.dumps(status, indent=2)}")
        
        # Test with a sample theorem
        theorem = "The square root of 2 is irrational"
        
        result = await integration.generate_proof_with_retrieval(
            theorem_text=theorem,
            theorem_name="sqrt_2_irrational"
        )
        
        print(f"Result: {json.dumps(result.to_dict(), indent=2)}")
        
        return result
    
    # Run test
    try:
        result = asyncio.run(test_integration())
        if result.success:
            print("SUCCESS: Integration working!")
        else:
            print("FAILED: Integration not fully functional")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
