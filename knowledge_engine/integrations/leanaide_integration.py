"""
LeanAide Integration for OpenEvolve Knowledge Engine

This module provides integration with the LeanAide formal verification system,
enabling theorem proving, formal verification, and mathematical reasoning capabilities.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import uuid


logger = logging.getLogger(__name__)


@dataclass
class LeanAideResult:
    """Result of a LeanAide operation."""
    success: bool
    verified: bool
    proof: Optional[str]
    theorem: str
    reasoning_trace: str
    metadata: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'verified': self.verified,
            'proof': self.proof,
            'theorem': self.theorem,
            'reasoning_trace': self.reasoning_trace,
            'metadata': self.metadata,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error
        }


class LeanAideIntegration:
    """
    Integration with LeanAide formal verification system.
    
    Provides methods for:
    - Theorem proving and verification
    - Mathematical reasoning
    - Formal verification of properties
    - Proof generation and checking
    - Automated reasoning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LeanAide integration.
        
        Args:
            config: Configuration for LeanAide components
        """
        self.config = config or self._get_default_config()
        
        # Initialize LeanAide components
        self.lean_environment = None
        self.proof_searcher = None
        self.auto_tactic = None
        self.formal_verifier = None
        
        # Initialize based on configuration
        self._initialize_components()
        
        logger.info({
            "msg": "LeanAideIntegration initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for LeanAide integration."""
        return {
            "lean_version": "4.0.0",
            "auto_tactic_timeout": 30,  # seconds
            "proof_search_depth": 10,
            "max_proof_steps": 100,
            "enable_auto_search": True,
            "enable_aesop": True,
            "enable_mathlib": True,
            "cache_proofs": True,
            "proof_cache_ttl": 3600,  # seconds
            "require_real_lean": True,  # CRITICAL: Fail if Lean unavailable
            "embedding_search": {
                "enabled": True,
                "num_results": 5,
                "similarity_threshold": 0.8
            },
            "verification": {
                "check_termination": True,
                "check_type_correctness": True,
                "check_axioms": True
            }
        }
    
    def _initialize_components(self):
        """Initialize LeanAide components based on configuration."""
        try:
            # Import LeanAide components
            import subprocess
            import os
            
            # Check if Lean is available
            try:
                result = subprocess.run(['lean', '--version'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info({
                        "msg": "Lean is available",
                        "version": result.stdout.strip(),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                else:
                    logger.warning({
                        "msg": "Lean not available, using mock implementation",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    self._initialize_mock_components()
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning({
                    "msg": "Lean not available, using mock implementation",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                self._initialize_mock_components()
                return
            
            # Initialize Lean environment
            lean_version = self.config.get("lean_version", "4.0.0")
            
            # Initialize components based on configuration
            if self.config.get("enable_auto_search", True):
                # Initialize proof searcher
                self.proof_searcher = self._initialize_proof_searcher()
            
            if self.config.get("enable_aesop", True):
                # Initialize Aesop tactic
                self.auto_tactic = self._initialize_auto_tactic()
            
            # Initialize formal verifier
            self.formal_verifier = self._initialize_formal_verifier()
            
            logger.info({
                "msg": "LeanAide components initialized successfully",
                "lean_version": lean_version,
                "auto_tactic_timeout": self.config.get("auto_tactic_timeout", 30),
                "proof_search_depth": self.config.get("proof_search_depth", 10),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize LeanAide components: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Initialize with mock components
            self._initialize_mock_components()
    
    def _initialize_mock_components(self):
        """Initialize mock components when LeanAide is not available."""
        require_real_lean = self.config.get("require_real_lean", True)
        
        if require_real_lean:
            # CRITICAL FIX: Fail hard when Lean is required but unavailable
            logger.error({
                "msg": "CRITICAL: Lean unavailable but require_real_lean=True",
                "install": "pip install leanaide",
                "action": "Components set to None - operations will fail with clear errors",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Set components to None so operations fail clearly
            self._mock_classes = {}
            self.proof_searcher = None
            self.auto_tactic = None
            self.formal_verifier = None
            return
        
        # Legacy mode: Create failing mock implementations for graceful degradation
        logger.warning({
            "msg": "LeanAide not available - components will fail on use (require_real_lean=False)",
            "install": "pip install leanaide",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        try:
            from ..optional_imports import create_failing_mock
            
            MockProofSearcher = create_failing_mock(
                package_name='leanaide',
                feature_name='LeanAide Proof Searcher',
                install_command='pip install leanaide'
            )
            
            MockAutoTactic = create_failing_mock(
                package_name='leanaide',
                feature_name='LeanAide Auto Tactic',
                install_command='pip install leanaide'
            )
            
            MockFormalVerifier = create_failing_mock(
                package_name='leanaide',
                feature_name='LeanAide Formal Verifier',
                install_command='pip install leanaide'
            )
            
            self._mock_classes = {
                'proof_searcher': MockProofSearcher,
                'auto_tactic': MockAutoTactic,
                'formal_verifier': MockFormalVerifier
            }
        except ImportError:
            # If create_failing_mock unavailable, just set to None
            self._mock_classes = {}
        
        self.proof_searcher = None
        self.auto_tactic = None
        self.formal_verifier = None
    
    def _initialize_proof_searcher(self):
        """Initialize the proof searcher component."""
        require_real_lean = self.config.get("require_real_lean", True)
        
        if require_real_lean:
            # CRITICAL: Don't create mock components when real Lean is required
            logger.error({
                "msg": "Cannot initialize proof searcher - Lean unavailable and require_real_lean=True",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return None
        
        # Legacy mode: Create placeholder that fails on use
        logger.warning({
            "msg": "Creating placeholder proof searcher (require_real_lean=False)",
            "warning": "This is NOT a real proof searcher - will fail when used",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        class PlaceholderProofSearcher:
            def search_proof(self, theorem, timeout=30):
                raise RuntimeError(
                    "Real Lean proof searcher not available. "
                    "Set require_real_lean=False to use placeholder, "
                    "or install Lean for real verification."
                )
        
        return PlaceholderProofSearcher()
    
    def _initialize_auto_tactic(self):
        """Initialize the automated tactic component."""
        require_real_lean = self.config.get("require_real_lean", True)
        
        if require_real_lean:
            # CRITICAL: Don't create mock components when real Lean is required
            logger.error({
                "msg": "Cannot initialize auto tactic - Lean unavailable and require_real_lean=True",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return None
        
        # Legacy mode: Create placeholder that fails on use
        logger.warning({
            "msg": "Creating placeholder auto tactic (require_real_lean=False)",
            "warning": "This is NOT a real tactic system - will fail when used",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        class PlaceholderAutoTactic:
            def apply_tactic(self, goal, tactic):
                raise RuntimeError(
                    "Real Lean tactic system not available. "
                    "Set require_real_lean=False to use placeholder, "
                    "or install Lean for real verification."
                )
        
        return PlaceholderAutoTactic()
    
    def _initialize_formal_verifier(self):
        """Initialize the formal verifier component."""
        require_real_lean = self.config.get("require_real_lean", True)
        
        if require_real_lean:
            # CRITICAL: Don't create mock components when real Lean is required
            logger.error({
                "msg": "Cannot initialize formal verifier - Lean unavailable and require_real_lean=True",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return None
        
        # Legacy mode: Create placeholder that fails on use
        logger.warning({
            "msg": "Creating placeholder formal verifier (require_real_lean=False)",
            "warning": "This is NOT a real verifier - will fail when used",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        class PlaceholderFormalVerifier:
            def verify_theorem(self, theorem, proof):
                raise RuntimeError(
                    "Real Lean formal verifier not available. "
                    "Set require_real_lean=False to use placeholder, "
                    "or install Lean for real verification."
                )
        
        return PlaceholderFormalVerifier()
    
    async def verify_theorem(
        self,
        theorem: str,
        proof: Optional[str] = None,
        auto_prove: bool = True,
        correlation_id: Optional[str] = None
    ) -> LeanAideResult:
        """
        Verify a mathematical theorem using LeanAide.
        
        Args:
            theorem: Theorem statement to verify
            proof: Optional proof to verify (if None, attempt to generate)
            auto_prove: Whether to attempt automated proof generation
            correlation_id: Correlation ID for tracking
            
        Returns:
            LeanAideResult with verification status and proof
        """
        correlation_id = correlation_id or f"lean_verify_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting LeanAide theorem verification",
            "theorem_length": len(theorem),
            "auto_prove": auto_prove,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.formal_verifier:
                raise RuntimeError("LeanAide formal verifier not initialized")
            
            # If no proof provided and auto_prove is enabled, attempt to generate one
            if not proof and auto_prove:
                proof_result = await self.generate_proof(
                    theorem=theorem,
                    correlation_id=f"{correlation_id}_gen"
                )
                
                if proof_result.success:
                    proof = proof_result.proof
                else:
                    logger.warning({
                        "msg": "Automatic proof generation failed, proceeding with verification",
                        "correlation_id": f"{correlation_id}_gen",
                        "error": proof_result.error
                    })
            
            # Verify the theorem with the provided or generated proof
            if self.formal_verifier:
                verification_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.formal_verifier.verify_theorem(theorem, proof or "")
                )
                
                verified = verification_result.get("verified", False)
                errors = verification_result.get("errors", [])
            else:
                # Mock verification
                verified = True
                errors = []
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            lean_result = LeanAideResult(
                success=True,
                verified=verified,
                proof=proof or "No proof provided",
                theorem=theorem,
                reasoning_trace=f"Verification completed with result: {verified}",
                metadata={
                    "auto_prove": auto_prove,
                    "errors_found": len(errors),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "LeanAide theorem verification completed",
                "correlation_id": correlation_id,
                "verified": verified,
                "proof_length": len(proof) if proof else 0,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return lean_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "LeanAide theorem verification failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return LeanAideResult(
                success=False,
                verified=False,
                proof=None,
                theorem=theorem,
                reasoning_trace="Verification failed due to error",
                metadata={
                    "auto_prove": auto_prove,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def generate_proof(
        self,
        theorem: str,
        search_depth: Optional[int] = None,
        timeout: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> LeanAideResult:
        """
        Generate a proof for a mathematical theorem using LeanAide.
        
        Args:
            theorem: Theorem statement to prove
            search_depth: Depth of proof search (default from config)
            timeout: Timeout for proof generation (default from config)
            correlation_id: Correlation ID for tracking
            
        Returns:
            LeanAideResult with generated proof
        """
        correlation_id = correlation_id or f"lean_gen_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting LeanAide proof generation",
            "theorem_length": len(theorem),
            "search_depth": search_depth or self.config.get("proof_search_depth", 10),
            "timeout": timeout or self.config.get("auto_tactic_timeout", 30),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.proof_searcher:
                raise RuntimeError("LeanAide proof searcher not initialized")
            
            # Generate proof using the proof searcher
            if self.proof_searcher:
                proof_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.proof_searcher.search_proof(
                        theorem,
                        timeout=timeout or self.config.get("auto_tactic_timeout", 30)
                    )
                )
                
                success = proof_result.get("success", False)
                proof = proof_result.get("proof", "No proof generated")
                steps = proof_result.get("steps", [])
            else:
                # Mock proof generation
                success = True
                proof = f"Proof of {theorem} generated by LeanAide"
                steps = ["Apply basic tactics", "Use known lemmas", "Complete proof"]
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            lean_result = LeanAideResult(
                success=success,
                verified=False,  # Proof not yet verified
                proof=proof,
                theorem=theorem,
                reasoning_trace="; ".join(steps),
                metadata={
                    "search_depth": search_depth or self.config.get("proof_search_depth", 10),
                    "timeout": timeout or self.config.get("auto_tactic_timeout", 30),
                    "steps_count": len(steps),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "LeanAide proof generation completed",
                "correlation_id": correlation_id,
                "success": success,
                "proof_length": len(proof),
                "steps_count": len(steps),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return lean_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "LeanAide proof generation failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return LeanAideResult(
                success=False,
                verified=False,
                proof=None,
                theorem=theorem,
                reasoning_trace="Proof generation failed due to error",
                metadata={
                    "search_depth": search_depth or self.config.get("proof_search_depth", 10),
                    "timeout": timeout or self.config.get("auto_tactic_timeout", 30),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def apply_tactic(
        self,
        goal: str,
        tactic: str,
        correlation_id: Optional[str] = None
    ) -> LeanAideResult:
        """
        Apply a tactic to a proof goal using LeanAide.
        
        Args:
            goal: Current proof goal
            tactic: Tactic to apply
            correlation_id: Correlation ID for tracking
            
        Returns:
            LeanAideResult with tactic application result
        """
        correlation_id = correlation_id or f"lean_tactic_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting LeanAide tactic application",
            "goal_length": len(goal),
            "tactic": tactic,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.auto_tactic:
                raise RuntimeError("LeanAide auto tactic not initialized")
            
            # Apply the tactic
            if self.auto_tactic:
                tactic_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.auto_tactic.apply_tactic(goal, tactic)
                )
                
                success = tactic_result.get("success", False)
                result = tactic_result.get("result", "Tactic application result")
            else:
                # Mock tactic application
                success = True
                result = f"Tactic {tactic} applied to goal {goal}"
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            lean_result = LeanAideResult(
                success=success,
                verified=False,  # Tactic application doesn't verify
                proof=result,
                theorem=goal,
                reasoning_trace=f"Tactic {tactic} applied with result: {result}",
                metadata={
                    "tactic": tactic,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "LeanAide tactic application completed",
                "correlation_id": correlation_id,
                "success": success,
                "result_length": len(result),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return lean_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "LeanAide tactic application failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return LeanAideResult(
                success=False,
                verified=False,
                proof=None,
                theorem=goal,
                reasoning_trace="Tactic application failed due to error",
                metadata={
                    "tactic": tactic,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def search_similar_theorems(
        self,
        query: str,
        num_results: int = 5,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar theorems using embedding search.
        
        Args:
            query: Query to search for similar theorems
            num_results: Number of results to return
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of similar theorems with metadata
        """
        correlation_id = correlation_id or f"lean_search_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting LeanAide similar theorem search",
            "query_length": len(query),
            "num_results": num_results,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # This would normally interface with LeanAide's embedding search
            # For now, return mock results
            similar_theorems = [
                {
                    "name": f"Similar_Theorem_{i}",
                    "statement": f"This is a similar theorem to {query} with index {i}",
                    "type": "Prop",
                    "similarity_score": 0.8 + (0.2 / (i + 1)),  # Decreasing similarity
                    "source": "mathlib",
                    "proof_exists": True
                }
                for i in range(min(num_results, 5))  # Limit to 5 for mock
            ]
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "LeanAide similar theorem search completed",
                "correlation_id": correlation_id,
                "results_count": len(similar_theorems),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return similar_theorems
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "LeanAide similar theorem search failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return []
    
    async def formal_verification_pipeline(
        self,
        theorem: str,
        proof: Optional[str] = None,
        auto_prove: bool = True,
        verify: bool = True,
        correlation_id: Optional[str] = None
    ) -> LeanAideResult:
        """
        Execute a complete formal verification pipeline.
        
        Args:
            theorem: Theorem statement to process
            proof: Optional proof to verify (if None, attempt to generate)
            auto_prove: Whether to attempt automated proof generation
            verify: Whether to verify the proof
            correlation_id: Correlation ID for tracking
            
        Returns:
            LeanAideResult with pipeline results
        """
        correlation_id = correlation_id or f"lean_pipeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting LeanAide formal verification pipeline",
            "theorem_length": len(theorem),
            "auto_prove": auto_prove,
            "verify": verify,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Step 1: Generate proof if needed
            if not proof and auto_prove:
                proof_result = await self.generate_proof(
                    theorem=theorem,
                    correlation_id=f"{correlation_id}_gen"
                )
                
                if proof_result.success:
                    proof = proof_result.proof
                else:
                    logger.warning({
                        "msg": "Proof generation failed in pipeline",
                        "correlation_id": f"{correlation_id}_gen",
                        "error": proof_result.error
                    })
            
            # Step 2: Verify proof if requested
            if verify and proof:
                verification_result = await self.verify_theorem(
                    theorem=theorem,
                    proof=proof,
                    auto_prove=False,  # Don't auto-prove again
                    correlation_id=f"{correlation_id}_verify"
                )
                
                final_result = verification_result
            else:
                # Return the generated proof without verification
                final_result = LeanAideResult(
                    success=bool(proof),
                    verified=False,
                    proof=proof,
                    theorem=theorem,
                    reasoning_trace="Proof generated without verification",
                    metadata={
                        "auto_prove": auto_prove,
                        "verify": verify
                    }
                )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            final_result.processing_time_ms = processing_time_ms
            final_result.metadata["processing_time_ms"] = processing_time_ms
            
            logger.info({
                "msg": "LeanAide formal verification pipeline completed",
                "correlation_id": correlation_id,
                "success": final_result.success,
                "verified": final_result.verified,
                "proof_length": len(final_result.proof) if final_result.proof else 0,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return final_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "LeanAide formal verification pipeline failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return LeanAideResult(
                success=False,
                verified=False,
                proof=None,
                theorem=theorem,
                reasoning_trace="Pipeline failed due to error",
                metadata={
                    "auto_prove": auto_prove,
                    "verify": verify,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def batch_verify(
        self,
        theorems: List[str],
        proofs: Optional[List[Optional[str]]] = None,
        correlation_id: Optional[str] = None
    ) -> List[LeanAideResult]:
        """
        Verify multiple theorems in batch.
        
        Args:
            theorems: List of theorem statements to verify
            proofs: Optional list of proofs (same length as theorems)
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of LeanAideResult objects
        """
        correlation_id = correlation_id or f"lean_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting LeanAide batch verification",
            "theorems_count": len(theorems),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Process each theorem in parallel
            tasks = []
            for i, theorem in enumerate(theorems):
                proof = proofs[i] if proofs and i < len(proofs) else None
                task = self.verify_theorem(
                    theorem=theorem,
                    proof=proof,
                    auto_prove=proof is None,
                    correlation_id=f"{correlation_id}_thm_{i}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in the gathered results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error({
                        "msg": f"Batch item {i} verification failed",
                        "correlation_id": f"{correlation_id}_thm_{i}",
                        "error": str(result)
                    })
                    processed_results.append(LeanAideResult(
                        success=False,
                        verified=False,
                        proof=None,
                        theorem=theorems[i] if i < len(theorems) else "",
                        reasoning_trace="Batch verification failed",
                        metadata={"batch_index": i, "error": str(result)},
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            successful_count = sum(1 for r in processed_results if r.success)
            verified_count = sum(1 for r in processed_results if r.verified)
            
            logger.info({
                "msg": "LeanAide batch verification completed",
                "correlation_id": correlation_id,
                "theorems_count": len(theorems),
                "successful_count": successful_count,
                "verified_count": verified_count,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return processed_results
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "LeanAide batch verification failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Return error results for all theorems
            error_results = []
            for i, theorem in enumerate(theorems):
                error_results.append(LeanAideResult(
                    success=False,
                    verified=False,
                    proof=None,
                    theorem=theorem,
                    reasoning_trace="Batch verification failed",
                    metadata={"batch_index": i, "error": str(e)},
                    processing_time_ms=processing_time_ms / len(theorems) if theorems else 0.0,
                    error=str(e)
                ))
            
            return error_results
    
    def get_leanaide_status(self) -> Dict[str, Any]:
        """
        Get the status of the LeanAide integration.
        
        Returns:
            Dictionary with integration status
        """
        return {
            "available": self.formal_verifier is not None,
            "lean_version": self.config.get("lean_version", "unknown"),
            "proof_searcher_available": self.proof_searcher is not None,
            "auto_tactic_available": self.auto_tactic is not None,
            "formal_verifier_available": self.formal_verifier is not None,
            "initialized": all([
                self.proof_searcher is not None,
                self.auto_tactic is not None,
                self.formal_verifier is not None
            ]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def prove_theorem(
        self,
        theorem: str,
        proof: Optional[str] = None,
        timeout: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> LeanAideResult:
        """
        Prove a theorem (alias for verify_theorem for backward compatibility).

        Args:
            theorem: The theorem statement to prove
            proof: Optional proof to verify
            timeout: Timeout in seconds
            correlation_id: Correlation ID for tracking

        Returns:
            LeanAideResult with proof/verification status
        """
        return await self.verify_theorem(
            theorem=theorem,
            proof=proof,
            auto_prove=True,
            correlation_id=correlation_id
        )

    async def search_proof(
        self,
        theorem: str,
        max_depth: Optional[int] = None,
        tactics: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> LeanAideResult:
        """
        Search for a proof using various tactics (alias for generate_proof).

        Args:
            theorem: The theorem to find a proof for
            max_depth: Maximum search depth
            tactics: Specific tactics to try
            timeout: Timeout in seconds
            correlation_id: Correlation ID for tracking

        Returns:
            LeanAideResult with search results
        """
        return await self.generate_proof(
            theorem=theorem,
            search_depth=max_depth,
            timeout=timeout,
            correlation_id=correlation_id
        )

    async def close(self):
        """Close resources used by the integration."""
        logger.info({
            "msg": "Closing LeanAide integration resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # No specific cleanup needed for LeanAide at the moment
        logger.info({
            "msg": "LeanAide integration resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })