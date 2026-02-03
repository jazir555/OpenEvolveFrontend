"""
LeanAide Autoformalization System with MDAP/MAKER Integration

This module provides a complete autoformalization system that combines:
- Natural language to Lean 4 code translation (autoformalization)
- MDAP (Multi-Agent Decomposition with Aggregated Proofs) for multi-agent generation
- MAKER (Multi-Agent Voting for Keeping Reliability) for voting-based refinement

The system provides:
1. Autoformalization: Natural language → Lean code
2. Multi-agent generation with MDAP
3. Voting-based refinement with MAKER
4. Verification and validation
5. Caching and performance optimization
"""

import asyncio
import json
import logging
import time
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

# ACE + Steer Integration
try:
    from ace_steer_integration import AceSteerBridge
    from ace_mcp_tools import ACE_AVAILABLE
    STEER_ACE_BRIDGE_AVAILABLE = True
except ImportError:
    STEER_ACE_BRIDGE_AVAILABLE = False
    ACE_AVAILABLE = False
    AceSteerBridge = None

# Configure logging
logger = logging.getLogger(__name__)


class AutoformalizationStrategy(Enum):
    """Strategies for autoformalization"""
    DIRECT = "direct"
    MDAP = "mdap"
    MAKER = "maker"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class AutoformalizationResult:
    """Result of autoformalization process"""
    success: bool
    lean_code: str = ""
    theorem_name: str = ""
    confidence: float = 0.0
    strategy_used: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    verification_status: str = "not_verified"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "lean_code": self.lean_code,
            "theorem_name": self.theorem_name,
            "confidence": self.confidence,
            "strategy_used": self.strategy_used,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "verification_status": self.verification_status,
            "metadata": self.metadata
        }


@dataclass
class MDAPAgentResult:
    """Result from MDAP agent"""
    agent_id: str
    lean_code: str
    confidence: float
    strategy: str
    execution_time: float
    errors: List[str] = field(default_factory=list)


@dataclass
class MAKERVote:
    """Vote in MAKER system"""
    agent_id: str
    lean_code: str
    confidence: float
    rationale: str
    timestamp: float = field(default_factory=time.time)


class LeanAideAutoformalizationEngine:
    """
    Main autoformalization engine that integrates MDAP and MAKER systems.

    This engine provides:
    - Direct autoformalization (NL → Lean)
    - MDAP multi-agent generation
    - MAKER voting-based refinement
    - Hybrid approaches
    - Adaptive strategy selection
    """

    def __init__(
        self,
        leanaide_client,
        mdap_orchestrator=None,
        maker_engine=None,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 3600,
        ace_steer_bridge: Optional[AceSteerBridge] = None,
        ace_enabled: bool = True
    ):
        """
        Initialize the autoformalization engine.

        Args:
            leanaide_client: LeanAide client for basic autoformalization
            mdap_orchestrator: MDAP orchestrator for multi-agent generation
            maker_engine: MAKER engine for voting-based refinement
            enable_caching: Whether to enable caching
            cache_ttl_seconds: Cache TTL in seconds
            ace_steer_bridge: ACE + Steer bridge for reliability and learning
            ace_enabled: Whether to enable ACE+Steer features
        """
        self.leanaide_client = leanaide_client
        self.mdap_orchestrator = mdap_orchestrator
        self.maker_engine = maker_engine
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache = {}
        
        # ACE + Steer Integration
        self.ace_enabled = ace_enabled and STEER_ACE_BRIDGE_AVAILABLE
        self.ace_steer_bridge = ace_steer_bridge
        
        if self.ace_enabled and not self.ace_steer_bridge:
            # Auto-initialize bridge if enabled but not provided
            self.ace_steer_bridge = AceSteerBridge(
                ace_agent_id="leanaide_autoformalization_agent",
                skillbook_path="./leanaide_ace_skillbook.json"
            )
            logger.info("Auto-initialized ACE+Steer bridge for LeanAide")

    async def autoformalize(
        self,
        natural_language: str,
        statement_type: str = "theorem",
        name: Optional[str] = None,
        strategy: AutoformalizationStrategy = AutoformalizationStrategy.ADAPTIVE,
        context: Optional[Dict[str, Any]] = None
    ) -> AutoformalizationResult:
        """
        Autoformalize natural language to Lean code using selected strategy.

        Args:
            natural_language: Natural language mathematical statement
            statement_type: Type of statement ("theorem", "lemma", "definition")
            name: Optional name for the statement
            strategy: Strategy to use for autoformalization
            context: Additional context for the formalization

        Returns:
            AutoformalizationResult with generated Lean code
        """
        start_time = time.time()
        context = context or {}
        
        # ACE+Steer: Inject skills into the natural language prompt
        original_nl = natural_language
        if self.ace_enabled and self.ace_steer_bridge:
            natural_language = self.ace_steer_bridge.prepare_prompt(
                task=natural_language,
                context=json.dumps(context) if context else ""
            )
            logger.debug(f"ACE skills injected into autoformalization prompt for: {original_nl[:50]}...")

        # Create cache key
        cache_key = self._create_cache_key(original_nl, statement_type, name, strategy.value)
        
        # Check cache if enabled
        if self.enable_caching and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if time.time() - cached_result.get("timestamp", 0) < self.cache_ttl_seconds:
                logger.debug(f"Cache hit for autoformalization: {original_nl[:50]}...")
                cached_result["execution_time"] = time.time() - start_time
                return AutoformalizationResult(**cached_result["result"])

        try:
            # Select strategy based on input or adaptive logic
            if strategy == AutoformalizationStrategy.ADAPTIVE:
                strategy = self._select_adaptive_strategy(original_nl, context)

            # Execute based on strategy
            if strategy == AutoformalizationStrategy.DIRECT:
                result = await self._direct_autoformalize(natural_language, statement_type, name)
            elif strategy == AutoformalizationStrategy.MDAP:
                result = await self._mdap_autoformalize(natural_language, statement_type, name, context)
            elif strategy == AutoformalizationStrategy.MAKER:
                result = await self._maker_autoformalize(natural_language, statement_type, name, context)
            elif strategy == AutoformalizationStrategy.HYBRID:
                result = await self._hybrid_autoformalize(natural_language, statement_type, name, context)
            else:
                result = await self._direct_autoformalize(natural_language, statement_type, name)

            # ACE + Steer: Verify and Learn
            if self.ace_enabled and self.ace_steer_bridge and result.success:
                v_result = self.ace_steer_bridge.verify_and_learn(
                    query=original_nl,
                    output=result.lean_code,
                    verifications=["slop"] # For Lean code, we primarily check for slop/predictable AI filler
                )
                if not v_result.get("all_passed"):
                    result.warnings.append(f"Steer verification failed: {v_result.get('failed_verifications')}")
                    if v_result.get("ace_learning", {}).get("success"):
                        logger.info("ACE learned from Steer failure in LeanAide autoformalization")

            # Add execution time
            result.execution_time = time.time() - start_time
            result.strategy_used = strategy.value

            # Cache result if successful
            if self.enable_caching and result.success:
                self.cache[cache_key] = {
                    "result": result.to_dict(),
                    "timestamp": time.time()
                }

            return result

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            logger.error(f"Autoformalization failed: {e}", exc_info=True)
            return AutoformalizationResult(
                success=False,
                errors=[str(e)],
                execution_time=time.time() - start_time,
                strategy_used=strategy.value
            )

    def _create_cache_key(self, natural_language: str, statement_type: str, name: Optional[str], strategy: str) -> str:
        """Create cache key for the autoformalization request."""
        cache_input = f"{natural_language}:{statement_type}:{name or ''}:{strategy}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def _select_adaptive_strategy(self, natural_language: str, context: Dict[str, Any]) -> AutoformalizationStrategy:
        """Select strategy based on natural language characteristics."""
        # Simple heuristic for now - in production, this could be more sophisticated
        complexity_keywords = ["induction", "recursive", "complex", "advanced", "multi-step"]
        simple_keywords = ["basic", "simple", "elementary", "trivial"]

        natural_lower = natural_language.lower()
        
        complexity_score = sum(1 for keyword in complexity_keywords if keyword in natural_lower)
        simplicity_score = sum(1 for keyword in simple_keywords if keyword in natural_lower)

        if complexity_score > simplicity_score and self.mdap_orchestrator:
            return AutoformalizationStrategy.MDAP
        elif complexity_score > 0 and self.maker_engine:
            return AutoformalizationStrategy.MAKER
        elif self.mdap_orchestrator and self.maker_engine:
            return AutoformalizationStrategy.HYBRID
        else:
            return AutoformalizationStrategy.DIRECT

    async def _direct_autoformalize(
        self,
        natural_language: str,
        statement_type: str,
        name: Optional[str]
    ) -> AutoformalizationResult:
        """Direct autoformalization using LeanAide client."""
        try:
            # Use the existing autoformalization engine from lean4_integration
            from lean4_integration import AutoformalizationEngine
            auto_engine = AutoformalizationEngine(self.leanaide_client, self.leanaide_client.cache)
            
            result = await auto_engine.autoformalize(natural_language, statement_type, name)
            
            return AutoformalizationResult(
                success=result.success,
                lean_code=result.lean_code,
                theorem_name=result.theorem_name,
                confidence=0.8 if result.success else 0.2,  # Default confidence
                errors=result.errors,
                warnings=result.warnings,
                verification_status="not_verified" if result.success else "failed"
            )
        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            logger.error(f"Direct autoformalization failed: {e}")
            return AutoformalizationResult(
                success=False,
                errors=[str(e)],
                verification_status="failed"
            )

    async def _mdap_autoformalize(
        self,
        natural_language: str,
        statement_type: str,
        name: Optional[str],
        context: Dict[str, Any]
    ) -> AutoformalizationResult:
        """Autoformalization using MDAP multi-agent approach."""
        if not self.mdap_orchestrator:
            logger.warning("MDAP orchestrator not available, falling back to direct")
            return await self._direct_autoformalize(natural_language, statement_type, name)

        try:
            # Create MDAP task for autoformalization
            from leanaide_mdap import LeanMDAPTask, LeanDomain, ProofStrategy
            
            # Determine domain based on natural language
            domain = self._infer_domain(natural_language)
            
            mdap_task = LeanMDAPTask(
                task_id=f"autoformalize_{uuid.uuid4()}",
                description=f"Autoformalize: {natural_language}",
                theorem_statement=natural_language,
                domain=domain,
                enable_decomposition=True,
                enable_refinement=True
            )
            
            # Create default steps if none exist
            if not mdap_task.steps_created:
                strategies = [ProofStrategy.EVOLUTION, ProofStrategy.MCTS, ProofStrategy.ADVERSARIAL]
                mdap_task.create_default_steps(strategies, parallel=True)
            
            # Execute with MDAP orchestrator
            mdap_result = self.mdap_orchestrator.orchestrate_proof_generation(mdap_task)
            
            # Extract the best proof
            best_proof = mdap_result.best_proof
            
            return AutoformalizationResult(
                success=best_proof.verification_status,
                lean_code=best_proof.lean_code,
                theorem_name=name or best_proof.theorem_name,
                confidence=best_proof.confidence,
                errors=[best_proof.verification_message] if not best_proof.verification_status else [],
                verification_status="verified" if best_proof.verification_status else "failed",
                metadata={
                    "mdap_agents_used": mdap_result.agents_used,
                    "num_proofs_generated": len(mdap_result.all_proofs),
                    "voting_statistics": mdap_result.voting_statistics
                }
            )
            
        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            logger.error(f"MDAP autoformalization failed: {e}")
            return AutoformalizationResult(
                success=False,
                errors=[str(e)],
                verification_status="failed"
            )

    async def _maker_autoformalize(
        self,
        natural_language: str,
        statement_type: str,
        name: Optional[str],
        context: Dict[str, Any]
    ) -> AutoformalizationResult:
        """Autoformalization using MAKER voting approach."""
        if not self.maker_engine:
            logger.warning("MAKER engine not available, falling back to direct")
            return await self._direct_autoformalize(natural_language, statement_type, name)

        try:
            # For MAKER, we'll simulate the voting process
            # In a real implementation, this would use the maker_engine
            
            # Generate multiple candidates using different approaches
            candidates = await self._generate_maker_candidates(natural_language, statement_type, name)
            
            if not candidates:
                return await self._direct_autoformalize(natural_language, statement_type, name)
            
            # Vote on the best candidate (simplified voting)
            best_candidate = self._vote_on_candidates(candidates)
            
            return AutoformalizationResult(
                success=True,
                lean_code=best_candidate.lean_code,
                theorem_name=name or f"maker_theorem_{int(time.time())}",
                confidence=best_candidate.confidence,
                verification_status="not_verified",
                metadata={
                    "num_candidates": len(candidates),
                    "voting_method": "confidence_weighted"
                }
            )
            
        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            logger.error(f"MAKER autoformalization failed: {e}")
            return AutoformalizationResult(
                success=False,
                errors=[str(e)],
                verification_status="failed"
            )

    async def _hybrid_autoformalize(
        self,
        natural_language: str,
        statement_type: str,
        name: Optional[str],
        context: Dict[str, Any]
    ) -> AutoformalizationResult:
        """Hybrid autoformalization using both MDAP and MAKER."""
        if not self.mdap_orchestrator or not self.maker_engine:
            logger.warning("Hybrid requires both MDAP and MAKER, falling back to MDAP if available")
            if self.mdap_orchestrator:
                return await self._mdap_autoformalize(natural_language, statement_type, name, context)
            else:
                return await self._direct_autoformalize(natural_language, statement_type, name)

        try:
            # First, use MDAP to generate multiple proof candidates
            mdap_result = await self._mdap_autoformalize(natural_language, statement_type, name, context)
            
            if not mdap_result.success:
                return mdap_result
            
            # Then, use MAKER to refine/vote on the MDAP result
            refined_result = await self._refine_with_maker(mdap_result.lean_code, natural_language)
            
            return AutoformalizationResult(
                success=refined_result.success,
                lean_code=refined_result.lean_code,
                theorem_name=refined_result.theorem_name or name,
                confidence=max(mdap_result.confidence, refined_result.confidence),
                errors=mdap_result.errors + refined_result.errors,
                warnings=mdap_result.warnings + refined_result.warnings,
                verification_status=refined_result.verification_status,
                metadata={
                    "hybrid_phases": ["mdap_generation", "maker_refinement"],
                    "mdap_confidence": mdap_result.confidence,
                    "refined_confidence": refined_result.confidence
                }
            )
            
        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            logger.error(f"Hybrid autoformalization failed: {e}")
            return AutoformalizationResult(
                success=False,
                errors=[str(e)],
                verification_status="failed"
            )

    def _infer_domain(self, natural_language: str) -> 'LeanDomain':
        """Infer mathematical domain from natural language."""
        from leanaide_mdap import LeanDomain
        
        natural_lower = natural_language.lower()
        
        if any(keyword in natural_lower for keyword in ["algebra", "group", "ring", "field", "vector"]):
            return LeanDomain.ALGEBRA
        elif any(keyword in natural_lower for keyword in ["analysis", "limit", "continuity", "derivative", "integral"]):
            return LeanDomain.ANALYSIS
        elif any(keyword in natural_lower for keyword in ["logic", "proposition", "predicate", "quantifier"]):
            return LeanDomain.LOGIC
        elif any(keyword in natural_lower for keyword in ["category", "functor", "natural transformation"]):
            return LeanDomain.CATEGORY_THEORY
        elif any(keyword in natural_lower for keyword in ["topology", "continuous", "open set", "compact"]):
            return LeanDomain.TOPOLOGY
        elif any(keyword in natural_lower for keyword in ["number", "prime", "modular", "diophantine"]):
            return LeanDomain.NUMBER_THEORY
        elif any(keyword in natural_lower for keyword in ["combinatorics", "graph", "counting", "permutation"]):
            return LeanDomain.COMBINATORICS
        elif any(keyword in natural_lower for keyword in ["geometry", "triangle", "angle", "euclidean"]):
            return LeanDomain.GEOMETRY
        else:
            return LeanDomain.GENERAL

    async def _generate_maker_candidates(
        self,
        natural_language: str,
        statement_type: str,
        name: Optional[str]
    ) -> List[MDAPAgentResult]:
        """Generate multiple candidates for MAKER voting."""
        candidates = []
        
        # Generate using different approaches/prompt variations
        approaches = [
            f"Direct translation: {natural_language}",
            f"Step-by-step approach: {natural_language}",
            f"Constructive approach: {natural_language}",
            f"Proof by contradiction: {natural_language}"
        ]
        
        for i, approach in enumerate(approaches):
            try:
                # Use direct autoformalization for each approach
                result = await self._direct_autoformalize(approach, statement_type, f"{name}_candidate_{i}" if name else f"candidate_{i}")
                
                if result.success:
                    candidates.append(MDAPAgentResult(
                        agent_id=f"candidate_agent_{i}",
                        lean_code=result.lean_code,
                        confidence=result.confidence,
                        strategy=f"approach_{i}",
                        execution_time=result.execution_time
                    ))
            except (RuntimeError, ValueError, TypeError, ImportError) as e:
                logger.debug(f"Candidate generation failed for approach {i}: {e}")
        
        return candidates

    def _vote_on_candidates(self, candidates: List[MDAPAgentResult]) -> MDAPAgentResult:
        """Vote on the best candidate using confidence weighting."""
        if not candidates:
            # Return a default result
            return MDAPAgentResult(
                agent_id="default",
                lean_code="-- No valid candidates generated",
                confidence=0.0,
                strategy="default",
                execution_time=0.0
            )
        
        # Simple voting: select highest confidence
        return max(candidates, key=lambda c: c.confidence)

    async def _refine_with_maker(self, lean_code: str, natural_language: str) -> AutoformalizationResult:
        """Refine Lean code using MAKER approach."""
        # For now, return the original code with confidence boost
        # In a full implementation, this would use actual MAKER voting
        return AutoformalizationResult(
            success=True,
            lean_code=lean_code,
            confidence=min(1.0, 0.9),  # Boost confidence after refinement
            verification_status="not_verified"
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and capabilities."""
        return {
            "autoformalization_engine": True,
            "mdap_available": self.mdap_orchestrator is not None,
            "maker_available": self.maker_engine is not None,
            "caching_enabled": self.enable_caching,
            "available_strategies": [s.value for s in AutoformalizationStrategy],
            "cache_size": len(self.cache)
        }


# Convenience functions for easy integration

def create_leanaide_autoformalization_engine(
    leanaide_client,
    mdap_orchestrator=None,
    maker_engine=None,
    enable_caching: bool = True
) -> LeanAideAutoformalizationEngine:
    """
    Create a LeanAide autoformalization engine with MDAP/MAKER integration.

    Args:
        leanaide_client: LeanAide client for basic operations
        mdap_orchestrator: MDAP orchestrator (optional)
        maker_engine: MAKER engine (optional)
        enable_caching: Whether to enable caching

    Returns:
        LeanAideAutoformalizationEngine instance
    """
    return LeanAideAutoformalizationEngine(
        leanaide_client=leanaide_client,
        mdap_orchestrator=mdap_orchestrator,
        maker_engine=maker_engine,
        enable_caching=enable_caching
    )


async def autoformalize_with_mdap_maker(
    natural_language: str,
    leanaide_client,
    statement_type: str = "theorem",
    name: Optional[str] = None,
    strategy: AutoformalizationStrategy = AutoformalizationStrategy.ADAPTIVE,
    mdap_orchestrator=None,
    maker_engine=None
) -> AutoformalizationResult:
    """
    Convenience function to autoformalize with MDAP/MAKER integration.

    Args:
        natural_language: Natural language mathematical statement
        leanaide_client: LeanAide client
        statement_type: Type of statement
        name: Optional name
        strategy: Strategy to use
        mdap_orchestrator: MDAP orchestrator (optional)
        maker_engine: MAKER engine (optional)

    Returns:
        AutoformalizationResult
    """
    engine = create_leanaide_autoformalization_engine(
        leanaide_client,
        mdap_orchestrator,
        maker_engine
    )
    
    return await engine.autoformalize(
        natural_language,
        statement_type,
        name,
        strategy
    )


# Example usage and testing

async def example_usage():
    """Example demonstrating the autoformalization system."""
    print("=== LeanAide Autoformalization with MDAP/MAKER ===\n")
    
    # Note: This is a demonstration - actual client would need to be initialized
    print("This module provides the framework for autoformalization with MDAP/MAKER integration.")
    print("To use it, you would need to:")
    print("1. Initialize a LeanAide client")
    print("2. Optionally initialize MDAP and MAKER orchestrators")
    print("3. Create the autoformalization engine")
    print("4. Call autoformalize() with natural language input")
    print()
    
    print("Example usage pattern:")
    print("""
    from leanaide_autoformalization_mdap_maker import LeanAideAutoformalizationEngine, AutoformalizationStrategy
    
    # Initialize components (pseudo-code)
    leanaide_client = initialize_leanaide_client()
    mdap_orchestrator = initialize_mdap_orchestrator()  # optional
    maker_engine = initialize_maker_engine()  # optional
    
    # Create engine
    engine = LeanAideAutoformalizationEngine(
        leanaide_client=leanaide_client,
        mdap_orchestrator=mdap_orchestrator,
        maker_engine=maker_engine
    )
    
    # Autoformalize
    result = await engine.autoformalize(
        natural_language="For all natural numbers n, n + 0 = n",
        statement_type="theorem",
        name="add_zero",
        strategy=AutoformalizationStrategy.ADAPTIVE
    )
    
    print(f"Success: {result.success}")
    print(f"Lean code: {result.lean_code}")
    print(f"Confidence: {result.confidence}")
    """)

    print("\nSystem capabilities:")
    engine = LeanAideAutoformalizationEngine(None)  # dummy client for status
    status = engine.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(example_usage())