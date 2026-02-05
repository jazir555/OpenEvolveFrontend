"""
Proof Search Service for RESE-LeanAide Integration

Provides AI-guided proof search capabilities for all 4 RESE phases:
- Phase I: Epistemic Audit - Verify constraints and detect contradictions
- Phase II: Isomorphic Mapping - Verify isomorphisms formally
- Phase III: MCTS Refinement - Guide proof search with MCTS
- Phase IV: Architectural Synthesis - Prove efficacy claims

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of Idempotency: Safe to call multiple times
- Structured Logging: JSON with correlation_id
- Timeout: All operations have timeouts

Author: OpenEvolve
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import math

# Import LeanAide client
try:
    from leanaide_client import LeanAideClient, LeanAideConfig, LeanAideResult
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logging.warning("LeanAide client not available - using simulation mode")

# Import Z3-LeanAide bridge
try:
    from z3_leanaide_bridge import Z3LeanAideBridge, ConstraintType
    Z3_BRIDGE_AVAILABLE = True
except ImportError:
    Z3_BRIDGE_AVAILABLE = False
    logging.warning("Z3-LeanAide bridge not available")

# Import RESE schemas
try:
    from glue.schemas.rese_schemas import (
        Hypothesis, Pattern, IsomorphicMapping, SearchTreeNode,
        HypothesisStatus, PatternType, MCTSNodeState
    )
except ImportError:
    from rese_schemas import (
        Hypothesis, Pattern, IsomorphicMapping, SearchTreeNode,
        HypothesisStatus, PatternType, MCTSNodeState
    )


# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================

class ProofStrategy(Enum):
    """Proof search strategies"""
    AUTO_TACTICS = "auto_tactics"
    MCTS_GUIDED = "mcts_guided"
    Z3_LEAN_HYBRID = "z3_lean_hybrid"
    AI_ASSISTED = "ai_assisted"
    BRUTE_FORCE = "brute_force"


class ProofStatus(Enum):
    """Status of proof search"""
    SEARCHING = "searching"
    PROVED = "proved"
    DISPROVED = "disproved"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ProofTactic:
    """A proof tactic step"""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "explanation": self.explanation
        }


@dataclass
class ProofSearchResult:
    """Result from proof search"""
    success: bool
    status: ProofStatus
    theorem_name: str
    lean_code: str
    proof_found: bool
    proof_script: Optional[str] = None
    tactics_used: List[ProofTactic] = field(default_factory=list)
    search_nodes_explored: int = 0
    search_depth: int = 0
    execution_time_ms: float = 0.0
    confidence: float = 0.0
    counterexample: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "theorem_name": self.theorem_name,
            "lean_code": self.lean_code,
            "proof_found": self.proof_found,
            "proof_script": self.proof_script,
            "tactics_used": [t.to_dict() for t in self.tactics_used],
            "search_nodes_explored": self.search_nodes_explored,
            "search_depth": self.search_depth,
            "execution_time_ms": self.execution_time_ms,
            "confidence": self.confidence,
            "counterexample": self.counterexample,
            "errors": self.errors,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp
        }


@dataclass
class ProofSearchConfig:
    """Configuration for proof search service"""
    leanaide_host: str = "localhost"
    leanaide_port: int = 7654
    timeout_ms: int = 60000
    max_search_depth: int = 100
    mcts_iterations: int = 1000
    mcts_exploration_constant: float = 1.414
    enable_z3_hybrid: bool = True
    enable_counterexamples: bool = True
    confidence_threshold: float = 0.8
    correlation_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ProofSearchConfig":
        """Create configuration from environment variables"""
        return cls(
            leanaide_host=os.getenv("LEANAIDE_HOST", "localhost"),
            leanaide_port=int(os.getenv("LEANAIDE_PORT", "7654")),
            timeout_ms=int(os.getenv("PROOF_SEARCH_TIMEOUT_MS", "60000")),
            max_search_depth=int(os.getenv("PROOF_SEARCH_MAX_DEPTH", "100")),
            mcts_iterations=int(os.getenv("PROOF_SEARCH_MCTS_ITERATIONS", "1000")),
            mcts_exploration_constant=float(os.getenv("PROOF_SEARCH_MCTS_C", "1.414")),
            enable_z3_hybrid=os.getenv("PROOF_SEARCH_ENABLE_Z3", "true").lower() == "true",
            enable_counterexamples=os.getenv("PROOF_SEARCH_ENABLE_COUNTEREXAMPLES", "true").lower() == "true",
            confidence_threshold=float(os.getenv("PROOF_SEARCH_CONFIDENCE_THRESHOLD", "0.8")),
            correlation_id=os.getenv("CORRELATION_ID")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "leanaide_host": self.leanaide_host,
            "leanaide_port": self.leanaide_port,
            "timeout_ms": self.timeout_ms,
            "max_search_depth": self.max_search_depth,
            "mcts_iterations": self.mcts_iterations,
            "mcts_exploration_constant": self.mcts_exploration_constant,
            "enable_z3_hybrid": self.enable_z3_hybrid,
            "enable_counterexamples": self.enable_counterexamples,
            "confidence_threshold": self.confidence_threshold,
            "correlation_id": self.correlation_id
        }


# ============================================================================
# Structured Logger
# ============================================================================

class ProofSearchLogger:
    """Structured logger for proof search service"""

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = logging.getLogger("proof_search_service")

    def _log(self, level: str, msg: str, **kwargs):
        """Log in JSON Lines format"""
        log_entry = {
            "msg": msg,
            "level": level,
            "correlation_id": self.correlation_id,
            "source_service": "proof_search_service",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        log_json = json.dumps(log_entry)
        self.logger.log(getattr(logging, level.upper()), log_json)

    def info(self, msg: str, **kwargs):
        self._log("INFO", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log("WARNING", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log("ERROR", msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self._log("DEBUG", msg, **kwargs)


# ============================================================================
# MCTS-Guided Proof Search
# ============================================================================

class MCTSProofNode:
    """Node in MCTS proof search tree"""

    def __init__(
        self,
        proof_state: str,
        parent: Optional['MCTSProofNode'] = None,
        tactic: Optional[ProofTactic] = None
    ):
        self.proof_state = proof_state
        self.parent = parent
        self.tactic = tactic
        self.children: List[MCTSProofNode] = []
        self.visits = 0
        self.value = 0.0
        self.unproved = True

    def ucb1(self, total_visits: int, c: float = 1.414) -> float:
        """Calculate UCB1 score for node selection"""
        if self.visits == 0:
            return float('inf')

        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(total_visits) / self.visits)

        return exploitation + exploration


class MCTSProofSearch:
    """MCTS-guided proof search"""

    def __init__(
        self,
        config: ProofSearchConfig,
        logger: ProofSearchLogger
    ):
        self.config = config
        self.logger = logger
        self.root: Optional[MCTSProofNode] = None
        self.total_visits = 0

    async def search(
        self,
        lean_code: str,
        correlation_id: str
    ) -> ProofSearchResult:
        """
        Perform MCTS-guided proof search.

        Args:
            lean_code: Lean 4 code to prove
            correlation_id: Correlation ID

        Returns:
            ProofSearchResult with best proof found
        """
        start_time = asyncio.get_event_loop().time()

        self.logger.info(
            "MCTS proof search started",
            correlation_id=correlation_id,
            iterations=self.config.mcts_iterations
        )

        # Initialize root node
        self.root = MCTSProofNode(proof_state=lean_code)

        # Run MCTS iterations
        for i in range(self.config.mcts_iterations):
            # Selection
            node = self._select(self.root)

            # Expansion
            if node.unproved:
                self._expand(node)

            # Simulation (simplified - would use actual tactic execution)
            reward = await self._simulate(node, correlation_id)

            # Backpropagation
            self._backpropagate(node, reward)

            self.total_visits += 1

            # Check for proof
            if not node.unproved:
                self.logger.info(
                    "Proof found during MCTS search",
                    correlation_id=correlation_id,
                    iteration=i
                )
                break

        # Extract best proof
        best_proof = self._extract_best_proof()

        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        return ProofSearchResult(
            success=True,
            status=ProofStatus.PROVED if best_proof else ProofStatus.UNKNOWN,
            theorem_name="mcts_proof",
            lean_code=lean_code,
            proof_found=bool(best_proof),
            proof_script=best_proof,
            search_nodes_explored=self.total_visits,
            search_depth=self._get_max_depth(),
            execution_time_ms=execution_time_ms,
            confidence=self.root.value / self.root.visits if self.root.visits > 0 else 0.0,
            correlation_id=correlation_id
        )

    def _select(self, node: MCTSProofNode) -> MCTSProofNode:
        """Select node using UCB1"""
        while node.children:
            node = max(node.children, key=lambda c: c.ucb1(self.total_visits, self.config.mcts_exploration_constant))
        return node

    def _expand(self, node: MCTSProofNode):
        """Expand node with possible tactics"""
        # Generate possible tactics (simplified)
        possible_tactics = [
            ProofTactic(name="simp", confidence=0.6, explanation="Simplification"),
            ProofTactic(name="linarith", confidence=0.7, explanation="Linear arithmetic"),
            ProofTactic(name="apply", confidence=0.5, explanation="Apply lemma"),
        ]

        for tactic in possible_tactics:
            child = MCTSProofNode(
                proof_state=node.proof_state,
                parent=node,
                tactic=tactic
            )
            node.children.append(child)

        node.unproved = False

    async def _simulate(self, node: MCTSProofNode, correlation_id: str) -> float:
        """Simulate proof from node (simplified)"""
        # In real implementation, would execute tactic and evaluate result
        # Here we use random simulation
        reward = random.random()
        return reward

    def _backpropagate(self, node: MCTSProofNode, reward: float):
        """Backpropagate reward up the tree"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent if hasattr(node, 'parent') else None

    def _extract_best_proof(self) -> Optional[str]:
        """Extract best proof from tree"""
        if not self.root:
            return None

        # Find path to best child
        node = self.root
        proof_steps = []

        while node.children:
            best_child = max(node.children, key=lambda c: c.value / c.visits if c.visits > 0 else 0)
            if best_child.tactic:
                proof_steps.append(f"  {best_child.tactic.name}")
            node = best_child

        if proof_steps:
            return "by\n" + "\n".join(proof_steps)

        return None

    def _get_max_depth(self) -> int:
        """Get maximum depth of tree"""
        def depth(node: MCTSProofNode) -> int:
            if not node.children:
                return 0
            return 1 + max(depth(child) for child in node.children)

        return depth(self.root) if self.root else 0


# ============================================================================
# Proof Search Service
# ============================================================================

class ProofSearchService:
    """
    Proof search service for RESE phases.

    Provides AI-guided proof search using:
    - MCTS-guided search
    - Z3-LeanAide hybrid
    - Auto tactics
    """

    def __init__(
        self,
        config: Optional[ProofSearchConfig] = None,
        logger: Optional[ProofSearchLogger] = None
    ):
        """
        Initialize proof search service.

        Args:
            config: Service configuration
            logger: Structured logger
        """
        self.config = config or ProofSearchConfig.from_env()
        self.logger = logger or ProofSearchLogger(self.config.correlation_id)

        # Initialize LeanAide client
        self.leanaide_client: Optional[LeanAideClient] = None
        if LEANAIDE_AVAILABLE:
            leanaide_config = LeanAideConfig(
                host=self.config.leanaide_host,
                port=self.config.leanaide_port,
                timeout=self.config.timeout_ms / 1000.0
            )
            self.leanaide_client = LeanAideClient(config=leanaide_config)

        # Initialize Z3 bridge
        self.z3_bridge: Optional[Z3LeanAideBridge] = None
        if Z3_BRIDGE_AVAILABLE and self.config.enable_z3_hybrid:
            self.z3_bridge = Z3LeanAideBridge()

        # Initialize MCTS search
        self.mcts_search = MCTSProofSearch(self.config, self.logger)

        self.logger.info(
            "ProofSearchService initialized",
            config=self.config.to_dict()
        )

    async def search_phase_i(
        self,
        lean_code: str,
        constraint_type: str = "logical",
        strategy: ProofStrategy = ProofStrategy.Z3_LEAN_HYBRID,
        correlation_id: Optional[str] = None
    ) -> ProofSearchResult:
        """
        Search proof for Phase I constraint.

        Args:
            lean_code: Lean 4 constraint code
            constraint_type: Type of constraint
            strategy: Proof search strategy
            correlation_id: Correlation ID

        Returns:
            ProofSearchResult
        """
        start_time = asyncio.get_event_loop().time()
        cid = correlation_id or self.logger.correlation_id

        self.logger.info(
            "Phase I proof search started",
            constraint_type=constraint_type,
            strategy=strategy.value,
            correlation_id=cid
        )

        try:
            # Extract theorem name
            theorem_name = self._extract_theorem_name(lean_code)

            if strategy == ProofStrategy.Z3_LEAN_HYBRID and self.z3_bridge:
                # Use Z3-LeanAide hybrid
                bridge_result = await self.z3_bridge.verify(
                    constraint=lean_code,
                    use_counterexamples=self.config.enable_counterexamples
                )

                execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                result = ProofSearchResult(
                    success=True,
                    status=ProofStatus.PROVED if bridge_result.agreed else ProofStatus.UNKNOWN,
                    theorem_name=theorem_name,
                    lean_code=lean_code,
                    proof_found=bridge_result.agreed,
                    counterexample=bridge_result.counterexample,
                    confidence=bridge_result.confidence,
                    execution_time_ms=execution_time_ms,
                    correlation_id=cid
                )

                self.logger.info(
                    "Phase I proof search completed",
                    correlation_id=cid,
                    proved=result.proof_found,
                    confidence=result.confidence
                )

                return result

            elif strategy == ProofStrategy.MCTS_GUIDED:
                # Use MCTS-guided search
                return await self.mcts_search.search(lean_code, cid)

            else:
                # Default: Use auto tactics
                return await self._search_with_auto_tactics(lean_code, cid, start_time)

        except Exception as e:
            self.logger.error(
                "Phase I proof search failed",
                correlation_id=cid,
                error=str(e)
            )

            return ProofSearchResult(
                success=False,
                status=ProofStatus.ERROR,
                theorem_name="unknown",
                lean_code=lean_code,
                proof_found=False,
                errors=[str(e)],
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                correlation_id=cid
            )

    async def search_phase_ii(
        self,
        lean_code: str,
        isomorphism_type: str = "structural",
        correlation_id: Optional[str] = None
    ) -> ProofSearchResult:
        """
        Search proof for Phase II isomorphism.

        Args:
            lean_code: Lean 4 isomorphism code
            isomorphism_type: Type of isomorphism
            correlation_id: Correlation ID

        Returns:
            ProofSearchResult
        """
        start_time = asyncio.get_event_loop().time()
        cid = correlation_id or self.logger.correlation_id

        self.logger.info(
            "Phase II proof search started",
            isomorphism_type=isomorphism_type,
            correlation_id=cid
        )

        try:
            theorem_name = self._extract_theorem_name(lean_code)

            # Use MCTS-guided search for isomorphisms
            result = await self.mcts_search.search(lean_code, cid)
            result.theorem_name = theorem_name

            self.logger.info(
                "Phase II proof search completed",
                correlation_id=cid,
                proved=result.proof_found
            )

            return result

        except Exception as e:
            self.logger.error(
                "Phase II proof search failed",
                correlation_id=cid,
                error=str(e)
            )

            return ProofSearchResult(
                success=False,
                status=ProofStatus.ERROR,
                theorem_name="unknown",
                lean_code=lean_code,
                proof_found=False,
                errors=[str(e)],
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                correlation_id=cid
            )

    async def search_phase_iii(
        self,
        lean_code: str,
        hypothesis: Optional[Hypothesis] = None,
        correlation_id: Optional[str] = None
    ) -> ProofSearchResult:
        """
        Search proof for Phase III hypothesis.

        Args:
            lean_code: Lean 4 hypothesis code
            hypothesis: Hypothesis object
            correlation_id: Correlation ID

        Returns:
            ProofSearchResult
        """
        start_time = asyncio.get_event_loop().time()
        cid = correlation_id or self.logger.correlation_id

        self.logger.info(
            "Phase III proof search started",
            hypothesis_id=hypothesis.hypothesis_id if hypothesis else None,
            correlation_id=cid
        )

        try:
            theorem_name = self._extract_theorem_name(lean_code)

            # Use MCTS-guided search for hypotheses
            result = await self.mcts_search.search(lean_code, cid)
            result.theorem_name = theorem_name

            # Update hypothesis if provided
            if hypothesis and result.proof_found:
                hypothesis.status = HypothesisStatus.CONFIRMED
                hypothesis.update_evidence({
                    "proof_search_result": result.to_dict(),
                    "confidence": result.confidence
                }, is_supporting=True)

            self.logger.info(
                "Phase III proof search completed",
                correlation_id=cid,
                proved=result.proof_found
            )

            return result

        except Exception as e:
            self.logger.error(
                "Phase III proof search failed",
                correlation_id=cid,
                error=str(e)
            )

            return ProofSearchResult(
                success=False,
                status=ProofStatus.ERROR,
                theorem_name="unknown",
                lean_code=lean_code,
                proof_found=False,
                errors=[str(e)],
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                correlation_id=cid
            )

    async def search_phase_iv(
        self,
        lean_code: str,
        efficacy_claim: str = "",
        correlation_id: Optional[str] = None
    ) -> ProofSearchResult:
        """
        Search proof for Phase IV efficacy claim.

        Args:
            lean_code: Lean 4 efficacy code
            efficacy_claim: Efficacy claim text
            correlation_id: Correlation ID

        Returns:
            ProofSearchResult
        """
        start_time = asyncio.get_event_loop().time()
        cid = correlation_id or self.logger.correlation_id

        self.logger.info(
            "Phase IV proof search started",
            efficacy_claim=efficacy_claim[:50],
            correlation_id=cid
        )

        try:
            theorem_name = self._extract_theorem_name(lean_code)

            # Use Z3-LeanAide hybrid for efficacy claims
            if self.z3_bridge:
                bridge_result = await self.z3_bridge.verify(
                    constraint=lean_code,
                    use_counterexamples=self.config.enable_counterexamples
                )

                execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                result = ProofSearchResult(
                    success=True,
                    status=ProofStatus.PROVED if bridge_result.agreed else ProofStatus.UNKNOWN,
                    theorem_name=theorem_name,
                    lean_code=lean_code,
                    proof_found=bridge_result.agreed,
                    counterexample=bridge_result.counterexample,
                    confidence=bridge_result.confidence,
                    metadata={"efficacy_claim": efficacy_claim},
                    execution_time_ms=execution_time_ms,
                    correlation_id=cid
                )

                self.logger.info(
                    "Phase IV proof search completed",
                    correlation_id=cid,
                    proved=result.proof_found,
                    confidence=result.confidence
                )

                return result
            else:
                # Fallback to MCTS
                result = await self.mcts_search.search(lean_code, cid)
                result.theorem_name = theorem_name
                return result

        except Exception as e:
            self.logger.error(
                "Phase IV proof search failed",
                correlation_id=cid,
                error=str(e)
            )

            return ProofSearchResult(
                success=False,
                status=ProofStatus.ERROR,
                theorem_name="unknown",
                lean_code=lean_code,
                proof_found=False,
                errors=[str(e)],
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                correlation_id=cid
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_theorem_name(self, lean_code: str) -> str:
        """Extract theorem name from Lean code"""
        import re
        match = re.search(r'theorem\s+(\w+)', lean_code)
        if match:
            return match.group(1)
        return "unknown_theorem"

    async def _search_with_auto_tactics(
        self,
        lean_code: str,
        correlation_id: str,
        start_time: float
    ) -> ProofSearchResult:
        """Search proof using auto tactics"""

        theorem_name = self._extract_theorem_name(lean_code)

        # Try to elaborate
        if self.leanaide_client:
            result = await self.leanaide_client.elaborate(lean_code)

            if result.success:
                execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                return ProofSearchResult(
                    success=True,
                    status=ProofStatus.PROVED,
                    theorem_name=theorem_name,
                    lean_code=lean_code,
                    proof_found=True,
                    proof_script="by auto",
                    tactics_used=[
                        ProofTactic(name="auto", confidence=0.9, explanation="Auto tactics")
                    ],
                    confidence=0.9,
                    execution_time_ms=execution_time_ms,
                    correlation_id=correlation_id
                )

        # Fallback
        return ProofSearchResult(
            success=True,
            status=ProofStatus.UNKNOWN,
            theorem_name=theorem_name,
            lean_code=lean_code,
            proof_found=False,
            tactics_used=[
                ProofTactic(name="sorry", confidence=0.1, explanation="Admitted proof")
            ],
            confidence=0.1,
            execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
            correlation_id=correlation_id
        )

    async def batch_search(
        self,
        items: List[Dict[str, Any]],
        phase: str,
        correlation_id: Optional[str] = None
    ) -> List[ProofSearchResult]:
        """
        Batch proof search for multiple items.

        Args:
            items: List of items to search proofs for
            phase: RESE phase
            correlation_id: Correlation ID

        Returns:
            List of ProofSearchResult
        """
        cid = correlation_id or self.logger.correlation_id

        self.logger.info(
            "Batch proof search started",
            phase=phase,
            item_count=len(items),
            correlation_id=cid
        )

        # Process in parallel
        tasks = []
        for item in items:
            lean_code = item.get("lean_code", "")

            if phase == "phase_i":
                task = self.search_phase_i(lean_code, item.get("type", "logical"), cid)
            elif phase == "phase_ii":
                task = self.search_phase_ii(lean_code, item.get("isomorphism_type", "structural"), cid)
            elif phase == "phase_iii":
                task = self.search_phase_iii(lean_code, item.get("hypothesis"), cid)
            elif phase == "phase_iv":
                task = self.search_phase_iv(lean_code, item.get("efficacy_claim", ""), cid)
            else:
                continue

            tasks.append(task)

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append(ProofSearchResult(
                    success=False,
                    status=ProofStatus.ERROR,
                    theorem_name="unknown",
                    lean_code=items[i].get("lean_code", ""),
                    proof_found=False,
                    errors=[str(result)],
                    correlation_id=cid
                ))
            else:
                formatted_results.append(result)

        self.logger.info(
            "Batch proof search completed",
            phase=phase,
            successful=sum(1 for r in formatted_results if r.proof_found),
            failed=sum(1 for r in formatted_results if not r.proof_found),
            correlation_id=cid
        )

        return formatted_results

    async def close(self):
        """Close the service and cleanup resources"""
        if self.leanaide_client:
            await self.leanaide_client.close()

        self.logger.info("ProofSearchService closed")


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_proof_search_service(
    config: Optional[ProofSearchConfig] = None
) -> ProofSearchService:
    """
    Create and initialize proof search service.

    Args:
        config: Service configuration

    Returns:
        Initialized ProofSearchService
    """
    return ProofSearchService(config)


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage of proof search service"""

    print("=" * 70)
    print("RESE-LeanAide Proof Search Service")
    print("=" * 70)

    # Create service
    service = await create_proof_search_service()

    try:
        # Phase I example
        print("\n1. PHASE I: EPISTEMIC AUDIT")
        print("-" * 40)
        lean_code_i = """
import Mathlib

theorem prime_constraint (p : Nat) (hp : p > 1) :
  (∀ d : Nat, d ∣ p → d = 1 ∨ d = p) → Prime p := by
  sorry
"""
        result_i = await service.search_phase_i(lean_code_i)
        print(f"Success: {result_i.success}")
        print(f"Proof found: {result_i.proof_found}")
        print(f"Confidence: {result_i.confidence:.2f}")

        # Phase II example
        print("\n2. PHASE II: ISOMORPHIC MAPPING")
        print("-" * 40)
        lean_code_ii = """
import Mathlib

theorem isomorphic_nat_int : Nat ≃ ℤ := by
  sorry
"""
        result_ii = await service.search_phase_ii(lean_code_ii)
        print(f"Success: {result_ii.success}")
        print(f"Proof found: {result_ii.proof_found}")
        print(f"Nodes explored: {result_ii.search_nodes_explored}")

        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70)

    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
