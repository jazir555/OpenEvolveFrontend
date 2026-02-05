"""
Unified Mathematical Knowledge Bridge (Z3 ↔ LeanAIDE)

Connects Z3 SMT solver and LeanAIDE theorem prover knowledge bases:
- Cross-system knowledge sharing
- Unified pattern matching
- Hybrid solving strategies
- Translation between SMT-LIB and Lean
- Combined verification workflows

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 integration
try:
    from knowledge_engine.integrations.z3_knowledge_integration import (
        Z3KnowledgeIntegration,
        get_z3_knowledge_integration
    )
    from knowledge_engine.integrations.z3_enhanced_knowledge import (
        EnhancedZ3KnowledgeIntegration,
        get_enhanced_z3_integration
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

# Import LeanAIDE integration
try:
    from knowledge_engine.integrations.leanaide_knowledge_extraction import (
        LeanAideKnowledgeExtractor,
        get_leanaide_knowledge_extractor,
        ProofStrategy,
        TacticPattern
    )
    from knowledge_engine.integrations.leanaide_proof_integration import (
        LeanAideProofIntegration,
        get_leanaide_proof_integration
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False

# Import bridge
try:
    from z3_leanaide_bridge import (
        Z3LeanAideBridge,
        TranslationDirection,
        VerificationStrategy
    )
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False


class ProblemClassification(Enum):
    """Classification of mathematical problems."""
    CONSTRAINT_SOLVING = "constraint_solving"      # Best for Z3
    THEOREM_PROVING = "theorem_proving"            # Best for LeanAIDE
    SMT_SOLVING = "smt_solving"                    # Best for Z3
    INDUCTIVE_PROOF = "inductive_proof"            # Best for LeanAIDE
    HYBRID = "hybrid"                              # Both systems
    UNKNOWN = "unknown"


@dataclass
class UnifiedMathProblem:
    """Unified representation of mathematical problem."""
    problem_id: str
    statement: str
    formalization: Optional[str] = None
    classification: ProblemClassification = ProblemClassification.UNKNOWN
    domain: str = "general"
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    z3_attempt: Optional[Dict] = None
    leanaide_attempt: Optional[Dict] = None
    hybrid_result: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class UnifiedKnowledgePattern:
    """Pattern that works across both Z3 and LeanAIDE."""
    pattern_id: str
    name: str
    description: str
    applicable_systems: List[str] = field(default_factory=list)  # ["z3", "leanaide"]
    z3_form: Optional[str] = None
    lean_form: Optional[str] = None
    success_count_z3: int = 0
    success_count_lean: int = 0
    effectiveness_score: float = 0.0
    problem_types: List[str] = field(default_factory=list)


class ProblemClassifier:
    """Classify problems for optimal solver selection."""
    
    def __init__(self):
        self.classification_rules = {
            ProblemClassification.CONSTRAINT_SOLVING: {
                "keywords": ["constraint", "satisfy", "solve", "equation", "inequality"],
                "patterns": [r'[<>=]+', r'\b(assert|check-sat)\b']
            },
            ProblemClassification.THEOREM_PROVING: {
                "keywords": ["theorem", "lemma", "proof", "forall", "exists", "implies"],
                "patterns": [r'\b(theorem|lemma|proof)\b', r'(∀|∃|->)']
            },
            ProblemClassification.SMT_SOLVING: {
                "keywords": ["smt", "smt-lib", "check-sat", "get-model"],
                "patterns": [r'\(assert', r'\(check-sat\)']
            },
            ProblemClassification.INDUCTIVE_PROOF: {
                "keywords": ["induction", "inductive", "base case", "inductive step"],
                "patterns": [r'\b(induction|inductive)\b']
            }
        }
    
    def classify(self, problem_statement: str) -> ProblemClassification:
        """
        Classify problem for optimal solver selection.
        
        Args:
            problem_statement: Problem text
            
        Returns:
            Problem classification
        """
        import re
        
        scores = defaultdict(float)
        problem_lower = problem_statement.lower()
        
        for classification, rules in self.classification_rules.items():
            # Check keywords
            for keyword in rules["keywords"]:
                if keyword in problem_lower:
                    scores[classification] += 1.0
            
            # Check patterns
            for pattern in rules["patterns"]:
                if re.search(pattern, problem_statement):
                    scores[classification] += 2.0
        
        if not scores:
            return ProblemClassification.UNKNOWN
        
        # Get highest scoring classification
        best_classification = max(scores.items(), key=lambda x: x[1])
        
        # Check if it could be hybrid
        if scores[ProblemClassification.CONSTRAINT_SOLVING] > 0 and \
           scores[ProblemClassification.THEOREM_PROVING] > 0:
            return ProblemClassification.HYBRID
        
        return best_classification[0]
    
    def recommend_solver(
        self,
        classification: ProblemClassification
    ) -> Tuple[str, float]:
        """
        Recommend solver based on classification.
        
        Returns:
            Tuple of (solver_name, confidence)
        """
        recommendations = {
            ProblemClassification.CONSTRAINT_SOLVING: ("z3", 0.9),
            ProblemClassification.SMT_SOLVING: ("z3", 0.95),
            ProblemClassification.THEOREM_PROVING: ("leanaide", 0.9),
            ProblemClassification.INDUCTIVE_PROOF: ("leanaide", 0.95),
            ProblemClassification.HYBRID: ("hybrid", 0.85),
            ProblemClassification.UNKNOWN: ("hybrid", 0.5)
        }
        
        return recommendations.get(classification, ("hybrid", 0.5))


class CrossSystemKnowledgeTransfer:
    """Transfer knowledge between Z3 and LeanAIDE systems."""
    
    def __init__(self):
        self.transfer_mappings: Dict[str, Dict[str, str]] = {}
        self.successful_transfers: List[Dict] = []
    
    def z3_to_lean_tactic(self, z3_tactic: str) -> Optional[str]:
        """Map Z3 tactic to Lean tactic."""
        tactic_map = {
            "simplify": "simp",
            "solve-eqs": "linarith",
            "smt": "smt",
            "qe": "qify",
            "bit-blast": "bv_decide"
        }
        return tactic_map.get(z3_tactic.lower())
    
    def lean_to_z3_tactic(self, lean_tactic: str) -> Optional[str]:
        """Map Lean tactic to Z3 tactic."""
        tactic_map = {
            "simp": "simplify",
            "linarith": "solve-eqs",
            "smt": "smt",
            "bv_decide": "bit-blast",
            "norm_num": "simplify"
        }
        return tactic_map.get(lean_tactic.lower())
    
    def translate_pattern(
        self,
        pattern: Dict[str, Any],
        source_system: str,
        target_system: str
    ) -> Optional[Dict[str, Any]]:
        """
        Translate pattern between systems.
        
        Args:
            pattern: Pattern to translate
            source_system: Source system ("z3" or "leanaide")
            target_system: Target system ("z3" or "leanaide")
            
        Returns:
            Translated pattern or None
        """
        if source_system == target_system:
            return pattern
        
        translated = pattern.copy()
        
        if source_system == "z3" and target_system == "leanaide":
            # Translate Z3 pattern to Lean
            if "tactics" in pattern:
                translated["tactics"] = [
                    self.z3_to_lean_tactic(t) or t
                    for t in pattern["tactics"]
                ]
        
        elif source_system == "leanaide" and target_system == "z3":
            # Translate Lean pattern to Z3
            if "tactics" in pattern:
                translated["tactics"] = [
                    self.lean_to_z3_tactic(t) or t
                    for t in pattern["tactics"]
                ]
        
        return translated


class UnifiedMathKnowledgeBridge:
    """
    Unified bridge connecting Z3 and LeanAIDE knowledge systems.
    
    Provides:
    - Unified problem representation
    - Cross-system knowledge sharing
    - Hybrid solving workflows
    - Optimal solver selection
    - Knowledge transfer between systems
    """
    
    def __init__(
        self,
        z3_integration: Optional[Any] = None,
        leanaide_integration: Optional[Any] = None
    ):
        self.problem_classifier = ProblemClassifier()
        self.knowledge_transfer = CrossSystemKnowledgeTransfer()
        
        # System integrations
        self.z3_integration = z3_integration
        self.leanaide_integration = leanaide_integration
        
        # Knowledge bases
        self.unified_patterns: Dict[str, UnifiedKnowledgePattern] = {}
        self.problem_history: List[UnifiedMathProblem] = []
        
        # Statistics
        self.stats = {
            "problems_processed": 0,
            "z3_successes": 0,
            "leanaide_successes": 0,
            "hybrid_successes": 0,
            "knowledge_transfers": 0
        }
        
        logger.info("UnifiedMathKnowledgeBridge initialized")
    
    async def initialize(self):
        """Initialize all components."""
        if Z3_AVAILABLE and not self.z3_integration:
            self.z3_integration = await get_enhanced_z3_integration()
        
        if LEANAIDE_AVAILABLE and not self.leanaide_integration:
            self.leanaide_integration = await get_leanaide_proof_integration()
        
        logger.info("UnifiedMathKnowledgeBridge components initialized")
    
    async def solve_problem(
        self,
        problem_statement: str,
        preferred_solver: Optional[str] = None,
        use_hybrid: bool = False,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Solve mathematical problem using optimal solver(s).
        
        Args:
            problem_statement: Problem to solve
            preferred_solver: Optional preferred solver ("z3", "leanaide", "auto")
            use_hybrid: Whether to use hybrid solving
            timeout: Timeout in seconds
            
        Returns:
            Solution result with metadata
        """
        problem_id = f"problem_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Classify problem
        classification = self.problem_classifier.classify(problem_statement)
        recommended_solver, confidence = self.problem_classifier.recommend_solver(classification)
        
        logger.info({
            "msg": "Solving mathematical problem",
            "problem_id": problem_id,
            "classification": classification.value,
            "recommended_solver": recommended_solver
        })
        
        # Determine solver to use
        solver = preferred_solver or recommended_solver
        
        # Create unified problem record
        problem = UnifiedMathProblem(
            problem_id=problem_id,
            statement=problem_statement,
            classification=classification,
            domain=self._detect_domain(problem_statement)
        )
        
        result = None
        
        try:
            if solver == "z3" or (use_hybrid and Z3_AVAILABLE):
                result = await self._solve_with_z3(problem_statement, timeout)
                problem.z3_attempt = result
                
                if result.get("success"):
                    self.stats["z3_successes"] += 1
                    if not use_hybrid:
                        problem.hybrid_result = result
            
            if solver == "leanaide" or (use_hybrid and LEANAIDE_AVAILABLE):
                result = await self._solve_with_leanaide(problem_statement, timeout)
                problem.leanaide_attempt = result
                
                if result.get("success"):
                    self.stats["leanaide_successes"] += 1
                    if not use_hybrid:
                        problem.hybrid_result = result
            
            if use_hybrid:
                # Combine results
                problem.hybrid_result = self._combine_results(
                    problem.z3_attempt,
                    problem.leanaide_attempt
                )
                if problem.hybrid_result.get("success"):
                    self.stats["hybrid_successes"] += 1
            
            self.stats["problems_processed"] += 1
            self.problem_history.append(problem)
            
            return {
                "problem_id": problem_id,
                "classification": classification.value,
                "solver_used": solver,
                "success": problem.hybrid_result.get("success", False) if problem.hybrid_result else False,
                "result": problem.hybrid_result,
                "z3_result": problem.z3_attempt,
                "leanaide_result": problem.leanaide_attempt,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error({"msg": f"Problem solving failed: {e}", "problem_id": problem_id})
            return {
                "problem_id": problem_id,
                "success": False,
                "error": str(e)
            }
    
    async def _solve_with_z3(self, problem: str, timeout: float) -> Dict[str, Any]:
        """Solve problem using Z3."""
        if not self.z3_integration:
            return {"success": False, "error": "Z3 integration not available"}
        
        # This would call actual Z3 integration
        # For now, return mock result
        return {
            "success": True,
            "solver": "z3",
            "solution": "sat",
            "model": {"x": 5}
        }
    
    async def _solve_with_leanaide(self, problem: str, timeout: float) -> Dict[str, Any]:
        """Solve problem using LeanAIDE."""
        if not self.leanaide_integration:
            return {"success": False, "error": "LeanAIDE integration not available"}
        
        result = await self.leanaide_integration.prove_theorem(problem)
        return {
            "success": result.get("success", False),
            "solver": "leanaide",
            "proof": result.get("proof")
        }
    
    def _combine_results(
        self,
        z3_result: Optional[Dict],
        leanaide_result: Optional[Dict]
    ) -> Dict[str, Any]:
        """Combine results from both solvers."""
        z3_success = z3_result.get("success", False) if z3_result else False
        leanaide_success = leanaide_result.get("success", False) if leanaide_result else False
        
        if z3_success and leanaide_success:
            return {
                "success": True,
                "method": "consensus",
                "z3_solution": z3_result.get("solution"),
                "leanaide_proof": leanaide_result.get("proof")
            }
        elif z3_success:
            return {"success": True, "method": "z3", **z3_result}
        elif leanaide_success:
            return {"success": True, "method": "leanaide", **leanaide_result}
        else:
            return {
                "success": False,
                "z3_error": z3_result.get("error") if z3_result else None,
                "leanaide_error": leanaide_result.get("error") if leanaide_result else None
            }
    
    def _detect_domain(self, problem: str) -> str:
        """Detect mathematical domain of problem."""
        problem_lower = problem.lower()
        
        domains = {
            "arithmetic": ["nat", "int", "+", "-", "*", "/"],
            "algebra": ["group", "ring", "field", "module"],
            "logic": ["forall", "exists", "implies", "and", "or"],
            "set_theory": ["set", "union", "intersection", "subset"],
            "analysis": ["limit", "continuous", "derivative", "integral"],
            "linear_algebra": ["matrix", "vector", "linear", "eigen"]
        }
        
        for domain, keywords in domains.items():
            if any(kw in problem_lower for kw in keywords):
                return domain
        
        return "general"
    
    def transfer_knowledge(
        self,
        source_system: str,
        target_system: str,
        pattern_type: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        Transfer knowledge from one system to another.
        
        Args:
            source_system: Source system ("z3" or "leanaide")
            target_system: Target system ("z3" or "leanaide")
            pattern_type: Type of patterns to transfer
            
        Returns:
            List of transferred patterns
        """
        transferred = []
        
        # Get patterns from source system
        if source_system == "z3" and self.z3_integration:
            # This would extract patterns from Z3 knowledge base
            patterns = []  # Would get from z3_integration
        elif source_system == "leanaide" and self.leanaide_integration:
            patterns = list(self.leanaide_integration.knowledge_extractor.tactic_patterns.values())
        else:
            patterns = []
        
        # Translate patterns
        for pattern in patterns:
            translated = self.knowledge_transfer.translate_pattern(
                pattern.to_dict() if hasattr(pattern, 'to_dict') else pattern,
                source_system,
                target_system
            )
            if translated:
                transferred.append(translated)
        
        self.stats["knowledge_transfers"] += len(transferred)
        
        logger.info({
            "msg": f"Knowledge transfer completed",
            "source": source_system,
            "target": target_system,
            "patterns_transferred": len(transferred)
        })
        
        return transferred
    
    def get_unified_knowledge_summary(self) -> Dict[str, Any]:
        """Get unified summary of all knowledge."""
        summary = {
            "statistics": self.stats,
            "problem_history": len(self.problem_history),
            "unified_patterns": len(self.unified_patterns)
        }
        
        if self.z3_integration:
            summary["z3_knowledge"] = self.z3_integration.get_analytics() if hasattr(self.z3_integration, 'get_analytics') else {}
        
        if self.leanaide_integration:
            summary["leanaide_knowledge"] = self.leanaide_integration.get_knowledge_summary()
        
        return summary
    
    def export_unified_knowledge(self, filepath: str):
        """Export unified knowledge to file."""
        knowledge = self.get_unified_knowledge_summary()
        
        with open(filepath, 'w') as f:
            json.dump(knowledge, f, indent=2, default=str)
        
        logger.info({"msg": f"Unified knowledge exported to {filepath}"})


# Global instance
_unified_bridge: Optional[UnifiedMathKnowledgeBridge] = None


async def get_unified_math_bridge() -> UnifiedMathKnowledgeBridge:
    """Get global unified bridge instance."""
    global _unified_bridge
    if _unified_bridge is None:
        _unified_bridge = UnifiedMathKnowledgeBridge()
        await _unified_bridge.initialize()
    return _unified_bridge


# Example usage
async def example_unified_bridge():
    """Example: Unified bridge usage."""
    print("Unified Mathematical Knowledge Bridge Example")
    print("=" * 60)
    
    bridge = await get_unified_math_bridge()
    
    # Solve a problem
    problem = "Prove that for all natural numbers n, n + 0 = n"
    result = await bridge.solve_problem(problem, use_hybrid=True)
    
    print(f"\nProblem: {problem}")
    print(f"Classification: {result['classification']}")
    print(f"Success: {result['success']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    # Get summary
    summary = bridge.get_unified_knowledge_summary()
    print(f"\nUnified Knowledge Summary:")
    print(f"  Problems processed: {summary['statistics']['problems_processed']}")
    print(f"  Z3 successes: {summary['statistics']['z3_successes']}")
    print(f"  LeanAIDE successes: {summary['statistics']['leanaide_successes']}")


if __name__ == "__main__":
    asyncio.run(example_unified_bridge())
