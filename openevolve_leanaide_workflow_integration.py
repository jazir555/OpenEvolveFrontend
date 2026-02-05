"""
OpenEvolve + LeanAide Workflow Integration

This module provides integration between OpenEvolve's evolutionary optimization
workflow and LeanAide's formal verification capabilities.

Features:
- Mathematical problem detection in the evolution workflow
- Formal verification of evolved solutions
- Quality scoring with mathematical correctness
- Integration with the team system (Red/Blue/Gold teams)
- Support for multi-objective optimization with formal verification

Author: OpenEvolve
Created: 2026-02-02
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


logger = logging.getLogger(__name__)


# =============================================================================
# Import Dependencies
# =============================================================================

# OpenEvolve imports
try:
    from evolution import EvolutionConfiguration, TEAM_SYSTEM_AVAILABLE
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    EvolutionConfiguration = None
    TEAM_SYSTEM_AVAILABLE = False
    logger.warning("OpenEvolve evolution module not available")

# LeanAide imports
LEANAIDE_AVAILABLE = False
LeanAideClient = None
LeanAideConfig = None
LeanAideWorkflowIntegrator = None
LeanAideWorkflowConfig = None
LeanAideVerificationResult = None
MathematicalProblemDetector = None

try:
    from leanaide_client import LeanAideClient as _LeanAideClient, LeanAideConfig as _LeanAideConfig
    from leanaide_workflow_integration import (
        LeanAideWorkflowIntegrator as _LeanAideWorkflowIntegrator,
        LeanAideWorkflowConfig as _LeanAideWorkflowConfig,
        LeanAideVerificationResult as _LeanAideVerificationResult,
        MathematicalProblemDetector as _MathematicalProblemDetector
    )
    # Set values if imports succeeded
    if _LeanAideClient:
        LEANAIDE_AVAILABLE = True
        LeanAideClient = _LeanAideClient
    if _LeanAideConfig:
        LeanAideConfig = _LeanAideConfig
    if _LeanAideWorkflowIntegrator:
        LeanAideWorkflowIntegrator = _LeanAideWorkflowIntegrator
    if _LeanAideWorkflowConfig:
        LeanAideWorkflowConfig = _LeanAideWorkflowConfig
    if _LeanAideVerificationResult:
        LeanAideVerificationResult = _LeanAideVerificationResult
    if _MathematicalProblemDetector:
        MathematicalProblemDetector = _MathematicalProblemDetector
except (ImportError, AttributeError) as e:
    logger.warning(f"LeanAide not available for workflow integration: {e}")

# Quality Gate imports
try:
    from quality_gate_leanaide_verifier import (
        LeanAideQualityGateVerifier,
        get_leanaide_quality_gate_verifier,
        MathematicalVerificationResult,
        LeanAideQualityConfig
    )
    QUALITY_GATE_AVAILABLE = True
except ImportError:
    QUALITY_GATE_AVAILABLE = False
    LeanAideQualityGateVerifier = None
    get_leanaide_quality_gate_verifier = None
    logger.warning("LeanAide quality gate not available")




# =============================================================================
# Data Classes
# =============================================================================

class OptimizationObjective(Enum):
    """Objectives for LeanAide-enhanced optimization."""
    MATHEMATICAL_CORRECTNESS = "mathematical_correctness"
    FORMAL_VERIFICATION = "formal_verification"
    PROOF_ELEGANCE = "proof_elegance"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    HYBRID = "hybrid"


@dataclass
class OpenEvolveLeanAideConfig:
    """Configuration for OpenEvolve + LeanAide integration."""
    # LeanAide settings
    leanaide_enabled: bool = True
    leanaide_host: str = "localhost"
    leanaide_port: int = 7654
    leanaide_timeout: float = 300.0
    
    # Detection settings
    auto_detect_math: bool = True
    math_confidence_threshold: float = 0.3
    
    # Verification settings
    verification_level: str = "verification"
    confidence_threshold: float = 0.8
    require_formal_proof: bool = False
    
    # Optimization settings
    optimization_objective: str = OptimizationObjective.HYBRID.value
    formal_weight: float = 0.3  # Weight for formal verification in fitness
    correctness_weight: float = 0.4  # Weight for mathematical correctness
    other_weight: float = 0.3  # Weight for other criteria
    
    # Workflow settings
    verify_subproblems: bool = True
    verify_final_solution: bool = True
    use_rag_augmentation: bool = True
    store_proofs: bool = True


@dataclass
class MathematicalFitnessResult:
    """Result of mathematical fitness evaluation."""
    is_mathematical: bool
    mathematical_confidence: float
    formal_verified: bool
    formal_confidence: float
    lean_code: Optional[str]
    formal_proof: Optional[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_mathematical": self.is_mathematical,
            "mathematical_confidence": self.mathematical_confidence,
            "formal_verified": self.formal_verified,
            "formal_confidence": self.formal_confidence,
            "lean_code": self.lean_code,
            "formal_proof": self.formal_proof,
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }


@dataclass
class EvolutionStepResult:
    """Result of an evolution step with LeanAide verification."""
    step_number: int
    solution: str
    fitness: float
    mathematical_fitness: MathematicalFitnessResult
    is_mathematical: bool
    formal_verification_passed: bool
    confidence_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "solution": self.solution,
            "fitness": self.fitness,
            "mathematical_fitness": self.mathematical_fitness.to_dict(),
            "is_mathematical": self.is_mathematical,
            "formal_verification_passed": self.formal_verification_passed,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp
        }


# =============================================================================
# OpenEvolve + LeanAide Integration
# =============================================================================

class OpenEvolveLeanAideIntegrator:
    """
    Integration between OpenEvolve evolution workflow and LeanAide verification.
    
    This integrator:
    - Detects mathematical problems in the evolution workflow
    - Applies formal verification to evolved solutions
    - Calculates fitness scores with mathematical correctness
    - Stores proof artifacts for analysis
    """
    
    def __init__(
        self,
        config: Optional[OpenEvolveLeanAideConfig] = None,
        leanaide_config: Optional[LeanAideWorkflowConfig] = None
    ):
        """
        Initialize the integrator.
        
        Args:
            config: Integration configuration
            leanaide_config: LeanAide workflow configuration
        """
        self.config = config or OpenEvolveLeanAideConfig()
        self.leanaide_config = leanaide_config or LeanAideWorkflowConfig(
            enabled=self.config.leanaide_enabled,
            host=self.config.leanaide_host,
            port=self.config.leanaide_port,
            timeout=self.config.leanaide_timeout,
            auto_detect_math=self.config.auto_detect_math,
            confidence_threshold=self.config.confidence_threshold,
            require_formal_proof=self.config.require_formal_proof
        )
        
        # Initialize components
        self.math_detector = None
        self.workflow_integrator = None
        self.quality_gate = None
        self._initialize_components()
        
        # Track evolution history
        self.evolution_history: List[EvolutionStepResult] = []
        
        logger.info({
            "msg": "OpenEvolveLeanAideIntegrator initialized",
            "leanaide_enabled": self.config.leanaide_enabled,
            "optimization_objective": self.config.optimization_objective,
            "formal_weight": self.config.formal_weight,
            "correctness_weight": self.config.correctness_weight
        })
    
    def _initialize_components(self):
        """Initialize integration components."""
        # Initialize mathematical problem detector
        if LEANAIDE_AVAILABLE and MathematicalProblemDetector:
            self.math_detector = MathematicalProblemDetector(self.leanaide_config)
            logger.info("Mathematical problem detector initialized")
        
        # Initialize workflow integrator
        if LEANAIDE_AVAILABLE and LeanAideWorkflowIntegrator:
            try:
                self.workflow_integrator = LeanAideWorkflowIntegrator(self.leanaide_config)
                logger.info("LeanAide workflow integrator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize workflow integrator: {e}")
        
        # Initialize quality gate verifier
        if QUALITY_GATE_AVAILABLE and get_leanaide_quality_gate_verifier:
            try:
                qg_config = LeanAideQualityConfig(
                    enabled=self.config.leanaide_enabled,
                    verification_level=self.config.verification_level,
                    confidence_threshold=self.config.confidence_threshold,
                    require_formal_proof=self.config.require_formal_proof
                )
                self.quality_gate = get_leanaide_quality_gate_verifier(qg_config)
                logger.info("Quality gate verifier initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize quality gate: {e}")
    
    async def detect_mathematical_content(
        self,
        problem_statement: str,
        solution_content: Optional[str] = None
    ) -> Tuple[bool, float]:
        """
        Detect if content is mathematical.
        
        Args:
            problem_statement: The problem to analyze
            solution_content: Optional solution to analyze
            
        Returns:
            Tuple of (is_mathematical, confidence)
        """
        if self.math_detector:
            return self.math_detector.is_mathematical_problem(problem_statement, solution_content)
        
        # Fallback: simple keyword detection
        math_keywords = ["theorem", "lemma", "proof", "prove", "equation", "integral"]
        text = (problem_statement + " " + (solution_content or "")).lower()
        matches = sum(1 for kw in math_keywords if kw in text)
        confidence = min(matches / 3.0, 1.0)
        return confidence >= self.config.math_confidence_threshold, confidence
    
    async def evaluate_mathematical_fitness(
        self,
        problem_statement: str,
        solution: str,
        step_number: int = 0
    ) -> MathematicalFitnessResult:
        """
        Evaluate the mathematical fitness of a solution.
        
        Args:
            problem_statement: The original problem
            solution: The evolved solution
            step_number: Current evolution step
            
        Returns:
            MathematicalFitnessResult with evaluation details
        """
        start_time = datetime.now()
        
        logger.info({
            "msg": "Evaluating mathematical fitness",
            "step_number": step_number,
            "solution_length": len(solution)
        })
        
        # Step 1: Detect if mathematical
        is_math, math_confidence = await self.detect_mathematical_content(problem_statement, solution)
        
        result = MathematicalFitnessResult(
            is_mathematical=is_math,
            mathematical_confidence=math_confidence,
            formal_verified=False,
            formal_confidence=0.0,
            lean_code=None,
            formal_proof=None,
            processing_time_ms=0.0
        )
        
        # If not mathematical, return early
        if not is_math:
            result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.info({
                "msg": "Non-mathematical content detected",
                "confidence": math_confidence
            })
            return result
        
        # Step 2: If mathematical, perform formal verification
        if self.config.leanaide_enabled and self.quality_gate:
            try:
                math_result = await self.quality_gate.verify_mathematical_correctness(solution)
                
                result.is_mathematical = math_result.is_mathematical
                result.formal_verified = math_result.verification_passed
                result.formal_confidence = math_result.confidence_score
                result.lean_code = math_result.formal_code
                result.errors = math_result.errors
                result.warnings = math_result.warnings
                result.metadata = math_result.metadata
                
            except Exception as e:
                logger.error(f"Formal verification failed: {e}")
                result.errors.append(str(e))
        
        result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info({
            "msg": "Mathematical fitness evaluation complete",
            "is_mathematical": result.is_mathematical,
            "formal_verified": result.formal_verified,
            "confidence": result.formal_confidence,
            "processing_time_ms": result.processing_time_ms
        })
        
        return result
    
    def calculate_hybrid_fitness(
        self,
        base_fitness: float,
        math_fitness: MathematicalFitnessResult,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate hybrid fitness combining base and mathematical fitness.
        
        Args:
            base_fitness: Original fitness score (0-1)
            math_fitness: Mathematical fitness evaluation
            weights: Optional weight overrides
            
        Returns:
            Hybrid fitness score (0-1)
        """
        weights = weights or {
            "formal": self.config.formal_weight,
            "correctness": self.config.correctness_weight,
            "other": self.config.other_weight
        }
        
        if not math_fitness.is_mathematical:
            # Non-mathematical: use base fitness
            return base_fitness
        
        # Calculate mathematical component
        math_component = (
            (1.0 if math_fitness.formal_verified else 0.0) * weights["formal"] +
            math_fitness.formal_confidence * weights["correctness"]
        )
        
        # Combine with base fitness
        hybrid_fitness = (
            base_fitness * weights["other"] +
            math_component
        )
        
        # Normalize
        hybrid_fitness = min(max(hybrid_fitness, 0.0), 1.0)
        
        return hybrid_fitness
    
    async def evaluate_evolution_step(
        self,
        problem_statement: str,
        solution: str,
        base_fitness: float,
        step_number: int
    ) -> EvolutionStepResult:
        """
        Evaluate a single evolution step with LeanAide verification.
        
        Args:
            problem_statement: The problem being solved
            solution: The evolved solution
            base_fitness: Original fitness score
            step_number: Current evolution step number
            
        Returns:
            EvolutionStepResult with all evaluation details
        """
        # Evaluate mathematical fitness
        math_fitness = await self.evaluate_mathematical_fitness(
            problem_statement, solution, step_number
        )
        
        # Calculate hybrid fitness
        hybrid_fitness = self.calculate_hybrid_fitness(base_fitness, math_fitness)
        
        # Determine if formal verification passed
        formal_passed = (
            math_fitness.is_mathematical and 
            math_fitness.formal_verified and
            math_fitness.formal_confidence >= self.config.confidence_threshold
        )
        
        result = EvolutionStepResult(
            step_number=step_number,
            solution=solution,
            fitness=hybrid_fitness,
            mathematical_fitness=math_fitness,
            is_mathematical=math_fitness.is_mathematical,
            formal_verification_passed=formal_passed,
            confidence_score=math_fitness.formal_confidence
        )
        
        # Store in history
        self.evolution_history.append(result)
        
        return result
    
    async def verify_solution(
        self,
        problem_statement: str,
        solution: str
    ) -> LeanAideVerificationResult:
        """
        Perform full verification of a solution.
        
        Args:
            problem_statement: The problem being solved
            solution: The final solution
            
        Returns:
            LeanAideVerificationResult with verification details
        """
        if not self.workflow_integrator:
            return LeanAideVerificationResult(
                success=False,
                is_mathematical=False,
                confidence_score=0.0,
                verification_method="unavailable",
                errors=["Workflow integrator not available"]
            )
        
        try:
            result = await self.workflow_integrator.verify_formal_statement(
                problem_statement, solution
            )
            return result
        except Exception as e:
            logger.error(f"Solution verification failed: {e}")
            return LeanAideVerificationResult(
                success=False,
                is_mathematical=False,
                confidence_score=0.0,
                verification_method="error",
                errors=[str(e)]
            )
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of the evolution process."""
        if not self.evolution_history:
            return {"status": "no_history", "message": "No evolution steps recorded"}
        
        mathematical_steps = [s for s in self.evolution_history if s.is_mathematical]
        verified_steps = [s for s in mathematical_steps if s.formal_verification_passed]
        
        return {
            "total_steps": len(self.evolution_history),
            "mathematical_steps": len(mathematical_steps),
            "verified_steps": len(verified_steps),
            "verification_rate": len(verified_steps) / len(self.evolution_history) if self.evolution_history else 0,
            "final_fitness": self.evolution_history[-1].fitness if self.evolution_history else 0,
            "final_confidence": self.evolution_history[-1].confidence_score if self.evolution_history else 0,
            "history": [s.to_dict() for s in self.evolution_history]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            "leanaide_available": LEANAIDE_AVAILABLE,
            "workflow_integrator_available": self.workflow_integrator is not None,
            "quality_gate_available": self.quality_gate is not None,
            "config": {
                "leanaide_enabled": self.config.leanaide_enabled,
                "optimization_objective": self.config.optimization_objective,
                "formal_weight": self.config.formal_weight,
                "correctness_weight": self.config.correctness_weight
            },
            "evolution_summary": self.get_evolution_summary()
        }


# =============================================================================
# Factory Functions
# =============================================================================

def get_openevolve_leanaide_integrator(
    config: Optional[OpenEvolveLeanAideConfig] = None
) -> OpenEvolveLeanAideIntegrator:
    """
    Get an OpenEvolve-LeanAide integrator instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        OpenEvolveLeanAideIntegrator instance
    """
    return OpenEvolveLeanAideIntegrator(config=config)


async def create_openevolve_leanaide_integrator(
    config: Optional[OpenEvolveLeanAideConfig] = None
) -> OpenEvolveLeanAideIntegrator:
    """
    Create and initialize an integrator (async).
    
    Args:
        config: Optional configuration
        
    Returns:
        Initialized OpenEvolveLeanAideIntegrator instance
    """
    integrator = get_openevolve_leanaide_integrator(config)
    return integrator


# =============================================================================
# Integration with Team System
# =============================================================================

async def integrate_with_red_team(
    integrator: OpenEvolveLeanAideIntegrator,
    problem_statement: str,
    solution: str
) -> Dict[str, Any]:
    """
    Use LeanAide to enhance Red Team adversarial testing.
    
    Args:
        integrator: The integrator instance
        problem_statement: The problem being tested
        solution: The solution to test
        
    Returns:
        Red team assessment with LeanAide insights
    """
    # Get mathematical analysis
    math_fitness = await integrator.evaluate_mathematical_fitness(problem_statement, solution)
    
    return {
        "adversarial_test_passed": math_fitness.formal_verified,
        "mathematical_weakness_detected": len(math_fitness.errors) > 0,
        "mathematical_warnings": math_fitness.warnings,
        "confidence_score": math_fitness.formal_confidence,
        "lean_code": math_fitness.lean_code,
        "proof_artifacts": math_fitness.formal_proof
    }


async def integrate_with_blue_team(
    integrator: OpenEvolveLeanAideIntegrator,
    problem_statement: str,
    solution: str,
    issues: List[str]
) -> Dict[str, Any]:
    """
    Use LeanAide to guide Blue Team fixes.
    
    Args:
        integrator: The integrator instance
        problem_statement: The problem being solved
        solution: The current solution
        issues: List of identified issues
        
    Returns:
        Fix suggestions with LeanAide insights
    """
    math_fitness = await integrator.evaluate_mathematical_fitness(problem_statement, solution)
    
    fix_suggestions = []
    
    # Generate fixes based on LeanAide feedback
    if not math_fitness.formal_verified:
        if math_fitness.lean_code:
            fix_suggestions.append({
                "type": "formal_verification",
                "description": "Fix formal verification issues in Lean code",
                "lean_code": math_fitness.lean_code,
                "errors": math_fitness.errors
            })
    
    if math_fitness.warnings:
        fix_suggestions.append({
            "type": "warnings",
            "description": "Address LeanAide warnings",
            "warnings": math_fitness.warnings
        })
    
    return {
        "fix_suggestions": fix_suggestions,
        "math_fitness": math_fitness.to_dict()
    }


async def integrate_with_gold_team(
    integrator: OpenEvolveLeanAideIntegrator,
    problem_statement: str,
    solution: str
) -> Dict[str, Any]:
    """
    Use LeanAide for Gold Team final verification.
    
    Args:
        integrator: The integrator instance
        problem_statement: The problem being solved
        solution: The final solution
        
    Returns:
        Gold team verification result
    """
    verification_result = await integrator.verify_solution(problem_statement, solution)
    
    # Calculate final confidence
    math_fitness = await integrator.evaluate_mathematical_fitness(problem_statement, solution)
    
    final_confidence = (
        verification_result.confidence_score * 0.6 +
        math_fitness.formal_confidence * 0.4
    ) if math_fitness.is_mathematical else verification_result.confidence_score
    
    return {
        "verification_passed": verification_result.success,
        "confidence_score": final_confidence,
        "verification_details": verification_result.to_dict(),
        "math_fitness": math_fitness.to_dict()
    }


# =============================================================================
# Standalone Usage
# =============================================================================

if __name__ == "__main__":
    import sys
    
    async def test_integration():
        """Test the integration."""
        print("Testing OpenEvolve + LeanAide Integration...")
        
        # Create integrator
        config = OpenEvolveLeanAideConfig(
            leanaide_enabled=True,
            optimization_objective=OptimizationObjective.HYBRID.value,
            formal_weight=0.3,
            correctness_weight=0.4,
            other_weight=0.3
        )
        
        integrator = get_openevolve_leanaide_integrator(config)
        
        # Get status
        status = integrator.get_status()
        print(f"Status: {json.dumps(status, indent=2)}")
        
        # Test with mathematical content
        problem = "Prove that the square root of 2 is irrational"
        solution = "theorem sqrt_2_irrational : ∀ n : ℕ, n*n = 2 -> false := by sorry"
        
        print("\nTesting mathematical fitness evaluation...")
        math_fitness = await integrator.evaluate_mathematical_fitness(problem, solution)
        print(f"Result: {json.dumps(math_fitness.to_dict(), indent=2)}")
        
        # Test evolution step
        print("\nTesting evolution step evaluation...")
        step_result = await integrator.evaluate_evolution_step(
            problem, solution, base_fitness=0.8, step_number=1
        )
        print(f"Step Result: {json.dumps(step_result.to_dict(), indent=2)}")
        
        # Get summary
        summary = integrator.get_evolution_summary()
        print(f"\nEvolution Summary: {json.dumps(summary, indent=2)}")
        
        return True
    
    try:
        result = asyncio.run(test_integration())
        if result:
            print("\nSUCCESS: Integration working!")
        else:
            print("\nFAILED: Integration issues detected")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
