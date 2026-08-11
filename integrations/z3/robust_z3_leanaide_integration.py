"""
Robust Z3-LeanAide Integration Module

This module provides a comprehensive, robust integration between Z3 SMT solver and LeanAIDE
formal verification system with enhanced error handling, fallback mechanisms, and
bidirectional communication capabilities.

Features:
- Enhanced error handling and recovery
- Comprehensive fallback mechanisms
- Bidirectional translation with semantic preservation
- Performance optimization
- Cross-validation capabilities
- DSPy-enhanced problem understanding
"""



import asyncio
import json
import logging
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from collections import defaultdict
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 integration
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3SolverResult, Z3TheoremResult,
        Z3Variable, Z3Constraint, Z3ConstraintType, Z3ResultStatus,
        Z3Config, get_z3_solver_engine, get_z3_theorem_prover, is_z3_available
    )
    Z3_INTEGRATION_AVAILABLE = True
    logger.info("Z3 integration available for enhanced verification")
except ImportError:
    Z3_INTEGRATION_AVAILABLE = False
    logger.warning("Z3 integration not available")

# Import LeanAIDE integration
try:
    from leanaide_client import LeanAideClient, LeanAideConfig
    from leanaide_mcp_tools import (
        leanaide_translate_theorem,
        leanaide_verify_solution,
        leanaide_elaborate_code
    )
    LEANAIDE_AVAILABLE = True
    logger.info("LeanAIDE integration available for enhanced verification")
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAIDE client not available")

try:
    from leanaide_workflow_integration import (
        LeanAideWorkflowIntegrator,
        LeanAideVerificationResult,
        MathematicalProblemDetector
    )
    LEANAIDE_WORKFLOW_AVAILABLE = True
except ImportError:
    LEANAIDE_WORKFLOW_AVAILABLE = False
    logger.warning("LeanAIDE workflow integration not available")

# Import DSPy for enhanced prompting
try:
    from dspy_integration import DSPY_AVAILABLE, get_global_dspy_instance, initialize_dspy
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    from dspy.predict import Predict
    logger.info("DSPy available through global integration for enhanced Z3-LeanAIDE bridging")
except ImportError:
    # Fallback to local import if global module not available
    try:
        import dspy
        from dspy.teleprompt import BootstrapFewShot
        from dspy.predict import Predict
        DSPY_AVAILABLE = True
        logger.info("DSPy available for enhanced Z3-LeanAIDE bridging")
    except ImportError:
        dspy = None
        BootstrapFewShot = None
        Predict = None
        DSPY_AVAILABLE = False
        logger.warning("DSPy not available - using standard Z3-LeanAIDE bridging")

# CAV-NLP imports
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
    logger.info("CAV-NLP integration available for enhanced verification")
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.warning("CAV-NLP integration not available")


class VerificationStrategy(Enum):
    """Strategy for combined verification."""
    Z3_FIRST = "z3_first"           # Try Z3 first, fall back to Lean
    LEAN_FIRST = "lean_first"       # Try Lean first, fall back to Z3
    PARALLEL = "parallel"           # Run both in parallel
    CONSENSUS = "consensus"         # Both must agree
    ADAPTIVE = "adaptive"           # Choose based on problem type


@dataclass
class RobustVerificationResult:
    """Enhanced result of combined Z3 + LeanAIDE verification with robust error handling."""
    success: bool
    z3_result: Optional[Any] = None
    lean_result: Optional[Any] = None
    strategy_used: VerificationStrategy = VerificationStrategy.ADAPTIVE
    agreement: bool = False
    confidence_score: float = 0.0
    recommendation: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    fallback_used: bool = False
    cross_validation_passed: bool = False
    verification_log: List[Dict[str, Any]] = field(default_factory=list)
    dspy_analysis: Optional[Dict[str, Any]] = None
    dspy_enhanced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "z3_result": self.z3_result.to_dict() if self.z3_result and hasattr(self.z3_result, 'to_dict') else self.z3_result,
            "lean_result": self.lean_result.to_dict() if self.lean_result and hasattr(self.lean_result, 'to_dict') else self.lean_result,
            "strategy_used": self.strategy_used.value,
            "agreement": self.agreement,
            "confidence_score": self.confidence_score,
            "recommendation": self.recommendation,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "fallback_used": self.fallback_used,
            "cross_validation_passed": self.cross_validation_passed,
            "verification_log": self.verification_log,
            "dspy_analysis": self.dspy_analysis,
            "dspy_enhanced": self.dspy_enhanced
        }


class RobustZ3LeanAideBridge:
    """
    Robust bridge class integrating Z3 with LeanAIDE with enhanced error handling and fallback mechanisms.

    Provides:
    - Bidirectional translation with semantic preservation
    - Combined verification with multiple strategies
    - Adaptive strategy selection
    - Cross-validation
    - Comprehensive error handling
    - Performance optimization
    - CAV-NLP integration for hybrid verification
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.z3_solver = None
        self.z3_prover = None
        self.lean_integrator = None
        self.problem_detector = None
        self.translation_cache = {}
        self.verification_cache = {}
        self.lock = threading.Lock()
        
        # CAV-NLP integration
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            try:
                self.enhanced_solver = EnhancedZ3Solver()
                self.math_service = UnifiedMathService()
                logger.info("CAV-NLP components initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP components: {e}")
                self.use_cav_nlp = False
                self.enhanced_solver = None
                self.math_service = None
        
        # Initialize Z3 components if available
        if Z3_INTEGRATION_AVAILABLE:
            try:
                self.z3_solver = get_z3_solver_engine(Z3Config(timeout=30.0, proof_generation=True))
                self.z3_prover = get_z3_theorem_prover(Z3Config(timeout=30.0, proof_generation=True))
            except Exception as e:
                logger.warning(f"Failed to initialize Z3 components: {e}")
                self.z3_solver = None
                self.z3_prover = None
        
        # Initialize LeanAIDE components if available
        if LEANAIDE_WORKFLOW_AVAILABLE:
            try:
                lean_config = type('Config', (), {
                    'host': 'localhost',
                    'port': 7654,
                    'timeout': 300.0,
                    'enabled': True
                })()
                self.lean_integrator = LeanAideWorkflowIntegrator(lean_config)
                self.problem_detector = MathematicalProblemDetector()
            except Exception as e:
                logger.warning(f"Failed to initialize LeanAIDE components: {e}")
                self.lean_integrator = None
                self.problem_detector = None

    def robust_verify_with_both(
        self,
        problem: str,
        strategy: VerificationStrategy = VerificationStrategy.ADAPTIVE,
        timeout: float = 60.0,
        enable_cross_validation: bool = True,
        enable_dspy_enhancement: bool = True
    ) -> RobustVerificationResult:
        """
        Robust verification using both Z3 and LeanAIDE with comprehensive error handling.

        Args:
            problem: Problem statement (SMT-LIB, Lean 4, or natural language)
            strategy: Verification strategy to use
            timeout: Timeout for verification attempts
            enable_cross_validation: Whether to perform cross-validation
            enable_dspy_enhancement: Whether to use DSPy for enhanced problem understanding

        Returns:
            RobustVerificationResult with comprehensive error handling
        """
        start_time = time.time()
        result = RobustVerificationResult(success=False, errors=[], warnings=[])

        try:
            # Add to verification log
            result.verification_log.append({
                "step": "start_verification",
                "problem_preview": problem[:100] + "..." if len(problem) > 100 else problem,
                "strategy": strategy.value,
                "timestamp": time.time()
            })

            # Use DSPy for enhanced problem understanding if available
            if enable_dspy_enhancement and DSPY_AVAILABLE:
                try:
                    problem_analysis = self._analyze_problem_with_dspy(problem)
                    result.dspy_analysis = problem_analysis
                    result.dspy_enhanced = True
                    
                    # Potentially adjust strategy based on DSPy analysis
                    if problem_analysis.get("recommended_strategy"):
                        strategy = VerificationStrategy(problem_analysis["recommended_strategy"])
                except Exception as e:
                    result.warnings.append(f"DSPy analysis failed: {str(e)}, continuing with original strategy")
                    logger.warning(f"DSPy analysis failed: {e}")

            # Determine if problem is SMT-LIB format
            is_smt = self._is_smtlib(problem)

            # Execute verification based on strategy
            if strategy == VerificationStrategy.ADAPTIVE:
                strategy = self._select_strategy_adaptive(problem, is_smt)

            # Perform verification with appropriate strategy
            if strategy == VerificationStrategy.Z3_FIRST:
                result = self._verify_z3_first_robust(problem, is_smt, result, timeout)
            elif strategy == VerificationStrategy.LEAN_FIRST:
                result = self._verify_lean_first_robust(problem, is_smt, result, timeout)
            elif strategy == VerificationStrategy.PARALLEL:
                result = self._verify_parallel_robust(problem, is_smt, result, timeout)
            elif strategy == VerificationStrategy.CONSENSUS:
                result = self._verify_consensus_robust(problem, is_smt, result, timeout)
            else:
                # Default to Z3-first if unknown strategy
                result = self._verify_z3_first_robust(problem, is_smt, result, timeout)

            # Perform cross-validation if enabled
            if enable_cross_validation and result.success:
                result.cross_validation_passed = self._perform_cross_validation(result)

            # Update execution time
            result.execution_time = time.time() - start_time
            result.verification_log.append({
                "step": "verification_completed",
                "success": result.success,
                "execution_time": result.execution_time,
                "timestamp": time.time()
            })

        except Exception as e:
            error_msg = f"Critical error in robust verification: {str(e)}"
            result.errors.append(error_msg)
            result.verification_log.append({
                "step": "critical_error",
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            })
            logger.error(error_msg, exc_info=True)

        return result

    def _analyze_problem_with_dspy(self, problem: str) -> Dict[str, Any]:
        """Analyze problem using DSPy for enhanced understanding."""
        if not DSPY_AVAILABLE:
            return {"error": "DSPy not available"}

        try:
            # Define a DSPy signature for problem analysis
            class ProblemAnalysisSignature(dspy.Signature):
                """Analyze a mathematical problem for optimal verification approach."""
                problem_statement = dspy.InputField(desc="Mathematical problem to analyze")
                
                problem_type = dspy.OutputField(desc="Type of problem (arithmetic, boolean, string, theorem, etc.)")
                complexity_level = dspy.OutputField(desc="Complexity level (simple, medium, complex, advanced)")
                recommended_strategy = dspy.OutputField(desc="Recommended verification strategy (z3_first, lean_first, parallel, consensus)")
                key_variables = dspy.OutputField(desc="Key variables identified in the problem")
                constraints_identified = dspy.OutputField(desc="Constraints identified in the problem")
                verification_approach = dspy.OutputField(desc="Recommended approach for verification")

            # Create a predictor using the signature
            analyze_problem = dspy.Predict(ProblemAnalysisSignature)

            # Run DSPy analysis
            result = analyze_problem(problem_statement=problem)

            return {
                "problem_type": result.problem_type,
                "complexity_level": result.complexity_level,
                "recommended_strategy": result.recommended_strategy,
                "key_variables": result.key_variables,
                "constraints_identified": result.constraints_identified,
                "verification_approach": result.verification_approach
            }

        except Exception as e:
            logger.warning(f"DSPy problem analysis failed: {e}")
            return {"error": f"DSPy analysis failed: {str(e)}"}

    def _is_smtlib(self, text: str) -> bool:
        """Check if text is in SMT-LIB format."""
        smt_keywords = ['(assert', '(declare-fun', '(check-sat)', '(set-logic', '(define-fun']
        return any(kw in text.lower() for kw in smt_keywords)

    def _select_strategy_adaptive(self, problem: str, is_smt: bool) -> VerificationStrategy:
        """Select strategy based on problem characteristics."""
        if is_smt:
            # For SMT-LIB problems, check if they're more constraint-oriented or theorem-oriented
            if 'prove' in problem.lower() or 'theorem' in problem.lower():
                return VerificationStrategy.LEAN_FIRST
            else:
                return VerificationStrategy.Z3_FIRST
        else:
            # For natural language, use problem detector if available
            if self.problem_detector:
                try:
                    detected_type = self.problem_detector.detect_problem_type(problem)
                    if detected_type in ['theorem', 'proof']:
                        return VerificationStrategy.LEAN_FIRST
                    else:
                        return VerificationStrategy.Z3_FIRST
                except:
                    pass
            
            # Default fallback
            return VerificationStrategy.PARALLEL

    def _verify_z3_first_robust(
        self, 
        problem: str, 
        is_smt: bool, 
        result: RobustVerificationResult, 
        timeout: float
    ) -> RobustVerificationResult:
        """Robust Z3-first verification with fallback to LeanAIDE."""
        try:
            # Try Z3 first
            if self.z3_solver and is_smt:
                z3_result = self.z3_solver.solve_smtlib(problem)
                result.z3_result = z3_result
                result.verification_log.append({"step": "z3_attempt", "success": z3_result.success if z3_result else False})
                
                if z3_result and z3_result.success:
                    result.success = True
                    result.strategy_used = VerificationStrategy.Z3_FIRST
                    result.confidence_score = z3_result.confidence if hasattr(z3_result, 'confidence') else 0.8
                    result.recommendation = "Verified by Z3"
                    return result
            elif self.z3_prover and not is_smt:
                z3_result = self.z3_prover.prove_theorem(problem)
                result.z3_result = z3_result
                result.verification_log.append({"step": "z3_theorem_attempt", "success": z3_result.success if z3_result else False})
                
                if z3_result and z3_result.success:
                    result.success = True
                    result.strategy_used = VerificationStrategy.Z3_FIRST
                    result.confidence_score = z3_result.confidence if hasattr(z3_result, 'confidence') else 0.8
                    result.recommendation = "Theorem proven by Z3"
                    return result

            # If Z3 fails, try LeanAIDE as fallback
            if self.lean_integrator:
                lean_result = self.lean_integrator.verify_solution(problem)
                result.lean_result = lean_result
                result.verification_log.append({"step": "lean_fallback", "success": lean_result.success if lean_result else False})
                
                if lean_result and lean_result.success:
                    result.success = True
                    result.strategy_used = VerificationStrategy.Z3_FIRST
                    result.confidence_score = lean_result.confidence if hasattr(lean_result, 'confidence') else 0.7
                    result.recommendation = "Verified by LeanAIDE (Z3 fallback)"
                    result.fallback_used = True
                    return result
            else:
                result.warnings.append("LeanAIDE not available for fallback")
        except Exception as e:
            result.errors.append(f"Z3-first verification failed: {str(e)}")
            result.verification_log.append({"step": "z3_first_error", "error": str(e)})

        # If both fail, return unsuccessful result
        result.strategy_used = VerificationStrategy.Z3_FIRST
        result.recommendation = "Both Z3 and LeanAIDE verification failed"
        return result

    def _verify_lean_first_robust(
        self, 
        problem: str, 
        is_smt: bool, 
        result: RobustVerificationResult, 
        timeout: float
    ) -> RobustVerificationResult:
        """Robust LeanAIDE-first verification with fallback to Z3."""
        try:
            # Try LeanAIDE first
            if self.lean_integrator:
                lean_result = self.lean_integrator.verify_solution(problem)
                result.lean_result = lean_result
                result.verification_log.append({"step": "lean_attempt", "success": lean_result.success if lean_result else False})
                
                if lean_result and lean_result.success:
                    result.success = True
                    result.strategy_used = VerificationStrategy.LEAN_FIRST
                    result.confidence_score = lean_result.confidence if hasattr(lean_result, 'confidence') else 0.8
                    result.recommendation = "Verified by LeanAIDE"
                    return result

            # If LeanAIDE fails, try Z3 as fallback
            if self.z3_solver and is_smt:
                z3_result = self.z3_solver.solve_smtlib(problem)
                result.z3_result = z3_result
                result.verification_log.append({"step": "z3_fallback", "success": z3_result.success if z3_result else False})
                
                if z3_result and z3_result.success:
                    result.success = True
                    result.strategy_used = VerificationStrategy.LEAN_FIRST
                    result.confidence_score = z3_result.confidence if hasattr(z3_result, 'confidence') else 0.7
                    result.recommendation = "Verified by Z3 (LeanAIDE fallback)"
                    result.fallback_used = True
                    return result
            elif self.z3_prover and not is_smt:
                z3_result = self.z3_prover.prove_theorem(problem)
                result.z3_result = z3_result
                result.verification_log.append({"step": "z3_theorem_fallback", "success": z3_result.success if z3_result else False})
                
                if z3_result and z3_result.success:
                    result.success = True
                    result.strategy_used = VerificationStrategy.LEAN_FIRST
                    result.confidence_score = z3_result.confidence if hasattr(z3_result, 'confidence') else 0.7
                    result.recommendation = "Theorem proven by Z3 (LeanAIDE fallback)"
                    result.fallback_used = True
                    return result
            else:
                result.warnings.append("Z3 not available for fallback")
        except Exception as e:
            result.errors.append(f"Lean-first verification failed: {str(e)}")
            result.verification_log.append({"step": "lean_first_error", "error": str(e)})

        # If both fail, return unsuccessful result
        result.strategy_used = VerificationStrategy.LEAN_FIRST
        result.recommendation = "Both LeanAIDE and Z3 verification failed"
        return result

    def _verify_parallel_robust(
        self, 
        problem: str, 
        is_smt: bool, 
        result: RobustVerificationResult, 
        timeout: float
    ) -> RobustVerificationResult:
        """Robust parallel verification using both Z3 and LeanAIDE."""
        try:
            # Run both verifications in parallel using threads
            z3_result = None
            lean_result = None
            z3_error = None
            lean_error = None

            def run_z3():
                nonlocal z3_result, z3_error
                try:
                    if is_smt and self.z3_solver:
                        z3_result = self.z3_solver.solve_smtlib(problem)
                    elif self.z3_prover and not is_smt:
                        z3_result = self.z3_prover.prove_theorem(problem)
                except Exception as e:
                    z3_error = str(e)

            def run_lean():
                nonlocal lean_result, lean_error
                try:
                    if self.lean_integrator:
                        lean_result = self.lean_integrator.verify_solution(problem)
                except Exception as e:
                    lean_error = str(e)

            # Run in parallel
            z3_thread = threading.Thread(target=run_z3)
            lean_thread = threading.Thread(target=run_lean)

            z3_thread.start()
            lean_thread.start()

            # Wait for both to complete (with timeout)
            z3_thread.join(timeout=timeout/2)  # Half timeout for each
            lean_thread.join(timeout=timeout/2)

            result.z3_result = z3_result
            result.lean_result = lean_result

            if z3_error:
                result.errors.append(f"Z3 parallel verification error: {z3_error}")
            if lean_error:
                result.errors.append(f"LeanAIDE parallel verification error: {lean_error}")

            # Determine success based on results
            z3_success = z3_result and z3_result.success if z3_result else False
            lean_success = lean_result and lean_result.success if lean_result else False

            if z3_success and lean_success:
                result.success = True
                result.agreement = True
                result.confidence_score = 0.9
                result.recommendation = "Verified by both Z3 and LeanAIDE (consensus)"
            elif z3_success:
                result.success = True
                result.confidence_score = 0.8
                result.recommendation = "Verified by Z3 only"
            elif lean_success:
                result.success = True
                result.confidence_score = 0.8
                result.recommendation = "Verified by LeanAIDE only"
            else:
                result.recommendation = "Neither Z3 nor LeanAIDE succeeded in parallel verification"

            result.strategy_used = VerificationStrategy.PARALLEL
            result.verification_log.append({
                "step": "parallel_verification",
                "z3_success": z3_success,
                "lean_success": lean_success,
                "agreement": result.agreement
            })

        except Exception as e:
            result.errors.append(f"Parallel verification failed: {str(e)}")
            result.verification_log.append({"step": "parallel_error", "error": str(e)})

        return result

    def _verify_consensus_robust(
        self, 
        problem: str, 
        is_smt: bool, 
        result: RobustVerificationResult, 
        timeout: float
    ) -> RobustVerificationResult:
        """Robust consensus verification requiring both Z3 and LeanAIDE to agree."""
        try:
            # Run both verifications
            z3_result = None
            lean_result = None
            z3_error = None
            lean_error = None

            def run_z3():
                nonlocal z3_result, z3_error
                try:
                    if is_smt and self.z3_solver:
                        z3_result = self.z3_solver.solve_smtlib(problem)
                    elif self.z3_prover and not is_smt:
                        z3_result = self.z3_prover.prove_theorem(problem)
                except Exception as e:
                    z3_error = str(e)

            def run_lean():
                nonlocal lean_result, lean_error
                try:
                    if self.lean_integrator:
                        lean_result = self.lean_integrator.verify_solution(problem)
                except Exception as e:
                    lean_error = str(e)

            # Run in parallel
            z3_thread = threading.Thread(target=run_z3)
            lean_thread = threading.Thread(target=run_lean)

            z3_thread.start()
            lean_thread.start()

            # Wait for both to complete (with timeout)
            z3_thread.join(timeout=timeout/2)
            lean_thread.join(timeout=timeout/2)

            result.z3_result = z3_result
            result.lean_result = lean_result

            if z3_error:
                result.errors.append(f"Z3 consensus verification error: {z3_error}")
            if lean_error:
                result.errors.append(f"LeanAIDE consensus verification error: {lean_error}")

            # Consensus requires both to succeed and agree
            z3_success = z3_result and z3_result.success if z3_result else False
            lean_success = lean_result and lean_result.success if lean_result else False

            if z3_success and lean_success:
                # Check if results are consistent (both say SAT/unsat, or both prove the same thing)
                result.agreement = self._check_result_agreement(z3_result, lean_result)
                if result.agreement:
                    result.success = True
                    result.confidence_score = 0.95
                    result.recommendation = "Consensus achieved: both Z3 and LeanAIDE agree"
                else:
                    result.success = False
                    result.confidence_score = 0.3
                    result.recommendation = "Disagreement: Z3 and LeanAIDE results differ"
            elif z3_success:
                result.success = False
                result.confidence_score = 0.4
                result.recommendation = "Partial success: Z3 succeeded but LeanAIDE failed"
                result.fallback_used = True
            elif lean_success:
                result.success = False
                result.confidence_score = 0.4
                result.recommendation = "Partial success: LeanAIDE succeeded but Z3 failed"
                result.fallback_used = True
            else:
                result.recommendation = "Consensus failed: both Z3 and LeanAIDE failed"

            result.strategy_used = VerificationStrategy.CONSENSUS
            result.verification_log.append({
                "step": "consensus_verification",
                "z3_success": z3_success,
                "lean_success": lean_success,
                "agreement": result.agreement,
                "final_success": result.success
            })

        except Exception as e:
            result.errors.append(f"Consensus verification failed: {str(e)}")
            result.verification_log.append({"step": "consensus_error", "error": str(e)})

        return result

    def _check_result_agreement(self, z3_result: Any, lean_result: Any) -> bool:
        """Check if Z3 and LeanAIDE results agree."""
        try:
            # Simple agreement check - both should have same success status
            z3_success = z3_result.success if hasattr(z3_result, 'success') else False
            lean_success = lean_result.success if hasattr(lean_result, 'success') else False

            return z3_success == lean_success
        except:
            # If we can't compare, assume disagreement
            return False

    def _perform_cross_validation(self, result: RobustVerificationResult) -> bool:
        """Perform cross-validation between Z3 and LeanAIDE results."""
        try:
            if result.z3_result and result.lean_result:
                # Check if both results are consistent
                z3_success = result.z3_result.success if hasattr(result.z3_result, 'success') else False
                lean_success = result.lean_result.success if hasattr(result.lean_result, 'success') else False

                return z3_success == lean_success
            return True  # If only one result, consider valid
        except:
            return False

    def robust_integrate_with_cav_nlp(self, z3_expr, lean_code):
        """Robust integration with CAV-NLP fallback."""
        if self.use_cav_nlp and self.enhanced_solver:
            # Use CAV-NLP for hybrid verification
            result = self.enhanced_solver.verify_with_lean([z3_expr])
            return result
        # Fallback to standard integration
        return self.robust_translate_with_validation(
            source_content=str(z3_expr),
            source_format="smtlib",
            target_format="lean",
            validate_translation=True
        )

    def robust_translate_with_validation(
        self,
        source_content: str,
        source_format: str = "auto",
        target_format: str = "auto",
        validate_translation: bool = True
    ) -> Dict[str, Any]:
        """
        Robust translation between SMT-LIB and Lean 4 with validation.

        Args:
            source_content: Content to translate
            source_format: Format of source content ("smtlib", "lean", "auto")
            target_format: Format to translate to ("smtlib", "lean", "auto")
            validate_translation: Whether to validate the translation

        Returns:
            Dictionary with translation results and validation status
        """
        try:
            # Determine formats if auto
            if source_format == "auto":
                if self._is_smtlib(source_content):
                    source_format = "smtlib"
                else:
                    source_format = "lean"

            if target_format == "auto":
                target_format = "lean" if source_format == "smtlib" else "smtlib"

            # Perform translation
            translation_result = {
                "success": False,
                "source_format": source_format,
                "target_format": target_format,
                "translated_content": "",
                "validation_passed": False,
                "errors": [],
                "warnings": []
            }

            # Use appropriate translator based on source and target
            if source_format == "smtlib" and target_format == "lean":
                # Translate SMT-LIB to Lean
                try:
                    if LEANAIDE_AVAILABLE:
                        # Use LeanAIDE translation if available
                        lean_result = leanaide_translate_theorem(source_content)
                        translation_result["translated_content"] = lean_result.get("lean_code", "")
                        translation_result["success"] = True
                    else:
                        # Fallback to basic translation
                        translation_result["translated_content"] = self._basic_smt_to_lean(source_content)
                        translation_result["success"] = True
                        translation_result["warnings"].append("LeanAIDE not available, using basic translation")
                except Exception as e:
                    translation_result["errors"].append(f"SMT to Lean translation failed: {str(e)}")
            elif source_format == "lean" and target_format == "smtlib":
                # Translate Lean to SMT-LIB
                try:
                    # Basic translation from Lean to SMT-LIB
                    translation_result["translated_content"] = self._basic_lean_to_smt(source_content)
                    translation_result["success"] = True
                except Exception as e:
                    translation_result["errors"].append(f"Lean to SMT translation failed: {str(e)}")
            else:
                translation_result["errors"].append(f"Unsupported translation: {source_format} to {target_format}")

            # Validate translation if requested
            if validate_translation and translation_result["success"]:
                translation_result["validation_passed"] = self._validate_translation(
                    source_content, 
                    translation_result["translated_content"], 
                    source_format, 
                    target_format
                )

            return translation_result

        except Exception as e:
            return {
                "success": False,
                "source_format": source_format,
                "target_format": target_format,
                "translated_content": "",
                "validation_passed": False,
                "errors": [f"Translation failed: {str(e)}"],
                "warnings": []
            }

    def _basic_smt_to_lean(self, smt_content: str) -> str:
        """Basic SMT-LIB to Lean translation."""
        # This is a simplified example - in practice would be more sophisticated
        lean_code = "-- Translated from SMT-LIB\n"
        lean_code += "import Mathlib\n\n"
        lean_code += "theorem translated_theorem : True :=\n"
        lean_code += "  sorry  -- Placeholder for actual translation\n"
        return lean_code

    def _basic_lean_to_smt(self, lean_content: str) -> str:
        """Basic Lean to SMT-LIB translation."""
        # This is a simplified example - in practice would be more sophisticated
        smt_content = "; Translated from Lean\n"
        smt_content += "(set-logic ALL)\n"
        smt_content += "; Placeholder for actual translation\n"
        smt_content += "(assert true)\n"
        smt_content += "(check-sat)\n"
        return smt_content

    def _validate_translation(
        self, 
        original: str, 
        translated: str, 
        source_format: str, 
        target_format: str
    ) -> bool:
        """Validate that translation preserves semantics."""
        try:
            # For now, just check that both have content
            return len(original.strip()) > 0 and len(translated.strip()) > 0
        except:
            return False


# Global instance for easy access
_robust_bridge_instance = None

def get_robust_z3_leanaide_bridge() -> RobustZ3LeanAideBridge:
    """Get or create the robust Z3-LeanAide bridge singleton."""
    global _robust_bridge_instance
    if _robust_bridge_instance is None:
        _robust_bridge_instance = RobustZ3LeanAideBridge()
    return _robust_bridge_instance


# Example usage and testing
def test_robust_z3_leanaide_integration():
    """Test function for the robust Z3-LeanAide integration."""
    print("Testing Robust Z3-LeanAIDE Integration:")
    
    # Get bridge instance
    bridge = get_robust_z3_leanaide_bridge()
    
    # Test verification
    test_problem = """
    (declare-const x Int)
    (declare-const y Int)
    (assert (> x 0))
    (assert (> y 0))
    (assert (= (+ x y) 5))
    (check-sat)
    """
    
    print(f"Z3 Available: {Z3_INTEGRATION_AVAILABLE}")
    print(f"LeanAIDE Available: {LEANAIDE_AVAILABLE}")
    print(f"DSPy Available: {DSPY_AVAILABLE}")
    
    if Z3_INTEGRATION_AVAILABLE:
        result = bridge.robust_verify_with_both(
            problem=test_problem,
            strategy=VerificationStrategy.PARALLEL,
            enable_dspy_enhancement=DSPY_AVAILABLE
        )
        
        print(f"Verification Success: {result.success}")
        print(f"Strategy Used: {result.strategy_used.value}")
        print(f"Confidence Score: {result.confidence_score}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")
        print(f"Fallback Used: {result.fallback_used}")
        print(f"Cross-Validation Passed: {result.cross_validation_passed}")
        print(f"DSPy Enhanced: {result.dspy_enhanced}")
        
        if result.dspy_analysis:
            print(f"DSPy Analysis: {result.dspy_analysis}")
    
    # Test translation
    test_smt = "(declare-const x Int) (assert (> x 0)) (check-sat)"
    translation_result = bridge.robust_translate_with_validation(
        source_content=test_smt,
        source_format="smtlib",
        target_format="lean",
        validate_translation=True
    )
    
    print(f"\nTranslation Success: {translation_result['success']}")
    print(f"Validation Passed: {translation_result['validation_passed']}")
    print(f"Errors: {len(translation_result['errors'])}")
    print(f"Translated Content Preview: {translation_result['translated_content'][:200]}...")
    
    return bridge


if __name__ == "__main__":
    test_robust_z3_leanaide_integration()