"""
ROMA-MDAP-MAKER + Associative Recomposition Integration

This module combines:
1. ROMA (Recursive Open Meta-Agents) - Hierarchical decomposition
2. MDAP (Multi-Agent Debate Protocol) - Multi-agent validation
3. MAKER - Structured workflow orchestration
4. Associative Recomposition - Domain-agnostic LLM + algorithmic verification

Complete pipeline:
    Problem → ROMA Decomposition → Associative Recomposition → MDAP Validation → Solution

Architecture:
    Layer 1: ROMA Hierarchical Decomposition
        ↓
    Layer 2: Associative Recomposition (LLM + Algorithmic)
        ↓
    Layer 3: MDAP Multi-Agent Validation
        ↓
    Layer 4: Ground Truth Verification

Author: OpenEvolve
Date: 2026-01-10
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Import ROMA+MDAP+MAKER components
try:
    from roma_mdap_maker_engine import (
        ROMAMDAPMakerEngine,
        ROMAMDAPMakerConfig,
        ROMARedFlagger,
        HierarchicalVotingStrategy,
        AdaptiveKSelector,
        ROMAIntrospectionEngine,
        create_roma_mdap_maker_config,
        get_roma_mdap_maker_status
    )
    from roma_mdap_maker_reliability_ssot import get_reliability_config
    ROMA_MDAP_MAKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ROMA-MDAP-MAKER not available: {e}")
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_reliability_config = None

# Import Associative Recomposition components
try:
    from associative_recomposition import (
        AssociativeRecomposer,
        AssemblyPlanJSON,
        DomainClassification
    )
    ASSOCIATIVE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Associative recomposition not available: {e}")
    ASSOCIATIVE_AVAILABLE = False

# Import Ground Truth Store
try:
    from ground_truth_store import GroundTruthStore, get_ground_truth_store
    GROUND_TRUTH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Ground truth store not available: {e}")
    GROUND_TRUTH_AVAILABLE = False

# Import Evaluator Team
try:
    from evaluator_team import (
        EvaluatorTeam,
        EvaluationMetric,
        EvaluationCriterion,
        EvaluatorAssessment,
        IntegratedEvaluation
    )
    EVALUATOR_TEAM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Evaluator Team not available: {e}")
    EVALUATOR_TEAM_AVAILABLE = False

# Import Adaptive Gauntlet System
try:
    from adaptive_gauntlet_system import (
        AdaptiveGauntletSystem,
        PerformanceTracker
    )
    from gauntlet_manager import GauntletManager
    from sovereign_data_models import (
        ProblemDefinition,
        SubProblem,
        ComplexityScore,
        ProblemType,
        SubProblemType,
        DomainContext,
        GauntletDefinition,
        GauntletRoundRule
    )
    GAUNTLET_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Adaptive Gauntlet System not available: {e}")
    GAUNTLET_SYSTEM_AVAILABLE = False        
# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ROMAMDAPMakerAssociativeConfig:
    """Configuration for ROMA-MDAP-MAKER + Associative integration"""

    # ROMA-MDAP-MAKER settings
    roma_max_depth_analysis: int = 3
    roma_max_depth_solving: int = 2
    roma_execution_mode: str = "recursive"
    roma_enable_checkpoints: bool = False
    roma_enable_logging: bool = True

    # MDAP/MAKER settings
    mdap_enabled: bool = True
    mdap_k_ahead: int = 3
    mdap_max_samples: int = 100
    mdap_enable_red_flagging: bool = True
    mdap_max_token_length: int = 750
    mdap_min_confidence: float = 0.2

    # Integration settings
    apply_maker_to_roma_atomic: bool = True
    apply_maker_to_roma_planning: bool = True
    aggregate_maker_results: bool = True
    enable_hierarchical_voting: bool = True
    enable_adaptive_k: bool = True

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10000

    # Fault tolerance
    max_retries: int = 3
    timeout_seconds: int = 300
    fallback_policy: str = "escalate_then_best_effort"

    # Associative Recomposition settings
    use_associative_recomposition: bool = True
    associative_max_retries: int = 3
    associative_use_agentjson: bool = True

    # Ground Truth settings
    enable_ground_truth: bool = True
    ground_truth_storage_path: str = "roma_mdap_maker_ground_truth.json"

    # Integration settings
    apply_mdap_to_recomposed: bool = True  # Apply MDAP validation after recomposition
    enable_hierarchical_validation: bool = True
    
    # Evaluator Team settings
    use_evaluator_team: bool = True
    evaluator_threshold: str = "standard_approval"  # "minimal_acceptance", "standard_approval", "high_quality"
    evaluator_num_members: int = 3
    
    # Gauntlet System settings
    use_gauntlet_system: bool = True
    gauntlet_difficulty: str = "adaptive"  # "easy", "medium", "hard", "adaptive"
    
    # Recursive Refinement settings
    max_refinement_attempts: int = 3
    min_acceptance_score: float = 75.0  # Out of 100 from Evaluator Team

    # Provider settings
    provider: str = "openai"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.1

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MAIN ENGINE
# =============================================================================

class ROMAMDAPMakerAssociativeEngine:
    """
    Main engine combining ROMA decomposition, associative recomposition,
    and MDAP multi-agent validation.
    """

    def __init__(
        self,
        config: ROMAMDAPMakerAssociativeConfig,
        team: Optional[Any] = None,
        ground_truth_store: Optional[GroundTruthStore] = None
    ):
        """
        Initialize the combined engine.

        Args:
            config: Configuration for all components
            team: Optional team for MDAP
            ground_truth_store: Optional ground truth store
        """
        self.config = config
        self.team = team

        # Initialize ROMA-MDAP-MAKER engine
        if ROMA_MDAP_MAKER_AVAILABLE:
            # Create ROMA-MDAP-MAKER config with all preserved parameters
            roma_mdap_maker_config = create_roma_mdap_maker_config(
                roma_max_depth_analysis=config.roma_max_depth_analysis,
                roma_max_depth_solving=config.roma_max_depth_solving,
                roma_execution_mode=config.roma_execution_mode,
                roma_enable_checkpoints=config.roma_enable_checkpoints,
                roma_enable_logging=config.roma_enable_logging,
                mdap_enabled=config.mdap_enabled,
                mdap_k_ahead=config.mdap_k_ahead,
                mdap_max_samples=config.mdap_max_samples,
                mdap_enable_red_flagging=config.mdap_enable_red_flagging,
                mdap_max_token_length=config.mdap_max_token_length,
                mdap_min_confidence=config.mdap_min_confidence,
                apply_maker_to_roma_atomic=config.apply_maker_to_roma_atomic,
                apply_maker_to_roma_planning=config.apply_maker_to_roma_planning,
                aggregate_maker_results=config.aggregate_maker_results,
                enable_hierarchical_voting=config.enable_hierarchical_voting,
                enable_adaptive_k=config.enable_adaptive_k,
                enable_caching=config.enable_caching,
                cache_ttl_seconds=config.cache_ttl_seconds,
                cache_max_size=config.cache_max_size,
                max_retries=config.max_retries,
                timeout_seconds=config.timeout_seconds,
                fallback_policy=config.fallback_policy,
                provider=config.provider,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                metadata=config.metadata
            )

            self.roma_mdap_maker_engine = ROMAMDAPMakerEngine(
                config=roma_mdap_maker_config,
                team=team
            )
        else:
            self.roma_mdap_maker_engine = None
            logger.warning("ROMA-MDAP-MAKER engine not available")

        # Initialize Associative Recomposer
        if ASSOCIATIVE_AVAILABLE and config.use_associative_recomposition:
            # Setup ground truth store
            if GROUND_TRUTH_AVAILABLE and config.enable_ground_truth:
                self.ground_truth_store = ground_truth_store or get_ground_truth_store()
            else:
                self.ground_truth_store = None

            self.associative_recomposer = AssociativeRecomposer(
                ground_truth_store=self.ground_truth_store,
                use_agentjson=config.associative_use_agentjson,
                max_retries=config.associative_max_retries
            )
        else:
            self.associative_recomposer = None
            logger.warning("Associative recomposer not available")

        # Initialize Evaluator Team
        if EVALUATOR_TEAM_AVAILABLE and config.use_evaluator_team:
            self.evaluator_team = EvaluatorTeam()
            logger.info("Evaluator Team initialized")
        else:
            self.evaluator_team = None
            if config.use_evaluator_team:
                logger.warning("Evaluator Team requested but not available")

        # Initialize Adaptive Gauntlet System
        if GAUNTLET_SYSTEM_AVAILABLE and config.use_gauntlet_system:
            self.gauntlet_system = AdaptiveGauntletSystem()
            self.gauntlet_manager = GauntletManager(evaluator_team=self.evaluator_team)
            logger.info("Adaptive Gauntlet System and Manager initialized")
        else:
            self.gauntlet_system = None
            self.gauntlet_manager = None
            if config.use_gauntlet_system:
                logger.warning("Gauntlet System requested but not available")

        # Metrics
        self.metrics = {
            "total_problems_solved": 0,
            "total_decomposition_time": 0.0,
            "total_recomposition_time": 0.0,
            "total_validation_time": 0.0,
            "avg_confidence": 0.0,
            "total_sub_solutions": 0,
            "successful_recompositions": 0,
            "failed_recompositions": 0
        }

    def solve_problem_recursive(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        llm_call_fn: Optional[Callable[[str], str]] = None
    ) -> Dict[str, Any]:
        """
        End-to-end recursive problem-solving loop:
        Decompose -> Solve -> Recompose -> Evaluate -> (Refine if needed) -> Repeat
        
        Args:
            problem: Problem statement
            context: Additional context
            llm_call_fn: LLM call function

        Returns:
            Final solution approved by the Evaluator Team or best attempt
        """
        logger.info(f"\n{'#'*80}")
        logger.info(f"STARTING RECURSIVE REFINEMENT LOOP")
        logger.info(f"{'#'*80}\n")
        
        context = context or {}
        best_result = None
        
        for attempt in range(self.config.max_refinement_attempts):
            logger.info(f"\n>>> REFINEMENT ATTEMPT {attempt + 1}/{self.config.max_refinement_attempts}")
            
            # Current problem might be refined by previous attempt's feedback
            current_problem = problem
            config_overrides = {}
            
            if best_result:
                feedback = best_result.get("evaluator_assessment", {}).get("recommendations", [])
                eval_score = best_result.get("evaluator_assessment", {}).get("consensus_score", 0.0)
                
                # Intelligent Refinement Strategy
                if feedback:
                    logger.info("Injecting evaluator feedback into next iteration...")
                    current_problem = f"{problem}\n\nPREVIOUS FEEDBACK TO ADDRESS:\n" + "\n".join(f"- {f}" for f in feedback)
                
                # If score is low, boost parameters
                if eval_score < 50.0:
                    # Low quality -> Needs deeper thought and more validation
                    logger.info("Low score detected. Increasing decomposition depth and voting threshold.")
                    config_overrides["roma_max_depth_analysis"] = self.config.roma_max_depth_analysis + 1
                    config_overrides["mdap_k_ahead"] = self.config.mdap_k_ahead + 2
                elif eval_score < 70.0:
                    # Moderate quality -> Needs more careful validation
                    logger.info("Moderate score detected. Increasing voting threshold.")
                    config_overrides["mdap_k_ahead"] = self.config.mdap_k_ahead + 1

            # Standard pipeline execution
            result = self.solve_problem(current_problem, context, llm_call_fn, config_overrides)
            
            if result.get("error"):
                logger.error(f"Attempt {attempt+1} failed with error: {result['error']}")
                if not best_result:
                    best_result = result
                continue

            # Check if accepted
            val_res = result.get("mdap_validation", {})
            eval_res = result.get("evaluator_assessment", {})
            
            score = eval_res.get("consensus_score", 0.0)
            is_approved = eval_res.get("verdict") == "APPROVED"
            
            if not best_result or score > best_result.get("evaluator_assessment", {}).get("consensus_score", 0.0):
                best_result = result
                
            if is_approved or score >= self.config.min_acceptance_score:
                logger.info(f"✓ Solution ACCEPTED with score {score:.1f}")
                best_result["final_attempt"] = attempt + 1
                return best_result
            
            logger.warning(f"✗ Solution REJECTED with score {score:.1f}. Refactoring...")
            
        logger.warning(f"Maximum refinement attempts reached ({self.config.max_refinement_attempts}). Returning best result.")
        best_result["final_attempt"] = self.config.max_refinement_attempts
        return best_result

    def solve_problem(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        llm_call_fn: Optional[Callable[[str], str]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete problem-solving pipeline:
        1. ROMA decomposition
        2. Associative recomposition
        3. MDAP validation

        Args:
            problem: Problem statement
            context: Additional context
            llm_call_fn: LLM call function (optional, uses default if not provided)
            config_overrides: Optional overrides for engine configuration

        Returns:
            Complete solution with all metadata
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ROMA-MDAP-MAKER + ASSOCIATIVE RECOMPOSITION PIPELINE")
        logger.info(f"{'='*80}\n")
        logger.info(f"Problem: {problem[:100]}...")

        start_time = time.time()
        context = context or {}
        
        # Apply overrides locally if provided
        active_config = self.config
        if config_overrides:
            # We'll handle overrides by passing them to sub-methods
            logger.info(f"Applying config overrides for current task: {config_overrides}")

        try:
            # Phase 1: ROMA Decomposition
            logger.info("\n[PHASE 1] ROMA Hierarchical Decomposition")
            phase1_start = time.time()

            roma_result = self._roma_decompose(problem, context, config_overrides)

            phase1_time = time.time() - phase1_start
            logger.info(f"✓ ROMA decomposition completed in {phase1_time:.2f}s")

            if roma_result.get("error"):
                return {
                    "error": roma_result["error"],
                    "phase": "roma_decomposition",
                    "problem": problem
                }

            # Phase 2: Associative Recomposition
            logger.info("\n[PHASE 2] Associative Recomposition")
            phase2_start = time.time()

            recomposition_result = self._associative_recompose(
                roma_result,
                problem,
                context,
                llm_call_fn
            )

            phase2_time = time.time() - phase2_start
            logger.info(f"✓ Associative recomposition completed in {phase2_time:.2f}s")

            if recomposition_result.get("error"):
                self.metrics["failed_recompositions"] += 1
                return {
                    "error": recomposition_result["error"],
                    "phase": "associative_recomposition",
                    "problem": problem,
                    "roma_result": roma_result
                }

            self.metrics["successful_recompositions"] += 1

            # Phase 3: Comprehensive Evaluation (Evaluator Team + Gauntlet + MDAP)
            logger.info("\n[PHASE 3] Comprehensive Evaluation (Evaluator Team + Gauntlet)")
            phase3_start = time.time()

            validation_result = self._evaluate_solution(
                recomposition_result,
                problem,
                context
            )

            phase3_time = time.time() - phase3_start
            logger.info(f"✓ Evaluation completed in {phase3_time:.2f}s")

            # Compile final result
            total_time = time.time() - start_time

            result = {
                "success": True,
                "problem": problem,
                "solution": recomposition_result.get("assembled_solution"),
                "final_solution": recomposition_result.get("assembled_solution"),  # Alias
                "confidence": validation_result.get("confidence", 0.5),

                # ROMA results
                "roma_decomposition": roma_result.get("decomposition"),
                "roma_hierarchy": roma_result.get("decomposition"),
                "roma_dag": roma_result.get("dag_info"),
                "roma_depth": roma_result.get("max_depth"),

                # Recomposition results
                "recomposition_metadata": recomposition_result.get("metadata"),
                "domain_classification": recomposition_result.get("metadata", {}).get("classification"),
                "assembly_plan": recomposition_result.get("metadata", {}).get("plan"),

                # MDAP results
                "mdap_validation": validation_result,
                "validation_details": validation_result.get("validation_details"),

                # Timing
                "decomposition_time": phase1_time,
                "recomposition_time": phase2_time,
                "validation_time": phase3_time,
                "total_time": total_time,

                # Metrics
                "num_sub_solutions": len(roma_result.get("sub_solutions", [])),
                "num_atomic_tasks": roma_result.get("total_atomic_tasks", 0),
                "error_free": validation_result.get("error_rate", 1.0) == 0.0
            }

            # Update metrics
            self.metrics["total_problems_solved"] += 1
            self.metrics["total_decomposition_time"] += phase1_time
            self.metrics["total_recomposition_time"] += phase2_time
            self.metrics["total_validation_time"] += phase3_time
            self.metrics["total_sub_solutions"] += result["num_sub_solutions"]
            self.metrics["avg_confidence"] = (
                (self.metrics["avg_confidence"] * (self.metrics["total_problems_solved"] - 1) +
                 result["confidence"]) / self.metrics["total_problems_solved"]
            )

            logger.info(f"\n{'='*80}")
            logger.info(f"PIPELINE COMPLETE")
            logger.info(f"{'='*80}")
            logger.info(f"Total Time: {total_time:.2f}s")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info(f"Sub-solutions: {result['num_sub_solutions']}")
            logger.info(f"Atomic Tasks: {result['num_atomic_tasks']}")
            logger.info(f"Error-Free: {result['error_free']}")
            logger.info(f"{'='*80}\n")

            return result

        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Error in solve_problem: {e}", exc_info=True)
            return {
                "error": str(e),
                "problem": problem,
                "phase": "unknown"
            }
        except Exception as e:
            logger.critical(f"Unexpected error in solve_problem: {e}", exc_info=True)
            return {
                "error": f"Unexpected error: {str(e)}",
                "problem": problem,
                "phase": "unknown"
            }

    def _roma_decompose(
        self,
        problem: str,
        context: Dict[str, Any],
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Phase 1: ROMA Decomposition"""
        if not self.roma_mdap_maker_engine:
            return {
                "error": "ROMA-MDAP-MAKER engine not available",
                "sub_solutions": [],
                "total_atomic_tasks": 0
            }

        # Apply overrides to engine config if they match
        if config_overrides:
            for k, v in config_overrides.items():
                if hasattr(self.roma_mdap_maker_engine.config, k):
                    setattr(self.roma_mdap_maker_engine.config, k, v)
                # Ensure we also pick up mdap_enable_red_flagging and other MAKER options
                if k == "mdap_enable_red_flagging" and hasattr(self.roma_mdap_maker_engine.config, "mdap_enable_red_flagging"):
                    setattr(self.roma_mdap_maker_engine.config, "mdap_enable_red_flagging", v)

        # Use ROMA-MDAP-MAKER to decompose
        result = self.roma_mdap_maker_engine.solve_with_roma_mdap_maker(
            task=problem,
            context=context
        )

        # Extract sub-solutions from ROMA hierarchy
        sub_solutions = self._extract_sub_solutions(result)

        return {
            "decomposition": result.get("roma_hierarchy"),
            "dag_info": result.get("roma_dag"),
            "max_depth": result.get("total_steps", 0),
            "sub_solutions": sub_solutions,
            "total_atomic_tasks": result.get("total_steps", 0),
            "roma_metadata": result
        }

    def _extract_sub_solutions(
        self,
        roma_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract sub-solutions from ROMA hierarchy"""
        hierarchy = roma_result.get("roma_hierarchy", {})
        sub_solutions = []

        def extract_from_node(node: Dict[str, Any], index: int = 0):
            """Recursively extract solutions from ROMA nodes"""
            if not node.get("subtasks") or len(node.get("subtasks", [])) == 0:
                # Atomic node - extract as sub-solution
                sub_solutions.append({
                    "id": f"sol_{len(sub_solutions) + 1}",
                    "description": node.get("description", ""),
                    "solution_content": node.get("result", ""),
                    "confidence": 0.8,  # Default confidence
                    "metadata": node
                })
            else:
                # Non-atomic - recurse into children
                for i, child in enumerate(node.get("subtasks", [])):
                    extract_from_node(child, i)

        if hierarchy:
            extract_from_node(hierarchy)

        return sub_solutions

    def _associative_recompose(
        self,
        roma_result: Dict[str, Any],
        problem: str,
        context: Dict[str, Any],
        llm_call_fn: Optional[Callable[[str], str]]
    ) -> Dict[str, Any]:
        """Phase 2: Associative Recomposition"""
        if not self.associative_recomposer:
            # Fallback: simple concatenation
            sub_solutions = roma_result.get("sub_solutions", [])
            assembled = self._simple_assemble(sub_solutions)
            return {
                "assembled_solution": assembled,
                "metadata": {
                    "fallback": True,
                    "method": "simple_concatenation"
                }
            }

        sub_solutions = roma_result.get("sub_solutions", [])

        if not sub_solutions:
            return {
                "error": "No sub-solutions to recompose",
                "assembled_solution": None
            }

        # Convert sub_solutions to associative format
        formatted_solutions = {}
        for i, sol in enumerate(sub_solutions):
            sol_id = sol.get("id", f"sol_{i+1}")
            formatted_solutions[sol_id] = {
                "description": sol.get("description", ""),
                "solution_content": sol.get("solution_content", ""),
                "confidence_score": sol.get("confidence", 0.8),
                "metadata": sol.get("metadata", {})
            }

        # Run associative recomposition
        assembled, metadata = self.associative_recomposer.recompose_with_verification(
            sub_solutions=formatted_solutions,
            conflicts=[],  # ROMA already handled dependencies
            problem_statement=problem,
            llm_call_fn=llm_call_fn or self._default_llm_call
        )

        if not assembled:
            return {
                "error": "Associative recomposition failed",
                "assembled_solution": None,
                "metadata": metadata
            }

        return {
            "assembled_solution": assembled,
            "metadata": metadata
        }

    def _evaluate_solution(
        self,
        recomposition_result: Dict[str, Any],
        problem: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Phase 3: Comprehensive Evaluation (Evaluator Team + Gauntlet + MDAP)
        
        Integrates:
        1. MDAP Multi-Agent Validation (Base Layer)
        2. Evaluator Team Assessment (Strategic Layer)
        3. Adaptive Gauntlet System (Stress Test Layer)
        """
        assembled_solution = recomposition_result.get("assembled_solution")
        if not assembled_solution:
            return {"error": "No solution to evaluate", "confidence": 0.0}

        # 1. MDAP Validation (Base Layer)
        # We still use MDAP validation as it leverages the ROMA engine for factual/logic checks
        mdap_result = self._mdap_validate(recomposition_result, problem, context)
        
        # 2. Evaluator Team Assessment (Strategic Layer)
        evaluator_result = {}
        if self.evaluator_team:
            try:
                # Determine content type from recomposition metadata
                metadata = recomposition_result.get("metadata", {})
                classification = metadata.get("classification", {})
                content_type = classification.get("solution_type", "general")
                if isinstance(content_type, dict): # Handle enum serialization
                     content_type = content_type.get("value", "general")
                elif hasattr(content_type, "value"):
                     content_type = content_type.value
                
                # Map config threshold
                from evaluator_team import EvaluationThreshold
                threshold_map = {
                    "minimal_acceptance": EvaluationThreshold.MINIMAL_ACCEPTANCE,
                    "standard_approval": EvaluationThreshold.STANDARD_APPROVAL,
                    "high_quality": EvaluationThreshold.HIGH_QUALITY,
                    "exceptional": EvaluationThreshold.EXCEPTIONAL
                }
                threshold = threshold_map.get(self.config.evaluator_threshold, EvaluationThreshold.STANDARD_APPROVAL)
                
                logger.info(f"Running Evaluator Team assessment for {content_type}...")
                
                evaluation = self.evaluator_team.evaluate_content(
                    content=assembled_solution,
                    content_type=str(content_type),
                    threshold=threshold,
                    num_evaluators=self.config.evaluator_num_members
                )
                
                evaluator_result = {
                    "consensus_score": evaluation.consensus_score,
                    "verdict": evaluation.final_verdict,
                    "recommendations": evaluation.recommendations,
                    "report": self.evaluator_team.generate_evaluation_report(evaluation),
                    "confidence_intervals": evaluation.confidence_intervals
                }
                logger.info(f"Evaluator Verdict: {evaluation.final_verdict} (Score: {evaluation.consensus_score:.1f})")
                
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.error(f"Evaluator Team assessment failed: {e}", exc_info=True)
                evaluator_result = {"error": str(e)}

        # 3. Adaptive Gauntlet System (Stress Test Layer)
        gauntlet_result = {}
        if self.gauntlet_system and GAUNTLET_SYSTEM_AVAILABLE:
            try:
                # Create artifacts
                prob_def, sub_prob = self._create_gauntlet_artifacts(problem, context)
                
                # Get mock team performance (in a real system this would persist)
                team_perf = {} 
                
                # Create adaptive gauntlet
                gauntlet_def = self.gauntlet_system.create_adaptive_gauntlet(
                    problem=prob_def,
                    sub_problem=sub_prob,
                    team_performance=team_perf
                )
                
                # Execute gauntlet using GauntletManager
                if self.gauntlet_manager:
                    gauntlet_result = self.gauntlet_manager.execute_gauntlet(
                        gauntlet_def,
                        assembled_solution,
                        context
                    )
                    
                    # Update performance tracker
                    self.gauntlet_system.update_performance_from_result(
                        gauntlet_id=gauntlet_def.gauntlet_id,
                        team_id="default_team",
                        domain=prob_def.domain_context.domain,
                        problem_type=prob_def.problem_type.value,
                        passed=gauntlet_result.get("passed", False),
                        score=gauntlet_result.get("final_score", 0.0)
                    )
                    
                    logger.info(f"Gauntlet Result: {'PASSED' if gauntlet_result.get('passed') else 'FAILED'} (Score: {gauntlet_result.get('final_score', 0.0):.2f})")
                
            except (RuntimeError, ValueError, KeyError) as e:
                logger.error(f"Gauntlet System run failed: {e}", exc_info=True)
                gauntlet_result = {"error": str(e)}

        # Calculate Unified Confidence
        # Mix MDAP confidence (0-1) and Evaluator score (0-100)
        unified_confidence = mdap_result.get("confidence", 0.5)
        if "consensus_score" in evaluator_result:
            # Weighted average: 60% Evaluator, 40% MDAP
            eval_conf = evaluator_result["consensus_score"] / 100.0
            mdap_conf = mdap_result.get("confidence", 0.5)
            unified_confidence = (eval_conf * 0.6) + (mdap_conf * 0.4)

        return {
            "confidence": unified_confidence,
            "mdap_validation": mdap_result,
            "evaluator_assessment": evaluator_result,
            "gauntlet_result": gauntlet_result,
            "error_rate": mdap_result.get("error_rate", 0.0),
            "validated": (evaluator_result.get("verdict") == "APPROVED") or (not self.evaluator_team and mdap_result.get("validated", False)),
            "validation_details": {
                "mdap": mdap_result.get("validation_details"),
                "evaluator": evaluator_result.get("report")
            }
        }

    def _create_gauntlet_artifacts(
        self,
        problem: str,
        context: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """Create sovereign data models for gauntlet"""
        if not GAUNTLET_SYSTEM_AVAILABLE:
            return None, None
            
        # Create minimal valid objects
        domain = context.get("domain", "general")
        
        # Estimate complexity
        complexity_val = 5.0
        if len(problem) > 500: complexity_val += 2.0
        if "requirements" in context: complexity_val += len(context["requirements"]) * 0.5
        complexity_val = min(10.0, complexity_val)
        
        complexity = ComplexityScore(
            explanation="Estimated from problem length and requirements",
            cognitive_complexity=complexity_val,
            computational_complexity=complexity_val,
            domain_complexity=complexity_val,
            integration_complexity=complexity_val,
            overall_complexity=complexity_val
        )
        
        prob_def = ProblemDefinition(
            id=f"prob_{int(time.time())}",
            title="Current Problem",
            description=problem,
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=DomainContext(domain=domain),
            complexity_score=complexity
        )
        
        sub_prob = SubProblem(
            id=f"sub_{int(time.time())}",
            parent_id=prob_def.id,
            title="Main Task",
            description=problem,
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=complexity
        )
        
        return prob_def, sub_prob

    def _mdap_validate(
        self,
        recomposition_result: Dict[str, Any],
        problem: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 3: MDAP Multi-Agent Validation"""
        if not self.roma_mdap_maker_engine or not self.config.apply_mdap_to_recomposed:
            # Skip MDAP validation
            return {
                "confidence": 0.8,
                "validation_details": "MDAP validation skipped",
                "error_rate": 0.0
            }

        # Extract metadata for validation
        assembled_solution = recomposition_result.get("assembled_solution")
        metadata = recomposition_result.get("metadata", {})

        # Create validation task
        validation_task = {
            "id": "validate_recomposed",
            "description": f"Validate the following solution for: {problem}",
            "result": assembled_solution,
            "schema": None,
            "task_type": "validation",
            "priority": 0,
            "metadata": {
                "domain": metadata.get("classification", {}),
                "plan": metadata.get("plan", {})
            }
        }

        # Use ROMA-MDAP-MAKER to validate
        validation_result = self.roma_mdap_maker_engine.solve_with_roma_mdap_maker(
            task=validation_task["description"],
            context={
                "solution_to_validate": assembled_solution,
                **context
            }
        )

        return {
            "confidence": validation_result.get("confidence", 0.5),
            "validation_details": validation_result,
            "error_rate": validation_result.get("error_rate", 0.0),
            "red_flags": validation_result.get("red_flags", 0),
            "validated": True
        }

    def _simple_assemble(self, sub_solutions: List[Dict[str, Any]]) -> str:
        """Simple fallback assembly (concatenation)"""
        parts = []
        for sol in sub_solutions:
            description = sol.get("description", "")
            content = sol.get("solution_content", "")
            parts.append(f"## {description}\n\n{content}\n")
        return "\n".join(parts)

    def _default_llm_call(self, prompt: str) -> str:
        """Default LLM call function"""
        import os

        # Check for API key
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No API key found, returning mock response")
            return json.dumps({
                "classification": {
                    "domain": "general",
                    "solution_type": "text",
                    "field": "unknown",
                    "complexity": "medium",
                    "confidence": 0.5
                },
                "target_solution_type": "text",
                "instructions": [],
                "intro": "Mock assembly plan",
                "conclusion": "Mock conclusion"
            })

        # Use llm_utils to make actual call
        try:
            from llm_utils import _compose_messages, _request_openai_compatible_chat

            messages = _compose_messages(
                system_prompt="You are a helpful assistant that analyzes problems and creates structured plans.",
                user_prompt=prompt
            )

            response = _request_openai_compatible_chat(
                api_key=api_key,
                base_url="https://api.openai.com/v1",
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=2000
            )

            return response or ""
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"LLM call failed: {e}")
            return json.dumps({
                "error": str(e),
                "classification": {
                    "domain": "general",
                    "solution_type": "text",
                    "field": "unknown",
                    "complexity": "low",
                    "confidence": 0.3
                }
            })

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset metrics"""
        self.metrics = {
            "total_problems_solved": 0,
            "total_decomposition_time": 0.0,
            "total_recomposition_time": 0.0,
            "total_validation_time": 0.0,
            "avg_confidence": 0.0,
            "total_sub_solutions": 0,
            "successful_recompositions": 0,
            "failed_recompositions": 0
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_romamdapmaker_associative_config(
    preset: str = "standard",
    roma_max_depth_analysis: Optional[int] = None,
    roma_max_depth_solving: Optional[int] = None,
    roma_execution_mode: Optional[str] = None,
    roma_enable_checkpoints: Optional[bool] = None,
    roma_enable_logging: Optional[bool] = None,
    mdap_enabled: Optional[bool] = None,
    mdap_k_ahead: Optional[int] = None,
    mdap_max_samples: Optional[int] = None,
    mdap_enable_red_flagging: Optional[bool] = None,
    mdap_max_token_length: Optional[int] = None,
    mdap_min_confidence: Optional[float] = None,
    apply_maker_to_roma_atomic: Optional[bool] = None,
    apply_maker_to_roma_planning: Optional[bool] = None,
    aggregate_maker_results: Optional[bool] = None,
    enable_hierarchical_voting: Optional[bool] = None,
    enable_adaptive_k: Optional[bool] = None,
    enable_caching: Optional[bool] = None,
    cache_ttl_seconds: Optional[int] = None,
    cache_max_size: Optional[int] = None,
    max_retries: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    fallback_policy: Optional[str] = None,
    use_associative_recomposition: Optional[bool] = None,
    associative_max_retries: Optional[int] = None,
    associative_use_agentjson: Optional[bool] = None,
    enable_ground_truth: Optional[bool] = None,
    ground_truth_storage_path: Optional[str] = None,
    apply_mdap_to_recomposed: Optional[bool] = None,
    enable_hierarchical_validation: Optional[bool] = None,
    use_evaluator_team: Optional[bool] = None,
    evaluator_threshold: Optional[str] = None,
    evaluator_num_members: Optional[int] = None,
    use_gauntlet_system: Optional[bool] = None,
    gauntlet_difficulty: Optional[str] = None,
    max_refinement_attempts: Optional[int] = None,
    min_acceptance_score: Optional[float] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ROMAMDAPMakerAssociativeConfig:
    """
    Create ROMA-MDAP-MAKER + Associative configuration.

    Uses the SSOT (Single Source of Truth) config from roma_mdap_maker_reliability_ssot.py
    for all ROMA-MDAP-MAKER parameters, with optional overrides.

    Args:
        preset: Configuration preset ("standard", "thorough", "fast", "validation", "recomposition")
        roma_max_depth_analysis: ROMA max depth for analysis (overrides preset)
        roma_max_depth_solving: ROMA max depth for solving (overrides preset)
        roma_execution_mode: ROMA execution mode (overrides preset)
        roma_enable_checkpoints: Enable ROMA checkpoints (overrides preset)
        roma_enable_logging: Enable ROMA logging (overrides preset)
        mdap_enabled: Enable MDAP validation (overrides preset)
        mdap_k_ahead: MAKER voting threshold (overrides preset)
        mdap_max_samples: Max samples per voting round (overrides preset)
        mdap_enable_red_flagging: Enable red-flagging (overrides preset)
        mdap_max_token_length: Max token length for MDAP (overrides preset)
        mdap_min_confidence: Min confidence for MDAP (overrides preset)
        apply_maker_to_roma_atomic: Apply MAKER to atomic tasks (overrides preset)
        apply_maker_to_roma_planning: Apply MAKER to planning (overrides preset)
        aggregate_maker_results: Aggregate voted results (overrides preset)
        enable_hierarchical_voting: Enable hierarchical voting (overrides preset)
        enable_adaptive_k: Enable adaptive k selection (overrides preset)
        enable_caching: Enable result caching (overrides preset)
        cache_ttl_seconds: Cache TTL in seconds (overrides preset)
        cache_max_size: Maximum cache size (overrides preset)
        max_retries: Max retries per task (overrides preset)
        timeout_seconds: Timeout per task (overrides preset)
        fallback_policy: Policy for task failures (overrides preset)
        use_associative_recomposition: Use associative recomposition
        associative_max_retries: Max retries for recomposition
        associative_use_agentjson: Use AgentJSON for parsing
        enable_ground_truth: Enable ground truth verification
        ground_truth_storage_path: Path for ground truth storage
        apply_mdap_to_recomposed: Apply MDAP after recomposition
        enable_hierarchical_validation: Enable hierarchical validation
        use_evaluator_team: Use Evaluator Team for assessment
        evaluator_threshold: Approval threshold for team
        evaluator_num_members: Number of members in Evaluator Team
        use_gauntlet_system: Use Adaptive Gauntlet System
        gauntlet_difficulty: Difficulty setting for Gauntlet
        max_refinement_attempts: Max attempts for recursive refinement
        min_acceptance_score: Minimum score to accept solution
        provider: LLM provider
        api_key: LLM API key
        model: Model name
        temperature: Sampling temperature
        metadata: Additional metadata
        **kwargs: Additional configuration

    Returns:
        ROMAMDAPMakerAssociativeConfig object
    """
    # Get SSOT config from reliability_ssot
    if get_reliability_config is not None:
        ssot_config = get_reliability_config(preset=preset)

        # Apply defaults from SSOT config if not explicitly provided
        if roma_max_depth_analysis is None:
            roma_max_depth_analysis = getattr(ssot_config, 'roma_max_depth_analysis', 3)
        if roma_max_depth_solving is None:
            roma_max_depth_solving = getattr(ssot_config, 'roma_max_depth_solving', 2)
        if roma_execution_mode is None:
            roma_execution_mode = getattr(ssot_config, 'roma_execution_mode', "recursive")
        if roma_enable_checkpoints is None:
            roma_enable_checkpoints = getattr(ssot_config, 'roma_enable_checkpoints', False)
        if roma_enable_logging is None:
            roma_enable_logging = getattr(ssot_config, 'roma_enable_logging', True)
        if mdap_enabled is None:
            mdap_enabled = getattr(ssot_config, 'mdap_enabled', True)
        if mdap_k_ahead is None:
            mdap_k_ahead = getattr(ssot_config, 'mdap_k_ahead', 3)
        if mdap_max_samples is None:
            mdap_max_samples = getattr(ssot_config, 'mdap_max_samples', 100)
        if mdap_enable_red_flagging is None:
            mdap_enable_red_flagging = getattr(ssot_config, 'mdap_enable_red_flagging', True)
        if mdap_max_token_length is None:
            mdap_max_token_length = getattr(ssot_config, 'mdap_max_token_length', 750)
        if mdap_min_confidence is None:
            mdap_min_confidence = getattr(ssot_config, 'mdap_min_confidence', 0.2)
        if apply_maker_to_roma_atomic is None:
            apply_maker_to_roma_atomic = getattr(ssot_config, 'apply_maker_to_roma_atomic', True)
        if apply_maker_to_roma_planning is None:
            apply_maker_to_roma_planning = getattr(ssot_config, 'apply_maker_to_roma_planning', True)
        if aggregate_maker_results is None:
            aggregate_maker_results = getattr(ssot_config, 'aggregate_maker_results', True)
        if enable_hierarchical_voting is None:
            enable_hierarchical_voting = getattr(ssot_config, 'enable_hierarchical_voting', True)
        if enable_adaptive_k is None:
            enable_adaptive_k = getattr(ssot_config, 'enable_adaptive_k', True)
        if enable_caching is None:
            enable_caching = getattr(ssot_config, 'enable_caching', True)
        if cache_ttl_seconds is None:
            cache_ttl_seconds = getattr(ssot_config, 'cache_ttl_seconds', 3600)
        if cache_max_size is None:
            cache_max_size = getattr(ssot_config, 'cache_max_size', 10000)
        if max_retries is None:
            max_retries = getattr(ssot_config, 'max_retries', 3)
        if timeout_seconds is None:
            timeout_seconds = getattr(ssot_config, 'timeout_seconds', 300)
        if fallback_policy is None:
            fallback_policy = getattr(ssot_config, 'fallback_policy', "escalate_then_best_effort")
        if provider is None:
            provider = getattr(ssot_config, 'provider', "openai")
        if model is None:
            model = getattr(ssot_config, 'model', "gpt-4o-mini")
        if temperature is None:
            temperature = getattr(ssot_config, 'temperature', 0.1)
    else:
        # Fallback to hardcoded defaults if SSOT not available
        if roma_max_depth_analysis is None:
            roma_max_depth_analysis = 3
        if roma_max_depth_solving is None:
            roma_max_depth_solving = 2
        if roma_execution_mode is None:
            roma_execution_mode = "recursive"
        if roma_enable_checkpoints is None:
            roma_enable_checkpoints = False
        if roma_enable_logging is None:
            roma_enable_logging = True
        if mdap_enabled is None:
            mdap_enabled = True
        if mdap_k_ahead is None:
            mdap_k_ahead = 3
        if mdap_max_samples is None:
            mdap_max_samples = 100
        if mdap_enable_red_flagging is None:
            mdap_enable_red_flagging = True
        if mdap_max_token_length is None:
            mdap_max_token_length = 750
        if mdap_min_confidence is None:
            mdap_min_confidence = 0.2
        if apply_maker_to_roma_atomic is None:
            apply_maker_to_roma_atomic = True
        if apply_maker_to_roma_planning is None:
            apply_maker_to_roma_planning = True
        if aggregate_maker_results is None:
            aggregate_maker_results = True
        if enable_hierarchical_voting is None:
            enable_hierarchical_voting = True
        if enable_adaptive_k is None:
            enable_adaptive_k = True
        if enable_caching is None:
            enable_caching = True
        if cache_ttl_seconds is None:
            cache_ttl_seconds = 3600
        if cache_max_size is None:
            cache_max_size = 10000
        if max_retries is None:
            max_retries = 3
        if timeout_seconds is None:
            timeout_seconds = 300
        if fallback_policy is None:
            fallback_policy = "escalate_then_best_effort"
        if provider is None:
            provider = "openai"
        if model is None:
            model = "gpt-4o-mini"
        if temperature is None:
            temperature = 0.1

    # Associative-specific defaults
    if use_associative_recomposition is None:
        use_associative_recomposition = True
    if associative_max_retries is None:
        associative_max_retries = 3
    if associative_use_agentjson is None:
        associative_use_agentjson = True
    if enable_ground_truth is None:
        enable_ground_truth = True
    if ground_truth_storage_path is None:
        ground_truth_storage_path = "roma_mdap_maker_ground_truth.json"
    if apply_mdap_to_recomposed is None:
        apply_mdap_to_recomposed = True
    if enable_hierarchical_validation is None:
        enable_hierarchical_validation = True
    if use_evaluator_team is None:
        use_evaluator_team = True
    if evaluator_threshold is None:
        evaluator_threshold = "standard_approval"
    if evaluator_num_members is None:
        evaluator_num_members = 3
    if use_gauntlet_system is None:
        use_gauntlet_system = True
    if gauntlet_difficulty is None:
        gauntlet_difficulty = "adaptive"
    if max_refinement_attempts is None:
        max_refinement_attempts = 3
    if min_acceptance_score is None:
        min_acceptance_score = 75.0
    return ROMAMDAPMakerAssociativeConfig(
        roma_max_depth_analysis=roma_max_depth_analysis,
        roma_max_depth_solving=roma_max_depth_solving,
        roma_execution_mode=roma_execution_mode,
        roma_enable_checkpoints=roma_enable_checkpoints,
        roma_enable_logging=roma_enable_logging,
        mdap_enabled=mdap_enabled,
        mdap_k_ahead=mdap_k_ahead,
        mdap_max_samples=mdap_max_samples,
        mdap_enable_red_flagging=mdap_enable_red_flagging,
        mdap_max_token_length=mdap_max_token_length,
        mdap_min_confidence=mdap_min_confidence,
        apply_maker_to_roma_atomic=apply_maker_to_roma_atomic,
        apply_maker_to_roma_planning=apply_maker_to_roma_planning,
        aggregate_maker_results=aggregate_maker_results,
        enable_hierarchical_voting=enable_hierarchical_voting,
        enable_adaptive_k=enable_adaptive_k,
        enable_caching=enable_caching,
        cache_ttl_seconds=cache_ttl_seconds,
        cache_max_size=cache_max_size,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        fallback_policy=fallback_policy,
        use_associative_recomposition=use_associative_recomposition,
        associative_max_retries=associative_max_retries,
        associative_use_agentjson=associative_use_agentjson,
        enable_ground_truth=enable_ground_truth,
        ground_truth_storage_path=ground_truth_storage_path,
        apply_mdap_to_recomposed=apply_mdap_to_recomposed,
        enable_hierarchical_validation=enable_hierarchical_validation,
        use_evaluator_team=use_evaluator_team,
        evaluator_threshold=evaluator_threshold,
        evaluator_num_members=evaluator_num_members,
        use_gauntlet_system=use_gauntlet_system,
        gauntlet_difficulty=gauntlet_difficulty,
        max_refinement_attempts=max_refinement_attempts,
        min_acceptance_score=min_acceptance_score,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        metadata=metadata or {},
        **kwargs
    )


def solve_with_romamdapmaker_associative(
    problem: str,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[ROMAMDAPMakerAssociativeConfig] = None,
    llm_call_fn: Optional[Callable[[str], str]] = None,
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for ROMA-MDAP-MAKER + Associative problem solving.

    Args:
        problem: Problem statement
        context: Additional context
        config: Configuration (uses default if not provided)
        llm_call_fn: LLM call function
        recursive: Whether to use the recursive refinement loop

    Returns:
        Complete solution with all metadata
    """
    if config is None:
        from roma_mdap_maker_reliability_ssot import get_standard_config
        config = get_standard_config()

    engine = ROMAMDAPMakerAssociativeEngine(config)

    if recursive:
        return engine.solve_problem_recursive(
            problem=problem,
            context=context,
            llm_call_fn=llm_call_fn
        )
    else:
        return engine.solve_problem(
            problem=problem,
            context=context,
            llm_call_fn=llm_call_fn
        )



def get_romamdapmaker_associative_status() -> Dict[str, Any]:
    """
    Get ROMA-MDAP-MAKER + Associative system status.

    Returns:
        Dict with availability and configuration info
    """
    return {
        "roma_mdap_maker_available": ROMA_MDAP_MAKER_AVAILABLE,
        "associative_available": ASSOCIATIVE_AVAILABLE,
        "ground_truth_available": GROUND_TRUTH_AVAILABLE,
        "full_system_available": (
            ROMA_MDAP_MAKER_AVAILABLE and
            ASSOCIATIVE_AVAILABLE and
            GROUND_TRUTH_AVAILABLE
        ),
        "components": {
            "roma_mdap_maker": ROMA_MDAP_MAKER_AVAILABLE,
            "associative_recomposition": ASSOCIATIVE_AVAILABLE,
            "ground_truth_store": GROUND_TRUTH_AVAILABLE
        },
        "description": "ROMA hierarchical decomposition + Associative recomposition + MDAP multi-agent validation"
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ROMAMDAPMakerAssociativeConfig",
    "ROMAMDAPMakerAssociativeEngine",
    "create_romamdapmaker_associative_config",
    "solve_with_romamdapmaker_associative",
    "get_romamdapmaker_associative_status",
    "ROMA_MDAP_MAKER_AVAILABLE",
    "ASSOCIATIVE_AVAILABLE",
    "GROUND_TRUTH_AVAILABLE"
]
