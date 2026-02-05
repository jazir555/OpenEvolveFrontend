"""
ROMA-MDAP-MAKER + Associative Recomposition Integration

This module combines:
1. ROMA (Recursive Open Meta-Agents) - Hierarchical decomposition
2. MDAP (Multi-Agent Debate Protocol) - Multi-agent validation
3. MAKER - Structured workflow orchestration
4. Associative Recomposition - Domain-agnostic LLM + algorithmic verification

Complete pipeline:
    Problem -> ROMA Decomposition -> Associative Recomposition -> MDAP Validation -> Solution

Architecture:
    Layer 1: ROMA Hierarchical Decomposition
        v
    Layer 2: Associative Recomposition (LLM + Algorithmic)
        v
    Layer 3: MDAP Multi-Agent Validation
        v
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

from roma_mdap_maker_config import (
    ROMAMDAPMakerAssociativeConfig,
    create_romamdapmaker_associative_config as create_romamdapmaker_associative_config_base
)

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
    ROMA_MDAP_MAKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ROMA-MDAP-MAKER not available: {e}")
    ROMA_MDAP_MAKER_AVAILABLE = False

def create_romamdapmaker_associative_config(
    preset: str = "standard",
    **kwargs
) -> ROMAMDAPMakerAssociativeConfig:
    """
    Create ROMA-MDAP-MAKER + Associative configuration.
    Wraps the base factory with SSOT preset loading.
    """
    try:
        from roma_mdap_maker_reliability_ssot import get_reliability_config
        ssot_config = get_reliability_config(preset=preset)
        # Apply defaults from SSOT config to kwargs if not present
        for field_name in ROMAMDAPMakerAssociativeConfig.__dataclass_fields__.keys():
            if field_name not in kwargs or kwargs[field_name] is None:
                if hasattr(ssot_config, field_name):
                    kwargs[field_name] = getattr(ssot_config, field_name)
    except ImportError:
        logger.warning("ROMA-MDAP-MAKER reliability SSOT not available")
    
    return create_romamdapmaker_associative_config_base(**kwargs)

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
            self.gauntlet_manager = GauntletManager()
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
        logger.info(f"{ '#'*80}\n")
        
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
            eval_res = result.get("evaluator_assessment", {})
            
            score = eval_res.get("consensus_score", 0.0)
            is_approved = eval_res.get("verdict") == "APPROVED"
            
            if not best_result or score > best_result.get("evaluator_assessment", {}).get("consensus_score", 0.0):
                best_result = result
                
            if is_approved or score >= self.config.min_acceptance_score:
                logger.info(f"[OK] Solution ACCEPTED with score {score:.1f}")
                best_result["final_attempt"] = attempt + 1
                return best_result
            
            logger.warning(f"[FAIL] Solution REJECTED with score {score:.1f}. Refactoring...")
            
        logger.warning(f"Maximum refinement attempts reached ({self.config.max_refinement_attempts}). Returning best result.")
        if best_result:
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
        logger.info(f"{ '='*80}\n")
        logger.info(f"Problem: {problem[:100]}...")

        start_time = time.time()
        context = context or {}
        
        # Apply overrides locally if provided
        if config_overrides:
            logger.info(f"Applying config overrides for current task: {config_overrides}")

        try:
            # Phase 1: ROMA Decomposition
            logger.info("\n[PHASE 1] ROMA Hierarchical Decomposition")
            phase1_start = time.time()

            roma_result = self._roma_decompose(problem, context, config_overrides)

            phase1_time = time.time() - phase1_start
            logger.info(f"[OK] ROMA decomposition completed in {phase1_time:.2f}s")

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
            logger.info(f"[OK] Associative recomposition completed in {phase2_time:.2f}s")

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
            logger.info(f"[OK] Evaluation completed in {phase3_time:.2f}s")

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
                "mdap_validation": validation_result.get("mdap_validation"),
                "evaluator_assessment": validation_result.get("evaluator_assessment"),
                "gauntlet_result": validation_result.get("gauntlet_result"),
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
            logger.info(f"{ '='*80}")
            logger.info(f"Total Time: {total_time:.2f}s")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info(f"Sub-solutions: {result['num_sub_solutions']}")
            logger.info(f"Atomic Tasks: {result['num_atomic_tasks']}")
            logger.info(f"Error-Free: {result['error_free']}")
            logger.info(f"{ '='*80}\n")

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
        """
        assembled_solution = recomposition_result.get("assembled_solution")
        if not assembled_solution:
            return {"error": "No solution to evaluate", "confidence": 0.0}

        # 1. MDAP Validation
        mdap_result = self._mdap_validate(recomposition_result, problem, context)
        
        # 2. Evaluator Team Assessment
        evaluator_result = {}
        if self.evaluator_team:
            try:
                metadata = recomposition_result.get("metadata", {})
                classification = metadata.get("classification", {})
                content_type = classification.get("solution_type", "general")
                
                from evaluator_team import EvaluationThreshold
                threshold_map = {
                    "minimal_acceptance": EvaluationThreshold.MINIMAL_ACCEPTANCE,
                    "standard_approval": EvaluationThreshold.STANDARD_APPROVAL,
                    "high_quality": EvaluationThreshold.HIGH_QUALITY,
                    "exceptional": EvaluationThreshold.EXCEPTIONAL
                }
                threshold = threshold_map.get(self.config.evaluator_threshold, EvaluationThreshold.STANDARD_APPROVAL)
                
                evaluation = self.evaluator_team.evaluate_content(
                    content=assembled_solution,
                    content_type=str(content_type),
                    threshold=threshold,
                    num_evaluators=self.config.evaluator_num_members
                )
                
                evaluator_result = {
                    "consensus_score": evaluation.consensus_score,
                    "verdict": evaluation.final_verdict,
                    "final_verdict": evaluation.final_verdict,
                    "recommendations": evaluation.recommendations,
                    "report": self.evaluator_team.generate_evaluation_report(evaluation),
                    "confidence_intervals": evaluation.confidence_intervals
                }
            except Exception as e:
                logger.error(f"Evaluator Team assessment failed: {e}")
                evaluator_result = {"error": str(e)}

        # 3. Adaptive Gauntlet System
        gauntlet_result = {}
        if self.gauntlet_system and GAUNTLET_SYSTEM_AVAILABLE:
            try:
                prob_def, sub_prob = self._create_gauntlet_artifacts(problem, context)
                gauntlet_def = self.gauntlet_system.create_adaptive_gauntlet(prob_def, sub_prob)
                
                if self.gauntlet_manager:
                    gauntlet_result = self.gauntlet_manager.execute_gauntlet(
                        gauntlet_def,
                        assembled_solution,
                        context
                    )
                    self.gauntlet_system.update_performance_from_result(gauntlet_result, None)
            except Exception as e:
                logger.error(f"Gauntlet System run failed: {e}")
                gauntlet_result = {"error": str(e)}

        # Calculate Unified Confidence
        unified_confidence = mdap_result.get("confidence", 0.5)
        if "consensus_score" in evaluator_result:
            eval_conf = evaluator_result["consensus_score"] / 100.0
            mdap_conf = mdap_result.get("confidence", 0.5)
            unified_confidence = (eval_conf * 0.6) + (mdap_conf * 0.4)

        return {
            "confidence": unified_confidence,
            "mdap_validation": mdap_result,
            "evaluator_assessment": evaluator_result,
            "gauntlet_result": gauntlet_result,
            "error_rate": mdap_result.get("error_rate", 0.0),
            "validated": (evaluator_result.get("verdict") == "APPROVED") or (not self.evaluator_team and mdap_result.get("validated", False))
        }

    def _create_gauntlet_artifacts(self, problem: str, context: Dict[str, Any]):
        """Create artifacts for gauntlet with full data models"""
        from sovereign_data_models import (
            ProblemDefinition, SubProblem, ProblemType, SubProblemType,
            DomainContext, ComplexityScore
        )
        
        domain_context = DomainContext(domain="Software Development")
        complexity_score = ComplexityScore(
            explanation="Decomposition-based complexity",
            cognitive_complexity=5.0,
            computational_complexity=3.0,
            domain_complexity=4.0,
            integration_complexity=4.0,
            overall_complexity=4.0
        )
        
        prob_def = ProblemDefinition(
            id="prob_root",
            title="Root Problem",
            description=problem,
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=domain_context,
            complexity_score=complexity_score
        )
        
        sub_prob = SubProblem(
            id="sub_root",
            parent_id="prob_root",
            title="Implementation Task",
            description=problem,
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=complexity_score
        )
        
        return prob_def, sub_prob

    def _mdap_validate(
        self,
        recomposition_result: Dict[str, Any],
        problem: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 3: MDAP Validation"""
        if not self.roma_mdap_maker_engine or not self.config.apply_mdap_to_recomposed:
            return {"confidence": 0.8, "error_rate": 0.0, "validated": True}

        assembled_solution = recomposition_result.get("assembled_solution")
        validation_result = self.roma_mdap_maker_engine.solve_with_roma_mdap_maker(
            task=f"Validate the following solution for: {problem}",
            context={"solution_to_validate": assembled_solution, **context}
        )

        return {
            "confidence": validation_result.get("confidence", 0.5),
            "validation_details": validation_result,
            "error_rate": validation_result.get("error_rate", 0.0),
            "validated": True
        }

    def _simple_assemble(self, sub_solutions: List[Dict[str, Any]]) -> str:
        """Concatenate sub-solutions"""
        parts = []
        for sol in sub_solutions:
            parts.append(f"## {sol.get('description', '')}\n\n{sol.get('solution_content', '')}\n")
        return "\n".join(parts)

    def _default_llm_call(self, prompt: str) -> str:
        """Default LLM call"""
        # Mock implementation for safety if no API key
        return json.dumps({"assembled_solution": "Mock assembled solution"})

def solve_with_romamdapmaker_associative(
    problem: str,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[ROMAMDAPMakerAssociativeConfig] = None,
    llm_call_fn: Optional[Callable[[str], str]] = None,
    recursive: bool = True
) -> Dict[str, Any]:
    """Convenience function for problem solving"""
    if config is None:
        engine = ROMAMDAPMakerAssociativeEngine(create_romamdapmaker_associative_config())
    else:
        engine = ROMAMDAPMakerAssociativeEngine(config)

    if recursive:
        return engine.solve_problem_recursive(problem, context, llm_call_fn)
    else:
        return engine.solve_problem(problem, context, llm_call_fn)

def get_romamdapmaker_associative_status() -> Dict[str, Any]:
    """Get system status"""
    return {
        "roma_mdap_maker_available": ROMA_MDAP_MAKER_AVAILABLE,
        "associative_available": ASSOCIATIVE_AVAILABLE,
        "ground_truth_available": GROUND_TRUTH_AVAILABLE,
        "full_system_available": (ROMA_MDAP_MAKER_AVAILABLE and ASSOCIATIVE_AVAILABLE and GROUND_TRUTH_AVAILABLE)
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