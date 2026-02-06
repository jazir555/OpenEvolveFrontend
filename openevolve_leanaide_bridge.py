#!/usr/bin/env python3
"""
OpenEvolve-LeanAIDE Bridge for Workflow Autoformalization

This module provides a comprehensive bridge between OpenEvolve's decomposition and evolution systems
and LeanAIDE's autoformalization capabilities. It enables automatic formalization of mathematical
problems within OpenEvolve workflows, supporting the full evolution lifecycle.

Key Features:
- Auto-detection of mathematical problems in OpenEvolve workflows
- Integration with LeanAIDE autoformalization engine
- Support for MDAP and MAKER workflows
- Formal verification of evolved solutions
- Confidence-based decision making
- Comprehensive error handling and fallback mechanisms

Author: OpenEvolve
Created: 2026-01-01
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import json
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Import OpenEvolve workflow structures
try:
    from workflow_structures import (
        MathematicalDomain, VerificationMethod, WorkflowState, SubProblem,
        SolutionAttempt, VerificationReport, DecompositionPlan
    )
    from workflow_engine import WorkflowEngine
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    logger.warning("OpenEvolve workflow structures not available - using fallback types")
    OPENEVOLVE_AVAILABLE = False
    
    # Fallback types for when OpenEvolve is not available
    class MathematicalDomain(Enum):
        ALGEBRA = "algebra"
        ANALYSIS = "analysis"
        LOGIC = "logic"
        GENERAL = "general"
    
    class VerificationMethod(Enum):
        LEANAIDE_FORMAL = "leanaide_formal"
        STANDARD_GAUNTLET = "standard_gauntlet"
        HYBRID = "hybrid"

# Import LeanAIDE autoformalization engine
try:
    from leanaide_autoformalization_mdap_maker import (
        LeanAideAutoformalizationEngine, AutoformalizationStrategy
    )
    from leanaide_client import LeanAideClient, LeanAideConfig
    LEANAIDE_AUTOFORMALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("LeanAIDE autoformalization engine not available")
    LEANAIDE_AUTOFORMALIZATION_AVAILABLE = False

    # Fallback AutoformalizationStrategy if not available
    class AutoformalizationStrategy(Enum):
        """Fallback autoformalization strategy enum."""
        ADAPTIVE = "adaptive"
        CONSERVATIVE = "conservative"
        AGGRESSIVE = "aggressive"

# Import LeanAIDE workflow integration
try:
    from leanaide_workflow_integration import (
        LeanAideWorkflowConfig, LeanAideVerificationResult
    )
    LEANAIDE_WORKFLOW_AVAILABLE = True
except ImportError:
    logger.warning("LeanAIDE workflow integration not available")
    LEANAIDE_WORKFLOW_AVAILABLE = False

    # Fallback types if not available
    class LeanAideWorkflowConfig:
        """Fallback workflow config."""
        pass

    class LeanAideVerificationResult:
        """Fallback verification result."""
        pass

    class LeanAideConfig:
        """Fallback LeanAIDE config."""
        pass

class AutoformalizationStage(Enum):
    """Stages where autoformalization can be applied in OpenEvolve workflows."""
    DECOMPOSITION = "decomposition"  # During initial problem decomposition
    SUB_PROBLEM_GENERATION = "sub_problem_generation"  # When generating sub-problems
    SOLUTION_ATTEMPT = "solution_attempt"  # When creating solution attempts
    VERIFICATION = "verification"  # During verification stages
    EVOLUTION = "evolution"  # During evolutionary refinement
    FINAL_INTEGRATION = "final_integration"  # During final solution integration


@dataclass
class OpenEvolveLeanAideConfig:
    """Configuration for OpenEvolve-LeanAIDE bridge."""
    # Autoformalization settings
    autoformalization_enabled: bool = True
    auto_detect_math_problems: bool = True
    default_strategy: AutoformalizationStrategy = AutoformalizationStrategy.ADAPTIVE
    
    # Confidence thresholds
    min_confidence_for_autoformalization: float = 0.6
    min_confidence_for_verification: float = 0.8
    
    # Integration settings
    integrate_with_decomposition: bool = True
    integrate_with_evolution: bool = True
    integrate_with_verification: bool = True
    
    # Performance settings
    max_autoformalization_time: float = 120.0  # seconds
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Fallback settings
    fallback_to_standard_workflow: bool = True
    hybrid_verification_enabled: bool = True
    
    # Domain-specific settings
    domain_specific_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # LeanAIDE client configuration
    leanaide_config: Optional["LeanAideConfig"] = None


@dataclass
class AutoformalizationResult:
    """Result from autoformalization of an OpenEvolve workflow component."""
    success: bool
    original_problem: str
    formalized_problem: Optional[str] = None
    lean_code: Optional[str] = None
    confidence_score: float = 0.0
    strategy_used: Optional[str] = None
    mathematical_domain: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "original_problem": self.original_problem,
            "formalized_problem": self.formalized_problem,
            "lean_code": self.lean_code,
            "confidence_score": self.confidence_score,
            "strategy_used": self.strategy_used,
            "mathematical_domain": self.mathematical_domain,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


class OpenEvolveLeanAideBridge:
    """Main bridge class connecting OpenEvolve workflows with LeanAIDE autoformalization."""
    
    def __init__(self, config: Optional[OpenEvolveLeanAideConfig] = None):
        """Initialize the bridge."""
        self.config = config or OpenEvolveLeanAideConfig()
        self.autoformalization_engine = None
        self.cache = {}
        self.setup_engine()
        
    def setup_engine(self):
        """Setup the LeanAIDE autoformalization engine."""
        if not LEANAIDE_AUTOFORMALIZATION_AVAILABLE:
            logger.warning("LeanAIDE autoformalization engine not available")
            return
            
        try:
            # Initialize LeanAIDE client
            leanaide_config = self.config.leanaide_config or LeanAideConfig()
            leanaide_client = LeanAideClient(leanaide_config)
            
            # Initialize autoformalization engine
            self.autoformalization_engine = LeanAideAutoformalizationEngine(
                leanaide_client=leanaide_client,
                enable_caching=self.config.enable_caching,
                cache_ttl_seconds=self.config.cache_ttl_seconds
            )
            
            logger.info("LeanAIDE autoformalization engine initialized successfully")
            
        except (ImportError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Failed to initialize LeanAIDE engine: {e}")
            self.autoformalization_engine = None

    def is_mathematical_problem(self, problem_text: str) -> bool:
        """Detect if a problem is mathematical in nature."""
        if not problem_text:
            return False
            
        # Check for mathematical keywords
        math_keywords = [
            "prove", "theorem", "lemma", "proof", "mathematical",
            "equation", "inequality", "function", "integral", "derivative",
            "limit", "series", "convergence", "divergence", "continuity",
            "differentiability", "optimization", "algorithm", "complexity",
            "graph theory", "number theory", "combinatorics", "probability",
            "algebra", "analysis", "topology", "geometry", "logic"
        ]
        
        text_lower = problem_text.lower()
        for keyword in math_keywords:
            if keyword in text_lower:
                return True
                
        # Check for mathematical symbols
        math_symbols = ["∀", "∃", "∈", "∉", "⊆", "⊂", "⊇", "⊃", "->", "⇒", "⇔", "∧", "∨", "¬", "∑", "∏", "∫", "∮", "∇", "∂"]
        for symbol in math_symbols:
            if symbol in problem_text:
                return True
                
        # Check for common mathematical expressions
        math_patterns = [
            r'\\forall|\\exists|\\in|\notin|\\subset|\\subseteq',
            r'\\sum|\\prod|\\int|\\iint|\\iiint',
            r'\\frac|\\sqrt|\\log|\\ln|\\exp',
            r'\\sin|\\cos|\\tan|\\cot|\\sec|\\csc',
            r'\\lim|\\infty|\nabla|\\partial',
            r'[A-Za-z]+\s*\d+\s*[+\-*/]\s*\d+',
            r'\d+\s*[+\-*/]\s*[A-Za-z]+'
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, problem_text):
                return True
                
        return False

    def detect_mathematical_domain(self, problem_text: str) -> Optional[str]:
        """Detect the mathematical domain of a problem."""
        if not self.is_mathematical_problem(problem_text):
            return None
            
        text_lower = problem_text.lower()
        
        # Domain detection rules
        domain_rules = {
            "algebra": ["group", "ring", "field", "vector space", "homomorphism", "isomorphism"],
            "analysis": ["continuous", "differentiable", "integral", "derivative", "limit", "convergence"],
            "topology": ["topological", "compact", "connected", "hausdorff", "metric space"],
            "number_theory": ["prime", "divisible", "gcd", "lcm", "modular", "fermat"],
            "combinatorics": ["permutation", "combination", "graph", "tree", "vertex", "edge"],
            "geometry": ["triangle", "circle", "polygon", "euclidean", "distance", "angle"],
            "logic": ["proposition", "predicate", "tautology", "contradiction", "proof theory"],
            "set_theory": ["set", "cardinality", "ordinal", "zfc", "axiom of choice"],
            "category_theory": ["category", "functor", "natural transformation", "adjunction"],
            "linear_algebra": ["matrix", "determinant", "eigenvalue", "eigenvector", "vector space"],
            "calculus": ["derivative", "integral", "differentiation", "integration", "taylor series"],
            "probability": ["probability", "random variable", "distribution", "expectation", "variance"]
        }
        
        for domain, keywords in domain_rules.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return domain
                    
        return "general"

    def generate_cache_key(self, problem_text: str, strategy: str = "adaptive") -> str:
        """Generate a cache key for autoformalization results."""
        cache_key = f"{problem_text}:{strategy}"
        return hashlib.md5(cache_key.encode('utf-8')).hexdigest()

    async def autoformalize_problem(
        self,
        problem_text: str,
        strategy: Optional[AutoformalizationStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AutoformalizationResult:
        """Autoformalize a problem using LeanAIDE."""
        start_time = time.time()
        result = AutoformalizationResult(
            success=False,
            original_problem=problem_text,
            confidence_score=0.0
        )
        
        # Check if autoformalization is enabled
        if not self.config.autoformalization_enabled:
            result.errors.append("Autoformalization is disabled in configuration")
            return result
            
        # Check if problem is mathematical
        if not self.is_mathematical_problem(problem_text):
            result.errors.append("Problem does not appear to be mathematical")
            return result
            
        # Use provided strategy or default
        strategy = strategy or self.config.default_strategy
        
        # Check cache
        if self.config.enable_caching:
            cache_key = self.generate_cache_key(problem_text, strategy.name)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                cached_result.execution_time = time.time() - start_time
                return cached_result
                
        # Check if engine is available
        if not self.autoformalization_engine:
            result.errors.append("LeanAIDE autoformalization engine not available")
            return result
            
        try:
            # Perform autoformalization
            autoformalization_result = await self.autoformalization_engine.autoformalize(
                natural_language=problem_text,
                strategy=strategy,
                context=context or {}
            )
            
            # Process the result
            result.success = autoformalization_result.success
            result.formalized_problem = autoformalization_result.formalized_statement
            result.lean_code = autoformalization_result.lean_code
            result.confidence_score = autoformalization_result.confidence_score
            result.strategy_used = strategy.name
            result.mathematical_domain = self.detect_mathematical_domain(problem_text)
            result.execution_time = time.time() - start_time
            
            # Add metadata
            result.metadata = {
                "strategy": strategy.name,
                "domain": result.mathematical_domain,
                "autoformalization_engine": "LeanAIDE",
                "timestamp": time.time()
            }
            
            # Cache the result
            if self.config.enable_caching and result.success:
                self.cache[self.generate_cache_key(problem_text, strategy.name)] = result
                
            return result
            
        except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
            result.errors.append(f"Autoformalization failed: {str(e)}")
            logger.error(f"Autoformalization error: {e}")
            return result

    async def autoformalize_subproblem(
        self,
        subproblem: Union[SubProblem, Dict[str, Any]],
        workflow_state: Optional[Dict[str, Any]] = None
    ) -> AutoformalizationResult:
        """Autoformalize a subproblem from OpenEvolve workflow."""
        if isinstance(subproblem, dict):
            problem_text = subproblem.get('description', '') or subproblem.get('problem_statement', '')
        else:
            problem_text = subproblem.description or subproblem.problem_statement
            
        # Use adaptive strategy for subproblems
        return await self.autoformalize_problem(
            problem_text=problem_text,
            strategy=AutoformalizationStrategy.ADAPTIVE,
            context={
                "workflow_stage": "subproblem_autoformalization",
                "workflow_state": workflow_state or {}
            }
        )

    async def autoformalize_solution_attempt(
        self,
        solution_attempt: Union[SolutionAttempt, Dict[str, Any]],
        original_problem: Optional[str] = None
    ) -> AutoformalizationResult:
        """Autoformalize a solution attempt."""
        if isinstance(solution_attempt, dict):
            solution_text = solution_attempt.get('solution', '') or solution_attempt.get('content', '')
        else:
            solution_text = solution_attempt.solution or solution_attempt.content
            
        problem_text = original_problem or solution_text
        
        return await self.autoformalize_problem(
            problem_text=problem_text,
            strategy=AutoformalizationStrategy.ADAPTIVE,
            context={
                "workflow_stage": "solution_attempt_autoformalization",
                "solution_type": "evolved"
            }
        )

    async def verify_formalized_solution(
        self,
        problem_text: str,
        lean_code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> LeanAideVerificationResult:
        """Verify a formalized solution using LeanAIDE."""
        if not LEANAIDE_WORKFLOW_AVAILABLE:
            raise ImportError("LeanAIDE workflow integration not available")
            
        from leanaide_workflow_integration import verify_with_leanaide
        
        return await verify_with_leanaide(
            problem_text=problem_text,
            lean_code=lean_code,
            context=context or {}
        )

    async def integrate_with_decomposition(
        self,
        decomposition_plan: DecompositionPlan,
        workflow_state: Dict[str, Any]
    ) -> DecompositionPlan:
        """Integrate autoformalization with decomposition process."""
        if not self.config.integrate_with_decomposition:
            return decomposition_plan
            
        # Autoformalize each subproblem
        autoformalized_subproblems = []
        
        for subproblem in decomposition_plan.subproblems:
            autoformalization_result = await self.autoformalize_subproblem(
                subproblem=subproblem,
                workflow_state=workflow_state
            )
            
            if autoformalization_result.success:
                # Add formalized version to subproblem metadata
                subproblem.metadata = subproblem.metadata or {}
                subproblem.metadata['autoformalization'] = {
                    'formalized_problem': autoformalization_result.formalized_problem,
                    'lean_code': autoformalization_result.lean_code,
                    'confidence_score': autoformalization_result.confidence_score,
                    'mathematical_domain': autoformalization_result.mathematical_domain
                }
                
                # Add verification method
                if subproblem.metadata.get('verification_methods'):
                    subproblem.metadata['verification_methods'].append('leanaide_formal')
                else:
                    subproblem.metadata['verification_methods'] = ['leanaide_formal']
                    
            autoformalized_subproblems.append(subproblem)
            
        # Create new decomposition plan with autoformalized subproblems
        return dataclasses.replace(
            decomposition_plan,
            subproblems=autoformalized_subproblems
        )

    async def integrate_with_evolution(
        self,
        solution_attempt: SolutionAttempt,
        workflow_state: Dict[str, Any]
    ) -> SolutionAttempt:
        """Integrate autoformalization with evolutionary process."""
        if not self.config.integrate_with_evolution:
            return solution_attempt
            
        # Autoformalize the evolved solution
        autoformalization_result = await self.autoformalize_solution_attempt(
            solution_attempt=solution_attempt,
            original_problem=workflow_state.get('original_problem')
        )
        
        if autoformalization_result.success:
            # Add formalized version to solution metadata
            solution_attempt.metadata = solution_attempt.metadata or {}
            solution_attempt.metadata['autoformalization'] = {
                'formalized_solution': autoformalization_result.formalized_problem,
                'lean_code': autoformalization_result.lean_code,
                'confidence_score': autoformalization_result.confidence_score,
                'mathematical_domain': autoformalization_result.mathematical_domain,
                'verification_status': 'pending'
            }
            
            # Verify the formalized solution
            if autoformalization_result.lean_code:
                verification_result = await self.verify_formalized_solution(
                    problem_text=autoformalization_result.original_problem,
                    lean_code=autoformalization_result.lean_code,
                    context={
                        'workflow_stage': 'evolution_verification',
                        'solution_id': getattr(solution_attempt, 'id', 'unknown')
                    }
                )

                solution_attempt.metadata['autoformalization']['verification_status'] = 'verified' if verification_result.success else 'failed'
                solution_attempt.metadata['autoformalization']['verification_confidence'] = verification_result.confidence_score
                    
        return solution_attempt

    async def integrate_with_verification(
        self,
        verification_report: VerificationReport,
        workflow_state: Dict[str, Any]
    ) -> VerificationReport:
        """Integrate autoformalization with verification process."""
        if not self.config.integrate_with_verification:
            return verification_report
            
        # Check if this is a mathematical problem that can benefit from formal verification
        original_problem = workflow_state.get('original_problem', '')
        if not self.is_mathematical_problem(original_problem):
            return verification_report
            
        # Get the solution that was verified
        solution_attempt = workflow_state.get('solution_attempt')
        if not solution_attempt:
            return verification_report
            
        # Check if we have autoformalization metadata
        autoformalization_data = None
        if isinstance(solution_attempt, dict):
            autoformalization_data = solution_attempt.get('metadata', {}).get('autoformalization')
        else:
            autoformalization_data = getattr(solution_attempt, 'metadata', {}).get('autoformalization')
            
        if autoformalization_data and autoformalization_data.get('lean_code'):
            # Perform formal verification
            verification_result = await self.verify_formalized_solution(
                problem_text=original_problem,
                lean_code=autoformalization_data['lean_code']
            )
            
            # Update verification report
            verification_report.leanaide_verification = {
                'success': verification_result.success,
                'confidence_score': verification_result.confidence_score,
                'formal_proof': verification_result.formal_proof,
                'errors': verification_result.errors,
                'warnings': verification_result.warnings
            }
            
            # Update overall verification status
            if verification_result.success and verification_result.confidence_score >= self.config.min_confidence_for_verification:
                verification_report.is_approved = True
                verification_report.confidence_score = max(
                    verification_report.confidence_score,
                    verification_result.confidence_score
                )
                verification_report.verification_methods.append('leanaide_formal')
            else:
                verification_report.is_approved = False
                verification_report.verification_methods.append('leanaide_formal_failed')
                
        return verification_report

    def get_autoformalization_strategy_recommendation(
        self,
        problem_text: str,
        workflow_stage: str
    ) -> AutoformalizationStrategy:
        """Recommend the best autoformalization strategy for a given problem and stage."""
        domain = self.detect_mathematical_domain(problem_text)
        
        # Strategy recommendations based on domain and stage
        recommendations = {
            'decomposition': {
                'algebra': AutoformalizationStrategy.MDAP,
                'analysis': AutoformalizationStrategy.HYBRID,
                'logic': AutoformalizationStrategy.DIRECT,
                'general': AutoformalizationStrategy.ADAPTIVE
            },
            'sub_problem_generation': {
                'algebra': AutoformalizationStrategy.HYBRID,
                'analysis': AutoformalizationStrategy.MDAP,
                'logic': AutoformalizationStrategy.DIRECT,
                'general': AutoformalizationStrategy.ADAPTIVE
            },
            'solution_attempt': {
                'algebra': AutoformalizationStrategy.HYBRID,
                'analysis': AutoformalizationStrategy.MAKER,
                'logic': AutoformalizationStrategy.DIRECT,
                'general': AutoformalizationStrategy.ADAPTIVE
            },
            'verification': {
                'algebra': AutoformalizationStrategy.MAKER,
                'analysis': AutoformalizationStrategy.HYBRID,
                'logic': AutoformalizationStrategy.DIRECT,
                'general': AutoformalizationStrategy.ADAPTIVE
            }
        }
        
        stage_recommendations = recommendations.get(workflow_stage, {})
        domain_recommendation = stage_recommendations.get(domain, AutoformalizationStrategy.ADAPTIVE)
        
        return domain_recommendation

    async def autoformalize_workflow_stage(
        self,
        workflow_state: Dict[str, Any],
        stage: AutoformalizationStage
    ) -> Dict[str, Any]:
        """Autoformalize a specific workflow stage."""
        stage_methods = {
            AutoformalizationStage.DECOMPOSITION: self.integrate_with_decomposition,
            AutoformalizationStage.SUB_PROBLEM_GENERATION: self.autoformalize_subproblem,
            AutoformalizationStage.SOLUTION_ATTEMPT: self.autoformalize_solution_attempt,
            AutoformalizationStage.VERIFICATION: self.integrate_with_verification,
            AutoformalizationStage.EVOLUTION: self.integrate_with_evolution,
            AutoformalizationStage.FINAL_INTEGRATION: self.autoformalize_problem
        }
        
        method = stage_methods.get(stage)
        if method:
            # Get the appropriate data for this stage
            stage_data = workflow_state.get('stage_data')
            if stage_data:
                result = await method(stage_data, workflow_state)
                workflow_state['autoformalization_results'] = workflow_state.get('autoformalization_results', {})
                workflow_state['autoformalization_results'][stage.value] = result
                
        return workflow_state

    def create_autoformalization_report(
        self,
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a comprehensive autoformalization report."""
        autoformalization_results = workflow_state.get('autoformalization_results', {})
        
        report = {
            'workflow_id': workflow_state.get('workflow_id', 'unknown'),
            'original_problem': workflow_state.get('original_problem', ''),
            'stages_processed': list(autoformalization_results.keys()),
            'successful_stages': [],
            'failed_stages': [],
            'overall_success': False,
            'highest_confidence': 0.0,
            'mathematical_domains': set(),
            'strategies_used': set(),
            'execution_times': {},
            'errors': [],
            'warnings': []
        }
        
        for stage, result in autoformalization_results.items():
            if isinstance(result, AutoformalizationResult):
                if result.success:
                    report['successful_stages'].append(stage)
                    report['highest_confidence'] = max(report['highest_confidence'], result.confidence_score)
                    if result.mathematical_domain:
                        report['mathematical_domains'].add(result.mathematical_domain)
                    if result.strategy_used:
                        report['strategies_used'].add(result.strategy_used)
                    report['execution_times'][stage] = result.execution_time
                else:
                    report['failed_stages'].append(stage)
                    report['errors'].extend(result.errors)
                    report['warnings'].extend(result.warnings)
            elif isinstance(result, dict) and result.get('is_approved') is True:
                report['successful_stages'].append(stage)
            else:
                report['failed_stages'].append(stage)
                
        report['overall_success'] = len(report['failed_stages']) == 0
        report['mathematical_domains'] = list(report['mathematical_domains'])
        report['strategies_used'] = list(report['strategies_used'])
        
        return report


# Global bridge instance
_openevolve_leanaide_bridge = None


def get_openevolve_leanaide_bridge(config: Optional[OpenEvolveLeanAideConfig] = None) -> OpenEvolveLeanAideBridge:
    """Get the global OpenEvolve-LeanAIDE bridge instance."""
    global _openevolve_leanaide_bridge
    if not _openevolve_leanaide_bridge:
        _openevolve_leanaide_bridge = OpenEvolveLeanAideBridge(config)
    return _openevolve_leanaide_bridge


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Initialize the bridge
        bridge = get_openevolve_leanaide_bridge()
        
        # Example problem
        problem = "Prove that for all natural numbers n, the sum of the first n odd numbers is n²"
        
        # Autoformalize the problem
        result = await bridge.autoformalize_problem(problem)
        
        print(f"Autoformalization Result:")
        print(f"Success: {result.success}")
        print(f"Confidence: {result.confidence_score}")
        print(f"Domain: {result.mathematical_domain}")
        print(f"Lean Code: {result.lean_code}")
        print(f"Strategy: {result.strategy_used}")
        
    asyncio.run(main())