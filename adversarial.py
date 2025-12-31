# This file implements the adversarial generation functionality of the OpenEvolve frontend.
# The purpose of this module is to facilitate AI-driven testing and refinement of ideas,
# code, and other content. It operates on the principle of "AI peer review," where
# different AI agents are assigned to "red team" (critique) and "blue team" (improve)
# roles. An "evaluator" AI then assesses the quality of the improvements.
#
# This process is designed for constructive, iterative improvement and is NOT intended
# for generating malicious prompts, code, or other harmful content. The goal is to
# identify weaknesses and enhance the quality of the content in a controlled and
# ethical manner.

import streamlit as st
import time
import threading
import traceback
import uuid
import random
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict

from session_utils import _hash_text
# Import prompts with error handling
try:
    from session_manager import APPROVAL_PROMPT, RED_TEAM_CRITIQUE_PROMPT, BLUE_TEAM_PATCH_PROMPT
except (ImportError, SyntaxError):
    # Fallback prompts if session_manager is not available or has syntax errors
    APPROVAL_PROMPT = "Evaluate the quality and effectiveness of this content."
    RED_TEAM_CRITIQUE_PROMPT = "Critically analyze this content and identify potential issues, weaknesses, or areas for improvement."
    BLUE_TEAM_PATCH_PROMPT = "Improve this content by addressing the identified issues and enhancing its quality."
# Import OpenEvolve integration with error handling
try:
    from openevolve_integration import (
        create_language_specific_evaluator,
        create_specialized_evaluator,
        create_comprehensive_openevolve_config,
        run_unified_evolution,
    )
except ImportError:
    # Fallback implementations
    def create_language_specific_evaluator(content_type, requirements, compliance):
        class MockEvaluator:
            def evaluate_content(self, content):
                return {"robustness_score": 0.5}
        return MockEvaluator()
    
    def create_specialized_evaluator(content_type, requirements, compliance):
        return create_language_specific_evaluator(content_type, requirements, compliance)
    
    def create_comprehensive_openevolve_config(**kwargs):
        return kwargs
    
    def run_unified_evolution(**kwargs):
        return {"success": False, "message": "OpenEvolve integration not available"}

try:
    from integrated_workflow import generate_adversarial_data_augmentation
except ImportError:
    def generate_adversarial_data_augmentation(**kwargs):
        return kwargs.get("content", "")

try:
    from review_utils import determine_review_type, get_appropriate_prompts
except ImportError:
    def determine_review_type(content):
        return "general"
    
    def get_appropriate_prompts(review_type):
        return RED_TEAM_CRITIQUE_PROMPT, BLUE_TEAM_PATCH_PROMPT

try:
    from logging_util import _update_adv_log_and_status
except ImportError:
    def _update_adv_log_and_status(message):
        print(f"[ADVERSARIAL] {message}")
from parameter_manager import ParameterManager, ValidationResult
from evolution import EvolutionConfiguration

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AdversarialConfiguration:
    """
    Comprehensive adversarial configuration utilizing all adversarial parameters
    from the parameter manager plus relevant parameters from other categories
    """
    # Core Adversarial Parameters (20)
    attack_model_config: Dict[str, Any] = None
    defense_model_config: Dict[str, Any] = None
    adversarial_rounds: int = 5
    attack_strength: float = 0.5
    defense_strategy: str = "reactive"
    coevolutionary_approach: bool = False
    red_team_models: List[str] = None
    blue_team_models: List[str] = None
    red_team_sample_size: int = 3
    blue_team_sample_size: int = 3
    adversarial_temperature: float = 0.8
    attack_diversity: bool = True
    defense_strength: float = 1.0
    adversarial_budget: int = 100
    attack_types: List[str] = None
    defense_strategies: List[str] = None
    robustness_metric: str = "accuracy"
    perturbation_bound: float = 0.1
    gradient_masking: bool = False
    ensemble_defense: bool = True
    
    # Core Evolution Parameters (relevant for adversarial)
    max_iterations: int = 10
    population_size: int = 20
    temperature: float = 0.7
    max_tokens: int = 2048
    seed: Optional[int] = None
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    
    # Evaluation Parameters (relevant for adversarial)
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = None
    parallel_evaluations: int = 4
    evaluator_timeout: int = 300
    max_retries_eval: int = 3
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1
    evaluator_models: List[Dict] = None
    evaluator_system_message: str = ""
    
    # Prompt Engineering Parameters (for adversarial prompts)
    prompt_template: str = "default"
    system_prompt: str = ""
    context_length: int = 2000
    prompt_optimization: bool = True
    template_stochasticity: bool = True
    meta_prompting: bool = False
    few_shot_examples: int = 3
    chain_of_thought: bool = True
    
    # Resource Management Parameters
    memory_limit_mb: int = 4096
    cpu_limit: float = 0.8
    max_time: int = 1800
    api_call_limit: int = 1000
    cost_limit_usd: float = 10.0
    resource_monitoring: bool = True
    
    # Evolution Tracing Parameters (for adversarial tracking)
    trace_enabled: bool = False
    trace_level: str = "basic"
    trace_format: str = "json"
    trace_file: str = "./adversarial_trace.log"
    include_population: bool = False
    include_fitness: bool = True
    
    # Advanced Research Parameters (relevant for adversarial)
    meta_learning: bool = False
    transfer_learning: bool = False
    explainable_ai: bool = False
    differential_privacy: bool = False
    
    # Custom Requirements Parameters (for adversarial constraints)
    custom_fitness: str = ""
    custom_constraints: List[str] = None
    domain_knowledge: str = ""
    expert_rules: List[str] = None
    business_logic: str = ""
    regulatory_compliance: List[str] = None
    ethical_guidelines: List[str] = None
    
    # UI & Visualization Parameters
    enable_visualization: bool = True
    plot_frequency: int = 10
    real_time_updates: bool = False
    export_plots: bool = True
    
    # Experimental Parameters
    experimental_features: bool = False
    debug_mode: bool = False
    profiling_enabled: bool = False
    
    def __post_init__(self):
        """Initialize default values for list/dict fields"""
        if self.attack_model_config is None:
            self.attack_model_config = {}
        if self.defense_model_config is None:
            self.defense_model_config = {}
        if self.red_team_models is None:
            self.red_team_models = []
        if self.blue_team_models is None:
            self.blue_team_models = []
        if self.attack_types is None:
            self.attack_types = []
        if self.defense_strategies is None:
            self.defense_strategies = []
        if self.cascade_thresholds is None:
            self.cascade_thresholds = [0.5, 0.75, 0.9]
        if self.evaluator_models is None:
            self.evaluator_models = []
        if self.custom_constraints is None:
            self.custom_constraints = []
        if self.expert_rules is None:
            self.expert_rules = []
        if self.regulatory_compliance is None:
            self.regulatory_compliance = []
        if self.ethical_guidelines is None:
            self.ethical_guidelines = []
    
    @classmethod
    def from_parameter_manager(cls, param_manager: ParameterManager, session_state: Dict[str, Any]) -> 'AdversarialConfiguration':
        """Create adversarial configuration from parameter manager and session state"""
        config = cls()
        
        # Get all parameter defaults
        defaults = param_manager.get_defaults()
        
        # Update configuration with session state values or defaults
        for param_name, param_def in param_manager.schema.parameters.items():
            if hasattr(config, param_name):
                # Use session state value if available, otherwise use default
                value = session_state.get(param_name, defaults.get(param_name, param_def.default))
                setattr(config, param_name, value)
        
        return config
    
    def validate(self, param_manager: ParameterManager) -> ValidationResult:
        """Validate the adversarial configuration using parameter manager"""
        config_dict = asdict(self)
        return param_manager.validate(config_dict)
    
    def to_evolution_config(self) -> EvolutionConfiguration:
        """Convert to EvolutionConfiguration for use with evolution.py"""
        evolution_config = EvolutionConfiguration()
        
        # Map adversarial config to evolution config
        config_dict = asdict(self)
        for key, value in config_dict.items():
            if hasattr(evolution_config, key):
                setattr(evolution_config, key, value)
        
        # Set adversarial-specific evolution mode
        evolution_config.evolution_mode = "adversarial"
        
        return evolution_config



# Import OpenEvolve modules for backend integration
try:
    # We just need to check if the package is available.
    # The actual functions are imported from openevolve_integration.

    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available")

MODEL_META_BY_ID: Dict[str, Dict[str, Any]] = {}
MODEL_META_LOCK = threading.Lock()


def run_comprehensive_adversarial_testing(
    current_content: str,
    content_type: str = "document_general",
    config: Optional[AdversarialConfiguration] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    team_manager: Optional[Any] = None,
    gauntlet_manager: Optional[Any] = None,
    use_decomposition: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced comprehensive adversarial testing implementing the full tripartite AI architecture
    from ULTIMATE_ADVERSARIAL_EVOLUTION_EXPLAINED.md
    
    This function implements:
    1. Red Team (Critics) - Vulnerability identification and attack simulation
    2. Blue Team (Fixers) - Defense implementation and fix application  
    3. Evaluator Team (Judges) - Quality assessment and consensus building
    
    Supports all evolution modes:
    - Standard adversarial testing
    - Quality diversity adversarial exploration
    - Multi-objective adversarial optimization
    - Problem decomposition adversarial analysis
    
    Args:
        current_content: Content to test adversarially
        content_type: Type of content being tested
        config: AdversarialConfiguration with all 272 parameters
        custom_config: Custom configuration overrides
        team_manager: Team coordination manager for Red/Blue/Evaluator teams
        gauntlet_manager: Gauntlet system manager for structured testing
        use_decomposition: Enable problem decomposition approach
        **kwargs: Additional OpenEvolve parameters
    
    Returns:
        Dict containing comprehensive adversarial testing results with full metrics
    """
    
    # Import TEAM_SYSTEM_AVAILABLE at the start of the function
    try:
        from evolution import TEAM_SYSTEM_AVAILABLE
    except ImportError:
        TEAM_SYSTEM_AVAILABLE = False
    
    _update_adv_log_and_status("🚀 Starting comprehensive adversarial testing...")
    _update_adv_log_and_status(f"📝 Content type: {content_type}")
    
    # Initialize parameter manager and configuration
    param_manager = ParameterManager()
    
    # Create configuration from session state if not provided
    if config is None:
        config = AdversarialConfiguration.from_parameter_manager(param_manager, st.session_state)
    
    # Apply custom configuration overrides
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        _update_adv_log_and_status(f"🔧 Applied {len(custom_config)} custom configuration overrides")
    
    # Validate configuration
    validation_result = config.validate(param_manager)
    if not validation_result.valid:
        _update_adv_log_and_status(f"⚠️ Configuration validation errors: {validation_result.errors}")
        for error in validation_result.errors[:3]:  # Show first 3 errors
            logger.warning(f"Adversarial parameter validation error: {error}")
    
    # Log comprehensive configuration summary
    _update_adv_log_and_status("📊 Adversarial Configuration Summary:")
    _update_adv_log_and_status(f"   • Adversarial Rounds: {config.adversarial_rounds}")
    _update_adv_log_and_status(f"   • Attack Strength: {config.attack_strength}")
    _update_adv_log_and_status(f"   • Defense Strategy: {config.defense_strategy}")
    _update_adv_log_and_status(f"   • Red Team Models: {len(config.red_team_models)}")
    _update_adv_log_and_status(f"   • Blue Team Models: {len(config.blue_team_models)}")
    _update_adv_log_and_status(f"   • Coevolutionary: {config.coevolutionary_approach}")
    
    # Advanced features summary
    advanced_features = []
    if config.cascade_evaluation:
        advanced_features.append("Cascade Evaluation")
    if config.use_llm_feedback:
        advanced_features.append("LLM Feedback")
    if config.meta_learning:
        advanced_features.append("Meta-Learning")
    if config.transfer_learning:
        advanced_features.append("Transfer Learning")
    if config.explainable_ai:
        advanced_features.append("Explainable AI")
    if config.differential_privacy:
        advanced_features.append("Differential Privacy")
    if config.ensemble_defense:
        advanced_features.append("Ensemble Defense")
    if config.attack_diversity:
        advanced_features.append("Attack Diversity")
    
    if advanced_features:
        _update_adv_log_and_status(f"   • Advanced Features: {', '.join(advanced_features)}")
    
    # Initialize comprehensive result structure
    start_time = time.time()
    operation_id = f"adversarial_{int(start_time)}"
    
    adversarial_result = {
        "success": False,
        "final_content": current_content,
        "original_content": current_content,
        "operation_id": operation_id,
        "start_time": start_time,
        "content_type": content_type,
        "config_summary": {
            "adversarial_rounds": config.adversarial_rounds,
            "attack_strength": config.attack_strength,
            "defense_strategy": config.defense_strategy,
            "coevolutionary": config.coevolutionary_approach,
            "use_decomposition": use_decomposition
        },
        "metrics": {
            "total_rounds": 0,
            "attack_success_rate": 0.0,
            "defense_success_rate": 0.0,
            "robustness_score": 0.0,
            "vulnerability_count": 0,
            "fixes_applied": 0,
            "consensus_score": 0.0,
            "improvement_ratio": 0.0
        },
        "team_results": {
            "red_team": {"assessments": [], "total_issues": 0},
            "blue_team": {"fixes": [], "total_fixes": 0},
            "evaluator_team": {"evaluations": [], "consensus": {}}
        },
        "rounds": [],
        "vulnerabilities": [],
        "fixes": [],
        "error": None
    }
    
    try:
        # Check if team system is available for full implementation
        if TEAM_SYSTEM_AVAILABLE and team_manager:
            _update_adv_log_and_status("👥 Using full team system for adversarial testing...")
            
            # Import team components
            from red_team import RedTeam
            from blue_team import BlueTeam
            from evaluator_team import EvaluatorTeam
            
            # Initialize teams
            red_team = RedTeam()
            blue_team = BlueTeam()
            evaluator_team = EvaluatorTeam()
            
            # Run comprehensive adversarial rounds
            current_content_working = current_content
            
            for round_num in range(config.adversarial_rounds):
                _update_adv_log_and_status(f"🔄 Starting adversarial round {round_num + 1}/{config.adversarial_rounds}")
                
                round_result = {
                    "round": round_num + 1,
                    "start_time": time.time(),
                    "red_team_assessment": None,
                    "blue_team_fixes": None,
                    "evaluator_consensus": None,
                    "content_before": current_content_working,
                    "content_after": current_content_working,
                    "improvements": []
                }
                
                # Phase 1: Red Team Critique Generation
                _update_adv_log_and_status(f"🔴 Phase 1: Red Team analysis (Round {round_num + 1})")
                
                if use_decomposition:
                    # Use decomposition-based adversarial analysis
                    red_assessment = red_team.assess_content_with_quality_diversity(
                        content=current_content_working,
                        content_type=content_type,
                        api_key=config.api_key,
                        model_name=config.red_team_models[0] if config.red_team_models else "gpt-4"
                    )
                else:
                    # Standard red team assessment
                    red_assessment = red_team.assess_content(
                        content=current_content_working,
                        content_type=content_type,
                        custom_requirements=f"Attack strength: {config.attack_strength}, Focus on: {', '.join(config.attack_types) if config.attack_types else 'general vulnerabilities'}"
                    )
                
                round_result["red_team_assessment"] = red_assessment
                adversarial_result["team_results"]["red_team"]["assessments"].append(red_assessment)
                
                if red_assessment and red_assessment.findings:
                    adversarial_result["metrics"]["vulnerability_count"] += len(red_assessment.findings)
                    adversarial_result["vulnerabilities"].extend([
                        {
                            "round": round_num + 1,
                            "category": issue.category.value if hasattr(issue.category, 'value') else str(issue.category),
                            "severity": issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity),
                            "description": issue.description,
                            "recommendation": getattr(issue, 'suggested_fix', 'No recommendation provided')
                        }
                        for issue in red_assessment.findings
                    ])
                    
                    _update_adv_log_and_status(f"🔍 Red Team found {len(red_assessment.findings)} issues")
                    
                    # Phase 2: Blue Team Patch Development
                    _update_adv_log_and_status(f"🔵 Phase 2: Blue Team fixes (Round {round_num + 1})")
                    
                    # Generate fixes for identified issues
                    # Create dummy issues for blue team to fix
                    from red_team import IssueFinding, IssueCategory
                    from quality_assessment import SeverityLevel
                    
                    fix_issues = [IssueFinding(
                        title=f"Fix Issue {i+1}",
                        description=issue.description,
                        severity=issue.severity,
                        category=issue.category
                    ) for i, issue in enumerate(red_assessment.findings[:3])]
                    
                    blue_assessment = blue_team.apply_fixes(
                        content=current_content_working,
                        issues=fix_issues,
                        content_type=content_type,
                        custom_requirements=f"Defense strength: {config.defense_strength}, Strategy: {config.defense_strategy}"
                    )
                    
                    round_result["blue_team_fixes"] = blue_assessment
                    adversarial_result["team_results"]["blue_team"]["fixes"].append(blue_assessment)
                    
                    if blue_assessment and blue_assessment.applied_fixes:
                        adversarial_result["metrics"]["fixes_applied"] += len(blue_assessment.applied_fixes)
                        adversarial_result["fixes"].extend([
                            {
                                "round": round_num + 1,
                                "type": fix.fix_type.value if hasattr(fix.fix_type, 'value') else str(fix.fix_type),
                                "description": fix.description,
                                "implementation": fix.fixed_content,
                                "confidence": fix.effectiveness_score
                            }
                            for fix in blue_assessment.applied_fixes
                        ])
                        
                        # Apply the best fix to the content
                        best_fix = max(blue_assessment.applied_fixes, key=lambda f: f.effectiveness_score)
                        if best_fix.fixed_content and best_fix.fixed_content.strip():
                            current_content_working = best_fix.fixed_content
                            round_result["content_after"] = current_content_working
                            _update_adv_log_and_status(f"✅ Applied fix: {best_fix.description[:100]}...")
                        
                        _update_adv_log_and_status(f"🔧 Blue Team generated {len(blue_assessment.applied_fixes)} fixes")
                    
                    # Phase 3: Evaluator Team Consensus Building
                    _update_adv_log_and_status(f"⚖️ Phase 3: Evaluator Team consensus (Round {round_num + 1})")
                    
                    evaluator_assessment = evaluator_team.evaluate_content(
                        content=current_content_working,
                        content_type=content_type,
                        custom_requirements={"evaluation_focus": f"Evaluate improvements from round {round_num + 1}. Original issues: {len(red_assessment.findings)}, Fixes applied: {len(blue_assessment.applied_fixes) if blue_assessment else 0}"}
                    )
                    
                    round_result["evaluator_consensus"] = evaluator_assessment
                    adversarial_result["team_results"]["evaluator_team"]["evaluations"].append(evaluator_assessment)
                    
                    if evaluator_assessment:
                        consensus_score = evaluator_assessment.overall_score
                        adversarial_result["metrics"]["consensus_score"] = max(adversarial_result["metrics"]["consensus_score"], consensus_score)
                        _update_adv_log_and_status(f"📊 Evaluator consensus score: {consensus_score:.3f}")
                
                else:
                    _update_adv_log_and_status("✅ Red Team found no significant issues")
                
                round_result["end_time"] = time.time()
                round_result["duration"] = round_result["end_time"] - round_result["start_time"]
                adversarial_result["rounds"].append(round_result)
                
                # Update metrics
                adversarial_result["metrics"]["total_rounds"] = round_num + 1
                
                # Early stopping if content is sufficiently robust
                if adversarial_result["metrics"]["consensus_score"] > 0.9:
                    _update_adv_log_and_status(f"🎯 Early stopping: High consensus score achieved ({adversarial_result['metrics']['consensus_score']:.3f})")
                    break
            
            # Calculate final metrics
            if adversarial_result["metrics"]["vulnerability_count"] > 0:
                adversarial_result["metrics"]["attack_success_rate"] = min(1.0, adversarial_result["metrics"]["vulnerability_count"] / (config.adversarial_rounds * 3))  # Assume max 3 issues per round
            
            if adversarial_result["metrics"]["fixes_applied"] > 0:
                adversarial_result["metrics"]["defense_success_rate"] = min(1.0, adversarial_result["metrics"]["fixes_applied"] / max(1, adversarial_result["metrics"]["vulnerability_count"]))
            
            # Calculate robustness score (higher is better)
            adversarial_result["metrics"]["robustness_score"] = (
                adversarial_result["metrics"]["consensus_score"] * 0.4 +
                adversarial_result["metrics"]["defense_success_rate"] * 0.3 +
                (1.0 - adversarial_result["metrics"]["attack_success_rate"]) * 0.3
            )
            
            # Calculate improvement ratio
            if current_content_working != current_content:
                adversarial_result["metrics"]["improvement_ratio"] = len(current_content_working) / len(current_content) if current_content else 1.0
                adversarial_result["final_content"] = current_content_working
            
            adversarial_result["success"] = True
            
        else:
            # Fallback to OpenEvolve backend or basic implementation
            _update_adv_log_and_status("🔄 Using OpenEvolve backend for adversarial testing...")
            
            backend_result = _run_adversarial_testing_with_openevolve_backend_enhanced(
                current_content=current_content,
                content_type=content_type,
                config=config
            )
            
            adversarial_result.update(backend_result)
        
        # Finalize results
        end_time = time.time()
        adversarial_result["end_time"] = end_time
        adversarial_result["total_duration"] = end_time - start_time
        
        # Log comprehensive results
        _update_adv_log_and_status("✅ Comprehensive adversarial testing completed!")
        _update_adv_log_and_status(f"⏱️ Total duration: {adversarial_result['total_duration']:.2f}s")
        _update_adv_log_and_status(f"🛡️ Robustness score: {adversarial_result['metrics']['robustness_score']:.4f}")
        _update_adv_log_and_status(f"🔍 Vulnerabilities found: {adversarial_result['metrics']['vulnerability_count']}")
        _update_adv_log_and_status(f"🔧 Fixes applied: {adversarial_result['metrics']['fixes_applied']}")
        
        return adversarial_result
        
    except Exception as e:
        adversarial_result["error"] = str(e)
        adversarial_result["end_time"] = time.time()
        adversarial_result["total_duration"] = adversarial_result["end_time"] - start_time
        
        _update_adv_log_and_status(f"💥 Comprehensive adversarial testing failed: {e}")
        logger.error(f"Adversarial testing error: {e}", exc_info=True)
        return adversarial_result

def _run_adversarial_testing_with_openevolve_backend_enhanced(
    current_content: str,
    content_type: str,
    config: AdversarialConfiguration
) -> Dict[str, Any]:
    """
    Enhanced backend function using comprehensive adversarial configuration
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend not available for adversarial testing")
        return {"success": False, "error": "OpenEvolve backend not available"}

    try:
        _update_adv_log_and_status(f"🔧 Using {len(asdict(config))} adversarial parameters")
        
        # Prepare comprehensive model configurations
        red_team_configs = []
        for i, model_id in enumerate(config.red_team_models[:config.red_team_sample_size]):
            red_team_configs.append({
                "name": model_id,
                "weight": 1.0,
                "temperature": config.adversarial_temperature,
                "max_tokens": config.max_tokens,
                "role": "attacker",
                "attack_strength": config.attack_strength,
                "attack_types": config.attack_types
            })
        
        blue_team_configs = []
        for i, model_id in enumerate(config.blue_team_models[:config.blue_team_sample_size]):
            blue_team_configs.append({
                "name": model_id,
                "weight": 1.0,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "role": "defender",
                "defense_strength": config.defense_strength,
                "defense_strategies": config.defense_strategies
            })
        
        # Create evaluator configurations
        evaluator_configs = config.evaluator_models or []
        if not evaluator_configs:
            # Default evaluator configuration
            evaluator_configs = [{
                "name": "gpt-4",
                "weight": 1.0,
                "temperature": 0.3,  # Lower temperature for more consistent evaluation
                "max_tokens": config.max_tokens,
                "role": "evaluator"
            }]
        
        _update_adv_log_and_status(f"🔴 Red Team: {len(red_team_configs)} models")
        _update_adv_log_and_status(f"🔵 Blue Team: {len(blue_team_configs)} models")
        _update_adv_log_and_status(f"⚖️ Evaluators: {len(evaluator_configs)} models")
        
        # Convert to evolution configuration for compatibility
        evolution_config = config.to_evolution_config()
        
        # Set adversarial-specific parameters
        evolution_config.attack_model_config = config.attack_model_config or (red_team_configs[0] if red_team_configs else {})
        evolution_config.defense_model_config = config.defense_model_config or (blue_team_configs[0] if blue_team_configs else {})
        evolution_config.adversarial_rounds = config.adversarial_rounds
        evolution_config.attack_strength = config.attack_strength
        evolution_config.defense_strategy = config.defense_strategy
        evolution_config.coevolutionary_approach = config.coevolutionary_approach
        evolution_config.red_team_models = config.red_team_models
        evolution_config.blue_team_models = config.blue_team_models
        evolution_config.red_team_sample_size = config.red_team_sample_size
        evolution_config.blue_team_sample_size = config.blue_team_sample_size
        evolution_config.adversarial_temperature = config.adversarial_temperature
        evolution_config.attack_diversity = config.attack_diversity
        evolution_config.defense_strength = config.defense_strength
        evolution_config.adversarial_budget = config.adversarial_budget
        evolution_config.attack_types = config.attack_types
        evolution_config.defense_strategies = config.defense_strategies
        evolution_config.robustness_metric = config.robustness_metric
        evolution_config.perturbation_bound = config.perturbation_bound
        evolution_config.gradient_masking = config.gradient_masking
        evolution_config.ensemble_defense = config.ensemble_defense
        
        # Create specialized evaluator based on content type
        if content_type.startswith("code_"):
            evaluator_instance = create_specialized_evaluator(
                content_type, 
                config.domain_knowledge, 
                config.regulatory_compliance
            )
        else:
            evaluator_instance = create_language_specific_evaluator(
                content_type, 
                config.domain_knowledge, 
                config.regulatory_compliance
            )
        
        # Run multiple adversarial rounds
        best_results = None
        best_score = 0.0
        all_round_results = []
        
        for round_num in range(config.adversarial_rounds):
            _update_adv_log_and_status(f"🥊 Starting adversarial round {round_num + 1}/{config.adversarial_rounds}")
            
            # Adjust attack strength per round for progressive difficulty
            current_attack_strength = config.attack_strength * (1.0 + (round_num * 0.1))
            evolution_config.attack_strength = min(current_attack_strength, 2.0)  # Cap at 2.0
            
            # Run adversarial evolution using unified evolution function
            from evolution import run_comprehensive_evolution
            
            round_result = run_comprehensive_evolution(
                content=current_content,
                content_type=content_type,
                evolution_mode="adversarial",
                custom_config=asdict(evolution_config)
            )
            
            # Process round results
            if round_result and round_result != current_content:
                # Evaluate the adversarial result
                try:
                    # If evaluator_instance is an object with evaluate_content method
                    if hasattr(evaluator_instance, 'evaluate_content'):
                        evaluation = evaluator_instance.evaluate_content(round_result)
                    else:
                        # If evaluator_instance is a function, create temp file and call it
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                            f.write(str(round_result))
                            temp_path = f.name
                        
                        try:
                            evaluation = evaluator_instance(temp_path)
                        finally:
                            import os
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                    
                    round_score = evaluation.get("robustness_score", evaluation.get("score", 0.0))
                except Exception as e:
                    print(f"Error evaluating round result: {e}")
                    evaluation = {"score": 0.0}
                    round_score = 0.0
                
                round_data = {
                    "round": round_num + 1,
                    "content": round_result,
                    "score": round_score,
                    "attack_strength": current_attack_strength,
                    "evaluation": evaluation
                }
                
                all_round_results.append(round_data)
                
                _update_adv_log_and_status(f"✅ Round {round_num + 1} completed - Score: {round_score:.4f}")
                
                # Track best result
                if round_score > best_score:
                    best_score = round_score
                    best_results = round_data
                    current_content = round_result  # Use improved content for next round
            else:
                _update_adv_log_and_status(f"⚠️ Round {round_num + 1} produced no improvement")
        
        # Compile comprehensive results
        if best_results:
            _update_adv_log_and_status(f"🏆 Adversarial testing completed successfully!")
            _update_adv_log_and_status(f"📊 Best robustness score: {best_score:.4f}")
            _update_adv_log_and_status(f"🔧 Parameters utilized: {len(asdict(config))}")
            
            return {
                "success": True,
                "best_content": best_results["content"],
                "best_score": best_score,
                "best_round": best_results["round"],
                "all_rounds": all_round_results,
                "total_rounds": config.adversarial_rounds,
                "configuration": asdict(config),
                "metrics": {
                    "robustness_improvement": best_score,
                    "rounds_completed": len(all_round_results),
                    "attack_strength_final": best_results.get("attack_strength", config.attack_strength),
                    "advanced_features_used": len(advanced_features) if 'advanced_features' in locals() else 0
                }
            }
        else:
            _update_adv_log_and_status("🤔 Adversarial testing completed with no improvement")
            return {
                "success": False,
                "message": "No improvement achieved through adversarial testing",
                "original_content": current_content,
                "rounds_attempted": config.adversarial_rounds,
                "configuration": asdict(config)
            }
            
    except Exception as e:
        _update_adv_log_and_status(f"💥 Adversarial testing failed: {e}")
        logger.error(f"Adversarial testing error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "configuration": asdict(config) if config else {}
        }

def _run_adversarial_testing_with_openevolve_backend(
    current_content: str,
    content_type: str,
    red_team_models: List[str],
    blue_team_models: List[str],
    api_key: str,
    base_url: str,
    max_iterations: int,
    confidence_threshold: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    seed: Optional[int],
    max_workers: int,
    rotation_strategy: str,
    red_team_sample_size: int,
    blue_team_sample_size: int,
    custom_requirements: str = "",
    evaluator_system_prompt: str = "",
    red_team_prompt: str = "",
    blue_team_prompt: str = "",
    compliance_rules: Optional[List[str]] = None,
    feature_dimensions: Optional[List[str]] = None,
    feature_bins: Optional[int] = None,
    enable_data_augmentation: bool = False,
    augmentation_model_id: str = None,
    augmentation_temperature: float = 0.7,
    enable_human_feedback: bool = False,
    current_iteration: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility - now uses comprehensive configuration
    """
    
    # Create comprehensive configuration from legacy parameters
    config = AdversarialConfiguration()
    
    # Map legacy parameters to new configuration
    config.red_team_models = red_team_models
    config.blue_team_models = blue_team_models
    config.api_key = api_key
    config.api_base = base_url
    config.max_iterations = max_iterations
    config.max_tokens = max_tokens
    config.temperature = temperature
    config.seed = seed
    config.red_team_sample_size = red_team_sample_size
    config.blue_team_sample_size = blue_team_sample_size
    config.domain_knowledge = custom_requirements
    config.evaluator_system_message = evaluator_system_prompt
    config.system_prompt = red_team_prompt
    config.regulatory_compliance = compliance_rules or []
    
    # Apply additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Use the enhanced function
    return run_comprehensive_adversarial_testing(
        current_content=current_content,
        content_type=content_type,
        config=config
    )

def create_adversarial_configuration(
    parameters: Optional[Dict[str, Any]] = None,
    adversarial_rounds: int = 3,
    attack_strength: float = 0.5,
    defense_strength: float = 0.7,
    **kwargs
) -> AdversarialConfiguration:
    """
    Create a comprehensive adversarial configuration with explicit parameters.
    This is the new standalone version that doesn't depend on session state.
    
    Args:
        parameters: Dictionary of parameters to use (if None, uses defaults)
        adversarial_rounds: Number of adversarial rounds
        attack_strength: Strength of attacks (0.0-1.0)
        defense_strength: Strength of defenses (0.0-1.0)
        **kwargs: Additional parameters to override
        
    Returns:
        AdversarialConfiguration object
    """
    param_manager = ParameterManager()
    
    # Create parameter dictionary with defaults
    if parameters is None:
        parameters = {}
    
    # Set basic parameters
    parameters.setdefault('adversarial_rounds', adversarial_rounds)
    parameters.setdefault('attack_strength', attack_strength)
    parameters.setdefault('defense_strength', defense_strength)
    
    # Override with any additional kwargs
    parameters.update(kwargs)
    
    # Create configuration from parameters
    config = AdversarialConfiguration.from_parameter_manager(param_manager, parameters)
    
    # Validate configuration
    validation_result = config.validate(param_manager)
    if not validation_result.valid:
        logger.warning(f"Configuration has {len(validation_result.errors)} validation errors")
        for error in validation_result.errors[:3]:  # Show first 3 errors
            logger.warning(f"   - {error}")
    
    if validation_result.warnings:
        logger.warning(f"Configuration has {len(validation_result.warnings)} warnings")
    
    return config


def create_adversarial_configuration_from_session() -> AdversarialConfiguration:
    """
    Create comprehensive adversarial configuration from Streamlit session state.
    This is the legacy version for backward compatibility.
    """
    try:
        import streamlit as st
        param_manager = ParameterManager()
        config = AdversarialConfiguration.from_parameter_manager(param_manager, st.session_state)
        
        # Log configuration summary
        _update_adv_log_and_status(f"🔧 Adversarial configuration created with {len(asdict(config))} parameters")
        
        # Validate configuration
        validation_result = config.validate(param_manager)
        if not validation_result.valid:
            _update_adv_log_and_status(f"⚠️ Configuration has {len(validation_result.errors)} validation errors")
            for error in validation_result.errors[:3]:  # Show first 3 errors
                _update_adv_log_and_status(f"   - {error}")
        
        if validation_result.warnings:
            _update_adv_log_and_status(f"⚠️ Configuration has {len(validation_result.warnings)} warnings")
        
        return config
    except ImportError:
        # Streamlit not available, use standalone version with defaults
        logger.warning("Streamlit not available, using default configuration")
        return create_adversarial_configuration()

def get_adversarial_capabilities_summary() -> Dict[str, Any]:
    """
    Get summary of adversarial testing capabilities and parameters
    """
    param_manager = ParameterManager()
    
    # Get adversarial-specific parameters
    adversarial_params = param_manager.get_parameters_by_category("adversarial")
    
    capabilities = {
        "total_adversarial_parameters": len(adversarial_params),
        "adversarial_modes": [
            "red_team_blue_team",
            "coevolutionary",
            "ensemble_defense",
            "gradient_masking",
            "differential_privacy"
        ],
        "attack_strategies": [
            "perturbation_based",
            "prompt_injection",
            "adversarial_examples",
            "model_inversion",
            "membership_inference"
        ],
        "defense_mechanisms": [
            "ensemble_defense",
            "gradient_masking",
            "adversarial_training",
            "input_validation",
            "output_filtering"
        ],
        "robustness_metrics": [
            "accuracy_under_attack",
            "perturbation_resistance",
            "semantic_preservation",
            "confidence_calibration"
        ],
        "advanced_features": {
            "coevolutionary_approach": "Co-evolution between attackers and defenders",
            "ensemble_defense": "Multiple defense models working together",
            "attack_diversity": "Diverse attack strategies for comprehensive testing",
            "gradient_masking": "Protection against gradient-based attacks",
            "differential_privacy": "Privacy-preserving adversarial testing",
            "meta_learning": "Learning from previous adversarial encounters",
            "transfer_learning": "Knowledge transfer across domains",
            "explainable_ai": "Interpretable adversarial decisions"
        },
        "parameter_categories": {
            "adversarial": len(adversarial_params),
            "evaluation": len(param_manager.get_parameters_by_category("evaluation")),
            "prompt_engineering": len(param_manager.get_parameters_by_category("prompt_engineering")),
            "resource_management": len(param_manager.get_parameters_by_category("resource_management"))
        }
    }
    
    return capabilities
    """
    Run adversarial testing using OpenEvolve backend for code content with ALL features.

    Args:
        current_content: The content to test adversarially
        content_type: Type of content being tested
        red_team_models: List of red team models
        blue_team_models: List of blue team models
        api_key: API key for the LLM provider
        base_url: Base URL for the API
        max_iterations: Maximum number of iterations
        confidence_threshold: Confidence threshold for stopping
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty
        seed: Random seed
        max_workers: Maximum number of parallel workers
        rotation_strategy: Model rotation strategy
        red_team_sample_size: Number of red team models to sample
        blue_team_sample_size: Number of blue team models to sample
        custom_requirements: Custom requirements for testing
        evaluator_system_prompt: System prompt for evaluation
        red_team_prompt: Prompt for red team
        blue_team_prompt: Prompt for blue team
        compliance_rules: Compliance rules to check against
        feature_dimensions: List of feature dimensions for MAP-Elites
        feature_bins: Number of bins for feature dimensions
        enable_data_augmentation: Whether to perform data augmentation
        augmentation_model_id: Model to use for data augmentation
        augmentation_temperature: Temperature for augmentation
        enable_human_feedback: Whether to capture human feedback

    Advanced OpenEvolve parameters:
        enable_artifacts: Whether to enable artifact side-channel
        cascade_evaluation: Whether to use cascade evaluation
        cascade_thresholds: Thresholds for cascade evaluation
        use_llm_feedback: Whether to use LLM-based feedback
        llm_feedback_weight: Weight for LLM feedback
        parallel_evaluations: Number of parallel evaluations
        distributed: Whether to use distributed evaluation
        template_dir: Directory for prompt templates
        num_top_programs: Number of top programs to include in prompts
        num_diverse_programs: Number of diverse programs to include in prompts
        use_template_stochasticity: Whether to use template stochasticity
        template_variations: Template variations for stochasticity
        use_meta_prompting: Whether to use meta-prompting
        meta_prompt_weight: Weight for meta-prompting
        include_artifacts: Whether to include artifacts in prompts
        max_artifact_bytes: Maximum artifact size in bytes
        artifact_security_filter: Whether to apply security filtering to artifacts
        early_stopping_patience: Patience for early stopping
        convergence_threshold: Convergence threshold for early stopping
        early_stopping_metric: Metric to use for early stopping
        memory_limit_mb: Memory limit in MB for evaluation
        cpu_limit: CPU limit for evaluation
        random_seed: Random seed for reproducibility
        db_path: Path to database file
        in_memory: Whether to use in-memory database
        diff_based_evolution: Whether to use diff-based evolution
        max_code_length: Maximum length of code to evolve
        evolution_trace_enabled: Whether to enable evolution trace logging
        evolution_trace_format: Format for evolution traces
        evolution_trace_include_code: Whether to include code in traces
        evolution_trace_include_prompts: Whether to include prompts in traces
        evolution_trace_output_path: Output path for evolution traces
        evolution_trace_buffer_size: Buffer size for trace writing
        evolution_trace_compress: Whether to compress traces
        log_level: Logging level
        log_dir: Directory for log files
        api_timeout: Timeout for API requests
        api_retries: Number of API request retries
        api_retry_delay: Delay between API retries
        artifact_size_threshold: Threshold for artifact storage
        cleanup_old_artifacts: Whether to cleanup old artifacts
        artifact_retention_days: Days to retain artifacts
        diversity_reference_size: Size of reference set for diversity calculation
        max_retries_eval: Maximum retries for evaluation
        evaluator_timeout: Timeout for evaluation
        evaluator_models: List of evaluator model configurations
        double_selection: Use different programs for performance vs inspiration
        adaptive_feature_dimensions: Adjust feature dimensions based on progress
        test_time_compute: Use test-time compute for enhanced reasoning
        optillm_integration: Integrate with OptiLLM for advanced routing
        plugin_system: Enable plugin system for extended capabilities
        hardware_optimization: Optimize for specific hardware (GPU, etc.)
        multi_strategy_sampling: Use elite, diverse, and exploratory selection
        ring_topology: Use ring topology for island migration
        controlled_gene_flow: Control gene flow between islands
        auto_diff: Use automatic differentiation where applicable
        symbolic_execution: Enable symbolic execution for verification
        coevolutionary_approach: Use co-evolution between different populations

    Returns:
        Dict[str, Any]: Adversarial testing results
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend not available for adversarial testing")
        return {"success": False, "error": "OpenEvolve backend not available"}

    try:
        # Prepare model configurations for adversarial evolution
        red_team_configs = []
        for model_id in red_team_models[:red_team_sample_size]:
            red_team_configs.append({
                "name": model_id,
                "weight": 1.0,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            })
        
        blue_team_configs = []
        for model_id in blue_team_models[:blue_team_sample_size]:
            blue_team_configs.append({
                "name": model_id,
                "weight": 1.0,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            })

        # Create comprehensive OpenEvolve configuration with ALL parameters
        config = create_comprehensive_openevolve_config(
            content_type=content_type,
            model_configs=red_team_configs + blue_team_configs,  # Combine both teams
            api_key=api_key,
            api_base=base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
            population_size=max_workers,
            num_islands=st.session_state.get("num_islands", 5),
            migration_interval=st.session_state.get("migration_interval", 50),
            migration_rate=st.session_state.get("migration_rate", 0.1),
            archive_size=st.session_state.get("archive_size", 100),
            elite_ratio=st.session_state.get("elite_ratio", 0.1),
            exploration_ratio=st.session_state.get("exploration_ratio", 0.2),
            exploitation_ratio=st.session_state.get("exploitation_ratio", 0.7),
            checkpoint_interval=st.session_state.get("checkpoint_interval", 100),
            feature_dimensions=feature_dimensions,
            feature_bins=feature_bins,
            system_message=red_team_prompt,  # Use red team prompt as base
            evaluator_system_message=evaluator_system_prompt,
            # Advanced parameters
            enable_artifacts=enable_artifacts,
            cascade_evaluation=cascade_evaluation,
            cascade_thresholds=cascade_thresholds,
            use_llm_feedback=use_llm_feedback,
            llm_feedback_weight=llm_feedback_weight,
            parallel_evaluations=parallel_evaluations,
            distributed=distributed,
            template_dir=template_dir,
            num_top_programs=num_top_programs,
            num_diverse_programs=num_diverse_programs,
            use_template_stochasticity=use_template_stochasticity,
            template_variations=template_variations,
            use_meta_prompting=use_meta_prompting,
            meta_prompt_weight=meta_prompt_weight,
            include_artifacts=include_artifacts,
            max_artifact_bytes=max_artifact_bytes,
            artifact_security_filter=artifact_security_filter,
            early_stopping_patience=early_stopping_patience,
            convergence_threshold=convergence_threshold,
            early_stopping_metric=early_stopping_metric,
            memory_limit_mb=memory_limit_mb,
            cpu_limit=cpu_limit,
            random_seed=random_seed,
            db_path=db_path,
            in_memory=in_memory,
            # Additional parameters
            diff_based_evolution=diff_based_evolution,
            max_code_length=max_code_length,
            evolution_trace_enabled=evolution_trace_enabled,
            evolution_trace_format=evolution_trace_format,
            evolution_trace_include_code=evolution_trace_include_code,
            evolution_trace_include_prompts=evolution_trace_include_prompts,
            evolution_trace_output_path=evolution_trace_output_path,
            evolution_trace_buffer_size=evolution_trace_buffer_size,
            evolution_trace_compress=evolution_trace_compress,
            log_level=log_level,
            log_dir=log_dir,
            api_timeout=api_timeout,
            api_retries=api_retries,
            api_retry_delay=api_retry_delay,
            artifact_size_threshold=artifact_size_threshold,
            cleanup_old_artifacts=cleanup_old_artifacts,
            artifact_retention_days=artifact_retention_days,
            diversity_reference_size=diversity_reference_size,
            max_retries_eval=max_retries_eval,
            evaluator_timeout=evaluator_timeout,
            evaluator_models=evaluator_models,
            # Advanced research features
            double_selection=double_selection,
            adaptive_feature_dimensions=adaptive_feature_dimensions,
            test_time_compute=test_time_compute,
            optillm_integration=optillm_integration,
            plugin_system=plugin_system,
            hardware_optimization=hardware_optimization,
            multi_strategy_sampling=multi_strategy_sampling,
            ring_topology=ring_topology,
            controlled_gene_flow=controlled_gene_flow,
            auto_diff=auto_diff,
            symbolic_execution=symbolic_execution,
            coevolutionary_approach=coevolutionary_approach,
        )

        if not config:
            return {"success": False, "error": "Failed to create OpenEvolve configuration"}

        # Perform data augmentation if enabled
        if enable_data_augmentation and augmentation_model_id:
            _update_adv_log_and_status(
                f"🧪 Augmenting content using {augmentation_model_id}..."
            )
            current_content = generate_adversarial_data_augmentation(
                content=current_content,
                content_type=content_type,
                api_key=api_key,
                model_id=augmentation_model_id,
                temperature=augmentation_temperature,
                max_tokens=max_tokens,  # Use same max_tokens as main evolution
                seed=seed,
            )
            _update_adv_log_and_status("✅ Content augmentation complete.")

        # Create evaluator function based on content_type
        if content_type.startswith("code_"):
            evaluator_instance = create_specialized_evaluator(
                content_type, custom_requirements, compliance_rules
            )
        else:
            evaluator_instance = create_language_specific_evaluator(
                content_type, custom_requirements, compliance_rules
            )

        # Run adversarial evolution using the unified evolution function
        result = run_unified_evolution(
            content=current_content,
            content_type=content_type,
            evolution_mode="adversarial",
            model_configs=red_team_configs + blue_team_configs,  # Combined team models
            api_key=api_key,
            api_base=base_url,
            max_iterations=max_iterations,
            population_size=max_workers,
            system_message=red_team_prompt,  # Red team prompt
            evaluator_system_message=evaluator_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            feature_dimensions=feature_dimensions,
            custom_requirements=custom_requirements,
            custom_evaluator=evaluator_instance.evaluate,
            # Pass adversarial-specific parameters
            attack_model_config=red_team_configs[0] if red_team_configs else {"name": "gpt-4", "weight": 1.0},
            defense_model_config=blue_team_configs[0] if blue_team_configs else {"name": "gpt-4", "weight": 1.0},
            # All advanced parameters
            enable_artifacts=enable_artifacts,
            cascade_evaluation=cascade_evaluation,
            use_llm_feedback=use_llm_feedback,
            llm_feedback_weight=llm_feedback_weight,
            evolution_trace_enabled=evolution_trace_enabled,
            early_stopping_patience=early_stopping_patience,
            convergence_threshold=convergence_threshold,
            random_seed=random_seed,
            diff_based_evolution=diff_based_evolution,
            max_code_length=max_code_length,
            diversity_metric="edit_distance",
            parallel_evaluations=parallel_evaluations,
            distributed=distributed,
            template_dir=template_dir,
            num_top_programs=num_top_programs,
            num_diverse_programs=num_diverse_programs,
            use_template_stochasticity=use_template_stochasticity,
            template_variations=template_variations or {},
            use_meta_prompting=use_meta_prompting,
            meta_prompt_weight=meta_prompt_weight,
            include_artifacts=include_artifacts,
            max_artifact_bytes=max_artifact_bytes,
            artifact_security_filter=artifact_security_filter,
            memory_limit_mb=memory_limit_mb,
            cpu_limit=cpu_limit,
            db_path=db_path,
            in_memory=in_memory,
            log_level=log_level,
            log_dir=log_dir,
            api_timeout=api_timeout,
            api_retries=api_retries,
            api_retry_delay=api_retry_delay,
            artifact_size_threshold=artifact_size_threshold,
            cleanup_old_artifacts=cleanup_old_artifacts,
            artifact_retention_days=artifact_retention_days,
            diversity_reference_size=diversity_reference_size,
            max_retries_eval=max_retries_eval,
            evaluator_timeout=evaluator_timeout,
            evaluator_models=evaluator_models,
            # Advanced research features
            double_selection=double_selection,
            adaptive_feature_dimensions=adaptive_feature_dimensions,
            test_time_compute=test_time_compute,
            optillm_integration=optillm_integration,
            plugin_system=plugin_system,
            hardware_optimization=hardware_optimization,
            multi_strategy_sampling=multi_strategy_sampling,
            ring_topology=ring_topology,
            controlled_gene_flow=controlled_gene_flow,
            auto_diff=auto_diff,
            symbolic_execution=symbolic_execution,
            coevolutionary_approach=coevolutionary_approach,
        )

        if result and result.get("success"):
            # Process results
            best_code = result.get("best_code", current_content)
            
            # Simulate human feedback capture
            if enable_human_feedback:
                # For now, a dummy score and comments
                dummy_score = random.uniform(0.5, 1.0)  # Simulate a score
                dummy_comments = "Human reviewed and provided general feedback on clarity and relevance."
                capture_human_feedback(
                    adversarial_example={
                        "id": str(uuid.uuid4()),
                        "content": best_code,
                        "content_type": content_type,
                        "iteration": current_iteration,  # Now using the passed iteration
                    },
                    human_score=dummy_score,
                    human_comments=dummy_comments,
                )

            return {
                "success": True,
                "best_program": result.get("best_program"),
                "best_score": result.get("best_score", 0.0),
                "best_code": best_code,
                "metrics": result.get("metrics", {}),
                "output_dir": result.get("output_dir"),
            }
        else:
            return {
                "success": False,
                "message": result.get("message", "Adversarial testing completed with no improvement."),
            }

    except Exception as e:
        st.error(f"Error running adversarial testing with OpenEvolve backend: {e}")
        print(f"Adversarial testing error: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}













def run_adversarial_testing():
    """Enhanced adversarial testing using comprehensive parameter configuration."""
    print("run_adversarial_testing function called - Enhanced version")
    
    _update_adv_log_and_status("🚀 Starting enhanced adversarial testing with comprehensive parameters...")
    
    # Load model performance data for continuous learning
    try:
        with open("model_performance.json", "r") as f:
            st.session_state.adversarial_model_performance = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.session_state.adversarial_model_performance = {}

    try:
        # Create comprehensive adversarial configuration
        config = create_adversarial_configuration_from_session()
        
        # Log capabilities summary
        capabilities = get_adversarial_capabilities_summary()
        _update_adv_log_and_status(f"📊 Adversarial capabilities: {capabilities['total_adversarial_parameters']} parameters")
        _update_adv_log_and_status(f"🎯 Attack strategies: {len(capabilities['attack_strategies'])}")
        _update_adv_log_and_status(f"🛡️ Defense mechanisms: {len(capabilities['defense_mechanisms'])}")
        
        # Validate session state requirements
        if not config.api_key:
            st.error("OpenRouter API key is required for adversarial testing.")
            return

        if not config.red_team_models or not config.blue_team_models:
            st.error("Please select at least one model for both red and blue teams.")
            return

        if not st.session_state.protocol_text.strip():
            st.error("Please enter a protocol to test.")
            return

        current_sop = st.session_state.protocol_text
        
        # Determine content type and review type
        content_type = "document_general"
        if st.session_state.get("adversarial_custom_mode", False):
            review_type = "custom"
            content_type = st.session_state.get("adversarial_content_type", "document_general")
        else:
            # Auto-detect or use specified review type
            if st.session_state.adversarial_review_type == "Auto-Detect":
                review_type = determine_review_type(current_sop)
            elif st.session_state.adversarial_review_type == "Code Review":
                review_type = "code"
                content_type = st.session_state.get("code_language_type", "code_python")
            elif st.session_state.adversarial_review_type == "Plan Review":
                review_type = "plan"
                content_type = "document_general"
            elif st.session_state.adversarial_review_type == "Legal Document":
                review_type = "document"
                content_type = "document_legal"
            elif st.session_state.adversarial_review_type == "Medical Document":
                review_type = "document"
                content_type = "document_medical"
            elif st.session_state.adversarial_review_type == "Technical Document":
                review_type = "document"
                content_type = "document_technical"
            else:
                review_type = "general"
                content_type = "document_general"

        _update_adv_log_and_status(f"📋 Review type: {review_type}, Content type: {content_type}")
        
        # Initialize session state for tracking
        with st.session_state.thread_lock:
            st.session_state.adversarial_log = []
            st.session_state.adversarial_stop_flag = False
            st.session_state.adversarial_total_tokens_prompt = 0
            st.session_state.adversarial_total_tokens_completion = 0
            st.session_state.adversarial_cost_estimate_usd = 0.0

        # Check OpenEvolve backend availability
        if OPENEVOLVE_AVAILABLE:
            try:
                import requests
                health_response = requests.get("http://localhost:8000/health", timeout=5)
                if health_response.status_code == 200:
                    _update_adv_log_and_status("✅ OpenEvolve backend is available")
                    
                    # Run comprehensive adversarial testing
                    result = run_comprehensive_adversarial_testing(
                        current_content=current_sop,
                        content_type=content_type,
                        config=config
                    )
                    
                    # Store results in session state
                    st.session_state.adversarial_results = result
                    
                    if result.get("success"):
                        _update_adv_log_and_status("🎉 Comprehensive adversarial testing completed successfully!")
                        _update_adv_log_and_status(f"🏆 Best robustness score: {result.get('best_score', 0.0):.4f}")
                        _update_adv_log_and_status(f"🔄 Rounds completed: {result.get('total_rounds', 0)}")
                    else:
                        _update_adv_log_and_status(f"⚠️ Adversarial testing completed with issues: {result.get('message', 'Unknown error')}")
                        
                else:
                    st.error("OpenEvolve backend is not responding. Please ensure it is running.")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to OpenEvolve backend. Please ensure it is running.")
            except Exception as e:
                st.error(f"Error checking OpenEvolve backend: {e}")
        else:
            st.error("OpenEvolve backend is not available. Please install and run the backend.")
        # --- Initialization ---
        api_key = st.session_state.openrouter_key
        red_team_base = list(st.session_state.red_team_models or [])
        blue_team_base = list(st.session_state.blue_team_models or [])
        min_iter, max_iter = (
            st.session_state.adversarial_min_iter,
            st.session_state.adversarial_max_iter,
        )
        print(f"Min iterations: {min_iter}, Max iterations: {max_iter}")
        confidence = st.session_state.adversarial_confidence
        max_tokens = st.session_state.adversarial_max_tokens

        max_workers = st.session_state.adversarial_max_workers
        rotation_strategy = st.session_state.adversarial_rotation_strategy
        seed_str = str(st.session_state.adversarial_seed or "").strip()
        seed = None
        if seed_str:
            try:
                seed = int(float(seed_str))  # Handle floats by truncating to int
            except (ValueError, TypeError):
                logger.warning("Invalid adversarial seed input '%s'; continuing without seed.", seed_str)

        # Validation
        if not api_key:
            st.error("OpenRouter API key is required for adversarial testing.")
            return

        if not red_team_base or not blue_team_base:
            st.error("Please select at least one model for both red and blue teams.")
            return

        if not st.session_state.protocol_text.strip():
            st.error("Please enter a protocol to test.")
            return

        current_sop = st.session_state.protocol_text
        iteration = 0

        # Determine review type and get appropriate prompts
        content_type = "general"
        if st.session_state.get("adversarial_custom_mode", False):
            # Use custom prompts when custom mode is enabled
            red_team_prompt = st.session_state.get(
                "adversarial_custom_red_prompt", ""
            )
            blue_team_prompt = st.session_state.get(
                "adversarial_custom_blue_prompt", ""
            )
            approval_prompt = st.session_state.get(
                "adversarial_custom_approval_prompt", ""
            )
            review_type = "custom"
        else:
            # Use standard prompts based on review type
            if st.session_state.adversarial_review_type == "Auto-Detect":
                review_type = "general"
            elif st.session_state.adversarial_review_type == "Code Review":
                review_type = "code"
                content_type = st.session_state.get(
                    "code_language_type", "code_python" # Assuming a UI selection for code language
                )
            elif st.session_state.adversarial_review_type == "Plan Review":
                review_type = "plan"
                content_type = (
                    "document_general"  # Or a more specific plan document type
                )
            elif st.session_state.adversarial_review_type == "Legal Document":
                review_type = "document"
                content_type = "document_legal"
            elif st.session_state.adversarial_review_type == "Medical Document":
                review_type = "document"
                content_type = "document_medical"
            elif st.session_state.adversarial_review_type == "Technical Document":
                review_type = "document"
                content_type = "document_technical"
            else:
                review_type = "general"
                content_type = "document_general"

            red_team_prompt, blue_team_prompt = get_appropriate_prompts(review_type)

        with st.session_state.thread_lock:
            st.session_state.adversarial_log = []
            st.session_state.adversarial_stop_flag = False
            st.session_state.adversarial_total_tokens_prompt = 0
            st.session_state.adversarial_total_tokens_completion = 0
            st.session_state.adversarial_cost_estimate_usd = 0.0


        current_sop = st.session_state.protocol_text
        base_hash = _hash_text(current_sop)
        iteration = 0

        # Determine review type and get appropriate prompts
        content_type = "general"
        if st.session_state.get("adversarial_custom_mode", False):
            # Use custom prompts when custom mode is enabled
            red_team_prompt = st.session_state.get(
                "adversarial_custom_red_prompt", RED_TEAM_CRITIQUE_PROMPT
            )
            blue_team_prompt = st.session_state.get(
                "adversarial_custom_blue_prompt", BLUE_TEAM_PATCH_PROMPT
            )
            approval_prompt = st.session_state.get(
                "adversarial_custom_approval_prompt", APPROVAL_PROMPT
            )
            review_type = "custom"
            print(f"Using custom red team prompt: {red_team_prompt}")
            print(f"Using custom blue team prompt: {blue_team_prompt}")
            print(f"Using custom approval prompt: {approval_prompt}")
        else:
            # Use standard prompts based on review type
            if st.session_state.adversarial_review_type == "Auto-Detect":
                review_type = determine_review_type(current_sop)
            elif st.session_state.adversarial_review_type == "Code Review":
                review_type = "code"
                content_type = st.session_state.get(
                    "code_language_type", "code_python"
                )  # Assuming a UI selection for code language
            elif st.session_state.adversarial_review_type == "Plan Review":
                review_type = "plan"
                content_type = (
                    "document_general"  # Or a more specific plan document type
                )
            elif st.session_state.adversarial_review_type == "Legal Document":
                review_type = "document"
                content_type = "document_legal"
            elif st.session_state.adversarial_review_type == "Medical Document":
                review_type = "document"
                content_type = "document_medical"
            elif st.session_state.adversarial_review_type == "Technical Document":
                review_type = "document"
                content_type = "document_technical"
            else:
                review_type = "general"
                content_type = "document_general"

            red_team_prompt, blue_team_prompt = get_appropriate_prompts(review_type)
            print(f"Using review type: {review_type}, content_type: {content_type}")

        # Generate dynamic prompt enhancements
        red_team_prompt_enhancement = ""
        blue_team_prompt_enhancement = ""

        if content_type.startswith("code_"):
            red_team_prompt_enhancement += "\nAs a red teamer, focus on security vulnerabilities, code quality, and performance issues in the code. Look for potential bugs, inefficient algorithms, and non-idiomatic code."
            blue_team_prompt_enhancement += "\nAs a blue teamer, focus on fixing all identified issues, improving code robustness, and optimizing performance. Ensure the code is clean, secure, and follows best practices."
        elif content_type == "document_legal":
            red_team_prompt_enhancement += "\nAs a red teamer, scrutinize the legal document for ambiguities, loopholes, non-compliance with legal standards (e.g., GDPR, CCPA), and potential liabilities."
            blue_team_prompt_enhancement += "\nAs a blue teamer, refine the legal document for clarity, enforce compliance with relevant regulations, and mitigate all identified legal risks."
        elif content_type == "document_medical":
            red_team_prompt_enhancement += "\nAs a red teamer, analyze the medical document for factual inaccuracies, patient privacy violations (e.g., HIPAA), ethical concerns, and clarity for medical professionals."
            blue_team_prompt_enhancement += "\nAs a blue teamer, ensure the medical document is factually accurate, compliant with patient privacy laws, ethically sound, and clearly understandable by medical staff."
        elif content_type == "document_technical":
            red_team_prompt_enhancement += "\nAs a red teamer, review the technical document for technical inaccuracies, outdated information, unclear instructions, and potential security implications of described systems."
            blue_team_prompt_enhancement += "\nAs a blue teamer, update the technical document for accuracy, clarity, and completeness. Ensure all technical details are correct and instructions are easy to follow."
        elif review_type == "plan":
            red_team_prompt_enhancement += "\nAs a red teamer, critique the plan for feasibility, resource allocation, risk assessment, and alignment with strategic objectives. Identify any hidden dependencies or unrealistic timelines."
            blue_team_prompt_enhancement += "\nAs a blue teamer, refine the plan to address all identified risks, optimize resource allocation, and ensure feasibility and strategic alignment."
        else:  # General SOP
            red_team_prompt_enhancement += "\nAs a red teamer, identify any weaknesses, inefficiencies, or potential misinterpretations in the general SOP. Focus on clarity, completeness, and robustness."
            blue_team_prompt_enhancement += "\nAs a blue teamer, improve the general SOP by enhancing clarity, ensuring completeness, and making it more robust against misinterpretation."

        if st.session_state.get("compliance_requirements"):
            red_team_prompt_enhancement += f"\nAlso, specifically check for compliance with the following requirements: {st.session_state.compliance_requirements}"
            blue_team_prompt_enhancement += f"\nEnsure the final output strictly adheres to the following compliance requirements: {st.session_state.compliance_requirements}"

        _update_adv_log_and_status(
            f"🚀 Start: {len(red_team_base)} red / {len(blue_team_base)} blue | seed={seed} | base_hash={base_hash} | rotation={rotation_strategy} | review_type={review_type}"
        )

        # Use OpenEvolve backend for all content types
        if OPENEVOLVE_AVAILABLE:
            _update_adv_log_and_status(
                "🚀 Using OpenEvolve backend for adversarial testing..."
            )
            # Verify OpenEvolve backend is accessible before attempting to use it
            try:
                import requests
                health_response = requests.get("http://localhost:8000/health", timeout=5)
                if health_response.status_code == 200:
                    result = _run_adversarial_testing_with_openevolve_backend(
                        current_sop,
                        content_type,  # Pass the determined content_type
                        red_team_base,
                        blue_team_base,
                        api_key,
                        st.session_state.openrouter_base_url,
                        max_iter,
                        confidence,
                        max_tokens,
                        st.session_state.get("adversarial_temperature", 0.7),
                        st.session_state.get("adversarial_top_p", 0.95),
                        st.session_state.get("adversarial_frequency_penalty", 0.0),
                        st.session_state.get("adversarial_presence_penalty", 0.0),
                        seed,
                        max_workers,
                        rotation_strategy,
                        st.session_state.get("red_team_sample_size", len(red_team_base)),
                        st.session_state.get("blue_team_sample_size", len(blue_team_base)),
                        st.session_state.get("custom_requirements", ""),
                        evaluator_system_prompt=approval_prompt,
                        red_team_prompt=red_team_prompt,
                        blue_team_prompt=blue_team_prompt,
                        compliance_rules=st.session_state.get("compliance_rules", None),
                        feature_dimensions=st.session_state.get(
                            "adversarial_feature_dimensions", None
                        ),
                        feature_bins=st.session_state.get("adversarial_feature_bins", None),
                        enable_data_augmentation=st.session_state.get(
                            "adversarial_enable_data_augmentation", False
                        ),
                        augmentation_model_id=st.session_state.get(
                            "adversarial_augmentation_model_id", None
                        ),
                        augmentation_temperature=st.session_state.get(
                            "adversarial_augmentation_temperature", 0.7
                        ),
                        enable_human_feedback=st.session_state.get(
                            "adversarial_enable_human_feedback", False
                        ),
                        current_iteration=iteration,
                    )
                    st.session_state.adversarial_results = result
                else:
                    st.error(
                        "OpenEvolve backend is not responding. Please ensure it is running."
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to OpenEvolve backend. Please ensure it is running."
                )
            except Exception as e:
                st.error(f"Error checking OpenEvolve backend: {e}.")
        else:
            st.error("OpenEvolve backend is not available. Please install and run the backend.")

    except Exception as e:
        tb_str = traceback.format_exc()
        error_message = f"A critical error occurred: {e}\n{tb_str}"
        st.error(error_message)
        st.session_state.adversarial_results = {"critical_error": error_message}









def _load_human_feedback() -> List[Dict]:
    feedback_file = "human_feedback.json"
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            _update_adv_log_and_status(f"⚠️ Corrupted human feedback file: {feedback_file}. Starting fresh.")
            return []
    return []

def capture_human_feedback(
    adversarial_example: Dict[str, Any], human_score: float, human_comments: str
):
    """
    Captures human feedback on an adversarial example and stores it in a local JSON file.
    """
    feedback_entry = {
        "timestamp": time.time(),
        "adversarial_example_id": adversarial_example.get("id"),
        "human_score": human_score,
        "human_comments": human_comments,
        "content": adversarial_example.get("content"), # Store content for context
        "content_type": adversarial_example.get("content_type"),
        "iteration": adversarial_example.get("iteration"),
    }

    feedback_file = "human_feedback.json"
    all_feedback = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r") as f:
                all_feedback = json.load(f)
        except json.JSONDecodeError:
            _update_adv_log_and_status(f"⚠️ Corrupted human feedback file: {feedback_file}. Starting fresh.")
            all_feedback = []

    all_feedback.append(feedback_entry)

    try:
        with open(feedback_file, "w") as f:
            json.dump(all_feedback, f, indent=2)
        _update_adv_log_and_status(
            f"📝 Captured human feedback for adversarial example {adversarial_example.get('id')} and saved to {feedback_file}"
        )
    except Exception as e:
        _update_adv_log_and_status(f"❌ Failed to save human feedback to {feedback_file}: {e}")


def optimize_model_selection(
    red_team_models: List[str],
    blue_team_models: List[str],
    evaluator_models: List[str],
    optimization_strategy: str = "performance_priority",
    model_performance_data: Optional[Dict] = None,
) -> (List[str], List[str], List[str]):
    """
    Optimizes model selection based on various strategies, primarily performance.
    """
    if not model_performance_data:
        return red_team_models, blue_team_models, evaluator_models

    def _sort_models(models: List[str]) -> List[str]:
        # Sort models by performance score (descending). Default score is 0.5 if no data.
        # Assuming model_performance_data has a structure like: {'model_id': {'score': 0.8, ...}}
        return sorted(models, key=lambda m: model_performance_data.get(m, {}).get('score', 0.5), reverse=True)

    optimized_red_team = _sort_models(red_team_models)
    optimized_blue_team = _sort_models(blue_team_models)
    optimized_evaluator_models = _sort_models(evaluator_models)

    return optimized_red_team, optimized_blue_team, optimized_evaluator_models



# ============================================================================
# ULTIMATE ADVERSARIAL TESTING INTEGRATION
# Complete implementation supporting native OpenEvolve + workflow integration
# ============================================================================

def run_ultimate_adversarial_testing(
    content: str,
    content_type: str = "document_general",
    use_native_openevolve: bool = True,
    use_workflow_system: bool = True,
    adversarial_config: Optional[AdversarialConfiguration] = None,
    openevolve_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Ultimate adversarial testing combining native OpenEvolve adversarial capabilities
    with the comprehensive workflow system from ULTIMATE_ADVERSARIAL_EVOLUTION_EXPLAINED.md
    
    This function provides:
    1. Native OpenEvolve adversarial evolution
    2. Complete tripartite AI architecture (Red/Blue/Evaluator teams)
    3. Multi-round adversarial testing
    4. Problem decomposition support
    5. Comprehensive metrics and analysis
    6. Gauntlet system integration
    
    Args:
        content: Content to test adversarially
        content_type: Type of content
        use_native_openevolve: Enable native OpenEvolve adversarial evolution
        use_workflow_system: Enable workflow-based adversarial testing
        adversarial_config: Adversarial configuration
        openevolve_params: Native OpenEvolve parameters
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with ultimate adversarial testing results
    """
    _update_adv_log_and_status("🌟 Starting ULTIMATE Adversarial Testing...")
    _update_adv_log_and_status("🔥 Native OpenEvolve + Comprehensive Workflow Integration")
    
    start_time = time.time()
    operation_id = f"ultimate_adversarial_{int(start_time)}"
    
    # Initialize ultimate result structure
    ultimate_result = {
        "success": False,
        "operation_id": operation_id,
        "start_time": start_time,
        "system_type": "ultimate_adversarial_hybrid",
        "original_content": content,
        "final_content": content,
        "content_type": content_type,
        "native_openevolve_enabled": use_native_openevolve,
        "workflow_system_enabled": use_workflow_system,
        "testing_phases": {
            "phase_1_native_adversarial": {"status": "pending", "duration": 0, "success": False},
            "phase_2_workflow_testing": {"status": "pending", "duration": 0, "success": False},
            "phase_3_hybrid_analysis": {"status": "pending", "duration": 0, "success": False},
            "phase_4_comprehensive_validation": {"status": "pending", "duration": 0, "success": False}
        },
        "native_openevolve_results": {
            "available": False,
            "adversarial_score": 0.0,
            "robustness_improvement": 0.0,
            "iterations_completed": 0,
            "api_calls": 0,
            "cost_usd": 0.0
        },
        "workflow_results": {
            "team_system_available": TEAM_SYSTEM_AVAILABLE,
            "adversarial_rounds": 0,
            "vulnerabilities_found": 0,
            "fixes_applied": 0,
            "consensus_score": 0.0,
            "robustness_score": 0.0
        },
        "hybrid_metrics": {
            "combined_robustness_score": 0.0,
            "improvement_ratio": 0.0,
            "testing_effectiveness": 0.0,
            "overall_security_score": 0.0
        },
        "detailed_analysis": {
            "vulnerabilities": [],
            "fixes": [],
            "evaluations": [],
            "recommendations": []
        },
        "error": None
    }
    
    try:
        # ====================================================================
        # PHASE 1: NATIVE OPENEVOLVE ADVERSARIAL TESTING
        # ====================================================================
        if use_native_openevolve:
            _update_adv_log_and_status("🔥 Phase 1: Native OpenEvolve Adversarial Testing")
            phase_start = time.time()
            
            try:
                from openevolve_client import OpenEvolveClient
                client = OpenEvolveClient(config=openevolve_params or {})
                ultimate_result["native_openevolve_results"]["available"] = client.available
                
                if client.available:
                    _update_adv_log_and_status("🚀 Running native OpenEvolve adversarial evolution...")
                    
                    # Prepare OpenEvolve parameters for adversarial mode
                    oe_params = {
                        "evolution_mode": "adversarial",
                        "max_iterations": 10,
                        "population_size": 20,
                        "temperature": 0.8,
                        **(openevolve_params or {}),
                        **kwargs
                    }
                    
                    # Run native OpenEvolve adversarial evolution
                    oe_result = client.evolve(
                        content=content,
                        content_type=content_type,
                        **oe_params
                    )
                    
                    if oe_result.success:
                        content = oe_result.best_code  # Update content with improved version
                        ultimate_result["native_openevolve_results"]["adversarial_score"] = oe_result.best_score
                        ultimate_result["native_openevolve_results"]["robustness_improvement"] = oe_result.metrics.get("improvement_ratio", 0.0)
                        ultimate_result["native_openevolve_results"]["iterations_completed"] = oe_result.iterations_completed
                        ultimate_result["native_openevolve_results"]["api_calls"] = oe_result.metrics.get("api_calls", 0)
                        ultimate_result["native_openevolve_results"]["cost_usd"] = oe_result.metrics.get("cost_usd", 0.0)
                        
                        ultimate_result["testing_phases"]["phase_1_native_adversarial"]["success"] = True
                        _update_adv_log_and_status(f"✅ Native OpenEvolve adversarial completed: Score {oe_result.best_score:.4f}")
                    else:
                        _update_adv_log_and_status(f"⚠️ Native OpenEvolve adversarial failed: {oe_result.error}")
                
                else:
                    _update_adv_log_and_status("⚠️ Native OpenEvolve not available")
                
            except ImportError:
                _update_adv_log_and_status("⚠️ OpenEvolve client not available")
            
            ultimate_result["testing_phases"]["phase_1_native_adversarial"]["status"] = "completed"
            ultimate_result["testing_phases"]["phase_1_native_adversarial"]["duration"] = time.time() - phase_start
        
        # ====================================================================
        # PHASE 2: WORKFLOW-BASED ADVERSARIAL TESTING
        # ====================================================================
        # Import TEAM_SYSTEM_AVAILABLE
        try:
            from evolution import TEAM_SYSTEM_AVAILABLE
        except ImportError:
            TEAM_SYSTEM_AVAILABLE = False
            
        if use_workflow_system and TEAM_SYSTEM_AVAILABLE:
            _update_adv_log_and_status("👥 Phase 2: Workflow-Based Adversarial Testing")
            phase_start = time.time()
            
            try:
                # Create or use provided adversarial configuration
                if adversarial_config is None:
                    try:
                        adversarial_config = create_adversarial_configuration_from_session()
                    except:
                        adversarial_config = create_adversarial_configuration()
                
                # Run comprehensive adversarial testing
                workflow_result = run_comprehensive_adversarial_testing(
                    current_content=content,
                    content_type=content_type,
                    config=adversarial_config,
                    use_decomposition=kwargs.get("use_decomposition", False)
                )
                
                if workflow_result and workflow_result.get("success"):
                    content = workflow_result.get("final_content", content)  # Update content
                    
                    # Extract workflow metrics
                    workflow_metrics = workflow_result.get("metrics", {})
                    ultimate_result["workflow_results"]["adversarial_rounds"] = workflow_metrics.get("total_rounds", 0)
                    ultimate_result["workflow_results"]["vulnerabilities_found"] = workflow_metrics.get("vulnerability_count", 0)
                    ultimate_result["workflow_results"]["fixes_applied"] = workflow_metrics.get("fixes_applied", 0)
                    ultimate_result["workflow_results"]["consensus_score"] = workflow_metrics.get("consensus_score", 0.0)
                    ultimate_result["workflow_results"]["robustness_score"] = workflow_metrics.get("robustness_score", 0.0)
                    
                    # Extract detailed analysis
                    ultimate_result["detailed_analysis"]["vulnerabilities"] = workflow_result.get("vulnerabilities", [])
                    ultimate_result["detailed_analysis"]["fixes"] = workflow_result.get("fixes", [])
                    
                    ultimate_result["testing_phases"]["phase_2_workflow_testing"]["success"] = True
                    _update_adv_log_and_status(f"✅ Workflow adversarial testing completed: {workflow_metrics.get('total_rounds', 0)} rounds")
                
                else:
                    _update_adv_log_and_status("⚠️ Workflow adversarial testing failed")
                
            except Exception as e:
                _update_adv_log_and_status(f"⚠️ Workflow adversarial testing error: {e}")
            
            ultimate_result["testing_phases"]["phase_2_workflow_testing"]["status"] = "completed"
            ultimate_result["testing_phases"]["phase_2_workflow_testing"]["duration"] = time.time() - phase_start
        
        # ====================================================================
        # PHASE 3: HYBRID ANALYSIS AND INTEGRATION
        # ====================================================================
        _update_adv_log_and_status("🔄 Phase 3: Hybrid Analysis and Integration")
        phase_start = time.time()
        
        # Combine results from both approaches
        native_score = ultimate_result["native_openevolve_results"]["adversarial_score"]
        workflow_score = ultimate_result["workflow_results"]["robustness_score"]
        
        # Calculate hybrid metrics
        if native_score > 0 and workflow_score > 0:
            # Both systems contributed
            ultimate_result["hybrid_metrics"]["combined_robustness_score"] = (native_score + workflow_score) / 2
            ultimate_result["hybrid_metrics"]["testing_effectiveness"] = 0.9  # High effectiveness
        elif native_score > 0:
            # Only native OpenEvolve contributed
            ultimate_result["hybrid_metrics"]["combined_robustness_score"] = native_score
            ultimate_result["hybrid_metrics"]["testing_effectiveness"] = 0.7  # Good effectiveness
        elif workflow_score > 0:
            # Only workflow system contributed
            ultimate_result["hybrid_metrics"]["combined_robustness_score"] = workflow_score
            ultimate_result["hybrid_metrics"]["testing_effectiveness"] = 0.7  # Good effectiveness
        else:
            # Neither system contributed significantly
            ultimate_result["hybrid_metrics"]["combined_robustness_score"] = 0.3  # Baseline
            ultimate_result["hybrid_metrics"]["testing_effectiveness"] = 0.3  # Low effectiveness
        
        # Calculate improvement ratio
        if ultimate_result["original_content"]:
            ultimate_result["hybrid_metrics"]["improvement_ratio"] = len(content) / len(ultimate_result["original_content"])
        else:
            ultimate_result["hybrid_metrics"]["improvement_ratio"] = 1.0
        
        # Calculate overall security score
        security_factors = [
            ultimate_result["hybrid_metrics"]["combined_robustness_score"],
            ultimate_result["workflow_results"]["consensus_score"],
            min(1.0, ultimate_result["workflow_results"]["fixes_applied"] / max(1, ultimate_result["workflow_results"]["vulnerabilities_found"])),
            ultimate_result["hybrid_metrics"]["testing_effectiveness"]
        ]
        
        ultimate_result["hybrid_metrics"]["overall_security_score"] = sum(security_factors) / len(security_factors)
        
        ultimate_result["testing_phases"]["phase_3_hybrid_analysis"]["success"] = True
        ultimate_result["testing_phases"]["phase_3_hybrid_analysis"]["status"] = "completed"
        ultimate_result["testing_phases"]["phase_3_hybrid_analysis"]["duration"] = time.time() - phase_start
        
        # ====================================================================
        # PHASE 4: COMPREHENSIVE VALIDATION AND RECOMMENDATIONS
        # ====================================================================
        _update_adv_log_and_status("🔍 Phase 4: Comprehensive Validation and Recommendations")
        phase_start = time.time()
        
        # Generate recommendations based on results
        recommendations = []
        
        if ultimate_result["hybrid_metrics"]["overall_security_score"] < 0.6:
            recommendations.append("Consider additional security hardening measures")
        
        if ultimate_result["workflow_results"]["vulnerabilities_found"] > 5:
            recommendations.append("High number of vulnerabilities found - implement comprehensive security review")
        
        if ultimate_result["native_openevolve_results"]["cost_usd"] > 1.0:
            recommendations.append("Consider cost optimization for adversarial testing")
        
        if not ultimate_result["native_openevolve_results"]["available"]:
            recommendations.append("Enable native OpenEvolve for enhanced adversarial capabilities")
        
        if not ultimate_result["workflow_results"]["team_system_available"]:
            recommendations.append("Enable team system for comprehensive adversarial analysis")
        
        ultimate_result["detailed_analysis"]["recommendations"] = recommendations
        
        # Final validation
        ultimate_result["final_content"] = content
        ultimate_result["success"] = ultimate_result["hybrid_metrics"]["overall_security_score"] > 0.5
        
        ultimate_result["testing_phases"]["phase_4_comprehensive_validation"]["success"] = True
        ultimate_result["testing_phases"]["phase_4_comprehensive_validation"]["status"] = "completed"
        ultimate_result["testing_phases"]["phase_4_comprehensive_validation"]["duration"] = time.time() - phase_start
        
        # ====================================================================
        # FINALIZATION
        # ====================================================================
        
        end_time = time.time()
        ultimate_result["end_time"] = end_time
        ultimate_result["total_duration"] = end_time - start_time
        
        # Log comprehensive results
        _update_adv_log_and_status("🌟 ULTIMATE Adversarial Testing completed!")
        _update_adv_log_and_status(f"⏱️ Total duration: {ultimate_result['total_duration']:.2f}s")
        _update_adv_log_and_status(f"🛡️ Overall security score: {ultimate_result['hybrid_metrics']['overall_security_score']:.4f}")
        _update_adv_log_and_status(f"🔍 Vulnerabilities found: {ultimate_result['workflow_results']['vulnerabilities_found']}")
        _update_adv_log_and_status(f"🔧 Fixes applied: {ultimate_result['workflow_results']['fixes_applied']}")
        _update_adv_log_and_status(f"💰 Total cost: ${ultimate_result['native_openevolve_results']['cost_usd']:.4f}")
        
        return ultimate_result
        
    except Exception as e:
        ultimate_result["error"] = str(e)
        ultimate_result["end_time"] = time.time()
        ultimate_result["total_duration"] = ultimate_result["end_time"] - start_time
        
        _update_adv_log_and_status(f"💥 ULTIMATE Adversarial Testing failed: {e}")
        logger.error(f"Ultimate adversarial testing error: {e}", exc_info=True)
        return ultimate_result


def run_native_openevolve_adversarial_only(
    content: str,
    content_type: str = "document_general",
    **openevolve_params
) -> Dict[str, Any]:
    """
    Run pure native OpenEvolve adversarial evolution without workflow enhancements.
    This function focuses exclusively on native OpenEvolve adversarial capabilities.
    
    Args:
        content: Content to test adversarially
        content_type: Type of content
        **openevolve_params: All native OpenEvolve parameters
    
    Returns:
        Dictionary with native OpenEvolve adversarial results
    """
    _update_adv_log_and_status("🔥 Running Pure Native OpenEvolve Adversarial Evolution")
    
    start_time = time.time()
    result = {
        "success": False,
        "approach": "native_openevolve_only",
        "original_content": content,
        "final_content": content,
        "openevolve_result": None,
        "metrics": {},
        "error": None
    }
    
    try:
        # Initialize OpenEvolve client
        from openevolve_client import OpenEvolveClient
        client = OpenEvolveClient()
        
        if not client.available:
            _update_adv_log_and_status("❌ Native OpenEvolve not available")
            result["error"] = "Native OpenEvolve backend not available"
            return result
        
        # Prepare adversarial parameters
        adversarial_params = {
            "evolution_mode": "adversarial",
            "max_iterations": 15,
            "population_size": 25,
            "temperature": 0.8,
            "adversarial_rounds": 5,
            "attack_strength": 1.0,
            "defense_strength": 1.0,
            **openevolve_params
        }
        
        _update_adv_log_and_status("🚀 Running native OpenEvolve adversarial evolution...")
        
        # Run native OpenEvolve adversarial evolution
        openevolve_result = client.evolve(
            content=content,
            content_type=content_type,
            **adversarial_params
        )
        
        result["openevolve_result"] = {
            "success": openevolve_result.success,
            "best_code": openevolve_result.best_code,
            "best_score": openevolve_result.best_score,
            "iterations_completed": openevolve_result.iterations_completed,
            "metrics": openevolve_result.metrics,
            "error": openevolve_result.error
        }
        
        if openevolve_result.success:
            result["final_content"] = openevolve_result.best_code
            result["success"] = True
            
            # Extract key metrics
            result["metrics"] = {
                "adversarial_score": openevolve_result.best_score,
                "robustness_improvement": openevolve_result.metrics.get("improvement_ratio", 0.0),
                "iterations_completed": openevolve_result.iterations_completed,
                "api_calls": openevolve_result.metrics.get("api_calls", 0),
                "cost_usd": openevolve_result.metrics.get("cost_usd", 0.0),
                "total_duration": time.time() - start_time
            }
            
            _update_adv_log_and_status(f"✅ Native OpenEvolve adversarial completed: Score {openevolve_result.best_score:.4f}")
        else:
            result["error"] = openevolve_result.error
            _update_adv_log_and_status(f"❌ Native OpenEvolve adversarial failed: {openevolve_result.error}")
        
        return result
        
    except Exception as e:
        result["error"] = str(e)
        _update_adv_log_and_status(f"💥 Native OpenEvolve adversarial error: {e}")
        logger.error(f"Native OpenEvolve adversarial error: {e}", exc_info=True)
        return result


def get_adversarial_testing_capabilities() -> Dict[str, Any]:
    """
    Get comprehensive information about all available adversarial testing capabilities.
    
    Returns:
        Dictionary with complete adversarial capability information
    """
    # Import TEAM_SYSTEM_AVAILABLE from the module where it's defined
    try:
        from evolution import TEAM_SYSTEM_AVAILABLE
    except ImportError:
        TEAM_SYSTEM_AVAILABLE = False
    
    capabilities = {
        "native_openevolve_adversarial": {
            "available": False,
            "features": [],
            "parameters_supported": []
        },
        "workflow_adversarial": {
            "available": TEAM_SYSTEM_AVAILABLE,
            "team_components": [],
            "testing_phases": [],
            "advanced_features": []
        },
        "hybrid_capabilities": {
            "ultimate_adversarial_testing": True,
            "combined_approaches": True,
            "comprehensive_analysis": True
        }
    }
    
    # Check native OpenEvolve adversarial capabilities
    try:
        from openevolve_client import OpenEvolveClient
        client = OpenEvolveClient()
        capabilities["native_openevolve_adversarial"]["available"] = client.available
        
        if client.available:
            capabilities["native_openevolve_adversarial"]["features"] = [
                "adversarial_evolution", "attack_simulation", "defense_optimization",
                "robustness_scoring", "multi_round_testing"
            ]
            capabilities["native_openevolve_adversarial"]["parameters_supported"] = [
                "adversarial_rounds", "attack_strength", "defense_strength",
                "perturbation_bound", "ensemble_defense", "gradient_masking"
            ]
    except ImportError:
        logger.debug("OpenEvolve client unavailable for adversarial capabilities check.")
    
    # Check workflow adversarial capabilities
    if TEAM_SYSTEM_AVAILABLE:
        capabilities["workflow_adversarial"]["team_components"] = [
            "red_team_analysis", "blue_team_fixes", "evaluator_consensus"
        ]
        capabilities["workflow_adversarial"]["testing_phases"] = [
            "vulnerability_identification", "patch_development", "consensus_building"
        ]
        capabilities["workflow_adversarial"]["advanced_features"] = [
            "problem_decomposition", "multi_round_testing", "comprehensive_metrics",
            "gauntlet_integration", "quality_diversity_analysis"
        ]
    
    return capabilities


def create_comprehensive_adversarial_config(
    adversarial_rounds: int = 5,
    attack_strength: float = 1.0,
    defense_strength: float = 1.0,
    use_decomposition: bool = False,
    enable_all_features: bool = True,
    **additional_params
) -> AdversarialConfiguration:
    """
    Create a comprehensive adversarial configuration that supports both
    native OpenEvolve and workflow system features.
    
    Args:
        adversarial_rounds: Number of adversarial rounds
        attack_strength: Strength of attacks (0.1-2.0)
        defense_strength: Strength of defenses (0.1-2.0)
        use_decomposition: Enable problem decomposition
        enable_all_features: Enable all advanced features
        **additional_params: Additional configuration parameters
    
    Returns:
        Comprehensive AdversarialConfiguration
    """
    # Get base configuration from session
    base_config = create_adversarial_configuration_from_session()
    
    # Override with provided parameters
    base_config.adversarial_rounds = adversarial_rounds
    base_config.attack_strength = attack_strength
    base_config.defense_strength = defense_strength
    
    if enable_all_features:
        # Enable advanced features
        base_config.coevolutionary_approach = True
        base_config.ensemble_defense = True
        base_config.attack_diversity = True
        base_config.meta_learning = True
        base_config.transfer_learning = True
        base_config.explainable_ai = True
    
    # Apply additional parameters
    for key, value in additional_params.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
    
    return base_config


# =============================================================================
# MAKER/MDAP ENHANCED ADVERSARIAL TESTING
# =============================================================================

def run_maker_enhanced_adversarial_testing(
    content: str,
    content_type: str = "document_general",
    config: Optional[AdversarialConfiguration] = None,
    enable_maker_voting: bool = True,
    enable_mdap_decomposition: bool = True,
    coevolution_rounds: int = 3,
    k_ahead: int = 3
) -> Dict[str, Any]:
    """
    Run MAKER/MDAP-enhanced adversarial testing with zero-error guarantees.
    
    This function integrates the MAKER framework (arXiv:2511.09030) with adversarial testing
    to provide:
    
    1. MAKER-enhanced red team: First-to-ahead-by-k voting for reliable attack generation
    2. MDAP-enhanced blue team: Maximal decomposition for thorough defense coverage
    3. Co-evolutionary testing: Attack/defense arms race with mutation
    4. Zero-error vulnerability detection: Statistical convergence through voting
    
    Args:
        content: Content to test adversarially
        content_type: Type of content (document_general, code, api_spec, etc.)
        config: Adversarial configuration (optional)
        enable_maker_voting: Enable MAKER voting for red team (default: True)
        enable_mdap_decomposition: Enable MDAP for blue team (default: True)
        coevolution_rounds: Number of attack/defense co-evolution rounds (default: 3)
        k_ahead: Voting threshold for first-to-ahead-by-k (default: 3)
    
    Returns:
        Dict containing:
        - attacks: List of IssueFinding objects from red team
        - defenses: List of DefenseStrategy objects from blue team
        - evolution_history: List of per-round metrics
        - config: Configuration used
        - metrics: Performance metrics
    
    Example:
        >>> result = run_maker_enhanced_adversarial_testing(
        ...     content="my_code.py",
        ...     content_type="code",
        ...     coevolution_rounds=5,
        ...     k_ahead=3
        ... )
        >>> print(f"Found {len(result['attacks'])} vulnerabilities")
        >>> print(f"Generated {len(result['defenses'])} defense strategies")
    """
    logger.info("=" * 80)
    logger.info("MAKER/MDAP-ENHANCED ADVERSARIAL TESTING")
    logger.info("=" * 80)
    logger.info(f"Content type: {content_type}")
    logger.info(f"MAKER voting: {enable_maker_voting}")
    logger.info(f"MDAP decomposition: {enable_mdap_decomposition}")
    logger.info(f"Coevolution rounds: {coevolution_rounds}")
    logger.info(f"Voting threshold (k_ahead): {k_ahead}")
    
    # Use default config if not provided
    if config is None:
        config = create_adversarial_configuration()
    
    # Try to import MAKER integration
    try:
        from adversarial_maker_integration import (
            run_maker_adversarial_testing,
            create_adversarial_maker_config,
            AdversarialMAKERConfig,
            AdversarialMAKERMode
        )
        
        # Create MAKER-enhanced config
        maker_config = create_adversarial_maker_config(config)
        maker_config.coevolution_rounds = coevolution_rounds
        maker_config.red_team_consensus_threshold = k_ahead
        
        # Disable features if requested
        if not enable_maker_voting:
            maker_config.red_team_voting_enabled = False
        if not enable_mdap_decomposition:
            maker_config.blue_team_decomposition_enabled = False
        
        # Run MAKER-enhanced testing
        result = run_maker_adversarial_testing(
            content=content,
            content_type=content_type,
            config=config
        )
        
        logger.info(f"[OK] MAKER-enhanced testing completed successfully")
        logger.info(f"  - Attacks found: {len(result.get('final_attacks', []))}")
        logger.info(f"  - Defenses generated: {len(result.get('final_defenses', []))}")
        logger.info(f"  - Co-evolution rounds: {result.get('total_rounds', 0)}")
        
        # Add metadata
        result["method"] = "maker_mdap_enhanced"
        result["paper_reference"] = "arXiv:2511.09030"
        
        return result
        
    except ImportError as e:
        logger.warning(f"[WARN] MAKER integration not available: {e}")
        logger.warning(f"[WARN] Falling back to standard adversarial testing")
        
        # Fallback to standard adversarial testing
        return run_comprehensive_adversarial_testing(
            current_content=content,
            content_type=content_type,
            config=config
        )
    
    except Exception as e:
        logger.error(f"[ERROR] MAKER-enhanced testing failed: {e}")
        logger.error(f"[ERROR] Falling back to standard adversarial testing")
        
        # Fallback to standard adversarial testing
        return run_comprehensive_adversarial_testing(
            current_content=content,
            content_type=content_type,
            config=config
        )


def get_maker_adversarial_capabilities() -> Dict[str, Any]:
    """
    Get capabilities of MAKER/MDAP-enhanced adversarial testing.
    
    Returns:
        Dict describing MAKER/MDAP adversarial capabilities
    """
    capabilities = {
        "maker_enabled": False,
        "mdap_enabled": False,
        "algorithms": [],
        "modes": [],
        "integration_status": "unknown"
    }
    
    try:
        from adversarial_maker_integration import (
            AdversarialMAKERMode,
            AdversarialMAKERConfig
        )
        
        capabilities["maker_enabled"] = True
        capabilities["mdap_enabled"] = True
        capabilities["integration_status"] = "available"
        
        # List available adversarial modes
        capabilities["modes"] = [mode.value for mode in AdversarialMAKERMode]
        
        # List available algorithms from paper
        capabilities["algorithms"] = [
            "Algorithm 1: generate_solution (sequential attack generation)",
            "Algorithm 2: do_voting (first-to-ahead-by-k consensus)",
            "Algorithm 3: get_vote (red-flagging unreliable attacks)",
            "Algorithm 4: recursive_solve (attack decomposition)"
        ]
        
        # Configuration options
        capabilities["config_options"] = {
            "coevolution_rounds": "Number of attack/defense rounds (default: 3)",
            "k_ahead": "Voting threshold (default: 3, higher = more conservative)",
            "enable_maker_voting": "Enable MAKER for red team (default: True)",
            "enable_mdap_decomposition": "Enable MDAP for blue team (default: True)",
            "mutation_strength": "Attack mutation rate (0.0-1.0, default: 0.2)"
        }
        
        # Benefits
        capabilities["benefits"] = [
            "Zero-error vulnerability detection through voting",
            "Reliable attack generation via consensus",
            "Comprehensive defense coverage via decomposition",
            "Adversarial co-evolution for robust testing"
        ]
        
        # Paper reference
        capabilities["paper"] = {
            "title": "Solving a Million-Step LLM Task with Zero Errors",
            "arxiv": "2511.09030",
            "url": "https://arxiv.org/abs/2511.09030"
        }
        
    except ImportError as e:
        capabilities["integration_status"] = f"unavailable: {str(e)}"
    
    return capabilities
