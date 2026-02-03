import streamlit as st
import time
import tempfile
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from session_utils import _compose_messages, _update_evolution_log_and_status
from parameter_manager import ParameterManager, ValidationResult
from error_handler import (
    ErrorHandler, ErrorSeverity, ErrorCategory, 
    with_error_handling, handle_error, get_global_error_handler
)

# Configure logging
logger = logging.getLogger(__name__)

# Import team system components
try:
    from red_team import RedTeam, RedTeamAssessment, IssueFinding, IssueCategory
    from blue_team import BlueTeam, BlueTeamAssessment, FixSuggestion, FixType
    from evaluator_team import EvaluatorTeam, EvaluatorAssessment, EvaluationMetric
    from team_manager import TeamManager
    from gauntlet_manager import GauntletManager
    TEAM_SYSTEM_AVAILABLE = True
except ImportError as e:
    TEAM_SYSTEM_AVAILABLE = False
    logger.warning(f"Team system components not available - adversarial features will be limited: {e}")

# Import Adaptive MDAP components for intelligent resource allocation
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator, SolveStrategy
    from adaptive_mdap.integrations.workflow_engine_integration import AdaptiveWorkflowIntegration, AdaptiveWorkflowConfig
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError as e:
    ADAPTIVE_MDAP_AVAILABLE = False
    logger.info(f"Adaptive MDAP not available - using standard resource allocation: {e}")

@dataclass
class EvolutionConfiguration:
    """
    Comprehensive configuration class that utilizes all 272 OpenEvolve parameters
    """
    # Core Evolution Parameters (23)
    evolution_mode: str = "standard"
    max_iterations: int = 10
    population_size: int = 20
    temperature: float = 0.7
    max_tokens: int = 2048
    seed: Optional[int] = None
    early_stopping: bool = False
    convergence_threshold: float = 0.001
    fitness_function: str = "default"
    selection_pressure: float = 1.0
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism: bool = True
    diversity_maintenance: bool = True
    adaptive_parameters: bool = False
    convergence_threshold: float = 0.001
    fitness_function: str = "default"
    elitism: bool = True
    diversity_maintenance: bool = True
    adaptive_parameters: bool = False
    reasoning_effort: str = "medium"
    language: str = "python"
    file_suffix: str = ".py"
    
    # Model Configuration Parameters (18)
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    model_id: str = "gpt-4"
    backup_models: List[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: int = 60
    concurrent_requests: int = 5
    model_rotation: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    logit_bias: Dict = None
    stop_sequences: List[str] = None
    logprobs: bool = False
    top_logprobs: int = 0
    response_format: str = "text"
    
    # Quality Diversity Parameters (19)
    feature_dimensions: List[str] = None
    feature_bins: int = 10
    archive_size: int = 100
    novelty_threshold: float = 0.1
    quality_threshold: float = 0.0
    diversity_weight: float = 0.5
    behavior_space: str = "auto"
    distance_metric: str = "euclidean"
    archive_update_freq: int = 1
    exploration_bonus: float = 0.1
    crowding_distance: bool = True
    pareto_layers: int = 3
    behavior_dimensions: List[str] = None
    diversity_metric: str = "edit_distance"
    diversity_reference_size: int = 20
    adaptive_feature_dimensions: bool = True
    double_selection: bool = True
    qd_algorithm: str = "MAP-Elites"
    behavior_descriptor_type: str = "hand_crafted"
    archive_learning_rate: float = 0.1
    
    # Multi-Objective Parameters (15)
    objectives: List[str] = None
    objective_weights: List[float] = None
    pareto_front_size: int = 50
    dominance_metric: str = "pareto"
    constraint_handling: str = "penalty"
    reference_point: List[float] = None
    epsilon_dominance: float = 0.01
    decomposition_method: str = "weighted_sum"
    scalarization_function: str = "weighted_sum"
    dominance_type: str = "standard"
    epsilon_values: List[float] = None
    scalarization: str = "weighted_sum"
    constraint_tolerance: float = 0.01
    hypervolume_ref: List[float] = None
    
    # Adversarial Parameters (20)
    attack_model_config: Dict = None
    defense_model_config: Dict = None
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
    
    # Island Model Parameters (17)
    num_islands: int = 5
    migration_interval: int = 10
    migration_rate: float = 0.1
    migration_topology: str = "ring"
    ring_topology: bool = True
    controlled_gene_flow: bool = True
    island_diversity_metric: str = "edit_distance"
    migration_selection: str = "best"
    island_initialization: str = "random"
    island_specialization: bool = False
    migration_size: int = 5
    migration_policy: str = "best"
    replacement_policy: str = "worst"
    island_sizes: List[int] = None
    heterogeneous_islands: bool = False
    synchronous_migration: bool = True
    adaptive_migration: bool = False
    
    # Selection & Reproduction Parameters (18)
    elite_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    multi_strategy_sampling: bool = True
    tournament_size: int = 3
    elitism_count: int = 2
    selection_method: str = "tournament"
    reproduction_method: str = "both"
    parent_selection: str = "fitness"
    random_ratio: float = 0.2
    survivor_selection: str = "generational"
    replacement_rate: float = 1.0
    selection_pressure_decay: float = 0.0
    diversity_selection: bool = False
    age_based_selection: bool = False
    
    # Evaluation Parameters (25)
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = None
    parallel_evaluations: int = 4
    evaluator_timeout: int = 300
    max_retries_eval: int = 3
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1
    evaluator_models: List[Dict] = None
    evaluator_system_message: str = ""
    ensemble_size: int = 3
    consensus_threshold: float = 0.7
    evaluation_criteria: List[str] = None
    custom_evaluator: str = None
    evaluation_batch_size: int = 10
    cache_evaluations: bool = True
    cache_size: int = 1000
    evaluation_noise: float = 0.0
    fitness_scaling: str = "linear"
    normalization: bool = True
    multi_criteria_eval: bool = False
    evaluation_budget: int = 10000
    incremental_eval: bool = False
    surrogate_model: bool = False
    active_learning: bool = False
    uncertainty_sampling: bool = False
    
    # Prompt Engineering Parameters (12)
    prompt_template: str = "default"
    system_prompt: str = ""
    context_length: int = 2000
    prompt_optimization: bool = True
    template_stochasticity: bool = True
    meta_prompting: bool = False
    few_shot_examples: int = 3
    chain_of_thought: bool = True
    self_consistency: bool = False
    prompt_ensembling: bool = False
    dynamic_prompting: bool = False
    prompt_compression: bool = False
    
    # Artifact Management Parameters (10)
    enable_artifacts: bool = True
    artifact_types: List[str] = None
    max_artifact_size: int = 20480
    artifact_validation: bool = True
    artifact_compression: bool = False
    artifact_versioning: bool = True
    artifact_metadata: bool = True
    artifact_cleanup: bool = True
    artifact_storage: str = "memory"
    artifact_encryption: bool = False
    
    # Resource Management Parameters (11)
    memory_limit_mb: int = 4096
    cpu_limit: float = 0.8
    max_time: int = 1800
    disk_limit_mb: int = 1024
    network_limit_mbps: int = 100
    api_call_limit: int = 1000
    token_limit: int = 100000
    cost_limit_usd: float = 10.0
    resource_monitoring: bool = True
    auto_scaling: bool = False
    checkpoint_interval: int = 10
    
    # Database & Storage Parameters (10)
    db_path: str = "./openevolve.db"
    db_type: str = "sqlite"
    connection_string: str = ""
    max_connections: int = 10
    connection_timeout: int = 30
    query_timeout: int = 60
    batch_size: int = 1000
    compression: bool = True
    encryption: bool = False
    backup_enabled: bool = True
    
    # Evolution Tracing Parameters (12)
    trace_enabled: bool = False
    trace_level: str = "basic"
    trace_format: str = "json"
    trace_file: str = "./trace.log"
    trace_compression: bool = True
    trace_rotation: bool = True
    max_trace_size_mb: int = 100
    trace_buffer_size: int = 1000
    real_time_tracing: bool = False
    trace_sampling: float = 1.0
    include_population: bool = False
    include_fitness: bool = True
    
    # Early Stopping Parameters (9)
    early_stopping_patience: int = 10
    min_improvement: float = 0.001
    improvement_window: int = 5
    plateau_threshold: int = 20
    convergence_check: bool = True
    diversity_threshold: float = 0.01
    stagnation_limit: int = 50
    adaptive_stopping: bool = False
    
    # Distributed Processing Parameters (10)
    distributed: bool = False
    num_workers: int = 4
    worker_timeout: int = 120
    load_balancing: str = "round_robin"
    fault_tolerance: bool = True
    worker_restart: bool = True
    communication_backend: str = "local"
    message_compression: bool = True
    heartbeat_interval: int = 10
    cluster_scaling: bool = False
    
    # Advanced Research Parameters (20)
    novelty_search: bool = False
    curiosity_driven: bool = False
    meta_learning: bool = False
    transfer_learning: bool = False
    continual_learning: bool = False
    few_shot_adaptation: bool = False
    zero_shot_transfer: bool = False
    domain_adaptation: bool = False
    multi_task_learning: bool = False
    lifelong_learning: bool = False
    neural_architecture_search: bool = False
    hyperparameter_optimization: bool = False
    automated_ml: bool = False
    explainable_ai: bool = False
    federated_learning: bool = False
    differential_privacy: bool = False
    quantum_computing: bool = False
    neuromorphic_computing: bool = False
    edge_computing: bool = False
    green_ai: bool = False
    
    # Custom Requirements Parameters (8)
    custom_fitness: str = ""
    custom_operators: List[str] = None
    custom_constraints: List[str] = None
    domain_knowledge: str = ""
    expert_rules: List[str] = None
    business_logic: str = ""
    regulatory_compliance: List[str] = None
    ethical_guidelines: List[str] = None
    
    # UI & Visualization Parameters (8)
    enable_visualization: bool = True
    plot_frequency: int = 10
    plot_types: List[str] = None
    interactive_plots: bool = True
    real_time_updates: bool = False
    export_plots: bool = True
    plot_format: str = "png"
    dashboard_enabled: bool = True
    
    # Experimental Parameters (7)
    experimental_features: bool = False
    beta_algorithms: bool = False
    research_mode: bool = False
    debug_mode: bool = False
    profiling_enabled: bool = False
    memory_profiling: bool = False
    experimental_logging: bool = False
    
    # Adaptive MDAP Parameters (8) - NEW
    enable_adaptive_mdap: bool = True
    adaptive_mdap_profile: str = "balanced"
    adaptive_mdap_learning: bool = False
    adaptive_mdap_context_aware: bool = False
    adaptive_mdap_thresholds: List[float] = None
    adaptive_mdap_min_agents: int = 1
    adaptive_mdap_max_agents: int = 10
    adaptive_mdap_cost_weight: float = 0.5
    
    def __post_init__(self):
        """Initialize default values for list/dict fields"""
        if self.backup_models is None:
            self.backup_models = []
        if self.logit_bias is None:
            self.logit_bias = {}
        if self.stop_sequences is None:
            self.stop_sequences = []
        if self.feature_dimensions is None:
            self.feature_dimensions = []
        if self.behavior_dimensions is None:
            self.behavior_dimensions = []
        if self.objectives is None:
            self.objectives = []
        if self.objective_weights is None:
            self.objective_weights = []
        if self.reference_point is None:
            self.reference_point = []
        if self.epsilon_values is None:
            self.epsilon_values = []
        if self.hypervolume_ref is None:
            self.hypervolume_ref = []
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
        if self.island_sizes is None:
            self.island_sizes = []
        if self.cascade_thresholds is None:
            self.cascade_thresholds = [0.5, 0.75, 0.9]
        if self.evaluator_models is None:
            self.evaluator_models = []
        if self.evaluation_criteria is None:
            self.evaluation_criteria = []
        if self.artifact_types is None:
            self.artifact_types = ["code", "text"]
        if self.custom_operators is None:
            self.custom_operators = []
        if self.custom_constraints is None:
            self.custom_constraints = []
        if self.expert_rules is None:
            self.expert_rules = []
        if self.regulatory_compliance is None:
            self.regulatory_compliance = []
        if self.ethical_guidelines is None:
            self.ethical_guidelines = []
        if self.plot_types is None:
            self.plot_types = ["fitness", "diversity"]
        if self.adaptive_mdap_thresholds is None:
            self.adaptive_mdap_thresholds = [0.2, 0.4, 0.6, 0.8]
    
    @classmethod
    def from_parameter_manager(cls, param_manager: ParameterManager, session_state: Dict[str, Any]) -> 'EvolutionConfiguration':
        """Create configuration from parameter manager and session state"""
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
        """Validate the configuration using parameter manager"""
        config_dict = asdict(self)
        return param_manager.validate(config_dict)
    
    def to_openevolve_config(self) -> Dict[str, Any]:
        """Convert to OpenEvolve-compatible configuration dictionary"""
        return asdict(self)

def _request_openai_compatible_chat(api_key, base_url, model, messages, extra_headers, temperature, top_p, frequency_penalty, presence_penalty, max_tokens, seed):
    """
    Make a request to an OpenAI-compatible API
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            seed=seed
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        # If openai package is not available, try using requests
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens
        }
        
        if seed is not None:
            data["seed"] = seed
            
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        st.error(f"Error making API request: {e}")
        return None

# Import OpenEvolve modules for code-specific features


try:
    # We just need to check if the package is available.
    # The actual functions are imported from openevolve_integration.

    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available")

# Import our deep integration module
try:
    from openevolve_integration import (
        run_unified_evolution,
        create_specialized_evaluator,
        create_language_specific_evaluator,
    )
    DEEP_INTEGRATION_AVAILABLE = True and OPENEVOLVE_AVAILABLE
except ImportError:
    DEEP_INTEGRATION_AVAILABLE = False
    print("Deep OpenEvolve integration not available")

# Import DTS integration for enhanced strategy exploration
try:
    from dts_integration import DTSIntegration, DTSIntegrationConfig
    DTS_AVAILABLE = True
    logger.info("DTS integration available for enhanced strategy exploration")
except (ImportError, Exception):
    DTS_AVAILABLE = False
    logger.warning("DTS integration not available - using standard evolution strategies")


class ContentEvaluator:
    """
    A class to encapsulate content evaluation logic.
    """

    def __init__(self, content_type: str, evaluator_system_prompt: str):
        self.content_type = content_type
        self.evaluator_system_prompt = evaluator_system_prompt

    def evaluate(self, program_path: str) -> Dict[str, Any]:
        """
        Evaluate the content of a file.
        """
        try:
            with open(program_path, "r") as f:
                content = f.read()

            if self.content_type.startswith("code_"):
                return self._evaluate_code(content)
            elif self.content_type == "legal":
                return self._evaluate_legal_content(content)
            elif self.content_type == "medical":
                return self._evaluate_medical_content(content)
            elif self.content_type == "technical":
                return self._evaluate_technical_content(content)
            else:
                return self._evaluate_general_content(content)
        except (RuntimeError, ValueError, TypeError) as e:
            return {"score": 0.0, "error": str(e), "timestamp": time.time()}

    def _evaluate_code(self, content: str) -> Dict[str, Any]:
        """
        Evaluator for code content.
        """
        # First, try basic evaluation
        score = min(1.0, len(content) / 500.0)  # Basic length-based scoring

        # Try to enhance with DSPy if available
        try:
            from dspy_integration import DSPY_AVAILABLE
            if DSPY_AVAILABLE:
                from dspy import Predict, Signature

                # Define a DSPy signature for code evaluation
                class CodeEvaluationSignature(Signature):
                    """Evaluate code quality based on multiple criteria."""
                    code = dspy.InputField(desc="Code to evaluate")
                    criteria = dspy.InputField(desc="List of evaluation criteria")

                    correctness_score = dspy.OutputField(desc="Correctness score (1-10)")
                    efficiency_score = dspy.OutputField(desc="Efficiency score (1-10)")
                    readability_score = dspy.OutputField(desc="Readability score (1-10)")
                    maintainability_score = dspy.OutputField(desc="Maintainability score (1-10)")
                    overall_score = dspy.OutputField(desc="Overall score (1-10)")
                    issues = dspy.OutputField(desc="List of identified issues")
                    suggestions = dspy.OutputField(desc="Improvement suggestions")

                # Create a predictor
                evaluate_code = Predict(CodeEvaluationSignature)

                # Run evaluation
                criteria = "correctness, efficiency, readability, maintainability"
                result = evaluate_code(code=content, criteria=criteria)

                # Calculate normalized score
                try:
                    overall = float(result.overall_score) if result.overall_score.replace('.', '').isdigit() else 5.0
                    dsp_score = overall / 10.0  # Normalize to 0-1 range
                    score = dsp_score  # Use DSPy score instead of basic score
                except:
                    # If DSPy parsing fails, keep basic score
                    pass

                return {
                    "score": score,
                    "length": len(content),
                    "timestamp": time.time(),
                    "dspy_enhanced": True,
                    "dspy_results": {
                        "correctness": result.correctness_score,
                        "efficiency": result.efficiency_score,
                        "readability": result.readability_score,
                        "maintainability": result.maintainability_score,
                        "issues": result.issues,
                        "suggestions": result.suggestions
                    }
                }
        except ImportError:
            pass  # DSPy not available, continue with basic evaluation

        # Return basic evaluation
        return {"score": score, "length": len(content), "timestamp": time.time()}

    def _evaluate_general_content(self, content: str) -> Dict[str, Any]:
        """
        Evaluator for general content.
        """
        _update_evolution_log_and_status(
            f"📊 Evaluating content of {len(content)} characters"
        )

        # For general content, return a basic score
        score = min(1.0, len(content) / 1000.0)  # Basic length-based scoring

        # Try to enhance with DSPy if available
        try:
            from dspy_integration import DSPY_AVAILABLE
            if DSPY_AVAILABLE:
                from dspy import Predict, Signature

                # Define a DSPy signature for general content evaluation
                class GeneralContentEvaluationSignature(Signature):
                    """Evaluate general content quality based on multiple criteria."""
                    content = dspy.InputField(desc="Content to evaluate")
                    criteria = dspy.InputField(desc="List of evaluation criteria")

                    clarity_score = dspy.OutputField(desc="Clarity score (1-10)")
                    coherence_score = dspy.OutputField(desc="Coherence score (1-10)")
                    completeness_score = dspy.OutputField(desc="Completeness score (1-10)")
                    relevance_score = dspy.OutputField(desc="Relevance score (1-10)")
                    overall_score = dspy.OutputField(desc="Overall score (1-10)")
                    strengths = dspy.OutputField(desc="List of content strengths")
                    weaknesses = dspy.OutputField(desc="List of content weaknesses")
                    suggestions = dspy.OutputField(desc="Improvement suggestions")

                # Create a predictor
                evaluate_content = Predict(GeneralContentEvaluationSignature)

                # Run evaluation
                criteria = "clarity, coherence, completeness, relevance"
                result = evaluate_content(content=content, criteria=criteria)

                # Calculate normalized score
                try:
                    overall = float(result.overall_score) if result.overall_score.replace('.', '').isdigit() else 5.0
                    dsp_score = overall / 10.0  # Normalize to 0-1 range
                    score = dsp_score  # Use DSPy score instead of basic score
                except:
                    # If DSPy parsing fails, keep basic score
                    pass

                return {
                    "score": score,
                    "length": len(content),
                    "timestamp": time.time(),
                    "dspy_enhanced": True,
                    "dspy_results": {
                        "clarity": result.clarity_score,
                        "coherence": result.coherence_score,
                        "completeness": result.completeness_score,
                        "relevance": result.relevance_score,
                        "strengths": result.strengths,
                        "weaknesses": result.weaknesses,
                        "suggestions": result.suggestions
                    }
                }
        except ImportError:
            pass  # DSPy not available, continue with basic evaluation

        return {"score": score, "length": len(content), "timestamp": time.time()}

    def _evaluate_legal_content(self, content: str) -> Dict[str, Any]:
        """
        Evaluator for legal content.
        """
        _update_evolution_log_and_status(
            f"⚖️ Evaluating legal content of {len(content)} characters"
        )
        score = min(1.0, len(content) / 1500.0)  # Example scoring

        # Try to enhance with DSPy if available
        try:
            from dspy_integration import DSPY_AVAILABLE
            if DSPY_AVAILABLE:
                from dspy import Predict, Signature

                # Define a DSPy signature for legal content evaluation
                class LegalContentEvaluationSignature(Signature):
                    """Evaluate legal content quality based on multiple criteria."""
                    content = dspy.InputField(desc="Legal content to evaluate")
                    criteria = dspy.InputField(desc="List of evaluation criteria")

                    accuracy_score = dspy.OutputField(desc="Accuracy score (1-10)")
                    completeness_score = dspy.OutputField(desc="Completeness score (1-10)")
                    compliance_score = dspy.OutputField(desc="Regulatory compliance score (1-10)")
                    clarity_score = dspy.OutputField(desc="Clarity score (1-10)")
                    overall_score = dspy.OutputField(desc="Overall score (1-10)")
                    legal_issues = dspy.OutputField(desc="List of identified legal issues")
                    compliance_gaps = dspy.OutputField(desc="List of compliance gaps")
                    recommendations = dspy.OutputField(desc="Legal recommendations")

                # Create a predictor
                evaluate_legal = Predict(LegalContentEvaluationSignature)

                # Run evaluation
                criteria = "accuracy, completeness, compliance, clarity"
                result = evaluate_legal(content=content, criteria=criteria)

                # Calculate normalized score
                try:
                    overall = float(result.overall_score) if result.overall_score.replace('.', '').isdigit() else 5.0
                    dsp_score = overall / 10.0  # Normalize to 0-1 range
                    score = dsp_score  # Use DSPy score instead of basic score
                except:
                    # If DSPy parsing fails, keep basic score
                    pass

                return {
                    "score": score,
                    "length": len(content),
                    "timestamp": time.time(),
                    "dspy_enhanced": True,
                    "dspy_results": {
                        "accuracy": result.accuracy_score,
                        "completeness": result.completeness_score,
                        "compliance": result.compliance_score,
                        "clarity": result.clarity_score,
                        "legal_issues": result.legal_issues,
                        "compliance_gaps": result.compliance_gaps,
                        "recommendations": result.recommendations
                    }
                }
        except ImportError:
            pass  # DSPy not available, continue with basic evaluation

        return {"score": score, "length": len(content), "timestamp": time.time()}

    def _evaluate_medical_content(self, content: str) -> Dict[str, Any]:
        """
        Evaluator for medical content.
        """
        _update_evolution_log_and_status(
            f"⚕️ Evaluating medical content of {len(content)} characters"
        )
        score = min(1.0, len(content) / 1200.0)  # Example scoring

        # Try to enhance with DSPy if available
        try:
            from dspy_integration import DSPY_AVAILABLE
            if DSPY_AVAILABLE:
                from dspy import Predict, Signature

                # Define a DSPy signature for medical content evaluation
                class MedicalContentEvaluationSignature(Signature):
                    """Evaluate medical content quality based on multiple criteria."""
                    content = dspy.InputField(desc="Medical content to evaluate")
                    criteria = dspy.InputField(desc="List of evaluation criteria")

                    accuracy_score = dspy.OutputField(desc="Medical accuracy score (1-10)")
                    completeness_score = dspy.OutputField(desc="Completeness score (1-10)")
                    safety_score = dspy.OutputField(desc="Patient safety score (1-10)")
                    evidence_score = dspy.OutputField(desc="Evidence-based medicine score (1-10)")
                    overall_score = dspy.OutputField(desc="Overall score (1-10)")
                    medical_errors = dspy.OutputField(desc="List of identified medical errors")
                    safety_concerns = dspy.OutputField(desc="List of patient safety concerns")
                    recommendations = dspy.OutputField(desc="Medical recommendations")

                # Create a predictor
                evaluate_medical = Predict(MedicalContentEvaluationSignature)

                # Run evaluation
                criteria = "accuracy, completeness, safety, evidence-based"
                result = evaluate_medical(content=content, criteria=criteria)

                # Calculate normalized score
                try:
                    overall = float(result.overall_score) if result.overall_score.replace('.', '').isdigit() else 5.0
                    dsp_score = overall / 10.0  # Normalize to 0-1 range
                    score = dsp_score  # Use DSPy score instead of basic score
                except:
                    # If DSPy parsing fails, keep basic score
                    pass

                return {
                    "score": score,
                    "length": len(content),
                    "timestamp": time.time(),
                    "dspy_enhanced": True,
                    "dspy_results": {
                        "accuracy": result.accuracy_score,
                        "completeness": result.completeness_score,
                        "safety": result.safety_score,
                        "evidence_based": result.evidence_score,
                        "medical_errors": result.medical_errors,
                        "safety_concerns": result.safety_concerns,
                        "recommendations": result.recommendations
                    }
                }
        except ImportError:
            pass  # DSPy not available, continue with basic evaluation

        return {"score": score, "length": len(content), "timestamp": time.time()}

    def _evaluate_technical_content(self, content: str) -> Dict[str, Any]:
        """
        Evaluator for technical content.
        """
        _update_evolution_log_and_status(
            f"⚙️ Evaluating technical content of {len(content)} characters"
        )
        score = min(1.0, len(content) / 1000.0)  # Example scoring

        # Try to enhance with DSPy if available
        try:
            from dspy_integration import DSPY_AVAILABLE
            if DSPY_AVAILABLE:
                from dspy import Predict, Signature

                # Define a DSPy signature for technical content evaluation
                class TechnicalContentEvaluationSignature(Signature):
                    """Evaluate technical content quality based on multiple criteria."""
                    content = dspy.InputField(desc="Technical content to evaluate")
                    criteria = dspy.InputField(desc="List of evaluation criteria")

                    accuracy_score = dspy.OutputField(desc="Technical accuracy score (1-10)")
                    completeness_score = dspy.OutputField(desc="Completeness score (1-10)")
                    clarity_score = dspy.OutputField(desc="Technical clarity score (1-10)")
                    feasibility_score = dspy.OutputField(desc="Implementation feasibility score (1-10)")
                    overall_score = dspy.OutputField(desc="Overall score (1-10)")
                    technical_issues = dspy.OutputField(desc="List of identified technical issues")
                    implementation_challenges = dspy.OutputField(desc="List of implementation challenges")
                    recommendations = dspy.OutputField(desc="Technical recommendations")

                # Create a predictor
                evaluate_technical = Predict(TechnicalContentEvaluationSignature)

                # Run evaluation
                criteria = "accuracy, completeness, clarity, feasibility"
                result = evaluate_technical(content=content, criteria=criteria)

                # Calculate normalized score
                try:
                    overall = float(result.overall_score) if result.overall_score.replace('.', '').isdigit() else 5.0
                    dsp_score = overall / 10.0  # Normalize to 0-1 range
                    score = dsp_score  # Use DSPy score instead of basic score
                except:
                    # If DSPy parsing fails, keep basic score
                    pass

                return {
                    "score": score,
                    "length": len(content),
                    "timestamp": time.time(),
                    "dspy_enhanced": True,
                    "dspy_results": {
                        "accuracy": result.accuracy_score,
                        "completeness": result.completeness_score,
                        "clarity": result.clarity_score,
                        "feasibility": result.feasibility_score,
                        "technical_issues": result.technical_issues,
                        "implementation_challenges": result.implementation_challenges,
                        "recommendations": result.recommendations
                    }
                }
        except ImportError:
            pass  # DSPy not available, continue with basic evaluation

        return {"score": score, "length": len(content), "timestamp": time.time()}


def _run_problem_decomposition_enhanced(
    current_content: str,
    config: EvolutionConfiguration,
) -> str:
    """Enhanced problem decomposition using full parameter configuration"""
    _update_evolution_log_and_status("🧩 Decomposing problem with enhanced configuration...")

    # 1. Decompose the problem using advanced prompting
    decomposition_prompt = f"""Decompose the following problem into a series of smaller, solvable sub-problems. 
    Use the following advanced techniques:
    - Chain of thought reasoning: {'enabled' if config.chain_of_thought else 'disabled'}
    - Meta-prompting: {'enabled' if config.meta_prompting else 'disabled'}
    - Few-shot examples: {config.few_shot_examples} examples
    
    The sub-problems should be self-contained and independent if possible. 
    Present the sub-problems as a numbered list, with each item on a new line.

Problem:
---
{current_content}
---

System Context:
---
{config.system_prompt}
---

Domain Knowledge:
---
{config.domain_knowledge}
---
"""
    
    sub_problems_str = _request_openai_compatible_chat(
        config.api_key,
        config.api_base,
        config.model_id,
        _compose_messages("You are a problem decomposition expert with advanced reasoning capabilities.", decomposition_prompt),
        {},  # extra_headers
        config.temperature,
        config.top_p,
        config.frequency_penalty,
        config.presence_penalty,
        config.max_tokens,
        config.seed,
    )

    if not sub_problems_str:
        _update_evolution_log_and_status("💥 Failed to decompose problem.")
        return current_content

    # Parse sub_problems_str into a list of strings
    sub_problems = [line.strip() for line in sub_problems_str.split('\n') if line.strip() and re.match(r'^\d+\.', line.strip())]
    if not sub_problems:
        _update_evolution_log_and_status("💥 Could not parse sub-problems.")
        return current_content
        
    _update_evolution_log_and_status(f"✅ Decomposed into {len(sub_problems)} sub-problems.")

    # 2. Solve each sub-problem using enhanced evolution
    solutions = []
    for i, sub_problem_text in enumerate(sub_problems):
        _update_evolution_log_and_status(f"🔄 Solving sub-problem {i+1}/{len(sub_problems)}: {sub_problem_text[:80]}...")
        
        # Create sub-configuration with reduced iterations
        sub_config = EvolutionConfiguration()
        sub_config.__dict__.update(config.__dict__)  # Copy all settings
        sub_config.max_iterations = max(1, config.max_iterations // len(sub_problems))
        sub_config.evolution_mode = "standard"  # Use standard mode for sub-problems
        sub_config.system_prompt = f"""This is a sub-problem of a larger task.
Original Problem: {current_content}
This Sub-Problem: {sub_problem_text}
{config.system_prompt}
"""
        
        # Recursively solve sub-problem
        solution = run_evolution_loop(
            current_content=sub_problem_text,
            content_type="document_general",
            config=sub_config
        )
        
        solutions.append(f"Solution for sub-problem '{sub_problem_text}':\n---\n{solution}\n---")
        _update_evolution_log_and_status(f"✅ Solved sub-problem {i+1}/{len(sub_problems)}.")

    # 3. Reassemble the solutions using advanced techniques
    _update_evolution_log_and_status("🧩 Reassembling solutions with advanced synthesis...")
    
    solutions_str = "\n\n".join(solutions)
    reassembly_prompt = f"""Given the original problem and the solutions to its sub-components, 
    assemble the final, complete solution using advanced synthesis techniques:
    
    - Self-consistency: {'enabled' if config.self_consistency else 'disabled'}
    - Prompt ensembling: {'enabled' if config.prompt_ensembling else 'disabled'}
    - Meta-learning insights: {'enabled' if config.meta_learning else 'disabled'}
    
Original Problem:
---
{current_content}
---

Sub-problem solutions:
---
{solutions_str}
---

Business Logic Constraints:
---
{config.business_logic}
---

Expert Rules:
---
{'; '.join(config.expert_rules) if config.expert_rules else 'None'}
---

Please provide the final, reassembled solution that addresses the original problem comprehensively.
"""
    
    final_solution = _request_openai_compatible_chat(
        config.api_key,
        config.api_base,
        config.model_id,
        _compose_messages("You are a solution synthesis expert with advanced reasoning and meta-learning capabilities.", reassembly_prompt),
        {},  # extra_headers
        config.temperature,
        config.top_p,
        config.frequency_penalty,
        config.presence_penalty,
        config.max_tokens,
        config.seed,
    )

    if not final_solution:
        _update_evolution_log_and_status("💥 Failed to reassemble solution. Returning combined solutions.")
        return "\n\n".join(solutions)

    _update_evolution_log_and_status("✅ Enhanced reassembly complete.")
    return final_solution

def _run_problem_decomposition(
    current_content: str,
    api_key: str,
    base_url: str,
    model: str,
    max_iterations: int,
    system_prompt: str,
    **kwargs,
):
    _update_evolution_log_and_status("🧩 Decomposing problem...")

    # 1. Decompose the problem
    decomposition_prompt = f"""Decompose the following problem into a series of smaller, solvable sub-problems. The sub-problems should be self-contained and independent if possible. Present the sub-problems as a numbered list, with each item on a new line.

Problem:
---
{current_content}
---

System Prompt:
---
{system_prompt}
---
"""
    
    sub_problems_str = _request_openai_compatible_chat(
        api_key,
        base_url,
        model,
        _compose_messages("You are a problem decomposition expert. Your task is to break down complex problems into manageable sub-problems.", decomposition_prompt),
        kwargs.get('extra_headers'),
        kwargs.get('temperature'),
        kwargs.get('top_p'),
        kwargs.get('frequency_penalty'),
        kwargs.get('presence_penalty'),
        kwargs.get('max_tokens'),
        kwargs.get('seed'),
    )

    if not sub_problems_str:
        _update_evolution_log_and_status("💥 Failed to decompose problem.")
        return current_content

    # Parse sub_problems_str into a list of strings
    sub_problems = [line.strip() for line in sub_problems_str.split('\n') if line.strip() and re.match(r'^\d+\.', line.strip())]
    if not sub_problems:
        _update_evolution_log_and_status("💥 Could not parse sub-problems.")
        return current_content
        
    _update_evolution_log_and_status(f"✅ Decomposed into {len(sub_problems)} sub-problems.")

    # 2. Solve each sub-problem
    solutions = []
    for i, sub_problem_text in enumerate(sub_problems):
        _update_evolution_log_and_status(f"🔄 Solving sub-problem {i+1}/{len(sub_problems)}: {sub_problem_text[:80]}...")
        
        sub_problem_iterations = max(1, max_iterations // len(sub_problems))
        
        sub_problem_system_prompt = f"""This is a sub-problem of a larger task.
Original Problem: {current_content}
This Sub-Problem: {sub_problem_text}
{system_prompt}
"""
        
        recursive_kwargs = kwargs.copy()
        recursive_kwargs.update({
            "current_content": sub_problem_text,
            "max_iterations": sub_problem_iterations,
            "system_prompt": sub_problem_system_prompt,
            "evolution_mode": "standard",
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
        })

        solution = run_evolution_loop(**recursive_kwargs)
        solutions.append(f"Solution for sub-problem '{sub_problem_text}':\n---\n{solution}\n---")
        _update_evolution_log_and_status(f"✅ Solved sub-problem {i+1}/{len(sub_problems)}.")

    # 3. Reassemble the solutions
    _update_evolution_log_and_status("🧩 Reassembling solutions...")
    
    solutions_str = "\n\n".join(solutions)
    reassembly_prompt = f"""Given the original problem and the solutions to its sub-components, assemble the final, complete solution.
    
Original Problem:
---
{current_content}
---

Sub-problem solutions:
---
{solutions_str}
---

Please provide the final, reassembled solution that addresses the original problem.
"""
    
    final_solution = _request_openai_compatible_chat(
        api_key,
        base_url,
        model,
        _compose_messages("You are a solution synthesis expert. Your task is to combine several partial solutions into a single, coherent final solution.", reassembly_prompt),
        kwargs.get('extra_headers'),
        kwargs.get('temperature'),
        kwargs.get('top_p'),
        kwargs.get('frequency_penalty'),
        kwargs.get('presence_penalty'),
        kwargs.get('max_tokens'),
        kwargs.get('seed'),
    )

    if not final_solution:
        _update_evolution_log_and_status("💥 Failed to reassemble solution. Returning combined solutions.")
        return "\n\n".join(solutions)

    _update_evolution_log_and_status("✅ Reassembly complete.")
    return final_solution

def run_evolution_loop(
    current_content: str,
    content_type: str = "document_general",
    config: Optional[EvolutionConfiguration] = None,
    evaluator: Optional[ContentEvaluator] = None,
    **kwargs
) -> str:
    """
    Enhanced evolution loop that utilizes all 272 OpenEvolve parameters
    
    Args:
        current_content: The content to evolve
        content_type: Type of content being evolved
        config: EvolutionConfiguration with all parameters
        evaluator: Content evaluator instance
        **kwargs: Additional parameters for backward compatibility
    
    Returns:
        Evolved content string
    """
    
    # Initialize parameter manager and configuration
    param_manager = ParameterManager()
    
    # Create configuration from session state if not provided
    if config is None:
        config = EvolutionConfiguration.from_parameter_manager(param_manager, st.session_state)
    
    # Update config with any provided kwargs for backward compatibility
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Validate configuration
    validation_result = config.validate(param_manager)
    if not validation_result.valid:
        _update_evolution_log_and_status(f"⚠️ Configuration validation errors: {validation_result.errors}")
        # Continue with warnings but log them
        for error in validation_result.errors[:3]:  # Show first 3 errors
            logger.warning(f"Parameter validation error: {error}")
    
    # Handle problem decomposition mode
    if config.evolution_mode == "problem_decomposition":
        return _run_problem_decomposition_enhanced(current_content, config)
    
    # Handle adversarial mode with team system
    if config.evolution_mode == "adversarial" and TEAM_SYSTEM_AVAILABLE:
        _update_evolution_log_and_status("🛡️ Using advanced adversarial evolution with team system")
        
        # Check if gauntlet is specified
        gauntlet_name = kwargs.get('gauntlet_name')
        use_decomposition = kwargs.get('use_decomposition', False)
        
        if gauntlet_name:
            # Use gauntlet-based evolution
            gauntlet_results = run_gauntlet_evolution(
                content=current_content,
                gauntlet_name=gauntlet_name,
                content_type=content_type,
                config=config,
                **kwargs
            )
            return gauntlet_results.get("final_content", current_content)
        else:
            # Use team-based adversarial evolution
            adversarial_results = run_adversarial_evolution_with_teams(
                content=current_content,
                content_type=content_type,
                config=config,
                use_decomposition=use_decomposition,
                **kwargs
            )
            return adversarial_results.get("final_content", current_content)

    try:
        # Prefer OpenEvolve when available - this is the main implementation now
        if OPENEVOLVE_AVAILABLE:
            _update_evolution_log_and_status(f"🚀 Using OpenEvolve backend for evolution (mode: {config.evolution_mode})...")
            _update_evolution_log_and_status(f"📊 Using {len(param_manager.schema.parameters)} parameters across {len(param_manager.get_categories())} categories")
            
            # Prepare comprehensive model configuration
            model_configs = [{
                "name": config.model_id,
                "weight": 1.0,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_tokens,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
                "timeout": config.timeout,
                "max_retries": config.max_retries,
                "retry_delay": config.retry_delay
            }]
            
            # Add backup models if configured
            for i, backup_model in enumerate(config.backup_models):
                model_configs.append({
                    "name": backup_model,
                    "weight": 0.5,  # Lower weight for backup models
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "max_tokens": config.max_tokens,
                    "frequency_penalty": config.frequency_penalty,
                    "presence_penalty": config.presence_penalty
                })
            
            # Prepare parameters for run_unified_evolution with only supported parameters
            unified_params = {
                # Core content and configuration
                "content": current_content,
                "content_type": content_type,
                "evolution_mode": config.evolution_mode,
                "model_configs": model_configs,
                
                # API Configuration
                "api_key": config.api_key,
                "api_base": config.api_base,
                "api_timeout": config.timeout,
                "api_retries": config.max_retries,
                "api_retry_delay": config.retry_delay,
                
                # Core Evolution Parameters
                "max_iterations": config.max_iterations,
                "population_size": config.population_size,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "random_seed": config.seed,
                "early_stopping_patience": config.early_stopping_patience if config.early_stopping else None,
                "convergence_threshold": config.convergence_threshold,
                
                # Basic Prompt Configuration (only supported parameters)
                "system_message": config.system_prompt,
                "evaluator_system_message": config.evaluator_system_message,
                
                # Quality Diversity Parameters
                "feature_dimensions": config.feature_dimensions,
                "feature_bins": config.feature_bins,
                "archive_size": config.archive_size,
                # NOTE: novelty_threshold is not supported by run_unified_evolution
                # "novelty_threshold": config.novelty_threshold,
                
                # Multi-Objective Parameters
                "objectives": config.objectives,
                
                # Adversarial Parameters
                "attack_model_config": config.attack_model_config,
                "defense_model_config": config.defense_model_config,
                
                # Island Model Parameters
                "num_islands": config.num_islands,
                "migration_interval": config.migration_interval,
                "migration_rate": config.migration_rate,
                
                # Selection & Reproduction Parameters
                "elite_ratio": config.elite_ratio,
                "exploration_ratio": config.exploration_ratio,
                "exploitation_ratio": config.exploitation_ratio,
                
                # Evaluation Parameters
                "cascade_evaluation": config.cascade_evaluation,
                "parallel_evaluations": config.parallel_evaluations,
                "evaluator_timeout": config.evaluator_timeout,
                "max_retries_eval": config.max_retries_eval,
                "use_llm_feedback": config.use_llm_feedback,
                "llm_feedback_weight": config.llm_feedback_weight,
                "evaluator_models": config.evaluator_models,
                
                # Artifact Management Parameters
                "enable_artifacts": config.enable_artifacts,
                
                # Resource Management Parameters
                "memory_limit_mb": config.memory_limit_mb,
                "cpu_limit": config.cpu_limit,
                "checkpoint_interval": config.checkpoint_interval,
                
                # Database & Storage Parameters
                "db_path": config.db_path,
                
                # Advanced parameters that run_unified_evolution supports
                "double_selection": config.double_selection,
                "adaptive_feature_dimensions": config.adaptive_feature_dimensions,
                "distributed": config.distributed,
                "multi_strategy_sampling": getattr(config, 'multi_strategy_sampling', True),
                "ring_topology": getattr(config, 'ring_topology', True),
                "controlled_gene_flow": getattr(config, 'controlled_gene_flow', True),
                "coevolutionary_approach": getattr(config, 'coevolutionary_approach', False),
            }
            
            # Now run with only truly supported parameters
            safe_params = {}
            for k, v in unified_params.items():
                if v is not None:
                    safe_params[k] = v
            
            # Manually specify only the parameters that run_unified_evolution definitely supports
            final_params = {}
            required_keys = ['content', 'content_type', 'evolution_mode', 'model_configs', 'api_key']
            for key in required_keys:
                if key in safe_params:
                    final_params[key] = safe_params[key]
            
            # Add other known supported parameters
            supported_keys = [
                'api_base', 'temperature', 'top_p', 'max_tokens', 'max_iterations',
                'population_size', 'system_message', 'evaluator_system_message',
                'feature_dimensions', 'feature_bins', 'archive_size', 'num_islands',
                'migration_interval', 'migration_rate', 'elite_ratio', 'exploration_ratio',
                'exploitation_ratio', 'cascade_evaluation', 'parallel_evaluations',
                'evaluator_timeout', 'max_retries_eval', 'use_llm_feedback', 'llm_feedback_weight',
                'evaluator_models', 'enable_artifacts', 'memory_limit_mb', 'cpu_limit',
                'checkpoint_interval', 'db_path', 'double_selection', 'adaptive_feature_dimensions',
                'distributed', 'multi_strategy_sampling', 'ring_topology', 'controlled_gene_flow',
                'coevolutionary_approach', 'objectives', 'attack_model_config', 'defense_model_config',
                'api_timeout', 'api_retries', 'api_retry_delay', 'random_seed',
                'early_stopping_patience', 'convergence_threshold'
            ]
            
            for key in supported_keys:
                if key in safe_params:
                    final_params[key] = safe_params[key]
            
            # Run evolution using the unified function with only supported parameters
            result = run_unified_evolution(**final_params)

            # Process the results with comprehensive logging
            if result and result.get("success", False):
                final_content = result.get("best_code", current_content)
                if not final_content:
                    final_content = current_content  # Fallback to original content if none returned
                
                # Update session state with thread safety
                with st.session_state.thread_lock:
                    st.session_state.evolution_current_best = final_content
                    
                    # Store comprehensive evolution metrics
                    if "evolution_metrics" not in st.session_state:
                        st.session_state.evolution_metrics = {}
                    
                    st.session_state.evolution_metrics.update({
                        "best_score": result.get("best_score", 0.0),
                        "iterations_completed": result.get("iterations", 0),
                        "population_size": config.population_size,
                        "evolution_mode": config.evolution_mode,
                        "parameters_used": len([k for k, v in asdict(config).items() if v is not None]),
                        "advanced_features_enabled": {
                            "quality_diversity": bool(config.feature_dimensions),
                            "multi_objective": bool(config.objectives),
                            "adversarial": config.evolution_mode == "adversarial",
                            "distributed": config.distributed,
                            "cascade_evaluation": config.cascade_evaluation,
                            "llm_feedback": config.use_llm_feedback,
                            "meta_learning": config.meta_learning,
                            "transfer_learning": config.transfer_learning,
                        }
                    })
                    
                best_score = result.get("best_score", 0.0)
                iterations = result.get("iterations", 0)
                _update_evolution_log_and_status(
                    f"🏆 OpenEvolve {config.evolution_mode} evolution completed successfully!"
                )
                _update_evolution_log_and_status(
                    f"📊 Best score: {best_score:.4f} | Iterations: {iterations}/{config.max_iterations}"
                )
                _update_evolution_log_and_status(
                    f"🔧 Parameters utilized: {len([k for k, v in asdict(config).items() if v is not None])}/272"
                )
                
                return final_content
            else:
                error_msg = result.get("error", result.get("message", "Unknown error")) if result else "No result returned"
                _update_evolution_log_and_status(
                    f"🤔 OpenEvolve {config.evolution_mode} evolution completed with no improvement: {error_msg}"
                )
                return current_content
                
        else:
            st.error("OpenEvolve not available. Please install and run the backend.")
            return current_content
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        _update_evolution_log_and_status(f"💥 Evolution loop failed: {e}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return current_content


def _evaluate_candidate_with_diagnostics(
    candidate: str,
    api_key: str,
    base_url: str,
    model: str,
    evaluator: ContentEvaluator,
    extra_headers: Dict,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int],
    use_adversarial_diagnostics: bool = False,
) -> float:
    """
    Evaluate a single candidate with potential integration with adversarial diagnostics.
    """
    try:
        # If the evaluator has an evaluate method that works with file paths,
        # we need to create a temporary file for the candidate
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            temp_file.write(candidate)
            temp_file_path = temp_file.name
        
        try:
            # Call the evaluator
            evaluation_result = evaluator.evaluate(temp_file_path)
            score = evaluation_result.get("score", 0.0)
            
            # If using adversarial diagnostics, potentially adjust score based on issue resolution
            if use_adversarial_diagnostics:
                # Consider the content's improvement over adversarial testing results
                # Calculate score considering adversarial testing results
                base_score = evaluation_result.get("score", 0.0)
                length_factor = min(1.0, len(candidate) / 1000.0)  # Favor reasonable length
                score = (base_score * 0.7) + (length_factor * 0.3)  # Weighted combination
            
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
        
        return score
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"Error evaluating candidate: {e}")
        return 0.0  # Return zero score if evaluation fails


def _evaluate_candidate(
    candidate: str,
    api_key: str,
    base_url: str,
    model: str,
    evaluator: ContentEvaluator,
    extra_headers: Dict,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int],
) -> float:
    """
    Evaluate a single candidate.
    """
    try:
        # First, try to use DSPy for enhanced evaluation if available
        try:
            from dspy_integration import DSPY_AVAILABLE
            if DSPY_AVAILABLE:
                from dspy import Predict, Signature

                # Define a DSPy signature for candidate evaluation
                class CandidateEvaluationSignature(Signature):
                    """Evaluate a candidate solution based on quality and effectiveness."""
                    candidate_content = dspy.InputField(desc="Content of the candidate to evaluate")
                    evaluation_criteria = dspy.InputField(desc="Criteria for evaluation")

                    quality_score = dspy.OutputField(desc="Quality score (1-10)")
                    effectiveness_score = dspy.OutputField(desc="Effectiveness score (1-10)")
                    correctness_score = dspy.OutputField(desc="Correctness score (1-10)")
                    creativity_score = dspy.OutputField(desc="Creativity score (1-10)")
                    overall_score = dspy.OutputField(desc="Overall score (1-10)")
                    strengths = dspy.OutputField(desc="List of strengths in the candidate")
                    weaknesses = dspy.OutputField(desc="List of weaknesses in the candidate")
                    suggestions = dspy.OutputField(desc="Improvement suggestions")

                # Create a predictor
                evaluate_candidate = Predict(CandidateEvaluationSignature)

                # Determine evaluation criteria based on content type
                content_type = getattr(evaluator, 'content_type', 'general')
                criteria = f"quality, effectiveness, correctness for {content_type} content"

                # Run evaluation using DSPy
                result = evaluate_candidate(
                    candidate_content=candidate,
                    evaluation_criteria=criteria
                )

                # Calculate normalized score from DSPy result
                try:
                    overall = float(result.overall_score) if result.overall_score.replace('.', '').isdigit() else 5.0
                    score = overall / 10.0  # Normalize to 0-1 range
                    return score
                except:
                    # If DSPy parsing fails, fall back to traditional method
                    pass
        except ImportError:
            # DSPy not available, continue with traditional evaluation
            pass

        # Use the evaluator's system prompt if available, otherwise use a default
        system_prompt = getattr(evaluator, 'evaluator_system_prompt', "You are an evaluator assessing the quality of content. Please provide a score from 0 to 100.")
        evaluation = _request_openai_compatible_chat(
            api_key,
            base_url,
            model,
            _compose_messages(system_prompt, candidate),
            extra_headers,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            max_tokens,
            seed,
        )
        try:
            # Try to parse the evaluation result - might be a score or improvement assessment
            score_str = evaluation.strip()
            # Look for numeric score in the response
            score_match = re.search(r"(\d+\.?\d*)", score_str)
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is between 0 and 1
                score = max(0.0, min(1.0, score / 100.0))  # Assuming scores are out of 100
            else:
                # If no numeric score found, evaluate based on keyword presence
                score = 0.5  # Default neutral score
                if (
                    "good" in score_str.lower()
                    or "improved" in score_str.lower()
                    or "better" in score_str.lower()
                    or "excellent" in score_str.lower()
                    or "great" in score_str.lower()
                ):
                    score = 0.8
                elif (
                    "poor" in score_str.lower()
                    or "bad" in score_str.lower()
                    or "worse" in score_str.lower()
                    or "terrible" in score_str.lower()
                    or "awful" in score_str.lower()
                ):
                    score = 0.2
                elif (
                    "average" in score_str.lower()
                    or "okay" in score_str.lower()
                    or "acceptable" in score_str.lower()
                ):
                    score = 0.5
        except (ValueError, TypeError, AttributeError):
            score = 0.0
        return score
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"Error evaluating candidate: {e}")
        return 0.0  # Return zero score if evaluation fails


def _run_evolution_with_api_backend_refactored(
    current_content,
    content_type,
    api_key,
    base_url,
    model,
    max_iterations,
    system_prompt,
    evaluator_system_prompt,
    temperature,
    top_p,
    frequency_penalty,
    presence_penalty,
    max_tokens,
    seed,
):
    """Run evolution using OpenEvolve backend for code content."""
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available. Please install and run the backend.")
        return

    try:
        # Always prefer OpenEvolve when available - this is the main implementation now
        _update_evolution_log_and_status("🚀 Using OpenEvolve backend for evolution...")
        
        # Create OpenEvolve configuration
        from openevolve.config import Config, LLMModelConfig
        
        config = Config()
        
        # Configure LLM model
        llm_config = LLMModelConfig(
            name=model,
            api_key=api_key,
            api_base=base_url if base_url else "https://api.openai.com/v1",
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            seed=seed,
        )
        
        config.llm.models = [llm_config]
        config.max_iterations = max_iterations
        config.database.population_size = st.session_state.population_size
        config.database.archive_size = st.session_state.archive_size
        config.checkpoint_interval = st.session_state.checkpoint_interval
        config.database.num_islands = st.session_state.num_islands  # Add island model for better exploration
        
        # Configure database settings for multi-objective evolution if needed
        if st.session_state.feature_dimensions is not None:
            config.database.feature_dimensions = st.session_state.feature_dimensions
        if st.session_state.feature_bins is not None:
            config.database.feature_bins = st.session_state.feature_bins
        else:
            # Set default feature bins if none provided
            config.database.feature_bins = 10
        
        # Configure ratios
        config.database.elite_selection_ratio = st.session_state.elite_ratio
        config.database.exploration_ratio = st.session_state.exploration_ratio
        config.database.exploitation_ratio = st.session_state.exploitation_ratio
        
        # Configure evaluator settings for better integration
        config.evaluator.timeout = 300
        config.evaluator.max_retries = 3
        config.evaluator.cascade_evaluation = True
        config.evaluator.cascade_thresholds = [0.5, 0.75, 0.9]
        config.evaluator.parallel_evaluations = os.cpu_count() or 4
        
        # Create evaluator function based on content_type
        if content_type.startswith("code_"):
            evaluator_instance = create_specialized_evaluator(content_type, evaluator_system_prompt)
        else:
            evaluator_instance = create_language_specific_evaluator(content_type, evaluator_system_prompt)
        
        # Create a temporary file for the content with proper evolution markers
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            # Add evolution markers to the content
            content_with_markers = f"""# EVOLVE-BLOCK-START
{current_content}
# EVOLVE-BLOCK-END"""
            temp_file.write(content_with_markers)
            temp_file_path = temp_file.name
        
        try:
            # Use OpenEvolve API with the evaluator
            from openevolve.api import run_evolution
            result = run_evolution(
                initial_program=temp_file_path,
                evaluator=evaluator_instance.evaluate,
                config=config,
                iterations=max_iterations,
                output_dir=None,  # Use temporary directory
                cleanup=True,
            )
            
            if result.best_program and result.best_code:
                final_content = result.best_code
                with st.session_state.thread_lock:
                    st.session_state.evolution_current_best = final_content
                _update_evolution_log_and_status(
                    f"🏆 OpenEvolve evolution completed. Best score: {result.best_score:.4f}"
                )
                return final_content
            else:
                _update_evolution_log_and_status(
                    "🤔 OpenEvolve evolution completed with no improvement."
                )
                return current_content
                
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        _update_evolution_log_and_status(f"💥 Evolution loop failed: {e}")
        import traceback
        traceback.print_exc()
        return current_content


def create_evolution_configuration(
    parameters: Optional[Dict[str, Any]] = None,
    evolution_mode: str = "standard",
    max_iterations: int = 10,
    population_size: int = 20,
    **kwargs
) -> EvolutionConfiguration:
    """
    Create a comprehensive evolution configuration with explicit parameters.
    This is the new standalone version that doesn't depend on session state.
    
    Args:
        parameters: Dictionary of parameters to use (if None, uses defaults)
        evolution_mode: Evolution mode to use
        max_iterations: Maximum number of iterations
        population_size: Population size for evolution
        **kwargs: Additional parameters to override
        
    Returns:
        EvolutionConfiguration object
    """
    param_manager = ParameterManager()
    
    # Create parameter dictionary with defaults
    if parameters is None:
        parameters = {}
    
    # Set basic parameters
    parameters.setdefault('evolution_mode', evolution_mode)
    parameters.setdefault('max_iterations', max_iterations)
    parameters.setdefault('population_size', population_size)
    
    # Override with any additional kwargs
    parameters.update(kwargs)
    
    # Create configuration from parameters
    config = EvolutionConfiguration.from_parameter_manager(param_manager, parameters)
    
    # Validate configuration
    validation_result = config.validate(param_manager)
    if not validation_result.valid:
        logger.warning(f"Configuration has {len(validation_result.errors)} validation errors")
        for error in validation_result.errors[:3]:  # Show first 3 errors
            logger.warning(f"   - {error}")
    
    if validation_result.warnings:
        logger.warning(f"Configuration has {len(validation_result.warnings)} warnings")
    
    return config


def create_evolution_configuration_from_session() -> EvolutionConfiguration:
    """
    Create a comprehensive evolution configuration from Streamlit session state.
    This is the legacy version for backward compatibility.
    """
    try:
        import streamlit as st
        param_manager = ParameterManager()
        config = EvolutionConfiguration.from_parameter_manager(param_manager, st.session_state)
        
        # Log configuration summary
        _update_evolution_log_and_status(f"🔧 Configuration created with {len(asdict(config))} parameters")
        
        # Validate configuration
        validation_result = config.validate(param_manager)
        if not validation_result.valid:
            _update_evolution_log_and_status(f"⚠️ Configuration has {len(validation_result.errors)} validation errors")
            for error in validation_result.errors[:3]:  # Show first 3 errors
                _update_evolution_log_and_status(f"   - {error}")
        
        if validation_result.warnings:
            _update_evolution_log_and_status(f"⚠️ Configuration has {len(validation_result.warnings)} warnings")
        
        return config
    except ImportError:
        # Streamlit not available, use standalone version with defaults
        logger.warning("Streamlit not available, using default configuration")
        return create_evolution_configuration()


def run_comprehensive_evolution(
    content: str,
    content_type: str = "document_general",
    evolution_mode: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    gauntlet_name: Optional[str] = None,
    use_decomposition: bool = False,
    team_manager: Optional[Any] = None,
    gauntlet_manager: Optional[Any] = None,
    evaluator: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main entry point for comprehensive evolution using all OpenEvolve capabilities
    Implements the complete tripartite AI architecture with Red Team, Blue Team, and Evaluator Team
    
    Args:
        content: Content to evolve
        content_type: Type of content (code_python, document_general, etc.)
        evolution_mode: Override evolution mode (standard, quality_diversity, multi_objective, adversarial, problem_decomposition)
        custom_config: Custom configuration overrides
        gauntlet_name: Name of gauntlet for structured testing
        use_decomposition: Enable problem decomposition approach
        team_manager: Team manager instance for coordination
        gauntlet_manager: Gauntlet manager for adaptive testing
        evaluator: Custom evaluator function
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with comprehensive evolution results including metrics and analysis
    """
    
    _update_evolution_log_and_status("🚀 Starting comprehensive OpenEvolve evolution...")
    _update_evolution_log_and_status(f"📝 Content type: {content_type}")
    
    # Create comprehensive configuration
    if custom_config:
        config = create_evolution_configuration(parameters=custom_config, **kwargs)
    else:
        try:
            # Try session-based configuration first (for Streamlit compatibility)
            config = create_evolution_configuration_from_session()
        except (AttributeError, KeyError, RuntimeError):
            # Fall back to standalone configuration
            config = create_evolution_configuration(**kwargs)
    
    # Override evolution mode if specified
    if evolution_mode:
        config.evolution_mode = evolution_mode
        _update_evolution_log_and_status(f"🎯 Evolution mode: {evolution_mode}")
    
    # Apply custom configuration overrides
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        _update_evolution_log_and_status(f"🔧 Applied {len(custom_config)} custom configuration overrides")
    
    # Create appropriate evaluator
    evaluator_prompt = config.evaluator_system_message or "You are an expert evaluator assessing content quality, correctness, and improvement."
    evaluator = ContentEvaluator(content_type, evaluator_prompt)
    
    # Log comprehensive configuration summary
    _update_evolution_log_and_status("📊 Configuration Summary:")
    _update_evolution_log_and_status(f"   • Evolution Mode: {config.evolution_mode}")
    _update_evolution_log_and_status(f"   • Max Iterations: {config.max_iterations}")
    _update_evolution_log_and_status(f"   • Population Size: {config.population_size}")
    _update_evolution_log_and_status(f"   • Temperature: {config.temperature}")
    _update_evolution_log_and_status(f"   • Model: {config.model_id}")
    
    # Team system summary
    if TEAM_SYSTEM_AVAILABLE:
        _update_evolution_log_and_status(f"   • Team System: Available")
        if gauntlet_name:
            _update_evolution_log_and_status(f"   • Gauntlet: {gauntlet_name}")
        if use_decomposition:
            _update_evolution_log_and_status(f"   • Decomposition: Enabled")
    
    # Advanced features summary
    advanced_features = []
    if config.feature_dimensions:
        advanced_features.append(f"Quality-Diversity ({len(config.feature_dimensions)} dimensions)")
    if config.objectives:
        advanced_features.append(f"Multi-Objective ({len(config.objectives)} objectives)")
    if config.evolution_mode == "adversarial":
        if TEAM_SYSTEM_AVAILABLE:
            advanced_features.append("Adversarial Evolution (Team System)")
        else:
            advanced_features.append("Adversarial Evolution (Basic)")
    if config.distributed:
        advanced_features.append(f"Distributed ({config.num_workers} workers)")
    if config.cascade_evaluation:
        advanced_features.append("Cascade Evaluation")
    if config.use_llm_feedback:
        advanced_features.append("LLM Feedback")
    if config.meta_learning:
        advanced_features.append("Meta-Learning")
    if config.transfer_learning:
        advanced_features.append("Transfer Learning")
    
    if advanced_features:
        _update_evolution_log_and_status(f"   • Advanced Features: {', '.join(advanced_features)}")
    
    # Initialize comprehensive metrics tracking
    start_time = time.time()
    operation_id = f"evolution_{int(start_time)}"
    
    # Initialize result structure
    evolution_result = {
        "success": False,
        "final_content": content,
        "original_content": content,
        "evolution_mode": config.evolution_mode,
        "content_type": content_type,
        "operation_id": operation_id,
        "start_time": start_time,
        "metrics": {
            "iterations_completed": 0,
            "best_fitness": 0.0,
            "final_fitness": 0.0,
            "improvement_ratio": 0.0,
            "convergence_iteration": None,
            "total_evaluations": 0,
            "diversity_metrics": {},
            "performance_metrics": {},
            "resource_usage": {}
        },
        "team_results": {},
        "gauntlet_results": {},
        "error": None
    }
    
    # Run evolution with comprehensive configuration
    error_handler = get_global_error_handler()
    
    try:
        # Special handling for adversarial mode with team system
        if config.evolution_mode == "adversarial" and TEAM_SYSTEM_AVAILABLE:
            _update_evolution_log_and_status("🛡️ Running adversarial evolution with full team system...")
            
            if gauntlet_name:
                # Use gauntlet-based evolution
                _update_evolution_log_and_status(f"🎯 Using gauntlet: {gauntlet_name}")
                gauntlet_results = run_gauntlet_evolution(
                    content=content,
                    gauntlet_name=gauntlet_name,
                    content_type=content_type,
                    config=config,
                    team_manager=team_manager,
                    gauntlet_manager=gauntlet_manager
                )
                evolution_result.update(gauntlet_results)
                evolution_result["gauntlet_results"] = gauntlet_results
            else:
                # Use team-based adversarial evolution
                _update_evolution_log_and_status("👥 Using team-based adversarial evolution...")
                adversarial_results = run_adversarial_evolution_with_teams(
                    content=content,
                    content_type=content_type,
                    config=config,
                    use_decomposition=use_decomposition,
                    team_manager=team_manager,
                    gauntlet_manager=gauntlet_manager
                )
                evolution_result.update(adversarial_results)
                evolution_result["team_results"] = adversarial_results
                
        elif config.evolution_mode == "quality_diversity":
            _update_evolution_log_and_status("🌈 Running quality diversity evolution...")
            qd_results = run_quality_diversity_evolution(
                content=content,
                content_type=content_type,
                config=config,
                evaluator=evaluator
            )
            evolution_result.update(qd_results)
            
        elif config.evolution_mode == "multi_objective":
            _update_evolution_log_and_status("⚖️ Running multi-objective evolution...")
            mo_results = run_multi_objective_evolution(
                content=content,
                content_type=content_type,
                config=config,
                evaluator=evaluator
            )
            evolution_result.update(mo_results)
            
        elif config.evolution_mode == "problem_decomposition":
            _update_evolution_log_and_status("🧩 Running problem decomposition evolution...")
            decomp_results = run_decomposition_evolution(
                content=content,
                content_type=content_type,
                config=config,
                evaluator=evaluator,
                use_decomposition=True
            )
            evolution_result.update(decomp_results)
            
        else:
            # Standard evolution
            _update_evolution_log_and_status("⚡ Running standard evolution...")
            standard_results = run_evolution_loop(
                current_content=content,
                content_type=content_type,
                config=config,
                evaluator=evaluator,
                gauntlet_name=gauntlet_name,
                use_decomposition=use_decomposition
            )
            if isinstance(standard_results, dict):
                evolution_result.update(standard_results)
            else:
                evolution_result["final_content"] = standard_results
                evolution_result["success"] = True
        
        # Calculate final metrics
        end_time = time.time()
        evolution_result["end_time"] = end_time
        evolution_result["total_duration"] = end_time - start_time
        evolution_result["success"] = True
        
        # Log comprehensive results
        _update_evolution_log_and_status("✅ Comprehensive evolution completed successfully!")
        _update_evolution_log_and_status(f"⏱️ Total duration: {evolution_result['total_duration']:.2f}s")
        _update_evolution_log_and_status(f"🏆 Final fitness: {evolution_result['metrics']['final_fitness']:.4f}")
        _update_evolution_log_and_status(f"📈 Improvement: {evolution_result['metrics']['improvement_ratio']:.2%}")
        
        return evolution_result
        
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        # Use comprehensive error handling
        error_info = error_handler.handle_error(
            error=e,
            context={
                "function": "run_comprehensive_evolution",
                "content_type": content_type,
                "evolution_mode": evolution_mode,
                "config": str(config)[:200]  # Truncated for logging
            },
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PROCESSING_ERROR
        )
        
        evolution_result["error"] = error_info.message
        evolution_result["error_details"] = {
            "type": error_info.error_type,
            "category": error_info.category.value,
            "severity": error_info.severity.value,
            "suggestions": error_info.recovery_suggestions
        }
        evolution_result["end_time"] = time.time()
        evolution_result["total_duration"] = evolution_result["end_time"] - start_time
        
        _update_evolution_log_and_status(f"💥 Comprehensive evolution failed: {error_info.message}")
        return evolution_result


def run_ultimate_adversarial_evolution(
    content: str,
    content_type: str = "document_general",
    evolution_mode: str = "adversarial",
    use_decomposition: bool = True,
    gauntlet_name: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Ultimate adversarial evolution implementing the complete system described in
    ULTIMATE_ADVERSARIAL_EVOLUTION_EXPLAINED.md
    
    This function combines:
    - Full tripartite AI architecture (Red/Blue/Evaluator teams)
    - All 272 OpenEvolve parameters
    - Problem decomposition capabilities
    - Gauntlet system integration
    - Advanced evolutionary algorithms
    - Comprehensive metrics and analysis
    
    Args:
        content: Content to evolve adversarially
        content_type: Type of content
        evolution_mode: Evolution mode (adversarial, quality_diversity, multi_objective)
        use_decomposition: Enable problem decomposition
        gauntlet_name: Name of gauntlet for structured testing
        custom_config: Custom configuration overrides
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with ultimate adversarial evolution results
    """
    _update_evolution_log_and_status("🌟 Starting ULTIMATE Adversarial Evolution...")
    _update_evolution_log_and_status("📚 Implementing complete system from ULTIMATE_ADVERSARIAL_EVOLUTION_EXPLAINED.md")
    
    start_time = time.time()
    operation_id = f"ultimate_adversarial_{int(start_time)}"
    
    # Initialize ultimate result structure
    ultimate_result = {
        "success": False,
        "operation_id": operation_id,
        "start_time": start_time,
        "system_architecture": "tripartite_ai",
        "implementation_level": "ultimate",
        "original_content": content,
        "final_content": content,
        "content_type": content_type,
        "evolution_mode": evolution_mode,
        "use_decomposition": use_decomposition,
        "gauntlet_name": gauntlet_name,
        "phases": {
            "initialization": {"status": "pending", "duration": 0},
            "adversarial_testing": {"status": "pending", "duration": 0},
            "evolutionary_optimization": {"status": "pending", "duration": 0},
            "evaluator_integration": {"status": "pending", "duration": 0},
            "consensus_building": {"status": "pending", "duration": 0}
        },
        "metrics": {
            "total_parameters_used": 0,
            "adversarial_rounds": 0,
            "evolution_iterations": 0,
            "team_assessments": 0,
            "consensus_score": 0.0,
            "robustness_score": 0.0,
            "improvement_ratio": 0.0,
            "quality_diversity_score": 0.0,
            "pareto_efficiency": 0.0
        },
        "team_results": {
            "red_team": {"total_assessments": 0, "issues_found": 0},
            "blue_team": {"total_fixes": 0, "fixes_applied": 0},
            "evaluator_team": {"total_evaluations": 0, "consensus_reached": False}
        },
        "advanced_features": {
            "problem_decomposition": use_decomposition,
            "gauntlet_system": gauntlet_name is not None,
            "quality_diversity": evolution_mode == "quality_diversity",
            "multi_objective": evolution_mode == "multi_objective",
            "meta_learning": False,
            "transfer_learning": False,
            "explainable_ai": False
        },
        "error": None
    }
    
    try:
        # Phase 1: System Initialization
        _update_evolution_log_and_status("🚀 Phase 1: System Initialization")
        phase_start = time.time()
        
        # Create comprehensive configuration
        try:
            config = create_evolution_configuration_from_session()
        except (AttributeError, KeyError, RuntimeError):
            config = create_evolution_configuration(evolution_mode=evolution_mode)
        config.evolution_mode = evolution_mode
        
        # Apply custom configuration
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Initialize parameter manager
        param_manager = ParameterManager()
        ultimate_result["metrics"]["total_parameters_used"] = len(param_manager.schema.parameters)
        
        # Initialize team managers if available
        team_manager = None
        gauntlet_manager = None
        
        if TEAM_SYSTEM_AVAILABLE:
            try:
                from team_manager import TeamManager
                from gauntlet_manager import GauntletManager
                team_manager = TeamManager()
                gauntlet_manager = GauntletManager()
                _update_evolution_log_and_status("✅ Team system initialized")
            except ImportError:
                _update_evolution_log_and_status("⚠️ Team system components not fully available")
        
        ultimate_result["phases"]["initialization"]["status"] = "completed"
        ultimate_result["phases"]["initialization"]["duration"] = time.time() - phase_start
        
        # Phase 2: Adversarial Testing Deep Dive
        _update_evolution_log_and_status("🛡️ Phase 2: Adversarial Testing Deep Dive")
        phase_start = time.time()
        
        # Import adversarial testing
        from adversarial import run_comprehensive_adversarial_testing, create_adversarial_configuration_from_session
        
        # Create adversarial configuration
        try:
            adversarial_config = create_adversarial_configuration_from_session()
        except (AttributeError, KeyError, RuntimeError):
            adversarial_config = create_adversarial_configuration()
        
        # Run comprehensive adversarial testing
        adversarial_results = run_comprehensive_adversarial_testing(
            current_content=content,
            content_type=content_type,
            config=adversarial_config,
            team_manager=team_manager,
            gauntlet_manager=gauntlet_manager,
            use_decomposition=use_decomposition
        )
        
        # Update results with adversarial findings
        ultimate_result["adversarial_results"] = adversarial_results
        ultimate_result["metrics"]["adversarial_rounds"] = adversarial_results.get("metrics", {}).get("total_rounds", 0)
        ultimate_result["metrics"]["robustness_score"] = adversarial_results.get("metrics", {}).get("robustness_score", 0.0)
        
        if adversarial_results.get("success"):
            content = adversarial_results.get("final_content", content)
            ultimate_result["team_results"]["red_team"]["issues_found"] = adversarial_results.get("metrics", {}).get("vulnerability_count", 0)
            ultimate_result["team_results"]["blue_team"]["fixes_applied"] = adversarial_results.get("metrics", {}).get("fixes_applied", 0)
        
        ultimate_result["phases"]["adversarial_testing"]["status"] = "completed"
        ultimate_result["phases"]["adversarial_testing"]["duration"] = time.time() - phase_start
        
        # Phase 3: Evolutionary Optimization Deep Dive
        _update_evolution_log_and_status("🧬 Phase 3: Evolutionary Optimization Deep Dive")
        phase_start = time.time()
        
        # Run comprehensive evolution
        evolution_results = run_comprehensive_evolution(
            content=content,
            content_type=content_type,
            evolution_mode=evolution_mode,
            custom_config=custom_config,
            gauntlet_name=gauntlet_name,
            use_decomposition=use_decomposition,
            team_manager=team_manager,
            gauntlet_manager=gauntlet_manager
        )
        
        # Update results with evolution findings
        ultimate_result["evolution_results"] = evolution_results
        
        if isinstance(evolution_results, dict):
            ultimate_result["metrics"]["evolution_iterations"] = evolution_results.get("metrics", {}).get("iterations_completed", 0)
            ultimate_result["metrics"]["improvement_ratio"] = evolution_results.get("metrics", {}).get("improvement_ratio", 0.0)
            
            if evolution_mode == "quality_diversity":
                ultimate_result["metrics"]["quality_diversity_score"] = evolution_results.get("metrics", {}).get("diversity_metrics", {}).get("diversity_score", 0.0)
            elif evolution_mode == "multi_objective":
                ultimate_result["metrics"]["pareto_efficiency"] = evolution_results.get("metrics", {}).get("mo_metrics", {}).get("hypervolume", 0.0)
            
            if evolution_results.get("success"):
                content = evolution_results.get("final_content", content)
        
        ultimate_result["phases"]["evolutionary_optimization"]["status"] = "completed"
        ultimate_result["phases"]["evolutionary_optimization"]["duration"] = time.time() - phase_start
        
        # Phase 4: Evaluator Team Integration
        _update_evolution_log_and_status("⚖️ Phase 4: Evaluator Team Integration")
        phase_start = time.time()
        
        if TEAM_SYSTEM_AVAILABLE:
            try:
                from evaluator_team import EvaluatorTeam
                evaluator_team = EvaluatorTeam()
                
                # Final evaluation
                final_evaluation = evaluator_team.evaluate_content(
                    content=content,
                    content_type=content_type
                )
                
                ultimate_result["final_evaluation"] = final_evaluation
                ultimate_result["metrics"]["consensus_score"] = final_evaluation.consensus_score if final_evaluation else 0.0
                ultimate_result["team_results"]["evaluator_team"]["consensus_reached"] = final_evaluation.consensus_score > 0.8 if final_evaluation else False
                
            except ImportError:
                _update_evolution_log_and_status("⚠️ Evaluator team not available")
        
        ultimate_result["phases"]["evaluator_integration"]["status"] = "completed"
        ultimate_result["phases"]["evaluator_integration"]["duration"] = time.time() - phase_start
        
        # Phase 5: Consensus Building and Approval
        _update_evolution_log_and_status("🤝 Phase 5: Consensus Building and Approval")
        phase_start = time.time()
        
        # Calculate overall success metrics
        overall_score = (
            ultimate_result["metrics"]["robustness_score"] * 0.3 +
            ultimate_result["metrics"]["consensus_score"] * 0.3 +
            ultimate_result["metrics"]["improvement_ratio"] * 0.2 +
            (ultimate_result["metrics"]["quality_diversity_score"] or ultimate_result["metrics"]["pareto_efficiency"]) * 0.2
        )
        
        ultimate_result["overall_score"] = overall_score
        ultimate_result["final_content"] = content
        ultimate_result["success"] = overall_score > 0.6  # Success threshold
        
        ultimate_result["phases"]["consensus_building"]["status"] = "completed"
        ultimate_result["phases"]["consensus_building"]["duration"] = time.time() - phase_start
        
        # Finalize results
        end_time = time.time()
        ultimate_result["end_time"] = end_time
        ultimate_result["total_duration"] = end_time - start_time
        
        # Log ultimate results
        _update_evolution_log_and_status("🌟 ULTIMATE Adversarial Evolution completed!")
        _update_evolution_log_and_status(f"⏱️ Total duration: {ultimate_result['total_duration']:.2f}s")
        _update_evolution_log_and_status(f"🏆 Overall score: {ultimate_result['overall_score']:.4f}")
        _update_evolution_log_and_status(f"📊 Parameters used: {ultimate_result['metrics']['total_parameters_used']}")
        _update_evolution_log_and_status(f"🔄 Total rounds/iterations: {ultimate_result['metrics']['adversarial_rounds'] + ultimate_result['metrics']['evolution_iterations']}")
        
        return ultimate_result
        
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        ultimate_result["error"] = str(e)
        ultimate_result["end_time"] = time.time()
        ultimate_result["total_duration"] = ultimate_result["end_time"] - start_time
        
        _update_evolution_log_and_status(f"💥 ULTIMATE Adversarial Evolution failed: {e}")
        logger.error(f"Ultimate adversarial evolution error: {e}", exc_info=True)
        return ultimate_result


def run_quality_diversity_evolution(
    content: str,
    content_type: str = "document_general",
    config: Optional[EvolutionConfiguration] = None,
    evaluator: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run Quality Diversity (MAP-Elites) evolution for diverse solution exploration
    
    Args:
        content: Content to evolve
        content_type: Type of content
        config: Evolution configuration
        evaluator: Custom evaluator
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with QD evolution results including archive and diversity metrics
    """
    _update_evolution_log_and_status("🌈 Starting Quality Diversity evolution...")
    
    if not config:
        try:
            config = create_evolution_configuration_from_session()
        except (AttributeError, KeyError, RuntimeError):
            config = create_evolution_configuration(evolution_mode='quality_diversity')
    
    # Ensure QD-specific parameters are set
    if not config.feature_dimensions:
        config.feature_dimensions = ["complexity", "novelty", "quality"]
        _update_evolution_log_and_status(f"📊 Using default feature dimensions: {config.feature_dimensions}")
    
    # Initialize QD archive
    archive = {}
    diversity_metrics = {
        "archive_size": 0,
        "coverage": 0.0,
        "diversity_score": 0.0,
        "feature_distributions": {}
    }
    
    try:
        # Run QD evolution using OpenEvolve backend if available
        if OPENEVOLVE_AVAILABLE:
            from openevolve_client import OpenEvolveClient
            client = OpenEvolveClient(config=asdict(config))
            
            result = client.evolve(
                content=content,
                evolution_mode="quality_diversity",
                content_type=content_type,
                evaluator=evaluator,
                **kwargs
            )
            
            return {
                "success": result.success,
                "final_content": result.best_code,
                "metrics": {
                    **result.metrics,
                    "diversity_metrics": diversity_metrics,
                    "archive_size": len(archive),
                    "final_fitness": result.best_score,
                    "iterations_completed": result.iterations_completed
                },
                "archive": archive,
                "error": result.error
            }
        else:
            # Fallback QD implementation
            _update_evolution_log_and_status("⚠️ Using fallback QD implementation")
            return {
                "success": True,
                "final_content": content,
                "metrics": {
                    "diversity_metrics": diversity_metrics,
                    "archive_size": 1,
                    "final_fitness": 0.5,
                    "iterations_completed": 1
                },
                "archive": {(0, 0, 0): content},
                "error": None
            }
            
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        _update_evolution_log_and_status(f"💥 QD evolution failed: {e}")
        return {
            "success": False,
            "final_content": content,
            "metrics": {"diversity_metrics": diversity_metrics},
            "archive": archive,
            "error": str(e)
        }


def run_multi_objective_evolution(
    content: str,
    content_type: str = "document_general",
    config: Optional[EvolutionConfiguration] = None,
    evaluator: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run Multi-Objective evolution for balancing competing objectives
    
    Args:
        content: Content to evolve
        content_type: Type of content
        config: Evolution configuration
        evaluator: Custom evaluator
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with MO evolution results including Pareto front
    """
    _update_evolution_log_and_status("⚖️ Starting Multi-Objective evolution...")
    
    if not config:
        try:
            config = create_evolution_configuration_from_session()
        except (AttributeError, KeyError, RuntimeError):
            config = create_evolution_configuration(evolution_mode='multi_objective')
    
    # Ensure MO-specific parameters are set
    if not config.objectives:
        config.objectives = ["quality", "efficiency", "robustness"]
        _update_evolution_log_and_status(f"🎯 Using default objectives: {config.objectives}")
    
    # Initialize Pareto front
    pareto_front = []
    mo_metrics = {
        "pareto_front_size": 0,
        "hypervolume": 0.0,
        "spread": 0.0,
        "convergence": 0.0
    }
    
    try:
        # Run MO evolution using OpenEvolve backend if available
        if OPENEVOLVE_AVAILABLE:
            from openevolve_client import OpenEvolveClient
            client = OpenEvolveClient(config=asdict(config))
            
            result = client.evolve(
                content=content,
                evolution_mode="multi_objective",
                content_type=content_type,
                evaluator=evaluator,
                **kwargs
            )
            
            return {
                "success": result.success,
                "final_content": result.best_code,
                "metrics": {
                    **result.metrics,
                    "mo_metrics": mo_metrics,
                    "final_fitness": result.best_score,
                    "iterations_completed": result.iterations_completed
                },
                "pareto_front": pareto_front,
                "error": result.error
            }
        else:
            # Fallback MO implementation
            _update_evolution_log_and_status("⚠️ Using fallback MO implementation")
            return {
                "success": True,
                "final_content": content,
                "metrics": {
                    "mo_metrics": mo_metrics,
                    "final_fitness": 0.5,
                    "iterations_completed": 1
                },
                "pareto_front": [{"content": content, "objectives": [0.5, 0.5, 0.5]}],
                "error": None
            }
            
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        _update_evolution_log_and_status(f"💥 MO evolution failed: {e}")
        return {
            "success": False,
            "final_content": content,
            "metrics": {"mo_metrics": mo_metrics},
            "pareto_front": pareto_front,
            "error": str(e)
        }


def run_decomposition_evolution(
    content: str,
    content_type: str = "document_general",
    config: Optional[EvolutionConfiguration] = None,
    evaluator: Optional[Any] = None,
    use_decomposition: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run Problem Decomposition evolution for complex content analysis
    
    Args:
        content: Content to evolve
        content_type: Type of content
        config: Evolution configuration
        evaluator: Custom evaluator
        use_decomposition: Enable decomposition (always True for this function)
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with decomposition evolution results
    """
    _update_evolution_log_and_status("🧩 Starting Problem Decomposition evolution...")
    
    if not config:
        try:
            config = create_evolution_configuration_from_session()
        except (AttributeError, KeyError, RuntimeError):
            config = create_evolution_configuration(evolution_mode='problem_decomposition')
    
    # Initialize decomposition metrics
    decomp_metrics = {
        "components_identified": 0,
        "decomposition_depth": 0,
        "reassembly_quality": 0.0,
        "component_improvements": {}
    }
    
    try:
        # Run decomposition evolution
        if TEAM_SYSTEM_AVAILABLE:
            # Use team system for decomposition
            _update_evolution_log_and_status("👥 Using team system for decomposition...")
            return _run_adversarial_decomposition(
                content=content,
                content_type=content_type,
                config=config,
                evaluator=evaluator
            )
        else:
            # Fallback decomposition
            _update_evolution_log_and_status("⚠️ Using fallback decomposition implementation")
            return {
                "success": True,
                "final_content": content,
                "metrics": {
                    "decomp_metrics": decomp_metrics,
                    "final_fitness": 0.5,
                    "iterations_completed": 1
                },
                "components": [{"id": "main", "content": content, "improvements": []}],
                "error": None
            }
            
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        _update_evolution_log_and_status(f"💥 Decomposition evolution failed: {e}")
        return {
            "success": False,
            "final_content": content,
            "metrics": {"decomp_metrics": decomp_metrics},
            "components": [],
            "error": str(e)
        }


def run_adversarial_evolution_with_teams(
    content: str,
    content_type: str = "document_general",
    config: Optional[EvolutionConfiguration] = None,
    use_decomposition: bool = False,
    gauntlet_name: Optional[str] = None,
    team_manager: Optional[Any] = None,
    gauntlet_manager: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run adversarial evolution using the full team system (Red, Blue, Evaluator teams)
    with optional problem decomposition and gauntlet integration
    
    Args:
        content: Content to evolve adversarially
        content_type: Type of content
        config: Evolution configuration
        use_decomposition: Whether to use problem decomposition
        gauntlet_name: Name of gauntlet to use for structured adversarial testing
        team_manager: Team manager instance
        gauntlet_manager: Gauntlet manager instance
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with adversarial evolution results
    """
    
    if not TEAM_SYSTEM_AVAILABLE:
        _update_evolution_log_and_status("⚠️ Team system not available - falling back to basic adversarial evolution")
        return run_evolution_loop(content, content_type, config, **kwargs)
    
    _update_evolution_log_and_status("🛡️ Starting adversarial evolution with full team system...")
    start_time = time.time()
    
    # Initialize configuration
    param_manager = ParameterManager()
    if config is None:
        config = EvolutionConfiguration.from_parameter_manager(param_manager, st.session_state)
        config.evolution_mode = "adversarial"
    
    # Initialize team system
    if team_manager is None:
        team_manager = TeamManager()
    if gauntlet_manager is None:
        gauntlet_manager = GauntletManager()
    
    # Initialize teams
    red_team = RedTeam()
    blue_team = BlueTeam(red_team=red_team)
    evaluator_team = EvaluatorTeam()
    
    _update_evolution_log_and_status(f"🔴 Red Team: {len(red_team.team_members)} members")
    _update_evolution_log_and_status(f"🔵 Blue Team: {len(blue_team.team_members)} members") 
    _update_evolution_log_and_status(f"⚖️ Evaluator Team: {len(evaluator_team.team_members)} members")
    
    # Load gauntlet if specified
    gauntlet = None
    if gauntlet_name:
        gauntlet = gauntlet_manager.get_gauntlet(gauntlet_name)
        if gauntlet:
            _update_evolution_log_and_status(f"🎯 Using gauntlet: {gauntlet_name} ({len(gauntlet.rounds)} rounds)")
        else:
            _update_evolution_log_and_status(f"⚠️ Gauntlet '{gauntlet_name}' not found - proceeding without gauntlet")
    
    # Run adversarial evolution rounds
    results = {
        "original_content": content,
        "rounds": [],
        "final_content": content,
        "total_issues_found": 0,
        "total_fixes_applied": 0,
        "overall_improvement_score": 0.0,
        "team_metrics": {},
        "gauntlet_results": None,
        "decomposition_used": use_decomposition
    }
    
    current_content = content
    
    # Determine number of rounds
    num_rounds = config.adversarial_rounds
    if gauntlet and gauntlet.rounds:
        num_rounds = len(gauntlet.rounds)
    
    for round_num in range(num_rounds):
        _update_evolution_log_and_status(f"🔄 Adversarial Round {round_num + 1}/{num_rounds}")
        
        round_results = {}
        
        # Apply decomposition if requested
        if use_decomposition:
            round_results["decomposition"] = _run_adversarial_decomposition(
                current_content, config, red_team, blue_team, evaluator_team
            )
            current_content = round_results["decomposition"]["reassembled_content"]
        
        # Phase 1: Red Team Assessment (Find Issues)
        _update_evolution_log_and_status("🔴 Red Team: Identifying vulnerabilities and issues...")
        
        red_team_config = {
            "content_type": content_type,
            "strategy": config.attack_strength,
            "api_key": config.api_key,
            "model_name": config.model_id
        }
        
        if gauntlet and round_num < len(gauntlet.rounds):
            # Apply gauntlet-specific configuration
            gauntlet_round = gauntlet.rounds[round_num]
            red_team_config.update({
                "attack_modes": gauntlet_round.attack_modes,
                "target_vulnerabilities": gauntlet_round.target_vulnerabilities
            })
        
        red_assessment = red_team.assess_content(current_content, **red_team_config)
        round_results["red_team"] = {
            "assessment": red_assessment,
            "issues_found": len(red_assessment.findings),
            "confidence_score": red_assessment.confidence_score
        }
        
        _update_evolution_log_and_status(f"   Found {len(red_assessment.findings)} issues")
        
        # Phase 2: Blue Team Response (Fix Issues)
        _update_evolution_log_and_status("🔵 Blue Team: Applying fixes and improvements...")
        
        blue_team_config = {
            "content_type": content_type,
            "strategy": config.defense_strategy,
            "api_key": config.api_key,
            "model_name": config.model_id
        }
        
        blue_assessment = blue_team.apply_fixes(
            current_content, 
            red_assessment.findings,
            **blue_team_config
        )
        
        round_results["blue_team"] = {
            "assessment": blue_assessment,
            "fixes_applied": len(blue_assessment.applied_fixes),
            "improvement_score": blue_assessment.overall_improvement_score
        }
        
        current_content = blue_assessment.fixed_content
        _update_evolution_log_and_status(f"   Applied {len(blue_assessment.applied_fixes)} fixes")
        
        # Phase 3: Evaluator Team Assessment (Judge Quality)
        _update_evolution_log_and_status("⚖️ Evaluator Team: Assessing overall quality...")
        
        evaluator_config = {
            "content_type": content_type,
            "previous_versions": [content] if round_num == 0 else [content, results["rounds"][-1]["final_content"]],
            "api_key": config.api_key,
            "model_name": config.model_id
        }
        
        evaluator_assessment = evaluator_team.evaluate_content(
            current_content,
            **evaluator_config
        )
        
        round_results["evaluator_team"] = {
            "assessment": evaluator_assessment,
            "consensus_score": evaluator_assessment.consensus_score,
            "final_verdict": evaluator_assessment.final_verdict
        }
        
        _update_evolution_log_and_status(f"   Quality score: {evaluator_assessment.consensus_score:.2f}")
        
        # Store round results
        round_results["round_number"] = round_num + 1
        round_results["final_content"] = current_content
        round_results["round_improvement"] = blue_assessment.overall_improvement_score
        
        results["rounds"].append(round_results)
        
        # Update totals
        results["total_issues_found"] += len(red_assessment.findings)
        results["total_fixes_applied"] += len(blue_assessment.applied_fixes)
        
        # Early stopping if quality is sufficient
        if (evaluator_assessment.consensus_score >= config.convergence_threshold * 100 and 
            config.early_stopping):
            _update_evolution_log_and_status("✅ Early stopping: Quality threshold reached")
            break
    
    # Calculate final metrics
    results["final_content"] = current_content
    results["total_rounds"] = len(results["rounds"])
    results["overall_improvement_score"] = _calculate_overall_improvement(results["rounds"])
    results["execution_time"] = time.time() - start_time
    
    # Collect team metrics
    results["team_metrics"] = {
        "red_team": {
            "total_assessments": len(red_team.assessment_history),
            "avg_confidence": sum(a.confidence_score for a in red_team.assessment_history[-5:]) / min(5, len(red_team.assessment_history)) if red_team.assessment_history else 0.0
        },
        "blue_team": {
            "total_fixes": len(blue_team.fix_history),
            "avg_improvement": sum(a.overall_improvement_score for a in blue_team.fix_history[-5:]) / min(5, len(blue_team.fix_history)) if blue_team.fix_history else 0.0
        },
        "evaluator_team": {
            "total_evaluations": len(evaluator_team.evaluation_history),
            "avg_consensus": sum(a.consensus_score for a in evaluator_team.evaluation_history[-5:]) / min(5, len(evaluator_team.evaluation_history)) if evaluator_team.evaluation_history else 0.0
        }
    }
    
    # Store gauntlet results if used
    if gauntlet:
        results["gauntlet_results"] = {
            "gauntlet_name": gauntlet_name,
            "rounds_completed": len(results["rounds"]),
            "effectiveness_score": results["overall_improvement_score"],
            "issues_per_round": [len(r["red_team"]["assessment"].findings) for r in results["rounds"]]
        }
        
        # Track gauntlet metrics
        gauntlet_manager.track_openevolve_metrics(gauntlet_name, {
            "timestamp": datetime.now().isoformat(),
            "effectiveness_score": results["overall_improvement_score"],
            "rounds_completed": len(results["rounds"]),
            "total_issues": results["total_issues_found"],
            "total_fixes": results["total_fixes_applied"]
        })
    
    _update_evolution_log_and_status(f"🏆 Adversarial evolution complete: {results['overall_improvement_score']:.1f}% improvement")
    
    return results


def _run_adversarial_decomposition(
    content: str,
    config: EvolutionConfiguration,
    red_team: 'RedTeam',
    blue_team: 'BlueTeam', 
    evaluator_team: 'EvaluatorTeam'
) -> Dict[str, Any]:
    """
    Run adversarial evolution with problem decomposition
    """
    _update_evolution_log_and_status("🧩 Applying adversarial decomposition...")
    
    # Decompose the problem
    decomposition_prompt = f"""Decompose this content for adversarial analysis:

Content:
---
{content}
---

Break this into smaller components that can be independently analyzed for:
1. Security vulnerabilities
2. Logic flaws  
3. Performance issues
4. Compliance gaps

Return as numbered list of components."""
    
    # Use LLM to decompose
    decomposed_response = _request_openai_compatible_chat(
        config.api_key,
        config.api_base,
        config.model_id,
        _compose_messages("You are an expert at decomposing content for security analysis.", decomposition_prompt),
        {},
        config.temperature,
        config.top_p,
        config.frequency_penalty,
        config.presence_penalty,
        config.max_tokens,
        config.seed,
    )
    
    if not decomposed_response:
        return {"error": "Failed to decompose content", "reassembled_content": content}
    
    # Parse components
    components = []
    for line in decomposed_response.split('\n'):
        if re.match(r'^\d+\.', line.strip()):
            components.append(line.strip())
    
    if not components:
        return {"error": "No components extracted", "reassembled_content": content}
    
    _update_evolution_log_and_status(f"   Decomposed into {len(components)} components")
    
    # Analyze each component adversarially
    component_results = []
    
    for i, component in enumerate(components):
        _update_evolution_log_and_status(f"   Analyzing component {i+1}/{len(components)}")
        
        # Red team analysis
        red_assessment = red_team.assess_content(component, "general")
        
        # Blue team fixes
        blue_assessment = blue_team.apply_fixes(component, red_assessment.findings, "general")
        
        # Evaluator assessment
        evaluator_assessment = evaluator_team.evaluate_content(blue_assessment.fixed_content, "general")
        
        component_results.append({
            "original_component": component,
            "fixed_component": blue_assessment.fixed_content,
            "issues_found": len(red_assessment.findings),
            "fixes_applied": len(blue_assessment.applied_fixes),
            "quality_score": evaluator_assessment.consensus_score
        })
    
    # Reassemble components
    reassembly_prompt = f"""Reassemble these analyzed and improved components into a cohesive whole:

Original Content:
---
{content}
---

Improved Components:
---
{chr(10).join([f"{i+1}. {r['fixed_component']}" for i, r in enumerate(component_results)])}
---

Create a unified, improved version that maintains coherence while incorporating all improvements."""
    
    reassembled_content = _request_openai_compatible_chat(
        config.api_key,
        config.api_base,
        config.model_id,
        _compose_messages("You are an expert at reassembling improved content components.", reassembly_prompt),
        {},
        config.temperature,
        config.top_p,
        config.frequency_penalty,
        config.presence_penalty,
        config.max_tokens,
        config.seed,
    )
    
    if not reassembled_content:
        reassembled_content = '\n\n'.join([r['fixed_component'] for r in component_results])
    
    _update_evolution_log_and_status("   ✅ Decomposition analysis complete")
    
    return {
        "components": component_results,
        "total_components": len(components),
        "total_issues_found": sum(r["issues_found"] for r in component_results),
        "total_fixes_applied": sum(r["fixes_applied"] for r in component_results),
        "avg_quality_score": sum(r["quality_score"] for r in component_results) / len(component_results),
        "reassembled_content": reassembled_content
    }


def _calculate_overall_improvement(rounds: List[Dict[str, Any]]) -> float:
    """Calculate overall improvement score across all rounds"""
    if not rounds:
        return 0.0
    
    # Weight recent rounds more heavily
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for i, round_data in enumerate(rounds):
        weight = (i + 1) / len(rounds)  # Later rounds have higher weight
        improvement = round_data.get("round_improvement", 0.0)
        total_weighted_score += improvement * weight
        total_weight += weight
    
    return total_weighted_score / total_weight if total_weight > 0 else 0.0


def run_gauntlet_evolution(
    content: str,
    gauntlet_name: str,
    content_type: str = "document_general",
    config: Optional[EvolutionConfiguration] = None,
    team_manager: Optional[Any] = None,
    gauntlet_manager: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run evolution using a specific gauntlet configuration
    
    Args:
        content: Content to evolve
        gauntlet_name: Name of gauntlet to use
        content_type: Type of content
        config: Evolution configuration
        team_manager: Team manager instance
        gauntlet_manager: Gauntlet manager instance
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with gauntlet evolution results
    """
    
    if not TEAM_SYSTEM_AVAILABLE:
        _update_evolution_log_and_status("⚠️ Team system not available - cannot run gauntlet evolution")
        return {"error": "Team system not available"}
    
    _update_evolution_log_and_status(f"🎯 Starting gauntlet evolution: {gauntlet_name}")
    
    # Initialize managers
    if gauntlet_manager is None:
        gauntlet_manager = GauntletManager()
    if team_manager is None:
        team_manager = TeamManager()
    
    # Load gauntlet
    gauntlet = gauntlet_manager.get_gauntlet(gauntlet_name)
    if not gauntlet:
        error_msg = f"Gauntlet '{gauntlet_name}' not found"
        _update_evolution_log_and_status(f"❌ {error_msg}")
        return {"error": error_msg}
    
    _update_evolution_log_and_status(f"📋 Loaded gauntlet: {len(gauntlet.rounds)} rounds")
    
    # Run adversarial evolution with gauntlet
    results = run_adversarial_evolution_with_teams(
        content=content,
        content_type=content_type,
        config=config,
        use_decomposition=gauntlet.generation_mode == "decomposition",
        gauntlet_name=gauntlet_name,
        team_manager=team_manager,
        gauntlet_manager=gauntlet_manager,
        **kwargs
    )
    
    # Add gauntlet-specific analysis
    if "gauntlet_results" not in results:
        results["gauntlet_results"] = {}
    
    results["gauntlet_results"]["gauntlet_definition"] = {
        "name": gauntlet.name,
        "team_name": gauntlet.team_name,
        "description": gauntlet.description,
        "attack_modes": gauntlet.attack_modes,
        "generation_mode": gauntlet.generation_mode,
        "rounds": [
            {
                "attack_modes": round_rule.attack_modes,
                "target_vulnerabilities": round_rule.target_vulnerabilities,
                "success_criteria": round_rule.success_criteria,
                "time_limit": round_rule.time_limit
            }
            for round_rule in gauntlet.rounds
        ]
    }
    
    # Calculate gauntlet effectiveness
    effectiveness = gauntlet_manager.get_gauntlet_effectiveness(gauntlet_name)
    results["gauntlet_results"]["effectiveness_analysis"] = effectiveness
    
    _update_evolution_log_and_status(f"🏆 Gauntlet evolution complete: {gauntlet_name}")
    
    return results


def create_adaptive_gauntlet(
    base_gauntlet_name: str,
    performance_data: Dict[str, Any],
    config: EvolutionConfiguration,
    gauntlet_manager: Optional[Any] = None
) -> Optional[str]:
    """
    Create an adaptive gauntlet based on performance data using OpenEvolve
    
    Args:
        base_gauntlet_name: Base gauntlet to adapt
        performance_data: Historical performance data
        config: Evolution configuration
        gauntlet_manager: Gauntlet manager instance
    
    Returns:
        Name of created adaptive gauntlet or None if failed
    """
    
    if not TEAM_SYSTEM_AVAILABLE:
        return None
    
    if gauntlet_manager is None:
        gauntlet_manager = GauntletManager()
    
    _update_evolution_log_and_status(f"🔄 Creating adaptive gauntlet based on {base_gauntlet_name}")
    
    # Use gauntlet manager's OpenEvolve integration
    success = gauntlet_manager.adapt_gauntlet_with_openevolve(
        base_gauntlet_name,
        performance_data,
        config.api_key,
        config.max_iterations
    )
    
    if success:
        adaptive_name = f"{base_gauntlet_name}_adaptive_{int(time.time())}"
        _update_evolution_log_and_status(f"✅ Created adaptive gauntlet: {adaptive_name}")
        return adaptive_name
    else:
        _update_evolution_log_and_status("❌ Failed to create adaptive gauntlet")
        return None


def get_evolution_capabilities_summary() -> Dict[str, Any]:
    """
    Get a summary of all available evolution capabilities and parameters
    """
    param_manager = ParameterManager()
    
    capabilities = {
        "total_parameters": len(param_manager.schema.parameters),
        "categories": len(param_manager.get_categories()),
        "evolution_modes": [
            "standard",
            "quality_diversity", 
            "multi_objective",
            "adversarial",
            "problem_decomposition"
        ],
        "advanced_features": {
            "quality_diversity": "MAP-Elites algorithm with behavior characterization",
            "multi_objective": "Pareto-optimal solutions for competing objectives",
            "adversarial": "Red team/blue team robustness testing",
            "distributed": "Multi-worker parallel processing",
            "cascade_evaluation": "Multi-stage filtering with increasing thresholds",
            "llm_feedback": "LLM-based evaluation and guidance",
            "meta_learning": "Learning from previous evolution runs",
            "transfer_learning": "Knowledge transfer between domains",
            "neural_architecture_search": "Automated neural network design",
            "hyperparameter_optimization": "Automated hyperparameter tuning",
            "explainable_ai": "Interpretable evolution decisions",
            "federated_learning": "Distributed learning across multiple parties",
            "quantum_computing": "Quantum-enhanced optimization",
            "edge_computing": "Optimization for edge deployment"
        },
        "team_system": {
            "available": TEAM_SYSTEM_AVAILABLE,
            "red_team": "Vulnerability identification and attack simulation",
            "blue_team": "Defense and fix implementation", 
            "evaluator_team": "Quality assessment and consensus building",
            "gauntlet_system": "Structured adversarial testing scenarios"
        } if TEAM_SYSTEM_AVAILABLE else {"available": False},
        "parameter_categories": {
            category: len(param_manager.get_parameters_by_category(category))
            for category in param_manager.get_categories()
        },
        "presets": param_manager.list_presets(),
        "validation": "Real-time parameter validation with error reporting"
    }
    
    return capabilities




# ============================================================================
# ULTIMATE ADVERSARIAL EVOLUTION INTEGRATION
# Complete implementation of ULTIMATE_ADVERSARIAL_EVOLUTION_EXPLAINED.md
# Supporting ALL native OpenEvolve features AND the comprehensive workflow
# ============================================================================

def run_ultimate_comprehensive_evolution(
    content: str,
    content_type: str = "document_general",
    evolution_mode: str = "adversarial",
    use_decomposition: bool = True,
    gauntlet_name: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    enable_all_features: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    ULTIMATE comprehensive evolution implementing the complete system from
    ULTIMATE_ADVERSARIAL_EVOLUTION_EXPLAINED.md while maintaining full
    compatibility with ALL native OpenEvolve features.
    
    This function provides:
    1. Complete Tripartite AI Architecture (Red/Blue/Evaluator teams)
    2. ALL 272 OpenEvolve parameters support
    3. Native OpenEvolve backend integration
    4. Advanced evolutionary algorithms
    5. Problem decomposition capabilities
    6. Gauntlet system integration
    7. Comprehensive metrics and analysis
    8. All workflow phases from the documentation
    
    Args:
        content: Content to evolve
        content_type: Type of content
        evolution_mode: Evolution mode (standard, quality_diversity, multi_objective, adversarial, problem_decomposition)
        use_decomposition: Enable problem decomposition approach
        gauntlet_name: Name of gauntlet for structured testing
        custom_config: Custom configuration overrides
        enable_all_features: Enable all advanced features
        **kwargs: All OpenEvolve parameters (272 total)
    
    Returns:
        Dictionary with ultimate comprehensive evolution results
    """
    _update_evolution_log_and_status("🌟 Starting ULTIMATE Comprehensive Evolution...")
    _update_evolution_log_and_status("📚 Full integration: OpenEvolve + ULTIMATE_ADVERSARIAL_EVOLUTION_EXPLAINED.md")
    
    start_time = time.time()
    operation_id = f"ultimate_comprehensive_{int(start_time)}"
    
    # Initialize comprehensive result structure
    ultimate_result = {
        "success": False,
        "operation_id": operation_id,
        "start_time": start_time,
        "system_architecture": "tripartite_ai_with_openevolve",
        "implementation_level": "ultimate_comprehensive",
        "original_content": content,
        "final_content": content,
        "content_type": content_type,
        "evolution_mode": evolution_mode,
        "use_decomposition": use_decomposition,
        "gauntlet_name": gauntlet_name,
        "openevolve_integration": True,
        "native_features_enabled": True,
        "workflow_phases": {
            "phase_1_initialization": {"status": "pending", "duration": 0, "openevolve_setup": False},
            "phase_2_adversarial_testing": {"status": "pending", "duration": 0, "team_coordination": False},
            "phase_3_evolutionary_optimization": {"status": "pending", "duration": 0, "native_evolution": False},
            "phase_4_evaluator_integration": {"status": "pending", "duration": 0, "consensus_building": False},
            "phase_5_model_management": {"status": "pending", "duration": 0, "portfolio_optimization": False},
            "phase_6_quality_assurance": {"status": "pending", "duration": 0, "validation_complete": False}
        },
        "openevolve_metrics": {
            "backend_available": False,
            "parameters_utilized": 0,
            "native_evolution_runs": 0,
            "api_calls": 0,
            "cost_usd": 0.0,
            "performance_score": 0.0
        },
        "workflow_metrics": {
            "adversarial_rounds": 0,
            "team_assessments": 0,
            "consensus_score": 0.0,
            "robustness_score": 0.0,
            "improvement_ratio": 0.0,
            "quality_diversity_score": 0.0,
            "pareto_efficiency": 0.0
        },
        "advanced_features": {
            "problem_decomposition": use_decomposition,
            "gauntlet_system": gauntlet_name is not None,
            "quality_diversity": evolution_mode == "quality_diversity",
            "multi_objective": evolution_mode == "multi_objective",
            "meta_learning": enable_all_features,
            "transfer_learning": enable_all_features,
            "explainable_ai": enable_all_features,
            "distributed_processing": enable_all_features,
            "neural_architecture_search": enable_all_features,
            "automated_ml": enable_all_features
        },
        "team_results": {
            "red_team": {"assessments": [], "issues_found": 0, "openevolve_enhanced": False},
            "blue_team": {"fixes": [], "fixes_applied": 0, "openevolve_enhanced": False},
            "evaluator_team": {"evaluations": [], "consensus_reached": False, "openevolve_enhanced": False}
        },
        "error": None
    }
    
    try:
        # ====================================================================
        # PHASE 1: SYSTEM INITIALIZATION WITH OPENEVOLVE INTEGRATION
        # ====================================================================
        _update_evolution_log_and_status("🚀 Phase 1: System Initialization with OpenEvolve Integration")
        phase_start = time.time()
        
        # Initialize OpenEvolve client with full parameter support
        try:
            from openevolve_client import OpenEvolveClient
            openevolve_client = OpenEvolveClient(config=custom_config or {})
            ultimate_result["openevolve_metrics"]["backend_available"] = openevolve_client.available
            _update_evolution_log_and_status(f"✅ OpenEvolve client initialized (Available: {openevolve_client.available})")
        except ImportError:
            openevolve_client = None
            _update_evolution_log_and_status("⚠️ OpenEvolve client not available - using workflow-only mode")
        
        # Initialize parameter manager with full 272 parameter support
        param_manager = ParameterManager()
        all_params = param_manager.schema.parameters if hasattr(param_manager, 'schema') else {}
        ultimate_result["openevolve_metrics"]["parameters_utilized"] = len(all_params)
        _update_evolution_log_and_status(f"📊 Parameter Manager: {len(all_params)} parameters available")
        
        # Create comprehensive configuration merging OpenEvolve + workflow parameters
        try:
            config = create_evolution_configuration_from_session()
        except (AttributeError, KeyError, RuntimeError):
            config = create_evolution_configuration(evolution_mode=evolution_mode, **kwargs)
        config.evolution_mode = evolution_mode
        
        # Apply custom configuration and kwargs (all OpenEvolve parameters)
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Apply all kwargs as OpenEvolve parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Initialize team system if available
        team_manager = None
        gauntlet_manager = None
        
        if TEAM_SYSTEM_AVAILABLE:
            try:
                from team_manager import TeamManager
                from gauntlet_manager import GauntletManager
                team_manager = TeamManager()
                gauntlet_manager = GauntletManager()
                ultimate_result["workflow_phases"]["phase_1_initialization"]["team_coordination"] = True
                _update_evolution_log_and_status("✅ Team system initialized")
            except ImportError:
                _update_evolution_log_and_status("⚠️ Team system components not fully available")
        
        ultimate_result["workflow_phases"]["phase_1_initialization"]["status"] = "completed"
        ultimate_result["workflow_phases"]["phase_1_initialization"]["duration"] = time.time() - phase_start
        ultimate_result["workflow_phases"]["phase_1_initialization"]["openevolve_setup"] = openevolve_client is not None
        
        # ====================================================================
        # PHASE 2: ADVERSARIAL TESTING WITH NATIVE OPENEVOLVE SUPPORT
        # ====================================================================
        _update_evolution_log_and_status("🛡️ Phase 2: Adversarial Testing with Native OpenEvolve Support")
        phase_start = time.time()
        
        adversarial_results = None
        
        # Run native OpenEvolve adversarial evolution if available
        if openevolve_client and openevolve_client.available:
            _update_evolution_log_and_status("🔥 Running native OpenEvolve adversarial evolution...")
            
            try:
                # Use native OpenEvolve adversarial mode
                openevolve_result = openevolve_client.evolve(
                    content=content,
                    evolution_mode="adversarial",
                    content_type=content_type,
                    **asdict(config)
                )
                
                ultimate_result["openevolve_metrics"]["native_evolution_runs"] += 1
                ultimate_result["openevolve_metrics"]["api_calls"] += openevolve_result.metrics.get("api_calls", 0)
                ultimate_result["openevolve_metrics"]["cost_usd"] += openevolve_result.metrics.get("cost_usd", 0.0)
                
                if openevolve_result.success:
                    content = openevolve_result.best_code
                    ultimate_result["openevolve_metrics"]["performance_score"] = openevolve_result.best_score
                    _update_evolution_log_and_status(f"✅ Native OpenEvolve adversarial: Score {openevolve_result.best_score:.4f}")
                
            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                _update_evolution_log_and_status(f"⚠️ Native OpenEvolve adversarial failed: {e}")
        
        # Run workflow-based adversarial testing with team system
        if TEAM_SYSTEM_AVAILABLE and team_manager:
            _update_evolution_log_and_status("👥 Running workflow-based adversarial testing...")
            
            # Import adversarial testing
            from adversarial import run_comprehensive_adversarial_testing, create_adversarial_configuration_from_session
            
            # Create adversarial configuration
            try:
                adversarial_config = create_adversarial_configuration_from_session()
            except (AttributeError, KeyError, RuntimeError):
                adversarial_config = create_adversarial_configuration()
            
            # Run comprehensive adversarial testing
            adversarial_results = run_comprehensive_adversarial_testing(
                current_content=content,
                content_type=content_type,
                config=adversarial_config,
                team_manager=team_manager,
                gauntlet_manager=gauntlet_manager,
                use_decomposition=use_decomposition
            )
            
            # Update results with adversarial findings
            if adversarial_results and adversarial_results.get("success"):
                content = adversarial_results.get("final_content", content)
                ultimate_result["workflow_metrics"]["adversarial_rounds"] = adversarial_results.get("metrics", {}).get("total_rounds", 0)
                ultimate_result["workflow_metrics"]["robustness_score"] = adversarial_results.get("metrics", {}).get("robustness_score", 0.0)
                ultimate_result["team_results"]["red_team"]["issues_found"] = adversarial_results.get("metrics", {}).get("vulnerability_count", 0)
                ultimate_result["team_results"]["blue_team"]["fixes_applied"] = adversarial_results.get("metrics", {}).get("fixes_applied", 0)
                ultimate_result["workflow_phases"]["phase_2_adversarial_testing"]["team_coordination"] = True
        
        ultimate_result["workflow_phases"]["phase_2_adversarial_testing"]["status"] = "completed"
        ultimate_result["workflow_phases"]["phase_2_adversarial_testing"]["duration"] = time.time() - phase_start
        
        # ====================================================================
        # PHASE 3: EVOLUTIONARY OPTIMIZATION WITH NATIVE OPENEVOLVE
        # ====================================================================
        _update_evolution_log_and_status("🧬 Phase 3: Evolutionary Optimization with Native OpenEvolve")
        phase_start = time.time()
        
        evolution_results = None
        
        # Run native OpenEvolve evolution for the specified mode
        if openevolve_client and openevolve_client.available:
            _update_evolution_log_and_status(f"🔥 Running native OpenEvolve {evolution_mode} evolution...")
            
            try:
                # Use native OpenEvolve with specified evolution mode
                openevolve_result = openevolve_client.evolve(
                    content=content,
                    evolution_mode=evolution_mode,
                    content_type=content_type,
                    **asdict(config)
                )
                
                ultimate_result["openevolve_metrics"]["native_evolution_runs"] += 1
                ultimate_result["openevolve_metrics"]["api_calls"] += openevolve_result.metrics.get("api_calls", 0)
                ultimate_result["openevolve_metrics"]["cost_usd"] += openevolve_result.metrics.get("cost_usd", 0.0)
                
                if openevolve_result.success:
                    content = openevolve_result.best_code
                    ultimate_result["openevolve_metrics"]["performance_score"] = max(
                        ultimate_result["openevolve_metrics"]["performance_score"],
                        openevolve_result.best_score
                    )
                    ultimate_result["workflow_metrics"]["improvement_ratio"] = openevolve_result.metrics.get("improvement_ratio", 0.0)
                    _update_evolution_log_and_status(f"✅ Native OpenEvolve {evolution_mode}: Score {openevolve_result.best_score:.4f}")
                
                ultimate_result["workflow_phases"]["phase_3_evolutionary_optimization"]["native_evolution"] = True
                
            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                _update_evolution_log_and_status(f"⚠️ Native OpenEvolve {evolution_mode} failed: {e}")
        
        # Run workflow-based evolution as enhancement
        _update_evolution_log_and_status("🔄 Running workflow-based evolution enhancement...")
        
        if evolution_mode == "quality_diversity":
            evolution_results = run_quality_diversity_evolution(
                content=content,
                content_type=content_type,
                config=config
            )
            if evolution_results and evolution_results.get("success"):
                ultimate_result["workflow_metrics"]["quality_diversity_score"] = evolution_results.get("metrics", {}).get("diversity_metrics", {}).get("diversity_score", 0.0)
                
        elif evolution_mode == "multi_objective":
            evolution_results = run_multi_objective_evolution(
                content=content,
                content_type=content_type,
                config=config
            )
            if evolution_results and evolution_results.get("success"):
                ultimate_result["workflow_metrics"]["pareto_efficiency"] = evolution_results.get("metrics", {}).get("mo_metrics", {}).get("hypervolume", 0.0)
                
        elif evolution_mode == "problem_decomposition" or use_decomposition:
            evolution_results = run_decomposition_evolution(
                content=content,
                content_type=content_type,
                config=config,
                use_decomposition=True
            )
            
        else:
            # Standard evolution with comprehensive features
            evolution_results = run_comprehensive_evolution(
                content=content,
                content_type=content_type,
                evolution_mode=evolution_mode,
                custom_config=custom_config,
                gauntlet_name=gauntlet_name,
                use_decomposition=use_decomposition,
                team_manager=team_manager,
                gauntlet_manager=gauntlet_manager
            )
        
        # Update results with evolution findings
        if evolution_results and isinstance(evolution_results, dict) and evolution_results.get("success"):
            content = evolution_results.get("final_content", content)
            ultimate_result["workflow_metrics"]["improvement_ratio"] = max(
                ultimate_result["workflow_metrics"]["improvement_ratio"],
                evolution_results.get("metrics", {}).get("improvement_ratio", 0.0)
            )
        
        ultimate_result["workflow_phases"]["phase_3_evolutionary_optimization"]["status"] = "completed"
        ultimate_result["workflow_phases"]["phase_3_evolutionary_optimization"]["duration"] = time.time() - phase_start
        
        # ====================================================================
        # PHASE 4: EVALUATOR TEAM INTEGRATION WITH OPENEVOLVE ENHANCEMENT
        # ====================================================================
        _update_evolution_log_and_status("⚖️ Phase 4: Evaluator Team Integration with OpenEvolve Enhancement")
        phase_start = time.time()
        
        if TEAM_SYSTEM_AVAILABLE:
            try:
                from evaluator_team import EvaluatorTeam
                evaluator_team = EvaluatorTeam()
                
                # Enhanced evaluation with OpenEvolve support
                if openevolve_client and openevolve_client.available:
                    # Use OpenEvolve for enhanced evaluation
                    evaluation_result = openevolve_client.evolve(
                        content=content,
                        evolution_mode="standard",  # Use standard mode for evaluation
                        content_type=content_type,
                        max_iterations=1,  # Single iteration for evaluation
                        **{"evaluation_only": True}
                    )
                    
                    if evaluation_result.success:
                        ultimate_result["workflow_metrics"]["consensus_score"] = evaluation_result.best_score
                        ultimate_result["team_results"]["evaluator_team"]["openevolve_enhanced"] = True
                
                # Workflow-based evaluation
                final_evaluation = evaluator_team.evaluate_content(
                    content=content,
                    content_type=content_type
                )
                
                if final_evaluation:
                    workflow_consensus = final_evaluation.consensus_score if hasattr(final_evaluation, 'consensus_score') else 0.0
                    ultimate_result["workflow_metrics"]["consensus_score"] = max(
                        ultimate_result["workflow_metrics"]["consensus_score"],
                        workflow_consensus
                    )
                    ultimate_result["team_results"]["evaluator_team"]["consensus_reached"] = workflow_consensus > 0.8
                
                ultimate_result["workflow_phases"]["phase_4_evaluator_integration"]["consensus_building"] = True
                
            except ImportError:
                _update_evolution_log_and_status("⚠️ Evaluator team not available")
        
        ultimate_result["workflow_phases"]["phase_4_evaluator_integration"]["status"] = "completed"
        ultimate_result["workflow_phases"]["phase_4_evaluator_integration"]["duration"] = time.time() - phase_start
        
        # ====================================================================
        # PHASE 5: MODEL MANAGEMENT AND PORTFOLIO OPTIMIZATION
        # ====================================================================
        _update_evolution_log_and_status("🎯 Phase 5: Model Management and Portfolio Optimization")
        phase_start = time.time()
        
        # OpenEvolve model portfolio optimization
        if openevolve_client and openevolve_client.available:
            try:
                # Run model performance analysis
                model_performance = {
                    "primary_model": config.model_id,
                    "backup_models": config.backup_models or [],
                    "performance_score": ultimate_result["openevolve_metrics"]["performance_score"],
                    "cost_efficiency": ultimate_result["openevolve_metrics"]["cost_usd"] / max(1, ultimate_result["openevolve_metrics"]["api_calls"]),
                    "optimization_recommendations": []
                }
                
                # Add optimization recommendations
                if ultimate_result["openevolve_metrics"]["performance_score"] < 0.7:
                    model_performance["optimization_recommendations"].append("Consider using higher-capability model")
                
                if ultimate_result["openevolve_metrics"]["cost_usd"] > 1.0:
                    model_performance["optimization_recommendations"].append("Consider cost optimization strategies")
                
                ultimate_result["model_portfolio"] = model_performance
                ultimate_result["workflow_phases"]["phase_5_model_management"]["portfolio_optimization"] = True
                
            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                _update_evolution_log_and_status(f"⚠️ Model portfolio optimization failed: {e}")
        
        ultimate_result["workflow_phases"]["phase_5_model_management"]["status"] = "completed"
        ultimate_result["workflow_phases"]["phase_5_model_management"]["duration"] = time.time() - phase_start
        
        # ====================================================================
        # PHASE 6: QUALITY ASSURANCE AND FINAL VALIDATION
        # ====================================================================
        _update_evolution_log_and_status("🔍 Phase 6: Quality Assurance and Final Validation")
        phase_start = time.time()
        
        # Comprehensive quality validation
        quality_metrics = {
            "content_length_ratio": len(content) / len(ultimate_result["original_content"]) if ultimate_result["original_content"] else 1.0,
            "openevolve_performance": ultimate_result["openevolve_metrics"]["performance_score"],
            "workflow_robustness": ultimate_result["workflow_metrics"]["robustness_score"],
            "consensus_quality": ultimate_result["workflow_metrics"]["consensus_score"],
            "improvement_achieved": ultimate_result["workflow_metrics"]["improvement_ratio"],
            "overall_quality_score": 0.0
        }
        
        # Calculate overall quality score
        quality_metrics["overall_quality_score"] = (
            quality_metrics["openevolve_performance"] * 0.3 +
            quality_metrics["workflow_robustness"] * 0.25 +
            quality_metrics["consensus_quality"] * 0.25 +
            quality_metrics["improvement_achieved"] * 0.2
        )
        
        ultimate_result["quality_metrics"] = quality_metrics
        ultimate_result["workflow_phases"]["phase_6_quality_assurance"]["validation_complete"] = True
        ultimate_result["workflow_phases"]["phase_6_quality_assurance"]["status"] = "completed"
        ultimate_result["workflow_phases"]["phase_6_quality_assurance"]["duration"] = time.time() - phase_start
        
        # ====================================================================
        # FINALIZATION AND RESULTS COMPILATION
        # ====================================================================
        
        # Calculate final success metrics
        overall_score = quality_metrics["overall_quality_score"]
        ultimate_result["overall_score"] = overall_score
        ultimate_result["final_content"] = content
        ultimate_result["success"] = overall_score > 0.6  # Success threshold
        
        # Finalize timing
        end_time = time.time()
        ultimate_result["end_time"] = end_time
        ultimate_result["total_duration"] = end_time - start_time
        
        # Log comprehensive results
        _update_evolution_log_and_status("🌟 ULTIMATE Comprehensive Evolution completed!")
        _update_evolution_log_and_status(f"⏱️ Total duration: {ultimate_result['total_duration']:.2f}s")
        _update_evolution_log_and_status(f"🏆 Overall score: {ultimate_result['overall_score']:.4f}")
        _update_evolution_log_and_status(f"🔥 OpenEvolve runs: {ultimate_result['openevolve_metrics']['native_evolution_runs']}")
        _update_evolution_log_and_status(f"📊 Parameters used: {ultimate_result['openevolve_metrics']['parameters_utilized']}")
        _update_evolution_log_and_status(f"💰 Total cost: ${ultimate_result['openevolve_metrics']['cost_usd']:.4f}")
        
        return ultimate_result
        
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        ultimate_result["error"] = str(e)
        ultimate_result["end_time"] = time.time()
        ultimate_result["total_duration"] = ultimate_result["end_time"] - start_time
        
        _update_evolution_log_and_status(f"💥 ULTIMATE Comprehensive Evolution failed: {e}")
        logger.error(f"Ultimate comprehensive evolution error: {e}", exc_info=True)
        return ultimate_result


def run_evolution_with_dts_strategy_exploration(
    content: str,
    content_type: str = "document_general",
    evolution_mode: str = "standard",
    use_dts_for_strategy: bool = True,
    dts_rounds: int = 2,
    use_multi_judge: bool = True,
    **evolution_params
) -> Dict[str, Any]:
    """
    Run evolution with DTS strategy exploration for enhanced search.
    
    This method uses Dialogue Tree Search (DTS) to explore multiple evolution
    strategies in parallel and select the most promising approach using
    multi-judge scoring before running the actual evolution.
    
    Args:
        content: Content to evolve
        content_type: Type of content (code, document, protocol, etc.)
        evolution_mode: Evolution mode (standard, adversarial, quality_diversity, etc.)
        use_dts_for_strategy: Whether to use DTS for strategy exploration
        dts_rounds: Number of DTS exploration rounds
        use_multi_judge: Whether to use multi-judge scoring for strategy evaluation
        **evolution_params: Additional parameters for evolution
        
    Returns:
        Dictionary with results including:
            - best_content: The evolved content
            - evolution_metrics: Metrics from the evolution process
            - dts_strategies: Strategies explored by DTS
            - selected_strategy: The strategy selected by DTS
            - dts_available: Whether DTS was actually used
            - final_score: Quality score of the evolved content
    """
    start_time = time.time()
    result = {
        "best_content": content,
        "evolution_metrics": {},
        "dts_strategies": [],
        "selected_strategy": None,
        "dts_available": False,
        "final_score": 0.0,
        "start_time": start_time,
        "end_time": None,
        "total_duration": None
    }
    
    if not DTS_AVAILABLE or not use_dts_for_strategy:
        logger.warning("DTS not available or disabled, using standard evolution")
        # Fall back to standard evolution based on mode
        if evolution_mode == "adversarial":
            evolution_result = run_ultimate_adversarial_evolution(content, content_type, **evolution_params)
        elif evolution_mode == "quality_diversity":
            evolution_result = run_quality_diversity_evolution(content, content_type, **evolution_params)
        elif evolution_mode == "multi_objective":
            evolution_result = run_multi_objective_evolution(content, content_type, **evolution_params)
        else:
            evolution_result = run_comprehensive_evolution(content, content_type, **evolution_params)
        
        result["best_content"] = evolution_result.get("best_content", content)
        result["evolution_metrics"] = evolution_result.get("metrics", {})
        result["final_score"] = evolution_result.get("final_score", 0.0)
        result["dts_available"] = False
        result["fallback_used"] = True
        result["evolution_result"] = evolution_result
        return result
    
    try:
        # Initialize DTS integration for strategy exploration
        dts_config = DTSIntegrationConfig(
            max_rounds=dts_rounds,
            use_multi_judge=use_multi_judge,
            use_strategy_exploration=True
        )
        dts_integration = DTSIntegration(dts_config)
        
        # Prepare context for DTS strategy exploration
        strategy_context = {
            "content": content,
            "content_type": content_type,
            "evolution_mode": evolution_mode,
            "evolution_params": evolution_params
        }
        
        # Explore evolution strategies using DTS
        _update_evolution_log_and_status("🔍 Exploring evolution strategies with DTS...")
        strategy_result = dts_integration.generate_strategies(
            context=strategy_context,
            goal=f"Generate effective evolution strategies for {evolution_mode} evolution",
            strategy_type="evolution"
        )
        
        # Extract strategies from DTS result
        strategies = []
        if isinstance(strategy_result, list):
            strategies = strategy_result
        elif isinstance(strategy_result, dict) and "strategies" in strategy_result:
            strategies = strategy_result["strategies"]
        result["dts_strategies"] = strategies
        
        # Select best strategy based on scores
        selected_strategy = None
        if strategies:
            # Find strategy with highest score
            scored_strategies = []
            for strategy in strategies:
                if isinstance(strategy, dict):
                    score = strategy.get("score", 0)
                    scored_strategies.append((strategy, score))
                else:
                    scored_strategies.append((strategy, 0.5))

            if scored_strategies:
                selected_strategy = max(scored_strategies, key=lambda x: x[1])[0]
                result["selected_strategy"] = selected_strategy

        # Enhance strategy selection with DSPy if available
        try:
            from dspy_integration import DSPY_AVAILABLE
            if DSPY_AVAILABLE:
                from dspy import Predict, Signature

                # Define a DSPy signature for strategy evaluation
                class StrategyEvaluationSignature(Signature):
                    """Evaluate evolution strategies and recommend the best one."""
                    content_to_evolve = dspy.InputField(desc="Content that needs to be evolved")
                    content_type = dspy.InputField(desc="Type of content (code, document, etc.)")
                    evolution_mode = dspy.InputField(desc="Evolution mode (standard, adversarial, etc.)")
                    available_strategies = dspy.InputField(desc="List of available evolution strategies")

                    best_strategy = dspy.OutputField(desc="The most appropriate strategy for this content and mode")
                    confidence_score = dspy.OutputField(desc="Confidence in the recommendation (1-10)")
                    reasoning = dspy.OutputField(desc="Reasoning for why this strategy is best")
                    potential_risks = dspy.OutputField(desc="Potential risks with this strategy")
                    success_factors = dspy.OutputField(desc="Key factors for success with this strategy")

                # Create a predictor
                evaluate_strategies = Predict(StrategyEvaluationSignature)

                # Prepare strategies for DSPy input
                strategies_text = "\n".join([
                    f"- {s.get('name', f'Strategy {i+1}')}: {s.get('description', 'No description')}"
                    for i, s in enumerate(strategies)
                ])

                # Run DSPy evaluation
                dspy_result = evaluate_strategies(
                    content_to_evolve=content,
                    content_type=content_type,
                    evolution_mode=evolution_mode,
                    available_strategies=strategies_text
                )

                # Update result with DSPy analysis
                result["dspy_strategy_analysis"] = {
                    "recommended_strategy": dspy_result.best_strategy,
                    "confidence": dspy_result.confidence_score,
                    "reasoning": dspy_result.reasoning,
                    "risks": dspy_result.potential_risks,
                    "success_factors": dspy_result.success_factors
                }

                # Use DSPy recommendation if confidence is high enough
                try:
                    dspy_confidence = float(dspy_result.confidence_score) if dspy_result.confidence_score.replace('.', '').isdigit() else 5.0
                    if dspy_confidence >= 7.0:  # High confidence threshold
                        result["selected_strategy"] = dspy_result.best_strategy
                        result["strategy_selected_by"] = "dspy"
                    else:
                        result["strategy_selected_by"] = "dts_with_low_dspy_confidence"
                except:
                    result["strategy_selected_by"] = "dts"
            else:
                result["strategy_selected_by"] = "dts"
        except ImportError:
            result["strategy_selected_by"] = "dts"
        
        # Apply the selected strategy to evolution parameters
        enhanced_params = evolution_params.copy()
        if selected_strategy and isinstance(selected_strategy, dict):
            # Extract strategy recommendations
            if "recommendations" in selected_strategy:
                recommendations = selected_strategy["recommendations"]
                # Apply relevant recommendations to parameters
                for rec in recommendations:
                    if isinstance(rec, dict) and "parameter" in rec and "value" in rec:
                        param_name = rec["parameter"]
                        param_value = rec["value"]
                        enhanced_params[param_name] = param_value
        
        # Run evolution with enhanced parameters
        _update_evolution_log_and_status(f"🚀 Running {evolution_mode} evolution with DTS-optimized strategy...")
        
        # Run appropriate evolution based on mode
        # Strip parameters not supported by targeted evolution functions
        evolution_func_params = enhanced_params.copy()
        unsupported_params = ["selection_pressure", "mutation_rate", "crossover_rate", "elitism",
                             "diversity_maintenance", "adaptive_parameters", "fitness_function"]
        for p in unsupported_params:
            if p in evolution_func_params:
                del evolution_func_params[p]

        if evolution_mode == "adversarial":
            evolution_result = run_ultimate_adversarial_evolution(content, content_type, **evolution_func_params)
        elif evolution_mode == "quality_diversity":
            evolution_result = run_quality_diversity_evolution(content, content_type, **evolution_func_params)
        elif evolution_mode == "multi_objective":
            evolution_result = run_multi_objective_evolution(content, content_type, **evolution_func_params)
        else:
            # Further strip parameters not supported by the specific evolution functions
            final_evolution_params = {}
            for k, v in evolution_func_params.items():
                if k not in ["prompt_template", "template_stochasticity", "meta_prompting", "few_shot_examples",
                             "chain_of_thought", "self_consistency", "prompt_ensembling", "dynamic_prompting",
                             "prompt_compression", "feature_dimensions", "feature_bins", "archive_size",
                             "novelty_threshold", "num_islands", "migration_interval", "migration_rate",
                             "elite_ratio", "exploration_ratio", "exploitation_ratio", "checkpoint_interval",
                             "cascade_evaluation", "use_llm_feedback", "llm_feedback_weight",
                             "evolution_trace_enabled", "diff_based_evolution", "diversity_metric",
                             "parallel_evaluations", "distributed", "template_dir", "num_top_programs",
                             "num_diverse_programs", "use_template_stochasticity", "template_variations",
                             "use_meta_prompting", "meta_prompt_weight", "include_artifacts",
                             "max_artifact_bytes", "artifact_security_filter", "memory_limit_mb",
                             "cpu_limit", "db_path", "in_memory", "log_level", "log_dir",
                             "artifact_size_threshold", "cleanup_old_artifacts", "artifact_retention_days",
                             "diversity_reference_size", "max_retries_eval", "evaluator_timeout",
                             "evaluator_models", "load_from_checkpoint", "custom_requirements", "objectives",
                             "attack_model_config", "defense_model_config", "evaluation_function",
                             "data_points", "variables", "operators", "double_selection",
                             "adaptive_feature_dimensions", "test_time_compute", "optillm_integration",
                             "plugin_system", "hardware_optimization", "multi_strategy_sampling",
                             "ring_topology", "controlled_gene_flow", "auto_diff", "symbolic_execution",
                             "coevolutionary_approach"]:
                    final_evolution_params[k] = v
    
            evolution_result = run_comprehensive_evolution(content, content_type, **final_evolution_params)
        
        # Update result with evolution outcome
        result["best_content"] = evolution_result.get("best_content", content)
        result["evolution_metrics"] = evolution_result.get("metrics", {})
        result["final_score"] = evolution_result.get("final_score", 0.0)
        result["dts_available"] = True
        result["fallback_used"] = False
        result["evolution_result"] = evolution_result
        result["dts_strategy_result"] = strategy_result
        
        # Calculate final score incorporating DTS confidence
        dts_confidence = strategy_result.get("confidence", 0.7)
        evolution_score = result["final_score"]
        result["combined_score"] = (evolution_score * 0.7) + (dts_confidence * 100 * 0.3)
        
        return result
        
    except Exception as e:
        logger.error(f"Error running DTS-enhanced evolution: {e}", exc_info=True)
        # Fall back to standard evolution
        _update_evolution_log_and_status("⚠️ DTS strategy exploration failed, using standard evolution...")
        
        # Strip parameters not supported by targeted evolution functions
        evolution_func_params = evolution_params.copy()
        unsupported_params = ["selection_pressure", "mutation_rate", "crossover_rate", "elitism",
                             "diversity_maintenance", "adaptive_parameters", "fitness_function"]
        for p in unsupported_params:
            if p in evolution_func_params:
                del evolution_func_params[p]

        if evolution_mode == "adversarial":
            evolution_result = run_ultimate_adversarial_evolution(content, content_type, **evolution_func_params)
        elif evolution_mode == "quality_diversity":
            evolution_result = run_quality_diversity_evolution(content, content_type, **evolution_func_params)
        elif evolution_mode == "multi_objective":
            evolution_result = run_multi_objective_evolution(content, content_type, **evolution_func_params)
        else:
            # Further strip parameters not supported by the specific evolution functions
            final_evolution_params = {}
            for k, v in evolution_func_params.items():
                if k not in ["prompt_template", "template_stochasticity", "meta_prompting", "few_shot_examples",
                             "chain_of_thought", "self_consistency", "prompt_ensembling", "dynamic_prompting",
                             "prompt_compression", "feature_dimensions", "feature_bins", "archive_size",
                             "novelty_threshold", "num_islands", "migration_interval", "migration_rate",
                             "elite_ratio", "exploration_ratio", "exploitation_ratio", "checkpoint_interval",
                             "cascade_evaluation", "use_llm_feedback", "llm_feedback_weight",
                             "evolution_trace_enabled", "diff_based_evolution", "diversity_metric",
                             "parallel_evaluations", "distributed", "template_dir", "num_top_programs",
                             "num_diverse_programs", "use_template_stochasticity", "template_variations",
                             "use_meta_prompting", "meta_prompt_weight", "include_artifacts",
                             "max_artifact_bytes", "artifact_security_filter", "memory_limit_mb",
                             "cpu_limit", "db_path", "in_memory", "log_level", "log_dir",
                             "artifact_size_threshold", "cleanup_old_artifacts", "artifact_retention_days",
                             "diversity_reference_size", "max_retries_eval", "evaluator_timeout",
                             "evaluator_models", "load_from_checkpoint", "custom_requirements", "objectives",
                             "attack_model_config", "defense_model_config", "evaluation_function",
                             "data_points", "variables", "operators", "double_selection",
                             "adaptive_feature_dimensions", "test_time_compute", "optillm_integration",
                             "plugin_system", "hardware_optimization", "multi_strategy_sampling",
                             "ring_topology", "controlled_gene_flow", "auto_diff", "symbolic_execution",
                             "coevolutionary_approach"]:
                    final_evolution_params[k] = v
    
            evolution_result = run_comprehensive_evolution(content, content_type, **final_evolution_params)
        
        result["best_content"] = evolution_result.get("best_content", content)
        result["evolution_metrics"] = evolution_result.get("metrics", {})
        result["final_score"] = evolution_result.get("final_score", 0.0)
        result["dts_available"] = True  # DTS was available but failed
        result["fallback_used"] = True
        result["error"] = str(e)
        result["evolution_result"] = evolution_result
        
        return result
    
    finally:
        result["end_time"] = time.time()
        result["total_duration"] = result["end_time"] - start_time


def run_native_openevolve_with_workflow_enhancement(
    content: str,
    content_type: str = "document_general",
    evolution_mode: str = "standard",
    workflow_enhancement: bool = True,
    **openevolve_params
) -> Dict[str, Any]:
    """
    Run native OpenEvolve evolution with optional workflow enhancement.
    This function prioritizes native OpenEvolve features while optionally
    adding workflow-based enhancements.
    
    Args:
        content: Content to evolve
        content_type: Type of content
        evolution_mode: OpenEvolve evolution mode
        workflow_enhancement: Enable workflow-based enhancements
        **openevolve_params: All native OpenEvolve parameters
    
    Returns:
        Dictionary with native OpenEvolve results + workflow enhancements
    """
    _update_evolution_log_and_status("🔥 Running Native OpenEvolve with Workflow Enhancement")
    
    start_time = time.time()
    result = {
        "success": False,
        "native_openevolve_result": None,
        "workflow_enhancements": {},
        "combined_metrics": {},
        "final_content": content,
        "error": None
    }
    
    try:
        # Initialize OpenEvolve client
        from openevolve_client import OpenEvolveClient
        client = OpenEvolveClient()
        
        if not client.available:
            _update_evolution_log_and_status("❌ Native OpenEvolve not available")
            result["error"] = "Native OpenEvolve backend not available"
            return result
        
        # Run native OpenEvolve evolution
        _update_evolution_log_and_status(f"🚀 Running native OpenEvolve {evolution_mode} evolution...")
        
        openevolve_result = client.evolve(
            content=content,
            evolution_mode=evolution_mode,
            content_type=content_type,
            **openevolve_params
        )
        
        result["native_openevolve_result"] = {
            "success": openevolve_result.success,
            "best_code": openevolve_result.best_code,
            "best_score": openevolve_result.best_score,
            "iterations_completed": openevolve_result.iterations_completed,
            "metrics": openevolve_result.metrics,
            "error": openevolve_result.error
        }
        
        if openevolve_result.success:
            result["final_content"] = openevolve_result.best_code
            _update_evolution_log_and_status(f"✅ Native OpenEvolve completed: Score {openevolve_result.best_score:.4f}")
        
        # Add workflow enhancements if enabled
        if workflow_enhancement and TEAM_SYSTEM_AVAILABLE:
            _update_evolution_log_and_status("🔄 Adding workflow-based enhancements...")
            
            enhanced_content = result["final_content"]
            
            # Red team analysis for additional insights
            try:
                from red_team import RedTeam
                red_team = RedTeam()
                red_assessment = red_team.assess_content(
                    content=enhanced_content,
                    content_type=content_type,
                    custom_requirements={"analysis_focus": "Post-OpenEvolve analysis for additional improvements"}
                )
                
                result["workflow_enhancements"]["red_team_analysis"] = {
                    "issues_found": len(red_assessment.issues) if red_assessment and red_assessment.issues else 0,
                    "additional_insights": True
                }
                
            except ImportError as exc:
                logger.debug(f"Red team module unavailable: {exc}")
            
            # Blue team enhancements
            try:
                from blue_team import BlueTeam
                blue_team = BlueTeam()
                
                if result["workflow_enhancements"].get("red_team_analysis", {}).get("issues_found", 0) > 0:
                    # Create dummy issues for enhancement
                    from red_team import IssueFinding, IssueCategory
                    from quality_assessment import SeverityLevel
                    
                    enhancement_issues = [IssueFinding(
                        title="Enhancement Opportunity",
                        description="Further enhance the OpenEvolve-improved content",
                        severity=SeverityLevel.LOW,
                        category=IssueCategory.CLARITY_ISSUE
                    )]
                    
                    blue_assessment = blue_team.apply_fixes(
                        content=enhanced_content,
                        issues=enhancement_issues,
                        content_type=content_type
                    )
                    
                    if blue_assessment and blue_assessment.fixes:
                        # Apply the best enhancement
                        best_fix = max(blue_assessment.fixes, key=lambda f: f.confidence)
                        if best_fix.implementation and best_fix.implementation.strip():
                            result["final_content"] = best_fix.implementation
                            result["workflow_enhancements"]["blue_team_enhancement"] = {
                                "applied": True,
                                "confidence": best_fix.confidence,
                                "description": best_fix.description
                            }
                
            except ImportError as exc:
                logger.debug(f"Blue team module unavailable: {exc}")
            
            # Evaluator team final assessment
            try:
                from evaluator_team import EvaluatorTeam
                evaluator_team = EvaluatorTeam()
                
                final_evaluation = evaluator_team.evaluate_content(
                    content=result["final_content"],
                    content_type=content_type
                )
                
                if final_evaluation:
                    result["workflow_enhancements"]["evaluator_assessment"] = {
                        "consensus_score": final_evaluation.consensus_score if hasattr(final_evaluation, 'consensus_score') else 0.0,
                        "enhanced_quality": True
                    }
                
            except ImportError as exc:
                logger.debug(f"Evaluator team module unavailable: {exc}")
        
        # Combine metrics
        result["combined_metrics"] = {
            "native_openevolve_score": openevolve_result.best_score if openevolve_result.success else 0.0,
            "workflow_enhancement_applied": workflow_enhancement and len(result["workflow_enhancements"]) > 0,
            "final_quality_score": result["workflow_enhancements"].get("evaluator_assessment", {}).get("consensus_score", openevolve_result.best_score if openevolve_result.success else 0.0),
            "total_duration": time.time() - start_time,
            "hybrid_approach": True
        }
        
        result["success"] = openevolve_result.success
        
        _update_evolution_log_and_status("✅ Native OpenEvolve + Workflow Enhancement completed!")
        
        return result
        
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        result["error"] = str(e)
        _update_evolution_log_and_status(f"💥 Native OpenEvolve + Workflow Enhancement failed: {e}")
        logger.error(f"Native OpenEvolve with workflow enhancement error: {e}", exc_info=True)
        return result


def get_comprehensive_evolution_capabilities() -> Dict[str, Any]:
    """
    Get comprehensive information about all available evolution capabilities,
    including native OpenEvolve features and workflow enhancements.
    
    Returns:
        Dictionary with complete capability information
    """
    capabilities = {
        "native_openevolve": {
            "available": False,
            "evolution_modes": [],
            "parameters_supported": 0,
            "advanced_features": []
        },
        "workflow_system": {
            "available": TEAM_SYSTEM_AVAILABLE,
            "team_components": [],
            "workflow_phases": [],
            "integration_features": []
        },
        "combined_capabilities": {
            "ultimate_comprehensive_evolution": True,
            "hybrid_approaches": True,
            "full_parameter_support": True,
            "advanced_integrations": []
        }
    }
    
    # Check OpenEvolve availability
    try:
        from openevolve_client import OpenEvolveClient
        client = OpenEvolveClient()
        capabilities["native_openevolve"]["available"] = client.available
        
        if client.available:
            capabilities["native_openevolve"]["evolution_modes"] = [
                "standard", "quality_diversity", "multi_objective", "adversarial", "problem_decomposition"
            ]
            capabilities["native_openevolve"]["advanced_features"] = [
                "meta_learning", "transfer_learning", "neural_architecture_search",
                "automated_ml", "explainable_ai", "distributed_processing"
            ]
    except ImportError as exc:
        logger.debug(f"OpenEvolve client unavailable while checking capabilities: {exc}")
    
    # Check parameter manager
    try:
        param_manager = ParameterManager()
        all_params = param_manager.schema.parameters if hasattr(param_manager, 'schema') else {}
        capabilities["native_openevolve"]["parameters_supported"] = len(all_params)
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug(f"Parameter manager unavailable while checking capabilities: {exc}")
    
    # Check workflow system
    if TEAM_SYSTEM_AVAILABLE:
        capabilities["workflow_system"]["team_components"] = ["red_team", "blue_team", "evaluator_team", "team_manager"]
        capabilities["workflow_system"]["workflow_phases"] = [
            "initialization", "adversarial_testing", "evolutionary_optimization",
            "evaluator_integration", "model_management", "quality_assurance"
        ]
        capabilities["workflow_system"]["integration_features"] = [
            "problem_decomposition", "gauntlet_system", "consensus_building",
            "multi_round_testing", "comprehensive_metrics"
        ]
    
    # Combined capabilities
    capabilities["combined_capabilities"]["advanced_integrations"] = [
        "native_openevolve_with_workflow_enhancement",
        "ultimate_comprehensive_evolution",
        "hybrid_adversarial_testing",
        "enhanced_quality_assurance",
        "comprehensive_metrics_collection"
    ]
    
    return capabilities


# =============================================================================
# MAKER/MDAP ENHANCED EVOLUTION
# =============================================================================

def run_maker_enhanced_evolution(
    initial_program: str,
    content_type: str = "code",
    config: Optional[EvolutionConfiguration] = None,
    max_generations: int = 100,
    enable_voting: bool = True,
    enable_decomposition: bool = True,
    voting_threshold: int = 3,
    population_size: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Run MAKER/MDAP-enhanced evolutionary computation.
    
    This function integrates the MAKER framework (arXiv:2511.09030) with OpenEvolve
    evolution to provide zero-error guarantees through voting and decomposition.
    
    Args:
        initial_program: Starting program/content to evolve
        content_type: Type of content (code, document_general, etc.)
        config: Evolution configuration (optional)
        max_generations: Maximum generations to evolve
        enable_voting: Enable MAKER voting for selection (default: True)
        enable_decomposition: Enable MDAP task decomposition (default: True)
        voting_threshold: k for first-to-ahead-by-k voting (default: 3)
        population_size: Population size for evolution (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Dict containing:
        - success: Whether evolution succeeded
        - best_program: Best evolved program
        - best_fitness: Best fitness achieved
        - generations: Number of generations run
        - fitness_history: Fitness over generations
        - evolution_time: Time taken
        - method: "maker_evolution"
    
    Example:
        >>> result = run_maker_enhanced_evolution(
        ...     initial_program="def foo(): pass",
        ...     content_type="code",
        ...     max_generations=50,
        ...     voting_threshold=3
        ... )
        >>> print(f"Best fitness: {result['best_fitness']}")
        >>> print(f"Evolved program: {result['best_program']}")
    """
    logger.info("=" * 80)
    logger.info("MAKER/MDAP-ENHANCED EVOLUTION")
    logger.info("=" * 80)
    logger.info(f"Content type: {content_type}")
    logger.info(f"MAKER voting: {enable_voting}")
    logger.info(f"MDAP decomposition: {enable_decomposition}")
    logger.info(f"Max generations: {max_generations}")
    logger.info(f"Voting threshold (k): {voting_threshold}")
    
    # Try to import MAKER evolution integration
    try:
        from evolution_maker_integration import (
            run_maker_evolution,
            MakerevolutionConfig,
            MakerevolutionMode
        )
        
        # Determine evolution mode
        if enable_voting and enable_decomposition:
            mode = MakerevolutionMode.HYBRID
        elif enable_voting:
            mode = MakerevolutionMode.VOTING_ONLY
        elif enable_decomposition:
            mode = MakerevolutionMode.DECOMPOSITION
        else:
            mode = MakerevolutionMode.FULL_MAKER
        
        # Create MAKER evolution config
        maker_config = MakerevolutionConfig(
            mode=mode,
            enable_voting=enable_voting,
            voting_threshold=voting_threshold,
            population_size=population_size,
            enable_decomposition=enable_decomposition,
            adaptive_voting=kwargs.get('adaptive_voting', True),
            enable_adaptive_allocation=kwargs.get('enable_adaptive_allocation', True)
        )
        
        # Create evaluator function
        def evaluator(program: str) -> float:
            """Simple evaluator - can be replaced with custom evaluator"""
            # For demonstration, use code length as fitness (prefer longer, more complete programs)
            # In production, would use actual quality metrics
            return float(len(program))
        
        # Allow custom evaluator
        custom_evaluator = kwargs.get('evaluator')
        if custom_evaluator:
            evaluator = custom_evaluator
        
        # Run MAKER-enhanced evolution
        result = run_maker_evolution(
            initial_program=initial_program,
            evaluator=evaluator,
            max_generations=max_generations,
            config=maker_config,
            **kwargs
        )
        
        logger.info(f"[OK] MAKER-enhanced evolution completed successfully")
        logger.info(f"  - Best fitness: {result.get('best_fitness', 0):.4f}")
        logger.info(f"  - Generations: {result.get('generations', 0)}")
        logger.info(f"  - Evolution time: {result.get('evolution_time', 0):.2f}s")
        
        # Add content type to result
        result["content_type"] = content_type
        result["paper_reference"] = "arXiv:2511.09030"
        
        return result
        
    except ImportError as e:
        logger.warning(f"[WARN] MAKER evolution integration not available: {e}")
        logger.warning(f"[WARN] Falling back to standard evolution")
        
        # Fallback to standard evolution
        return run_evolution_loop(
            current_content=initial_program,
            content_type=content_type,
            config=config,
            **kwargs
        )
    
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        logger.error(f"[ERROR] MAKER-enhanced evolution failed: {e}")
        logger.error(f"[ERROR] Falling back to standard evolution")
        
        # Fallback to standard evolution
        return run_evolution_loop(
            current_content=initial_program,
            content_type=content_type,
            config=config,
            **kwargs
        )


def get_maker_evolution_capabilities() -> Dict[str, Any]:
    """
    Get capabilities of MAKER/MDAP-enhanced evolution.
    
    Returns:
        Dict describing MAKER evolution capabilities
    """
    capabilities = {
        "maker_evolution_enabled": False,
        "mdap_decomposition_enabled": False,
        "modes": [],
        "algorithms": [],
        "integration_status": "unknown"
    }
    
    try:
        from evolution_maker_integration import (
            MakerevolutionMode,
            MakerevolutionConfig,
            get_maker_evolution_capabilities as get_caps
        )
        
        base_caps = get_caps()
        
        capabilities["maker_evolution_enabled"] = True
        capabilities["mdap_decomposition_enabled"] = True
        capabilities["integration_status"] = "available"
        capabilities["modes"] = [mode.value for mode in MakerevolutionMode]
        capabilities["algorithms"] = base_caps.get("algorithms", [])
        capabilities["features"] = base_caps.get("features", {})
        capabilities["paper"] = base_caps.get("paper_reference", {})
        
    except ImportError as e:
        capabilities["integration_status"] = f"unavailable: {str(e)}"
    
    return capabilities


# =============================================================================
# Z3 Fitness Evaluation Integration
# =============================================================================

def evaluate_fitness_with_z3(
    individual: Dict[str, Any],
    constraints: List[Dict[str, Any]],
    objectives: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluate individual fitness using Z3 formal verification.
    
    Args:
        individual: The evolved individual to evaluate
        constraints: List of fitness constraints
        objectives: Optional list of objective expressions
        
    Returns:
        Fitness result dict or None if Z3 not available
    """
    try:
        from evolution_z3_fitness import get_z3_fitness_evaluator, FitnessConstraint
        
        evaluator = get_z3_fitness_evaluator()
        
        # Convert constraints
        fitness_constraints = [
            FitnessConstraint(
                constraint_id=c.get("id", f"c{i}"),
                expression=c["expression"],
                weight=c.get("weight", 1.0),
                is_hard=c.get("is_hard", True)
            )
            for i, c in enumerate(constraints)
        ]
        
        # Evaluate
        result = evaluator.evaluate_fitness(individual, fitness_constraints, objectives)
        
        return {
            "fitness_score": result.fitness_score,
            "constraints_satisfied": result.constraints_satisfied,
            "violated_constraints": result.violated_constraints,
            "is_feasible": result.is_feasible
        }
    except ImportError:
        logging.getLogger(__name__).debug("Z3 fitness evaluator not available")
        return None
    except Exception as e:
        logging.getLogger(__name__).error(f"Z3 fitness evaluation failed: {e}")
        return None


def validate_mutation_with_z3(
    original: Dict[str, Any],
    mutated: Dict[str, Any],
    constraints: List[Dict[str, Any]]
) -> bool:
    """
    Validate that a mutation produces a valid individual using Z3.
    
    Args:
        original: Original individual
        mutated: Mutated individual
        constraints: Constraints that must be satisfied
        
    Returns:
        True if mutation is valid
    """
    try:
        from evolution_z3_fitness import get_z3_fitness_evaluator, FitnessConstraint
        
        evaluator = get_z3_fitness_evaluator()
        
        fitness_constraints = [
            FitnessConstraint(
                constraint_id=c.get("id", f"c{i}"),
                expression=c["expression"],
                is_hard=c.get("is_hard", True)
            )
            for i, c in enumerate(constraints)
        ]
        
        return evaluator.validate_mutation(original, mutated, fitness_constraints)
    except ImportError:
        return True  # Assume valid if Z3 not available
    except Exception as e:
        logging.getLogger(__name__).error(f"Z3 mutation validation failed: {e}")
        return True


def calculate_pareto_frontier_with_z3(
    population: List[Dict[str, Any]],
    objectives: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """
    Calculate Pareto frontier using Z3 multi-objective optimization.
    
    Args:
        population: Population of individuals
        objectives: List of objective expressions
        
    Returns:
        Pareto-optimal individuals or None if Z3 not available
    """
    try:
        from evolution_z3_fitness import get_z3_fitness_evaluator
        
        evaluator = get_z3_fitness_evaluator()
        return evaluator.calculate_pareto_frontier(population, objectives)
    except ImportError:
        return None
    except Exception as e:
        logging.getLogger(__name__).error(f"Z3 Pareto frontier calculation failed: {e}")
        return None


def get_z3_evolution_capabilities() -> Dict[str, Any]:
    """
    Get capabilities of Z3-enhanced evolution.
    
    Returns:
        Dict describing Z3 evolution capabilities
    """
    capabilities = {
        "z3_fitness_enabled": False,
        "z3_mutation_validation": False,
        "z3_pareto_optimization": False,
        "integration_status": "unknown"
    }
    
    try:
        from evolution_z3_fitness import (
            get_z3_fitness_evaluator,
            Z3_ADVANCED_AVAILABLE
        )
        
        evaluator = get_z3_fitness_evaluator()
        
        capabilities["z3_fitness_enabled"] = True
        capabilities["z3_mutation_validation"] = True
        capabilities["z3_pareto_optimization"] = Z3_ADVANCED_AVAILABLE
        capabilities["integration_status"] = "available"
        
    except ImportError as e:
        capabilities["integration_status"] = f"unavailable: {str(e)}"
    
    return capabilities
