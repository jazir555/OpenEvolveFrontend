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
from typing import List, Dict, Any, Optional

from session_utils import _hash_text
from session_manager import APPROVAL_PROMPT, RED_TEAM_CRITIQUE_PROMPT, BLUE_TEAM_PATCH_PROMPT
from openevolve_integration import (
    create_language_specific_evaluator,
    create_specialized_evaluator,
    create_comprehensive_openevolve_config,
    run_unified_evolution,
)
from integrated_workflow import generate_adversarial_data_augmentation
from review_utils import determine_review_type, get_appropriate_prompts
from logging_util import _update_adv_log_and_status



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
    # Advanced OpenEvolve parameters
    enable_artifacts: bool = True,
    cascade_evaluation: bool = True,
    cascade_thresholds: Optional[List[float]] = None,
    use_llm_feedback: bool = False,
    llm_feedback_weight: float = 0.1,
    parallel_evaluations: int = 4,
    distributed: bool = False,
    template_dir: Optional[str] = None,
    num_top_programs: int = 3,
    num_diverse_programs: int = 2,
    use_template_stochasticity: bool = True,
    template_variations: Optional[Dict[str, List[str]]] = None,
    use_meta_prompting: bool = False,
    meta_prompt_weight: float = 0.1,
    include_artifacts: bool = True,
    max_artifact_bytes: int = 20 * 1024,
    artifact_security_filter: bool = True,
    early_stopping_patience: Optional[int] = None,
    convergence_threshold: float = 0.001,
    early_stopping_metric: str = "combined_score",
    memory_limit_mb: Optional[int] = None,
    cpu_limit: Optional[float] = None,
    random_seed: Optional[int] = 42,
    db_path: Optional[str] = None,
    in_memory: bool = True,
    # Additional parameters from OpenEvolve
    diff_based_evolution: bool = True,
    max_code_length: int = 10000,
    evolution_trace_enabled: bool = False,
    evolution_trace_format: str = "jsonl",
    evolution_trace_include_code: bool = False,
    evolution_trace_include_prompts: bool = True,
    evolution_trace_output_path: Optional[str] = None,
    evolution_trace_buffer_size: int = 10,
    evolution_trace_compress: bool = False,
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    api_timeout: int = 60,
    api_retries: int = 3,
    api_retry_delay: int = 5,
    artifact_size_threshold: int = 32 * 1024,
    cleanup_old_artifacts: bool = True,
    artifact_retention_days: int = 30,
    diversity_reference_size: int = 20,
    max_retries_eval: int = 3,
    evaluator_timeout: int = 300,
    evaluator_models: Optional[List[Dict[str, any]]] = None,
    # Advanced research features
    double_selection: bool = True,
    adaptive_feature_dimensions: bool = True,
    test_time_compute: bool = False,
    optillm_integration: bool = False,
    plugin_system: bool = False,
    hardware_optimization: bool = False,
    multi_strategy_sampling: bool = True,
    ring_topology: bool = True,
    controlled_gene_flow: bool = True,
    auto_diff: bool = True,
    symbolic_execution: bool = False,
    coevolutionary_approach: bool = False,
) -> Dict[str, Any]:
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
    """Run adversarial testing with content-type-aware routing to OpenEvolve backend."""
    print("run_adversarial_testing function called")
    # Load model performance data for continuous learning
    try:
        with open("model_performance.json", "r") as f:
            st.session_state.adversarial_model_performance = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.session_state.adversarial_model_performance = {}

    try:
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
                pass  # Invalid input, keep seed as None

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


