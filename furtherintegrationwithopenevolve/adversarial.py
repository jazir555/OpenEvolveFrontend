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
import requests
import json
import time
import threading
import traceback
import uuid
import random
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import OpenEvolve modules for backend integration

try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig

    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available - using API-based adversarial testing only")

from openevolve_integration import (
    create_language_specific_evaluator,
    create_specialized_evaluator,
)

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

            red_team_prompt, blue_team_prompt = "", ""

        # Prioritize OpenEvolve backend when available for all content types
        if OPENEVOLVE_AVAILABLE:
            st.info("Using OpenEvolve backend for adversarial testing...")
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
            st.error("OpenEvolve not available. Please install and run the backend.")

    except Exception as e:
        tb_str = traceback.format_exc()
        error_message = f"A critical error occurred: {e}\n{tb_str}"
        st.error(error_message)
        st.session_state.adversarial_results = {"critical_error": error_message}

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
) -> Dict[str, Any]:
    """
    Run adversarial testing using OpenEvolve backend for code content.

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

    Returns:
        Dict[str, Any]: Adversarial testing results
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend not available for adversarial testing")
        return {"success": False, "error": "OpenEvolve backend not available"}

    try:
        # Create OpenEvolve configuration
        config = Config()

        # Configure LLM models
        llm_models = []

        # Add red team models
        for model_id in red_team_models[:red_team_sample_size]:
            llm_config = LLMModelConfig(
                name=model_id,
                api_key=api_key,
                api_base=base_url if base_url else "https://api.openai.com/v1",
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                seed=seed,
            )
            llm_models.append(llm_config)

        # Add blue team models
        for model_id in blue_team_models[:blue_team_sample_size]:
            llm_config = LLMModelConfig(
                name=model_id,
                api_key=api_key,
                api_base=base_url if base_url else "https://api.openai.com/v1",
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                seed=seed,
            )
            llm_models.append(llm_config)

        config.llm.models = llm_models
        config.evolution.max_iterations = max_iterations
        config.evolution.population_size = max_workers
        config.evolution.num_islands = st.session_state.num_islands
        config.evolution.elite_ratio = st.session_state.elite_ratio
        config.evolution.exploration_ratio = st.session_state.exploration_ratio
        config.evolution.exploitation_ratio = st.session_state.exploitation_ratio
        config.evolution.archive_size = st.session_state.archive_size
        config.evolution.checkpoint_interval = st.session_state.checkpoint_interval

        # Configure database settings for multi-objective evolution
        if feature_dimensions is not None:
            config.database.feature_dimensions = feature_dimensions
        if feature_bins is not None:
            config.database.feature_bins = feature_bins

        # Create evaluator function based on content_type
        if content_type.startswith("code_"):
            evaluator_instance = create_specialized_evaluator(
                content_type, custom_requirements, compliance_rules
            )
        else:
            evaluator_instance = create_language_specific_evaluator(
                content_type, custom_requirements, compliance_rules
            )
        # Create temporary file for the content with proper evolution markers
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            # Add evolution markers to the content
            content_with_markers = f"""# EVOLVE-BLOCK-START
{current_content}
# EVOLVE-BLOCK-END"""
            temp_file.write(content_with_markers)
            temp_file_path = temp_file.name

        try:
            # Run adversarial testing using OpenEvolve API
            result = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=evaluator_instance.evaluate,
                config=config,
                iterations=max_iterations,
                output_dir=None,  # Use temporary directory
                cleanup=True,
            )

            # Process results
            if result.best_program and result.best_code:
                # Remove evolution markers from the final result
                best_code = result.best_code
                if "# EVOLVE-BLOCK-START" in best_code:
                    start_idx = best_code.find("# EVOLVE-BLOCK-START") + len(
                        "# EVOLVE-BLOCK-START"
                    )
                    end_idx = best_code.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_code = best_code[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": result.best_program,
                    "best_score": result.best_score,
                    "best_code": best_code,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                }
            else:
                return {
                    "success": False,
                    "message": "Adversarial testing completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running adversarial testing with OpenEvolve backend: {e}")
        print(f"Adversarial testing error: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}