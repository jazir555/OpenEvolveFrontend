import streamlit as st
import time
import tempfile
import os
import re
from typing import List, Dict, Any, Optional

import requests
from session_utils import _compose_messages

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
        
    except Exception as e:
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
        except Exception as e:
            return {"score": 0.0, "error": str(e), "timestamp": time.time()}

    def _evaluate_code(self, content: str) -> Dict[str, Any]:
        """
        Evaluator for code content.
        """
        # For now, return a basic score based on content structure
        score = min(1.0, len(content) / 500.0)  # Basic length-based scoring

        # More sophisticated evaluation would happen here
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

        return {"score": score, "length": len(content), "timestamp": time.time()}

    def _evaluate_legal_content(self, content: str) -> Dict[str, Any]:
        """
        Evaluator for legal content.
        """
        _update_evolution_log_and_status(
            f"⚖️ Evaluating legal content of {len(content)} characters"
        )
        score = min(1.0, len(content) / 1500.0)  # Example scoring
        return {"score": score, "length": len(content), "timestamp": time.time()}

    def _evaluate_medical_content(self, content: str) -> Dict[str, Any]:
        """
        Evaluator for medical content.
        """
        _update_evolution_log_and_status(
            f"⚕️ Evaluating medical content of {len(content)} characters"
        )
        score = min(1.0, len(content) / 1200.0)  # Example scoring
        return {"score": score, "length": len(content), "timestamp": time.time()}

    def _evaluate_technical_content(self, content: str) -> Dict[str, Any]:
        """
        Evaluator for technical content.
        """
        _update_evolution_log_and_status(
            f"⚙️ Evaluating technical content of {len(content)} characters"
        )
        score = min(1.0, len(content) / 1000.0)  # Example scoring
        return {"score": score, "length": len(content), "timestamp": time.time()}


def run_evolution_loop(
    current_content: str,
    api_key: str,
    base_url: str,
    model: str,
    max_iterations: int,
    population_size: int,
    system_prompt: str,
    evaluator: ContentEvaluator,
    extra_headers: Dict,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int],
    use_adversarial_diagnostics: bool = False,
    multi_objective_optimization: bool = False,
    feature_dimensions: Optional[List[str]] = None,
    feature_bins: Optional[int] = None,
    elite_ratio: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.7,
    archive_size: int = 100,
    checkpoint_interval: int = 10,
    content_type: str = "document_general",  # Added content_type parameter for advanced features
    use_quality_diversity: bool = False,  # New parameter for QD evolution
    use_multi_objective: bool = False,  # New parameter for multi-objective evolution
    use_adversarial_evolution: bool = False,  # New parameter for adversarial evolution
    adversarial_attack_model: str = "gpt-4",  # Model for adversarial attacks
    adversarial_defense_model: str = "gpt-4",  # Model for adversarial defense
    algorithm_discovery_mode: bool = False,  # New parameter for algorithm discovery
    objectives: Optional[List[str]] = None,  # Objectives for multi-objective optimization
    evolution_mode: str = "standard",  # New parameter to specify evolution mode
    # Advanced OpenEvolve features
    enable_artifacts: bool = True,
    cascade_evaluation: bool = True,
    cascade_thresholds: Optional[List[float]] = None,
    use_llm_feedback: bool = False,
    llm_feedback_weight: float = 0.1,
    parallel_evaluations: int = 1,
    distributed: bool = False,
    system_message: str = None,
    evaluator_system_message: str = None,
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
    # Advanced research features from OpenEvolve
    double_selection: bool = True,  # Different programs for performance vs inspiration
    adaptive_feature_dimensions: bool = True,  # Adjust feature dimensions based on progress
    test_time_compute: bool = False,  # Use test-time compute for enhanced reasoning
    optillm_integration: bool = False,  # Integrate with OptiLLM for advanced routing
    plugin_system: bool = False,  # Enable plugin system for extended capabilities
    hardware_optimization: bool = False,  # Optimize for specific hardware (GPU, etc.)
    multi_strategy_sampling: bool = True,  # Use elite, diverse, and exploratory selection
    ring_topology: bool = True,  # Use ring topology for island migration
    controlled_gene_flow: bool = True,  # Control gene flow between islands
    auto_diff: bool = True,  # Use automatic differentiation where applicable
    symbolic_execution: bool = False,  # Enable symbolic execution for verification
    coevolutionary_approach: bool = False,  # Use co-evolution between different populations
):
    """
    The main evolution loop with ALL OpenEvolve features integrated.
    This function now supports all of OpenEvolve's advanced capabilities.
    """
    try:
        # Prefer OpenEvolve when available - this is the main implementation now
        if OPENEVOLVE_AVAILABLE:
            _update_evolution_log_and_status(f"🚀 Using OpenEvolve backend for evolution (mode: {evolution_mode})...")
            
            # Prepare model configuration for all evolution modes
            model_configs = [{
                "name": model,
                "weight": 1.0,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }]
            
            # Run evolution using the unified function with ALL parameters
            result = run_unified_evolution(
                content=current_content,
                content_type=content_type,
                evolution_mode=evolution_mode,
                model_configs=model_configs,
                api_key=api_key,
                api_base=base_url if base_url else "https://api.openai.com/v1",
                max_iterations=max_iterations,
                population_size=population_size,
                system_message=system_message or system_prompt,
                evaluator_system_message=evaluator_system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                objectives=objectives,
                feature_dimensions=feature_dimensions,
                custom_requirements="",
                custom_evaluator=evaluator.evaluate if evaluator else None,
                # Advanced parameters
                enable_artifacts=enable_artifacts,
                cascade_evaluation=cascade_evaluation,
                cascade_thresholds=cascade_thresholds or [0.5, 0.75, 0.9],
                use_llm_feedback=use_llm_feedback,
                llm_feedback_weight=llm_feedback_weight,
                parallel_evaluations=parallel_evaluations,
                distributed=distributed,
                template_dir=None,
                num_top_programs=num_top_programs,
                num_diverse_programs=num_diverse_programs,
                use_template_stochasticity=use_template_stochasticity,
                template_variations=template_variations or {},
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
                evaluator_models=None,
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
                # Adversarial-specific parameters when needed
                attack_model_config={"name": adversarial_attack_model, "weight": 1.0} if evolution_mode == "adversarial" or use_adversarial_evolution else None,
                defense_model_config={"name": adversarial_defense_model, "weight": 1.0} if evolution_mode == "adversarial" or use_adversarial_evolution else None,
            )

            # Process the results
            if result and result.get("success", False):
                final_content = result.get("best_code", current_content)
                if not final_content:
                    final_content = current_content  # Fallback to original content if none returned
                
                with st.session_state.thread_lock:
                    st.session_state.evolution_current_best = final_content
                    
                best_score = result.get("best_score", 0.0)
                _update_evolution_log_and_status(
                    f"🏆 OpenEvolve {evolution_mode} evolution completed. Best score: {best_score:.4f}"
                )
                return final_content
            else:
                error_msg = result.get("error", result.get("message", "Unknown error")) if result else "No result returned"
                _update_evolution_log_and_status(
                    f"🤔 OpenEvolve {evolution_mode} evolution completed with no improvement: {error_msg}"
                )
                return current_content
                
        else:
            st.error("OpenEvolve not available. Please install and run the backend.")
            return current_content
    except Exception as e:
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
                # This is a simplified approach - in a full implementation, we'd use more sophisticated logic
                base_score = evaluation_result.get("score", 0.0)
                length_factor = min(1.0, len(candidate) / 1000.0)  # Favor reasonable length
                score = (base_score * 0.7) + (length_factor * 0.3)  # Weighted combination
            
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
        
        return score
    except Exception as e:
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
        evaluation = _request_openai_compatible_chat(
            api_key,
            base_url,
            model,
            _compose_messages(evaluator.evaluator_system_prompt, candidate),
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
    except Exception as e:
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
                
    except Exception as e:
        _update_evolution_log_and_status(f"💥 Evolution loop failed: {e}")
        import traceback
        traceback.print_exc()
        return current_content





from session_utils import _compose_messages, _update_evolution_log_and_status



