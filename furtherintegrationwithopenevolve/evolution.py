import streamlit as st
import requests
import json
import time
import tempfile
import os
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import OpenEvolve modules for code-specific features

try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig

    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available - using API-based evolution only")

# Import our deep integration module
try:
    from openevolve_integration import (
        run_advanced_code_evolution,
        OpenEvolveAPI,
    )
    DEEP_INTEGRATION_AVAILABLE = True and OPENEVOLVE_AVAILABLE
except ImportError:
    DEEP_INTEGRATION_AVAILABLE = False
    print("Deep OpenEvolve integration not available")


class ContentEvaluator:
    """
    A class to encapsulate content evaluation logic compatible with OpenEvolve.
    """

    def __init__(self, content_type: str, evaluator_system_prompt: str):
        self.content_type = content_type
        self.evaluator_system_prompt = evaluator_system_prompt

    def evaluate(self, program_path: str) -> Dict[str, Any]:
        """
        Evaluate the content of a file with metrics compatible with OpenEvolve.
        """
        try:
            with open(program_path, "r") as f:
                content = f.read()

            return self._evaluate_code(content)

        except Exception as e:
            # Return error metrics in a format that OpenEvolve can process
            return {
                "score": 0.0, 
                "error": str(e), 
                "timestamp": time.time(),
                "combined_score": 0.0,  # Critical for OpenEvolve
                "length": 0,
                "complexity": 0.0,
                "diversity": 0.0,
            }

    def _evaluate_code(self, content: str) -> Dict[str, Any]:
        """
        Evaluator for code content with OpenEvolve-compatible metrics.
        """
        # Calculate various metrics for OpenEvolve
        length = len(content)
        word_count = len(content.split())
        unique_words = len(set(content.split()))
        complexity = word_count / 50.0 if word_count > 0 else 0.0  # Basic complexity measure
        diversity = unique_words / word_count if word_count > 0 else 0.0  # Vocabulary diversity
        
        # Calculate score based on various factors
        length_score = min(1.0, length / 1000.0)  # Favor reasonable length
        diversity_score = min(1.0, diversity * 2.0)  # Encourage diversity
        score = (length_score * 0.4) + (diversity_score * 0.3) + (complexity * 0.01 * 0.3)
        score = min(1.0, score)  # Cap at 1.0
        
        return {
            "score": score,
            "length": length,
            "timestamp": time.time(),
            "combined_score": score,  # OpenEvolve compatibility
            "complexity": complexity,
            "diversity": diversity,
            "word_count": word_count,
            "unique_words": unique_words
        }

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
    checkpoint_interval: int = 10
):
    """
    The main evolution loop with enhanced capabilities that can utilize adversarial testing diagnostics.
    This function uses the OpenEvolve API for more sophisticated evolution.
    """
    try:
        # Always prefer OpenEvolve when available - this is the main implementation now
        if OPENEVOLVE_AVAILABLE:
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
            config.evolution.max_iterations = max_iterations
            config.evolution.population_size = population_size
            config.evolution.archive_size = archive_size
            config.evolution.checkpoint_interval = checkpoint_interval
            config.evolution.num_islands = st.session_state.get("num_islands", 1)  # Add island model for better exploration
            
            # Configure database settings for multi-objective evolution if needed
            if feature_dimensions is not None:
                config.database.feature_dimensions = feature_dimensions
            if feature_bins is not None:
                config.database.feature_bins = feature_bins
            else:
                # Set default feature bins if none provided
                config.database.feature_bins = 10
            
            # Configure ratios
            config.database.elite_selection_ratio = elite_ratio
            config.database.exploration_ratio = exploration_ratio
            config.database.exploitation_ratio = exploitation_ratio
            
            # Configure evaluator settings for better integration
            config.evaluator.timeout = 300
            config.evaluator.max_retries = 3
            config.evaluator.cascade_evaluation = True
            config.evaluator.cascade_thresholds = [0.5, 0.75, 0.9]
            config.evaluator.parallel_evaluations = os.cpu_count() or 4
            
            # Use adversarial diagnostics to inform the evolution process if enabled
            if use_adversarial_diagnostics:
                adversarial_history = st.session_state.get("integrated_adversarial_history", [])
                if adversarial_history:
                    # Add adversarial-relevant dimensions if they're not already present
                    if config.database.feature_dimensions is not None:
                        if "issues_resolved" not in config.database.feature_dimensions:
                            config.database.feature_dimensions.append("issues_resolved")
                        if "mitigation_effectiveness" not in config.database.feature_dimensions:
                            config.database.feature_dimensions.append("mitigation_effectiveness")

            # Create a temporary file for the content
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
                temp_file.write(current_content)
                temp_file_path = temp_file.name
            
            try:
                # Use OpenEvolve API with the evaluator
                from openevolve.api import run_evolution
                result = run_evolution(
                    initial_program=temp_file_path,
                    evaluator=evaluator.evaluate,
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
        else:
            st.error("OpenEvolve not available. Please install and run the backend.")
            return current_content
    except Exception as e:
        _update_evolution_log_and_status(f"💥 Evolution loop failed: {e}")
        import traceback
        traceback.print_exc()
        return current_content

def _update_evolution_log_and_status(msg: str):
    """Thread-safe way to update logs and status message with optimization."""
    # Only update if we're actually running evolution
    if not st.session_state.get("evolution_running", False):
        return
    
    current_time = time.strftime('%H:%M:%S')
    log_entry = f"[{current_time}] {msg}"
    
    # Throttling: Only allow updates every 100ms to prevent flooding
    last_update_key = "_last_evolution_log_update"
    current_timestamp = time.time()
    
    with st.session_state.thread_lock:
        # Check if this is actually a new message to avoid duplicates
        if st.session_state.evolution_log and st.session_state.evolution_log[-1] == log_entry:
            return
            
        # Throttling check
        if last_update_key in st.session_state:
            last_timestamp = st.session_state[last_update_key]
            if current_timestamp - last_timestamp < 0.1:  # 100ms minimum interval
                return
        
        st.session_state.evolution_log.append(log_entry)
        st.session_state.evolution_status_message = msg
        st.session_state[last_update_key] = current_timestamp
        
        # Limit log size to prevent memory issues
        if len(st.session_state.evolution_log) > 100:
            st.session_state.evolution_log = st.session_state.evolution_log[-100:]
    
    # Only update log queue if it exists and we have new content
    if "log_queue" in st.session_state and st.session_state.log_queue:
        try:
            st.session_state.log_queue.put(log_entry)
        except:
            pass  # Silently fail if queue is full or closed
