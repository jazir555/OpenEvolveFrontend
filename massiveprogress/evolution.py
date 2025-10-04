import streamlit as st
import requests
import json
import time
import tempfile
import os
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from session_utils import _clamp, _rand_jitter_ms, _compose_messages

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
    checkpoint_interval: int = 10
):
    """
    The main evolution loop with enhanced capabilities that can utilize adversarial testing diagnostics.
    """
    try:
        for i in range(max_iterations):
            # Check if we have adversarial stop flag (for integrated workflows)
            if st.session_state.get("adversarial_stop_flag", False):
                _update_evolution_log_and_status("⏹️ Evolution stopped due to adversarial stop flag.")
                break
                
            if st.session_state.evolution_stop_flag:
                _update_evolution_log_and_status("⏹️ Evolution stopped by user.")
                break

            _update_evolution_log_and_status(f"🔄 Iteration {i + 1}/{max_iterations}")
            _update_evolution_log_and_status("🧬 Generating new population...")

            if i % checkpoint_interval == 0:
                _update_evolution_log_and_status(f"💾 Saving checkpoint at iteration {i}")
                print(f"Saving checkpoint at iteration {i}")

            # Generate new population
            with ThreadPoolExecutor(max_workers=population_size) as executor:
                futures = [
                    executor.submit(
                        _request_openai_compatible_chat,
                        api_key,
                        base_url,
                        model,
                        _compose_messages(system_prompt, current_content),
                        extra_headers,
                        temperature,
                        top_p,
                        frequency_penalty,
                        presence_penalty,
                        max_tokens,
                        seed + i if seed is not None else None,
                    )
                    for i in range(population_size)
                ]

                new_population = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        new_population.append(result)
                    except Exception as e:
                        _update_evolution_log_and_status(f"❌ Error generating candidate: {e}")
                        # Add a placeholder if generation fails
                        new_population.append(current_content)

            _update_evolution_log_and_status("🔍 Evaluating new population...")

            best_candidate = ""
            best_score = -1

            # Use adversarial diagnostics to inform the evolution process if enabled
            if use_adversarial_diagnostics:
                adversarial_history = st.session_state.get("integrated_adversarial_history", [])
                if adversarial_history:
                    # Get the latest adversarial diagnostics to inform evolution
                    latest_diag = adversarial_history[-1]
                    # If there were many issues found in adversarial testing, 
                    # we should focus more on fixing those issues in evolution
                    if latest_diag.get("issues_found", 0) > latest_diag.get("mitigations", 0):
                        # Adjust system prompt to focus on fixing issues
                        improvement_system_prompt = f"{system_prompt}\n\nBased on adversarial testing results, there are still more issues than have been mitigated. Focus on addressing remaining issues and vulnerabilities identified during adversarial testing."
                        
                        # Generate alternative candidates specifically focused on addressing issues
                        with ThreadPoolExecutor(max_workers=max(1, population_size//2)) as executor:
                            improvement_futures = [
                                executor.submit(
                                    _request_openai_compatible_chat,
                                    api_key,
                                    base_url,
                                    model,
                                    _compose_messages(improvement_system_prompt, current_content),
                                    extra_headers,
                                    temperature,
                                    top_p,
                                    frequency_penalty,
                                    presence_penalty,
                                    max_tokens,
                                    seed + population_size + i if seed is not None else None,
                                )
                                for i in range(max(1, population_size//2))
                            ]
                            
                            for future in as_completed(improvement_futures):
                                try:
                                    result = future.result()
                                    new_population.append(result)
                                except Exception as e:
                                    _update_evolution_log_and_status(f"❌ Error generating improvement candidate: {e}")

            # Evaluate new population
            with ThreadPoolExecutor(max_workers=population_size) as executor:
                futures = {
                    executor.submit(
                        _evaluate_candidate_with_diagnostics,
                        candidate,
                        api_key,
                        base_url,
                        model,
                        evaluator,
                        extra_headers,
                        temperature,
                        top_p,
                        frequency_penalty,
                        presence_penalty,
                        max_tokens,
                        seed,
                        use_adversarial_diagnostics
                    ): candidate
                    for candidate in new_population
                }

                for future in as_completed(futures):
                    try:
                        candidate = futures[future]
                        score = future.result()

                        if score > best_score:
                            best_score = score
                            best_candidate = candidate
                    except Exception as e:
                        _update_evolution_log_and_status(f"❌ Error evaluating candidate: {e}")
                        continue

            if best_candidate and best_score > 0:
                current_content = best_candidate
                with st.session_state.thread_lock:
                    st.session_state.evolution_current_best = current_content
                _update_evolution_log_and_status(
                    f"🏆 New best candidate found with score: {best_score:.2f}"
                )
            else:
                _update_evolution_log_and_status("🤔 No improvement in this iteration.")

        _update_evolution_log_and_status("🏁 Evolution finished.")
        return current_content
    except Exception as e:
        _update_evolution_log_and_status(f"💥 Evolution loop failed: {e}")
        import traceback
        traceback.print_exc()
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


def _run_evolution_with_openevolve_backend_refactored(
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
    try:
        # Check if we have deep integration available
        if DEEP_INTEGRATION_AVAILABLE:
            _update_evolution_log_and_status(
                "🚀 Using advanced OpenEvolve integration..."
            )

            checkpoint_path = None
            if st.session_state.get(
                "load_checkpoint_triggered"
            ) and st.session_state.get("selected_checkpoint"):
                checkpoint_path = st.session_state.selected_checkpoint
                _update_evolution_log_and_status(
                    f"Loading evolution from checkpoint: {checkpoint_path}"
                )
                st.session_state.load_checkpoint_triggered = False

            # Run advanced code evolution with enhanced settings
            result = run_advanced_code_evolution(
                content=current_content,
                content_type=content_type,
                model_name=model,
                api_key=api_key,
                api_base=base_url if base_url else "https://api.openai.com/v1",
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_iterations=max_iterations,
                population_size=st.session_state.population_size,
                num_islands=st.session_state.num_islands,
                archive_size=st.session_state.archive_size,
                elite_ratio=st.session_state.elite_ratio,
                exploration_ratio=st.session_state.exploration_ratio,
                exploitation_ratio=st.session_state.exploitation_ratio,
                checkpoint_interval=st.session_state.checkpoint_interval,
                language=st.session_state.language,
                file_suffix=st.session_state.file_suffix,
                reasoning_effort=st.session_state.reasoning_effort,
                feature_dimensions=st.session_state.feature_dimensions,
                feature_bins=st.session_state.feature_bins,
                custom_requirements=evaluator_system_prompt,
                frequency_penalty=st.session_state.frequency_penalty,
                presence_penalty=st.session_state.presence_penalty,
                seed=st.session_state.seed,
                diversity_metric=st.session_state.diversity_metric,
                checkpoint_path=checkpoint_path,
            )

            if result and result.get("success"):
                evolution_id = result.get("evolution_id")
                if st.session_state.get("save_checkpoint_triggered") and evolution_id:
                    api = OpenEvolveAPI(
                        base_url=st.session_state.openevolve_base_url,
                        api_key=st.session_state.openevolve_api_key,
                    )
                    if api.save_checkpoint(evolution_id):
                        _update_evolution_log_and_status(
                            f"Checkpoint saved for evolution ID: {evolution_id}"
                        )
                    else:
                        _update_evolution_log_and_status(
                            f"Failed to save checkpoint for evolution ID: {evolution_id}"
                        )
                    st.session_state.save_checkpoint_triggered = False

                best_code = result.get("best_code", "")
                best_score = result.get("best_score", 0.0)

                with st.session_state.thread_lock:
                    st.session_state.evolution_current_best = best_code

                _update_evolution_log_and_status(
                    f"🏆 Advanced OpenEvolve evolution completed. Best score: {best_score:.4f}"
                )
                _update_evolution_log_and_status(
                    f"📄 Best content length: {len(best_code)} characters"
                )
                return
            elif result:
                _update_evolution_log_and_status(
                    result.get(
                        "message",
                        "🤔 OpenEvolve evolution completed with no improvement.",
                    )
                )
                return
        else:
            # Fall back to basic OpenEvolve integration
            _update_evolution_log_and_status("🚀 Using basic OpenEvolve integration...")

            # Create OpenEvolve configuration
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
            config.evolution.population_size = st.session_state.population_size
            config.evolution.num_islands = st.session_state.num_islands
            config.evolution.elite_ratio = st.session_state.elite_ratio
            config.evolution.exploration_ratio = st.session_state.exploration_ratio
            config.evolution.exploitation_ratio = st.session_state.exploitation_ratio
            config.evolution.archive_size = st.session_state.archive_size
            config.evolution.checkpoint_interval = st.session_state.checkpoint_interval

            # Create evaluator
            evaluator = ContentEvaluator(content_type, evaluator_system_prompt)

            # Create temporary file for the content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as temp_file:
                # Add evolution markers for code content
                content_with_markers = f"""# EVOLVE-BLOCK-START
{current_content}
# EVOLVE-BLOCK-END"""
                temp_file.write(content_with_markers)
                temp_file_path = temp_file.name

            try:
                # Run evolution using OpenEvolve API
                result = openevolve_run_evolution(
                    initial_program=temp_file_path,
                    evaluator=evaluator.evaluate,
                    config=config,
                    iterations=max_iterations,
                    output_dir=None,  # Use temporary directory
                    cleanup=True,
                )

                # Update session state with results
                if result.best_program and result.best_code:
                    with st.session_state.thread_lock:
                        # Remove evolution markers from the final result
                        best_code = result.best_code
                        if "# EVOLVE-BLOCK-START" in best_code:
                            start_idx = best_code.find("# EVOLVE-BLOCK-START") + len(
                                "# EVOLVE-BLOCK-START"
                            )
                            end_idx = best_code.find("# EVOLVE-BLOCK-END")
                            if end_idx != -1:
                                best_code = best_code[start_idx:end_idx].strip()
                        st.session_state.evolution_current_best = best_code
                    _update_evolution_log_and_status(
                        f"🏆 OpenEvolve evolution completed. Best score: {result.best_score:.4f}"
                    )
                    _update_evolution_log_and_status(
                        f"📄 Best content length: {len(best_code)} characters"
                    )
                else:
                    _update_evolution_log_and_status(
                        "🤔 OpenEvolve evolution completed with no improvement."
                    )

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

    except Exception as e:
        _update_evolution_log_and_status(f"💥 OpenEvolve error: {str(e)}")
        print(f"OpenEvolve error: {e}")
        import traceback

        traceback.print_exc()
        # Fallback to API-based approach
        _update_evolution_log_and_status("⚠️ Falling back to API-based evolution...")
        api_key = st.session_state.api_key
        base_url = st.session_state.base_url
        model = st.session_state.model
        extra_headers = json.loads(st.session_state.extra_headers)
        max_iterations = st.session_state.max_iterations
        population_size = st.session_state.population_size

        _run_evolution_with_api_backend_refactored(
            current_content,
            api_key,
            base_url,
            model,
            max_iterations,
            population_size,
            system_prompt,
            evaluator_system_prompt,
            extra_headers,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            max_tokens,
            seed,
        )


def _run_evolution_with_api_backend_refactored(
    current_content,
    api_key,
    base_url,
    model,
    max_iterations,
    population_size,
    system_prompt,
    evaluator_system_prompt,
    extra_headers,
    temperature,
    top_p,
    frequency_penalty,
    presence_penalty,
    max_tokens,
    seed,
):
    """Run evolution using API-based approach for general content."""
    evaluator = ContentEvaluator("general", evaluator_system_prompt)
    run_evolution_loop(
        current_content,
        api_key,
        base_url,
        model,
        max_iterations,
        population_size,
        system_prompt,
        evaluator,
        extra_headers,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        max_tokens,
        seed,
    )


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


@st.cache_data(ttl=3600) # Cache for 1 hour
def _request_openai_compatible_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: List,
    extra_headers: Dict,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    seed: Optional[int],
    req_timeout: int = 60,
    max_retries: int = 5,
    provider: str = "OpenAI",
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **extra_headers,
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": _clamp(temperature, 0.0, 2.0),
        "max_tokens": max_tokens,
        "top_p": _clamp(top_p, 0.0, 1.0),
        "frequency_penalty": _clamp(frequency_penalty, -2.0, 2.0),
        "presence_penalty": _clamp(presence_penalty, -2.0, 2.0),
    }
    if seed is not None:
        payload["seed"] = int(seed)

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
            if r.status_code in {429, 500, 502, 503, 504}:
                sleep_s = (2**attempt) + _rand_jitter_ms()
                time.sleep(sleep_s)
                last_err = Exception(f"Transient error {r.status_code}: Retrying...")
                continue
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                content = choice.get("message", {}).get("content", "")
            else:
                content = ""
            return content or ""
        except Exception as e:
            last_err = e
            sleep_s = (2**attempt) + _rand_jitter_ms()
            time.sleep(sleep_s)
    raise RuntimeError(
        f"Request failed for {model} after {max_retries} attempts: {last_err}"
    )