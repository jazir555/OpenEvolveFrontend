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
):
    """
    The main evolution loop.
    """
    for i in range(max_iterations):
        if st.session_state.evolution_stop_flag:
            _update_evolution_log_and_status("⏹️ Evolution stopped by user.")
            break

        _update_evolution_log_and_status(f"🔄 Iteration {i + 1}/{max_iterations}")
        _update_evolution_log_and_status("🧬 Generating new population...")

        if i % st.session_state.checkpoint_interval == 0:
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

            new_population = [future.result() for future in as_completed(futures)]

        _update_evolution_log_and_status("🔍 Evaluating new population...")

        best_candidate = ""
        best_score = -1

        # Evaluate new population
        with ThreadPoolExecutor(max_workers=population_size) as executor:
            futures = {
                executor.submit(
                    _evaluate_candidate,
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
                ): candidate
                for candidate in new_population
            }

            for future in as_completed(futures):
                candidate = futures[future]
                score = future.result()

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

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
        else:
            # If no numeric score found, evaluate based on keyword presence
            score = 0.5  # Default neutral score
            if (
                "good" in score_str.lower()
                or "improved" in score_str.lower()
                or "better" in score_str.lower()
            ):
                score = 0.8
            elif (
                "poor" in score_str.lower()
                or "bad" in score_str.lower()
                or "worse" in score_str.lower()
            ):
                score = 0.2
    except (ValueError, TypeError):
        score = 0.0
    return score


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
    with st.session_state.thread_lock:
        st.session_state.evolution_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    if "log_queue" in st.session_state and st.session_state.log_queue:
        st.session_state.log_queue.put(f"[{time.strftime('%H:%M:%S')}] {msg}")


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

def render_evolution_settings():
    """
    Placeholder function to render the evolution settings section in the Streamlit UI.
    This would typically allow users to configure parameters for the evolutionary algorithm.
    """
    st.header("🧬 Evolution Settings")
    st.info("Evolution settings management features are under development. Stay tuned!")
    # Example of how you might display evolution parameters:
    # st.subheader("Algorithm Parameters")
    # st.slider("Max Iterations", 1, 100, st.session_state.max_iterations)
    # st.slider("Population Size", 1, 50, st.session_state.population_size)
    # st.slider("Temperature", 0.0, 2.0, st.session_state.temperature)
