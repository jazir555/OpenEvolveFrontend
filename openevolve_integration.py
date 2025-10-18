"""
Deep integration with OpenEvolve backend for enhanced code evolution capabilities.
"""

import streamlit as st
import tempfile
import os
import json as json_module
import time
from typing import Dict, Any, Optional, List, Callable, Iterator, Union
from dataclasses import asdict
import requests

# Import OpenEvolve modules
try:
    from openevolve.api import (
        run_evolution as openevolve_run_evolution,
        evolve_function as openevolve_evolve_function,
        evolve_algorithm as openevolve_evolve_algorithm,
        evolve_code as openevolve_evolve_code,
        EvolutionResult,
    )
    from openevolve.config import (
        Config,
        LLMModelConfig,
        DatabaseConfig,
        EvaluatorConfig,
        PromptConfig,
        EvolutionTraceConfig,
    )


    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    # Define fallback types when OpenEvolve is not available
    class Config:
        """Fallback Config class when OpenEvolve is not available"""
        pass
    
    st.warning("OpenEvolve backend not available - using API-based evolution only")


from llm_utils import _request_openai_compatible_chat


class OpenEvolveAPI:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key # Store api_key as an attribute
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get(self, path: str) -> requests.Response:
        """Makes a GET request to the OpenEvolve backend."""
        url = f"{self.base_url}{path}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)  # Add timeout
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError:
            st.error(f"Connection error: Could not connect to OpenEvolve backend at {url}")
            raise
        except requests.exceptions.Timeout:
            st.error(f"Timeout error: Request to {url} timed out")
            raise
        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {e}")
            raise

    def start_evolution(
        self, config: Dict, checkpoint_path: Optional[str] = None
    ) -> Optional[str]:
        try:
            payload = {"config": config}
            if checkpoint_path:
                payload["checkpoint_path"] = checkpoint_path
            url = f"{self.base_url}/evolutions"
            response = requests.post(
                url, json=payload, headers=self.headers, timeout=30
            )
            response.raise_for_status()
            return response.json().get("evolution_id")
        except requests.exceptions.ConnectionError:
            st.error(f"Connection error: Could not connect to OpenEvolve backend at {self.base_url}/evolutions")
            return None
        except requests.exceptions.Timeout:
            st.error(f"Timeout error: Request to {self.base_url}/evolutions timed out")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error starting evolution: {e}")
            return None

    @st.cache_data(ttl=300) # Cache for 5 minutes
    def get_checkpoints(_self) -> List[str]:
        # For local checkpointing, we list files in a predefined directory
        checkpoint_dir = os.path.join(os.getcwd(), "openevolve_checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            return []
        
        checkpoints = []
        for item in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, item)):
                checkpoints.append(item)
        return sorted(checkpoints, reverse=True)

    def get_evolution_status(self, evolution_id: str) -> Optional[Dict]:
        try:
            url = f"{self.base_url}/evolutions/{evolution_id}"
            response = requests.get(
                url, headers=self.headers, timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Connection error: Could not connect to OpenEvolve backend at {self.base_url}/evolutions/{evolution_id}")
            return None
        except requests.exceptions.Timeout:
            st.error(f"Timeout error: Request to {self.base_url}/evolutions/{evolution_id} timed out")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting evolution status: {e}")
            return None

    def get_best_solution(self, evolution_id: str) -> Optional[Dict]:
        try:
            url = f"{self.base_url}/evolutions/{evolution_id}/best"
            response = requests.get(
                url, headers=self.headers, timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Connection error: Could not connect to OpenEvolve backend at {self.base_url}/evolutions/{evolution_id}/best")
            return None
        except requests.exceptions.Timeout:
            st.error(f"Timeout error: Request to {self.base_url}/evolutions/{evolution_id}/best timed out")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting best solution: {e}")
            return None

    def get_evolution_history(self, evolution_id: str) -> Optional[List[Dict]]:
        try:
            url = f"{self.base_url}/evolutions/{evolution_id}/history"
            response = requests.get(
                url,
                headers=self.headers,
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Connection error: Could not connect to OpenEvolve backend at {self.base_url}/evolutions/{evolution_id}/history")
            return None
        except requests.exceptions.Timeout:
            st.error(f"Timeout error: Request to {self.base_url}/evolutions/{evolution_id}/history timed out")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting evolution history: {e}")
            return None

    def stream_evolution_logs(self, evolution_id: str) -> Iterator[str]:
        try:
            url = f"{self.base_url}/evolutions/{evolution_id}/logs"
            with requests.get(
                url,
                headers=self.headers,
                stream=True,
                timeout=30,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    yield chunk.decode("utf-8")
        except requests.exceptions.ConnectionError:
            st.error(f"Connection error: Could not connect to OpenEvolve backend at {self.base_url}/evolutions/{evolution_id}/logs")
            return
        except requests.exceptions.Timeout:
            st.error(f"Timeout error: Request to {self.base_url}/evolutions/{evolution_id}/logs timed out")
            return
        except requests.exceptions.RequestException as e:
            st.error(f"Error streaming evolution logs: {e}")
            return




    def upload_evaluator(self, evaluator_code: str) -> Optional[str]:
        try:
            url = f"{self.base_url}/evaluators"
            response = requests.post(
                url,
                json={"code": evaluator_code},
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("evaluator_id")
        except requests.exceptions.ConnectionError:
            st.error(f"Connection error: Could not connect to OpenEvolve backend at {self.base_url}/evaluators")
            return None
        except requests.exceptions.Timeout:
            st.error(f"Timeout error: Request to {self.base_url}/evaluators timed out")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading evaluator: {e}")
            return None

    @st.cache_data(ttl=3600) # Cache the result for 1 hour
    def save_custom_prompt(self, prompt_name: str, prompt_content: str) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/prompts",
                json={"name": prompt_name, "content": prompt_content},
                headers=self.headers,
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"Error saving custom prompt: {e}")
            return False

    @st.cache_data(ttl=3600) # Cache the result for 1 hour
    def get_custom_prompts(_self) -> Optional[Dict[str, str]]:
        try:
            response = requests.get(f"{_self.base_url}/prompts", headers=_self.headers)
            response.raise_for_status()
            return response.json().get("prompts")
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting custom prompts: {e}")
            return None

    @st.cache_data(ttl=3600) # Cache the result for 1 hour
    def get_custom_evaluators(_self) -> Optional[Dict[str, str]]:
        try:
            response = requests.get(f"{_self.base_url}/evaluators", headers=_self.headers)
            response.raise_for_status()
            return response.json().get("evaluators")
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting custom evaluators: {e}")
            return None

    def delete_evaluator(self, evaluator_id: str) -> bool:
        try:
            url = f"{self.base_url}/evaluators/{evaluator_id}"
            response = requests.delete(
                url, headers=self.headers, timeout=30
            )
            response.raise_for_status()
            return True
        except requests.exceptions.ConnectionError:
            st.error(f"Connection error: Could not connect to OpenEvolve backend at {self.base_url}/evaluators/{evaluator_id}")
            return False
        except requests.exceptions.Timeout:
            st.error(f"Timeout error: Request to {self.base_url}/evaluators/{evaluator_id} timed out")
            return False
        except requests.exceptions.RequestException as e:
            st.error(f"Error deleting evaluator: {e}")
            return False

    def save_checkpoint(self, evolution_id: str) -> bool:
        st.info(f"Checkpointing is handled automatically by OpenEvolve. Evolution '{evolution_id}' state is saved periodically.")
        return True

    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Loads an evolution from a specified checkpoint."""
        checkpoint_dir = os.path.join(os.getcwd(), "openevolve_checkpoints", checkpoint_name)
        if not os.path.exists(checkpoint_dir):
            st.error(f"Checkpoint '{checkpoint_name}' not found.")
            return None
        
        # When loading a checkpoint, we need to re-run the evolution from that checkpoint.
        # This is a simplified approach. A more robust solution would involve
        # re-initializing the entire evolution state from the checkpoint.
        # For now, we'll just return the path to the checkpoint.
        st.info(f"Loading evolution from checkpoint: {checkpoint_name}. This will restart the evolution from the saved state.")
        return {"checkpoint_path": checkpoint_dir}


def create_advanced_openevolve_config(
    model_name: str,
    api_key: str,
    api_base: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_iterations: int = 100,
    population_size: int = 1000,
    num_islands: int = 5,
    migration_interval: int = 50,
    migration_rate: float = 0.1,
    archive_size: int = 100,
    elite_ratio: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.7,
    checkpoint_interval: int = 100,
    language: str = None,
    file_suffix: str = ".py",
    reasoning_effort: str = None,
    feature_dimensions: Optional[List[str]] = None,
    feature_bins: Optional[int] = None,
    evaluator_id: Optional[str] = None,
    diversity_metric: str = "edit_distance",
    # Additional parameters for advanced features
    enable_artifacts: bool = True,
    cascade_evaluation: bool = True,
    cascade_thresholds: Optional[List[float]] = None,
    use_llm_feedback: bool = False,
    llm_feedback_weight: float = 0.1,
    parallel_evaluations: int = 1,
    distributed: bool = False,
    template_dir: Optional[str] = None,
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
    # Additional advanced parameters from OpenEvolve
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
    # Ensemble model configuration
    evaluator_models: Optional[List[Dict[str, any]]] = None,
    output_dir: Optional[str] = None,
) -> Optional[Config]:
    """
    Create an advanced OpenEvolve configuration with enhanced settings.

    Args:
        model_name: Name of the LLM model to use
        api_key: API key for the LLM provider
        api_base: Base URL for the API (optional)
        temperature: Temperature for generation (0.0-2.0)
        top_p: Top-p sampling parameter (0.0-1.0)
        max_tokens: Maximum tokens to generate
        max_iterations: Maximum number of evolution iterations
        population_size: Size of the population
        num_islands: Number of islands for island-based evolution
        migration_interval: Interval for migration between islands
        migration_rate: Rate of migration between islands
        archive_size: Size of the archive for storing best solutions
        elite_ratio: Ratio of elite individuals to preserve
        exploration_ratio: Ratio for exploration in evolution
        exploitation_ratio: Ratio for exploitation in evolution
        checkpoint_interval: Interval for saving checkpoints
        language: Programming language (optional)
        file_suffix: File suffix for the language
        reasoning_effort: Reasoning effort level (optional)
        feature_dimensions: List of feature dimensions for MAP-Elites
        feature_bins: Number of bins for feature dimensions
        evaluator_id: Optional evaluator ID
        diversity_metric: Metric for measuring diversity
        enable_artifacts: Whether to enable artifact side-channel
        cascade_evaluation: Whether to use cascade evaluation
        cascade_thresholds: Thresholds for cascade evaluation
        use_llm_feedback: Whether to use LLM-based feedback
        llm_feedback_weight: Weight for LLM feedback
        parallel_evaluations: Number of parallel evaluations
        distributed: Whether to use distributed evaluation
        template_dir: Directory for prompt templates
        system_message: System message for LLM
        evaluator_system_message: System message for evaluator
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

    Returns:
        Config object or None if OpenEvolve is not available
    """
    if not OPENEVOLVE_AVAILABLE:
        return None

    try:
        # Create configuration
        config = Config()

        # Set general settings
        config.max_iterations = max_iterations
        config.checkpoint_interval = checkpoint_interval
        config.language = language
        config.file_suffix = file_suffix
        config.random_seed = random_seed
        config.early_stopping_patience = early_stopping_patience
        config.convergence_threshold = convergence_threshold
        config.early_stopping_metric = early_stopping_metric
        config.diff_based_evolution = diff_based_evolution
        config.max_code_length = max_code_length
        config.log_level = log_level
        config.log_dir = log_dir

        # Configure LLM model
        llm_config = LLMModelConfig(
            name=model_name,
            api_key=api_key,
            api_base=api_base if api_base else "https://api.openai.com/v1",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=api_timeout,
            retries=api_retries,
            retry_delay=api_retry_delay,
            reasoning_effort=reasoning_effort,
            random_seed=random_seed,
        )

        # Add the model to the config
        config.llm.models = [llm_config]
        config.llm.system_message = system_message or config.llm.system_message

        # Configure evaluator models if provided
        if evaluator_models:
            evaluator_llm_configs = []
            for eval_model in evaluator_models:
                evaluator_config = LLMModelConfig(
                    name=eval_model['name'],
                    api_key=eval_model.get('api_key', api_key),
                    api_base=eval_model.get('api_base', api_base if api_base else "https://api.openai.com/v1"),
                    temperature=eval_model.get('temperature', 0.3),  # Lower temp for more consistent evaluation
                    top_p=eval_model.get('top_p', 0.9),
                    max_tokens=eval_model.get('max_tokens', 1024),
                    timeout=eval_model.get('timeout', api_timeout),
                    retries=eval_model.get('retries', api_retries),
                    retry_delay=eval_model.get('retry_delay', api_retry_delay),
                    reasoning_effort=eval_model.get('reasoning_effort', reasoning_effort),
                    random_seed=random_seed,
                )
                evaluator_llm_configs.append(evaluator_config)
            config.llm.evaluator_models = evaluator_llm_configs
        else:
            # Use the same models as evaluation models if none specified
            config.llm.evaluator_models = [llm_config]

        # Configure prompt settings
        config.prompt = PromptConfig(
            template_dir=template_dir,
            system_message=system_message or config.prompt.system_message,
            evaluator_system_message=evaluator_system_message or config.prompt.evaluator_system_message,
            num_top_programs=num_top_programs,
            num_diverse_programs=num_diverse_programs,
            use_template_stochasticity=use_template_stochasticity,
            template_variations=template_variations or {},
            use_meta_prompting=use_meta_prompting,
            meta_prompt_weight=meta_prompt_weight,
            include_artifacts=include_artifacts,
            max_artifact_bytes=max_artifact_bytes,
            artifact_security_filter=artifact_security_filter,
            suggest_simplification_after_chars=500,
            include_changes_under_chars=100,
            concise_implementation_max_lines=10,
            comprehensive_implementation_min_lines=50,
        )

        # Configure database settings for enhanced evolution
        config.database = DatabaseConfig(
            db_path=output_dir,
            in_memory=in_memory,
            population_size=population_size,
            archive_size=archive_size,
            num_islands=num_islands,
            elite_selection_ratio=elite_ratio,
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            diversity_metric=diversity_metric,
            feature_dimensions=feature_dimensions
            if feature_dimensions is not None
            else ["complexity", "diversity"],
            feature_bins=feature_bins if feature_bins is not None else 10,
            migration_interval=migration_interval,
            migration_rate=migration_rate,
            random_seed=random_seed,
            log_prompts=True,
            diversity_reference_size=diversity_reference_size,
            artifacts_base_path=os.path.join(output_dir, "artifacts") if output_dir else None,
            artifact_size_threshold=artifact_size_threshold,
            cleanup_old_artifacts=cleanup_old_artifacts,
            artifact_retention_days=artifact_retention_days,
        )

        # Configure evaluator settings
        config.evaluator = EvaluatorConfig(
            timeout=evaluator_timeout,
            max_retries=max_retries_eval,
            memory_limit_mb=memory_limit_mb,
            cpu_limit=cpu_limit,
            cascade_evaluation=cascade_evaluation,
            cascade_thresholds=cascade_thresholds or [0.5, 0.75, 0.9],
            parallel_evaluations=parallel_evaluations,
            distributed=distributed,
            use_llm_feedback=use_llm_feedback,
            llm_feedback_weight=llm_feedback_weight,
            enable_artifacts=enable_artifacts,
            max_artifact_storage=100 * 1024 * 1024,  # 100MB per program
        )

        # Configure evolution trace settings
        config.evolution_trace = EvolutionTraceConfig(
            enabled=evolution_trace_enabled,
            format=evolution_trace_format,
            include_code=evolution_trace_include_code,
            include_prompts=evolution_trace_include_prompts,
            output_path=evolution_trace_output_path,
            buffer_size=evolution_trace_buffer_size,
            compress=evolution_trace_compress,
        )

        return config

    except Exception as e:
        st.error(f"Error creating OpenEvolve configuration: {e}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None


def create_language_specific_evaluator(
    content_type: str,
    custom_requirements: str = "",
    compliance_rules: Optional[List[str]] = None,
) -> Callable:
    """
    Create a language-specific evaluator for code content.

    Args:
        content_type: Type of content (e.g., 'code_python', 'code_js', etc.)
        custom_requirements: Custom requirements to check for

    Returns:
        Callable evaluator function
    """

    def code_evaluator(program_path: str) -> Dict[str, Any]:
        """Evaluator for code content."""
        try:
            with open(program_path, "r") as f:
                content = f.read()

            metrics = {
                "timestamp": time.time(),
                "length": len(content),
                "compliance_score": 1.0,  # Start with full compliance
                "compliance_violations": [],
                # OpenEvolve-specific metrics that can be used by the evolution process
                "complexity": len(content.split()) / 100.0,  # Simple complexity measure
                "diversity": len(set(content.split())) / max(1, len(content.split())),  # Vocabulary diversity
            }

            # Perform compliance checks if rules are provided
            if compliance_rules and len(compliance_rules) > 0:
                for rule in compliance_rules:
                    if (
                        rule.lower() not in content.lower()
                    ):  # Simple keyword check for now
                        metrics["compliance_score"] -= 1.0 / len(compliance_rules)
                        metrics["compliance_violations"].append(
                            f"Missing compliance rule: {rule}"
                        )

            # Add language-specific metrics
            if content_type == "code_python":
                # Python-specific checks
                metrics.update(
                    {
                        "has_imports": "import" in content.lower(),
                        "has_functions": "def " in content,
                        "has_classes": "class " in content,
                        "has_main_guard": "if __name__ ==" in content,
                        "comment_ratio": content.count("#")
                        / max(1, len(content.split())),
                    }
                )
            elif content_type == "code_js":
                # JavaScript-specific checks
                metrics.update(
                    {
                        "has_imports": "import" in content or "require" in content,
                        "has_functions": "function" in content or "=>" in content,
                        "has_classes": "class" in content,
                        "has_exports": "export" in content,
                        "comment_ratio": (content.count("//") + content.count("/*"))
                        / max(1, len(content.split())),
                    }
                )
            elif content_type == "code_java":
                # Java-specific checks
                metrics.update(
                    {
                        "has_imports": "import " in content,
                        "has_classes": "class " in content,
                        "has_package": "package " in content,
                        "has_public_class": "public class" in content,
                        "comment_ratio": (content.count("//") + content.count("/*"))
                        / max(1, len(content.split())),
                    }
                )
            elif content_type == "code_cpp":
                # C++-specific checks
                metrics.update(
                    {
                        "has_includes": "#include" in content,
                        "has_namespaces": "namespace" in content,
                        "has_classes": "class " in content or "struct " in content,
                        "has_templates": "template" in content,
                        "comment_ratio": (content.count("//") + content.count("/*"))
                        / max(1, len(content.split())),
                    }
                )
            elif content_type == "code_csharp":
                # C#-specific checks
                metrics.update(
                    {
                        "has_using": "using " in content,
                        "has_namespaces": "namespace" in content,
                        "has_classes": "class " in content,
                        "has_public_class": "public class" in content,
                        "comment_ratio": (content.count("//") + content.count("/*"))
                        / max(1, len(content.split())),
                    }
                )
            elif content_type == "code_go":
                # Go-specific checks
                metrics.update(
                    {
                        "has_package": "package " in content,
                        "has_imports": "import " in content,
                        "has_functions": "func " in content,
                        "has_structs": "struct " in content,
                        "comment_ratio": content.count("//")
                        / max(1, len(content.split())),
                    }
                )
            elif content_type == "code_rust":
                # Rust-specific checks
                metrics.update(
                    {
                        "has_mod": "mod " in content,
                        "has_use": "use " in content,
                        "has_functions": "fn " in content,
                        "has_structs": "struct " in content,
                        "comment_ratio": content.count("//")
                        / max(1, len(content.split())),
                    }
                )
            elif content_type == "code_swift":
                # Swift-specific checks
                metrics.update(
                    {
                        "has_imports": "import " in content,
                        "has_functions": "func " in content,
                        "has_classes": "class " in content or "struct " in content,
                        "has_protocols": "protocol " in content,
                        "comment_ratio": (content.count("//") + content.count("/*"))
                        / max(1, len(content.split())),
                    }
                )
            elif content_type == "code_kotlin":
                # Kotlin-specific checks
                metrics.update(
                    {
                        "has_imports": "import " in content,
                        "has_functions": "fun " in content,
                        "has_classes": "class " in content,
                        "has_package": "package " in content,
                        "comment_ratio": (content.count("//") + content.count("/*"))
                        / max(1, len(content.split())),
                    }
                )
            elif content_type == "code_typescript":
                # TypeScript-specific checks
                metrics.update(
                    {
                        "has_imports": "import " in content,
                        "has_functions": "function " in content or "=>" in content,
                        "has_classes": "class " in content,
                        "has_interfaces": "interface " in content,
                        "comment_ratio": (content.count("//") + content.count("/*"))
                        / max(1, len(content.split())),
                    }
                )
            elif content_type == "document_legal":
                # Legal-specific checks
                metrics.update(
                    {
                        "has_contract_terms": any(
                            term in content.lower()
                            for term in [
                                "contract",
                                "agreement",
                                "clause",
                                "party",
                                "jurisdiction",
                            ]
                        ),
                        "has_legal_citations": any(
                            citation in content
                            for citation in ["§", "et al.", "v.", "supra"]
                        ),
                        "document_length_words": len(content.split()),
                    }
                )
            elif content_type == "document_medical":
                # Medical-specific checks
                metrics.update(
                    {
                        "has_medical_terms": any(
                            term in content.lower()
                            for term in [
                                "patient",
                                "diagnosis",
                                "treatment",
                                "symptom",
                                "medication",
                                "hospital",
                                "clinic",
                            ]
                        ),
                        "has_patient_id_format": "PID-"
                        in content,  # Simple check for a common pattern
                        "document_length_words": len(content.split()),
                    }
                )
            elif content_type == "document_technical":
                # Technical-specific checks
                metrics.update(
                    {
                        "has_code_snippets": any(
                            keyword in content
                            for keyword in [
                                "def ",
                                "class ",
                                "import ",
                                "function ",
                                "{",
                                "}",
                                ";",
                            ]
                        ),
                        "has_technical_terms": any(
                            term in content.lower()
                            for term in [
                                "api",
                                "database",
                                "server",
                                "client",
                                "algorithm",
                                "framework",
                                "library",
                            ]
                        ),
                        "document_length_words": len(content.split()),
                    }
                )
            elif content_type == "code_php":
                metrics.update({
                    "has_functions": "function " in content,
                    "has_classes": "class " in content,
                    "has_variables": "$" in content,
                    "comment_ratio": content.count("//") + content.count("/*") / max(1, len(content.split())),
                })
            elif content_type == "code_ruby":
                metrics.update({
                    "has_methods": "def " in content,
                    "has_classes": "class " in content,
                    "has_modules": "module " in content,
                    "comment_ratio": content.count("#") / max(1, len(content.split())),
                })
            elif content_type == "code_perl":
                metrics.update({
                    "has_subroutines": "sub " in content,
                    "has_packages": "package " in content,
                    "has_variables": "$" in content or "@" in content or "%" in content,
                    "comment_ratio": content.count("#") / max(1, len(content.split())),
                })
            elif content_type == "code_scala":
                metrics.update({
                    "has_functions": "def " in content,
                    "has_classes": "class " in content,
                    "has_objects": "object " in content,
                    "comment_ratio": content.count("//") + content.count("/*") / max(1, len(content.split())),
                })
            elif content_type == "code_haskell":
                metrics.update({
                    "has_functions": "::" in content,
                    "has_data_types": "data " in content,
                    "has_modules": "module " in content,
                    "comment_ratio": content.count("--") / max(1, len(content.split())),
                })
            elif content_type == "code_r":
                metrics.update({
                    "has_functions": "function(" in content,
                    "has_variables": "<-" in content or "=" in content,
                    "comment_ratio": content.count("#") / max(1, len(content.split())),
                })
            elif content_type == "code_matlab":
                metrics.update({
                    "has_functions": "function " in content,
                    "has_scripts": ".m" in content,
                    "comment_ratio": content.count("%") / max(1, len(content.split())),
                })
            elif content_type == "code_assembly":
                metrics.update({
                    "has_labels": ":" in content,
                    "has_instructions": "mov " in content or "add " in content,
                    "comment_ratio": content.count(";") / max(1, len(content.split())),
                })
            elif content_type == "code_shell":
                metrics.update({
                    "has_shebang": "#!" in content,
                    "has_variables": "$" in content,
                    "has_functions": "() {" in content,
                    "comment_ratio": content.count("#") / max(1, len(content.split())),
                })
            elif content_type == "code_sql":
                metrics.update({
                    "has_select": "SELECT" in content.upper(),
                    "has_from": "FROM" in content.upper(),
                    "has_where": "WHERE" in content.upper(),
                    "comment_ratio": content.count("--") + content.count("/*") / max(1, len(content.split())),
                })
            elif content_type == "code_html":
                metrics.update({
                    "has_doctype": "<!DOCTYPE html>" in content.lower(),
                    "has_head": "<head>" in content.lower(),
                    "has_body": "<body>" in content.lower(),
                    "comment_ratio": content.count("<!--") / max(1, len(content.split())),
                })
            elif content_type == "code_css":
                metrics.update({
                    "has_selectors": "{" in content and "}" in content,
                    "has_properties": ":" in content and ";" in content,
                    "comment_ratio": content.count("/*") / max(1, len(content.split())),
                })
            elif content_type == "code_json":
                metrics.update({
                    "is_valid_json": True, # Basic check, full validation is complex
                    "has_objects": "{" in content,
                    "has_arrays": "[" in content,
                })
            elif content_type == "code_xml":
                metrics.update({
                    "has_xml_declaration": "<?xml" in content.lower(),
                    "has_tags": "<" in content and ">" in content,
                })
            elif content_type == "code_yaml":
                metrics.update({
                    "has_key_value_pairs": ":" in content,
                    "has_lists": "-" in content,
                })
            elif content_type == "code_markdown":
                metrics.update({
                    "has_headers": "#" in content,
                    "has_lists": "-" in content or "*" in content or "1." in content,
                    "has_links": "[" in content and "]" in content and "(" in content and ")" in content,
                })
            elif content_type == "document_general":
                metrics.update({
                    "document_length_words": len(content.split()),
                    "has_sections": "##" in content or "###" in content,
                })
            elif content_type == "document_report":
                metrics.update({
                    "document_length_words": len(content.split()),
                    "has_title": "#" in content,
                    "has_summary": "summary" in content.lower(),
                })
            elif content_type == "code_lean4":
                metrics.update({
                    "has_imports": "import " in content,
                    "has_theorems": "theorem " in content,
                    "has_definitions": "def " in content or "abbrev " in content,
                    "comment_ratio": content.count("--") / max(1, len(content.split())),
                })

            # Calculate a composite score based on various factors
            score_components = []

            # Length score (favor moderate length)
            length_score = min(1.0, len(content) / 1000.0)
            score_components.append(length_score * 0.2)

            # Structure score (favor well-structured code)
            structure_score = 0.0
            if metrics.get("has_functions") or metrics.get("has_classes"):
                structure_score += 0.3
            if metrics.get("has_imports"):
                structure_score += 0.2
            score_components.append(structure_score * 0.3)

            # Comment score (favor reasonable commenting)
            comment_ratio = metrics.get("comment_ratio", 0.0)
            if 0.01 <= comment_ratio <= 0.3:  # Reasonable comment ratio
                comment_score = 0.5
            elif comment_ratio > 0.3:  # Too many comments
                comment_score = 0.3
            else:  # Too few comments
                comment_score = comment_ratio * 10
            score_components.append(comment_score * 0.3)

            # Custom requirements score
            if custom_requirements:
                # Simple keyword matching for now
                req_matches = sum(
                    1
                    for req in custom_requirements.split(",")
                    if req.strip().lower() in content.lower()
                )
                req_score = min(
                    1.0, req_matches / max(1, len(custom_requirements.split(",")))
                )
                score_components.append(req_score * 0.2)

            # Domain relevance score (for document types)
            domain_relevance_score = 0.0
            if content_type.startswith("document_"):
                if content_type == "document_legal" and metrics.get(
                    "has_contract_terms"
                ):
                    domain_relevance_score += 0.5
                if content_type == "document_medical" and metrics.get(
                    "has_medical_terms"
                ):
                    domain_relevance_score += 0.5
                if content_type == "document_technical" and metrics.get(
                    "has_technical_terms"
                ):
                    domain_relevance_score += 0.5
                # Add more sophisticated checks here if needed
                score_components.append(
                    domain_relevance_score * 0.3
                )  # Give it a moderate weight

            # Calculate final score as average of components
            final_score = sum(score_components) if score_components else 0.5

            metrics["combined_score"] = final_score
            metrics["score_components"] = {
                "length": length_score,
                "structure": structure_score,
                "comments": comment_score,
                "requirements": score_components[-1] if custom_requirements else 0.0,
                "domain_relevance": domain_relevance_score,
                "compliance": metrics.get("compliance_score", 0.0),
            }

            return metrics

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

    return code_evaluator


def run_advanced_code_evolution(
    content: str,
    content_type: str,
    model_name: str,
    api_key: str,
    api_base: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_iterations: int = 100,
    population_size: int = 1000,
    num_islands: int = 5,
    migration_interval: int = 50,
    migration_rate: float = 0.1,
    archive_size: int = 100,
    elite_ratio: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.7,
    checkpoint_interval: int = 100,
    custom_requirements: str = "",
    custom_evaluator: Optional[Callable] = None,
    compliance_rules: Optional[List[str]] = None,
    # Additional advanced parameters
    enable_artifacts: bool = True,
    cascade_evaluation: bool = True,
    cascade_thresholds: Optional[List[float]] = None,
    use_llm_feedback: bool = False,
    llm_feedback_weight: float = 0.1,
    parallel_evaluations: int = 4,
    distributed: bool = False,
    template_dir: Optional[str] = None,
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
    feature_dimensions: Optional[List[str]] = None,
    feature_bins: Optional[int] = None,
    diversity_metric: str = "edit_distance",
    # Additional parameters from create_advanced_openevolve_config
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
    # Ensemble model configuration
    evaluator_models: Optional[List[Dict[str, any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run advanced code evolution using the OpenEvolve backend with enhanced settings.

    Args:
        content: The code content to evolve
        content_type: Type of content (e.g., 'code_python', 'code_js', etc.)
        model_name: Name of the LLM model to use
        api_key: API key for the LLM provider
        api_base: Base URL for the API (optional)
        temperature: Temperature for generation (0.0-2.0)
        top_p: Top-p sampling parameter (0.0-1.0)
        max_tokens: Maximum tokens to generate
        max_iterations: Maximum number of evolution iterations
        population_size: Size of the population
        num_islands: Number of islands for island-based evolution
        migration_interval: Interval for migration between islands
        migration_rate: Rate of migration between islands
        archive_size: Size of the archive for storing best solutions
        elite_ratio: Ratio of elite individuals to preserve
        exploration_ratio: Ratio for exploration in evolution
        exploitation_ratio: Ratio for exploitation in evolution
        checkpoint_interval: Interval for saving checkpoints
        custom_requirements: Custom requirements to check for
        custom_evaluator: Custom evaluator function (optional)
        compliance_rules: Compliance rules to check against
        enable_artifacts: Whether to enable artifact side-channel
        cascade_evaluation: Whether to use cascade evaluation
        cascade_thresholds: Thresholds for cascade evaluation
        use_llm_feedback: Whether to use LLM-based feedback
        llm_feedback_weight: Weight for LLM feedback
        parallel_evaluations: Number of parallel evaluations
        distributed: Whether to use distributed evaluation
        template_dir: Directory for prompt templates
        system_message: System message for LLM
        evaluator_system_message: System message for evaluator
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
        feature_dimensions: List of feature dimensions for MAP-Elites
        feature_bins: Number of bins for feature dimensions
        diversity_metric: Metric for measuring diversity
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

    Returns:
        Dictionary with evolution results or None if failed
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available.")
        return None

    try:
        # Create comprehensive configuration with ALL OpenEvolve features
        model_configs = [{"name": model_name, "weight": 1.0, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}]
        
        config = create_comprehensive_openevolve_config(
            content_type=content_type,
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
            population_size=population_size,
            num_islands=num_islands,
            migration_interval=migration_interval,
            migration_rate=migration_rate,
            archive_size=archive_size,
            elite_ratio=elite_ratio,
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            checkpoint_interval=checkpoint_interval,
            feature_dimensions=feature_dimensions or ["complexity", "diversity"],
            feature_bins=feature_bins or 10,
            diversity_metric=diversity_metric,
            system_message=system_message,
            evaluator_system_message=evaluator_system_message,
            # Advanced parameters
            enable_artifacts=enable_artifacts,
            cascade_evaluation=cascade_evaluation,
            cascade_thresholds=cascade_thresholds or [0.5, 0.75, 0.9],
            use_llm_feedback=use_llm_feedback,
            llm_feedback_weight=llm_feedback_weight,
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
            early_stopping_patience=early_stopping_patience,
            convergence_threshold=convergence_threshold,
            early_stopping_metric=early_stopping_metric,
            memory_limit_mb=memory_limit_mb,
            cpu_limit=cpu_limit,
            random_seed=random_seed,
            db_path=db_path,
            in_memory=in_memory,
            # Additional advanced parameters
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
        )

        if not config:
            return None

        # Create evaluator based on content type and custom requirements
        if custom_evaluator:
            evaluator = custom_evaluator
        else:
            evaluator = create_specialized_evaluator(content_type, custom_requirements, compliance_rules)

        # Create a temporary file with the content to evolve
        with tempfile.NamedTemporaryFile(mode="w", suffix=get_file_suffix_from_content_type(content_type), delete=False) as temp_file:
            content_with_markers = f"""# EVOLVE-BLOCK-START
{content}
# EVOLVE-BLOCK-END"""
            temp_file.write(content_with_markers)
            temp_file_path = temp_file.name

        try:
            # Run evolution using OpenEvolve API
            from openevolve.api import run_evolution
            result = run_evolution(
                initial_program=temp_file_path,
                evaluator=evaluator,
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
                    start_idx = best_code.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                    end_idx = best_code.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_code = best_code[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": asdict(result.best_program) if result.best_program else None,
                    "best_score": result.best_score,
                    "best_code": best_code,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                    "trace_enabled": evolution_trace_enabled,
                }
            else:
                return {
                    "success": False,
                    "message": "Evolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running advanced code evolution: {e}")
        import traceback

        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

def run_ensemble_evolution(
    content: str,
    content_type: str,
    primary_models: List[Dict[str, any]],
    api_key: str,
    api_base: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_iterations: int = 100,
    population_size: int = 1000,
    num_islands: int = 5,
    migration_interval: int = 50,
    migration_rate: float = 0.1,
    archive_size: int = 100,
    elite_ratio: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.7,
    checkpoint_interval: int = 100,
    custom_requirements: str = "",
    custom_evaluator: Optional[Callable] = None,
    compliance_rules: Optional[List[str]] = None,
    # Additional advanced parameters
    enable_artifacts: bool = True,
    cascade_evaluation: bool = True,
    cascade_thresholds: Optional[List[float]] = None,
    use_llm_feedback: bool = False,
    llm_feedback_weight: float = 0.1,
    parallel_evaluations: int = 4,
    distributed: bool = False,
    template_dir: Optional[str] = None,
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
    feature_dimensions: Optional[List[str]] = None,
    feature_bins: Optional[int] = None,
    diversity_metric: str = "edit_distance",
    # Additional parameters from create_advanced_openevolve_config
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
) -> Optional[Dict[str, Any]]:
    """
    Run advanced code evolution using an ensemble of models with enhanced settings.

    Args:
        content: The code content to evolve
        content_type: Type of content (e.g., 'code_python', 'code_js', etc.)
        primary_models: List of primary model configurations with weights
        api_key: API key for the LLM provider
        api_base: Base URL for the API (optional)
        temperature: Temperature for generation (0.0-2.0)
        top_p: Top-p sampling parameter (0.0-1.0)
        max_tokens: Maximum tokens to generate
        max_iterations: Maximum number of evolution iterations
        population_size: Size of the population
        num_islands: Number of islands for island-based evolution
        migration_interval: Interval for migration between islands
        migration_rate: Rate of migration between islands
        archive_size: Size of the archive for storing best solutions
        elite_ratio: Ratio of elite individuals to preserve
        exploration_ratio: Ratio for exploration in evolution
        exploitation_ratio: Ratio for exploitation in evolution
        checkpoint_interval: Interval for saving checkpoints
        custom_requirements: Custom requirements to check for
        custom_evaluator: Custom evaluator function (optional)
        compliance_rules: Compliance rules to check against
        enable_artifacts: Whether to enable artifact side-channel
        cascade_evaluation: Whether to use cascade evaluation
        cascade_thresholds: Thresholds for cascade evaluation
        use_llm_feedback: Whether to use LLM-based feedback
        llm_feedback_weight: Weight for LLM feedback
        parallel_evaluations: Number of parallel evaluations
        distributed: Whether to use distributed evaluation
        template_dir: Directory for prompt templates
        system_message: System message for LLM
        evaluator_system_message: System message for evaluator
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
        feature_dimensions: List of feature dimensions for MAP-Elites
        feature_bins: Number of bins for feature dimensions
        diversity_metric: Metric for measuring diversity
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

    Returns:
        Dictionary with evolution results or None if failed
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available.")
        return None

    try:
        # Create configuration from the primary models list
        from openevolve.config import Config, LLMModelConfig
        
        config = Config()
        
        # Configure LLM models from the provided list
        llm_configs = []
        for model_info in primary_models:
            llm_config = LLMModelConfig(
                name=model_info['name'],
                api_key=api_key,
                api_base=api_base if api_base else "https://api.openai.com/v1",
                temperature=model_info.get('temperature', temperature),
                top_p=model_info.get('top_p', top_p),
                max_tokens=model_info.get('max_tokens', max_tokens),
                weight=model_info.get('weight', 1.0),
                timeout=model_info.get('timeout', api_timeout),
                retries=model_info.get('retries', api_retries),
                retry_delay=model_info.get('retry_delay', api_retry_delay),
                reasoning_effort=model_info.get('reasoning_effort', None),
                random_seed=random_seed,
            )
            llm_configs.append(llm_config)

        # Add all models to the config
        config.llm.models = llm_configs
        config.llm.system_message = system_message or config.llm.system_message

        # Configure evaluator models with the same models for consistency
        config.llm.evaluator_models = llm_configs

        # Set general settings
        config.max_iterations = max_iterations
        config.checkpoint_interval = checkpoint_interval
        config.language = get_language_from_content_type(content_type)
        config.file_suffix = get_file_suffix_from_content_type(content_type)
        config.random_seed = random_seed
        config.early_stopping_patience = early_stopping_patience
        config.convergence_threshold = convergence_threshold
        config.early_stopping_metric = early_stopping_metric
        config.diff_based_evolution = diff_based_evolution
        config.max_code_length = max_code_length
        config.log_level = log_level
        config.log_dir = log_dir

        # Configure prompt settings
        config.prompt = PromptConfig(
            template_dir=template_dir,
            system_message=system_message or config.prompt.system_message,
            evaluator_system_message=evaluator_system_message or config.prompt.evaluator_system_message,
            num_top_programs=num_top_programs,
            num_diverse_programs=num_diverse_programs,
            use_template_stochasticity=use_template_stochasticity,
            template_variations=template_variations or {},
            use_meta_prompting=use_meta_prompting,
            meta_prompt_weight=meta_prompt_weight,
            include_artifacts=include_artifacts,
            max_artifact_bytes=max_artifact_bytes,
            artifact_security_filter=artifact_security_filter,
            suggest_simplification_after_chars=500,
            include_changes_under_chars=100,
            concise_implementation_max_lines=10,
            comprehensive_implementation_min_lines=50,
        )

        # Configure database settings for enhanced evolution
        config.database = DatabaseConfig(
            db_path=db_path,
            in_memory=in_memory,
            population_size=population_size,
            archive_size=archive_size,
            num_islands=num_islands,
            elite_selection_ratio=elite_ratio,
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            diversity_metric=diversity_metric,
            feature_dimensions=feature_dimensions or ["complexity", "diversity"],
            feature_bins=feature_bins or 10,
            migration_interval=migration_interval,
            migration_rate=migration_rate,
            random_seed=random_seed,
            log_prompts=True,
            diversity_reference_size=diversity_reference_size,
            artifacts_base_path=os.path.join(db_path, "artifacts") if db_path else None,
            artifact_size_threshold=artifact_size_threshold,
            cleanup_old_artifacts=cleanup_old_artifacts,
            artifact_retention_days=artifact_retention_days,
        )

        # Configure evaluator settings
        config.evaluator = EvaluatorConfig(
            timeout=evaluator_timeout,
            max_retries=max_retries_eval,
            memory_limit_mb=memory_limit_mb,
            cpu_limit=cpu_limit,
            cascade_evaluation=cascade_evaluation,
            cascade_thresholds=cascade_thresholds or [0.5, 0.75, 0.9],
            parallel_evaluations=parallel_evaluations,
            distributed=distributed,
            use_llm_feedback=use_llm_feedback,
            llm_feedback_weight=llm_feedback_weight,
            enable_artifacts=enable_artifacts,
            max_artifact_storage=100 * 1024 * 1024,  # 100MB per program
        )

        # Configure evolution trace settings
        config.evolution_trace = EvolutionTraceConfig(
            enabled=evolution_trace_enabled,
            format=evolution_trace_format,
            include_code=evolution_trace_include_code,
            include_prompts=evolution_trace_include_prompts,
            output_path=evolution_trace_output_path,
            buffer_size=evolution_trace_buffer_size,
            compress=evolution_trace_compress,
        )

        if not config:
            return None

        # Create evaluator based on content type and custom requirements
        if custom_evaluator:
            evaluator = custom_evaluator
        else:
            evaluator = create_specialized_evaluator(content_type, custom_requirements, compliance_rules)

        # Create a temporary file with the content to evolve
        with tempfile.NamedTemporaryFile(mode="w", suffix=get_file_suffix_from_content_type(content_type), delete=False) as temp_file:
            content_with_markers = f"""# EVOLVE-BLOCK-START
{content}
# EVOLVE-BLOCK-END"""
            temp_file.write(content_with_markers)
            temp_file_path = temp_file.name

        try:
            # Run evolution using OpenEvolve API
            from openevolve.api import run_evolution
            result = run_evolution(
                initial_program=temp_file_path,
                evaluator=evaluator,
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
                    start_idx = best_code.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                    end_idx = best_code.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_code = best_code[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": asdict(result.best_program) if result.best_program else None,
                    "best_score": result.best_score,
                    "best_code": best_code,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                    "trace_enabled": evolution_trace_enabled,
                    "models_used": [m.name for m in config.llm.models],
                }
            else:
                return {
                    "success": False,
                    "message": "Evolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running ensemble code evolution: {e}")
        import traceback

        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def create_specialized_evaluator(
    content_type: str,
    custom_requirements: str = "",
    compliance_rules: Optional[List[str]] = None,
) -> Callable:
    """
    Create a specialized evaluator for code content using linters with artifact support.

    Args:
        content_type: Type of content (e.g., 'code_python', 'code_js', etc.)
        custom_requirements: Custom requirements to check for
        compliance_rules: Compliance rules to check against

    Returns:
        Callable evaluator function
    """

    def code_evaluator(program_path: str) -> Dict[str, Any]:
        """Evaluator for code content using linters with artifact support."""
        try:
            with open(program_path, "r") as f:
                content = f.read()

            metrics = {
                "timestamp": time.time(),
                "length": len(content),
                "linter_score": 0.0,
                "linter_errors": [],
                "compliance_score": 1.0,  # Start with full compliance
                "compliance_violations": [],
                # OpenEvolve-specific metrics that can be used by the evolution process
                "complexity": len(content.split()) / 100.0,  # Simple complexity measure
                "diversity": len(set(content.split())) / max(1, len(content.split())),  # Vocabulary diversity
                "performance": 0.0,  # Performance score placeholder
                "readability": 0.0,  # Readability score placeholder
            }

            # Perform compliance checks if rules are provided
            if compliance_rules and len(compliance_rules) > 0:
                for rule in compliance_rules:
                    if (
                        rule.lower() not in content.lower()
                    ):  # Simple keyword check for now
                        metrics["compliance_score"] -= 1.0 / len(compliance_rules)
                        metrics["compliance_violations"].append(
                            f"Missing compliance rule: {rule}"
                        )

            # Language-specific linter integration for better evaluation
            artifacts = {}  # Store execution artifacts for feedback
            
            # Enhanced artifact collection with more detailed feedback
            if content_type == "code_python":
                try:
                    import subprocess
                    import sys
                    import tempfile
                    
                    # Check Python syntax and collect artifacts
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                        temp_file.write(content)
                        temp_file_path = temp_file.name
                    
                    try:
                        # Run Python syntax check
                        result = subprocess.run([
                            sys.executable, "-m", "py_compile", temp_file_path
                        ], capture_output=True, timeout=10)
                        
                        if result.returncode != 0:
                            artifacts["python_syntax_error"] = result.stderr.decode("utf-8")
                        else:
                            # Run python code to capture runtime artifacts
                            exec_result = subprocess.run([
                                sys.executable, temp_file_path
                            ], capture_output=True, timeout=15)
                            
                            if exec_result.returncode != 0:
                                artifacts["python_runtime_error"] = exec_result.stderr.decode("utf-8")
                            elif exec_result.stderr:
                                artifacts["python_warnings"] = exec_result.stderr.decode("utf-8")
                            
                            # Run pylint if available for detailed code analysis
                            try:
                                from pylint.lint import Run
                                from pylint.reporters.text import TextReporter
                                import io

                                reporter = TextReporter(io.StringIO())
                                run = Run([temp_file_path], reporter=reporter, exit=False)
                                linter = run.linter
                                score = linter.stats.global_note if hasattr(linter.stats, 'global_note') else 0
                                errors = reporter.out.getvalue()
                                metrics["linter_score"] = score
                                metrics["linter_errors"] = errors
                            except ImportError:
                                # Fallback: simple static analysis
                                error_count = 0
                                if "import" not in content and ("print" in content or len(content.split()) > 50):
                                    error_count += 1  # Missing import statement for longer scripts
                                metrics["linter_score"] = max(0, 10 - error_count)
                        
                        # Performance and readability metrics
                        lines = content.splitlines()
                        metrics["performance"] = min(1.0, len(lines) / 100.0)  # Favor shorter code
                        metrics["readability"] = 1.0 - (content.count("  ") / len(content))  # Indentation consistency
                        
                        # Check for common Python code quality metrics
                        metrics.update({
                            "has_try_except": "try:" in content or "except:" in content,
                            "has_type_hints": "->" in content or ": " in content,
                            "has_docstrings": '"""' in content or "'''" in content,
                            "has_unittests": "unittest" in content.lower(),
                            "has_main_guard": "__name__ == \"__main__\"" in content,
                            "has_imports": "import " in content,
                            "has_functions": "def " in content,
                            "has_classes": "class " in content,
                        })
                    finally:
                        # Clean up temporary file
                        import os
                        os.unlink(temp_file_path)
                        
                except subprocess.TimeoutExpired:
                    artifacts["execution_timeout"] = "Python execution timed out"
                except Exception as e:
                    artifacts["execution_error"] = str(e)

            elif content_type == "code_js":
                try:
                    import subprocess
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_file:
                        temp_file.write(content)
                        temp_file_path = temp_file.name
                    
                    try:
                        # Check JavaScript syntax with Node.js
                        result = subprocess.run([
                            "node", "-c", temp_file_path
                        ], capture_output=True, timeout=10)
                        
                        if result.returncode != 0:
                            artifacts["js_syntax_error"] = result.stderr.decode("utf-8")
                        else:
                            # Execute JavaScript to capture runtime artifacts
                            exec_result = subprocess.run([
                                "node", temp_file_path
                            ], capture_output=True, timeout=15)
                            
                            if exec_result.returncode != 0:
                                artifacts["js_runtime_error"] = exec_result.stderr.decode("utf-8")
                            elif exec_result.stderr:
                                artifacts["js_warnings"] = exec_result.stderr.decode("utf-8")
                                
                            # Try to run eslint if available for detailed analysis
                            try:
                                result = subprocess.run(
                                    ["eslint", "--format", "json", temp_file_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=15
                                )
                                if result.returncode == 0:
                                    eslint_output = json_module.loads(result.stdout)
                                    if eslint_output and isinstance(eslint_output, list) and len(eslint_output) > 0:
                                        errors = eslint_output[0].get("messages", [])
                                        error_count = len(errors)
                                        score = max(0, 10 - error_count)  # a simple scoring metric
                                        metrics["linter_score"] = score
                                        metrics["linter_errors"] = errors
                                else:
                                    # If eslint command failed, try a basic check
                                    error_count = content.count("undefined") + content.count("Uncaught")
                                    metrics["linter_score"] = max(0, 10 - error_count)
                            except (ImportError, FileNotFoundError):
                                # Basic JavaScript checks
                                error_count = 0
                                if content.count("{") != content.count("}"):
                                    error_count += 1
                                if content.count("(") != content.count(")"):
                                    error_count += 1
                                metrics["linter_score"] = max(0, 10 - error_count)
                                
                        # Update JavaScript metrics
                        metrics.update({
                            "has_functions": "function" in content or "=>" in content or "function(" in content,
                            "has_classes": "class " in content,
                            "has_imports": "import" in content or "require" in content,
                            "has_async": "async" in content or "await" in content,
                            "has_arrow_functions": "=>" in content,
                        })
                    finally:
                        # Clean up temporary file
                        import os
                        os.unlink(temp_file_path)
                        
                except subprocess.TimeoutExpired:
                    artifacts["execution_timeout"] = "JavaScript execution timed out"
                except Exception as e:
                    artifacts["execution_error"] = str(e)
                    
            elif content_type == "code_java":
                try:
                    import subprocess
                    import tempfile
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Write content to a temporary Java file
                        class_name = "MainClass"  # Default class name - needs to match content
                        # Try to extract class name from content
                        import re
                        class_match = re.search(r'class\s+(\w+)', content)
                        if class_match:
                            class_name = class_match.group(1)
                        
                        java_file = os.path.join(temp_dir, f"{class_name}.java")
                        with open(java_file, 'w') as f:
                            f.write(content)
                        
                        # Try to compile Java code
                        result = subprocess.run([
                            "javac", java_file
                        ], capture_output=True, timeout=30)
                        
                        if result.returncode != 0:
                            artifacts["java_compile_error"] = result.stderr.decode("utf-8")
                        else:
                            # Try to run the compiled Java class
                            class_dir = os.path.dirname(java_file)
                            run_result = subprocess.run([
                                "java", "-cp", class_dir, class_name
                            ], capture_output=True, timeout=15)
                            
                            if run_result.returncode != 0:
                                artifacts["java_runtime_error"] = run_result.stderr.decode("utf-8")
                            elif run_result.stderr:
                                artifacts["java_warnings"] = run_result.stderr.decode("utf-8")
                                
                        # Add Java-specific metrics
                        metrics.update({
                            "has_imports": "import " in content,
                            "has_classes": "class " in content,
                            "has_main_method": "public static void main" in content,
                            "has_package": "package " in content,
                            "has_public_class": "public class" in content,
                        })
                except subprocess.TimeoutExpired:
                    artifacts["compilation_timeout"] = "Java compilation or execution timed out"
                except Exception as e:
                    artifacts["compilation_error"] = str(e)
                    
            elif content_type == "code_cpp":
                try:
                    import subprocess
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as temp_file:
                        temp_file.write(content)
                        temp_file_path = temp_file.name
                    
                    try:
                        # Try to compile C++ code with g++
                        result = subprocess.run([
                            "g++", "-std=c++11", "-c", temp_file_path, 
                            "-o", temp_file_path.replace('.cpp', '.o')
                        ], capture_output=True, timeout=30)
                        
                        if result.returncode != 0:
                            artifacts["cpp_compile_error"] = result.stderr.decode("utf-8")
                        else:
                            # Compilation successful
                            # Add C++ specific metrics
                            metrics.update({
                                "has_includes": "#include" in content,
                                "has_namespaces": "namespace" in content,
                                "has_classes": "class " in content or "struct " in content,
                                "has_templates": "template" in content,
                                "has_std": "std::" in content,
                            })
                    finally:
                        # Clean up temp files
                        import os
                        o_file = temp_file_path.replace('.cpp', '.o')
                        if os.path.exists(o_file):
                            os.unlink(o_file)
                        os.unlink(temp_file_path)
                except subprocess.TimeoutExpired:
                    artifacts["compilation_timeout"] = "C++ compilation timed out"
                except Exception as e:
                    artifacts["compilation_error"] = str(e)
                    
            # Language-agnostic checks - comprehensive list
            if content_type == "code_python":
                metrics.update({
                    "has_functions": "def " in content,
                    "has_classes": "class " in content,
                    "has_imports": "import " in content,
                    "has_main_guard": "if __name__ == \"__main__\":" in content,
                    "comment_ratio": content.count("#") / max(1, len(content.split())),
                })
            elif content_type == "code_js":
                metrics.update({
                    "has_functions": "function" in content or "=>" in content or "function(" in content,
                    "has_classes": "class " in content,
                    "has_imports": "import" in content or "require" in content,
                    "comment_ratio": (content.count("//") + content.count("/*")) / max(1, len(content.split())),
                })
            elif content_type == "code_java":
                metrics.update({
                    "has_imports": "import " in content,
                    "has_classes": "class " in content,
                    "has_package": "package " in content,
                    "has_public_class": "public class" in content,
                    "comment_ratio": (content.count("//") + content.count("/*")) / max(1, len(content.split())),
                })
            elif content_type == "code_cpp":
                metrics.update({
                    "has_includes": "#include" in content,
                    "has_namespaces": "namespace" in content,
                    "has_classes": "class " in content or "struct " in content,
                    "has_templates": "template" in content,
                    "comment_ratio": (content.count("//") + content.count("/*")) / max(1, len(content.split())),
                })
            elif content_type == "code_csharp":
                metrics.update({
                    "has_using": "using " in content,
                    "has_namespaces": "namespace" in content,
                    "has_classes": "class " in content,
                    "has_public_class": "public class" in content,
                    "comment_ratio": (content.count("//") + content.count("/*")) / max(1, len(content.split())),
                })
            elif content_type == "code_go":
                metrics.update({
                    "has_package": "package " in content,
                    "has_imports": "import " in content,
                    "has_functions": "func " in content,
                    "has_structs": "struct " in content,
                    "comment_ratio": content.count("//") / max(1, len(content.split())),
                })
            elif content_type == "code_rust":
                metrics.update({
                    "has_mod": "mod " in content,
                    "has_use": "use " in content,
                    "has_functions": "fn " in content,
                    "has_structs": "struct " in content,
                    "comment_ratio": content.count("//") / max(1, len(content.split())),
                })
            elif content_type == "code_swift":
                metrics.update({
                    "has_imports": "import " in content,
                    "has_functions": "func " in content,
                    "has_classes": "class " in content or "struct " in content,
                    "has_protocols": "protocol " in content,
                    "comment_ratio": (content.count("//") + content.count("/*")) / max(1, len(content.split())),
                })
            elif content_type == "code_kotlin":
                metrics.update({
                    "has_imports": "import " in content,
                    "has_functions": "fun " in content,
                    "has_classes": "class " in content,
                    "has_package": "package " in content,
                    "comment_ratio": (content.count("//") + content.count("/*")) / max(1, len(content.split())),
                })
            elif content_type == "code_typescript":
                metrics.update({
                    "has_imports": "import " in content,
                    "has_functions": "function " in content or "=>" in content,
                    "has_classes": "class " in content,
                    "has_interfaces": "interface " in content,
                    "comment_ratio": (content.count("//") + content.count("/*")) / max(1, len(content.split())),
                })
            elif content_type == "code_php":
                metrics.update({
                    "has_functions": "function " in content,
                    "has_classes": "class " in content,
                    "has_variables": "$" in content,
                    "comment_ratio": content.count("//") + content.count("/*") / max(1, len(content.split())),
                })
            elif content_type == "code_ruby":
                metrics.update({
                    "has_methods": "def " in content,
                    "has_classes": "class " in content,
                    "has_modules": "module " in content,
                    "comment_ratio": content.count("#") / max(1, len(content.split())),
                })
            elif content_type == "code_perl":
                metrics.update({
                    "has_subroutines": "sub " in content,
                    "has_packages": "package " in content,
                    "has_variables": "$" in content or "@" in content or "%" in content,
                    "comment_ratio": content.count("#") / max(1, len(content.split())),
                })
            elif content_type == "code_scala":
                metrics.update({
                    "has_functions": "def " in content,
                    "has_classes": "class " in content,
                    "has_objects": "object " in content,
                    "comment_ratio": content.count("//") + content.count("/*") / max(1, len(content.split())),
                })
            elif content_type == "code_haskell":
                metrics.update({
                    "has_functions": "::" in content,
                    "has_data_types": "data " in content,
                    "has_modules": "module " in content,
                    "comment_ratio": content.count("--") / max(1, len(content.split())),
                })
            elif content_type == "code_r":
                metrics.update({
                    "has_functions": "function(" in content,
                    "has_variables": "<-" in content or "=" in content,
                    "comment_ratio": content.count("#") / max(1, len(content.split())),
                })
            elif content_type == "code_matlab":
                metrics.update({
                    "has_functions": "function " in content,
                    "has_scripts": ".m" in content,
                    "comment_ratio": content.count("%") / max(1, len(content.split())),
                })
            elif content_type == "code_assembly":
                metrics.update({
                    "has_labels": ":" in content,
                    "has_instructions": "mov " in content or "add " in content,
                    "comment_ratio": content.count(";") / max(1, len(content.split())),
                })
            elif content_type == "code_shell":
                metrics.update({
                    "has_shebang": "#!" in content,
                    "has_variables": "$" in content,
                    "has_functions": "() {" in content,
                    "comment_ratio": content.count("#") / max(1, len(content.split())),
                })
            elif content_type == "code_sql":
                metrics.update({
                    "has_select": "SELECT" in content.upper(),
                    "has_from": "FROM" in content.upper(),
                    "has_where": "WHERE" in content.upper(),
                    "comment_ratio": content.count("--") + content.count("/*") / max(1, len(content.split())),
                })
            elif content_type == "code_html":
                metrics.update({
                    "has_doctype": "<!DOCTYPE html>" in content.lower(),
                    "has_head": "<head>" in content.lower(),
                    "has_body": "<body>" in content.lower(),
                    "comment_ratio": content.count("<!--") / max(1, len(content.split())),
                })
            elif content_type == "code_css":
                metrics.update({
                    "has_selectors": "{" in content and "}" in content,
                    "has_properties": ":" in content and ";" in content,
                    "comment_ratio": content.count("/*") / max(1, len(content.split())),
                })
            elif content_type == "code_json":
                try:
                    import json
                    json.loads(content)
                    metrics.update({
                        "is_valid_json": True,
                        "has_objects": "{" in content,
                        "has_arrays": "[" in content,
                    })
                except json.JSONDecodeError:
                    metrics.update({
                        "is_valid_json": False,
                        "parse_error": "Invalid JSON format"
                    })
            elif content_type == "code_xml":
                metrics.update({
                    "has_xml_declaration": "<?xml" in content.lower(),
                    "has_tags": "<" in content and ">" in content,
                })
            elif content_type == "code_yaml":
                metrics.update({
                    "has_key_value_pairs": ":" in content,
                    "has_lists": "-" in content,
                })
            elif content_type == "code_markdown":
                metrics.update({
                    "has_headers": "#" in content,
                    "has_lists": "-" in content or "*" in content or "1." in content,
                    "has_links": "[" in content and "]" in content and "(" in content and ")" in content,
                })
            elif content_type == "document_legal":
                metrics.update({
                    "has_contract_terms": any(
                        term in content.lower()
                        for term in [
                            "contract",
                            "agreement",
                            "clause",
                            "party",
                            "jurisdiction",
                        ]
                    ),
                    "has_legal_citations": any(
                        citation in content
                        for citation in ["§", "et al.", "v.", "supra"]
                    ),
                    "document_length_words": len(content.split()),
                })
            elif content_type == "document_medical":
                metrics.update({
                    "has_medical_terms": any(
                        term in content.lower()
                        for term in [
                            "patient",
                            "diagnosis",
                            "treatment",
                            "symptom",
                            "medication",
                            "hospital",
                            "clinic",
                        ]
                    ),
                    "has_patient_id_format": "PID-"
                    in content,  # Simple check for a common pattern
                    "document_length_words": len(content.split()),
                })
            elif content_type == "document_technical":
                metrics.update({
                    "has_code_snippets": any(
                        keyword in content
                        for keyword in [
                            "def ",
                            "class ",
                            "import ",
                            "function ",
                            "{",
                            "}",
                            ";",
                        ]
                    ),
                    "has_technical_terms": any(
                        term in content.lower()
                        for term in [
                            "api",
                            "database",
                            "server",
                            "client",
                            "algorithm",
                            "framework",
                            "library",
                        ]
                    ),
                    "document_length_words": len(content.split()),
                })
            elif content_type == "code_lean4":
                metrics.update({
                    "has_imports": "import " in content,
                    "has_theorems": "theorem " in content,
                    "has_definitions": "def " in content or "abbrev " in content,
                    "comment_ratio": content.count("--") / max(1, len(content.split())),
                })

            # Calculate a composite score based on various factors
            score_components = []

            # Length score (favor moderate length)
            length_score = min(1.0, len(content) / 1000.0)
            score_components.append(length_score * 0.2)

            # Linter score
            linter_score = metrics.get("linter_score", 0.0) / 10.0  # Normalize to 0-1
            score_components.append(linter_score * 0.3)

            # Structure score (favor well-structured code)
            structure_score = 0.0
            structure_elements = []
            for key, value in metrics.items():
                if key.startswith(("has_", "is_")) and isinstance(value, bool):
                    if value:
                        structure_elements.append(key)
            structure_score = min(1.0, len(structure_elements) / 10.0)  # Max 10 structural elements
            score_components.append(structure_score * 0.2)

            # Comment score (favor reasonable commenting)
            comment_ratio = metrics.get("comment_ratio", 0.0)
            if 0.01 <= comment_ratio <= 0.3:  # Reasonable comment ratio
                comment_score = 0.5
            elif comment_ratio > 0.3:  # Too many comments
                comment_score = 0.3
            else:  # Too few comments
                comment_score = comment_ratio * 10
            score_components.append(comment_score * 0.15)

            # Custom requirements score
            if custom_requirements:
                # Simple keyword matching for now
                req_matches = sum(
                    1
                    for req in custom_requirements.split(",")
                    if req.strip().lower() in content.lower()
                )
                req_score = min(
                    1.0, req_matches / max(1, len(custom_requirements.split(",")))
                )
                score_components.append(req_score * 0.15)

            # Domain relevance score (for document types)
            domain_relevance_score = 0.0
            if content_type.startswith("document_"):
                if content_type == "document_legal" and metrics.get("has_contract_terms"):
                    domain_relevance_score += 0.3
                if content_type == "document_medical" and metrics.get("has_medical_terms"):
                    domain_relevance_score += 0.3
                if content_type == "document_technical" and metrics.get("has_technical_terms"):
                    domain_relevance_score += 0.3
                # Add more sophisticated checks here if needed
                score_components.append(domain_relevance_score * 0.3)  # Give it a moderate weight

            # Compliance score
            compliance_score = metrics.get("compliance_score", 1.0)
            score_components.append(compliance_score * 0.2)

            # Calculate final score as weighted average of components
            if score_components:
                # Normalize weights to sum to 1
                final_score = sum(score_components) / len(score_components)
            else:
                final_score = 0.5  # Default neutral score

            metrics["combined_score"] = final_score
            metrics["score_components"] = {
                "length": length_score,
                "linter": linter_score,
                "structure": structure_score,
                "comments": comment_score,
                "requirements": score_components[-2] if len(score_components) >= 2 else 0.0,
                "compliance": compliance_score,
                "domain_relevance": domain_relevance_score,
            }

            # Add artifacts to metrics if they exist
            if artifacts:
                metrics["artifacts"] = artifacts
                # Add artifact feedback to stderr for OpenEvolve to use in next generation
                for artifact_type, artifact_data in artifacts.items():
                    if "error" in artifact_type.lower():
                        if "stderr" not in metrics:
                            metrics["stderr"] = ""
                        metrics["stderr"] += f"⚠️ {artifact_type.replace('_', ' ').title()}: {artifact_data}\n"
                    elif "warning" in artifact_type.lower():
                        if "warnings" not in metrics:
                            metrics["warnings"] = ""
                        metrics["warnings"] += f"⚠️ {artifact_type.replace('_', ' ').title()}: {artifact_data}\n"

            # Add LLM-based feedback if enabled and available
            try:
                # Check if we should use LLM feedback (this requires access to session state)
                # For this integration, we'll attempt to get LLM feedback if the evaluator supports it
                llm_feedback = _get_llm_feedback(content, content_type, custom_requirements)
                if llm_feedback:
                    metrics["llm_feedback"] = llm_feedback
                    if "llm_feedback_score" in llm_feedback:
                        # Adjust the combined score based on LLM feedback
                        # Only apply if we have a valid LLM score
                        llm_score = llm_feedback["llm_feedback_score"]
                        if isinstance(llm_score, (int, float)) and 0 <= llm_score <= 1:
                            # Blend LLM feedback with existing metrics
                            llm_weight = 0.15  # 15% weight to LLM feedback
                            metrics["combined_score"] = (1 - llm_weight) * final_score + llm_weight * llm_score
            except Exception:
                # If LLM feedback fails, continue with regular evaluation
                pass

            return metrics

        except Exception as e:
            # Return error metrics in a format that OpenEvolve can process
            error_metrics = {
                "score": 0.0, 
                "error": str(e), 
                "timestamp": time.time(),
                "combined_score": 0.0,  # Critical for OpenEvolve
                "length": 0,
                "complexity": 0.0,
                "diversity": 0.0,
                "artifacts": {"execution_error": str(e)},
                "stderr": f"Error during evaluation: {str(e)}"
            }
            return error_metrics

    return code_evaluator


def get_language_from_content_type(content_type: str) -> str:
    """Get programming language from content type."""
    language_map = {
        "code_python": "python",
        "code_js": "javascript",
        "code_java": "java",
        "code_cpp": "cpp",
        "code_csharp": "csharp",
        "code_go": "go",
        "code_rust": "rust",
        "code_swift": "swift",
        "code_kotlin": "kotlin",
        "code_typescript": "typescript",
        "code_php": "php",
        "code_ruby": "ruby",
        "code_perl": "perl",
        "code_scala": "scala",
        "code_haskell": "haskell",
        "code_r": "r",
        "code_matlab": "matlab",
        "code_assembly": "assembly",
        "code_shell": "shell",
        "code_sql": "sql",
        "code_html": "html",
        "code_css": "css",
        "code_json": "json",
        "code_xml": "xml",
        "code_yaml": "yaml",
        "code_markdown": "markdown",
        "document_legal": "document",
        "document_medical": "document",
        "document_technical": "document",
        "document_general": "document",
        "document_report": "document",
        "code_lean4": "lean4",
    }
    return language_map.get(content_type, "python")


def get_file_suffix_from_content_type(content_type: str) -> str:
    """Get file suffix from content type."""
    suffix_map = {
        "code_python": ".py",
        "code_js": ".js",
        "code_java": ".java",
        "code_cpp": ".cpp",
        "code_csharp": ".cs",
        "code_go": ".go",
        "code_rust": ".rs",
        "code_swift": ".swift",
        "code_kotlin": ".kt",
        "code_typescript": ".ts",
        "code_php": ".php",
        "code_ruby": ".rb",
        "code_perl": ".pl",
        "code_scala": ".scala",
        "code_haskell": ".hs",
        "code_r": ".r",
        "code_matlab": ".m",
        "code_assembly": ".asm",
        "code_shell": ".sh",
        "code_sql": ".sql",
        "code_html": ".html",
        "code_css": ".css",
        "code_json": ".json",
        "code_xml": ".xml",
        "code_yaml": ".yaml",
        "code_markdown": ".md",
        "document_legal": ".txt",
        "document_medical": ".txt",
        "document_technical": ".txt",
        "document_general": ".txt",
        "document_report": ".txt",
        "code_lean4": ".lean",
    }
    return suffix_map.get(content_type, ".py")


def run_specialized_code_evolution(
    content: str,
    content_type: str,
    model_name: str,
    api_key: str,
    api_base: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_iterations: int = 100,
    population_size: int = 1000,
    num_islands: int = 5,
    archive_size: int = 100,
    elite_ratio: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.7,
    checkpoint_interval: int = 100,
    custom_requirements: str = "",
    custom_evaluator: Optional[Callable] = None,
    compliance_rules: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run specialized code evolution using the OpenEvolve backend with linters.

    Args:
        content: The code content to evolve
        content_type: Type of content (e.g., 'code_python', 'code_js', etc.)
        model_name: Name of the LLM model to use
        api_key: API key for the LLM provider
        api_base: Base URL for the API (optional)
        temperature: Temperature for generation (0.0-2.0)
        top_p: Top-p sampling parameter (0.0-1.0)
        max_tokens: Maximum tokens to generate
        max_iterations: Maximum number of evolution iterations
        population_size: Size of the population
        num_islands: Number of islands for island-based evolution
        archive_size: Size of the archive for storing best solutions
        elite_ratio: Ratio of elite individuals to preserve
        exploration_ratio: Ratio for exploration in evolution
        exploitation_ratio: Ratio for exploitation in evolution
        checkpoint_interval: Interval for saving checkpoints
        custom_requirements: Custom requirements to check for
        custom_evaluator: Custom evaluator function (optional)

    Returns:
        Dictionary with evolution results or None if failed
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available.")
        return None

    try:
        # Create advanced configuration
        config = create_advanced_openevolve_config(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
            population_size=population_size,
            num_islands=num_islands,
            archive_size=archive_size,
            elite_ratio=elite_ratio,
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            checkpoint_interval=checkpoint_interval,
            language=get_language_from_content_type(content_type),
            file_suffix=get_file_suffix_from_content_type(content_type),
        )

        if not config:
            return None

        # Create evaluator
        if custom_evaluator:
            evaluator = custom_evaluator
        else:
            evaluator = create_specialized_evaluator(
                content_type, custom_requirements, compliance_rules
            )

        # Create temporary file for the content with proper evolution markers
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=get_file_suffix_from_content_type(content_type),
            delete=False,
        ) as temp_file:
            # Add evolution markers to the content
            content_with_markers = f"""# EVOLVE-BLOCK-START
{content}
# EVOLVE-BLOCK-END"""
            temp_file.write(content_with_markers)
            temp_file_path = temp_file.name

        try:
            # Run evolution using OpenEvolve API
            result: EvolutionResult = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=evaluator,
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
                    "best_program": asdict(result.best_program)
                    if result.best_program
                    else None,
                    "best_score": result.best_score,
                    "best_code": best_code,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                }
            else:
                return {
                    "success": False,
                    "message": "Evolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running specialized code evolution: {e}")
        import traceback

        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def get_industry_template(industry_type: str, template_name: str) -> str:
    """
    Provides industry-specific content templates.

    Args:
        industry_type: The industry type (e.g., "legal", "medical", "technical").
        template_name: The specific template name (e.g., "contract_draft", "patient_report", "api_docs").

    Returns:
        A string containing the template content.
    """
    templates = {
        "legal": {
            "contract_draft": """# Draft Legal Contract\n\nThis Contract, made and entered into this [Day] day of [Month], [Year], by and between [Party A Name] ("Party A") and [Party B Name] ("Party B").\n\n**WHEREAS,** Party A is engaged in the business of [Party A Business];\n**WHEREAS,** Party B desires to engage Party A for [Service Description];\n\n**NOW, THEREFORE,** in consideration of the mutual covenants and agreements contained herein, the parties agree as follows:\n\n1.  **Term.** The term of this Contract shall commence on [Start Date] and continue until [End Date].\n2.  **Services.** Party A shall provide the following services: [Detailed Service List].\n3.  **Compensation.** Party B shall pay Party A the sum of [Amount] for the services rendered.\n\n**IN WITNESS WHEREOF,** the parties have executed this Contract as of the date first written above.\n\n_________________________\nParty A Signature\n\n_________________________\nParty B Signature\n""",
            "privacy_policy_draft": """# Draft Privacy Policy\n\nThis Privacy Policy describes how [Your Company Name] collects, uses, and discloses your personal information when you visit our website [Your Website URL].\n\n**1. Information We Collect**\nWe collect various types of information in connection with the services we provide, including:\n*   **Personal Data:** Name, email address, phone number, etc.\n*   **Usage Data:** IP address, browser type, pages visited, etc.\n\n**2. How We Use Your Information**\nWe use the collected information for various purposes, including to:\n*   Provide and maintain our Service\n*   Notify you about changes to our Service\n*   Allow you to participate in interactive features of our Service when you choose to do so\n\n**3. Disclosure of Your Information**\nWe may disclose your personal information in the good faith belief that such action is necessary to:\n*   Comply with a legal obligation\n*   Protect and defend the rights or property of [Your Company Name]\n\n**4. Your Data Protection Rights**\nDepending on your location, you may have the following data protection rights:\n*   The right to access, update or to delete the information we have on you.\n*   The right to rectify any inaccurate information.\n\n**5. Changes to This Privacy Policy**\nWe may update our Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page.\n\n*Last updated: [Date]*\n""",
        },
        "medical": {
            "patient_report_template": """# Patient Medical Report\n\n**Patient Name:** [Patient Name]\n**Date of Birth:** [DOB]\n**Patient ID:** [Patient ID]\n**Date of Report:** [Date]\n\n--- \n\n**Chief Complaint:** [Patient's primary reason for visit]\n\n**History of Present Illness (HPI):**\n[Detailed description of the illness, including onset, duration, character, associated symptoms, relieving/aggravating factors.]\n\n**Past Medical History (PMH):**\n*   **Conditions:** [List of past medical conditions]\n*   **Surgeries:** [List of past surgeries with dates]\n*   **Medications:** [List of current medications with dosage and frequency]\n*   **Allergies:** [List of known allergies]\n\n**Physical Examination:**\n*   **General:** [General appearance, vital signs]\n*   **HEENT:** [Head, Eyes, Ears, Nose, Throat findings]\n*   **Cardiovascular:** [Heart sounds, rhythm]\n*   **Respiratory:** [Lung sounds, effort]\n*   **Abdominal:** [Abdomen findings]\n*   **Neurological:** [Neurological assessment]\n\n**Assessment:**\n[Summary of findings and differential diagnoses.]\n\n**Plan:**\n[Treatment plan, including medications, further investigations, referrals, and follow-up instructions.]\n\n--- \n\n**Physician Signature:** _________________________\n**Date:** [Date]""",
            "research_protocol_template": """# Medical Research Protocol Draft\n\n**Protocol Title:** [Title of Research Study]\n**Principal Investigator:** [PI Name]\n**Institution:** [Institution Name]\n**Version:** [Version Number]\n**Date:** [Date]\n\n--- \n\n**1. Introduction**\n*   **Background:** [Brief overview of the research area and rationale for the study.]\n*   **Study Objectives:** [Primary and secondary objectives of the study.]\n\n**2. Study Design**\n*   **Type of Study:** [e.g., Randomized Controlled Trial, Observational Study, Case-Control]\n*   **Study Population:** [Inclusion and exclusion criteria for participants.]\n*   **Sample Size:** [Justification for the sample size.]\n\n**3. Study Procedures**\n*   **Recruitment:** [How participants will be recruited.]\n*   **Interventions/Assessments:** [Detailed description of all procedures, interventions, and assessments.]\n*   **Data Collection:** [Methods for collecting data, including forms and instruments.]\n\n**4. Data Analysis**\n*   **Statistical Methods:** [Planned statistical analyses.]\n\n**5. Ethical Considerations**\n*   **IRB/Ethics Committee Approval:** [Statement regarding ethical approval.]\n*   **Informed Consent:** [Process for obtaining informed consent.]\n\n**6. Dissemination Plan**\n*   [How study results will be disseminated.]\n\n--- \n\n**Approval Signature:** _________________________\n**Date:** [Date]""",
        },
        "technical": {
            "api_documentation_template": """# API Documentation Draft: [API Name]\n\n**Version:** 1.0.0\n**Date:** [Date]\n**Author:** [Author Name]\n\n--- \n\n## 1. Introduction\n[Brief overview of the API's purpose and functionality.]\n\n## 2. Authentication\n[Describe how users authenticate with the API (e.g., API keys, OAuth 2.0).]\n\n## 3. Endpoints\n
### `GET /resource`\n*   **Description:** Retrieves a list of resources.\n*   **Parameters:**\n    *   `limit` (optional, integer): Maximum number of resources to return. Default is 10.\n    *   `offset` (optional, integer): Number of resources to skip.\n*   **Responses:**\n    *   `200 OK`: Successfully retrieved resources.\n        ```json\n        [\n            {\n                "id": 1,\n                "name": "Resource 1"\n            },\n            {\n                "id": 2,\n                "name": "Resource 2"\n            }\n        ]\n        ```\n    *   `401 Unauthorized`: Authentication failed.\n
### `POST /resource`\n*   **Description:** Creates a new resource.\n*   **Request Body:**\n    ```json\n    {\n        "name": "New Resource Name"\n    }\n    ```\n*   **Responses:**\n    *   `201 Created`: Resource successfully created.\n        ```json\n        {\n            "id": 3,\n            "name": "New Resource Name"\n        }\n        ```\n    *   `400 Bad Request`: Invalid input.\n
## 4. Error Codes\n[List common error codes and their meanings.]\n\n--- \n\n**Example Usage (Python):**\n```python\nimport requests\n\nAPI_BASE_URL = "https://api.example.com"\nAPI_KEY = "your_api_key"\n\nheaders = {\n    "Authorization": f"Bearer {API_KEY}",\n    "Content-Type": "application/json"\n}\n\n# Get resources\nresponse = requests.get(f"{API_BASE_URL}/resource", headers=headers)\nprint(response.json())\n\n# Create resource\nnew_resource_data = {\"name\": \"My New Resource\"}\nresponse = requests.post(f"{API_BASE_URL}/resource", headers=headers, json=new_resource_data)\nprint(response.json())\n```
""",
            "software_design_doc_template": """# Software Design Document: [Project Name]\n\n**1. Introduction**\n    *   **Purpose:** [Briefly describe the purpose of this document.]\n    *   **Scope:** [Define the boundaries of the system being designed.]\n    *   **Definitions, Acronyms, and Abbreviations:** [List any terms that need clarification.]\n\n**2. System Overview**\n    *   **High-Level Architecture:** [Diagram or description of the main components and their interactions.]\n    *   **Key Features:** [List the primary functionalities of the system.]\n\n**3. Detailed Design**\n    *   **Component Design:** [For each major component, describe its responsibilities, interfaces, and internal structure.]\n        *   **Component A:**\n            *   **Responsibilities:**\n            *   **Interfaces:**\n            *   **Dependencies:**\n    *   **Data Model:** [Description of data structures, database schemas, or object models.]\n    *   **User Interface Design (if applicable):** [Mockups or descriptions of the UI.]\n\n**4. Technical Considerations**\n    *   **Technologies Used:** [List programming languages, frameworks, libraries, databases.]\n    *   **Performance Requirements:** [Expected response times, throughput.]\n    *   **Security Considerations:** [Authentication, authorization, data protection.]\n    *   **Scalability:** [How the system will handle increased load.]\n\n**5. Future Work/Open Issues**\n    *   [Any known limitations, future enhancements, or unresolved design questions.]\n\n--- \n\n**Author:** [Your Name]\n**Date:** [Date]""",
        },
    }

    if industry_type in templates and template_name in templates[industry_type]:
        return templates[industry_type][template_name]
    return f"No template found for industry '{industry_type}' and template '{template_name}'."


def create_multi_model_config(
    models: List[Dict[str, any]],
    api_key: str,
    api_base: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> Optional[Config]:
    """
    Create a configuration with multiple models for ensemble evolution with intelligent fallback.

    Args:
        models: List of dictionaries with 'name', 'weight', 'temperature', 'top_p', 'max_tokens' keys
        api_key: API key for the LLM provider
        api_base: Base URL for the API (optional)
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate

    Returns:
        Config object or None if OpenEvolve is not available
    """
    if not OPENEVOLVE_AVAILABLE:
        return None

    try:
        config = Config()

        # Configure LLM models from the provided list
        llm_configs = []
        for model_info in models:
            llm_config = LLMModelConfig(
                name=model_info['name'],
                api_key=api_key,
                api_base=api_base if api_base else "https://api.openai.com/v1",
                temperature=model_info.get('temperature', temperature),
                top_p=model_info.get('top_p', top_p),
                max_tokens=model_info.get('max_tokens', max_tokens),
                weight=model_info.get('weight', 1.0),
                # Add reasoning effort if provided
                reasoning_effort=model_info.get('reasoning_effort', None)
            )
            llm_configs.append(llm_config)

        # Add all models to the config
        config.llm.models = llm_configs

        # Configure ensemble-specific settings
        config.llm.api_base = api_base if api_base else "https://api.openai.com/v1"

        # Configure evaluator models with the same models for consistency
        config.llm.evaluator_models = llm_configs

        # Enable intelligent fallback by setting up different weights for different models
        # The ensemble will try different models and combine their results
        config.evaluator.use_llm_feedback = True
        config.evaluator.llm_feedback_weight = 0.1

        return config

    except Exception as e:
        st.error(f"Error creating multi-model configuration: {e}")
        return None


def create_ensemble_config_with_fallback(
    primary_models: List[str],
    fallback_models: List[str],
    api_key: str,
    api_base: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    primary_weight: float = 1.0,
    fallback_weight: float = 0.3,
) -> Optional[Config]:
    """
    Create a configuration with primary models and fallback models for intelligent ensemble operation.

    Args:
        primary_models: List of primary model names
        fallback_models: List of fallback model names
        api_key: API key for the LLM provider
        api_base: Base URL for the API (optional)
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        primary_weight: Weight for primary models
        fallback_weight: Weight for fallback models

    Returns:
        Config object or None if OpenEvolve is not available
    """
    if not OPENEVOLVE_AVAILABLE:
        return None

    try:
        config = Config()

        # Configure primary models with higher weights
        primary_configs = []
        for model_name in primary_models:
            primary_config = LLMModelConfig(
                name=model_name,
                api_key=api_key,
                api_base=api_base if api_base else "https://api.openai.com/v1",
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                weight=primary_weight,
            )
            primary_configs.append(primary_config)

        # Configure fallback models with lower weights
        fallback_configs = []
        for model_name in fallback_models:
            fallback_config = LLMModelConfig(
                name=model_name,
                api_key=api_key,
                api_base=api_base if api_base else "https://api.openai.com/v1",
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                weight=fallback_weight,
            )
            fallback_configs.append(fallback_config)

        # Combine models (primary models first, then fallback)
        all_configs = primary_configs + fallback_configs
        config.llm.models = all_configs

        # Configure ensemble-specific settings for intelligent fallback
        config.evaluator.use_llm_feedback = True
        config.evaluator.llm_feedback_weight = 0.15
        config.prompt.use_template_stochasticity = True

        # Enable cascade evaluation for intelligent model selection
        config.evaluator.cascade_evaluation = True
        config.evaluator.cascade_thresholds = [0.6, 0.8, 0.9]

        return config

    except Exception as e:
        st.error(f"Error creating ensemble configuration with fallback: {e}")
        return None


def _get_llm_feedback(content: str, content_type: str, custom_requirements: str = "") -> Optional[Dict[str, Any]]:
    """
    Get feedback from an LLM on the quality of content.
    
    Args:
        content: The content to evaluate
        content_type: Type of content being evaluated
        custom_requirements: Custom requirements to check for

    Returns:
        Dictionary with LLM feedback or None if unavailable
    """
    try:
        # Import here to avoid circular imports
        import openai
        
        # Use the API key from session state or environment
        api_key = os.getenv("OPENAI_API_KEY") or getattr(st.session_state, 'api_key', None)
        if not api_key:
            return None
            
        client = openai.OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
        
        # Create a prompt for LLM-based evaluation
        prompt = f"""
        Evaluate the following {content_type.replace('_', ' ')} content based on quality, correctness, and best practices:

        Content:
        {content}

        Evaluation criteria:
        1. Correctness/accuracy
        2. Code quality (for code) or clarity (for documents)
        3. Adherence to best practices
        4. Structure and organization
        5. Completeness

        Provide your feedback in the following JSON format:
        {{
            "llm_feedback_score": float between 0 and 1,
            "feedback_summary": "Brief summary of feedback",
            "strengths": ["list", "of", "strengths"],
            "areas_for_improvement": ["list", "of", "improvements"],
            "suggestions": ["specific", "actionable", "suggestions"]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use a more cost-effective model for feedback
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more consistent feedback
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        import json
        feedback_data = json.loads(response.choices[0].message.content)
        
        return feedback_data
        
    except Exception as e:
        print(f"Error getting LLM feedback: {e}")
        return None


def create_comprehensive_openevolve_config(
    content_type: str,
    model_configs: List[Dict[str, any]],
    api_key: str,
    api_base: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_iterations: int = 100,
    population_size: int = 1000,
    num_islands: int = 5,
    migration_interval: int = 50,
    migration_rate: float = 0.1,
    archive_size: int = 100,
    elite_ratio: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.7,
    checkpoint_interval: int = 100,
    feature_dimensions: Optional[List[str]] = None,
    feature_bins: Optional[int] = None,
    diversity_metric: str = "edit_distance",
    reasoning_effort: Optional[str] = None,
    system_message: str = None,
    evaluator_system_message: str = None,
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
    # Additional advanced parameters for sophisticated evolution
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
    frequency_penalty: float = 0.0,  # Add frequency_penalty parameter
    presence_penalty: float = 0.0,  # Add presence_penalty parameter
) -> Optional[Config]:
    """
    Create a comprehensive OpenEvolve configuration with ALL features enabled.
    This is a full-featured configuration that enables all of OpenEvolve's capabilities
    including advanced research-grade features.
    """
    if not OPENEVOLVE_AVAILABLE:
        return None

    try:
        # Create comprehensive configuration
        config = Config()

        # Set general settings
        config.max_iterations = max_iterations
        config.checkpoint_interval = checkpoint_interval
        config.language = get_language_from_content_type(content_type)
        config.file_suffix = get_file_suffix_from_content_type(content_type)
        config.random_seed = random_seed
        config.early_stopping_patience = early_stopping_patience
        config.convergence_threshold = convergence_threshold
        config.early_stopping_metric = early_stopping_metric
        config.diff_based_evolution = diff_based_evolution
        config.max_code_length = max_code_length
        config.log_level = log_level
        config.log_dir = log_dir

        # Configure LLM models from the provided list (ensemble configuration)
        llm_configs = []
        for model_info in model_configs:
            llm_config = LLMModelConfig(
                name=model_info['name'],
                api_key=model_info.get('api_key', api_key),
                api_base=model_info.get('api_base', api_base if api_base else "https://api.openai.com/v1"),
                temperature=model_info.get('temperature', temperature),
                top_p=model_info.get('top_p', top_p),
                max_tokens=model_info.get('max_tokens', max_tokens),
                frequency_penalty=model_info.get('frequency_penalty', frequency_penalty),
                presence_penalty=model_info.get('presence_penalty', presence_penalty),
                timeout=model_info.get('timeout', api_timeout),
                retries=model_info.get('retries', api_retries),
                retry_delay=model_info.get('retry_delay', api_retry_delay),
                reasoning_effort=model_info.get('reasoning_effort'),
                random_seed=random_seed,
                weight=model_info.get('weight', 1.0),
            )
            llm_configs.append(llm_config)

        # Add all models to the config
        config.llm.models = llm_configs
        config.llm.system_message = system_message or config.llm.system_message

        # Configure evaluator models if provided
        if evaluator_models:
            evaluator_llm_configs = []
            for eval_model in evaluator_models:
                evaluator_config = LLMModelConfig(
                    name=eval_model['name'],
                    api_key=eval_model.get('api_key', api_key),
                    api_base=eval_model.get('api_base', api_base if api_base else "https://api.openai.com/v1"),
                    temperature=eval_model.get('temperature', 0.3),  # Lower temp for more consistent evaluation
                    top_p=eval_model.get('top_p', 0.9),
                    max_tokens=eval_model.get('max_tokens', 1024),
                    frequency_penalty=eval_model.get('frequency_penalty', frequency_penalty),
                    presence_penalty=eval_model.get('presence_penalty', presence_penalty),
                    timeout=eval_model.get('timeout', api_timeout),
                    retries=eval_model.get('retries', api_retries),
                    retry_delay=eval_model.get('retry_delay', api_retry_delay),
                    random_seed=random_seed,
                    weight=eval_model.get('weight', 1.0),
                )
                evaluator_llm_configs.append(evaluator_config)
            config.llm.evaluator_models = evaluator_llm_configs
        else:
            # Use the same models as evaluation models if none specified
            config.llm.evaluator_models = llm_configs

        # Configure prompt settings with all advanced features
        config.prompt = PromptConfig(
            template_dir=template_dir,
            system_message=system_message or config.prompt.system_message,
            evaluator_system_message=evaluator_system_message or config.prompt.evaluator_system_message,
            num_top_programs=num_top_programs,
            num_diverse_programs=num_diverse_programs,
            use_template_stochasticity=use_template_stochasticity,
            template_variations=template_variations or {},
            use_meta_prompting=use_meta_prompting,
            meta_prompt_weight=meta_prompt_weight,
            include_artifacts=include_artifacts,
            max_artifact_bytes=max_artifact_bytes,
            artifact_security_filter=artifact_security_filter,
            suggest_simplification_after_chars=500,
            include_changes_under_chars=100,
            concise_implementation_max_lines=10,
            comprehensive_implementation_min_lines=50,
        )

        # Configure database settings with all advanced features for enhanced evolution
        feature_dims = None
        if adaptive_feature_dimensions and feature_dimensions is not None:
            # For adaptive feature dimensions, we may need to adjust based on content type
            if content_type.startswith("code_"):
                feature_dims = feature_dimensions + ["performance", "efficiency", "readability"]
            elif content_type.startswith("document_"):
                feature_dims = feature_dimensions + ["clarity", "completeness", "accuracy"]
            else:
                feature_dims = feature_dimensions
        else:
            feature_dims = feature_dimensions or ["complexity", "diversity"]

        config.database = DatabaseConfig(
            db_path=db_path,
            in_memory=in_memory,
            population_size=population_size,
            archive_size=archive_size,
            num_islands=num_islands,
            elite_selection_ratio=elite_ratio,
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            diversity_metric=diversity_metric,
            feature_dimensions=feature_dims,
            feature_bins=feature_bins if feature_bins is not None else 10,
            migration_interval=migration_interval,
            migration_rate=migration_rate,
            random_seed=random_seed,
            log_prompts=True,
            diversity_reference_size=diversity_reference_size,
            artifacts_base_path=os.path.join(db_path, "artifacts") if db_path else None,
            artifact_size_threshold=artifact_size_threshold,
            cleanup_old_artifacts=cleanup_old_artifacts,
            artifact_retention_days=artifact_retention_days,
        )

        # Configure evaluator settings with all advanced features
        config.evaluator = EvaluatorConfig(
            timeout=evaluator_timeout,
            max_retries=max_retries_eval,
            memory_limit_mb=memory_limit_mb,
            cpu_limit=cpu_limit,
            cascade_evaluation=cascade_evaluation,
            cascade_thresholds=cascade_thresholds or [0.5, 0.75, 0.9],
            parallel_evaluations=parallel_evaluations,
            distributed=distributed,
            use_llm_feedback=use_llm_feedback,
            llm_feedback_weight=llm_feedback_weight,
            enable_artifacts=enable_artifacts,
            max_artifact_storage=100 * 1024 * 1024,  # 100MB per program
        )

        # Configure evolution trace settings for full logging and analysis
        config.evolution_trace = EvolutionTraceConfig(
            enabled=evolution_trace_enabled,
            format=evolution_trace_format,
            include_code=evolution_trace_include_code,
            include_prompts=evolution_trace_include_prompts,
            output_path=evolution_trace_output_path,
            buffer_size=evolution_trace_buffer_size,
            compress=evolution_trace_compress,
        )

        # Apply advanced research-grade features
        if double_selection:
            # Configure for double selection (different programs for performance vs inspiration)
            config.prompt.num_top_programs = max(3, num_top_programs)  # Ensure we have enough top programs
            config.prompt.num_diverse_programs = max(2, num_diverse_programs)  # Ensure we have diverse programs

        if test_time_compute:
            # When using test-time compute, we might want to adjust parameters
            config.llm.system_message = (
                f"{config.llm.system_message} [Enhanced reasoning through test-time compute enabled]"
            )

        if hardware_optimization:
            # For hardware optimization, adjust parameters appropriately
            config.prompt.system_message = (
                f"{config.prompt.system_message} [Optimize specifically for target hardware: GPU, TPU, etc.]"
            )
            if "performance" not in config.database.feature_dimensions:
                config.database.feature_dimensions.append("performance")
                
        if coevolutionary_approach:
            # For coevolution, we might need to adjust how populations interact
            config.database.num_islands = max(3, config.database.num_islands)
            config.database.migration_rate = min(0.2, config.database.migration_rate)  # Lower rate for co-evolution

        if multi_strategy_sampling:
            # Use multiple sampling strategies
            config.database.elite_selection_ratio = elite_ratio
            config.database.exploration_ratio = exploration_ratio  
            config.database.exploitation_ratio = exploitation_ratio
        
        if ring_topology:
            # Apply ring topology for migration
            # This is handled by OpenEvolve internally, but we can set appropriate parameters
            pass

        return config

    except Exception as e:
        st.error(f"Error creating comprehensive OpenEvolve configuration: {e}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None


def run_quality_diversity_evolution(
    content: str,
    content_type: str,
    model_configs: List[Dict[str, any]],
    api_key: str,
    api_base: str = None,
    max_iterations: int = 100,
    population_size: int = 1000,
    archive_size: int = 100,
    feature_dimensions: Optional[List[str]] = None,
    feature_bins: Optional[int] = 10,
    system_message: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    custom_requirements: str = "",
    custom_evaluator: Optional[Callable] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run Quality-Diversity (QD) evolution using MAP-Elites algorithm.
    This creates a diverse archive of high-performing solutions across feature dimensions.
    
    Args:
        content: The content to evolve
        content_type: Type of content
        model_configs: List of model configurations for ensemble evolution
        api_key: API key for the LLM provider
        api_base: Base URL for the API
        max_iterations: Number of evolution iterations
        population_size: Size of the population
        archive_size: Size of the MAP-Elites archive
        feature_dimensions: Feature dimensions for QD search
        feature_bins: Number of bins for each feature dimension
        system_message: Custom system message for the LLM
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        custom_requirements: Custom requirements to check for
        custom_evaluator: Custom evaluator function (optional)
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available.")
        return None

    # Default feature dimensions for QD search
    if feature_dimensions is None:
        if content_type.startswith("code_"):
            feature_dimensions = ["complexity", "performance", "readability"]
        elif content_type.startswith("document_"):
            feature_dimensions = ["complexity", "clarity", "completeness"]
        else:
            feature_dimensions = ["complexity", "diversity", "quality"]

    try:
        # Create QD-focused configuration
        config = create_comprehensive_openevolve_config(
            content_type=content_type,
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            archive_size=archive_size,
            feature_dimensions=feature_dimensions,
            feature_bins=feature_bins,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            # Enable advanced features for QD search
            num_islands=3,  # Use multiple islands for better exploration
            migration_interval=25,  # Enable migration between islands
            migration_rate=0.05,
            exploration_ratio=0.4,  # Higher exploration for diversity
            exploitation_ratio=0.5,
            elite_ratio=0.1,
            cascade_evaluation=True,
            use_llm_feedback=True,
            llm_feedback_weight=0.15,
            evolution_trace_enabled=True,
            evolution_trace_include_prompts=True,
        )

        if not config:
            return None

        # Create evaluator based on content type and custom requirements
        if custom_evaluator:
            evaluator = custom_evaluator
        else:
            evaluator = create_specialized_evaluator(content_type, custom_requirements)

        # Create a temporary file with the content to evolve
        with tempfile.NamedTemporaryFile(mode="w", suffix=get_file_suffix_from_content_type(content_type), delete=False) as temp_file:
            content_with_markers = f"""# EVOLVE-BLOCK-START
{content}
# EVOLVE-BLOCK-END"""
            temp_file.write(content_with_markers)
            temp_file_path = temp_file.name

        try:
            # Run QD evolution using OpenEvolve API
            result = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=evaluator,
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
                    start_idx = best_code.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                    end_idx = best_code.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_code = best_code[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": asdict(result.best_program) if result.best_program else None,
                    "best_score": result.best_score,
                    "best_code": best_code,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                    "config": {
                        "feature_dimensions": feature_dimensions,
                        "feature_bins": feature_bins,
                        "archive_size": archive_size,
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "QD evolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running Quality-Diversity evolution: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def run_multi_objective_evolution(
    content: str,
    content_type: str,
    objectives: List[str],  # List of objectives to optimize (e.g., ["performance", "readability", "maintainability"])
    model_configs: List[Dict[str, any]],
    api_key: str,
    api_base: str = None,
    max_iterations: int = 100,
    population_size: int = 1000,
    system_message: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    custom_requirements: str = "",
    custom_evaluator: Optional[Callable] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run multi-objective evolution to optimize for multiple objectives simultaneously.
    This creates a Pareto front of solutions that balance competing objectives.
    
    Args:
        content: The content to evolve
        content_type: Type of content
        objectives: List of objectives to optimize for
        model_configs: List of model configurations for ensemble evolution
        api_key: API key for the LLM provider
        api_base: Base URL for the API
        max_iterations: Number of evolution iterations
        population_size: Size of the population
        system_message: Custom system message for the LLM
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        custom_requirements: Custom requirements to check for
        custom_evaluator: Custom evaluator function (optional)
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available.")
        return None

    # Map objectives to feature dimensions
    objective_to_dimension = {
        "performance": "performance",
        "readability": "readability", 
        "maintainability": "maintainability",
        "complexity": "complexity",
        "efficiency": "efficiency",
        "clarity": "clarity",
        "completeness": "completeness",
        "security": "security",
        "robustness": "robustness",
    }
    
    feature_dimensions = []
    for obj in objectives:
        if obj in objective_to_dimension:
            feature_dimensions.append(objective_to_dimension[obj])
        else:
            feature_dimensions.append(obj)  # Use as-is if not in mapping

    try:
        # Create multi-objective focused configuration
        config = create_comprehensive_openevolve_config(
            content_type=content_type,
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            archive_size=population_size,  # Use population size as archive for Pareto front
            feature_dimensions=feature_dimensions,
            feature_bins=15,  # More bins for better objective resolution
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            # Enable advanced features for multi-objective optimization
            num_islands=5,  # Multiple islands to find diverse Pareto solutions
            migration_interval=20,
            migration_rate=0.1,
            exploration_ratio=0.3,
            exploitation_ratio=0.5,
            elite_ratio=0.2,  # Higher elite ratio to preserve Pareto solutions
            cascade_evaluation=True,
            use_llm_feedback=True,
            llm_feedback_weight=0.2,
            evolution_trace_enabled=True,
            evolution_trace_include_prompts=True,
        )

        if not config:
            return None

        # Create evaluator based on content type and custom requirements
        if custom_evaluator:
            evaluator = custom_evaluator
        else:
            evaluator = create_specialized_evaluator(content_type, custom_requirements)

        # Create a temporary file with the content to evolve
        with tempfile.NamedTemporaryFile(mode="w", suffix=get_file_suffix_from_content_type(content_type), delete=False) as temp_file:
            content_with_markers = f"""# EVOLVE-BLOCK-START
{content}
# EVOLVE-BLOCK-END"""
            temp_file.write(content_with_markers)
            temp_file_path = temp_file.name

        try:
            # Run multi-objective evolution using OpenEvolve API
            result = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=evaluator,
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
                    start_idx = best_code.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                    end_idx = best_code.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_code = best_code[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": asdict(result.best_program) if result.best_program else None,
                    "best_score": result.best_score,
                    "best_code": best_code,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                    "objectives": objectives,
                    "config": {
                        "feature_dimensions": feature_dimensions,
                        "archive_size": population_size,
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "Multi-objective evolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running Multi-Objective evolution: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def run_adversarial_evolution(
    content: str,
    content_type: str,
    attack_model_config: Dict[str, any],
    defense_model_config: Dict[str, any],
    api_key: str,
    api_base: str = None,
    max_iterations: int = 50,
    population_size: int = 500,
    system_message: str = None,
    temperature: float = 0.8,  # Higher temperature for more diverse attacks
    max_tokens: int = 4096,
    custom_requirements: str = "",
    custom_evaluator: Optional[Callable] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run adversarial evolution using Red Team/Blue Team approach.
    This uses two model configurations - one for attack (find vulnerabilities) 
    and one for defense (patch vulnerabilities).
    
    Args:
        content: The content to evolve (make more robust)
        content_type: Type of content
        attack_model_config: Model configuration for red team (attack/fault finding)
        defense_model_config: Model configuration for blue team (defense/fixing)
        api_key: API key for the LLM provider
        api_base: Base URL for the API
        max_iterations: Number of evolution iterations
        population_size: Size of the population
        system_message: Custom system message for the LLM
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        custom_requirements: Custom requirements to check for
        custom_evaluator: Custom evaluator function (optional)
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available.")
        return None

    try:
        # Create model configurations for red team/blue team
        model_configs = [attack_model_config, defense_model_config]
        
        # Create adversarial-focused configuration
        config = create_comprehensive_openevolve_config(
            content_type=content_type,
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            archive_size=100,
            # Use custom feature dimensions relevant to adversarial robustness
            feature_dimensions=["vulnerability_count", "robustness_score", "functionality_preserved"],
            feature_bins=10,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            # Enable advanced features for adversarial evolution
            num_islands=3,  # Use multiple islands for diverse attack/defense strategies
            migration_interval=15,
            migration_rate=0.15,
            exploration_ratio=0.4,  # Higher exploration for finding diverse attack vectors
            exploitation_ratio=0.4,
            elite_ratio=0.2,
            cascade_evaluation=True,
            use_llm_feedback=True,
            llm_feedback_weight=0.25,  # Higher weight for adversarial feedback
            evolution_trace_enabled=True,
            evolution_trace_include_prompts=True,
            # Focus on artifacts to identify vulnerabilities
            include_artifacts=True,
            max_artifact_bytes=30 * 1024,  # Allow more artifact space for vulnerability reports
        )

        if not config:
            return None

        # Create adversarial evaluator that simulates red team/blue team dynamics
        def adversarial_evaluator(program_path):
            """Evaluator that simulates adversarial testing with red team/blue team dynamics."""
            with open(program_path, "r") as f:
                content = f.read()
            
            # Base metrics
            metrics = {
                "timestamp": time.time(),
                "length": len(content),
                "functionality_preserved": 1.0,  # Start optimistic
                "vulnerability_count": 0,
                "robustness_score": 0.5,  # Neutral starting point
                "complexity": len(content.split()) / 100.0,
                "diversity": len(set(content.split())) / max(1, len(content.split())),
            }
            
            # In a real implementation, we would perform actual adversarial testing
            # For now, we'll simulate it with basic checks
            vulnerabilities_found = []
            
            # Simulate basic vulnerability checks
            if content_type.startswith("code_"):
                # Check for common security issues
                if "eval(" in content or "exec(" in content:
                    vulnerabilities_found.append("Code execution vulnerability")
                if "password" in content.lower() and "=" in content:
                    vulnerabilities_found.append("Hardcoded credential")
                if "sql" in content.lower() and "input" in content.lower():
                    vulnerabilities_found.append("Potential SQL injection")
                
                metrics["vulnerability_count"] = len(vulnerabilities_found)
                metrics["robustness_score"] = max(0.1, 1.0 - (len(vulnerabilities_found) * 0.2))
            
            # Calculate combined score based on robustness and functionality
            functionality_score = min(1.0, len(content) / 500.0)  # Basic functionality measure
            combined_score = (
                metrics["robustness_score"] * 0.6 + 
                functionality_score * 0.4
            )
            
            metrics["combined_score"] = combined_score
            metrics["vulnerabilities"] = vulnerabilities_found
            
            return metrics

        # Create evaluator based on content type and custom requirements
        if custom_evaluator:
            evaluator = custom_evaluator
        else:
            evaluator = adversarial_evaluator

        # Create a temporary file with the content to evolve
        with tempfile.NamedTemporaryFile(mode="w", suffix=get_file_suffix_from_content_type(content_type), delete=False) as temp_file:
            content_with_markers = f"""# EVOLVE-BLOCK-START
{content}
# EVOLVE-BLOCK-END"""
            temp_file.write(content_with_markers)
            temp_file_path = temp_file.name

        try:
            # Run adversarial evolution using OpenEvolve API
            result = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=evaluator,
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
                    start_idx = best_code.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                    end_idx = best_code.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_code = best_code[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": asdict(result.best_program) if result.best_program else None,
                    "best_score": result.best_score,
                    "best_code": best_code,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                    "config": {
                        "attack_model": attack_model_config["name"],
                        "defense_model": defense_model_config["name"],
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "Adversarial evolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running Adversarial evolution: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def run_prompt_evolution(
    initial_prompt: str,
    evaluation_function: Callable[[str], Dict[str, float]],  # Function that evaluates prompt quality
    model_configs: List[Dict[str, any]],
    api_key: str,
    api_base: str = None,
    max_iterations: int = 50,
    population_size: int = 200,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Optional[Dict[str, Any]]:
    """
    Evolve prompts to optimize for specific outcomes (prompt optimization).
    This can be used to improve LLM prompt effectiveness.
    
    Args:
        initial_prompt: The initial prompt to evolve
        evaluation_function: Function that takes a prompt string and returns metrics dict
        model_configs: List of model configurations for ensemble evolution
        api_key: API key for the LLM provider
        api_base: Base URL for the API
        max_iterations: Number of evolution iterations
        population_size: Size of the population
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available.")
        return None

    try:
        # Create prompt evolution-focused configuration
        config = create_comprehensive_openevolve_config(
            content_type="document_general",  # Prompts are general documents
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            archive_size=50,
            # Focus on prompt quality dimensions
            feature_dimensions=["clarity", "effectiveness", "conciseness"],
            feature_bins=10,
            system_message="You are an expert prompt engineer. Improve the prompt to make it more effective for the target task.",
            temperature=temperature,
            max_tokens=max_tokens,
            # Enable advanced features for prompt evolution
            num_islands=2,
            migration_interval=25,
            migration_rate=0.1,
            exploration_ratio=0.3,
            exploitation_ratio=0.6,
            elite_ratio=0.1,
            cascade_evaluation=True,
            use_llm_feedback=True,
            llm_feedback_weight=0.1,
            evolution_trace_enabled=True,
            evolution_trace_include_prompts=True,
            # Prompt-specific settings
            suggest_simplification_after_chars=2000,  # Longer threshold for prompts
            concise_implementation_max_lines=20,  # More lines allowed for detailed prompts
        )

        if not config:
            return None

        # Create evaluator that uses the provided evaluation function
        def prompt_evaluator(program_path):
            with open(program_path, "r") as f:
                prompt_content = f.read()
            
            # Remove evolution markers
            if "# EVOLVE-BLOCK-START" in prompt_content:
                start_idx = prompt_content.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                end_idx = prompt_content.find("# EVOLVE-BLOCK-END")
                if end_idx != -1:
                    prompt_content = prompt_content[start_idx:end_idx].strip()
            
            # Get evaluation from the provided function
            try:
                eval_result = evaluation_function(prompt_content)
                # Ensure we have a combined score
                if "combined_score" not in eval_result:
                    # Calculate based on available metrics
                    score_values = [v for v in eval_result.values() if isinstance(v, (int, float))]
                    if score_values:
                        eval_result["combined_score"] = sum(score_values) / len(score_values)
                    else:
                        eval_result["combined_score"] = 0.5  # Default neutral score
                return eval_result
            except Exception as e:
                return {
                    "combined_score": 0.0,
                    "error": str(e),
                    "timestamp": time.time(),
                    "length": len(prompt_content),
                }

        # Create a temporary file with the initial prompt
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            prompt_with_markers = f"""# EVOLVE-BLOCK-START
{initial_prompt}
# EVOLVE-BLOCK-END"""
            temp_file.write(prompt_with_markers)
            temp_file_path = temp_file.name

        try:
            # Run prompt evolution using OpenEvolve API
            result = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=prompt_evaluator,
                config=config,
                iterations=max_iterations,
                output_dir=None,  # Use temporary directory
                cleanup=True,
            )

            # Process results
            if result.best_program and result.best_code:
                # Remove evolution markers from the final result
                best_prompt = result.best_code
                if "# EVOLVE-BLOCK-START" in best_prompt:
                    start_idx = best_prompt.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                    end_idx = best_prompt.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_prompt = best_prompt[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": asdict(result.best_program) if result.best_program else None,
                    "best_score": result.best_score,
                    "best_prompt": best_prompt,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                }
            else:
                return {
                    "success": False,
                    "message": "Prompt evolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running Prompt evolution: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def run_algorithm_discovery_evolution(
    problem_description: str,
    evaluation_function: Callable[[str], Dict[str, any]],  # Function that evaluates algorithm quality
    model_configs: List[Dict[str, any]],
    api_key: str,
    api_base: str = None,
    max_iterations: int = 100,
    population_size: int = 1000,
    temperature: float = 0.8,  # Higher temperature for more creative solutions
    max_tokens: int = 4096,
    language: str = "python",
) -> Optional[Dict[str, Any]]:
    """
    Run algorithm discovery evolution to find novel algorithmic solutions.
    This mode focuses on discovering entirely new algorithms rather than just optimizing existing ones.
    
    Args:
        problem_description: Description of the problem to solve
        evaluation_function: Function that evaluates algorithm implementations
        model_configs: List of model configurations for ensemble evolution
        api_key: API key for the LLM provider
        api_base: Base URL for the API
        max_iterations: Number of evolution iterations
        population_size: Size of the population
        temperature: Temperature for generation (higher for more creativity)
        max_tokens: Maximum tokens to generate
        language: Programming language for the implementation
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available.")
        return None

    try:
        # Create algorithm discovery-focused configuration
        config = create_comprehensive_openevolve_config(
            content_type=f"code_{language}",  # Use appropriate language
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            archive_size=200,  # Larger archive for diverse algorithmic approaches
            # Use feature dimensions that promote algorithmic diversity
            feature_dimensions=["algorithm_approach", "computational_complexity", "solution_quality"],
            feature_bins=20,  # More bins for fine-grained algorithmic differences
            system_message=f"""You are an expert algorithm designer. Discover novel algorithmic approaches to solve: {problem_description}

            Explore different algorithmic paradigms:
            - Divide and conquer
            - Dynamic programming
            - Greedy approaches
            - Graph algorithms
            - Mathematical optimizations
            - Heuristic methods
            - Probabilistic algorithms
            - Approximation algorithms

            Focus on finding unique approaches that might not be obvious.""",
            temperature=temperature,
            max_tokens=max_tokens,
            # Enable advanced features for algorithm discovery
            num_islands=7,  # More islands for diverse algorithm exploration
            migration_interval=15,
            migration_rate=0.05,  # Lower migration rate to maintain island diversity
            exploration_ratio=0.5,  # Higher exploration for novel approaches
            exploitation_ratio=0.4,
            elite_ratio=0.1,
            cascade_evaluation=True,
            use_llm_feedback=True,
            llm_feedback_weight=0.1,
            evolution_trace_enabled=True,
            evolution_trace_include_prompts=True,
            evolution_trace_include_code=True,  # Important to capture discovered algorithms
            # Algorithm-specific settings
            suggest_simplification_after_chars=3000,  # Allow more complex algorithms
            comprehensive_implementation_min_lines=20,  # Encourage more detailed implementations
        )

        if not config:
            return None

        # Create evaluator that uses the provided evaluation function
        def algorithm_evaluator(program_path):
            with open(program_path, "r") as f:
                algorithm_content = f.read()
            
            # Remove evolution markers
            if "# EVOLVE-BLOCK-START" in algorithm_content:
                start_idx = algorithm_content.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                end_idx = algorithm_content.find("# EVOLVE-BLOCK-END")
                if end_idx != -1:
                    algorithm_content = algorithm_content[start_idx:end_idx].strip()
            
            # Get evaluation from the provided function
            try:
                eval_result = evaluation_function(algorithm_content)
                
                # Add algorithm-specific metrics
                lines = algorithm_content.splitlines()
                eval_result["lines_of_code"] = len(lines)
                eval_result["function_count"] = algorithm_content.count("def ")
                eval_result["algorithm_approach"] = hash(algorithm_content[:100]) % 100  # Simple diversity measure
                eval_result["computational_complexity"] = min(10, len(lines) / 10)  # Rough complexity estimate
                
                # Ensure we have a combined score
                if "combined_score" not in eval_result:
                    # Calculate based on available metrics
                    score_values = [v for v in eval_result.values() if isinstance(v, (int, float))]
                    if score_values:
                        eval_result["combined_score"] = sum(score_values) / len(score_values)
                    else:
                        eval_result["combined_score"] = 0.5  # Default neutral score
                        
                return eval_result
            except Exception as e:
                return {
                    "combined_score": 0.0,
                    "error": str(e),
                    "timestamp": time.time(),
                    "length": len(algorithm_content),
                    "lines_of_code": 0,
                    "function_count": 0,
                    "algorithm_approach": 0,
                    "computational_complexity": 0,
                }

        # Create a temporary file with the problem description as initial code
        with tempfile.NamedTemporaryFile(mode="w", suffix=get_file_suffix_from_content_type(f"code_{language}"), delete=False) as temp_file:
            # Start with a basic algorithm template for the problem
            initial_code = f"""
# EVOLVE-BLOCK-START
{problem_description}

# Current implementation placeholder - evolve this
def solve_problem():
    # Algorithm discovery in progress
    pass

# EVOLVE-BLOCK-END"""
            temp_file.write(initial_code)
            temp_file_path = temp_file.name

        try:
            # Run algorithm discovery evolution using OpenEvolve API
            result = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=algorithm_evaluator,
                config=config,
                iterations=max_iterations,
                output_dir=None,  # Use temporary directory
                cleanup=True,
            )

            # Process results
            if result.best_program and result.best_code:
                # Remove evolution markers from the final result
                best_algorithm = result.best_code
                if "# EVOLVE-BLOCK-START" in best_algorithm:
                    start_idx = best_algorithm.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                    end_idx = best_algorithm.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_algorithm = best_algorithm[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": asdict(result.best_program) if result.best_program else None,
                    "best_score": result.best_score,
                    "best_algorithm": best_algorithm,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                }
            else:
                return {
                    "success": False,
                    "message": "Algorithm discovery evolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running Algorithm Discovery evolution: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def run_symbolic_regression_evolution(
    data_points: List[tuple[float, float]],  # List of (input, output) pairs
    variables: List[str],
    operators: List[str],
    model_configs: List[Dict[str, any]],
    api_key: str,
    api_base: str = None,
    max_iterations: int = 100,
    population_size: int = 1000,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> Optional[Dict[str, Any]]:
    """
    Run symbolic regression evolution to discover mathematical expressions that fit data.
    This discovers equations and mathematical relationships from data points.
    
    Args:
        data_points: List of (input, output) pairs to fit
        variables: List of variable names in the equation
        operators: List of mathematical operators to use (e.g., ['+', '-', '*', '/'])
        model_configs: List of model configurations for evolution
        api_key: API key for the LLM provider
        api_base: Base URL for the API
        max_iterations: Number of evolution iterations
        population_size: Size of the population
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available.")
        return None

    try:
        # Create symbolic regression-focused configuration
        config = create_comprehensive_openevolve_config(
            content_type="code_python",  # We'll generate Python functions
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            archive_size=200,  # Larger archive for diverse mathematical expressions
            feature_dimensions=["accuracy", "complexity", "interpretability"],
            feature_bins=20,
            system_message=f"""You are an expert mathematical modeler. Discover mathematical expressions that fit the following data points: {data_points}
            
            Variables available: {variables}
            Operators allowed: {operators}
            
            The expression should be in the form of a Python function like:
            def model(x):
                return expression
                
            Focus on finding concise, interpretable mathematical relationships that accurately fit the data.""",
            temperature=temperature,
            max_tokens=max_tokens,
            num_islands=5,  # Multiple islands for diverse mathematical approaches
            migration_interval=20,
            migration_rate=0.08,
            exploration_ratio=0.4,  # Balance exploration of different mathematical forms
            exploitation_ratio=0.5,
            elite_ratio=0.1,
            cascade_evaluation=True,
            evolution_trace_enabled=True,
            evolution_trace_include_code=True,
            suggest_simplification_after_chars=2000,  # Allow more complex expressions
            concise_implementation_max_lines=15,
        )

        if not config:
            return None

        # Create evaluator that tests the mathematical expressions
        def symbolic_regression_evaluator(program_path):
            with open(program_path, "r") as f:
                code = f.read()
            
            # Remove evolution markers
            if "# EVOLVE-BLOCK-START" in code:
                start_idx = code.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                end_idx = code.find("# EVOLVE-BLOCK-END")
                if end_idx != -1:
                    code = code[start_idx:end_idx].strip()
            
            accuracy = 0.0
            fitness = 0.0
            complexity = len(code.split())
            
            try:
                # Create a local namespace to execute the function safely
                local_namespace = {}
                exec(code, {"__builtins__": {}}, local_namespace)
                
                if 'model' in local_namespace:
                    model_func = local_namespace['model']
                    
                    # Test the model against the data points
                    total_error = 0
                    valid_points = 0
                    
                    for x, y_actual in data_points:
                        try:
                            y_predicted = model_func(x)
                            error = abs(y_actual - y_predicted)
                            total_error += error
                            valid_points += 1
                        except Exception:
                            # If evaluation fails, assign low fitness
                            pass
                    
                    if valid_points > 0:
                        avg_error = total_error / valid_points
                        # Convert error to accuracy (lower error = higher accuracy)
                        accuracy = 1.0 / (1.0 + avg_error)  # Sigmoid-like transformation
                        fitness = accuracy
                    else:
                        accuracy = 0.0
                        fitness = 0.0
                else:
                    accuracy = 0.0
                    fitness = 0.0
            except Exception:
                # If code execution fails, assign low fitness
                accuracy = 0.0
                fitness = 0.0
            
            return {
                "accuracy": accuracy,
                "complexity": complexity,
                "fitness": fitness,
                "combined_score": fitness,
                "data_points_fitted": len(data_points) if fitness > 0 else 0
            }

        # Create a temporary file with initial symbolic regression code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            initial_code = f"""
# EVOLVE-BLOCK-START
# Symbolic Regression: Find mathematical expression for data points
# Data: {data_points}

def model(x):
    # Attempt to fit the following data points:
    # {data_points}
    # Using variables: {variables}
    # Using operators: {operators}
    return x  # Placeholder - to be evolved

# EVOLVE-BLOCK-END"""
            temp_file.write(initial_code)
            temp_file_path = temp_file.name

        try:
            # Run symbolic regression evolution using OpenEvolve API
            result = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=symbolic_regression_evaluator,
                config=config,
                iterations=max_iterations,
                output_dir=None,  # Use temporary directory
                cleanup=True,
            )

            # Process results
            if result.best_program and result.best_code:
                # Remove evolution markers from the final result
                best_model = result.best_code
                if "# EVOLVE-BLOCK-START" in best_model:
                    start_idx = best_model.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                    end_idx = best_model.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_model = best_model[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": asdict(result.best_program) if result.best_program else None,
                    "best_score": result.best_score,
                    "best_model": best_model,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                }
            else:
                return {
                    "success": False,
                    "message": "Symbolic regression evolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running Symbolic Regression evolution: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def run_neuroevolution(
    problem_description: str,
    fitness_function: Callable[[str], Dict[str, float]],  # Function that evaluates neural network
    model_configs: List[Dict[str, any]],
    api_key: str,
    api_base: str = None,
    max_iterations: int = 100,
    population_size: int = 1000,
    temperature: float = 0.8,  # Higher temperature for more creative architectures
    max_tokens: int = 4096,
) -> Optional[Dict[str, Any]]:
    """
    Run neuroevolution to evolve neural network architectures and parameters.
    This discovers effective neural network structures for specific problems.
    
    Args:
        problem_description: Description of the problem to solve with neural networks
        fitness_function: Function that evaluates neural network implementations
        model_configs: List of model configurations for evolution
        api_key: API key for the LLM provider
        api_base: Base URL for the API
        max_iterations: Number of evolution iterations
        population_size: Size of the population
        temperature: Temperature for generation (higher for more creativity)
        max_tokens: Maximum tokens to generate
    """
    if not OPENEVOLVE_AVAILABLE:
        st.error("OpenEvolve backend is not available.")
        return None

    try:
        # Create neuroevolution-focused configuration
        config = create_comprehensive_openevolve_config(
            content_type="code_python",  # Generate Python neural network code
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            archive_size=150,  # Archive for diverse network architectures
            feature_dimensions=["accuracy", "efficiency", "complexity"],
            feature_bins=15,
            system_message=f"""You are an expert neural network architect. Design neural network architectures for: {problem_description}

            Consider different approaches:
            - Feedforward networks
            - Convolutional networks (if applicable)
            - Recurrent networks (if applicable)
            - Transformer architectures (if applicable)
            - Custom hybrid architectures
            
            The network should be implemented in Python using popular libraries like PyTorch or TensorFlow.
            Focus on creating architectures that balance performance and efficiency.""",
            temperature=temperature,
            max_tokens=max_tokens,
            num_islands=6,  # Multiple islands for diverse architectures
            migration_interval=15,
            migration_rate=0.05,
            exploration_ratio=0.5,  # High exploration for novel architectures
            exploitation_ratio=0.4,
            elite_ratio=0.1,
            cascade_evaluation=True,
            use_llm_feedback=True,
            evolution_trace_enabled=True,
            evolution_trace_include_code=True,
            suggest_simplification_after_chars=3000,
            comprehensive_implementation_min_lines=25,
        )

        if not config:
            return None

        # Create evaluator for neural network code
        def neuroevolution_evaluator(program_path):
            with open(program_path, "r") as f:
                nn_code = f.read()
            
            # Remove evolution markers
            if "# EVOLVE-BLOCK-START" in nn_code:
                start_idx = nn_code.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                end_idx = nn_code.find("# EVOLVE-BLOCK-END")
                if end_idx != -1:
                    nn_code = nn_code[start_idx:end_idx].strip()
            
            # Use the provided fitness function or default
            try:
                fitness_result = fitness_function(nn_code)
                if "combined_score" not in fitness_result:
                    # Calculate based on available metrics
                    score_values = [v for v in fitness_result.values() if isinstance(v, (int, float))]
                    if score_values:
                        fitness_result["combined_score"] = sum(score_values) / len(score_values)
                    else:
                        fitness_result["combined_score"] = 0.3  # Default low score
                return fitness_result
            except Exception as e:
                return {
                    "accuracy": 0.0,
                    "efficiency": 0.0,
                    "complexity": 0.0,
                    "combined_score": 0.0,
                    "error": str(e),
                    "timestamp": time.time(),
                }

        # Create a temporary file with initial neural network code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            initial_code = f"""
# EVOLVE-BLOCK-START
# Neuroevolution: Neural Network for {problem_description}

# Import necessary libraries
import numpy as np
# PyTorch example (can be evolved to TensorFlow, etc.)
# import torch
# import torch.nn as nn
# import torch.optim as optim

# Placeholder neural network implementation - to be evolved
class NeuralNetwork:
    def __init__(self):
        # Initialize network
        pass
    
    def forward(self, x):
        # Forward pass
        return x  # Placeholder

# EVOLVE-BLOCK-END"""
            temp_file.write(initial_code)
            temp_file_path = temp_file.name

        try:
            # Run neuroevolution using OpenEvolve API
            result = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=neuroevolution_evaluator,
                config=config,
                iterations=max_iterations,
                output_dir=None,  # Use temporary directory
                cleanup=True,
            )

            # Process results
            if result.best_program and result.best_code:
                # Remove evolution markers from the final result
                best_network = result.best_code
                if "# EVOLVE-BLOCK-START" in best_network:
                    start_idx = best_network.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                    end_idx = best_network.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_network = best_network[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": asdict(result.best_program) if result.best_program else None,
                    "best_score": result.best_score,
                    "best_network": best_network,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                }
            else:
                return {
                    "success": False,
                    "message": "Neuroevolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        st.error(f"Error running Neuroevolution: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def run_unified_evolution(
    content: str,
    content_type: str,
    evolution_mode: str,
    model_configs: List[Dict[str, any]],
    api_key: str,
    api_base: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_iterations: int = 100,
    population_size: int = 1000,
    system_message: str = "",
    evaluator_system_message: str = "",
    feature_dimensions: Optional[List[str]] = None,
    feature_bins: Optional[int] = None,
    num_islands: int = 5,
    migration_interval: int = 50,
    migration_rate: float = 0.1,
    archive_size: int = 100,
    elite_ratio: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.7,
    checkpoint_interval: int = 100,
    enable_artifacts: bool = True,
    cascade_evaluation: bool = True,
    use_llm_feedback: bool = False,
    llm_feedback_weight: float = 0.1,
    evolution_trace_enabled: bool = False,
    early_stopping_patience: Optional[int] = None,
    convergence_threshold: float = 0.001,
    random_seed: Optional[int] = 42,
    diff_based_evolution: bool = True,
    max_code_length: int = 10000,
    diversity_metric: str = "edit_distance",
    parallel_evaluations: int = 1,
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
    memory_limit_mb: Optional[int] = None,
    cpu_limit: Optional[float] = None,
    db_path: Optional[str] = None,
    in_memory: bool = True,
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
    output_dir: Optional[str] = None,
    load_from_checkpoint: Optional[str] = None,
    custom_evaluator: Optional[Callable] = None,
    custom_requirements: str = "",
    objectives: Optional[List[str]] = None,
    attack_model_config: Optional[Dict[str, any]] = None,
    defense_model_config: Optional[Dict[str, any]] = None,
    evaluation_function: Optional[Callable] = None,
    data_points: Optional[List[tuple[float, float]]] = None,
    variables: Optional[List[str]] = None,
    operators: Optional[List[str]] = None,
    fitness_function: Optional[Callable] = None,
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
) -> Optional[Dict[str, Any]]:
    """
    Unified function to run any type of evolution supported by OpenEvolve.
    This provides a single entry point for all evolution modes.
    
    Args:
        content: The content to evolve
        content_type: Type of content (e.g., 'code_python', 'document_general', etc.)
        evolution_mode: Type of evolution to run
        model_configs: List of model configurations for evolution
        api_key: API key for the LLM provider
        api_base: Base URL for the API
        max_iterations: Number of evolution iterations
        population_size: Size of the population
        system_message: System message for the LLM
        evaluator_system_message: System message for the evaluator
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        objectives: Objectives for multi-objective evolution
        feature_dimensions: Feature dimensions for QD/multi-objective evolution
        custom_requirements: Custom requirements to check for
        custom_evaluator: Custom evaluator function (optional)
        **kwargs: Additional parameters passed to specific evolution functions
        
    Returns:
        Dictionary with evolution results
    """
    if evolution_mode == "quality_diversity":
        return run_quality_diversity_evolution(
            content=content,
            content_type=content_type,
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            feature_dimensions=feature_dimensions,
            custom_requirements=custom_requirements,
            custom_evaluator=custom_evaluator,
        )
    
    elif evolution_mode == "multi_objective":
        return run_multi_objective_evolution(
            content=content,
            content_type=content_type,
            objectives=objectives or ["performance", "readability"],
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_requirements=custom_requirements,
            custom_evaluator=custom_evaluator,
        )
    
    elif evolution_mode == "adversarial":
        attack_model_config = attack_model_config or (model_configs[0] if model_configs else {"name": "gpt-4", "weight": 1.0})
        defense_model_config = defense_model_config or (model_configs[0] if model_configs else {"name": "gpt-4", "weight": 1.0})
        
        return run_adversarial_evolution(
            content=content,
            content_type=content_type,
            attack_model_config=attack_model_config,
            defense_model_config=defense_model_config,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_requirements=custom_requirements,
            custom_evaluator=custom_evaluator,
        )
    
    elif evolution_mode == "prompt_optimization":
        def default_prompt_evaluator(prompt_text):
            """Default prompt evaluation using an LLM to assess prompt quality."""
            try:
                evaluation_prompt = f"""Evaluate the following prompt based on clarity, effectiveness, and conciseness. 
                Provide a score from 0.0 to 1.0 for each category, and a brief justification.
                Return the output as a JSON object with keys 'clarity', 'effectiveness', 'conciseness', and 'justification'.

                Prompt to evaluate:
                ---
                {prompt_text}
                ---
                """
                
                response = _request_openai_compatible_chat(
                    api_key=api_key,
                    base_url=api_base,
                    model=model_configs[0]['name'],
                    messages=[{"role": "user", "content": evaluation_prompt}],
                    temperature=0.2,
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )
                
                if response:
                    eval_result = json.loads(response)
                    clarity = eval_result.get("clarity", 0.5)
                    effectiveness = eval_result.get("effectiveness", 0.5)
                    conciseness = eval_result.get("conciseness", 0.5)
                    
                    return {
                        "clarity": clarity,
                        "effectiveness": effectiveness,
                        "conciseness": conciseness,
                        "combined_score": (clarity + effectiveness + conciseness) / 3.0
                    }
            except Exception as e:
                print(f"Error in default_prompt_evaluator: {e}") 
            
            return {"clarity": 0.5, "effectiveness": 0.5, "conciseness": 0.5, "combined_score": 0.5}
        
        evaluation_func = evaluation_function or default_prompt_evaluator
        
        return run_prompt_evolution(
            initial_prompt=content,
            evaluation_function=evaluation_func,
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    elif evolution_mode == "algorithm_discovery":
        def default_algorithm_evaluator(algorithm_code):
            # Default algorithm evaluation - in a real use case, this would run tests/benchmarks
            return {
                "correctness": 0.7,
                "efficiency": 0.6,
                "readability": 0.8,
                "combined_score": 0.7
            }
        
        evaluation_func = evaluation_function or default_algorithm_evaluator
        
        return run_algorithm_discovery_evolution(
            problem_description=content,
            evaluation_function=evaluation_func,
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            temperature=temperature,
            max_tokens=max_tokens,
            language=content_type.replace("code_", "") if content_type.startswith("code_") else "python",
        )
    
    elif evolution_mode == "symbolic_regression":
        # Extract required parameters for symbolic regression
        data_points = data_points or [(x, x**2) for x in range(10)]  # Default quadratic
        variables = variables or ["x"]
        operators = operators or ["+", "-", "*", "/"]
        
        return run_symbolic_regression_evolution(
            data_points=data_points,
            variables=variables,
            operators=operators,
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    elif evolution_mode == "neuroevolution":
        def default_neural_evaluator(nn_code):
            # Default neural network evaluation
            return {
                "accuracy": 0.5,
                "efficiency": 0.6,
                "complexity": 0.4,
                "combined_score": 0.5
            }
        
        fitness_function = fitness_function or default_neural_evaluator
        
        return run_neuroevolution(
            problem_description=content,
            fitness_function=fitness_function,
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    else:  # Standard evolution
        # Create evaluator based on content type and custom requirements
        if custom_evaluator:
            evaluator = custom_evaluator
        else:
            evaluator = create_specialized_evaluator(content_type, custom_requirements)
        
        # Create comprehensive configuration for standard evolution
        config = create_comprehensive_openevolve_config(
            content_type=content_type,
            model_configs=model_configs,
            api_key=api_key,
            api_base=api_base,
            max_iterations=max_iterations,
            population_size=population_size,
            archive_size=kwargs.get("archive_size", 100),
            feature_dimensions=feature_dimensions or ["complexity", "diversity"],
            feature_bins=kwargs.get("feature_bins", 10),
            system_message=system_message,
            evaluator_system_message=evaluator_system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            num_islands=kwargs.get("num_islands", 3),
            migration_interval=kwargs.get("migration_interval", 25),
            migration_rate=kwargs.get("migration_rate", 0.1),
            exploration_ratio=kwargs.get("exploration_ratio", 0.2),
            exploitation_ratio=kwargs.get("exploitation_ratio", 0.7),
            elite_ratio=kwargs.get("elite_ratio", 0.1),
            cascade_evaluation=kwargs.get("cascade_evaluation", True),
            use_llm_feedback=kwargs.get("use_llm_feedback", False),
            llm_feedback_weight=kwargs.get("llm_feedback_weight", 0.1),
            evolution_trace_enabled=kwargs.get("evolution_trace_enabled", False),
            evolution_trace_include_prompts=True,
            random_seed=kwargs.get("random_seed", 42),
        )

        if not config:
            return None

        # Create a temporary file with the content to evolve
        with tempfile.NamedTemporaryFile(mode="w", suffix=get_file_suffix_from_content_type(content_type), delete=False) as temp_file:
            content_with_markers = f"""# EVOLVE-BLOCK-START
{content}
# EVOLVE-BLOCK-END"""
            temp_file.write(content_with_markers)
            temp_file_path = temp_file.name

        try:
            # Run standard evolution using OpenEvolve API
            result = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=evaluator,
                config=config,
                iterations=max_iterations,
                output_dir=output_dir,
                cleanup=True,
            )

            # Process results
            if result.best_program and result.best_code:
                # Remove evolution markers from the final result
                best_code = result.best_code
                if "# EVOLVE-BLOCK-START" in best_code:
                    start_idx = best_code.find("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
                    end_idx = best_code.find("# EVOLVE-BLOCK-END")
                    if end_idx != -1:
                        best_code = best_code[start_idx:end_idx].strip()

                return {
                    "success": True,
                    "best_program": asdict(result.best_program) if result.best_program else None,
                    "best_score": result.best_score,
                    "best_code": best_code,
                    "metrics": result.metrics,
                    "output_dir": result.output_dir,
                }
            else:
                return {
                    "success": False,
                    "message": "Standard evolution completed with no improvement.",
                }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
