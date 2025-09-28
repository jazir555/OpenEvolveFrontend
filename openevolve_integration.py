"""
Deep integration with OpenEvolve backend for enhanced code evolution capabilities.
"""

import streamlit as st
import tempfile
import os
import json
import time
from typing import Dict, Any, Optional, List, Callable, Iterator
from dataclasses import asdict
import requests

# Import OpenEvolve modules
try:
    from openevolve.api import (
        run_evolution as openevolve_run_evolution,
        EvolutionResult,
    )
    from openevolve.config import (
        Config,
        LLMModelConfig,
        DatabaseConfig,
        EvaluatorConfig,
    )


    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    st.warning("OpenEvolve backend not available - using API-based evolution only")


class OpenEvolveAPI:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get(self, path: str) -> requests.Response:
        """Makes a GET request to the OpenEvolve backend."""
        return requests.get(f"{self.base_url}{path}", headers=self.headers)

    def start_evolution(
        self, config: Dict, checkpoint_path: Optional[str] = None
    ) -> Optional[str]:
        try:
            payload = {"config": config}
            if checkpoint_path:
                payload["checkpoint_path"] = checkpoint_path
            response = requests.post(
                f"{self.base_url}/evolutions", json=payload, headers=self.headers
            )
            response.raise_for_status()
            return response.json().get("evolution_id")
        except requests.exceptions.RequestException as e:
            st.error(f"Error starting evolution: {e}")
            return None

    def get_checkpoints(self) -> Optional[List[str]]:
        try:
            response = requests.get(
                f"{self.base_url}/checkpoints", headers=self.headers
            )
            response.raise_for_status()
            return response.json().get("checkpoints")
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting checkpoints: {e}")
            return None

    def get_evolution_status(self, evolution_id: str) -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.base_url}/evolutions/{evolution_id}", headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting evolution status: {e}")
            return None

    def get_best_solution(self, evolution_id: str) -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.base_url}/evolutions/{evolution_id}/best", headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting best solution: {e}")
            return None

    def get_evolution_history(self, evolution_id: str) -> Optional[List[Dict]]:
        try:
            response = requests.get(
                f"{self.base_url}/evolutions/{evolution_id}/history",
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting evolution history: {e}")
            return None

    def stream_evolution_logs(self, evolution_id: str) -> Iterator[str]:
        try:
            with requests.get(
                f"{self.base_url}/evolutions/{evolution_id}/logs",
                headers=self.headers,
                stream=True,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    yield chunk.decode("utf-8")
        except requests.exceptions.RequestException as e:
            st.error(f"Error streaming evolution logs: {e}")
            return




    def upload_evaluator(self, evaluator_code: str) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.base_url}/evaluators",
                json={"code": evaluator_code},
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json().get("evaluator_id")
        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading evaluator: {e}")
            return None

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

    def get_custom_prompts(self) -> Optional[Dict[str, str]]:
        try:
            response = requests.get(f"{self.base_url}/prompts", headers=self.headers)
            response.raise_for_status()
            return response.json().get("prompts")
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting custom prompts: {e}")
            return None

    def get_custom_evaluators(self) -> Optional[Dict[str, str]]:
        try:
            response = requests.get(f"{self.base_url}/evaluators", headers=self.headers)
            response.raise_for_status()
            return response.json().get("evaluators")
        except requests.exceptions.RequestException as e:
            st.error(f"Error getting custom evaluators: {e}")
            return None

    def delete_evaluator(self, evaluator_id: str) -> bool:
        try:
            response = requests.delete(
                f"{self.base_url}/evaluators/{evaluator_id}", headers=self.headers
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"Error deleting evaluator: {e}")
            return False

    def save_checkpoint(self, evolution_id: str) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/evolutions/{evolution_id}/checkpoint",
                headers=self.headers,
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"Error saving checkpoint: {e}")
            return False


def create_advanced_openevolve_config(
    model_name: str,
    api_key: str,
    api_base: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_iterations: int = 100,
    population_size: int = 1000,
    num_islands: int = 1,
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
        archive_size: Size of the archive for storing best solutions
        elite_ratio: Ratio of elite individuals to preserve
        exploration_ratio: Ratio for exploration in evolution
        exploitation_ratio: Ratio for exploitation in evolution
        checkpoint_interval: Interval for saving checkpoints
        language: Programming language (optional)
        file_suffix: File suffix for the language
        reasoning_effort: Reasoning effort level (optional)

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

        # Configure LLM model
        llm_config = LLMModelConfig(
            name=model_name,
            api_key=api_key,
            api_base=api_base if api_base else "https://api.openai.com/v1",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

        # Add the model to the config
        config.llm.models = [llm_config]

        # Configure database settings for enhanced evolution
        config.database = DatabaseConfig(
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
            migration_interval=50,
            migration_rate=0.1,
            random_seed=42,
        )

        # Configure evaluator settings
        config.evaluator = EvaluatorConfig(
            timeout=300,
            max_retries=3,
            cascade_evaluation=True,
            cascade_thresholds=[0.5, 0.75, 0.9],
            parallel_evaluations=os.cpu_count() or 4,
            use_llm_feedback=False,
            enable_artifacts=True,
            evaluator_id=evaluator_id,
        )

        return config

    except Exception as e:
        st.error(f"Error creating OpenEvolve configuration: {e}")
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
            return {"score": 0.0, "error": str(e), "timestamp": time.time()}

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
            num_islands=st.session_state.num_islands,
            migration_interval=st.session_state.migration_interval,
            migration_rate=st.session_state.migration_rate,
            archive_size=archive_size,
            elite_ratio=elite_ratio,
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            checkpoint_interval=checkpoint_interval,
            language=get_language_from_content_type(content_type),
            file_suffix=get_file_suffix_from_content_type(content_type),
            feature_dimensions=st.session_state.feature_dimensions,
            feature_bins=st.session_state.feature_bins,
            evaluator_id=st.session_state.get("custom_evaluator_id"),
        )

        if not config:
            return None

        api = OpenEvolveAPI(
            base_url=st.session_state.openevolve_base_url,
            api_key=st.session_state.openevolve_api_key,
        )
        evolution_id = api.start_evolution(config=asdict(config))

        if evolution_id:
            st.session_state.evolution_id = evolution_id
            return {"success": True, "evolution_id": evolution_id}
        else:
            return {"success": False, "message": "Failed to start evolution."}

    except Exception as e:
        st.error(f"Error running advanced code evolution: {e}")
        import traceback

        st.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def create_specialized_evaluator(
    content_type: str,
    custom_requirements: str = "",
    compliance_rules: Optional[List[str]] = None,
) -> Callable:
    """
    Create a specialized evaluator for code content using linters.

    Args:
        content_type: Type of content (e.g., 'code_python', 'code_js', etc.)
        custom_requirements: Custom requirements to check for

    Returns:
        Callable evaluator function
    """

    def code_evaluator(program_path: str) -> Dict[str, Any]:
        """Evaluator for code content using linters."""
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

            if content_type == "code_python":
                try:
                    from pylint.lint import Run
                    from pylint.reporters.text import TextReporter
                    import io

                    reporter = TextReporter(io.StringIO())
                    run = Run([program_path], reporter=reporter, exit=False)
                    linter = run.linter
                    score = linter.stats.global_note
                    errors = reporter.out.getvalue()
                    metrics["linter_score"] = score
                    metrics["linter_errors"] = errors
                except ImportError:
                    pass  # pylint not installed

            elif content_type == "code_js":
                try:
                    import subprocess

                    result = subprocess.run(
                        ["eslint", "-f", "json", program_path],
                        capture_output=True,
                        text=True,
                    )
                    eslint_output = json.loads(result.stdout)
                    errors = eslint_output[0]["messages"]
                    error_count = len(errors)
                    score = 10 - error_count  # a simple scoring metric
                    metrics["linter_score"] = score
                    metrics["linter_errors"] = errors
                except (ImportError, FileNotFoundError):
                    pass  # eslint not installed

            # Calculate a composite score based on various factors
            score_components = []

            # Length score (favor moderate length)
            length_score = min(1.0, len(content) / 1000.0)
            score_components.append(length_score * 0.2)

            # Linter score
            linter_score = metrics.get("linter_score", 0.0) / 10.0  # Normalize to 0-1
            score_components.append(linter_score * 0.5)

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
                score_components.append(req_score * 0.3)
            # Calculate final score as average of components
            final_score = sum(score_components) if score_components else 0.5

            metrics["combined_score"] = final_score
            metrics["score_components"] = {
                "length": length_score,
                "linter": linter_score,
                "requirements": score_components[-1] if custom_requirements else 0.0,
                "compliance": metrics.get("compliance_score", 0.0),
            }

            return metrics

        except Exception as e:
            return {"score": 0.0, "error": str(e), "timestamp": time.time()}

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
        "document_legal": "document",
        "document_medical": "document",
        "document_technical": "document",
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
        "document_legal": ".txt",
        "document_medical": ".txt",
        "document_technical": ".txt",
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
    primary_model: str,
    secondary_model: str,
    api_key: str,
    api_base: str = None,
    primary_weight: float = 1.0,
    secondary_weight: float = 0.5,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> Optional[Config]:
    """
    Create a configuration with multiple models for ensemble evolution.

    Args:
        primary_model: Name of the primary LLM model
        secondary_model: Name of the secondary LLM model
        api_key: API key for the LLM provider
        api_base: Base URL for the API (optional)
        primary_weight: Weight for the primary model
        secondary_weight: Weight for the secondary model
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

        # Configure primary model
        primary_llm_config = LLMModelConfig(
            name=primary_model,
            api_key=api_key,
            api_base=api_base if api_base else "https://api.openai.com/v1",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            weight=primary_weight,
        )

        # Configure secondary model
        secondary_llm_config = LLMModelConfig(
            name=secondary_model,
            api_key=api_key,
            api_base=api_base if api_base else "https://api.openai.com/v1",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            weight=secondary_weight,
        )

        # Add both models to the config
        config.llm.models = [primary_llm_config, secondary_llm_config]

        return config

    except Exception as e:
        st.error(f"Error creating multi-model configuration: {e}")
        return None
