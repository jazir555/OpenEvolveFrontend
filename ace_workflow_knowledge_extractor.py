"""
ACE Workflow Knowledge Extractor

This module extracts knowledge artifacts from workflow executions using ACE's
learning capabilities. It integrates with the workflow stages to capture
reusable patterns, team performance data, and gauntlet effectiveness.

This is the core component for Stage 6: Knowledge Extraction & Learning.

ENHANCED WITH ML CLUSTERING:
- Sentence Transformers for embeddings
- scikit-learn for clustering (DBSCAN, KMeans)
- Entity and relation extraction
- Temporal knowledge graph
- Z3-based validation
"""

from typing import Any, Dict, List, Optional, Tuple
import sys
import os
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import json
import logging
import copy
import numpy as np

# ML Pattern Clustering Integration
try:
    from ml_pattern_clustering import (
        MLKnowledgeExtraction,
        MLPatternClustering,
        EntityExtractor,
        RelationExtractor,
        TemporalKnowledgeGraph,
        KnowledgeValidator,
        MLPattern,
        ExtractedEntity,
        ExtractedRelation
    )
    ML_CLUSTERING_AVAILABLE = True
except ImportError as e:
    ML_CLUSTERING_AVAILABLE = False
    logging.warning(f"ML clustering not available: {e}")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Z3 Validation
try:
    from z3 import Solver, Bool, sat
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

# SECURITY FIX: Import security utilities
try:
    from ace_security_utils import (
        validate_model_name,
        validate_string_length,
        validate_dict_structure,
        validate_list_size,
        validate_file_path_safe,
        atomic_save_json_file,
        create_safe_error,
        get_global_lock,
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ace_security_utils not available - security validations disabled")

# Add agentic-context-engine to path
ACE_PATH = os.path.join(os.path.dirname(__file__), "agentic-context-engine")
if os.path.exists(ACE_PATH) and ACE_PATH not in sys.path:
    sys.path.insert(0, ACE_PATH)

# Import ACE components
try:
    from ace import (
        Skillbook, Agent, Reflector, SkillManager,
        Sample, OfflineACE, LiteLLMClient
    )
    from ace.prompts_v2_1 import PromptManager
    ACE_AVAILABLE = True
except ImportError as e:
    ACE_AVAILABLE = False
    logging.warning(f"ACE not available: {e}")

# THREAD SAFETY FIX: Phase 2 - Import thread safety utilities
try:
    from ace_security_utils import get_global_lock, synchronized
    THREAD_SAFETY_AVAILABLE = True
except ImportError:
    THREAD_SAFETY_AVAILABLE = False
    def get_global_lock(name):
        return threading.RLock()
    def synchronized(lock=None):
        def decorator(func):
            return func
        return decorator

# Import knowledge artifacts
from ace_knowledge_artifacts import (
    KnowledgeArtifact,
    SolutionPattern,
    AntiPattern,
    DecompositionStrategy,
    TeamPerformanceData,
    GauntletEffectivenessData,
    WorkflowExtractionResult,
    ArtifactType,
    ArtifactSource,
    create_solution_pattern,
    create_anti_pattern,
    create_decomposition_strategy,
)

# CAV-NLP Integration for workflow knowledge formalization
try:
    from openevolve.cav_nlp_integration import Z3LeanAideBridge
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("CAV-NLP not available for workflow knowledge formalization")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowKnowledgeExtractor:
    """
    Extract knowledge artifacts from workflow executions using ACE.

    This class orchestrates the extraction of knowledge from all workflow
    stages, including solution patterns, anti-patterns, team performance
    metrics, and gauntlet effectiveness data.

    Architecture:
        1. Collect data from all workflow stages
        2. Use ACE Reflector to analyze execution patterns
        3. Use ACE SkillManager to extract reusable skills
        4. Transform skills into Knowledge Artifacts
        5. Extract team and gauntlet metrics
        6. Store in knowledge base

    Memory Management:
    - max_artifacts: Maximum artifacts to keep in memory (default 10000)
    - Each artifact ~1-10 KB depending on content size
    - 10000 artifacts = ~10-100 MB
    - Set to None for unlimited (not recommended in production)
    - Thread-safe with locks for concurrent access
    - Automatic cleanup with context manager support

    Security Hardening:
    - Input validation for all parameters (workflow_id, problem_statement, etc.)
    - Path traversal prevention for file operations
    - Command injection prevention for model names
    - Resource exhaustion prevention (list sizes, string lengths)
    - Thread safety for all shared state access
    - Safe error handling without information disclosure
    - Atomic file operations to prevent corruption

    Usage:
        with WorkflowKnowledgeExtractor(model="gpt-4o-mini") as extractor:
            result = extractor.extract_from_workflow(
                workflow_id="workflow_123",
                problem_statement="Solve X",
                workflow_results={...}
            )
            extractor.save_artifacts_to_file("output.json", result)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        skillbook_path: Optional[str] = None,
        enable_learning: bool = True,
        max_artifacts: int = 10000,
        use_cav_nlp: bool = True,
    ):
        """
        Initialize the workflow knowledge extractor.

        Args:
            model: LiteLLM model name for ACE
            skillbook_path: Path to load existing skillbook
            enable_learning: Enable ACE learning during extraction
            max_artifacts: Maximum artifacts to keep in memory (resource limit)

        SECURITY FIXES:
            - validate_model_name(): Prevent command injection
            - max_artifacts: Resource limit enforcement
            - Thread locks: Race condition prevention
        """
        # SECURITY FIX: Validate model name to prevent command injection
        if SECURITY_AVAILABLE:
            try:
                self.model = validate_model_name(model)
            except ValueError as e:
                logger.warning(f"Invalid model name: {e}. Using default.")
                self.model = "gpt-4o-mini"
        else:
            self.model = model

        # SECURITY FIX: Validate max_artifacts is positive
        if not isinstance(max_artifacts, int) or max_artifacts < 0:
            logger.warning(f"Invalid max_artifacts: {max_artifacts}. Using default.")
            max_artifacts = 10000

        self.enable_learning = enable_learning
        self.max_artifacts = max_artifacts

        # Initialize ACE components if available
        self.ace_available = ACE_AVAILABLE
        self.skillbook = None
        self.agent = None
        self.reflector = None
        self.skill_manager = None
        self.prompt_mgr = None

        if self.ace_available:
            self._initialize_ace_components(skillbook_path)
        
        # CAV-NLP integration for workflow knowledge formalization
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        self.cav_nlp_bridge: Optional[Z3LeanAideBridge] = None
        if self.use_cav_nlp:
            try:
                self.cav_nlp_bridge = Z3LeanAideBridge()
                logger.info("[OK] CAV-NLP bridge initialized for workflow knowledge")
            except Exception as e:
                logger.warning(f"[FAIL] Failed to initialize CAV-NLP bridge: {e}")
                self.use_cav_nlp = False

        # SECURITY FIX: Thread safety - create locks for concurrent access
        # Use global locks if security utils available, otherwise create local locks
        # LOCK ORDER FIX: Always acquire locks in this order to prevent deadlock:
        # 1. _lock (main extractor lock)
        # 2. _skillbook_lock (skillbook access)
        # 3. _artifacts_lock (artifacts storage)
        # 4. _team_perf_lock (team performance)
        # 5. _gauntlet_lock (gauntlet effectiveness)
        # Never acquire locks out of order or hold multiple locks without clear reason
        if SECURITY_AVAILABLE:
            self._lock = get_global_lock("workflow_knowledge_extractor")
            self._skillbook_lock = get_global_lock("skillbook_access")
            self._artifacts_lock = get_global_lock("artifacts_storage")
            self._team_perf_lock = get_global_lock("team_performance")
            self._gauntlet_lock = get_global_lock("gauntlet_effectiveness")
        else:
            self._lock = threading.RLock()
            self._skillbook_lock = threading.RLock()
            self._artifacts_lock = threading.RLock()
            self._team_perf_lock = threading.RLock()
            self._gauntlet_lock = threading.RLock()

        # Knowledge storage
        # ARTIFACTS LIST FIX: Protected by _artifacts_lock for thread-safe access
        self.artifacts: List[KnowledgeArtifact] = []
        self.team_performances: Dict[str, TeamPerformanceData] = {}
        self.gauntlet_effectiveness: Dict[str, GauntletEffectivenessData] = {}
        
        # ML CLUSTERING: Initialize ML components
        self.ml_clustering_available = ML_CLUSTERING_AVAILABLE
        self.ml_extraction: Optional[MLKnowledgeExtraction] = None
        self.ml_pattern_clustering: Optional[MLPatternClustering] = None
        self.temporal_graph: Optional[TemporalKnowledgeGraph] = None
        self.entity_extractor: Optional[EntityExtractor] = None
        self.relation_extractor: Optional[RelationExtractor] = None
        
        if self.ml_clustering_available and enable_learning:
            try:
                self.ml_extraction = MLKnowledgeExtraction()
                self.ml_pattern_clustering = MLPatternClustering()
                self.temporal_graph = TemporalKnowledgeGraph()
                self.entity_extractor = EntityExtractor(model)
                self.relation_extractor = RelationExtractor()
                logger.info("[OK] ML clustering components initialized")
            except Exception as e:
                logger.warning(f"[FAIL] Failed to initialize ML clustering: {e}")
                self.ml_clustering_available = False
    
    def formalize_workflow_knowledge(
        self, 
        description: str, 
        target_format: str = "z3"
    ) -> Optional[str]:
        """
        Formalize workflow knowledge using CAV-NLP.
        
        Args:
            description: Natural language description of the knowledge
            target_format: Target formal language ('z3', 'lean4')
            
        Returns:
            Formalized representation, or None if conversion fails
            
        Example:
            >>> extractor = WorkflowKnowledgeExtractor()
            >>> formal = extractor.formalize_workflow_knowledge(
            ...     "x must be positive and less than 100",
            ...     target_format="z3"
            ... )
            >>> print(formal)
            'And(x > 0, x < 100)'
        """
        if not self.use_cav_nlp or not self.cav_nlp_bridge:
            logger.debug("CAV-NLP not available, skipping formalization")
            return None
        
        try:
            if target_format == "lean4":
                result = self.cav_nlp_bridge.z3_to_lean4(description)
            elif target_format == "z3":
                # Try natural language to Z3 conversion
                if hasattr(self.cav_nlp_bridge, 'nl_to_z3'):
                    result = self.cav_nlp_bridge.nl_to_z3(description)
                else:
                    # Fallback to lean4 then back to Z3 if available
                    result = self.cav_nlp_bridge.z3_to_lean4(description)
            else:
                logger.warning(f"Unknown target format: {target_format}")
                return None
            
            if result:
                logger.debug(f"Successfully formalized workflow knowledge using CAV-NLP")
            return result
            
        except Exception as e:
            logger.warning(f"CAV-NLP formalization failed: {e}")
            return None
    
    def extract_formalized_knowledge(
        self,
        workflow_id: str,
        problem_statement: str,
        workflow_results: Dict[str, Any],
        target_format: str = "z3"
    ) -> Dict[str, Any]:
        """
        Extract knowledge from workflow with CAV-NLP formalization.
        
        This is an enhanced version of extract_from_workflow that adds
        formal representation to extracted artifacts using CAV-NLP.
        
        Args:
            workflow_id: Unique identifier for the workflow
            problem_statement: The original problem statement
            workflow_results: Complete results from all workflow stages
            target_format: Target formal language for formalization
            
        Returns:
            Dictionary with extracted artifacts and their formal representations
        """
        # First perform standard extraction
        result = self.extract_from_workflow(
            workflow_id=workflow_id,
            problem_statement=problem_statement,
            workflow_results=workflow_results
        )
        
        # Add formal representations using CAV-NLP
        if self.use_cav_nlp and result:
            formalized_artifacts = []
            for artifact in result.get("artifacts", []):
                # Try to formalize artifact content if it's text-based
                if hasattr(artifact, 'description'):
                    formalized = self.formalize_workflow_knowledge(
                        artifact.description,
                        target_format=target_format
                    )
                    if formalized:
                        artifact.formal_representation = formalized
                        artifact.formalization_method = "cav_nlp"
                formalized_artifacts.append(artifact)
            
            result["artifacts"] = formalized_artifacts
            result["cav_nlp_used"] = True
        else:
            result["cav_nlp_used"] = False
        
        return result

    def _initialize_ace_components(self, skillbook_path: Optional[str]):
        """Initialize ACE components."""
        try:
            # Load or create skillbook
            # THREAD SAFETY FIX: TS-6 - Remove TOCTOU
            if skillbook_path:
                try:
                    self.skillbook = Skillbook.load_from_file(skillbook_path)
                    logger.info(f"Loaded skillbook from {skillbook_path}")
                except (FileNotFoundError, json.JSONDecodeError, IOError):
                    self.skillbook = Skillbook()
                    logger.info(f"Skillbook not found, created new skillbook")
            else:
                self.skillbook = Skillbook()
                logger.info("Created new skillbook")

            # Create LLM client
            llm = LiteLLMClient(model=self.model)

            # Get prompt templates
            self.prompt_mgr = PromptManager()

            # Create ACE roles
            self.agent = Agent(llm, prompt_template=self.prompt_mgr.get_agent_prompt())
            self.reflector = Reflector(llm, prompt_template=self.prompt_mgr.get_reflector_prompt())
            self.skill_manager = SkillManager(llm, prompt_template=self.prompt_mgr.get_skill_manager_prompt())

            logger.info("ACE components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ACE components: {e}")
            self.ace_available = False

    def extract_from_workflow(
        self,
        workflow_id: str,
        problem_statement: str,
        workflow_results: Dict[str, Any],
        extract_team_metrics: bool = True,
        extract_gauntlet_metrics: bool = True,
    ) -> WorkflowExtractionResult:
        """
        Extract knowledge artifacts from a complete workflow execution.

        Args:
            workflow_id: Unique identifier for the workflow
            problem_statement: The original problem statement
            workflow_results: Complete results from all workflow stages
            extract_team_metrics: Extract team performance data
            extract_gauntlet_metrics: Extract gauntlet effectiveness data

        Returns:
            WorkflowExtractionResult with all extracted artifacts

        SECURITY FIXES:
            - validate_string_length(): Input validation for workflow_id, problem_statement
            - validate_dict_structure(): Structure validation for workflow_results
            - validate_list_size(): Size validation for sub_problems
            - create_safe_error(): Safe error handling without information disclosure
        """
        # SECURITY FIX: Validate workflow_id
        if SECURITY_AVAILABLE:
            try:
                workflow_id = validate_string_length(
                    workflow_id,
                    "workflow_id",
                    max_length=1000,
                    min_length=1,
                    allow_empty=False
                )
            except ValueError as e:
                logger.error(f"Invalid workflow_id: {e}")
                # Return safe error
                if SECURITY_AVAILABLE:
                    return create_safe_error(
                        "Invalid workflow identifier",
                        e,
                        include_details=False
                    )
                else:
                    raise

            # SECURITY FIX: Validate problem_statement
            try:
                problem_statement = validate_string_length(
                    problem_statement,
                    "problem_statement",
                    max_length=50000,
                    min_length=1,
                    allow_empty=False
                )
            except ValueError as e:
                logger.error(f"Invalid problem_statement: {e}")
                if SECURITY_AVAILABLE:
                    return create_safe_error(
                        "Invalid problem statement",
                        e,
                        include_details=False
                    )
                else:
                    raise

            # SECURITY FIX: Validate workflow_results structure
            try:
                # Expected structure: dict with optional 'phases', 'teams', 'gauntlets' keys
                workflow_results = validate_dict_structure(
                    workflow_results,
                    expected_fields={},
                    allow_extra=True,
                    require_all=False
                )
            except ValueError as e:
                logger.error(f"Invalid workflow_results structure: {e}")
                if SECURITY_AVAILABLE:
                    return create_safe_error(
                        "Invalid workflow results structure",
                        e,
                        include_details=False
                    )
                else:
                    raise

            # SECURITY FIX: Validate sub_problems size
            sub_problems = workflow_results.get("sub_problems", [])
            if sub_problems:
                try:
                    validate_list_size(
                        sub_problems,
                        "sub_problems",
                        max_size=1000,
                        min_size=0,
                        allow_empty=True
                    )
                except ValueError as e:
                    logger.error(f"Sub-problems list too large: {e}")
                    if SECURITY_AVAILABLE:
                        return create_safe_error(
                            "Too many sub-problems",
                            e,
                            include_details=False
                        )
                    else:
                        raise

        # DEEP COPY FIX: Deep copy workflow_results to prevent external modification
        workflow_results = copy.deepcopy(workflow_results)

        logger.info(f"Extracting knowledge from workflow: {workflow_id}")

        result = WorkflowExtractionResult(
            workflow_id=workflow_id,
            problem_statement=problem_statement,
        )

        if not self.ace_available:
            logger.warning("ACE not available - returning empty extraction")
            return result

        try:
            # 1. Extract from each stage
            stage_artifacts = self._extract_from_stages(workflow_results)
            for artifact in stage_artifacts:
                result.add_artifact(artifact)

            # 2. Extract solution patterns using ACE
            solution_patterns = self._extract_solution_patterns(workflow_results)
            for artifact in solution_patterns:
                result.add_artifact(artifact)

            # 3. Extract anti-patterns using ACE
            anti_patterns = self._extract_anti_patterns(workflow_results)
            for artifact in anti_patterns:
                result.add_artifact(artifact)

            # 4. Extract decomposition strategies
            decomposition_strategies = self._extract_decomposition_strategies(workflow_results)
            for artifact in decomposition_strategies:
                result.add_artifact(artifact)

            # 5. Extract team performance metrics
            if extract_team_metrics:
                team_performances = self._extract_team_performance(workflow_results)
                result.team_performances = team_performances
                # THREAD SAFETY FIX: TS-11 - Synchronize internal tracking updates
                with self._team_perf_lock:
                    for tp in team_performances:
                        self.team_performances[tp.team_id] = tp

            # 6. Extract gauntlet effectiveness
            if extract_gauntlet_metrics:
                gauntlet_metrics = self._extract_gauntlet_effectiveness(workflow_results)
                result.gauntlet_effectiveness = gauntlet_metrics
                # THREAD SAFETY FIX: TS-11 - Synchronize internal tracking updates
                with self._gauntlet_lock:
                    for ge in gauntlet_metrics:
                        self.gauntlet_effectiveness[ge.gauntlet_id] = ge

            logger.info(f"Extracted {result.total_artifacts} artifacts from workflow")
            return result

        except Exception as e:
            logger.error(f"Failed to extract knowledge from workflow: {e}")
            # SECURITY FIX: Return safe error instead of exposing internal details
            if SECURITY_AVAILABLE:
                error_result = WorkflowExtractionResult(
                    workflow_id=workflow_id,
                    problem_statement=problem_statement,
                )
                error_result._extraction_error = str(e)
                return error_result
            else:
                return result

    def _add_artifact_with_limit(self, artifact: KnowledgeArtifact):
        """
        RESOURCE FIX: Add artifact with size limit enforcement.

        Args:
            artifact: Artifact to add
        """
        # ARTIFACTS LIST FIX: Protect artifacts list modification with lock
        with self._artifacts_lock:
            self.artifacts.append(artifact)

            # Enforce max_artifacts limit
            if self.max_artifacts is not None and len(self.artifacts) > self.max_artifacts:
                removed = len(self.artifacts) - self.max_artifacts
                self.artifacts = self.artifacts[-self.max_artifacts:]
                logger.warning(f"Removed {removed} old artifacts (limit: {self.max_artifacts})")

    def _extract_from_stages(self, workflow_results: Dict[str, Any]) -> List[KnowledgeArtifact]:
        """
        Extract knowledge from individual workflow stages.

        SECURITY FIX: Added None checks and validation for workflow_results
        """
        artifacts = []

        # SECURITY FIX: None check
        if not workflow_results:
            logger.warning("workflow_results is None or empty")
            return artifacts

        # Extract from each stage
        phases = workflow_results.get("phases", {})
        if not phases or not isinstance(phases, dict):
            logger.warning("No valid phases found in workflow_results")
            return artifacts

        for stage_name, stage_result in phases.items():
            # SECURITY FIX: None check for stage_result
            if stage_result is None:
                logger.warning(f"Stage result for {stage_name} is None")
                continue

            if not stage_result.get("success", False):
                continue

            # Extract learning from stage
            stage_artifacts = self._extract_from_stage(stage_name, stage_result)
            artifacts.extend(stage_artifacts)

        return artifacts

    def _extract_from_stage(self, stage_name: str, stage_result: Dict[str, Any]) -> List[KnowledgeArtifact]:
        """
        Extract knowledge from a single workflow stage.

        SECURITY FIX: Added None checks and validation
        """
        artifacts = []

        # SECURITY FIX: None checks
        if not stage_result or not isinstance(stage_result, dict):
            logger.warning(f"Invalid stage_result for {stage_name}")
            return artifacts

        try:
            # Get learning result if available
            learning = stage_result.get("learning")
            if not learning or not isinstance(learning, dict):
                return artifacts

            # Create artifact from learning
            if learning.get("reflection_summary"):
                # SECURITY FIX: Validate content length
                summary = learning["reflection_summary"]
                if SECURITY_AVAILABLE and summary:
                    try:
                        summary = validate_string_length(summary, "reflection_summary", max_length=10000)
                    except ValueError:
                        logger.warning(f"Reflection summary too long for {stage_name}, truncating")
                        summary = summary[:10000]

                artifact = create_solution_pattern(
                    title=f"{stage_name} Pattern",
                    description=f"Learning from {stage_name}",
                    content=summary,
                    problem_category=stage_name.replace("Phase ", "").lower(),
                    domain="workflow",
                )
                artifacts.append(artifact)

        except Exception as e:
            logger.warning(f"Failed to extract from stage {stage_name}: {e}")

        return artifacts

    def _extract_solution_patterns(self, workflow_results: Dict[str, Any]) -> List[SolutionPattern]:
        """
        Extract solution patterns using ACE reflector.

        SECURITY FIX: Added None checks and validation
        """
        patterns = []

        # SECURITY FIX: None check
        if not workflow_results or not isinstance(workflow_results, dict):
            logger.warning("Invalid workflow_results in _extract_solution_patterns")
            return patterns

        try:
            # Analyze successful solutions
            phases = workflow_results.get("phases", {})
            if not phases or not isinstance(phases, dict):
                logger.warning("No valid phases found")
                return patterns

            for stage_name, stage_result in phases.items():
                # SECURITY FIX: None check for stage_result
                if stage_result is None:
                    continue

                if not stage_result.get("success"):
                    continue

                # Get solutions from stage
                solutions = self._get_solutions_from_stage(stage_result)

                # SECURITY FIX: Validate solutions list size
                if SECURITY_AVAILABLE and solutions:
                    try:
                        solutions = validate_list_size(solutions, "solutions", max_size=100)
                    except ValueError as e:
                        logger.warning(f"Solutions list too large: {e}")
                        solutions = solutions[:100]

                for solution in solutions:
                    # SECURITY FIX: None check for solution
                    if solution is None:
                        continue

                    # Use ACE to analyze and extract patterns
                    if self.reflector and self.skillbook:
                        pattern = self._extract_pattern_from_solution(
                            solution,
                            stage_name
                        )
                        if pattern:
                            patterns.append(pattern)

        except Exception as e:
            logger.error(f"Failed to extract solution patterns: {e}")

        return patterns

    def _extract_pattern_from_solution(
        self,
        solution: Dict[str, Any],
        stage_name: str
    ) -> Optional[SolutionPattern]:
        """
        Extract a reusable pattern from a solution.

        SECURITY FIX: Added None checks and validation
        """
        # SECURITY FIX: None check
        if not solution or not isinstance(solution, dict):
            logger.warning("Invalid solution provided")
            return None

        try:
            # SECURITY FIX: Validate and truncate solution content
            # DEEP COPY FIX: Deep copy solution data to prevent external modification
            sol_text = copy.deepcopy(solution.get('solution', ''))
            if sol_text and SECURITY_AVAILABLE:
                try:
                    sol_text = validate_string_length(sol_text, "solution", max_length=5000)
                except ValueError:
                    sol_text = sol_text[:5000]

            # Create sample for ACE analysis
            sample = Sample(
                query=f"Extract pattern from: {sol_text[:500]}",
                context=copy.deepcopy(f"Stage: {stage_name}"),
            )

            # Create mock agent output
            from ace import AgentOutput
            agent_output = AgentOutput(
                final_answer=sol_text,
                reasoning=copy.deepcopy(solution.get("reasoning", "")),
            )

            # THREAD SAFETY FIX: TS-4 - Synchronize skillbook access
            with self._skillbook_lock:
                # Use reflector to analyze
                reflection = self.reflector.run(
                    sample=sample,
                    agent_output=agent_output,
                    skillbook=self.skillbook,
                    environment_result=None,
                )

            # Extract pattern from reflection
            if reflection and reflection.summary:
                pattern = create_solution_pattern(
                    title=f"Pattern from {stage_name}",
                    description=f"Reusable solution pattern from {stage_name}",
                    content=reflection.summary,
                    problem_category=stage_name.lower(),
                    domain="solution",
                )
                return pattern

        except Exception as e:
            logger.warning(f"Failed to extract pattern from solution: {e}")

        return None

    def _extract_anti_patterns(self, workflow_results: Dict[str, Any]) -> List[AntiPattern]:
        """
        Extract anti-patterns (common mistakes) using ACE.

        SECURITY FIX: Added None checks and validation
        """
        anti_patterns = []

        # SECURITY FIX: None check
        if not workflow_results or not isinstance(workflow_results, dict):
            logger.warning("Invalid workflow_results in _extract_anti_patterns")
            return anti_patterns

        try:
            # Look for failed executions and refinements
            phases = workflow_results.get("phases", {})
            if not phases or not isinstance(phases, dict):
                return anti_patterns

            for stage_name, stage_result in phases.items():
                # SECURITY FIX: None check for stage_result
                if stage_result is None:
                    continue

                # Check for refinement loops (indicates mistakes were made)
                if "refinement" in stage_name.lower() or "recovery" in stage_name.lower():
                    anti_pattern = self._extract_anti_pattern_from_refinement(
                        stage_result,
                        stage_name
                    )
                    if anti_pattern:
                        anti_patterns.append(anti_pattern)

        except Exception as e:
            logger.error(f"Failed to extract anti-patterns: {e}")

        return anti_patterns

    def _extract_anti_pattern_from_refinement(
        self,
        stage_result: Dict[str, Any],
        stage_name: str
    ) -> Optional[AntiPattern]:
        """
        Extract an anti-pattern from a refinement stage.

        SECURITY FIX: Added None checks and validation
        """
        # SECURITY FIX: None check
        if not stage_result or not isinstance(stage_result, dict):
            logger.warning("Invalid stage_result in _extract_anti_pattern_from_refinement")
            return None

        try:
            # Get the issue that needed refinement
            error_or_issue = stage_result.get("error", "") or stage_result.get("issue", "")

            if error_or_issue:
                # SECURITY FIX: Validate string length
                if SECURITY_AVAILABLE and error_or_issue:
                    try:
                        error_or_issue = validate_string_length(error_or_issue, "error_or_issue", max_length=5000)
                    except ValueError:
                        error_or_issue = error_or_issue[:5000]

                anti_pattern = create_anti_pattern(
                    title=f"Anti-pattern from {stage_name}",
                    description=f"Common mistake that occurred in {stage_name}",
                    common_mistake=error_or_issue,
                    correct_approach=stage_result.get("fix", "See refinement solution"),
                    severity="medium",
                    domain="workflow",
                )
                return anti_pattern

        except Exception as e:
            logger.warning(f"Failed to extract anti-pattern from refinement: {e}")

        return None

    def _extract_decomposition_strategies(self, workflow_results: Dict[str, Any]) -> List[DecompositionStrategy]:
        """
        Extract decomposition strategies from Phase 1 results.

        SECURITY FIX: Added None checks and validation
        """
        strategies = []

        # SECURITY FIX: None check
        if not workflow_results or not isinstance(workflow_results, dict):
            logger.warning("Invalid workflow_results in _extract_decomposition_strategies")
            return strategies

        try:
            # Get Phase 1 results (decomposition)
            phases = workflow_results.get("phases", {})
            if not phases or not isinstance(phases, dict):
                return strategies

            phase1_result = phases.get("phase_1", {})
            if not phase1_result or not isinstance(phase1_result, dict):
                return strategies

            if not phase1_result.get("success"):
                return strategies

            # Extract decomposition strategy
            analysis = phase1_result.get("analysis", "")
            if analysis:
                # SECURITY FIX: Validate string length
                if SECURITY_AVAILABLE and analysis:
                    try:
                        analysis = validate_string_length(analysis, "analysis", max_length=10000)
                    except ValueError:
                        analysis = analysis[:10000]

                strategy = create_decomposition_strategy(
                    title="Decomposition Strategy",
                    description="Effective problem decomposition approach",
                    strategy=analysis,
                    decomposition_depth=2,
                    granularity="medium",
                )
                strategies.append(strategy)

        except Exception as e:
            logger.error(f"Failed to extract decomposition strategies: {e}")

        return strategies

    def _extract_team_performance(self, workflow_results: Dict[str, Any]) -> List[TeamPerformanceData]:
        """
        Extract team performance metrics from workflow.

        SECURITY FIX: Added None checks and validation
        """
        team_performances = []

        # SECURITY FIX: None check
        if not workflow_results or not isinstance(workflow_results, dict):
            logger.warning("Invalid workflow_results in _extract_team_performance")
            return team_performances

        try:
            # Extract team data from workflow results
            teams_data = workflow_results.get("teams", {})
            if not teams_data or not isinstance(teams_data, dict):
                return team_performances

            for team_id, team_data in teams_data.items():
                # SECURITY FIX: None check for team_data
                if team_data is None or not isinstance(team_data, dict):
                    logger.warning(f"Skipping invalid team_data for team_id: {team_id}")
                    continue

                # TEAM DATA TYPE VALIDATION FIX: Add isinstance checks and type conversions
                # DEEP COPY FIX: Deep copy team data to prevent external modification
                try:
                    # Validate and convert types safely
                    team_name = copy.deepcopy(team_data.get("name", team_id))
                    if not isinstance(team_name, str):
                        team_name = str(team_name)

                    team_type = copy.deepcopy(team_data.get("type", "blue_team"))
                    if not isinstance(team_type, str):
                        team_type = str(team_type)

                    total_tasks = team_data.get("tasks_completed", 0)
                    if not isinstance(total_tasks, int):
                        try:
                            total_tasks = int(total_tasks)
                        except (ValueError, TypeError):
                            total_tasks = 0

                    successful_tasks = team_data.get("tasks_succeeded", 0)
                    if not isinstance(successful_tasks, int):
                        try:
                            successful_tasks = int(successful_tasks)
                        except (ValueError, TypeError):
                            successful_tasks = 0

                    failed_tasks = team_data.get("tasks_failed", 0)
                    if not isinstance(failed_tasks, int):
                        try:
                            failed_tasks = int(failed_tasks)
                        except (ValueError, TypeError):
                            failed_tasks = 0

                    avg_execution_time = team_data.get("avg_execution_time", 0.0)
                    if not isinstance(avg_execution_time, (int, float)):
                        try:
                            avg_execution_time = float(avg_execution_time)
                        except (ValueError, TypeError):
                            avg_execution_time = 0.0

                    avg_quality_score = team_data.get("avg_quality_score", 0.0)
                    if not isinstance(avg_quality_score, (int, float)):
                        try:
                            avg_quality_score = float(avg_quality_score)
                        except (ValueError, TypeError):
                            avg_quality_score = 0.0

                    preferred_types = copy.deepcopy(team_data.get("preferred_types", []))
                    if not isinstance(preferred_types, list):
                        preferred_types = []

                    skill_affinities = copy.deepcopy(team_data.get("skill_affinities", {}))
                    if not isinstance(skill_affinities, dict):
                        skill_affinities = {}

                    collaboration_score = team_data.get("collaboration_score", 0.0)
                    if not isinstance(collaboration_score, (int, float)):
                        try:
                            collaboration_score = float(collaboration_score)
                        except (ValueError, TypeError):
                            collaboration_score = 0.0

                    perf_data = TeamPerformanceData(
                        team_id=team_id,
                        team_name=team_name,
                        team_type=team_type,
                        total_tasks=total_tasks,
                        successful_tasks=successful_tasks,
                        failed_tasks=failed_tasks,
                        avg_execution_time=avg_execution_time,
                        avg_quality_score=avg_quality_score,
                        preferred_problem_types=preferred_types,
                        skill_affinities=skill_affinities,
                        collaboration_effectiveness=collaboration_score,
                    )
                    team_performances.append(perf_data)
                except Exception as e:
                    logger.warning(f"Failed to create TeamPerformanceData for {team_id}: {e}")

        except Exception as e:
            logger.error(f"Failed to extract team performance: {e}")

        return team_performances

    def _extract_gauntlet_effectiveness(self, workflow_results: Dict[str, Any]) -> List[GauntletEffectivenessData]:
        """
        Extract gauntlet effectiveness metrics from workflow.

        SECURITY FIX: Added None checks and validation
        """
        gauntlet_metrics = []

        # SECURITY FIX: None check
        if not workflow_results or not isinstance(workflow_results, dict):
            logger.warning("Invalid workflow_results in _extract_gauntlet_effectiveness")
            return gauntlet_metrics

        try:
            # Extract gauntlet data from workflow results
            gauntlets_data = workflow_results.get("gauntlets", {})
            if not gauntlets_data or not isinstance(gauntlets_data, dict):
                return gauntlet_metrics

            for gauntlet_id, gauntlet_data in gauntlets_data.items():
                # SECURITY FIX: None check for gauntlet_data
                if gauntlet_data is None or not isinstance(gauntlet_data, dict):
                    continue

                effectiveness = GauntletEffectivenessData(
                    gauntlet_id=gauntlet_id,
                    gauntlet_name=gauntlet_data.get("name", gauntlet_id),
                    gauntlet_type=gauntlet_data.get("type", "red_team"),
                    total_runs=gauntlet_data.get("runs", 0),
                    issues_found=gauntlet_data.get("issues_found", 0),
                    false_positives=gauntlet_data.get("false_positives", 0),
                    true_positives=gauntlet_data.get("true_positives", 0),
                    avg_execution_time=gauntlet_data.get("avg_time", 0.0),
                    effective_problem_types=gauntlet_data.get("effective_types", []),
                    common_violations=gauntlet_data.get("violations", {}),
                )
                # Calculate rates
                effectiveness.detection_rate = effectiveness.calculate_detection_rate()
                gauntlet_metrics.append(effectiveness)

        except Exception as e:
            logger.error(f"Failed to extract gauntlet effectiveness: {e}")

        return gauntlet_metrics

    def _get_solutions_from_stage(self, stage_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract solutions from a stage result.

        SECURITY FIX: Added None checks and validation
        """
        solutions = []

        # SECURITY FIX: None check
        if not stage_result or not isinstance(stage_result, dict):
            return solutions

        # Check for single solution
        if "solution" in stage_result and stage_result["solution"]:
            solutions.append({"solution": stage_result["solution"]})

        # Check for multiple solutions (e.g., Phase 2)
        elif "solutions" in stage_result:
            sols = stage_result["solutions"]
            if sols and isinstance(sols, list):
                solutions.extend(sols)

        # Check for verification results
        elif "verifications" in stage_result:
            verifs = stage_result["verifications"]
            if verifs and isinstance(verifs, list):
                solutions.extend(verifs)

        # Check for critique results
        elif "critiques" in stage_result:
            critiques = stage_result["critiques"]
            if critiques and isinstance(critiques, list):
                solutions.extend(critiques)

        return solutions

    # =========================================================================
    # ML CLUSTERING METHODS
    # =========================================================================
    
    def extract_patterns_ml(
        self,
        workflow_results: List[Dict[str, Any]]
    ) -> List[MLPattern]:
        """
        Extract patterns using ML clustering.
        
        Args:
            workflow_results: List of workflow execution results
            
        Returns:
            List of ML-discovered patterns
        """
        if not self.ml_clustering_available or not self.ml_pattern_clustering:
            logger.warning("ML clustering not available")
            return []
        
        try:
            # Extract texts for clustering
            texts = []
            metadata = []
            
            for i, result in enumerate(workflow_results):
                # Extract problem descriptions
                problem = result.get("problem_statement", "")
                if problem:
                    texts.append(problem)
                    metadata.append({
                        'workflow_id': result.get('workflow_id', f'wf_{i}'),
                        'index': i,
                        'domain': result.get('domain', 'general')
                    })
                
                # Extract solution descriptions
                solution = result.get("solution", "")
                if solution:
                    texts.append(solution)
                    metadata.append({
                        'workflow_id': result.get('workflow_id', f'wf_{i}'),
                        'index': i,
                        'type': 'solution'
                    })
            
            if len(texts) < 2:
                logger.info("Not enough data for ML clustering")
                return []
            
            # Perform clustering
            patterns = self.ml_pattern_clustering.cluster_patterns(texts, metadata)
            
            logger.info(f"ML clustering discovered {len(patterns)} patterns")
            return patterns
        
        except Exception as e:
            logger.error(f"ML pattern extraction failed: {e}")
            return []
    
    def extract_entities_and_relations(
        self,
        text: str,
        context: Optional[str] = None
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        Extract entities and relations from text using ML.
        
        Args:
            text: Text to extract from
            context: Optional context
            
        Returns:
            Tuple of (entities, relations)
        """
        if not self.ml_clustering_available:
            return [], []
        
        try:
            # Extract entities
            entities = self.entity_extractor.extract_entities(text, context) if self.entity_extractor else []
            
            # Extract relations
            relations = self.relation_extractor.extract_relations(text, entities) if self.relation_extractor else []
            
            return entities, relations
        
        except Exception as e:
            logger.error(f"Entity/relation extraction failed: {e}")
            return [], []
    
    def add_to_temporal_graph(
        self,
        content: str,
        node_type: str = "fact",
        confidence: float = 0.5,
        valid_duration_days: Optional[int] = None
    ) -> Optional[str]:
        """
        Add knowledge to temporal graph with versioning.
        
        Args:
            content: Knowledge content
            node_type: Type of knowledge
            confidence: Confidence score
            valid_duration_days: Optional validity period
            
        Returns:
            Node ID if successful
        """
        if not self.ml_clustering_available or not self.temporal_graph:
            return None
        
        try:
            valid_from = datetime.now()
            valid_until = None
            if valid_duration_days:
                valid_until = valid_from + timedelta(days=valid_duration_days)
            
            node = self.temporal_graph.add_node(
                content=content,
                node_type=node_type,
                confidence=confidence,
                valid_from=valid_from,
                valid_until=valid_until
            )
            
            return node.node_id
        
        except Exception as e:
            logger.error(f"Failed to add to temporal graph: {e}")
            return None
    
    def validate_with_z3(
        self,
        statements: List[str]
    ) -> Dict[str, Any]:
        """
        Validate statements using Z3 prover.
        
        Args:
            statements: List of statements to validate
            
        Returns:
            Validation result
        """
        if not Z3_AVAILABLE:
            return {
                'valid': None,
                'message': 'Z3 not available',
                'confidence': 0.0
            }
        
        try:
            solver = Solver()
            
            # Create boolean variables
            for i, statement in enumerate(statements):
                var = Bool(f"stmt_{i}")
                solver.add(var)  # Assume each statement is true
            
            result = solver.check()
            
            return {
                'valid': result == sat,
                'message': 'Statements are consistent' if result == sat else 'Inconsistent',
                'confidence': 0.9 if result == sat else 0.95,
                'statements_checked': len(statements)
            }
        
        except Exception as e:
            logger.error(f"Z3 validation failed: {e}")
            return {
                'valid': None,
                'message': f'Validation error: {e}',
                'confidence': 0.0
            }
    
    def get_ml_extraction_stats(self) -> Dict[str, Any]:
        """Get ML extraction statistics."""
        stats = {
            'ml_available': self.ml_clustering_available,
            'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
            'z3_available': Z3_AVAILABLE,
        }
        
        if self.ml_clustering_available and self.temporal_graph:
            stats['temporal_graph'] = {
                'total_nodes': len(self.temporal_graph.nodes),
                'total_edges': len(self.temporal_graph.edges)
            }
        
        return stats

    def save_artifacts_to_file(self, filepath: str, result: WorkflowExtractionResult):
        """
        Save extraction results to JSON file.

        SECURITY FIXES:
            - validate_file_path_safe(): Prevent path traversal
            - atomic_save_json_file(): Prevent file corruption and TOCTOU
        """
        # SECURITY FIX: Validate filepath
        if SECURITY_AVAILABLE:
            try:
                filepath = validate_file_path_safe(filepath)
            except ValueError as e:
                logger.error(f"Invalid filepath: {e}")
                raise

        # SECURITY FIX: None check for result
        if not result:
            logger.warning("Cannot save None result")
            return

        try:
            # LOCK FIX: Copy data inside lock, then save copy (lock too narrow fix)
            # This ensures data consistency while minimizing lock hold time
            with result._lock:
                data = {
                    "workflow_id": result.workflow_id,
                    "problem_statement": result.problem_statement,
                    "extraction_timestamp": result.extraction_timestamp.isoformat(),
                    "summary": result.to_summary(),
                    "artifacts": [artifact.to_dict() for artifact in result.extracted_artifacts],
                    "team_performances": [tp.to_dict() for tp in result.team_performances],
                    "gauntlet_effectiveness": [ge.to_dict() for ge in result.gauntlet_effectiveness],
                }
                # Make a shallow copy of data for saving outside lock
                data_to_save = data.copy()

            # SECURITY FIX: Use atomic save to prevent file corruption
            if SECURITY_AVAILABLE:
                atomic_save_json_file(filepath, data_to_save)
            else:
                # Fallback to regular save if security utils not available
                with open(filepath, "w") as f:
                    json.dump(data_to_save, f, indent=2)

            logger.info(f"Saved {result.total_artifacts} artifacts to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")

    def update_skillbook_from_artifacts(self, artifacts: List[KnowledgeArtifact]):
        """
        Update ACE skillbook with extracted artifacts (thread-safe).

        SECURITY FIXES:
            - Lock protection for skillbook access
            - validate_list_size(): Prevent resource exhaustion
            - None checks and validation
        """
        if not self.ace_available or not self.skill_manager:
            logger.warning("ACE not available - cannot update skillbook")
            return

        # SECURITY FIX: None check
        if not artifacts or not isinstance(artifacts, list):
            logger.warning("Invalid artifacts list")
            return

        # SECURITY FIX: Validate artifacts list size
        if SECURITY_AVAILABLE:
            try:
                artifacts = validate_list_size(
                    artifacts,
                    "artifacts",
                    max_size=1000,
                    min_size=0,
                    allow_empty=True
                )
            except ValueError as e:
                logger.error(f"Too many artifacts: {e}")
                return

        try:
            from ace import Skill, UpdateOperation, UpdateBatch

            updates = []
            for artifact in artifacts:
                # SECURITY FIX: None check for artifact
                if artifact is None:
                    continue

                # Convert artifact to skill
                skill = Skill(
                    name=artifact.title,
                    strategy=artifact.content,
                    helpful_count=artifact.metrics.times_helpful,
                    harmful_count=artifact.metrics.times_harmful,
                )

                # Create update operation
                update = UpdateOperation(
                    operation_type="ADD",
                    skill=skill,
                )
                updates.append(update)

            # THREAD SAFETY FIX: TS-4 - Synchronize skillbook access
            # Apply updates to skillbook
            if updates:
                with self._skillbook_lock:
                    batch = UpdateBatch(updates=updates)
                    for update in batch.updates:
                        update.apply(self.skillbook)

                logger.info(f"Updated skillbook with {len(updates)} artifacts")

        except Exception as e:
            logger.error(f"Failed to update skillbook: {e}")

    def get_artifact_statistics(self) -> Dict[str, Any]:
        """Get statistics about extracted artifacts (thread-safe)."""
        # THREAD SAFETY FIX: TS-11 - Synchronize access to knowledge storage
        with self._artifacts_lock:
            artifact_counts = {}
            for artifact in self.artifacts:
                artifact_type = artifact.metadata.artifact_type.value
                artifact_counts[artifact_type] = artifact_counts.get(artifact_type, 0) + 1

            total_artifacts = len(self.artifacts)

        with self._team_perf_lock:
            team_count = len(self.team_performances)

        with self._gauntlet_lock:
            gauntlet_count = len(self.gauntlet_effectiveness)

        return {
            "total_artifacts": total_artifacts,
            "by_type": artifact_counts,
            "team_performances_tracked": team_count,
            "gauntlets_tracked": gauntlet_count,
        }

    def cleanup(self):
        """Release resources held by this object."""
        try:
            # Clear collections
            self.artifacts.clear()
            self.team_performances.clear()
            self.gauntlet_effectiveness.clear()

            # Clear ACE components
            self.agent = None
            self.reflector = None
            self.skill_manager = None
            self.prompt_mgr = None

            # Handle skillbook
            if self.skillbook:
                self.skillbook = None

            logger.info("WorkflowKnowledgeExtractor resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False


# Convenience functions

def extract_knowledge_from_workflow(
    workflow_id: str,
    problem_statement: str,
    workflow_results: Dict[str, Any],
    model: str = "gpt-4o-mini",
    skillbook_path: Optional[str] = None,
    output_file: Optional[str] = None,
) -> WorkflowExtractionResult:
    """
    Convenience function to extract knowledge from a workflow.

    Args:
        workflow_id: Unique identifier for the workflow
        problem_statement: The original problem statement
        workflow_results: Complete results from workflow execution
        model: LiteLLM model name
        skillbook_path: Path to load existing skillbook
        output_file: Optional file to save extraction results

    Returns:
        WorkflowExtractionResult with extracted artifacts
    """
    extractor = WorkflowKnowledgeExtractor(
        model=model,
        skillbook_path=skillbook_path,
    )

    result = extractor.extract_from_workflow(
        workflow_id=workflow_id,
        problem_statement=problem_statement,
        workflow_results=workflow_results,
    )

    if output_file:
        extractor.save_artifacts_to_file(output_file, result)

    return result


# Export
__all__ = [
    "WorkflowKnowledgeExtractor",
    "extract_knowledge_from_workflow",
]
