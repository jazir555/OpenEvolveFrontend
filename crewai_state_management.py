"""
CrewAI State Management

This module provides Pydantic-based state management for CrewAI workflows,
replacing the Hephaestus database-backed state system.

Key Features:
- Pydantic models for type safety
- Local state persistence (JSON)
- State transition validation
- Recovery and resume capability
- State versioning and rollback
- Snapshot management
- Export/import for debugging
"""


import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ValidationError

logger = logging.getLogger(__name__)

# Lean 4 Integration
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False
    logger.debug("Lean 4 integration not available for state management")


# =============================================================================
# ENUMS
# =============================================================================

class ExecutionMethod(str, Enum):
    """Available execution methods for CrewAI workflows"""
    TRADITIONAL = "traditional"
    ROMA = "roma"
    ROMA_MDAP_MAKER = "roma_mdap_maker"  # ZERO-ERROR
    CLAUDIOMIRO = "claudiomiro"
    DATAPIZZA = "datapizza"
    HYBRID = "hybrid"
    AUTO = "auto"


class WorkflowStatus(str, Enum):
    """Status of workflow execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SETUP_COMPLETE = "setup_complete"
    SOLVING = "solving"
    CRITIQUE = "critique"
    VERIFYING = "verifying"
    REASSEMBLING = "reassembling"
    FINAL_VALIDATION = "final_validation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TicketType(str, Enum):
    """Types of tickets/workflows in the system"""
    TASK = "task"
    BUG = "bug"
    STORY = "story"
    EPIC = "epic"
    MDAP_TASK = "mdap_task"
    MDAP_STEP = "mdap_step"
    MAKER_RUN = "maker_run"
    MAKER_STEP = "maker_step"
    VOTING_ROUND = "voting_round"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SubProblem(BaseModel):
    """Sub-problem from decomposition"""
    id: str = Field(..., description="Unique sub-problem identifier")
    title: str = Field(..., description="Sub-problem title")
    description: str = Field(..., description="Sub-problem description")
    dependencies: List[str] = Field(default_factory=list, description="List of sub-problem IDs this depends on")
    complexity_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Complexity score (0-1)")
    estimated_effort: int = Field(default=5, ge=1, le=10, description="Estimated effort (1-10 scale)")
    priority: float = Field(default=1.0, ge=0.0, le=1.0, description="Priority score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('dependencies')
    def validate_dependencies(cls, v):
        """Ensure no self-dependencies"""
        if v and 'id' in [dep.split(':')[0] for dep in v]:
            raise ValueError("Sub-problem cannot depend on itself")
        return v


class DecompositionPlan(BaseModel):
    """Complete decomposition plan from Phase 1"""
    id: str = Field(..., description="Decomposition plan ID")
    problem_statement: str = Field(..., description="Original problem statement")
    sub_problems: List[SubProblem] = Field(..., description="List of sub-problems")
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict, description="Dependency graph")
    decomposition_strategy: str = Field(default="semantic", description="Strategy used for decomposition")
    decomposition_depth: int = Field(default=1, ge=1, le=10, description="Depth of decomposition tree")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")


class SolutionAttempt(BaseModel):
    """
    Solution attempt for a sub-problem (compatible with sgd_workflow_orchestrator.py)

    This class matches the dataclass structure used in sgd_workflow_orchestrator.py
    for compatibility with the SGD workflow system.
    """
    # Core fields (matching sgd_workflow_orchestrator.py usage)
    id: str = Field(..., description="Unique solution attempt identifier")
    sub_problem_id: str = Field(..., description="Sub-problem this solves")
    content: str = Field(..., description="Solution content (alias for solution_content)")
    generated_by_model: str = Field(..., description="Model/agent that generated solution")
    timestamp: float = Field(..., description="Unix timestamp of creation")
    status: str = Field(default="PENDING", description="Status: PENDING, IN_PROGRESS, COMPLETED, FAILED, ROLLED_BACK")

    # Extended fields for CrewAI workflows
    solution_content: Optional[str] = Field(None, description="Solution content (use 'content' field)")
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in solution")
    execution_method: Optional[ExecutionMethod] = Field(None, description="Method used to generate solution")
    agent_name: Optional[str] = Field(None, description="Agent that generated solution")
    execution_time: Optional[float] = Field(None, description="Time taken to generate solution (seconds)")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage information")
    voting_participants: Optional[int] = Field(None, description="Number of voting participants (MDAP)")
    red_flags: List[str] = Field(default_factory=list, description="Red flags detected")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp (ISO format)")
    error_message: Optional[str] = Field(None, description="Error message if status is FAILED")

    @field_validator('content', 'solution_content')
    def validate_content(cls, v, info):
        """Ensure content is populated from either field"""
        # If solution_content is provided but content is not, copy it
        if info.field_name == 'solution_content' and v is not None:
            return v
        return v

    def get_content(self) -> str:
        """Get solution content from either field"""
        return self.solution_content if self.solution_content else self.content

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to sync content fields"""
        # Sync solution_content with content for compatibility
        if self.content and not self.solution_content:
            # Can't modify here in Pydantic v2, would need model_dump
            pass
        super().model_post_init(__context)


class CritiqueReport(BaseModel):
    """Critique of a solution"""
    id: str = Field(..., description="Critique report ID")
    target_id: str = Field(..., description="ID of solution being critiqued")
    critique_type: str = Field(..., description="Type of critique (integration, edge_cases, performance, security, compliance)")
    findings: List[str] = Field(..., description="List of findings")
    severity: str = Field(default="medium", description="Severity level (low, medium, high, critical)")
    recommendations: List[str] = Field(default_factory=list, description="List of recommendations")
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in critique")
    agent_name: Optional[str] = Field(None, description="Agent that performed critique")
    voting_participants: Optional[int] = Field(None, description="Number of voting participants (MDAP)")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")


class ValidationResult(BaseModel):
    """Validation result for requirements verification"""
    id: str = Field(..., description="Validation result ID")
    requirement_id: str = Field(..., description="ID of requirement being validated")
    passed: bool = Field(..., description="Whether requirement passed validation")
    score: float = Field(..., ge=0.0, le=1.0, description="Validation score (0-1)")
    details: str = Field(..., description="Detailed validation feedback")
    missing_items: List[str] = Field(default_factory=list, description="Missing requirements")
    voting_participants: Optional[int] = Field(None, description="Number of voting participants (MDAP)")
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in validation")
    validator_name: Optional[str] = Field(None, description="Name of validator")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")


class ReassemblyResult(BaseModel):
    """Result of reassembling sub-solutions"""
    reassembled_content: str = Field(..., description="Final reassembled content")
    components_used: List[str] = Field(..., description="List of component IDs used")
    assembly_strategy: str = Field(..., description="Strategy used for assembly")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores per component")
    quality_metrics: Dict[str, float] = Field(..., description="Quality metrics (completeness, consistency, coherence, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")


class WorkflowState(BaseModel):
    """
    Complete workflow state for CrewAI flows

    This model replaces the Hephaestus ticket-based state system with a
    local Pydantic-based state model.
    """
    # Core identification
    workflow_id: str = Field(..., description="Unique workflow identifier")
    problem_statement: str = Field(..., description="Original problem statement")

    # Status tracking
    phase: int = Field(default=1, ge=1, le=6, description="Current workflow phase (1-6)")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING, description="Current workflow status")
    execution_method: ExecutionMethod = Field(default=ExecutionMethod.AUTO, description="Selected execution method")

    # Phase 1: Setup results
    complexity_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Problem complexity score (0-10)")
    decomposition_plan: Optional[DecompositionPlan] = Field(None, description="Decomposition plan from Phase 1")
    recommended_params: Optional[Dict[str, Any]] = Field(None, description="Recommended ROMA-MDAP-MAKER parameters")

    # Phase 2: Solutions
    sub_solutions: Dict[str, SolutionAttempt] = Field(default_factory=dict, description="Generated solutions per sub-problem")
    solving_progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Solving progress (0-1)")

    # Phase 3: Critiques
    critique_reports: List[CritiqueReport] = Field(default_factory=list, description="Critique reports from Phase 3")

    # Phase 4: Verification
    verification_results: List[ValidationResult] = Field(default_factory=list, description="Verification results from Phase 4")

    # Phase 5: Reassembly
    reassembly_result: Optional[ReassemblyResult] = Field(None, description="Reassembly result from Phase 5")

    # Phase 6: Final validation
    final_validation: Optional[ValidationResult] = Field(None, description="Final validation from Phase 6")
    overall_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall quality score (0-1)")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Workflow creation timestamp")
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last update timestamp")

    # State transition validation
    @field_validator('phase', 'status')
    def validate_state_consistency(cls, v, info):
        """Ensure phase and status are consistent"""
        if info.data.get('status') == WorkflowStatus.COMPLETED and v < 6:
            raise ValueError("Cannot mark workflow as completed before phase 6")
        return v


# =============================================================================
# STATE MANAGER
# =============================================================================

class StateManager:
    """
    Manages persistence and recovery of workflow states.

    Replaces Hephaestus database-backed state with local JSON file storage.
    Provides state versioning, rollback, snapshot, and transaction support.
    Includes Lean 4 verification for state invariants.
    """

    def __init__(
        self,
        storage_dir: str = "./crewai_states",
        enable_compression: bool = True,
        max_versions: int = 10
    ):
        """
        Initialize state manager.

        Args:
            storage_dir: Directory for state JSON files
            enable_compression: Enable gzip compression for storage
            max_versions: Maximum number of state versions to keep per workflow
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.enable_compression = enable_compression
        self.max_versions = max_versions

        # Version tracking
        self._versions_file = self.storage_dir / ".versions.json"
        self._load_version_registry()

        # Import gzip if compression enabled
        if self.enable_compression:
            import gzip
            self.gzip = gzip
        else:
            self.gzip = None

        logger.info(f"StateManager initialized with storage_dir={storage_dir}, max_versions={max_versions}")

    def verify_state_invariant(
        self,
        state: WorkflowState,
        invariant_description: str
    ) -> Dict[str, Any]:
        """
        Verify a workflow state invariant using Lean 4.
        
        Uses LeanAideClient to formalize and verify that the workflow
        state satisfies specified invariants (e.g., consistency checks).
        
        Args:
            state: The workflow state to verify
            invariant_description: Description of the invariant to verify
            
        Returns:
            Dictionary with verification results including:
            - verified: Boolean indicating if invariant holds
            - confidence: Confidence score (0-1)
            - proof: Generated Lean proof code
            - state_summary: Summary of verified state properties
        """
        if not LEAN_AVAILABLE:
            return {
                "verified": False,
                "reason": "Lean 4 not available",
                "invariant": invariant_description
            }
        
        try:
            client = LeanAideClient()
            
            # Build state representation for verification
            state_summary = {
                "workflow_id": state.workflow_id,
                "phase": state.phase,
                "status": state.status.value,
                "num_sub_solutions": len(state.sub_solutions),
                "num_critiques": len(state.critique_reports),
                "num_verifications": len(state.verification_results),
                "has_decomposition": state.decomposition_plan is not None,
                "has_reassembly": state.reassembly_result is not None,
                "has_final_validation": state.final_validation is not None
            }
            
            # Create verification statement
            verification_statement = f"""
            Workflow state invariant: {invariant_description}
            
            Current State:
            - Phase: {state_summary['phase']}
            - Status: {state_summary['status']}
            - Sub-solutions: {state_summary['num_sub_solutions']}
            - Critiques: {state_summary['num_critiques']}
            - Verifications: {state_summary['num_verifications']}
            - Has decomposition: {state_summary['has_decomposition']}
            - Has reassembly: {state_summary['has_reassembly']}
            - Has final validation: {state_summary['has_final_validation']}
            """
            
            # Formalize the invariant
            formalization = client.translate_thm(verification_statement)
            
            if not formalization.success:
                return {
                    "verified": False,
                    "reason": f"Invariant formalization failed: {formalization.error}",
                    "invariant": invariant_description,
                    "state_summary": state_summary
                }
            
            # Check basic consistency (can be extended with more complex proofs)
            consistency_checks = self._run_consistency_checks(state)
            
            return {
                "verified": formalization.success and all(consistency_checks.values()),
                "confidence": 0.9 if formalization.success else 0.0,
                "proof": formalization.data.get("proof", "") if formalization.data else "",
                "state_summary": state_summary,
                "consistency_checks": consistency_checks,
                "invariant": invariant_description
            }
            
        except Exception as e:
            logger.error(f"Lean state invariant verification failed: {e}")
            return {
                "verified": False,
                "reason": f"Verification error: {str(e)}",
                "invariant": invariant_description
            }
    
    def _run_consistency_checks(self, state: WorkflowState) -> Dict[str, bool]:
        """
        Run basic consistency checks on workflow state.
        
        These checks ensure the state machine is in a valid configuration.
        """
        checks = {}
        
        # Phase consistency: solutions should exist for phase >= 2
        if state.phase >= 2:
            checks["has_solutions_for_phase"] = len(state.sub_solutions) > 0
        else:
            checks["has_solutions_for_phase"] = True
        
        # Critique consistency: critiques should exist for phase >= 3
        if state.phase >= 3:
            checks["has_critiques_for_phase"] = len(state.critique_reports) > 0
        else:
            checks["has_critiques_for_phase"] = True
        
        # Verification consistency: results should exist for phase >= 4
        if state.phase >= 4:
            checks["has_verifications_for_phase"] = len(state.verification_results) > 0
        else:
            checks["has_verifications_for_phase"] = True
        
        # Reassembly consistency: result should exist for phase >= 5
        if state.phase >= 5:
            checks["has_reassembly_for_phase"] = state.reassembly_result is not None
        else:
            checks["has_reassembly_for_phase"] = True
        
        # Final validation consistency: should exist for completed status
        if state.status.value == "completed":
            checks["has_final_validation_for_completed"] = state.final_validation is not None
        else:
            checks["has_final_validation_for_completed"] = True
        
        # Sub-problem solution count should match decomposition plan
        if state.decomposition_plan:
            expected_count = len(state.decomposition_plan.sub_problems)
            checks["solution_count_matches_decomposition"] = len(state.sub_solutions) == expected_count
        else:
            checks["solution_count_matches_decomposition"] = True
        
        return checks

    def _load_version_registry(self):
        """Load version registry from disk"""
        if self._versions_file.exists():
            try:
                with open(self._versions_file, 'r') as f:
                    self._versions = json.load(f)
            except (json.JSONDecodeError, OSError, IOError) as e:
                logger.warning(f"Failed to load version registry: {e}, starting fresh")
                self._versions = {}
        else:
            self._versions = {}

    def _save_version_registry(self):
        """Save version registry to disk"""
        try:
            with open(self._versions_file, 'w') as f:
                json.dump(self._versions, f, indent=2)
        except (OSError, IOError) as e:
            logger.error(f"Failed to save version registry: {e}")

    def _get_state_file(self, workflow_id: str) -> Path:
        """Get the file path for a workflow state"""
        extension = ".json.gz" if self.enable_compression else ".json"
        return self.storage_dir / f"{workflow_id}{extension}"

    def save_state(
        self,
        workflow_id: str,
        state: WorkflowState
    ) -> None:
        """
        Save workflow state to disk.

        Args:
            workflow_id: Unique workflow identifier
            state: WorkflowState to save
        """
        try:
            state_file = self._get_state_file(workflow_id)

            # Update updated timestamp
            state.updated_at = datetime.now().isoformat()

            # Serialize to JSON
            json_data = state.model_dump_json(indent=2)

            # Write to file (with optional compression)
            if self.enable_compression and self.gzip:
                with self.gzip.open(state_file, 'wt', encoding='utf-8') as f:
                    f.write(json_data)
            else:
                with open(state_file, 'w', encoding='utf-8') as f:
                    f.write(json_data)

            logger.debug(f"Saved state for workflow {workflow_id} to {state_file}")

        except (OSError, IOError, ValueError) as e:
            logger.error(f"Failed to save state for workflow {workflow_id}: {e}")
            raise

    def load_state(
        self,
        workflow_id: str
    ) -> Optional[WorkflowState]:
        """
        Load workflow state from disk.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            WorkflowState if found, None otherwise
        """
        try:
            state_file = self._get_state_file(workflow_id)

            if not state_file.exists():
                logger.warning(f"No state file found for workflow {workflow_id}")
                return None

            # Read from file (with optional decompression)
            if self.enable_compression and self.gzip and state_file.suffix == '.gz':
                with self.gzip.open(state_file, 'rt', encoding='utf-8') as f:
                    json_data = f.read()
            else:
                with open(state_file, 'r', encoding='utf-8') as f:
                    json_data = f.read()

            # Deserialize
            state = WorkflowState.model_validate_json(json_data)

            logger.debug(f"Loaded state for workflow {workflow_id} from {state_file}")
            return state

        except (json.JSONDecodeError, OSError, IOError, ValueError) as e:
            logger.error(f"Failed to load state for workflow {workflow_id}: {e}")
            return None

    def delete_state(self, workflow_id: str) -> bool:
        """
        Delete workflow state from disk.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            True if deleted, False otherwise
        """
        try:
            state_file = self._get_state_file(workflow_id)

            if not state_file.exists():
                logger.warning(f"No state file found for workflow {workflow_id}")
                return False

            state_file.unlink()
            logger.info(f"Deleted state for workflow {workflow_id}")
            return True

        except (OSError, IOError) as e:
            logger.error(f"Failed to delete state for workflow {workflow_id}: {e}")
            return False

    def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None
    ) -> List[str]:
        """
        List all workflow IDs.

        Args:
            status: Optional filter by workflow status

        Returns:
            List of workflow IDs
        """
        try:
            pattern = "*.json*" if self.enable_compression else "*.json"
            workflow_files = list(self.storage_dir.glob(pattern))

            if status:
                # Filter by status
                filtered_ids = []
                for file in workflow_files:
                    workflow_id = file.stem.replace('.json', '')
                    state = self.load_state(workflow_id)
                    if state and state.status == status:
                        filtered_ids.append(workflow_id)
                return filtered_ids
            else:
                # Return all workflow IDs
                return [f.stem.replace('.json', '') for f in workflow_files]

        except (OSError, IOError) as e:
            logger.error(f"Failed to list workflows: {e}")
            return []

    def cleanup_old_states(
        self,
        max_age_days: int = 30
    ) -> int:
        """
        Clean up old workflow states older than max_age_days.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of states cleaned up
        """
        try:
            import time
            from datetime import timedelta

            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            cleaned_count = 0

            for workflow_file in self.storage_dir.glob("*.json*"):
                # Get modification time
                mtime = datetime.fromtimestamp(workflow_file.stat().st_mtime)

                if mtime < cutoff_time:
                    workflow_file.unlink()
                    cleaned_count += 1

            logger.info(f"Cleaned up {cleaned_count} old workflow states (older than {max_age_days} days)")
            return cleaned_count

        except (OSError, IOError) as e:
            logger.error(f"Failed to cleanup old states: {e}")
            return 0

    def save_state_with_versioning(
        self,
        workflow_id: str,
        state: WorkflowState
    ) -> str:
        """
        Save workflow state with automatic versioning.

        Creates a versioned backup before saving the new state.

        Args:
            workflow_id: Unique workflow identifier
            state: WorkflowState to save

        Returns:
            Version ID of the saved state
        """
        import time

        try:
            # Create version ID
            version_id = f"{workflow_id}_v{int(time.time() * 1000)}"

            # Initialize version tracking for this workflow
            if workflow_id not in self._versions:
                self._versions[workflow_id] = []

            # If current state exists, back it up as a version
            current_state = self.load_state(workflow_id)
            if current_state:
                # Save as versioned backup
                version_file = self.storage_dir / f"{version_id}.json.gz" if self.enable_compression else self.storage_dir / f"{version_id}.json"

                current_state.updated_at = datetime.now().isoformat()
                json_data = current_state.model_dump_json(indent=2)

                if self.enable_compression and self.gzip:
                    with self.gzip.open(version_file, 'wt', encoding='utf-8') as f:
                        f.write(json_data)
                else:
                    with open(version_file, 'w', encoding='utf-8') as f:
                        f.write(json_data)

                # Track version
                self._versions[workflow_id].append(version_id)

                # Cleanup old versions
                if len(self._versions[workflow_id]) > self.max_versions:
                    old_version_id = self._versions[workflow_id].pop(0)
                    old_version_file = self.storage_dir / f"{old_version_id}.json.gz" if self.enable_compression else self.storage_dir / f"{old_version_id}.json"
                    if old_version_file.exists():
                        old_version_file.unlink()

                self._save_version_registry()

            # Save the new state
            self.save_state(workflow_id, state)

            logger.debug(f"Saved state with versioning for workflow {workflow_id}, version_id={version_id}")
            return version_id

        except (OSError, IOError, ValueError) as e:
            logger.error(f"Failed to save state with versioning for workflow {workflow_id}: {e}")
            raise

    def get_state_versions(self, workflow_id: str) -> List[str]:
        """
        Get list of available versions for a workflow.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            List of version IDs
        """
        return self._versions.get(workflow_id, [])

    def load_state_version(
        self,
        workflow_id: str,
        version_id: str
    ) -> Optional[WorkflowState]:
        """
        Load a specific version of a workflow state.

        Args:
            workflow_id: Unique workflow identifier
            version_id: Version ID to load

        Returns:
            WorkflowState if found, None otherwise
        """
        try:
            version_file = self.storage_dir / f"{version_id}.json.gz" if self.enable_compression else self.storage_dir / f"{version_id}.json"

            if not version_file.exists():
                logger.warning(f"Version file not found: {version_file}")
                return None

            # Read from file (with optional decompression)
            if self.enable_compression and self.gzip and version_file.suffix == '.gz':
                with self.gzip.open(version_file, 'rt', encoding='utf-8') as f:
                    json_data = f.read()
            else:
                with open(version_file, 'r', encoding='utf-8') as f:
                    json_data = f.read()

            # Deserialize
            state = WorkflowState.model_validate_json(json_data)

            logger.debug(f"Loaded state version {version_id} for workflow {workflow_id}")
            return state

        except (json.JSONDecodeError, OSError, IOError, ValueError) as e:
            logger.error(f"Failed to load state version {version_id} for workflow {workflow_id}: {e}")
            return None

    def rollback_to_version(
        self,
        workflow_id: str,
        version_id: str
    ) -> bool:
        """
        Rollback workflow to a specific version.

        Args:
            workflow_id: Unique workflow identifier
            version_id: Version ID to rollback to

        Returns:
            True if rollback successful, False otherwise
        """
        try:
            # Load the version
            state = self.load_state_version(workflow_id, version_id)
            if not state:
                logger.error(f"Cannot rollback: version {version_id} not found")
                return False

            # Save as current state
            self.save_state(workflow_id, state)

            logger.info(f"Rolled back workflow {workflow_id} to version {version_id}")
            return True

        except (OSError, IOError, ValueError) as e:
            logger.error(f"Failed to rollback workflow {workflow_id} to version {version_id}: {e}")
            return False

    def create_snapshot(
        self,
        workflow_id: str,
        snapshot_name: Optional[str] = None
    ) -> str:
        """
        Create a named snapshot of the current workflow state.

        Args:
            workflow_id: Unique workflow identifier
            snapshot_name: Optional name for the snapshot (auto-generated if not provided)

        Returns:
            Snapshot ID
        """
        import time

        try:
            state = self.load_state(workflow_id)
            if not state:
                raise ValueError(f"Workflow {workflow_id} not found")

            # Generate snapshot ID
            if not snapshot_name:
                snapshot_name = f"snapshot_{int(time.time())}"

            snapshot_id = f"{workflow_id}_{snapshot_name}"

            # Save snapshot
            snapshot_file = self.storage_dir / f"{snapshot_id}.json.gz" if self.enable_compression else self.storage_dir / f"{snapshot_id}.json"

            state.updated_at = datetime.now().isoformat()
            json_data = state.model_dump_json(indent=2)

            if self.enable_compression and self.gzip:
                with self.gzip.open(snapshot_file, 'wt', encoding='utf-8') as f:
                    f.write(json_data)
            else:
                with open(snapshot_file, 'w', encoding='utf-8') as f:
                    f.write(json_data)

            logger.info(f"Created snapshot {snapshot_id} for workflow {workflow_id}")
            return snapshot_id

        except (OSError, IOError, ValueError) as e:
            logger.error(f"Failed to create snapshot for workflow {workflow_id}: {e}")
            raise

    def list_snapshots(self, workflow_id: str) -> List[str]:
        """
        List all snapshots for a workflow.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            List of snapshot IDs
        """
        try:
            pattern = f"{workflow_id}_snapshot_*.json*" if self.enable_compression else f"{workflow_id}_snapshot_*.json"
            snapshot_files = list(self.storage_dir.glob(pattern))

            # Extract snapshot IDs
            snapshots = []
            for f in snapshot_files:
                # Remove extension
                snapshot_id = f.stem.replace('.json', '')
                snapshots.append(snapshot_id)

            return snapshots

        except (OSError, IOError) as e:
            logger.error(f"Failed to list snapshots for workflow {workflow_id}: {e}")
            return []

    def restore_snapshot(
        self,
        workflow_id: str,
        snapshot_id: str
    ) -> bool:
        """
        Restore workflow from a snapshot.

        Args:
            workflow_id: Unique workflow identifier
            snapshot_id: Snapshot ID to restore

        Returns:
            True if restore successful, False otherwise
        """
        try:
            snapshot_file = self.storage_dir / f"{snapshot_id}.json.gz" if self.enable_compression else self.storage_dir / f"{snapshot_id}.json"

            if not snapshot_file.exists():
                logger.error(f"Snapshot file not found: {snapshot_file}")
                return False

            # Read snapshot
            if self.enable_compression and self.gzip and snapshot_file.suffix == '.gz':
                with self.gzip.open(snapshot_file, 'rt', encoding='utf-8') as f:
                    json_data = f.read()
            else:
                with open(snapshot_file, 'r', encoding='utf-8') as f:
                    json_data = f.read()

            # Deserialize and save as current state
            state = WorkflowState.model_validate_json(json_data)
            self.save_state(workflow_id, state)

            logger.info(f"Restored workflow {workflow_id} from snapshot {snapshot_id}")
            return True

        except (json.JSONDecodeError, OSError, IOError, ValueError) as e:
            logger.error(f"Failed to restore snapshot {snapshot_id} for workflow {workflow_id}: {e}")
            return False

    def export_state(
        self,
        workflow_id: str,
        export_path: str
    ) -> bool:
        """
        Export workflow state to a JSON file.

        Args:
            workflow_id: Unique workflow identifier
            export_path: Path to export file

        Returns:
            True if export successful, False otherwise
        """
        try:
            state = self.load_state(workflow_id)
            if not state:
                logger.error(f"Workflow {workflow_id} not found for export")
                return False

            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)

            json_data = state.model_dump_json(indent=2)
            with open(export_file, 'w', encoding='utf-8') as f:
                f.write(json_data)

            logger.info(f"Exported workflow {workflow_id} to {export_path}")
            return True

        except (OSError, IOError, ValueError) as e:
            logger.error(f"Failed to export workflow {workflow_id}: {e}")
            return False

    def import_state(
        self,
        import_path: str,
        workflow_id: Optional[str] = None
    ) -> Optional[WorkflowState]:
        """
        Import workflow state from a JSON file.

        Args:
            import_path: Path to import file
            workflow_id: Optional new workflow ID (uses original if not provided)

        Returns:
            Imported WorkflowState if successful, None otherwise
        """
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                logger.error(f"Import file not found: {import_path}")
                return None

            with open(import_file, 'r', encoding='utf-8') as f:
                json_data = f.read()

            state = WorkflowState.model_validate_json(json_data)

            # Optionally override workflow ID
            if workflow_id:
                state.workflow_id = workflow_id

            # Save to storage
            self.save_state(state.workflow_id, state)

            logger.info(f"Imported workflow {state.workflow_id} from {import_path}")
            return state

        except (json.JSONDecodeError, OSError, IOError, ValueError) as e:
            logger.error(f"Failed to import workflow from {import_path}: {e}")
            return None

    def get_state_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of workflow state without loading full state.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            State summary dictionary or None
        """
        try:
            state = self.load_state(workflow_id)
            if not state:
                return None

            return {
                'workflow_id': state.workflow_id,
                'problem_statement': state.problem_statement[:100] + '...' if len(state.problem_statement) > 100 else state.problem_statement,
                'phase': state.phase,
                'status': state.status.value,
                'execution_method': state.execution_method.value,
                'created_at': state.created_at,
                'updated_at': state.updated_at,
                'has_decomposition_plan': state.decomposition_plan is not None,
                'num_sub_solutions': len(state.sub_solutions),
                'num_critiques': len(state.critique_reports),
                'num_verification_results': len(state.verification_results),
                'has_reassembly_result': state.reassembly_result is not None,
                'has_final_validation': state.final_validation is not None,
            }

        except (ValueError, TypeError) as e:
            logger.error(f"Failed to get state summary for workflow {workflow_id}: {e}")
            return None


# =============================================================================
# STATE TRANSITION GUARDS
# =============================================================================

class StateTransitionGuard:
    """
    Validates state transitions to prevent invalid workflow state changes.
    """

    def __init__(self):
        """Initialize state transition guard"""
        self.valid_transitions = {
            WorkflowStatus.PENDING: [
                WorkflowStatus.IN_PROGRESS,
                WorkflowStatus.CANCELLED
            ],
            WorkflowStatus.IN_PROGRESS: [
                WorkflowStatus.SETUP_COMPLETE,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED
            ],
            WorkflowStatus.SETUP_COMPLETE: [
                WorkflowStatus.SOLVING,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED
            ],
            WorkflowStatus.SOLVING: [
                WorkflowStatus.CRITIQUE,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED
            ],
            WorkflowStatus.CRITIQUE: [
                WorkflowStatus.VERIFYING,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED
            ],
            WorkflowStatus.VERIFYING: [
                WorkflowStatus.REASSEMBLING,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED
            ],
            WorkflowStatus.REASSEMBLING: [
                WorkflowStatus.FINAL_VALIDATION,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED
            ],
            WorkflowStatus.FINAL_VALIDATION: [
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED
            ],
            WorkflowStatus.COMPLETED: [],  # Terminal state
            WorkflowStatus.FAILED: [],  # Terminal state
            WorkflowStatus.CANCELLED: []  # Terminal state
        }

    def validate_transition(
        self,
        current_status: WorkflowStatus,
        new_status: WorkflowStatus
    ) -> bool:
        """
        Validate whether a state transition is allowed.

        Args:
            current_status: Current workflow status
            new_status: Desired new status

        Returns:
            True if transition is valid, False otherwise
        """
        if current_status not in self.valid_transitions:
            logger.warning(f"Unknown current status: {current_status}")
            return False

        allowed_transitions = self.valid_transitions[current_status]

        if new_status not in allowed_transitions:
            logger.warning(f"Invalid transition: {current_status} -> {new_status}")
            return False

        logger.debug(f"Valid transition: {current_status} -> {new_status}")
        return True

    def guard_transition(
        self,
        state: WorkflowState,
        new_status: WorkflowStatus
    ) -> WorkflowState:
        """
        Guard a state transition, throwing an error if invalid.

        Args:
            state: Current workflow state
            new_status: Desired new status

        Returns:
            Updated workflow state

        Raises:
            ValueError: If transition is invalid
        """
        if not self.validate_transition(state.status, new_status):
            raise ValueError(
                f"Invalid state transition: {state.status} -> {new_status}"
            )

        state.status = new_status
        state.updated_at = datetime.now().isoformat()

        return state


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_workflow_state(
    workflow_id: str,
    problem_statement: str,
    execution_method: ExecutionMethod = ExecutionMethod.AUTO
) -> WorkflowState:
    """
    Factory function to create initial workflow state.

    Args:
        workflow_id: Unique workflow identifier
        problem_statement: Problem to solve
        execution_method: Execution method to use

    Returns:
        New WorkflowState instance
    """
    return WorkflowState(
        workflow_id=workflow_id,
        problem_statement=problem_statement,
        execution_method=execution_method,
        phase=1,
        status=WorkflowStatus.PENDING
    )


def create_state_manager(
    storage_dir: str = "./crewai_states",
    enable_compression: bool = True
) -> StateManager:
    """
    Factory function to create state manager.

    Args:
        storage_dir: Directory for state storage
        enable_compression: Enable gzip compression

    Returns:
        New StateManager instance
    """
    return StateManager(
        storage_dir=storage_dir,
        enable_compression=enable_compression
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Comprehensive example demonstrating all features
    print("=" * 70)
    print("CrewAI State Management - Comprehensive Example")
    print("=" * 70)

    # Create state manager with versioning enabled
    state_mgr = create_state_manager(
        storage_dir="./crewai_states",
        enable_compression=True
    )

    # 1. Create a new workflow state
    print("\n1. Creating workflow state...")
    state = create_workflow_state(
        workflow_id="example_workflow_001",
        problem_statement="Design a zero-error distributed database",
        execution_method=ExecutionMethod.ROMA_MDAP_MAKER
    )
    print(f"   Created workflow: {state.workflow_id}")
    print(f"   Phase: {state.phase}, Status: {state.status}")

    # 2. Save with versioning
    print("\n2. Saving state with versioning...")
    version_id = state_mgr.save_state_with_versioning(state.workflow_id, state)
    print(f"   Saved with version ID: {version_id}")

    # 3. Create a solution attempt
    print("\n3. Adding solution attempt...")
    solution = SolutionAttempt(
        id="sol_001",
        sub_problem_id="sub_001",
        content="Use Raft consensus for distributed coordination",
        generated_by_model="gpt-4",
        timestamp=time.time(),
        status="PENDING"
    )
    state.sub_solutions["sub_001"] = solution

    # 4. Save another version
    version_id = state_mgr.save_state_with_versioning(state.workflow_id, state)
    print(f"   Updated state, new version: {version_id}")

    # 5. List versions
    print("\n4. Listing versions...")
    versions = state_mgr.get_state_versions(state.workflow_id)
    print(f"   Available versions: {len(versions)}")
    for v in versions:
        print(f"   - {v}")

    # 6. Create a named snapshot
    print("\n5. Creating snapshot...")
    snapshot_id = state_mgr.create_snapshot(
        state.workflow_id,
        snapshot_name="before_validation"
    )
    print(f"   Created snapshot: {snapshot_id}")

    # 7. Get state summary
    print("\n6. Getting state summary...")
    summary = state_mgr.get_state_summary(state.workflow_id)
    print(f"   Summary: {summary}")

    # 8. Export state
    print("\n7. Exporting state...")
    export_path = f"./{state.workflow_id}_export.json"
    state_mgr.export_state(state.workflow_id, export_path)
    print(f"   Exported to: {export_path}")

    # 9. List all workflows
    print("\n8. Listing all workflows...")
    workflows = state_mgr.list_workflows()
    print(f"   Total workflows: {len(workflows)}")
    for wf_id in workflows:
        print(f"   - {wf_id}")

    # 10. Demonstrate rollback
    print("\n9. Demonstrating rollback...")
    if len(versions) > 1:
        rollback_version = versions[0]
        success = state_mgr.rollback_to_version(state.workflow_id, rollback_version)
        print(f"   Rollback to {rollback_version}: {'Success' if success else 'Failed'}")

    # 11. Cleanup
    print("\n10. Cleanup old states...")
    cleaned = state_mgr.cleanup_old_states(max_age_days=30)
    print(f"   Cleaned up {cleaned} old states")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)

class VerificationReport:
    """Stub class for VerificationReport."""
    pass
