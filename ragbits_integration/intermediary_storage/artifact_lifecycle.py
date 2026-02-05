"""
Artifact Lifecycle Manager

Manages the lifecycle state transitions of workflow artifacts.
Provides validation and state machine enforcement for artifact status changes.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ArtifactStatus(Enum):
    """Artifact lifecycle states"""
    DRAFT = "draft"
    PENDING = "pending"
    VERIFIED = "verified"
    FINAL = "final"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ArtifactType(Enum):
    """Types of artifacts in the workflow"""
    CONTENT_ANALYSIS = "content_analysis"
    DECOMPOSITION_PLAN = "decomposition_plan"
    SOLUTION_DRAFT = "solution_draft"
    CRITIQUE = "critique"
    VERIFICATION = "verification"
    ASSEMBLED_SOLUTION = "assembled_solution"
    FINAL_VERIFICATION = "final_verification"


class ArtifactLifecycleManager:
    """
    Manages artifact lifecycle state transitions.

    Enforces valid state transitions and tracks artifact history.
    Provides validation for status changes and maintains audit trails.

    Valid transitions:
    - draft -> pending
    - pending -> verified
    - pending -> rejected
    - verified -> final
    - verified -> rejected
    - any -> superseded

    Usage:
        lifecycle = ArtifactLifecycleManager(storage_manager)

        # Create new artifact in draft state
        await lifecycle.create_draft(
            artifact_type="solution_draft",
            content="...",
            metadata={"team": "blue"}
        )

        # Transition to pending
        await lifecycle.transition_to_pending(artifact_id)

        # Transition to verified
        await lifecycle.transition_to_verified(artifact_id)
    """

    # Valid state transitions
    VALID_TRANSITIONS = {
        ArtifactStatus.DRAFT: [ArtifactStatus.PENDING, ArtifactStatus.REJECTED],
        ArtifactStatus.PENDING: [ArtifactStatus.VERIFIED, ArtifactStatus.REJECTED],
        ArtifactStatus.VERIFIED: [ArtifactStatus.FINAL, ArtifactStatus.REJECTED],
        ArtifactStatus.FINAL: [ArtifactStatus.SUPERSEDED],
        ArtifactStatus.REJECTED: [ArtifactStatus.DRAFT],  # Can be reworked
        ArtifactStatus.SUPERSEDED: []  # Terminal state
    }

    def __init__(self, storage_manager):
        """
        Initialize the lifecycle manager.

        Args:
            storage_manager: IntermediaryStorageManager instance
        """
        self.storage = storage_manager
        self._transition_history = {}  # Track all state transitions
        logger.info("ArtifactLifecycleManager initialized")

    async def create_draft(
        self,
        artifact_type: str,
        content: str,
        metadata: Dict[str, Any],
        links_to: Optional[List[str]] = None
    ) -> str:
        """
        Create a new artifact in draft state.

        Args:
            artifact_type: Type of artifact
            content: Artifact content
            metadata: Additional metadata
            links_to: Related artifact IDs

        Returns:
            New artifact ID
        """
        metadata["status"] = ArtifactStatus.DRAFT.value
        metadata["created_at"] = datetime.utcnow().timestamp()

        artifact_id = await self.storage.store_artifact(
            artifact_type=artifact_type,
            content=content,
            metadata=metadata,
            links_to=links_to
        )

        # Record creation in transition history
        self._record_transition(
            artifact_id,
            None,
            ArtifactStatus.DRAFT.value,
            "created"
        )

        logger.info(f"Created draft artifact {artifact_id}")
        return artifact_id

    async def transition_to_pending(
        self,
        artifact_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Transition artifact from draft to pending.

        Used when an artifact is ready for review or processing.

        Args:
            artifact_id: Artifact to transition
            reason: Optional reason for transition

        Returns:
            True if transition successful
        """
        return await self._transition_status(
            artifact_id,
            ArtifactStatus.DRAFT,
            ArtifactStatus.PENDING,
            reason or "submitted for review"
        )

    async def transition_to_verified(
        self,
        artifact_id: str,
        verification_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition artifact from pending to verified.

        Used when an artifact has passed verification/review.

        Args:
            artifact_id: Artifact to transition
            verification_details: Optional verification metadata

        Returns:
            True if transition successful
        """
        return await self._transition_status(
            artifact_id,
            ArtifactStatus.PENDING,
            ArtifactStatus.VERIFIED,
            "verified",
            extra_metadata=verification_details
        )

    async def transition_to_final(
        self,
        artifact_id: str,
        finalization_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition artifact from verified to final.

        Used when an artifact is approved as final.

        Args:
            artifact_id: Artifact to transition
            finalization_details: Optional finalization metadata

        Returns:
            True if transition successful
        """
        return await self._transition_status(
            artifact_id,
            ArtifactStatus.VERIFIED,
            ArtifactStatus.FINAL,
            "approved as final",
            extra_metadata=finalization_details
        )

    async def transition_to_rejected(
        self,
        artifact_id: str,
        rejection_reason: str
    ) -> bool:
        """
        Transition artifact to rejected state.

        Used when an artifact fails review or verification.

        Args:
            artifact_id: Artifact to transition
            rejection_reason: Reason for rejection

        Returns:
            True if transition successful
        """
        current_status = await self._get_current_status(artifact_id)
        if current_status in [ArtifactStatus.DRAFT, ArtifactStatus.PENDING, ArtifactStatus.VERIFIED]:
            return await self._transition_status(
                artifact_id,
                current_status,
                ArtifactStatus.REJECTED,
                f"rejected: {rejection_reason}",
                extra_metadata={"rejection_reason": rejection_reason}
            )
        return False

    async def transition_to_superseded(
        self,
        artifact_id: str,
        superseded_by: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Transition artifact to superseded state.

        Used when a new version replaces this artifact.

        Args:
            artifact_id: Artifact to transition
            superseded_by: ID of the new artifact
            reason: Optional reason for supersession

        Returns:
            True if transition successful
        """
        return await self._transition_status(
            artifact_id,
            None,  # Any state can be superseded
            ArtifactStatus.SUPERSEDED,
            reason or f"superseded by {superseded_by}",
            extra_metadata={"superseded_by": superseded_by}
        )

    async def _transition_status(
        self,
        artifact_id: str,
        from_status: Optional[ArtifactStatus],
        to_status: ArtifactStatus,
        reason: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Perform a status transition with validation.

        Args:
            artifact_id: Artifact to transition
            from_status: Expected current status (None for any)
            to_status: Target status
            reason: Transition reason
            extra_metadata: Additional metadata for the transition

        Returns:
            True if transition successful
        """
        # Get current status
        current_status = await self._get_current_status(artifact_id)

        # Validate from_status if specified
        if from_status and current_status != from_status:
            logger.warning(
                f"Invalid transition for {artifact_id}: "
                f"expected {from_status.value}, found {current_status.value if current_status else 'unknown'}"
            )
            return False

        # Validate transition is allowed
        if current_status and to_status not in self.VALID_TRANSITIONS.get(current_status, []):
            logger.warning(
                f"Invalid transition for {artifact_id}: "
                f"{current_status.value} -> {to_status.value}"
            )
            return False

        # Perform transition
        metadata_update = {
            "status": to_status.value,
            "status_reason": reason,
            "status_updated_at": datetime.utcnow().timestamp()
        }

        if extra_metadata:
            metadata_update.update(extra_metadata)

        success = await self.storage.update_artifact_status(artifact_id, to_status.value)

        if success:
            # Record transition
            self._record_transition(
                artifact_id,
                current_status.value if current_status else None,
                to_status.value,
                reason
            )

            logger.info(
                f"Transitioned {artifact_id}: "
                f"{current_status.value if current_status else 'None'} -> {to_status.value}"
            )

        return success

    async def _get_current_status(
        self,
        artifact_id: str
    ) -> Optional[ArtifactStatus]:
        """Get current status of an artifact"""
        artifact = await self.storage.retrieve_artifact(artifact_id, use_cache=False)

        if not artifact:
            return None

        status_str = artifact.get("metadata", {}).get("status")
        try:
            return ArtifactStatus(status_str)
        except (ValueError, KeyError):
            logger.warning(f"Unknown status for artifact {artifact_id}: {status_str}")
            return None

    def _record_transition(
        self,
        artifact_id: str,
        from_status: Optional[str],
        to_status: str,
        reason: str
    ):
        """Record a state transition in the history"""
        if artifact_id not in self._transition_history:
            self._transition_history[artifact_id] = []

        self._transition_history[artifact_id].append({
            "from": from_status,
            "to": to_status,
            "reason": reason,
            "timestamp": datetime.utcnow().timestamp()
        })

    async def get_transition_history(
        self,
        artifact_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get the complete transition history for an artifact.

        Args:
            artifact_id: Artifact identifier

        Returns:
            List of transition events in chronological order
        """
        if artifact_id not in self._transition_history:
            # Try to reconstruct history from stored versions
            await self._reconstruct_history(artifact_id)

        return self._transition_history.get(artifact_id, [])

    async def _reconstruct_history(self, artifact_id: str):
        """
        Reconstruct transition history from stored artifact versions.

        Called when history is not in memory but artifact exists.
        """
        # Get all versions of the artifact
        # This would require vector store queries for versions
        # For now, create a basic history entry
        artifact = await self.storage.retrieve_artifact(artifact_id, use_cache=False)
        if artifact:
            status = artifact.get("metadata", {}).get("status", "unknown")
            self._record_transition(artifact_id, None, status, "reconstructed")

    async def can_transition_to(
        self,
        artifact_id: str,
        target_status: ArtifactStatus
    ) -> bool:
        """
        Check if an artifact can transition to a target status.

        Args:
            artifact_id: Artifact to check
            target_status: Desired target status

        Returns:
            True if transition is valid
        """
        current_status = await self._get_current_status(artifact_id)

        if not current_status:
            return target_status == ArtifactStatus.DRAFT

        return target_status in self.VALID_TRANSITIONS.get(current_status, [])

    async def get_artifacts_by_status(
        self,
        status: ArtifactStatus,
        artifact_type: Optional[ArtifactType] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all artifacts with a specific status.

        Args:
            status: Status to filter by
            artifact_type: Optional artifact type filter

        Returns:
            List of artifacts with the specified status
        """
        filters = {"status": status.value}
        if artifact_type:
            filters["type"] = artifact_type.value

        return await self.storage._search_artifacts(
            query=f"artifacts with status {status.value}",
            filters=filters,
            top_k=100
        )

    async def get_pending_artifacts(
        self,
        artifact_type: Optional[ArtifactType] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all artifacts pending review/processing.

        Args:
            artifact_type: Optional artifact type filter

        Returns:
            List of pending artifacts
        """
        return await self.get_artifacts_by_status(
            ArtifactStatus.PENDING,
            artifact_type
        )

    async def get_rejected_artifacts(
        self,
        artifact_type: Optional[ArtifactType] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all rejected artifacts (for potential rework).

        Args:
            artifact_type: Optional artifact type filter

        Returns:
            List of rejected artifacts
        """
        return await self.get_artifacts_by_status(
            ArtifactStatus.REJECTED,
            artifact_type
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get lifecycle statistics.

        Returns:
            Statistics dict with counts and metrics
        """
        total_artifacts = len(self._transition_history)
        total_transitions = sum(len(history) for history in self._transition_history.values())

        return {
            "total_artifacts_tracked": total_artifacts,
            "total_transitions": total_transitions,
            "average_transitions_per_artifact": (
                total_transitions / total_artifacts if total_artifacts > 0 else 0
            )
        }
