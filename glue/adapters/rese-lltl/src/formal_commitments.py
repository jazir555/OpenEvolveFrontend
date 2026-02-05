"""
RESE LLTL Formal Commitments Handler

Manages Formal Propositional Commitments for DEE -> SCE auditability.

Following CLAUDE.md principles:
- Law of Idempotency: Same input produces same commitment
- Law of Configuration Explicitness: All config via env vars
- Structured Logging: JSON logs with correlation_id
- Law of UTC: All timestamps in UTC

From RESE Technical Manual §2.2:
"DEE -> SCE (Auditability): The DEE's statistical results are converted
into auditable Formal Propositional Commitments by assigning explicit
Confidence Thresholds that the SCE can integrate into its logic graph
for contradiction detection."

Author: RESE Team
Created: 2026-02-04
"""

import os
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Import confidence tracker
try:
    from confidence_tracker import (
        ConfidenceTracker,
        ConfidenceThreshold,
        ConfidenceLevel,
        ConfidenceLogger
    )
except ImportError:
    # For testing
    ConfidenceTracker = None
    ConfidenceLogger = None


# Configure structured logging
class CommitmentLogger:
    """Structured logger for formal commitments."""

    def __init__(self):
        self.logger = logging.getLogger("formal_commitments")
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"component": "formal_commitments", "message": "%(message)s"}'
        ))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def log(self, level: str, msg: str, **kwargs):
        """Log structured message."""
        log_data = {
            "correlation_id": kwargs.get("correlation_id"),
            "operation": kwargs.get("operation"),
            "proposition_id": kwargs.get("proposition_id"),
            "source_hypothesis": kwargs.get("source_hypothesis"),
            "confidence": kwargs.get("confidence"),
            "threshold": kwargs.get("threshold"),
            "message": msg
        }
        log_data = {k: v for k, v in log_data.items() if v is not None}
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_data))


logger = CommitmentLogger()


class CommitmentStatus(Enum):
    """Status of a formal commitment."""
    PENDING = "pending"           # Not yet integrated into SCE
    INTEGRATED = "integrated"     # Successfully integrated
    REJECTED = "rejected"         # Rejected by SCE
    CONTRADICTED = "contradicted" # Found to contradict other commitments


@dataclass
class FormalCommitment:
    """
    A formal propositional commitment for SCE integration.

    Represents a statistical result as a formal logical proposition
    that can be integrated into the SCE logic graph for contradiction
    detection and auditability.

    From RESE Technical Manual §2.2:
    "DEE -> SCE (Auditability): The DEE's statistical results are converted
    into auditable Formal Propositional Commitments by assigning explicit
    Confidence Thresholds that the SCE can integrate into its logic graph
    for contradiction detection."

    Attributes:
        proposition_id: Unique identifier for this proposition
        statement: Formal logical statement
        confidence_threshold: Minimum confidence to accept (0-1)
        statistical_evidence: Statistical evidence (p-value, CI, etc.)
        source_hypothesis: ID of hypothesis
        derivation_method: How this was derived (e.g., "mcts_validation")
        timestamp: UTC ISO-8601 timestamp
        correlation_id: For tracing
        status: Current status of commitment
        lean4_theorem: Lean 4 formalization (future)
        metadata: Additional metadata
    """
    proposition_id: str
    statement: str
    confidence_threshold: float
    statistical_evidence: Dict[str, float]
    source_hypothesis: str
    derivation_method: str
    timestamp: str
    correlation_id: str
    status: CommitmentStatus = CommitmentStatus.PENDING
    lean4_theorem: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_sce_constraint(self) -> Dict[str, Any]:
        """
        Convert to SCE constraint format.

        Returns constraint dict that SCE can integrate into logic graph.

        Returns:
            SCE constraint dictionary
        """
        return {
            "constraint_id": self.proposition_id,
            "formal_statement": self.statement,
            "confidence": self.confidence_threshold,
            "evidence": self.statistical_evidence,
            "type": "statistical_commitment",
            "source_hypothesis": self.source_hypothesis,
            "derivation_method": self.derivation_method
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "proposition_id": self.proposition_id,
            "statement": self.statement,
            "confidence_threshold": self.confidence_threshold,
            "statistical_evidence": self.statistical_evidence,
            "source_hypothesis": self.source_hypothesis,
            "derivation_method": self.derivation_method,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "status": self.status.value,
            "lean4_theorem": self.lean4_theorem,
            "metadata": self.metadata
        }

    def update_status(self, new_status: CommitmentStatus):
        """Update commitment status."""
        self.status = new_status


@dataclass
class ContradictionReport:
    """
    Report of contradictions between commitments.

    Generated when SCE detects contradictions in the logic graph.
    """
    report_id: str
    contradiction_type: str  # "z3", "naive", "manual"
    contradicted_commitments: List[str]  # List of proposition IDs
    reason: str
    detected_at: str  # UTC ISO-8601
    correlation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FormalCommitmentsHandler:
    """
    Manages Formal Propositional Commitments for DEE -> SCE auditability.

    From RESE Technical Manual §2.2:
    "DEE -> SCE (Auditability): The DEE's statistical results are converted
    into auditable Formal Propositional Commitments by assigning explicit
    Confidence Thresholds that the SCE can integrate into its logic graph
    for contradiction detection."

    Features:
    - Convert DEE statistical results to formal commitments
    - Track all commitments for auditability
    - Generate SCE-compatible constraints
    - Manage commitment lifecycle
    - Detect contradictions (with confidence thresholds)
    """

    def __init__(self, confidence_tracker: Optional[ConfidenceTracker] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize formal commitments handler.

        Args:
            confidence_tracker: ConfidenceTracker instance for threshold calculation
            config: Optional configuration dict
        """
        self.config = self._load_config(config)
        self._validate_config()

        # Confidence tracker for threshold calculation
        self.confidence_tracker = confidence_tracker or ConfidenceTracker()

        # Commitment storage
        self.commitments: Dict[str, FormalCommitment] = {}

        # Contradiction reports
        self.contradiction_reports: List[ContradictionReport] = []

        logger.log("INFO", "Formal commitments handler initialized",
                  operation="initialize",
                  significance_level=self.config["significance_level"])

    def _load_config(self, override_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        CLAUDE.md: Law of Configuration Explicitness.
        """
        config = {
            "significance_level": float(os.getenv("LLTL_SIGNIFICANCE_LEVEL", "0.05")),
            "enable_history": os.getenv("LLTL_ENABLE_COMMITMENT_HISTORY", "true").lower() == "true",
            "max_commitments": int(os.getenv("LLTL_MAX_COMMITMENTS", "10000")),
            "enable_contradiction_detection": os.getenv("LLTL_ENABLE_CONTRADICTION_DETECTION", "true").lower() == "true"
        }

        # Apply overrides
        if override_config:
            config.update(override_config)

        return config

    def _validate_config(self):
        """Validate configuration."""
        errors = []

        # Validate significance level
        if not (0 < self.config["significance_level"] < 1):
            errors.append("SIGNIFICANCE_LEVEL must be between 0 and 1")

        # Validate max commitments
        if self.config["max_commitments"] <= 0:
            errors.append("MAX_COMMITMENTS must be positive")

        if errors:
            error_msg = f"Configuration validation failed: {', '.join(errors)}"
            logger.log("ERROR", error_msg, operation="validate_config")
            raise RuntimeError(error_msg)

    def create_commitment(
        self,
        statistical_result: Dict[str, Any],
        source_hypothesis: str,
        derivation_method: str,
        correlation_id: Optional[str] = None
    ) -> Tuple[Optional[FormalCommitment], Optional[str]]:
        """
        Create a formal commitment from DEE statistical result.

        This is the core DEE -> SCE conversion method.

        Args:
            statistical_result: Dict with:
                - hypothesis_statement: str
                - confidence: float (0-1)
                - p_value: float
                - confidence_interval: Tuple[float, float] or List[float]
                - expected_value: float
                - (optional) evidence: List[Dict]

            source_hypothesis: ID of hypothesis
            derivation_method: How result was derived (e.g., "mcts_validation")
            correlation_id: For tracing

        Returns:
            Tuple of (FormalCommitment, error_message)
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        try:
            # Validate required fields
            required_fields = ['hypothesis_statement', 'confidence']
            missing_fields = [f for f in required_fields if f not in statistical_result]
            if missing_fields:
                return None, f"Missing required fields: {missing_fields}"

            # Extract statistical evidence
            confidence = float(statistical_result.get('confidence', 0.0))
            p_value = float(statistical_result.get('p_value', 1.0))
            confidence_interval = statistical_result.get('confidence_interval', (0.0, 1.0))

            # Handle confidence_interval as tuple or list
            if isinstance(confidence_interval, (list, tuple)) and len(confidence_interval) >= 2:
                ci_lower, ci_upper = float(confidence_interval[0]), float(confidence_interval[1])
            else:
                ci_lower, ci_upper = 0.0, 1.0

            # Calculate confidence threshold
            threshold = self.confidence_tracker.calculate_threshold(
                confidence=confidence,
                derivation_method=derivation_method,
                correlation_id=correlation_id
            )

            # Construct formal logical statement
            hypothesis_stmt = str(statistical_result['hypothesis_statement'])
            formal_statement = self._construct_formal_statement(
                hypothesis=hypothesis_stmt,
                confidence=confidence,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper)
            )

            # Create formal commitment
            commitment = FormalCommitment(
                proposition_id=str(uuid.uuid4()),
                statement=formal_statement,
                confidence_threshold=threshold.threshold,
                statistical_evidence={
                    'confidence': confidence,
                    'p_value': p_value,
                    'confidence_interval_lower': ci_lower,
                    'confidence_interval_upper': ci_upper,
                    'expected_value': float(statistical_result.get('expected_value', 0.0))
                },
                source_hypothesis=source_hypothesis,
                derivation_method=derivation_method,
                timestamp=datetime.now(timezone.utc).isoformat(),
                correlation_id=correlation_id,
                metadata={
                    'confidence_level': threshold.level.value,
                    'significance_level': threshold.significance_level
                }
            )

            # Store commitment
            self.commitments[commitment.proposition_id] = commitment

            # Track in confidence history
            if self.config["enable_history"]:
                try:
                    self.confidence_tracker.track_threshold(
                        proposition_id=commitment.proposition_id,
                        input_confidence=confidence,
                        threshold=threshold,
                        correlation_id=correlation_id
                    )
                except RuntimeError:
                    # History tracking disabled
                    pass

            logger.log("INFO", f"Created formal commitment: {commitment.proposition_id}",
                      correlation_id=correlation_id,
                      operation="create_commitment",
                      proposition_id=commitment.proposition_id,
                      source_hypothesis=source_hypothesis,
                      confidence=confidence,
                      threshold=threshold.threshold)

            return commitment, None

        except Exception as e:
            error_msg = f"Failed to create commitment: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="create_commitment")
            return None, error_msg

    def _construct_formal_statement(
        self,
        hypothesis: str,
        confidence: float,
        p_value: float,
        confidence_interval: Tuple[float, float]
    ) -> str:
        """
        Construct formal logical statement from statistical evidence.

        Format: "H ∧ (confidence ≥ T) ∧ (p ≤ α) -> Accept(H)"

        Args:
            hypothesis: Hypothesis statement
            confidence: Statistical confidence
            p_value: Statistical significance
            confidence_interval: (lower, upper) bounds

        Returns:
            Formal logical statement string
        """
        # Significance level
        α = self.config["significance_level"]

        # Truncate hypothesis for readability
        hypothesis_short = hypothesis[:50] + "..." if len(hypothesis) > 50 else hypothesis

        # Construct statement (using ASCII for Windows compatibility)
        statement = (
            f"({hypothesis}) AND "
            f"(confidence >= {confidence:.3f}) AND "
            f"(p_value <= {α:.3f}) AND "
            f"(CI in [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]) "
            f"IMPLIES Accept({hypothesis_short})"
        )

        return statement

    def get_commitment(self, proposition_id: str) -> Optional[FormalCommitment]:
        """
        Get a specific commitment by ID.

        Args:
            proposition_id: ID of commitment to retrieve

        Returns:
            FormalCommitment or None if not found
        """
        return self.commitments.get(proposition_id)

    def get_all_commitments(
        self,
        status: Optional[CommitmentStatus] = None,
        limit: int = 100
    ) -> List[FormalCommitment]:
        """
        Get all commitments, optionally filtered by status.

        Args:
            status: Filter by status (None = all)
            limit: Maximum number of commitments to return

        Returns:
            List of FormalCommitment objects
        """
        commitments = list(self.commitments.values())

        # Filter by status
        if status:
            commitments = [c for c in commitments if c.status == status]

        # Sort by timestamp (most recent first) and limit
        commitments = sorted(commitments, key=lambda c: c.timestamp, reverse=True)[:limit]

        return commitments

    def get_commitments_by_hypothesis(
        self,
        hypothesis_id: str,
        limit: int = 100
    ) -> List[FormalCommitment]:
        """
        Get all commitments for a specific hypothesis.

        Args:
            hypothesis_id: Hypothesis ID
            limit: Maximum number of commitments to return

        Returns:
            List of FormalCommitment objects
        """
        commitments = [
            c for c in self.commitments.values()
            if c.source_hypothesis == hypothesis_id
        ]

        # Sort by timestamp (most recent first) and limit
        commitments = sorted(commitments, key=lambda c: c.timestamp, reverse=True)[:limit]

        return commitments

    def update_commitment_status(
        self,
        proposition_id: str,
        new_status: CommitmentStatus,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Update the status of a commitment.

        Args:
            proposition_id: ID of commitment
            new_status: New status
            correlation_id: For tracing

        Returns:
            True if successful, False if commitment not found
        """
        commitment = self.commitments.get(proposition_id)
        if not commitment:
            logger.log("WARNING", f"Commitment not found: {proposition_id}",
                      correlation_id=correlation_id,
                      operation="update_commitment_status")
            return False

        old_status = commitment.status
        commitment.update_status(new_status)

        logger.log("INFO", f"Updated commitment status: {proposition_id} ({old_status.value} -> {new_status.value})",
                  correlation_id=correlation_id,
                  operation="update_commitment_status",
                  proposition_id=proposition_id,
                  old_status=old_status.value,
                  new_status=new_status.value)

        return True

    def detect_contradictions(
        self,
        correlation_id: Optional[str] = None
    ) -> List[ContradictionReport]:
        """
        Detect contradictions between commitments.

        Uses confidence thresholds in contradiction detection:
        - Low confidence commitments are less likely to trigger contradictions
        - High confidence commitments are more strictly checked

        Args:
            correlation_id: For tracing

        Returns:
            List of ContradictionReport objects
        """
        if not self.config["enable_contradiction_detection"]:
            logger.log("DEBUG", "Contradiction detection disabled",
                      correlation_id=correlation_id,
                      operation="detect_contradictions")
            return []

        correlation_id = correlation_id or str(uuid.uuid4())
        contradictions = []

        # Get all integrated commitments
        integrated = [c for c in self.commitments.values() if c.status == CommitmentStatus.INTEGRATED]

        # Pairwise contradiction detection
        seen_pairs: Set[Tuple[str, str]] = set()

        for i, c1 in enumerate(integrated):
            for c2 in integrated[i+1:]:
                if self._check_contradiction(c1, c2):
                    pair_id = tuple(sorted([c1.proposition_id, c2.proposition_id]))
                    if pair_id not in seen_pairs:
                        seen_pairs.add(pair_id)

                        # Create contradiction report
                        report = ContradictionReport(
                            report_id=str(uuid.uuid4()),
                            contradiction_type="confidence_aware",
                            contradicted_commitments=[c1.proposition_id, c2.proposition_id],
                            reason=f"Contradiction detected between commitments with thresholds {c1.confidence_threshold:.2f} and {c2.confidence_threshold:.2f}",
                            detected_at=datetime.now(timezone.utc).isoformat(),
                            correlation_id=correlation_id,
                            metadata={
                                "commitment_1_threshold": c1.confidence_threshold,
                                "commitment_2_threshold": c2.confidence_threshold,
                                "commitment_1_statement": c1.statement,
                                "commitment_2_statement": c2.statement
                            }
                        )

                        contradictions.append(report)

                        # Mark as contradicted
                        c1.update_status(CommitmentStatus.CONTRADICTED)
                        c2.update_status(CommitmentStatus.CONTRADICTED)

                        logger.log("WARNING", f"Contradiction detected between {c1.proposition_id} and {c2.proposition_id}",
                                  correlation_id=correlation_id,
                                  operation="detect_contradictions",
                                  report_id=report.report_id)

        # Store reports
        self.contradiction_reports.extend(contradictions)

        return contradictions

    def _check_contradiction(self, c1: FormalCommitment, c2: FormalCommitment) -> bool:
        """
        Check if two commitments contradict each other.

        Uses confidence thresholds in contradiction detection:
        - If both have high confidence (>0.8), strict checking
        - If one has low confidence (<0.6), lenient checking

        Args:
            c1: First commitment
            c2: Second commitment

        Returns:
            True if contradictions detected
        """
        # Extract statements
        stmt1 = c1.statement.lower()
        stmt2 = c2.statement.lower()

        # Check for direct negation
        if f"not {stmt2}" in stmt1 or f"not {stmt1}" in stmt2:
            # High confidence contradictions are more serious
            if c1.confidence_threshold > 0.8 and c2.confidence_threshold > 0.8:
                return True

        # Check for opposite inequalities
        if ('<' in stmt1 and '>' in stmt2) or ('>' in stmt1 and '<' in stmt2):
            # Extract variables and check if they're the same variable
            var1 = self._extract_variable_from_statement(stmt1)
            var2 = self._extract_variable_from_statement(stmt2)

            if var1 and var2 and var1 == var2:
                # Same variable with opposite inequalities
                # Only flag as contradiction if both have high confidence
                if c1.confidence_threshold > 0.75 and c2.confidence_threshold > 0.75:
                    return True

        # Check for opposite thresholds
        # One is very confident, one is not - might indicate contradiction
        if abs(c1.confidence_threshold - c2.confidence_threshold) > 0.4:
            # But only if both reference the same hypothesis
            if c1.source_hypothesis == c2.source_hypothesis:
                return True

        return False

    def _extract_variable_from_statement(self, statement: str) -> Optional[str]:
        """
        Extract variable name from formal statement.

        Simple heuristic extraction.

        Args:
            statement: Formal statement string

        Returns:
            Variable name or None
        """
        # Look for patterns like "(x > 5)" or "(confidence ≥ 0.9)"
        match = re.search(r'\(([a-zA-Z_]\w*)\s*[<>=]', statement)
        if match:
            return match.group(1)

        return None

    def get_contradiction_reports(
        self,
        limit: int = 100
    ) -> List[ContradictionReport]:
        """
        Get all contradiction reports.

        Args:
            limit: Maximum number of reports to return

        Returns:
            List of ContradictionReport objects
        """
        reports = sorted(
            self.contradiction_reports,
            key=lambda r: r.detected_at,
            reverse=True
        )[:limit]

        return reports

    def get_stats(self) -> Dict[str, Any]:
        """
        Get handler statistics.

        Returns:
            Dictionary with statistics
        """
        # Count by status
        status_counts = {status.value: 0 for status in CommitmentStatus}
        for commitment in self.commitments.values():
            status_counts[commitment.status.value] += 1

        return {
            "commitments": {
                "total": len(self.commitments),
                "by_status": status_counts
            },
            "contradictions": {
                "total_reports": len(self.contradiction_reports),
                "detection_enabled": self.config["enable_contradiction_detection"]
            },
            "config": {
                "significance_level": self.config["significance_level"],
                "max_commitments": self.config["max_commitments"]
            }
        }

    def clear_commitments(self) -> int:
        """
        Clear all commitments.

        Useful for testing and isolation.

        Returns:
            Number of commitments cleared
        """
        count = len(self.commitments)
        self.commitments.clear()
        self.contradiction_reports.clear()

        logger.log("INFO", f"Cleared {count} commitments",
                  operation="clear_commitments")

        return count
