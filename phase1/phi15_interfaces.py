"""
Phi 1.5 Integration Interfaces

Integration with RESE Stages 6, 1, and 7 for the Phi 1.5 system.

- Stage 6 → Phi 1.5: Receive null results and error classifications
- Phi 1.5 → Stage 1: Send inferred assumptions as constraints
- Phi 1.5 → Stage 7: Request assumption validation
- Stage 7 → Phi 1.5: Receive validation results and update confidence

Author: Agent B1 (Phi 1/Phi 1.5 Specialist)
Created: 2025-12-31
Status: Green - Active Implementation
"""

from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from tacit_assumption_miner import (
    Phi15Engine, NullResult, TacitAssumption,
    ParadigmShiftRecommendation, ErrorType
)
from failure_database import FailureDatabase, DatabaseManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Stage 6 → Φ₁.₅: Input Interface
# ============================================================================

class Phi15Stage6Interface:
    """
    Interface for receiving null results from Stage 6 (Error Source Analysis).

    This is the primary input point for the Φ₁.₅ system. It receives
    failed attempts with error classifications and stores them for
    assumption mining.
    """

    def __init__(self, phi15_engine: Phi15Engine,
                 database: Optional[FailureDatabase] = None):
        """
        Initialize Stage 6 interface.

        Args:
            phi15_engine: Φ₁.₅ engine instance
            database: Optional database instance (creates new if None)
        """
        self.phi15 = phi15_engine
        self.database = database or DatabaseManager().db

        # Configuration for incremental processing
        self.incremental_threshold = 10  # Process after N new failures
        self.incremental_time_hours = 1   # Process every N hours
        self.anomaly_rate_threshold = 0.3 # Process if anomaly rate > threshold

        # Tracking
        self.last_processing_time = datetime.now()
        self.unprocessed_count = 0

    def receive_null_result(self, result: NullResult) -> None:
        """
        Receive a single null result from Stage 6.

        Args:
            result: Null result from Stage 6 Error Source Analysis

        Flow:
            1. Validate input
            2. Add to database
            3. Extract features
            4. Check if should trigger incremental processing
        """
        try:
            # Validate
            self._validate_null_result(result)

            # Add to database
            self.database.add_failure(result)
            self.unprocessed_count += 1

            logger.info(f"Received null result: {result.attempt_id} "
                       f"(error_type: {result.error_type.value})")

            # Check if should process incrementally
            if self.should_process_incrementally():
                logger.info("Incremental processing threshold reached")
                self.trigger_incremental_processing()

        except Exception as e:
            logger.error(f"Error receiving null result {result.attempt_id}: {e}")
            raise

    def receive_batch_null_results(self, results: List[NullResult]) -> int:
        """
        Receive batch of null results from Stage 6.

        Args:
            results: List of null results

        Returns:
            Number of results successfully added
        """
        success_count = 0

        for result in results:
            try:
                self.receive_null_result(result)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to add null result {result.attempt_id}: {e}")

        logger.info(f"Received batch: {success_count}/{len(results)} successful")

        # Trigger full processing after batch
        if success_count > 0:
            self.trigger_full_processing()

        return success_count

    def should_process_incrementally(self) -> bool:
        """
        Decide whether to process incrementally.

        Returns:
            True if any threshold is met
        """
        # Threshold 1: Enough new results
        count_threshold = self.unprocessed_count >= self.incremental_threshold

        # Threshold 2: Time since last processing
        time_since_last = (datetime.now() - self.last_processing_time).total_seconds() / 3600
        time_threshold = time_since_last >= self.incremental_time_hours

        # Threshold 3: High anomaly rate (simplified)
        anomaly_threshold = False  # Would compute from recent failures

        return count_threshold or time_threshold or anomaly_threshold

    def trigger_incremental_processing(self) -> None:
        """Trigger incremental processing of new failures"""
        try:
            # Get unprocessed failures
            unprocessed = self.database.get_unprocessed_failures()

            if not unprocessed:
                return

            logger.info(f"Processing {len(unprocessed)} unprocessed failures")

            # Convert to NullResult objects if needed
            null_results = unprocessed  # Already NullResult objects

            # Process through Φ₁.₅
            assumptions, paradigm_rec = self.phi15.process_null_results(null_results)

            # Mark as processed
            for nr in null_results:
                self.database.mark_as_processed(nr.attempt_id)

            self.unprocessed_count = 0
            self.last_processing_time = datetime.now()

            logger.info(f"Inferred {len(assumptions)} new assumptions")
            logger.info(f"Paradigm crisis: {paradigm_rec.trigger}")

            # Trigger Stage 1 notification if assumptions found
            if assumptions:
                self.notify_stage1(assumptions, paradigm_rec)

        except Exception as e:
            logger.error(f"Error in incremental processing: {e}")
            raise

    def trigger_full_processing(self) -> None:
        """Trigger full re-processing of all failures"""
        try:
            # Get all failures
            all_failures = self.database.get_failures_since(
                datetime.now() - __import__('datetime').timedelta(days=365)
            )

            logger.info(f"Full processing: {len(all_failures)} failures")

            # Process through Φ₁.₅
            assumptions, paradigm_rec = self.phi15.process_null_results(all_failures)

            logger.info(f"Inferred {len(assumptions)} total assumptions")
            logger.info(f"Paradigm crisis: {paradigm_rec.trigger}")

            # Notify Stage 1
            if assumptions:
                self.notify_stage1(assumptions, paradigm_rec)

        except Exception as e:
            logger.error(f"Error in full processing: {e}")
            raise

    def notify_stage1(self, assumptions: List[TacitAssumption],
                     paradigm_rec: ParadigmShiftRecommendation) -> None:
        """Notify Stage 1 about new assumptions"""
        # This would be implemented by Stage 1 interface
        logger.info(f"Notifying Stage 1: {len(assumptions)} assumptions, "
                   f"paradigm_crisis={paradigm_rec.trigger}")

    def _validate_null_result(self, result: NullResult) -> None:
        """Validate null result input"""
        if not result.attempt_id:
            raise ValueError("Null result must have attempt_id")

        if not result.error_type:
            raise ValueError("Null result must have error_type")

        if not result.error_message:
            raise ValueError("Null result must have error_message")


# ============================================================================
# Φ₁.₅ → Stage 1: Output Interface
# ============================================================================

class Phi15Stage1Interface:
    """
    Interface for sending inferred assumptions to Stage 1 (Prompt Analysis).

    Converts inferred tacit assumptions to SCE constraints and sends
    them to Stage 1 for integration into the problem formulation.
    """

    def __init__(self, phi15_engine: Phi15Engine,
                 stage1_callback: Optional[Callable] = None):
        """
        Initialize Stage 1 interface.

        Args:
            phi15_engine: Φ₁.₅ engine instance
            stage1_callback: Optional callback function for Stage 1 integration
        """
        self.phi15 = phi15_engine
        self.stage1_callback = stage1_callback
        self.confidence_threshold = 0.6  # Only send high-confidence assumptions

    def send_assumptions(self, assumptions: List[TacitAssumption]) -> int:
        """
        Send inferred assumptions to Stage 1.

        Args:
            assumptions: List of inferred assumptions

        Returns:
            Number of assumptions sent

        Flow:
            1. Filter by confidence threshold
            2. Convert to SCE constraints
            3. Send to Stage 1 (via callback or direct API)
            4. Store in database
        """
        # Filter by confidence
        high_confidence = [
            a for a in assumptions
            if a.confidence >= self.confidence_threshold
        ]

        logger.info(f"Sending {len(high_confidence)}/{len(assumptions)} "
                   f"high-confidence assumptions to Stage 1")

        success_count = 0
        for assumption in high_confidence:
            try:
                # Convert to SCE constraint
                sce_constraint = assumption.to_sce_constraint()

                # Send to Stage 1
                if self.stage1_callback:
                    self.stage1_callback(sce_constraint)
                else:
                    # Default: just log
                    logger.info(f"Would send to Stage 1: {assumption.description}")

                success_count += 1

            except Exception as e:
                logger.error(f"Error sending assumption {assumption.id}: {e}")

        return success_count

    def send_paradigm_shift_recommendation(self,
                                          recommendation: ParadigmShiftRecommendation) -> None:
        """
        Send paradigm shift recommendation to Stage 1.

        Args:
            recommendation: Paradigm shift recommendation

        Action:
            Stage 1 will:
            - Flag high-priority paradigm issue
            - Present alternatives to user
            - Request guidance on paradigm selection
        """
        if not recommendation.trigger:
            return

        logger.warning(f"PARADIGM CRISIS RECOMMENDATION (confidence: {recommendation.confidence:.2f})")
        logger.warning(f"Explanation:\n{recommendation.explanation}")

        if recommendation.trigger and recommendation.confidence > 0.8:
            logger.critical("HIGH PRIORITY: Paradigm shift recommended")

        # Send to Stage 1
        if self.stage1_callback:
            try:
                self.stage1_callback({
                    'type': 'paradigm_shift',
                    'data': recommendation.to_dict()
                })
            except Exception as e:
                logger.error(f"Error sending paradigm shift: {e}")

    def format_for_stage1(self, assumption: TacitAssumption) -> Dict:
        """
        Format assumption for Stage 1 consumption.

        Args:
            assumption: Tacit assumption

        Returns:
            Dictionary formatted for Stage 1 API
        """
        return {
            'constraint_id': assumption.id,
            'type': 'soft',  # Inferred constraints start as soft
            'description': f"[INFERRED by Φ₁.₅] {assumption.description}",
            'formalization': assumption.formalization,
            'source': 'phi15_inferred',
            'confidence': assumption.confidence,
            'support': assumption.support,
            'relaxation': assumption.constraint_relaxation,
            'paradigm_implication': assumption.paradigm_implication
        }


# ============================================================================
# Φ₁.₅ ↔ Stage 7: Validation Interface
# ============================================================================

@dataclass
class ValidationResult:
    """Result from Stage 7 validation"""
    assumption_id: str
    success: bool
    improvement_score: float  # How much this improved things [0, 1]
    validation_type: str
    timestamp: datetime


class Phi15Stage7Interface:
    """
    Interface for Stage 7 (Validation) integration.

    Sends assumptions to Stage 7 for validation and receives feedback
    to update confidence scores.
    """

    def __init__(self, phi15_engine: Phi15Engine,
                 database: Optional[FailureDatabase] = None):
        """
        Initialize Stage 7 interface.

        Args:
            phi15_engine: Φ₁.₅ engine instance
            database: Optional database instance
        """
        self.phi15 = phi15_engine
        self.database = database or DatabaseManager().db

    def request_validation(self, assumption: TacitAssumption) -> Dict:
        """
        Request Stage 7 to validate an assumption.

        Args:
            assumption: Assumption to validate

        Returns:
            Validation request dictionary
        """
        request = {
            'assumption_id': assumption.id,
            'description': assumption.description,
            'formalization': assumption.formalization,
            'confidence': assumption.confidence,
            'validation_type': self._determine_validation_type(assumption),
            'prediction': self._generate_validation_prediction(assumption),
            'test_protocol': self._generate_test_protocol(assumption)
        }

        logger.info(f"Requesting validation for assumption {assumption.id}")

        return request

    def receive_validation_result(self, result: ValidationResult) -> None:
        """
        Receive validation result from Stage 7 and update confidence.

        Args:
            result: Validation result

        Action:
            - Update assumption confidence based on validation
            - Mark assumption as verified if successful
            - Adjust confidence up or down
        """
        try:
            # Find assumption
            assumption = self.phi15.assumptions.get(result.assumption_id)

            if not assumption:
                # Try database
                assumption = self.database.get_assumption(result.assumption_id)

            if not assumption:
                logger.warning(f"Assumption {result.assumption_id} not found")
                return

            # Update confidence based on validation
            old_confidence = assumption.confidence

            if result.success:
                # Boost confidence
                assumption.confidence = min(1.0, assumption.confidence * 1.2)
                assumption.verified = True
                logger.info(f"Validated assumption {assumption.id}: "
                           f"{old_confidence:.2f} → {assumption.confidence:.2f}")
            else:
                # Reduce confidence
                assumption.confidence = max(0.0, assumption.confidence * 0.7)
                logger.warning(f"Validation failed for {assumption.id}: "
                             f"{old_confidence:.2f} → {assumption.confidence:.2f}")

            # Update in database
            self.database.update_assumption_confidence(
                assumption.id,
                assumption.confidence
            )

        except Exception as e:
            logger.error(f"Error processing validation result: {e}")

    def _determine_validation_type(self, assumption: TacitAssumption) -> str:
        """Determine how to validate this assumption"""
        if assumption.pattern_type.value == 'systematic_violation':
            return 'relaxation'
        elif assumption.paradigm_implication:
            return 'counterexample'
        else:
            return 'simulation'

    def _generate_validation_prediction(self, assumption: TacitAssumption) -> str:
        """Generate prediction for validation"""
        return f"If {assumption.description} is relaxed/removed, "
        f"success rate should increase"

    def _generate_test_protocol(self, assumption: TacitAssumption) -> str:
        """Generate test protocol for validation"""
        return f"Test: Reformulate problem without {assumption.description} "


# ============================================================================
# Integrated Interface Manager
# ============================================================================

class Phi15InterfaceManager:
    """
    Manages all Φ₁.₅ interfaces for seamless integration with RESE.

    Provides a single point of access for all Stage 6, 1, and 7 interactions.
    """

    def __init__(self, phi15_engine: Optional[Phi15Engine] = None,
                 database_path: str = "rese/data/phi15_failures.db"):
        """
        Initialize interface manager.

        Args:
            phi15_engine: Optional Φ₁.₅ engine (creates new if None)
            database_path: Path to database
        """
        # Create engine if not provided
        self.phi15 = phi15_engine or Phi15Engine()

        # Create database
        self.database = DatabaseManager(database_path)

        # Create interfaces
        self.stage6 = Phi15Stage6Interface(self.phi15, self.database.db)
        self.stage1 = Phi15Stage1Interface(self.phi15)
        self.stage7 = Phi15Stage7Interface(self.phi15, self.database.db)

        logger.info("Φ₁.₅ Interface Manager initialized")

    def process_stage6_input(self, null_results: List[NullResult]) -> Dict:
        """
        Process input from Stage 6.

        Args:
            null_results: List of null results

        Returns:
            Processing results summary
        """
        logger.info(f"Processing {len(null_results)} null results from Stage 6")

        # Add to database via Stage 6 interface
        count = self.stage6.receive_batch_null_results(null_results)

        # Get new assumptions
        assumptions = self.phi15.get_top_assumptions(k=10)

        # Send to Stage 1
        if assumptions:
            sent_count = self.stage1.send_assumptions(assumptions)

            # Check for paradigm shift
            recent_paradigm = self.phi15.paradigm_history[-1] if self.phi15.paradigm_history else None
            if recent_paradigm:
                self.stage1.send_paradigm_shift_recommendation(recent_paradigm)

            return {
                'processed': count,
                'assumptions_sent': sent_count,
                'paradigm_crisis': recent_paradigm.trigger if recent_paradigm else False
            }

        return {
            'processed': count,
            'assumptions_sent': 0,
            'paradigm_crisis': False
        }

    def validate_assumption(self, assumption_id: str,
                           success: bool,
                           improvement_score: float) -> None:
        """
        Validate an assumption (called by Stage 7).

        Args:
            assumption_id: ID of assumption to validate
            success: Whether validation was successful
            improvement_score: How much this improved things [0, 1]
        """
        result = ValidationResult(
            assumption_id=assumption_id,
            success=success,
            improvement_score=improvement_score,
            validation_type='external',
            timestamp=datetime.now()
        )

        self.stage7.receive_validation_result(result)

    def get_status(self) -> Dict:
        """Get current status of Φ₁.₅ system"""
        stats = self.database.get_statistics()

        return {
            'total_failures': stats['total_failures'],
            'total_assumptions': stats['total_assumptions'],
            'recent_assumptions': stats['recent_assumptions_30d'],
            'high_confidence_assumptions': stats['high_confidence_assumptions'],
            'paradigm_crisis_detected': len([p for p in self.phi15.paradigm_history if p.trigger]) > 0
        }

    def shutdown(self) -> None:
        """Shutdown and cleanup"""
        self.database.close()
        logger.info("Φ₁.₅ Interface Manager shutdown complete")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_interface_manager(config: Optional[Dict] = None) -> Phi15InterfaceManager:
    """
    Create a Φ₁.₅ interface manager with optional configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized Phi15InterfaceManager
    """
    phi15_config = config.get('phi15', None) if config else None
    db_path = config.get('database_path', 'rese/data/phi15_failures.db') if config else 'rese/data/phi15_failures.db'

    engine = Phi15Engine(phi15_config) if phi15_config else None

    return Phi15InterfaceManager(engine, db_path)


if __name__ == "__main__":
    # Quick test
    print("Φ₁.₅ Integration Interfaces - Agent B1")
    print("=" * 50)

    # Create interface manager
    manager = create_interface_manager()

    # Get status
    status = manager.get_status()
    print(f"\nSystem Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    print(f"\nInterfaces ready for integration with Stages 6, 1, and 7")

    # Cleanup
    manager.shutdown()
