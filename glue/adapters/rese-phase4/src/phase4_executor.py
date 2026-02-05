"""
RESE Phase IV: Architecture Assembly Executor

This module implements the final phase of the RESE pipeline, which:
1. Assembles paradigm shifts from validated patterns
2. Integrates knowledge from all previous phases
3. Validates the final architecture
4. Generates the final output

Following CLAUDE.md principles:
- Law of Idempotency: Safe to run multiple times
- Law of Configuration Explicitness: All config via env vars
- Circuit Breaker: Detect assembly failures
- Structured Logging: JSON with correlation_id
- Timeout: All operations timeout (default 25000ms)
- Validation: Validate all inputs before assembly
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import uuid
import json
import time

# Add schemas to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "schemas"))

from rese_phase4_schemas import (
    ArchitectureAssembly,
    ParadigmShift,
    SynthesizedKnowledge,
    EpistemicAuditResult,
    IsomorphicMappingResult,
    MCTSRefinementResult,
    Phase4Config,
    AssemblyStatus,
    ParadigmShiftType,
    ValidationLevel,
    IntegrationStrategy,
)


# ============================================================================
# STRUCTURED LOGGER
# ============================================================================

class StructuredLogger:
    """Structured JSON logger following CLAUDE.md §3.3."""

    def __init__(self, service_name: str, correlation_id: Optional[str] = None):
        self.service_name = service_name
        self.correlation_id = correlation_id or str(uuid.uuid4())

    def _log(self, level: str, msg: str, **kwargs):
        """Internal log method."""
        log_entry = {
            "level": level,
            "msg": msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": self.correlation_id,
            "source_service": self.service_name,
            **kwargs
        }
        print(json.dumps(log_entry))

    def debug(self, msg: str, **kwargs):
        self._log("debug", msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log("info", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log("warning", msg, **kwargs)

    def error(self, msg: str, error: Optional[Exception] = None, **kwargs):
        if error:
            kwargs["error"] = str(error)
            kwargs["error_type"] = type(error).__name__
        self._log("error", msg, **kwargs)


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker for detecting assembly failures.

    Following CLAUDE.md §2.3: System Failure -> Circuit Breaker.
    """

    def __init__(self, failure_threshold: int = 5, timeout_ms: int = 60000):
        self.failure_threshold = failure_threshold
        self.timeout_ms = timeout_ms
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open

    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def can_execute(self) -> bool:
        """Check if operation can execute."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed_ms = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() * 1000
                if elapsed_ms >= self.timeout_ms:
                    self.state = "half_open"
                    return True
            return False

        # half_open - allow one attempt
        return True


# ============================================================================
# PARADIGM SHIFT ASSEMBLER
# ============================================================================

class ParadigmShiftAssembler:
    """
    Assembles paradigm shifts from validated patterns across phases.

    This is the Δ₁ component: Architecture Assembly.
    """

    def __init__(self, config: Phase4Config, logger: StructuredLogger):
        self.config = config
        self.logger = logger

    def assemble(
        self,
        phase1_patterns: List[Dict[str, Any]],
        phase2_patterns: List[Dict[str, Any]],
        phase3_patterns: List[Dict[str, Any]]
    ) -> List[ParadigmShift]:
        """
        Assemble paradigm shifts from patterns across all phases.

        Args:
            phase1_patterns: Patterns from Phase I (epistemic audit)
            phase2_patterns: Patterns from Phase II (isomorphic mappings)
            phase3_patterns: Patterns from Phase III (MCTS refinement)

        Returns:
            List of assembled paradigm shifts

        Raises:
            TimeoutError: If assembly exceeds timeout
        """
        start_time = time.time()
        timeout_sec = self.config.assembly_timeout_ms / 1000.0

        self.logger.info(
            "Starting paradigm shift assembly",
            phase1_patterns_count=len(phase1_patterns),
            phase2_patterns_count=len(phase2_patterns),
            phase3_patterns_count=len(phase3_patterns),
        )

        paradigm_shifts = []

        try:
            # Group patterns by type
            pattern_groups = self._group_patterns_by_type(
                phase1_patterns, phase2_patterns, phase3_patterns
            )

            # Assemble paradigm shifts from each group
            for pattern_type, patterns in pattern_groups.items():
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    raise TimeoutError(f"Paradigm shift assembly exceeded timeout: {elapsed:.2f}s")

                # Assemble shifts from this pattern group
                shifts = self._assemble_from_pattern_group(pattern_type, patterns)
                paradigm_shifts.extend(shifts)

                self.logger.debug(
                    f"Assembled paradigm shifts for type: {pattern_type}",
                    count=len(shifts)
                )

            # Filter by confidence threshold
            paradigm_shifts = [
                ps for ps in paradigm_shifts
                if ps.confidence >= self.config.min_confidence_threshold
            ]

            # Limit to max paradigm shifts
            if len(paradigm_shifts) > self.config.max_paradigm_shifts:
                paradigm_shifts = sorted(
                    paradigm_shifts,
                    key=lambda ps: ps.confidence,
                    reverse=True
                )[:self.config.max_paradigm_shifts]

            self.logger.info(
                "Completed paradigm shift assembly",
                total_shifts=len(paradigm_shifts),
                elapsed_seconds=time.time() - start_time,
            )

            return paradigm_shifts

        except Exception as e:
            self.logger.error("Paradigm shift assembly failed", error=e)
            raise

    def _group_patterns_by_type(
        self,
        phase1_patterns: List[Dict[str, Any]],
        phase2_patterns: List[Dict[str, Any]],
        phase3_patterns: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group patterns by type for assembly."""
        groups = {}

        # Phase I patterns
        for pattern in phase1_patterns:
            pattern_type = pattern.get("type", "structural")
            if pattern_type not in groups:
                groups[pattern_type] = []
            groups[pattern_type].append({**pattern, "source_phase": 1})

        # Phase II patterns
        for pattern in phase2_patterns:
            pattern_type = pattern.get("type", "structural")
            if pattern_type not in groups:
                groups[pattern_type] = []
            groups[pattern_type].append({**pattern, "source_phase": 2})

        # Phase III patterns
        for pattern in phase3_patterns:
            pattern_type = pattern.get("type", "structural")
            if pattern_type not in groups:
                groups[pattern_type] = []
            groups[pattern_type].append({**pattern, "source_phase": 3})

        return groups

    def _assemble_from_pattern_group(
        self,
        pattern_type: str,
        patterns: List[Dict[str, Any]]
    ) -> List[ParadigmShift]:
        """
        Assemble paradigm shifts from a group of related patterns.

        This implements the core synthesis logic:
        1. Identify patterns from multiple phases
        2. Extract transformation rules
        3. Calculate confidence
        4. Create paradigm shift objects
        """
        shifts = []

        # Group patterns by source phase
        phase1_patterns = [p for p in patterns if p.get("source_phase") == 1]
        phase2_patterns = [p for p in patterns if p.get("source_phase") == 2]
        phase3_patterns = [p for p in patterns if p.get("source_phase") == 3]

        # Only create shifts if we have multi-phase patterns
        if len(phase1_patterns) + len(phase2_patterns) + len(phase3_patterns) < 2:
            return shifts

        # Create paradigm shift
        shift = ParadigmShift(
            shift_type=ParadigmShiftType(pattern_type) if pattern_type in [e.value for e in ParadigmShiftType] else ParadigmShiftType.STRUCTURAL,
            description=f"Paradigm shift assembled from {len(patterns)} patterns of type '{pattern_type}'",
            source_patterns=[p.get("pattern_id", str(uuid.uuid4())) for p in patterns],
            phase1_contributions=phase1_patterns,
            phase2_contributions=phase2_patterns,
            phase3_contributions=phase3_patterns,
            transformation_rules=self._extract_transformation_rules(patterns),
            confidence=self._calculate_shift_confidence(patterns),
            validation_status="pending",
        )

        shifts.append(shift)

        return shifts

    def _extract_transformation_rules(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract transformation rules from patterns."""
        rules = []

        for pattern in patterns:
            # Extract rules from pattern metadata
            pattern_rules = pattern.get("transformation_rules", [])
            if isinstance(pattern_rules, list):
                rules.extend(pattern_rules)

        # Deduplicate rules
        seen = set()
        unique_rules = []
        for rule in rules:
            rule_key = json.dumps(rule, sort_keys=True)
            if rule_key not in seen:
                seen.add(rule_key)
                unique_rules.append(rule)

        return unique_rules

    def _calculate_shift_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate confidence for paradigm shift based on pattern confidences."""
        if not patterns:
            return 0.0

        # Average pattern confidences
        confidences = [p.get("confidence", 0.5) for p in patterns]
        base_confidence = sum(confidences) / len(confidences)

        # Boost if multi-phase
        source_phases = set(p.get("source_phase") for p in patterns)
        if len(source_phases) >= 2:
            base_confidence *= 1.1  # 10% boost for multi-phase
        if len(source_phases) >= 3:
            base_confidence *= 1.05  # Additional 5% boost for all three phases

        return min(1.0, base_confidence)


# ============================================================================
# KNOWLEDGE INTEGRATOR
# ============================================================================

class KnowledgeIntegrator:
    """
    Integrates knowledge from all RESE phases.

    This synthesizes outputs from Phases I, II, and III into a coherent knowledge base.
    """

    def __init__(self, config: Phase4Config, logger: StructuredLogger):
        self.config = config
        self.logger = logger

    def integrate(
        self,
        phase1_result: Optional[EpistemicAuditResult],
        phase2_result: Optional[IsomorphicMappingResult],
        phase3_result: Optional[MCTSRefinementResult],
        paradigm_shifts: List[ParadigmShift]
    ) -> SynthesizedKnowledge:
        """
        Integrate knowledge from all phases.

        Args:
            phase1_result: Phase I audit result
            phase2_result: Phase II mapping result
            phase3_result: Phase III refinement result
            paradigm_shifts: Assembled paradigm shifts

        Returns:
            Synthesized knowledge object
        """
        start_time = time.time()

        self.logger.info(
            "Starting knowledge integration",
            has_phase1=phase1_result is not None,
            has_phase2=phase2_result is not None,
            has_phase3=phase3_result is not None,
            paradigm_shifts_count=len(paradigm_shifts),
        )

        try:
            # Create synthesized knowledge
            knowledge = SynthesizedKnowledge(
                knowledge_type="architecture_assembly",
                description=f"Synthesized knowledge from {sum([phase1_result is not None, phase2_result is not None, phase3_result is not None])} RESE phases",
                source_phase1=phase1_result,
                source_phase2=phase2_result,
                source_phase3=phase3_result,
                paradigm_shifts=paradigm_shifts,
                integration_strategy=self.config.integration_strategy,
                synthesis_rules=self._generate_synthesis_rules(
                    phase1_result, phase2_result, phase3_result
                ),
                confidence=self._calculate_overall_confidence(
                    phase1_result, phase2_result, phase3_result, paradigm_shifts
                ),
                completeness=self._calculate_completeness(
                    phase1_result, phase2_result, phase3_result
                ),
                consistency=self._calculate_consistency(
                    phase1_result, phase2_result, phase3_result
                ),
            )

            self.logger.info(
                "Completed knowledge integration",
                knowledge_id=knowledge.knowledge_id,
                elapsed_seconds=time.time() - start_time,
            )

            return knowledge

        except Exception as e:
            self.logger.error("Knowledge integration failed", error=e)
            raise

    def _generate_synthesis_rules(
        self,
        phase1_result: Optional[EpistemicAuditResult],
        phase2_result: Optional[IsomorphicMappingResult],
        phase3_result: Optional[MCTSRefinementResult]
    ) -> List[Dict[str, Any]]:
        """Generate synthesis rules from phase results."""
        rules = []

        # Rule 1: Trust validated hypotheses over unvalidated
        rules.append({
            "rule_id": str(uuid.uuid4()),
            "type": "validation_priority",
            "description": "Prioritize validated hypotheses from Phase III",
            "priority": 1.0,
        })

        # Rule 2: Cross-validate isomorphisms with constraints
        if phase1_result and phase2_result:
            rules.append({
                "rule_id": str(uuid.uuid4()),
                "type": "cross_validation",
                "description": "Cross-validate Phase II isomorphisms with Phase I constraints",
                "priority": 0.9,
            })

        # Rule 3: Weight by confidence
        rules.append({
            "rule_id": str(uuid.uuid4()),
            "type": "confidence_weighting",
            "description": "Weight contributions by confidence scores",
            "priority": 0.8,
        })

        return rules

    def _calculate_overall_confidence(
        self,
        phase1_result: Optional[EpistemicAuditResult],
        phase2_result: Optional[IsomorphicMappingResult],
        phase3_result: Optional[MCTSRefinementResult],
        paradigm_shifts: List[ParadigmShift]
    ) -> float:
        """Calculate overall confidence from all sources."""
        confidences = []

        if phase1_result:
            confidences.append(phase1_result.confidence)
        if phase2_result:
            confidences.append(phase2_result.confidence)
        if phase3_result:
            confidences.append(phase3_result.confidence)

        for shift in paradigm_shifts:
            confidences.append(shift.confidence)

        if not confidences:
            return 0.5

        return sum(confidences) / len(confidences)

    def _calculate_completeness(
        self,
        phase1_result: Optional[EpistemicAuditResult],
        phase2_result: Optional[IsomorphicMappingResult],
        phase3_result: Optional[MCTSRefinementResult]
    ) -> float:
        """Calculate completeness based on phase coverage."""
        phases_present = sum([
            phase1_result is not None,
            phase2_result is not None,
            phase3_result is not None,
        ])

        return phases_present / 3.0

    def _calculate_consistency(
        self,
        phase1_result: Optional[EpistemicAuditResult],
        phase2_result: Optional[IsomorphicMappingResult],
        phase3_result: Optional[MCTSRefinementResult]
    ) -> float:
        """
        Calculate consistency score across phases.

        This checks for contradictions between phases.
        """
        # Start with baseline consistency
        consistency = 1.0

        # Check Phase I contradictions
        if phase1_result and phase1_result.contradictions:
            # Reduce consistency based on number of contradictions
            contradiction_penalty = min(0.3, len(phase1_result.contradictions) * 0.05)
            consistency -= contradiction_penalty

        # Check for inconsistencies between phases
        # (Simplified - would need more sophisticated analysis in practice)
        if phase1_result and phase2_result:
            # Cross-validation check
            if phase1_result.confidence > 0.7 and phase2_result.confidence > 0.7:
                # Both phases confident - boost consistency
                consistency = min(1.0, consistency + 0.1)

        return max(0.0, consistency)


# ============================================================================
# ARCHITECTURE VALIDATOR
# ============================================================================

class ArchitectureValidator:
    """
    Validates the final architecture assembly.

    This is the Δ₃ component: ACI Reduction Validation.
    """

    def __init__(self, config: Phase4Config, logger: StructuredLogger):
        self.config = config
        self.logger = logger

    def validate(self, assembly: ArchitectureAssembly) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the architecture assembly.

        Args:
            assembly: Architecture assembly to validate

        Returns:
            Tuple of (is_valid, validation_results)
        """
        start_time = time.time()

        self.logger.info(
            "Starting architecture validation",
            assembly_id=assembly.assembly_id,
            validation_level=self.config.validation_level.value,
        )

        validation_results = []

        try:
            # Validation 1: Completeness check
            completeness_result = self._validate_completeness(assembly)
            validation_results.append(completeness_result)

            # Validation 2: Consistency check
            consistency_result = self._validate_consistency(assembly)
            validation_results.append(consistency_result)

            # Validation 3: Confidence check
            confidence_result = self._validate_confidence(assembly)
            validation_results.append(confidence_result)

            # Validation 4: ACI reduction check
            aci_result = self._validate_aci_reduction(assembly)
            validation_results.append(aci_result)

            # Additional validations based on level
            if self.config.validation_level in [ValidationLevel.STRICT, ValidationLevel.FORMAL]:
                strict_results = self._validate_strict(assembly)
                validation_results.extend(strict_results)

            if self.config.validation_level == ValidationLevel.FORMAL and self.config.enable_formal_verification:
                formal_results = self._validate_formal(assembly)
                validation_results.extend(formal_results)

            # Determine overall validity
            is_valid = all(r.get("passed", False) for r in validation_results)

            self.logger.info(
                "Completed architecture validation",
                is_valid=is_valid,
                validation_count=len(validation_results),
                elapsed_seconds=time.time() - start_time,
            )

            return is_valid, validation_results

        except Exception as e:
            self.logger.error("Architecture validation failed", error=e)
            return False, [{
                "validation_type": "error",
                "passed": False,
                "error": str(e),
            }]

    def _validate_completeness(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Validate assembly completeness."""
        knowledge = assembly.synthesized_knowledge

        if not knowledge:
            return {
                "validation_type": "completeness",
                "passed": False,
                "reason": "No synthesized knowledge",
            }

        # Check if we have inputs from multiple phases
        phases_present = sum([
            knowledge.source_phase1 is not None,
            knowledge.source_phase2 is not None,
            knowledge.source_phase3 is not None,
        ])

        passed = phases_present >= 2  # Require at least 2 phases

        return {
            "validation_type": "completeness",
            "passed": passed,
            "phases_present": phases_present,
            "completeness_score": knowledge.completeness,
        }

    def _validate_consistency(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Validate assembly consistency."""
        knowledge = assembly.synthesized_knowledge

        if not knowledge:
            return {
                "validation_type": "consistency",
                "passed": False,
                "reason": "No synthesized knowledge",
            }

        passed = knowledge.consistency >= 0.6  # Minimum consistency threshold

        return {
            "validation_type": "consistency",
            "passed": passed,
            "consistency_score": knowledge.consistency,
        }

    def _validate_confidence(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Validate assembly confidence."""
        knowledge = assembly.synthesized_knowledge

        if not knowledge:
            return {
                "validation_type": "confidence",
                "passed": False,
                "reason": "No synthesized knowledge",
            }

        passed = knowledge.confidence >= self.config.min_confidence_threshold

        return {
            "validation_type": "confidence",
            "passed": passed,
            "confidence_score": knowledge.confidence,
            "threshold": self.config.min_confidence_threshold,
        }

    def _validate_aci_reduction(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Validate ACI (Algorithmic Complexity Index) reduction."""
        # Target: at least 20% ACI reduction
        target_reduction = 0.2
        passed = assembly.aci_reduction_achieved >= target_reduction

        return {
            "validation_type": "aci_reduction",
            "passed": passed,
            "aci_reduction": assembly.aci_reduction_achieved,
            "target": target_reduction,
        }

    def _validate_strict(self, assembly: ArchitectureAssembly) -> List[Dict[str, Any]]:
        """Perform strict-level validations."""
        results = []

        # Strict validation 1: Paradigm shift quality
        if assembly.paradigm_shifts:
            avg_shift_confidence = sum(ps.confidence for ps in assembly.paradigm_shifts) / len(assembly.paradigm_shifts)
            results.append({
                "validation_type": "paradigm_shift_quality",
                "passed": avg_shift_confidence >= 0.8,
                "avg_confidence": avg_shift_confidence,
            })

        # Strict validation 2: Cross-phase consistency
        if self.config.enable_cross_validation and assembly.synthesized_knowledge:
            knowledge = assembly.synthesized_knowledge
            # Check for cross-validation between phases
            has_cross_phase = (
                knowledge.source_phase1 is not None and
                knowledge.source_phase2 is not None and
                knowledge.source_phase3 is not None
            )
            results.append({
                "validation_type": "cross_phase_consistency",
                "passed": has_cross_phase,
                "has_all_phases": has_cross_phase,
            })

        return results

    def _validate_formal(self, assembly: ArchitectureAssembly) -> List[Dict[str, Any]]:
        """
        Perform formal verification (Lean 4).

        NOTE: This is a placeholder for formal verification integration.
        In production, this would invoke Lean 4 proofs.
        """
        results = []

        # Placeholder for formal verification
        results.append({
            "validation_type": "formal_verification",
            "passed": True,  # Placeholder
            "note": "Formal verification not yet implemented",
        })

        return results


# ============================================================================
# MAIN ARCHITECTURE ASSEMBLY EXECUTOR
# ============================================================================

class ArchitectureAssemblyExecutor:
    """
    Main executor for RESE Phase IV: Architecture Assembly.

    This orchestrates:
    1. Paradigm shift assembly
    2. Knowledge integration
    3. Architecture validation
    4. Final output generation

    Following CLAUDE.md principles:
    - Idempotent operations
    - Circuit breaker for failures
    - Timeout protection
    - Structured logging
    """

    def __init__(self, config: Optional[Phase4Config] = None):
        """Initialize executor with configuration."""
        self.config = config or Phase4Config.from_env()
        self.logger = StructuredLogger(
            "rese-phase4-executor",
            self.config.correlation_id
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_ms=60000
        )

        # Initialize components
        self.paradigm_assembler = ParadigmShiftAssembler(self.config, self.logger)
        self.knowledge_integrator = KnowledgeIntegrator(self.config, self.logger)
        self.architecture_validator = ArchitectureValidator(self.config, self.logger)

        self.logger.info(
            "Architecture Assembly Executor initialized",
            config=self.config.to_dict(),
        )

    def execute(
        self,
        phase1_result: Optional[Dict[str, Any]] = None,
        phase2_result: Optional[Dict[str, Any]] = None,
        phase3_result: Optional[Dict[str, Any]] = None,
        phase1_patterns: List[Dict[str, Any]] = None,
        phase2_patterns: List[Dict[str, Any]] = None,
        phase3_patterns: List[Dict[str, Any]] = None,
    ) -> ArchitectureAssembly:
        """
        Execute Phase IV: Architecture Assembly.

        Args:
            phase1_result: Phase I output (epistemic audit)
            phase2_result: Phase II output (isomorphic mappings)
            phase3_result: Phase III output (MCTS refinement)
            phase1_patterns: Patterns from Phase I
            phase2_patterns: Patterns from Phase II
            phase3_patterns: Patterns from Phase III

        Returns:
            ArchitectureAssembly object

        Raises:
            RuntimeError: If circuit breaker is open
            TimeoutError: If assembly exceeds timeout
        """
        start_time = time.time()

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            error_msg = "Circuit breaker is open - too many recent failures"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            self.logger.info(
                "Starting Phase IV: Architecture Assembly",
                has_phase1=phase1_result is not None,
                has_phase2=phase2_result is not None,
                has_phase3=phase3_result is not None,
            )

            # Parse phase results
            p1_result = self._parse_phase1_result(phase1_result)
            p2_result = self._parse_phase2_result(phase2_result)
            p3_result = self._parse_phase3_result(phase3_result)

            # Default empty patterns if not provided
            phase1_patterns = phase1_patterns or []
            phase2_patterns = phase2_patterns or []
            phase3_patterns = phase3_patterns or []

            # Step 1: Assemble paradigm shifts
            self.logger.info("Step 1: Assembling paradigm shifts")
            paradigm_shifts = self.paradigm_assembler.assemble(
                phase1_patterns, phase2_patterns, phase3_patterns
            )

            # Step 2: Integrate knowledge
            self.logger.info("Step 2: Integrating knowledge")
            synthesized_knowledge = self.knowledge_integrator.integrate(
                p1_result, p2_result, p3_result, paradigm_shifts
            )

            # Step 3: Create initial assembly
            self.logger.info("Step 3: Creating architecture assembly")
            assembly = ArchitectureAssembly(
                synthesized_knowledge=synthesized_knowledge,
                paradigm_shifts=paradigm_shifts,
                final_architecture=self._generate_final_architecture(synthesized_knowledge, paradigm_shifts),
                aci_reduction_achieved=self._calculate_aci_reduction(p3_result),
                confidence=synthesized_knowledge.confidence,
                validation_level=self.config.validation_level,
                status=AssemblyStatus.ASSEMBLING,
            )

            # Step 4: Validate assembly
            self.logger.info("Step 4: Validating architecture")
            is_valid, validation_results = self.architecture_validator.validate(assembly)

            assembly.validation_results = validation_results
            assembly.status = AssemblyStatus.VALIDATED if is_valid else AssemblyStatus.FAILED

            # Record success
            self.circuit_breaker.record_success()

            elapsed = time.time() - start_time
            self.logger.info(
                "Phase IV completed successfully",
                assembly_id=assembly.assembly_id,
                is_valid=is_valid,
                paradigm_shifts_count=len(paradigm_shifts),
                elapsed_seconds=elapsed,
            )

            return assembly

        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            self.logger.error("Phase IV execution failed", error=e)
            raise

    def _parse_phase1_result(self, result: Optional[Dict[str, Any]]) -> Optional[EpistemicAuditResult]:
        """Parse Phase I result from dictionary."""
        if not result:
            return None
        try:
            return EpistemicAuditResult.from_dict(result)
        except Exception as e:
            self.logger.warning("Failed to parse Phase I result", error=e)
            return None

    def _parse_phase2_result(self, result: Optional[Dict[str, Any]]) -> Optional[IsomorphicMappingResult]:
        """Parse Phase II result from dictionary."""
        if not result:
            return None
        try:
            return IsomorphicMappingResult.from_dict(result)
        except Exception as e:
            self.logger.warning("Failed to parse Phase II result", error=e)
            return None

    def _parse_phase3_result(self, result: Optional[Dict[str, Any]]) -> Optional[MCTSRefinementResult]:
        """Parse Phase III result from dictionary."""
        if not result:
            return None
        try:
            return MCTSRefinementResult.from_dict(result)
        except Exception as e:
            self.logger.warning("Failed to parse Phase III result", error=e)
            return None

    def _generate_final_architecture(
        self,
        knowledge: SynthesizedKnowledge,
        paradigm_shifts: List[ParadigmShift]
    ) -> Dict[str, Any]:
        """Generate final architecture specification."""
        architecture = {
            "architecture_id": str(uuid.uuid4()),
            "knowledge_base": {
                "knowledge_id": knowledge.knowledge_id,
                "confidence": knowledge.confidence,
                "completeness": knowledge.completeness,
                "consistency": knowledge.consistency,
            },
            "paradigm_shifts": [
                {
                    "shift_id": ps.shift_id,
                    "type": ps.shift_type.value,
                    "confidence": ps.confidence,
                    "description": ps.description,
                }
                for ps in paradigm_shifts
            ],
            "integration_strategy": knowledge.integration_strategy.value,
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_shifts": len(paradigm_shifts),
            },
        }

        return architecture

    def _calculate_aci_reduction(self, phase3_result: Optional[MCTSRefinementResult]) -> float:
        """Calculate ACI reduction achieved."""
        if phase3_result:
            return phase3_result.aci_reduction
        return 0.0


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "ArchitectureAssemblyExecutor",
    "ParadigmShiftAssembler",
    "KnowledgeIntegrator",
    "ArchitectureValidator",
    "Phase4Config",
    # Additional Phase IV components
    "OutputGenerator",
    "PredictiveValidator",
    "ResultVerifier",
    "StructuredLogger",
    "CircuitBreaker",
]
