"""
RESE Phase IV: Output Generator

This module generates the final solution architecture output from Phase IV,
including formatted results, ACI validation, and predictive metrics.

Following CLAUDE.md principles:
- Law of Idempotency: Same inputs produce same outputs
- Law of Configuration Explicitness: All config via env vars
- Circuit Breaker: Detect generation failures
- Structured Logging: JSON with correlation_id
- UTC: All timestamps in UTC ISO-8601

Author: RESE Team
Created: 2026-02-04
Phase: IV - Architectural Synthesis and Validation
"""

import os
import sys
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

# Add schemas to path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "..", "..", "schemas")))

from rese_phase4_schemas import (
    ArchitectureAssembly,
    ParadigmShift,
    SynthesizedKnowledge,
    EpistemicAuditResult,
    IsomorphicMappingResult,
    MCTSRefinementResult,
    Phase4Config,
    AssemblyStatus,
    ValidationLevel,
)


# ============================================================================
# OUTPUT FORMATS
# ============================================================================

class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    YAML = "yaml"
    PRETTY = "pretty"


# ============================================================================
# STRUCTURED LOGGER (REUSED)
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
# OUTPUT GENERATOR
# ============================================================================

class OutputGenerator:
    """
    Generates formatted output from ArchitectureAssembly.

    This is responsible for:
    1. Formatting assembly results in multiple formats
    2. Extracting key metrics and insights
    3. Generating human-readable summaries
    4. Preparing data for downstream consumers
    """

    def __init__(self, config: Phase4Config, logger: Optional[StructuredLogger] = None):
        """
        Initialize output generator.

        Args:
            config: Phase IV configuration
            logger: Optional logger (creates own if not provided)
        """
        self.config = config
        self.logger = logger or StructuredLogger(
            "rese-phase4-output-generator",
            self.config.correlation_id
        )

        self.logger.info(
            "Output Generator initialized",
            config=self.config.to_dict(),
        )

    def generate(
        self,
        assembly: ArchitectureAssembly,
        output_format: OutputFormat = OutputFormat.JSON
    ) -> Dict[str, Any]:
        """
        Generate formatted output from architecture assembly.

        Args:
            assembly: Architecture assembly to format
            output_format: Desired output format

        Returns:
            Formatted output dictionary with:
                - formatted_output: The formatted content
                - metrics: Key metrics extracted
                - metadata: Output metadata
                - validation_summary: Summary of validation results

        Raises:
            ValueError: If assembly is invalid
            TimeoutError: If generation exceeds timeout
        """
        import time
        start_time = time.time()
        timeout_sec = self.config.assembly_timeout_ms / 1000.0

        self.logger.info(
            "Generating output",
            assembly_id=assembly.assembly_id,
            output_format=output_format.value,
        )

        try:
            # Validate assembly
            if not assembly.synthesized_knowledge:
                raise ValueError("Assembly has no synthesized knowledge")

            # Extract metrics
            metrics = self._extract_metrics(assembly)

            # Generate formatted content
            formatted_content = self._format_content(assembly, output_format)

            # Generate validation summary
            validation_summary = self._generate_validation_summary(assembly)

            # Generate predictions
            predictions = self._generate_predictions(assembly)

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                raise TimeoutError(f"Output generation exceeded timeout: {elapsed:.2f}s")

            result = {
                "formatted_output": formatted_content,
                "metrics": metrics,
                "validation_summary": validation_summary,
                "predictions": predictions,
                "metadata": {
                    "assembly_id": assembly.assembly_id,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "output_format": output_format.value,
                    "generation_time_seconds": elapsed,
                    "confidence": assembly.confidence,
                    "aci_reduction": assembly.aci_reduction_achieved,
                },
            }

            self.logger.info(
                "Output generation completed",
                assembly_id=assembly.assembly_id,
                elapsed_seconds=elapsed,
            )

            return result

        except Exception as e:
            self.logger.error("Output generation failed", error=e)
            raise

    def _extract_metrics(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Extract key metrics from assembly."""
        knowledge = assembly.synthesized_knowledge

        metrics = {
            "overall_confidence": assembly.confidence,
            "aci_reduction_achieved": assembly.aci_reduction_achieved,
            "completeness": knowledge.completeness if knowledge else 0.0,
            "consistency": knowledge.consistency if knowledge else 0.0,
            "paradigm_shift_count": len(assembly.paradigm_shifts),
            "validation_passed": assembly.status == AssemblyStatus.VALIDATED,
        }

        # Extract paradigm shift metrics
        if assembly.paradigm_shifts:
            shift_confidences = [ps.confidence for ps in assembly.paradigm_shifts]
            metrics["paradigm_shift_avg_confidence"] = sum(shift_confidences) / len(shift_confidences)
            metrics["paradigm_shift_max_confidence"] = max(shift_confidences)
            metrics["paradigm_shift_min_confidence"] = min(shift_confidences)

        # Extract phase contributions
        if knowledge:
            metrics["phase_contributions"] = {
                "phase1": knowledge.source_phase1 is not None,
                "phase2": knowledge.source_phase2 is not None,
                "phase3": knowledge.source_phase3 is not None,
            }

        # Extract validation metrics
        if assembly.validation_results:
            passed = sum(1 for r in assembly.validation_results if r.get("passed", False))
            metrics["validation_checks_passed"] = passed
            metrics["validation_checks_total"] = len(assembly.validation_results)

        return metrics

    def _format_content(
        self,
        assembly: ArchitectureAssembly,
        output_format: OutputFormat
    ) -> Dict[str, Any]:
        """Format content based on output format."""
        if output_format == OutputFormat.JSON:
            return self._format_json(assembly)
        elif output_format == OutputFormat.MARKDOWN:
            return self._format_markdown(assembly)
        elif output_format == OutputFormat.YAML:
            return self._format_yaml(assembly)
        elif output_format == OutputFormat.PRETTY:
            return self._format_pretty(assembly)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _format_json(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Format as JSON structure."""
        return {
            "format": "json",
            "content": assembly.to_dict(),
        }

    def _format_markdown(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Format as Markdown document."""
        lines = []
        lines.append("# RESE Phase IV: Architecture Assembly")
        lines.append("")
        lines.append(f"**Assembly ID:** `{assembly.assembly_id}`")
        lines.append(f"**Status:** {assembly.status.value}")
        lines.append(f"**Confidence:** {assembly.confidence:.2%}")
        lines.append(f"**ACI Reduction:** {assembly.aci_reduction_achieved:.2%}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Synthesized Knowledge
        if assembly.synthesized_knowledge:
            knowledge = assembly.synthesized_knowledge
            lines.append("## Synthesized Knowledge")
            lines.append("")
            lines.append(f"- **Knowledge ID:** `{knowledge.knowledge_id}`")
            lines.append(f"- **Type:** {knowledge.knowledge_type}")
            lines.append(f"- **Description:** {knowledge.description}")
            lines.append(f"- **Completeness:** {knowledge.completeness:.2%}")
            lines.append(f"- **Consistency:** {knowledge.consistency:.2%}")
            lines.append(f"- **Integration Strategy:** {knowledge.integration_strategy.value}")
            lines.append("")

        # Paradigm Shifts
        if assembly.paradigm_shifts:
            lines.append(f"## Paradigm Shifts ({len(assembly.paradigm_shifts)})")
            lines.append("")

            for i, shift in enumerate(assembly.paradigm_shifts, 1):
                lines.append(f"### {i}. {shift.shift_type.value.title()} Shift")
                lines.append("")
                lines.append(f"- **ID:** `{shift.shift_id}`")
                lines.append(f"- **Confidence:** {shift.confidence:.2%}")
                lines.append(f"- **Description:** {shift.description}")
                lines.append("")

                if shift.transformation_rules:
                    lines.append("**Transformation Rules:**")
                    for rule in shift.transformation_rules:
                        lines.append(f"  - {rule.get('description', 'N/A')}")
                    lines.append("")

        # Validation Results
        if assembly.validation_results:
            lines.append("## Validation Results")
            lines.append("")

            for result in assembly.validation_results:
                status = "[OK]" if result.get("passed", False) else "[FAIL]"
                lines.append(f"- {status} **{result.get('validation_type', 'Unknown')}**")
                if "reason" in result:
                    lines.append(f"  - Reason: {result['reason']}")
                lines.append("")

        # Final Architecture
        if assembly.final_architecture:
            lines.append("## Final Architecture")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(assembly.final_architecture, indent=2))
            lines.append("```")
            lines.append("")

        # Metadata
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"- **Created:** {assembly.created_at.isoformat()}")
        lines.append(f"- **Updated:** {assembly.updated_at.isoformat()}")
        lines.append("")

        return {
            "format": "markdown",
            "content": "\n".join(lines),
        }

    def _format_yaml(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Format as YAML structure (simplified)."""
        # Note: In production, would use PyYAML
        yaml_content = f"""assembly_id: {assembly.assembly_id}
status: {assembly.status.value}
confidence: {assembly.confidence:.4f}
aci_reduction_achieved: {assembly.aci_reduction_achieved:.4f}
paradigm_shifts_count: {len(assembly.paradigm_shifts)}
validation_level: {assembly.validation_level.value}
"""

        return {
            "format": "yaml",
            "content": yaml_content,
        }

    def _format_pretty(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Format as human-readable pretty output."""
        lines = []
        lines.append("=" * 80)
        lines.append("RESE PHASE IV: ARCHITECTURE ASSEMBLY".center(80))
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append(f"Assembly ID: {assembly.assembly_id}")
        lines.append(f"Status:       {assembly.status.value.upper()}")
        lines.append(f"Confidence:   {assembly.confidence:.2%}")
        lines.append(f"ACI Reduction: {assembly.aci_reduction_achieved:.2%}")
        lines.append("")

        # Knowledge
        if assembly.synthesized_knowledge:
            knowledge = assembly.synthesized_knowledge
            lines.append("-" * 80)
            lines.append("SYNTHESIZED KNOWLEDGE")
            lines.append("-" * 80)
            lines.append(f"Completeness: {knowledge.completeness:.2%}")
            lines.append(f"Consistency:  {knowledge.consistency:.2%}")
            lines.append(f"Strategy:     {knowledge.integration_strategy.value}")
            lines.append("")

        # Paradigm Shifts
        if assembly.paradigm_shifts:
            lines.append("-" * 80)
            lines.append(f"PARADIGM SHIFTS ({len(assembly.paradigm_shifts)})")
            lines.append("-" * 80)
            for shift in assembly.paradigm_shifts:
                lines.append(f"  [{shift.shift_type.value}] {shift.description[:60]}...")
                lines.append(f"    Confidence: {shift.confidence:.2%}")
            lines.append("")

        # Validation
        if assembly.validation_results:
            lines.append("-" * 80)
            lines.append("VALIDATION RESULTS")
            lines.append("-" * 80)
            for result in assembly.validation_results:
                status = "PASS" if result.get("passed", False) else "FAIL"
                lines.append(f"  [{status}] {result.get('validation_type', 'Unknown')}")
            lines.append("")

        lines.append("=" * 80)

        return {
            "format": "pretty",
            "content": "\n".join(lines),
        }

    def _generate_validation_summary(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Generate validation summary."""
        if not assembly.validation_results:
            return {
                "total_checks": 0,
                "passed": 0,
                "failed": 0,
                "status": "no_validation",
            }

        passed = sum(1 for r in assembly.validation_results if r.get("passed", False))
        failed = len(assembly.validation_results) - passed

        # Group by validation type
        by_type = {}
        for result in assembly.validation_results:
            vtype = result.get("validation_type", "unknown")
            if vtype not in by_type:
                by_type[vtype] = {"passed": 0, "failed": 0}
            if result.get("passed", False):
                by_type[vtype]["passed"] += 1
            else:
                by_type[vtype]["failed"] += 1

        return {
            "total_checks": len(assembly.validation_results),
            "passed": passed,
            "failed": failed,
            "status": "passed" if failed == 0 else "failed",
            "by_type": by_type,
        }

    def _generate_predictions(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """
        Generate testable predictions from the assembly.

        Per RESE spec §6.3: "The final architecture must generate a set of
        testable predictions that, when verified, demonstrate a statistically
        significant reduction in the Anomaly Characterization Index (ACI)
        relative to the incumbent paradigm."
        """
        predictions = {
            "aci_reduction_prediction": {
                "predicted_reduction": assembly.aci_reduction_achieved,
                "confidence": assembly.confidence,
                "statistical_significance": self._assess_significance(assembly),
                "sample_size": self._estimate_sample_size(assembly),
            },
            "paradigm_shift_predictions": [],
            "constraint_satisfaction_predictions": [],
        }

        # Generate predictions for each paradigm shift
        for shift in assembly.paradigm_shifts:
            pred = {
                "shift_id": shift.shift_id,
                "shift_type": shift.shift_type.value,
                "prediction": f"Applying {shift.shift_type.value} paradigm shift will reduce complexity",
                "confidence": shift.confidence,
                "testable": True,
            }
            predictions["paradigm_shift_predictions"].append(pred)

        # Generate constraint satisfaction predictions
        if assembly.synthesized_knowledge and assembly.synthesized_knowledge.source_phase1:
            phase1 = assembly.synthesized_knowledge.source_phase1
            if phase1.constraints:
                pred = {
                    "prediction": f"All {len(phase1.constraints)} constraints will be satisfied",
                    "confidence": phase1.confidence,
                    "testable": True,
                }
                predictions["constraint_satisfaction_predictions"].append(pred)

        return predictions

    def _assess_significance(self, assembly: ArchitectureAssembly) -> Dict[str, Any]:
        """Assess statistical significance of ACI reduction."""
        # Simplified assessment - in production would use proper statistical tests
        aci_reduction = assembly.aci_reduction_achieved
        confidence = assembly.confidence

        # Thresholds for significance
        alpha = 0.05  # Significance level
        min_effect_size = 0.2  # Minimum 20% reduction

        is_significant = (
            aci_reduction >= min_effect_size and
            confidence >= (1.0 - alpha)
        )

        return {
            "is_significant": is_significant,
            "alpha": alpha,
            "effect_size": aci_reduction,
            "min_effect_size": min_effect_size,
            "confidence_level": confidence,
            "note": "Simplified assessment - production would use proper hypothesis testing",
        }

    def _estimate_sample_size(self, assembly: ArchitectureAssembly) -> int:
        """Estimate sample size for validation."""
        # Base sample size on complexity of assembly
        base_size = 100

        # Adjust by paradigm shifts
        multiplier = 1 + len(assembly.paradigm_shifts) * 0.1

        # Adjust by confidence (lower confidence = larger sample needed)
        if assembly.confidence < 0.8:
            multiplier *= 1.5
        elif assembly.confidence < 0.9:
            multiplier *= 1.2

        return int(base_size * multiplier)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "OutputGenerator",
    "OutputFormat",
]
