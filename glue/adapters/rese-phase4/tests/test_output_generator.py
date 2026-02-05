"""
Unit tests for RESE Phase IV Output Generator

Tests cover:
- JSON output generation
- Markdown output generation
- YAML output generation
- Pretty output generation
- Metrics extraction
- Validation summary generation
- Prediction generation

Following CLAUDE.md principles:
- Law of Runtime Truth: Test against actual behavior
- Law of Idempotency: Verify reproducible outputs
"""

import pytest
import sys
import os
from datetime import datetime, timezone

# Add src and schemas to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "schemas"))

from src.output_generator import OutputGenerator, OutputFormat, StructuredLogger
from rese_phase4_schemas import (
    ArchitectureAssembly,
    SynthesizedKnowledge,
    ParadigmShift,
    ParadigmShiftType,
    Phase4Config,
    AssemblyStatus,
    ValidationLevel,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return Phase4Config(
        assembly_timeout_ms=25000,
        validation_level=ValidationLevel.STANDARD,
        min_confidence_threshold=0.7,
        correlation_id="test-correlation-123",
    )


@pytest.fixture
def sample_paradigm_shift():
    """Create sample paradigm shift."""
    return ParadigmShift(
        shift_type=ParadigmShiftType.STRUCTURAL,
        description="Test structural shift",
        source_patterns=["pattern-1", "pattern-2"],
        phase1_contributions=[{"test": "data"}],
        phase2_contributions=[{"test": "data"}],
        transformation_rules=[{"rule": "test"}],
        confidence=0.85,
        validation_status="validated",
    )


@pytest.fixture
def sample_synthesized_knowledge(sample_paradigm_shift):
    """Create sample synthesized knowledge."""
    return SynthesizedKnowledge(
        knowledge_type="architecture_assembly",
        description="Test knowledge",
        paradigm_shifts=[sample_paradigm_shift],
        confidence=0.82,
        completeness=0.9,
        consistency=0.88,
    )


@pytest.fixture
def sample_assembly(sample_synthesized_knowledge, sample_paradigm_shift):
    """Create sample architecture assembly."""
    return ArchitectureAssembly(
        synthesized_knowledge=sample_synthesized_knowledge,
        paradigm_shifts=[sample_paradigm_shift],
        validation_results=[
            {"validation_type": "completeness", "passed": True},
            {"validation_type": "consistency", "passed": True},
            {"validation_type": "confidence", "passed": True},
            {"validation_type": "aci_reduction", "passed": True, "aci_reduction": 0.35},
        ],
        final_architecture={"architecture_id": "test-arch-1"},
        aci_reduction_achieved=0.35,
        confidence=0.82,
        validation_level=ValidationLevel.STANDARD,
        status=AssemblyStatus.VALIDATED,
    )


# ============================================================================
# TEST: OUTPUT GENERATOR INITIALIZATION
# ============================================================================

def test_output_generator_initialization(config):
    """Test OutputGenerator initializes correctly."""
    generator = OutputGenerator(config)

    assert generator.config == config
    assert generator.logger is not None
    assert isinstance(generator.logger, StructuredLogger)


def test_output_generator_without_logger(config):
    """Test OutputGenerator creates its own logger if none provided."""
    generator = OutputGenerator(config, logger=None)

    assert generator.logger is not None
    assert isinstance(generator.logger, StructuredLogger)


# ============================================================================
# TEST: JSON OUTPUT GENERATION
# ============================================================================

def test_generate_json_output(sample_assembly, config):
    """Test JSON output generation."""
    generator = OutputGenerator(config)
    result = generator.generate(sample_assembly, OutputFormat.JSON)

    assert result is not None
    assert "formatted_output" in result
    assert "metrics" in result
    assert "validation_summary" in result
    assert "predictions" in result
    assert "metadata" in result

    # Check format
    assert result["formatted_output"]["format"] == "json"
    assert "content" in result["formatted_output"]


# ============================================================================
# TEST: MARKDOWN OUTPUT GENERATION
# ============================================================================

def test_generate_markdown_output(sample_assembly, config):
    """Test Markdown output generation."""
    generator = OutputGenerator(config)
    result = generator.generate(sample_assembly, OutputFormat.MARKDOWN)

    assert result is not None
    assert result["formatted_output"]["format"] == "markdown"
    assert "content" in result["formatted_output"]

    # Check Markdown content
    content = result["formatted_output"]["content"]
    assert "# RESE Phase IV: Architecture Assembly" in content
    assert "## Synthesized Knowledge" in content
    assert "## Paradigm Shifts" in content
    assert "## Validation Results" in content


# ============================================================================
# TEST: YAML OUTPUT GENERATION
# ============================================================================

def test_generate_yaml_output(sample_assembly, config):
    """Test YAML output generation."""
    generator = OutputGenerator(config)
    result = generator.generate(sample_assembly, OutputFormat.YAML)

    assert result is not None
    assert result["formatted_output"]["format"] == "yaml"
    assert "content" in result["formatted_output"]

    # Check YAML content
    content = result["formatted_output"]["content"]
    assert "assembly_id:" in content
    assert "status:" in content
    assert "confidence:" in content


# ============================================================================
# TEST: PRETTY OUTPUT GENERATION
# ============================================================================

def test_generate_pretty_output(sample_assembly, config):
    """Test pretty output generation."""
    generator = OutputGenerator(config)
    result = generator.generate(sample_assembly, OutputFormat.PRETTY)

    assert result is not None
    assert result["formatted_output"]["format"] == "pretty"
    assert "content" in result["formatted_output"]

    # Check pretty content
    content = result["formatted_output"]["content"]
    assert "RESE PHASE IV: ARCHITECTURE ASSEMBLY" in content
    assert "SYNTHESIZED KNOWLEDGE" in content
    assert "PARADIGM SHIFTS" in content
    assert "VALIDATION RESULTS" in content


# ============================================================================
# TEST: METRICS EXTRACTION
# ============================================================================

def test_extract_metrics(sample_assembly, config):
    """Test metrics extraction."""
    generator = OutputGenerator(config)
    result = generator.generate(sample_assembly, OutputFormat.JSON)

    metrics = result["metrics"]

    # Check basic metrics
    assert "overall_confidence" in metrics
    assert "aci_reduction_achieved" in metrics
    assert "completeness" in metrics
    assert "consistency" in metrics
    assert "paradigm_shift_count" in metrics
    assert "validation_passed" in metrics

    # Check values
    assert metrics["overall_confidence"] == 0.82
    assert metrics["aci_reduction_achieved"] == 0.35
    assert metrics["paradigm_shift_count"] == 1
    assert metrics["validation_passed"] is True

    # Check paradigm shift metrics
    assert "paradigm_shift_avg_confidence" in metrics
    assert metrics["paradigm_shift_avg_confidence"] == 0.85


# ============================================================================
# TEST: VALIDATION SUMMARY
# ============================================================================

def test_validation_summary(sample_assembly, config):
    """Test validation summary generation."""
    generator = OutputGenerator(config)
    result = generator.generate(sample_assembly, OutputFormat.JSON)

    summary = result["validation_summary"]

    assert "total_checks" in summary
    assert "passed" in summary
    assert "failed" in summary
    assert "status" in summary
    assert "by_type" in summary

    # Check counts
    assert summary["total_checks"] == 4
    assert summary["passed"] == 4
    assert summary["failed"] == 0
    assert summary["status"] == "passed"


# ============================================================================
# TEST: PREDICTIONS GENERATION
# ============================================================================

def test_predictions_generation(sample_assembly, config):
    """Test predictions generation."""
    generator = OutputGenerator(config)
    result = generator.generate(sample_assembly, OutputFormat.JSON)

    predictions = result["predictions"]

    assert "aci_reduction_prediction" in predictions
    assert "paradigm_shift_predictions" in predictions
    assert "constraint_satisfaction_predictions" in predictions

    # Check ACI prediction
    aci_pred = predictions["aci_reduction_prediction"]
    assert "predicted_reduction" in aci_pred
    assert "confidence" in aci_pred
    assert "statistical_significance" in aci_pred
    assert aci_pred["predicted_reduction"] == 0.35

    # Check paradigm shift predictions
    shift_preds = predictions["paradigm_shift_predictions"]
    assert len(shift_preds) == 1
    assert shift_preds[0]["confidence"] == 0.85
    assert shift_preds[0]["testable"] is True


# ============================================================================
# TEST: METADATA GENERATION
# ============================================================================

def test_metadata_generation(sample_assembly, config):
    """Test metadata generation."""
    generator = OutputGenerator(config)
    result = generator.generate(sample_assembly, OutputFormat.JSON)

    metadata = result["metadata"]

    assert "assembly_id" in metadata
    assert "generated_at" in metadata
    assert "output_format" in metadata
    assert "generation_time_seconds" in metadata
    assert "confidence" in metadata
    assert "aci_reduction" in metadata

    # Check values
    assert metadata["assembly_id"] == sample_assembly.assembly_id
    assert metadata["output_format"] == "json"
    assert metadata["confidence"] == 0.82
    assert metadata["aci_reduction"] == 0.35
    assert isinstance(metadata["generation_time_seconds"], float)


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================

def test_generate_without_knowledge(config):
    """Test error handling when assembly has no knowledge."""
    assembly = ArchitectureAssembly(
        synthesized_knowledge=None,
        paradigm_shifts=[],
        aci_reduction_achieved=0.0,
        confidence=0.0,
    )

    generator = OutputGenerator(config)

    with pytest.raises(ValueError, match="Assembly has no synthesized knowledge"):
        generator.generate(assembly, OutputFormat.JSON)


# ============================================================================
# TEST: IDEMPOTENCY
# ============================================================================

def test_generation_idempotency(sample_assembly, config):
    """Test that generation is idempotent (Law of Idempotency)."""
    generator = OutputGenerator(config)

    # Generate twice
    result1 = generator.generate(sample_assembly, OutputFormat.JSON)
    result2 = generator.generate(sample_assembly, OutputFormat.JSON)

    # Check metadata is different (timestamps)
    assert result1["metadata"]["generated_at"] != result2["metadata"]["generated_at"]

    # Check core data is the same
    assert result1["metrics"] == result2["metrics"]
    assert result1["validation_summary"] == result1["validation_summary"]


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
