"""
Integration tests for RESE Phase IV

End-to-end tests covering:
- Complete Phase IV pipeline
- Integration with Phase I, II, III outputs
- Full verification workflow
- Output generation with all components
- Error handling across components

Following CLAUDE.md principles:
- Law of Runtime Truth: Test actual pipeline behavior
- Law of Idempotency: Verify reproducible results
- Circuit Breaker: Test failure recovery

Author: RESE Team
"""

import pytest
import sys
import os
import time
from datetime import datetime, timezone

# Add src and schemas to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "schemas"))

from src.phase4_executor import ArchitectureAssemblyExecutor
from src.adapter import Phase4Adapter
from src.output_generator import OutputGenerator, OutputFormat
from src.predictive_validator import PredictiveValidator, StatisticalTest
from src.result_verifier import (
    ResultVerifier,
    ConstraintSatisfactionCheck,
    ProofCompletenessCheck,
    ACIReductionCheck,
)
from rese_phase4_schemas import (
    ArchitectureAssembly,
    SynthesizedKnowledge,
    ParadigmShift,
    ParadigmShiftType,
    EpistemicAuditResult,
    IsomorphicMappingResult,
    MCTSRefinementResult,
    Phase4Config,
    AssemblyStatus,
    ValidationLevel,
    IntegrationStrategy,
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
        integration_strategy=IntegrationStrategy.SYNTHESIZE,
        min_confidence_threshold=0.7,
        correlation_id="test-correlation-integration-123",
    )


@pytest.fixture
def phase1_result():
    """Create Phase I result."""
    return {
        "audit_id": "audit-001",
        "constraints": [
            {"constraint_id": "c1", "type": "equation", "description": "Constraint 1"},
            {"constraint_id": "c2", "type": "inequality", "description": "Constraint 2"},
        ],
        "contradictions": [],
        "tacit_assumptions": [{"id": "a1", "description": "Assumption 1"}],
        "cognitive_biases": [],
        "validation_status": "validated",
        "confidence": 0.85,
        "metadata": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def phase2_result():
    """Create Phase II result."""
    return {
        "mapping_id": "map-001",
        "isomorphisms": [
            {"isomorphism_id": "iso1", "similarity": 0.9},
            {"isomorphism_id": "iso2", "similarity": 0.85},
        ],
        "functional_dependencies": [],
        "domain_mappings": [],
        "similarity_scores": [],
        "validation_results": [],
        "confidence": 0.78,
        "metadata": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def phase3_result():
    """Create Phase III result."""
    return {
        "refinement_id": "ref-001",
        "search_results": [],
        "validated_hypotheses": [
            {
                "hypothesis_id": "hyp1",
                "statement": "Test hypothesis",
                "status": "validated",
                "confidence": 0.88,
            }
        ],
        "convergence_metrics": {"iterations": 100},
        "aci_reduction": 0.35,
        "statistical_validation": [],
        "confidence": 0.82,
        "metadata": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def phase1_patterns():
    """Create Phase I patterns."""
    return [
        {
            "pattern_id": "p1-1",
            "type": "structural",
            "description": "Structural pattern from Phase I",
            "confidence": 0.85,
            "transformation_rules": [{"rule": "transform1"}],
        }
    ]


@pytest.fixture
def phase2_patterns():
    """Create Phase II patterns."""
    return [
        {
            "pattern_id": "p2-1",
            "type": "functional",
            "description": "Functional pattern from Phase II",
            "confidence": 0.78,
            "transformation_rules": [{"rule": "transform2"}],
        }
    ]


@pytest.fixture
def phase3_patterns():
    """Create Phase III patterns."""
    return [
        {
            "pattern_id": "p3-1",
            "type": "causal",
            "description": "Causal pattern from Phase III",
            "confidence": 0.88,
            "transformation_rules": [{"rule": "transform3"}],
        }
    ]


# ============================================================================
# TEST: COMPLETE PHASE IV EXECUTION
# ============================================================================

def test_complete_phase_iv_execution(
    config,
    phase1_result,
    phase2_result,
    phase3_result,
    phase1_patterns,
    phase2_patterns,
    phase3_patterns
):
    """Test complete Phase IV execution."""
    executor = ArchitectureAssemblyExecutor(config)

    assembly = executor.execute(
        phase1_result=phase1_result,
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase1_patterns=phase1_patterns,
        phase2_patterns=phase2_patterns,
        phase3_patterns=phase3_patterns,
    )

    # Check assembly
    assert isinstance(assembly, ArchitectureAssembly)
    assert assembly.assembly_id is not None
    assert assembly.synthesized_knowledge is not None
    assert len(assembly.paradigm_shifts) > 0
    assert assembly.status == AssemblyStatus.VALIDATED
    assert assembly.aci_reduction_achieved == 0.35


# ============================================================================
# TEST: ADAPTER INTEGRATION
# ============================================================================

def test_adapter_integration(
    config,
    phase1_result,
    phase2_result,
    phase3_result,
    phase1_patterns,
    phase2_patterns,
    phase3_patterns
):
    """Test adapter integration."""
    adapter = Phase4Adapter(config)

    request = {
        "request_id": "req-001",
        "phase1_result": phase1_result,
        "phase2_result": phase2_result,
        "phase3_result": phase3_result,
        "phase1_patterns": phase1_patterns,
        "phase2_patterns": phase2_patterns,
        "phase3_patterns": phase3_patterns,
    }

    response = adapter.assemble_architecture(request)

    # Check response
    assert response["status"] == "success"
    assert "assembly" in response
    assert "metadata" in response
    assert response["metadata"]["validation_passed"] is True


# ============================================================================
# TEST: OUTPUT GENERATION INTEGRATION
# ============================================================================

def test_output_generation_integration(
    config,
    phase1_result,
    phase2_result,
    phase3_result,
    phase1_patterns,
    phase2_patterns,
    phase3_patterns
):
    """Test output generation integration."""
    executor = ArchitectureAssemblyExecutor(config)
    assembly = executor.execute(
        phase1_result=phase1_result,
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase1_patterns=phase1_patterns,
        phase2_patterns=phase2_patterns,
        phase3_patterns=phase3_patterns,
    )

    # Generate outputs in all formats
    output_gen = OutputGenerator(config)

    json_output = output_gen.generate(assembly, OutputFormat.JSON)
    assert json_output is not None
    assert json_output["formatted_output"]["format"] == "json"

    markdown_output = output_gen.generate(assembly, OutputFormat.MARKDOWN)
    assert markdown_output is not None
    assert markdown_output["formatted_output"]["format"] == "markdown"

    pretty_output = output_gen.generate(assembly, OutputFormat.PRETTY)
    assert pretty_output is not None
    assert pretty_output["formatted_output"]["format"] == "pretty"


# ============================================================================
# TEST: PREDICTIVE VALIDATION INTEGRATION
# ============================================================================

def test_predictive_validation_integration(
    config,
    phase1_result,
    phase2_result,
    phase3_result,
    phase1_patterns,
    phase2_patterns,
    phase3_patterns
):
    """Test predictive validation integration."""
    executor = ArchitectureAssemblyExecutor(config)
    assembly = executor.execute(
        phase1_result=phase1_result,
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase1_patterns=phase1_patterns,
        phase2_patterns=phase2_patterns,
        phase3_patterns=phase3_patterns,
    )

    # Validate predictions
    validator = PredictiveValidator(config, test_type=StatisticalTest.WILCOXON)

    incumbent_aci = [0.85, 0.82, 0.88, 0.90, 0.87, 0.83, 0.86, 0.89, 0.84, 0.88]
    new_aci = [0.55, 0.52, 0.58, 0.50, 0.56, 0.53, 0.54, 0.51, 0.57, 0.52]

    validation_result = validator.validate(assembly, incumbent_aci, new_aci)

    # Check validation
    assert validation_result.is_valid is True
    assert validation_result.aci_reduction > 0
    assert validation_result.statistical_significance["is_significant"] is True


# ============================================================================
# TEST: RESULT VERIFICATION INTEGRATION
# ============================================================================

def test_result_verification_integration(
    config,
    phase1_result,
    phase2_result,
    phase3_result,
    phase1_patterns,
    phase2_patterns,
    phase3_patterns
):
    """Test result verification integration."""
    executor = ArchitectureAssemblyExecutor(config)
    assembly = executor.execute(
        phase1_result=phase1_result,
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase1_patterns=phase1_patterns,
        phase2_patterns=phase2_patterns,
        phase3_patterns=phase3_patterns,
    )

    # Verify results
    verifier = ResultVerifier(config)
    verification_result = verifier.verify(assembly)

    # Check verification
    assert verification_result.verification_id is not None
    assert verification_result.is_valid is True
    assert len(verification_result.results) > 0
    assert verification_result.checks_passed > 0
    assert verification_result.checks_failed == 0


# ============================================================================
# TEST: END-TO-END WORKFLOW
# ============================================================================

def test_end_to_end_workflow(
    config,
    phase1_result,
    phase2_result,
    phase3_result,
    phase1_patterns,
    phase2_patterns,
    phase3_patterns
):
    """Test complete end-to-end workflow."""
    # Step 1: Execute assembly
    executor = ArchitectureAssemblyExecutor(config)
    assembly = executor.execute(
        phase1_result=phase1_result,
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase1_patterns=phase1_patterns,
        phase2_patterns=phase2_patterns,
        phase3_patterns=phase3_patterns,
    )
    assert assembly.status == AssemblyStatus.VALIDATED

    # Step 2: Generate output
    output_gen = OutputGenerator(config)
    output = output_gen.generate(assembly, OutputFormat.JSON)
    assert output["metrics"]["validation_passed"] is True

    # Step 3: Validate predictions
    validator = PredictiveValidator(config)
    incumbent_aci = [0.85, 0.82, 0.88, 0.90, 0.87, 0.83, 0.86, 0.89, 0.84, 0.88]
    new_aci = [0.55, 0.52, 0.58, 0.50, 0.56, 0.53, 0.54, 0.51, 0.57, 0.52]
    pred_result = validator.validate(assembly, incumbent_aci, new_aci)
    assert pred_result.is_valid is True

    # Step 4: Verify results
    verifier = ResultVerifier(config)
    verify_result = verifier.verify(assembly)
    assert verify_result.is_valid is True


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================

def test_error_handling_invalid_request(config):
    """Test error handling for invalid request."""
    adapter = Phase4Adapter(config)

    # Missing all phase data
    request = {
        "request_id": "req-invalid",
    }

    with pytest.raises(ValueError, match="at least one phase"):
        adapter.assemble_architecture(request)


def test_error_handling_timeout(config, phase1_result):
    """Test timeout handling."""
    # Set very short timeout
    config.assembly_timeout_ms = 1
    executor = ArchitectureAssemblyExecutor(config)

    # This should timeout
    with pytest.raises(TimeoutError):
        executor.execute(
            phase1_result=phase1_result,
            phase1_patterns=[],
        )


# ============================================================================
# TEST: CIRCUIT BREAKER
# ============================================================================

def test_circuit_breaker(config):
    """Test circuit breaker functionality."""
    adapter = Phase4Adapter(config)

    # Check initial state
    health = adapter.health_check()
    assert health["circuit_breaker_state"] == "closed"
    assert health["status"] == "healthy"

    # Simulate failures
    for _ in range(6):
        try:
            adapter.assemble_architecture({"request_id": "req-fail"})
        except:
            pass

    # Check circuit breaker is open
    health = adapter.health_check()
    assert health["circuit_breaker_state"] == "open"
    assert health["failure_count"] >= 5


# ============================================================================
# TEST: HEALTH CHECK
# ============================================================================

def test_health_check(config):
    """Test health check endpoint."""
    adapter = Phase4Adapter(config)

    health = adapter.health_check()

    assert "status" in health
    assert "circuit_breaker_state" in health
    assert "failure_count" in health
    assert "config" in health
    assert "timestamp" in health


# ============================================================================
# TEST: PARTIAL DATA HANDLING
# ============================================================================

def test_partial_data_handling(config, phase1_result, phase1_patterns):
    """Test handling of partial data (only Phase I)."""
    executor = ArchitectureAssemblyExecutor(config)

    assembly = executor.execute(
        phase1_result=phase1_result,
        phase1_patterns=phase1_patterns,
    )

    # Should still work but with lower completeness
    assert assembly is not None
    assert assembly.synthesized_knowledge is not None
    assert assembly.synthesized_knowledge.completeness < 1.0


# ============================================================================
# TEST: IDEMPOTENCY
# ============================================================================

def test_idempotency(
    config,
    phase1_result,
    phase2_result,
    phase3_result,
    phase1_patterns,
    phase2_patterns,
    phase3_patterns
):
    """Test idempotency (Law of Idempotency)."""
    executor = ArchitectureAssemblyExecutor(config)

    # Execute twice with same inputs
    assembly1 = executor.execute(
        phase1_result=phase1_result,
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase1_patterns=phase1_patterns,
        phase2_patterns=phase2_patterns,
        phase3_patterns=phase3_patterns,
    )

    assembly2 = executor.execute(
        phase1_result=phase1_result,
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase1_patterns=phase1_patterns,
        phase2_patterns=phase2_patterns,
        phase3_patterns=phase3_patterns,
    )

    # Different assemblies (different IDs)
    assert assembly1.assembly_id != assembly2.assembly_id

    # But same core results
    assert assembly1.aci_reduction_achieved == assembly2.aci_reduction_achieved
    assert assembly1.confidence == assembly2.confidence
    assert len(assembly1.paradigm_shifts) == len(assembly2.paradigm_shifts)


# ============================================================================
# TEST: PERFORMANCE
# ============================================================================

def test_performance(
    config,
    phase1_result,
    phase2_result,
    phase3_result,
    phase1_patterns,
    phase2_patterns,
    phase3_patterns
):
    """Test performance characteristics."""
    executor = ArchitectureAssemblyExecutor(config)

    start_time = time.time()
    assembly = executor.execute(
        phase1_result=phase1_result,
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase1_patterns=phase1_patterns,
        phase2_patterns=phase2_patterns,
        phase3_patterns=phase3_patterns,
    )
    elapsed = time.time() - start_time

    # Should complete in reasonable time
    assert elapsed < 5.0  # 5 seconds max

    # And produce valid assembly
    assert assembly.status == AssemblyStatus.VALIDATED


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
