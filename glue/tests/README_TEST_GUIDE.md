# RESE Pipeline Testing Guide

Quick reference for running RESE pipeline tests.

## Quick Start

```bash
# Run complete end-to-end test
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
python glue/tests/test_rese_complete_pipeline.py
```

## Test Files

### Main Test Suite
- **File:** `glue/tests/test_rese_complete_pipeline.py`
- **Purpose:** Complete 4-phase pipeline integration test
- **Duration:** ~12 seconds
- **Coverage:** All phases + all integrations

### Existing Tests (Phase-Specific)
- `glue/adapters/rese-phase1/tests/test_*.py`
- `glue/adapters/rese-phase2/tests/test_*.py`
- `glue/adapters/rese-phase3/tests/test_*.py`
- `glue/adapters/rese-lltl/tests/test_*.py`

## Test Results

### Results Files
- **JSON:** `RESE_PIPELINE_VERIFICATION_REPORT.json`
- **Markdown:** `RESE_PIPELINE_VERIFICATION_REPORT.md`
- **Summary:** `RESE_TEST_DELIVERABLES_SUMMARY.md`

### Reading Results
1. Check markdown report for human-readable summary
2. Check JSON report for detailed metrics
3. Check summary report for executive overview

## Test Phases

### Phase I: Epistemic Audit
- Tests constraint hardening
- Tests assumption mining
- Tests debiasing (Φ₂)
- Tests contradiction detection (SCE)
- Tests red team protocol

### Phase II: Isomorphic Mapping
- Tests structure identification
- Tests FDG construction
- Tests I_mech calculation
- Tests constraint inversion
- Tests Z3 behavioral equivalence

### Phase III: MCTS Search
- Tests search tree construction
- Tests UCB1 selection
- Tests Z3 constraint checking
- Tests ACI calculation
- Tests convergence detection

### Phase IV: Architecture Assembly
- Tests paradigm shift assembly
- Tests knowledge integration
- Tests architecture validation
- Tests ACI reduction

## Troubleshooting

### Import Errors
```bash
# Ensure paths are correct
export PYTHONPATH="${PYTHONPATH}:$(pwd)/glue/schemas:$(pwd)/glue/lib"
```

### Encoding Errors
```bash
# Fix Unicode encoding
export PYTHONIOENCODING=utf-8
```

### Z3 Not Available
```bash
# Install Z3 solver
pip install z3-solver
```

## Continuous Integration

### Run in CI/CD
```bash
# Set environment variables
export PHASE1_TIMEOUT_MS=15000
export PHASE2_TIMEOUT_MS=20000
export PHASE3_TIMEOUT_MS=30000
export PHASE4_TIMEOUT_MS=25000

# Run test
python glue/tests/test_rese_complete_pipeline.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Some tests failed"
    exit 1
fi
```

## Performance Benchmarks

Expected execution times:
- Phase I: <50ms
- Phase II: <50ms
- Phase III: <5000ms
- Phase IV: <100ms
- **Total:** <5200ms (5.2 seconds)

Current actual times:
- Phase I: 9ms ✅
- Phase II: 2ms ✅
- Phase III: 165ms ✅
- Phase IV: 4ms (failed)
- **Total:** 180ms ✅

## Support

For issues or questions:
1. Check the markdown report for detailed analysis
2. Check the JSON report for specific error details
3. Review phase-specific test logs
4. Consult RESE Technical Manual

## Next Steps

1. Fix Phase IV bug (see report)
2. Install Z3 solver for formal verification
3. Implement health check endpoints
4. Run tests again to verify 95%+ success rate
