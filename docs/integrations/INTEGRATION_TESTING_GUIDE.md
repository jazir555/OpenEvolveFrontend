# E2E Integration Testing Quick Reference

## Running Integration Tests

### All Integration Tests
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese
python -m pytest tests/test_integration/test_all_stage_integrations.py -v -o addopts=""
```

### Specific Stage Tests
```bash
# Stage 1: Prompt Analysis
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestStage1Integration -v -o addopts=""

# Stage 2: Isomorphic Mapping
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestStage2Integration -v -o addopts=""

# Stage 3: Monte Carlo
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestStage3Integration -v -o addopts=""

# Stage 5: Physics/Logic
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestStage5Integration -v -o addopts=""

# Stage 6: Error Analysis
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestStage6Integration -v -o addopts=""

# Stage 7: Red/Blue Team
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestStage7Integration -v -o addopts=""

# Stage 8: Predictive Models
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestStage8Integration -v -o addopts=""

# Stage 9: Convergence
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestStage9Integration -v -o addopts=""
```

### End-to-End Pipeline Tests
```bash
# Full pipeline execution
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestEndToEndPipeline::test_full_pipeline_execution -v -o addopts=""

# Pipeline performance
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestEndToEndPipeline::test_pipeline_performance -v -o addopts=""
```

### Data Flow Tests
```bash
# Stage 1 → Stage 3
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestDataFlow::test_stage1_to_stage3_data_flow -v -o addopts=""

# Stage 3 → Stage 5
python -m pytest tests/test_integration/test_all_stage_integrations.py::TestDataFlow::test_stage3_to_stage5_data_flow -v -o addopts=""
```

---

## Test Results Summary

**Latest Run**: 2025-12-31
**Total Tests**: 28
**Passed**: 28 (100%)
**Failed**: 0
**Execution Time**: ~2.4 seconds

### Stage-by-Stage Results

| Stage | Tests | Status | Notes |
|-------|-------|--------|-------|
| Stage 1 (Prompt Analysis) | 4 | ✅ PASS | SCE + Φ₁.₅ + Φ₂ working |
| Stage 2 (Isomorphic Mapping) | 2 | ✅ PASS | Ψ₂ + Ψ₃ + I_mech (graceful degradation) |
| Stage 3 (Monte Carlo) | 3 | ✅ PASS | Γ₁ + Γ₂ + N_max working |
| Stage 5 (Physics/Logic) | 3 | ✅ PASS | LLTL + Φ₂ working |
| Stage 6 (Error Analysis) | 3 | ✅ PASS | Φ₁.₅ + Γ₁ working |
| Stage 7 (Red/Blue Team) | 3 | ✅ PASS | Φ₁.₅ + adversarial working |
| Stage 8 (Predictive Models) | 3 | ✅ PASS | Δ₁ + Δ₂ working |
| Stage 9 (Convergence) | 3 | ✅ PASS | Γ₁ + Δ₃ working |
| E2E Pipeline | 2 | ✅ PASS | Full pipeline validated |
| Data Flow | 2 | ✅ PASS | Inter-stage data flow validated |

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    E2E INVENTION ENGINE                     │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │ Stage 1  │───▶│ Stage 2  │───▶│ Stage 3  │            │
│  │  Prompt  │    │ Isomorphic│    │   Monte  │            │
│  │ Analysis │    │  Mapping  │    │  Carlo   │            │
│  └──────────┘    └──────────┘    └──────────┘            │
│        │                                │                  │
│        │                           ┌──────────┐            │
│        └──────────────────────────▶│ Stage 5  │            │
│                                    │ Physics/ │            │
│                                    │   Logic  │            │
│                                    └──────────┘            │
│                                         │                  │
│                                    ┌──────────┐            │
│                                    │ Stage 6  │            │
│                                    │   Error  │            │
│                                    │ Analysis │            │
│                                    └──────────┘            │
│                                         │                  │
│                                    ┌──────────┐            │
│                                    │ Stage 7  │            │
│                                    │ Red/Blue │            │
│                                    │   Team   │            │
│                                    └──────────┘            │
│                                         │                  │
│                                    ┌──────────┐            │
│                                    │ Stage 8  │            │
│                                    │Predictive│            │
│                                    │  Models  │            │
│                                    └──────────┘            │
│                                         │                  │
│                                    ┌──────────┐            │
│                                    │ Stage 9  │            │
│                                    │Convergence│           │
│                                    └──────────┘            │
└─────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │   RESE MODULES  │
                        ├─────────────────┤
                        │ SCE (Φ₁)        │
                        │ Φ₁.₅            │
                        │ Φ₂ (Biases)     │
                        │ Ψ₂, Ψ₃         │
                        │ I_mech          │
                        │ Γ₁, Γ₂          │
                        │ LLTL            │
                        │ Δ₁, Δ₂, Δ₃      │
                        │ DITO            │
                        └─────────────────┘
```

---

## Common Issues and Solutions

### Import Errors
**Issue**: `ImportError: cannot import name 'TacitAssumptionMiner'`
**Solution**: Use `Phi15Engine` instead (fixed in all integration files)

### Attribute Errors
**Issue**: `'BiasReport' object has no attribute 'detected_biases'`
**Solution**: Use `bias_report.detections` instead (fixed in Stage 1)

### Missing Parameters
**Issue**: `ErrorReport.__init__() missing 1 required positional argument: 'timestamp'`
**Solution**: Add `timestamp=datetime.now()` to ErrorReport initialization

### Enum Value Mismatches
**Issue**: `AttributeError: COMPLETED` (enum value doesn't exist)
**Solution**: Check valid enum values and use appropriate status

---

## Module Dependencies

### Required (Core)
- ✅ `symbolic_constraint_engine` (SCE)
- ✅ `tacit_assumption_miner` (Φ₁.₅)
- ✅ `cognitive_biases` (Φ₂)
- ✅ `mcts_search` (Γ₂)

### Optional (Graceful Degradation)
- ⚠️ `imech` (Isomorphic Mechanism) - Stage 2 degrades if missing
- ⚠️ `gamma1` (ACI Calculator) - Uses fallback if missing
- ⚠️ `psi3` (Constraint Inverter) - Stage 2 degrades if missing

### Not Required
- ✅ Stage 5 (LLTL) - Self-contained
- ✅ Stage 7 (Adversarial) - Self-contained
- ✅ Stage 8 (Δ₁, Δ₂) - Self-contained
- ✅ Stage 9 (Δ₃) - Self-contained

---

## Test File Locations

### Main Test File
```
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\tests\test_integration\test_all_stage_integrations.py
```

### Integration Module Files
```
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\integrations\stage1.py
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\integrations\stage2.py
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\integrations\stage3.py
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\integrations\stage5.py
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\integrations\stage6.py
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\integrations\stage7.py
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\integrations\stage8.py
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\integrations\stage9.py
```

### Validation Report
```
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\E2E_INTEGRATION_VALIDATION_REPORT.md
```

---

## Quick Status Check

```bash
# Run all tests and show summary
python -m pytest tests/test_integration/test_all_stage_integrations.py -v -o addopts="" --tb=no | tail -1

# Expected output:
# ============================= 28 passed in 2.XXs ==============================
```

---

## Next Steps

1. ✅ All integration tests passing
2. ✅ Data flow validated
3. ✅ Error handling verified
4. 📋 Review full validation report: `E2E_INTEGRATION_VALIDATION_REPORT.md`
5. 🚀 System ready for production use

---

**Last Updated**: 2025-12-31
**Status**: ✅ ALL SYSTEMS OPERATIONAL
