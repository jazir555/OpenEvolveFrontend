# RESE Testing - Quick Start

Quick reference guide for running RESE tests.

---

## Essential Commands

### Run All Tests
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

### Run by Type
```bash
# Unit tests
pytest tests/ -m unit -v

# Integration tests
pytest tests/ -m integration -v

# Performance tests
pytest tests/ -m performance -v

# Validation tests
pytest tests/ -m validation -v
```

### Run by Phase
```bash
# Phase I (Φ₁.₅)
pytest tests/ -m phase1 -v

# Phase II (I_mech, Ψ₃)
pytest tests/ -m phase2 -v

# Phase III (Γ₁)
pytest tests/ -m phase3 -v

# Phase IV (Δ₃, DITO)
pytest tests/ -m phase4 -v
```

### Run Specific Test File
```bash
pytest tests/test_phi15.py -v
pytest tests/test_integration/test_full_pipeline.py -v
pytest tests/test_validation/test_key_innovations.py -v
```

### Run Specific Test
```bash
pytest tests/test_phi15.py::TestPhi15Engine::test_engine_initialization -v
```

### Options
```bash
# Verbose with print statements
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x -v

# Run in parallel
pytest tests/ -n auto

# Show local variables on failure
pytest tests/ -v -l

# Run slow tests
pytest tests/ --runslow
```

---

## KEY INNOVATIONS Validation

```bash
# Run all validation tests
pytest tests/test_validation/ -v -s

# Individual innovations
pytest tests/test_validation/test_key_innovations.py::TestPhi15Validation -v
pytest tests/test_validation/test_key_innovations.py::TestImechValidation -v
pytest tests/test_validation/test_key_innovations.py::TestGamma1Validation -v
pytest tests/test_validation/test_key_innovations.py::TestDelta3Validation -v
pytest tests/test_validation/test_key_innovations.py::TestPsi3Validation -v
pytest tests/test_validation/test_key_innovations.py::TestDitoValidation -v
```

---

## Performance Testing

```bash
# Load testing
pytest tests/test_performance/ -k "load" -v

# Stress testing
pytest tests/test_performance/ -k "stress" -v

# Benchmarking
pytest tests/test_performance/ --benchmark-only

# Memory profiling
pytest tests/test_performance/ -k "memory" -v
```

---

## Coverage Reports

```bash
# HTML report
pytest tests/ --cov=. --cov-report=html

# Terminal report
pytest tests/ --cov=. --cov-report=term-missing

# XML report
pytest tests/ --cov=. --cov-report=xml

# Check threshold
pytest tests/ --cov=. --cov-fail-under=80
```

---

## CI/CD

### Trigger Tests
```bash
# Push to trigger
git push origin main

# Pull request to trigger
gh pr create

# Manual trigger
gh workflow run test_pipeline.yml

# View status
gh workflow list
gh run list --workflow=test_pipeline.yml
```

---

## Troubleshooting

### Import Errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Database Lock
```bash
rm -rf tests/test_databases/*.db
```

### Debug Mode
```bash
pytest tests/ -v --pdb
pytest tests/ -v --log-cli-level=DEBUG
```

---

## Documentation

- **Full Guide:** `TESTING_GUIDE.md`
- **QA Procedures:** `QA_PROCEDURES.md`
- **Bug Report:** `BUG_REPORT_TEMPLATE.md`
- **Summary:** `TEST_INFRASTRUCTURE_SUMMARY.md`
- **README:** `README.md`

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total Tests | 500+ |
| Coverage | 82% |
| Unit Tests | 300+ |
| Integration Tests | 150+ |
| Performance Tests | 30+ |
| Validation Tests | 20+ |

---

## Contact

- **QA Lead:** Agent Z2 (Testing/QA Specialist)
- **Issues:** GitHub Issues
- **Documentation:** `TESTING_GUIDE.md`

---

**Last Updated:** 2025-12-31
**Version:** 1.0.0
