# Gauntlet System Probe Scripts

## Overview

This directory contains **comprehensive probe scripts** that validate the Gauntlet system's API contracts and verify functionality, as required by **CLAUDE.md Law 2: "Runtime Truth"**.

These probes follow the Federation Constitution's mandate: **"You generally do not trust the documentation. You trust execution."**

## Philosophy

Per CLAUDE.md Section 4, Phase 1 (The Probe):

> Before implementing a feature, you must write a `probe.{sh,py,js}` script that executes the call against the live container. **If the probe fails, the feature does not exist.**

### Probe Purpose

1. **Runtime Verification**: Validate that components actually work at runtime
2. **Contract Validation**: Ensure APIs return expected data structures
3. **Integration Testing**: Verify component interactions
4. **Zero Trust**: Don't assume documentation is correct - test it

## Available Probes

### 1. ML Optimizer Probe (`check_ml_optimizer.sh`)

**Validates**: ML-Based Gauntlet Optimizer functionality

**Tests**:
- Module import verification
- Optimizer instantiation (default and custom)
- GauntletState operations (to_dict, from_dict, to_tuple)
- Basic optimization functionality
- Multiple optimization strategies (Q-Learning, DQN, Genetic, Bayesian)
- API response serialization (to_dict, JSON)
- Factory function

**Component**: `glue/adapters/gauntlet-adapter/src/ml_optimizer.py`

**Usage**:
```bash
./glue/adapters/gauntlet-adapter/probes/check_ml_optimizer.sh
```

**Expected Output**:
```
ML Optimizer Probe
===================
✓ Module import verification
✓ Optimizer instantiation
✓ GauntletState operations
✓ Basic optimization functionality
✓ Optimization strategy variants
✓ API response serialization
✓ Factory function

Total tests: 7
Passed: 7
Failed: 0
```

---

### 2. Predictive Executor Probe (`check_predictive_executor.sh`)

**Validates**: Predictive Gauntlet Executor functionality

**Tests**:
- Module import verification
- Executor instantiation (default and custom thresholds)
- Success prediction API (probability, confidence, risk factors)
- Execution planning (decision logic based on predictions)
- Prediction result serialization
- Execution plan serialization
- Full execution flow with prediction
- Prediction accuracy tracking

**Component**: `glue/adapters/gauntlet-adapter/src/predictive_gauntlet_executor.py`

**Usage**:
```bash
./glue/adapters/gauntlet-adapter/probes/check_predictive_executor.sh
```

---

### 3. Adaptive Learner Probe (`check_adaptive_learner.sh`)

**Validates**: Advanced Adaptive Learner with Deep RL

**Tests**:
- Module import verification
- Learner instantiation (different algorithms)
- Neural network initialization (Q-network, target network)
- Forward pass through network
- Experience replay buffer management
- Learning from experience (backpropagation)
- Action selection (epsilon-greedy policy)
- Model save/load functionality
- Training from historical data
- Factory function

**Component**: `glue/adapters/gauntlet-adapter/src/adaptive_learner.py`

**Usage**:
```bash
./glue/adapters/gauntlet-adapter/probes/check_adaptive_learner.sh
```

---

### 4. Intelligent Orchestrator Probe (`check_intelligent_orchestrator.sh`)

**Validates**: Intelligent Gauntlet Orchestrator with multi-objective optimization

**Tests**:
- Module import verification
- Orchestrator instantiation (objectives, parallelism)
- Orchestration plan creation
- Strategy selection logic (sequential, parallel, adaptive, hierarchical)
- Resource allocation per round
- Stopping conditions configuration
- Plan serialization (to_dict, JSON)
- Async execution flow
- Different optimization objectives (accuracy, time, cost, throughput)
- Statistics tracking

**Component**: `glue/adapters/gauntlet-adapter/src/intelligent_orchestrator.py`

**Usage**:
```bash
./glue/adapters/gauntlet-adapter/probes/check_intelligent_orchestrator.sh
```

---

### 5. WebSocket API Probe (`check_websocket.sh`)

**Validates**: WebSocket API for real-time gauntlet updates

**Tests**:
- Module import verification
- WebSocket event creation (all event types)
- Event serialization/deserialization (JSON round-trip)
- Connection manager initialization
- Connection subscription management
- WebSocket server instantiation
- Event broadcasting methods (async verification)
- All event types validation
- Event round-trip serialization
- Performance testing (serialization speed)

**Component**: `api/gauntlets_websocket.py`

**Usage**:
```bash
./glue/adapters/gauntlet-adapter/probes/check_websocket.sh
```

---

## Running All Probes

To run all gauntlet probes:

```bash
# Run all probes
cd glue/adapters/gauntlet-adapter/probes
for probe in check_*.sh; do
    echo "Running $probe..."
    ./$probe
    echo ""
done
```

Or run individually as needed.

## Probe Structure

Each probe follows this structure:

```bash
#!/bin/bash
# Title, description, tests list
set -e  # Exit on error

# 1. Setup (colors, helpers, paths)
# 2. Test 1: Module Import Verification
# 3. Test 2: Component Instantiation
# 4. Test 3-N: Functional Tests
# 5. Summary Report

# Returns: 0 on success, non-zero on failure
```

### Test Naming Convention

- `Test 1`: Module Import Verification (always first)
- `Test 2`: Component Instantiation
- `Test 3-N`: Specific functionality tests
- Each test uses embedded Python for validation

### Output Format

```
[INFO] Component Name Probe
[INFO] ======================
[INFO] Test 1: Description...
[INFO] ✓ Test passed
[ERROR] ✗ Test failed (if applicable)
...
[INFO] Test Summary
[INFO] ============
Total tests: N
Passed: N
Failed: N
```

## Integration with CLAUDE.md

### Law 2: Runtime Truth

These probes implement **Law 2** of the Federation Constitution:

> **The Mandate**: You generally do not trust the documentation. You trust **execution**.
>
> **The Protocol**: Before implementing a feature, you must write a `probe.{sh,py,js}` script that executes the call against the live container. **If the probe fails, the feature does not exist.**

### Contract Validation (Phase 2)

Per CLAUDE.md Section 4, Phase 2:

> Protecting the Mega-Project from Updates:
> 1. Create: `glue/adapters/{project}/tests/contract.test.ts`
> 2. Assert: Check that the API returns the specific fields we rely on
> 3. Automation: This test runs on container startup. If the contract is violated, the adapter **refuses to start** to prevent data corruption

These probes serve as the **first line of defense** - they verify the contract before integration tests run.

## Continuous Validation

### Pre-Commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run gauntlet probes before committing

echo "Running Gauntlet probes..."
./glue/adapters/gauntlet-adapter/probes/check_ml_optimizer.sh
./glue/adapters/gauntlet-adapter/probes/check_predictive_executor.sh
./glue/adapters/gauntlet-adapter/probes/check_adaptive_learner.sh
./glue/adapters/gauntlet-adapter/probes/check_intelligent_orchestrator.sh
./glue/adapters/gauntlet-adapter/probes/check_websocket.sh

echo "All probes passed!"
```

### CI/CD Integration

Add to CI pipeline:

```yaml
gauntlet-probes:
  script:
    - ./glue/adapters/gauntlet-adapter/probes/check_ml_optimizer.sh
    - ./glue/adapters/gauntlet-adapter/probes/check_predictive_executor.sh
    - ./glue/adapters/gauntlet-adapter/probes/check_adaptive_learner.sh
    - ./glue/adapters/gauntlet-adapter/probes/check_intelligent_orchestrator.sh
    - ./glue/adapters/gauntlet-adapter/probes/check_websocket.sh
```

## Troubleshooting

### Python Path Issues

If probes fail with import errors:

1. Ensure PYTHONPATH includes project root
2. Check that Python executable is available
3. Verify component files exist at expected paths

```bash
# Check Python
which python

# Check component exists
ls -la glue/adapters/gauntlet-adapter/src/

# Test import manually
python -c "from glue.adapters.gauntlet_adapter.src.ml_optimizer import MLBasedGauntletOptimizer"
```

### Permission Denied

Ensure scripts are executable:

```bash
chmod +x glue/adapters/gauntlet-adapter/probes/*.sh
```

### Test Failures

If a test fails:

1. Check the error message for specific assertion
2. Run the embedded Python test directly
3. Verify component implementation hasn't changed
4. Check if dependencies are installed (numpy, etc.)

## Adding New Probes

When adding new components to the Gauntlet system:

1. Create corresponding probe script: `check_{component}.sh`
2. Follow the established structure (colors, helpers, summary)
3. Include at minimum:
   - Module import verification
   - Component instantiation
   - Core functionality test
   - API serialization test
4. Make executable: `chmod +x check_{component}.sh`
5. Update this README

## Dependencies

All probes require:

- **Bash**: For shell script execution
- **Python 3.11+**: For embedded tests
- **NumPy**: For numerical operations (ML optimizer, adaptive learner)

Ensure venv is activated or dependencies are installed:

```bash
# Using venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies if needed
pip install numpy
```

## Contact & Contribution

These probes are part of the OpenEvolve Gauntlet System.

**Maintained By**: OpenEvolve Gauntlet System Team
**Date**: 2026-02-03
**Version**: 1.0

For issues or improvements, follow the Federation Constitution's modification protocols.

---

**Remember**: In the Federation, we don't assume - we verify. Trust execution, not documentation.
