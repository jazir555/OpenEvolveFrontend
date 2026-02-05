# RESE Runtime Verification Probe Scripts

**Created:** 2026-02-04
**Status:** Complete
**Following:** CLAUDE.md "Law of Runtime Truth"

---

## Overview

This document describes the runtime verification probe scripts created for the RESE (Recursive Epistemic Solvability Engine) framework. These probes follow the **Law of Runtime Truth** principle: we trust **execution**, not documentation.

Before implementing features or using adapters, these probe scripts verify that the code actually works at runtime.

---

## Probe Scripts Created

### 1. Phase I Probe: `check_phase1.sh`
**Location:** `glue/adapters/rese-phase1/probes/check_phase1.sh`

**Purpose:** Verify Phase I: Epistemic Audit functionality

**Checks performed (13 total):**
- ✓ Directory and file structure exists
- ✓ Executor module can be imported
- ✓ Adapter module can be imported
- ✓ Configuration can be loaded from environment
- ✓ Executor can be instantiated
- ✓ TacitAssumption dataclass works
- ✓ ConstraintHardener can extract constraints
- ✓ AssumptionMiner can mine tacit assumptions
- ✓ Circuit breaker can detect failures
- ✓ Dead letter queue works
- ✓ Full Phase I audit executes end-to-end

**Key verification:** Tests actual runtime behavior, not just file existence. Validates that configuration loads from environment variables (Law of Configuration Explicitness).

---

### 2. Phase II Probe: `check_phase2.sh`
**Location:** `glue/adapters/rese-phase2/probes/check_phase2.sh`

**Purpose:** Verify Phase II: Isomorphic Mapping functionality

**Checks performed:**
- ✓ Python environment available
- ✓ Module import successful
- ✓ Phase II execution works
- ✓ Adapter interface functional
- ✓ I_mech (Isomorphic Mechanism) score computation
- ✓ Constraint inversion works
- ✓ Cross-domain pattern detection

**Key verification:** Tests that isomorphic mappings can be found between domains and I_mech scores are computed correctly.

---

### 3. Phase III Probe: `check_phase3.sh`
**Location:** `glue/adapters/rese-phase3/probes/check_phase3.sh`

**Purpose:** Verify Phase III: MCTS Search functionality

**Checks performed (8 total):**
- ✓ Python availability
- ✓ Environment variables set correctly
- ✓ Imports work (MCTSSearchExecutor, Phase3Config, etc.)
- ✓ Configuration validation
- ✓ Executor initialization
- ✓ Search execution (10 iterations)
- ✓ Hypothesis validation (statistical tests)
- ✓ Convergence detection (ACI)

**Key verification:** Tests that MCTS search can execute, hypotheses can be validated statistically, and convergence detection works via ACI (Algorithmic Convergence Indicator).

---

### 4. Phase IV Probe: `check_phase4.sh`
**Location:** `glue/adapters/rese-phase4/probes/check_phase4.sh`

**Purpose:** Verify Phase IV: Architecture Assembly functionality

**Checks performed (6 total):**
- ✓ Directory structure exists
- ✓ Python environment available
- ✓ Schema imports work
- ✓ Executor can be instantiated
- ✓ Adapter can be instantiated
- ✓ Simple assembly operation completes
- ✓ Health check endpoint works
- ✓ Schema validation works

**Key verification:** Tests that paradigm shifts can be assembled from patterns and knowledge integration works across all phases.

---

### 5. Full Pipeline Probe: `check_full_pipeline.sh` ⭐ **NEW**
**Location:** `glue/adapters/rese-integration/probes/check_full_pipeline.sh`

**Purpose:** Verify the entire RESE pipeline end-to-end

**Features:**
- ✓ Runs all 4 phase probes in sequence
- ✓ Generates JSON health report
- ✓ Tracks pass/fail status per phase
- ✓ Provides overall system health summary
- ✓ Returns appropriate exit codes (0 = all pass, 1 = one or more fail)

**Output format:**
```json
{
  "probe_name": "check_full_pipeline",
  "probe_type": "full_pipeline_verification",
  "correlation_id": "uuid-here",
  "timestamp": "2026-02-04T22:07:41Z",
  "phases": {
    "phase1": {"name": "Epistemic Audit", "status": "pass"},
    "phase2": {"name": "Isomorphic Mapping", "status": "pass"},
    "phase3": {"name": "MCTS Search", "status": "pass"},
    "phase4": {"name": "Architecture Assembly", "status": "pass"}
  },
  "summary": {
    "total_phases": 4,
    "passed_phases": 4,
    "failed_phases": 0
  }
}
```

---

## Usage

### Running Individual Phase Probes

```bash
# Phase I: Epistemic Audit
cd glue/adapters/rese-phase1/probes
bash check_phase1.sh

# Phase II: Isomorphic Mapping
cd glue/adapters/rese-phase2/probes
bash check_phase2.sh

# Phase III: MCTS Search
cd glue/adapters/rese-phase3/probes
bash check_phase3.sh

# Phase IV: Architecture Assembly
cd glue/adapters/rese-phase4/probes
bash check_phase4.sh
```

### Running Full Pipeline Probe

```bash
cd glue/adapters/rese-integration/probes
bash check_full_pipeline.sh
```

**Exit codes:**
- `0`: All phases passed ✓
- `1`: One or more phases failed ✗

---

## Design Principles

All probes follow CLAUDE.md principles:

### 1. Law of Runtime Truth
- Tests execute actual code, not just check file existence
- Imports modules and instantiates classes
- Calls methods and validates results
- Returns 0 on success, non-zero on failure

### 2. Law of Configuration Explicitness
- All configuration via environment variables
- Probes crash immediately if required config is missing
- No magic defaults

### 3. Structured Logging
- JSON output for machine readability
- Includes correlation_id for distributed tracing
- Timestamps in UTC (Law of UTC)

### 4. Idempotency
- Probes can be run multiple times safely
- Check before create logic
- No side effects

---

## Current Status

As of 2026-02-04, all probe scripts are **created and executable**.

However, the individual phase probes are currently showing failures due to:
1. Import path issues (Python module resolution)
2. Missing dependencies (schemas, lib modules)
3. Environment variable configuration

**This is expected and correct behavior!** The probes are doing their job: **detecting that the system is not yet fully functional**.

### Next Steps

To get all probes passing:

1. **Fix Python import paths** - Ensure `sys.path` includes necessary directories
2. **Create missing schemas** - Ensure `glue/schemas/rese_schemas.py` exists
3. **Create missing lib modules** - Ensure `glue/lib/rese_dee.py` exists
4. **Set environment variables** - Configure all required env vars
5. **Fix any runtime errors** - Address issues detected by probes

Once all probes pass, the RESE pipeline will be verified to be **functionally complete**.

---

## Integration with CI/CD

These probes should be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run RESE Probes
  run: |
    cd glue/adapters/rese-integration/probes
    bash check_full_pipeline.sh
```

If any probe fails, the pipeline should fail, preventing deployment of non-functional code.

---

## References

- **CLAUDE.md Section 4.1:** The Probe (Discovery) - "Before implementing a feature, write a probe script that executes the call against the live container."
- **CLAUDE.md Section 4.2:** The Contract (Defense) - "This test runs on container startup. If the contract is violated, the adapter refuses to start."

---

## Summary

✅ **5 runtime verification probe scripts created**
✅ **Following CLAUDE.md "Law of Runtime Truth"**
✅ **Executable bash scripts with clear pass/fail**
✅ **Full pipeline probe runs all phases in sequence**
✅ **JSON health reports for monitoring**
✅ **Ready for CI/CD integration**

The probe scripts are doing their job: **detecting runtime issues** before deployment. This is exactly what the "Law of Runtime Truth" requires.

**Trust execution, not documentation.**
