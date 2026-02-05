# RESE Probe Scripts - Quick Reference

## What Are These?

**Runtime verification probes** following CLAUDE.md "Law of Runtime Truth":

> "Before implementing a feature, you must write a probe script that executes the call against the live container. If the probe fails, the feature does not exist."

**We trust EXECUTION, not documentation.**

---

## Available Probes

### Individual Phase Probes

| Phase | Script | Location | Purpose |
|-------|--------|----------|---------|
| I | `check_phase1.sh` | `../rese-phase1/probes/` | Verify Epistemic Audit (13 checks) |
| II | `check_phase2.sh` | `../rese-phase2/probes/` | Verify Isomorphic Mapping |
| III | `check_phase3.sh` | `../rese-phase3/probes/` | Verify MCTS Search (8 checks) |
| IV | `check_phase4.sh` | `../rese-phase4/probes/` | Verify Architecture Assembly (6 checks) |

### Full Pipeline Probe

| Script | Location | Purpose |
|--------|----------|---------|
| `check_full_pipeline.sh` | `./` | Run ALL phase probes, report system health |

---

## Quick Start

### Test Everything
```bash
cd glue/adapters/rese-integration/probes
bash check_full_pipeline.sh
```

### Test Individual Phase
```bash
# Example: Test Phase I
cd glue/adapters/rese-phase1/probes
bash check_phase1.sh
```

---

## Exit Codes

- `0` = All checks passed ✓
- `1` = One or more checks failed ✗

**Never deploy code that fails probes.**

---

## What Gets Checked

### Phase I (Epistemic Audit)
- Executor imports and initializes
- Configuration loads from environment
- ConstraintHardener works
- AssumptionMiner works
- Circuit breaker triggers
- Dead letter queue operates
- Full audit executes end-to-end

### Phase II (Isomorphic Mapping)
- Module imports work
- I_mech scores compute
- Constraint inversion works
- Cross-domain patterns detected

### Phase III (MCTS Search)
- Configuration validates
- Executor initializes all components
- Search executes (UCB1 selection)
- Hypothesis validation (statistical tests)
- Convergence detection (ACI)

### Phase IV (Architecture Assembly)
- Executor and adapter instantiate
- Health check works
- Assembly operation completes
- Schema validation passes

---

## Philosophy

From CLAUDE.md:

> "The Mandate: You generally do not trust the documentation. You trust **execution**."

> "Before implementing a feature, write a `probe.{sh,py,js}` script that executes the call against the live container."

> "If you cannot get a 200 OK from the shell, you cannot write the code."

**These probes embody that philosophy.**

---

## Status

| Probe | Status | Notes |
|-------|--------|-------|
| Phase I | ✅ Created | 9.5KB, 13 checks |
| Phase II | ✅ Created | 4.3KB |
| Phase III | ✅ Created | 12KB, 8 checks |
| Phase IV | ✅ Created | 8.9KB, 6 checks |
| Full Pipeline | ✅ Created | 8.1KB, runs all phases |

**All probes created and executable.** Integration work in progress.

---

## Contact

Questions? See `RESE_PROBE_SUMMARY.md` for detailed documentation.
