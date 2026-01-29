# OpenEvolve Parameter Migration Dashboard

## Migration Status Overview

```
PROGRESS TRACKER
================================================================================
Phase 1: Quick Wins                    [░░░░░░░░░░] 0% (0/19 files)
Phase 2: Session State Refactoring     [░░░░░░░░░░] 0% (0/21 files)
Phase 3: ParameterManager Migration     [░░░░░░░░░░] 0% (0/17 files)
Phase 4: Direct Parameter Migration     [░░░░░░░░░░] 0% (0/80 files)
Phase 5: Core Files                    [░░░░░░░░░░] 0% (0/4 files)

TOTAL PROGRESS: [░░░░░░░░░░] 0% (0/144 files)
================================================================================
```

## Pattern Distribution

```
Direct Parameter Definitions  ████████████████████ 80 files (56%)
Old Evolution Import          ████                 23 files (16%)
Session State Usage           ███                  21 files (15%)
ParameterManager Import       ███                  17 files (12%)
Old Adversarial Import        ██                   13 files (9%)
```

## Risk Assessment Matrix

```
CRITICAL RISK (Core Files)     ████                 4 files  (3%)  ⚠️
HIGH RISK                      █                    1 file   (1%)  ⚠️
MEDIUM RISK                    ████████████████████ 75 files (52%)
LOW RISK                       ███████████████████  64 files (44%) ✓
```

## Quick Reference Guides

### 1. Quick Win Files (Start Here!)
**Effort: 1-2 lines per file | Risk: LOW | Time: 1-2 days**

```
Priority Files:
✓ demo_evolution_maker.py
✓ evolution_adapter.py
✓ test_critical_blockers_resolved.py
✓ test_error_handling.py
✓ verify_fix.py
... and 14 more test/demo files
```

**Action:** Replace `from evolution import` with `from openevolve_imports import`

---

### 2. Session State Files
**Effort: 5-10 lines per file | Risk: MEDIUM | Time: 3-5 days**

```
Priority Files:
⚠ mainlayout.py (UI main layout)
⚠ sidebar.py (Parameter UI)
⚠ configuration_system.py (Config management)
⚠ bubblelabs_ui_component.py (UI component)
... and 17 more files
```

**Action:** Replace `st.session_state[]` with `UnifiedConfiguration.from_session()`

---

### 3. ParameterManager Files
**Effort: 5-10 lines per file | Risk: MEDIUM | Time: 2-3 days**

```
Priority Files:
⚠ adversarial.py (Core - see Phase 5)
⚠ evolution.py (Core - see Phase 5)
⚠ base_configuration.py (Core - see Phase 5)
⚠ unified_configuration.py (Core - see Phase 5)
... and 13 more files
```

**Action:** Replace `from parameter_manager import ParameterManager`

---

### 4. Direct Parameter Files
**Effort: 20+ lines per file | Risk: MEDIUM-HIGH | Time: 7-10 days**

```
High Priority:
⚠⚠ adversarial_maker_integration.py
⚠⚠ adversarial_mdap_mcts.py
⚠⚠ blue_team.py
⚠⚠ evaluator_team.py
⚠⚠ leanaide_mdap.py
⚠⚠ mcts_evolved_policies.py
⚠⚠ openevolve_bubblelabs_api.py
... and 73 more files
```

**Action:** Replace function parameters with `config: UnifiedConfiguration`

---

### 5. Core Files (DO LAST!)
**Effort: 50+ lines per file | Risk: CRITICAL | Time: 5-7 days**

```
⚠️⚠️⚠️ CRITICAL - DO NOT MODIFY WITHOUT COMPREHENSIVE TESTS ⚠️⚠️⚠️

1. evolution.py (4,012 lines)
   - Core evolution engine
   - 272 parameter definitions
   - All evolution algorithms

2. adversarial.py (2,544 lines)
   - Adversarial testing engine
   - Integration with evolution
   - Team system coordination

3. base_configuration.py
   - Base configuration classes
   - Parameter validation
   - Type definitions

4. unified_configuration.py
   - Unified configuration system
   - ParameterManager integration
   - Session state management
```

## Migration Checklists

### Phase 1 Checklist
- [ ] Create feature branch: `feature/phase1-migration`
- [ ] Update all test files with openevolve_imports
- [ ] Update demo files with openevolve_imports
- [ ] Update adapter files with openevolve_imports
- [ ] Run unit tests: `pytest tests/`
- [ ] Verify no import errors
- [ ] Submit PR for review

### Phase 2 Checklist
- [ ] Create feature branch: `feature/phase2-session-state`
- [ ] Audit all st.session_state usage
- [ ] Replace with UnifiedConfiguration.from_session()
- [ ] Update UI components
- [ ] Test Streamlit app locally
- [ ] Verify state persistence
- [ ] Submit PR for review

### Phase 3 Checklist
- [ ] Create feature branch: `feature/phase3-param-manager`
- [ ] Replace ParameterManager imports
- [ ] Update configuration creation
- [ ] Remove deprecated patterns
- [ ] Run integration tests
- [ ] Verify backward compatibility
- [ ] Submit PR for review

### Phase 4 Checklist
- [ ] Create feature branch: `feature/phase4-direct-params`
- [ ] Create config classes for each module
- [ ] Replace direct parameters systematically
- [ ] Update MCP tools
- [ ] Update integration points
- [ ] Run full test suite
- [ ] Performance testing
- [ ] Submit PR for review

### Phase 5 Checklist
- [ ] Create feature branch: `feature/phase5-core-files`
- [ ] CRITICAL: Create comprehensive test suite FIRST
- [ ] Set up feature flags
- [ ] Migrate evolution.py incrementally
- [ ] Migrate adversarial.py incrementally
- [ ] Migrate base_configuration.py
- [ ] Migrate unified_configuration.py
- [ ] Run full regression suite
- [ ] Load testing
- [ ] Security review
- [ ] Submit PR for review

## Code Pattern Reference

### Import Migration

**OLD:**
```python
from evolution import run_evolution_loop, EvolutionConfiguration
from adversarial import run_comprehensive_adversarial_testing
```

**NEW:**
```python
from openevolve_imports import (
    EvolutionAPI,
    AdversarialAPI,
    EVOLUTION_AVAILABLE,
    ADVERSARIAL_AVAILABLE
)

if EVOLUTION_AVAILABLE:
    result = EvolutionAPI.run_evolution_loop(content, config)
```

---

### Parameter Migration

**OLD:**
```python
def evolve_content(
    content: str,
    max_iterations: int = 10,
    temperature: float = 0.7,
    population_size: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elite_count: int = 2,
    # ... 265 more parameters
):
    config = {
        'max_iterations': max_iterations,
        'temperature': temperature,
        # ... duplicate all parameters
    }
```

**NEW:**
```python
def evolve_content(
    content: str,
    config: Optional[UnifiedConfiguration] = None
):
    if config is None:
        config = UnifiedConfiguration.get_default()

    # Access via config.max_iterations, config.temperature, etc.
    # No duplication, single source of truth
```

---

### Session State Migration

**OLD:**
```python
max_iter = st.session_state.get('max_iterations', 10)
temp = st.session_state.get('temperature', 0.7)
pop_size = st.session_state.get('population_size', 50)

config = {
    'max_iterations': max_iter,
    'temperature': temp,
    'population_size': pop_size
}
```

**NEW:**
```python
config = UnifiedConfiguration.from_session()
# config automatically loads from session state with defaults
# Access via config.max_iterations, config.temperature, etc.
```

## Success Metrics

### Quantitative Goals
- [ ] 100% of files using openevolve_imports
- [ ] 0 direct parameter definitions in production code
- [ ] 100% backward compatibility maintained
- [ ] <5% performance degradation acceptable
- [ ] 0 critical bugs post-migration
- [ ] Test coverage >80%

### Qualitative Goals
- [ ] Cleaner, more maintainable codebase
- [ ] Easier onboarding for new developers
- [ ] Centralized configuration management
- [ ] Better type safety and validation
- [ ] Improved documentation

## Rollback Procedures

### If Phase 1 Fails:
```bash
git checkout main
git branch -D feature/phase1-migration
# Loss: 1-2 days work, low risk
```

### If Phase 2 Fails:
```bash
git checkout main
git branch -D feature/phase2-session-state
# Loss: 3-5 days work, medium risk
# Session state changes can be user-facing
```

### If Phase 3 Fails:
```bash
git checkout main
git branch -D feature/phase3-param-manager
# Loss: 2-3 days work, medium risk
# Affects configuration system
```

### If Phase 4 Fails:
```bash
git checkout main
git branch -D feature/phase4-direct-params
# Loss: 7-10 days work, high risk
# Significant code changes across many files
```

### If Phase 5 Fails:
```bash
git checkout main
git branch -D feature/phase5-core-files
# Loss: 5-7 days work, CRITICAL risk
# CORE ENGINE FILES - SYSTEM-WIDE IMPACT
# MUST HAVE COMPREHENSIVE ROLLBACK PLAN
```

## Contact & Support

**Migration Team:**
- Project Lead: [To be assigned]
- Core Systems Owner: [To be assigned]
- QA Lead: [To be assigned]

**Documentation:**
- Full Analysis: `PARAMETER_MIGRATION_ANALYSIS.md`
- File Inventory: `COMPLETE_FILE_INVENTORY.md`
- Migration Guide: `MIGRATION_GUIDE_COMPLETE.md`

**Issue Tracking:**
- GitHub Issues: [Link to project board]
- Slack Channel: #[To be created]

---

**Last Updated:** 2026-01-03
**Next Review:** After Phase 1 completion
**Status:** Planning Complete - Ready to Execute Phase 1
