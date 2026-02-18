# 🚀 Decomposition/Recomposition - Quick Reference Guide
**Everything you need to know in 5 minutes**

**Last Updated:** 2026-01-27
**System Status:** 🟡 **76% Complete** - Production ready with critical gaps

---

## ⚡ Quick Start

```python
# 1. Decompose a problem
from decomposition_engine import DecompositionEngine
engine = DecompositionEngine()
plan = engine.decompose(problem)

# 2. Solve sub-problems (using CrewAI bridge)
from decomposition_crewai_bridge import execute_phase_2_solve
result = execute_phase_2_solve(plan, execution_method="roma_mdap_maker")

# 3. Reassemble solution
from problem_recomposition import SolutionAssembler
assembler = SolutionAssembler()
solution = assembler.assemble_solution(plan, result['solutions'])

# 4. Validate
from final_solution import FinalSolutionValidator
validator = FinalSolutionValidator()
validation = validator.validate_final_solution(solution, problem)
```

---

## 🎯 Component Selection Guide

| Use Case | Use This | File |
|----------|----------|------|
| **Mathematical Proofs** | LeanAide Decomposition | `leanaide_decomposition_architecture.md` |
| **Zero-Error Critical Tasks** | ROMA-MDAP-MAKER | `ROMA_MDAP_MAKER_QUICK_REFERENCE.md` |
| **Visual Bubble Flows** | Flow Decomposition | `FLOW_DECOMPOSITION_QUICK_REFERENCE.md` |
| **Research Problems** | Sovereign Workflow | `Decomposition_Workflow.md` |
| **General Problems** | Hybrid (Auto-Select) | `decomposition_engine_lean_quick_reference.md` |
| **Production Code** | ROMA Deterministic | See below |
| **Documentation** | ROMA Creative | See below |

---

## 🔴 Critical Issues (Top 3)

### 1. Core Decomposition Broken (60% functional)
**Impact:** Only 2/5 engines working
**Fix:** Fix `DependencyDecomposition` class in `decomposition_engine.py`
**Effort:** 4 hours
**Status:** 🔴 BLOCKER

### 2. No Automated Tests (0% coverage)
**Impact:** Cannot deploy to production
**Fix:** Create pytest infrastructure + 110 tests
**Effort:** 2-3 weeks
**Status:** 🔴 BLOCKER

### 3. No Persistent Storage (100% memory-only)
**Impact:** All data lost on restart
**Fix:** Implement database integration
**Effort:** 1-2 weeks
**Status:** 🟡 HIGH PRIORITY

---

## 📊 System Health

```
┌────────────────────────────────────────────┐
│  Component                 │ Status │ Score│
├────────────────────────────────────────────┤
│  Core Decomposition      │ 🟡    │  60% │
│  Recomposition           │ 🟢    │  85% │
│  Subproblem Management    │ 🟢    │  85% │
│  Integration Bridges     │ 🟢    │  85% │
│  Documentation           │ 🟢    │  80% │
├────────────────────────────────────────────┤
│  OVERALL                 │ 🟡    │  76% │
└────────────────────────────────────────────┘
```

---

## 🛠️ Assembly Strategies (Recomposition)

### Available Strategies

1. **Hierarchical** ⭐ Recommended for complex dependencies
   - Builds dependency tree
   - Topological sort
   - Quality: 95%

2. **Linear** ⚡ Fastest
   - Simple sequential
   - No dependency optimization
   - Quality: 90%

3. **Parallel** 🚀 Most scalable
   - ThreadPoolExecutor-based
   - Concurrent assembly
   - Quality: 90%

4. **Adaptive** 🤖 Smart
   - Analyzes problem characteristics
   - Auto-selects best strategy
   - Quality: 92%

5. **ROMA Deterministic** 🔒 Production code
   - AI-powered structure planning
   - Sub-solutions inserted VERBATIM
   - Code integrity preserved
   - Quality: 95%

6. **ROMA Creative** ✨ Enhanced flow
   - AI-powered rewriting
   - Enhanced coherence
   - Use for documentation
   - Quality: 95%

### Usage

```python
# For code (preserves integrity)
solution = assembler.assemble_solution(
    plan, sub_solutions,
    assembly_strategy="roma",
    roma_deterministic=True  # ← Verbatim insertion
)

# For documentation (enhanced flow)
solution = assembler.assemble_solution(
    plan, sub_solutions,
    assembly_strategy="roma",
    roma_deterministic=False,  # ← May rewrite
    roma_temperature=0.7
)
```

---

## 🧩 Integration Bridges

### ROMA Integration ✅
```python
from roma_decomposition_hybrid import ROMADecompositionHybrid

hybrid = ROMADecompositionHybrid()
result = hybrid.execute_hybrid_workflow(
    problem=problem,
    max_depth=2,
    execution_mode="recursive"
)
```

### CrewAI Bridge (Recommended - MIT) ✅
```python
from decomposition_crewai_bridge import execute_phase_2_solve

result = execute_phase_2_solve(
    decomposition_plan=plan,
    execution_method="roma_mdap_maker"  # Zero-error
)
```

### LeanAide Integration (Mathematics) ✅
```python
from leanaide_decomposition_integration import LeanDecomposer

decomposer = LeanDecomposer()
plan = await decomposer.decompose_mathematical_problem(
    problem="Prove that every prime > 3 is of form 6k±1",
    strategy="hybrid"
)
```

---

## 📈 Quality Metrics

All solutions scored on 0.0-1.0 scale:

- **Completeness:** Are all sub-problems addressed?
- **Consistency:** Are there contradictions?
- **Coherence:** Does it flow well?
- **Integration Quality:** Seamless transitions?
- **Overall Score:** Weighted combination

```python
solution.quality_metrics.overall_score  # 0.0 to 1.0
```

**Target:** ≥0.80 for production

---

## 🚨 Common Issues & Fixes

### Issue 1: Import Error for LeanAide
```python
# ❌ ERROR: ModuleNotFoundError: leanaide_evolution

# ✅ FIX: Use graceful degradation
try:
    from leanaide_evolution import LeanProofEvolutionEngine
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAide not available - using fallback")
```

### Issue 2: PersistentDecompositionEngine Won't Instantiate
```python
# ❌ ERROR: Missing WorkflowState

# ✅ FIX: Add to sovereign_data_models.py
@dataclass
class WorkflowState:
    workflow_id: str
    current_stage: str
    status: str
    # ... see master roadmap for full definition
```

### Issue 3: HybridDecomposition Fails
```python
# ❌ ERROR: DependencyDecomposition class missing

# ✅ FIX: Extract lines 436-598 from decomposition_engine.py
# into proper DependencyDecomposition class
# See master roadmap lines 411-598
```

---

## ✅ Checklist: Ready for Production?

- [ ] All P0 tasks complete (master roadmap)
- [ ] Test coverage ≥80%
- [ ] All data persists across restart
- [ ] Security audit passed
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Monitoring configured

**If all checked:** ✅ **Ready to deploy!**

**If any unchecked:** ❌ **Not ready** - see master roadmap

---

## 📚 Key Files

**Core:**
- `problem_decomposition.py` ✅ Working (100%)
- `problem_recomposition.py` ✅ Production-ready (85%)
- `sovereign_data_models.py` ✅ Excellent data models

**Bridges:**
- `roma_decomposition_hybrid.py` ✅ ROMA integration
- `decomposition_crewai_bridge.py` ✅ CrewAI (MIT - recommended)
- `leanaide_decomposition_integration.py` ✅ Mathematics

**Documentation:**
- `DECOMPOSITION_RECOMPOSITION_MASTER_ROADMAP.md` ⭐ **THIS FILE**
- `Decomposition_Workflow.md` - Sovereign workflow design
- `leanaide_decomposition_architecture.md` - Math decomposition

---

## 🎓 Best Practices

### 1. Always Use MIT-Licensed Components
```python
# ✅ GOOD (MIT)
from decomposition_crewai_bridge import execute_workflow

# ❌ AVOID (AGPL)
from decomposition_crewai_bridge import execute_workflow
```

### 2. Use Deterministic ROMA for Code
```python
assembler.assemble_solution(
    plan, solutions,
    assembly_strategy="roma",
    roma_deterministic=True  # Preserves code
)
```

### 3. Always Validate Solutions
```python
validator = FinalSolutionValidator()
result = validator.validate_final_solution(solution, problem)
if not result.is_valid:
    logger.error(f"Validation failed: {result.validation_message}")
```

### 4. Handle Optional Dependencies Gracefully
```python
try:
    import optional_module
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    logger.warning("Optional module not available")
```

---

## 🚀 Next Steps

### For Developers:
1. Read `DECOMPOSITION_RECOMPOSITION_MASTER_ROADMAP.md` (this file)
2. Complete P0 tasks (Week 1-2)
3. Set up pytest infrastructure (Week 3)
4. Implement persistent storage (Week 5-6)

### For Project Managers:
1. Review 8-week implementation timeline
2. Allocate 2-3 developers
3. Track progress using P0/P1/P2 task list
4. Plan production deployment for Week 8

### For Architects:
1. Review data model consistency issue
2. Approve database schema design
3. Review security audit results
4. Approve production configuration

---

## 📞 Support

**Questions?** See these resources:

1. **Master Roadmap:** `DECOMPOSITION_RECOMPOSITION_MASTER_ROADMAP.md`
2. **Architecture:** `leanaide_decomposition_architecture.md`
3. **Workflow Design:** `Decomposition_Workflow.md`
4. **Integration Guides:** `docs/guides/`

**Need Help?**
- Check the troubleshooting section (Appendix C in master roadmap)
- Review test files for examples
- Read inline documentation in source files

---

**Quick Reference Version:** 1.0
**Generated:** 2026-01-27
**For Full Details:** See `DECOMPOSITION_RECOMPOSITION_MASTER_ROADMAP.md`
