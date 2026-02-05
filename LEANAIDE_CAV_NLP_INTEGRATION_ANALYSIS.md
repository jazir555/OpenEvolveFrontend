# LeanAide + CAV-NLP Integration Analysis

## Executive Summary

With CAV-NLP now integrated as the primary mathematical formalization system, we need to redefine LeanAide's role. The two systems have **overlapping formalization capabilities** but **complementary strengths**.

| System | Primary Role | Strengths | Status |
|--------|--------------|-----------|--------|
| **CAV-NLP** | Primary Formalization | 100% test coverage, canonicalization, Z3 validation, CEGIS | ✅ Production-ready |
| **LeanAide** | Verification & Elaboration | Lean server integration, proof automation, documentation | ⚠️ Re-evaluate role |
| **Z3-to-Lean** | Proof Verification | Proof certificate validation | 🔧 70% complete |

---

## Current State Analysis

### LeanAide Components (35 files, ~25,000 lines)

| File | Lines | Purpose | Overlap with CAV-NLP |
|------|-------|---------|---------------------|
| `leanaide_client.py` | ~800 | Async client for LeanAide server | **HIGH** - Translation tasks |
| `lean4_integration.py` | ~1000 | Verification service | **LOW** - Verification only |
| `leanaide_evolution.py` | 3048 | Evolution integration | **MEDIUM** - Workflow |
| `leanaide_mcts*.py` | ~7000 | MCTS strategies | **HIGH** - Search strategies |
| `leanaide_strategies.py` | 2392 | Proof strategies | **MEDIUM** - Tactics |
| `leanaide_mdap*.py` | ~5500 | MDAP integration | **HIGH** - Workflow |
| `leanaide_maker.py` | 1845 | Maker integration | **MEDIUM** - Code gen |
| `leanaide_continuous_math.py` | 1685 | Continuous math | **LOW** - Domain specific |

### CAV-NLP Components (7 core files, ~9,000 lines)

| File | Lines | Purpose | Unique Strength |
|------|-------|---------|----------------|
| `flexible_semantic_parsing.py` | 654 | Parse NL/LaTeX | Production-tested parser |
| `dependency_dag.py` | 514 | Extract dependencies | Complete DAG extraction |
| `z3_semantic_synthesis.py` | 3650 | Synthesize to Lean | Z3-validated synthesis |
| `canonical_lean_generator.py` | 519 | Generate Lean code | Canonical form output |
| `z3_canonicalizer.py` | 309 | Canonicalization | Z3 UNSAT equivalence |
| `cegis_learner.py` | 1129 | Learning loop | Continuous improvement |

---

## Functional Overlap Analysis

### 1. Formalization (NL/LaTeX → Lean 4)

| Feature | LeanAide | CAV-NLP | Winner |
|---------|----------|---------|--------|
| NL parsing | ✅ Basic | ✅ Advanced (Z3-validated) | **CAV-NLP** |
| LaTeX parsing | ✅ Limited | ✅ Full | **CAV-NLP** |
| Dependency extraction | ⚠️ Partial | ✅ Complete DAG | **CAV-NLP** |
| Canonicalization | ❌ None | ✅ Z3-based | **CAV-NLP** |
| Test coverage | ⚠️ Unknown | ✅ 20/20 (100%) | **CAV-NLP** |
| Production readiness | ⚠️ Beta | ✅ Production | **CAV-NLP** |

**Decision**: CAV-NLP becomes primary formalization engine.

### 2. Verification (Lean 4 → Result)

| Feature | LeanAide | CAV-NLP | Winner |
|---------|----------|---------|--------|
| Lean server integration | ✅ Full | ❌ None | **LeanAide** |
| Proof automation | ✅ Yes | ❌ No | **LeanAide** |
| Counterexample gen | ⚠️ Via Z3 | ✅ Via Z3 | **Tie** |
| Batch verification | ✅ Yes | ❌ No | **LeanAide** |

**Decision**: LeanAide continues as verification service.

### 3. Elaboration & Documentation

| Feature | LeanAide | CAV-NLP | Winner |
|---------|----------|---------|--------|
| Lean code elaboration | ✅ Yes | ❌ No | **LeanAide** |
| Theorem documentation | ✅ Yes | ❌ No | **LeanAide** |
| Definition naming | ✅ Yes | ❌ No | **LeanAide** |
| Math query | ✅ Yes | ❌ No | **LeanAide** |

**Decision**: LeanAide retains these unique capabilities.

---

## Proposed Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenEvolve Unified System                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Formalization Pipeline                           │   │
│  │                                                                       │   │
│  │   Input (NL/LaTeX)                                                    │   │
│  │        │                                                              │   │
│  │        ▼                                                              │   │
│  │   ┌──────────────────────────────────────────────────────────────┐   │   │
│  │   │  PRIMARY: CAV-NLP Formalization Engine                        │   │   │
│  │   │  - flexible_semantic_parsing.py                               │   │   │
│  │   │  - dependency_dag.py                                          │   │   │
│  │   │  - z3_semantic_synthesis.py                                   │   │   │
│  │   │  - canonical_lean_generator.py                                │   │   │
│  │   └──────────────────────────────────────────────────────────────┘   │   │
│  │        │                                                              │   │
│  │        ▼                                                              │   │
│  │   Output: Canonical Lean 4 Code                                       │   │
│  │        │                                                              │   │
│  │        ▼                                                              │   │
│  │   ┌──────────────────────────────────────────────────────────────┐   │   │
│  │   │  SECONDARY: LeanAide Elaboration Service                      │   │   │
│  │   │  - Elaborate Lean code                                        │   │   │
│  │   │  - Generate documentation                                     │   │   │
│  │   │  - Suggest theorem names                                      │   │   │
│  │   └──────────────────────────────────────────────────────────────┘   │   │
│  │        │                                                              │   │
│  │        ▼                                                              │   │
│  │   Output: Elaborated, Documented Lean 4                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Verification Pipeline                            │   │
│  │                                                                       │   │
│  │   ┌──────────────────────────────────────────────────────────────┐   │   │
│  │   │  LeanAide Verification Service                                │   │   │
│  │   │  - Lean server integration                                    │   │   │
│  │   │  - Proof automation (aesop, etc.)                             │   │   │
│  │   │  - Batch verification                                         │   │   │
│  │   └──────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │   ┌──────────────────────────────────────────────────────────────┐   │   │
│  │   │  Z3-to-Lean Proof Verification (optional)                     │   │   │
│  │   │  - Proof certificate validation                               │   │   │
│  │   │  - Formal proof checking                                      │   │   │
│  │   └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Unified Interface                                │   │
│  │                                                                       │   │
│  │   UnifiedMathService                                                  │   │
│  │   ├── formalize(text) → Lean 4 code     [uses CAV-NLP]               │   │
│  │   ├── elaborate(code) → Elaborated code [uses LeanAide]              │   │
│  │   ├── verify(code) → VerificationResult [uses LeanAide]              │   │
│  │   ├── prove(goal) → Proof + Code        [uses both]                  │   │
│  │   └── document(code) → Documentation    [uses LeanAide]              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LeanAide Roles After Integration

### Role 1: Verification Service (Primary)

**Kept**: `lean4_integration.py` - Verification service

```python
# CAV-NLP generates code, LeanAide verifies it
from openevolve.cav_nlp_integration import CAVNLPFormalizer
from openevolve.lean4_integration import LeanAideService

formalizer = CAVNLPFormalizer()
lean_service = LeanAideService()

# Formalize with CAV-NLP
lean_code = formalizer.formalize("For all x > 0, x + 1 > 1")

# Verify with LeanAide
result = await lean_service.verify(lean_code)
```

### Role 2: Elaboration Service (Primary)

**Kept**: Elaboration tasks from `leanaide_client.py`

```python
# Elaborate CAV-NLP generated code
elaborated = await leanaide_client.elaborate(lean_code)
```

### Role 3: Documentation Generation (Primary)

**Kept**: Documentation tasks

```python
# Generate docs for CAV-NLP generated theorems
docs = await leanaide_client.generate_documentation(lean_code)
```

### Role 4: Proof Automation (Complementary)

**Modified**: Use for proof completion when CAV-NLP synthesis is insufficient

```python
# CAV-NLP generates sketch, LeanAide completes proof
sketch = cav_nlp.generate_proof_sketch(theorem)
completed = await lean_service.complete_proof(sketch)
```

### Role 5: Legacy Translation (Deprecated)

**Removed**: Direct NL→Lean translation

- `TRANSLATE_THM` → Use CAV-NLP instead
- `TRANSLATE_DEF` → Use CAV-NLP instead
- Keep for backward compatibility only

---

## Implementation Plan

### Phase 1: Unified Service Interface (Week 1)

Create `unified_math_service.py`:

```python
class UnifiedMathService:
    """
    Unified interface for mathematical formalization and verification.
    
    Uses:
    - CAV-NLP for formalization (primary)
    - LeanAide for verification (primary)
    - LeanAide for elaboration/documentation
    """
    
    def __init__(self):
        self.cav_nlp = CAVNLPFormalizer()
        self.lean_service = LeanAideService()
        self.lean_client = LeanAideClient()
    
    async def formalize(self, text: str) -> FormalizationResult:
        """Formalize natural language to Lean 4 using CAV-NLP."""
        # Primary: CAV-NLP
        result = self.cav_nlp.formalize(text)
        
        # Secondary: Elaborate with LeanAide
        if result.success:
            result.elaborated = await self.lean_service.elaborate(result.code)
        
        return result
    
    async def verify(self, code: str) -> VerificationResult:
        """Verify Lean 4 code using LeanAide."""
        return await self.lean_service.verify(code)
    
    async def prove(self, theorem: str) -> ProofResult:
        """Prove theorem using hybrid CAV-NLP + LeanAide approach."""
        # CAV-NLP: Generate proof sketch
        sketch = self.cav_nlp.generate_proof_sketch(theorem)
        
        # LeanAide: Complete the proof
        completed = await self.lean_service.complete_proof(sketch)
        
        return ProofResult(sketch=sketch, proof=completed)
```

### Phase 2: LeanAide Integration Refactor (Week 2-3)

1. **Update `leanaide_client.py`**:
   - Mark translation tasks as deprecated
   - Keep verification/elaboration tasks
   - Add unified service integration

2. **Update workflow integrations**:
   - `leanaide_evolution.py` - Use unified service
   - `leanaide_mcts*.py` - Use unified service
   - `leanaide_mdap*.py` - Use unified service

3. **Create adapter layer**:
   - `leanaide_cav_nlp_bridge.py` - Bridge between systems

### Phase 3: Deprecation & Migration (Week 4)

1. Add deprecation warnings to LeanAide translation methods
2. Create migration guide
3. Update documentation
4. Notify users

---

## File Migration Strategy

### Keep (LeanAide Unique Capabilities)

| File | Reason |
|------|--------|
| `leanaide_client.py` | Server client, but mark translation as deprecated |
| `lean4_integration.py` | Verification service - PRIMARY ROLE |
| `leanaide_api_routes.py` | API routes for verification/elaboration |
| `leanaide_config.py` | Configuration |

### Refactor (Integration Points)

| File | Action |
|------|--------|
| `leanaide_evolution.py` | Use unified service instead of direct translation |
| `leanaide_mcts*.py` | Use CAV-NLP for formalization in search |
| `leanaide_strategies.py` | Keep proof strategies, update formalization calls |
| `leanaide_mdap*.py` | Use unified service |
| `leanaide_maker.py` | Use CAV-NLP for code generation |

### Deprecate (Replaced by CAV-NLP)

| Pattern | Replacement |
|---------|-------------|
| `LeanAideClient.translate_thm()` | `CAVNLPFormalizer.formalize()` |
| `LeanAideClient.translate_def()` | `CAVNLPFormalizer.formalize()` |
| Direct NL→Lean in workflows | Unified service formalize() |

---

## API Compatibility

### Before (LeanAide-only)

```python
from leanaide_client import LeanAideClient, TaskType

client = LeanAideClient()
result = await client.execute_task(
    TaskType.TRANSLATE_THM,
    {"text": "For all x > 0, x^2 > 0"}
)
lean_code = result.data["lean_code"]
```

### After (Unified Service)

```python
from openevolve.unified_math_service import UnifiedMathService

service = UnifiedMathService()

# Formalize with CAV-NLP (primary)
result = await service.formalize("For all x > 0, x^2 > 0")
lean_code = result.code

# Verify with LeanAide (complementary)
verification = await service.verify(lean_code)

# Elaborate with LeanAide (complementary)
elaborated = await service.elaborate(lean_code)
```

### Backward Compatibility

```python
# Old code still works with deprecation warning
from leanaide_client import LeanAideClient
client = LeanAideClient()
result = await client.translate_thm("For all x > 0, x^2 > 0")
# → Delegates to CAV-NLP, warns about deprecation
```

---

## Benefits of This Integration

1. **Best of Both Worlds**:
   - CAV-NLP's robust formalization (100% test coverage)
   - LeanAide's verification and elaboration capabilities

2. **Reduced Maintenance**:
   - Single formalization path (CAV-NLP)
   - Clear separation of concerns

3. **Improved Quality**:
   - Z3-validated canonicalization
   - Dependency DAG extraction
   - CEGIS learning loop

4. **Backward Compatible**:
   - Existing code continues to work
   - Gradual migration path

5. **Future-Proof**:
   - CAV-NLP's arXiv corpus learning
   - Continuous improvement via CEGIS

---

## Conclusion

**Recommendation**: 
1. CAV-NLP becomes the **primary formalization engine**
2. LeanAide becomes the **verification and elaboration service**
3. Create a **unified interface** that uses both optimally
4. **Deprecate** LeanAide's translation capabilities over 6 months
5. **Preserve** all verification, elaboration, and documentation features

This gives us a best-of-breed system: CAV-NLP's robust formalization combined with LeanAide's Lean server integration for verification.
