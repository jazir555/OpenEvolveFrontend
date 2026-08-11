# Z3-CAV-NLP Integration Complete Documentation

> **Comprehensive guide to the CAV-NLP (Canonical Arithmetic Verification via Natural Language Processing) integration with Z3 workflows throughout OpenEvolve.**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Files Modified/Created](#2-files-modifiedcreated)
3. [New Integration Modules](#3-new-integration-modules)
4. [Integration Points](#4-integration-points)
5. [New Capabilities Added](#5-new-capabilities-added)
6. [Usage Examples](#6-usage-examples)
7. [Configuration Reference](#7-configuration-reference)
8. [Migration Guide](#8-migration-guide)
9. [Testing & Verification](#9-testing--verification)
10. [Next Steps](#10-next-steps)

---

## 1. Executive Summary

### What Was Integrated

The Z3-CAV-NLP integration connects **CAV-NLP** (Canonical Arithmetic Verification via Natural Language Processing) with **Z3 SMT solver** workflows across the OpenEvolve platform. This integration enables natural language mathematical statements to be automatically formalized, verified, and proved using a hybrid approach combining Z3's efficient SMT solving with Lean 4's formal verification capabilities.

### Key Integration Components

| Component | Purpose |
|-----------|---------|
| **EnhancedZ3Solver** | Drop-in replacement for Z3 Solver with CAV-NLP capabilities |
| **UnifiedMathService** | Central service for formalization and verification |
| **LeanAideCAVNLPBridge** | Migration bridge from LeanAide to CAV-NLP |
| **MCP Tools** | 5 new MCP tools for external AI system integration |
| **BubbleLabs Nodes** | 4 enhanced nodes with CAV-NLP capabilities |
| **Evolution Fitness** | CAV-NLP enhanced fitness evaluation |

### Why CAV-NLP Enhances Z3 Workflows

**Before CAV-NLP:**
```python
# Manual formalization required
solver = z3.Solver()
x, y = z3.Ints('x y')
solver.add(x > 0, y > 0, x + y > 10)  # Manual constraint encoding
```

**After CAV-NLP:**
```python
# Natural language formalization
solver = EnhancedZ3Solver()
constraint = solver.formalize_constraint(
    "x and y are positive integers whose sum exceeds 10"
)
solver.add(constraint)
```

### Key Benefits

1. **Natural Language Input**: Write mathematical statements in plain English or LaTeX
2. **Hybrid Verification**: Dual verification using Z3 (speed) + Lean (rigor)
3. **Canonical Forms**: Automatic canonicalization for equivalence detection
4. **Proof Export**: Export Z3 results to Lean 4 for formal verification
5. **Higher Confidence**: Hybrid confidence scoring (up to 100%)
6. **Drop-in Compatibility**: Works with existing Z3 code

---

## 2. Files Modified/Created

### New Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `openevolve/z3_cav_nlp_integration.py` | 1,808 | Core integration classes and decorators |
| `openevolve/unified_math_service.py` | 1,050+ | Unified formalization/verification service |
| `openevolve/leanaide_cav_nlp_bridge.py` | 901 | Migration bridge from LeanAide |

### Files Enhanced with CAV-NLP

| File | Integration Type | New Capabilities |
|------|-----------------|------------------|
| `z3_mcp_tools.py` | MCP Tools | 5 new CAV-NLP enhanced tools |
| `evolution_z3_fitness.py` | Fitness Evaluation | Hybrid verification, canonicalization |
| `automated_proof_engine.py` | Proof Engine | CAV-NLP enhanced proving strategies |
| `blue_team_solver_engine.py` | Solver Engine | Mathematical constraint solving |
| `bubblelabs_nodes/z3_constraint_solving_node.py` | Bubble Node | NL constraint formalization |
| `bubblelabs_nodes/z3_theorem_proving_node.py` | Bubble Node | NL theorem proving |
| `bubblelabs_nodes/math_verification_pipeline_node.py` | Bubble Node | Full verification pipeline |
| `bubblelabs_nodes/lean_autoformalization_node.py` | Bubble Node | CAV-NLP autoformalization |

---

## 3. New Integration Modules

### 3.1 `openevolve/z3_cav_nlp_integration.py`

**Purpose:** Drop-in enhancement for existing Z3-based code

**Key Classes:**

#### EnhancedZ3Solver

```python
class EnhancedZ3Solver:
    """Z3 Solver with CAV-NLP formalization capabilities."""
    
    def __init__(
        self,
        use_cav_nlp: bool = True,
        lean_service: Optional[Any] = None,
        enable_logging: bool = True
    )
```

**Methods:**
- `formalize_constraint(natural_language, context)` - Convert NL to Z3
- `verify_with_lean(constraints)` - Hybrid Z3 + Lean verification
- `find_counterexample(theorem)` - Find counterexamples
- `prove(theorem, variables)` - Prove theorems
- `get_capabilities()` - Query available capabilities

**Example:**
```python
from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver

solver = EnhancedZ3Solver()

# Formalize natural language constraint
constraint = solver.formalize_constraint(
    "for all positive x, x squared is positive",
    context={"paper_title": "Number Theory Basics"}
)

solver.add(constraint)
result = solver.check()

# Hybrid verification
verification = solver.verify_with_lean()
print(f"Confidence: {verification.confidence}")
```

#### ConstraintFormalizer

```python
class ConstraintFormalizer:
    """Formalize NL/LaTeX constraints to Z3."""
    
    def formalize(text, context=None, target="z3") -> FormalizationResult
    def formalize_latex(latex, context=None) -> FormalizationResult
    def batch_formalize(texts, context=None) -> List[FormalizationResult]
```

#### ProofExporter

```python
class ProofExporter:
    """Export Z3 proofs to Lean 4."""
    
    def export_proof(z3_proof, theorem_name, generate_tactics=True) -> str
    def export_constraints(constraints, theorem_name) -> str
    def export_with_verification(z3_proof, theorem_name) -> Tuple[str, VerificationResult]
```

#### CanonicalConstraintManager

```python
class CanonicalConstraintManager:
    """Manage canonical forms of constraints."""
    
    def canonicalize(constraint) -> CanonicalForm
    def are_equivalent(c1, c2) -> bool
    def find_redundant_constraints(constraints) -> List[int]
    def simplify_constraint_set(constraints) -> List
```

**Decorators:**

```python
@with_cav_nlp(auto_formalize=True, auto_canonicalize=False)
def solve_constraint(constraint):
    # String arguments automatically formalized
    pass

@auto_formalize
def analyze(constraint):
    # Automatically formalize string inputs
    pass

@auto_canonicalize
def get_constraint():
    # Automatically canonicalize return value
    pass
```

**Context Managers:**

```python
from openevolve.z3_cav_nlp_integration import cav_nlp_scope, enhanced_solver

# Scoped CAV-NLP enhanced solving
with cav_nlp_scope() as solver:
    constraint = solver.formalize_constraint("x > 0")
    solver.add(constraint)
    result = solver.check()
    verification = solver.verify_with_lean()

# With configuration
with enhanced_solver(use_cav_nlp=True) as solver:
    # Use enhanced solver
    pass
```

### 3.2 `openevolve/unified_math_service.py`

**Purpose:** Single entry point for all mathematical operations

**Architecture:**
- **Primary:** CAV-NLP for formalization (NL/LaTeX → Lean 4)
- **Secondary:** LeanAide for verification and elaboration
- **Optional:** Z3 for counterexample generation

```python
class UnifiedMathService:
    """
    Unified service for mathematical formalization and verification.
    
    Uses:
    - CAV-NLP as the primary formalization engine
    - LeanAide for verification and elaboration
    - Z3 for counterexample generation
    """
```

**Key Methods:**

```python
# Formalization (CAV-NLP)
async def formalize(
    text: str,
    context: Optional[CAVNLPContext] = None,
    elaborate: bool = True,
    generate_docs: bool = False
) -> FormalizationResult

# Verification (LeanAide)
async def verify(code: str) -> Optional[VerificationResult]

# Elaboration (LeanAide)
async def elaborate(code: str, timeout: Optional[float] = None) -> ElaborationResult

# Documentation (LeanAide)
async def generate_documentation(
    code: str,
    theorem_name: Optional[str] = None
) -> DocumentationResult

# Hybrid Proof (CAV-NLP + LeanAide)
async def prove(
    theorem: str,
    variables: Optional[Dict[str, str]] = None
) -> ProofResult
```

**CAV-NLP Pipeline:**

```
Natural Language/LaTeX
         ↓
flexible_semantic_parsing.py (SemanticNormalizer)
         ↓
dependency_dag.py (PaperStructureExtractor)
         ↓
z3_semantic_synthesis.py (Z3SemanticSynthesizer)
         ↓
canonical_lean_generator.py (CanonicalLeanGenerator)
         ↓
Canonical Lean 4 Code
```

**Example:**
```python
from openevolve.unified_math_service import create_unified_math_service

service = create_unified_math_service()

# Formalize
result = await service.formalize(
    "For all x > 0, x² > 0",
    elaborate=True,
    generate_docs=True
)
print(result.code)

# Verify
verification = await service.verify(result.code)
print(f"Verified: {verification.success}")

# Generate proof
proof = await service.prove("∀ x > 0, x² > 0")
print(proof.proof_code)
```

### 3.3 `openevolve/leanaide_cav_nlp_bridge.py`

**Purpose:** Smooth migration path from LeanAide to CAV-NLP

```python
class LeanAideCAVNLPBridge:
    """
    Bridge for migrating from LeanAide to CAV-NLP.
    
    Routes formalization requests to CAV-NLP while preserving
    LeanAide's verification and elaboration capabilities.
    """
```

**Migration Pattern:**

```python
# Old way (deprecated)
client = LeanAideClient()
result = await client.translate_thm("x + 0 = x")

# New way (recommended)
bridge = LeanAideCAVNLPBridge()
result = await bridge.translate_thm("x + 0 = x")  # Uses CAV-NLP
```

**Key Features:**
- Redirects `translate_thm()` and `translate_def()` to CAV-NLP
- Preserves `elaborate()`, `verify()`, `generate_documentation()` with LeanAide
- Issues deprecation warnings for old methods
- Provides detailed formalization output (semantic primitives, DAG, canonical form)

---

## 4. Integration Points

### 4.1 MCP Tools (`z3_mcp_tools.py`)

**5 New CAV-NLP Enhanced Tools:**

| Tool | Description |
|------|-------------|
| `z3_formalize_constraint` | Formalize NL to Z3/SMT-LIB/Lean using CAV-NLP |
| `z3_verify_hybrid` | Verify using hybrid Z3 + Lean approach |
| `z3_canonicalize_constraint` | Return canonical form using CAV-NLP |
| `z3_enhanced_prove` | Prove theorem with CAV-NLP enhanced verification |
| `z3_analyze_problem` | Analyze problem with optional CAV-NLP enhancement |

**Example Usage:**

```python
from z3_mcp_tools import get_z3_mcp_server

server = get_z3_mcp_server()

# Formalize natural language
result = server.call_tool("z3_formalize_constraint", {
    "natural_language": "For all natural numbers n, n + 0 = n",
    "target_format": "lean",
    "elaborate": True
})

# Hybrid verification
result = server.call_tool("z3_verify_hybrid", {
    "constraint": "For all integers x, x + 0 = x",
    "input_format": "natural_language"
})

# Enhanced proving
result = server.call_tool("z3_enhanced_prove", {
    "theorem": "For all x > 0, x² > 0",
    "use_cav_nlp": True,
    "generate_proof": True
})
```

### 4.2 BubbleLabs Nodes (4 Files)

| Node | New Operations | CAV-NLP Features |
|------|---------------|------------------|
| `z3_constraint_solving_node.py` | `formalize_constraints`, `nl_optimize` | NL constraint solving |
| `z3_theorem_proving_node.py` | `formalize_and_prove`, `hybrid_verify` | NL theorem proving |
| `math_verification_pipeline_node.py` | `hybrid_verify`, `cav_nlp_formalize` | Full verification pipeline |
| `lean_autoformalization_node.py` | All operations | CAV-NLP autoformalization |

**Configuration Options:**

```python
{
    "use_cav_nlp": True,           # Enable CAV-NLP
    "use_lean_verification": True, # Enable Lean verification
    "cav_nlp_timeout": 30.0,       # Timeout for formalization
    "elaborate_formalization": True,
    "generate_documentation": False,
    "use_hybrid_scoring": True,    # Hybrid confidence scoring
    "confidence_threshold": 0.8
}
```

### 4.3 Blue Team Solver (`blue_team_solver_engine.py`)

**Integration:**
- Mathematical constraint solving with CAV-NLP
- Natural language constraint formalization
- Hybrid verification for solution validation

### 4.4 Automated Proof Engine (`automated_proof_engine.py`)

**Integration:**
- CAV-NLP enhanced proving strategies
- Natural language theorem input
- Hybrid Z3 + Lean proof attempts

### 4.5 Evolution Fitness (`evolution_z3_fitness.py`)

**Integration:**

```python
class Z3FitnessEvaluator:
    def __init__(self, config=None, use_cav_nlp=True):
        self.use_cav_nlp = use_cav_nlp
        self.enhanced_solver = EnhancedZ3Solver() if use_cav_nlp else None
```

**New Capabilities:**
- Hybrid verification of evolved solutions
- Candidate canonicalization for duplicate detection
- Population deduplication using CAV-NLP
- Canonical form caching for performance

```python
evaluator = Z3FitnessEvaluator(use_cav_nlp=True)

# Evaluate with hybrid verification
fitness = evaluator.evaluate_fitness(
    individual,
    constraints,
    use_verification=True  # Enables CAV-NLP hybrid verification
)

# Canonicalize for duplicate detection
canonical = evaluator.canonicalize_candidate(individual)

# Deduplicate population
unique_pop = evaluator.deduplicate_by_canonical_form(population)
```

---

## 5. New Capabilities Added

### 5.1 Natural Language Formalization

**Before:**
```python
import z3
x, y = z3.Ints('x y')
solver = z3.Solver()
solver.add(x > 0, y > 0, x + y < 100)
```

**After:**
```python
from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver

solver = EnhancedZ3Solver()
constraint = solver.formalize_constraint(
    "x and y are positive integers summing to less than 100"
)
solver.add(constraint)
```

**Supported Input Formats:**
- Plain English ("for all x > 0, x² > 0")
- LaTeX ("\\forall x > 0, x^2 > 0")
- Mixed notation ("for all x \\in \\mathbb{R}, x² ≥ 0")

### 5.2 Hybrid Verification (Z3 + Lean)

```python
solver = EnhancedZ3Solver()
solver.add(constraint)

# Hybrid verification
result = solver.verify_with_lean()

print(f"Success: {result.success}")
print(f"Z3 Result: {result.z3_result}")
print(f"Lean Result: {result.lean_result}")
print(f"Confidence: {result.confidence}")  # 0.0 - 1.0
print(f"Counterexample: {result.counterexample}")
print(f"Proof: {result.proof}")
```

**Confidence Scoring:**
- Z3 verification: +0.4 confidence
- Lean verification: +0.6 confidence
- Both agree: 1.0 confidence

### 5.3 Constraint Canonicalization

```python
from openevolve.z3_cav_nlp_integration import CanonicalConstraintManager

manager = CanonicalConstraintManager()

# Canonicalize constraints
c1 = manager.canonicalize(x > 0)
c2 = manager.canonicalize(0 < x)  # Semantically equivalent

# Check equivalence
print(manager.are_equivalent(c1.original, c2.original))  # True

# Find redundant constraints
constraints = [x > 0, x > -5, y > 0]
redundant = manager.find_redundant_constraints(constraints)
# Returns indices of redundant constraints
```

### 5.4 Proof Export to Lean 4

```python
from openevolve.z3_cav_nlp_integration import ProofExporter

exporter = ProofExporter()

# Export Z3 proof to Lean
lean_code = exporter.export_proof(
    solver,
    theorem_name="positive_sum",
    generate_tactics=True
)

print(lean_code)
# Output:
# import Mathlib
#
# theorem positive_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
#     x + y > 0 := by
#   linarith
```

### 5.5 Equivalence Checking

```python
from openevolve.z3_cav_nlp_integration import check_equivalence, quick_canonicalize

# Check if two constraints are equivalent
c1 = x > 0
c2 = 0 < x
c3 = x >= 1

print(check_equivalence(c1, c2))  # True
print(check_equivalence(c1, c3))  # False

# Get canonical form
canonical = quick_canonicalize(c1)
```

---

## 6. Usage Examples

### 6.1 Basic Usage

```python
from openevolve.z3_cav_nlp_integration import (
    EnhancedZ3Solver,
    formalize_to_z3,
    cav_nlp_scope
)

# Method 1: Enhanced Z3 Solver
solver = EnhancedZ3Solver()
constraint = solver.formalize_constraint("x is positive")
solver.add(constraint)
result = solver.check()

# Method 2: Quick formalization
expr = formalize_to_z3("x and y are positive")
solver = EnhancedZ3Solver()
solver.add(expr)

# Method 3: Context manager
with cav_nlp_scope() as solver:
    solver.add(solver.formalize_constraint("x > 0"))
    result = solver.check()
```

### 6.2 Advanced Usage

```python
from openevolve.z3_cav_nlp_integration import (
    EnhancedZ3Solver,
    ConstraintFormalizer,
    ProofExporter,
    CanonicalConstraintManager
)

# Formalize with context
formalizer = ConstraintFormalizer()
result = formalizer.formalize(
    text="For all x > 0, x² > 0",
    context={
        "paper_title": "Real Analysis Basics",
        "section_context": "Properties of squares",
        "theorem_number": "1.2"
    },
    target="lean"
)

if result.success:
    print(f"Variables: {result.variables}")
    print(f"Type: {result.constraint_type}")
    print(f"Canonical: {result.canonical_form}")

# Export to Lean with verification
exporter = ProofExporter()
lean_code, verification = exporter.export_with_verification(
    solver,
    theorem_name="square_positive"
)

# Manage canonical forms
manager = CanonicalConstraintManager()
canonical = manager.canonicalize(constraint)
simplified = manager.simplify_constraint_set(constraints)
```

### 6.3 Configuration Examples

```python
# Configuration for different use cases

# 1. High-performance (minimal CAV-NLP)
fast_solver = EnhancedZ3Solver(
    use_cav_nlp=False,  # Disable for speed
    enable_logging=False
)

# 2. Maximum verification (full CAV-NLP)
verified_solver = EnhancedZ3Solver(
    use_cav_nlp=True,
    lean_service=lean_aide_service,
    enable_logging=True
)

# 3. Unified Math Service
from openevolve.unified_math_service import create_unified_math_service

service = create_unified_math_service(
    use_cav_nlp=True,
    use_leanaide=True
)
```

---

## 7. Configuration Reference

### 7.1 EnhancedZ3Solver Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_cav_nlp` | bool | True | Enable CAV-NLP features |
| `lean_service` | Any | None | Optional LeanAide service |
| `enable_logging` | bool | True | Enable operation logging |

### 7.2 UnifiedMathService Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_cav_nlp` | bool | True | Use CAV-NLP for formalization |
| `use_leanaide` | bool | True | Use LeanAide for verification |
| `lean_service` | Any | None | Pre-configured LeanAide service |
| `cav_nlp_bridge` | Any | None | Pre-configured CAV-NLP bridge |

### 7.3 MCP Tools Configuration

| Tool | Parameters |
|------|-----------|
| `z3_formalize_constraint` | `natural_language`, `target_format`, `elaborate` |
| `z3_verify_hybrid` | `constraint`, `input_format`, `timeout` |
| `z3_canonicalize_constraint` | `constraint`, `input_type` |
| `z3_enhanced_prove` | `theorem`, `use_cav_nlp`, `generate_proof`, `input_format` |
| `z3_analyze_problem` | `problem`, `use_cav_nlp` |

### 7.4 BubbleLabs Node Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_cav_nlp` | bool | True | Enable CAV-NLP |
| `use_lean_verification` | bool | True | Enable Lean verification |
| `cav_nlp_timeout` | float | 30.0 | Formalization timeout |
| `elaborate_formalization` | bool | True | Elaborate with LeanAide |
| `generate_documentation` | bool | False | Generate docs |
| `use_hybrid_scoring` | bool | True | Hybrid confidence scoring |
| `confidence_threshold` | float | 0.8 | Minimum confidence |

### 7.5 Performance Tuning

```python
# For high-throughput scenarios
config = {
    "use_cav_nlp": True,
    "cache_canonical_forms": True,  # Enable caching
    "cav_nlp_timeout": 10.0,  # Shorter timeout
    "elaborate_formalization": False  # Skip elaboration
}

# For maximum accuracy
config = {
    "use_cav_nlp": True,
    "use_lean_verification": True,
    "use_hybrid_scoring": True,
    "confidence_threshold": 0.95,
    "elaborate_formalization": True,
    "generate_documentation": True
}
```

---

## 8. Migration Guide

### 8.1 From Pure Z3

**Before:**
```python
import z3

solver = z3.Solver()
x, y = z3.Ints('x y')
solver.add(x > 0, y > 0)
result = solver.check()
```

**After:**
```python
from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver

solver = EnhancedZ3Solver()
x, y = solver.formalize_constraint("x and y are positive")
solver.add(x, y)
result = solver.check()
verification = solver.verify_with_lean()
```

### 8.2 From LeanAide

**Before:**
```python
from leanaide_client import LeanAideClient

client = LeanAideClient()
result = await client.translate_thm("x + 0 = x")
```

**After (Bridge):**
```python
from openevolve.leanaide_cav_nlp_bridge import create_migration_bridge

bridge = create_migration_bridge()
result = await bridge.translate_thm("x + 0 = x")  # Uses CAV-NLP
```

**After (Direct):**
```python
from openevolve.unified_math_service import create_unified_math_service

service = create_unified_math_service()
result = await service.formalize("x + 0 = x")
```

### 8.3 Backward Compatibility

- **EnhancedZ3Solver** is fully backward compatible with `z3.Solver`
- **LeanAideCAVNLPBridge** maintains the same API as `LeanAideClient`
- All existing code continues to work without modification
- CAV-NLP features are opt-in via `use_cav_nlp` parameter

### 8.4 Deprecation Timeline

| Feature | Status | Replacement |
|---------|--------|-------------|
| `LeanAideClient.translate_thm()` | Deprecated | `UnifiedMathService.formalize()` |
| `LeanAideClient.translate_def()` | Deprecated | `UnifiedMathService.formalize()` |
| Pure Z3 workflows | Supported | `EnhancedZ3Solver` recommended |

---

## 9. Testing & Verification

### 9.1 Unit Tests

```python
# Test EnhancedZ3Solver
from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver

def test_formalize_constraint():
    solver = EnhancedZ3Solver()
    constraint = solver.formalize_constraint("x is positive")
    assert constraint is not None
    
def test_verify_with_lean():
    solver = EnhancedZ3Solver()
    solver.add(solver.formalize_constraint("x > 0"))
    result = solver.verify_with_lean()
    assert result.confidence >= 0.0
```

### 9.2 Integration Tests

```python
# Test MCP tools
from z3_mcp_tools import get_z3_mcp_server

def test_formalize_mcp_tool():
    server = get_z3_mcp_server()
    result = server.call_tool("z3_formalize_constraint", {
        "natural_language": "x > 0",
        "target_format": "lean"
    })
    assert result["success"] is True
    assert "code" in result
```

### 9.3 Expected Outputs

**Formalization Output:**
```json
{
    "success": true,
    "code": "import Mathlib\n\ntheorem formalized_statement (x : ℝ) (hx : x > 0) : x > 0 := by\n  sorry",
    "source": "cav_nlp",
    "elaborated_code": "...",
    "warnings": [],
    "metadata": {
        "timestamp": "2026-02-05T13:06:06Z",
        "cav_nlp_used": true
    }
}
```

**Hybrid Verification Output:**
```json
{
    "success": true,
    "verified": true,
    "z3_result": {"proven": true, "status": "verified"},
    "lean_verification": {"success": true, "status": "SUCCESS"},
    "hybrid_confidence": 1.0,
    "errors": []
}
```

---

## 10. Next Steps

### 10.1 Future Enhancements

1. **Extended Domain Support**
   - Geometry formalization
   - Set theory support
   - Category theory primitives

2. **Performance Optimizations**
   - Parallel formalization
   - Model quantization for faster inference
   - Distributed CAV-NLP processing

3. **Additional Integrations**
   - Jupyter notebook extension
   - VS Code plugin
   - Web UI for interactive formalization

4. **Advanced Features**
   - Incremental formalization
   - Proof repair suggestions
   - Counterexample visualization

### 10.2 Known Limitations

| Limitation | Workaround |
|------------|------------|
| CAV-NLP requires internet for some models | Use local model deployment |
| Complex nested quantifiers may fail | Break into smaller statements |
| Large SMT-LIB files may timeout | Increase timeout or chunk |
| LaTeX parsing is heuristic-based | Validate output manually |

### 10.3 Documentation References

- [Z3 Integration Guide](docs/knowledge_engine/Z3_INTEGRATION_README.md)
- [CAV-NLP Architecture](CAV_NLP_INTEGRATION_STRATEGY.md)
- [Unified Math Service API](openevolve/unified_math_service.py)
- [Migration Examples](openevolve/leanaide_cav_nlp_bridge.py)

---

## Appendix: Quick Reference

### Import Cheat Sheet

```python
# Core integration
from openevolve.z3_cav_nlp_integration import (
    EnhancedZ3Solver,
    ConstraintFormalizer,
    ProofExporter,
    CanonicalConstraintManager,
    cav_nlp_scope,
    formalize_to_z3,
    check_equivalence
)

# Unified service
from openevolve.unified_math_service import (
    UnifiedMathService,
    create_unified_math_service
)

# Migration bridge
from openevolve.leanaide_cav_nlp_bridge import (
    LeanAideCAVNLPBridge,
    create_migration_bridge
)
```

### Common Patterns

```python
# Pattern 1: Quick formalization
expr = formalize_to_z3("x > 0")

# Pattern 2: Hybrid verification
with cav_nlp_scope() as solver:
    solver.add(constraint)
    verification = solver.verify_with_lean()

# Pattern 3: Batch formalization
formalizer = ConstraintFormalizer()
results = formalizer.batch_formalize(["x > 0", "y < 5"])

# Pattern 4: Proof export
exporter = ProofExporter()
lean_code = exporter.export_proof(solver)
```

---

**Document Version:** 1.0.0  
**Last Updated:** 2026-02-05  
**Author:** OpenEvolve Team  
**License:** Apache-2.0
