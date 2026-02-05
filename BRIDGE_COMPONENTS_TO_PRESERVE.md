# Components to Preserve from Current Bridge

## Analysis of z3_leanaide_bridge.py

### ✅ Components to KEEP (Valuable)

#### 1. Data Structures (Lines 62-155)
**Rationale**: Well-designed dataclasses that provide clean interfaces

| Class | Purpose | Preservation Strategy |
|-------|---------|----------------------|
| `TranslationDirection` | Enum for translation direction | Keep as-is |
| `ConstraintType` | Enum for constraint types | Keep as-is |
| `Z3Constraint` | Z3 constraint representation | Keep, add CAV-NLP integration |
| `Lean4Constraint` | Lean 4 constraint representation | Keep, add CAV-NLP integration |
| `TranslationResult` | Translation result metadata | Keep, extend with CAV-NLP fields |
| `VerificationBridgeResult` | Dual verification result | Keep, add DAG tracking |
| `HybridProofResult` | Hybrid proof result | Keep, integrate with CAV-NLP |

#### 2. Type & Operator Mappings (Lines 173-198, 333-342)
**Rationale**: Comprehensive mappings between Z3 and Lean notation

```python
# Keep these mappings
type_mappings = {
    "Bool": "Prop",
    "Int": "ℤ",
    "Real": "ℝ",
    "Array": "Array"
}

operator_mappings = {
    "And": "∧",
    "Or": "∨",
    "Not": "¬",
    "Implies": "->",
    "Eq": "=",
    "Lt": "<",
    "Le": "≤",
    "Gt": ">",
    "Ge": "≥",
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
    "Mod": "%",
    "Neg": "-"
}
```

#### 3. Main Bridge API (Lines 774-844)
**Rationale**: Clean, well-documented public API - maintain backward compatibility

Methods to preserve:
- `z3_to_lean4()` - Translate Z3 to Lean 4
- `lean4_to_z3()` - Translate Lean 4 to Z3
- `verify()` - Hybrid verification
- `find_counterexample()` - Counterexample generation
- `prove()` - Hybrid proof
- `is_z3_available()` - Capability checking
- `is_lean_available()` - Capability checking
- `get_capabilities()` - Feature detection

#### 4. Convenience Functions (Lines 851-859)
**Rationale**: Simple entry points for common use cases

- `create_z3_lean_bridge()` - Factory function
- `quick_verify()` - Quick verification helper

#### 5. Tactic Selection Logic (Lines 306-316)
**Rationale**: Good default tactics for different constraint types

```python
tactics_map = {
    ConstraintType.BOOLEAN: ["tauto"],
    ConstraintType.ARITHMETIC: ["linarith"],
    ConstraintType.NONLINEAR: ["nlinarith", "ring_nf"],
    ConstraintType.ARRAY: ["simp", "aesop"],
    ConstraintType.QUANTIFIED: ["intro", "simp"]
}
```

#### 6. Confidence Calculation (Lines 578-596)
**Rationale**: Useful heuristic for verification confidence

```python
def _calculate_confidence(z3_result, lean_result, agreed):
    confidence = 0.5
    if z3_result is not None:
        confidence += 0.2
    if lean_result is not None:
        confidence += 0.2
    if agreed:
        confidence += 0.3
    return min(confidence, 1.0)
```

---

### ❌ Components to REPLACE with CAV-NLP

#### 1. Translation Logic (Z3ToLeanTranslator.translate)
**Current**: Template-based translation (Lines 200-239)
**Replace with**: CAV-NLP semantic synthesis

```python
# OLD (template-based)
def translate(self, z3_expr, constraint_type):
    lean_expr = self._translate_expr(z3_expr)  # Simple string replacement
    lean_code = self._generate_lean_code(theorem_stmt, variables, constraint_type)
    return Lean4Constraint(...)

# NEW (CAV-NLP semantic synthesis)
def translate(self, z3_expr, constraint_type):
    # Convert Z3 to IR
    ir = self._z3_to_ir(z3_expr)
    # Use CAV-NLP for semantic synthesis
    synthesized = cav_nlp.synthesize_lean(ir, context=self.context)
    # Generate with canonical form
    lean_code = cav_nlp.generate_canonical_lean(synthesized)
    return Lean4Constraint(...)
```

#### 2. Expression Parsing (LeanToZ3Translator._translate_lean_expr)
**Current**: Regex-based parsing with eval() (Lines 414-449)
**Replace with**: CAV-NLP flexible semantic parser

```python
# OLD (regex + eval)
def _translate_lean_expr(self, expr, z3_vars):
    result = expr
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)
    return eval(result, {"__builtins__": {}}, context)

# NEW (CAV-NLP parser)
def _translate_lean_expr(self, lean_code):
    ast = cav_nlp.parse_mathematical_text(lean_code)
    return cav_nlp.convert_to_z3(ast)
```

#### 3. Code Generation (_generate_lean_code)
**Current**: Template-based code generation (Lines 281-304)
**Replace with**: CAV-NLP canonical lean generator

```python
# OLD (template-based)
def _generate_lean_code(self, theorem_stmt, variables, constraint_type):
    imports = ["import Mathlib"]
    tactics = self._select_tactics(constraint_type)
    theorem_with_proof = theorem_stmt.replace("sorry", "\n  ".join(tactics))
    return "\n".join(imports) + "\n\n" + theorem_with_proof

# NEW (CAV-NLP canonical generator)
def _generate_lean_code(self, ir_expr, dag=None):
    return cav_nlp.canonical_lean_generator.generate(
        ir=ir_expr,
        dag=dag or self.dependency_graph,
        canonicalize=True
    )
```

#### 4. Variable Extraction
**Current**: Recursive traversal with string matching (Lines 240-256)
**Replace with**: CAV-NLP dependency DAG extraction

```python
# OLD (recursive string matching)
def _extract_variables(self, expr):
    variables = set()
    def collect_vars(e):
        if hasattr(e, 'children'):
            for child in e.children():
                collect_vars(child)
        elif hasattr(e, 'decl'):
            name = str(e.decl())
            if name not in ['true', 'false', ...]:
                variables.add(name)
    collect_vars(expr)
    return sorted(list(variables))

# NEW (CAV-NLP DAG extraction)
def _extract_variables(self, ir_expr):
    return cav_nlp.dependency_dag.extract_variables(ir_expr)
```

---

## Integration Architecture

### Preserved API with CAV-NLP Backend

```python
class Z3LeanAideBridge:
    """
    Main bridge class - API preserved, backend replaced with CAV-NLP.
    """
    
    def __init__(self, lean_service=None, use_cav_nlp=True):
        self.use_cav_nlp = use_cav_nlp
        
        if use_cav_nlp:
            # New CAV-NLP backend
            self.translator = CAVNLPTranslator()  # Replaces Z3ToLeanTranslator
            self.verification = CAVNLPVerificationBridge()  # Enhanced verification
        else:
            # Legacy backend (for backward compatibility)
            self.z3_to_lean = Z3ToLeanTranslator()
            self.lean_to_z3 = LeanToZ3Translator()
            self.verification = Z3LeanVerificationBridge(lean_service)
        
        self.hybrid_proof = HybridProofEngine(self.verification)
    
    def z3_to_lean4(self, z3_expr, constraint_type=ConstraintType.BOOLEAN):
        """API preserved - uses CAV-NLP internally"""
        if self.use_cav_nlp:
            return self.translator.translate_with_cav_nlp(z3_expr, constraint_type)
        else:
            return self.z3_to_lean.translate(z3_expr, constraint_type)
    
    async def verify(self, constraint, use_counterexamples=True):
        """API preserved - enhanced with DAG tracking"""
        # Always use enhanced verification with CAV-NLP
        return await self.verification.verify_hybrid(
            constraint, 
            use_counterexamples=use_counterexamples,
            track_dependencies=True  # New: CAV-NLP feature
        )
```

---

## Migration Checklist

### Phase 1: Data Structures (Keep)
- [ ] Copy all dataclasses to new module
- [ ] Add CAV-NLP-specific fields (dag, canonical_form, etc.)
- [ ] Maintain backward compatibility

### Phase 2: Mappings (Keep)
- [ ] Copy type_mappings
- [ ] Copy operator_mappings
- [ ] Add CAV-NLP canonicalization rules

### Phase 3: API (Keep Interface, Replace Implementation)
- [ ] Create adapter class with same methods
- [ ] Implement methods using CAV-NLP backend
- [ ] Add feature flag for legacy mode

### Phase 4: Translation Logic (Replace)
- [ ] Replace Z3ToLeanTranslator with CAVNLPTranslator
- [ ] Replace LeanToZ3Translator with CAVNLPReverseTranslator
- [ ] Replace code generation with canonical generator

### Phase 5: Verification (Enhance)
- [ ] Keep VerificationBridgeResult structure
- [ ] Add DAG tracking to verification
- [ ] Add canonicalization verification
- [ ] Keep confidence calculation

### Phase 6: Testing
- [ ] Ensure all existing tests pass
- [ ] Add CAV-NLP specific tests
- [ ] Verify backward compatibility

---

## File Structure

```
openevolve/
├── cav_nlp_integration/              # NEW: CAV-NLP integration
│   ├── __init__.py
│   ├── adapter.py                    # Main adapter (preserves API)
│   ├── translator.py                 # CAV-NLP based translation
│   ├── data_structures.py            # Preserved dataclasses
│   ├── mappings.py                   # Preserved mappings
│   └── verification.py               # Enhanced verification
├── legacy/                           # OLD: Legacy code
│   └── z3_leanaide_bridge.py         # Original (deprecated)
└── z3_leanaide_bridge.py             # COMPAT: Thin wrapper to adapter
```

## Backward Compatibility

```python
# openevolve/z3_leanaide_bridge.py
"""Backward compatibility wrapper - delegates to CAV-NLP adapter."""

import warnings
from cav_nlp_integration.adapter import Z3LeanAideBridge as NewBridge

warnings.warn(
    "z3_leanaide_bridge is deprecated. Use cav_nlp_integration directly.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new module
__all__ = ['Z3LeanAideBridge', 'create_z3_lean_bridge', 'quick_verify']

class Z3LeanAideBridge(NewBridge):
    """Backward compatibility wrapper."""
    pass

# Re-export convenience functions
from cav_nlp_integration.adapter import create_z3_lean_bridge, quick_verify
```
