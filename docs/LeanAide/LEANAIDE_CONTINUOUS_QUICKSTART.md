# LeanAide Continuous Mathematics - Quick Start Guide

**Get started with the LeanAide Continuous Mathematics System in 5 minutes**

## What is LeanAide Continuous?

LeanAide Continuous is a powerful system for:
- 🔍 **Detecting** continuous mathematics in natural language
- 🔄 **Translating** math to formal Lean 4 code
- 📚 **Providing** domain-specific knowledge
- ✅ **Verifying** generated code correctness

Perfect for:
- Mathematicians formalizing problems
- Students learning differential equations
- Researchers working with scientific computing
- Anyone needing to bridge natural language and formal proofs

---

## Installation

### Prerequisites

```bash
# Python 3.9+ required
python --version

# Install dependencies
pip install sympy
```

### Files Required

Ensure you have these files in your project:
```
continuous_math_detector.py
ode_pde_translator.py
scientific_domain_patterns.py
verification_methods.py
leanaide_continuous_mcp.py
```

---

## Quick Start (3 Steps)

### Step 1: Detect Mathematics

```python
from continuous_math_detector import detect_continuous_math

text = "Solve dy/dx + y = 0 with y(0) = 1"
result = detect_continuous_math(text)

print(f"Type: {result.math_type}")        # ordinary_differential_equation
print(f"Domain: {result.domain}")          # general
print(f"Confidence: {result.confidence}")  # 0.95
```

### Step 2: Translate to Lean 4

```python
from ode_pde_translator import translate_to_lean4

translation = translate_to_lean4(result)

print(translation.lean4_code)
```

Output:
```lean
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Calculus.Deriv

namespace SimpleODE

open Real

/-- The ODE: dy/dx + y = 0 -/
def ode_eq (y : Real → Real) : Prop :=
  ∀ x, deriv y x + y x = 0

/-- Initial condition: y(0) = 1 -/
def initial_cond (y : Real → Real) : Prop :=
  y 0 = 1

/-- Existence and uniqueness theorem -/
theorem existence_uniqueness :
  ∃ y : Real → Real, ode_eq y ∧ initial_cond y :=
  by
    sorry

end SimpleODE
```

### Step 3: Verify Code

```python
from verification_methods import verify_lean4_code

verification = verify_lean4_code(translation.lean4_code)

print(f"Status: {verification.overall_status}")  # passed/failed/warning
print(f"Valid: {verification.is_valid}")         # True/False
print(f"Issues: {len(verification.issues)}")      # Number of issues
```

---

## Common Use Cases

### 1. Simple ODE

```python
from leanaide_continuous_mcp import get_mcp_tools

mcp = get_mcp_tools()

# Quick one-shot translation
result = mcp.execute_tool(
    "translate_ode",
    {"equation": "dy/dx = y", "initial_condition": "y(0) = 1"}
)

print(result.data["lean4_code"])
```

### 2. Heat Equation (PDE)

```python
text = "Solve the heat equation ∂u/∂t = α ∂²u/∂x²"

# Use complete pipeline
result = mcp.execute_tool(
    "complete_pipeline",
    {"text": text, "verify": False}  # Skip verification for speed
)

print(f"Math Type: {result.data['detection']['math_type']}")
print(f"Lean 4 Code:\n{result.data['translation']['lean4_code']}")
```

### 3. Biology Problem (Lotka-Volterra)

```python
text = """
Predator-prey model:
dx/dt = αx - βxy
dy/dt = δxy - γy
"""

# Detect
detect_result = mcp.execute_tool("detect_math", {"text": text})

# Get domain knowledge
templates = mcp.execute_tool(
    "get_equation_templates",
    {"domain": "biology"}
)

print(f"Found {templates.data['count']} biology templates")

# Translate
translate_result = mcp.execute_tool("translate_to_lean4", {"text": text})
print(translate_result.data["lean4_code"])
```

### 4. Check if Something is an ODE

```python
# Quick check
result = mcp.execute_tool("is_ode", {"text": "dy/dx = x + y"})

if result.data["is_ode"]:
    print("✓ Yes, this is an ODE!")
else:
    print("✗ Not an ODE")
```

---

## MCP Tool Reference

### All Available Tools

```python
from leanaide_continuous_mcp import get_mcp_tools

mcp = get_mcp_tools()
tools = mcp.list_tools()

print(tools)
# ['detect_math', 'is_ode', 'is_pde', 'translate_to_lean4',
#  'translate_ode', 'translate_pde', 'get_equation_templates',
#  'get_solution_methods', 'recommend_solution_method',
#  'verify_lean4_code', 'complete_pipeline']
```

### Tool Categories

#### Detection Tools
- `detect_math` - Full detection and classification
- `is_ode` - Quick ODE check
- `is_pde` - Quick PDE check

#### Translation Tools
- `translate_to_lean4` - Translate with detection
- `translate_ode` - Translate standalone ODE
- `translate_pde` - Translate standalone PDE

#### Domain Knowledge Tools
- `get_equation_templates` - Get equation templates
- `get_solution_methods` - Get solution methods
- `recommend_solution_method` - Get recommendation

#### Verification Tools
- `verify_lean4_code` - Verify Lean 4 code

#### Workflow Tools
- `complete_pipeline` - Full detection → translation → verification

---

## Tips & Tricks

### 1. Add Domain Keywords

If domain detection isn't working, add domain-specific keywords:

```python
# Before
text = "Solve ∂u/∂t = ∂²u/∂x²"
# → Detects as "general"

# After
text = "Solve the heat equation ∂u/∂t = α ∂²u/∂x² in physics"
# → Detects as "physics"
```

### 2. Disable Verification for Speed

```python
result = mcp.execute_tool(
    "complete_pipeline",
    {"text": "dy/dx + y = 0", "verify": False}
)
```

### 3. Get Domain Knowledge First

```python
# See what's available for your domain
templates = mcp.execute_tool(
    "get_equation_templates",
    {"domain": "physics"}
)

for template in templates.data["templates"]:
    print(f"- {template['name']}: {template['equation_pattern']}")
```

### 4. Use Specific Tools for Better Performance

```python
# Slower but more comprehensive
result = mcp.execute_tool("detect_math", {"text": "dy/dx = y"})

# Faster if you know what you're looking for
result = mcp.execute_tool("is_ode", {"text": "dy/dx = y"})
```

---

## Supported Mathematics

### Math Types
- ✅ ODEs (Ordinary Differential Equations)
- ✅ PDEs (Partial Differential Equations)
- ✅ Integrals
- ✅ Derivatives
- ✅ Limits
- 🔄 DAEs (Differential-Algebraic Equations) - *experimental*
- 🔄 SDEs (Stochastic Differential Equations) - *experimental*

### Scientific Domains
- 🔬 **Physics** - Heat equation, wave equation, Schrödinger, etc.
- 🧬 **Biology** - Lotka-Volterra, SIR model, logistic growth, etc.
- ⚗️ **Chemistry** - Rate equations, Michaelis-Menten, etc.
- ⚙️ **Engineering** - Control systems, RLC circuits, etc.
- 📈 **Economics** - Black-Scholes, Solow model, etc.

### Problem Types
- IVP (Initial Value Problems)
- BVP (Boundary Value Problems)
- Eigenvalue problems
- Control problems
- Optimization problems

---

## Example Workflows

### Researcher Workflow

```python
from leanaide_continuous_mcp import get_mcp_tools

mcp = get_mcp_tools()

# 1. You have a problem
problem = """
I need to model heat diffusion in a metal rod.
The equation is ∂T/∂t = α ∂²T/∂x²
"""

# 2. Understand what type of math it is
detection = mcp.execute_tool("detect_math", {"text": problem})
print(f"Math Type: {detection.data['math_type']}")  # PDE
print(f"Domain: {detection.data['domain']}")        # Physics

# 3. Get relevant knowledge
templates = mcp.execute_tool(
    "get_equation_templates",
    {"domain": "physics", "category": "thermodynamics"}
)
print(f"Found {templates.data['count']} templates")

# 4. See solution methods
methods = mcp.execute_tool(
    "get_solution_methods",
    {"domain": "physics", "math_type": "PDE"}
)
print(f"Available methods: {len(methods.data['solution_methods'])}")

# 5. Formalize in Lean 4
translation = mcp.execute_tool("translate_to_lean4", {"text": problem})
print(translation.data["lean4_code"])

# 6. Verify correctness
verification = mcp.execute_tool(
    "verify_lean4_code",
    {"code": translation.data["lean4_code"], "domain": "physics"}
)
print(f"Verification: {verification.data['status']}")
```

### Student Learning Workflow

```python
# Learning about differential equations

problems = [
    "dy/dx = y (exponential growth)",
    "dy/dx + y = 0 (decay)",
    "d²y/dx² + y = 0 (harmonic oscillator)"
]

for problem in problems:
    print(f"\n{'='*60}")
    print(f"Problem: {problem}")
    print('='*60)

    # What type of equation?
    result = mcp.execute_tool("detect_math", {"text": problem})
    print(f"Type: {result.data['math_type']}")

    # Get formalization
    translation = mcp.execute_tool("translate_to_lean4", {"text": problem})
    print(f"\nLean 4:\n{translation.data['lean4_code']}")

    # See the theorems
    print(f"\nTheorems: {len(translation.data['theorems'])}")
    for i, theorem in enumerate(translation.data['theorems'], 1):
        print(f"{i}. {theorem[:80]}...")
```

---

## Troubleshooting

### Problem: Detection returns "unknown_math_type"

**Solution**: Add more mathematical notation or domain keywords

```python
# Before
text = "Solve the equation"
# → unknown_math_type

# After
text = "Solve the differential equation dy/dx = y"
# → ordinary_differential_equation
```

### Problem: Translation fails

**Solution**: Check that detection succeeded first

```python
detect_result = mcp.execute_tool("detect_math", {"text": text})

if detect_result.data["math_type"] == "unknown_math_type":
    print("⚠️ Detection failed - cannot translate")
else:
    translate_result = mcp.execute_tool("translate_to_lean4", {"text": text})
```

### Problem: Domain detection is wrong

**Solution**: Manually specify domain in translation

```python
# Override domain detection
translation = mcp.execute_tool(
    "translate_to_lean4",
    {"text": "dy/dx = y", "domain": "physics"}
)
```

### Problem: Verification takes too long

**Solution**: Disable LeanAide (uses only static checks)

```python
from verification_methods import Lean4Verifier

verifier = Lean4Verifier(enable_leanaide=False)
# This is much faster but doesn't run Lean
```

---

## Next Steps

### Learn More
- 📖 [Complete Documentation](LEANAIDE_CONTINUOUS_MCP.md)
- 🔧 [API Reference](LEANAIDE_CONTINUOUS_MCP.md#api-reference)
- 📚 [Architecture](ARCHITECTURE.md)

### Advanced Topics
- Custom domain patterns
- Extending verification checks
- Integration with LeanAide
- MCP server deployment

### Examples
- [Physics Examples](../examples/physics/README.md)
- [Biology Examples](../examples/biology/README.md)
- [Engineering Examples](../examples/engineering/README.md)

---

## Cheat Sheet

```python
# Import
from lenaide_continuous_mcp import get_mcp_tools

# Initialize
mcp = get_mcp_tools()

# Detect
result = mcp.execute_tool("detect_math", {"text": "dy/dx = y"})

# Translate
result = mcp.execute_tool("translate_to_lean4", {"text": "dy/dx = y"})

# Verify
result = mcp.execute_tool("verify_lean4_code", {"code": lean4_code})

# Complete pipeline
result = mcp.execute_tool("complete_pipeline", {"text": "dy/dx = y"})

# Check result
if result.success:
    print(result.data)
else:
    print(result.error)
```

---

**Ready to dive deeper?** Check out the [full documentation](LEANAIDE_CONTINUOUS_MCP.md)!

**Author**: OpenEvolve
**Created**: 2026-01-09
**Phase**: 2 - LeanAide Enhancement
