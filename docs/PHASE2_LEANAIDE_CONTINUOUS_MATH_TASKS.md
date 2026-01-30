# Phase 2: LeanAide Enhancement for Continuous Mathematics

**Priority**: P1 (HIGH VALUE)
**Estimated Effort**: 2-3 weeks
**Status**: READY TO START
**Source**: FRM Integration Analysis Recommendation

---

## Executive Summary

**Why This is High Value**: LeanAide is fully integrated (90%+) but **underutilized**. It currently only handles discrete mathematics (proofs, algebra, logic). By adding continuous mathematics support (ODE/PDE/DAE/SDE), LeanAide can provide **80% of FRM's value** with **20% of the effort**.

**Value Proposition**:
- Covers FRM's unique value (continuous math modeling)
- No architectural changes (same Python tech stack)
- Leverages existing LeanAide integration
- Low maintenance burden
- 2-3 weeks vs 3-5 weeks for FRM integration

---

## Background: LeanAide Current Capabilities

**Current Focus**: Discrete Mathematics
- Algebraic proofs
- Number theory
- Topology
- Logic and set theory
- Combinatorics
- Geometry

**Current Integration Points**:
- Stage 0: Mathematical content detection
- Stage 1: Formal decomposition of mathematical problems
- Stage 3: Formal verification of solutions
- Stage 3B: Mathematical critique of proofs
- Stage 5: Final formal verification
- Stage 6: Extract verified theorems

**Gap**: Does not handle continuous mathematics (ODE, PDE, DAE, SDE)

---

## Component 1: Continuous Math Detection

**Effort**: 3-4 days
**Priority**: P1 (enables all other work)
**Dependencies**: None
**Files**: `leanaide_client.py`, `leanaide_hephaestus_bridge.py`

### Tasks

#### 1.1 Extend MathematicalDomain Enum
**File**: `leanaide_hephaestus_bridge.py`

```python
class MathematicalDomain(Enum):
    """Mathematical domains for classification"""
    # Existing discrete domains
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    TOPOLOGY = "topology"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    LOGIC = "logic"
    SET_THEORY = "set_theory"
    GENERAL = "general"

    # NEW: Continuous mathematics domains
    ODE = "ordinary_differential_equations"
    PDE = "partial_differential_equations"
    DAE = "differential_algebraic_equations"
    SDE = "stochastic_differential_equations"
    CONTINUOUS_MODELING = "continuous_modeling"
    DYNAMICAL_SYSTEMS = "dynamical_systems"
```

#### 1.2 Implement Continuous Math Detector
**File**: `leanaide_client.py`

```python
class LeanAideClient:
    async def detect_mathematics_type(
        self,
        problem_statement: str
    ) -> Tuple[MathematicalDomain, float]:
        """
        Detect if problem involves discrete or continuous mathematics

        Returns:
            (domain_type, confidence)
        """
        prompt = f"""
        Analyze this problem statement and classify the mathematical domain:

        Problem: {problem_statement}

        Classify as one of:
        - discrete: algebra, number theory, logic, combinatorics, topology, geometry
        - continuous: ODE, PDE, DAE, SDE, dynamical systems, modeling

        Provide classification and confidence (0.0 to 1.0).
        """

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response
        domain = self._parse_domain(response)
        confidence = self._parse_confidence(response)

        return domain, confidence

    def _parse_domain(self, response: str) -> MathematicalDomain:
        """Parse domain from LLM response"""
        # Look for keywords
        response_lower = response.lower()

        continuous_keywords = ["differential", "ode", "pde", "derivative", "integral",
                              "continuous", "dynamical", "stochastic"]

        if any(kw in response_lower for kw in continuous_keywords):
            # Determine specific continuous type
            if "ordinary differential" in response_lower or " ode " in response_lower:
                return MathematicalDomain.ODE
            elif "partial differential" in response_lower or " pde " in response_lower:
                return MathematicalDomain.PDE
            elif "stochastic" in response_lower:
                return MathematicalDomain.SDE
            else:
                return MathematicalDomain.CONTINUOUS_MODELING
        else:
            # Default to general (will use existing discrete logic)
            return MathematicalDomain.GENERAL
```

**Deliverable**: Continuous math detection in `leanaide_client.py`

---

## Component 2: ODE/PDE Translation to Lean 4

**Effort**: 1 week
**Priority**: P1 (core continuous math support)
**Dependencies**: Component 1
**Files**: `leanaide_client.py`, `LeanAide/translate.lean`

### Tasks

#### 2.1 Define ODE/PDE Data Structures
**File**: `leanaide_client.py`

```python
@dataclass
class DifferentialEquation:
    """Representation of a differential equation"""
    equation_type: Literal["ode", "pde", "dae", "sde"]
    equation: str  # LaTeX or plain text representation
    variables: List[str]  # Independent variables (e.g., ["t", "x"])
    unknowns: List[str]  # Dependent variables (e.g., ["y", "u"])
    parameters: Dict[str, float]  # Constants
    initial_conditions: Optional[Dict[str, float]] = None
    boundary_conditions: Optional[Dict[str, float]] = None
    order: int = 1  # Order of derivative

@dataclass
class ContinuousMathProblem:
    """Continuous mathematics problem"""
    problem_type: Literal["ode", "pde", "dae", "sde", "optimization", "modeling"]
    equations: List[DifferentialEquation]
    domain: str  # Application domain (medicine, physics, biology, etc.)
    goal: str  # What we're solving for
    constraints: List[str]
```

#### 2.2 Implement ODE Translation
**File**: `leanaide_client.py`

```python
class LeanAideClient:
    async def translate_ode_to_lean4(
        self,
        ode: DifferentialEquation,
        problem: ContinuousMathProblem
    ) -> str:
        """
        Translate ordinary differential equation to Lean 4

        Example:
        Input:  dy/dt = -ky (exponential decay)
        Output: Lean 4 formalization
        """
        prompt = f"""
        Translate this ordinary differential equation (ODE) to Lean 4 formal mathematics:

        ODE: {ode.equation}
        Variables: {ode.variables}
        Unknowns: {ode.unknowns}
        Initial Conditions: {ode.initial_conditions}

        Provide:
        1. Lean 4 type definition for the function
        2. Lean 4 definition of the differential equation
        3. Lean 4 statement of the initial value problem
        4. If possible, a proof sketch for existence/uniqueness

        Use mathlib4 definitions for derivatives, limits, etc.
        """

        response = await self._call_llm(prompt)
        return self._extract_lean4_code(response)

    async def solve_ode_in_lean4(
        self,
        ode: DifferentialEquation,
        lean4_code: str
    ) -> Dict[str, Any]:
        """
        Attempt to solve ODE in Lean 4 (if tractable) or provide formalization
        """
        # For many ODEs, exact solutions may not exist in Lean 4
        # Focus on formalization and verification of properties

        result = {
            "formalization": lean4_code,
            "solvable": await self._check_if_solvable(ode),
            "properties": await self._verify_ode_properties(ode, lean4_code),
            "existence_proof": await self._prove_existence(ode, lean4_code)
        }

        return result
```

#### 2.3 Implement PDE Translation
**File**: `leanaide_client.py`

```python
class LeanAideClient:
    async def translate_pde_to_lean4(
        self,
        pde: DifferentialEquation,
        problem: ContinuousMathProblem
    ) -> str:
        """
        Translate partial differential equation to Lean 4

        Example:
        Input:  ∂u/∂t = α ∂²u/∂x² (heat equation)
        Output: Lean 4 formalization
        """
        prompt = f"""
        Translate this partial differential equation (PDE) to Lean 4 formal mathematics:

        PDE: {pde.equation}
        Variables: {pde.variables}  # e.g., ["t", "x", "y"]
        Unknowns: {pde.unknowns}    # e.g., ["u"]
        Boundary Conditions: {pde.boundary_conditions}
        Initial Conditions: {pde.initial_conditions}

        Provide:
        1. Lean 4 type definition for the function (multi-variable)
        2. Lean 4 definition of the partial differential equation
        3. Lean 4 statement of the boundary/initial value problem
        4. Classification of PDE type (elliptic, parabolic, hyperbolic)

        Use mathlib4 definitions for partial derivatives, etc.
        """

        response = await self._call_llm(prompt)
        return self._extract_lean4_code(response)
```

**Deliverable**: ODE/PDE translation in `leanaide_client.py`

---

## Component 3: Scientific Domain Patterns

**Effort**: 3-4 days
**Priority**: P1 (domain-specific handling)
**Dependencies**: Component 1, Component 2
**Files**: `leanaide_client.py`, `leanaide_hephaestus_bridge.py`

### Tasks

#### 3.1 Add Domain-Specific Solvers
**File**: `leanaide_client.py`

```python
class ScientificDomainSolver:
    """Domain-specific solution strategies for continuous math"""

    DOMAIN_PATTERNS = {
        "medicine": {
            "epidemiology": ["SIR_model", "SEIR_model", "compartmental_models"],
            "pharmacokinetics": ["absorption", "distribution", "metabolism", "excretion"],
            "physiology": ["cardiovascular", "neural", "respiratory"]
        },
        "physics": {
            "mechanics": ["newton_laws", "lagrangian", "hamiltonian"],
            "electromagnetism": ["maxwell_equations"],
            "thermodynamics": ["heat_equation", "diffusion"],
            "quantum": ["schrodinger_equation"]
        },
        "biology": {
            "population": ["logistic_growth", "predator_prey", "competition"],
            "biochemistry": ["michaelis_menten", "enzyme_kinetics"],
            "neuroscience": ["hodgkin_huxley", "fitzhugh_nagumo"]
        },
        "engineering": {
            "control": ["pid_controller", "state_space", "transfer_function"],
            "circuits": ["rlc_circuit", "op_amp"],
            "fluids": ["navier_stokes", "bernoulli"]
        }
    }

    async def identify_domain_pattern(
        self,
        problem: ContinuousMathProblem
    ) -> Tuple[str, str]:
        """
        Identify domain and specific pattern

        Returns:
            (domain, pattern)
        """
        # Use LLM to classify
        prompt = f"""
        Classify this continuous mathematics problem:

        Domain: {problem.domain}
        Equations: {[eq.equation for eq in problem.equations]}
        Goal: {problem.goal}

        Classify into:
        1. Domain: medicine, physics, biology, engineering, chemistry, climate, etc.
        2. Pattern: specific sub-domain (e.g., epidemiology, thermodynamics, control)

        Provide classification and confidence.
        """

        response = await self._call_llm(prompt)
        return self._parse_classification(response)

    async def get_domain_knowledge(
        self,
        domain: str,
        pattern: str
    ) -> Dict[str, Any]:
        """
        Retrieve domain-specific knowledge:
        - Common equation forms
        - Standard solution methods
        - Typical constraints
        - Known theorems
        """
        if domain in self.DOMAIN_PATTERNS and pattern in self.DOMAIN_PATTERNS[domain]:
            return {
                "common_forms": self._get_common_forms(domain, pattern),
                "solution_methods": self._get_solution_methods(domain, pattern),
                "theorems": self._get_theorems(domain, pattern)
            }
        else:
            # Fallback to general knowledge
            return await self._fetch_general_knowledge(domain, pattern)
```

#### 3.2 Update Hephaestus Bridge for Continuous Math
**File**: `leanaide_hephaestus_bridge.py`

```python
class LeanAideHephaestusBridge:
    async def execute_phase_1_setup(self, **kwargs) -> Dict[str, Any]:
        """
        Phase 1: Analysis - Enhanced for continuous mathematics
        """
        problem_text = kwargs.get("problem_text", "")

        # Detect math type
        math_type, confidence = await self.client.detect_mathematics_type(problem_text)

        if math_type in [MathematicalDomain.ODE, MathematicalDomain.PDE,
                         MathematicalDomain.DAE, MathematicalDomain.SDE]:
            # Handle continuous mathematics
            continuous_problem = await self._parse_continuous_problem(problem_text)
            domain, pattern = await self.domain_solver.identify_domain_pattern(continuous_problem)

            return {
                "math_type": "continuous",
                "specific_type": math_type.value,
                "domain": domain,
                "pattern": pattern,
                "equations": [eq.equation for eq in continuous_problem.equations],
                "confidence": confidence,
                "lean4_ready": True
            }
        else:
            # Use existing discrete math logic
            return await self._analyze_discrete_problem(problem_text)

    async def _parse_continuous_problem(
        self,
        problem_text: str
    ) -> ContinuousMathProblem:
        """Parse continuous mathematics problem from text"""
        # Extract equations
        equations = await self._extract_differential_equations(problem_text)

        # Classify problem type
        problem_type = await self._classify_problem_type(problem_text, equations)

        # Extract domain
        domain = await self._extract_application_domain(problem_text)

        return ContinuousMathProblem(
            problem_type=problem_type,
            equations=equations,
            domain=domain,
            goal=await self._extract_goal(problem_text),
            constraints=await self._extract_constraints(problem_text)
        )
```

**Deliverable**: Domain patterns in `leanaide_client.py` and bridge updates

---

## Component 4: Verification for Continuous Math

**Effort**: 4-5 days
**Priority**: P1 (verification capabilities)
**Dependencies**: Component 1, Component 2
**Files**: `leanaide_client.py`

### Tasks

#### 4.1 Implement Property Verification
**File**: `leanaide_client.py`

```python
class LeanAideClient:
    async def verify_continuous_solution(
        self,
        problem: ContinuousMathProblem,
        proposed_solution: str,
        lean4_formalization: str
    ) -> Dict[str, Any]:
        """
        Verify a proposed solution to a continuous math problem

        For continuous mathematics, full verification is often intractable.
        Instead, verify:
        1. Solution satisfies the equation (substitution check)
        2. Solution satisfies initial/boundary conditions
        3. Solution has expected properties (continuity, differentiability, etc.)
        4. Domain-specific properties (positivity, boundedness, etc.)
        """
        verification_result = {
            "equation_satisfied": await self._verify_equation_satisfied(
                problem, proposed_solution, lean4_formalization
            ),
            "conditions_satisfied": await self._verify_conditions_satisfied(
                problem, proposed_solution
            ),
            "properties_verified": await self._verify_properties(
                problem, proposed_solution
            ),
            "domain_properties": await self._verify_domain_properties(
                problem.domain, proposed_solution
            ),
            "formal_verification_status": await self._check_lean4_proof(
                lean4_formalization
            )
        }

        return verification_result

    async def _verify_equation_satisfied(
        self,
        problem: ContinuousMathProblem,
        solution: str,
        lean4_code: str
    ) -> bool:
        """
        Check if solution satisfies the differential equation

        Method:
        1. Parse solution to get function definition
        2. Symbolically differentiate solution
        3. Substitute into equation
        4. Check if LHS = RHS (within tolerance)
        """
        # Use sympy or numerical checking
        from sympy import symbols, Function, dsolve, Eq, diff

        # Parse and verify
        pass

    async def _verify_domain_properties(
        self,
        domain: str,
        solution: str
    ) -> Dict[str, bool]:
        """
        Verify domain-specific properties:

        Medicine:
        - Positivity (populations, concentrations can't be negative)
        - Boundedness (values stay within physical limits)

        Physics:
        - Conservation laws (energy, mass, momentum)
        - Causality (effects don't precede causes)

        Biology:
        - Positivity (populations, concentrations)
        - Stability (solutions don't blow up)
        """
        domain_checks = {
            "medicine": ["positivity", "boundedness"],
            "physics": ["conservation", "causality"],
            "biology": ["positivity", "stability"],
            "engineering": ["stability", "performance"]
        }

        if domain in domain_checks:
            return await self._run_domain_checks(domain, domain_checks[domain], solution)
        else:
            return {}
```

**Deliverable**: Verification methods in `leanaide_client.py`

---

## Component 5: MCP Tools for Continuous Math

**Effort**: 2-3 days
**Priority**: P2 (nice to have)
**Dependencies**: Component 1, Component 2, Component 4
**Files**: `leanaide_mcp_tools.py`

### Tasks

#### 5.1 Add Continuous Math MCP Tools
**File**: `leanaide_mcp_tools.py`

```python
# Existing tools...

@mcp_tool
async def leanaide_detect_continuous_math(
    problem_statement: str
) -> Dict[str, Any]:
    """
    Detect if problem involves continuous mathematics (ODE/PDE/DAE/SDE)

    Args:
        problem_statement: The problem to analyze

    Returns:
        {
            "is_continuous": bool,
            "equation_type": str ("ode", "pde", "dae", "sde", "none"),
            "confidence": float,
            "domain": str,
            "equations": list[str]
        }
    """
    client = LeanAideClient()
    domain, confidence = await client.detect_mathematics_type(problem_statement)

    is_continuous = domain in [
        MathematicalDomain.ODE,
        MathematicalDomain.PDE,
        MathematicalDomain.DAE,
        MathematicalDomain.SDE
    ]

    return {
        "is_continuous": is_continuous,
        "equation_type": domain.value if is_continuous else "none",
        "confidence": confidence,
        "domain": "continuous" if is_continuous else "discrete"
    }

@mcp_tool
async def leanaide_translate_ode(
    equation: str,
    variables: list[str],
    unknowns: list[str],
    initial_conditions: Optional[dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Translate ordinary differential equation to Lean 4

    Args:
        equation: The ODE (LaTeX or plain text)
        variables: Independent variables (e.g., ["t"])
        unknowns: Dependent variables (e.g., ["y"])
        initial_conditions: Optional initial conditions

    Returns:
        {
            "lean4_code": str,
            "formalization": str,
            "existence_proof": Optional[str]
        }
    """
    ode = DifferentialEquation(
        equation_type="ode",
        equation=equation,
        variables=variables,
        unknowns=unknowns,
        initial_conditions=initial_conditions
    )

    client = LeanAideClient()
    lean4_code = await client.translate_ode_to_lean4(ode, None)

    return {
        "lean4_code": lean4_code,
        "formalization": lean4_code,
        "existence_proof": None
    }

@mcp_tool
async def leanaide_verify_continuous_solution(
    equation: str,
    solution: str,
    domain: str
) -> Dict[str, Any]:
    """
    Verify a solution to a continuous mathematics problem

    Args:
        equation: The differential equation
        solution: Proposed solution
        domain: Application domain (medicine, physics, biology, etc.)

    Returns:
        {
            "equation_satisfied": bool,
            "conditions_satisfied": bool,
            "properties_verified": dict[str, bool],
            "domain_properties": dict[str, bool],
            "overall_valid": bool
        }
    """
    # Parse and verify
    client = LeanAideClient()
    result = await client.verify_continuous_solution(
        problem=None,  # Would parse from equation
        proposed_solution=solution,
        lean4_formalization=None
    )

    return result
```

**Deliverable**: New MCP tools in `leanaide_mcp_tools.py`

---

## Integration Tasks

### Task 6.1: Update Workflow Stage Integration
**File**: `workflow_enhanced_stages.py`

```python
# Stage 0: Enhanced content analysis
async def run_content_analysis(problem_statement: str) -> ContentAnalysisResult:
    """Now includes continuous math detection"""
    # Check for continuous mathematics
    leanaide_client = LeanAideClient()
    math_type, confidence = await leanaide_client.detect_mathematics_type(problem_statement)

    result.is_continuous_math = (math_type in [
        MathematicalDomain.ODE, MathematicalDomain.PDE,
        MathematicalDomain.DAE, MathematicalDomain.SDE
    ])

    if result.is_continuous_math:
        result.continuous_math_type = math_type.value
        result.requires_continuous_solver = True

    return result

# Stage 1: Enhanced decomposition
async def run_ai_decomposition(plan: DecompositionPlan) -> DecompositionPlan:
    """Now handles continuous math decomposition"""
    if plan.content_analysis.is_continuous_math:
        # Use continuous math decomposition strategy
        continuous_decomp = await _decompose_continuous_problem(plan)
        plan.sub_problems.extend(continuous_decomp)
    else:
        # Use existing discrete math decomposition
        pass

    return plan
```

---

## Testing Tasks

### Task 7.1: Unit Tests
**File**: `tests/test_leanaide_continuous_math.py`

```python
import pytest

@pytest.mark.asyncio
async def test_continuous_math_detection():
    """Test detection of continuous mathematics"""
    client = LeanAideClient()

    # ODE example
    ode_problem = "Solve dy/dt = -ky with y(0) = y0"
    domain, confidence = await client.detect_mathematics_type(ode_problem)
    assert domain == MathematicalDomain.ODE
    assert confidence > 0.8

    # PDE example
    pde_problem = "Solve the heat equation ∂u/∂t = α ∂²u/∂x²"
    domain, confidence = await client.detect_mathematics_type(pde_problem)
    assert domain == MathematicalDomain.PDE

    # Discrete math example (should not be detected as continuous)
    discrete_problem = "Prove that √2 is irrational"
    domain, confidence = await client.detect_mathematics_type(discrete_problem)
    assert domain not in [MathematicalDomain.ODE, MathematicalDomain.PDE]

@pytest.mark.asyncio
async def test_ode_translation():
    """Test ODE to Lean 4 translation"""
    client = LeanAideClient()

    ode = DifferentialEquation(
        equation_type="ode",
        equation="dy/dt = -ky",
        variables=["t"],
        unknowns=["y"],
        parameters={"k": 1.0},
        initial_conditions={"y0": 1.0}
    )

    lean4_code = await client.translate_ode_to_lean4(ode, None)
    assert "def" in lean4_code or "theorem" in lean4_code

@pytest.mark.asyncio
async def test_domain_pattern_identification():
    """Test scientific domain pattern identification"""
    solver = ScientificDomainSolver()

    # Epidemiology problem
    epi_problem = ContinuousMathProblem(
        problem_type="ode",
        equations=[],
        domain="medicine",
        goal="Model disease spread"
    )

    domain, pattern = await solver.identify_domain_pattern(epi_problem)
    assert domain == "medicine"
    assert pattern == "epidemiology"
```

### Task 7.2: Integration Tests
**File**: `tests/test_leanaide_continuous_integration.py`

```python
@pytest.mark.asyncio
async def test_end_to_end_continuous_workflow():
    """Test complete continuous math workflow"""
    # Setup
    problem = "Model the spread of a disease using SIR model"
    bridge = LeanAideHephaestusBridge()

    # Phase 1: Analysis
    phase1_result = await bridge.execute_phase_1_setup(problem_text=problem)
    assert phase1_result["math_type"] == "continuous"
    assert phase1_result["domain"] == "medicine"

    # Phase 2: Translation
    phase2_result = await bridge.execute_phase_2_translation(
        continuous_problem=phase1_result["continuous_problem"]
    )
    assert phase2_result["lean4_code"] is not None

    # Phase 5: Verification
    phase5_result = await bridge.execute_phase_5_verification(
        solution=proposed_solution,
        formalization=phase2_result["lean4_code"]
    )
    assert phase5_result["equation_satisfied"] == True
```

---

## Documentation Tasks

### Task 8.1: Update LeanAide Documentation
**File**: `docs/LEANAIDE_CONTINUOUS_MATH.md`

Document:
- New continuous math capabilities
- API reference for ODE/PDE translation
- Domain patterns supported
- Verification capabilities
- MCP tools for continuous math
- Examples and use cases

### Task 8.2: Create Examples
**File**: `examples/leanaide_continuous_math_examples.py`

```python
# Example 1: SIR Model (Epidemiology)
async def sir_model_example():
    """Translate SIR model to Lean 4"""
    problem = """
    SIR epidemic model:
    dS/dt = -βSI
    dI/dt = βSI - γI
    dR/dt = γI

    Where S=susceptible, I=infected, R=recovered
    """
    client = LeanAideClient()
    lean4_code = await client.translate_system_to_lean4(problem)
    print(lean4_code)

# Example 2: Heat Equation (Physics)
async def heat_equation_example():
    """Translate heat equation to Lean 4"""
    problem = "∂u/∂t = α ∂²u/∂x² with u(x,0) = f(x)"
    client = LeanAideClient()
    lean4_code = await client.translate_pde_to_lean4(problem)
    print(lean4_code)
```

---

## Success Criteria

Phase 2 is complete when:

- [ ] Continuous math detection implemented and tested
- [ ] ODE translation to Lean 4 working
- [ ] PDE translation to Lean 4 working
- [ ] Scientific domain patterns implemented (medicine, physics, biology, engineering)
- [ ] Verification for continuous solutions working
- [ ] MCP tools for continuous math added
- [ ] All unit and integration tests passing
- [ ] Documentation complete
- [ ] Examples provided
- [ ] LeanAide now handles 80% of FRM's mathematical value

---

## Timeline

| Week | Component | Status |
|------|-----------|--------|
| 1 (Days 1-4) | Continuous Math Detection | Pending |
| 1 (Days 5-7) + Week 2 | ODE/PDE Translation | Pending |
| Week 2 (Days 4-7) | Scientific Domain Patterns | Pending |
| Week 3 (Days 1-5) | Verification for Continuous Math | Pending |
| Week 3 (Days 6-7) | MCP Tools & Integration | Pending |

---

## Dependencies

**Required Python Packages** (likely already present):
```txt
sympy>=1.11  # For symbolic math verification
numpy>=1.24  # For numerical checks
```

**External Services**: None (uses existing LLM integration)

---

## Comparison with FRM Integration

| Aspect | LeanAide Enhancement | FRM Integration |
|--------|---------------------|-----------------|
| **Effort** | 2-3 weeks | 3-5 weeks |
| **Maintenance** | Low (same tech stack) | High (separate tech stack) |
| **Math Coverage** | Continuous + Discrete | Continuous only |
| **Architecture** | No changes | REST/MCP bridge required |
| **Integration** | Already integrated (90%+) | Not integrated |
| **Value** | 80% of FRM value | 100% of FRM value |

---

## Notes

- **Higher ROI**: 2-3 weeks for 80% of FRM's value vs 3-5 weeks for full FRM
- **Leverages existing**: Builds on LeanAide's existing integration and infrastructure
- **No architectural debt**: Same Python tech stack, no separate service
- **Complementary**: Works alongside existing discrete math capabilities

---

**Task File Created**: 2025-12-31
**Source**: FRM Integration Analysis Recommendation
**Status**: READY FOR IMPLEMENTATION
