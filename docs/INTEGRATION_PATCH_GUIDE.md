# Integration Guide: Adding LeanAide to Existing Decomposition Engine

## Quick Start Integration

This guide shows how to integrate the LeanAide enhancements into the existing decomposition engine with minimal code changes.

## Option 1: Drop-in Replacement (Recommended)

Simply replace the import and instantiation:

```python
# Before
from decomposition_engine import DecompositionEngine

engine = DecompositionEngine(problem_analyzer, knowledge_manager)
plan = engine.decompose(problem)

# After
from decomposition_engine_lean_enhanced import LeanEnhancedDecompositionEngine

engine = LeanEnhancedDecompositionEngine(
    problem_analyzer=problem_analyzer,
    knowledge_manager=knowledge_manager,
    enable_lean_detection=True,    # Enable LeanAide
    enable_evolution=True          # Enable evolutionary
)
plan = await engine.decompose_with_leanaide(problem)  # New method
```

**Benefits:**
- ✅ Only 2 lines changed
- ✅ Automatic mathematical problem detection
- ✅ Backward compatible (existing code still works)
- ✅ Graceful fallback if LeanAide unavailable

## Option 2: Gradual Integration

Add LeanAide as an optional enhancement:

```python
from decomposition_engine import DecompositionEngine
from decomposition_engine_lean_enhanced import LeanEnhancedDecompositionEngine

# Use standard decomposition
engine = DecompositionEngine(problem_analyzer, knowledge_manager)

# Check if problem is mathematical
from decomposition_engine_lean_enhanced import LeanMathematicalDetector

detector = LeanMathematicalDetector()
metadata = detector.detect_mathematical_problem(problem.description, problem.title)

if metadata.is_mathematical:
    # Use Lean-enhanced engine for mathematical problems
    lean_engine = LeanEnhancedDecompositionEngine(
        problem_analyzer=problem_analyzer,
        knowledge_manager=knowledge_manager,
        enable_lean_detection=True
    )
    plan = await lean_engine.decompose_with_leanaide(problem)
else:
    # Use standard engine for non-mathematical problems
    plan = engine.decompose(problem)
```

**Benefits:**
- ✅ Gradual rollout
- ✅ Test LeanAide on specific problems
- ✅ Easy to disable if issues arise

## Option 3: Configuration-Based Integration

Use configuration to control LeanAide integration:

```python
import yaml

# Load configuration
with open("decomposition_config_lean.yaml") as f:
    config = yaml.safe_load(f)

# Conditional instantiation
if config["leanaide"]["enabled"]:
    from decomposition_engine_lean_enhanced import LeanEnhancedDecompositionEngine
    engine = LeanEnhancedDecompositionEngine(
        problem_analyzer=problem_analyzer,
        knowledge_manager=knowledge_manager,
        enable_lean_detection=config["leanaide"]["enabled"],
        enable_evolution=config["evolutionary"]["enabled"]
    )
else:
    from decomposition_engine import DecompositionEngine
    engine = DecompositionEngine(problem_analyzer, knowledge_manager)
```

**Benefits:**
- ✅ Configuration-driven
- ✅ No code changes to enable/disable
- ✅ Easy to tune parameters

## Minimal Integration Example

The smallest possible integration:

```python
# Single line change to enable LeanAide detection
from decomposition_engine_lean_enhanced import detect_and_route_mathematical_problem

# Use in place of standard decomposition
plan, metadata = await detect_and_route_mathematical_problem(problem)

if metadata.is_mathematical:
    print(f"Using LeanAide decomposition for {metadata.domain.value} problem")
else:
    print("Using standard decomposition")

# plan is either a DecompositionPlan (if mathematical) or None (if not)
if plan is None:
    # Fall back to standard decomposition
    from decomposition_engine import DecompositionEngine
    engine = DecompositionEngine()
    plan = engine.decompose(problem)
```

## Adding to Existing Workflow

### Step 1: Update Imports

```python
# Add these imports to your existing file
from decomposition_engine_lean_enhanced import (
    LeanMathematicalDetector,
    LeanEnhancedDecompositionEngine,
    detect_and_route_mathematical_problem,
    generate_evolutionary_config
)
```

### Step 2: Update Initialization

```python
# In your initialization code
class ProblemDecompositionService:
    def __init__(self):
        # Existing initialization
        self.problem_analyzer = ProblemAnalyzer()
        self.knowledge_manager = KnowledgeManager()

        # Add Lean-enhanced engine
        self.lean_engine = LeanEnhancedDecompositionEngine(
            problem_analyzer=self.problem_analyzer,
            knowledge_manager=self.knowledge_manager,
            enable_lean_detection=True,
            enable_evolution=True
        )

        # Keep standard engine for fallback
        self.standard_engine = DecompositionEngine(
            self.problem_analyzer,
            self.knowledge_manager
        )
```

### Step 3: Update Decomposition Method

```python
    async def decompose_problem(self, problem: ProblemDefinition):
        """Decompose a problem with LeanAide integration."""
        # Try Lean-enhanced decomposition first
        plan = await self.lean_engine.decompose_with_leanaide(problem)

        # Check if Lean decomposition was used
        if plan.metadata.get("lean_decomposition"):
            logger.info(f"Used LeanAide decomposition for {problem.id}")
            domain = plan.metadata.get("mathematical_domain")
            logger.info(f"Mathematical domain: {domain}")

            # Process Lean sub-problems
            lean_subproblems = [
                sp for sp in plan.sub_problems
                if sp.metadata.get("lean_formalization")
            ]
            logger.info(f"Created {len(lean_subproblems)} Lean-formalizable sub-problems")

        return plan
```

### Step 4: Add Evolutionary Processing (Optional)

```python
    async def process_lean_subproblem(self, sub_problem):
        """Process a Lean sub-problem with evolutionary generation."""
        metadata = sub_problem.metadata.get("mathematical_metadata")

        if not metadata or not metadata.get("requires_evolution"):
            # No evolution needed, process normally
            return await self.standard_process(sub_problem)

        # Generate evolutionary configuration
        config = await generate_evolutionary_config(metadata)

        if not config.get("enable_evolution"):
            return await self.standard_process(sub_problem)

        # Use LeanAide evolutionary engine
        from leanaide_evolution import LeanProofEvolutionEngine

        evolution_engine = LeanProofEvolutionEngine(
            theorem=sub_problem.description,
            **config
        )

        result = await evolution_engine.evolve()

        if result.success:
            logger.info(f"Evolution succeeded: {result.best_proof.theorem_name}")
            return result.best_proof
        else:
            logger.warning(f"Evolution failed for {sub_problem.id}")
            return await self.standard_process(sub_problem)
```

## Testing Integration

### Create Simple Test

```python
import asyncio
from decomposition_engine_lean_enhanced import detect_and_route_mathematical_problem
from sovereign_data_models import ProblemDefinition, DomainContext, ComplexityScore

async def test_leanaide_integration():
    """Test LeanAide integration."""
    # Create a mathematical problem
    problem = ProblemDefinition(
        id="test_001",
        title="Infinite Primes",
        description="Prove that there are infinitely many prime numbers.",
        problem_type="theorem_proof",
        domain_context=DomainContext(
            domain="number_theory",
            subdomain=None,
            related_domains=[],
            domain_knowledge={}
        ),
        complexity_score=ComplexityScore(
            cognitive_complexity=6.0,
            computational_complexity=2.0,
            domain_complexity=5.0,
            integration_complexity=3.0,
            overall_complexity=4.0,
            explanation="Test"
        ),
        constraints=[],
        success_criteria=[],
        stakeholders=[],
        resources_available={}
    )

    # Test detection and routing
    plan, metadata = await detect_and_route_mathematical_problem(problem)

    # Verify results
    assert metadata is not None, "Metadata should not be None"
    assert metadata.is_mathematical, "Should be detected as mathematical"
    assert metadata.domain.value == "number_theory", "Should be number theory"
    assert plan is not None, "Plan should be created"

    print("✓ LeanAide integration test passed")
    print(f"  Domain: {metadata.domain.value}")
    print(f"  Difficulty: {metadata.proof_difficulty}/10")
    print(f"  Sub-problems: {len(plan.sub_problems)}")

if __name__ == "__main__":
    asyncio.run(test_leanaide_integration())
```

## Migration Checklist

- [ ] Install new files in project directory
  - [ ] `decomposition_engine_lean_enhanced.py`
  - [ ] `decomposition_config_lean.yaml`
  - [ ] `test_decomposition_lean_integration.py`

- [ ] Review documentation
  - [ ] Read `LEANAIDE_DECOMPOSITION_INTEGRATION.md`
  - [ ] Review `decomposition_engine_lean_quick_reference.md`
  - [ ] Check configuration options

- [ ] Update code
  - [ ] Add imports for Lean-enhanced engine
  - [ ] Update initialization (if using Option 1)
  - [ ] Add conditional routing (if using Option 2)
  - [ ] Update configuration loading (if using Option 3)

- [ ] Test integration
  - [ ] Run test suite: `pytest test_decomposition_lean_integration.py -v`
  - [ ] Test with mathematical problems
  - [ ] Test with non-mathematical problems (verify fallback)
  - [ ] Test error handling

- [ ] Configure
  - [ ] Adjust detection thresholds (if needed)
  - [ ] Configure LeanAide server URL
  - [ ] Set evolutionary parameters
  - [ ] Enable/disable ROMA integration
  - [ ] Enable/disable Hephaestus integration

- [ ] Deploy
  - [ ] Deploy to development environment
  - [ ] Monitor logs for issues
  - [ ] Test with real problems
  - [ ] Adjust configuration based on results

- [ ] Document
  - [ ] Update team documentation
  - [ ] Add examples to wiki
  - [ ] Share configuration guidelines
  - [ ] Document any custom settings

## Common Integration Patterns

### Pattern 1: Side-by-Side Engines

Run both engines and compare results:

```python
# Decompose with both engines
standard_plan = standard_engine.decompose(problem)
lean_plan = await lean_engine.decompose_with_leanaide(problem)

# Compare and choose best
if lean_plan.quality_scores.overall_score > standard_plan.quality_scores.overall_score:
    chosen_plan = lean_plan
    strategy = "LeanAide"
else:
    chosen_plan = standard_plan
    strategy = "Standard"

logger.info(f"Selected {strategy} decomposition (score: {chosen_plan.quality_scores.overall_score})")
```

### Pattern 2: Progressive Enhancement

Start with standard, enhance with LeanAide:

```python
# Start with standard decomposition
plan = standard_engine.decompose(problem)

# Enhance with LeanAide if mathematical
if metadata.is_mathematical:
    # Enhance each sub-problem with Lean metadata
    for sp in plan.sub_problems:
        lean_metadata = detector.detect_mathematical_problem(sp.description, sp.title)
        if lean_metadata.is_mathematical:
            sp.metadata["mathematical_metadata"] = lean_metadata.to_dict()
            sp.metadata["lean_formalization"] = True
```

### Pattern 3: Hybrid Approach

Use LeanAide for decomposition, standard for execution:

```python
# Decompose with LeanAide
lean_plan = await lean_engine.decompose_with_leanaide(problem)

# Extract sub-problems
lean_subproblems = [
    sp for sp in lean_plan.sub_problems
    if sp.metadata.get("lean_formalization")
]

# Execute with standard workflow
for sp in lean_subproblems:
    # Process with standard workflow
    result = await workflow.execute_subproblem(sp)
```

## Troubleshooting Integration Issues

### Issue: Import Errors

```python
# Error: ImportError: No module named 'decomposition_engine_lean_enhanced'

# Solution: Make sure the file is in the correct location
import sys
sys.path.append("/path/to/Frontend")  # Add to path if needed

# Or verify file location
import os
print(os.path.exists("decomposition_engine_lean_enhanced.py"))
```

### Issue: Async/Await Errors

```python
# Error: await outside of async function

# Solution: Make sure calling function is async
async def decompose_with_leanaide(self, problem):
    # This works
    plan = await self.lean_engine.decompose_with_leanaide(problem)
    return plan

# This doesn't work
def decompose_with_leanaide(self, problem):
    plan = await self.lean_engine.decompose_with_leanaide(problem)  # ERROR!
    return plan
```

### Issue: Missing Dependencies

```python
# Error: NameError: name 'LeanEnhancedDecompositionEngine' is not defined

# Solution: Import the class
from decomposition_engine_lean_enhanced import LeanEnhancedDecompositionEngine

# Or use try/except for graceful degradation
try:
    from decomposition_engine_lean_enhanced import LeanEnhancedDecompositionEngine
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False
    print("Warning: LeanAide integration not available")
```

## Verification

After integration, verify:

```python
# Verification script
async def verify_integration():
    """Verify LeanAide integration is working."""

    # 1. Check imports
    try:
        from decomposition_engine_lean_enhanced import (
            LeanMathematicalDetector,
            LeanEnhancedDecompositionEngine
        )
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    # 2. Check detector
    detector = LeanMathematicalDetector()
    metadata = detector.detect_mathematical_problem(
        "Prove that sqrt(2) is irrational"
    )
    if metadata.is_mathematical:
        print("✓ Detector working")
    else:
        print("✗ Detector not working")
        return False

    # 3. Check engine
    try:
        engine = LeanEnhancedDecompositionEngine(
            enable_lean_detection=True,
            enable_evolution=True
        )
        print("✓ Engine initialization successful")
    except Exception as e:
        print(f"✗ Engine initialization failed: {e}")
        return False

    # 4. Check full workflow
    try:
        problem = ProblemDefinition(...)  # Create test problem
        plan = await engine.decompose_with_leanaide(problem)
        if plan:
            print("✓ Full workflow working")
        else:
            print("✗ Workflow returned None")
            return False
    except Exception as e:
        print(f"✗ Workflow failed: {e}")
        return False

    print("\n✓ All integration checks passed!")
    return True

if __name__ == "__main__":
    asyncio.run(verify_integration())
```

## Summary

The LeanAide integration can be added with:

- **Option 1:** 2 lines changed (drop-in replacement)
- **Option 2:** 10-15 lines (gradual integration)
- **Option 3:** 20-30 lines (configuration-based)

All options are:
- ✅ Backward compatible
- ✅ Gracefully degradable
- ✅ Fully tested
- ✅ Production ready

Choose the option that best fits your use case and deployment strategy.
