# LLTL Integration Guide

**Quick Start Guide for Integrating Logic-to-Loss Translation Layer**

---

## Installation

No additional installation required! LLTL uses dependencies already in the project:

- PyTorch (optional but recommended): `pip install torch`
- NumPy: Already installed
- NetworkX: Already installed

---

## Quick Start (5 Minutes)

### Step 1: Import LLTL

```python
from rese.core.symbolic_constraint_engine import (
    Constraint, ConstraintType, SymbolicConstraintEngine
)
from rese.core.logic_to_loss_translation import create_lltl_from_sce
```

### Step 2: Create Constraints

```python
# Create constraint engine
sce = SymbolicConstraintEngine()

# Add a hard constraint
sce.add_constraint(Constraint(
    id="temp_limit",
    type=ConstraintType.HARD,
    description="Temperature < 1000°C",
    formalization="forall (T : Temperature), T < 1000",
    source="user_prompt"
))

# Add a soft constraint
sce.add_constraint(Constraint(
    id="pressure_limit",
    type=ConstraintType.SOFT,
    description="Pressure < 10 bar",
    formalization="forall (P : Pressure), P < 10",
    source="system_inferred"
))
```

### Step 3: Translate to Loss Functions

```python
# One-line translation
lltl = create_lltl_from_sce(sce)

# Check results
print(f"Translated {len(lltl.loss_functions)} constraints")
```

### Step 4: Use Loss Functions

```python
import torch

# Create input variables
inputs = {
    "temperature": torch.tensor([900.0]),
    "pressure": torch.tensor([8.0]),
}

# Compute total loss
loss = lltl.compute_total_loss(inputs)
print(f"Loss: {loss.item():.4f}")

# Check violations
violations = lltl.get_loss_violations(inputs)
for cid, viol in violations.items():
    status = "VIOLATED" if viol["violated"] else "OK"
    print(f"{cid}: {status}")
```

---

## Integration Patterns

### Pattern 1: Batch Validation

Best for: Post-generation validation

```python
from rese.core.stage5_integration import GeneratorValidator

# Create validator
validator = GeneratorValidator(
    sce=sce,
    feedback_mode=FeedbackMode.BATCH,
)

# Validate after generation
def generate_with_validation(prompt):
    output = my_generator.generate(prompt)

    # Validate
    should_continue, state, signal = validator.validate_step(output)

    if not should_continue:
        print("Constraint violation detected!")
        return None

    return output
```

### Pattern 2: Real-Time Optimization

Best for: Iterative improvement during generation

```python
from rese.core.stage5_integration import Stage5Integration

# Create integration
integration = Stage5Integration(
    lltl=lltl,
    sce=sce,
    feedback_mode=FeedbackMode.REALTIME,
    feedback_strategy=FeedbackStrategy.BACKPROPAGATE,
)

def generate_with_optimization(prompt):
    # Initial generation
    output = my_generator.generate(prompt)

    # Optimize to satisfy constraints
    for iteration in range(100):
        # Monitor
        state = integration.monitor_generation(output)
        signal = integration.generate_feedback(state)

        # Check if satisfied
        if not signal.should_backpropagate:
            break

        # Backpropagate and adjust
        state.loss.backward()

        with torch.no_grad():
            for var in output.values():
                if var.grad is not None:
                    var -= 0.01 * var.grad

    return output
```

### Pattern 3: Early Stopping

Best for: Preventing invalid generations

```python
from rese.core.stage5_integration import FeedbackStrategy

# Create integration with early stopping
integration = Stage5Integration(
    lltl=lltl,
    sce=sce,
    feedback_strategy=FeedbackStrategy.STOP_ON_HARD,
    max_violations=3,
)

def generate_with_early_stopping(prompt):
    output = None

    for step in range(10):
        # Generate partial output
        output = my_generator.generate_step(prompt, step)

        # Validate
        state = integration.monitor_generation(output, step=step)
        signal = integration.generate_feedback(state)

        # Stop if constraint violated
        if signal.should_stop:
            print(f"Stopping at step {step}")
            return output

    return output
```

---

## Common Integration Scenarios

### Scenario 1: E2E Invention Engine

```python
from end_to_end_invention_planner import EndToEndInventionPlanner

class IntegratedInventionEngine:
    def __init__(self):
        # Create components
        self.planner = EndToEndInventionPlanner()
        self.validator = GeneratorValidator(
            sce=self.planner.constraint_engine,
        )

    def plan_invention(self, prompt):
        # Generate plan (Stages 1-4)
        plan = self.planner.generate_plan(prompt)

        # Validate each step (Stage 5)
        for step in plan.steps:
            should_continue, state, signal = self.validator.validate_step(
                step.variables
            )

            if not should_continue:
                # Fix constraint violation
                step = self.fix_step(step, signal)

        return plan
```

### Scenario 2: LLM Output Validation

```python
class LLMWithConstraints:
    def __init__(self, sce):
        self.lltl = create_lltl_from_sce(sce)
        self.llm = MyLLM()

    def generate_constrained(self, prompt):
        # Generate text
        text = self.llm.generate(prompt)

        # Extract variables
        variables = self.extract_variables(text)

        # Check constraints
        violations = self.lltl.get_loss_violations(variables)

        if any(v["violated"] for v in violations.values()):
            print("Constraint violations detected:")
            for cid, viol in violations.items():
                if viol["violated"]:
                    print(f"  - {cid}: {viol['description']}")

            # Regenerate or adjust
            text = self.adjust_text(text, violations)

        return text
```

### Scenario 3: Physics Simulation

```python
class ConstrainedSimulation:
    def __init__(self, sce):
        self.lltl = create_lltl_from_sce(sce)

    def run_simulation(self, initial_conditions):
        state = initial_conditions

        for t in range(1000):
            # Physics step
            state = self.physics_step(state)

            # Check constraints
            loss = self.lltl.compute_total_loss(state)

            if loss > 100:
                print(f"Constraint violation at t={t}")
                break

            # Constrain state
            if loss > 0:
                loss.backward()
                with torch.no_grad():
                    for var in state.values():
                        if var.grad is not None:
                            var -= 0.1 * var.grad

        return state
```

---

## Configuration Guide

### Choosing Aggregation Method

| Method | Use Case | Example |
|--------|----------|---------|
| `WEIGHTED_SUM` | General purpose | Most applications |
| `LEXICOGRAPHIC` | Prioritize constraints | Hard > Soft > Preference |
| `MAX` | Minimize worst violation | Safety-critical systems |
| `PRODUCT` | Multiple violations | All constraints important |
| `ADAPTIVE` | Dynamic weighting | Unknown constraint importance |

### Choosing Feedback Strategy

| Strategy | Use Case | Behavior |
|----------|----------|----------|
| `STOP_ON_HARD` | Safety-critical | Stop immediately on hard violations |
| `BACKPROPAGATE` | Neural systems | Use gradients to fix violations |
| `REGENERATE` | Symbolic systems | Regenerate violating portions |
| `ADJUST_WEIGHTS` | Unknown importance | Learn weights during generation |
| `IGNORE_PREFERENCE` | Focus on essentials | Only enforce hard/soft constraints |

### Choosing Feedback Mode

| Mode | Use Case | Performance |
|------|----------|-------------|
| `REALTIME` | Continuous monitoring | Higher overhead |
| `BATCH` | Post-step validation | Balanced |
| `ON_VIOLATION` | Sparse monitoring | Lower overhead |
| `ADAPTIVE` | Dynamic monitoring | Variable overhead |

---

## Debugging Tips

### Enable Logging

```python
import logging

# Enable LLTL logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rese.core.logic_to_loss_translation")
logger.setLevel(logging.DEBUG)
```

### Export Loss Functions

```python
# Export to JSON for inspection
lltl.export_loss_functions("loss_functions.json")

# View the file to see:
# - Constraint IDs
# - Descriptions
# - Weights
# - Fuzzy types
```

### Export Generation History

```python
# After generation
integration.export_history("generation_history.json")

# Analyze:
# - Loss over time
# - Violation patterns
# - Constraint satisfaction
```

### Inspect Violations

```python
violations = lltl.get_loss_violations(inputs)

# Detailed inspection
for cid, viol in violations.items():
    print(f"\nConstraint: {cid}")
    print(f"  Description: {viol['description']}")
    print(f"  Type: {viol['type']}")
    print(f"  Violated: {viol['violated']}")
    print(f"  Loss: {viol['loss']:.4f}")
    print(f"  Severity: {viol['severity']:.2f}")
```

---

## Performance Optimization

### Tip 1: Use GPU

```python
lltl = create_lltl_from_sce(
    sce,
    device="cuda",  # Use GPU if available
)
```

### Tip 2: Limit Constraints

```python
# Only translate what you need
results = lltl.translate_sce(
    sce,
    constraint_filter=lambda c: c.type in [ConstraintType.HARD, ConstraintType.SOFT]
)
```

### Tip 3: Batch Processing

```python
# Validate multiple steps at once
batch = [step1_vars, step2_vars, step3_vars]
results = validator.validate_batch(batch)
```

### Tip 4: Clear Cache

```python
# Periodically clear to save memory
if step % 100 == 0:
    lltl.clear_cache()
    integration.reset()
```

---

## Testing Your Integration

### Unit Test Example

```python
import unittest

class TestMyIntegration(unittest.TestCase):
    def setUp(self):
        self.sce = create_my_constraints()
        self.lltl = create_lltl_from_sce(self.sce)

    def test_loss_computation(self):
        inputs = {"x": torch.tensor([1.0])}
        loss = self.lltl.compute_total_loss(inputs)
        self.assertIsInstance(loss, torch.Tensor)

    def test_constraint_satisfaction(self):
        inputs = {"x": torch.tensor([1.0])}
        violations = self.lltl.get_loss_violations(inputs)
        self.assertFalse(
            any(v["violated"] for v in violations.values())
        )
```

### Integration Test Example

```python
def test_end_to_end():
    # Create system
    system = MyConstrainedSystem()

    # Generate
    output = system.generate("test prompt")

    # Validate
    assert output is not None
    assert system.validator.get_summary()["total_violations"] == 0
```

---

## FAQ

**Q: Do I need PyTorch?**
A: No, LLTL falls back to NumPy. But PyTorch is recommended for automatic differentiation.

**Q: How many constraints can I use?**
A: Tested up to 1000. For more, consider constraint grouping.

**Q: Can I use LLTL with my existing generator?**
A: Yes! LLTL works with any generator that produces numerical outputs.

**Q: How do I handle conflicting constraints?**
A: Use weights to prioritize, or wait for DITO (Agent A3) for automatic resolution.

**Q: Can I use LLTL for discrete variables?**
A: LLTL works best with continuous variables. For discrete, use relaxation or embeddings.

---

## Getting Help

1. Check the [Full Documentation](LLTL_DOCUMENTATION.md)
2. Run the test suite for examples
3. Export loss functions and history for debugging
4. Enable logging for detailed information

---

## Checklist for Integration

- [ ] Import LLTL modules
- [ ] Create SymbolicConstraintEngine
- [ ] Add constraints
- [ ] Create LogicToLossTranslator
- [ ] Test loss computation
- [ ] Choose integration pattern
- [ ] Implement validation
- [ ] Test with sample data
- [ ] Optimize performance
- [ ] Add logging/debugging
- [ ] Write tests
- [ ] Document your integration

---

**Ready to integrate!** Follow the Quick Start (5 minutes) or choose an Integration Pattern above.

For advanced usage, see [Full Documentation](LLTL_DOCUMENTATION.md).
