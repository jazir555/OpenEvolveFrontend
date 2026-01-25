# RESE Examples

This directory contains comprehensive examples demonstrating how to use the RESE (Recursive Epistemic Solvability Engine) framework.

## Overview

RESE is a four-phase formal methodology that transforms intractable problems into tractable ones:
- **Phase I:** Epistemic Audit - Systematic falsification of assumptions
- **Phase II:** Isomorphic Resonance - Cross-domain knowledge transfer
- **Phase III:** Monte Carlo Refinement - ACI-guided adaptive search
- **Phase IV:** Architectural Synthesis - Validated solution assembly

## Examples List

### Basic Examples

#### Example 01: Quick Start ([example01_quickstart.py](example01_quickstart.py))
**Level:** Beginner
**Time:** 5 minutes

Your first RESE pipeline! Demonstrates basic usage to solve a simple routing problem.
```bash
python example01_quickstart.py
```

**What you'll learn:**
- How to define a problem
- How to specify constraints
- How to run the RESE pipeline
- How to interpret results

---

#### Example 02: Symbolic Constraint Engine ([example02_sce_basic.py](example02_sce_basic.py))
**Level:** Beginner
**Time:** 10 minutes

Direct usage of the Symbolic Constraint Engine (SCE) - the foundation of RESE.
```bash
python example02_sce_basic.py
```

**What you'll learn:**
- Creating and managing constraints
- Detecting conflicts
- Understanding constraint types (HARD, SOFT, PREFERENCE)
- Using dependency tracking

---

#### Example 03: Cognitive Bias Detection ([example03_cognitive_biases.py](example03_cognitive_biases.py))
**Level:** Beginner
**Time:** 10 minutes

Detect and mitigate cognitive biases in problem formulation.
```bash
python example03_cognitive_biases.py
```

**What you'll learn:**
- Using Φ₂ (Cognitive Bias Detector)
- Identifying common biases (confirmation bias, sunk cost, etc.)
- Applying debiasing interventions
- Interpreting bias reports

---

### Intermediate Examples

#### Example 04: ACI Calculator ([example04_aci_calculator.py](example04_aci_calculator.py))
**Level:** Intermediate
**Time:** 15 minutes

Calculate the Algorithmic Complexity Index (ACI) for constraint satisfaction problems.
```bash
python example04_aci_calculator.py
```

**What you'll learn:**
- Understanding ACI components (H, C, S)
- Calculating disorder entropy
- Calculating causal coherence
- Calculating solvability index
- Interpreting ACI scores

---

#### Example 05: Isomorphism Validation ([example05_imech.py](example05_imech.py))
**Level:** Intermediate
**Time:** 15 minutes

Use I_mech to validate mechanistic similarity between domains for knowledge transfer.
```bash
python example05_imech.py
```

**What you'll learn:**
- Defining problem domains
- Comparing domains for similarity
- Transferring knowledge across domains
- Interpreting isomorphism scores

---

#### Example 06: MCTS Search ([example06_mcts_search.py](example06_mcts_search.py))
**Level:** Intermediate
**Time:** 15 minutes

ACI-guided Monte Carlo Tree Search for optimization problems.
```bash
python example06_mcts_search.py
```

**What you'll learn:**
- Setting up MCTS search
- Using ACI to guide exploration
- Balancing exploration vs exploitation
- Monitoring search convergence

---

#### Example 07: Custom Integration ([example07_custom_integration.py](example07_custom_integration.py))
**Level:** Intermediate
**Time:** 20 minutes

Create custom phases and integrate them into the RESE pipeline.
```bash
python example07_custom_integration.py
```

**What you'll learn:**
- Creating custom phase executors
- Implementing phase logic
- Integrating with RESE pipeline
- Extending RESE functionality

---

#### Example 08: Configuration Management ([example08_configuration.py](example08_configuration.py))
**Level:** Intermediate
**Time:** 15 minutes

Configure and customize RESE behavior for different environments.
```bash
python example08_configuration.py
```

**What you'll learn:**
- Loading configuration
- Customizing phase parameters
- Environment-specific settings
- Saving and loading configs
- Feature flags

---

### Advanced Examples

#### Example 09: Solution Validation ([example09_validation.py](example09_validation.py))
**Level:** Advanced
**Time:** 20 minutes

Validate solutions using ACI reduction and statistical testing.
```bash
python example09_validation.py
```

**What you'll learn:**
- Using Δ₃ (ACI Reduction Validator)
- Calculating ACI reduction
- Statistical significance testing
- Comparing multiple solutions
- Non-circular validation

---

#### Example 10: End-to-End Pipeline ([example10_end_to_end.py](example10_end_to_end.py))
**Level:** Advanced
**Time:** 30 minutes

Complete end-to-end RESE pipeline for solving a resource allocation problem.
```bash
python example10_end_to_end.py
```

**What you'll learn:**
- Defining complex problems
- Configuring the pipeline
- Running all 4 phases
- Interpreting comprehensive results
- Making recommendations

---

#### Example 11: Error Handling ([example11_error_handling.py](example11_error_handling.py))
**Level:** All Levels
**Time:** 20 minutes

Error handling and debugging techniques for RESE pipelines.
```bash
python example11_error_handling.py
```

**What you'll learn:**
- Common errors and solutions
- Error handling patterns
- Debugging with logging
- Performance profiling
- Best practices

---

## Running the Examples

### Prerequisites

1. Install RESE:
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
pip install -e rese/
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

Run any example directly:
```bash
python example01_quickstart.py
```

### From Python

```python
import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

# Import and run example
from examples import example01_quickstart
example01_quickstart.main()
```

---

## Learning Path

We recommend following this order:

### Beginner (Getting Started)
1. **Example 01** - Quick Start
2. **Example 02** - SCE Basic
3. **Example 03** - Cognitive Biases

### Intermediate (Building Skills)
4. **Example 04** - ACI Calculator
5. **Example 05** - Isomorphism Validation
6. **Example 06** - MCTS Search
7. **Example 07** - Custom Integration
8. **Example 08** - Configuration

### Advanced (Mastering RESE)
9. **Example 09** - Solution Validation
10. **Example 10** - End-to-End Pipeline
11. **Example 11** - Error Handling

---

## Common Tasks

### Solve an Optimization Problem

See **Example 01** or **Example 10**

### Detect Biases in Constraints

See **Example 03**

### Calculate ACI for a Problem

See **Example 04**

### Transfer Knowledge Between Domains

See **Example 05**

### Create Custom Phase

See **Example 07**

### Validate a Solution

See **Example 09**

---

## Troubleshooting

### Import Errors

If you get import errors, make sure to add the path:
```python
import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')
```

### Missing Dependencies

Install required packages:
```bash
pip install networkx numpy torch
```

### Slow Performance

Enable caching in configuration (see **Example 08**)

---

## Additional Resources

- **User Guide:** `../docs/user_guide.md`
- **Developer Guide:** `../docs/developer_guide.md`
- **API Reference:** `../docs/api_reference.md`
- **Integration Guide:** `../docs/e2e_integration.md`
- **Troubleshooting:** `../docs/troubleshooting.md`

---

## Contributing Examples

Have a great example? We'd love to add it!

1. Create a new file: `example12_<name>.py`
2. Follow the existing format
3. Include comprehensive comments
4. Add to this README
5. Submit a pull request

---

**Happy Solving! 🚀**
