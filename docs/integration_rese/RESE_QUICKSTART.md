<<<<<<< HEAD
# RESE Quick Start Guide

## Table of Contents

1. [Installation](#installation)
2. [Setup and Configuration](#setup-and-configuration)
3. [Your First Example](#your-first-example)
4. [Common Use Cases](#common-use-cases)
5. [Next Steps](#next-steps)
6. [FAQ](#faq)

---

## Installation

### Prerequisites

**Required:**
- Python 3.9 or higher
- pip (Python package manager)

**Check your Python version:**
```bash
python --version
# Should show Python 3.9.x or higher
```

### Install RESE

**Option 1: Install from source (recommended for development)**
```bash
cd /path/to/OpenEvolve/Frontend
pip install -e .
```

**Option 2: Install dependencies only**
```bash
pip install numpy fastapi uvicorn pydantic networkx scipy
```

### Verify Installation

```bash
# Run quick start script
python rese/quickstart.py
```

**Expected output:**
```
================================================================================
 RESE Quick Start
================================================================================

Recursive Epistemic Solvability Engine - Quick Start Guide
This script will verify your installation and run a quick demo.

================================================================================
 Checking Dependencies
================================================================================
✓ numpy (required)
✓ fastapi (required)
✓ pydantic (required)
⚠ psutil (optional) - not installed
⚠ networkx (optional) - not installed
⚠ scipy (optional) - not installed

...

================================================================================
 Summary
================================================================================

Test Results:
--------------------------------------------------------------------------------
  Dependencies............................................ ✓ PASS
  Configuration........................................... ✓ PASS
  Pipeline................................................. ✓ PASS
  Monitoring............................................... ✓ PASS
  Demo.................................................... ✓ PASS

--------------------------------------------------------------------------------

🎉 All tests passed! RESE is ready to use.

Next Steps:
  1. Review configuration in config.json
  2. Start the API server: python -m rese.api
  3. View API docs: http://localhost:8000/docs
  4. Read the documentation: rese/docs/
```

---

## Setup and Configuration

### 1. Create Configuration File

**Generate default configuration:**
```python
from rese.config import create_default_config

config = create_default_config('config.json')
print("Configuration created at config.json")
```

**Or create manually (`config.json`):**
```json
{
  "environment": "development",
  "project_name": "rese",
  "version": "1.0.0",
  "phase1": {
    "sce_max_constraints": 10000,
    "phi15_enabled": true,
    "phi15_assumption_threshold": 0.6,
    "phi2_enabled": true,
    "phi2_bias_threshold": 0.5
  },
  "phase2": {
    "psi2_similarity_threshold": 0.7,
    "psi3_target_accuracy": 0.8,
    "imech_algorithm": "weisfeiler_lehman"
  },
  "phase3": {
    "gamma2_iterations": 1000,
    "gamma2_aci_guided": true,
    "convergence_enabled": true
  },
  "phase4": {
    "delta3_min_aci_reduction": 0.2,
    "delta3_validation_threshold": 0.7
  },
  "pipeline": {
    "enable_caching": true,
    "cache_ttl_seconds": 3600
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "enable_auth": false
  },
  "monitoring": {
    "log_level": "INFO"
  }
}
```

### 2. Load Configuration

```python
from rese.config import load_config

# Load from default location
config = load_config()

# Or from specific file
config = load_config(Path('my_config.json'))
```

### 3. Start API Server (Optional)

**Development mode:**
```bash
python -m rese.api
```

**Production mode:**
```bash
uvicorn rese.api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Verify API is running:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-12-31T12:00:00Z",
  "uptime_seconds": 1.5
}
```

---

## Your First Example

### Example 1: Simple Constraint Satisfaction

**Create file `my_first_rese.py`:**

```python
#!/usr/bin/env python3
"""
My First RESE Analysis
"""

from rese.rese_pipeline import run_rese

# Define problem
result = run_rese(
    problem_description="Find optimal allocation of resources",
    constraints=[
        {
            'id': 'c1',
            'type': 'hard',
            'description': 'Total cost must be less than $1000',
            'formalization': 'sum(cost_i) < 1000'
        },
        {
            'id': 'c2',
            'type': 'hard',
            'description': 'All resources must be positive',
            'formalization': 'all(x_i > 0)'
        },
        {
            'id': 'c3',
            'type': 'soft',
            'description': 'Maximize total utility',
            'formalization': 'maximize(sum(utility_i))'
        }
    ],
    variables={
        'cost': 'float',
        'utility': 'float',
        'quantity': 'integer'
    }
)

# Print results
print("=" * 80)
print("RESE Analysis Results")
print("=" * 80)
print(f"Status: {result.status.value}")
print(f"Elapsed: {result.elapsed_seconds:.2f} seconds")
print()

# Phase results
for phase_name, phase_result in result.phase_results.items():
    print(f"{phase_name.upper()}:")
    print(f"  Status: {phase_result.status.value}")
    print(f"  Elapsed: {phase_result.elapsed_seconds:.2f}s")

    if phase_result.metrics:
        print(f"  Metrics:")
        for key, value in phase_result.metrics.items():
            print(f"    {key}: {value}")
    print()

# Final solution
if result.final_solution:
    print("FINAL SOLUTION:")
    for key, value in result.final_solution.items():
        print(f"  {key}: {value}")
    print()

# ACI history
if result.aci_history:
    print("ACI PROGRESSION:")
    print(f"  Initial: {result.aci_history[0]:.3f}")
    print(f"  Final: {result.aci_history[-1]:.3f}")
    reduction = (result.aci_history[0] - result.aci_history[-1]) / result.aci_history[0]
    print(f"  Reduction: {reduction * 100:.1f}%")
```

**Run it:**
```bash
python my_first_rese.py
```

---

### Example 2: Using Φ₁.₅ to Discover Assumptions

**Create file `discover_assumptions.py`:**

```python
#!/usr/bin/env python3
"""
Discover hidden assumptions using Φ₁.₅
"""

from rese.rese_pipeline import RESEPipeline, ProblemInput

# Create pipeline
pipeline = RESEPipeline()

# Define problem (intentionally vague)
problem = ProblemInput(
    id="vague_problem",
    description="Design a car",
    constraints=[
        {'id': 'c1', 'type': 'hard', 'description': 'must have wheels'}
    ],
    variables={'wheels': 'integer'}
)

# Run Phase I only (epistemic audit)
result = pipeline.run(problem, phases=['phase1'])

# Extract assumptions
assumptions = result.phase_results['phase1'].output['assumptions']

print("=" * 80)
print("Hidden Assumptions Discovered by Φ₁.₅")
print("=" * 80)
print(f"\nProblem: {problem.description}")
print(f"Explicit Constraints: {len(problem.constraints)}")
print(f"Hidden Assumptions Found: {len(assumptions)}\n")

for i, assumption in enumerate(assumptions, 1):
    print(f"{i}. {assumption['description']}")
    print(f"   Type: {assumption['type']}")
    print(f"   Confidence: {assumption['confidence']:.2f}")
    print(f"   Source: {assumption['source']}")
    print()
```

**Run it:**
```bash
python discover_assumptions.py
```

**Expected output:**
```
================================================================================
Hidden Assumptions Discovered by Φ₁.₅
================================================================================

Problem: Design a car
Explicit Constraints: 1
Hidden Assumptions Found: 7

1. Car must carry passengers
   Type: Functional
   Confidence: 0.94
   Source: failure_db

2. Car must meet safety regulations
   Type: Regulatory
   Confidence: 0.97
   Source: domain_kb

3. Car must be manufacturable at scale
   Type: Practical
   Confidence: 0.89
   Source: inference

4. Car must have propulsion system
   Type: Functional
   Confidence: 0.96
   Source: domain_kb

5. Car must be affordable to target market
   Type: Economic
   Confidence: 0.82
   Source: failure_db

6. Car must be reliable (low maintenance)
   Type: Quality
   Confidence: 0.91
   Source: domain_kb

7. Car must have reasonable fuel efficiency
   Type: Performance
   Confidence: 0.87
   Source: inference
```

---

### Example 3: Validating Isomorphisms with I_mech

**Create file `validate_isomorphism.py`:**

```python
#!/usr/bin/env python3
"""
Validate mechanistic isomorphisms with I_mech
"""

from rese.phase2.imech import IMechValidator, Domain

# Create validator
validator = IMechValidator()

# Define source domain (chemical reactor optimization)
source_domain = Domain(
    id='chemical_reactor',
    name='Chemical Reactor Optimization',
    variables={
        'temperature': 'continuous',
        'pressure': 'continuous',
        'catalyst_amount': 'continuous',
        'yield': 'objective'
    },
    constraints=[
        'yield = f(temperature, pressure, catalyst_amount)',
        'yield increases with temperature (up to limit)',
        'yield increases with pressure (up to limit)',
        'optimal catalyst_amount depends on temperature and pressure'
    ]
)

# Define target domain (neural network optimization)
target_domain = Domain(
    id='neural_network',
    name='Neural Network Architecture Optimization',
    variables={
        'learning_rate': 'continuous',
        'batch_size': 'continuous',
        'dropout_rate': 'continuous',
        'accuracy': 'objective'
    },
    constraints=[
        'accuracy = f(learning_rate, batch_size, dropout_rate)',
        'accuracy increases with learning_rate (up to limit)',
        'accuracy depends on batch_size (non-monotonic)',
        'optimal dropout_rate depends on learning rate and batch size'
    ]
)

# Compare domains
print("=" * 80)
print("Mechanistic Isomorphism Validation (I_mech)")
print("=" * 80)
print(f"\nSource: {source_domain.name}")
print(f"Target: {target_domain.name}\n")

similarity = validator.compare_domains(source_domain, target_domain)

print("Similarity Analysis:")
print(f"  Overall Score: {similarity.score:.3f}")
print(f"  Structural Similarity: {similarity.structural_similarity:.3f}")
print(f"  Causal Similarity: {similarity.causal_similarity:.3f}")
print(f"  Interventional Similarity: {similarity.interventional_similarity:.3f}")
print(f"  Is Isomorphic: {similarity.is_isomorphic}")
print(f"  Confidence: {similarity.confidence:.3f}")

if similarity.score > 0.8:
    print("\n✓ HIGH ISOMORPHISM: Solution transfer is recommended")

    # Show variable mapping
    print("\nVariable Mapping:")
    for source_var, target_var in similarity.mapping.items():
        print(f"  {source_var} → {target_var}")

else:
    print(f"\n✗ LOW ISOMORPHISM: Score {similarity.score:.3f} below threshold 0.8")
    print("  Solution transfer NOT recommended without manual validation")
```

**Run it:**
```bash
python validate_isomorphism.py
```

---

### Example 4: Using the REST API

**Start the API server:**
```bash
python -m rese.api
```

**Submit problem via curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Optimize neural network training",
    "constraints": [
      {
        "id": "c1",
        "type": "hard",
        "description": "accuracy >= 0.9"
      },
      {
        "id": "c2",
        "type": "soft",
        "description": "minimize training time"
      }
    ],
    "variables": {
      "learning_rate": "float",
      "batch_size": "int",
      "layers": "int"
    }
  }'
```

**Response:**
```json
{
  "pipeline_id": "rese_abc123",
  "problem_id": "problem_def456",
  "status": "completed",
  "final_solution": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "layers": 3
  },
  "aci_history": [0.85, 0.72, 0.55, 0.28, 0.15],
  "validation_score": 0.87,
  "confidence": 0.85
}
```

**Check status:**
```bash
curl "http://localhost:8000/api/v1/pipeline/rese_abc123/status"
```

**Get final result:**
```bash
curl "http://localhost:8000/api/v1/pipeline/rese_abc123/result"
```

---

### Example 5: Real-Time Updates with WebSocket

**Create file `websocket_client.py`:**

```python
#!/usr/bin/env python3
"""
WebSocket client for real-time RESE updates
"""

import asyncio
import websockets
import json

async def monitor_pipeline(pipeline_id):
    uri = f"ws://localhost:8000/ws/pipeline/{pipeline_id}"

    async with websockets.connect(uri) as websocket:
        # Subscribe to pipeline
        subscribe_msg = {
            "type": "subscribe",
            "pipeline_id": pipeline_id
        }
        await websocket.send(json.dumps(subscribe_msg))

        # Receive updates
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                if data['type'] == 'subscribed':
                    print(f"✓ Subscribed to pipeline {pipeline_id}")

                elif data['type'] == 'pipeline_update':
                    status = data['status']
                    progress = data['progress']

                    print(f"\nStatus: {status}")

                    if 'phase_results' in progress:
                        for phase_name, phase_info in progress['phase_results'].items():
                            print(f"  {phase_name}: {phase_info['status']}")

                    if 'aci_history' in progress and progress['aci_history']:
                        aci = progress['aci_history'][-1]
                        print(f"  ACI: {aci:.3f}")

                elif data['type'] == 'pong':
                    # Keep-alive
                    pass

            except websockets.exceptions.ConnectionClosed:
                print("\n✗ Connection closed")
                break

# Run monitor
if __name__ == "__main__":
    pipeline_id = "rese_abc123"  # Use your actual pipeline_id
    asyncio.run(monitor_pipeline(pipeline_id))
```

**Run it:**
```bash
python websocket_client.py
```

---

## Common Use Cases

### Use Case 1: Optimization Problems

```python
from rese.rese_pipeline import run_rese

# Define optimization problem
result = run_rese(
    problem_description="Minimize production cost while meeting quality standards",
    constraints=[
        {'id': 'c1', 'type': 'hard', 'formalization': 'quality >= 0.9'},
        {'id': 'c2', 'type': 'hard', 'formalization': 'cost < 10000'},
        {'id': 'c3', 'type': 'soft', 'formalization': 'minimize(cost)'}
    ],
    variables={'cost': 'float', 'quality': 'float'}
)

# Get optimal solution
print(f"Optimal Cost: {result.final_solution['cost']}")
print(f"Quality: {result.final_solution['quality']}")
print(f"Confidence: {result.confidence}")
```

### Use Case 2: Engineering Design

```python
from rese.rese_pipeline import RESEPipeline, ProblemInput

# Engineering design problem
problem = ProblemInput(
    id="bridge_design",
    description="Design a bridge spanning 100m",
    constraints=[
        {'id': 'c1', 'type': 'hard', 'formalization': 'span = 100'},
        {'id': 'c2', 'type': 'hard', 'formalization': 'max_load >= 50'},
        {'id': 'c3', 'type': 'soft', 'formalization': 'minimize_cost'}
    ],
    variables={
        'span': 'float',
        'max_load': 'float',
        'cost': 'float',
        'material': 'categorical'
    },
    domain="civil_engineering"
)

pipeline = RESEPipeline()
result = pipeline.run(problem)
```

### Use Case 3: Knowledge Transfer

```python
from rese.phase2.imech import IMechValidator, Domain

# Transfer solution from known domain to new domain
validator = IMechValidator()

# Known problem (solved)
known_domain = Domain(
    id='known_problem',
    name='Known Optimization Problem',
    variables={...},
    constraints=[...]
)

# New problem (unsolved)
new_domain = Domain(
    id='new_problem',
    name='New Problem Domain',
    variables={...},
    constraints=[...]
)

# Validate isomorphism
similarity = validator.compare_domains(known_domain, new_domain)

if similarity.is_isomorphic:
    print("✓ Domains are isomorphic - solution transfer recommended")
    print(f"Confidence: {similarity.confidence}")
else:
    print("✗ Domains are not isomorphic - solve from scratch")
```

---

## Next Steps

### 1. Learn More

- **User Guide**: `RESE_USER_GUIDE.md` - Comprehensive system overview
- **API Reference**: `RESE_API_REFERENCE.md` - Complete API documentation
- **Integration Guide**: `RESE_INTEGRATION_GUIDE.md` - E2E integration
- **Developer Guide**: `RESE_DEVELOPER_GUIDE.md` - Architecture and development

### 2. Explore Examples

Check the `rese/examples/` directory for more examples:
- Constraint satisfaction
- Isomorphism validation
- MCTS optimization
- API usage
- WebSocket monitoring

### 3. Configure for Your Use Case

Edit `config.json` to optimize RESE for your specific needs:
- Adjust phase thresholds
- Tune MCTS parameters
- Enable/disable features
- Set performance limits

### 4. Integrate with Your System

See `RESE_INTEGRATION_GUIDE.md` for:
- Stage-by-stage integration
- Data flow diagrams
- Configuration options
- Best practices

---

## FAQ

### Q: How long does a typical RESE analysis take?

**A:** Depends on problem complexity and phases:

- **Phase I only**: 5-30 seconds
- **All phases (simple)**: 1-5 minutes
- **All phases (complex)**: 10-30 minutes

Use caching to speed up repeated analyses.

### Q: Can I run RESE without all phases?

**A:** Yes! Specify which phases to run:

```python
# Run only Phase I and III
result = pipeline.run(problem, phases=['phase1', 'phase3'])
```

### Q: What does ACI actually measure?

**A:** ACI (Algorithmic Complexity Index) measures uncertainty/complexity:
- **ACI = 1.0**: Maximum uncertainty (random)
- **ACI = 0.5**: Partial information
- **ACI < 0.2**: Good, validated solution

RESE targets ACI < 0.2 before execution.

### Q: How accurate is I_mech isomorphism detection?

**A:** Benchmarked at >80% transfer success correlation.

Key points:
- Validated on 50+ domain pairs
- Requires causal structure similarity (not just superficial)
- Generates Lean 4 proofs for high-confidence matches
- Always validate critical transfers manually

### Q: Can I use RESE for my specific domain?

**A:** RESE is domain-agnostic but works best when:
- Problem can be formalized as constraints
- Similar problems exist in knowledge base
- Quantifiable metrics are available

For domain-specific optimization:
1. Add domain knowledge to failure database (for Φ₁.₅)
2. Add known solutions to knowledge base (for I_mech)
3. Tune ACI calculation for domain (custom calculator)

### Q: How do I interpret confidence scores?

**A:**

- **0.9-1.0**: Very high confidence - proceed with execution
- **0.7-0.9**: High confidence - good for most cases
- **0.5-0.7**: Moderate confidence - review carefully
- **<0.5**: Low confidence - not recommended for execution

### Q: What if RESE doesn't converge?

**A:**

1. Check problem formulation (are constraints consistent?)
2. Increase MCTS iterations
3. Relax convergence criteria
4. Review intermediate phase outputs

```python
# Debug non-convergence
config.phase3.gamma2_iterations = 10000  # Increase
config.phase3.convergence_patience = 100  # More patient
```

### Q: Can I use RESE in production?

**A:** Yes, with proper setup:

1. **Use production config:**
   ```python
   config = config.for_environment(Environment.PRODUCTION)
   ```

2. **Enable monitoring:**
   ```python
   config.monitoring.enable_metrics = True
   config.monitoring.enable_alerts = True
   ```

3. **Set up API server properly:**
   ```bash
   uvicorn rese.api:app --workers 4 --host 0.0.0.0 --port 8000
   ```

4. **Implement proper error handling and logging**

### Q: How much does RESE cost to run?

**A:** RESE is open-source and free to use. Costs:
- **Development**: $0 (local machine)
- **Production**: Cloud compute costs only
  - Small instance: ~$50/month
  - Large instance: ~$500/month

No licensing fees, no per-call charges.

---

**Quick Start Guide Version:** 1.0.0
**Last Updated:** 2025-12-31
**Authors:** RESE Development Team

**Need Help?**
- GitHub Issues: https://github.com/your-org/rese/issues
- Documentation: See other RESE_*.md files
- Examples: `rese/examples/`
=======
# RESE Quick Start Guide

## Table of Contents

1. [Installation](#installation)
2. [Setup and Configuration](#setup-and-configuration)
3. [Your First Example](#your-first-example)
4. [Common Use Cases](#common-use-cases)
5. [Next Steps](#next-steps)
6. [FAQ](#faq)

---

## Installation

### Prerequisites

**Required:**
- Python 3.9 or higher
- pip (Python package manager)

**Check your Python version:**
```bash
python --version
# Should show Python 3.9.x or higher
```

### Install RESE

**Option 1: Install from source (recommended for development)**
```bash
cd /path/to/OpenEvolve/Frontend
pip install -e .
```

**Option 2: Install dependencies only**
```bash
pip install numpy fastapi uvicorn pydantic networkx scipy
```

### Verify Installation

```bash
# Run quick start script
python rese/quickstart.py
```

**Expected output:**
```
================================================================================
 RESE Quick Start
================================================================================

Recursive Epistemic Solvability Engine - Quick Start Guide
This script will verify your installation and run a quick demo.

================================================================================
 Checking Dependencies
================================================================================
✓ numpy (required)
✓ fastapi (required)
✓ pydantic (required)
⚠ psutil (optional) - not installed
⚠ networkx (optional) - not installed
⚠ scipy (optional) - not installed

...

================================================================================
 Summary
================================================================================

Test Results:
--------------------------------------------------------------------------------
  Dependencies............................................ ✓ PASS
  Configuration........................................... ✓ PASS
  Pipeline................................................. ✓ PASS
  Monitoring............................................... ✓ PASS
  Demo.................................................... ✓ PASS

--------------------------------------------------------------------------------

🎉 All tests passed! RESE is ready to use.

Next Steps:
  1. Review configuration in config.json
  2. Start the API server: python -m rese.api
  3. View API docs: http://localhost:8000/docs
  4. Read the documentation: rese/docs/
```

---

## Setup and Configuration

### 1. Create Configuration File

**Generate default configuration:**
```python
from rese.config import create_default_config

config = create_default_config('config.json')
print("Configuration created at config.json")
```

**Or create manually (`config.json`):**
```json
{
  "environment": "development",
  "project_name": "rese",
  "version": "1.0.0",
  "phase1": {
    "sce_max_constraints": 10000,
    "phi15_enabled": true,
    "phi15_assumption_threshold": 0.6,
    "phi2_enabled": true,
    "phi2_bias_threshold": 0.5
  },
  "phase2": {
    "psi2_similarity_threshold": 0.7,
    "psi3_target_accuracy": 0.8,
    "imech_algorithm": "weisfeiler_lehman"
  },
  "phase3": {
    "gamma2_iterations": 1000,
    "gamma2_aci_guided": true,
    "convergence_enabled": true
  },
  "phase4": {
    "delta3_min_aci_reduction": 0.2,
    "delta3_validation_threshold": 0.7
  },
  "pipeline": {
    "enable_caching": true,
    "cache_ttl_seconds": 3600
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "enable_auth": false
  },
  "monitoring": {
    "log_level": "INFO"
  }
}
```

### 2. Load Configuration

```python
from rese.config import load_config

# Load from default location
config = load_config()

# Or from specific file
config = load_config(Path('my_config.json'))
```

### 3. Start API Server (Optional)

**Development mode:**
```bash
python -m rese.api
```

**Production mode:**
```bash
uvicorn rese.api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Verify API is running:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-12-31T12:00:00Z",
  "uptime_seconds": 1.5
}
```

---

## Your First Example

### Example 1: Simple Constraint Satisfaction

**Create file `my_first_rese.py`:**

```python
#!/usr/bin/env python3
"""
My First RESE Analysis
"""

from rese.rese_pipeline import run_rese

# Define problem
result = run_rese(
    problem_description="Find optimal allocation of resources",
    constraints=[
        {
            'id': 'c1',
            'type': 'hard',
            'description': 'Total cost must be less than $1000',
            'formalization': 'sum(cost_i) < 1000'
        },
        {
            'id': 'c2',
            'type': 'hard',
            'description': 'All resources must be positive',
            'formalization': 'all(x_i > 0)'
        },
        {
            'id': 'c3',
            'type': 'soft',
            'description': 'Maximize total utility',
            'formalization': 'maximize(sum(utility_i))'
        }
    ],
    variables={
        'cost': 'float',
        'utility': 'float',
        'quantity': 'integer'
    }
)

# Print results
print("=" * 80)
print("RESE Analysis Results")
print("=" * 80)
print(f"Status: {result.status.value}")
print(f"Elapsed: {result.elapsed_seconds:.2f} seconds")
print()

# Phase results
for phase_name, phase_result in result.phase_results.items():
    print(f"{phase_name.upper()}:")
    print(f"  Status: {phase_result.status.value}")
    print(f"  Elapsed: {phase_result.elapsed_seconds:.2f}s")

    if phase_result.metrics:
        print(f"  Metrics:")
        for key, value in phase_result.metrics.items():
            print(f"    {key}: {value}")
    print()

# Final solution
if result.final_solution:
    print("FINAL SOLUTION:")
    for key, value in result.final_solution.items():
        print(f"  {key}: {value}")
    print()

# ACI history
if result.aci_history:
    print("ACI PROGRESSION:")
    print(f"  Initial: {result.aci_history[0]:.3f}")
    print(f"  Final: {result.aci_history[-1]:.3f}")
    reduction = (result.aci_history[0] - result.aci_history[-1]) / result.aci_history[0]
    print(f"  Reduction: {reduction * 100:.1f}%")
```

**Run it:**
```bash
python my_first_rese.py
```

---

### Example 2: Using Φ₁.₅ to Discover Assumptions

**Create file `discover_assumptions.py`:**

```python
#!/usr/bin/env python3
"""
Discover hidden assumptions using Φ₁.₅
"""

from rese.rese_pipeline import RESEPipeline, ProblemInput

# Create pipeline
pipeline = RESEPipeline()

# Define problem (intentionally vague)
problem = ProblemInput(
    id="vague_problem",
    description="Design a car",
    constraints=[
        {'id': 'c1', 'type': 'hard', 'description': 'must have wheels'}
    ],
    variables={'wheels': 'integer'}
)

# Run Phase I only (epistemic audit)
result = pipeline.run(problem, phases=['phase1'])

# Extract assumptions
assumptions = result.phase_results['phase1'].output['assumptions']

print("=" * 80)
print("Hidden Assumptions Discovered by Φ₁.₅")
print("=" * 80)
print(f"\nProblem: {problem.description}")
print(f"Explicit Constraints: {len(problem.constraints)}")
print(f"Hidden Assumptions Found: {len(assumptions)}\n")

for i, assumption in enumerate(assumptions, 1):
    print(f"{i}. {assumption['description']}")
    print(f"   Type: {assumption['type']}")
    print(f"   Confidence: {assumption['confidence']:.2f}")
    print(f"   Source: {assumption['source']}")
    print()
```

**Run it:**
```bash
python discover_assumptions.py
```

**Expected output:**
```
================================================================================
Hidden Assumptions Discovered by Φ₁.₅
================================================================================

Problem: Design a car
Explicit Constraints: 1
Hidden Assumptions Found: 7

1. Car must carry passengers
   Type: Functional
   Confidence: 0.94
   Source: failure_db

2. Car must meet safety regulations
   Type: Regulatory
   Confidence: 0.97
   Source: domain_kb

3. Car must be manufacturable at scale
   Type: Practical
   Confidence: 0.89
   Source: inference

4. Car must have propulsion system
   Type: Functional
   Confidence: 0.96
   Source: domain_kb

5. Car must be affordable to target market
   Type: Economic
   Confidence: 0.82
   Source: failure_db

6. Car must be reliable (low maintenance)
   Type: Quality
   Confidence: 0.91
   Source: domain_kb

7. Car must have reasonable fuel efficiency
   Type: Performance
   Confidence: 0.87
   Source: inference
```

---

### Example 3: Validating Isomorphisms with I_mech

**Create file `validate_isomorphism.py`:**

```python
#!/usr/bin/env python3
"""
Validate mechanistic isomorphisms with I_mech
"""

from rese.phase2.imech import IMechValidator, Domain

# Create validator
validator = IMechValidator()

# Define source domain (chemical reactor optimization)
source_domain = Domain(
    id='chemical_reactor',
    name='Chemical Reactor Optimization',
    variables={
        'temperature': 'continuous',
        'pressure': 'continuous',
        'catalyst_amount': 'continuous',
        'yield': 'objective'
    },
    constraints=[
        'yield = f(temperature, pressure, catalyst_amount)',
        'yield increases with temperature (up to limit)',
        'yield increases with pressure (up to limit)',
        'optimal catalyst_amount depends on temperature and pressure'
    ]
)

# Define target domain (neural network optimization)
target_domain = Domain(
    id='neural_network',
    name='Neural Network Architecture Optimization',
    variables={
        'learning_rate': 'continuous',
        'batch_size': 'continuous',
        'dropout_rate': 'continuous',
        'accuracy': 'objective'
    },
    constraints=[
        'accuracy = f(learning_rate, batch_size, dropout_rate)',
        'accuracy increases with learning_rate (up to limit)',
        'accuracy depends on batch_size (non-monotonic)',
        'optimal dropout_rate depends on learning rate and batch size'
    ]
)

# Compare domains
print("=" * 80)
print("Mechanistic Isomorphism Validation (I_mech)")
print("=" * 80)
print(f"\nSource: {source_domain.name}")
print(f"Target: {target_domain.name}\n")

similarity = validator.compare_domains(source_domain, target_domain)

print("Similarity Analysis:")
print(f"  Overall Score: {similarity.score:.3f}")
print(f"  Structural Similarity: {similarity.structural_similarity:.3f}")
print(f"  Causal Similarity: {similarity.causal_similarity:.3f}")
print(f"  Interventional Similarity: {similarity.interventional_similarity:.3f}")
print(f"  Is Isomorphic: {similarity.is_isomorphic}")
print(f"  Confidence: {similarity.confidence:.3f}")

if similarity.score > 0.8:
    print("\n✓ HIGH ISOMORPHISM: Solution transfer is recommended")

    # Show variable mapping
    print("\nVariable Mapping:")
    for source_var, target_var in similarity.mapping.items():
        print(f"  {source_var} → {target_var}")

else:
    print(f"\n✗ LOW ISOMORPHISM: Score {similarity.score:.3f} below threshold 0.8")
    print("  Solution transfer NOT recommended without manual validation")
```

**Run it:**
```bash
python validate_isomorphism.py
```

---

### Example 4: Using the REST API

**Start the API server:**
```bash
python -m rese.api
```

**Submit problem via curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Optimize neural network training",
    "constraints": [
      {
        "id": "c1",
        "type": "hard",
        "description": "accuracy >= 0.9"
      },
      {
        "id": "c2",
        "type": "soft",
        "description": "minimize training time"
      }
    ],
    "variables": {
      "learning_rate": "float",
      "batch_size": "int",
      "layers": "int"
    }
  }'
```

**Response:**
```json
{
  "pipeline_id": "rese_abc123",
  "problem_id": "problem_def456",
  "status": "completed",
  "final_solution": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "layers": 3
  },
  "aci_history": [0.85, 0.72, 0.55, 0.28, 0.15],
  "validation_score": 0.87,
  "confidence": 0.85
}
```

**Check status:**
```bash
curl "http://localhost:8000/api/v1/pipeline/rese_abc123/status"
```

**Get final result:**
```bash
curl "http://localhost:8000/api/v1/pipeline/rese_abc123/result"
```

---

### Example 5: Real-Time Updates with WebSocket

**Create file `websocket_client.py`:**

```python
#!/usr/bin/env python3
"""
WebSocket client for real-time RESE updates
"""

import asyncio
import websockets
import json

async def monitor_pipeline(pipeline_id):
    uri = f"ws://localhost:8000/ws/pipeline/{pipeline_id}"

    async with websockets.connect(uri) as websocket:
        # Subscribe to pipeline
        subscribe_msg = {
            "type": "subscribe",
            "pipeline_id": pipeline_id
        }
        await websocket.send(json.dumps(subscribe_msg))

        # Receive updates
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                if data['type'] == 'subscribed':
                    print(f"✓ Subscribed to pipeline {pipeline_id}")

                elif data['type'] == 'pipeline_update':
                    status = data['status']
                    progress = data['progress']

                    print(f"\nStatus: {status}")

                    if 'phase_results' in progress:
                        for phase_name, phase_info in progress['phase_results'].items():
                            print(f"  {phase_name}: {phase_info['status']}")

                    if 'aci_history' in progress and progress['aci_history']:
                        aci = progress['aci_history'][-1]
                        print(f"  ACI: {aci:.3f}")

                elif data['type'] == 'pong':
                    # Keep-alive
                    pass

            except websockets.exceptions.ConnectionClosed:
                print("\n✗ Connection closed")
                break

# Run monitor
if __name__ == "__main__":
    pipeline_id = "rese_abc123"  # Use your actual pipeline_id
    asyncio.run(monitor_pipeline(pipeline_id))
```

**Run it:**
```bash
python websocket_client.py
```

---

## Common Use Cases

### Use Case 1: Optimization Problems

```python
from rese.rese_pipeline import run_rese

# Define optimization problem
result = run_rese(
    problem_description="Minimize production cost while meeting quality standards",
    constraints=[
        {'id': 'c1', 'type': 'hard', 'formalization': 'quality >= 0.9'},
        {'id': 'c2', 'type': 'hard', 'formalization': 'cost < 10000'},
        {'id': 'c3', 'type': 'soft', 'formalization': 'minimize(cost)'}
    ],
    variables={'cost': 'float', 'quality': 'float'}
)

# Get optimal solution
print(f"Optimal Cost: {result.final_solution['cost']}")
print(f"Quality: {result.final_solution['quality']}")
print(f"Confidence: {result.confidence}")
```

### Use Case 2: Engineering Design

```python
from rese.rese_pipeline import RESEPipeline, ProblemInput

# Engineering design problem
problem = ProblemInput(
    id="bridge_design",
    description="Design a bridge spanning 100m",
    constraints=[
        {'id': 'c1', 'type': 'hard', 'formalization': 'span = 100'},
        {'id': 'c2', 'type': 'hard', 'formalization': 'max_load >= 50'},
        {'id': 'c3', 'type': 'soft', 'formalization': 'minimize_cost'}
    ],
    variables={
        'span': 'float',
        'max_load': 'float',
        'cost': 'float',
        'material': 'categorical'
    },
    domain="civil_engineering"
)

pipeline = RESEPipeline()
result = pipeline.run(problem)
```

### Use Case 3: Knowledge Transfer

```python
from rese.phase2.imech import IMechValidator, Domain

# Transfer solution from known domain to new domain
validator = IMechValidator()

# Known problem (solved)
known_domain = Domain(
    id='known_problem',
    name='Known Optimization Problem',
    variables={...},
    constraints=[...]
)

# New problem (unsolved)
new_domain = Domain(
    id='new_problem',
    name='New Problem Domain',
    variables={...},
    constraints=[...]
)

# Validate isomorphism
similarity = validator.compare_domains(known_domain, new_domain)

if similarity.is_isomorphic:
    print("✓ Domains are isomorphic - solution transfer recommended")
    print(f"Confidence: {similarity.confidence}")
else:
    print("✗ Domains are not isomorphic - solve from scratch")
```

---

## Next Steps

### 1. Learn More

- **User Guide**: `RESE_USER_GUIDE.md` - Comprehensive system overview
- **API Reference**: `RESE_API_REFERENCE.md` - Complete API documentation
- **Integration Guide**: `RESE_INTEGRATION_GUIDE.md` - E2E integration
- **Developer Guide**: `RESE_DEVELOPER_GUIDE.md` - Architecture and development

### 2. Explore Examples

Check the `rese/examples/` directory for more examples:
- Constraint satisfaction
- Isomorphism validation
- MCTS optimization
- API usage
- WebSocket monitoring

### 3. Configure for Your Use Case

Edit `config.json` to optimize RESE for your specific needs:
- Adjust phase thresholds
- Tune MCTS parameters
- Enable/disable features
- Set performance limits

### 4. Integrate with Your System

See `RESE_INTEGRATION_GUIDE.md` for:
- Stage-by-stage integration
- Data flow diagrams
- Configuration options
- Best practices

---

## FAQ

### Q: How long does a typical RESE analysis take?

**A:** Depends on problem complexity and phases:

- **Phase I only**: 5-30 seconds
- **All phases (simple)**: 1-5 minutes
- **All phases (complex)**: 10-30 minutes

Use caching to speed up repeated analyses.

### Q: Can I run RESE without all phases?

**A:** Yes! Specify which phases to run:

```python
# Run only Phase I and III
result = pipeline.run(problem, phases=['phase1', 'phase3'])
```

### Q: What does ACI actually measure?

**A:** ACI (Algorithmic Complexity Index) measures uncertainty/complexity:
- **ACI = 1.0**: Maximum uncertainty (random)
- **ACI = 0.5**: Partial information
- **ACI < 0.2**: Good, validated solution

RESE targets ACI < 0.2 before execution.

### Q: How accurate is I_mech isomorphism detection?

**A:** Benchmarked at >80% transfer success correlation.

Key points:
- Validated on 50+ domain pairs
- Requires causal structure similarity (not just superficial)
- Generates Lean 4 proofs for high-confidence matches
- Always validate critical transfers manually

### Q: Can I use RESE for my specific domain?

**A:** RESE is domain-agnostic but works best when:
- Problem can be formalized as constraints
- Similar problems exist in knowledge base
- Quantifiable metrics are available

For domain-specific optimization:
1. Add domain knowledge to failure database (for Φ₁.₅)
2. Add known solutions to knowledge base (for I_mech)
3. Tune ACI calculation for domain (custom calculator)

### Q: How do I interpret confidence scores?

**A:**

- **0.9-1.0**: Very high confidence - proceed with execution
- **0.7-0.9**: High confidence - good for most cases
- **0.5-0.7**: Moderate confidence - review carefully
- **<0.5**: Low confidence - not recommended for execution

### Q: What if RESE doesn't converge?

**A:**

1. Check problem formulation (are constraints consistent?)
2. Increase MCTS iterations
3. Relax convergence criteria
4. Review intermediate phase outputs

```python
# Debug non-convergence
config.phase3.gamma2_iterations = 10000  # Increase
config.phase3.convergence_patience = 100  # More patient
```

### Q: Can I use RESE in production?

**A:** Yes, with proper setup:

1. **Use production config:**
   ```python
   config = config.for_environment(Environment.PRODUCTION)
   ```

2. **Enable monitoring:**
   ```python
   config.monitoring.enable_metrics = True
   config.monitoring.enable_alerts = True
   ```

3. **Set up API server properly:**
   ```bash
   uvicorn rese.api:app --workers 4 --host 0.0.0.0 --port 8000
   ```

4. **Implement proper error handling and logging**

### Q: How much does RESE cost to run?

**A:** RESE is open-source and free to use. Costs:
- **Development**: $0 (local machine)
- **Production**: Cloud compute costs only
  - Small instance: ~$50/month
  - Large instance: ~$500/month

No licensing fees, no per-call charges.

---

**Quick Start Guide Version:** 1.0.0
**Last Updated:** 2025-12-31
**Authors:** RESE Development Team

**Need Help?**
- GitHub Issues: https://github.com/your-org/rese/issues
- Documentation: See other RESE_*.md files
- Examples: `rese/examples/`
>>>>>>> 1cb9c5e35 (update)
