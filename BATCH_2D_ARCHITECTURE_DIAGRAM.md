# OpenEvolve Frontend Architecture - Current State

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CONSUMER LAYER                                 │
│  (Application code that uses evolution/adversarial functionality)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────┐ ┌─────────────┐ ┌──────────────────┐
        │   Pattern 1:  │ │  Pattern 2: │ │   Pattern 3:     │
        │ Integration   │ │   Client    │ │ Dependency       │
        │   Layer       │ │   Layer     │ │ Injection        │
        └───────────────┘ └─────────────┘ └──────────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          ADAPTER LAYER                                  │
│  (evolution_adapter.py, adversarial_adapter.py)                       │
│  - Provides clean API surface                                         │
│  - Handles configuration                                              │
│  - Manages error handling                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────┐ ┌─────────────┐ ┌──────────────────┐
        │   OpenEvolve  │ │  Adversarial│ │   Generic        │
        │   Integration │ │   Unified   │ │   MAKER         │
        │               │ │   Framework │ │                  │
        └───────────────┘ └─────────────┘ └──────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CORE ENGINE LAYER                               │
│  (evolution.py, adversarial.py, mdap_engine.py, etc.)                  │
│  - Actual implementation logic                                         │
│  - NOT directly imported by consumer code                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Consumer Code Patterns

### Pattern 1: Integration Layer Usage ✅ RECOMMENDED

**File:** `adversarial_testing.py`

```python
from openevolve_integration import run_unified_evolution

def run_comprehensive_adversarial_testing(...):
    # Phases 1-2: Red/Blue team analysis
    red_team_results = run_red_team_analysis(...)
    blue_team_results = run_blue_team_resolution(...)

    # Phase 3: Evolution via integration layer
    evolution_results = run_unified_evolution(
        content=improved_content,
        evolution_mode="adversarial",
        model_configs=[...],
        max_iterations=max_iterations,
        ...
    )
```

**Advantages:**
- ✅ Simple, high-level API
- ✅ Unified entry point for all modes
- ✅ Automatic configuration management
- ✅ Built-in error handling

**Use Cases:**
- General content evolution
- Adversarial testing
- Multi-modal optimization
- Standard application code

---

### Pattern 2: Client Layer Usage ✅ FOR COMPLEX WORKFLOWS

**Files:** `problem_analyzer.py`, `decomposition_engine.py`

```python
from openevolve_client import OpenEvolveClient

class ProblemAnalyzer:
    def __init__(self, openevolve_client=None):
        self.openevolve_client = openevolve_client or OpenEvolveClient()

    def analyze_problem(self, problem_text: str):
        # Use client for LLM-based analysis
        domain_context = self.openevolve_client.analyze(
            content=problem_text,
            task="domain_classification"
        )
        ...
```

**Advantages:**
- ✅ Stateful operations
- ✅ Complex workflows
- ✅ Fine-grained control
- ✅ Connection pooling
- ✅ Advanced configuration

**Use Cases:**
- Complex multi-step workflows
- Stateful operations
- Batch processing
- Advanced configuration needs

---

### Pattern 3: Dependency Injection ✅ FOR FRAME CODE

**File:** `multi_round_testing.py`

```python
def run_multi_round_evolution(
    content: str,
    evolution_function: Callable,  # <-- Injected!
    rounds: int = 5,
    ...
) -> MultiRoundResult:
    """
    Run multi-round evolution testing.

    Args:
        evolution_function: ANY evolution function that accepts (content, **params)
        ...
    """
    tester = MultiRoundTester()
    test_function = create_evolution_test_function(evolution_function)
    return tester.run_multi_round_test(content, test_function, ...)
```

**Advantages:**
- ✅ Provider-agnostic
- ✅ Highly testable
- ✅ Maximum flexibility
- ✅ No coupling to specific implementations

**Use Cases:**
- Framework code
- Testing infrastructure
- Generic utilities
- Provider-agnostic systems

---

## Provider Layer Architecture

### Adapters (Internal/Advanced Use)

```python
# evolution_adapter.py
from evolution_adapter import create_evolution_adapter

adapter = create_evolution_adapter(
    max_iterations=50,
    temperature=0.7,
    population_size=1000
)

result = adapter.run_evolution(content)
final_content = result.final_content
```

**When to Use:**
- Fine-grained control over evolution parameters
- Custom evaluation metrics
- Advanced features not exposed in integration layer
- Building new integrations

---

### Core Engines (DO NOT IMPORT DIRECTLY)

```
┌─────────────────────────────────────────────┐
│          CORE ENGINE LAYER                  │
│  (Protected by adapter pattern)             │
├─────────────────────────────────────────────┤
│  evolution.py                               │
│  - run_evolution_loop()                     │
│  - EvolutionConfiguration                   │
│  - EvolutionResult                          │
│                                             │
│  adversarial.py                             │
│  - run_comprehensive_adversarial_testing()  │
│  - RedTeamEngine                            │
│  - BlueTeamEngine                           │
│                                             │
│  mdap_engine.py                             │
│  - MDAPEngine                               │
│  - MDAPConfiguration                        │
│                                             │
│  maker_engine.py                            │
│  - MAKEREngine                              │
│  - MAKERConfiguration                       │
└─────────────────────────────────────────────┘
```

**Rule:** ❌ Consumer code should NOT import these directly
**Exception:** ✅ Adapters and integration layer can import these

---

## File Classification

### ✅ Consumer Files (Use Integration/Client)

1. **adversarial_testing.py**
   - Uses: `openevolve_integration.run_unified_evolution`
   - Pattern: Integration Layer
   - Status: ✅ Correct

2. **problem_analyzer.py**
   - Uses: `OpenEvolveClient`
   - Pattern: Client Layer
   - Status: ✅ Correct

3. **decomposition_engine.py**
   - Uses: `OpenEvolveClient`
   - Pattern: Client Layer
   - Status: ✅ Correct

4. **multi_round_testing.py**
   - Uses: Dependency injection
   - Pattern: Provider-agnostic
   - Status: ✅ Correct

### ✅ Provider Files (Implement Functionality)

1. **evolution.py**
   - Role: Core evolution engine
   - Status: Core implementation

2. **adversarial.py**
   - Role: Core adversarial testing
   - Status: Core implementation

3. **adversarial_unified.py**
   - Role: Unified adversarial framework (NEW)
   - Status: Modern implementation

4. **evolution_adapter.py**
   - Role: Adapter for evolution engine
   - Status: Adapter layer

5. **adversarial_adapter.py**
   - Role: Adapter for adversarial engine
   - Status: Adapter layer

6. **openevolve_integration.py**
   - Role: Unified integration layer
   - Status: Integration layer

7. **openevolve_client.py**
   - Role: High-level client API
   - Status: Client layer

### ✅ Workflow Files (Different Domain)

1. **openevolve_workflow_manager_integrated.py**
   - Role: Sovereign decomposition workflow
   - Uses: `workflow_engine` (not evolution/adversarial)
   - Status: ✅ Correct (different domain)

2. **end_to_end_invention_planner.py**
   - Role: Invention planning system
   - Uses: SOP, LeanAide, decomposition
   - Status: ✅ Correct (different domain)

---

## Data Flow Examples

### Example 1: Simple Evolution (Integration Layer)

```
User Request
    │
    ▼
adversarial_testing.py::run_comprehensive_adversarial_testing()
    │
    ├── Phase 1: Red Team Analysis (direct LLM calls)
    │
    ├── Phase 2: Blue Team Resolution (direct LLM calls)
    │
    └── Phase 3: Evolution
            │
            ▼
openevolve_integration.py::run_unified_evolution()
    │
    ├── Create configuration
    │
    ├── evolution_adapter.py::create_evolution_adapter()
    │       │
    │       └── evolution.py::run_evolution_loop()
    │
    └── Return results
```

### Example 2: Problem Analysis (Client Layer)

```
User Request
    │
    ▼
problem_analyzer.py::ProblemAnalyzer
    │
    ▼
OpenEvolveClient::analyze()
    │
    ├── Manage connection
    │
    ├── Call appropriate engine
    │
    └── Return analysis
```

### Example 3: Multi-Round Testing (Dependency Injection)

```
User Request
    │
    ▼
multi_round_testing.py::run_multi_round_evolution()
    │
    ├── Accept ANY evolution_function
    │
    ├── Wrap it with test harness
    │
    └── Run multiple rounds
```

---

## Migration Guidelines

### For NEW Code

**Question:** Which pattern should I use?

**Decision Tree:**

```
┌─────────────────────────────────────┐
│  Do you need evolution/adversarial? │
└─────────────────────────────────────┘
              │ NO
              ├──────────► Use workflow_engine or other modules
              │
              │ YES
              ▼
┌─────────────────────────────────────┐
│  Is this a framework/utility code? │
└─────────────────────────────────────┘
              │ YES
              ├──────────► Use dependency injection
              │
              │ NO
              ▼
┌─────────────────────────────────────┐
│  Do you need complex workflows?     │
└─────────────────────────────────────┘
              │ YES
              ├──────────► Use OpenEvolveClient
              │
              │ NO
              ▼
┌─────────────────────────────────────┐
│  Use openevolve_integration        │
│  run_unified_evolution()            │
└─────────────────────────────────────┘
```

### Examples

**Standard Application Code:**
```python
from openevolve_integration import run_unified_evolution

result = run_unified_evolution(
    content=my_content,
    evolution_mode="evolution",
    model_configs=[{"name": "gpt-4", "weight": 1.0}],
    max_iterations=50
)
```

**Complex Workflow:**
```python
from openevolve_client import OpenEvolveClient

client = OpenEvolveClient()
result1 = client.evolve(content1, ...)
result2 = client.evolve(content2, ...)
result3 = client.combine_results(result1, result2)
```

**Framework Code:**
```python
def my_framework_function(optimization_func: Callable):
    # Works with ANY optimization function
    return optimization_func(data, param=value)
```

---

## Testing Strategy

### Unit Testing

```python
# Test consumer code with mock dependencies
def test_adversarial_testing():
    mock_integration = Mock(return_value={"success": True, "best_code": "..."})
    with patch('openevolve_integration.run_unified_evolution', mock_integration):
        result = run_comprehensive_adversarial_testing(...)
        assert result["success"]
```

### Integration Testing

```python
# Test integration layer with real adapters
def test_integration_layer():
    result = run_unified_evolution(
        content="test content",
        evolution_mode="evolution",
        max_iterations=5  # Small number for testing
    )
    assert result["success"]
```

### Adapter Testing

```python
# Test adapter layer directly
def test_evolution_adapter():
    adapter = create_evolution_adapter(max_iterations=5)
    result = adapter.run_evolution("test content")
    assert result.final_content != "test content"
```

---

## Best Practices Summary

### ✅ DO

1. **Use integration layer** for standard application code
2. **Use client layer** for complex workflows
3. **Use dependency injection** for framework/utility code
4. **Import from adapters** only when building new integrations
5. **Follow the decision tree** when choosing patterns

### ❌ DON'T

1. **Direct imports** from evolution.py in consumer code
2. **Direct imports** from adversarial.py in consumer code
3. **Bypass the adapter layer** without good reason
4. **Mix patterns** unnecessarily in the same module
5. **Create tight coupling** to specific implementations

---

## Conclusion

The OpenEvolve Frontend architecture is **HEALTHY** and **WELL-DESIGNED**:

- ✅ Clear separation of concerns
- ✅ Proper layering (Consumer → Adapter → Core)
- ✅ Multiple integration patterns for different use cases
- ✅ No architectural violations found
- ✅ All consumer code follows best practices

**No migration work needed at this time.**
