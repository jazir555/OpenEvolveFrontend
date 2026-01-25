# Tutorial 2: Using Decomposition Strategies

**Level:** Intermediate
**Time:** 45 minutes
**Prerequisites:** Tutorial 1 (Getting Started)

---

## Learning Objectives

After this tutorial, you will be able to:
- Understand the three decomposition strategies
- Choose the right strategy for your problem
- Customize strategy parameters
- Combine multiple strategies
- Handle complex dependencies

---

## Overview of Strategies

The decomposition engine offers three built-in strategies:

```
┌─────────────────────────────────────────────────────────┐
│                   Strategy Selection                    │
├──────────────────┬──────────────────┬──────────────────┤
│   Semantic       │   Hierarchical   │   Flow-Based     │
│                  │                  │                  │
│ Best for:        │ Best for:        │ Best for:        │
│ • Conceptual     │ • Systems        │ • Processes      │
│ • Research       │ • Architecture   │ • Pipelines      │
│ • Design         │ • Organizations  │ • Workflows      │
│                  │                  │                  │
│ Pros:            │ Pros:            │ Pros:            │
│ • Intelligent    │ • Structured     │ • Natural        │
│ • Context-aware  │ • Clear layers   │ • Sequential     │
│ • Flexible       │ • Scalable       │ • Parallelizable │
│                  │                  │                  │
│ Cons:            │ Cons:            │ Cons:            │
│ • Requires LLM   │ • Rigid          │ • Linear         │
│ • Slower         │ • Less flexible  │ • Order-specific │
└──────────────────┴──────────────────┴──────────────────┘
```

---

## Strategy 1: Semantic Decomposition

### When to Use

Semantic decomposition is best for:
- **Research problems**: Exploratory investigations
- **Design problems**: Creative solutions
- **Conceptual problems**: Abstract concepts
- **Novel domains**: No existing structure

### How It Works

1. **LLM Analysis**: Uses GPT-4 to understand problem semantics
2. **Concept Clustering**: Identifies natural conceptual boundaries
3. **Context Awareness**: Considers domain, constraints, success criteria
4. **Smart Boundaries**: Creates sub-problems with minimal overlap

### Example: Research Problem

```python
# example_semantic_research.py
from decomposition_engine import SemanticDecomposition, DecompositionEngine
from sovereign_data_models import ProblemDefinition, ProblemType, DomainContext, ComplexityScore

# Define research problem
research_problem = ProblemDefinition(
    id="research-001",
    title="Quantum Machine Learning Algorithms",
    description="""Investigate and develop quantum machine learning algorithms
    that can provide speedup over classical algorithms for:
    - Classification tasks
    - Clustering problems
    - Optimization challenges

    Focus on near-term quantum computers (NISQ era) with 50-100 qubits.
    Consider noise resilience and practical implementation constraints.
    """,
    problem_type=ProblemType.RESEARCH,
    domain_context=DomainContext(
        domain="Quantum Computing",
        subdomain="Machine Learning"
    ),
    complexity_score=ComplexityScore(
        overall_complexity=9,
        cognitive_complexity=10,
        computational_complexity=8,
        domain_complexity=10,
        integration_complexity=7
    )
)

# Create semantic strategy
strategy = SemanticDecomposition()

# Or use via engine
engine = DecompositionEngine()
result = engine.decompose(research_problem, strategy="semantic")

# Print results
print(f"=== Semantic Decomposition ===")
print(f"Generated {len(result.sub_problems)} sub-problems:\n")

for sp in result.sub_problems:
    print(f"📚 {sp.title}")
    print(f"   Type: {sp.type.value}")
    print(f"   Focus: {sp.description[:150]}...")
    print()
```

**Expected Output:**
```
=== Semantic Decomposition ===
Generated 5 sub-problems:

📚 Literature Review on QML Algorithms
   Type: research
   Focus: Survey existing quantum machine learning algorithms, identify
   promising approaches for NISQ devices, categorize by quantum advantage...

📚 Quantum Data Encoding Investigation
   Type: research
   Focus: Research efficient data encoding schemes for quantum computers,
   compare amplitude, basis, and angle encoding, analyze resource requirements...

📚 Noise-Resilient Algorithm Design
   Type: analysis
   Focus: Design QML algorithms robust to NISQ-era noise, explore error
   mitigation techniques, develop noise-aware training procedures...

📚 Benchmark Development
   Type: implementation
   Focus: Create quantum simulation framework for benchmarking QML
   algorithms, implement classical baselines, design performance metrics...

📚 Experimental Validation
   Type: validation
   Focus: Validate algorithms on real quantum hardware, analyze results
   from quantum computers, compare against theoretical predictions...
```

### Customizing Semantic Decomposition

```python
from openevolve_client import OpenEvolveClient

# Create custom OpenEvolve client
client = OpenEvolveClient(
    model="gpt-4",  # Use GPT-4 for better quality
    temperature=0.3,  # Lower for more consistent decomposition
    max_tokens=6000  # Allow longer responses
)

# Create strategy with custom client
strategy = SemanticDecomposition(openevolve_client=client)

# Decompose
sub_problems = strategy.decompose(research_problem)
```

---

## Strategy 2: Hierarchical Decomposition

### When to Use

Hierarchical decomposition is best for:
- **System architecture**: Multi-tier systems
- **Organizational structure**: Company hierarchies
- **Complex software**: Large codebases
- **Infrastructure**: Cloud architecture

### How It Works

1. **Top-Level Identification**: Identifies main components
2. **Recursive Breakdown**: Recursively breaks down each component
3. **Layer Creation**: Creates hierarchical layers
4. **Dependency Tracking**: Maintains parent-child relationships

### Example: System Architecture

```python
# example_hierarchical_system.py
from decomposition_engine import HierarchicalDecomposition, DecompositionEngine
from sovereign_data_models import ProblemDefinition, ProblemType, DomainContext, ComplexityScore

# Define system architecture problem
system_problem = ProblemDefinition(
    id="system-001",
    title="E-Commerce Microservices Architecture",
    description="""Design and implement a scalable e-commerce platform using
    microservices architecture:

    Business Requirements:
    - Handle 10,000 concurrent users
    - Support 1M+ products
    - Process 1000 orders/minute
    - 99.99% uptime
    - Global deployment

    Technical Requirements:
    - Microservices architecture
    - Event-driven communication
    - Caching layer
    - CDN for static content
    - Database replication
    - Auto-scaling
    - Monitoring and alerting
    """,
    problem_type=ProblemType.DESIGN,
    domain_context=DomainContext(
        domain="Software Architecture",
        subdomain="Distributed Systems"
    ),
    complexity_score=ComplexityScore(
        overall_complexity=9,
        cognitive_complexity=7,
        computational_complexity=8,
        domain_complexity=9,
        integration_complexity=10
    )
)

# Create hierarchical strategy
strategy = HierarchicalDecomposition(
    max_depth=3,  # Maximum hierarchy depth
    min_sub_problems=4,  # Minimum sub-problems per level
    balance=True  # Balance workload across branches
)

# Or use via engine
engine = DecompositionEngine()
result = engine.decompose(system_problem, strategy="hierarchical")

# Print hierarchical structure
def print_hierarchy(sub_problems, level=0):
    """Print sub-problems as hierarchy tree"""
    indent = "  " * level

    for sp in sub_problems:
        # Find children (sub-problems that depend on this one)
        children = [child for child in sub_problems if sp.id in child.dependencies]

        # Print this node
        icon = "🏗️" if level == 0 else "📦"
        print(f"{indent}{icon} {sp.title}")
        print(f"{indent}   Complexity: {sp.complexity_score.overall_complexity}/10")

        # Recursively print children
        if children and level < 3:
            print_hierarchy(children, level + 1)

print("\n=== Hierarchical Decomposition ===")
print_hierarchy(result.sub_problems)
```

**Expected Output:**
```
=== Hierarchical Decomposition ===
🏗️ Core Infrastructure Layer
   Complexity: 9/10
  📦 Network Architecture Design
    Complexity: 8/10
  📦 Database Cluster Setup
    Complexity: 9/10
  📦 Caching Layer Implementation
    Complexity: 7/10

🏗️ Application Services Layer
   Complexity: 8/10
  📦 User Service Development
    Complexity: 7/10
  📦 Product Service Development
    Complexity: 7/10
  📦 Order Service Development
    Complexity: 8/10
  📦 Payment Service Development
    Complexity: 9/10

🏗️ API Gateway Layer
   Complexity: 7/10
  📦 Gateway Configuration
    Complexity: 6/10
  📦 Rate Limiting Implementation
    Complexity: 5/10
  📦 Authentication Integration
    Complexity: 8/10

🏗️ Monitoring and Operations
   Complexity: 7/10
  📦 Monitoring System Setup
    Complexity: 7/10
  📦 Alerting Configuration
    Complexity: 6/10
  📦 Logging Infrastructure
    Complexity: 5/10
```

### Customizing Hierarchical Decomposition

```python
# Create custom hierarchical strategy
strategy = HierarchicalDecomposition(
    max_depth=4,  # Allow deeper hierarchy
    min_sub_problems=3,  # Fewer sub-problems per branch
    balance=True,  # Balance complexity
    preserve_cross_cutting=True  # Keep cross-cutting concerns visible
)

# Decompose
sub_problems = strategy.decompose(system_problem)
```

---

## Strategy 3: Flow-Based Decomposition

### When to Use

Flow-based decomposition is best for:
- **Business processes**: Order processing, workflows
- **Data pipelines**: ETL, data processing
- **Manufacturing**: Production lines
- **Scientific workflows**: Experiment protocols

### How It Works

1. **Stage Identification**: Identifies key process stages
2. **Flow Mapping**: Maps data/control flow
3. **Dependency Creation**: Establishes input/output dependencies
4. **Parallelization**: Identifies parallelizable stages

### Example: Data Pipeline

```python
# example_flow_pipeline.py
from decomposition_engine import FlowBasedDecomposition, DecompositionEngine
from sovereign_data_models import ProblemDefinition, ProblemType, DomainContext, ComplexityScore

# Define data pipeline problem
pipeline_problem = ProblemDefinition(
    id="pipeline-001",
    title="Real-Time Fraud Detection Pipeline",
    description="""Build a real-time fraud detection pipeline for financial transactions:

    Pipeline Stages:
    1. Transaction ingestion (1000 transactions/second)
    2. Data validation and normalization
    3. Feature extraction
    4. Model inference (ML model)
    5. Risk scoring
    6. Alert generation
    7. Result storage and reporting

    Requirements:
    - Latency < 100ms per transaction
    - Throughput: 1000 tx/second
    - Scalable to 10,000 tx/second
    - 99.9% availability
    - Real-time monitoring
    """,
    problem_type=ProblemType.IMPLEMENTATION,
    domain_context=DomainContext(
        domain="Data Engineering",
        subdomain="Real-Time Processing"
    ),
    complexity_score=ComplexityScore(
        overall_complexity=8,
        cognitive_complexity=6,
        computational_complexity=10,
        domain_complexity=7,
        integration_complexity=9
    )
)

# Create flow-based strategy
strategy = FlowBasedDecomposition(
    preserve_order=True,  # Maintain stage order
    allow_parallel=True,  # Identify parallel opportunities
    batch_processing=True  # Consider batch optimization
)

# Or use via engine
engine = DecompositionEngine()
result = engine.decompose(pipeline_problem, strategy="flow")

# Print pipeline flow
def print_pipeline_flow(sub_problems):
    """Print sub-problems as pipeline stages"""
    print("\n=== Pipeline Flow ===\n")

    # Group by stage
    stages = {}
    for sp in sub_problems:
        # Extract stage from description or use order
        stage_num = sub_problems.index(sp) + 1
        stages[stage_num] = sp

    # Print stages
    for stage_num in sorted(stages.keys()):
        sp = stages[stage_num]
        parallelizable = "⚡ (parallelizable)" if not sp.dependencies else ""

        print(f"Stage {stage_num}: {sp.title} {parallelizable}")
        print(f"  Input: {sp.description.split('Input:')[1].split('Output:')[0].strip() if 'Input:' in sp.description else 'N/A'}")
        print(f"  Output: {sp.description.split('Output:')[1].strip() if 'Output:' in sp.description else 'N/A'}")
        print(f"  Complexity: {sp.complexity_score.overall_complexity}/10")
        print(f"  Estimated Latency: {sp.estimated_effort}ms")
        print()

print_pipeline_flow(result.sub_problems)
```

**Expected Output:**
```
=== Pipeline Flow ===

Stage 1: Transaction Ingestion Service
  Input: Raw transactions from payment gateway
  Output: Validated transaction stream
  Complexity: 7/10
  Estimated Latency: 10ms

Stage 2: Data Validation and Normalization
  Input: Validated transaction stream
  Output: Normalized transaction data
  Complexity: 6/10
  Estimated Latency: 15ms

Stage 3: Feature Extraction Engine ⚡ (parallelizable)
  Input: Normalized transaction data
  Output: Feature vector (150+ features)
  Complexity: 8/10
  Estimated Latency: 20ms

Stage 4: ML Model Inference ⚡ (parallelizable)
  Input: Feature vector
  Output: Fraud probability score
  Complexity: 9/10
  Estimated Latency: 30ms

Stage 5: Risk Scoring Algorithm
  Input: Fraud probability + transaction metadata
  Output: Final risk score (0-100)
  Complexity: 7/10
  Estimated Latency: 5ms

Stage 6: Alert Generation Service ⚡ (parallelizable)
  Input: High-risk transactions
  Output: Alert notifications
  Complexity: 6/10
  Estimated Latency: 10ms

Stage 7: Result Storage and Reporting
  Input: All processed transactions
  Output: Database records + analytics reports
  Complexity: 7/10
  Estimated Latency: 10ms

Total Pipeline Latency: ~100ms ✓
```

### Customizing Flow-Based Decomposition

```python
# Create custom flow-based strategy
strategy = FlowBasedDecomposition(
    preserve_order=True,  # Maintain sequential order
    allow_parallel=True,  # Mark parallelizable stages
    batch_processing=True,  # Consider batch optimization
    buffer_sizing=True  # Add buffer stages between components
)

# Decompose
sub_problems = strategy.decompose(pipeline_problem)
```

---

## Choosing the Right Strategy

### Decision Tree

```
START
  │
  ├─ Is the problem a workflow/pipeline?
  │   └─ YES → Use FLOW-BASED
  │
  ├─ Is the problem a system/architecture?
  │   └─ YES → Use HIERARCHICAL
  │
  ├─ Is the problem research/design?
  │   └─ YES → Use SEMANTIC
  │
  └─ Not sure?
      └─ Use SEMANTIC (most flexible)
```

### Comparison Matrix

| Aspect | Semantic | Hierarchical | Flow-Based |
|--------|----------|--------------|------------|
| **Problem Types** | Research, Design | Systems, Architecture | Pipelines, Workflows |
| **Structure** | Conceptual clusters | Tree hierarchy | Sequential stages |
| **Dependencies** | Cross-cutting | Parent-child | Sequential |
| **Flexibility** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |
| **Scalability** | ★★★★☆ | ★★★★★ | ★★★★☆ |
| **LLM Required** | Yes | Optional | Optional |
| **Speed** | Slow | Fast | Fast |

---

## Combining Strategies

### Hybrid Approach: Semantic + Hierarchical

```python
# example_hybrid_strategy.py
from decomposition_engine import SemanticDecomposition, HierarchicalDecomposition
from sovereign_data_models import ProblemDefinition

# For a complex system, use semantic first, then hierarchical
complex_system_problem = ProblemDefinition(
    id="complex-001",
    title="Autonomous Vehicle Software System",
    description="Design software architecture for self-driving car...",
    # ... other fields
)

# Step 1: Use semantic to identify major components
semantic_strategy = SemanticDecomposition()
top_level_components = semantic_strategy.decompose(complex_system_problem)

# Step 2: For each major component, use hierarchical to break down further
for component in top_level_components[:3]:  # Just first 3 as example
    print(f"\n=== Decomposing: {component.title} ===")

    # Create sub-problem for this component
    component_problem = ProblemDefinition(
        id=component.id,
        title=component.title,
        description=component.description,
        problem_type=component.type,
        domain_context=complex_system_problem.domain_context,
        complexity_score=component.complexity_score
    )

    # Hierarchical decomposition
    hierarchical_strategy = HierarchicalDecomposition(max_depth=2)
    sub_components = hierarchical_strategy.decompose(component_problem)

    print(f"Generated {len(sub_components)} sub-components:")
    for sc in sub_components:
        print(f"  - {sc.title}")
```

---

## Advanced: Custom Strategy

```python
# example_custom_strategy.py
from decomposition_engine import DecompositionStrategyBase
from sovereign_data_models import SubProblem, ProblemDefinition
from typing import List

class DomainSpecificStrategy(DecompositionStrategyBase):
    """Custom strategy for domain-specific decomposition"""

    def __init__(self, domain_templates: dict):
        self.domain_templates = domain_templates

    def get_strategy_name(self) -> str:
        return "domain_specific"

    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose using domain-specific templates"""

        # Get template for this domain
        domain = problem.domain_context.domain
        template = self.domain_templates.get(domain)

        if not template:
            raise ValueError(f"No template for domain: {domain}")

        # Create sub-problems from template
        sub_problems = []
        for i, component in enumerate(template['components'], 1):
            sp = SubProblem(
                id=f"{problem.id}-sub-{i:03d}",
                parent_id=problem.id,
                title=component['title'],
                description=component['description'],
                type=SubProblemType[component['type'].upper()],
                complexity_score=problem.complexity_score,
                dependencies=[],
                success_criteria=[],
                validation_gauntlet="",
                priority=component.get('priority', 5),
                estimated_effort=component.get('effort', 8)
            )
            sub_problems.append(sp)

        return sub_problems

# Define domain templates
ML_DOMAIN_TEMPLATE = {
    'components': [
        {
            'title': 'Data Collection and Preparation',
            'description': 'Gather and preprocess training data',
            'type': 'implementation',
            'priority': 9,
            'effort': 16
        },
        {
            'title': 'Feature Engineering',
            'description': 'Design and extract relevant features',
            'type': 'analysis',
            'priority': 8,
            'effort': 12
        },
        {
            'title': 'Model Selection and Training',
            'description': 'Train and validate ML models',
            'type': 'implementation',
            'priority': 9,
            'effort': 24
        },
        {
            'title': 'Evaluation and Testing',
            'description': 'Evaluate model performance',
            'type': 'validation',
            'priority': 7,
            'effort': 8
        },
        {
            'title': 'Deployment and Monitoring',
            'description': 'Deploy model to production',
            'type': 'integration',
            'priority': 7,
            'effort': 12
        }
    ]
}

# Use custom strategy
strategy = DomainSpecificStrategy(domain_templates={
    'Machine Learning': ML_DOMAIN_TEMPLATE,
    # Add more domain templates...
})

# Decompose ML problem
ml_problem = ProblemDefinition(
    id="ml-001",
    title="Customer Churn Prediction",
    description="Build ML model to predict customer churn",
    problem_type=ProblemType.IMPLEMENTATION,
    domain_context=DomainContext(domain="Machine Learning"),
    complexity_score=ComplexityScore(overall_complexity=6, cognitive_complexity=5, computational_complexity=7, domain_complexity=6, integration_complexity=5)
)

sub_problems = strategy.decompose(ml_problem)
print(f"Generated {len(sub_problems)} sub-problems using ML template")
```

---

## Exercise: Strategy Selection

Given these problems, which strategy would you choose and why?

### Problem 1: Research Project
*"Investigate quantum computing applications for drug discovery"*

**Solution:**
```
Strategy: SEMANTIC
Reasoning:
- Research problem requires conceptual understanding
- No existing structure
- Needs flexibility to explore different directions
- Domain is novel (quantum + biology)
```

### Problem 2: E-Commerce Platform
*"Build a scalable e-commerce platform with microservices"*

**Solution:**
```
Strategy: HIERARCHICAL
Reasoning:
- Clear system architecture
- Well-defined layers (API, services, database)
- Multiple components need structured breakdown
- Scalability is key requirement
```

### Problem 3: ETL Pipeline
*"Create data pipeline for processing sales data"*

**Solution:**
```
Strategy: FLOW-BASED
Reasoning:
- Sequential data flow (extract → transform → load)
- Clear stages with input/output
- Opportunity for parallelization
- Performance targets (latency, throughput)
```

---

## Summary

In this tutorial, you learned:

✓ Three decomposition strategies and when to use them
✓ How to customize strategy parameters
✓ How to combine multiple strategies
✓ How to create custom domain-specific strategies
✓ Decision-making framework for strategy selection

---

## Next Steps

**Next Tutorial:** [Tutorial 3: Quality Assessment](tutorial_03_quality.md)

In the next tutorial, you'll learn how to:
- Assess decomposition quality
- Identify and fix quality issues
- Validate sub-problems
- Measure decomposition effectiveness

---

**Tutorial Version:** 1.0.0
**Last Updated:** 2025-01-03
