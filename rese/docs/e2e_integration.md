# RESE E2E Integration Guide

**Recursive Epistemic Solvability Engine**
**Version:** 1.0.0
**Last Updated:** 2025-12-31

---

## Table of Contents

1. [Overview](#overview)
2. [E2E Stage Integration](#e2e-stage-integration)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Interface Specifications](#interface-specifications)
5. [Integration Examples](#integration-examples)
6. [Best Practices](#best-practices)

---

## Overview

### What is E2E Integration?

The **E2E (End-to-End) Stage** is the integration layer that connects RESE with external systems:
- Invention Engine
- Knowledge Graphs
- External Solvers
- User Interfaces
- Data Sources

### Integration Architecture

```
┌──────────────────────────────────────────────────────┐
│                 E2E Invention Engine                   │
│              (External System)                         │
└───────────────────┬──────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────┐
│              RESE-E2E Integration Layer               │
│  ┌──────────────────────────────────────────────┐   │
│  │  Input Adapter                              │   │
│  │  - Convert E2E format → RESE format          │   │
│  │  - Validate inputs                           │   │
│  └───────────────────┬──────────────────────────┘   │
│                      │                               │
│  ┌───────────────────▼──────────────────────────┐   │
│  │  RESE Pipeline                              │   │
│  │  - Phase I: Epistemic Audit                  │   │
│  │  - Phase II: Isomorphic Resonance            │   │
│  │  - Phase III: Monte Carlo Refinement         │   │
│  │  - Phase IV: Architectural Synthesis         │   │
│  └───────────────────┬──────────────────────────┘   │
│                      │                               │
│  ┌───────────────────▼──────────────────────────┐   │
│  │  Output Adapter                             │   │
│  │  - Convert RESE format → E2E format          │   │
│  │  - Format results                            │   │
│  └───────────────────┬──────────────────────────┘   │
└──────────────────────┼──────────────────────────────┘
                       │
                       ↓
              ┌─────────────────┐
              │  Back to E2E     │
              │  Engine         │
              └─────────────────┘
```

---

## E2E Stage Integration

### Integration Points

RESE integrates with E2E at **5 key points**:

1. **Input Stage:** Problem definition from E2E → RESE
2. **Phase I:** Bias detection results → E2E feedback
3. **Phase II:** Isomorphism results → E2E knowledge transfer
4. **Phase III:** ACI values → E2E search guidance
5. **Output Stage:** Solution from RESE → E2E engine

---

### Stage 5 Integration Module

**File:** `rese/core/stage5_integration.py`

**Purpose:** Integrates RESE with E2E Stage 5 (Formal Reasoning).

```python
"""
RESE-E2E Stage 5 Integration

Connects RESE pipeline with E2E Stage 5 (Formal Reasoning Mode).
Handles bidirectional communication and format conversion.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class IntegrationMode(Enum):
    """Integration modes"""
    SYNCHRONOUS = "synchronous"     # Blocking calls
    ASYNCHRONOUS = "asynchronous"   # Non-blocking
    STREAMING = "streaming"         # Streaming results


@dataclass
class E2EInput:
    """Input from E2E Engine"""
    problem_id: str
    problem_description: str
    domain: str
    constraints: List[Dict[str, Any]]
    variables: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class E2EOutput:
    """Output to E2E Engine"""
    problem_id: str
    solution: Dict[str, Any]
    confidence: float
    aci_history: List[float]
    phase_results: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class Stage5Integrator:
    """
    Integrates RESE with E2E Stage 5.

    Handles:
    - Input format conversion (E2E → RESE)
    - Output format conversion (RESE → E2E)
    - Bidirectional communication
    - Error handling and recovery
    """

    def __init__(self, mode: IntegrationMode = IntegrationMode.SYNCHRONOUS):
        """
        Initialize integrator.

        Args:
            mode: Integration mode (synchronous, asynchronous, streaming)
        """
        self.mode = mode
        self.input_adapter = E2EInputAdapter()
        self.output_adapter = E2EOutputAdapter()

    def process(
        self,
        e2e_input: E2EInput,
        rese_config: Optional[Dict] = None
    ) -> E2EOutput:
        """
        Process E2E input through RESE pipeline.

        Args:
            e2e_input: Input from E2E Engine
            rese_config: Optional RESE configuration overrides

        Returns:
            E2EOutput: Formatted output for E2E Engine
        """
        # Step 1: Convert E2E input to RESE format
        rese_problem = self.input_adapter.adapt(e2e_input)

        # Step 2: Run RESE pipeline
        from rese.rese_pipeline import RESEPipeline

        pipeline = RESEPipeline()
        result = pipeline.run(rese_problem)

        # Step 3: Convert RESE output to E2E format
        e2e_output = self.output_adapter.adapt(result, e2e_input)

        return e2e_output

    def stream_process(
        self,
        e2e_input: E2EInput,
        progress_callback: Optional[callable] = None
    ):
        """
        Stream RESE progress back to E2E.

        Args:
            e2e_input: Input from E2E Engine
            progress_callback: Optional callback for progress updates

        Yields:
            Progress updates
        """
        from rese.rese_pipeline import RESEPipeline

        rese_problem = self.input_adapter.adapt(e2e_input)
        pipeline = RESEPipeline()

        # Add progress callback
        if progress_callback:
            pipeline.add_progress_callback(
                lambda r: progress_callback(
                    self.output_adapter.adapt_progress(r)
                )
            )

        # Run pipeline
        result = pipeline.run(rese_problem)

        # Yield final result
        yield self.output_adapter.adapt(result, e2e_input)


class E2EInputAdapter:
    """Converts E2E input format to RESE format"""

    def adapt(self, e2e_input: E2EInput) -> 'ProblemInput':
        """
        Convert E2E input to RESE ProblemInput.

        Args:
            e2e_input: Input from E2E Engine

        Returns:
            ProblemInput: RESE-formatted input
        """
        from rese.rese_pipeline import ProblemInput

        return ProblemInput(
            id=e2e_input.problem_id,
            description=e2e_input.problem_description,
            constraints=self._convert_constraints(e2e_input.constraints),
            variables=e2e_input.variables,
            domain=e2e_input.domain,
            metadata={
                **e2e_input.metadata,
                'e2e_context': e2e_input.context
            }
        )

    def _convert_constraints(
        self,
        e2e_constraints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert E2E constraints to RESE format"""
        converted = []

        for constraint in e2e_constraints:
            converted.append({
                'id': constraint.get('id', ''),
                'type': constraint.get('type', 'soft').upper(),
                'description': constraint.get('description', ''),
                'formalization': constraint.get('formalization', ''),
                'source': 'e2e_engine'
            })

        return converted


class E2EOutputAdapter:
    """Converts RESE output format to E2E format"""

    def adapt(
        self,
        rese_result: 'PipelineResult',
        original_input: E2EInput
    ) -> E2EOutput:
        """
        Convert RESE result to E2E output.

        Args:
            rese_result: Result from RESE pipeline
            original_input: Original E2E input

        Returns:
            E2EOutput: E2E-formatted output
        """
        return E2EOutput(
            problem_id=rese_result.problem_id,
            solution=self._extract_solution(rese_result),
            confidence=rese_result.confidence,
            aci_history=rese_result.aci_history,
            phase_results=self._extract_phase_results(rese_result),
            metadata={
                'status': rese_result.status.value,
                'elapsed_seconds': rese_result.elapsed_seconds,
                'num_phases_completed': len(rese_result.phase_results)
            },
            recommendations=self._generate_recommendations(rese_result)
        )

    def adapt_progress(self, rese_result: 'PipelineResult') -> Dict[str, Any]:
        """Adapt progress update for E2E"""
        return {
            'problem_id': rese_result.problem_id,
            'status': rese_result.status.value,
            'elapsed_seconds': rese_result.elapsed_seconds,
            'phases_completed': len(rese_result.phase_results),
            'current_phase': self._get_current_phase(rese_result)
        }

    def _extract_solution(
        self,
        rese_result: 'PipelineResult'
    ) -> Dict[str, Any]:
        """Extract solution from RESE result"""
        if not rese_result.final_solution:
            return {}

        return {
            'architecture': rese_result.final_solution,
            'validation_score': rese_result.validation_score,
            'confidence': rese_result.confidence
        }

    def _extract_phase_results(
        self,
        rese_result: 'PipelineResult'
    ) -> Dict[str, Any]:
        """Extract phase-wise results"""
        phase_summaries = {}

        for phase_name, phase_result in rese_result.phase_results.items():
            phase_summaries[phase_name] = {
                'status': phase_result.status.value,
                'elapsed_seconds': phase_result.elapsed_seconds,
                'metrics': phase_result.metrics,
                'has_errors': len(phase_result.errors) > 0,
                'num_warnings': len(phase_result.warnings)
            }

        return phase_summaries

    def _generate_recommendations(
        self,
        rese_result: 'PipelineResult'
    ) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []

        # Analyze ACI trend
        if len(rese_result.aci_history) > 1:
            initial_aci = rese_result.aci_history[0]
            final_aci = rese_result.aci_history[-1]
            reduction = initial_aci - final_aci

            if reduction > 0.3:
                recommendations.append(
                    f"Strong ACI reduction detected ({reduction:.2f}). "
                    "Solution is highly promising."
                )
            elif reduction > 0.1:
                recommendations.append(
                    f"Moderate ACI reduction ({reduction:.2f}). "
                    "Solution shows improvement."
                )
            else:
                recommendations.append(
                    f"Limited ACI reduction ({reduction:.2f}). "
                    "Consider additional refinement."
                )

        # Check validation score
        if rese_result.validation_score < 0.7:
            recommendations.append(
                "Validation score below threshold. "
                "Review phase results for potential issues."
            )

        # Check for errors
        if rese_result.errors:
            recommendations.append(
                f"{len(rese_result.errors)} errors detected. "
                "Review error logs for details."
            )

        return recommendations

    def _get_current_phase(self, rese_result: 'PipelineResult') -> Optional[str]:
        """Get current executing phase"""
        for phase_name, phase_result in rese_result.phase_results.items():
            if phase_result.status.value == 'running':
                return phase_name
        return None
```

---

## Data Flow Diagrams

### Complete E2E-RESE Data Flow

```
┌────────────────────────────────────────────────────────────┐
│                    E2E Invention Engine                     │
│  - User provides problem description                        │
│  - Domain knowledge from knowledge graph                    │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ↓
┌────────────────────────────────────────────────────────────┐
│                 Input: E2EInput Format                      │
│  {                                                         │
│    "problem_id": "tsp_50",                                 │
│    "problem_description": "50-city TSP",                   │
│    "domain": "logistics",                                  │
│    "constraints": [...],                                   │
│    "variables": {...}                                      │
│  }                                                         │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ↓
┌────────────────────────────────────────────────────────────┐
│              RESE-E2E Input Adapter                        │
│  - Validates input format                                  │
│  - Converts constraints to RESE format                     │
│  - Adds metadata                                           │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ↓
┌────────────────────────────────────────────────────────────┐
│                 RESE Pipeline Execution                     │
│                                                              │
│  ┌────────────────────────────────────────────┐           │
│  │ Phase I: Epistemic Audit                   │           │
│  │  - SCE validates constraints               │           │
│  │  - Φ₁.₅ mines tacit assumptions            │           │
│  │  - Φ₂ detects cognitive biases             │           │
│  │  → Bias report ← E2E feedback point        │           │
│  └─────────────────┬──────────────────────────┘           │
│                    ↓                                        │
│  ┌────────────────────────────────────────────┐           │
│  │ Phase II: Isomorphic Resonance             │           │
│  │  - Ψ₁ inverts constraints                  │           │
│  │  - Ψ₂ maps ontologies                      │           │
│  │  - Ψ₃ validates isomorphisms               │           │
│  │  → Isomorphisms ← E2E knowledge transfer   │           │
│  └─────────────────┬──────────────────────────┘           │
│                    ↓                                        │
│  ┌────────────────────────────────────────────┐           │
│  │ Phase III: Monte Carlo Refinement          │           │
│  │  - Γ₁ calculates ACI                       │           │
│  │  - Γ₂ runs MCTS search                     │           │
│  │  - Γ₃ validates statistically              │           │
│  │  → ACI values ← E2E search guidance        │           │
│  └─────────────────┬──────────────────────────┘           │
│                    ↓                                        │
│  ┌────────────────────────────────────────────┐           │
│  │ Phase IV: Architectural Synthesis          │           │
│  │  - Δ₁ assembles architecture               │           │
│  │  - Δ₂ generates predictions                │           │
│  │  - Δ₃ validates ACI reduction              │           │
│  └─────────────────┬──────────────────────────┘           │
└────────────────────┼───────────────────────────────────────┘
                     │
                     ↓
┌────────────────────────────────────────────────────────────┐
│              RESE-E2E Output Adapter                       │
│  - Extracts solution                                       │
│  - Formats results                                         │
│  - Generates recommendations                               │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ↓
┌────────────────────────────────────────────────────────────┐
│                 Output: E2EOutput Format                    │
│  {                                                         │
│    "problem_id": "tsp_50",                                 │
│    "solution": {...},                                      │
│    "confidence": 0.87,                                     │
│    "aci_history": [0.8, 0.65, 0.43, 0.30],                │
│    "phase_results": {...},                                 │
│    "recommendations": [...]                                │
│  }                                                         │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ↓
┌────────────────────────────────────────────────────────────┐
│                    E2E Invention Engine                     │
│  - Integrates solution into invention workflow             │
│  - Updates knowledge graph with new solution               │
│  - Presents results to user                                │
└────────────────────────────────────────────────────────────┘
```

---

### ACI-Guided Search Integration

```
┌────────────────────────────────────────────────────────┐
│              E2E Search Controller                      │
│  - Manages overall search process                       │
│  - Coordinates multiple strategies                      │
└────────────────────────┬───────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────┐
│         Request: Calculate ACI for State               │
│  {                                                     │
│    "state": current_search_state,                      │
│    "context": search_context                           │
│  }                                                     │
└────────────────────────┬───────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────┐
│              RESE Γ₁ ACI Calculator                     │
│  - Calculates disorder entropy (H)                     │
│  - Calculates causal coherence (C)                     │
│  - Calculates solvability index (S)                    │
│  - Returns ACI = α·(1-H) + β·C + γ·S                   │
└────────────────────────┬───────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────┐
│         Response: ACI Value + Guidance                 │
│  {                                                     │
│    "ACI": 0.65,                                        │
│    "confidence": 0.85,                                 │
│    "recommendation": "continue_search",                │
│    "components": {H: 0.3, C: 0.7, S: 0.8}             │
│  }                                                     │
└────────────────────────┬───────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────┐
│              E2E Search Controller                      │
│  - Uses ACI to guide node selection                    │
│  - Prioritizes high-ACI branches                       │
│  - Allocates more resources to promising regions       │
└────────────────────────────────────────────────────────┘
```

---

## Interface Specifications

### E2E → RESE Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "E2EInput",
  "type": "object",
  "required": ["problem_id", "problem_description", "domain", "constraints", "variables"],
  "properties": {
    "problem_id": {
      "type": "string",
      "description": "Unique problem identifier"
    },
    "problem_description": {
      "type": "string",
      "description": "Human-readable problem description"
    },
    "domain": {
      "type": "string",
      "description": "Problem domain (e.g., 'logistics', 'engineering')"
    },
    "constraints": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "type": {"type": "string", "enum": ["hard", "soft", "preference"]},
          "description": {"type": "string"},
          "formalization": {"type": "string"}
        },
        "required": ["id", "type", "description"]
      }
    },
    "variables": {
      "type": "object",
      "description": "Problem variables (key-value pairs)"
    },
    "context": {
      "type": "object",
      "description": "Additional context from E2E"
    },
    "metadata": {
      "type": "object",
      "description": "Metadata (timestamp, user, etc.)"
    }
  }
}
```

### RESE → E2E Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "E2EOutput",
  "type": "object",
  "required": ["problem_id", "solution", "confidence", "aci_history"],
  "properties": {
    "problem_id": {
      "type": "string",
      "description": "Problem identifier (matches input)"
    },
    "solution": {
      "type": "object",
      "description": "Solution found by RESE",
      "properties": {
        "architecture": {"type": "object"},
        "validation_score": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
      }
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Overall confidence in solution"
    },
    "aci_history": {
      "type": "array",
      "items": {"type": "number"},
      "description": "ACI values at each phase"
    },
    "phase_results": {
      "type": "object",
      "description": "Results from each phase"
    },
    "metadata": {
      "type": "object",
      "properties": {
        "status": {"type": "string"},
        "elapsed_seconds": {"type": "number"},
        "num_phases_completed": {"type": "integer"}
      }
    },
    "recommendations": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Recommendations for next steps"
    }
  }
}
```

### REST API Interface

**Base URL:** `http://localhost:8000/api/v1`

---

#### POST /process

Process problem through RESE pipeline.

**Request:**
```json
POST /api/v1/process
Content-Type: application/json

{
  "problem_id": "tsp_50",
  "problem_description": "50-city Traveling Salesman Problem",
  "domain": "logistics",
  "constraints": [...],
  "variables": {...}
}
```

**Response:**
```json
{
  "problem_id": "tsp_50",
  "solution": {...},
  "confidence": 0.87,
  "aci_history": [0.8, 0.65, 0.43, 0.30],
  "recommendations": [...]
}
```

---

#### GET /status/{problem_id}

Get processing status.

**Response:**
```json
{
  "problem_id": "tsp_50",
  "status": "running",
  "current_phase": "phase3",
  "progress": 0.75,
  "elapsed_seconds": 45.2
}
```

---

#### GET /aci

Calculate ACI for a state.

**Request:**
```json
POST /api/v1/aci
Content-Type: application/json

{
  "state": {...},
  "csp_instance": {...}
}
```

**Response:**
```json
{
  "ACI": 0.65,
  "components": {
    "disorder_entropy": 0.30,
    "causal_coherence": 0.70,
    "solvability_index": 0.80
  },
  "confidence": 0.85,
  "recommendation": "continue_search"
}
```

---

## Integration Examples

### Example 1: Basic Synchronous Integration

```python
from rese.core.stage5_integration import (
    Stage5Integrator,
    E2EInput,
    IntegrationMode
)

# Create E2E input
e2e_input = E2EInput(
    problem_id="tsp_50",
    problem_description="50-city Traveling Salesman Problem",
    domain="logistics",
    constraints=[
        {
            "id": "visit_all",
            "type": "hard",
            "description": "Visit all cities exactly once",
            "formalization": "∀ city ∈ cities: visited(city) = 1"
        },
        {
            "id": "minimize_distance",
            "type": "soft",
            "description": "Minimize total distance",
            "formalization": "minimize Σ dist(city_i, city_j)"
        }
    ],
    variables={
        "num_cities": 50,
        "coordinates": city_coordinates
    }
)

# Create integrator
integrator = Stage5Integrator(mode=IntegrationMode.SYNCHRONOUS)

# Process
e2e_output = integrator.process(e2e_input)

# Use results
print(f"Solution confidence: {e2e_output.confidence:.2f}")
print(f"ACI reduction: {e2e_output.aci_history[0] - e2e_output.aci_history[-1]:.2f}")
print(f"Recommendations:")
for rec in e2e_output.recommendations:
    print(f"  - {rec}")
```

---

### Example 2: Streaming Integration

```python
from rese.core.stage5_integration import Stage5Integrator, E2EInput

# Create integrator
integrator = Stage5Integrator(mode=IntegrationMode.STREAMING)

# Stream processing
for progress_update in integrator.stream_process(
    e2e_input,
    progress_callback=lambda p: print(f"Progress: {p['status']}")
):
    print(f"Phase: {progress_update['current_phase']}")
    print(f"Status: {progress_update['status']}")
    print(f"Completed: {progress_update['phases_completed']}/4")

# Final result available in last update
```

---

### Example 3: ACI-Guided Search Integration

```python
# E2E search controller
class E2ESearchController:
    def __init__(self):
        self.rese_client = RESEClient()
        self.search_queue = PriorityQueue()

    def select_next_node(self, nodes):
        """Use ACI to select most promising node"""
        # Calculate ACI for each node
        node_aci_pairs = []

        for node in nodes:
            aci_response = self.rese_client.calculate_aci(
                state=node.state
            )
            aci_value = aci_response['ACI']
            node_aci_pairs.append((node, aci_value))

        # Sort by ACI (higher is better)
        node_aci_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return highest ACI node
        return node_aci_pairs[0][0]

    def guided_search(self, initial_state):
        """Run ACI-guided search"""
        current = initial_state
        visited = set()

        while not self.is_terminal(current):
            # Get successors
            successors = self.get_successors(current)

            # Filter unvisited
            unvisited = [s for s in successors if s not in visited]

            if not unvisited:
                break

            # Use ACI to select next
            current = self.select_next_node(unvisited)
            visited.add(current)

        return current
```

---

### Example 4: Knowledge Transfer Integration

```python
from rese.phase2.imech import IMechValidator, Domain

# E2E maintains knowledge graph of solved problems
class E2EKnowledgeGraph:
    def __init__(self):
        self.solved_problems = {}

    def find_similar_solved(self, current_problem):
        """Find similar solved problems using I_mech"""
        validator = IMechValidator()

        # Create domain for current problem
        current_domain = Domain(
            id=current_problem.id,
            name=current_problem.name,
            variables=current_problem.variables,
            constraints=current_problem.constraints
        )

        # Compare with all solved problems
        similarities = []

        for solved_id, solved_problem in self.solved_problems.items():
            solved_domain = Domain(
                id=solved_id,
                name=solved_problem.name,
                variables=solved_problem.variables,
                constraints=solved_problem.constraints
            )

            comparison = validator.compare_domains(
                source=solved_domain,
                target=current_domain
            )

            if comparison.score > 0.7:
                similarities.append({
                    'solved_id': solved_id,
                    'similarity': comparison.score,
                    'confidence': comparison.confidence
                })

        # Return most similar
        if similarities:
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[0]

        return None
```

---

## Best Practices

### 1. Error Handling

**DO:**
```python
try:
    e2e_output = integrator.process(e2e_input)
except ValidationError as e:
    # Handle validation errors
    logger.error(f"Invalid input: {e}")
    return error_response("Invalid input format")
except PipelineError as e:
    # Handle pipeline errors
    logger.error(f"Pipeline failed: {e}")
    return error_response("Processing failed")
```

**DON'T:**
```python
# Don't silently catch exceptions
try:
    e2e_output = integrator.process(e2e_input)
except:
    pass
```

---

### 2. Asynchronous Processing

**DO:**
```python
# Use async for long-running pipelines
import asyncio

async def process_async(e2e_input):
    loop = asyncio.get_event_loop()
    e2e_output = await loop.run_in_executor(
        None,
        integrator.process,
        e2e_input
    )
    return e2e_output
```

**DON'T:**
```python
# Don't block the event loop
e2e_output = integrator.process(e2e_input)  # Blocking!
```

---

### 3. Caching

**DO:**
```python
# Cache ACI calculations
@lru_cache(maxsize=1000)
def calculate_aci_cached(state_hash):
    return rese_client.calculate_aci(state=state)
```

**DON'T:**
```python
# Don't recalculate ACI for same state
aci = rese_client.calculate_aci(state)
# ... later ...
aci = rese_client.calculate_aci(state)  # Wasteful!
```

---

### 4. Monitoring

**DO:**
```python
# Track phase durations
import time

start = time.time()
e2e_output = integrator.process(e2e_input)
elapsed = time.time() - start

logger.info(f"RESE processing took {elapsed:.2f}s")

# Alert if too slow
if elapsed > 300:  # 5 minutes
    send_alert(f"RESE processing slow: {elapsed:.2f}s")
```

**DON'T:**
```python
# Don't ignore performance
e2e_output = integrator.process(e2e_input)  # No monitoring!
```

---

### 5. Input Validation

**DO:**
```python
# Validate E2E input before sending to RESE
def validate_e2e_input(e2e_input):
    required_fields = ['problem_id', 'problem_description',
                      'domain', 'constraints', 'variables']

    for field in required_fields:
        if field not in e2e_input:
            raise ValueError(f"Missing required field: {field}")

    # Validate constraints
    if not isinstance(e2e_input['constraints'], list):
        raise ValueError("Constraints must be a list")

    if len(e2e_input['constraints']) == 0:
        raise ValueError("At least one constraint required")

    return True
```

**DON'T:**
```python
# Don't send unvalidated input
e2e_output = integrator.process(e2e_input)  # May fail!
```

---

## End of Integration Guide

For more details, see:
- [User Guide](user_guide.md)
- [Developer Guide](developer_guide.md)
- [API Reference](api_reference.md)
