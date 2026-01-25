# Γ₁ Implementation Plan
**Agent D1 - ACI Analyzer Specialist**

**Date:** 2025-12-31
**Target Implementation:** Week 36
**Status:** Planning Phase

---

## Executive Summary

This document outlines the complete implementation plan for the Algorithmic Complexity Index (ACI) system, including data structures, integration strategy, development phases, and resource allocation.

**Implementation Goals:**
- Complete Γ₁ system by Week 36
- Achieve >85% ACI signal correlation
- Integrate with Stages 3, 6, and 9
- Support real-time ACI monitoring (<100ms updates)

---

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Data Structures](#2-data-structures)
3. [Implementation Phases](#3-implementation-phases)
4. [Integration Strategy](#4-integration-strategy)
5. [Testing Strategy](#5-testing-strategy)
6. [Deployment Plan](#6-deployment-plan)
7. [Resource Allocation](#7-resource-allocation)

---

## 1. System Architecture

### 1.1 Module Structure

```
rese/
├── gamma1/                           # Γ₁ ACI System
│   ├── __init__.py                   # Package initialization
│   ├── core/                         # Core ACI calculation
│   │   ├── __init__.py
│   │   ├── aci_calculator.py         # Main ACI calculation
│   │   ├── entropy_engine.py         # Disorder entropy (H)
│   │   ├── coherence_engine.py       # Causal coherence (C)
│   │   └── solvability_engine.py     # Solvability index (S)
│   ├── signal/                       # Signal extraction
│   │   ├── __init__.py
│   │   ├── signal_extractor.py       # SNR, correlation metrics
│   │   ├── threshold_learner.py      # Optimal threshold learning
│   │   └── validator.py              # Validation metrics
│   ├── adaptive/                     # Adaptive search integration
│   │   ├── __init__.py
│   │   ├── strategy_generator.py     # Search strategy recommendations
│   │   ├── mcts_guide.py             # MCTS parameter tuning
│   │   └── monitor.py                # Real-time ACI monitoring
│   ├── utils/                        # Utilities
│   │   ├── __init__.py
│   │   ├── graph_utils.py            # Constraint graph operations
│   │   ├── cache.py                  # Caching utilities
│   │   └── parallel.py               # Parallel computation
│   └── config/                       # Configuration
│       ├── __init__.py
│       ├── weights.py                # ACI weight parameters
│       └── thresholds.py             # Classification thresholds
└── tests/                            # Test suite
    ├── test_aci.py                   # ACI calculation tests
    ├── test_signal.py                # Signal extraction tests
    └── test_adaptive.py              # Adaptive integration tests
```

### 1.2 Technology Stack

**Core Dependencies:**
- Python 3.9+
- NumPy (numerical computations)
- NetworkX (graph algorithms)
- SciPy (statistical functions)
- Scikit-learn (machine learning utilities)

**Optional Dependencies:**
- Numba (JIT compilation for speed)
- Multiprocessing (parallel computation)
- Joblib (caching and parallelism)

---

## 2. Data Structures

### 2.1 Core Data Models

```python
# ============================================================================
# CSP Representation
# ============================================================================

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import numpy as np

@dataclass
class Variable:
    """CSP Variable"""
    name: str
    domain: List[Any]  # Domain values

@dataclass
class Constraint:
    """CSP Constraint"""
    variables: List[str]  # Variable names involved
    allowed_tuples: Set[Tuple]  # Set of allowed value tuples
    tightness: float = None  # Precomputed tightness (optional)

@dataclass
class CSPInstance:
    """
    Complete CSP Instance
    """
    variables: List[Variable]
    constraints: List[Constraint]
    metadata: Dict = None  # Optional metadata

    def __post_init__(self):
        """Build constraint graph"""
        self.constraint_graph = self._build_constraint_graph()
        self.variable_map = {v.name: v for v in self.variables}

    def _build_constraint_graph(self):
        """Build NetworkX graph from constraints"""
        import networkx as nx
        G = nx.Graph()

        # Add all variables as nodes
        for var in self.variables:
            G.add_node(var.name, domain_size=len(var.domain))

        # Add edges for constraints
        for constraint in self.constraints:
            vars_in_constraint = constraint.variables
            # Connect all pairs in constraint
            for i in range(len(vars_in_constraint)):
                for j in range(i+1, len(vars_in_constraint)):
                    G.add_edge(vars_in_constraint[i],
                              vars_in_constraint[j],
                              constraint=constraint)

        return G


# ============================================================================
# ACI Result
# ============================================================================

@dataclass
class ACIResult:
    """
    Complete ACI Calculation Result
    """
    # Primary score
    ACI: float  # ∈ [0, 1]

    # Component breakdown
    components: Dict[str, float]  # {'H': H, 'C': C, 'S': S}

    # Metadata
    confidence: float  # ∈ [0, 1]
    interpretation: Dict
    recommendation: Dict

    # Computation info
    computation_time: float  # seconds
    cached: bool = False

    def __str__(self):
        return f"ACI={self.ACI:.3f (confidence={self.confidence:.2f})"


# ============================================================================
# Entropy Components
# ============================================================================

@dataclass
class EntropyComponents:
    """Disorder Entropy (H) Components"""
    local: float  # Local domain entropy
    constraint: float  # Constraint entropy
    structural: float  # Structural entropy

    def total(self, weights=(0.3, 0.4, 0.3)) -> float:
        """Calculate total H with given weights"""
        return (weights[0] * self.local +
                weights[1] * self.constraint +
                weights[2] * self.structural)


# ============================================================================
# Coherence Components
# ============================================================================

@dataclass
class CoherenceComponents:
    """Causal Coherence (C) Components"""
    graph: float  # Graph structure coherence
    flow: float  # Information flow regularity
    stability: float  # Intervention stability

    def total(self, weights=(0.4, 0.3, 0.3)) -> float:
        """Calculate total C with given weights"""
        return (weights[0] * self.graph +
                weights[1] * self.flow +
                weights[2] * self.stability)


# ============================================================================
# Solvability Components
# ============================================================================

@dataclass
class SolvabilityComponents:
    """Solvability Index (S) Components"""
    phase_distance: float  # Distance from phase transition
    propagation: float  # Propagation effectiveness
    structure: float  # Constraint structure quality
    heuristic: float  # Heuristic effectiveness

    def total(self, weights=(0.3, 0.3, 0.2, 0.2)) -> float:
        """Calculate total S with given weights"""
        return (weights[0] * self.phase_distance +
                weights[1] * self.propagation +
                weights[2] * self.structure +
                weights[3] * self.heuristic)


# ============================================================================
# Signal Quality Metrics
# ============================================================================

@dataclass
class SignalQuality:
    """Signal Extraction Quality Metrics"""
    signal_to_noise: float  # SNR
    correlation: float  # Pearson correlation
    accuracy: float  # Classification accuracy
    auc: float  # ROC AUC

    mean_solvable_aci: float
    mean_intractable_aci: float

    def meets_target(self, target_correlation=0.85) -> bool:
        """Check if metrics meet target"""
        return (self.correlation >= target_correlation and
                self.accuracy >= 0.85 and
                self.auc >= 0.90)


# ============================================================================
# MCTS Guidance
# ============================================================================

@dataclass
class MCTSGuidance:
    """MCTS Parameter Adjustments"""
    c_param: float  # UCB exploration parameter
    max_depth: int  # Max simulation depth
    rollout_strategy: str  # 'RANDOM' or 'CAUSALLY_GUIDED'
    confidence: float
```

### 2.2 Cache Structures

```python
# ============================================================================
# ACI Cache
# ============================================================================

from functools import lru_cache
import hashlib
import pickle

class ACICache:
    """
    Multi-level cache for ACI calculations
    """

    def __init__(self, max_size=1000):
        self.memory_cache = {}  # In-memory cache
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def get_csp_hash(self, csp: CSPInstance) -> str:
        """Generate hash for CSP instance"""
        # Serialize CSP
        csp_str = pickle.dumps({
            'vars': [(v.name, tuple(v.domain)) for v in csp.variables],
            'constraints': [(c.variables, tuple(c.allowed_tuples))
                           for c in csp.constraints]
        })
        return hashlib.md5(csp_str).hexdigest()

    def get(self, csp: CSPInstance) -> ACIResult:
        """Get cached ACI result"""
        key = self.get_csp_hash(csp)
        if key in self.memory_cache:
            self.hit_count += 1
            result = self.memory_cache[key]
            result.cached = True
            return result
        self.miss_count += 1
        return None

    def set(self, csp: CSPInstance, result: ACIResult):
        """Cache ACI result"""
        key = self.get_csp_hash(csp)
        # Evict if at capacity
        if len(self.memory_cache) >= self.max_size:
            # Simple FIFO eviction
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        self.memory_cache[key] = result

    def clear(self):
        """Clear cache"""
        self.memory_cache.clear()
        self.hit_count = 0
        self.miss_count = 0

    def statistics(self):
        """Cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'cache_size': len(self.memory_cache)
        }
```

---

## 3. Implementation Phases

### Phase 1: Core ACI Engine (Weeks 1-2)

**Objectives:**
- Implement basic ACI calculation
- Test on small CSP instances
- Validate component calculations

**Tasks:**

#### Week 1: Foundation
- [ ] Create package structure
- [ ] Implement CSP data structures
- [ ] Implement graph utilities
- [ ] Implement basic entropy calculator
- [ ] Unit tests for entropy

**Deliverables:**
- `gamma1/core/` module structure
- Working entropy engine
- Test suite for entropy

#### Week 2: Components
- [ ] Implement coherence engine
- [ ] Implement solvability engine
- [ ] Implement main ACI calculator
- [ ] Integration tests for ACI
- [ ] Performance benchmarks

**Deliverables:**
- Complete ACI calculation pipeline
- Integration test suite
- Initial performance metrics

---

### Phase 2: Signal Extraction (Weeks 3-4)

**Objectives:**
- Implement signal quality metrics
- Learn optimal thresholds
- Validate on benchmark problems

**Tasks:**

#### Week 3: Signal Metrics
- [ ] Implement signal extractor
- [ ] Implement threshold learner
- [ ] Implement validation metrics
- [ ] Create benchmark dataset

**Deliverables:**
- Signal extraction module
- Benchmark dataset
- Initial validation results

#### Week 4: Validation
- [ ] Run validation on benchmarks
- [ ] Analyze signal quality
- [ ] Tune ACI weights
- [ ] Document findings

**Deliverables:**
- Validation report
- Optimized ACI weights
- Updated documentation

---

### Phase 3: Adaptive Integration (Weeks 5-6)

**Objectives:**
- Integrate with Monte Carlo (Stage 3)
- Integrate with Error Analysis (Stage 6)
- Integrate with Convergence Validation (Stage 9)

**Tasks:**

#### Week 5: Stage Integration
- [ ] Implement strategy generator
- [ ] Implement MCTS guide
- [ ] Implement ACI monitor
- [ ] Integrate with Stage 3 (Monte Carlo)

**Deliverables:**
- Adaptive search module
- Stage 3 integration
- Integration tests

#### Week 6: Full Integration
- [ ] Integrate with Stage 6 (Error Analysis)
- [ ] Integrate with Stage 9 (Convergence)
- [ ] End-to-end testing
- [ ] Performance optimization

**Deliverables:**
- Complete integrated system
- End-to-end test suite
- Performance report

---

### Phase 4: Optimization & Hardening (Weeks 7-8)

**Objectives:**
- Optimize computation speed
- Add caching
- Implement parallel processing
- Final validation

**Tasks:**

#### Week 7: Optimization
- [ ] Implement incremental ACI updates
- [ ] Add multi-level caching
- [ ] Parallelize component calculations
- [ ] Numba JIT compilation

**Deliverables:**
- Optimized ACI engine
- Performance benchmarks
- Optimization report

#### Week 8: Hardening
- [ ] Comprehensive error handling
- [ ] Edge case testing
- [ ] Documentation completion
- [ ] Final validation

**Deliverables:**
- Production-ready Γ₁ system
- Complete documentation
- Final validation report

---

## 4. Integration Strategy

### 4.1 Stage 3 Integration (Monte Carlo Nest)

**Integration Point:** ACI-guided sampling

```python
# rese/gamma1/adaptive/monte_carlo_guide.py

class MonteCarloGuide:
    """
    Use ACI to guide Monte Carlo sampling
    """

    def __init__(self, aci_calculator):
        self.aci_calculator = aci_calculator

    def guide_sampling(self, csp: CSPInstance) -> Dict:
        """
        Generate sampling strategy based on ACI
        """
        aci_result = self.aci_calculator.calculate(csp)
        aci = aci_result.ACI

        if aci > 0.7:
            # High ACI: Smart sampling
            return {
                'strategy': 'SMART',
                'sample_size': 1000,
                'use_heuristics': True,
                'guided_by': 'causal_structure'
            }
        elif aci > 0.4:
            # Medium ACI: Hybrid
            return {
                'strategy': 'HYBRID',
                'sample_size': 5000,
                'use_heuristics': True,
                'guided_by': 'mixed'
            }
        else:
            # Low ACI: Pure random
            return {
                'strategy': 'RANDOM',
                'sample_size': 10000,
                'use_heuristics': False,
                'guided_by': 'none'
            }

    def adapt_during_search(self, csp: CSPInstance,
                           current_assignments: Dict,
                           initial_aci: float) -> Dict:
        """
        Adapt sampling strategy during search
        """
        # Recalculate ACI with current assignments
        current_csp = self._apply_assignments(csp, current_assignments)
        current_aci_result = self.aci_calculator.calculate(current_csp)
        aci_change = current_aci_result.ACI - initial_aci

        if aci_change > 0.05:
            # ACI improving: continue strategy
            return {'action': 'CONTINUE', 'confidence': 'HIGH'}
        elif aci_change < -0.05:
            # ACI declining: change strategy
            return {'action': 'CHANGE_STRATEGY', 'confidence': 'HIGH'}
        else:
            # ACI stable: monitor
            return {'action': 'MONITOR', 'confidence': 'MEDIUM'}

    def _apply_assignments(self, csp: CSPInstance,
                          assignments: Dict) -> CSPInstance:
        """Apply variable assignments to create reduced CSP"""
        # Implementation details...
        pass
```

### 4.2 Stage 6 Integration (Error Source Analysis)

**Integration Point:** ACI-based error diagnosis

```python
# rese/gamma1/adaptive/error_diagnosis.py

class ErrorDiagnosis:
    """
    Use ACI components to diagnose error sources
    """

    def __init__(self, aci_calculator):
        self.aci_calculator = aci_calculator

    def diagnose(self, csp: CSPInstance,
                 errors: List[Dict]) -> Dict:
        """
        Diagnose error sources using ACI
        """
        aci_result = self.aci_calculator.calculate(csp)
        H = aci_result.components['disorder_entropy']
        C = aci_result.components['causal_coherence']
        S = aci_result.components['solvability_index']

        diagnosis = {
            'primary_cause': None,
            'recommendations': [],
            'confidence': 0.0
        }

        # High disorder
        if H > 0.7:
            diagnosis['primary_cause'] = 'HIGH_DISORDER'
            diagnosis['recommendations'].append({
                'action': 'ADD_CONSTRAINTS',
                'reason': 'Reduce search space entropy',
                'priority': 'HIGH'
            })
            diagnosis['confidence'] += 0.4

        # Low coherence
        if C < 0.3:
            if not diagnosis['primary_cause']:
                diagnosis['primary_cause'] = 'LOW_COHERENCE'
            diagnosis['recommendations'].append({
                'action': 'RESTRUCTURE_CONSTRAINTS',
                'reason': 'Improve causal structure',
                'priority': 'HIGH'
            })
            diagnosis['confidence'] += 0.4

        # Low solvability
        if S < 0.3:
            if not diagnosis['primary_cause']:
                diagnosis['primary_cause'] = 'INHERENTLY_DIFFICULT'
            diagnosis['recommendations'].append({
                'action': 'REFORMULATE_PROBLEM',
                'reason': 'Problem near phase transition',
                'priority': 'MEDIUM'
            })
            diagnosis['confidence'] += 0.2

        return diagnosis

    def suggest_reformulation(self, csp: CSPInstance) -> Dict:
        """
        Suggest problem reformulation based on ACI
        """
        aci_result = self.aci_calculator.calculate(csp)

        suggestions = []

        # If constraint graph is disconnected
        if not nx.is_connected(csp.constraint_graph):
            components = list(nx.connected_components(csp.constraint_graph))
            suggestions.append({
                'type': 'DECOMPOSE',
                'description': f'Decompose into {len(components)} independent subproblems',
                'expected_aci_improvement': 0.2
            })

        # If domains are very large
        avg_domain_size = mean([len(v.domain) for v in csp.variables])
        if avg_domain_size > 100:
            suggestions.append({
                'type': 'DOMAIN_REDUCTION',
                'description': 'Apply symmetry breaking or value ordering',
                'expected_aci_improvement': 0.15
            })

        return {
            'reformulation_suggestions': suggestions,
            'expected_improvement': sum([s['expected_aci_improvement']
                                       for s in suggestions])
        }
```

### 4.3 Stage 9 Integration (Convergence Validation)

**Integration Point:** Convergence prediction

```python
# rese/gamma1/adaptive/convergence_predictor.py

class ConvergencePredictor:
    """
    Predict search convergence using ACI
    """

    def __init__(self, aci_calculator):
        self.aci_calculator = aci_calculator
        self.history = []

    def predict(self, csp: CSPInstance,
               search_progress: Dict) -> Dict:
        """
        Predict if search will converge
        """
        aci_result = self.aci_calculator.calculate(csp)
        aci = aci_result.ACI
        progress_rate = search_progress.get('improvement_rate', 0.0)
        steps = search_progress.get('steps', 0)

        prediction = {
            'will_converge': False,
            'confidence': 0.0,
            'expected_remaining_steps': None,
            'recommendation': None
        }

        # High ACI + good progress
        if aci > 0.8 and progress_rate > 0.1:
            prediction['will_converge'] = True
            prediction['confidence'] = 0.9
            prediction['expected_remaining_steps'] = int(steps / progress_rate)
            prediction['recommendation'] = 'CONTINUE'

        # Low ACI + poor progress
        elif aci < 0.3 and progress_rate < 0.01:
            prediction['will_converge'] = False
            prediction['confidence'] = 0.8
            prediction['recommendation'] = 'ABORT_OR_REFORMULATE'

        # Medium ACI or uncertain progress
        else:
            prediction['will_converge'] = 'UNCERTAIN'
            prediction['confidence'] = 0.5
            prediction['recommendation'] = 'CONTINUE_MONITORING'

        return prediction

    def update_history(self, aci: float, progress: float):
        """Update prediction history for learning"""
        self.history.append({
            'aci': aci,
            'progress': progress,
            'timestamp': time.time()
        })

        # Keep only recent history
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

```python
# tests/test_aci.py

import unittest
from rese.gamma1.core import ACICalculator
from rese.gamma1.utils import create_test_csp

class TestACICalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = ACICalculator()

    def test_trivial_csp(self):
        """Test on trivially solvable CSP"""
        csp = create_test_csp(
            n_variables=5,
            domain_size=2,
            n_constraints=0  # No constraints
        )
        result = self.calculator.calculate(csp)

        # Should have high ACI
        self.assertGreater(result.ACI, 0.7)

    def test_over_constrained_csp(self):
        """Test on over-constrained CSP"""
        csp = create_test_csp(
            n_variables=5,
            domain_size=2,
            n_constraints=20  # Many constraints
        )
        result = self.calculator.calculate(csp)

        # Should have low ACI
        self.assertLess(result.ACI, 0.4)

    def test_tree_csp(self):
        """Test on tree-structured CSP (high coherence)"""
        csp = create_tree_csp(n_variables=10, domain_size=3)
        result = self.calculator.calculate(csp)

        # Should have high coherence
        self.assertGreater(result.components['causal_coherence'], 0.6)

    def test_dense_csp(self):
        """Test on dense CSP (low coherence)"""
        csp = create_dense_csp(n_variables=10, domain_size=3)
        result = self.calculator.calculate(csp)

        # Should have lower coherence
        self.assertLess(result.components['causal_coherence'], 0.5)

    def test_aci_bounds(self):
        """Test that ACI is always in [0, 1]"""
        for _ in range(100):
            csp = create_random_csp()
            result = self.calculator.calculate(csp)
            self.assertGreaterEqual(result.ACI, 0.0)
            self.assertLessEqual(result.ACI, 1.0)


class TestEntropyEngine(unittest.TestCase):

    def test_uniform_domain_entropy(self):
        """Test entropy of uniform distribution"""
        from rese.gamma1.core import shannon_entropy

        # Domain of size 8: H = 3 bits
        domain = list(range(8))
        H = shannon_entropy([1/8] * 8)
        self.assertAlmostEqual(H, 3.0, places=5)

    def test_deterministic_domain_entropy(self):
        """Test entropy of deterministic distribution"""
        from rese.gamma1.core import shannon_entropy

        # Single value: H = 0 bits
        H = shannon_entropy([1.0])
        self.assertAlmostEqual(H, 0.0, places=5)
```

### 5.2 Integration Tests

```python
# tests/test_integration.py

import unittest
from rese.gamma1.core import ACICalculator
from rese.gamma1.adaptive import MonteCarloGuide, ErrorDiagnosis
from rese.stage3 import MonteCarloNest  # Assuming exists
from rese.stage6 import ErrorAnalyzer  # Assuming exists

class TestStage3Integration(unittest.TestCase):

    def test_monte_carlo_guidance(self):
        """Test ACI guidance for Monte Carlo"""
        aci_calc = ACICalculator()
        guide = MonteCarloGuide(aci_calc)

        # Create test CSP
        csp = create_test_csp(n_variables=10, domain_size=3)

        # Get guidance
        strategy = guide.guide_sampling(csp)

        # Should return valid strategy
        self.assertIn(strategy['strategy'], ['SMART', 'HYBRID', 'RANDOM'])
        self.assertGreater(strategy['sample_size'], 0)

    def test_adaptive_sampling(self):
        """Test adaptive sampling during search"""
        aci_calc = ACICalculator()
        guide = MonteCarloGuide(aci_calc)

        csp = create_test_csp(n_variables=10, domain_size=3)
        initial_result = aci_calc.calculate(csp)

        # Simulate some assignments
        assignments = {csp.variables[0].name: csp.variables[0].domain[0]}

        # Get adaptation
        adaptation = guide.adapt_during_search(csp, assignments, initial_result.ACI)

        # Should return valid action
        self.assertIn(adaptation['action'],
                     ['CONTINUE', 'CHANGE_STRATEGY', 'MONITOR'])


class TestStage6Integration(unittest.TestCase):

    def test_error_diagnosis(self):
        """Test ACI-based error diagnosis"""
        aci_calc = ACICalculator()
        diagnosis = ErrorDiagnosis(aci_calc)

        # Create problematic CSP
        csp = create_over_constrained_csp()
        errors = [
            {'type': 'backtrack', 'count': 1000},
            {'type': 'timeout', 'time': 60}
        ]

        # Diagnose
        result = diagnosis.diagnose(csp, errors)

        # Should identify issue
        self.assertIsNotNone(result['primary_cause'])
        self.assertGreater(len(result['recommendations']), 0)


class TestStage9Integration(unittest.TestCase):

    def test_convergence_prediction(self):
        """Test ACI-based convergence prediction"""
        aci_calc = ACICalculator()
        predictor = ConvergencePredictor(aci_calc)

        # Create test CSP
        csp = create_test_csp(n_variables=10, domain_size=3)

        # Simulate good progress
        progress = {
            'improvement_rate': 0.15,
            'steps': 100
        }

        # Predict
        prediction = predictor.predict(csp, progress)

        # Should make prediction
        self.assertIn(prediction['will_converge'], [True, False, 'UNCERTAIN'])
        self.assertIsNotNone(prediction['recommendation'])
```

### 5.3 Validation Tests

```python
# tests/test_validation.py

import unittest
from rese.gamma1.signal import SignalExtractor
from rese.gamma1.core import ACICalculator

class TestSignalExtraction(unittest.TestCase):

    def test_signal_quality(self):
        """Test signal extraction quality"""
        # Create benchmark instances
        solvable_instances = [create_easy_csp() for _ in range(50)]
        intractable_instances = [create_hard_csp() for _ in range(50)]

        calculator = ACICalculator()

        # Calculate ACI for all instances
        aci_results = []
        solve_times = []

        for csp in solvable_instances:
            result = calculator.calculate(csp)
            aci_results.append(result)
            solve_times.append(1.0)  # Fast

        for csp in intractable_instances:
            result = calculator.calculate(csp)
            aci_results.append(result)
            solve_times.append(float('inf'))  # Intractable

        # Extract signal
        extractor = SignalExtractor()
        quality = extractor.extract_signal(aci_results, solve_times)

        # Should meet targets
        self.assertGreater(quality['correlation'], 0.7)  # At least 0.7
        self.assertGreater(quality['accuracy'], 0.7)
        self.assertGreater(quality['auc'], 0.8)
```

---

## 6. Deployment Plan

### 6.1 Development Environment

**Setup:**
```bash
# Create virtual environment
python -m venv venv/gamma1
source venv/gamma1/bin/activate  # On Windows: venv\gamma1\Scripts\activate

# Install dependencies
pip install -r requirements-gamma1.txt

# Run tests
pytest tests/ -v

# Run benchmarks
python benchmarks/run_benchmarks.py
```

**requirements-gamma1.txt:**
```
numpy>=1.21.0
networkx>=2.6.0
scipy>=1.7.0
scikit-learn>=1.0.0
numba>=0.56.0  # Optional for JIT
```

### 6.2 Deployment Checklist

**Pre-deployment:**
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Validation targets met (>85% correlation)
- [ ] Documentation complete
- [ ] Performance benchmarks satisfactory
- [ ] Code review completed

**Deployment Steps:**

1. **Stage 1: Deploy to Development**
   - Install in dev environment
   - Run full test suite
   - Performance profiling
   - Fix any issues

2. **Stage 2: Deploy to Staging**
   - Integration testing with full system
   - Load testing
   - Monitor for issues

3. **Stage 3: Deploy to Production**
   - Gradual rollout
   - Continuous monitoring
   - Rollback plan ready

### 6.3 Monitoring

**Metrics to Monitor:**
- ACI calculation time (target: <100ms)
- Cache hit rate (target: >60%)
- Signal quality (correlation, accuracy, AUC)
- Memory usage
- Error rates

**Alerting:**
- ACI calculation time >500ms
- Signal quality <80% target
- Cache hit rate <40%
- Error rate >5%

---

## 7. Resource Allocation

### 7.1 Development Effort

| Phase | Duration | Full-Time Equivalents | Total Effort |
|-------|----------|----------------------|--------------|
| Phase 1: Core Engine | 2 weeks | 1 developer | 80 hours |
| Phase 2: Signal Extraction | 2 weeks | 1 developer | 80 hours |
| Phase 3: Adaptive Integration | 2 weeks | 1 developer | 80 hours |
| Phase 4: Optimization | 2 weeks | 1 developer | 80 hours |
| **Total** | **8 weeks** | **1 developer** | **320 hours** |

### 7.2 Computational Resources

**Development:**
- CPU: 4 cores minimum
- RAM: 8 GB minimum
- Storage: 10 GB

**Testing/Benchmarks:**
- CPU: 8+ cores recommended
- RAM: 16+ GB recommended
- Storage: 50 GB for benchmark datasets

**Production:**
- CPU: Depends on integration
- RAM: 4+ GB per instance
- Storage: Minimal (data is transient)

### 7.3 Timeline

```
Week 1-2:  Phase 1 (Core Engine)
Week 3-4:  Phase 2 (Signal Extraction)
Week 5-6:  Phase 3 (Adaptive Integration)
Week 7-8:  Phase 4 (Optimization & Hardening)
Week 9:    Final Validation & Deployment
```

**Target Completion:** Week 36

---

## 8. Risk Management

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ACI doesn't correlate well | Medium | High | Extensive validation, weight tuning |
| Performance too slow | Low | Medium | Parallelization, caching, JIT |
| Integration issues | Low | Medium | Early integration testing |
| Validation targets not met | Medium | High | Iterative refinement, fallback plans |

### 8.2 Contingency Plans

**If ACI correlation <85%:**
- Increase feature engineering
- Use machine learning for weight optimization
- Consider hybrid approaches (ACI + ML)

**If performance too slow:**
- Implement approximations
- Use sampling for large problems
- Optimize hotspots with Numba

**If integration fails:**
- Simplify integration points
- Create API abstractions
- Extend timeline if needed

---

## 9. Success Criteria

**Minimum Viable Product (MVP):**
- ACI calculation working
- Signal quality >75% correlation
- Basic Stage 3 integration

**Full Success:**
- ACI calculation <100ms
- Signal quality >85% correlation
- Full integration with Stages 3, 6, 9
- Comprehensive documentation
- Validation report

**Stretch Goals:**
- Signal quality >90% correlation
- Real-time ACI monitoring
- Automatic weight learning
- Publication-quality results

---

## 10. Next Steps

1. **Review and approve** this implementation plan
2. **Set up development environment**
3. **Begin Phase 1** (Core Engine)
4. **Weekly progress reports**
5. **Adjust timeline as needed**

---

**Document Status:** Complete
**Next Document:** `gamma1_validation_strategy.md`
**Agent:** D1 (Γ₁ Specialist)
**Date:** 2025-12-31
