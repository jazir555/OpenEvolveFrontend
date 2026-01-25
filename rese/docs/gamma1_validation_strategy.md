# Γ₁ Validation Strategy
**Agent D1 - ACI Analyzer Specialist**

**Date:** 2025-12-31
**Target:** >85% ACI Signal Correlation
**Status:** Validation Planning

---

## Executive Summary

This document defines the comprehensive validation strategy for the Algorithmic Complexity Index (ACI) system, including success metrics, benchmark design, evaluation methodology, and continuous monitoring procedures.

**Primary Target:** >85% correlation between ACI scores and actual solvability

---

## Table of Contents
1. [Success Metrics](#1-success-metrics)
2. [Benchmark Design](#2-benchmark-design)
3. [Evaluation Methodology](#3-evaluation-methodology)
4. [Validation Experiments](#4-validation-experiments)
5. [Statistical Analysis](#5-statistical-analysis)
6. [Continuous Monitoring](#6-continuous-monitoring)
7. [Failure Analysis](#7-failure-analysis)

---

## 1. Success Metrics

### 1.1 Primary Metrics

#### 1.1.1 Correlation Coefficient
```python
def pearson_correlation(aci_scores, solve_times):
    """
    Calculate Pearson correlation between ACI and solve time

    Target: r > 0.85

    Interpretation:
        r > 0.9:  Excellent
        r > 0.8:  Good
        r > 0.7:  Acceptable
        r < 0.7:  Needs improvement
    """
    from scipy.stats import pearsonr
    r, p_value = pearsonr(aci_scores, solve_times)
    return r, p_value
```

#### 1.1.2 Classification Accuracy
```python
def classification_accuracy(aci_scores, actual_solvability, threshold=0.5):
    """
    Calculate binary classification accuracy

    Target: Accuracy > 0.85

    Classes:
        - Solvable: ACI > threshold
        - Intractable: ACI ≤ threshold
    """
    predictions = [1 if aci > threshold else 0 for aci in aci_scores]
    actual = [1 if solvable else 0 for solvable in actual_solvability]

    accuracy = sum(p == a for p, a in zip(predictions, actual)) / len(predictions)

    return accuracy
```

#### 1.1.3 Signal-to-Noise Ratio (SNR)
```python
def signal_to_noise_ratio(solvable_aci, intractable_aci):
    """
    Calculate SNR of ACI signal

    Target: SNR > 3.0

    SNR = (mean_solvable - mean_intractable) / std_combined
    """
    signal = mean(solvable_aci) - mean(intractable_aci)
    noise = (std(solvable_aci) + std(intractable_aci)) / 2

    snr = signal / noise if noise > 0 else float('inf')

    return snr
```

#### 1.1.4 ROC AUC
```python
def roc_auc_score(aci_scores, actual_solvability):
    """
    Calculate Area Under ROC Curve

    Target: AUC > 0.90

    Interpretation:
        AUC = 1.0: Perfect classifier
        AUC > 0.9: Excellent
        AUC > 0.8: Good
        AUC = 0.5: Random guessing
    """
    from sklearn.metrics import roc_auc_score

    actual_binary = [1 if s else 0 for s in actual_solvability]
    auc = roc_auc_score(actual_binary, aci_scores)

    return auc
```

### 1.2 Secondary Metrics

#### 1.2.1 Confidence Calibration
```python
def confidence_calibration(aci_results, actual_performance):
    """
    Check if ACI confidence scores are well-calibrated

    Target: Calibration error < 0.1
    """
    # Bin by confidence
    confidence_bins = {}
    for result, actual in zip(aci_results, actual_performance):
        conf = int(result.confidence * 10) / 10  # Round to 0.1
        if conf not in confidence_bins:
            confidence_bins[conf] = []
        confidence_bins[conf].append(actual)

    # Calculate actual success rate in each bin
    calibration_errors = []
    for conf, outcomes in confidence_bins.items():
        actual_rate = mean(outcomes)
        error = abs(actual_rate - conf)
        calibration_errors.append(error)

    return mean(calibration_errors)
```

#### 1.2.2 Component Agreement
```python
def component_agreement(aci_results):
    """
    Measure how well ACI components agree with each other

    Higher agreement = more reliable ACI
    """
    agreements = []

    for result in aci_results:
        H = result.components['disorder_entropy']
        C = result.components['causal_coherence']
        S = result.components['solvability_index']

        # All oriented: higher = more solvable
        components = [1-H, C, S]
        component_variance = variance(components)

        # Lower variance = higher agreement
        agreement = 1.0 - component_variance
        agreements.append(agreement)

    return mean(agreements)
```

### 1.3 Performance Metrics

#### 1.3.1 Computation Time
```python
def measure_computation_time(aci_calculator, test_instances):
    """
    Measure ACI calculation time

    Target:
        - Median < 100ms
        - 95th percentile < 500ms
        - 99th percentile < 1s
    """
    times = []

    for instance in test_instances:
        start = time.time()
        aci_calculator.calculate(instance)
        end = time.time()

        times.append((end - start) * 1000)  # Convert to ms

    return {
        'median_ms': median(times),
        'p95_ms': percentile(times, 95),
        'p99_ms': percentile(times, 99),
        'max_ms': max(times)
    }
```

#### 1.3.2 Cache Effectiveness
```python
def cache_effectiveness(aci_calculator, repeated_instances):
    """
    Measure cache hit rate

    Target: Hit rate > 60%
    """
    # First pass (cold cache)
    for instance in repeated_instances:
        aci_calculator.calculate(instance)

    stats_before = aci_calculator.cache.statistics()

    # Second pass (warm cache)
    for instance in repeated_instances:
        aci_calculator.calculate(instance)

    stats_after = aci_calculator.cache.statistics()

    return {
        'hit_rate': stats_after['hit_rate'],
        'speedup': stats_before['miss_count'] / stats_after['miss_count']
    }
```

---

## 2. Benchmark Design

### 2.1 Benchmark Categories

#### Category 1: Tractable Problems (High ACI Expected)

**2.1.1 Tree-Structured CSP**
```python
def generate_tree_csp(n_variables=20, domain_size=5):
    """
    Generate tree-structured CSP
    Expected ACI: 0.7 - 0.9
    """
    # Create tree structure
    variables = [Variable(f'v{i}', list(range(domain_size)))
                for i in range(n_variables)]

    # Add constraints forming a tree
    constraints = []
    for i in range(n_variables - 1):
        # Connect v_i to v_{i+1}
        allowed_tuples = generate_random_allowed_tuples(
            domain_size, domain_size, tightness=0.3  # Low tightness
        )
        constraints.append(Constraint(
            variables=[f'v{i}', f'v{i+1}'],
            allowed_tuples=allowed_tuples
        ))

    return CSPInstance(variables=variables, constraints=constraints)
```

**2.1.2 Loose CSP**
```python
def generate_loose_csp(n_variables=15, domain_size=4):
    """
    Generate CSP with loose constraints
    Expected ACI: 0.6 - 0.8
    """
    variables = [Variable(f'v{i}', list(range(domain_size)))
                for i in range(n_variables)]

    # Add few, loose constraints
    constraints = []
    n_constraints = int(n_variables * 0.3)  # Sparse
    for _ in range(n_constraints):
        v1, v2 = random.sample(range(n_variables), 2)
        allowed_tuples = generate_random_allowed_tuples(
            domain_size, domain_size, tightness=0.2  # Very loose
        )
        constraints.append(Constraint(
            variables=[f'v{v1}', f'v{v2}'],
            allowed_tuples=allowed_tuples
        ))

    return CSPInstance(variables=variables, constraints=constraints)
```

#### Category 2: Challenging Problems (Medium ACI Expected)

**2.2.1 Near Phase Transition**
```python
def generate_phase_transition_csp(n_variables=20, domain_size=5):
    """
    Generate CSP near phase transition
    Expected ACI: 0.3 - 0.6
    """
    variables = [Variable(f'v{i}', list(range(domain_size)))
                for i in range(n_variables)]

    # Critical tightness ≈ 0.5, critical density ≈ 0.5
    constraints = []
    n_constraints = int(n_variables * (n_variables - 1) * 0.5 / 2)

    for _ in range(n_constraints):
        v1, v2 = random.sample(range(n_variables), 2)
        allowed_tuples = generate_random_allowed_tuples(
            domain_size, domain_size, tightness=0.5  # Critical
        )
        constraints.append(Constraint(
            variables=[f'v{v1}', f'v{v2}'],
            allowed_tuples=allowed_tuples
        ))

    return CSPInstance(variables=variables, constraints=constraints)
```

**2.2.2 Dense Graph CSP**
```python
def generate_dense_csp(n_variables=15, domain_size=4):
    """
    Generate CSP with dense constraint graph
    Expected ACI: 0.2 - 0.5
    """
    variables = [Variable(f'v{i}', list(range(domain_size)))
                for i in range(n_variables)]

    # Add many constraints (dense)
    constraints = []
    n_constraints = int(n_variables * (n_variables - 1) * 0.7 / 2)

    for _ in range(n_constraints):
        v1, v2 = random.sample(range(n_variables), 2)
        allowed_tuples = generate_random_allowed_tuples(
            domain_size, domain_size, tightness=0.4
        )
        constraints.append(Constraint(
            variables=[f'v{v1}', f'v{v2}'],
            allowed_tuples=allowed_tuples
        ))

    return CSPInstance(variables=variables, constraints=constraints)
```

#### Category 3: Intractable Problems (Low ACI Expected)

**2.3.1 Over-Constrained CSP**
```python
def generate_over_constrained_csp(n_variables=10, domain_size=3):
    """
    Generate over-constrained CSP (likely unsatisfiable)
    Expected ACI: 0.0 - 0.3
    """
    variables = [Variable(f'v{i}', list(range(domain_size)))
                for i in range(n_variables)]

    # Add many tight constraints
    constraints = []
    n_constraints = int(n_variables * (n_variables - 1) * 0.8 / 2)

    for _ in range(n_constraints):
        v1, v2 = random.sample(range(n_variables), 2)
        allowed_tuples = generate_random_allowed_tuples(
            domain_size, domain_size, tightness=0.8  # Very tight
        )
        constraints.append(Constraint(
            variables=[f'v{v1}', f'v{v2}'],
            allowed_tuples=allowed_tuples
        ))

    return CSPInstance(variables=variables, constraints=constraints)
```

**2.3.2 High Entropy CSP**
```python
def generate_high_entropy_csp(n_variables=20, domain_size=10):
    """
    Generate CSP with large domains and random constraints
    Expected ACI: 0.0 - 0.4
    """
    # Large domains = high entropy
    variables = [Variable(f'v{i}', list(range(domain_size)))
                for i in range(n_variables)]

    # Random constraints
    constraints = []
    n_constraints = int(n_variables * (n_variables - 1) * 0.6 / 2)

    for _ in range(n_constraints):
        v1, v2 = random.sample(range(n_variables), 2)
        allowed_tuples = generate_random_allowed_tuples(
            domain_size, domain_size, tightness=0.5
        )
        constraints.append(Constraint(
            variables=[f'v{v1}', f'v{v2}'],
            allowed_tuples=allowed_tuples
        ))

    return CSPInstance(variables=variables, constraints=constraints)
```

### 2.2 Benchmark Suite Composition

**Complete Benchmark Set:**
```python
def create_benchmark_suite():
    """
    Create comprehensive benchmark suite
    """
    benchmarks = {
        'tractable': {
            'tree_structured': [generate_tree_csp(20, 5) for _ in range(20)],
            'loose_constraints': [generate_loose_csp(15, 4) for _ in range(20)],
            'nqueasy': [generate_nqueasy(n) for n in [8, 10, 12, 14] for _ in range(5)]
        },
        'challenging': {
            'phase_transition': [generate_phase_transition_csp(20, 5) for _ in range(30)],
            'dense_graph': [generate_dense_csp(15, 4) for _ in range(30)],
            'graph_coloring': [generate_graph_coloring(n) for n in [10, 15, 20] for _ in range(10)]
        },
        'intractable': {
            'over_constrained': [generate_over_constrained_csp(10, 3) for _ in range(20)],
            'high_entropy': [generate_high_entropy_csp(20, 10) for _ in range(20)],
            'random_unsatisfiable': [generate_random_unsatisfiable_csp() for _ in range(30)]
        }
    }

    return benchmarks
```

### 2.3 Real-World Benchmarks

**Standard CSP Libraries:**
- **CSPLib:** http://www.csplib.org/
  - Problem 1: Quasigroup Completion
  - Problem 3: Graph Coloring
  - Problem 6: Golomb Ruler
  - Problem 10: Job Shop Scheduling

**Integration:**
```python
def load_csplib_problem(problem_number, instance_id):
    """
    Load problem from CSPLib
    """
    from csplib import load_problem

    raw_problem = load_problem(problem_number, instance_id)

    # Convert to our CSP format
    variables = [Variable(name, domain)
                for name, domain in raw_problem.variables.items()]

    constraints = [Constraint(vars, tuples)
                  for vars, tuples in raw_problem.constraints.items()]

    return CSPInstance(variables=variables, constraints=constraints)
```

---

## 3. Evaluation Methodology

### 3.1 Experimental Protocol

#### 3.1.1 Standard Evaluation Procedure

```python
def standard_evaluation(aci_calculator, benchmark_suite, solver):
    """
    Standard evaluation protocol

    Returns:
        dict with all validation metrics
    """
    results = {
        'instances': [],
        'aci_scores': [],
        'solve_times': [],
        'actual_solvability': []
    }

    for category, problems in benchmark_suite.items():
        for problem_type, instances in problems.items():
            for instance in instances:
                # Calculate ACI
                aci_result = aci_calculator.calculate(instance)
                aci_score = aci_result.ACI

                # Attempt to solve
                start_time = time.time()
                try:
                    solution = solver.solve(instance, timeout=300)  # 5 min timeout
                    solve_time = time.time() - start_time
                    solvable = solution is not None
                except TimeoutError:
                    solve_time = float('inf')
                    solvable = False

                # Record results
                results['instances'].append({
                    'category': category,
                    'type': problem_type,
                    'aci': aci_score,
                    'components': aci_result.components,
                    'solve_time': solve_time,
                    'solvable': solvable
                })

                results['aci_scores'].append(aci_score)
                results['solve_times'].append(solve_time)
                results['actual_solvability'].append(solvable)

    # Calculate metrics
    metrics = calculate_validation_metrics(results)

    return {
        'raw_results': results,
        'metrics': metrics
    }

def calculate_validation_metrics(results):
    """
    Calculate all validation metrics
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import roc_auc_score, accuracy_score

    aci_scores = results['aci_scores']
    solve_times = results['solve_times']
    actual_solvability = results['actual_solvability']

    # Convert infinite times to large number for correlation
    measurable_times = [t if t < float('inf') else 1000 for t in solve_times]

    # Primary metrics
    correlation, p_value = pearsonr(aci_scores, measurable_times)

    predictions = [1 if aci > 0.5 else 0 for aci in aci_scores]
    actual_binary = [1 if s else 0 for s in actual_solvability]
    accuracy = accuracy_score(actual_binary, predictions)

    auc = roc_auc_score(actual_binary, aci_scores)

    solvable_aci = [aci for aci, solv in zip(aci_scores, actual_solvability) if solv]
    intractable_aci = [aci for aci, solv in zip(aci_scores, actual_solvability) if not solv]
    snr = signal_to_noise_ratio(solvable_aci, intractable_aci)

    return {
        'correlation': correlation,
        'correlation_p_value': p_value,
        'accuracy': accuracy,
        'auc': auc,
        'snr': snr,
        'mean_solvable_aci': mean(solvable_aci) if solvable_aci else 0,
        'mean_intractable_aci': mean(intractable_aci) if intractable_aci else 0,
        'meets_target': correlation > 0.85 and accuracy > 0.85 and auc > 0.90
    }
```

### 3.2 Cross-Validation

```python
def cross_validate(aci_calculator, benchmark_suite, k_folds=5):
    """
    K-fold cross-validation for robustness

    Tests:
        1. Generalization across different problem types
        2. Sensitivity to training data
        3. Consistency of ACI scores
    """
    all_instances = []
    for category, problems in benchmark_suite.items():
        for problem_type, instances in problems.items():
            for instance in instances:
                all_instances.append((instance, category, problem_type))

    # Shuffle
    random.shuffle(all_instances)

    # Split into K folds
    fold_size = len(all_instances) // k_folds
    folds = [all_instances[i*fold_size:(i+1)*fold_size]
            for i in range(k_folds)]

    cv_results = []

    for i in range(k_folds):
        # Train on all folds except i
        train_folds = [folds[j] for j in range(k_folds) if j != i]
        test_fold = folds[i]

        # Learn ACI weights from training set
        # (In practice, this might involve adjusting α, β, γ)
        weights = learn_aci_weights(train_folds)
        aci_calculator.set_weights(weights)

        # Evaluate on test fold
        fold_results = []
        for instance, category, problem_type in test_fold:
            aci_result = aci_calculator.calculate(instance)
            fold_results.append({
                'instance': instance,
                'category': category,
                'problem_type': problem_type,
                'aci': aci_result.ACI
            })

        cv_results.append({
            'fold': i,
            'results': fold_results,
            'weights': weights
        })

    # Analyze consistency across folds
    aci_variations = []
    for i in range(k_folds):
        for j in range(i+1, k_folds):
            # Compare ACI scores on same instances
            fold_i_acis = [r['aci'] for r in cv_results[i]['results']]
            fold_j_acis = [r['aci'] for r in cv_results[j]['results']]
            variation = mean([abs(a - b) for a, b in zip(fold_i_acis, fold_j_acis)])
            aci_variations.append(variation)

    return {
        'cv_results': cv_results,
        'mean_variation': mean(aci_variations),
        'max_variation': max(aci_variations),
        'is_consistent': mean(aci_variations) < 0.1  # Less than 0.1 variation
    }
```

### 3.3 Ablation Studies

```python
def ablation_study(aci_calculator, benchmark_suite):
    """
    Ablation study: test importance of each ACI component

    Tests:
        1. ACI with only H (disorder entropy)
        2. ACI with only C (causal coherence)
        3. ACI with only S (solvability index)
        4. Full ACI (H + C + S)
    """
    results = {}

    # Full ACI
    full_results = standard_evaluation(aci_calculator, benchmark_suite, solver)
    results['full_aci'] = full_results['metrics']

    # Ablation 1: Only H
    aci_calculator.set_weights({'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0})
    h_only_results = standard_evaluation(aci_calculator, benchmark_suite, solver)
    results['h_only'] = h_only_results['metrics']

    # Ablation 2: Only C
    aci_calculator.set_weights({'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0})
    c_only_results = standard_evaluation(aci_calculator, benchmark_suite, solver)
    results['c_only'] = c_only_results['metrics']

    # Ablation 3: Only S
    aci_calculator.set_weights({'alpha': 0.0, 'beta': 0.0, 'gamma': 1.0})
    s_only_results = standard_evaluation(aci_calculator, benchmark_suite, solver)
    results['s_only'] = s_only_results['metrics']

    # Reset to full ACI
    aci_calculator.reset_weights()

    return results
```

---

## 4. Validation Experiments

### 4.1 Experiment 1: Baseline Validation

**Objective:** Establish baseline performance

**Procedure:**
1. Generate benchmark suite (200 instances)
2. Calculate ACI for all instances
3. Solve all instances with baseline solver
4. Calculate metrics

**Success Criteria:**
- Correlation > 0.75 (initial target)
- Accuracy > 0.75
- AUC > 0.80

**Duration:** 1 week

### 4.2 Experiment 2: Weight Optimization

**Objective:** Optimize ACI weights (α, β, γ)

**Procedure:**
1. Split benchmark: 150 train, 50 test
2. Grid search over weight space:
   - α from 0.0 to 1.0 step 0.1
   - β from 0.0 to (1-α) step 0.1
   - γ = 1 - α - β
3. Train on training set
4. Evaluate on test set
5. Select weights with best test correlation

**Success Criteria:**
- Correlation > 0.85
- Improved over baseline

**Duration:** 1 week

```python
def grid_search_weights(benchmark_suite):
    """
    Grid search for optimal ACI weights
    """
    # Split data
    train_set, test_set = split_benchmark(benchmark_suite, ratio=0.75)

    best_weights = None
    best_correlation = -1

    results = []

    for alpha in [i/10 for i in range(0, 11)]:
        for beta in [j/10 for j in range(0, 11 - int(alpha*10))]:
            gamma = 1.0 - alpha - beta

            weights = {'alpha': alpha, 'beta': beta, 'gamma': gamma}

            # Evaluate
            metrics = evaluate_with_weights(weights, test_set)
            correlation = metrics['correlation']

            results.append({
                'weights': weights,
                'correlation': correlation,
                'accuracy': metrics['accuracy'],
                'auc': metrics['auc']
            })

            if correlation > best_correlation:
                best_correlation = correlation
                best_weights = weights

    return best_weights, results
```

### 4.3 Experiment 3: Stress Testing

**Objective:** Test ACI on extreme cases

**Test Cases:**
1. Very large problems (n > 100 variables)
2. Very large domains (domain_size > 1000)
3. Highly asymmetric problems
4. Degenerate cases (0 constraints, all variables constrained, etc.)

**Success Criteria:**
- No crashes or errors
- ACI remains in [0, 1]
- Reasonable computation time (<10s)

**Duration:** 3 days

```python
def stress_test(aci_calculator):
    """
    Stress test ACI calculator
    """
    test_cases = {
        'very_large': generate_large_csp(n_variables=200, domain_size=5),
        'huge_domains': generate_large_csp(n_variables=20, domain_size=10000),
        'asymmetric': generate_asymmetric_csp(),
        'no_constraints': generate_no_constraints_csp(),
        'all_constrained': generate_fully_connected_csp(),
        'single_variable': generate_single_variable_csp()
    }

    results = {}

    for test_name, csp in test_cases.items():
        try:
            start = time.time()
            aci_result = aci_calculator.calculate(csp)
            elapsed = time.time() - start

            results[test_name] = {
                'status': 'SUCCESS',
                'aci': aci_result.ACI,
                'in_bounds': 0 <= aci_result.ACI <= 1,
                'time': elapsed
            }
        except Exception as e:
            results[test_name] = {
                'status': 'ERROR',
                'error': str(e)
            }

    return results
```

### 4.4 Experiment 4: Real-World Validation

**Objective:** Validate on real-world problems

**Data Sources:**
- CSPLib problems
- SAT competition instances
- Industry case studies

**Success Criteria:**
- Correlation > 0.80 on real-world data
- No significant degradation from synthetic benchmarks

**Duration:** 1 week

### 4.5 Experiment 5: Longitudinal Validation

**Objective:** Track ACI performance over time

**Procedure:**
1. Collect weekly metrics on new instances
2. Monitor for drift
3. Retrain weights if necessary

**Success Criteria:**
- Stable performance over 8 weeks
- No significant performance degradation

**Duration:** Ongoing

---

## 5. Statistical Analysis

### 5.1 Significance Testing

```python
def statistical_significance(aci_scores, solve_times, n_bootstrap=1000):
    """
    Test statistical significance of ACI correlation

    Uses bootstrap to estimate confidence intervals
    """
    from scipy.stats import pearsonr
    import numpy as np

    # Observed correlation
    r_observed, _ = pearsonr(aci_scores, solve_times)

    # Bootstrap
    bootstrap_correlations = []
    n = len(aci_scores)

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        boot_aci = [aci_scores[i] for i in indices]
        boot_time = [solve_times[i] for i in indices]

        r_boot, _ = pearsonr(boot_aci, boot_time)
        bootstrap_correlations.append(r_boot)

    # Calculate confidence interval
    ci_lower = percentile(bootstrap_correlations, 2.5)
    ci_upper = percentile(bootstrap_correlations, 97.5)

    # Test if significantly different from 0
    p_value = sum([abs(r) > abs(r_observed) for r in bootstrap_correlations]) / n_bootstrap

    return {
        'correlation': r_observed,
        'confidence_interval': (ci_lower, ci_upper),
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### 5.2 Effect Size Analysis

```python
def cohens_d(solvable_aci, intractable_aci):
    """
    Calculate Cohen's d effect size

    Interpretation:
        d < 0.2: Small effect
        d < 0.5: Medium effect
        d < 0.8: Large effect
        d >= 0.8: Very large effect
    """
    mean_diff = mean(solvable_aci) - mean(intractable_aci)
    pooled_std = sqrt(
        ((len(solvable_aci) - 1) * std(solvable_aci)**2 +
         (len(intractable_aci) - 1) * std(intractable_aci)**2) /
        (len(solvable_aci) + len(intractable_aci) - 2)
    )

    d = mean_diff / pooled_std

    return {
        'cohens_d': d,
        'effect_size': 'VERY_LARGE' if d >= 0.8 else
                       'LARGE' if d >= 0.5 else
                       'MEDIUM' if d >= 0.2 else
                       'SMALL'
    }
```

### 5.3 Calibration Analysis

```python
def reliability_diagram(aci_results, actual_solvability, n_bins=10):
    """
    Generate reliability diagram for ACI confidence calibration

    Plots:
        - X-axis: Predicted ACI (binned)
        - Y-axis: Actual solvability rate
        - Well-calibrated: Points follow diagonal
    """
    bins = [(i/n_bins, (i+1)/n_bins) for i in range(n_bins)]

    calibration_points = []

    for bin_low, bin_high in bins:
        # Find instances in this bin
        in_bin = [(r, s) for r, s in zip(aci_results, actual_solvability)
                 if bin_low <= r.ACI < bin_high]

        if in_bin:
            # Calculate actual solvability rate
            actual_rate = mean([s for _, s in in_bin])
            predicted_aci = (bin_low + bin_high) / 2

            calibration_points.append({
                'predicted_aci': predicted_aci,
                'actual_rate': actual_rate,
                'count': len(in_bin)
            })

    return calibration_points
```

---

## 6. Continuous Monitoring

### 6.1 Production Monitoring

```python
class ACIMonitor:
    """
    Monitor ACI performance in production
    """

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.history = []

    def record_prediction(self, aci_result, actual_outcome):
        """
        Record ACI prediction and actual outcome

        actual_outcome: dict with {
            'solvable': bool,
            'solve_time': float
        }
        """
        self.history.append({
            'timestamp': time.time(),
            'aci': aci_result.ACI,
            'components': aci_result.components,
            'confidence': aci_result.confidence,
            'actual_solvable': actual_outcome['solvable'],
            'actual_solve_time': actual_outcome['solve_time']
        })

        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def get_current_metrics(self):
        """
        Calculate current performance metrics
        """
        if len(self.history) < 100:
            return {'status': 'INSUFFICIENT_DATA'}

        aci_scores = [h['aci'] for h in self.history]
        solvable = [h['actual_solvable'] for h in self.history]
        solve_times = [h['actual_solve_time'] for h in self.history]

        # Calculate metrics
        correlation, _ = pearsonr(aci_scores, solve_times)

        predictions = [1 if aci > 0.5 else 0 for aci in aci_scores]
        actual = [1 if s else 0 for s in solvable]
        accuracy = sum(p == a for p, a in zip(predictions, actual)) / len(predictions)

        return {
            'correlation': correlation,
            'accuracy': accuracy,
            'sample_size': len(self.history),
            'meets_target': correlation > 0.85 and accuracy > 0.85
        }

    def check_drift(self, baseline_metrics):
        """
        Check for performance drift

        Alerts if:
            - Correlation drops by >10%
            - Accuracy drops by >10%
        """
        current = self.get_current_metrics()

        if current['status'] == 'INSUFFICIENT_DATA':
            return {'drift_detected': False, 'reason': 'Insufficient data'}

        drift_detected = False
        reasons = []

        if current['correlation'] < baseline_metrics['correlation'] * 0.9:
            drift_detected = True
            reasons.append('Correlation dropped by >10%')

        if current['accuracy'] < baseline_metrics['accuracy'] * 0.9:
            drift_detected = True
            reasons.append('Accuracy dropped by >10%')

        return {
            'drift_detected': drift_detected,
            'reasons': reasons
        }
```

### 6.2 Alerting System

```python
class ACIAlertSystem:
    """
    Alert system for ACI performance issues
    """

    def __init__(self, monitor):
        self.monitor = monitor
        self.alert_thresholds = {
            'min_correlation': 0.75,
            'min_accuracy': 0.75,
            'max_computation_time': 1.0,  # seconds
            'min_cache_hit_rate': 0.4
        }

    def check_alerts(self):
        """
        Check if any thresholds are breached
        """
        metrics = self.monitor.get_current_metrics()
        alerts = []

        if metrics['correlation'] < self.alert_thresholds['min_correlation']:
            alerts.append({
                'severity': 'HIGH',
                'type': 'LOW_CORRELATION',
                'message': f"Correlation {metrics['correlation']:.2f} below threshold {self.alert_thresholds['min_correlation']}"
            })

        if metrics['accuracy'] < self.alert_thresholds['min_accuracy']:
            alerts.append({
                'severity': 'HIGH',
                'type': 'LOW_ACCURACY',
                'message': f"Accuracy {metrics['accuracy']:.2f} below threshold {self.alert_thresholds['min_accuracy']}"
            })

        return alerts
```

---

## 7. Failure Analysis

### 7.1 Error Categorization

```python
def analyze_failures(aci_results, actual_solvability):
    """
    Analyze cases where ACI prediction was wrong

    Categories:
        1. False Positives: ACI high, but intractable
        2. False Negatives: ACI low, but solvable
    """
    errors = {
        'false_positives': [],
        'false_negatives': []
    }

    for result, actual in zip(aci_results, actual_solvability):
        predicted = result.ACI > 0.5

        if predicted and not actual:
            errors['false_positives'].append({
                'aci': result.ACI,
                'components': result.components,
                'confidence': result.confidence
            })
        elif not predicted and actual:
            errors['false_negatives'].append({
                'aci': result.ACI,
                'components': result.components,
                'confidence': result.confidence
            })

    return errors
```

### 7.2 Error Diagnosis

```python
def diagnose_error_patterns(errors):
    """
    Find patterns in ACI errors
    """
    fp = errors['false_positives']
    fn = errors['false_negatives']

    diagnosis = {}

    # False Positive Analysis
    if fp:
        # Check if high entropy cases
        fp_entropy = mean([e['components']['disorder_entropy'] for e in fp])

        # Check if low coherence cases
        fp_coherence = mean([e['components']['causal_coherence'] for e in fp])

        diagnosis['false_positive_cause'] = {
            'high_entropy': fp_entropy > 0.7,
            'low_coherence': fp_coherence < 0.3,
            'avg_entropy': fp_entropy,
            'avg_coherence': fp_coherence
        }

    # False Negative Analysis
    if fn:
        fn_entropy = mean([e['components']['disorder_entropy'] for e in fn])
        fn_coherence = mean([e['components']['causal_coherence'] for e in fn])

        diagnosis['false_negative_cause'] = {
            'low_entropy': fn_entropy < 0.3,
            'high_coherence': fn_coherence > 0.7,
            'avg_entropy': fn_entropy,
            'avg_coherence': fn_coherence
        }

    return diagnosis
```

### 7.3 Remediation Strategies

```python
def suggest_remediations(error_diagnosis):
    """
    Suggest improvements based on error analysis
    """
    remediations = []

    if error_diagnosis.get('false_positive_cause', {}).get('high_entropy'):
        remediations.append({
            'priority': 'HIGH',
            'action': 'INCREASE_ENTROPY_WEIGHT',
            'reason': 'High entropy causes false positives'
        })

    if error_diagnosis.get('false_negative_cause', {}).get('high_coherence'):
        remediations.append({
            'priority': 'MEDIUM',
            'action': 'DECREASE_COHERENCE_WEIGHT',
            'reason': 'Over-reliance on coherence causes false negatives'
        })

    return remediations
```

---

## 8. Validation Report Template

```python
def generate_validation_report(validation_results):
    """
    Generate comprehensive validation report
    """
    report = {
        'summary': {
            'target_correlation': 0.85,
            'actual_correlation': validation_results['metrics']['correlation'],
            'target_accuracy': 0.85,
            'actual_accuracy': validation_results['metrics']['accuracy'],
            'target_auc': 0.90,
            'actual_auc': validation_results['metrics']['auc'],
            'meets_target': validation_results['metrics']['meets_target']
        },

        'detailed_metrics': validation_results['metrics'],

        'category_breakdown': {
            'tractable': {
                'count': len([r for r in validation_results['raw_results']
                             if r['category'] == 'tractable']),
                'mean_aci': mean([r['aci'] for r in validation_results['raw_results']
                                 if r['category'] == 'tractable'])
            },
            'challenging': {...},
            'intractable': {...}
        },

        'recommendations': [],

        'next_steps': []
    }

    # Add recommendations based on performance
    if not report['summary']['meets_target']:
        report['recommendations'].append({
            'priority': 'HIGH',
            'action': 'WEIGHT_OPTIMIZATION',
            'details': 'ACI does not meet target. Perform weight optimization.'
        })

    return report
```

---

## 9. Summary

This validation strategy provides:

1. **Comprehensive Metrics:** Correlation, accuracy, SNR, AUC
2. **Benchmark Suite:** Tractable, challenging, and intractable problems
3. **Rigorous Methodology:** Cross-validation, ablation studies, significance testing
4. **Continuous Monitoring:** Production monitoring and alerting
5. **Failure Analysis:** Detailed error categorization and remediation

**Validation Timeline:**
- Week 1: Baseline validation
- Week 2: Weight optimization
- Week 3: Stress testing
- Week 4: Real-world validation
- Week 5+: Continuous monitoring

**Success Criteria:**
- Correlation > 0.85
- Accuracy > 0.85
- AUC > 0.90
- Stable over time

---

**Document Status:** Complete
**All Γ₁ Documents:** Complete (4/4)
**Agent:** D1 (Γ₁ Specialist)
**Date:** 2025-12-31
