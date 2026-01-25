# Convergence Control Research Document
**Agent:** D3 (N_max Specialist)
**Date:** 2025-12-31
**Status:** Research Phase Complete
**Target:** Adaptive Convergence Control for Monte Carlo Refinement

---

## Executive Summary

This document provides comprehensive research on convergence control systems for Monte Carlo Tree Search (MCTS) in the context of constraint satisfaction problems. It covers statistical convergence tests, adaptive stopping criteria, N_max estimation, and integration with Γ₁ ACI prediction.

---

## Table of Contents

1. [Convergence Criteria](#1-convergence-criteria)
2. [Adaptive Stopping Strategies](#2-adaptive-stopping-strategies)
3. [N_max Estimation](#3-n_max-estimation)
4. [Γ₁ Integration](#4-γ₁-integration)
5. [Implementation Design](#5-implementation-design)
6. [Validation Strategy](#6-validation-strategy)
7. [References](#7-references)

---

## 1. Convergence Criteria

### 1.1 Statistical Convergence Tests

#### 1.1.1 Moving Window Variance
**Principle:** Monitor variance in a sliding window of recent iterations.

**Method:**
- Maintain window of last W iterations
- Calculate variance: σ² = Var({x_{n-W+1}, ..., x_n})
- Converged if: σ² < threshold

**Advantages:**
- Simple to implement
- Computationally efficient
- Responsive to changes

**Disadvantages:**
- Sensitive to window size
- May oscillate near threshold

**Optimal Window Size:**
- W = 20-50 for fast convergence
- W = 50-100 for stable problems
- Adaptive: W = max(20, n/10) where n = current iteration

**Threshold Selection:**
- Absolute: threshold = 0.001-0.01
- Relative: threshold = μ × 0.01 (1% of mean)
- Adaptive: threshold = μ × (1 - ACI) × 0.05

#### 1.1.2 Gradient-Based Convergence
**Principle:** Monitor rate of improvement over iterations.

**Method:**
- Calculate gradient: g_n = |x_n - x_{n-1}|
- Average over window: ḡ = mean({g_{n-W+1}, ..., g_n})
- Converged if: ḡ < threshold

**Advantages:**
- Directly measures improvement
- Good for optimization problems
- Early detection of plateau

**Disadvantages:**
- May miss oscillatory convergence
- Sensitive to noise

**Threshold Selection:**
- Absolute: threshold = 0.001
- Relative: threshold = |x_n| × 0.001
- Adaptive: threshold = |x_n| × (1 - ACI) × 0.01

#### 1.1.3 Gelman-Rubin Diagnostic (R-hat)
**Principle:** Compare within-chain and between-chain variance.

**Method:**
- Run M chains in parallel
- Calculate:
  - Within-chain variance: W = Var(Chains)
  - Between-chain variance: B = Var(Mean(Chains))
  - R-hat = sqrt((W + B/M) / W)
- Converged if: R-hat < 1.1 (or 1.05 for strict)

**Advantages:**
- Gold standard for MCMC convergence
- Detects mode collapse
- Theoretically grounded

**Disadvantages:**
- Requires multiple chains
- Computationally expensive
- May be conservative for MCTS

**Adaptation for MCTS:**
- Use multiple parallel MCTS workers
- Calculate R-hat on value estimates
- Combine with other methods

#### 1.1.4 Statistical Process Control (SPC)
**Principle:** Control charts with 3-sigma limits.

**Method:**
- Calculate mean μ and std σ of recent window
- Control limits: [μ - 3σ, μ + 3σ]
- Converged if: All points in window within limits

**Advantages:**
- Industry-standard method
- Detects outliers
- Visual and interpretable

**Disadvantages:**
- Assumes normality
- Conservative (may overestimate)

**Variants:**
- CUSUM (Cumulative Sum): Detects small shifts
- EWMA (Exponentially Weighted Moving Average): Smooths noise

#### 1.1.5 ACI Stabilization
**Principle:** Monitor stability of ACI estimates during search.

**Method:**
- Track ACI at each iteration
- Calculate variance: Var(ACI_{n-W+1}, ..., ACI_n)
- Converged if: ACI variance < threshold

**Advantages:**
- Unique to RESE framework
- Integrates with Γ₁
- Predictive of solution quality

**Disadvantages:**
- Requires ACI computation
- May stabilize prematurely

**Threshold:**
- ACI variance < 0.01 for high stability
- ACI variance < 0.05 for moderate stability

---

### 1.2 Solution Stability

#### 1.2.1 Best Solution Stability
**Principle:** Monitor if best solution changes.

**Method:**
- Track best solution over iterations
- Count iterations since last improvement
- Converged if: No improvement for K iterations

**Parameters:**
- K = 50 for fast convergence
- K = 100-200 for thorough search
- Adaptive: K = W × 2 (twice window size)

#### 1.2.2 Solution Frequency
**Principle:** Monitor frequency of best solution in samples.

**Method:**
- Maintain histogram of visited solutions
- Calculate frequency of best solution
- Converged if: freq(best) > threshold (e.g., > 0.8)

**Advantages:**
- Direct measure of concentration
- Robust to noise

---

### 1.3 Diminishing Returns

#### 1.3.1 Improvement Rate
**Principle:** Monitor rate of improvement.

**Method:**
- Calculate improvement ratio: r_n = (x_n - x_{n-W}) / x_{n-W}
- Converged if: r_n < threshold

**Threshold:**
- r_n < 0.01 for 1% improvement
- r_n < 0.001 for 0.1% improvement

#### 1.3.2 Cost-Benefit Analysis
**Principle:** Stop when marginal cost exceeds marginal benefit.

**Method:**
- Estimate cost per iteration: C (time)
- Estimate benefit per iteration: Δx
- Converged if: C / Δx > threshold

**Adaptive:**
- threshold = (1 - ACI) × cost_tolerance

---

## 2. Adaptive Stopping Strategies

### 2.1 Sequential Analysis

#### 2.1.1 Sequential Probability Ratio Test (SPRT)
**Principle:** Test hypotheses sequentially as data arrives.

**Method:**
- H0: No convergence (variance > threshold)
- H1: Convergence (variance < threshold)
- Calculate cumulative log-likelihood ratio
- Stop if: LLR > upper_bound (accept H1) or LLR < lower_bound (accept H0)

**Advantages:**
- Optimal for sequential testing
- Minimizes expected samples
- Theoretically grounded

**Disadvantages:**
- Requires likelihood model
- Complex to implement

**Simplified Version:**
- Monitor CI width sequentially
- Stop when CI < target width

#### 2.1.2 Confidence Interval Sequential
**Principle:** Stop when confidence interval is sufficiently narrow.

**Method:**
- Calculate CI after every K iterations
- Stop if: CI_width < target_width
- Otherwise: Continue

**Advantages:**
- Simple to implement
- Directly interpretable
- Works with bootstrap

**Parameters:**
- Check interval: K = 50-100 iterations
- Target width: 0.01-0.05
- Confidence level: 95%

### 2.2 Bayesian Stopping

#### 2.2.1 Bayesian Sequential Analysis
**Principle:** Update posterior distribution and stop when posterior is concentrated.

**Method:**
- Prior: p(θ) (distribution over true value)
- Likelihood: p(data|θ)
- Posterior: p(θ|data) ∝ p(data|θ) × p(θ)
- Stop if: Credible interval width < target

**Advantages:**
- Natural sequential updating
- Incorporates prior knowledge
- Principled uncertainty

**Disadvantages:**
- Requires conjugate priors or MCMC
- Computationally expensive

**Approximation:**
- Use normal approximation for posterior
- Update mean and variance incrementally

#### 2.2.2 Value of Information (VOI)
**Principle:** Stop when expected value of additional information < cost.

**Method:**
- Estimate posterior predictive distribution
- Calculate expected reduction in uncertainty
- Stop if: E[reduction] × value < cost_per_iteration

**Advantages:**
- Economically optimal
- Incorporates costs
- Decision-theoretic

**Disadvantages:**
- Complex to compute
- Requires cost model

---

### 2.3 Multi-Criteria Stopping

#### 2.3.1 Composite Stopping Rule
**Principle:** Combine multiple convergence signals.

**Method:**
- Define criteria: C1, C2, ..., Cm
- Calculate weighted score: S = Σ w_i × I(C_i satisfied)
- Stop if: S > threshold

**Weights:**
- ACI-based: w_i ∝ ACI_component
- Adaptive: Update weights based on performance

**Aggregation:**
- ANY: Stop if any criterion met (fast)
- ALL: Stop if all criteria met (thorough)
- MAJORITY: Stop if majority met (balanced)
- WEIGHTED: Stop if weighted sum > threshold

#### 2.3.2 Hierarchical Stopping
**Principle:** Check cheaper tests first, expensive tests later.

**Hierarchy:**
1. Level 1: Fast checks (iteration count, time)
2. Level 2: Moderate checks (moving window, gradient)
3. Level 3: Expensive checks (R-hat, bootstrap)

**Flow:**
```
if max_iterations or max_time:
    stop
elif ACI < 0.3 and no_improvement:
    stop (early stopping)
else:
    if level_2_checks_pass:
        if level_3_checks_pass:
            stop
```

---

## 3. N_max Estimation

### 3.1 Initial Estimation

#### 3.1.1 Based on ACI
**Principle:** Use ACI to predict required iterations.

**Formula:**
```
N_max = base_N × f(ACI)
```

**Function:**
- Linear: f(ACI) = 2 - ACI (high ACI → low N)
- Exponential: f(ACI) = exp(-2 × ACI)
- Step-wise:
  - ACI > 0.8: N = 100
  - 0.6 < ACI ≤ 0.8: N = 500
  - 0.4 < ACI ≤ 0.6: N = 1000
  - ACI ≤ 0.4: N = 5000

**Base Iterations:**
- Small problems (n < 50): base_N = 100
- Medium problems (50 ≤ n < 200): base_N = 500
- Large problems (n ≥ 200): base_N = 1000

#### 3.1.2 Based on Problem Structure
**Factors:**
- Number of variables: n
- Domain size: d
- Constraint density: ρ
- Graph structure: treewidth, cycles

**Heuristic:**
```
complexity = n × log(d) × (1 + treewidth) × (1 + ρ)
N_max = base_N × complexity / 1000
```

**Capped:**
- min_N = 100
- max_N = 10000
- N_max = clip(N_max, min_N, max_N)

### 3.2 Dynamic Adjustment

#### 3.2.1 Progress-Based Adjustment
**Principle:** Adjust N_max based on observed progress.

**Method:**
- Estimate expected progress per iteration: p̂
- Estimate remaining progress: P_remain = target - current
- Adjusted N_max: N_adj = current + P_remain / p̂

**Progress Estimation:**
- Linear: p̂ = mean(recent improvements)
- Exponential decay: p̂_t = p̂_0 × exp(-t/τ)
- ACI-adjusted: p̂ = p̂_base × ACI

#### 3.2.2 Variance-Based Adjustment
**Principle:** Increase N_max if variance is high.

**Method:**
- Estimate sample variance: s²
- Target variance: σ²_target
- Required samples: n = (s² / σ²_target) × z²
- Adjusted N_max: N_adj = current + n

**Where:**
- z = z-score for confidence level (e.g., 1.96 for 95%)

#### 3.2.3 ACI-Based Adjustment
**Principle:** Monitor ACI evolution and adjust accordingly.

**Method:**
- Track ACI trajectory: ACI_0, ACI_1, ..., ACI_t
- Estimate ACI rate of change: dACI/dt
- If ACI improving: Extend N_max
- If ACI stable: Maintain N_max
- If ACI degrading: Reduce N_max

**Rules:**
```python
if dACI_dt > 0.01:
    N_max = N_max × 1.5  # Improving, extend
elif dACI_dt < -0.01:
    N_max = N_max × 0.8  # Degrading, reduce
else:
    pass  # Stable, maintain
```

### 3.3 Early Stopping

#### 3.3.1 Low ACI Early Stop
**Principle:** Stop early if ACI indicates intractability.

**Condition:**
- ACI < 0.3 AND no improvement for K iterations

**Rationale:**
- Low ACI → likely intractable
- No improvement → unlikely to find solution
- Save computational resources

#### 3.3.2 Diminishing Returns Early Stop
**Principle:** Stop when marginal improvement is too small.

**Condition:**
- Improvement rate < threshold AND CI width acceptable

**Threshold:**
- Absolute: improvement < 0.001
- Relative: improvement < 0.01 × current_value

#### 3.3.3 Convergence Early Stop
**Principle:** Stop when convergence detected.

**Condition:**
- Multiple convergence criteria satisfied

**Requirements:**
- At least 2 of: variance, gradient, SPC converged
- Minimum iterations: min_iterations

---

## 4. Γ₁ Integration

### 4.1 ACI Prediction of Convergence

#### 4.1.1 ACI-Convergence Relationship
**Hypothesis:** Higher ACI → Faster convergence

**Mechanisms:**
1. **Low Entropy:** Structured search space → faster exploration
2. **High Coherence:** Clear dependencies → efficient propagation
3. **High Solvability:** Solution exists → reachable

**Empirical Model:**
```
expected_iterations = base_N × (1 - ACI)^α
```

**Parameters:**
- α ∈ [1, 2] (estimated from data)
- Higher α → stronger ACI effect

#### 4.1.2 ACI Component Analysis
**Disorder Entropy (H):**
- High H → slow convergence
- Expect: N_max ∝ H

**Causal Coherence (C):**
- High C → fast convergence
- Expect: N_max ∝ 1/C

**Solvability (S):**
- High S → fast convergence
- Expect: N_max ∝ 1/S

**Combined:**
```
N_max = base_N × H / (C + S)
```

### 4.2 Real-Time ACI Monitoring

#### 4.2.1 ACI Computation During Search
**When to Compute:**
- At root: Initial ACI
- At expansion: Node ACI
- Periodically: Every K iterations

**Optimization:**
- Cache ACI for repeated states
- Incremental updates for local changes
- Approximate ACI for large subtrees

#### 4.2.2 ACI Trajectory Analysis
**Track:**
- ACI at current node: ACI_current
- Best node ACI: ACI_best
- Average ACI: ACI_avg
- ACI trend: dACI/dt

**Use for Convergence:**
- Stabilizing ACI → search settling
- Improving ACI → finding better regions
- Degrading ACI → entering chaotic region

**Stopping Rules:**
```python
if ACI_stable() AND solution_stable():
    stop()

if ACI_degrading() AND no_improvement():
    stop()
```

### 4.3 ACI-Guided N_max Adjustment

#### 4.3.1 Predictive Adjustment
**Before Search:**
```python
N_max = estimate_from_aci(ACI_initial)
```

**During Search:**
```python
ACI_current = compute_aci(current_node)
N_max = adjust_n_max(N_max, ACI_initial, ACI_current)
```

**Adjustment Formula:**
```python
def adjust_n_max(N_max, ACI_init, ACI_curr):
    ratio = ACI_curr / ACI_init
    if ratio > 1.1:  # Improved
        return N_max * 0.8  # Can reduce
    elif ratio < 0.9:  # Degrading
        return N_max * 1.2  # Need more
    else:
        return N_max  # Maintain
```

#### 4.3.2 Component-Specific Adjustment
**Based on which components changed:**
- Entropy decreasing → Reduce N_max
- Coherence improving → Reduce N_max
- Solvability increasing → Reduce N_max

**Formula:**
```python
delta_H = H_curr - H_init
delta_C = C_curr - C_init
delta_S = S_curr - S_init

adjustment = 1.0 - 0.3*delta_H + 0.3*delta_C + 0.4*delta_S
N_max = N_max * clip(adjustment, 0.5, 2.0)
```

---

## 5. Implementation Design

### 5.1 Architecture

```
ConvergenceController
├── ConvergenceDetectors
│   ├── ACIStabilityDetector
│   ├── SolutionStabilityDetector
│   ├── VarianceDetector
│   ├── GradientDetector
│   └── GelmanRubinDetector
├── NMaxEstimator
│   ├── ACIBasedEstimator
│   ├── StructuralEstimator
│   └── DynamicAdjuster
├── AdaptiveStopping
│   ├── SequentialAnalyzer
│   ├── BayesianAnalyzer
│   └── CompositeRule
└── Integration
    ├── ACIMonitor (Γ₁)
    └── Stage9Reporter (E2E)
```

### 5.2 Class Design

#### 5.2.1 ConvergenceController (Main)
```python
class ConvergenceController:
    """
    Main convergence control interface.

    Responsibilities:
    - Coordinate all detectors
    - Make stopping decisions
    - Adjust N_max dynamically
    - Report to Stage 9
    """

    def __init__(self, config):
        self.detectors = [...]  # All detectors
        self.n_max_estimator = NMaxEstimator()
        self.stopping_rule = CompositeStoppingRule()
        self.aci_monitor = ACIMonitor()
        self.stage9_reporter = Stage9Reporter()

    def should_stop(self, search_state):
        """Make stopping decision"""
        # Run all detectors
        # Apply stopping rule
        # Report to Stage 9
        # Return decision + reason

    def get_n_max(self, csp, initial_aci):
        """Get initial N_max estimate"""
        # Estimate based on ACI and structure
        # Return N_max

    def adjust_n_max(self, search_state):
        """Dynamically adjust N_max"""
        # Monitor progress
        # Adjust N_max
        # Return new N_max
```

#### 5.2.2 ConvergenceDetectors
```python
class ConvergenceDetector(ABC):
    """Base class for convergence detectors"""

    @abstractmethod
    def detect(self, search_state) -> ConvergenceResult:
        """Detect convergence"""
        pass

class ACIStabilityDetector(ConvergenceDetector):
    """Detect ACI stabilization"""

    def detect(self, search_state):
        # Get ACI history
        # Calculate variance
        # Return convergence result

class SolutionStabilityDetector(ConvergenceDetector):
    """Detect solution stability"""

    def detect(self, search_state):
        # Check if best solution changed
        # Return convergence result

class VarianceDetector(ConvergenceDetector):
    """Detect variance-based convergence"""

    def detect(self, search_state):
        # Calculate rolling variance
        # Compare to threshold
        # Return convergence result

class GradientDetector(ConvergenceDetector):
    """Detect gradient-based convergence"""

    def detect(self, search_state):
        # Calculate gradient
        # Return convergence result

class GelmanRubinDetector(ConvergenceDetector):
    """Detect Gelman-Rubin convergence"""

    def detect(self, search_state):
        # Require multiple chains
        # Calculate R-hat
        # Return convergence result
```

#### 5.2.3 NMaxEstimator
```python
class NMaxEstimator:
    """Estimate and adjust N_max"""

    def __init__(self):
        self.aci_based = ACIBasedEstimator()
        self.structural = StructuralEstimator()
        self.dynamic_adjuster = DynamicAdjuster()

    def estimate_initial(self, csp, aci_result, problem_size):
        """Estimate initial N_max"""
        # Combine ACI-based and structural estimates
        # Return N_max

    def adjust_dynamic(self, search_state, current_n_max):
        """Adjust N_max based on progress"""
        # Monitor progress
        # Adjust N_max
        # Return adjusted N_max

class ACIBasedEstimator:
    """ACI-based N_max estimation"""

    def estimate(self, aci_score, problem_size):
        # Formula: N = base * f(ACI)
        # Return estimate

class StructuralEstimator:
    """Structural complexity-based estimation"""

    def estimate(self, n_vars, domain_size, constraint_density, treewidth):
        # Calculate complexity score
        # Return estimate

class DynamicAdjuster:
    """Dynamic N_max adjustment"""

    def adjust(self, search_state, current_n_max):
        # Check progress rate
        # Check ACI trajectory
        # Adjust N_max
        # Return new N_max
```

#### 5.2.4 AdaptiveStopping
```python
class AdaptiveStoppingRule(ABC):
    """Base class for stopping rules"""

    @abstractmethod
    def should_stop(self, search_state, detectors) -> Tuple[bool, str]:
        """Decide whether to stop"""
        pass

class CompositeStoppingRule(AdaptiveStoppingRule):
    """Combine multiple stopping criteria"""

    def __init__(self, strategy='MAJORITY'):
        self.strategy = strategy  # ANY, ALL, MAJORITY, WEIGHTED
        self.weights = [...]  # For WEIGHTED strategy

    def should_stop(self, search_state, detectors):
        # Run all detectors
        # Combine results based on strategy
        # Return decision + reason

class EarlyStoppingRule(AdaptiveStoppingRule):
    """Early stopping for low ACI or no progress"""

    def should_stop(self, search_state, detectors):
        # Check ACI
        # Check progress
        # Return decision + reason
```

#### 5.2.5 Γ₁ Integration
```python
class ACIMonitor:
    """Monitor ACI during search"""

    def __init__(self, aci_calculator):
        self.aci_calculator = aci_calculator
        self.aci_history = []

    def compute_aci(self, state):
        """Compute ACI for current state"""
        # Use ACI calculator
        # Cache result
        # Return ACI

    def get_trajectory(self):
        """Get ACI trajectory"""
        return self.aci_history

    def analyze_stability(self):
        """Analyze ACI stability"""
        # Calculate variance, trend
        # Return analysis
```

#### 5.2.6 Stage 9 Integration
```python
class Stage9Reporter:
    """Report convergence status to Stage 9"""

    def __init__(self, stage9_validator):
        self.stage9 = stage9_validator

    def report_convergence(self, conv_result):
        """Report convergence result"""
        # Send to Stage 9
        pass

    def report_n_max(self, n_max, adjustment_reason):
        """Report N_max adjustment"""
        # Send to Stage 9
        pass

    def report_stopping(self, decision, reason):
        """Report stopping decision"""
        # Send to Stage 9
        pass
```

### 5.3 Configuration

```python
@dataclass
class ConvergenceConfig:
    """Configuration for convergence control"""

    # Detection methods
    use_aci_stability: bool = True
    use_solution_stability: bool = True
    use_variance: bool = True
    use_gradient: bool = True
    use_gelman_rubin: bool = False  # Expensive

    # Thresholds
    variance_threshold: float = 0.001
    gradient_threshold: float = 0.001
    aci_variance_threshold: float = 0.01
    r_hat_threshold: float = 1.1

    # Window sizes
    convergence_window: int = 20
    stability_window: int = 50
    aci_window: int = 30

    # N_max estimation
    base_n_max: int = 1000
    min_n_max: int = 100
    max_n_max: int = 10000
    aci_weight_n_max: float = 0.7  # Weight for ACI vs structure
    use_dynamic_adjustment: bool = True

    # Early stopping
    enable_early_stopping: bool = True
    low_aci_threshold: float = 0.3
    no_improvement_iterations: int = 100
    diminishing_returns_threshold: float = 0.001

    # Stopping strategy
    stopping_strategy: str = 'MAJORITY'  # ANY, ALL, MAJORITY, WEIGHTED
    min_iterations_before_stop: int = 50

    # Integration
    aci_computation_interval: int = 10  # Compute ACI every N iterations
    report_to_stage9: bool = True
```

---

## 6. Validation Strategy

### 6.1 Unit Tests

#### 6.1.1 Detector Tests
- Test each detector independently
- Known convergent sequences
- Known non-convergent sequences
- Edge cases (empty, single point)

#### 6.1.2 N_max Estimator Tests
- Test ACI-based estimation
- Test structural estimation
- Test dynamic adjustment
- Test capping (min/max)

#### 6.1.3 Stopping Rule Tests
- Test ALL strategy
- Test ANY strategy
- Test MAJORITY strategy
- Test WEIGHTED strategy

### 6.2 Integration Tests

#### 6.2.1 MCTS Integration
- Run full MCTS with convergence control
- Verify stopping at convergence
- Verify N_max adjustment
- Verify ACI integration

#### 6.2.2 Γ₁ Integration
- Test ACI computation during search
- Test ACI trajectory analysis
- Test ACI-based adjustments

#### 6.2.3 Stage 9 Integration
- Test reporting to Stage 9
- Test result aggregation
- Test validation

### 6.3 Performance Validation

#### 6.3.1 Benchmark Problems
- Easy: Tree-structured CSPs (high ACI)
- Medium: Random CSPs (medium ACI)
- Hard: Dense CSPs (low ACI)

#### 6.3.2 Metrics
- Time to convergence
- Iterations to convergence
- Solution quality
- N_max accuracy
- Early stopping effectiveness

#### 6.3.3 Comparison
- vs. Fixed N_max
- vs. Variance-only stopping
- vs. Gradient-only stopping
- vs. Manual tuning

### 6.4 Ablation Studies

#### 6.4.1 Detector Ablation
- Remove each detector
- Measure impact on performance
- Identify most important detectors

#### 6.4.2 Component Ablation
- Remove ACI guidance
- Remove dynamic adjustment
- Remove early stopping
- Measure impact

---

## 7. References

### 7.1 Convergence Detection
1. Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences. *Statistical Science*, 7(4), 457-472.

2. Cowles, M. K., & Carlin, B. P. (1996). Markov chain Monte Carlo convergence diagnostics: A comparative review. *Journal of the American Statistical Association*, 91(434), 883-904.

3. Brooks, S. P., & Gelman, A. (1998). General methods for monitoring convergence of iterative simulations. *Journal of Computational and Graphical Statistics*, 7(4), 434-455.

### 7.2 Sequential Analysis
4. Wald, A. (1945). Sequential tests of statistical hypotheses. *The Annals of Mathematical Statistics*, 16(2), 117-186.

5. Siegmund, D. (1985). *Sequential analysis: Tests and confidence intervals*. Springer.

### 7.3 MCTS Convergence
6. Browne, C. B., et al. (2012). A survey of monte carlo tree search methods. *IEEE Transactions on Computational Intelligence and AI in Games*, 4(1), 1-43.

7. Chaslot, G., et al. (2008). Monte-carlo tree search: A new framework for game AI. *AIIDE*.

### 7.4 Adaptive Stopping
8. Lai, T. L. (2001). Sequential analysis: Some classical problems and new challenges. *Statistica Sinica*, 11(2), 303-350.

9. Feller, W. (2008). *An introduction to probability theory and its applications* (Vol. 2). John Wiley & Sons.

### 7.5 ACI and Γ₁
10. ACI Research Documents (Internal)
    - gamma1_aci_research.md
    - gamma1_algorithm_design.md
    - gamma1_implementation_plan.md

---

## Appendix A: Pseudocode

### A.1 Main Convergence Control Loop

```python
def convergence_controlled_mcts(csp, aci_calculator):
    # Initialize
    aci_result = aci_calculator.calculate(csp)
    controller = ConvergenceController(config)
    n_max = controller.get_n_max(csp, aci_result)
    mcts = MCTSSearch(config)

    search_state = SearchState(
        csp=csp,
        initial_aci=aci_result,
        n_max=n_max
    )

    # Main loop
    for iteration in range(n_max):
        # MCTS iteration
        mcts.step(search_state)

        # Check convergence (periodically)
        if iteration % config.check_interval == 0:
            should_stop, reason = controller.should_stop(search_state)

            if should_stop:
                logger.info(f"Stopping: {reason}")
                break

        # Dynamic N_max adjustment
        if config.use_dynamic_adjustment and iteration % config.adjust_interval == 0:
            new_n_max = controller.adjust_n_max(search_state)
            if new_n_max != n_max:
                logger.info(f"Adjusting N_max: {n_max} -> {new_n_max}")
                n_max = new_n_max

    # Return results
    return search_state.best_solution, search_state.statistics
```

### A.2 Composite Stopping Decision

```python
def composite_stopping_decision(detectors, strategy='MAJORITY'):
    results = []
    for detector in detectors:
        result = detector.detect(search_state)
        results.append(result)

    if strategy == 'ANY':
        decision = any(r.converged for r in results)
        reason = "Any detector converged"

    elif strategy == 'ALL':
        decision = all(r.converged for r in results)
        reason = "All detectors converged"

    elif strategy == 'MAJORITY':
        converged_count = sum(r.converged for r in results)
        decision = converged_count > len(results) / 2
        reason = f"{converged_count}/{len(results)} detectors converged"

    elif strategy == 'WEIGHTED':
        score = sum(w * r.converged for w, r in zip(weights, results))
        decision = score > 0.5
        reason = f"Weighted score: {score:.2f}"

    return decision, reason
```

---

## Appendix B: Configuration Examples

### B.1 Fast Configuration (Development)

```python
config = ConvergenceConfig(
    # Aggressive thresholds
    variance_threshold=0.01,
    convergence_window=10,
    min_n_max=50,
    max_n_max=500,

    # Fast stopping
    stopping_strategy='ANY',
    enable_early_stopping=True,
    min_iterations_before_stop=10,

    # Minimal computation
    use_gelman_rubin=False,
    aci_computation_interval=50
)
```

### B.2 Balanced Configuration (Production)

```python
config = ConvergenceConfig(
    # Balanced thresholds
    variance_threshold=0.001,
    convergence_window=20,
    min_n_max=100,
    max_n_max=5000,

    # Majority voting
    stopping_strategy='MAJORITY',
    enable_early_stopping=True,
    min_iterations_before_stop=50,

    # Moderate computation
    use_gelman_rubin=False,
    aci_computation_interval=20
)
```

### B.3 Thorough Configuration (Validation)

```python
config = ConvergenceConfig(
    # Strict thresholds
    variance_threshold=0.0001,
    convergence_window=50,
    min_n_max=500,
    max_n_max=10000,

    # Require all detectors
    stopping_strategy='ALL',
    enable_early_stopping=False,  # Don't stop early
    min_iterations_before_stop=200,

    # Full computation
    use_gelman_rubin=True,
    aci_computation_interval=10
)
```

---

## Appendix C: Troubleshooting

### C.1 Common Issues

**Issue:** Premature stopping
- **Cause:** Thresholds too loose
- **Fix:** Tighten variance/gradient thresholds, use ALL strategy

**Issue:** Never stops
- **Cause:** Thresholds too strict or problem genuinely not converging
- **Fix:** Loosen thresholds, enable early stopping, check ACI

**Issue:** N_max too small
- **Cause:** Underestimation from ACI or structure
- **Fix:** Increase base_n_max, adjust ACI weight

**Issue:** N_max too large
- **Cause:** Overestimation, not adjusting downward
- **Fix:** Enable early stopping, tighten diminishing returns

**Issue:** High computational cost
- **Cause:** Computing ACI too frequently, using Gelman-Rubin
- **Fix:** Increase aci_computation_interval, disable Gelman-Rubin

---

## Conclusion

This research document provides a comprehensive foundation for implementing adaptive convergence control for Monte Carlo refinement in the RESE framework. The implementation will integrate with Γ₁ (ACI calculation) and Stage 9 (E2E validation), providing intelligent, adaptive stopping based on multiple convergence criteria and dynamic N_max adjustment.

**Key Innovations:**
1. ACI-based convergence prediction
2. Composite multi-criteria stopping
3. Dynamic N_max adjustment
4. Early stopping for intractable problems
5. Seamless integration with Γ₁ and Stage 9

**Next Steps:**
1. Implement `convergence_controller.py`
2. Integrate with Γ₁ ACI calculator
3. Integrate with Stage 9 validator
4. Create comprehensive tests
5. Validate on benchmark problems
