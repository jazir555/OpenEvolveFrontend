# Φ₁.₅ Validation Strategy: Achieving >70% Assumption Mining Accuracy

**Agent**: B1 (Φ₁/Φ₁.₅ Specialist)
**Date**: 2025-12-31
**Status**: KEY INNOVATION - Validation Strategy
**Target**: >70% assumption mining accuracy

---

## Executive Summary

This document defines a comprehensive validation strategy for Φ₁.₅ (Tacit Assumption Mining) with clear success metrics, test cases, and evaluation methodology. The strategy uses a three-phase approach: historical validation (retrospective), synthetic validation (controlled), and real-world validation (prospective). The target is **>70% assumption mining accuracy** measured by precision, recall, and F1 score against ground truth.

---

## Table of Contents

1. [Success Metrics](#success-metrics)
2. [Phase 1: Historical Validation](#phase-1-historical-validation)
3. [Phase 2: Synthetic Validation](#phase-2-synthetic-validation)
4. [Phase 3: Real-World Validation](#phase-3-real-world-validation)
5. [Test Case Design](#test-case-design)
6. [Evaluation Methodology](#evaluation-methodology)
7. [Accuracy Improvement Plan](#accuracy-improvement-plan)
8. [Failure Analysis](#failure-analysis)
9. [Benchmarking](#benchmarking)
10. [Validation Timeline](#validation-timeline)

---

## 1. Success Metrics

### 1.1 Primary Metrics

**Assumption Mining Accuracy**:

1. **Precision** (Of inferred assumptions, how many are correct?)
   ```
   Precision = TP / (TP + FP)
   Where:
   - TP = True Positive (correctly inferred assumption)
   - FP = False Positive (incorrectly inferred assumption)
   ```
   **Target**: >0.70

2. **Recall** (Of actual hidden assumptions, how many did we find?)
   ```
   Recall = TP / (TP + FN)
   Where:
   - FN = False Negative (missed assumption)
   ```
   **Target**: >0.65

3. **F1 Score** (Harmonic mean of precision and recall)
   ```
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   ```
   **Target**: >0.68

4. **Confidence Calibration** (Are confidence scores accurate?)
   ```
   Calibration = |Predicted Confidence - Actual Accuracy|
   Target: < 0.10
   ```

### 1.2 Secondary Metrics

**Paradigm Shift Detection**:

1. **Detection Rate**: How many actual paradigm shifts did we detect?
   ```
   Detection Rate = Detected Paradigm Shifts / Actual Paradigm Shifts
   Target: >0.60
   ```

2. **False Positive Rate**: How many paradigm shifts were incorrectly predicted?
   ```
   FPR = False Paradigm Shifts / Total Paradigm Shift Predictions
   Target: <0.20
   ```

3. **Timeliness**: How quickly were paradigm shifts detected?
   ```
   Timeliness = (Actual Detection Time - Earliest Possible Time) / (Actual Time - Earliest Time)
   Target: <0.50 (Detected before halfway point)
   ```

**Operational Metrics**:

1. **Processing Speed**: Time to process 100 failures
   - Target: <10 seconds

2. **Scalability**: Maximum failures in database before degradation
   - Target: >10,000 failures

3. **Integration Success**: Rate of successful constraint additions to SCE
   - Target: >95%

### 1.3 Success Criteria

**Minimum Viable Performance**:
- Precision ≥ 0.60
- Recall ≥ 0.60
- F1 ≥ 0.60

**Target Performance**:
- Precision ≥ 0.70
- Recall ≥ 0.65
- F1 ≥ 0.68

**Stretch Goal**:
- Precision ≥ 0.80
- Recall ≥ 0.75
- F1 ≥ 0.77

---

## 2. Phase 1: Historical Validation

### 2.1 Objective

Validate Φ₁.₅ on historical paradigm shifts where we know the tacit assumptions in hindsight.

### 2.2 Historical Case Database

**Build database of 50+ historical paradigm shifts**:

**Scientific Revolutions** (20 cases):
1. **Michelson-Morley Experiment** (1887)
   - Explicit: "Light travels through a medium (ether)"
   - Tacit: "Space has a preferred reference frame"
   - Paradigm shift: Special relativity

2. **Blackbody Radiation** (1900)
   - Explicit: "Energy is continuous"
   - Tacit: "Classical physics applies at all scales"
   - Paradigm shift: Quantum mechanics

3. **Photoelectric Effect** (1905)
   - Explicit: "Light is purely wave-like"
   - Tacit: "Energy transfer is continuous"
   - Paradigm shift: Photons, quantization

4. **Atomic Spectra** (1913)
   - Explicit: "Electrons orbit like planets"
   - Tacit: "Continuous energy levels"
   - Paradigm shift: Bohr model, quantized orbits

5. **Double-Slit Experiment** (various)
   - Explicit: "Particles and waves are distinct"
   - Tacit: "Nature is either particle or wave"
   - Paradigm shift: Wave-particle duality

6. **Bacterial Transformation** (1928)
   - Explicit: "Genetic material is protein"
   - Tacit: "Traits are transmitted directly"
   - Paradigm shift: DNA as genetic material

7. **Continental Drift** (1912)
   - Explicit: "Continents are fixed"
   - Tacit: "Earth's surface is static"
   - Paradigm shift: Plate tectonics

8. **Heliocentrism** (1543)
   - Explicit: "Earth is center of universe"
   - Tacit: "Celestial bodies move in perfect circles"
   - Paradigm shift: Heliocentric model

9. **Spontaneous Generation** (1668)
   - Explicit: "Life arises from non-living matter"
   - Tacit: "Vital force exists"
   - Paradigm shift: Biogenesis

10. **Phlogiston Theory** (1770s)
    - Explicit: "Flammable materials contain phlogiston"
    - Tacit: "Fire releases a substance"
    - Paradigm shift: Oxygen theory of combustion

... and 10 more

**Computer Science Paradigm Shifts** (15 cases):
1. **Randomized Algorithms** (1970s-80s)
   - Explicit: "Algorithms must be deterministic"
   - Tacit: "Randomness reduces reliability"
   - Paradigm shift: Randomized algorithms (Miller, Rabin, Karp)

2. **Approximation Algorithms** (1970s)
   - Explicit: "Exact solutions required"
   - Tacit: "Approximation is unacceptable"
   - Paradigm shift: Polynomial approximation schemes

3. **NP-Completeness** (1971)
   - Explicit: "All problems can be solved efficiently"
   - Tacit: "Exponential time is unacceptable"
   - Paradigm shift: Complexity classes, hardness

4. **Machine Learning** (1950s-80s)
   - Explicit: "Systems must be explicitly programmed"
   - Tacit: "Rules must be hand-crafted"
   - Paradigm shift: Learning from data

5. **Deep Learning** (2012)
   - Explicit: "Neural networks don't scale"
   - Tacit: "Feature engineering is essential"
   - Paradigm shift: End-to-end learning, representation learning

6. **Parallel Computing** (1960s-70s)
   - Explicit: "Sequential processing is optimal"
   - Tacit: "Parallelism adds too much complexity"
   - Paradigm shift: Parallel algorithms, distributed computing

7. **Functional Programming** (1950s-70s)
   - Explicit: "Imperative programming is natural"
   - Tacit: "Mutation is necessary"
   - Paradigm shift: Pure functions, immutability

8. **Relational Databases** (1970)
   - Explicit: "Data must be hierarchical/network"
   - Tacit: "Structure must reflect application"
   - Paradigm shift: Relational model, SQL

9. **Internet Protocols** (1970s-80s)
   - Explicit: "Centralized control required"
   - Tacit: "Network needs central authority"
   - Paradigm shift: Decentralized routing, TCP/IP

10. **Cryptocurrencies** (2008)
    - Explicit: "Digital money requires central authority"
    - Tacit: "Consensus impossible without trusted third party"
    - Paradigm shift: Blockchain, proof-of-work

... and 5 more

**Engineering/Technology Paradigm Shifts** (15 cases):
1. **Steam Engine** (1776)
   - Explicit: "Power comes from animals/water/wind"
   - Tacit: "Heat cannot be converted to motion efficiently"
   - Paradigm shift: Steam power, industrial revolution

2. **Electric Light** (1879)
   - Explicit: "Gas/coal lighting is optimal"
   - Tacit: "Electricity too dangerous for lighting"
   - Paradigm shift: Electrical lighting, power grids

3. **Flight** (1903)
   - Explicit: "Heavier-than-air flight impossible"
   - Tacit: "Wings must flap like birds"
   - Paradigm shift: Fixed-wing aircraft

4. **Semiconductors** (1947)
   - Explicit: "Vacuum tubes are only amplification method"
   - Tacit: "Solids cannot control current"
   - Paradigm shift: Transistors, integrated circuits

5. **Digital Photography** (1975)
   - Explicit: "Film is required for imaging"
   - Tacit: "Chemical process necessary for quality"
   - Paradigm shift: Digital sensors, computational photography

... and 10 more

### 2.3 Validation Protocol

**For Each Historical Case**:

1. **Data Preparation**:
   - Collect "before" state: Failed attempts, null results
   - Identify actual tacit assumption (ground truth)
   - Document paradigm shift

2. **Input to Φ₁.₅**:
   - Feed "before" failures (without knowing the outcome)
   - Run Φ₁.₅ assumption mining

3. **Evaluation**:
   - Compare inferred assumptions to ground truth
   - Calculate precision/recall
   - Document paradigm shift detection

4. **Metrics**:
   - Did Φ₁.₅ identify the correct tacit assumption?
   - How confident was it?
   - Did it suggest the correct paradigm shift?
   - How many false positives?

### 2.4 Expected Outcomes

**Success Criteria for Historical Validation**:
- ≥70% of cases: Correct assumption inferred (TP or in top-3)
- ≥60% of paradigm shift cases: Correct paradigm shift detected
- Confidence scores properly calibrated (±0.15)

**Example Success**:

**Case**: Michelson-Morley Experiment
- **Ground Truth**: "Ether does not exist"
- **Φ₁.₅ Output**:
  - Assumption 1: "Assuming ether exists causes systematic failures" (confidence: 0.85) ✓
  - Assumption 2: "Speed of light is constant" (confidence: 0.72) ✓
  - Paradigm shift: Special relativity suggested ✓
- **Result**: TP (correct assumption detected)

---

## 3. Phase 2: Synthetic Validation

### 3.1 Objective

Validate Φ₁.₅ on controlled synthetic problems where we embed known hidden constraints.

### 3.2 Synthetic Problem Design

**Design 50+ synthetic problems with known hidden constraints**:

**Template**:
```
Problem: [Problem Description]
Explicit Constraints: [Known constraints]
Hidden Constraint: [Embedded tacit assumption]
Expected Φ₁.₅ Output: [What Φ₁.₅ should discover]
```

**Example Problems**:

**Problem Set 1: Computational Problems** (20 problems)

1. **Exact Solution Problem**
   - Problem: "Solve NP-hard optimization problem exactly"
   - Explicit: "Find optimal solution"
   - Hidden: "Must solve exactly (no approximation allowed)"
   - Expected: "Approximation is acceptable"

2. **Determinism Problem**
   - Problem: "TSP using deterministic methods only"
   - Explicit: "Use deterministic algorithms"
   - Hidden: "Randomness is prohibited"
   - Expected: "Randomization can help"

3. **Time Limit Problem**
   - Problem: "Solve in polynomial time"
   - Explicit: "Must run in O(n^k)"
   - Hidden: "Exponential time is unacceptable"
   - Expected: "Exponential with pruning may work"

4. **Space Limit Problem**
   - Problem: "Solve with linear space"
   - Explicit: "Use O(n) memory"
   - Hidden: "Cannot use exponential space"
   - Expected: "More space allows better solutions"

5. **Sequential Problem**
   - Problem: "Solve using sequential processing"
   - Explicit: "Use one processor"
   - Hidden: "Parallelism not allowed"
   - Expected: "Parallel processing can help"

... and 15 more

**Problem Set 2: Engineering Design Problems** (15 problems)

1. **Material Limit Problem**
   - Problem: "Design bridge with steel only"
   - Explicit: "Use steel as material"
   - Hidden: "Must use traditional materials"
   - Expected: "Composite materials work better"

2. **Power Source Problem**
   - Problem: "Power device with batteries only"
   - Explicit: "Use battery power"
   - Hidden: "Must be portable energy source"
   - Expected: "Alternative power sources available"

3. **Cooling Problem**
   - Problem: "Cool system with air only"
   - Explicit: "Use air cooling"
   - Hidden: "Liquid cooling not considered"
   - Expected: "Liquid cooling more effective"

... and 12 more

**Problem Set 3: Algorithm Design Problems** (15 problems)

1. **Greedy Problem**
   - Problem: "Solve using greedy approach only"
   - Explicit: "Use greedy algorithm"
   - Hidden: "Local optima are global optima"
   - Expected: "Dynamic programming/backtracking needed"

2. **Linear Problem**
   - Problem: "Solve using linear methods only"
   - Explicit: "Use linear algorithms"
   - Hidden: "Nonlinear methods not considered"
   - Expected: "Nonlinear approaches may work"

3. **Deterministic Search Problem**
   - Problem: "Search using deterministic heuristics"
   - Explicit: "Use deterministic search"
   - Hidden: "Randomized search not allowed"
   - Expected: "Monte Carlo methods effective"

... and 12 more

### 3.3 Validation Protocol

**For Each Synthetic Problem**:

1. **Generation**:
   - Create problem with embedded hidden constraint
   - Generate 20-50 null results that violate the hidden constraint
   - Ensure null results show systematic failure pattern

2. **Input to Φ₁.₅**:
   - Feed null results to Φ₁.₅
   - Don't provide hidden constraint information

3. **Evaluation**:
   - Check if Φ₁.₅ infers hidden constraint
   - Measure confidence
   - Count false positives

4. **Metrics**:
   - Accuracy: % correct
   - Confidence calibration
   - False positive rate

### 3.4 Difficulty Levels

**Level 1 (Easy)**: Single obvious hidden constraint
- 15 problems
- Target accuracy: >85%

**Level 2 (Medium)**: Multiple interacting hidden constraints
- 20 problems
- Target accuracy: >75%

**Level 3 (Hard)**: Subtle, context-dependent hidden constraints
- 15 problems
- Target accuracy: >60%

---

## 4. Phase 3: Real-World Validation

### 4.1 Objective

Validate Φ₁.₅ on current research problems where hidden constraints are unknown.

### 4.2 Real-World Problem Sources

**Collaboration with Research Groups**:

1. **University Research Labs**:
   - CS/AI departments
   - Engineering departments
   - Physics/Chemistry departments

2. **Industry R&D**:
   - Tech companies (Google, Microsoft, etc.)
   - Engineering companies
   - Pharmaceutical companies

3. **Online Platforms**:
   - Stack Overflow (unsolved problems)
   - GitHub (failed attempts)
   - Research forums

### 4.3 Validation Protocol

**For Each Real-World Problem**:

1. **Problem Collection**:
   - Find problem with multiple failed attempts
   - Document null results
   - Current status: Unsolved or stuck

2. **Φ₁.₅ Analysis**:
   - Run Φ₁.₅ on null results
   - Generate tacit assumptions
   - Suggest paradigm shifts (if applicable)

3. **Expert Evaluation**:
   - Present assumptions to domain experts
   - Get expert ratings:
     - Is this assumption correct? (Yes/No/Maybe)
     - How novel? (1-5 scale)
     - How useful? (1-5 scale)
   - Collect feedback

4. **Validation**:
   - Test suggestions in practice
   - Track if they lead to progress
   - Document outcomes

### 4.4 Metrics

**Expert Agreement Rate**:
```
Agreement = (Experts saying "Yes") / (Total experts)
Target: >0.60
```

**Usefulness Rating**:
```
Average Usefulness = Σ(Expert Ratings) / (Total experts)
Target: >3.5 / 5.0
```

**Success Rate**:
```
Success = (Problems where suggestion helped) / (Total problems)
Target: >0.50
```

---

## 5. Test Case Design

### 5.1 Minimal Test Set

**20 diverse test cases** spanning:
- Different domains (CS, physics, engineering)
- Different difficulty levels
- Different assumption types (ontological, methodological, etc.)

### 5.2 Comprehensive Test Set

**100+ test cases**:
- 50 historical paradigm shifts
- 50 synthetic problems
- (Optional) Real-world problems

### 5.3 Test Case Template

```python
@dataclass
class Phi15TestCase:
    """Test case for Φ₁.₅ validation"""

    # Problem description
    case_id: str
    domain: str
    difficulty: str  # "easy", "medium", "hard"

    # Ground truth
    hidden_constraints: List[str]
    paradigm_shift: Optional[str]

    # Input data
    null_results: List[NullResult]
    explicit_constraints: List[str]

    # Expected output
    expected_assumptions: List[str]  # In order of priority
    expected_paradigm_shift: Optional[str]

    # Metadata
    source: str  # "historical", "synthetic", "real_world"
    reference: str  # Citation or source URL
```

### 5.4 Example Test Case

```python
michelson_morley_case = Phi15TestCase(
    case_id="H001",
    domain="physics",
    difficulty="medium",

    hidden_constraints=[
        "Ether exists as medium for light propagation",
        "Speed of light is not constant across reference frames"
    ],
    paradigm_shift="Special relativity",

    null_results=[
        # 50+ null results from failed ether detection attempts
        # Each with: no difference in speed of light in perpendicular directions
        # ... (would include actual experimental data)
    ],
    explicit_constraints=[
        "Light is a wave",
        "Waves require a medium"
    ],

    expected_assumptions=[
        "Assumption: Ether exists causes systematic detection failures",
        "Assumption: Speed of light is constant explains null results"
    ],
    expected_paradigm_shift="Special relativity (constancy of light speed)",

    source="historical",
    reference="Michelson, A. A., & Morley, E. W. (1887). 'On the Relative Motion of the Earth and the Luminiferous Ether'. American Journal of Science, 34, 333-345."
)
```

---

## 6. Evaluation Methodology

### 6.1 Automated Evaluation

**For historical and synthetic cases** (where ground truth is known):

```python
def evaluate_phi15(test_case: Phi15TestCase,
                   phi15_output: Tuple[List[TacitAssumption], ParadigmShiftRecommendation]) -> Dict:
    """
    Evaluate Φ₁.₅ output against ground truth.

    Returns:
        Evaluation metrics
    """
    inferred_assumptions, paradigm_rec = phi15_output

    # 1. Assumption-level evaluation
    tp = 0  # True positives (correct assumptions)
    fp = 0  # False positives (incorrect assumptions)
    fn = 0  # False negatives (missed assumptions)

    # Match inferred to expected
    for expected in test_case.expected_assumptions:
        # Check if any inferred assumption matches
        matches = [a for a in inferred_assumptions
                  if semantic_similarity(a.description, expected) > 0.7]

        if matches:
            tp += 1
        else:
            fn += 1

    # Count extra assumptions (false positives)
    fp = len(inferred_assumptions) - tp

    # 2. Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 3. Paradigm shift evaluation
    paradigm_correct = (
        test_case.expected_paradigm_shift is not None and
        paradigm_rec.trigger and
        semantic_similarity(paradigm_rec.suggested_alternatives[0] if paradigm_rec.suggested_alternatives else "",
                          test_case.expected_paradigm_shift) > 0.7
    )

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'paradigm_detected': paradigm_correct,
        'assumptions': inferred_assumptions,
        'paradigm_recommendation': paradigm_rec
    }
```

### 6.2 Human Evaluation

**For real-world cases** (where ground truth is unknown):

**Expert Rating Form**:

```
Φ₁.₅ Assumption Evaluation
===========================

Case ID: ___________
Expert Name: ___________
Expertise: ___________

Assumption: [Assumption description]
Confidence: [Confidence score]

Questions:
1. Is this assumption correct?
   □ Yes
   □ No
   □ Maybe / Partially correct

2. How novel is this assumption? (1 = Obvious, 5 = Very novel)
   □ 1  □ 2  □ 3  □ 4  □ 5

3. How useful is this assumption for solving the problem? (1 = Not useful, 5 = Very useful)
   □ 1  □ 2  □ 3  □ 4  □ 5

4. Is this assumption new to you, or were you already aware of it?
   □ New
   □ Already aware

5. If this is a paradigm shift recommendation, is it reasonable?
   □ Yes, very reasonable
   □ Somewhat reasonable
   □ Not reasonable
   □ N/A (not a paradigm shift)

Comments:
_________________________
_________________________
_________________________
```

### 6.3 Aggregate Metrics

**Combine multiple test cases**:

```python
def aggregate_evaluation(evaluations: List[Dict]) -> Dict:
    """
    Aggregate evaluation metrics across multiple test cases.
    """
    # Average metrics
    avg_precision = np.mean([e['precision'] for e in evaluations])
    avg_recall = np.mean([e['recall'] for e in evaluations])
    avg_f1 = np.mean([e['f1'] for e in evaluations])

    # Total counts
    total_tp = sum([e['true_positives'] for e in evaluations])
    total_fp = sum([e['false_positives'] for e in evaluations])
    total_fn = sum([e['false_negatives'] for e in evaluations])

    # Overall metrics (micro-averaged)
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)

    # Paradigm shift detection rate
    paradigm_cases = [e for e in evaluations if e['test_case'].expected_paradigm_shift is not None]
    paradigm_detection_rate = (
        sum([e['paradigm_detected'] for e in paradigm_cases]) / len(paradigm_cases)
        if paradigm_cases else 0
    )

    return {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'paradigm_detection_rate': paradigm_detection_rate,
        'num_cases': len(evaluations)
    }
```

---

## 7. Accuracy Improvement Plan

### 7.1 Iterative Refinement

**If accuracy < 70%**:

1. **Analyze Failures**:
   - Which test cases are failing?
   - What types of assumptions are missed?
   - What causes false positives?

2. **Identify Bottlenecks**:
   - Feature extraction issues?
   - Clustering problems?
   - Abduction logic flaws?
   - Confidence scoring errors?

3. **Targeted Improvements**:
   - Improve feature engineering for specific domains
   - Adjust clustering algorithms
   - Enhance abduction rules
   - Calibrate confidence weights

4. **Retest**:
   - Run validation again
   - Compare to baseline
   - Document improvements

### 7.2 Failure Mode Analysis

**Common Failure Modes**:

1. **Missing Assumptions** (Low Recall):
   - **Cause**: Clustering not grouping related failures
   - **Fix**: Improve similarity metrics, adjust clustering parameters

2. **Incorrect Assumptions** (Low Precision):
   - **Cause**: Abduction generating too many candidates
   - **Fix**: Stricter candidate filtering, better priors

3. **Overconfidence** (Poor Calibration):
   - **Cause**: Confidence scores too high overall
   - **Fix**: Recalibrate using Platt scaling or isotonic regression

4. **Paradigm Shift Missed**:
   - **Cause**: Crisis threshold too high
   - **Fix**: Lower threshold or improve signal aggregation

### 7.3 A/B Testing

**Compare algorithm variants**:

```python
# Variant A: Original algorithm
assumptions_a = phi15_variant_a.process(test_case.null_results)

# Variant B: Improved feature extraction
assumptions_b = phi15_variant_b.process(test_case.null_results)

# Compare
eval_a = evaluate_phi15(test_case, assumptions_a)
eval_b = evaluate_phi15(test_case, assumptions_b)

if eval_b['f1'] > eval_a['f1']:
    print("Variant B is better")
```

---

## 8. Failure Analysis

### 8.1 Error Analysis Framework

**For each failed test case**, document:

1. **Expected vs. Actual**:
   - What assumption should have been found?
   - What was actually found?
   - Why did it fail?

2. **Failure Category**:
   - **Clustering failure**: Failures not grouped correctly
   - **Abduction failure**: Correct explanation not generated
   - **Scoring failure**: Good explanation scored too low
   - **Paradigm failure**: Paradigm shift not detected

3. **Systematic Issue?**:
   - Is this a one-off error?
   - Or does it indicate a systematic problem?

### 8.2 Error Analysis Template

```python
@dataclass
class FailureAnalysis:
    """Analysis of a Φ₁.₅ failure"""

    case_id: str
    failure_type: str  # "clustering", "abduction", "scoring", "paradigm"

    # Expected vs. actual
    expected_assumption: str
    actual_output: List[TacitAssumption]

    # Why did it fail?
    root_cause: str
    systematic: bool  # Is this a systematic issue?

    # How to fix?
    suggested_fix: str
    priority: str  # "high", "medium", "low"
```

### 8.3 Systematic Issues Tracking

**Create issue tracker for systematic problems**:

| Issue ID | Description | Cases Affected | Priority | Status |
|----------|-------------|----------------|----------|--------|
| F001 | Physics domain assumptions poorly detected | 5/20 | High | Open |
| F002 | Overconfidence on engineering problems | 8/15 | Medium | Open |
| F003 | Paradigm shifts detected too early | 3/10 | Low | Open |

---

## 9. Benchmarking

### 9.1 Baseline Methods

**Compare Φ₁.₅ to alternative approaches**:

1. **Random Baseline**:
   - Randomly select assumptions from candidates
   - Expected: Low accuracy

2. **Frequency Baseline**:
   - Select most frequently violated constraints
   - Expected: Moderate accuracy

3. **Keyword Baseline**:
   - Extract assumptions from error messages using keywords
   - Expected: Low-moderate accuracy

4. **Manual Analysis**:
   - Human expert analysis (time-consuming but accurate)
   - Expected: High accuracy

### 9.2 Benchmark Results Template

```
Φ₁.₅ Benchmark Results
======================

Test Set: [Name and size]
Date: [Date]

Method              | Precision | Recall | F1    | Time (sec)
--------------------|-----------|--------|-------|----------
Φ₁.₅ (Full)         | 0.75      | 0.70   | 0.72  | 8.5
Φ₁.₅ (No abduction) | 0.65      | 0.60   | 0.62  | 5.2
Frequency Baseline  | 0.45      | 0.55   | 0.49  | 1.0
Keyword Baseline    | 0.35      | 0.40   | 0.37  | 2.0
Random Baseline     | 0.15      | 0.20   | 0.17  | 0.5
Manual Expert       | 0.85      | 0.80   | 0.82  | 300.0

Conclusion: Φ₁.₅ achieves [X] improvement over baselines while being [Y]x faster than manual analysis.
```

---

## 10. Validation Timeline

### 10.1 Phase 1: Historical Validation (Week 11-12)

**Week 11**:
- [ ] Build historical case database (20+ cases)
- [ ] Implement evaluation framework
- [ ] Run Φ₁.₅ on historical cases
- [ ] Collect results

**Week 12**:
- [ ] Analyze historical validation results
- [ ] Document accuracy metrics
- [ ] Identify systematic issues
- [ ] Make targeted improvements
- [ ] Re-validate

**Target**: ≥70% accuracy on historical cases

### 10.2 Phase 2: Synthetic Validation (Week 13)

**Week 13**:
- [ ] Design synthetic problems (50+)
- [ ] Generate null results for each problem
- [ ] Run Φ₁.₅ on synthetic cases
- [ ] Evaluate accuracy by difficulty level
- [ ] Document results

**Target**: ≥70% overall accuracy, >85% on easy cases

### 10.3 Phase 3: Real-World Validation (Week 14-16)

**Week 14-15**:
- [ ] Recruit expert collaborators
- [ ] Collect real-world problem cases
- [ ] Run Φ₁.₅ on real cases
- [ ] Present results to experts
- [ ] Collect expert ratings

**Week 16**:
- [ ] Analyze expert feedback
- [ ] Calculate expert agreement rate
- [ ] Document success stories
- [ ] Write final validation report

**Target**: >60% expert agreement, >3.5/5.0 usefulness

### 10.4 Final Validation Report

**Deliverable**: Comprehensive validation report including:

1. **Executive Summary**
   - Overall accuracy achieved
   - Comparison to targets
   - Key findings

2. **Detailed Results**
   - Historical validation results
   - Synthetic validation results
   - Real-world validation results

3. **Failure Analysis**
   - Common failure modes
   - Systematic issues
   - Recommendations for improvement

4. **Benchmarking**
   - Comparison to baselines
   - Performance analysis
   - Strengths and weaknesses

5. **Conclusions**
   - Is Φ₁.₅ ready for production?
   - What improvements are needed?
   - Future work

---

## Summary

**Validation Strategy Complete**:

✅ **Success Metrics**: Precision, recall, F1, calibration defined
✅ **Phase 1 - Historical**: 50+ paradigm shift cases for retrospective validation
✅ **Phase 2 - Synthetic**: 50+ controlled problems with known hidden constraints
✅ **Phase 3 - Real-World**: Expert evaluation on current problems
✅ **Test Cases**: 100+ diverse test cases spanning domains and difficulties
✅ **Evaluation**: Automated and human evaluation methodologies
✅ **Improvement Plan**: Iterative refinement based on failure analysis
✅ **Benchmarking**: Comparison to baseline methods
✅ **Timeline**: 6-week validation schedule

**Target Achievement**:
- Primary: >70% assumption mining accuracy (F1 score)
- Secondary: >60% paradigm shift detection rate
- Calibration: <0.10 confidence error

**Next Steps**:
- Begin implementation (Week 11)
- Concurrent validation during implementation
- Iterative refinement based on validation results

**Agent**: B1 (Φ₁/Φ₁.₅ Specialist)
**Date**: 2025-12-31
**Status**: Validation strategy complete, ready for implementation
