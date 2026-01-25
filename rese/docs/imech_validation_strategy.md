# I_mech: Validation Strategy

**Agent:** G3 (I_mech Specialist)
**Date:** 2025-12-31
**Target:** Week 32 Validation
**Goal:** Achieve >80% transfer success correlation

---

## Executive Summary

This document outlines a comprehensive validation strategy for I_mech, including:
1. **Success Metrics** - quantitative measures of performance
2. **Benchmark Datasets** - known analogies from science and engineering
3. **Evaluation Methodology** - testing protocols and statistical analysis
4. **Ablation Studies** - component contribution analysis
5. **Failure Analysis** - understanding edge cases

**Primary Success Criterion:** >80% correlation on benchmark analogies (historical successful technology transfers)

---

## 1. Success Metrics

### 1.1 Primary Metrics

#### Metric 1: Transfer Success Rate

**Definition:** Proportion of benchmark analogies where I_mech successfully transfers a solution that satisfies target domain constraints.

**Formula:**
```
Transfer Success Rate = (Number of successful transfers) / (Total number of benchmark cases)
```

**Target:** ≥ 0.80 (80%)

**Measurement:**
- For each benchmark case (source domain with solution, target domain):
  1. Run I_mech to detect isomorphism
  2. If similarity > threshold, transfer solution
  3. Validate transferred solution against target constraints
  4. Count as success if all constraints satisfied (within tolerance)

**Success Criterion:** Transfer Success Rate ≥ 0.80

---

#### Metric 2: Isomorphism Detection Accuracy

**Definition:** Proportion of benchmark cases where I_mech correctly identifies presence/absence of mechanistic isomorphism.

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

where:
  TP = True Positive (isomorphic, correctly detected)
  TN = True Negative (not isomorphic, correctly rejected)
  FP = False Positive (not isomorphic, incorrectly detected)
  FN = False Negative (isomorphic, incorrectly rejected)
```

**Target:** ≥ 0.85 (85%)

**Measurement:**
- Benchmark cases annotated with ground truth (isomorphic/not)
- Compare I_mech predictions against annotations
- Compute confusion matrix

**Success Criterion:** Accuracy ≥ 0.85

---

#### Metric 3: Similarity Score Correlation

**Definition:** Pearson correlation between I_mech similarity scores and human-rated similarity.

**Formula:**
```
Correlation = Cov(S_imech, S_human) / (σ_imech × σ_human)

where:
  S_imech = I_mech similarity scores
  S_human = human expert ratings
```

**Target:** ≥ 0.80

**Measurement:**
- Select 50 benchmark domain pairs
- Have human experts rate similarity (1-5 scale)
- Compute I_mech similarity scores (0-1 scale)
- Compute Pearson correlation

**Success Criterion:** Correlation ≥ 0.80

---

### 1.2 Secondary Metrics

#### Metric 4: Solution Quality Score

**Definition:** Quality of transferred solution compared to ground truth (if available).

**Formula:**
```
Quality Score = 1 - (||S_transferred - S_optimal|| / ||S_optimal||)
```

**Target:** ≥ 0.70

#### Metric 5: Computational Efficiency

**Definition:** Average time to compare two domains.

**Target:**
- Small domains (<100 nodes): < 1 second
- Medium domains (100-1000 nodes): < 10 seconds
- Large domains (>1000 nodes): < 60 seconds

#### Metric 6: Proof Verification Rate

**Definition:** Proportion of generated proofs that pass Lean 4 verification.

**Target:** ≥ 0.90

---

## 2. Benchmark Datasets

### 2.1 Historical Analogies Dataset (Primary)

**Source:** Curated collection of 100 historical technology transfers based on mechanistic analogy.

#### Categories

##### 1. Mechanical Systems (25 cases)

**Examples:**
1. **Steam Engine → Internal Combustion Engine**
   - Source: Expanding gas drives piston (steam)
   - Target: Expanding gas drives piston (controlled explosion)
   - Mechanism: Thermodynamic expansion → mechanical work
   - Year: 1876 (Otto)
   - Expected Isomorphism: High (0.9+)

2. **Water Wheel → Turbine**
   - Source: Fluid flow drives rotation
   - Target: Fluid flow drives rotation (optimized)
   - Mechanism: Fluid momentum transfer
   - Year: 1884 (Parsons)
   - Expected Isomorphism: High (0.85+)

3. **Clock Escapement → Watch Escapement**
   - Source: Regulated energy release mechanism
   - Target: Miniaturized regulated release
   - Mechanism: Oscillator-controlled gear advance
   - Year: 1675 (Hooke)
   - Expected Isomorphism: Very High (0.95+)

... (22 more cases)

##### 2. Electrical Systems (20 cases)

**Examples:**
1. **Telegraph → Telephone**
   - Source: Electrical signal transmission (on/off)
   - Target: Electrical signal transmission (continuous)
   - Mechanism: Electrical conduction over distance
   - Year: 1876 (Bell)
   - Expected Isomorphism: Medium (0.7+)

2. **DC Motor → AC Motor**
   - Source: Magnetic field rotates rotor
   - Target: Rotating magnetic field rotates rotor
   - Mechanism: Electromagnetic induction
   - Year: 1887 (Tesla)
   - Expected Isomorphism: High (0.85+)

3. **Vacuum Tube → Transistor**
   - Source: Control current flow with grid voltage
   - Target: Control current flow with base voltage
   - Mechanism: Electron flow modulation
   - Year: 1947 (Bardeen, Brattain, Shockley)
   - Expected Isomorphism: Very High (0.95+)

... (17 more cases)

##### 3. Biological Systems (15 cases)

**Examples:**
1. **Bird Wing → Airplane Wing**
   - Source: Lift generation through airflow
   - Target: Lift generation through airflow
   - Mechanism: Bernoulli principle / pressure differential
   - Year: 1903 (Wright Brothers)
   - Expected Isomorphism: High (0.8+)

2. **Human Arm → Robotic Arm**
   - Source: Articulated linkage with tendons
   - Target: Articulated linkage with actuators
   - Mechanism: Lever systems with rotational joints
   - Year: 1961 (Unimate)
   - Expected Isomorphism: High (0.85+)

3. **Bird Flocking → Swarm Robotics**
   - Source: Distributed coordination without central control
   - Target: Distributed robot coordination
   - Mechanism: Local interaction rules → global behavior
   - Year: 1980s+
   - Expected Isomorphism: Medium (0.75+)

... (12 more cases)

##### 4. Chemical Systems (15 cases)

**Examples:**
1. **Natural Dyes → Synthetic Dyes**
   - Source: Molecular structure absorbs light
   - Target: Engineered molecular structure absorbs light
   - Mechanism: Conjugated electron systems
   - Year: 1856 (Perkin)
   - Expected Isomorphism: High (0.85+)

2. **Natural Enzymes → Industrial Catalysts**
   - Source: Protein lowers activation energy
   - Target: Metal/chemical lowers activation energy
   - Mechanism: Alternative reaction pathway
   - Year: 1900s (Haber, Bosch)
   - Expected Isomorphism: Medium (0.7+)

... (13 more cases)

##### 5. Information Systems (25 cases)

**Examples:**
1. **Library Catalog → Database Index**
   - Source: Organized access to information
   - Target: Organized access to data
   - Mechanism: Hierarchical classification + lookup
   - Year: 1960s+
   - Expected Isomorphism: High (0.8+)

2. **Postal System → Packet Switching**
   - Source: Message routing through intermediate nodes
   - Target: Data routing through intermediate nodes
   - Mechanism: Store-and-forward with addressing
   - Year: 1960s (ARPANET)
   - Expected Isomorphism: Very High (0.9+)

3. **Biological Neuron → Artificial Neuron**
   - Source: Summation + threshold activation
   - Target: Weighted sum + activation function
   - Mechanism: Information integration and threshold-based output
   - Year: 1943 (McCulloch-Pitts)
   - Expected Isomorphism: High (0.85+)

... (22 more cases)

#### Dataset Structure

```python
# File: rese/tests/benchmarks/data/historical_analogies.json

{
  "analogies": [
    {
      "id": "steam_to_ic",
      "name": "Steam Engine to Internal Combustion",
      "category": "mechanical",
      "year": 1876,
      "inventor": "Nikolaus Otto",
      "source_domain": {
        "id": "steam_engine",
        "name": "Steam Engine",
        "description": "External combustion engine using steam",
        "constraints": ["PV = nRT", "efficiency < 1 - T_cold/T_hot"],
        "fdg": "steam_engine_fdg.json",
        "solution": "steam_engine_solution.json"
      },
      "target_domain": {
        "id": "ic_engine",
        "name": "Internal Combustion Engine",
        "description": "Internal combustion using fuel-air mixture",
        "constraints": ["PV = nRT", "efficiency < 1 - T_cold/T_hot"],
        "fdg": "ic_engine_fdg.json",
        "solution": "ic_engine_solution.json"
      },
      "expected_similarity": 0.92,
      "ground_truth_isomorphism": true,
      "mechanism_preserved": "thermodynamic_expansion",
      "transfer_success": true
    },
    ... // 99 more cases
  ],
  "metadata": {
    "total_cases": 100,
    "categories": ["mechanical", "electrical", "biological", "chemical", "information"],
    "source": "Historical patent and invention records"
  }
}
```

---

### 2.2 Synthetic Analogies Dataset (Secondary)

**Purpose:** Generate controlled test cases with known ground truth.

**Construction:**
- Start with base domain FDG
- Apply transformations:
  1. **Node renaming** (labels changed, structure same)
  2. **Node addition** (extra nodes added)
  3. **Edge rewiring** (some edges changed)
  4. **Parameter scaling** (parameters scaled)
- Create 500 synthetic pairs with varying similarity levels

**Categories:**
- 100 pairs: Near-identical (similarity 0.95-1.0)
- 100 pairs: High similarity (0.85-0.95)
- 100 pairs: Medium similarity (0.70-0.85)
- 100 pairs: Low similarity (0.50-0.70)
- 100 pairs: Not similar (0.0-0.50)

---

### 2.3 Negative Examples Dataset

**Purpose:** Test ability to reject non-analogous domains.

**Content:** 50 domain pairs that are NOT mechanistically isomorphic:
- Different constraint structures
- Different causal mechanisms
- Incompatible solution approaches

**Expected Result:** I_mech should correctly reject (similarity < 0.5)

---

## 3. Evaluation Methodology

### 3.1 Cross-Validation Setup

**K-Fold Cross-Validation (K=5):**
- Split 100 historical analogies into 5 folds (20 cases each)
- For each fold:
  1. Use 80 cases as "seen" (train threshold weights)
  2. Test on 20 cases as "unseen" (evaluate generalization)
- Report mean and standard deviation across folds

**Stratification:** Ensure each fold has proportional representation from all categories

### 3.2 Baseline Comparisons

Compare I_mech against:

#### Baseline 1: Random
- Random similarity scores
- Expected: 50% accuracy (binary), 0 correlation (continuous)

#### Baseline 2: Feature Matching
- Simple feature-based similarity (constraint types, variable counts)
- No structural or causal analysis
- Expected: 60-65% accuracy

#### Baseline 3: Graph Isomorphism Only
- WL + VF2 without causal or semantic analysis
- Expected: 70-75% accuracy

#### Baseline 4: SME (Structure-Mapping Engine)
- Original structure-mapping implementation
- Expected: 75-80% accuracy

**I_mech Target:** >80% accuracy (significant improvement over all baselines)

### 3.3 Statistical Testing

**Hypothesis Test:**
- H₀: I_mech accuracy ≤ baseline accuracy
- H₁: I_mech accuracy > baseline accuracy

**Test:** McNemar's test (for paired binary outcomes)

**Significance:** α = 0.05

**Required Sample Size:** For 80% power to detect 10% improvement, need n ≥ 82 cases (we have 100)

### 3.4 Human Evaluation

**Expert Panel:** 5 domain experts (engineering, cognitive science, history of technology)

**Protocol:**
1. Present 50 random domain pairs from benchmark
2. Experts rate: (a) Isomorphic? (yes/no), (b) Similarity (1-5 scale)
3. Compute inter-rater reliability (Fleiss' κ)
4. Compare expert consensus to I_mech predictions

**Expected:** High agreement (κ ≥ 0.7) between I_mech and experts

---

## 4. Ablation Studies

### 4.1 Purpose

Quantify contribution of each I_mech component to overall performance.

### 4.2 Experimental Design

Test each component independently:

| Configuration | Components Active | Expected Accuracy |
|--------------|-------------------|-------------------|
| Full Model | All (Struct + Causal + Semantic + Intervention + Proof) | ≥ 0.80 |
| No Causal | Struct + Semantic + Intervention + Proof | 0.70-0.75 |
| No Semantic | Struct + Causal + Intervention + Proof | 0.72-0.77 |
| No Intervention | Struct + Causal + Semantic + Proof | 0.68-0.73 |
| No Proof | Struct + Causal + Semantic + Intervention | 0.78-0.80 |
| Structural Only | Struct only | 0.65-0.70 |
| Random | - | 0.50 |

**Analysis:**
- Compute accuracy drop when each component removed
- Identify most critical components
- Inform optimization priorities

### 4.3 Weight Sensitivity Analysis

Test different scoring weight configurations:

| Config | w_struct | w_causal | w_semantic | w_intervention | Expected Accuracy |
|--------|----------|----------|------------|----------------|-------------------|
| Default | 0.3 | 0.3 | 0.2 | 0.2 | ≥ 0.80 |
| Causal-Heavy | 0.2 | 0.5 | 0.15 | 0.15 | 0.78-0.82 |
| Structural-Heavy | 0.5 | 0.2 | 0.15 | 0.15 | 0.75-0.80 |
| Balanced | 0.25 | 0.25 | 0.25 | 0.25 | 0.77-0.81 |

**Optimization:** Grid search for optimal weights on training set

---

## 5. Failure Analysis

### 5.1 Error Categories

Categorize failures by type:

1. **False Positives (FP)**
   - I_mech predicts isomorphism, but domains not actually isomorphic
   - **Concern:** Could lead to invalid solution transfers

2. **False Negatives (FN)**
   - I_mech predicts non-isomorphic, but domains actually isomorphic
   - **Concern:** Missed opportunities for analogy

3. **Borderline Cases**
   - Similarity scores near threshold (0.65-0.75)
   - **Analysis:** Should threshold be adjusted?

### 5.2 Failure Case Documentation

For each failure, document:

```python
# File: rese/tests/benchmarks/failure_analysis.json

{
  "failures": [
    {
      "case_id": "telegraph_to_radio",
      "predicted_similarity": 0.72,
      "ground_truth": false,
      "error_type": "false_positive",
      "analysis": {
        "why_failed": "Both use electromagnetic waves, but causal mechanisms differ (signal propagation vs radiation)",
        "component_scores": {
          "structural": 0.65,
          "causal": 0.55,
          "semantic": 0.85,
          "intervention": 0.50
        },
        "root_cause": "Semantic similarity masked mechanistic differences",
        "recommended_fix": "Increase weight of causal and intervention scores"
      }
    },
    ... // more failures
  ]
}
```

### 5.3 Error Analysis Protocol

**Weekly Review:**
1. Collect all failures from validation run
2. Categorize by error type
3. Identify patterns
4. Propose fixes
5. Iterate on algorithm

---

## 6. Performance Benchmarks

### 6.1 Computational Performance

**Test Domains:**
- Small: 10-50 nodes
- Medium: 50-500 nodes
- Large: 500-2000 nodes
- Very Large: 2000-10000 nodes

**Metrics:**
- Extraction time (domain → FDG)
- Similarity computation time
- Proof generation time (if enabled)
- Total time

**Targets:**

| Domain Size | Extraction | Similarity | Proof | Total |
|-------------|------------|------------|-------|-------|
| Small (<50) | < 0.1s | < 0.5s | < 1s | < 2s |
| Medium (50-500) | < 1s | < 5s | < 10s | < 20s |
| Large (500-2000) | < 5s | < 30s | < 60s | < 120s |
| Very Large (>2000) | < 30s | < 300s | < 600s | < 1000s |

### 6.2 Scalability Analysis

**Test:** Run I_mech on progressively larger synthetic FDGs

**Measure:** Time complexity, space complexity

**Plot:**
- X-axis: Number of nodes (log scale)
- Y-axis: Time (seconds, log scale)
- Fit to: O(n^α), estimate α

**Target:** α < 2 (quadratic or better)

---

## 7. Validation Timeline

### Week 31: Implementation (see imech_implementation_plan.md)
- Day 1-2: Data structures
- Day 3-4: Isomorphism detection
- Day 5: Causal similarity
- Day 6: Scoring and transfer
- Day 7: Proofs and integration

### Week 32: Validation

**Day 1-2: Dataset Preparation**
- [ ] Finalize historical analogies dataset
- [ ] Generate synthetic analogies
- [ ] Prepare ground truth annotations
- [ ] Set up evaluation infrastructure

**Day 3-4: Initial Validation**
- [ ] Run I_mech on historical analogies (100 cases)
- [ ] Compute primary metrics (transfer success, accuracy, correlation)
- [ ] Compare against baselines
- [ ] Document initial results

**Day 5: Ablation Studies**
- [ ] Run ablation experiments
- [ ] Analyze component contributions
- [ ] Weight sensitivity analysis
- [ ] Optimize hyperparameters

**Day 6: Human Evaluation**
- [ ] Recruit expert panel
- [ ] Conduct expert rating session
- [ ] Compute inter-rater reliability
- [ ] Compare experts vs I_mech

**Day 7: Failure Analysis and Iteration**
- [ ] Categorize all failures
- [ ] Root cause analysis
- [ ] Implement fixes
- [ ] Re-run validation

**End of Week 32 Deliverable:** Validation report with final metrics

---

## 8. Success Criteria Summary

### Primary Success Gates

| Metric | Target | Minimum | Status |
|--------|--------|---------|--------|
| Transfer Success Rate | ≥ 0.80 | ≥ 0.75 | ___ |
| Isomorphism Detection Accuracy | ≥ 0.85 | ≥ 0.80 | ___ |
| Similarity Correlation | ≥ 0.80 | ≥ 0.75 | ___ |
| Expert Agreement (κ) | ≥ 0.70 | ≥ 0.65 | ___ |
| Computational Efficiency | < 10s (medium) | < 20s | ___ |

### Must Pass: At least 4 of 5 primary criteria
### All Minimum Criteria Must Be Met

---

## 9. Validation Report Template

```markdown
# I_mech Validation Report

**Date:** [Week 32]
**Version:** [Implementation version]

## Executive Summary
- [ ] Pass / Fail
- Overall performance: [metrics]

## Primary Metrics
1. Transfer Success Rate: [value] (target: ≥ 0.80)
2. Isomorphism Detection Accuracy: [value] (target: ≥ 0.85)
3. Similarity Score Correlation: [value] (target: ≥ 0.80)
4. Expert Agreement (κ): [value] (target: ≥ 0.70)
5. Computational Efficiency: [value] (target: < 10s)

## Baseline Comparisons
- Random: [accuracy]
- Feature Matching: [accuracy]
- Graph Isomorphism Only: [accuracy]
- SME: [accuracy]
- **I_mech: [accuracy]**

## Ablation Study Results
[Component contribution analysis]

## Failure Analysis
[Failure categorization and proposed fixes]

## Performance Benchmarks
[Timing and scalability results]

## Recommendations
[For integration and future improvements]

## Conclusion
[Pass/Fail determination, next steps]
```

---

## 10. Risk Mitigation

### Risk 1: Low Transfer Success Rate (< 75%)

**Mitigation:**
- Lower threshold for borderline cases (0.7 → 0.65)
- Implement solution repair mechanism
- Increase weight of causal similarity
- Add more training data to scoring model

### Risk 2: High False Positive Rate

**Mitigation:**
- Add additional validation step (simulation)
- Increase intervention weight in scoring
- Require proof verification for high-stakes transfers
- Add confidence intervals with wider bounds

### Risk 3: Computational Performance Too Slow

**Mitigation:**
- Implement aggressive caching
- Use approximation for large graphs
- Parallelize independent computations
- Pre-compute FDGs for common domains

---

## 11. Next Steps After Validation

### If Pass (≥ 4 of 5 primary criteria):
1. Integrate into production OpenEvolve pipeline
2. Deploy to staging environment
3. Begin integration testing with full system
4. Prepare for Week 33-34 user acceptance testing

### If Fail (< 4 primary criteria):
1. Analyze failures
2. Implement targeted improvements
3. Re-run validation (Week 33)
4. Consider fallback to simpler baseline if necessary

---

## 12. Conclusion

This validation strategy provides a rigorous framework for evaluating I_mech against the >80% transfer success target. By combining:

- **Historical analogies** (real-world validation)
- **Synthetic tests** (controlled experiments)
- **Human evaluation** (expert validation)
- **Ablation studies** (component analysis)
- **Performance benchmarks** (scalability)

We ensure comprehensive assessment before deployment.

**Key Success Indicator:** >80% transfer success on historical technology analogies, demonstrating that I_mech can reliably identify and transfer mechanistically isomorphic solutions across domains.

---

**Appendix: Benchmark Data Collection Plan**

**Historical Analogies Sourcing:**
1. Books: "The Innovator's Dilemma", "The Nature of Technology"
2. Patent databases: USPTO, Google Patents
3. Academic papers: History of science and technology
4. Case studies: MIT OpenCourseWare, Harvard Business Review

**Annotation Protocol:**
1. Two independent researchers annotate each case
2. Resolve discrepancies through consensus
3. Third researcher adjudicates if needed
4. Compute inter-annotator agreement (target κ ≥ 0.8)

**Data Quality Assurance:**
- Verify each analogy has documented historical evidence
- Ensure FDGs are accurately extracted from domain descriptions
- Validate ground truth similarity scores with domain experts
