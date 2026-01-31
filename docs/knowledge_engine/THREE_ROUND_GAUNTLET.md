# 3-Round Gauntlet Orchestrator Documentation

## Overview

The 3-Round Gauntlet Orchestrator implements a progressive, multi-stage evaluation system that combines the speed of AI-based evaluation with the rigor of adversarial testing and consensus verification. This document provides comprehensive guidance on using, configuring, and extending the system.

## Table of Contents

1. [Architecture](#architecture)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Domain-Specific Configurations](#domain-specific-configurations)
6. [API Reference](#api-reference)
7. [Extending the System](#extending-the-system)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Architecture

### Design Philosophy

The 3-round gauntlet system is built on three core principles:

1. **Progressive Filtering**: Each round acts as a filter, allowing only high-quality solutions to proceed. This saves computational resources by terminating evaluations early for poor solutions.

2. **Weighted Scoring**: Each round contributes to the final score based on its configured weight, enabling nuanced assessment across multiple dimensions.

3. **Configurable Rigor**: Thresholds and weights can be adjusted per domain, ensuring appropriate standards for different use cases.

### Evaluation Flow

```
Solution Input
    ↓
┌─────────────────────────────────────────┐
│ Round 1: LoongFlow AI Evaluation       │
│ - Weight: 20%                           │
│ - Target: <30 seconds                   │
│ - Threshold: 0.5 (configurable)         │
│ Purpose: Quick quality screen           │
└────────────────┬────────────────────────┘
                 │
                 ↓
            Pass? (Score ≥ Threshold)
                 │
        No ──────┴────── Yes
        ↓                   ↓
    TERMINATE      ┌─────────────────────────────────────────┐
                   │ Round 2: Red Team Adversarial          │
                   │ - Weight: 30%                           │
                   │ - Target: <2 minutes                    │
                   │ - Threshold: 0.6 (configurable)         │
                   │ Purpose: Robustness testing             │
                   └────────────────┬────────────────────────┘
                                    │
                                    ↓
                               Pass?
                                    │
                           No ──────┴────── Yes
                           ↓                   ↓
                       TERMINATE      ┌─────────────────────────────────────────┐
                                      │ Round 3: Gold Team Consensus           │
                                      │ - Weight: 50%                           │
                                      │ - Target: <5 minutes                    │
                                      │ - Threshold: 0.7 (configurable)         │
                                      │ Purpose: Multi-model verification       │
                                      └────────────────┬────────────────────────┘
                                                       │
                                                       ↓
                                                  Final Score
                                            (Weighted aggregate)
```

### Round Details

#### Round 1: LoongFlow AI Evaluation

**Purpose**: Fast AI-based quality assessment to filter obvious failures

**Characteristics**:
- Single-pass evaluation using LoongFlow's GeneralEvaluator
- Returns score (0.0-1.0+), confidence, and qualitative feedback
- Typical execution time: 10-30 seconds
- Evaluates: correctness, quality, completeness

**Advantages**:
- Extremely fast compared to full evaluation
- Consistent scoring with calibrated AI models
- Provides actionable feedback for improvement
- Low computational cost

**When to use**: Always enabled as the first line of defense

---

#### Round 2: Red Team Adversarial Evaluation

**Purpose**: Stress-test solutions against adversarial attacks and edge cases

**Characteristics**:
- Multiple attack vectors tailored to domain
- Returns robustness score and attack success rate
- Typical execution time: 1-2 minutes
- Evaluates: resilience, error handling, edge cases

**Attack Vectors** (examples):
- **Finance**: Market crashes, liquidity crises, extreme volatility
- **Science**: Outlier sensitivity, noise resistance, parameter variations
- **Engineering**: Safety violations, load limits, environmental stressors
- **Web**: SQL injection, XSS, rate limiting, accessibility

**Advantages**:
- Finds hidden vulnerabilities
- Tests error handling and recovery
- Validates solution robustness
- Realistic failure scenario testing

**When to use**: Critical for production systems, safety-critical applications

---

#### Round 3: Gold Team Consensus Verification

**Purpose**: Multi-evaluator consensus for high-confidence assessment

**Characteristics**:
- Multiple specialized evaluators (domain experts)
- Consensus scoring and voting mechanisms
- Optional formal verification (Lean 4 for mathematics)
- Typical execution time: 3-5 minutes
- Evaluates: overall quality, consensus, formal correctness

**Evaluators** (examples):
- **Finance**: Financial analyst, risk manager, quant researcher, compliance
- **Science**: Domain expert, methodology reviewer, statistician
- **Engineering**: Safety engineer, structural analyst, materials expert
- **Web**: UX designer, frontend/backend engineers, accessibility specialist

**Advantages**:
- Reduces individual evaluator bias
- Provides comprehensive assessment
- High confidence in final score
- Optional formal verification for mathematical correctness

**When to use**: Final gate for high-stakes deployments, research publication

---

## Quick Start

### Installation

The orchestrator is part of the OpenEvolve gauntlet system:

```bash
# Already included in openevolve/gauntlets/
# No additional installation required
```

### Basic Usage

```python
from openevolve.gauntlets.three_round_orchestrator import (
    ThreeRoundGauntletOrchestrator,
    create_balanced_config
)

# Create configuration
config = create_balanced_config()

# Initialize orchestrator
orchestrator = ThreeRoundGauntletOrchestrator(config=config)

# Run evaluation
result = await orchestrator.run_full_gauntlet(
    solution="def solve(): ...",
    problem="Optimize portfolio allocation",
    domain="finance"
)

# Check results
print(f"Passed: {result.passed}")
print(f"Final Score: {result.final_score:.3f}")
print(f"Rounds Completed: {result.rounds_completed}")
print(result.comprehensive_report)
```

### With Domain-Specific Configuration

```python
from openevolve.gauntlets.three_round_orchestrator import create_domain_config

# Get domain-tuned configuration
config = create_domain_config('finance')  # or 'science', 'web', etc.

orchestrator = ThreeRoundGauntletOrchestrator(config=config)

result = await orchestrator.run_full_gauntlet(
    solution=trading_strategy_code,
    problem="Develop momentum-based trading strategy",
    domain="finance"
)
```

---

## Configuration

### ThreeRoundConfig Parameters

```python
@dataclass
class ThreeRoundConfig:
    # Round 1 (LoongFlow)
    round1_config: Dict[str, Any]         # LLM config, timeout, etc.
    round1_weight: float = 0.2            # Weight in final score
    round1_threshold: float = 0.5         # Min score to proceed
    round1_enabled: bool = True           # Enable/disable round

    # Round 2 (Red Team)
    round2_config: Dict[str, Any]         # Attack vectors, intensity
    round2_weight: float = 0.3
    round2_threshold: float = 0.6
    round2_enabled: bool = True

    # Round 3 (Gold Team)
    round3_config: Dict[str, Any]         # Evaluators, consensus threshold
    round3_weight: float = 0.5
    round3_threshold: float = 0.7
    round3_enabled: bool = True

    # Global settings
    enable_early_termination: bool = True     # Stop if fails early
    enable_parallel_execution: bool = False   # Parallel rounds (experimental)
    aggregate_artifacts: bool = True          # Collect artifacts
    generate_detailed_report: bool = True     # Generate report
```

### Configuration Examples

#### Strict Configuration (High-Stakes)

```python
strict_config = ThreeRoundConfig(
    round1_threshold=0.7,
    round2_threshold=0.8,
    round3_threshold=0.9,
    enable_early_termination=True
)
```

**Use Cases**:
- Financial trading systems
- Medical/Pharma applications
- Safety-critical engineering
- Production deployments

#### Lenient Configuration (Exploration)

```python
lenient_config = ThreeRoundConfig(
    round1_threshold=0.3,
    round2_threshold=0.5,
    round3_threshold=0.6,
    enable_early_termination=False  # Run all rounds for feedback
)
```

**Use Cases**:
- Research and development
- Prototyping and experimentation
- Learning and education
- Low-stakes applications

#### Balanced Configuration (Default)

```python
balanced_config = ThreeRoundConfig(
    round1_threshold=0.5,
    round2_threshold=0.6,
    round3_threshold=0.7,
    enable_early_termination=True
)
```

**Use Cases**:
- General-purpose development
- Standard business applications
- Most domains requiring quality assurance

### Round 1 Configuration (LoongFlow)

```python
round1_config = {
    'llm_config': {
        'model': 'claude-3-5-sonnet-20241022',
        'api_key': os.getenv('ANTHROPIC_API_KEY'),
        'url': 'http://localhost:8001',
        'temperature': 0.3,  # Lower = more consistent
        'max_tokens': 4096
    },
    'timeout': 60,  # seconds
    'domain': 'general'
}
```

**Temperature Guide**:
- `0.1-0.3`: High consistency (finance, safety-critical)
- `0.4-0.6`: Balanced (science, engineering)
- `0.7-0.9`: More creative (web, prototyping)

### Round 2 Configuration (Red Team)

```python
round2_config = {
    'attack_vectors': [
        'edge_case_scenarios',
        'performance_stress',
        'data_corruption'
    ],
    'attack_intensity': 'moderate',  # 'low', 'moderate', 'high', 'extreme'
    'timeout': 120
}
```

**Attack Intensity Guide**:
- `low`: Basic edge cases, quick testing
- `moderate`: Standard adversarial testing
- `high`: Aggressive attack scenarios
- `extreme`: Worst-case scenarios, safety validation

### Round 3 Configuration (Gold Team)

```python
round3_config = {
    'evaluators': [
        'domain_expert',
        'methodology_reviewer',
        'quality_assurance'
    ],
    'consensus_threshold': 0.75,  # Agreement level required
    'formal_verification': False,  # Lean 4 verification (math only)
    'timeout': 300
}
```

---

## Usage Examples

### Example 1: Finance Trading Strategy

```python
from openevolve.gauntlets.three_round_orchestrator import ThreeRoundGauntletOrchestrator
from examples.gauntlet_configs.finance_config import get_finance_config

# Get trading-specific configuration
config = get_finance_config('trading')

orchestrator = ThreeRoundGauntletOrchestrator(config=config)

# Evaluate trading strategy
trading_strategy = """
def momentum_strategy(returns, lookback=20):
    import numpy as np

    # Calculate momentum
    momentum = returns.rolling(lookback).mean()

    # Generate signals
    signals = np.where(momentum > 0, 1, -1)

    return signals
"""

result = await orchestrator.run_full_gauntlet(
    solution=trading_strategy,
    problem="Develop momentum-based trading strategy",
    domain="finance"
)

if result.passed:
    print(f"✓ Strategy approved with score {result.final_score:.3f}")
    print(f"  Round 1: {result.round1_result.score:.3f}")
    print(f"  Round 2: {result.round2_result.score:.3f}")
    print(f"  Round 3: {result.round3_result.score:.3f}")
else:
    print(f"✗ Strategy failed: {result.termination_reason}")
    print(f"  Completed {result.rounds_completed} rounds")
```

### Example 2: Scientific Experimental Design

```python
from examples.gauntlet_configs.science_config import get_science_config

config = get_science_config('experimental_design')
orchestrator = ThreeRoundGauntletOrchestrator(config=config)

experimental_protocol = """
# Experimental Design: Drug Efficacy Trial

1. Sample Size Calculation
   - Power analysis: 80% power, α=0.05
   - Effect size: d=0.5 (medium)
   - Required n = 128 per group

2. Randomization
   - Block randomization (block size=8)
   - Stratified by age and severity

3. Control Measures
   - Double-blind protocol
   - Placebo control group
   - Standardized outcome measures

4. Analysis Plan
   - Primary endpoint: symptom reduction at 4 weeks
   - Secondary endpoints: quality of life, adverse events
   - Statistical test: ANCOVA with baseline adjustment
"""

result = await orchestrator.run_full_gauntlet(
    solution=experimental_protocol,
    problem="Design randomized controlled trial for new drug",
    domain="science"
)

print(result.comprehensive_report)
```

### Example 3: Web Frontend Component

```python
from examples.gauntlet_configs.web_config import get_web_config

config = get_web_config('frontend')
orchestrator = ThreeRoundGauntletOrchestrator(config=config)

react_component = """
import React, { useState, useEffect } from 'react';

const UserProfile = ({ userId }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchUser(userId)
      .then(data => {
        setUser(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [userId]);

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <div className="user-profile">
      <Avatar src={user.avatar} alt={user.name} />
      <h2>{user.name}</h2>
      <p>{user.bio}</p>
    </div>
  );
};
"""

result = await orchestrator.run_full_gauntlet(
    solution=react_component,
    problem="Create user profile component with error handling",
    domain="web"
)

# Even with lenient config, get detailed feedback
if not result.passed:
    print("Areas for improvement:")
    if result.round1_result:
        print(f"  R1: {result.round1_result.feedback}")
    if result.round2_result:
        print(f"  R2: {result.round2_result.feedback}")
```

### Example 4: Custom Configuration

```python
from openevolve.gauntlets.three_round_orchestrator import ThreeRoundConfig

# Create custom configuration
custom_config = ThreeRoundConfig(
    # Round 1: Faster but more permissive
    round1_config={
        'llm_config': {
            'model': 'claude-3-5-sonnet-20241022',
            'temperature': 0.5
        },
        'timeout': 30  # Quick evaluation
    },
    round1_weight=0.15,
    round1_threshold=0.4,

    # Round 2: Emphasize robustness
    round2_config={
        'attack_vectors': ['performance', 'security', 'accessibility'],
        'attack_intensity': 'high'
    },
    round2_weight=0.35,  # Higher weight for robustness
    round2_threshold=0.7,

    # Round 3: Standard consensus
    round3_config={
        'evaluators': ['senior_developer', 'ux_reviewer', 'qa_engineer'],
        'consensus_threshold': 0.8
    },
    round3_weight=0.5,
    round3_threshold=0.75,

    # Run all rounds even on early failures (for learning)
    enable_early_termination=False
)

orchestrator = ThreeRoundGauntletOrchestrator(config=custom_config)
```

---

## Domain-Specific Configurations

### Available Domains

#### Finance
```python
from examples.gauntlet_configs.finance_config import get_finance_config

# Sub-domains: 'general', 'trading', 'risk'
config = get_finance_config('trading')
```

**Characteristics**:
- High thresholds (0.7-0.9)
- Aggressive adversarial testing
- Multiple specialized evaluators
- Early termination enabled

#### Science
```python
from examples.gauntlet_configs.science_config import get_science_config

# Sub-domains: 'general', 'experimental_design', 'data_analysis'
config = get_science_config('experimental_design')
```

**Characteristics**:
- Moderate thresholds (0.5-0.7)
- Methodological rigor focus
- Statistical validation
- Peer review style evaluation

#### Web
```python
from examples.gauntlet_configs.web_config import get_web_config

# Sub-domains: 'general', 'frontend', 'backend'
config = get_web_config('frontend')
```

**Characteristics**:
- Low thresholds (0.3-0.6)
- Focus on UX and functionality
- Accessibility consideration
- Early termination disabled (feedback focus)

### Creating Custom Domain Configurations

```python
from openevolve.gauntlets.three_round_orchestrator import ThreeRoundConfig

# Define domain configuration
MEDICAL_CONFIG = ThreeRoundConfig(
    round1_config={'llm_config': {'temperature': 0.1}, 'timeout': 90},
    round1_threshold=0.8,  # Very high
    round2_config={'attack_intensity': 'extreme'},
    round2_threshold=0.9,
    round3_config={
        'evaluators': [
            'medical_professional',
            'clinical_researcher',
            'regulatory_specialist',
            'ethics_committee'
        ],
        'consensus_threshold': 0.95
    },
    round3_threshold=0.95,  # Near-perfect required
    enable_early_termination=True
)
```

---

## API Reference

### Classes

#### ThreeRoundGauntletOrchestrator

Main orchestrator class for 3-round gauntlet evaluation.

**Constructor**:
```python
ThreeRoundGauntletOrchestrator(config: ThreeRoundConfig)
```

**Methods**:

```python
async def run_full_gauntlet(
    self,
    solution: str,
    problem: str,
    domain: str
) -> FullGauntletResult
```
Run complete 3-round gauntlet evaluation.

**Parameters**:
- `solution`: Solution code/text to evaluate
- `problem`: Problem statement or requirements
- `domain`: Application domain (affects evaluator behavior)

**Returns**: `FullGauntletResult` with complete evaluation results

---

```python
async def run_round1(
    self,
    solution: str,
    problem: str,
    domain: str
) -> Round1Result
```
Execute Round 1 (LoongFlow evaluation) only.

---

```python
async def run_round2(
    self,
    solution: str,
    problem: str,
    domain: str
) -> Round2Result
```
Execute Round 2 (Red Team evaluation) only.

---

```python
async def run_round3(
    self,
    solution: str,
    problem: str,
    domain: str
) -> Round3Result
```
Execute Round 3 (Gold Team evaluation) only.

---

```python
def should_continue_to_round2(
    self,
    round1_result: Round1Result
) -> bool
```
Determine if solution should proceed to Round 2.

---

```python
def should_continue_to_round3(
    self,
    round2_result: Round2Result
) -> bool
```
Determine if solution should proceed to Round 3.

---

```python
def calculate_final_score(
    self,
    round1: Optional[Round1Result],
    round2: Optional[Round2Result],
    round3: Optional[Round3Result]
) -> float
```
Calculate weighted aggregate final score.

---

```python
def generate_comprehensive_report(
    self,
    full_result: FullGauntletResult
) -> str
```
Generate comprehensive evaluation report.

### Data Classes

#### FullGauntletResult

Complete result from gauntlet evaluation.

**Attributes**:
- `solution` (str): Evaluated solution
- `problem` (str): Problem statement
- `domain` (str): Domain
- `round1_result` (Optional[Round1Result]): Round 1 results
- `round2_result` (Optional[Round2Result]): Round 2 results
- `round3_result` (Optional[Round3Result]): Round 3 results
- `passed` (bool): Whether solution passed all attempted rounds
- `final_score` (float): Weighted aggregate score
- `rounds_completed` (int): Number of rounds completed (1, 2, or 3)
- `termination_reason` (Optional[str]): Reason for early termination
- `artifacts_from_all_rounds` (List[Any]): Collected artifacts
- `total_time` (float): Total evaluation time (seconds)
- `comprehensive_report` (str): Generated report

#### Round1Result

Round 1 (LoongFlow) results.

**Attributes**:
- `passed` (bool): Passed/failed
- `score` (float): Achieved score
- `confidence` (float): Evaluation confidence
- `evaluation_time` (float): Execution time
- `feedback` (str): Feedback text
- `artifacts` (List[Any]): Artifacts
- `evaluator_type` (str): Evaluator used

#### Round2Result

Round 2 (Red Team) results.

**Attributes**:
- `passed` (bool): Survived attacks
- `score` (float): Achieved score
- `attacks_attempted` (int): Number of attacks
- `attacks_successful` (int): Successful attacks
- `robustness_score` (float): Robustness measure
- `evaluation_time` (float): Execution time
- `feedback` (str): Feedback text
- `artifacts` (List[Any]): Artifacts
- `attack_details` (List[Dict]): Attack details

#### Round3Result

Round 3 (Gold Team) results.

**Attributes**:
- `passed` (bool): Achieved consensus
- `score` (float): Final score
- `consensus_score` (float): Agreement level
- `formal_verification_passed` (bool): Lean 4 verification
- `evaluation_time` (float): Execution time
- `feedback` (str): Feedback text
- `artifacts` (List[Any]): Artifacts
- `evaluator_votes` (List[Dict]): Individual votes

### Factory Functions

```python
def create_strict_config() -> ThreeRoundConfig
```
Create strict configuration for high-stakes domains.

```python
def create_lenient_config() -> ThreeRoundConfig
```
Create lenient configuration for exploration.

```python
def create_balanced_config() -> ThreeRoundConfig
```
Create balanced configuration for general use.

```python
def create_domain_config(domain: str) -> ThreeRoundConfig
```
Create domain-specific configuration.

---

## Extending the System

### Adding Custom Evaluators

#### Custom Round 1 Evaluator

```python
from evaluators.loongflow_adapter import LoongFlowEvaluatorAdapter

class CustomLoongFlowEvaluator(LoongFlowEvaluatorAdapter):
    async def evaluate_round(self, solution, round_rule, context):
        # Custom pre-processing
        solution = self.preprocess(solution)

        # Run standard evaluation
        result = await super().evaluate_round(
            solution, round_rule, context
        )

        # Custom post-processing
        result.score = self.adjust_score(result.score, context)

        return result

    def preprocess(self, solution):
        # Custom preprocessing logic
        return solution

    def adjust_score(self, score, context):
        # Custom score adjustment
        if context.get('domain') == 'high_risk':
            return score * 0.9  # Penalize for high-risk domain
        return score
```

#### Custom Round 2 Evaluator

```python
class CustomRedTeamEvaluator:
    async def evaluate(self, solution, problem, domain, config):
        attacks_attempted = 0
        attacks_successful = 0
        attack_details = []

        for attack_vector in config.get('attack_vectors', []):
            attacks_attempted += 1

            result = await self.run_attack(
                solution, attack_vector, domain
            )

            if result['successful']:
                attacks_successful += 1

            attack_details.append(result)

        robustness_score = 1.0 - (attacks_successful / attacks_attempted)
        score = robustness_score

        return Round2Result(
            passed=score >= config.get('threshold', 0.6),
            score=score,
            attacks_attempted=attacks_attempted,
            attacks_successful=attacks_successful,
            robustness_score=robustness_score,
            evaluation_time=result['time'],
            feedback=self.generate_feedback(attack_details),
            artifacts=[attack_details],
            attack_details=attack_details
        )
```

#### Custom Round 3 Evaluator

```python
class CustomGoldTeamEvaluator:
    async def evaluate(self, solution, problem, domain, config):
        evaluator_votes = []

        for evaluator_name in config.get('evaluators', []):
            vote = await self.get_evaluator_vote(
                solution, evaluator_name, domain
            )
            evaluator_votes.append(vote)

        # Calculate consensus
        scores = [v['score'] for v in evaluator_votes]
        consensus_score = self.calculate_consensus(scores)

        # Formal verification if applicable
        formal_pass = False
        if config.get('formal_verification'):
            formal_pass = await self.run_formal_verification(solution)

        score = consensus_score

        return Round3Result(
            passed=score >= config.get('threshold', 0.7),
            score=score,
            consensus_score=consensus_score,
            formal_verification_passed=formal_pass,
            evaluation_time=sum(v['time'] for v in evaluator_votes),
            feedback=self.generate_feedback(evaluator_votes),
            artifacts=[evaluator_votes],
            evaluator_votes=evaluator_votes
        )
```

### Integration with Existing Systems

#### Integration with BubbleLab

```python
from BubbleLab.services.openevolve_api.api.gauntlets import GauntletSystem
from openevolve.gauntlets.three_round_orchestrator import ThreeRoundGauntletOrchestrator

class EnhancedGauntletSystem(GauntletSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.three_round_orchestrator = ThreeRoundGauntletOrchestrator(
            config=create_balanced_config()
        )

    async def execute_gauntlet(self, solution, problem, domain):
        # Use 3-round orchestrator
        result = await self.three_round_orchestrator.run_full_gauntlet(
            solution=solution,
            problem=problem,
            domain=domain
        )

        # Convert to BubbleLab format
        return self.to_bubblelab_format(result)
```

---

## Best Practices

### 1. Choose Appropriate Thresholds

**Guidelines**:
- **High-Stakes** (Finance, Medical, Safety): 0.7-0.9
- **Moderate** (Science, Engineering): 0.5-0.7
- **Low-Stakes** (Web, Prototyping): 0.3-0.6

### 2. Configure Weights Based on Priorities

```python
# Emphasize robustness (Round 2)
config = ThreeRoundConfig(
    round1_weight=0.15,
    round2_weight=0.45,  # Higher weight
    round3_weight=0.40
)

# Emphasize consensus (Round 3)
config = ThreeRoundConfig(
    round1_weight=0.15,
    round2_weight=0.25,
    round3_weight=0.60   # Highest weight
)
```

### 3. Use Early Termination for Efficiency

```python
# Enable for production (save compute)
config.enable_early_termination = True

# Disable for development (get feedback)
config.enable_early_termination = False
```

### 4. Set Appropriate Timeouts

```python
# Quick evaluations (prototyping)
round1_config = {'timeout': 30}
round2_config = {'timeout': 60}
round3_config = {'timeout': 120}

# Thorough evaluations (production)
round1_config = {'timeout': 90}
round2_config = {'timeout': 180}
round3_config = {'timeout': 300}
```

### 5. Analyze Comprehensive Reports

Always review the detailed reports to understand:

1. **Why solutions failed**: Check feedback from each round
2. **Attack patterns**: Review Round 2 attack details
3. **Consensus breakdown**: Examine Round 3 individual votes
4. **Performance**: Check evaluation times for optimization

### 6. Iterate Based on Feedback

```python
result = await orchestrator.run_full_gauntlet(...)

if not result.passed:
    # Extract feedback
    feedback = []
    if result.round1_result:
        feedback.append(result.round1_result.feedback)
    if result.round2_result:
        feedback.append(result.round2_result.feedback)

    # Use feedback to improve solution
    improved_solution = improve_solution(solution, feedback)

    # Re-evaluate
    new_result = await orchestrator.run_full_gauntlet(
        solution=improved_solution,
        problem=problem,
        domain=domain
    )
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Round 1 Evaluator Not Initialized

**Symptom**: `Round 1 evaluator not initialized` error

**Solution**:
```python
# Check LoongFlow is available
import sys
import os

loongflow_path = os.path.join(os.path.dirname(__file__), '..', 'LoongFlow')
if os.path.exists(loongflow_path):
    sys.path.insert(0, loongflow_path)

# Or use fallback mode
config = ThreeRoundConfig(
    round1_enabled=False  # Disable Round 1
)
```

#### Issue 2: Evaluation Timeout

**Symptom**: Evaluations taking too long

**Solution**:
```python
# Increase timeouts
config.round1_config['timeout'] = 120
config.round2_config['timeout'] = 180
config.round3_config['timeout'] = 300

# Or simplify evaluation
config.round1_config['llm_config']['max_tokens'] = 2048
```

#### Issue 3: Memory Issues

**Symptom**: Out of memory errors

**Solution**:
```python
# Disable artifact aggregation
config.aggregate_artifacts = False

# Reduce batch sizes
# Process solutions one at a time instead of batches
```

#### Issue 4: All Solutions Failing Round 1

**Symptom**: Zero pass rate at Round 1

**Solution**:
```python
# Lower threshold
config.round1_threshold = 0.4

# Or adjust temperature for more lenient evaluation
config.round1_config['llm_config']['temperature'] = 0.5
```

#### Issue 5: Poor Consensus in Round 3

**Symptom**: Low consensus scores even for good solutions

**Solution**:
```python
# Reduce consensus threshold
config.round3_config['consensus_threshold'] = 0.6

# Or adjust evaluators
config.round3_config['evaluators'] = [
    'general_expert',  # More generalist
    'specialist'       # Fewer specialized evaluators
]
```

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('openevolve.gauntlets')
logger.setLevel(logging.DEBUG)
```

### Performance Profiling

Track evaluation times:

```python
import time

result = await orchestrator.run_full_gauntlet(...)

print(f"Total time: {result.total_time:.2f}s")
if result.round1_result:
    print(f"Round 1: {result.round1_result.evaluation_time:.2f}s")
if result.round2_result:
    print(f"Round 2: {result.round2_result.evaluation_time:.2f}s")
if result.round3_result:
    print(f"Round 3: {result.round3_result.evaluation_time:.2f}s")
```

---

## Contributing

To contribute new evaluators, domain configurations, or improvements:

1. Follow existing code structure
2. Add comprehensive tests
3. Update documentation
4. Submit PR with clear description

---

## License

Part of OpenEvolve Gauntlet System. See main LICENSE file.

---

## Support

For issues, questions, or contributions:
- GitHub: [OpenEvolve repository]
- Documentation: [OpenEvolve docs]
- Discord: [Community server]

---

**Last Updated**: 2026-01-30
**Version**: 1.0.0
