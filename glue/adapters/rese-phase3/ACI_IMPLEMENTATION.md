# ACI Implementation Documentation

## Anomaly Characterization Index (ACI) for RESE Phase III

### Overview

This document describes the complete implementation of the **Anomaly Characterization Index (ACI)** for RESE Phase III (Monte Carlo Refinement) as specified in RESE Technical Manual §5.2.

**Purpose:** The ACI guides MCTS refinement by identifying high-potential signals in experimental data that warrant deeper investigation.

---

## Table of Contents

1. [Specification](#specification)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Usage](#usage)
6. [Integration with MCTS](#integration-with-mcts)
7. [Testing](#testing)
8. [Configuration](#configuration)
9. [CLAUDE.md Compliance](#claudemd-compliance)

---

## Specification

From RESE Technical Manual §5.2:

> "The ACI is a composite measure that guides search refinement"

### Two Components

1. **Disorder Entropy (𝔈_D)**: Measures randomness/uncertainty in time-series data
   - Uses Shannon entropy to quantify disorder
   - Normalized to [0, 1] range
   - 0 = perfectly ordered, 1 = maximum disorder

2. **Causal Coherence (𝔍_C)**: Statistical correlation between high 𝔈_D and input variables
   - Pearson or Spearman correlation
   - Identifies which input variables correlate with high-entropy regions
   - Normalized to [0, 1] range

### Signal Flagging

**High-potential signal** = High 𝔈_D **AND** High 𝔍_C

These signals indicate anomalous behavior worth investigating with MCTS refinement.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ACI Calculator                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Input: Experiment Data (time-series)                │  │
│  │  - Output variable                                   │  │
│  │  - Input variables (X₁, X₂, ..., Xₙ)                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Sliding Window Processing                           │  │
│  │  - Window size: ω (configurable)                    │  │
│  │  - Step size: ω (non-overlapping)                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│           ┌───────────────┴───────────────┐                │
│           ▼                               ▼                │
│  ┌────────────────────┐        ┌────────────────────┐     │
│  │ Calculate 𝔈_D      │        │ Calculate 𝔍_C      │     │
│  │ (Disorder Entropy) │        │ (Causal Coherence) │     │
│  │                    │        │                    │     │
│  │ • Histogram bins   │        │ • Correlation with │     │
│  │ • Shannon entropy  │        │   input variables  │     │
│  │ • Normalize        │        │ • Pearson/Spearman  │     │
│  └────────────────────┘        └────────────────────┘     │
│           │                               │                │
│           └───────────────┬───────────────┘                │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Composite ACI Score                                 │  │
│  │  ACI = (𝔈_D + 𝔍_C) / 2                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Signal Flagging                                     │  │
│  │  IF 𝔈_D ≥ threshold AND 𝔍_C ≥ threshold:            │  │
│  │    → High-potential signal                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Output: ACI Results                                 │  │
│  │  - Disorder entropy values                           │  │
│  │  - Causal coherence values                           │  │
│  │  - ACI scores                                        │  │
│  │  - High-potential signals (flagged)                  │  │
│  │  - Causal variables identified                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. ACIConfig

Configuration class for ACI calculator.

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `PHASE3_ACI_WINDOW_SIZE` | Sliding window size (ω) | 100 |
| `PHASE3_ACI_ENTROPY_BINS` | Bins for entropy histogram | 10 |
| `PHASE3_ACI_COHERENCE_THRESHOLD` | Threshold for high 𝔍_C | 0.5 |
| `PHASE3_ACI_ENTROPY_THRESHOLD` | Threshold for high 𝔈_D | 0.7 |
| `PHASE3_ACI_TIMEOUT_MS` | Calculation timeout | 3000 |
| `PHASE3_ACI_MIN_SAMPLES` | Minimum samples for correlation | 30 |
| `PHASE3_ACI_CORRELATION_METHOD` | Correlation method ('pearson' or 'spearman') | 'pearson' |

**Example:**

```python
from aci_calculator import ACIConfig

config = ACIConfig.from_env()

print(f"Window size: {config.window_size}")
print(f"Entropy threshold: {config.entropy_threshold}")
print(f"Coherence threshold: {config.coherence_threshold}")
```

---

### 2. AnomalyCharacterizationIndex

Main ACI calculator class.

**Key Methods:**

#### `calculate_disorder_entropy(time_series, bins=None)`

Calculate Disorder Entropy (𝔈_D).

**Parameters:**
- `time_series` (np.ndarray): 1D array of time-series values
- `bins` (int, optional): Number of histogram bins

**Returns:**
- `float`: Normalized 𝔈_D in [0, 1]

**Example:**

```python
import numpy as np
from aci_calculator import AnomalyCharacterizationIndex

aci = AnomalyCharacterizationIndex()

# White noise has high entropy
noise = np.random.rand(1000)
entropy = aci.calculate_disorder_entropy(noise)
print(f"Entropy: {entropy:.3f}")  # Should be > 0.7

# Sine wave has low entropy
t = np.arange(1000)
sine = 0.5 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
entropy = aci.calculate_disorder_entropy(sine)
print(f"Entropy: {entropy:.3f}")  # Should be < 0.5
```

---

#### `calculate_causal_coherence(entropy_data, input_variables, threshold=None)`

Calculate Causal Coherence (𝔍_C).

**Parameters:**
- `entropy_data` (np.ndarray): Array of entropy values over time
- `input_variables` (Dict[str, np.ndarray]): Input variable time-series
- `threshold` (float, optional): Threshold for "high" coherence

**Returns:**
- `Tuple[float, List[str]]`: (𝔍_C score, list of high-correlation variables)

**Example:**

```python
import numpy as np
from aci_calculator import AnomalyCharacterizationIndex

aci = AnomalyCharacterizationIndex()

# Create correlated data
entropy = np.linspace(0, 1, 100)
var1 = entropy * 0.8  # Correlated
var2 = np.random.rand(100)  # Uncorrelated

coherence, causal_vars = aci.calculate_causal_coherence(
    entropy,
    {'var1': var1, 'var2': var2}
)

print(f"Coherence: {coherence:.3f}")
print(f"Causal variables: {causal_vars}")  # Should include 'var1'
```

---

#### `detect_high_entropy_signals(experiment_data, time_series_key='output')`

Detect high-potential signals using sliding window.

**Parameters:**
- `experiment_data` (Dict[str, np.ndarray]): Experimental time-series data
- `time_series_key` (str): Key for output variable
- `correlation_id` (str, optional): Distributed tracing ID

**Returns:**
- `List[ACIResult]`: Detected signals with ACI metrics

**Example:**

```python
import numpy as np
from aci_calculator import AnomalyCharacterizationIndex

aci = AnomalyCharacterizationIndex()

# Generate experimental data
length = 1000
data = {
    'output': np.random.rand(length),
    'temperature': np.random.rand(length),
    'pressure': np.random.rand(length),
}

# Detect signals
results = aci.detect_high_entropy_signals(data, time_series_key='output')

# Process results
for result in results:
    if result.is_high_entropy_signal:
        print(f"High-entropy signal detected:")
        print(f"  Window: {result.window_start_idx} - {result.window_end_idx}")
        print(f"  Entropy: {result.disorder_entropy:.3f}")
        print(f"  Coherence: {result.causal_coherence:.3f}")
        print(f"  ACI Score: {result.aci_score:.3f}")
        print(f"  Causal variables: {result.causal_variables}")
```

---

#### `get_high_priority_signals(aci_results, top_n=None)`

Get high-priority signals for MCTS exploration.

**Parameters:**
- `aci_results` (List[ACIResult]): List of ACI results
- `top_n` (int, optional): Return top N signals

**Returns:**
- `List[ACIResult]`: High-priority signals, sorted by ACI score

**Example:**

```python
# Get top 5 high-priority signals
high_priority = aci.get_high_priority_signals(results, top_n=5)

for signal in high_priority:
    print(f"Signal ACI: {signal.aci_score:.3f}")
    print(f"  Entropy: {signal.disorder_entropy:.3f}")
    print(f"  Coherence: {signal.causal_coherence:.3f}")
```

---

#### `calculate_aci_reduction(initial_aci, final_aci)`

Calculate ACI reduction (measure of paradigm improvement).

**Parameters:**
- `initial_aci` (float): ACI before intervention
- `final_aci` (float): ACI after intervention

**Returns:**
- `float`: Percentage reduction (0-100)

**Example:**

```python
reduction = aci.calculate_aci_reduction(initial_aci=0.8, final_aci=0.4)
print(f"ACI reduction: {reduction:.1f}%")  # 50.0%
```

---

### 3. ACIResult

Data class for ACI calculation results.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `disorder_entropy` | float | 𝔈_D value [0, 1] |
| `causal_coherence` | float | 𝔍_C value [0, 1] |
| `aci_score` | float | Composite score [0, 1] |
| `is_high_entropy_signal` | bool | High-potential flag |
| `causal_variables` | List[str] | Variables with high 𝔍_C |
| `correlation_id` | str | Distributed tracing ID |
| `timestamp` | str | UTC ISO-8601 timestamp |
| `window_start_idx` | int | Window start index |
| `window_end_idx` | int | Window end index |
| `metadata` | Dict[str, Any] | Additional metadata |

**Example:**

```python
result = ACIResult(
    disorder_entropy=0.8,
    causal_coherence=0.7,
    aci_score=0.75,
    is_high_entropy_signal=True,
    causal_variables=['temperature', 'pressure'],
    correlation_id='test-123',
    timestamp='2026-02-04T12:00:00Z',
    window_start_idx=0,
    window_end_idx=100,
)

# Serialize to dict
result_dict = result.to_dict()

# Deserialize from dict
result2 = ACIResult.from_dict(result_dict)
```

---

### 4. SyntheticDataGenerator

Generate synthetic experimental data for testing.

**Methods:**

- `generate_constant_signal(length)`: Constant signal (zero entropy)
- `generate_sine_wave(length, frequency)`: Periodic signal (low entropy)
- `generate_random_walk(length)`: Random walk (medium entropy)
- `generate_white_noise(length)`: White noise (high entropy)
- `generate_multi_variable_experiment(length, num_variables)`: Multi-variable data with causal relationships

**Example:**

```python
from aci_calculator import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)

# Generate different signal types
constant = generator.generate_constant_signal(1000)
sine = generator.generate_sine_wave(1000, frequency=0.1)
noise = generator.generate_white_noise(1000)

# Generate multi-variable experiment
data = generator.generate_multi_variable_experiment(length=1000, num_variables=5)

# Use for ACI testing
aci = AnomalyCharacterizationIndex()
results = aci.detect_high_entropy_signals(data, time_series_key='output')
```

---

## Mathematical Foundation

### Disorder Entropy (𝔈_D)

**Shannon Entropy:**

$$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

Where:
- $P(x_i)$ = Probability of value in bin $i$
- $n$ = Number of bins

**Normalization:**

$$\mathfrak{E}_D = \frac{H(X)}{\log_2(\text{bins})}$$

Normalized to [0, 1] for comparison across different bin counts.

---

### Causal Coherence (𝔍_C)

**Pearson Correlation:**

$$\rho = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}$$

**Spearman Correlation:**

$$\rho_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

Where $d_i$ = rank difference.

**Causal Coherence:**

$$\mathfrak{C}_C = \max(\{|\rho_i| : \rho_i \text{ is significant}\})$$

Maximum significant correlation across all input variables.

---

### Composite ACI Score

$$\text{ACI} = \frac{\mathfrak{E}_D + \mathfrak{C}_C}{2}$$

Average of normalized entropy and coherence.

---

### Signal Flagging

$$\text{High-Potential} = (\mathfrak{E}_D \geq \tau_E) \land (\mathfrak{C}_C \geq \tau_C)$$

Where:
- $\tau_E$ = Entropy threshold (default: 0.7)
- $\tau_C$ = Coherence threshold (default: 0.5)

---

## Usage

### Basic Usage

```python
from aci_calculator import AnomalyCharacterizationIndex
import numpy as np

# Initialize ACI calculator
aci = AnomalyCharacterizationIndex()

# Prepare experimental data
data = {
    'output': np.random.rand(1000),
    'var1': np.random.rand(1000),
    'var2': np.random.rand(1000),
}

# Detect high-entropy signals
results = aci.detect_high_entropy_signals(data, time_series_key='output')

# Process results
for result in results:
    if result.is_high_entropy_signal:
        print(f"High-potential signal found at window {result.window_start_idx}")
        print(f"  ACI: {result.aci_score:.3f}")
        print(f"  Causal variables: {result.causal_variables}")
```

---

### Integration with MCTS

```python
from phase3_executor import MCTSSearchExecutor, Phase3Config
from aci_calculator import AnomalyCharacterizationIndex
import numpy as np

# Initialize Phase III executor
config = Phase3Config.from_env()
config.aci_enabled = True  # Enable ACI
executor = MCTSSearchExecutor(config)

# Initialize ACI calculator
aci = AnomalyCharacterizationIndex()

# Analyze experimental data to identify high-potential regions
experiment_data = {
    'output': load_time_series(),
    'temperature': load_temperature_data(),
    'pressure': load_pressure_data(),
}

# Detect high-entropy signals
aci_results = aci.detect_high_entropy_signals(experiment_data)

# Get high-priority signals for MCTS
high_priority = aci.get_high_priority_signals(aci_results, top_n=10)

# Use ACI results to guide MCTS exploration
# For example, adjust exploration strategy based on high-entropy regions
for signal in high_priority:
    # Focus MCTS search on regions with high ACI
    print(f"Exploring region {signal.window_start_idx}-{signal.window_end_idx}")
    print(f"  Priority: {signal.aci_score:.3f}")
    print(f"  Causal variables: {signal.causal_variables}")
```

---

### ACI Reduction Calculation (Phase IV Validation)

```python
from aci_calculator import AnomalyCharacterizationIndex

aci = AnomalyCharacterizationIndex()

# Calculate ACI before paradigm intervention
initial_results = aci.detect_high_entropy_signals(initial_data)
initial_aci = np.mean([r.aci_score for r in initial_results])

# Apply paradigm intervention
# ... (run experiment with new paradigm)

# Calculate ACI after intervention
final_results = aci.detect_high_entropy_signals(final_data)
final_aci = np.mean([r.aci_score for r in final_results])

# Calculate reduction (measure of paradigm improvement)
reduction = aci.calculate_aci_reduction(initial_aci, final_aci)

print(f"ACI before: {initial_aci:.3f}")
print(f"ACI after: {final_aci:.3f}")
print(f"Reduction: {reduction:.1f}%")

# Statistically significant reduction indicates successful paradigm
if reduction > 20:  # Example threshold
    print("Paradigm intervention successful!")
```

---

## Integration with MCTS

### Phase III Γ₁: High-Entropy Data Analysis

The ACI is used in Phase III to guide MCTS node selection:

1. **Analyze Experimental Data**
   - Calculate 𝔈_D over sliding windows
   - Identify high-entropy regions

2. **Identify Causal Variables**
   - Calculate 𝔍_C for each input variable
   - Determine which variables correlate with high entropy

3. **Flag High-Potential Signals**
   - High 𝔈_D AND High 𝔍_C
   - Prioritize for MCTS exploration

4. **Guide MCTS Search**
   - Focus exploration on high-entropy regions
   - Use causal variables for hypothesis generation
   - Adjust exploration/exploitation balance

### Example Integration

```python
# In MCTS search loop
for iteration in range(iterations):
    # Use ACI to guide node selection
    if aci_calculator and iteration % 100 == 0:
        # Analyze recent exploration history
        recent_rewards = extract_reward_history()
        recent_inputs = extract_input_history()

        experiment_data = {
            'output': recent_rewards,
            **recent_inputs,
        }

        # Detect high-entropy signals
        aci_results = aci_calculator.detect_high_entropy_signals(experiment_data)
        high_priority = aci_calculator.get_high_priority_signals(aci_results, top_n=5)

        # Adjust MCTS strategy based on ACI
        for signal in high_priority:
            # Boost exploration for high-ACI regions
            if signal.aci_score > 0.7:
                increase_exploration_weight(signal.causal_variables)

    # Continue MCTS selection, expansion, simulation, backpropagation
    ...
```

---

## Testing

### Run ACI Tests

```bash
cd glue/adapters/rese-phase3
python tests/test_aci_calculator.py
```

### Test Coverage

The test suite covers:

1. **Configuration Tests**
   - Environment variable loading
   - Default values
   - Validation

2. **Disorder Entropy Tests**
   - Constant signal (zero entropy)
   - White noise (high entropy)
   - Sine wave (low entropy)
   - Idempotency (same input → same output)

3. **Causal Coherence Tests**
   - Perfect correlation detection
   - No correlation case
   - Multiple variables
   - Length mismatch handling

4. **Signal Detection Tests**
   - High-entropy signal flagging
   - Sliding window processing
   - Timeout enforcement
   - Circuit breaker

5. **ACI Reduction Tests**
   - Reduction calculation
   - Edge cases (zero initial, increase)

6. **Synthetic Data Tests**
   - Signal generation
   - Reproducibility with seeds

7. **Integration Tests**
   - MCTS-guided selection
   - High-priority signal extraction

---

## Configuration

### Environment Variables

```bash
# Enable/disable ACI (default: true)
export PHASE3_ACI_ENABLED=true

# Sliding window parameters
export PHASE3_ACI_WINDOW_SIZE=100
export PHASE3_ACI_ENTROPY_BINS=10

# Thresholds
export PHASE3_ACI_COHERENCE_THRESHOLD=0.5
export PHASE3_ACI_ENTROPY_THRESHOLD=0.7

# Correlation analysis
export PHASE3_ACI_MIN_SAMPLES=30
export PHASE3_ACI_CORRELATION_METHOD=pearson

# Timeout and circuit breaker
export PHASE3_ACI_TIMEOUT_MS=3000
export PHASE3_ACI_CB_THRESHOLD=5
export PHASE3_ACI_CB_TIMEOUT_MS=60000
```

---

## CLAUDE.md Compliance

### ✅ Law of Configuration Explicitness

All configuration via environment variables. No magic defaults.

**Example:**
```python
config = ACIConfig.from_env()  # Loads from env vars
# Crashes immediately if invalid
```

---

### ✅ Law of Idempotency

Same input → same output.

**Test:**
```python
entropy1 = aci.calculate_disorder_entropy(signal)
entropy2 = aci.calculate_disorder_entropy(signal)
assert entropy1 == entropy2  # Always equal
```

---

### ✅ Law of UTC

All timestamps in UTC ISO-8601 format.

**Example:**
```python
result = ACIResult(
    ...
    timestamp=datetime.now(timezone.utc).isoformat(),  # UTC
)
```

---

### ✅ Circuit Breaker

Detects and handles calculation failures.

**Example:**
```python
# After multiple failures
try:
    results = aci.detect_high_entropy_signals(data)
except RuntimeError:
    # Circuit breaker is OPEN
    # Too many recent failures
```

---

### ✅ Structured Logging

JSON logs with correlation_id.

**Example:**
```json
{
  "level": "info",
  "component": "ACICalculator",
  "timestamp": "2026-02-04T12:00:00Z",
  "message": "High-entropy signal detected",
  "correlation_id": "abc-123",
  "entropy": 0.8,
  "coherence": 0.7
}
```

---

### ✅ Timeout Enforcement

All operations bounded by configurable timeout.

**Example:**
```python
# Will timeout after 3 seconds
config.timeout_ms = 3000
aci = AnomalyCharacterizationIndex(config)

# TimeoutError raised if exceeded
try:
    results = aci.detect_high_entropy_signals(large_data)
except TimeoutError:
    # Calculation exceeded timeout
```

---

## Performance Considerations

### Optimization Strategies

1. **Window Size**
   - Larger windows = more accurate, slower
   - Smaller windows = faster, less accurate
   - Default: 100 samples

2. **Entropy Bins**
   - More bins = finer granularity, slower
   - Fewer bins = coarser, faster
   - Default: 10 bins

3. **Correlation Method**
   - Pearson: Faster, assumes linear
   - Spearman: Slower, handles non-linear

4. **Timeout**
   - Prevents runaway calculations
   - Default: 3000ms

---

## Troubleshooting

### Issue: All signals have low entropy

**Solution:**
- Check signal quality (may be constant or periodic)
- Lower `PHASE3_ACI_ENTROPY_THRESHOLD`
- Increase `PHASE3_ACI_ENTROPY_BINS` for finer granularity

---

### Issue: No causal variables identified

**Solution:**
- Check input variables are correlated with output
- Lower `PHASE3_ACI_COHERENCE_THRESHOLD`
- Ensure sufficient sample size (`PHASE3_ACI_MIN_SAMPLES`)

---

### Issue: Calculation timeout

**Solution:**
- Increase `PHASE3_ACI_TIMEOUT_MS`
- Reduce `PHASE3_ACI_WINDOW_SIZE`
- Process data in batches

---

### Issue: Circuit breaker open

**Solution:**
- Check for systematic failures
- Increase `PHASE3_ACI_CB_THRESHOLD`
- Wait for recovery timeout (`PHASE3_ACI_CB_TIMEOUT_MS`)

---

## References

- **RESE Technical Manual §5.2**: Anomaly Characterization Index
- **RESE Technical Manual §6.3**: ACI reduction for Phase IV validation
- **CLAUDE.md**: Project constitution and principles
- **Phase III Executor**: `glue/adapters/rese-phase3/src/phase3_executor.py`
- **ACI Tests**: `glue/adapters/rese-phase3/tests/test_aci_calculator.py`

---

## Authors

RESE Team

---

## Version History

- **v1.0** (2026-02-04): Initial implementation
  - Disorder Entropy (𝔈_D) calculation
  - Causal Coherence (𝔍_C) calculation
  - High-entropy signal detection
  - Synthetic data generation
  - Comprehensive testing
  - CLAUDE.md compliance

---

**Document Status:** Complete

**Last Updated:** 2026-02-04

**Phase:** III - Monte Carlo Refinement
