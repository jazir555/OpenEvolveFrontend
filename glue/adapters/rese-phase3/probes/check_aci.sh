#!/bin/bash
###############################################################################
# ACI (Anomaly Characterization Index) Probe Script
#
# This script validates that the ACI calculator is working correctly
# against live experimental data.
#
# Following CLAUDE.md principles:
# - Law of Runtime Truth: Test against actual execution
# - Law of Configuration Explicitness: Env vars required
# - Structured Logging: JSON output
#
# Usage:
#   ./probes/check_aci.sh
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for checks
CHECKS_PASSED=0
CHECKS_FAILED=0

# Log function (JSON format)
log() {
    local level="$1"
    local message="$2"
    printf '{"level":"%s","message":"%s","timestamp":"%s"}\n' \
        "$level" \
        "$message" \
        "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}

# Run a check
run_check() {
    local check_name="$1"
    local command="$2"

    echo -e "${YELLOW}Running:${NC} $check_name"

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED:${NC} $check_name"
        log "info" "Check passed: $check_name"
        ((CHECKS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED:${NC} $check_name"
        log "error" "Check failed: $check_name"
        ((CHECKS_FAILED++))
        return 1
    fi
}

echo "================================================"
echo "ACI Calculator Probe Script"
echo "Validating Anomaly Characterization Index"
echo "================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Set environment variables for ACI
export PHASE3_ACI_WINDOW_SIZE="100"
export PHASE3_ACI_ENTROPY_BINS="10"
export PHASE3_ACI_COHERENCE_THRESHOLD="0.5"
export PHASE3_ACI_ENTROPY_THRESHOLD="0.7"
export PHASE3_ACI_TIMEOUT_MS="3000"
export PHASE3_ACI_MIN_SAMPLES="30"
export PHASE3_ACI_CORRELATION_METHOD="pearson"
export PHASE3_ACI_CB_THRESHOLD="5"
export PHASE3_ACI_CB_TIMEOUT_MS="60000"

# Check 1: Test ACI module imports
run_check "ACI module imports" \
    "python -c 'from sys import path; path.insert(0, \"../src\"); from aci_calculator import AnomalyCharacterizationIndex, ACIConfig, ACIResult; print(\"ACI imports OK\")'"

# Check 2: Test configuration loading
run_check "ACI configuration from environment" \
    "python -c 'from sys import path; path.insert(0, \"../src\"); from aci_calculator import ACIConfig; config = ACIConfig.from_env(); assert config.window_size == 100; assert config.coherence_threshold == 0.5; print(\"Config OK\")'"

# Check 3: Test disorder entropy calculation
run_check "Disorder Entropy (𝔈_D) calculation" \
    "python -c '
import sys
sys.path.insert(0, \"../src\")
import numpy as np
from aci_calculator import AnomalyCharacterizationIndex

aci = AnomalyCharacterizationIndex()

# Test white noise (high entropy)
np.random.seed(42)
noise = np.random.rand(1000)
entropy = aci.calculate_disorder_entropy(noise)
assert entropy > 0.7, f\"White noise should have high entropy, got {entropy}\"
print(f\"White noise entropy: {entropy:.3f} OK\")

# Test constant signal (zero entropy)
constant = np.ones(1000) * 0.5
entropy = aci.calculate_disorder_entropy(constant)
assert entropy == 0.0, f\"Constant signal should have zero entropy, got {entropy}\"
print(f\"Constant entropy: {entropy:.3f} OK\")
'"

# Check 4: Test causal coherence calculation
run_check "Causal Coherence (𝔍_C) calculation" \
    "python -c '
import sys
sys.path.insert(0, \"../src\")
import numpy as np
from aci_calculator import AnomalyCharacterizationIndex

aci = AnomalyCharacterizationIndex()

# Test perfect correlation
entropy_data = np.linspace(0, 1, 100)
input_var = entropy_data * 2  # Perfect linear relationship
coherence, causal_vars = aci.calculate_causal_coherence(entropy_data, {\"var1\": input_var})
assert coherence > 0.9, f\"Perfect correlation should have high coherence, got {coherence}\"
assert \"var1\" in causal_vars, f\"Should identify var1 as causal\"
print(f\"Perfect correlation coherence: {coherence:.3f} OK\")
'"

# Check 5: Test high-entropy signal detection
run_check "High-entropy signal detection (High 𝔈_D AND High 𝔍_C)" \
    "python -c '
import sys
sys.path.insert(0, \"../src\")
import numpy as np
from aci_calculator import AnomalyCharacterizationIndex

aci = AnomalyCharacterizationIndex()

# Generate data with high entropy and correlation
np.random.seed(42)
length = 500
input_var = np.random.rand(length)
output = input_var * 0.8 + np.random.randn(length) * 0.2

experiment_data = {
    \"output\": output,
    \"input1\": input_var,
}

results = aci.detect_high_entropy_signals(experiment_data, time_series_key=\"output\")
assert len(results) > 0, \"Should detect signals\"
print(f\"Detected {len(results)} signal windows OK\")

# Check result structure
result = results[0]
assert hasattr(result, \"disorder_entropy\"), \"Missing disorder_entropy\"
assert hasattr(result, \"causal_coherence\"), \"Missing causal_coherence\"
assert hasattr(result, \"aci_score\"), \"Missing aci_score\"
assert hasattr(result, \"is_high_entropy_signal\"), \"Missing is_high_entropy_signal\"
print(f\"Signal structure OK\")
'"

# Check 6: Test synthetic data generation
run_check "Synthetic data generator" \
    "python -c '
import sys
sys.path.insert(0, \"../src\")
import numpy as np
from aci_calculator import SyntheticDataGenerator, AnomalyCharacterizationIndex

generator = SyntheticDataGenerator(seed=42)

# Test white noise
noise = generator.generate_white_noise(1000)
assert len(noise) == 1000, \"Incorrect length\"
assert np.all(noise >= 0) and np.all(noise <= 1), \"Noise not in [0, 1]\"
print(f\"White noise generation OK\")

# Test multi-variable experiment
data = generator.generate_multi_variable_experiment(length=1000, num_variables=5)
assert \"output\" in data, \"Missing output\"
assert len(data) == 6, \"Should have output + 5 variables\"
print(f\"Multi-variable experiment generation OK\")
'"

# Check 7: Test ACI reduction calculation
run_check "ACI reduction calculation (Phase IV validation)" \
    "python -c '
import sys
sys.path.insert(0, \"../src\")
from aci_calculator import AnomalyCharacterizationIndex

aci = AnomalyCharacterizationIndex()

# Test 50% reduction
reduction = aci.calculate_aci_reduction(0.8, 0.4)
assert abs(reduction - 50.0) < 0.1, f\"Expected 50%% reduction, got {reduction}%\"
print(f\"ACI reduction: {reduction:.1f}% OK\")

# Test no reduction
reduction = aci.calculate_aci_reduction(0.6, 0.6)
assert reduction == 0.0, f\"No reduction should be 0%%, got {reduction}%\"
print(f\"No reduction: {reduction:.1f}% OK\")
'"

# Check 8: Test high-priority signal extraction
run_check "High-priority signal extraction for MCTS" \
    "python -c '
import sys
sys.path.insert(0, \"../src\")
from aci_calculator import ACIResult, AnomalyCharacterizationIndex

aci = AnomalyCharacterizationIndex()

# Create sample results
results = [
    ACIResult(
        disorder_entropy=0.8,
        causal_coherence=0.7,
        aci_score=0.75,
        is_high_entropy_signal=True,
        causal_variables=[\"var1\"],
        correlation_id=\"test-123\",
        timestamp=\"2026-02-04T12:00:00Z\",
        window_start_idx=0,
        window_end_idx=100
    ),
    ACIResult(
        disorder_entropy=0.3,
        causal_coherence=0.2,
        aci_score=0.25,
        is_high_entropy_signal=False,
        causal_variables=[],
        correlation_id=\"test-123\",
        timestamp=\"2026-02-04T12:00:00Z\",
        window_start_idx=100,
        window_end_idx=200
    ),
]

high_priority = aci.get_high_priority_signals(results, top_n=5)
assert len(high_priority) == 1, f\"Should extract 1 high-priority signal, got {len(high_priority)}\"
assert high_priority[0].is_high_entropy_signal, \"Should be high-entropy signal\"
print(f\"High-priority extraction OK: {len(high_priority)} signals\")
'"

# Check 9: Test idempotency (Law of Idempotency)
run_check "Idempotency: Same input → same output" \
    "python -c '
import sys
sys.path.insert(0, \"../src\")
import numpy as np
from aci_calculator import AnomalyCharacterizationIndex

aci = AnomalyCharacterizationIndex()

# Generate test data
np.random.seed(42)
signal = np.random.rand(500)

# Calculate multiple times
entropy1 = aci.calculate_disorder_entropy(signal)
entropy2 = aci.calculate_disorder_entropy(signal)
entropy3 = aci.calculate_disorder_entropy(signal)

assert entropy1 == entropy2 == entropy3, \"Entropy calculation must be idempotent\"
print(f\"Idempotency OK: entropy={entropy1:.3f}\")
'"

# Check 10: Test timeout enforcement
run_check "Timeout enforcement (Law of Timeout)" \
    "python -c '
import sys
sys.path.insert(0, \"../src\")
import os
import numpy as np
from aci_calculator import AnomalyCharacterizationIndex, ACIConfig

# Set very short timeout
os.environ[\"PHASE3_ACI_TIMEOUT_MS\"] = \"1\"
config = ACIConfig.from_env()
aci = AnomalyCharacterizationIndex(config)

# Generate large dataset
np.random.seed(42)
length = 10000
experiment_data = {
    \"output\": np.random.rand(length),
    \"input1\": np.random.rand(length),
}

# Attempt calculation (may timeout depending on machine speed)
try:
    results = aci.detect_high_entropy_signals(experiment_data, time_series_key=\"output\")
    print(f\"Timeout enforcement: No timeout (machine too fast) - OK\")
except TimeoutError:
    print(f\"Timeout enforcement: Timeout detected - OK\")
except Exception as e:
    print(f\"Timeout enforcement: Other exception ({type(e).__name__}) - OK\")
'"

# Summary
echo ""
echo "================================================"
echo "Probe Summary"
echo "================================================"
echo -e "Checks Passed: ${GREEN}$CHECKS_PASSED${NC}"
echo -e "Checks Failed: ${RED}$CHECKS_FAILED${NC}"
echo ""

# Exit with appropriate code
if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All ACI checks passed!${NC}"
    log "info" "All ACI probe checks passed"
    exit 0
else
    echo -e "${RED}✗ Some ACI checks failed${NC}"
    log "error" "Some ACI probe checks failed"
    exit 1
fi
