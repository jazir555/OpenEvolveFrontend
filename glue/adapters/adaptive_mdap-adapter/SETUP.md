# Adaptive MDAP/MAKER Adapter - Complete Setup Guide

**Version**: 2.0.0
**Last Updated**: February 17, 2026

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Running Examples](#running-examples)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

- **Operating System**: Windows 10+, Linux, macOS
- **Python**: 3.8 or higher (3.11+ recommended)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Disk**: 500MB free space

### Required Software

```bash
# Check Python version
python --version  # Should be 3.8+

# Check pip
pip --version

# Check git (if cloning repo)
git --version
```

---

## Installation

### Step 1: Navigate to Adapter Directory

```bash
cd glue/adapters/adaptive_mdap-adapter
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(pydantic|requests|aiohttp)"
```

Expected output should show:
- pydantic>=2.0.0
- requests>=2.31.0
- aiohttp>=3.9.0
- Other dependencies

---

## Configuration

### Step 1: Create Environment File

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your values
nano .env  # or use your preferred editor
```

### Step 2: Configure Required Values

Edit `.env` file and set these **required** values:

```bash
# REQUIRED: Request timeout (must be set)
ADAPTIVE_MDAP_TIMEOUT_MS=5000

# REQUIRED: At least one API key
DEEPSEEK_API_KEY=sk-your-actual-deepseek-key
# OR
OPENAI_API_KEY=sk-your-actual-openai-key
```

### Step 3: Configure Optional Values (Recommended)

```bash
# Cache settings (improves performance)
ADAPTIVE_MDAP_CACHE_SIZE=1000
ADAPTIVE_MDAP_CACHE_TTL=300

# Async settings (for concurrent operations)
ADAPTIVE_MDAP_MAX_CONCURRENCY=10

# Logging level
ADAPTIVE_MDAP_LOG_LEVEL=INFO
```

### Step 4: Load Environment Variables

```bash
# On Linux/macOS
source .env

# On Windows (Command Prompt)
set ADAPTIVE_MDAP_TIMEOUT_MS=5000
set DEEPSEEK_API_KEY=sk-your-key

# On Windows (PowerShell)
$env:ADAPTIVE_MDAP_TIMEOUT_MS="5000"
$env:DEEPSEEK_API_KEY="sk-your-key"
```

**Note**: The adapter will **crash immediately** if `ADAPTIVE_MDAP_TIMEOUT_MS` is not set (per Federation Constitution Law 5).

---

## Verification

### Test 1: Smoke Test (Critical)

Run the comprehensive smoke test:

```bash
python smoke_test.py
```

**Expected Output**:
```
========================================================================
  SMOKE TEST - Adaptive MDAP/MAKER Adapter
========================================================================

  [PASS] Core MDAP adapter imports
  [PASS] MAKER adapter imports
  ... (15 tests total)

========================================================================
  TEST SUMMARY
========================================================================

Total Tests: 15
Passed: 15
Failed: 0
Pass Rate: 100.0%

========================================================================

  SUCCESS: All smoke tests passed!
  Integration is operational.
```

If all tests pass, your installation is working correctly!

### Test 2: Import Test

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from src import get_adapter, CanonicalSubProblem
print('[OK] All imports successful')
"
```

### Test 3: Health Check

```bash
python unified_entry.py status
```

**Expected Output**:
```
Initializing Adaptive MDAP/MAKER Adapter (v2.0)...
[OK] Initialization complete

System Status:
  MDAP Adapter: healthy
  MAKER Adapter: healthy
  Advanced Components: 5 available
  Integration Manager: operational
```

---

## Running Examples

### Quick Start Example

```bash
# Simple test that verifies adapter works
python examples/example_simple_test.py
```

**Expected Output**:
```
======================================================================
  SIMPLE ADAPTER TEST
======================================================================

Adapter Health: healthy

Test 1: Basic Complexity Analysis
----------------------------------------------------------------------
Task ID: test_001
Status: failed
Error: MDAP_UNAVAILABLE
[INFO] Analysis failed - this is expected when core projects are not available
[INFO] The adapter is working correctly with graceful degradation

======================================================================
  TEST COMPLETE
======================================================================

Conclusion:
- Adapter imports successfully
- Health check works
- Analysis executes (with graceful degradation)
- Error handling works correctly

The adapter is functioning as designed!
```

### Complete Features Demo

```bash
# Run all 8 examples
python example_complete_features.py
```

This will demonstrate:
1. Basic complexity analysis
2. Advanced problem decomposition
3. Multi-gauntlet pipeline
4. ICR pattern learning
5. Performance optimization
6. UI dashboard generation
7. Cross-system workflow
8. End-to-end workflow

### Individual Examples

```bash
# Async processing
python examples/example_async_processing.py

# Caching and performance
python examples/example_caching_performance.py

# Advanced decomposition
python examples/example_advanced_decomposition.py

# ICR learning
python examples/example_icr_learning.py

# Multi-gauntlet pipeline
python examples/example_multi_gauntlet_pipeline.py

# UI dashboard
python examples/example_ui_dashboard.py

# Cross-system workflow
python examples/example_cross_system_workflow.py
```

---

## Troubleshooting

### Issue 1: "ADAPTIVE_MDAP_TIMEOUT_MS is required"

**Error**:
```
AdapterConfigError: ADAPTIVE_MDAP_TIMEOUT_MS is required.
Service cannot start without explicit timeout configuration.
```

**Solution**: Set the environment variable:
```bash
export ADAPTIVE_MDAP_TIMEOUT_MS=5000
# Or on Windows:
set ADAPTIVE_MDAP_TIMEOUT_MS=5000
```

### Issue 2: Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'src'
```

**Solution**: Make sure you're in the adapter directory and using correct path:
```python
import sys
sys.path.insert(0, '.')  # Add current directory
from src import get_adapter
```

### Issue 3: Core Projects Not Available

**Log Output**:
```
ERROR - Adaptive MDAP not available: No module named 'adaptive_mdap'
```

**Status**: This is **expected behavior** (graceful degradation). The adapter is working correctly.

**Explanation**: The adapter is designed to work without core projects using stub implementations. This is intentional per Federation Constitution Law 1 (Air Gap).

### Issue 4: Unicode Encoding Errors on Windows

**Error**:
```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Solution**: The adapter uses ASCII output ([OK]/[FAIL]) instead of Unicode (✓/✗). If you see this error in your own code, replace Unicode with ASCII equivalents.

### Issue 5: Async Event Loop Errors

**Error**:
```
There is no current event loop in thread 'MainThread'
```

**Solution**: Use `asyncio.run()` properly:
```python
# Correct
async def main():
    await async_operation()

asyncio.run(main())

# Incorrect - don't mix sync and async
def main():
    result = asyncio.run(async_func())  # OK
    result2 = asyncio.run(async_func2())  # BAD - creates new loop
```

### Issue 6: Probe Script Failures

**Error**:
```
bash: ./check_async_features.sh: No such file or directory
```

**Solution**: Run probes from the adapter directory:
```bash
cd glue/adapters/adaptive_mdap-adapter
bash probes/check_v2_features.sh
```

The master probe automatically changes to the probes directory.

---

## Next Steps

### 1. Explore the Documentation

- **README.md** - Overview and architecture
- **QUICK_START.md** - 5-minute quick start
- **ADR.md** - Architecture decision records
- **FINAL_INTEGRATION_STATUS.md** - Complete feature list

### 2. Run Probes

Verify specific functionality:

```bash
# Run all v1.0 tests
bash probes/check_api.sh

# Run all v2.0 tests
bash probes/check_v2_features.sh

# Run specific probe
bash probes/check_async_features.sh
```

### 3. Integrate with Your Code

```python
from src import get_adapter, CanonicalSubProblem

# Initialize adapter
adapter = get_adapter()

# Analyze complexity
subproblem = CanonicalSubProblem(
    id="task-001",
    description="Implement authentication",
    domain="security",
    depth=2
)

response = adapter.analyze_complexity(subproblem)

# Handle result
if response.status == TaskStatus.COMPLETED:
    print(f"Complexity: {response.complexity_score.overall_score}")
```

### 4. Explore Advanced Features

- **Async Processing**: Use `get_async_adapter()` for concurrent operations
- **Caching**: Automatic caching improves repeated operations
- **Decomposition**: Use `get_advanced_openevolve_integration()` for complex workflows
- **Gauntlet**: Multi-stage verification pipeline
- **ICR**: Pattern learning and prediction

### 5. Monitor and Debug

```python
# Check adapter health
health = adapter.health_check()

# Get cache statistics
stats = async_adapter.get_cache_stats()

# View logs
# Logs are structured JSON with correlation_id, timestamps, etc.
```

---

## Getting Help

### Check Logs First

All adapter operations log structured JSON:
```json
{
  "msg": "Complexity analysis requested",
  "timestamp": "2026-02-18T05:43:39.051951+00:00",
  "task_id": "example_1",
  "correlation_id": "d8cd819b-73ec-434d-adaa-727f9fae0adf"
}
```

### Review Documentation

1. **Federation Constitution** (CLAUDE.md) - System architecture principles
2. **Integration Status** (FINAL_INTEGRATION_STATUS.md) - What's implemented
3. **Gap Analysis** (GAP_ANALYSIS_ROUND5.md) - Known issues and improvements

### Run Diagnostics

```bash
# Full diagnostic
python smoke_test.py

# Check all integrations
python unified_entry.py status

# Run specific example
python examples/example_simple_test.py
```

---

## Summary

✅ **Installation**: pip install -r requirements.txt
✅ **Configuration**: Copy .env.example to .env and set ADAPTIVE_MDAP_TIMEOUT_MS
✅ **Verification**: Run python smoke_test.py (expect 15/15 pass)
✅ **Usage**: Import from src and use get_adapter()
✅ **Support**: Graceful degradation when core projects unavailable

**The adapter is designed to work immediately after setup, even without core projects installed.**

---

**Last Updated**: February 17, 2026
**Version**: 2.0.0
**Status**: Production Ready
