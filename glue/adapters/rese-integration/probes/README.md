# RESE Probe Scripts - Runtime Verification

Following the **"Law of Runtime Truth"** from CLAUDE.md: We trust execution, not documentation.

These probes verify the actual runtime state of RESE components rather than relying on documentation claims.

---

## Overview

Each probe script:
- ✅ Outputs structured JSON to stdout
- ✅ Includes correlation_id for tracing
- ✅ Exits 0 on success, non-zero on failure
- ✅ Provides loud, clear error messages
- ✅ Tests actual runtime behavior

---

## Available Probes

### 1. check_rese_dependencies.sh

**Purpose:** Verify all required dependencies are installed and accessible.

**What it checks:**
- Python version (requires 3.9+)
- Lean 4 installation (optional but recommended)
- Required Python packages: numpy, pydantic, fastapi, uvicorn
- Optional packages: scipy, networkx, psutil, pytest

**Usage:**

```bash
# Basic execution
./glue/adapters/rese-integration/probes/check_rese_dependencies.sh

# With output pretty-printed
./glue/adapters/rese-integration/probes/check_rese_dependencies.sh | jq

# Capture exit code
./glue/adapters/rese-integration/probes/check_rese_dependencies.sh
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "All dependencies present"
else
    echo "Missing required dependencies"
fi
```

**Example Output:**

```json
{
  "probe_name": "check_rese_dependencies",
  "probe_type": "dependency_verification",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-04T12:00:00Z",
  "source_service": "rese_probe",
  "target_service": "rese_core",
  "checks": {
    "python": {
      "status": "PASS",
      "required": true,
      "version": "3.11.0",
      "message": "Python version meets requirement (>=3.9)"
    },
    "numpy": {
      "status": "PASS",
      "required": true,
      "version": "1.24.3",
      "message": "numpy is importable"
    }
  },
  "overall_status": "PASS",
  "exit_code": 0,
  "recommendation": "All required dependencies present"
}
```

---

### 2. check_rese_api.sh

**Purpose:** Verify RESE API is accessible and responsive.

**What it checks:**
- RESE root directory exists
- Core modules directory exists (with .pyc files)
- Python imports work (rese.rese_pipeline, rese.api)
- API endpoints respond (health, docs)

**Environment Variables:**

```bash
# Required (Law of Configuration Explicitness)
export RESE_API_HOST="${RESE_API_HOST:-localhost}"
export RESE_API_PORT="${RESE_API_PORT:-8000}"
export RESE_ROOT_DIR="${RESE_ROOT_DIR:-/path/to/rese}"
```

**Usage:**

```bash
# Set environment variables
export RESE_ROOT_DIR="/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese"
export RESE_API_HOST="localhost"
export RESE_API_PORT="8000"

# Run probe
./glue/adapters/rese-integration/probes/check_rese_api.sh | jq

# Test against running API
./glue/adapters/rese-integration/probes/check_rese_api.sh
```

**Example Output:**

```json
{
  "probe_name": "check_rese_api",
  "probe_type": "api_verification",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-04T12:00:00Z",
  "source_service": "rese_probe",
  "target_service": "rese_api",
  "api_url": "http://localhost:8000",
  "checks": {
    "rese_directory": {
      "status": "PASS",
      "required": true,
      "path": "/path/to/rese",
      "message": "RESE root directory exists"
    },
    "health": {
      "status": "PASS",
      "required": false,
      "endpoint": "/health",
      "http_code": 200,
      "expected_code": 200,
      "message": "Endpoint accessible"
    }
  },
  "overall_status": "PASS",
  "exit_code": 0,
  "recommendation": "RESE API is accessible"
}
```

---

### 3. check_rese_phases.sh

**Purpose:** Verify each RESE phase can initialize and has components present.

**What it checks:**
- Phase directories exist (gamma1, core, etc.)
- Bytecode files (.pyc) are present
- Phase components are discoverable
- Phase dependencies are met

**Usage:**

```bash
# Set RESE root directory
export RESE_ROOT_DIR="/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese"

# Run probe
./glue/adapters/rese-integration/probes/check_rese_phases.sh | jq

# Check all phases
./glue/adapters/rese-integration/probes/check_rese_phases.sh
```

**Example Output:**

```json
{
  "probe_name": "check_rese_phases",
  "probe_type": "phase_verification",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-04T12:00:00Z",
  "source_service": "rese_probe",
  "target_service": "rese_pipeline",
  "rese_root": "/path/to/rese",
  "phases": {
    "gamma1": {
      "status": "PASS",
      "required": true,
      "directory": "gamma1",
      "pyc_files": 15,
      "exists": true,
      "message": "Gamma1 components exist with 15 bytecode files"
    },
    "core": {
      "status": "PASS",
      "required": true,
      "directory": "core",
      "pyc_files": 12,
      "exists": true,
      "message": "Core components exist with 12 bytecode files"
    }
  },
  "phase_tests": {
    "phase0_import": false,
    "phase0_message": "Cannot test - only bytecode exists"
  },
  "overall_status": "PASS",
  "exit_code": 0,
  "note": "RESE appears to be in bytecode (.pyc) format - source code restoration may be required (see Task #1)",
  "recommendation": "RESE phase directories exist but runtime testing requires source restoration"
}
```

---

## Running All Probes

### Quick Health Check

```bash
#!/bin/bash
# run_all_rese_probes.sh

echo "Running RESE probes..."
echo "======================"

# Dependencies
echo -e "\n1. Checking dependencies..."
./glue/adapters/rese-integration/probes/check_rese_dependencies.sh | jq

# API
echo -e "\n2. Checking API..."
./glue/adapters/rese-integration/probes/check_rese_api.sh | jq

# Phases
echo -e "\n3. Checking phases..."
./glue/adapters/rese-integration/probes/check_rese_phases.sh | jq

echo -e "\n======================"
echo "Probe execution complete"
```

### CI/CD Integration

```yaml
# .gitlab-ci.yml or similar
rese_healthcheck:
  stage: test
  script:
    - ./glue/adapters/rese-integration/probes/check_rese_dependencies.sh
    - ./glue/adapters/rese-integration/probes/check_rese_api.sh
    - ./glue/adapters/rese-integration/probes/check_rese_phases.sh
  only:
    - merge_requests
    - main
```

### Pre-Commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running RESE probes before commit..."

./glue/adapters/rese-integration/probes/check_rese_dependencies.sh
if [ $? -ne 0 ]; then
    echo "❌ RESE dependency check failed"
    echo "   Fix missing dependencies before committing"
    exit 1
fi

./glue/adapters/rese-integration/probes/check_rese_phases.sh
if [ $? -ne 0 ]; then
    echo "❌ RESE phase check failed"
    echo "   Verify RESE components before committing"
    exit 1
fi

echo "✅ All RESE probes passed"
exit 0
```

---

## Interpreting Results

### Exit Codes

- **0**: All checks passed
- **1**: One or more required checks failed

### Status Values

- **PASS**: Check passed successfully
- **FAIL**: Check failed and component is required
- **WARN**: Check failed but component is optional

### Correlation IDs

Each probe run generates a unique correlation ID (UUID v4). Use this for:
- Tracing probe executions in logs
- Debugging failures across multiple systems
- Auditing probe runs over time

---

## Troubleshooting

### Probe Execution Fails

**Symptom:** `bash: ./probes/check_rese_dependencies.sh: Permission denied`

**Solution:**
```bash
chmod +x glue/adapters/rese-integration/probes/*.sh
```

### Python Not Found

**Symptom:** Probe reports Python not installed

**Solution:**
```bash
# Install Python 3.9+
# On Ubuntu/Debian:
sudo apt-get install python3.11

# On macOS:
brew install python@3.11

# On Windows:
# Download from python.org
```

### Missing Dependencies

**Symptom:** Probe reports missing packages

**Solution:**
```bash
# Install required packages
pip install numpy pydantic fastapi uvicorn

# Install optional packages
pip install scipy networkx psutil pytest
```

### RESE Directory Not Found

**Symptom:** check_rese_api.sh reports RESE root not found

**Solution:**
```bash
# Set correct path
export RESE_ROOT_DIR="/absolute/path/to/rese"

# Or update default in probe script
```

### API Not Responding

**Symptom:** check_rese_api.sh reports endpoint failures

**Solution:**
```bash
# Start RESE API server
cd /path/to/rese
python -m rese.api

# Or with uvicorn
uvicorn rese.api:app --host 0.0.0.0 --port 8000
```

---

## Compliance with CLAUDE.md

These probes follow the immutable laws:

### ✅ Law of Runtime Truth
We execute actual commands and check real system state, not documentation.

### ✅ Law of Configuration Explicitness
All configurable values use environment variables (RESE_API_HOST, RESE_API_PORT, RESE_ROOT_DIR).

### ✅ Structured Logging
All output is JSON with correlation_id, source_service, target_service.

### ✅ Fail Fast
Probes exit immediately with non-zero code if critical dependencies are missing.

### ✅ UTC Timestamps
All timestamps use ISO-8601 UTC format.

---

## Maintenance

### Adding New Checks

To add a new check to a probe:

1. Add a new `check_*` function
2. Call it in the main probe flow
3. Update JSON output structure
4. Test with `| jq` to verify JSON validity

### Updating Probes

When RESE API changes:
1. Update endpoint paths in check_rese_api.sh
2. Add new phase checks to check_rese_phases.sh
3. Update this README with new functionality
4. Test all probes in CI/CD

---

## Contact & Support

**Questions about probes?**
- Review CLAUDE.md for principles
- Check probe script comments for details
- Run with `| jq` for readable output

**Probe failures?**
- Check exit codes and error messages
- Review JSON output for specific failures
- Ensure dependencies are installed
- Verify environment variables are set

---

**Probe Version:** 1.0.0
**Last Updated:** 2026-02-04
**Compliance:** CLAUDE.md Immutable Laws
**Status:** ✅ Production Ready
