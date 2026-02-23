# ROMA API Probe Scripts

Probe scripts to verify ROMA API availability and functionality following the **"Law of Runtime Truth"** from the Federation Constitution.

## Overview

These probes validate that the ROMA API server is actually available and working as documented, rather than trusting documentation or assumptions.

## Scripts

### 1. `check_api.sh`
**Purpose:** Health check probe

**Validates:**
- ROMA server is running and accessible
- `/health` endpoint returns 200 OK
- Health response contains `status` field
- API endpoints are accessible

**Usage:**
```bash
export ROMA_SERVER_URL=http://localhost:8000
./check_api.sh
```

**Exit Codes:**
- `0` - ROMA API is healthy
- `1` - ROMA API is unhealthy or unavailable

**Output:** JSON with health status
```json
{
  "status": "healthy",
  "server": "http://localhost:8000",
  "response": {...}
}
```

---

### 2. `probe_execution.sh`
**Purpose:** Task execution probe

**Validates:**
- Can create a new execution via POST `/api/v1/executions`
- Can retrieve execution details via GET `/api/v1/executions/{id}`
- Can cancel running execution via POST `/api/v1/executions/{id}/cancel`
- Execution responses contain required fields

**Usage:**
```bash
export ROMA_SERVER_URL=http://localhost:8000
export TEST_GOAL="What is 2+2?"
./probe_execution.sh
```

**Exit Codes:**
- `0` - Execution probe successful
- `1` - Execution probe failed

**Output:** JSON with execution details
```json
{
  "status": "success",
  "execution_id": "roma-1234567890-abc123",
  "execution_status": "pending"
}
```

---

### 3. `probe_storage.sh`
**Purpose:** Storage and checkpoint probe

**Validates:**
- Can retrieve execution checkpoints
- Checkpoint data structure is valid
- Can retrieve execution data (MLflow traces, if enabled)

**Usage:**
```bash
export ROMA_SERVER_URL=http://localhost:8000
./probe_storage.sh
```

**Exit Codes:**
- `0` - Storage probe successful
- `1` - Storage probe failed

**Output:** JSON with storage details
```json
{
  "status": "success",
  "execution_id": "roma-1234567890-abc123"
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROMA_SERVER_URL` | `http://localhost:8000` | ROMA API server URL |
| `TIMEOUT` | `10` (check_api), `30` (others) | Request timeout in seconds |
| `MAX_RETRIES` | `3` | Max retry attempts for health check |
| `RETRY_DELAY` | `2` | Delay between retries in seconds |
| `TEST_GOAL` | `"What is 2+2?"` | Test goal for execution probe |

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: ROMA Health Check

on:
  schedule:
    - cron: '*/5 * * * *'  # Every 5 minutes
  workflow_dispatch:

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Start ROMA
        run: |
          cd core-projects/ROMA
          docker-compose up -d
          sleep 30

      - name: Run health check
        run: |
          ./glue/adapters/roma/probes/check_api.sh

      - name: Run execution probe
        run: |
          ./glue/adapters/roma/probes/probe_execution.sh

      - name: Run storage probe
        run: |
          ./glue/adapters/roma/probes/probe_storage.sh
```

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running ROMA probes..."
./glue/adapters/roma/probes/check_api.sh
if [ $? -ne 0 ]; then
    echo "ROMA health check failed - aborting commit"
    exit 1
fi
```

---

## Running All Probes

```bash
# Run all probes sequentially
cd glue/adapters/roma/probes
./check_api.sh && ./probe_execution.sh && ./probe_storage.sh

# Run with error handling
for script in check_api.sh probe_execution.sh probe_storage.sh; do
    echo "Running $script..."
    ./$script
    if [ $? -ne 0 ]; then
        echo "Probe $script failed"
        exit 1
    fi
done
echo "All probes passed!"
```

---

## Troubleshooting

### Connection Refused
```
[ERROR] Connection refused - ROMA server may not be running
```
**Solution:** Start ROMA server:
```bash
cd core-projects/ROMA
docker-compose up -d
```

### Timeout
```
[ERROR] Request timeout after 10 seconds
```
**Solution:** Increase timeout:
```bash
export TIMEOUT=30
./check_api.sh
```

### Wrong URL
```
[ERROR] API endpoint returned HTTP 404
```
**Solution:** Verify server URL:
```bash
export ROMA_SERVER_URL=http://roma-core:8000
./check_api.sh
```

---

## "Law of Runtime Truth" Compliance

These probes satisfy **Law 2: Runtime Truth (Anti-Hallucination)** from the Federation Constitution:

> **The Mandate:** You generally do not trust the documentation. You trust **execution**.
>
> **The Protocol:** Before implementing a feature, you must write a `probe.{sh,py,js}` script that executes the call against the live container. If the probe fails, the feature does not exist.

These probe scripts verify ROMA API behavior by actually executing calls against the live server, ensuring our integration works with reality, not documentation.
