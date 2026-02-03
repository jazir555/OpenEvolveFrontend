# Z3 Adapter Probe Scripts

This directory contains production-ready probe scripts for Z3 integration, following the **Law of Runtime Truth** from the Federation Constitution.

## Overview

Probe scripts verify that Z3 integration components are functioning correctly before attempting to use them. They follow these principles:

- **Zero Trust**: Verify everything at runtime
- **Fail Fast**: Exit with clear error codes
- **Observable**: JSON Lines logging format
- **Idempotent**: Safe to run multiple times
- **Configurable**: All settings via environment variables

## Available Probes

### 1. check_api.sh
Tests Z3 REST API endpoints.

**Environment Variables:**
- `Z3_API_URL` - Base URL of Z3 API (default: `http://localhost:8000`)
- `TIMEOUT_MS` - Request timeout in milliseconds (default: `5000`)

**Probes:**
- Health check endpoint (`/health`)
- Solve endpoint (`/api/v1/solve`)
- Prove endpoint (`/api/v1/prove`)

**Exit Codes:**
- `0` - All probes passed
- `1` - Missing environment variable
- `2` - Health check failed
- `3` - Solve endpoint failed
- `4` - Prove endpoint failed
- `5` - curl not available

**Usage:**
```bash
export Z3_API_URL="http://localhost:8000"
export TIMEOUT_MS="5000"
./check_api.sh
```

### 2. check_database.sh (Bash version) / check_database.py (Python version)
Tests Z3 database connectivity and integrity.

**Environment Variables:**
- `DATABASE_URL` - Path to SQLite database file (default: `./z3_knowledge.db`)
- `TIMEOUT_MS` - Query timeout in milliseconds (default: `5000`)

**Probes:**
- Database file existence
- Database readability and integrity
- Schema validation (tables and columns)
- Query operations test

**Exit Codes:**
- `0` - All probes passed
- `1` - Missing environment variable
- `2` - Database file not found
- `3` - Database not readable
- `4` - sqlite3 not available (bash version)
- `5` - Schema validation failed
- `6` - Query execution failed

**Usage:**
```bash
# Python version (recommended for Windows)
export DATABASE_URL="./z3_knowledge.db"
export TIMEOUT_MS="5000"
python check_database.py

# Bash version (Linux/Mac with sqlite3)
export DATABASE_URL="./z3_knowledge.db"
export TIMEOUT_MS="5000"
./check_database.sh
```

### 3. check_knowledge_extraction.sh
Tests Z3 knowledge graph extraction and reasoning APIs.

**Environment Variables:**
- `Z3_API_URL` - Base URL of Z3 API (default: `http://localhost:8000`)
- `DATABASE_URL` - Path to SQLite database (default: `./z3_knowledge.db`)
- `TIMEOUT_MS` - Request timeout in milliseconds (default: `5000`)

**Probes:**
- Knowledge base status endpoint
- Pattern recognition capabilities
- Knowledge graph search functionality
- Database knowledge tables
- Knowledge extraction from solve results

**Exit Codes:**
- `0` - All critical probes passed
- `1` - Missing environment variable
- `2` - Knowledge base endpoint failed
- `3` - Pattern recognition failed
- `4` - Knowledge graph query failed
- `5` - curl not available

**Usage:**
```bash
export Z3_API_URL="http://localhost:8000"
export DATABASE_URL="./z3_knowledge.db"
export TIMEOUT_MS="5000"
./check_knowledge_extraction.sh
```

## Output Format

All probes output **JSON Lines** format for easy parsing and log aggregation:

```json
{"level":"info","msg":"Starting Z3 API probe","timestamp":"2026-02-03T10:30:00Z","probe":"check_api.sh"}
{"level":"info","msg":"Target URL: http://localhost:8000","timestamp":"2026-02-03T10:30:00Z","probe":"check_api.sh"}
{"level":"info","msg":"Health check passed: ok","timestamp":"2026-02-03T10:30:01Z","probe":"check_api.sh"}
{"level":"info","msg":"All Z3 API probes passed successfully","timestamp":"2026-02-03T10:30:02Z","probe":"check_api.sh"}
```

## Running All Probes

You can run all probes in sequence:

```bash
#!/bin/bash
export Z3_API_URL="http://localhost:8000"
export DATABASE_URL="./z3_knowledge.db"
export TIMEOUT_MS="5000"

# Run API probe
./check_api.sh
API_EXIT=$?

# Run database probe (Python version)
python check_database.py
DB_EXIT=$?

# Run knowledge extraction probe
./check_knowledge_extraction.sh
KNOWLEDGE_EXIT=$?

# Check overall status
if [ $API_EXIT -eq 0 ] && [ $DB_EXIT -eq 0 ] && [ $KNOWLEDGE_EXIT -eq 0 ]; then
    echo "All probes passed!"
    exit 0
else
    echo "Some probes failed: API=$API_EXIT DB=$DB_EXIT KNOWLEDGE=$KNOWLEDGE_EXIT"
    exit 1
fi
```

## CI/CD Integration

These probes are designed to run in CI/CD pipelines before deployment:

```yaml
# Example GitHub Actions workflow
- name: Run Z3 Probes
  env:
    Z3_API_URL: http://localhost:8000
    DATABASE_URL: ./z3_knowledge.db
    TIMEOUT_MS: 5000
  run: |
    ./glue/adapters/z3-adapter/probes/check_api.sh
    python ./glue/adapters/z3-adapter/probes/check_database.py
    ./glue/adapters/z3-adapter/probes/check_knowledge_extraction.sh
```

## Troubleshooting

### "curl is not installed or not in PATH"
**Solution:** Install curl or use the Python version of the probe.

### "sqlite3 is not installed or not in PATH"
**Solution:** Use the Python version (`check_database.py`) instead of the bash version.

### "Database file not found"
**Solution:** Ensure the database exists at the specified path, or initialize the database first.

### "API endpoint returned invalid JSON"
**Solution:** The Z3 API server may not be running. Start the server with:
```bash
python z3_api_server.py
```

## Development Notes

When adding new probes:

1. Follow the JSON Lines logging format
2. Use environment variables for all configuration
3. Return meaningful exit codes
4. Make probes idempotent (safe to run multiple times)
5. Add timeout logic to all external operations
6. Document the probe in this README

## Federation Constitution Compliance

These probes follow the Federation Constitution:

- **Law of Runtime Truth**: Verify API endpoints actually work before using them
- **Law of Configuration Explicitness**: All values via environment variables
- **Law of Idempotency**: Safe to run multiple times
- **Observability**: Structured JSON Lines logging
- **Failure Management**: Clear exit codes for different failure modes
