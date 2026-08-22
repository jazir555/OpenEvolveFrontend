# OpenEvolve API Testing Guide

Complete guide for testing the OpenEvolve FastAPI service and frontend client.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Running the Service](#running-the-service)
3. [Backend Tests](#backend-tests)
4. [Frontend Tests](#frontend-tests)
5. [Integration Tests](#integration-tests)
6. [Continuous Testing](#continuous-testing)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

**Python**:
```bash
# Check Python version (requires 3.10+)
python --version

# Install dependencies
cd BubbleLab/services/openevolve-api
pip install -r requirements.txt
```

**Node.js** (for frontend tests):
```bash
# Check Node version
node --version

# Install dependencies
cd BubbleLab/apps/bubble-studio
npm install
```

### Optional Tools

```bash
# For monitoring
pip install pytest-xdist  # Parallel test execution
pip install pytest-cov    # Coverage reports
```

---

## Running the Service

### Development Mode

```bash
# Option 1: Using Python module
cd BubbleLab/services
python -m uvicorn openevolve-api.main:app --host 0.0.0.0 --port 8001 --reload

# Option 2: Using Makefile
cd BubbleLab/services/openevolve-api
make dev

# Option 3: Using docker-compose
cd BubbleLab/services/openevolve-api
docker-compose up
```

### Verify Service is Running

```bash
# Health check
curl http://localhost:8001/health

# Or using Python
python -c "import httpx; print(httpx.get('http://localhost:8001/health').json())"

# Expected response:
# {
#   "status": "healthy",
#   "service": "openevolve-api",
#   "version": "0.1.0",
#   "features": {
#     "evolution": true,
#     "adversarial": true,
#     "sovereign": true
#   }
# }
```

---

## Backend Tests

### Run All Tests

```bash
# Using pytest
cd BubbleLab/services/openevolve-api
pytest tests/ -v

# Using test runner script
python scripts/run_tests.py
```

### Run Specific Test Classes

```bash
# Health tests only
pytest tests/test_api_integration.py::TestHealthAndInfo -v

# Workflow tests only
pytest tests/test_api_integration.py::TestWorkflows -v

# Execution tests only
pytest tests/test_api_integration.py::TestExecution -v
```

### Run Specific Test Methods

```bash
# Single test
pytest tests/test_api_integration.py::TestWorkflows::test_create_evolution_workflow -v

# Multiple specific tests
pytest tests/test_api_integration.py::TestWorkflows::test_create_evolution_workflow tests/test_api_integration.py::TestWorkflows::test_list_workflows -v
```

### With Coverage Report

```bash
# Generate coverage report
pytest tests/ --cov=. --cov-report=term-missing --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
pytest tests/ -n auto  # Uses all CPU cores
pytest tests/ -n 4     # Uses 4 workers
```

### Verbose Output

```bash
# Detailed output
pytest tests/ -vv

# Show print statements
pytest tests/ -vv -s
```

---

## Frontend Tests

### Run All Frontend Tests

```bash
cd BubbleLab/apps/bubble-studio

# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test:watch
```

### Run Specific Test Files

```bash
# OpenEvolve API client tests
npm test -- openevolveApi.test.ts

# Filter by test name
npm test -- --testNamePattern="should execute workflow"
```

### Type Checking

```bash
# Run TypeScript type checker
npm run type-check

# Or using tsc directly
npx tsc --noEmit
```

---

## Integration Tests

### End-to-End Workflow Test

```bash
# This test creates a complete workflow execution
pytest tests/test_api_integration.py::TestExecution::test_execute_workflow -v
```

### Manual Integration Testing

```python
import httpx

async def test_workflow():
    """Manual integration test"""
    client = httpx.AsyncClient(timeout=30.0)

    # 1. Create workflow
    workflow = await client.post(
        "http://localhost:8001/api/workflows",
        json={
            "name": "Test Workflow",
            "description": "Integration test",
            "workflow_type": "evolution",
            "parameters": {
                "max_iterations": 5,
                "population_size": 3
            }
        }
    )
    workflow_id = workflow.json()["id"]

    # 2. Execute workflow
    execution = await client.post(
        "http://localhost:8001/api/executions",
        json={
            "workflow_id": workflow_id,
            "problem_statement": "Create a function to add two numbers"
        }
    )
    execution_id = execution.json()["execution_id"]

    # 3. Check status
    status = await client.get(
        f"http://localhost:8001/api/executions/{execution_id}"
    )

    print(f"Status: {status.json()['status']}")

    await client.aclose()

# Run the test
import asyncio
asyncio.run(test_workflow())
```

---

## Continuous Testing

### Watch Mode

```bash
# Python tests with file watcher
pip install pytest-watch
ptw tests/ --ignore=htmlcov/

# Frontend tests in watch mode
npm run test:watch
```

### Pre-commit Hooks

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run tests before commit

echo "Running tests..."
pytest tests/ -q

if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Commit aborted."
    exit 1
fi

echo "✅ Tests passed. Proceeding with commit."
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## Test Coverage

### Current Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| Core Engines | 80% | TBD |
| API Endpoints | 90% | TBD |
| Frontend Client | 85% | TBD |
| Error Handling | 95% | TBD |

### Generate Coverage Report

```bash
# Backend coverage
pytest tests/ --cov=. --cov-report=html --cov-report=term

# Frontend coverage
npm run test:coverage

# Combined report
echo "Backend Coverage:" && pytest tests/ --cov=. --cov-report=term-missing -q
echo -e "\nFrontend Coverage:" && npm run test:coverage -- --silent
```

---

## Troubleshooting

### Service Not Running

**Error**: `Connection refused` or `Service not running`

**Solution**:
```bash
# Start the service
cd BubbleLab/services
python -m uvicorn openevolve-api.main:app --host 0.0.0.0 --port 8001

# Verify it's running
curl http://localhost:8001/health
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port 8001
lsof -i :8001  # macOS/Linux
netstat -ano | findstr :8001  # Windows

# Kill the process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use a different port
python -m uvicorn openevolve-api.main:app --port 8002
```

### Import Errors

**Error**: `ModuleNotFoundError` or `ImportError`

**Solution**:
```bash
# Ensure you're in the correct directory
cd BubbleLab/services/openevolve-api

# Install dependencies
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Timeout Errors

**Error**: `Timeout during request`

**Solution**:
```bash
# Increase timeout in tests
# Edit test_api_integration.py:
# BASE_URL = "http://localhost:8001"
# TEST_TIMEOUT = 60.0  # Increase from 30.0
```

### Test Dependencies Missing

**Error**: `No module named 'pytest'`

**Solution**:
```bash
# Install test dependencies
cd BubbleLab/services/openevolve-api
pip install -e ".[dev]"

# Or install manually
pip install pytest pytest-asyncio pytest-cov httpx
```

---

## Best Practices

### Writing Tests

1. **Arrange-Act-Assert Pattern**:
```python
def test_create_workflow():
    # Arrange
    workflow_data = {...}

    # Act
    response = client.post("/api/workflows", json=workflow_data)

    # Assert
    assert response.status_code == 200
    assert response.json()["name"] == workflow_data["name"]
```

2. **Use Fixtures for Common Setup**:
```python
@pytest.fixture
async def workflow(client):
    response = await client.post("/api/workflows", json={...})
    return response.json()

async def test_workflow_operations(workflow):
    # Use the workflow fixture
    assert workflow["id"] is not None
```

3. **Test Error Cases**:
```python
async def test_invalid_workflow(client):
    response = await client.post("/api/workflows", json={...})
    assert response.status_code == 422  # Validation error
```

### Running Tests in CI/CD

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          cd BubbleLab/services/openevolve-api
          pip install -r requirements.txt

      - name: Run tests
        run: |
          cd BubbleLab/services/openevolve-api
          pytest tests/ -v --cov=. --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Quick Reference

### Common Commands

```bash
# Start service
make dev

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test
pytest tests/test_api_integration.py::TestWorkflows::test_create_workflow -v

# Frontend tests
npm test -- openevolveApi.test.ts

# Type checking
npm run type-check
```

### Test Files

- `tests/test_api_integration.py` - Backend integration tests
- `apps/bubble-studio/src/services/__tests__/openevolveApi.test.ts` - Frontend tests
- `tests/conftest.py` - Pytest configuration

### Documentation

- `README.md` - Service overview
- `API_DOCUMENTATION.md` - API reference
- `TESTING_GUIDE.md` - This file

---

## Need Help?

- Check logs: `docker-compose logs -f openevolve-api`
- Run in debug mode: `make dev-debug`
- Enable verbose logging: Edit `main.py`, set log level to DEBUG

**Last Updated**: 2026-01-27
