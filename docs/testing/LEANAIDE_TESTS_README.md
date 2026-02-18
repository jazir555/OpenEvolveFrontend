# LeanAide Integration Test Suite

Comprehensive integration tests for LeanAide, covering client connections, MCP tools, CrewAI bridge phases, workflow integration, error handling, and performance.

## Overview

The test suite provides confidence that LeanAide integration works correctly across all components:

- **Client Connection & Health Checks** - Connection pooling, retries, health monitoring
- **8 MCP Tools** - All Model Context Protocol tools fully tested
- **6 CrewAI Bridge Phases** - Complete workflow from analysis to knowledge extraction
- **Workflow Integration** - Stage 3C, Stage 5 integration points
- **Error Handling** - Edge cases, timeouts, connection failures
- **Performance & Caching** - Concurrent requests, cache performance

## Quick Start

### Installation

```bash
# Install required dependencies
pip install pytest pytest-asyncio pytest-cov pytest-xdist

# Optional: For parallel test execution
pip install pytest-xdist
```

### Running Tests

```bash
# Run all tests
python test_leanaide_integration.py

# Or use the test runner
python run_leanaide_tests.py

# Run unit tests only
python run_leanaide_tests.py --unit

# Run integration tests only
python run_leanaide_tests.py --integration

# Run offline (mock) tests only - no server required
python run_leanaide_tests.py --mock

# Run with verbose output
python run_leanaide_tests.py --verbose

# Run with coverage report
python run_leanaide_tests.py --coverage

# Run fast tests only (skip slow tests)
python run_leanaide_tests.py --fast

# Run in parallel
python run_leanaide_tests.py --parallel
```

### Using pytest directly

```bash
# Run all tests
pytest test_leanaide_integration.py -v

# Run only unit tests
pytest test_leanaide_integration.py -v -m unit

# Run only integration tests
pytest test_leanaide_integration.py -v -m integration

# Run only mock tests (offline)
pytest test_leanaide_integration.py -v -m mock

# Skip slow tests
pytest test_leanaide_integration.py -v -m "not slow"

# Run with coverage
pytest test_leanaide_integration.py --cov=leanaide_client --cov=leanaide_mcp_tools --cov=leanaide_crewai_bridge --cov-report=html

# List all tests
pytest test_leanaide_integration.py --collect-only
```

## Test Organization

### Markers

Tests are organized using pytest markers:

- `@mark.unit` - Unit tests for individual components (fast, no external dependencies)
- `@mark.integration` - Integration tests for end-to-end workflows (slower, may require server)
- `@mark.mock` - Tests using mocking for offline testing (no server required)
- `@mark.server` - Tests requiring LeanAide server to be running
- `@mark.slow` - Tests that take longer to run (can be skipped with `-m "not slow"`)
- `@mark.async` - Async tests that require pytest-asyncio

### Test Structure

```
test_leanaide_integration.py
├── Configuration & Fixtures
│   ├── pytest_configure()
│   ├── test_data_dir
│   ├── sample_theorems
│   ├── sample_definitions
│   ├── sample_lean_code
│   └── mock_client
│
├── Unit Tests
│   ├── TestLeanAideClientInitialization
│   ├── TestLeanAideClientHealthChecks
│   ├── TestMCPToolRegistry
│   ├── TestMCPTool1_TranslateTheorem
│   ├── TestMCPTool2_TranslateDefinition
│   ├── TestMCPTool3_GenerateProof
│   ├── TestMCPTool4_VerifySolution
│   ├── TestMCPTool5_MathQuery
│   ├── TestMCPTool6_GenerateDocumentation
│   ├── TestMCPTool7_ElaborateCode
│   ├── TestMCPTool8_GetStatus
│   ├── TestMathematicalProblemDetector
│   ├── TestBridgePhase1_Analysis
│   ├── TestBridgePhase2_Translate
│   ├── TestBridgePhase3_Verify
│   ├── TestBridgePhase4_ProofCheck
│   ├── TestBridgePhase5_FormalVerification
│   └── TestBridgePhase6_KnowledgeExtraction
│
├── Integration Tests
│   ├── TestFullWorkflowIntegration
│   │   ├── test_full_6_phase_workflow
│   │   └── test_workflow_with_non_mathematical_content
│   └── TestBatchOperations
│       └── test_batch_translate_theorems
│
├── Error Handling Tests
│   ├── TestErrorHandling
│   │   ├── test_connection_error_handling
│   │   ├── test_timeout_handling
│   │   ├── test_empty_input_handling
│   │   ├── test_extremely_long_input_handling
│   │   └── test_malformed_response_handling
│
└── Performance Tests
    └── TestPerformanceAndCaching
        ├── test_cache_hit_performance
        └── test_concurrent_requests
```

## Test Coverage

### MCP Tools (8 Tools)

All 8 MCP tools are tested:

1. **leanaide_translate_theorem** - Translate natural language theorems to Lean
2. **leanaide_translate_definition** - Translate natural language definitions to Lean
3. **leanaide_generate_proof** - Generate proofs for theorems
4. **leanaide_verify_solution** - Verify Lean code correctness
5. **leanaide_math_query** - Answer mathematical questions
6. **leanaide_generate_documentation** - Generate documentation for Lean code
7. **leanaide_elaborate_code** - Elaborate Lean code and check errors
8. **get_leanaide_status** - Get server connection status

### CrewAI Bridge Phases (6 Phases)

All 6 phases of the CrewAI workflow are tested:

1. **Phase 1: Analysis** - Detect and classify mathematical content
2. **Phase 2: Translate** - Natural language to Lean 4 translation
3. **Phase 3: Verify** - Solution verification using Lean 4
4. **Phase 4: Proof Check** - Proof validity and completeness checks
5. **Phase 5: Formal Verification** - Comprehensive formal verification
6. **Phase 6: Knowledge Extraction** - Extract verified theorems

### Workflow Integration

Integration with workflow stages:
- Stage 3C: LeanAide integration for mathematical problem-solving
- Stage 5: Formal verification and knowledge extraction

## Test Data

Sample data is provided via pytest fixtures:

- `sample_theorems` - Mathematical theorems of varying complexity
- `sample_definitions` - Mathematical definitions
- `sample_lean_code` - Lean code examples
- `mock_server_response` - Template for mocking server responses

## Requirements

### Required

- Python 3.8+
- pytest >= 7.0
- pytest-asyncio

### Optional

- pytest-cov - Coverage reports
- pytest-xdist - Parallel test execution

### LeanAide Components

The tests require these LeanAide modules (with fallbacks if not available):

- `leanaide_client.py` - Async client for LeanAide server
- `leanaide_mcp_tools.py` - MCP tools for CrewAI agents
- `leanaide_crewai_bridge.py` - Bridge between LeanAide and CrewAI

## Server Requirements

### Tests marked with `@mark.server` require:

1. LeanAide server running on `localhost:7654`
2. Lean 4 installed and configured
3. Required dependencies available

### Running the server

```bash
# From the LeanAide directory
cd LeanAide
python leanaide_server.py

# Or with custom configuration
LEANAIDE_HOST=localhost LEANAIDE_PORT=7654 python leanaide_server.py
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: LeanAide Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install pytest pytest-asyncio pytest-cov
      - name: Run unit tests
        run: |
          python run_leanaide_tests.py --unit --coverage
      - name: Run integration tests
        run: |
          python run_leanaide_tests.py --integration --fast
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### Import Errors

If you see import errors for LeanAide modules:

```bash
# Add LeanAide to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/LeanAide"

# Or use pytest with custom path
PYTHONPATH=/path/to/Frontend pytest test_leanaide_integration.py
```

### Server Connection Failures

Tests marked with `@mark.server` will be skipped if the server is not available:

```bash
# Check server status
python -c "from leanaide_mcp_tools import get_leanaide_status; print(get_leanaide_status())"

# Start server if needed
cd LeanAide && python leanaide_server.py
```

### Async Test Failures

Make sure pytest-asyncio is installed:

```bash
pip install pytest-asyncio>=0.21.0
```

### Timeout Errors

Some tests may timeout on slower machines. Increase timeout in `leanaide_client.py`:

```python
config = LeanAideConfig(timeout=300.0)  # Increase from default
```

## Contributing

### Adding New Tests

1. Use appropriate test class organization
2. Add pytest markers (`@mark.unit`, `@mark.integration`, etc.)
3. Include docstrings explaining what is being tested
4. Use fixtures for test data
5. Test both success and failure cases
6. Make tests independent (can run in any order)

### Test Naming Convention

- `test_<component>_<action>_<outcome>` - For unit tests
- `test_full_<workflow>_<outcome>` - For integration tests
- `test_edge_case_<scenario>` - For edge case tests

Example:
- `test_translate_theorem_success`
- `test_full_6_phase_workflow`
- `test_edge_case_empty_input`

## Performance Benchmarks

Expected performance on typical hardware:

- Unit tests: < 1 second
- Integration tests: 5-30 seconds (depends on server)
- Full test suite: 30-120 seconds

## License

Same as parent OpenEvolve project.

## Contact

For issues or questions about LeanAide integration tests, please refer to the main OpenEvolve documentation.
