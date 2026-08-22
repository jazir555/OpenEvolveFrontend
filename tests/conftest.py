"""
Root Pytest Configuration for OpenEvolve Frontend Test Suite

This module provides shared fixtures, configuration, and setup for all tests.
It addresses common configuration/setup issues:

1. Environment variable validation and defaults
2. Path resolution for imports
3. Database/mocking setup
4. Logging configuration
5. Async test support
6. Common test fixtures
7. Test data management

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import asyncio
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator, Dict, Any
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock
import logging
import json

# ============================================================================
# PATH CONFIGURATION - Resolve imports from project root
# ============================================================================

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add core-projects/openevolve to path for OpenEvolve imports
# The structure is: core-projects/openevolve/openevolve/
core_projects_openevolve = project_root / "core-projects" / "openevolve"
if str(core_projects_openevolve) not in sys.path:
    sys.path.insert(0, str(core_projects_openevolve))

# Add knowledge_engine to path
knowledge_engine_path = project_root / "knowledge_engine"
if str(knowledge_engine_path) not in sys.path:
    sys.path.insert(0, str(knowledge_engine_path))

# Add glue layer to path
glue_path = project_root / "glue"
if str(glue_path) not in sys.path:
    sys.path.insert(0, str(glue_path))


# ============================================================================
# ENVIRONMENT CONFIGURATION - Set required defaults
# ============================================================================

def set_test_environment_defaults():
    """Set default environment variables for testing if not present."""
    test_env_defaults = {
        # Database configuration
        "DATABASE_URL": "sqlite:///:memory:",
        "TEST_DATABASE_URL": "sqlite:///:memory:",

        # API configuration
        "API_HOST": "localhost",
        "API_PORT": "8000",
        "API_TIMEOUT": "5",

        # Knowledge engine configuration
        "KNOWLEDGE_GRAPH_ENABLED": "false",
        "ENTITY_EXTRACTION_ENABLED": "false",

        # ROMA configuration
        "ROMA_ENABLED": "false",
        "ROMA_DECOMPOSITION_DEPTH": "3",

        # DSPy configuration
        "DSPY_ENABLED": "false",

        # DeepKE configuration
        "DEEPKE_ENABLED": "false",

        # Testing configuration
        "TESTING": "true",
        "TEST_LOG_LEVEL": "WARNING",
        "PYTEST_CURRENT_TEST": "",

        # Logging configuration
        "LOG_LEVEL": "WARNING",
        "LOG_FORMAT": "json",

        # Timezone
        "TZ": "UTC",
    }

    for key, value in test_env_defaults.items():
        if key not in os.environ:
            os.environ[key] = value


# Set defaults at import time
set_test_environment_defaults()


# ============================================================================
# LOGGING CONFIGURATION - Structured JSON logging for tests
# ============================================================================

def configure_test_logging():
    """Configure structured logging for tests."""
    log_level = os.environ.get("TEST_LOG_LEVEL", "WARNING").upper()

    # Configure logging to capture test output
    logging.basicConfig(
        level=getattr(logging, log_level, logging.WARNING),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Override any existing configuration
    )

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


configure_test_logging()


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.
    """
    # Register custom markers
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require external services)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (isolated, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (deselect with '-m \"not performance\"')"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as async tests"
    )
    config.addinivalue_line(
        "markers", "requires_env(var): marks tests requiring specific environment variable"
    )
    config.addinivalue_line(
        "markers", "skipif_import_exists(module): skip test if module can be imported"
    )


# ============================================================================
# ASYNC TEST SUPPORT - Proper event loop handling
# ============================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Create event loop policy for async tests."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def event_loop():
    """
    Create and yield an event loop for async tests.
    Ensures proper cleanup after tests complete.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
async def async_setup():
    """
    Generic async setup/teardown fixture.
    Use this in async tests that need setup/teardown.
    """
    # Setup code here
    yield
    # Teardown code here
    await asyncio.sleep(0)  # Allow pending tasks to complete


# ============================================================================
# DATABASE FIXTURES - In-memory test databases
# ============================================================================

@pytest.fixture
def test_db_path():
    """
    Provide a temporary database file path.
    Cleans up automatically after test.
    """
    db_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
    db_path = db_file.name
    db_file.close()

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def temp_directory():
    """
    Provide a temporary directory for test files.
    Cleans up automatically after test.
    """
    temp_dir = tempfile.mkdtemp(prefix="openevolve_test_")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# MOCK FIXTURES - Common mocking patterns
# ============================================================================

@pytest.fixture
def mock_logger():
    """
    Provide a mock logger for testing.
    Ensures tests don't inadvertently create real log files.
    """
    logger = Mock(spec=logging.Logger)
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture
def mock_config():
    """
    Provide a default test configuration.
    Override specific values in your test as needed.
    """
    return {
        "api": {
            "host": "localhost",
            "port": 8000,
            "timeout": 5,
            "retry_attempts": 3,
            "retry_delay": 0.1
        },
        "database": {
            "url": "sqlite:///:memory:",
            "pool_size": 1,
            "max_overflow": 0
        },
        "knowledge_engine": {
            "enabled": False,
            "entity_extraction": False,
            "relationship_extraction": False
        },
        "roma": {
            "enabled": False,
            "max_decomposition_depth": 3,
            "max_parallel_problems": 5
        },
        "testing": {
            "mock_external_services": True,
            "use_in_memory_db": True,
            "disable_async": False
        },
        "logging": {
            "level": "WARNING",
            "format": "json",
            "include_timestamp": True,
            "include_correlation_id": True
        }
    }


@pytest.fixture
def mock_async_response():
    """
    Provide a mock async HTTP response.
    Useful for testing API calls without making real requests.
    """
    async def _create_response(
        status_code: int = 200,
        json_data: Dict[str, Any] = None,
        text: str = "",
        headers: Dict[str, str] = None
    ):
        response = AsyncMock()
        response.status = status_code
        response.status_code = status_code
        response.json = AsyncMock(return_value=json_data or {})
        response.text = AsyncMock(return_value=text)
        response.headers = headers or {}
        return response

    return _create_response


# ============================================================================
# TEST DATA FIXTURES - Common test data
# ============================================================================

@pytest.fixture
def sample_entity_data():
    """Provide sample entity data for testing."""
    return {
        "entity_id": "test-entity-001",
        "entity_type": "test_type",
        "name": "Test Entity",
        "properties": {
            "description": "A test entity",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active"
        },
        "metadata": {
            "source": "test",
            "confidence": 0.95
        }
    }


@pytest.fixture
def sample_relationship_data():
    """Provide sample relationship data for testing."""
    return {
        "relationship_id": "test-rel-001",
        "source_id": "entity-001",
        "target_id": "entity-002",
        "relationship_type": "test_relationship",
        "properties": {
            "weight": 0.8,
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        "metadata": {
            "source": "test",
            "confidence": 0.9
        }
    }


@pytest.fixture
def sample_problem_data():
    """Provide sample problem data for ROMA-style decomposition testing."""
    return {
        "problem_id": "test-problem-001",
        "problem": "Test problem description",
        "domain": "test_domain",
        "complexity": "medium",
        "metadata": {
            "priority": "high",
            "deadline": datetime.now(timezone.utc).isoformat()
        }
    }


# ============================================================================
# KNOWLEDGE ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def mock_entity_knowledge_graph():
    """
    Provide a mock EntityKnowledgeGraph for testing.
    Avoids database initialization issues.
    """
    from unittest.mock import Mock

    kg = Mock()
    kg.add_entity = Mock(return_value="entity-001")
    kg.add_relationship = Mock(return_value="rel-001")
    kg.get_entity = Mock(return_value=None)
    kg.get_related_entities = Mock(return_value=[])
    kg.query = Mock(return_value=[])
    kg.exists = Mock(return_value=False)
    kg.delete = Mock(return_value=True)

    return kg


@pytest.fixture
def mock_knowledge_artifact():
    """Provide a mock knowledge artifact for testing."""
    from datetime import datetime, timezone

    artifact = Mock()
    artifact.artifact_id = "artifact-001"
    artifact.artifact_type = "test_artifact"
    artifact.content = {"test": "data"}
    artifact.source = "test"
    artifact.created_at = datetime.now(timezone.utc).isoformat()
    artifact.metadata = {"test": "metadata"}

    return artifact


# ============================================================================
# ROMA INTEGRATION FIXTURES
# ============================================================================

@pytest.fixture
def mock_roma_integration():
    """
    Provide a mock ROMAIntegration for testing.
    Avoids dependency on actual ROMA core project.
    """
    from unittest.mock import Mock, AsyncMock

    roma = Mock()
    roma.decompose = AsyncMock(return_value=Mock(
        decomposition_id="decomp-001",
        problem="test problem",
        sub_problems=[],
        is_atomic=True,
        depth=0
    ))
    roma.solve = AsyncMock(return_value=Mock(
        solution_id="solution-001",
        problem_id="prob-001",
        solution="test solution",
        confidence=0.9,
        reasoning="test reasoning"
    ))
    roma.verify = AsyncMock(return_value=Mock(
        verification_id="verify-001",
        solution_id="solution-001",
        passed=True,
        score=0.85,
        feedback="Test feedback",
        requirements_met={}
    ))
    roma.reassemble = AsyncMock(return_value=Mock(
        success=True,
        solution="reassembled solution"
    ))

    return roma


# ============================================================================
# INTEGRATION-SPECIFIC FIXTURES
# ============================================================================

@pytest.fixture
def integration_availability():
    """
    Check which integrations are available for testing.
    Tests can skip based on integration availability.
    """
    available = {}

    # Check ROMA
    try:
        from knowledge_engine.integrations.roma_integration import ROMAIntegration
        available["roma"] = True
    except ImportError:
        available["roma"] = False

    # Check DSPy
    try:
        from knowledge_engine.integrations.dspy_integration import DSPyIntegration
        available["dspy"] = True
    except ImportError:
        available["dspy"] = False

    # Check DeepKE
    try:
        from knowledge_engine.integrations.deepke_integration import DeepKEIntegration
        available["deepke"] = True
    except ImportError:
        available["deepke"] = False

    # Check RAGbits
    try:
        from knowledge_engine.integrations.ragbits_integration import RagbitsIntegration
        available["ragbits"] = True
    except ImportError:
        available["ragbits"] = False

    # Check CrewAI
    try:
        from knowledge_engine.integrations.crewai_integration import CrewAIIntegration
        available["crewai"] = True
    except ImportError:
        available["crewai"] = False

    return available


# ============================================================================
# TIME FIXTURES - Consistent timestamps
# ============================================================================

@pytest.fixture
def fixed_timestamp():
    """
    Provide a fixed timestamp for testing.
    Ensures consistent time-related assertions.
    """
    return datetime(2026, 2, 6, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_time():
    """
    Mock time functions for testing.
    Provides deterministic time-related behavior.
    """
    fixed_time = datetime(2026, 2, 6, 12, 0, 0, tzinfo=timezone.utc)

    with pytest.mock.patch('knowledge_engine.schemas.base.datetime') as mock_dt:
        mock_dt.now.return_value = fixed_time
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield mock_dt


# ============================================================================
# VALIDATION FIXTURES
# ============================================================================

@pytest.fixture
def validation_success():
    """Provide a successful validation result."""
    return Mock(
        is_valid=True,
        errors=[],
        warnings=[],
        metadata={"validated_at": datetime.now(timezone.utc).isoformat()}
    )


@pytest.fixture
def validation_failure():
    """Provide a failed validation result with errors."""
    return Mock(
        is_valid=False,
        errors=[
            "Invalid entity type",
            "Missing required property"
        ],
        warnings=["Deprecated field used"],
        metadata={"validated_at": datetime.now(timezone.utc).isoformat()}
    )


# ============================================================================
# SKIP HELPERS
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to skip tests based on conditions.
    """
    # Skip integration tests unless explicitly requested
    if not config.getoption("-m", default="") or "integration" not in config.getoption("-m", default=""):
        for item in items:
            if "integration" in item.keywords and "not integration" not in config.getoption("-m", default=""):
                continue

    # Skip slow tests unless explicitly requested
    marker_expr = config.getoption("-m", default="")
    if "slow" not in marker_expr and "not slow" not in marker_expr:
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Slow test - run with '-m slow' to include"))


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_state():
    """
    Automatic cleanup after each test.
    Resets global state, clears caches, etc.
    """
    yield

    # Reset any global state here
    # Clear caches
    # Close connections
    # etc.

    # Ensure all async tasks are completed
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
    except RuntimeError:
        pass  # No event loop


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

@pytest.fixture(autouse=True)
def validate_test_environment():
    """
    Validate that the test environment is properly configured.
    Runs automatically before each test.
    """
    # Set required environment variables for tests
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing")
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-for-testing")
    os.environ.setdefault("MODEL_ID", "gpt-4-test")
    os.environ.setdefault("API_KEY", "test-api-key")
    os.environ.setdefault("TESTING", "true")

    # Check required environment variables
    required_vars = ["TESTING"]

    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        pytest.fail(f"Missing required environment variables: {missing_vars}")

    # Validate Python path
    if str(project_root) not in sys.path:
        pytest.fail(f"Project root not in sys.path: {project_root}")

    yield

    # Post-test validation
    # Add any cleanup checks here



