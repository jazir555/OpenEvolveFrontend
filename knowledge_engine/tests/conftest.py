"""
Pytest configuration and fixtures for Knowledge Engine tests.

Following CLAUDE.md principles:
- All fixtures are idempotent
- Explicit configuration via environment variables
- Structured logging
"""

import asyncio
import json
import logging
import os
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Generator
from unittest.mock import AsyncMock, MagicMock
import sys

# Configure logging first before any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try importing, provide clear error if missing
# Import ONLY what we absolutely need for fixtures
CORE_AVAILABLE = False
DOCUMENT_LOADER_AVAILABLE = False
EXTRACTOR_AVAILABLE = False

try:
    # Import directly from the file, not through __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "core",
        project_root / "knowledge_engine" / "core.py"
    )
    if spec and spec.loader:
        core_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_module)
        KnowledgeState = core_module.KnowledgeState
        EntityKnowledgeGraph = core_module.EntityKnowledgeGraph
        CORE_AVAILABLE = True
        logger.info("Successfully imported core module directly")
    else:
        logger.warning("Could not load core module directly")
except Exception as e:
    logger.warning(f"Could not import knowledge_engine.core: {e}")
    logger.warning("Tests requiring core will be skipped")
    CORE_AVAILABLE = False

# Try importing document loader
try:
    spec = importlib.util.spec_from_file_location(
        "document_loader",
        project_root / "knowledge_engine" / "document_loader.py"
    )
    if spec and spec.loader:
        dl_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dl_module)
        DocumentLoader = dl_module.DocumentLoader
        DOCUMENT_LOADER_AVAILABLE = True
        logger.info("Successfully imported document_loader module")
except Exception as e:
    logger.warning(f"Could not import document_loader: {e}")
    DOCUMENT_LOADER_AVAILABLE = False

# Try importing knowledge extractor
try:
    spec = importlib.util.spec_from_file_location(
        "knowledge_extractor",
        project_root / "knowledge_engine" / "knowledge_extractor.py"
    )
    if spec and spec.loader:
        ke_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ke_module)
        KnowledgeExtractor = ke_module.KnowledgeExtractor
        EXTRACTOR_AVAILABLE = True
        logger.info("Successfully imported knowledge_extractor module")
except Exception as e:
    logger.warning(f"Could not import knowledge_extractor: {e}")
    EXTRACTOR_AVAILABLE = False


# Structured logging configuration
def setup_structured_logging():
    """Configure JSON structured logging for tests."""
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(
        json_formatter if (json_formatter := _get_json_formatter()) else logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)


def _get_json_formatter():
    """Try to import python-json-logger if available."""
    try:
        from pythonjsonlogger import jsonlogger
        return jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    except ImportError:
        return None


setup_structured_logging()


# Environment validation fixture
@pytest.fixture(scope="session", autouse=True)
def validate_environment():
    """
    Validate required environment variables at startup.
    Following CLAUDE.md: crash immediately if required config is missing.
    """
    required_vars = []
    optional_vars = [
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]

    missing_required = [var for var in required_vars if not os.getenv(var)]
    if missing_required:
        pytest.fail(f"Missing required environment variables: {missing_required}")

    # Log optional vars that are missing (warnings only)
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    if missing_optional:
        logger = logging.getLogger(__name__)
        logger.warning(
            json.dumps({
                "msg": "Optional environment variables missing",
                "missing_vars": missing_optional,
                "level": "WARNING"
            })
        )


# Mock data fixtures
@pytest.fixture
def sample_document() -> str:
    """Sample document text for testing."""
    return """
    Artificial Intelligence (AI) has revolutionized machine learning.
    Deep learning, a subset of AI, uses neural networks with multiple layers.
    Neural networks were inspired by biological neurons in the human brain.
    The human brain contains approximately 86 billion neurons.
    """


@pytest.fixture
def sample_entities() -> Dict[str, Any]:
    """Sample entity data for testing."""
    return {
        "Artificial Intelligence": {"type": "Concept", "confidence": 0.95},
        "Machine Learning": {"type": "Field", "confidence": 0.92},
        "Deep Learning": {"type": "Technique", "confidence": 0.89},
        "Neural Networks": {"type": "Architecture", "confidence": 0.94},
        "Human Brain": {"type": "Biological System", "confidence": 0.97},
    }


@pytest.fixture
def sample_relationships() -> list:
    """Sample relationship data for testing."""
    return [
        {"subject": "Deep Learning", "predicate": "subset_of", "object": "AI"},
        {"subject": "Neural Networks", "predicate": "uses", "object": "Deep Learning"},
        {"subject": "Neural Networks", "predicate": "inspired_by", "object": "Human Brain"},
    ]


@pytest.fixture
def sample_knowledge_state() -> KnowledgeState:
    """Sample KnowledgeState instance for testing."""
    state = KnowledgeState(query="What is deep learning?")
    state.add_fact("Deep learning is a subset of machine learning")
    state.add_fact("Deep learning uses neural networks")
    state.add_uncertainty("The exact number of layers is unclear")
    state.set_current_understanding("Deep learning uses multi-layered neural networks")
    return state


# Core fixtures
@pytest.fixture
def knowledge_graph() -> EntityKnowledgeGraph:
    """Create a fresh EntityKnowledgeGraph for each test."""
    return EntityKnowledgeGraph()


@pytest.fixture
async def populated_graph(
    knowledge_graph: EntityKnowledgeGraph,
    sample_entities: Dict[str, Any],
    sample_relationships: list
) -> EntityKnowledgeGraph:
    """
    Create a knowledge graph populated with sample data.
    Idempotent: can be called multiple times safely.
    """
    for entity_name, attrs in sample_entities.items():
        await knowledge_graph.add_entity_async(entity_name, attrs)

    for rel in sample_relationships:
        await knowledge_graph.add_relationship_async(
            rel["subject"],
            rel["predicate"],
            rel["object"],
            rel.get("attributes")
        )

    return knowledge_graph


# Async event loop fixture
@pytest.fixture
def event_loop() -> Generator:
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Mock service fixtures
@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = AsyncMock()
    client.generate.return_value = "Sample generated text"
    client.embed.return_value = [0.1, 0.2, 0.3]
    return client


@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4j client for testing."""
    client = AsyncMock()
    client.execute_query.return_value = [
        {"entity": "AI", "type": "Concept"},
        {"entity": "ML", "type": "Field"}
    ]
    return client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = AsyncMock()
    client.search.return_value = [
        {"id": "1", "score": 0.95, "payload": {"text": "Sample text"}}
    ]
    client.upsert.return_value = True
    return client


# Temporary directory fixture
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Test data generators
@pytest.fixture
def generate_test_entities():
    """Factory function to generate test entities."""
    def _generate(count: int = 10) -> Dict[str, Any]:
        return {
            f"Entity_{i}": {
                "type": f"Type_{i % 3}",
                "confidence": 0.5 + (i % 5) * 0.1,
                "metadata": {"index": i}
            }
            for i in range(count)
        }
    return _generate


@pytest.fixture
def generate_test_documents():
    """Factory function to generate test documents."""
    def _generate(count: int = 5) -> list:
        templates = [
            "Document {} discusses {} in detail.",
            "The concept of {} is explored in Document {}.",
            "{} plays a crucial role in Document {}.",
        ]
        topics = ["AI", "machine learning", "neural networks", "deep learning", "data science"]

        return [
            templates[i % len(templates)].format(i, topics[i % len(topics)])
            for i in range(count)
        ]
    return _generate


# Performance tracking
@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    metrics = {
        "start_time": None,
        "end_time": None,
        "operation_count": 0,
        "errors": []
    }

    class Tracker:
        def __init__(self, metrics_dict):
            self.metrics = metrics_dict

        def start(self):
            self.metrics["start_time"] = datetime.now()

        def stop(self):
            self.metrics["end_time"] = datetime.now()

        def record_operation(self):
            self.metrics["operation_count"] += 1

        def record_error(self, error: str):
            self.metrics["errors"].append(error)

        def get_duration_ms(self) -> float:
            if self.metrics["start_time"] and self.metrics["end_time"]:
                delta = self.metrics["end_time"] - self.metrics["start_time"]
                return delta.total_seconds() * 1000
            return 0.0

        def get_metrics(self) -> Dict[str, Any]:
            return {
                "duration_ms": self.get_duration_ms(),
                "operation_count": self.metrics["operation_count"],
                "error_count": len(self.metrics["errors"]),
                "errors": self.metrics["errors"]
            }

    return Tracker(metrics)


# Database cleanup fixture
@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """
    Cleanup test data after each test.
    Ensures tests are idempotent.
    """
    yield

    # Add cleanup logic here if using real databases
    # For now, this is a placeholder for future implementation
    pass


# Circuit breaker test fixture
@pytest.fixture
def circuit_breaker_test():
    """
    Test circuit breaker functionality.
    """
    class CircuitBreakerTester:
        def __init__(self):
            self.failure_count = 0
            self.success_count = 0
            self.state = "closed"  # closed, open, half-open

        def record_failure(self):
            self.failure_count += 1
            if self.failure_count >= 3:
                self.state = "open"

        def record_success(self):
            self.success_count += 1
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0

        def reset(self):
            self.failure_count = 0
            self.success_count = 0
            self.state = "closed"

    return CircuitBreakerTester()


# Retry logic test fixture
@pytest.fixture
def retry_tracker():
    """Track retry attempts during tests."""
    class RetryTracker:
        def __init__(self):
            self.attempts = []
            self.successes = 0
            self.failures = 0

        def record_attempt(self, attempt_num: int, success: bool, error: str = None):
            self.attempts.append({
                "attempt": attempt_num,
                "success": success,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
            if success:
                self.successes += 1
            else:
                self.failures += 1

        def get_stats(self) -> Dict[str, Any]:
            return {
                "total_attempts": len(self.attempts),
                "successes": self.successes,
                "failures": self.failures,
                "attempts": self.attempts
            }

    return RetryTracker()


# Additional test fixtures for comprehensive testing

@pytest.fixture
def sample_multilingual_documents():
    """Sample documents in multiple languages."""
    return {
        "en": "Artificial Intelligence is transforming healthcare and medicine.",
        "zh": "人工智能正在彻底改变医疗保健和医学。",
        "es": "La inteligencia artificial está transformando la atención médica.",
        "fr": "L'intelligence artificielle transforme les soins de santé."
    }

@pytest.fixture
def sample_temporal_data():
    """Sample temporal knowledge data."""
    return [
        {
            "episode_id": "ep_001",
            "content": "AI research began in the 1950s",
            "timestamp": "1950-01-01T00:00:00",
            "entities": ["AI", "research"]
        },
        {
            "episode_id": "ep_002",
            "content": "Deep learning emerged in the 2010s",
            "timestamp": "2010-01-01T00:00:00",
            "entities": ["Deep learning", "AI"]
        }
    ]

@pytest.fixture
def sample_bilingual_entities():
    """Sample bilingual entity pairs."""
    return [
        {"en": "Artificial Intelligence", "zh": "人工智能", "type": "Concept"},
        {"en": "Machine Learning", "zh": "机器学习", "type": "Field"},
        {"en": "Deep Learning", "zh": "深度学习", "type": "Technique"},
        {"en": "Neural Networks", "zh": "神经网络", "type": "Architecture"}
    ]

@pytest.fixture
def sample_large_document():
    """Sample large document for stress testing."""
    base_text = "AI and machine learning are transforming industries. "
    return base_text * 1000  # ~40KB document

@pytest.fixture
def mock_graphiti_bridge():
    """Mock Graphiti temporal bridge for testing."""
    class MockGraphitiBridge:
        def __init__(self):
            self.episodes = []

        async def add_episode(self, episode):
            self.episodes.append(episode)
            return True

        async def search(self, query, **kwargs):
            return self.episodes[:10]  # Return first 10

        async def get_temporal_relations(self, entity):
            return [
                {"entity": entity, "relation": "evolved_in", "target": "2024"}
            ]

    return MockGraphitiBridge()

@pytest.fixture
def mock_kggen_pipeline():
    """Mock KG-Gen pipeline for testing."""
    class MockKGGenPipeline:
        async def extract(self, text):
            # Simple mock extraction
            words = text.split()
            entities = [w for w in words if w[0].isupper() and len(w) > 3][:5]
            return {
                "entities": [{"name": e, "type": "Concept"} for e in entities],
                "relationships": []
            }

        async def process_batch(self, documents):
            results = []
            for doc in documents:
                result = await self.extract(doc)
                results.append(result)
            return results

    return MockKGGenPipeline()

@pytest.fixture
def mock_oneke_extractor():
    """Mock OneKE bilingual extractor for testing."""
    class MockOneKEExtractor:
        async def extract_bilingual(self, text, source_lang="auto"):
            # Detect and extract in both languages
            return {
                "language": source_lang,
                "entities": [
                    {"name": "AI", "language": "en", "type": "Concept"},
                    {"name": "人工智能", "language": "zh", "type": "Concept"}
                ],
                "relationships": []
            }

        async def translate_entity(self, entity, target_lang):
            translations = {
                "AI": "人工智能",
                "Machine Learning": "机器学习",
                "Deep Learning": "深度学习"
            }
            return translations.get(entity, entity)

    return MockOneKEExtractor()

@pytest.fixture
def mock_visualization_generator():
    """Mock visualization generator for testing."""
    class MockVisualizationGenerator:
        async def generate_graph_viz(self, knowledge_graph):
            # Convert KG to visualization format
            return {
                "nodes": [
                    {"id": k, "label": k, **v}
                    for k, v in knowledge_graph.get("entities", {}).items()
                ],
                "edges": knowledge_graph.get("relationships", []),
                "layout": "force_directed"
            }

        async def generate_temporal_viz(self, temporal_data):
            return {
                "timeline": [
                    {"time": d["timestamp"], "event": d["content"]}
                    for d in temporal_data
                ],
                "layout": "timeline"
            }

    return MockVisualizationGenerator()

@pytest.fixture
def test_database_config():
    """Test database configuration."""
    return {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "test_user",
        "neo4j_password": "test_password",
        "qdrant_url": "http://localhost:6333",
        "qdrant_api_key": "test_api_key"
    }

@pytest.fixture
async def sample_populated_graph(sample_entities, sample_relationships):
    """Create a populated graph for testing."""
    if not CORE_AVAILABLE:
        pytest.skip("Core module not available")

    graph = EntityKnowledgeGraph()
    for entity_name, attrs in sample_entities.items():
        await graph.add_entity_async(entity_name, attrs)

    for rel in sample_relationships:
        await graph.add_relationship_async(
            rel["subject"],
            rel["predicate"],
            rel["object"],
            rel.get("attributes")
        )

    return graph

@pytest.fixture
def mock_llm_responses():
    """Mock LLM responses for testing."""
    return {
        "entity_extraction": {
            "entities": ["AI", "Machine Learning", "Neural Networks"],
            "confidence": 0.95
        },
        "relationship_extraction": {
            "relationships": [
                {"source": "ML", "relation": "subset_of", "target": "AI"}
            ]
        },
        "summarization": {
            "summary": "AI is a broad field including ML and DL."
        }
    }
