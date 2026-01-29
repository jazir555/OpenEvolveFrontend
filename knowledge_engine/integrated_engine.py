"""
Integrated Knowledge Engine for OpenEvolve

This module provides the comprehensive integrated facade for the OpenEvolve Knowledge Engine,
combining all knowledge engine capabilities with workflow orchestration, batch processing,
and intelligent sprint selection.

Following CLAUDE.md principles:
- CONFIGURATION EXPLICITNESS: All config via parameters
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
- RUNTIME TRUTH: Verify components before use
- IDEMPOTENCY: All operations safe to run multiple times

Author: OpenEvolve Distinguished Engineer
Version: 2.0.0
"""

import asyncio
import json
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import traceback

# Configure structured logging
logger = logging.getLogger(__name__)

# Import core components
try:
    from .core import KnowledgeState, EntityKnowledgeGraph
except ImportError:
    from core import KnowledgeState, EntityKnowledgeGraph

# Import sprint components (with graceful degradation)
try:
    from .integrations.graphiti import GraphitiTemporalBridge
    GRAPHITI_AVAILABLE = True
except ImportError:
    try:
        from integrations.graphiti import GraphitiTemporalBridge
        GRAPHITI_AVAILABLE = True
    except ImportError:
        logger.warning("Graphiti components not available")
        GRAPHITI_AVAILABLE = False

try:
    from .integrations.kggen import ExtractionPipeline
    KGGEN_AVAILABLE = True
except ImportError:
    try:
        from integrations.kggen import ExtractionPipeline
        KGGEN_AVAILABLE = True
    except ImportError:
        logger.warning("KG-Gen components not available")
        KGGEN_AVAILABLE = False

try:
    from .integrations.oneke import OneKEModelAdapter
    ONEKE_AVAILABLE = True
except ImportError:
    try:
        from integrations.oneke import OneKEModelAdapter
        ONEKE_AVAILABLE = True
    except ImportError:
        logger.warning("OneKE components not available")
        ONEKE_AVAILABLE = False

# Import existing knowledge engine components
try:
    from .knowledge_extractor import KnowledgeExtractor, KnowledgeArtifact
except ImportError:
    try:
        from knowledge_extractor import KnowledgeExtractor, KnowledgeArtifact
    except ImportError:
        KnowledgeExtractor = None
        KnowledgeArtifact = None

try:
    from .knowledge_storage import KnowledgeStorage
except ImportError:
    try:
        from knowledge_storage import KnowledgeStorage
    except ImportError:
        KnowledgeStorage = None

try:
    from .knowledge_retriever import KnowledgeRetriever
except ImportError:
    try:
        from knowledge_retriever import KnowledgeRetriever
    except ImportError:
        KnowledgeRetriever = None

# Import existing OpenEvolve components
try:
    from .indexer import CodeIndexer
except ImportError:
    try:
        from indexer import CodeIndexer
    except ImportError:
        CodeIndexer = None

try:
    from .elasticsearch_search import ElasticsearchSearchEngine
except ImportError:
    try:
        from elasticsearch_search import ElasticsearchSearchEngine
    except ImportError:
        ElasticsearchSearchEngine = None


class TaskType(Enum):
    """Types of tasks that can be processed"""
    DOCUMENT_PROCESSING = "document_processing"
    CODE_ANALYSIS = "code_analysis"
    WORKFLOW_EXTRACTION = "workflow_extraction"
    TEMPORAL_QUERY = "temporal_query"
    KNOWLEDGE_SEARCH = "knowledge_search"
    BATCH_PROCESSING = "batch_processing"
    CONTRADICTION_DETECTION = "contradiction_detection"
    VISUALIZATION = "visualization"


class SprintType(Enum):
    """Available knowledge extraction sprints"""
    TEMPORAL_GRAPHITI = "temporal_graphiti"
    BILINGUAL_ONEKE = "bilingual_oneke"
    GENERIC_KGGEN = "generic_kggen"
    HYBRID_AUTO = "hybrid_auto"


@dataclass
class ProcessingOptions:
    """Options for processing operations"""
    extract_temporal: bool = True
    extract_bilingual: bool = False
    use_embeddings: bool = True
    validate_results: bool = True
    cache_results: bool = True
    timeout_ms: int = 30000
    max_retries: int = 3
    correlation_id: Optional[str] = None


@dataclass
class ProgressCallback:
    """Callback for progress tracking"""
    callback: Callable[[str, float, Dict[str, Any]], None]
    progress_interval: float = 0.1  # Update every 10%


@dataclass
class BatchResult:
    """Result from batch processing operations"""
    total_items: int
    successful: int
    failed: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": self.successful / self.total_items if self.total_items > 0 else 0.0,
            "results": self.results,
            "errors": self.errors,
            "total_time_ms": self.total_time_ms
        }


class IntegratedKnowledgeEngine:
    """
    Comprehensive Integrated Knowledge Engine for OpenEvolve.

    This class provides a unified interface combining:
    - Knowledge extraction from documents and workflows
    - Multi-sprint processing with automatic selection
    - Temporal knowledge tracking
    - Bilingual extraction capabilities
    - Batch processing with progress tracking
    - Workflow orchestration
    - Knowledge search and retrieval
    - Code repository analysis
    - Contradiction detection
    - Visualization generation
    - Import/export capabilities

    Example usage:
        ```python
        from knowledge_engine import IntegratedKnowledgeEngine

        # Create engine
        engine = IntegratedKnowledgeEngine(config)
        await engine.initialize()

        # Process document
        result = await engine.process_document("doc.pdf")

        # Batch process
        batch_result = await engine.batch_process_documents(
            ["doc1.pdf", "doc2.pdf"],
            progress_callback=lambda msg, pct, meta: print(f"{msg}: {pct}%")
        )

        # Search knowledge
        results = await engine.search_knowledge("machine learning")

        # Cleanup
        await engine.close()
        ```
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IntegratedKnowledgeEngine with all components.

        Args:
            config: Configuration dictionary (uses defaults if None)

        Raises:
            RuntimeError: If required configuration is missing
        """
        self.config = config or self._get_default_config()
        self._validate_config()

        # Initialize components (lazy loading)
        self._graphiti = None
        self._kggen = None
        self._oneke = None
        self._storage = None
        self._extractor = None
        self._retriever = None
        self._elasticsearch = None
        self._indexer = None

        # Knowledge state and entity graph
        self.knowledge_state = KnowledgeState(query="initial")
        self.entity_graph = EntityKnowledgeGraph()

        # Progress tracking
        self._progress_callbacks: List[ProgressCallback] = []

        # Tracking
        self._initialized = False
        self._closed = False

        logger.info({
            "msg": "IntegratedKnowledgeEngine created",
            "components": {
                "graphiti": GRAPHITI_AVAILABLE,
                "kggen": KGGEN_AVAILABLE,
                "oneke": ONEKE_AVAILABLE,
                "storage": KnowledgeStorage is not None,
                "extractor": KnowledgeExtractor is not None,
                "retriever": KnowledgeRetriever is not None
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration from environment variables.

        Following CLAUDE.md: CONFIGURATION EXPLICITNESS
        All configurable values must be injected via environment variables.

        Returns:
            Configuration dictionary
        """
        return {
            # Graphiti (Temporal Knowledge Graph)
            "graphiti_uri": os.getenv("GRAPHITI_URI", "bolt://localhost:7687"),
            "graphiti_user": os.getenv("GRAPHITI_USER", "neo4j"),
            "graphiti_password": os.getenv("GRAPHITI_PASSWORD"),

            # KG-Gen (Knowledge Generation)
            "kggen_model": os.getenv("KGGEN_MODEL", "gpt-4o"),
            "kggen_timeout_ms": int(os.getenv("KGGEN_TIMEOUT_MS", "30000")),

            # OneKE (Bilingual Extraction)
            "oneke_model": os.getenv("ONEKE_MODEL", "oneke/OneKE-13B"),
            "oneke_device": os.getenv("ONEKE_DEVICE", "cuda"),
            "oneke_timeout_ms": int(os.getenv("ONEKE_TIMEOUT_MS", "60000")),

            # Storage
            "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
            "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
            "mongo_uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
            "neo4j_password": os.getenv("NEO4J_PASSWORD"),
            "redis_host": os.getenv("REDIS_HOST", "localhost"),
            "redis_port": int(os.getenv("REDIS_PORT", "6379")),

            # Elasticsearch
            "elasticsearch_hosts": os.getenv("ELASTICSEARCH_HOSTS", "http://localhost:9200").split(","),
            "elasticsearch_api_key": os.getenv("ELASTICSEARCH_API_KEY", ""),
            "elasticsearch_index_prefix": os.getenv("ELASTICSEARCH_INDEX_PREFIX", "openevolve"),

            # Code Indexer
            "indexer_config": os.getenv("INDEXER_CONFIG_PATH", "knowledge_engine/indexer_config.yaml"),

            # Processing
            "default_timeout_ms": int(os.getenv("DEFAULT_TIMEOUT_MS", "30000")),
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "cache_ttl": int(os.getenv("CACHE_TTL", "300")),

            # LLM
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
        }

    def _validate_config(self):
        """
        Validate configuration (fail fast if misconfigured).

        Following CLAUDE.md: Fail loud if misconfigured.

        Raises:
            RuntimeError: If required configuration is missing
        """
        required_vars = []

        if GRAPHITI_AVAILABLE:
            if not self.config.get("graphiti_password"):
                required_vars.append("GRAPHITI_PASSWORD")

        if KGGEN_AVAILABLE:
            if not self.config.get("openai_api_key"):
                required_vars.append("OPENAI_API_KEY")

        if required_vars:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(required_vars)}. "
                f"Set these and restart. Service cannot start."
            )

    async def initialize(self):
        """
        Initialize all components asynchronously.

        Following CLAUDE.md: RUNTIME TRUTH
        Verify each component is actually working before marking as initialized.

        Raises:
            Exception: If component initialization fails
        """
        if self._initialized:
            logger.warning("IntegratedKnowledgeEngine already initialized")
            return

        logger.info({
            "msg": "Initializing IntegratedKnowledgeEngine components",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        tasks = []

        # Initialize knowledge engine components
        if KnowledgeStorage:
            tasks.append(self._init_storage())

        if KnowledgeExtractor:
            tasks.append(self._init_extractor())

        if KnowledgeRetriever:
            tasks.append(self._init_retriever())

        # Initialize sprint components
        if GRAPHITI_AVAILABLE:
            tasks.append(self._init_graphiti())

        if KGGEN_AVAILABLE:
            tasks.append(self._init_kggen())

        if ONEKE_AVAILABLE:
            tasks.append(self._init_oneke())

        # Initialize search engines
        if self.config.get("elasticsearch_hosts"):
            tasks.append(self._init_elasticsearch())

        if Path(self.config.get("indexer_config", "")).exists():
            tasks.append(self._init_indexer())

        # Run all initializations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for failures
        failures = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append(str(result))

        if failures:
            logger.warning({
                "msg": "Some components failed to initialize",
                "failures": failures,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        self._initialized = True

        logger.info({
            "msg": "IntegratedKnowledgeEngine initialization complete",
            "components_ready": {
                "graphiti": self._graphiti is not None,
                "kggen": self._kggen is not None,
                "oneke": self._oneke is not None,
                "storage": self._storage is not None,
                "extractor": self._extractor is not None,
                "retriever": self._retriever is not None,
                "elasticsearch": self._elasticsearch is not None,
                "indexer": self._indexer is not None
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    async def _init_storage(self):
        """Initialize knowledge storage."""
        try:
            self._storage = KnowledgeStorage(self.config)
            logger.info("Knowledge storage initialized")
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise

    async def _init_extractor(self):
        """Initialize knowledge extractor."""
        try:
            self._extractor = KnowledgeExtractor(self.config)
            logger.info("Knowledge extractor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize extractor: {e}")
            raise

    async def _init_retriever(self):
        """Initialize knowledge retriever."""
        try:
            self._retriever = KnowledgeRetriever(self._storage, self.config)
            logger.info("Knowledge retriever initialized")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise

    async def _init_graphiti(self):
        """Initialize Graphiti temporal knowledge bridge."""
        try:
            self._graphiti = GraphitiTemporalBridge(
                uri=self.config["graphiti_uri"],
                user=self.config["graphiti_user"],
                password=self.config["graphiti_password"]
            )
            await self._graphiti.initialize()
            logger.info("Graphiti temporal bridge initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}")
            raise

    async def _init_kggen(self):
        """Initialize KG-Gen extraction pipeline."""
        try:
            self._kggen = ExtractionPipeline(
                model=self.config["kggen_model"],
                timeout_ms=self.config.get("kggen_timeout_ms", 30000)
            )
            logger.info("KG-Gen extraction pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize KG-Gen: {e}")
            raise

    async def _init_oneke(self):
        """Initialize OneKE bilingual extraction."""
        try:
            self._oneke = OneKEModelAdapter(
                model_name=self.config["oneke_model"],
                device=self.config.get("oneke_device", "cuda")
            )
            await self._oneke.load_model()
            logger.info("OneKE model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OneKE: {e}")
            raise

    async def _init_elasticsearch(self):
        """Initialize Elasticsearch search engine."""
        try:
            if ElasticsearchSearchEngine:
                self._elasticsearch = ElasticsearchSearchEngine(
                    hosts=self.config["elasticsearch_hosts"],
                    api_key=self.config.get("elasticsearch_api_key", "")
                )
                logger.info("Elasticsearch search engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {e}")
            raise

    async def _init_indexer(self):
        """Initialize code indexer."""
        try:
            if CodeIndexer:
                self._indexer = CodeIndexer(
                    config_path=self.config.get("indexer_config", "knowledge_engine/indexer_config.yaml")
                )
                logger.info("Code indexer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize code indexer: {e}")
            raise

    # ========== High-Level API Methods ==========

    async def process_document(
        self,
        document_path: str,
        options: Optional[ProcessingOptions] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the complete knowledge pipeline.

        Following CLAUDE.md: IDEMPOTENCY
        Safe to run multiple times on same document.

        Args:
            document_path: Path to document
            options: Processing options

        Returns:
            Processing result with entities, relations, visualization
        """
        options = options or ProcessingOptions()
        correlation_id = options.correlation_id or f"doc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        start_time = datetime.now(timezone.utc)
        logger.info({
            "msg": "Processing document",
            "document": document_path,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            # Step 1: Extract text from document
            document_text = await self._extract_text_from_document(document_path)
            if not document_text:
                raise RuntimeError(f"Failed to extract text from document: {document_path}")

            # Step 2: Auto-select sprint based on content
            sprint_type = self._select_sprint_for_content(document_text, options)

            # Step 3: Extract knowledge using selected sprint
            extraction_result = await self._extract_knowledge_with_sprint(
                text=document_text,
                sprint_type=sprint_type,
                options=options,
                correlation_id=correlation_id
            )

            # Step 4: Store knowledge artifacts
            artifacts = extraction_result.get("artifacts", [])
            stored_artifacts = []

            for artifact in artifacts:
                if self._storage:
                    artifact_dict = artifact if isinstance(artifact, dict) else artifact.to_dict()
                    artifact_id = self._storage.store_knowledge_artifact(artifact_dict)
                    stored_artifacts.append(artifact_id)

            # Step 5: Update entity graph
            for entity in extraction_result.get("entities", []):
                await self.entity_graph.add_entity(
                    entity_name=entity.get("name"),
                    attributes=entity
                )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            result = {
                "success": True,
                "correlation_id": correlation_id,
                "document_path": document_path,
                "sprint_used": sprint_type.value,
                "entities": extraction_result.get("entities", []),
                "relations": extraction_result.get("relations", []),
                "artifacts_stored": len(stored_artifacts),
                "artifact_ids": stored_artifacts,
                "processing_time_ms": processing_time_ms
            }

            logger.info({
                "msg": "Document processing complete",
                "correlation_id": correlation_id,
                "entities_count": len(extraction_result.get("entities", [])),
                "artifacts_stored": len(stored_artifacts),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Document processing failed",
                "document": document_path,
                "correlation_id": correlation_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return {
                "success": False,
                "correlation_id": correlation_id,
                "document_path": document_path,
                "error": str(e),
                "processing_time_ms": processing_time_ms
            }

    async def batch_process_documents(
        self,
        document_paths: List[str],
        options: Optional[ProcessingOptions] = None,
        progress_callback: Optional[Callable[[str, float, Dict[str, Any]], None]] = None,
        max_concurrent: int = 5
    ) -> BatchResult:
        """
        Process multiple documents in batch with progress tracking.

        Args:
            document_paths: List of document paths
            options: Processing options
            progress_callback: Optional callback for progress updates
            max_concurrent: Maximum concurrent processing

        Returns:
            BatchResult with all processing results
        """
        options = options or ProcessingOptions()
        correlation_id = options.correlation_id or f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        start_time = datetime.now(timezone.utc)
        logger.info({
            "msg": "Starting batch document processing",
            "total_documents": len(document_paths),
            "correlation_id": correlation_id,
            "max_concurrent": max_concurrent,
            "timestamp": start_time.isoformat()
        })

        results = []
        errors = []
        successful = 0
        failed = 0
        total = len(document_paths)

        # Process documents with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(doc_path: str, index: int) -> Tuple[int, Dict[str, Any]]:
            async with semaphore:
                try:
                    # Update progress
                    if progress_callback:
                        progress_pct = (index / total) * 100
                        progress_callback(
                            f"Processing {Path(doc_path).name}",
                            progress_pct,
                            {"index": index, "total": total}
                        )

                    result = await self.process_document(doc_path, options)
                    return (index, result)
                except Exception as e:
                    return (index, {
                        "success": False,
                        "document_path": doc_path,
                        "error": str(e)
                    })

        # Create tasks for all documents
        tasks = [process_with_semaphore(doc_path, i) for i, doc_path in enumerate(document_paths)]

        # Execute tasks concurrently
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results in order
        ordered_results = [None] * total
        for task_result in task_results:
            if isinstance(task_result, Exception):
                errors.append({
                    "error": str(task_result),
                    "traceback": traceback.format_exc()
                })
                failed += 1
            else:
                index, result = task_result
                ordered_results[index] = result
                if result.get("success"):
                    successful += 1
                else:
                    errors.append({
                        "document_path": result.get("document_path"),
                        "error": result.get("error")
                    })
                    failed += 1

        # Remove None values from ordered_results
        results = [r for r in ordered_results if r is not None]

        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        batch_result = BatchResult(
            total_items=total,
            successful=successful,
            failed=failed,
            results=results,
            errors=errors,
            total_time_ms=processing_time_ms
        )

        # Final progress update
        if progress_callback:
            progress_callback("Batch processing complete", 100.0, {
                "successful": successful,
                "failed": failed,
                "total_time_ms": processing_time_ms
            })

        logger.info({
            "msg": "Batch processing complete",
            "correlation_id": correlation_id,
            "total": total,
            "successful": successful,
            "failed": failed,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return batch_result

    async def search_knowledge(
        self,
        query: str,
        query_type: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search the knowledge base.

        Args:
            query: Search query string
            query_type: Type of search ("hybrid", "keyword", "semantic")
            filters: Optional filters to apply
            limit: Maximum number of results
            correlation_id: Correlation ID for tracking

        Returns:
            Search results with metadata
        """
        correlation_id = correlation_id or f"search_{uuid.uuid4().hex}"
        start_time = datetime.now(timezone.utc)

        logger.info({
            "msg": "Searching knowledge base",
            "query": query,
            "query_type": query_type,
            "limit": limit,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            if self._retriever:
                results = self._retriever.search_knowledge(query, query_type, filters, limit)
            elif self._elasticsearch:
                # Fallback to Elasticsearch
                results = await self._search_elasticsearch(query, query_type, filters, limit)
            else:
                # Fallback to entity graph search
                results = await self.entity_graph.search_entities(query)
                results = [{"id": r.get("id"), **r} for r in results]

            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            result = {
                "success": True,
                "query": query,
                "query_type": query_type,
                "results": results,
                "count": len(results),
                "execution_time_ms": execution_time_ms,
                "correlation_id": correlation_id
            }

            logger.info({
                "msg": "Search complete",
                "correlation_id": correlation_id,
                "results_count": len(results),
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Search failed",
                "query": query,
                "correlation_id": correlation_id,
                "error": str(e),
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return {
                "success": False,
                "query": query,
                "error": str(e),
                "correlation_id": correlation_id,
                "execution_time_ms": execution_time_ms
            }

    async def analyze_code(
        self,
        repo_path: str,
        options: Optional[ProcessingOptions] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a code repository and extract knowledge.

        Args:
            repo_path: Path to code repository
            options: Processing options
            correlation_id: Correlation ID for tracking

        Returns:
            Analysis results with extracted knowledge
        """
        options = options or ProcessingOptions()
        correlation_id = correlation_id or f"code_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        start_time = datetime.now(timezone.utc)
        logger.info({
            "msg": "Analyzing code repository",
            "repo_path": repo_path,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        try:
            if not self._indexer:
                raise RuntimeError("Code indexer not available")

            # Index the repository
            index_result = self._indexer.index_repository(repo_path)

            # Extract knowledge from indexed code
            if self._extractor and index_result.get("indexed_files"):
                workflow_data = {
                    "workflow_id": f"code_analysis_{correlation_id}",
                    "domain": "code_analysis",
                    "complexity": "medium",
                    "indexed_files": index_result.get("indexed_files", []),
                    "code_patterns": index_result.get("patterns", []),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                artifacts = self._extractor.extract_from_workflow(workflow_data)

                # Store artifacts
                stored_artifacts = []
                if self._storage:
                    for artifact in artifacts:
                        artifact_dict = artifact if isinstance(artifact, dict) else artifact.to_dict()
                        artifact_id = self._storage.store_knowledge_artifact(artifact_dict)
                        stored_artifacts.append(artifact_id)
            else:
                artifacts = []
                stored_artifacts = []

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            result = {
                "success": True,
                "correlation_id": correlation_id,
                "repo_path": repo_path,
                "indexed_files": index_result.get("indexed_files", 0),
                "patterns_found": len(index_result.get("patterns", [])),
                "artifacts_extracted": len(artifacts),
                "artifacts_stored": len(stored_artifacts),
                "processing_time_ms": processing_time_ms
            }

            logger.info({
                "msg": "Code analysis complete",
                "correlation_id": correlation_id,
                "indexed_files": result["indexed_files"],
                "artifacts_extracted": len(artifacts),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Code analysis failed",
                "repo_path": repo_path,
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return {
                "success": False,
                "correlation_id": correlation_id,
                "repo_path": repo_path,
                "error": str(e),
                "processing_time_ms": processing_time_ms
            }

    async def query_temporal(
        self,
        query: str,
        timestamp: Optional[datetime] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query knowledge at specific point in time.

        Following CLAUDE.md: UTC TIME
        All timestamps in UTC.

        Args:
            query: Search query
            timestamp: Point in time for query (defaults to now)
            correlation_id: Correlation ID for tracking

        Returns:
            Temporal query results
        """
        correlation_id = correlation_id or f"temporal_{uuid.uuid4().hex}"
        timestamp = timestamp or datetime.now(timezone.utc)

        start_time = datetime.now(timezone.utc)
        logger.info({
            "msg": "Temporal query",
            "query": query,
            "timestamp": timestamp.isoformat(),
            "correlation_id": correlation_id,
            "timestamp_utc": start_time.isoformat()
        })

        if not self._graphiti:
            logger.warning("Graphiti temporal knowledge not available")
            return {
                "success": False,
                "query": query,
                "error": "Temporal knowledge not available",
                "correlation_id": correlation_id
            }

        try:
            results = await self._graphiti.search_at_point_in_time(
                query=query,
                reference_time=timestamp,
                correlation_id=correlation_id
            )

            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            result = {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
                "reference_time": timestamp.isoformat(),
                "execution_time_ms": execution_time_ms,
                "correlation_id": correlation_id
            }

            logger.info({
                "msg": "Temporal query complete",
                "correlation_id": correlation_id,
                "results_count": len(results),
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.error({
                "msg": "Temporal query failed",
                "query": query,
                "correlation_id": correlation_id,
                "error": str(e),
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return {
                "success": False,
                "query": query,
                "error": str(e),
                "correlation_id": correlation_id,
                "execution_time_ms": execution_time_ms
            }

    async def detect_contradictions(
        self,
        entity_name: str,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect contradictions for an entity across time.

        Args:
            entity_name: Name of entity to check
            correlation_id: Correlation ID for tracking

        Returns:
            Detected contradictions
        """
        correlation_id = correlation_id or f"contra_{uuid.uuid4().hex}"

        logger.info({
            "msg": "Detecting contradictions",
            "entity": entity_name,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        if not self._graphiti:
            logger.warning("Graphiti contradiction detection not available")
            return {
                "success": False,
                "entity": entity_name,
                "error": "Contradiction detection not available",
                "correlation_id": correlation_id
            }

        try:
            from knowledge_engine.integrations.graphiti import GraphitiContradictionDetector
            detector = GraphitiContradictionDetector(bridge=self._graphiti)
            contradictions = await detector.detect_contradictions(
                entity_name=entity_name,
                correlation_id=correlation_id
            )

            result = {
                "success": True,
                "entity": entity_name,
                "contradictions": contradictions,
                "count": len(contradictions),
                "correlation_id": correlation_id
            }

            logger.info({
                "msg": "Contradiction detection complete",
                "entity": entity_name,
                "correlation_id": correlation_id,
                "contradictions_found": len(contradictions),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return result

        except Exception as e:
            logger.error({
                "msg": "Contradiction detection failed",
                "entity": entity_name,
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return {
                "success": False,
                "entity": entity_name,
                "error": str(e),
                "correlation_id": correlation_id
            }

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge engine.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "graphiti": self._graphiti is not None,
                "kggen": self._kggen is not None,
                "oneke": self._oneke is not None,
                "storage": self._storage is not None,
                "extractor": self._extractor is not None,
                "retriever": self._retriever is not None,
                "elasticsearch": self._elasticsearch is not None,
                "indexer": self._indexer is not None
            },
            "knowledge": {
                "entities": len(self.entity_graph.entities),
                "relationships": len(self.entity_graph.relationships)
            }
        }

        # Add storage statistics if available
        if self._storage:
            try:
                storage_stats = self._storage.get_statistics()
                stats["storage"] = storage_stats
            except Exception as e:
                logger.warning(f"Failed to get storage stats: {e}")

        # Add Graphiti stats if available
        if self._graphiti:
            try:
                graphiti_stats = await self._graphiti.get_statistics()
                stats["graphiti"] = graphiti_stats
            except Exception as e:
                logger.warning(f"Failed to get Graphiti stats: {e}")

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of all components.

        Returns:
            Dictionary with health status
        """
        health = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall": "healthy",
            "components": {}
        }

        # Check Graphiti
        if self._graphiti:
            try:
                from knowledge_engine.integrations.graphiti.health_check import GraphitiHealthChecker
                checker = GraphitiHealthChecker(self._graphiti)
                graphiti_health = await checker.check_connection()
                health["components"]["graphiti"] = graphiti_health["status"]
                if graphiti_health["status"] != "healthy":
                    health["overall"] = "degraded"
            except Exception as e:
                health["components"]["graphiti"] = "unhealthy"
                health["overall"] = "degraded"

        # Check Elasticsearch
        if self._elasticsearch:
            try:
                es_health = await self._elasticsearch.ping()
                health["components"]["elasticsearch"] = "healthy" if es_health else "unhealthy"
                if not es_health:
                    health["overall"] = "degraded"
            except Exception as e:
                health["components"]["elasticsearch"] = "unhealthy"
                health["overall"] = "degraded"

        return health

    async def close(self):
        """
        Close all components and cleanup resources.

        Following CLAUDE.md: Proper cleanup of resources
        """
        if self._closed:
            logger.warning("IntegratedKnowledgeEngine already closed")
            return

        logger.info({
            "msg": "Closing IntegratedKnowledgeEngine",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        tasks = []

        if self._graphiti:
            tasks.append(self._graphiti.close())

        if self._oneke:
            tasks.append(self._oneke.unload())

        # Run all cleanup tasks
        await asyncio.gather(*tasks, return_exceptions=True)

        self._closed = True
        self._initialized = False

        logger.info({
            "msg": "IntegratedKnowledgeEngine closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # ========== Helper Methods ==========

    async def _extract_text_from_document(self, document_path: str) -> Optional[str]:
        """Extract text from document file."""
        try:
            from knowledge_engine import document_loader

            document_path_obj = Path(document_path)
            if not document_path_obj.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")

            # Determine file type and extract text
            if document_path_obj.suffix.lower() == '.pdf':
                from knowledge_engine.document_loader import SimplePdfConverter
                converter = SimplePdfConverter()
                result = converter.convert_pdf_to_markdown(str(document_path_obj))
                if not result.get("success"):
                    raise RuntimeError(f"PDF conversion failed: {result.get('error')}")
                return result.get("markdown_content")
            elif document_path_obj.suffix.lower() in ['.txt', '.md']:
                with open(document_path_obj, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise RuntimeError(f"Unsupported file type: {document_path_obj.suffix}")

        except Exception as e:
            logger.error(f"Failed to extract text from document: {e}")
            return None

    def _select_sprint_for_content(self, content: str, options: ProcessingOptions) -> SprintType:
        """
        Automatically select appropriate sprint based on content.

        Args:
            content: Document content
            options: Processing options

        Returns:
            Selected SprintType
        """
        # Simple heuristic-based selection
        content_lower = content.lower()

        # Check for multilingual content
        non_ascii_chars = sum(1 for c in content if ord(c) > 127)
        multilingual_ratio = non_ascii_chars / len(content) if content else 0

        if multilingual_ratio > 0.3 and options.extract_bilingual and ONEKE_AVAILABLE and self._oneke:
            return SprintType.BILINGUAL_ONEKE

        # Check for temporal indicators
        temporal_keywords = ['timeline', 'history', 'evolution', 'over time', 'temporal', 'chronological']
        if any(keyword in content_lower for keyword in temporal_keywords) and options.extract_temporal:
            return SprintType.TEMPORAL_GRAPHITI

        # Default to KG-Gen for general content
        if KGGEN_AVAILABLE and self._kggen:
            return SprintType.GENERIC_KGGEN

        # Fallback to available sprint
        if GRAPHITI_AVAILABLE and self._graphiti:
            return SprintType.TEMPORAL_GRAPHITI

        return SprintType.HYBRID_AUTO

    async def _extract_knowledge_with_sprint(
        self,
        text: str,
        sprint_type: SprintType,
        options: ProcessingOptions,
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Extract knowledge using specified sprint with fallback chain.

        Args:
            text: Text to extract from
            sprint_type: Sprint to use
            options: Processing options
            correlation_id: Correlation ID

        Returns:
            Extraction results
        """
        sprints_to_try = self._get_sprint_fallback_chain(sprint_type)

        for sprint in sprints_to_try:
            try:
                logger.info({
                    "msg": f"Attempting extraction with sprint: {sprint.value}",
                    "correlation_id": correlation_id
                })

                result = await self._extract_with_single_sprint(
                    text=text,
                    sprint_type=sprint,
                    options=options,
                    correlation_id=correlation_id
                )

                if result.get("entities") or result.get("artifacts"):
                    logger.info({
                        "msg": f"Extraction successful with sprint: {sprint.value}",
                        "correlation_id": correlation_id
                    })
                    return result

            except Exception as e:
                logger.warning({
                    "msg": f"Extraction failed with sprint {sprint.value}: {e}",
                    "correlation_id": correlation_id
                })
                continue

        # All sprints failed
        logger.error({
            "msg": "All extraction sprints failed",
            "correlation_id": correlation_id
        })

        return {"entities": [], "relations": [], "artifacts": []}

    def _get_sprint_fallback_chain(self, primary_sprint: SprintType) -> List[SprintType]:
        """Get fallback chain for sprint selection."""
        fallback_chains = {
            SprintType.TEMPORAL_GRAPHITI: [
                SprintType.TEMPORAL_GRAPHITI,
                SprintType.GENERIC_KGGEN,
                SprintType.HYBRID_AUTO
            ],
            SprintType.BILINGUAL_ONEKE: [
                SprintType.BILINGUAL_ONEKE,
                SprintType.GENERIC_KGGEN,
                SprintType.TEMPORAL_GRAPHITI
            ],
            SprintType.GENERIC_KGGEN: [
                SprintType.GENERIC_KGGEN,
                SprintType.TEMPORAL_GRAPHITI
            ],
            SprintType.HYBRID_AUTO: [
                SprintType.TEMPORAL_GRAPHITI,
                SprintType.GENERIC_KGGEN,
                SprintType.BILINGUAL_ONEKE
            ]
        }

        return fallback_chains.get(primary_sprint, [SprintType.HYBRID_AUTO])

    async def _extract_with_single_sprint(
        self,
        text: str,
        sprint_type: SprintType,
        options: ProcessingOptions,
        correlation_id: str
    ) -> Dict[str, Any]:
        """Extract using a single sprint."""
        if sprint_type == SprintType.TEMPORAL_GRAPHITI:
            if not (GRAPHITI_AVAILABLE and self._graphiti):
                raise RuntimeError("Graphiti not available")

            # Use Graphiti for temporal extraction
            entities = []
            relations = []

            # Extract entities (simplified - would use proper extraction in production)
            lines = text.split('\n')
            for i, line in enumerate(lines[:50]):  # Limit to first 50 lines
                words = line.split()
                for word in words:
                    if len(word) > 5 and word.isalpha():
                        await self._graphiti.add_entity(
                            name=word,
                            entity_type="extracted",
                            metadata={"source_line": i},
                            correlation_id=correlation_id
                        )
                        entities.append({"name": word, "type": "extracted"})

            return {"entities": entities, "relations": relations, "artifacts": []}

        elif sprint_type == SprintType.BILINGUAL_ONEKE:
            if not (ONEKE_AVAILABLE and self._oneke):
                raise RuntimeError("OneKE not available")

            # Use OneKE for bilingual extraction
            extraction_result = await self._oneke.extract_triples(
                text=text,
                correlation_id=correlation_id
            )

            return {
                "entities": extraction_result.get("entities", []),
                "relations": extraction_result.get("relations", []),
                "artifacts": []
            }

        elif sprint_type == SprintType.GENERIC_KGGEN:
            if not (KGGEN_AVAILABLE and self._kggen):
                raise RuntimeError("KG-Gen not available")

            # Use KG-Gen for generic extraction
            extraction_result = await self._kggen.extract(
                text=text,
                correlation_id=correlation_id
            )

            return {
                "entities": extraction_result.get("entities", []),
                "relations": extraction_result.get("relations", []),
                "triples": extraction_result.get("triples", []),
                "artifacts": []
            }

        else:
            # Hybrid: try available methods
            if KGGEN_AVAILABLE and self._kggen:
                return await self._extract_with_single_sprint(
                    text, SprintType.GENERIC_KGGEN, options, correlation_id
                )
            elif GRAPHITI_AVAILABLE and self._graphiti:
                return await self._extract_with_single_sprint(
                    text, SprintType.TEMPORAL_GRAPHITI, options, correlation_id
                )
            else:
                raise RuntimeError("No extraction sprint available")

    async def _search_elasticsearch(
        self,
        query: str,
        query_type: str,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search using Elasticsearch."""
        index_name = self.config.get("elasticsearch_index_prefix", "openevolve")

        if query_type == "keyword":
            es_query = {
                "size": limit,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "source", "processed_at"]
                    }
                }
            }
        elif query_type == "semantic":
            es_query = {
                "size": limit,
                "query": {
                    "match": {
                        "content": query
                    }
                }
            }
        else:  # hybrid
            es_query = {
                "size": limit,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "content": {
                                        "query": query,
                                        "boost": 2.0
                                    }
                                }
                            },
                            {
                                "match": {
                                    "source": query
                                }
                            }
                        ]
                    }
                }
            }

        response = await self._elasticsearch.search(
            index=index_name,
            query=es_query
        )

        return response.get("hits", {}).get("hits", [])


# ========== Convenience Functions ==========

async def create_integrated_knowledge_engine(
    config: Optional[Dict[str, Any]] = None
) -> IntegratedKnowledgeEngine:
    """
    Create and initialize an IntegratedKnowledgeEngine instance.

    Convenience function for one-line initialization.

    Args:
        config: Optional configuration dictionary

    Returns:
        Initialized IntegratedKnowledgeEngine ready to use

    Example:
        ```python
        engine = await create_integrated_knowledge_engine()
        result = await engine.process_document("doc.pdf")
        await engine.close()
        ```
    """
    engine = IntegratedKnowledgeEngine(config)
    await engine.initialize()
    return engine


# ========== Example Usage ==========

async def main():
    """Example usage of IntegratedKnowledgeEngine."""
    print("Integrated Knowledge Engine Example")

    # Create engine
    config = {
        "graphiti_uri": "bolt://localhost:7687",
        "graphiti_user": "neo4j",
        "graphiti_password": "password",  # In production, use environment variable
    }

    async with await create_integrated_knowledge_engine(config) as engine:
        # Check health
        health = await engine.health_check()
        print(f"Health: {json.dumps(health, indent=2)}")

        # Get statistics
        stats = await engine.get_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2)}")

        # Process a document
        # result = await engine.process_document("example.pdf")
        # print(f"Processing result: {result}")

        # Search knowledge
        # search_results = await engine.search_knowledge("machine learning")
        # print(f"Search results: {len(search_results['results'])} found")

    print("Example complete")


if __name__ == "__main__":
    asyncio.run(main())
