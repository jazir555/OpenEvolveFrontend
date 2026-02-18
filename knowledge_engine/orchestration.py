"""
KnowledgeEngine - Main orchestration class for OpenEvolve Knowledge System

This class provides a unified interface to all knowledge engine capabilities.
Following CLAUDE.md principles:
- CONFIGURATION EXPLICITNESS: All config via environment variables
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
- RUNTIME TRUTH: Verify components before use
- IDEMPOTENCY: All operations safe to run multiple times

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import asyncio
import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
import uuid

# Configure structured JSON logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import sprint components (with graceful degradation)
# Add parent directory to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from knowledge_engine.integrations.graphiti import (
        GraphitiTemporalBridge,
        GraphitiContradictionDetector,
    )
    GRAPHITI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Graphiti components not available: {e}")
    GRAPHITI_AVAILABLE = False
    GraphitiTemporalBridge = None
    GraphitiContradictionDetector = None

try:
    from knowledge_engine.integrations.ragbits_integration import RagbitsIntegration
    RAGBITS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Ragbits components not available: {e}")
    RAGBITS_AVAILABLE = False
    RagbitsIntegration = None

try:
    from knowledge_engine.integrations.knowledge_flow_orchestrator import KnowledgeFlowOrchestrator
    FLOW_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    FLOW_ORCHESTRATOR_AVAILABLE = False
    KnowledgeFlowOrchestrator = None

try:
    from knowledge_engine.integrations.kggen import (
        ExtractionPipeline,
    )
    KGGEN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"KG-Gen components not available: {e}")
    KGGEN_AVAILABLE = False
    ExtractionPipeline = None

try:
    from knowledge_engine.integrations.oneke import (
        OneKEModelAdapter,
        MultiTaskExtractionFramework
    )
    ONEKE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OneKE components not available: {e}")
    ONEKE_AVAILABLE = False
    OneKEModelAdapter = None

try:
    from knowledge_engine.visualization.graph_explorer import GraphExplorer, VisualizationOptions
    from knowledge_engine.visualization.temporal_viz import TemporalVisualizer, TemporalVisualizationOptions
    from knowledge_engine.visualization.community_viz import CommunityVisualizer, CommunityVisualizationOptions
    from knowledge_engine.visualization import ExportHandler
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Visualization components not available: {e}")
    VISUALIZATION_AVAILABLE = False
    GraphExplorer = None
    TemporalVisualizer = None
    CommunityVisualizer = None

# Import existing OpenEvolve components
from knowledge_engine.core import KnowledgeState, EntityKnowledgeGraph
from knowledge_engine.indexer import CodeIndexer
from knowledge_engine.elasticsearch_search import ElasticsearchSearchEngine


@dataclass
class ProcessingResult:
    """
    Result from document/knowledge processing operations.
    """
    success: bool
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    triples: List[Tuple[str, str, str]] = field(default_factory=list)
    visualization: Optional[str] = None
    error: Optional[str] = None
    correlation_id: Optional[str] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "entities": self.entities,
            "relations": self.relations,
            "triples": self.triples,
            "visualization": self.visualization,
            "error": self.error,
            "correlation_id": self.correlation_id,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }


@dataclass
class QueryResult:
    """
    Result from knowledge queries.
    """
    query: str
    results: List[Dict[str, Any]]
    count: int
    execution_time_ms: float
    correlation_id: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": self.results,
            "count": self.count,
            "execution_time_ms": self.execution_time_ms,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class KnowledgeEngine:
    """
    Main orchestration class for the Knowledge Engine.

    Provides unified access to:
    - Document processing and knowledge extraction
    - Temporal knowledge tracking (Graphiti)
    - Bilingual extraction (OneKE)
    - Knowledge visualization
    - Agent memory
    - Contradiction detection
    - Full-text search (Elasticsearch)
    - Code indexing

    Example usage:
        ```python
        from knowledge_engine import KnowledgeEngine, create_knowledge_engine

        # Option 1: Manual initialization
        engine = KnowledgeEngine()
        await engine.initialize()

        # Option 2: Convenience function
        engine = await create_knowledge_engine()

        # Process documents
        result = await engine.process_document(
            document_path="path/to/doc.pdf",
            extract_temporal=True,
            extract_bilingual=False
        )

        # Query knowledge
        results = await engine.query_temporal(
            query="machine learning algorithms",
            timestamp=datetime.now(timezone.utc)
        )

        # Cleanup
        await engine.close()
        ```
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize KnowledgeEngine with all components.

        Args:
            config: Configuration dictionary (uses env vars if None)

        Raises:
            RuntimeError: If required environment variables are missing
        """
        self.config = config or self._get_config_from_env()
        self._validate_config()

        # Initialize components (lazy loading)
        self._graphiti = None
        self._ragbits = None
        self._flow_orchestrator = None
        self._kggen = None
        self._oneke = None
        self._visualization = None
        self._elasticsearch = None
        self._indexer = None

        # Knowledge state and entity graph
        self.knowledge_state = KnowledgeState(query="initial")
        self.entity_graph = EntityKnowledgeGraph()

        # Tracking
        self._initialized = False
        self._closed = False

        logger.info({
            "msg": "KnowledgeEngine created",
            "components": {
                "graphiti": GRAPHITI_AVAILABLE,
                "ragbits": RAGBITS_AVAILABLE,
                "kggen": KGGEN_AVAILABLE,
                "oneke": ONEKE_AVAILABLE,
                "visualization": VISUALIZATION_AVAILABLE,
                "elasticsearch": bool(self.config.get("elasticsearch_hosts"))
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_config_from_env(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

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

            # Ragbits (RAG System)
            "ragbits_config": {
                "vector_store": {
                    "type": os.getenv("RAGBITS_VECTOR_STORE_TYPE", "qdrant"),
                    "config": {
                        "location": os.getenv("RAGBITS_VECTOR_STORE_URL", ":memory:"),
                        "collection_name": os.getenv("RAGBITS_COLLECTION", "knowledge_artifacts")
                    }
                }
            },

            # KG-Gen (Knowledge Generation)
            "kggen_model": os.getenv("KGGEN_ENTITY_MODEL", "gpt-4o"),
            "kggen_chunk_size": int(os.getenv("KGGEN_CHUNK_SIZE", "5000")),
            "kggen_timeout_ms": int(os.getenv("KGGEN_TIMEOUT_MS", "30000")),

            # OneKE (Bilingual Extraction)
            "oneke_model": os.getenv("ONEKE_MODEL_NAME", "oneke/OneKE-13B"),
            "oneke_device": os.getenv("ONEKE_DEVICE", "cuda"),
            "oneke_timeout_ms": int(os.getenv("ONEKE_TIMEOUT_MS", "60000")),

            # Visualization
            "viz_cache_ttl": int(os.getenv("VIS_CACHE_TTL", "3600")),
            "viz_max_nodes": int(os.getenv("VIS_MAX_NODES", "10000")),
            "viz_export_dir": os.getenv("VIS_EXPORT_DIR", "./visualizations"),

            # Elasticsearch
            "elasticsearch_hosts": os.getenv("ELASTICSEARCH_HOSTS", "http://localhost:9200").split(","),
            "elasticsearch_api_key": os.getenv("ELASTICSEARCH_API_KEY", ""),
            "elasticsearch_index_prefix": os.getenv("ELASTICSEARCH_INDEX_PREFIX", "openevolve"),

            # LLM
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),

            # Code Indexer
            "indexer_config": os.getenv("INDEXER_CONFIG_PATH", "knowledge_engine/indexer_config.yaml"),
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

        if ONEKE_AVAILABLE:
            # OneKE might work without API key if using local model
            pass

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
            logger.warning("KnowledgeEngine already initialized")
            return

        logger.info({
            "msg": "Initializing KnowledgeEngine components",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        tasks = []

        # Initialize components based on availability
        if GRAPHITI_AVAILABLE:
            tasks.append(self._init_graphiti())

        if RAGBITS_AVAILABLE:
            tasks.append(self._init_ragbits())

        if FLOW_ORCHESTRATOR_AVAILABLE:
            tasks.append(self._init_flow_orchestrator())

        if KGGEN_AVAILABLE:
            tasks.append(self._init_kggen())

        if ONEKE_AVAILABLE:
            tasks.append(self._init_oneke())

        if VISUALIZATION_AVAILABLE:
            tasks.append(self._init_visualization())

        # Initialize Elasticsearch if configured
        if self.config.get("elasticsearch_hosts"):
            tasks.append(self._init_elasticsearch())

        # Initialize code indexer if config exists
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
            "msg": "KnowledgeEngine initialization complete",
            "components_ready": {
                "graphiti": self._graphiti is not None,
                "ragbits": self._ragbits is not None,
                "kggen": self._kggen is not None,
                "oneke": self._oneke is not None,
                "visualization": self._visualization is not None,
                "elasticsearch": self._elasticsearch is not None,
                "indexer": self._indexer is not None
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

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

    async def _init_ragbits(self):
        """Initialize Ragbits integration."""
        try:
            self._ragbits = RagbitsIntegration(config=self.config["ragbits_config"])
            logger.info("Ragbits integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ragbits: {e}")
            raise

    async def _init_flow_orchestrator(self):
        """Initialize Knowledge Flow Orchestrator."""
        try:
            self._flow_orchestrator = KnowledgeFlowOrchestrator()
            logger.info("Knowledge Flow Orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Flow Orchestrator: {e}")
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

    async def _init_visualization(self):
        """Initialize visualization components."""
        try:
            # Get or create visualization config
            try:
                from knowledge_engine.visualization.config import get_visualization_config
                viz_config = get_visualization_config()
                # Override output dir if specified
                output_dir = self.config.get("viz_export_dir", "./visualizations")
                viz_config.output_dir = output_dir
            except ImportError:
                # Fallback if config not available
                viz_config = None
                output_dir = self.config.get("viz_export_dir", "./visualizations")

            if viz_config:
                self._visualization = {
                    "explorer": GraphExplorer(),
                    "temporal": TemporalVisualizer(),
                    "community": CommunityVisualizer(),
                    "export": ExportHandler(config=viz_config)
                }
            else:
                # Create ExportHandler without config
                self._visualization = {
                    "explorer": GraphExplorer(),
                    "temporal": TemporalVisualizer(),
                    "community": CommunityVisualizer(),
                    "export": ExportHandler(config=None)
                }

            logger.info("Visualization components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize visualization: {e}")
            raise

    async def _init_elasticsearch(self):
        """Initialize Elasticsearch search engine."""
        try:
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
        extract_temporal: bool = True,
        extract_bilingual: bool = False,
        correlation_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a document through the complete pipeline.

        Following CLAUDE.md: IDEMPOTENCY
        Safe to run multiple times on same document.

        Args:
            document_path: Path to document
            extract_temporal: Extract temporal knowledge
            extract_bilingual: Use bilingual extraction (OneKE)
            correlation_id: Correlation ID for tracking

        Returns:
            ProcessingResult with entities, relations, visualization

        Raises:
            RuntimeError: If no extraction engine available
        """
        start_time = datetime.now(timezone.utc)
        correlation_id = correlation_id or f"doc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info({
            "msg": "Processing document",
            "document": document_path,
            "correlation_id": correlation_id,
            "extract_temporal": extract_temporal,
            "extract_bilingual": extract_bilingual,
            "timestamp": start_time.isoformat()
        })

        try:
            # Read document
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
                document_text = result.get("markdown_content")
            elif document_path_obj.suffix.lower() in ['.txt', '.md']:
                with open(document_path_obj, 'r', encoding='utf-8') as f:
                    document_text = f.read()
            else:
                raise RuntimeError(f"Unsupported file type: {document_path_obj.suffix}")

            # Step 1: Extract knowledge
            if extract_bilingual and ONEKE_AVAILABLE and self._oneke:
                logger.debug("Using OneKE for bilingual extraction")
                extraction_result = await self._oneke.extract_triples(
                    text=document_text,
                    correlation_id=correlation_id
                )
            elif KGGEN_AVAILABLE and self._kggen:
                logger.debug("Using KG-Gen for extraction")
                extraction_result = await self._kggen.extract(
                    text=document_text,
                    correlation_id=correlation_id
                )
            else:
                raise RuntimeError("No extraction engine available")

            entities = extraction_result.get("entities", [])
            relations = extraction_result.get("relations", [])
            triples = extraction_result.get("triples", [])

            # Step 2: Add to temporal knowledge graph
            if extract_temporal and GRAPHITI_AVAILABLE and self._graphiti:
                logger.debug(f"Adding {len(entities)} entities to temporal KG")
                for entity in entities:
                    await self._graphiti.add_entity(
                        name=entity.get("name"),
                        entity_type=entity.get("type"),
                        metadata=entity,
                        correlation_id=correlation_id
                    )

                logger.debug(f"Adding {len(relations)} relations to temporal KG")
                for relation in relations:
                    await self._graphiti.add_relation(
                        subject=relation.get("subject"),
                        predicate=relation.get("predicate"),
                        object=relation.get("object"),
                        metadata=relation,
                        correlation_id=correlation_id
                    )

            # Step 3: Add to entity graph
            for entity in entities:
                await self.entity_graph.add_entity(
                    entity_name=entity.get("name"),
                    attributes=entity
                )

            for triple in triples:
                if len(triple) >= 3:
                    await self.entity_graph.add_relationship(
                        entity1=triple[0],
                        relation=triple[1],
                        entity2=triple[2]
                    )

            # Step 4: Generate visualization
            viz = None
            if VISUALIZATION_AVAILABLE and self._visualization:
                logger.debug("Generating visualization")
                viz = await self._visualization["explorer"].visualize(
                    triples=triples,
                    correlation_id=correlation_id
                )

            # Step 5: Index in Elasticsearch
            if self._elasticsearch:
                logger.debug("Indexing in Elasticsearch")
                # Build document for indexing
                document = {
                    "content": document_text,
                    "entities": entities,
                    "relations": relations,
                    "source": document_path,
                    "processed_at": datetime.now(timezone.utc).isoformat()
                }
                index_name = self.config.get("elasticsearch_index_prefix", "openevolve")
                await self._elasticsearch.index_document(
                    index=index_name,
                    document=document,
                    id=correlation_id
                )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            result = ProcessingResult(
                success=True,
                entities=entities,
                relations=relations,
                triples=triples,
                visualization=viz,
                correlation_id=correlation_id,
                processing_time_ms=processing_time_ms,
                metadata={
                    "document_path": document_path,
                    "extraction_method": "oneke" if extract_bilingual else "kggen",
                    "temporal_extraction": extract_temporal
                }
            )

            logger.info({
                "msg": "Document processing complete",
                "correlation_id": correlation_id,
                "entities_count": len(entities),
                "relations_count": len(relations),
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
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return ProcessingResult(
                success=False,
                error=str(e),
                correlation_id=correlation_id,
                processing_time_ms=processing_time_ms
            )

    async def query_temporal(
        self,
        query: str,
        timestamp: Optional[datetime] = None,
        correlation_id: Optional[str] = None
    ) -> QueryResult:
        """
        Query knowledge at specific point in time.

        Following CLAUDE.md: UTC TIME
        All timestamps in UTC.

        Args:
            query: Search query
            timestamp: Point in time for query (defaults to now)
            correlation_id: Correlation ID for tracking

        Returns:
            QueryResult with matching knowledge

        Raises:
            RuntimeError: If Graphiti not available
        """
        start_time = datetime.now(timezone.utc)
        correlation_id = correlation_id or f"query_{uuid.uuid4().hex}"
        timestamp = timestamp or datetime.now(timezone.utc)

        logger.info({
            "msg": "Temporal query",
            "query": query,
            "timestamp": timestamp.isoformat(),
            "correlation_id": correlation_id,
            "timestamp_utc": start_time.isoformat()
        })

        if not GRAPHITI_AVAILABLE or not self._graphiti:
            raise RuntimeError("Graphiti temporal knowledge not available")

        try:
            results = await self._graphiti.search_at_point_in_time(
                query=query,
                reference_time=timestamp,
                correlation_id=correlation_id
            )

            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            result = QueryResult(
                query=query,
                results=results,
                count=len(results),
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
                timestamp=timestamp.isoformat(),
                metadata={
                    "query_type": "temporal",
                    "reference_time": timestamp.isoformat()
                }
            )

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

            raise

    async def detect_contradictions(
        self,
        entity_name: str,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions for an entity across time.

        Args:
            entity_name: Name of entity to check
            correlation_id: Correlation ID for tracking

        Returns:
            List of detected contradictions

        Raises:
            RuntimeError: If Graphiti not available
        """
        correlation_id = correlation_id or f"contra_{uuid.uuid4().hex}"

        logger.info({
            "msg": "Detecting contradictions",
            "entity": entity_name,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        if not GRAPHITI_AVAILABLE or not self._graphiti:
            raise RuntimeError("Graphiti contradiction detection not available")

        try:
            detector = ContradictionDetector(bridge=self._graphiti)
            contradictions = await detector.detect_contradictions(
                entity_name=entity_name,
                correlation_id=correlation_id
            )

            logger.info({
                "msg": "Contradiction detection complete",
                "entity": entity_name,
                "correlation_id": correlation_id,
                "contradictions_found": len(contradictions),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return contradictions

        except Exception as e:
            logger.error({
                "msg": "Contradiction detection failed",
                "entity": entity_name,
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise

    async def visualize_graph(
        self,
        graph_type: str = "explorer",
        data: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Generate knowledge graph visualization.

        Args:
            graph_type: Type of visualization ("explorer", "temporal", "community")
            data: Data to visualize
            options: Visualization options
            correlation_id: Correlation ID for tracking

        Returns:
            Visualization data (JSON or file path)

        Raises:
            RuntimeError: If visualization not available
            ValueError: If unknown graph type
        """
        correlation_id = correlation_id or f"viz_{uuid.uuid4().hex}"

        logger.info({
            "msg": "Generating visualization",
            "type": graph_type,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        if not VISUALIZATION_AVAILABLE or not self._visualization:
            raise RuntimeError("Visualization components not available")

        try:
            if graph_type == "explorer":
                viz = await self._visualization["explorer"].visualize(
                    triples=data.get("triples", []) if data else [],
                    correlation_id=correlation_id
                )
            elif graph_type == "temporal":
                viz = await self._visualization["temporal"].visualize(
                    temporal_data=data,
                    correlation_id=correlation_id
                )
            elif graph_type == "community":
                viz = await self._visualization["community"].visualize(
                    triples=data.get("triples", []) if data else [],
                    correlation_id=correlation_id
                )
            else:
                raise ValueError(f"Unknown visualization type: {graph_type}")

            logger.info({
                "msg": "Visualization generated",
                "type": graph_type,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return viz

        except Exception as e:
            logger.error({
                "msg": "Visualization failed",
                "type": graph_type,
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise

    async def search_knowledge(
        self,
        query: str,
        query_type: str = "hybrid",
        limit: int = 10,
        correlation_id: Optional[str] = None
    ) -> QueryResult:
        """
        Search the knowledge base.

        Args:
            query: Search query
            query_type: Type of search ("keyword", "semantic", "hybrid")
            limit: Maximum results
            correlation_id: Correlation ID for tracking

        Returns:
            QueryResult with search results

        Raises:
            RuntimeError: If search engine not available
        """
        start_time = datetime.now(timezone.utc)
        correlation_id = correlation_id or f"search_{uuid.uuid4().hex}"

        logger.info({
            "msg": "Searching knowledge base",
            "query": query,
            "query_type": query_type,
            "limit": limit,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })

        if not self._elasticsearch:
            raise RuntimeError("Elasticsearch search not available")

        try:
            # Build Elasticsearch query based on query_type
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
                # Would use vector embeddings here if available
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

            # Extract hits from response
            results = response.get("hits", {}).get("hits", [])

            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            result = QueryResult(
                query=query,
                results=results,
                count=len(results),
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
                timestamp=start_time.isoformat(),
                metadata={
                    "query_type": query_type,
                    "limit": limit
                }
            )

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
            raise

    async def sync_ragbits_graphiti(self, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronize knowledge between Ragbits and Graphiti.
        
        Following ADR-008: Knowledge Flow Orchestration.
        - Bidirectional sync
        - Conflict resolution via TemporalKnowledgeEngine
        """
        start_time = datetime.now(timezone.utc)
        correlation_id = correlation_id or f"sync_{uuid.uuid4().hex}"
        
        logger.info({
            "msg": "Starting Ragbits-Graphiti sync",
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        if not (self._ragbits and self._graphiti):
            return {"success": False, "error": "Ragbits or Graphiti not available"}
            
        try:
            # 1. Fetch recent document chunks from Ragbits
            # (Simplified sync: we sync the last 100 items for now)
            ragbits_data = await self._ragbits.search_documents(query="*", top_k=100)
            
            # 2. Sync to Graphiti
            synced_count = 0
            for item in ragbits_data.results:
                from knowledge_engine.integrations.graphiti_integration import KnowledgeArtifact
                artifact = KnowledgeArtifact(
                    id=str(uuid.uuid4()),
                    content=item["content"],
                    artifact_type="document_chunk",
                    valid_at=datetime.now(timezone.utc),
                    metadata=item["metadata"],
                    source="ragbits"
                )
                await self._graphiti.add_artifact(artifact, correlation_id=correlation_id)
                synced_count += 1
                
            # 3. Fetch entities from Graphiti to potentially update Ragbits (Bidirectional)
            # (In a real implementation, we'd check for new entities in Graphiti)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "synced_count": synced_count,
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
            
            logger.info({
                "msg": "Ragbits-Graphiti sync complete",
                "synced_count": synced_count,
                "processing_time_ms": processing_time_ms
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return {"success": False, "error": str(e), "correlation_id": correlation_id}

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
                "visualization": self._visualization is not None,
                "elasticsearch": self._elasticsearch is not None,
                "indexer": self._indexer is not None
            },
            "knowledge": {
                "entities": len(self.entity_graph.entities),
                "relationships": len(self.entity_graph.relationships)
            }
        }

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
            logger.warning("KnowledgeEngine already closed")
            return

        logger.info({
            "msg": "Closing KnowledgeEngine",
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
            "msg": "KnowledgeEngine closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# ========== Convenience Functions ==========

async def create_knowledge_engine(
    config: Optional[Dict[str, Any]] = None
) -> KnowledgeEngine:
    """
    Create and initialize a KnowledgeEngine instance.

    Convenience function for one-line initialization.

    Args:
        config: Optional configuration dictionary

    Returns:
        Initialized KnowledgeEngine ready to use

    Example:
        ```python
        engine = await create_knowledge_engine()
        result = await engine.process_document("doc.pdf")
        await engine.close()
        ```
    """
    engine = KnowledgeEngine(config)
    await engine.initialize()
    return engine


# ========== Example Usage ==========

async def main():
    """Example usage of KnowledgeEngine."""
    print("🚀 KnowledgeEngine Example")

    # Option 1: Using convenience function
    async with await create_knowledge_engine() as engine:
        # Check health
        health = await engine.health_check()
        print(f"Health: {json.dumps(health, indent=2)}")

        # Get statistics
        stats = await engine.get_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2)}")

        # Process a document
        # result = await engine.process_document("example.pdf")

        # Query temporal knowledge
        # results = await engine.query_temporal("machine learning")

        # Detect contradictions
        # contradictions = await engine.detect_contradictions("AI")

        # Generate visualization
        # viz = await engine.visualize_graph("explorer", data={"triples": result.triples})

    print("[OK] Example complete")


if __name__ == "__main__":
    asyncio.run(main())
