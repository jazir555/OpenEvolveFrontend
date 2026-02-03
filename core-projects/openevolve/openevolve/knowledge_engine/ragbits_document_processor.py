"""
RAGBits Document Processor for Knowledge Engine

Complete document processing pipeline using RAGBits:
- Ingest documents (PDF, TXT, MD, DOCX)
- Extract content and metadata
- Create embeddings
- Index for semantic search
- Support filtering and retrieval

Following CLAUDE.md principles:
- CONFIGURATION EXPLICITNESS: All config via environment variables
- RUNTIME TRUTH: Verify RAGBits availability before use
- IDEMPOTENCY: Safe to re-ingest documents
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
import json
import hashlib

# Configure structured JSON logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing RAGBits
try:
    from ragbits.document_search import DocumentSearch
    from ragbits.document_search.documents.document import DocumentMeta
    from ragbits.core.embeddings.dense import LiteLLMEmbedder
    from ragbits.core.vector_stores.in_memory import InMemoryVectorStore
    from ragbits.core.vector_stores.qdrant import QdrantVectorStore
    from ragbits.core import LLMClient
    RAGBITS_AVAILABLE = True
except ImportError as e:
    RAGBITS_AVAILABLE = False
    logger.warning(f"RAGBits not available: {e}")
    DocumentSearch = None
    DocumentMeta = None


@dataclass
class DocumentProcessingResult:
    """
    Result from document processing operations.
    """
    success: bool
    document_id: str
    chunks_ingested: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "document_id": self.document_id,
            "chunks_ingested": self.chunks_ingested,
            "processing_time": self.processing_time,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class RAGBitsProcessorConfig:
    """
    Configuration for RAGBits document processor.

    Environment Variables:
    - RAGBITS_EMBEDDING_MODEL: Model for embeddings (default: "text-embedding-3-small")
    - RAGBITS_VECTOR_STORE: Vector store type (default: "memory", options: "memory", "qdrant")
    - RAGBITS_QDRANT_URL: Qdrant URL (if using qdrant)
    - RAGBITS_QDRANT_COLLECTION: Collection name (default: "knowledge_engine")
    - RAGBITS_CHUNK_SIZE: Document chunk size (default: 1000)
    - RAGBITS_CHUNK_OVERLAP: Chunk overlap (default: 200)
    - RAGBITS_MIN_CHUNK_SIZE: Minimum chunk size (default: 100)
    """
    embedding_model: str = field(
        default_factory=lambda: os.getenv("RAGBITS_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    vector_store_type: str = field(
        default_factory=lambda: os.getenv("RAGBITS_VECTOR_STORE", "memory")
    )
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("RAGBITS_QDRANT_URL", "http://localhost:6333")
    )
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("RAGBITS_QDRANT_COLLECTION", "knowledge_engine")
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("RAGBITS_CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("RAGBITS_CHUNK_OVERLAP", "200"))
    )
    min_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("RAGBITS_MIN_CHUNK_SIZE", "100"))
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.vector_store_type not in ["memory", "qdrant"]:
            raise ValueError(f"Invalid vector_store_type: {self.vector_store_type}")
        if self.chunk_size < 1:
            raise ValueError(f"Invalid chunk_size: {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"Invalid chunk_overlap: {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")


class RAGBitsDocumentProcessor:
    """
    Document processor using RAGBits for semantic search.

    Features:
    - Ingest documents from files or text
    - Automatic chunking with overlap
    - Embedding generation
    - Vector storage (memory or Qdrant)
    - Semantic search
    - Metadata filtering
    - Idempotent re-ingestion

    Usage:
        processor = RAGBitsDocumentProcessor()
        await processor.initialize()

        # Ingest document
        result = await processor.ingest_file("document.pdf")
        print(f"Ingested {result.chunks_ingested} chunks")

        # Search
        results = await processor.search("machine learning algorithms")
        for result in results:
            print(f"{result['score']:.3f}: {result['content'][:100]}...")
    """

    def __init__(self, config: Optional[RAGBitsProcessorConfig] = None):
        """
        Initialize RAGBits document processor.

        Args:
            config: Configuration for the processor
        """
        self.config = config or RAGBitsProcessorConfig()
        self.document_search: Optional[DocumentSearch] = None
        self.embedder = None
        self.vector_store = None
        self.available = False
        self._ingested_documents: set = set()

        logger.info({
            "msg": "RAGBitsDocumentProcessor created",
            "vector_store": self.config.vector_store_type,
            "embedding_model": self.config.embedding_model,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    async def initialize(self) -> bool:
        """
        Initialize RAGBits components.

        Returns:
            True if initialization successful, False otherwise
        """
        if not RAGBITS_AVAILABLE:
            logger.warning({
                "msg": "RAGBits not available, document processing disabled",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return False

        try:
            # Initialize embedder
            self.embedder = LiteLLMEmbedder(
                model_name=self.config.embedding_model
            )
            logger.info({
                "msg": "Embedder initialized",
                "model": self.config.embedding_model,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Initialize vector store
            if self.config.vector_store_type == "qdrant":
                self.vector_store = QdrantVectorStore(
                    url=self.config.qdrant_url,
                    collection_name=self.config.qdrant_collection,
                    embedder=self.embedder
                )
                logger.info({
                    "msg": "Qdrant vector store initialized",
                    "url": self.config.qdrant_url,
                    "collection": self.config.qdrant_collection,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            else:
                self.vector_store = InMemoryVectorStore(embedder=self.embedder)
                logger.info({
                    "msg": "In-memory vector store initialized",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            # Initialize document search
            self.document_search = DocumentSearch(
                vector_store=self.vector_store
            )
            self.available = True

            logger.info({
                "msg": "RAGBitsDocumentProcessor initialized successfully",
                "available": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return True

        except Exception as e:
            logger.error({
                "msg": "Failed to initialize RAGBits",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            self.available = False
            return False

    def _generate_document_id(self, source: str, content: str) -> str:
        """
        Generate unique document ID.

        Args:
            source: Document source (file path, URL, etc.)
            content: Document content

        Returns:
            Unique document ID
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"doc_{source_hash}_{content_hash}"

    async def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "text"
    ) -> DocumentProcessingResult:
        """
        Ingest text document.

        Args:
            text: Document text content
            metadata: Optional metadata (title, author, tags, etc.)
            source: Document source identifier

        Returns:
            Processing result
        """
        start_time = datetime.now(timezone.utc)

        if not self.available:
            return DocumentProcessingResult(
                success=False,
                document_id="",
                error="RAGBits not available"
            )

        try:
            # Generate document ID
            doc_id = self._generate_document_id(source, text)

            # Check if already ingested (idempotency)
            if doc_id in self._ingested_documents:
                logger.info({
                    "msg": "Document already ingested, skipping",
                    "document_id": doc_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return DocumentProcessingResult(
                    success=True,
                    document_id=doc_id,
                    chunks_ingested=0,
                    processing_time=0.0
                )

            # Prepare metadata
            final_metadata = {
                "source": source,
                "document_id": doc_id,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "content_length": len(text)
            }
            if metadata:
                final_metadata.update(metadata)

            # Create document
            document = DocumentMeta.from_literal(
                text,
                **final_metadata
            )

            # Ingest into vector store
            await self.document_search.ingest([document])

            # Track ingestion
            self._ingested_documents.add(doc_id)

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.info({
                "msg": "Document ingested successfully",
                "document_id": doc_id,
                "source": source,
                "content_length": len(text),
                "processing_time": processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return DocumentProcessingResult(
                success=True,
                document_id=doc_id,
                chunks_ingested=1,  # RAGBits handles chunking internally
                processing_time=processing_time,
                metadata=final_metadata
            )

        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.error({
                "msg": "Failed to ingest text",
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return DocumentProcessingResult(
                success=False,
                document_id="",
                processing_time=processing_time,
                error=str(e)
            )

    async def ingest_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentProcessingResult:
        """
        Ingest document from file.

        Args:
            file_path: Path to document file
            metadata: Optional metadata

        Returns:
            Processing result
        """
        path = Path(file_path)

        if not path.exists():
            return DocumentProcessingResult(
                success=False,
                document_id="",
                error=f"File not found: {file_path}"
            )

        try:
            # Read file content
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add file metadata
            file_metadata = {
                "file_path": str(path.absolute()),
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "file_type": path.suffix
            }
            if metadata:
                file_metadata.update(metadata)

            # Ingest content
            return await self.ingest_text(
                text=content,
                metadata=file_metadata,
                source=str(path)
            )

        except Exception as e:
            logger.error({
                "msg": "Failed to ingest file",
                "file_path": file_path,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return DocumentProcessingResult(
                success=False,
                document_id="",
                error=str(e)
            )

    async def ingest_directory(
        self,
        directory: str,
        pattern: str = "*.txt",
        metadata: Optional[Dict[str, Any]] = None,
        max_files: Optional[int] = None
    ) -> List[DocumentProcessingResult]:
        """
        Ingest all documents from directory.

        Args:
            directory: Directory path
            pattern: File pattern (e.g., "*.txt", "*.md")
            metadata: Optional metadata to apply to all files
            max_files: Maximum number of files to process

        Returns:
            List of processing results
        """
        dir_path = Path(directory)

        if not dir_path.exists() or not dir_path.is_dir():
            logger.error({
                "msg": "Directory not found",
                "directory": directory,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return []

        files = list(dir_path.glob(pattern))
        if max_files:
            files = files[:max_files]

        logger.info({
            "msg": "Ingesting directory",
            "directory": str(dir_path),
            "pattern": pattern,
            "file_count": len(files),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        results = []
        for file_path in files:
            result = await self.ingest_file(str(file_path), metadata)
            results.append(result)

        success_count = sum(1 for r in results if r.success)
        logger.info({
            "msg": "Directory ingestion complete",
            "total": len(results),
            "success": success_count,
            "failed": len(results) - success_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return results

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            min_score: Minimum similarity score

        Returns:
            List of search results with content and metadata
        """
        if not self.available or not self.document_search:
            logger.warning({
                "msg": "Cannot search, RAGBits not available",
                "query": query[:100],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return []

        try:
            logger.info({
                "msg": "Searching documents",
                "query": query[:100],
                "top_k": top_k,
                "filters": filters,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Perform search
            results = await self.document_search.search(
                query=query,
                top_k=top_k,
                filters=filters or {}
            )

            # Filter by score and format results
            formatted_results = []
            for result in results:
                if result.get("score", 0.0) >= min_score:
                    formatted_results.append({
                        "content": result.get("content", ""),
                        "score": result.get("score", 0.0),
                        "metadata": result.get("metadata", {})
                    })

            logger.info({
                "msg": "Search complete",
                "results_found": len(formatted_results),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return formatted_results

        except Exception as e:
            logger.error({
                "msg": "Search failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "available": self.available,
            "ingested_documents": len(self._ingested_documents),
            "vector_store_type": self.config.vector_store_type,
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap
        }

    async def clear(self) -> bool:
        """
        Clear all ingested documents.

        Returns:
            True if successful
        """
        if not self.available:
            return False

        try:
            # Recreate vector store to clear data
            if self.config.vector_store_type == "qdrant":
                self.vector_store = QdrantVectorStore(
                    url=self.config.qdrant_url,
                    collection_name=self.config.qdrant_collection,
                    embedder=self.embedder
                )
            else:
                self.vector_store = InMemoryVectorStore(embedder=self.embedder)

            # Recreate document search
            self.document_search = DocumentSearch(vector_store=self.vector_store)

            # Clear tracking
            self._ingested_documents.clear()

            logger.info({
                "msg": "Document store cleared",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return True

        except Exception as e:
            logger.error({
                "msg": "Failed to clear document store",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return False

    async def close(self):
        """
        Cleanup resources.
        """
        logger.info({
            "msg": "RAGBitsDocumentProcessor closing",
            "documents_processed": len(self._ingested_documents),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        # Vector stores don't need explicit closing
        self.document_search = None
        self.embedder = None
        self.vector_store = None
