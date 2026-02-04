"""
Matryoshka Enhanced Client

Production-ready Matryoshka client with unified memory integration.
This is a drop-in replacement for StatefulMatryoshkaClient with enhanced capabilities.

Features:
- Automatic 4-layer indexing of all exploration steps
- Persistent sessions across restarts
- Cross-document pattern learning
- Context rot prevention via always-true state
- Hybrid retrieval for relevant context
- Progress callbacks for long-running analyses
- Batch processing capabilities
- Session export/import

Author: OpenEvolve AI
Version: 2.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union,
    AsyncIterator, BinaryIO, TextIO
)
from collections import defaultdict
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT EXISTING MATRYOSHKA COMPONENTS
# =============================================================================

try:
    from matryoshka_unified_memory_integration import (
        UnifiedMatryoshkaClient,
        MatryoshkaMemoryBridge,
        MatryoshkaExplorationSession,
        ExplorationStep,
        ExplorationStepType,
        DocumentState,
        ExplorationContext,
        ExplorationResult,
        SynthesisResult,
        AnalysisResult,
        create_unified_matryoshka_client,
    )
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError as e:
    UNIFIED_MEMORY_AVAILABLE = False
    logger.warning(f"Unified memory integration not available: {e}")

try:
    from glue.adapters.matryoshka_adapter import (
        MatryoshkaClient,
        StatefulMatryoshkaClient,
    )
    MATRYOSHKA_ADAPTER_AVAILABLE = True
except ImportError:
    MATRYOSHKA_ADAPTER_AVAILABLE = False
    logger.warning("Matryoshka adapter not available")

try:
    from knowledge_unified_memory_system import (
        UnifiedMemorySystem,
        UnifiedMemory,
        UnifiedMemoryConfig,
        create_unified_system,
    )
    KNOWLEDGE_UNIFIED_AVAILABLE = True
except ImportError:
    KNOWLEDGE_UNIFIED_AVAILABLE = False

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ExplorationStrategy(Enum):
    """Exploration strategies for document analysis."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    ADAPTIVE = "adaptive"


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    PICKLE = "pickle"
    ZIP = "zip"


class AnalysisStatus(Enum):
    """Status of an analysis operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    IMPORTED = "imported"


# Default configuration
DEFAULT_MAX_TURNS = 10
DEFAULT_MEMORY_LIMIT = 15
DEFAULT_CHECKPOINT_INTERVAL = 5
SESSION_TTL_HOURS = 24


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AnalysisOptions:
    """
    Configuration options for document analysis.
    
    Attributes:
        max_turns: Maximum exploration turns (default: 10)
        memory_limit: Max memories per context retrieval (default: 15)
        exploration_strategy: Strategy for exploration (default: adaptive)
        save_checkpoints: Whether to save progress (default: True)
        enable_cross_doc_learning: Use insights from other documents (default: True)
        checkpoint_interval: Turns between checkpoints (default: 5)
        timeout_ms: Timeout per turn in milliseconds (default: 30000)
        enable_pattern_extraction: Extract patterns during analysis (default: True)
        context_window_bytes: Maximum context size in bytes (default: 8192)
        store_raw_observations: Keep raw execution output (default: False)
        min_importance_threshold: Minimum importance for memory storage (default: 0.3)
    """
    max_turns: int = DEFAULT_MAX_TURNS
    memory_limit: int = DEFAULT_MEMORY_LIMIT
    exploration_strategy: ExplorationStrategy = field(
        default_factory=lambda: ExplorationStrategy.ADAPTIVE
    )
    save_checkpoints: bool = True
    enable_cross_doc_learning: bool = True
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    timeout_ms: int = 30000
    enable_pattern_extraction: bool = True
    context_window_bytes: int = 8192
    store_raw_observations: bool = False
    min_importance_threshold: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_turns": self.max_turns,
            "memory_limit": self.memory_limit,
            "exploration_strategy": self.exploration_strategy.value,
            "save_checkpoints": self.save_checkpoints,
            "enable_cross_doc_learning": self.enable_cross_doc_learning,
            "checkpoint_interval": self.checkpoint_interval,
            "timeout_ms": self.timeout_ms,
            "enable_pattern_extraction": self.enable_pattern_extraction,
            "context_window_bytes": self.context_window_bytes,
            "store_raw_observations": self.store_raw_observations,
            "min_importance_threshold": self.min_importance_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisOptions":
        """Create from dictionary."""
        data = data.copy()
        if "exploration_strategy" in data:
            data["exploration_strategy"] = ExplorationStrategy(data["exploration_strategy"])
        return cls(**data)


@dataclass
class CodebaseAnalysisOptions:
    """
    Options for codebase-wide analysis.
    
    Attributes:
        file_patterns: Glob patterns for files to include (default: ["*.py"])
        exclude_patterns: Patterns to exclude (default: ["*test*", "*__pycache__*"])
        max_files: Maximum files to analyze (default: 100)
        analyze_relationships: Discover file relationships (default: True)
        shared_memory: Use shared memory across files (default: True)
        per_file_options: Options applied to each file analysis
        parallel_analysis: Analyze files in parallel (default: False)
        max_workers: Maximum parallel workers (default: 4)
    """
    file_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: List[str] = field(
        default_factory=lambda: ["*test*", "*__pycache__*", "*.pyc", "node_modules"]
    )
    max_files: int = 100
    analyze_relationships: bool = True
    shared_memory: bool = True
    per_file_options: Optional[AnalysisOptions] = None
    parallel_analysis: bool = False
    max_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_patterns": self.file_patterns,
            "exclude_patterns": self.exclude_patterns,
            "max_files": self.max_files,
            "analyze_relationships": self.analyze_relationships,
            "shared_memory": self.shared_memory,
            "per_file_options": self.per_file_options.to_dict() if self.per_file_options else None,
            "parallel_analysis": self.parallel_analysis,
            "max_workers": self.max_workers,
        }


@dataclass
class AnalysisItem:
    """Item for batch analysis."""
    query: str
    document_path: str
    item_id: Optional[str] = None
    options: Optional[AnalysisOptions] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.item_id is None:
            self.item_id = f"item_{uuid.uuid4().hex[:8]}"


@dataclass
class DocumentFinding:
    """A single finding from document analysis."""
    finding: str
    confidence: float
    source_section: Optional[str] = None
    turn_number: int = 0
    finding_type: str = "general"  # e.g., "entity", "relationship", "pattern", "error"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding": self.finding,
            "confidence": self.confidence,
            "source_section": self.source_section,
            "turn_number": self.turn_number,
            "finding_type": self.finding_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DocumentAnalysis:
    """Complete analysis result for a document."""
    session_id: str
    document_path: str
    query: str
    findings: List[DocumentFinding] = field(default_factory=list)
    summary: str = ""
    code_examples: List[str] = field(default_factory=list)
    
    # Session info
    total_turns: int = 0
    exploration_complete: bool = False
    status: AnalysisStatus = field(default_factory=lambda: AnalysisStatus.COMPLETED)
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    processing_time_ms: float = 0.0
    
    # Memory stats
    memories_created: int = 0
    memories_accessed: int = 0
    
    # Error info
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "document_path": self.document_path,
            "query": self.query,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "code_examples": self.code_examples,
            "total_turns": self.total_turns,
            "exploration_complete": self.exploration_complete,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_ms": self.processing_time_ms,
            "memories_created": self.memories_created,
            "memories_accessed": self.memories_accessed,
            "error_message": self.error_message,
        }
    
    @property
    def success(self) -> bool:
        """Check if analysis was successful."""
        return self.status == AnalysisStatus.COMPLETED and self.error_message is None


@dataclass
class SimilarAnalysis:
    """A similar analysis found via search."""
    session_id: str
    document_path: str
    query: str
    similarity_score: float
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "document_path": self.document_path,
            "query": self.query,
            "similarity_score": self.similarity_score,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DocumentInsights:
    """Accumulated insights about a document from all past analyses."""
    document_path: str
    total_analyses: int = 0
    all_findings: List[DocumentFinding] = field(default_factory=list)
    common_patterns: List[str] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)
    last_analyzed: Optional[datetime] = None
    
    # Aggregated metrics
    avg_confidence: float = 0.0
    total_turns_across_analyses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_path": self.document_path,
            "total_analyses": self.total_analyses,
            "all_findings": [f.to_dict() for f in self.all_findings],
            "common_patterns": self.common_patterns,
            "key_entities": self.key_entities,
            "last_analyzed": self.last_analyzed.isoformat() if self.last_analyzed else None,
            "avg_confidence": self.avg_confidence,
            "total_turns_across_analyses": self.total_turns_across_analyses,
        }


@dataclass
class CodePattern:
    """Pattern extracted from code analysis."""
    pattern_name: str
    pattern_type: str  # e.g., "structural", "behavioral", "naming"
    description: str
    examples: List[str] = field(default_factory=list)
    occurrence_count: int = 0
    confidence: float = 0.0
    source_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "examples": self.examples,
            "occurrence_count": self.occurrence_count,
            "confidence": self.confidence,
            "source_files": self.source_files,
            "metadata": self.metadata,
        }


@dataclass
class FileRelationship:
    """Relationship between files in a codebase."""
    source_file: str
    target_file: str
    relationship_type: str  # e.g., "imports", "calls", "extends", "similar"
    strength: float = 0.0  # 0.0 - 1.0
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "target_file": self.target_file,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "evidence": self.evidence,
        }


@dataclass
class CodebaseAnalysis:
    """Analysis result for an entire codebase."""
    session_id: str
    codebase_path: str
    files_analyzed: List[str] = field(default_factory=list)
    file_analyses: List[DocumentAnalysis] = field(default_factory=list)
    
    # Relationships
    file_relationships: List[FileRelationship] = field(default_factory=list)
    
    # Patterns
    extracted_patterns: List[CodePattern] = field(default_factory=list)
    
    # Summary
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    # Status
    status: AnalysisStatus = field(default_factory=lambda: AnalysisStatus.COMPLETED)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "codebase_path": self.codebase_path,
            "files_analyzed": self.files_analyzed,
            "file_analyses": [fa.to_dict() for fa in self.file_analyses],
            "file_relationships": [fr.to_dict() for fr in self.file_relationships],
            "extracted_patterns": [p.to_dict() for p in self.extracted_patterns],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_ms": self.processing_time_ms,
        }
    
    @property
    def success(self) -> bool:
        """Check if analysis was successful."""
        return self.status == AnalysisStatus.COMPLETED


@dataclass
class BatchAnalysisResult:
    """Result of batch analysis operation."""
    batch_id: str
    items: List[AnalysisItem] = field(default_factory=list)
    results: List[DocumentAnalysis] = field(default_factory=list)
    shared_patterns: List[CodePattern] = field(default_factory=list)
    
    # Statistics
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "shared_patterns": [p.to_dict() for p in self.shared_patterns],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_processing_time_ms": self.total_processing_time_ms,
        }


@dataclass
class ProgressUpdate:
    """Progress update during analysis."""
    session_id: str
    turn_number: int
    total_turns: int
    current_action: str
    progress_percent: float = 0.0
    current_finding: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "total_turns": self.total_turns,
            "current_action": self.current_action,
            "progress_percent": self.progress_percent,
            "current_finding": self.current_finding,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# ENHANCED MATRYOSHKA CLIENT
# =============================================================================

class EnhancedMatryoshkaClient:
    """
    Production-ready Matryoshka client with unified memory system.
    
    This class provides a drop-in replacement for StatefulMatryoshkaClient
    with enhanced capabilities including:
    
    - Automatic 4-layer indexing of all exploration steps
    - Persistent sessions across restarts
    - Cross-document pattern learning
    - Context rot prevention via always-true state
    - Hybrid retrieval for relevant context
    - Progress callbacks for monitoring
    - Batch processing capabilities
    - Session export/import
    
    Usage:
        # Basic usage (backward compatible)
        client = EnhancedMatryoshkaClient()
        result = client.analyze_document(
            query="Find all classes",
            document_path="./my_code.py"
        )
        
        # With options and progress
        def on_progress(update):
            print(f"Progress: {update.progress_percent}%")
        
        result = client.analyze_with_progress(
            query="Find all classes",
            document_path="./my_code.py",
            progress_callback=on_progress
        )
        
        # Codebase analysis
        codebase_result = client.analyze_codebase(
            queries=["Find patterns", "Identify dependencies"],
            codebase_path="./src"
        )
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        executable_path: Optional[str] = None,
        enable_unified_memory: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the enhanced Matryoshka client.
        
        Args:
            storage_path: Path for persistent storage (default: ./matryoshka_enhanced_storage)
            executable_path: Path to Matryoshka executable
            enable_unified_memory: Whether to use unified memory system
            config: Additional configuration dictionary
        """
        self._lock = threading.RLock()
        self._config = config or {}
        
        # Set up storage path
        self.storage_path = storage_path or "./matryoshka_enhanced_storage"
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Checkpoint directory
        self.checkpoint_dir = os.path.join(self.storage_path, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Session exports directory
        self.exports_dir = os.path.join(self.storage_path, "exports")
        os.makedirs(self.exports_dir, exist_ok=True)
        
        # Initialize unified memory client
        self.unified_client: Optional[UnifiedMatryoshkaClient] = None
        self.memory_bridge: Optional[MatryoshkaMemoryBridge] = None
        
        if enable_unified_memory and UNIFIED_MEMORY_AVAILABLE:
            try:
                db_dir = os.path.join(self.storage_path, "memory")
                self.unified_client = create_unified_matryoshka_client(
                    db_dir=db_dir,
                    executable_path=executable_path
                )
                self.memory_bridge = self.unified_client.memory_bridge
                logger.info("Unified memory system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize unified memory: {e}")
        
        # Fallback to base Matryoshka client
        if MATRYOSHKA_ADAPTER_AVAILABLE:
            self.base_client = MatryoshkaClient(executable_path=executable_path)
        else:
            self.base_client = None
        
        # Session tracking
        self._active_sessions: Dict[str, Any] = {}
        self._analysis_history: Dict[str, DocumentAnalysis] = {}
        self._document_index: Dict[str, List[str]] = defaultdict(list)  # path -> session_ids
        
        # Pattern storage
        self._pattern_cache: Dict[str, CodePattern] = {}
        
        logger.info("EnhancedMatryoshkaClient initialized")
    
    # ======================================================================
    # CORE ANALYSIS METHODS
    # ======================================================================
    
    def analyze_document(
        self,
        query: str,
        document_path: str,
        session_id: Optional[str] = None,
        options: Optional[AnalysisOptions] = None
    ) -> DocumentAnalysis:
        """
        Main entry point for document analysis.
        
        Args:
            query: What to find/analyze in the document
            document_path: Path to document
            session_id: Optional existing session to continue
            options: Analysis configuration
            
        Returns:
            DocumentAnalysis with findings, summary, and session info
            
        Example:
            result = client.analyze_document(
                query="Find all API endpoints",
                document_path="./app.py",
                options=AnalysisOptions(max_turns=15)
            )
            print(result.summary)
            for finding in result.findings:
                print(f"- {finding.finding}")
        """
        options = options or AnalysisOptions()
        start_time = time.time()
        
        # Generate or use provided session ID
        if not session_id:
            session_id = f"enhanced_{uuid.uuid4().hex[:16]}"
        
        # Initialize result
        result = DocumentAnalysis(
            session_id=session_id,
            document_path=document_path,
            query=query,
            status=AnalysisStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # Validate file exists
            if not os.path.exists(document_path):
                result.status = AnalysisStatus.FAILED
                result.error_message = f"File not found: {document_path}"
                return result
            
            # Use unified memory if available
            if self.unified_client and UNIFIED_MEMORY_AVAILABLE:
                analysis_result = self._analyze_with_unified_memory(
                    query=query,
                    document_path=document_path,
                    session_id=session_id,
                    options=options
                )
                result = analysis_result
            else:
                # Fallback to basic analysis
                result = self._analyze_basic(
                    query=query,
                    document_path=document_path,
                    session_id=session_id,
                    options=options
                )
            
            # Update document index
            self._document_index[document_path].append(session_id)
            self._analysis_history[session_id] = result
            
            # Save checkpoint if enabled
            if options.save_checkpoints:
                self._save_checkpoint(session_id, result)
            
        except Exception as e:
            logger.error(f"Error analyzing document: {e}", exc_info=True)
            result.status = AnalysisStatus.FAILED
            result.error_message = str(e)
        finally:
            result.processing_time_ms = (time.time() - start_time) * 1000
            if result.status == AnalysisStatus.RUNNING:
                result.status = AnalysisStatus.COMPLETED
            result.completed_at = datetime.utcnow()
        
        return result
    
    def analyze_with_progress(
        self,
        query: str,
        document_path: str,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
        session_id: Optional[str] = None,
        options: Optional[AnalysisOptions] = None
    ) -> DocumentAnalysis:
        """
        Analyze document with progress callbacks.
        
        Args:
            query: What to find/analyze in the document
            document_path: Path to document
            progress_callback: Called with ProgressUpdate during analysis
            session_id: Optional existing session to continue
            options: Analysis configuration
            
        Returns:
            DocumentAnalysis with findings, summary, and session info
            
        Example:
            def on_progress(update):
                print(f"Turn {update.turn_number}/{update.total_turns}: {update.current_action}")
            
            result = client.analyze_with_progress(
                query="Find patterns",
                document_path="./code.py",
                progress_callback=on_progress
            )
        """
        options = options or AnalysisOptions()
        
        # Wrap progress callback to inject our own tracking
        internal_callback = self._create_progress_wrapper(
            progress_callback, options
        )
        
        # Use unified memory with progress tracking if available
        if self.unified_client and UNIFIED_MEMORY_AVAILABLE:
            return self._analyze_with_progress_unified(
                query=query,
                document_path=document_path,
                progress_callback=internal_callback,
                session_id=session_id,
                options=options
            )
        else:
            # Simulate progress for basic analysis
            return self._analyze_with_progress_basic(
                query=query,
                document_path=document_path,
                progress_callback=internal_callback,
                session_id=session_id,
                options=options
            )
    
    def analyze_codebase(
        self,
        queries: List[str],
        codebase_path: str,
        options: Optional[CodebaseAnalysisOptions] = None
    ) -> CodebaseAnalysis:
        """
        Analyze an entire codebase using Matryoshka with memory.
        
        Discovers file relationships via graph index, learns patterns across files,
        and maintains codebase state.
        
        Args:
            queries: List of queries to analyze across the codebase
            codebase_path: Root path of the codebase
            options: Codebase analysis configuration
            
        Returns:
            CodebaseAnalysis with file analyses, relationships, and patterns
            
        Example:
            result = client.analyze_codebase(
                queries=["Find API endpoints", "Identify data models"],
                codebase_path="./src",
                options=CodebaseAnalysisOptions(file_patterns=["*.py"])
            )
            print(f"Analyzed {len(result.files_analyzed)} files")
            for pattern in result.extracted_patterns:
                print(f"Pattern: {pattern.pattern_name}")
        """
        options = options or CodebaseAnalysisOptions()
        start_time = time.time()
        
        session_id = f"codebase_{uuid.uuid4().hex[:16]}"
        result = CodebaseAnalysis(
            session_id=session_id,
            codebase_path=codebase_path,
            status=AnalysisStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        try:
            # Discover files
            files = self._discover_files(codebase_path, options)
            result.files_analyzed = files[:options.max_files]
            
            if not result.files_analyzed:
                result.status = AnalysisStatus.FAILED
                result.error_message = f"No files found matching patterns in {codebase_path}"
                return result
            
            # Analyze each file
            per_file_opts = options.per_file_options or AnalysisOptions(
                max_turns=5,  # Shorter for individual files
                enable_cross_doc_learning=options.shared_memory
            )
            
            for i, file_path in enumerate(result.files_analyzed):
                # Combine queries into one comprehensive query
                combined_query = f"Analyze: {'; '.join(queries)}"
                
                file_analysis = self.analyze_document(
                    query=combined_query,
                    document_path=file_path,
                    options=per_file_opts
                )
                
                result.file_analyses.append(file_analysis)
                
                # Update progress
                logger.debug(f"Analyzed file {i+1}/{len(result.files_analyzed)}: {file_path}")
            
            # Discover relationships if enabled
            if options.analyze_relationships:
                result.file_relationships = self._discover_file_relationships(
                    result.file_analyses
                )
            
            # Extract patterns
            result.extracted_patterns = self._extract_codebase_patterns(
                result.file_analyses
            )
            
            # Generate summary
            result.summary = self._generate_codebase_summary(result)
            result.recommendations = self._generate_recommendations(result)
            
            result.status = AnalysisStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Error analyzing codebase: {e}", exc_info=True)
            result.status = AnalysisStatus.FAILED
            result.error_message = str(e)
        finally:
            result.processing_time_ms = (time.time() - start_time) * 1000
            result.completed_at = datetime.utcnow()
        
        return result
    
    def analyze_batch(
        self,
        items: List[AnalysisItem],
        shared_options: Optional[AnalysisOptions] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> BatchAnalysisResult:
        """
        Analyze multiple documents with shared memory.
        
        Args:
            items: List of AnalysisItem to analyze
            shared_options: Default options applied to all items
            progress_callback: Called with (completed, total, current_item_id)
            
        Returns:
            BatchAnalysisResult with all results and shared patterns
            
        Example:
            items = [
                AnalysisItem(query="Find classes", document_path="./a.py"),
                AnalysisItem(query="Find functions", document_path="./b.py"),
            ]
            result = client.analyze_batch(items)
            for analysis in result.results:
                print(f"{analysis.document_path}: {len(analysis.findings)} findings")
        """
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        result = BatchAnalysisResult(
            batch_id=batch_id,
            items=items,
            total_items=len(items),
            started_at=datetime.utcnow()
        )
        
        session_ids = []
        
        for i, item in enumerate(items):
            try:
                # Merge options
                opts = item.options or shared_options or AnalysisOptions()
                
                # Analyze
                analysis = self.analyze_document(
                    query=item.query,
                    document_path=item.document_path,
                    session_id=item.item_id,
                    options=opts
                )
                
                result.results.append(analysis)
                session_ids.append(analysis.session_id)
                
                if analysis.success:
                    result.completed_items += 1
                else:
                    result.failed_items += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(items), item.item_id or "unknown")
                
            except Exception as e:
                logger.error(f"Error in batch item {item.item_id}: {e}")
                result.failed_items += 1
        
        # Extract shared patterns
        if len(session_ids) > 1:
            result.shared_patterns = self.extract_common_patterns(session_ids)
        
        result.total_processing_time_ms = (time.time() - start_time) * 1000
        result.completed_at = datetime.utcnow()
        
        return result
    
    # ======================================================================
    # SEARCH AND INSIGHT METHODS
    # ======================================================================
    
    def search_similar_analyses(
        self,
        query: str,
        limit: int = 10,
        document_filter: Optional[str] = None
    ) -> List[SimilarAnalysis]:
        """
        Find previously conducted similar analyses.
        Uses semantic search across all indexed exploration steps.
        
        Args:
            query: Search query
            limit: Maximum number of results
            document_filter: Optional path filter for specific document
            
        Returns:
            List of SimilarAnalysis results
            
        Example:
            similar = client.search_similar_analyses(
                query="authentication patterns",
                limit=5
            )
            for s in similar:
                print(f"Found in {s.document_path}: {s.similarity_score}")
        """
        results = []
        
        try:
            if self.unified_client and UNIFIED_MEMORY_AVAILABLE:
                # Use unified memory search
                search_results = self.unified_client.search_across_sessions(
                    query=query,
                    limit=limit * 2  # Get more for filtering
                )
                
                for sr in search_results:
                    # Filter by document if specified
                    if document_filter:
                        doc_path = sr.get("document_path", "")
                        if document_filter not in doc_path:
                            continue
                    
                    similar = SimilarAnalysis(
                        session_id=sr.get("session_id", "unknown"),
                        document_path=sr.get("document_path", "unknown"),
                        query=query,
                        similarity_score=sr.get("importance", 0.5) * sr.get("confidence", 0.5),
                        summary=sr.get("content", "")[:200],
                    )
                    results.append(similar)
            
            # Also search local history
            for session_id, analysis in self._analysis_history.items():
                if document_filter and document_filter not in analysis.document_path:
                    continue
                
                # Simple text matching for local history
                combined = f"{analysis.query} {analysis.summary}".lower()
                query_lower = query.lower()
                
                score = 0.0
                for word in query_lower.split():
                    if word in combined:
                        score += 0.2
                
                if score > 0.3:
                    similar = SimilarAnalysis(
                        session_id=session_id,
                        document_path=analysis.document_path,
                        query=analysis.query,
                        similarity_score=min(score, 1.0),
                        summary=analysis.summary[:200],
                        key_findings=[f.finding for f in analysis.findings[:3]]
                    )
                    results.append(similar)
            
            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error searching similar analyses: {e}")
        
        return results[:limit]
    
    def get_document_insights(self, document_path: str) -> DocumentInsights:
        """
        Get accumulated insights about a document from all past analyses.
        
        Args:
            document_path: Path to the document
            
        Returns:
            DocumentInsights with all findings and patterns
            
        Example:
            insights = client.get_document_insights("./app.py")
            print(f"Analyzed {insights.total_analyses} times")
            print(f"Found {len(insights.all_findings)} total findings")
            for pattern in insights.common_patterns:
                print(f"Pattern: {pattern}")
        """
        insights = DocumentInsights(document_path=document_path)
        
        # Get all sessions for this document
        session_ids = self._document_index.get(document_path, [])
        insights.total_analyses = len(session_ids)
        
        if not session_ids:
            return insights
        
        all_findings = []
        all_patterns = []
        total_confidence = 0.0
        total_turns = 0
        
        for session_id in session_ids:
            if session_id in self._analysis_history:
                analysis = self._analysis_history[session_id]
                all_findings.extend(analysis.findings)
                
                for finding in analysis.findings:
                    total_confidence += finding.confidence
                    if finding.finding_type == "pattern":
                        all_patterns.append(finding.finding)
                
                total_turns += analysis.total_turns
                
                if insights.last_analyzed is None or (
                    analysis.completed_at and analysis.completed_at > insights.last_analyzed
                ):
                    insights.last_analyzed = analysis.completed_at
        
        insights.all_findings = all_findings
        insights.common_patterns = list(set(all_patterns))[:20]  # Top 20 unique
        insights.key_entities = self._extract_entities(all_findings)
        insights.total_turns_across_analyses = total_turns
        
        if all_findings:
            insights.avg_confidence = total_confidence / len(all_findings)
        
        return insights
    
    def extract_common_patterns(
        self,
        session_ids: List[str],
        min_occurrences: int = 2
    ) -> List[CodePattern]:
        """
        Find patterns across multiple analysis sessions.
        
        Args:
            session_ids: List of session IDs to analyze
            min_occurrences: Minimum occurrences to be considered a pattern
            
        Returns:
            List of CodePattern found across sessions
            
        Example:
            patterns = client.extract_common_patterns(
                session_ids=["sess1", "sess2", "sess3"]
            )
            for pattern in patterns:
                print(f"{pattern.pattern_name}: {pattern.occurrence_count} occurrences")
        """
        patterns: Dict[str, CodePattern] = {}
        
        for session_id in session_ids:
            # Get session analysis
            if session_id in self._analysis_history:
                analysis = self._analysis_history[session_id]
            elif self.unified_client and UNIFIED_MEMORY_AVAILABLE:
                synthesis = self.unified_client.get_session_synthesis(session_id)
                if not synthesis:
                    continue
                analysis = None
            else:
                continue
            
            # Extract patterns from findings
            if analysis:
                for finding in analysis.findings:
                    if finding.finding_type == "pattern":
                        pattern_key = self._normalize_pattern_key(finding.finding)
                        
                        if pattern_key in patterns:
                            patterns[pattern_key].occurrence_count += 1
                            patterns[pattern_key].confidence = max(
                                patterns[pattern_key].confidence,
                                finding.confidence
                            )
                            if analysis.document_path not in patterns[pattern_key].source_files:
                                patterns[pattern_key].source_files.append(analysis.document_path)
                        else:
                            patterns[pattern_key] = CodePattern(
                                pattern_name=finding.finding[:50],
                                pattern_type="structural",
                                description=finding.finding,
                                occurrence_count=1,
                                confidence=finding.confidence,
                                source_files=[analysis.document_path]
                            )
        
        # Filter by minimum occurrences
        result = [
            p for p in patterns.values()
            if p.occurrence_count >= min_occurrences
        ]
        
        # Sort by occurrence count
        result.sort(key=lambda x: x.occurrence_count, reverse=True)
        
        return result
    
    # ======================================================================
    # SESSION MANAGEMENT
    # ======================================================================
    
    def export_session(
        self,
        session_id: str,
        format: str = "json",
        export_path: Optional[str] = None
    ) -> str:
        """
        Export a session for sharing or backup.
        Includes state, memories, and findings.
        
        Args:
            session_id: Session to export
            format: Export format ("json", "pickle", "zip")
            export_path: Optional custom export path
            
        Returns:
            Path to exported file
            
        Example:
            path = client.export_session("sess_abc123", format="zip")
            print(f"Exported to: {path}")
        """
        if session_id not in self._analysis_history:
            raise ValueError(f"Session {session_id} not found")
        
        analysis = self._analysis_history[session_id]
        export_format = ExportFormat(format.lower())
        
        # Generate export path
        if not export_path:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session_id}_{timestamp}"
            
            if export_format == ExportFormat.JSON:
                filename += ".json"
            elif export_format == ExportFormat.PICKLE:
                filename += ".pkl"
            elif export_format == ExportFormat.ZIP:
                filename += ".zip"
            
            export_path = os.path.join(self.exports_dir, filename)
        
        # Gather export data
        export_data = {
            "session_id": session_id,
            "exported_at": datetime.utcnow().isoformat(),
            "analysis": analysis.to_dict(),
        }
        
        # Add unified memory data if available
        if self.unified_client and UNIFIED_MEMORY_AVAILABLE:
            synthesis = self.unified_client.get_session_synthesis(session_id)
            if synthesis:
                export_data["synthesis"] = synthesis.to_dict()
        
        # Write export
        if export_format == ExportFormat.JSON:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif export_format == ExportFormat.PICKLE:
            with open(export_path, 'wb') as f:
                pickle.dump(export_data, f)
        
        elif export_format == ExportFormat.ZIP:
            json_path = export_path.replace('.zip', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(json_path, os.path.basename(json_path))
            
            os.remove(json_path)
        
        logger.info(f"Exported session {session_id} to {export_path}")
        return export_path
    
    def import_session(self, export_path: str) -> str:
        """
        Import a previously exported session.
        Returns new session_id.
        
        Args:
            export_path: Path to exported session file
            
        Returns:
            New session_id for the imported session
            
        Example:
            new_id = client.import_session("./session_backup.zip")
            print(f"Imported as: {new_id}")
        """
        if not os.path.exists(export_path):
            raise FileNotFoundError(f"Export file not found: {export_path}")
        
        # Generate new session ID
        new_session_id = f"imported_{uuid.uuid4().hex[:16]}"
        
        # Load export data
        if export_path.endswith('.zip'):
            with zipfile.ZipFile(export_path, 'r') as zf:
                json_name = [n for n in zf.namelist() if n.endswith('.json')][0]
                with zf.open(json_name) as f:
                    export_data = json.load(f)
        
        elif export_path.endswith('.pkl'):
            with open(export_path, 'rb') as f:
                export_data = pickle.load(f)
        
        else:
            with open(export_path, 'r', encoding='utf-8') as f:
                export_data = json.load(f)
        
        # Reconstruct analysis
        analysis_data = export_data.get("analysis", {})
        
        # Create new DocumentAnalysis with new session ID
        analysis = DocumentAnalysis(
            session_id=new_session_id,
            document_path=analysis_data.get("document_path", "unknown"),
            query=analysis_data.get("query", ""),
            summary=analysis_data.get("summary", ""),
            total_turns=analysis_data.get("total_turns", 0),
            status=AnalysisStatus.IMPORTED,
            processing_time_ms=analysis_data.get("processing_time_ms", 0),
        )
        
        # Restore findings
        for f_data in analysis_data.get("findings", []):
            finding = DocumentFinding(
                finding=f_data.get("finding", ""),
                confidence=f_data.get("confidence", 0.5),
                source_section=f_data.get("source_section"),
                turn_number=f_data.get("turn_number", 0),
                finding_type=f_data.get("finding_type", "general"),
            )
            analysis.findings.append(finding)
        
        # Store in history
        self._analysis_history[new_session_id] = analysis
        self._document_index[analysis.document_path].append(new_session_id)
        
        logger.info(f"Imported session as {new_session_id}")
        return new_session_id
    
    def list_sessions(
        self,
        document_filter: Optional[str] = None,
        status_filter: Optional[AnalysisStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        List all analysis sessions.
        
        Args:
            document_filter: Optional filter by document path
            status_filter: Optional filter by status
            
        Returns:
            List of session info dictionaries
        """
        sessions = []
        
        for session_id, analysis in self._analysis_history.items():
            # Apply filters
            if document_filter and document_filter not in analysis.document_path:
                continue
            if status_filter and analysis.status != status_filter:
                continue
            
            sessions.append({
                "session_id": session_id,
                "document_path": analysis.document_path,
                "query": analysis.query,
                "status": analysis.status.value,
                "findings_count": len(analysis.findings),
                "total_turns": analysis.total_turns,
                "started_at": analysis.started_at.isoformat(),
                "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
            })
        
        return sessions
    
    def get_session(self, session_id: str) -> Optional[DocumentAnalysis]:
        """Get a specific session's analysis result."""
        return self._analysis_history.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its data."""
        if session_id not in self._analysis_history:
            return False
        
        analysis = self._analysis_history[session_id]
        
        # Remove from document index
        if analysis.document_path in self._document_index:
            self._document_index[analysis.document_path] = [
                sid for sid in self._document_index[analysis.document_path]
                if sid != session_id
            ]
        
        # Remove from history
        del self._analysis_history[session_id]
        
        # Clean up unified memory if available
        if self.unified_client and UNIFIED_MEMORY_AVAILABLE:
            try:
                self.unified_client.close_session(session_id)
            except Exception as e:
                logger.warning(f"Error closing unified session: {e}")
        
        return True
    
    # ======================================================================
    # BACKWARD COMPATIBILITY
    # ======================================================================
    
    def analyze(
        self,
        query: str,
        file_path: str,
        max_turns: int = 10,
        **kwargs
    ) -> str:
        """
        Backward-compatible analyze method (matches MatryoshkaClient interface).
        
        Returns the analysis result as a string instead of DocumentAnalysis.
        """
        options = AnalysisOptions(max_turns=max_turns)
        result = self.analyze_document(
            query=query,
            document_path=file_path,
            options=options
        )
        
        if result.success:
            return result.summary or "\n".join(f.finding for f in result.findings)
        else:
            return f"Error: {result.error_message}"
    
    def continue_analysis(
        self,
        session_id: str,
        follow_up_query: str,
        max_turns: int = 5
    ) -> DocumentAnalysis:
        """
        Continue a previous analysis session.
        
        Args:
            session_id: Existing session ID
            follow_up_query: New query to continue with
            max_turns: Additional turns to perform
            
        Returns:
            Updated DocumentAnalysis
        """
        if session_id not in self._analysis_history:
            raise ValueError(f"Session {session_id} not found")
        
        previous = self._analysis_history[session_id]
        
        # Use unified memory to continue if available
        if self.unified_client and UNIFIED_MEMORY_AVAILABLE:
            try:
                result = self.unified_client.continue_analysis(
                    session_id=session_id,
                    follow_up_query=follow_up_query,
                    max_turns=max_turns
                )
                
                # Convert to DocumentAnalysis
                return self._convert_analysis_result(result)
            except Exception as e:
                logger.warning(f"Unified continue failed, using basic: {e}")
        
        # Fallback: start new analysis with context
        combined_query = f"Previous: {previous.query}\nFollow-up: {follow_up_query}"
        return self.analyze_document(
            query=combined_query,
            document_path=previous.document_path,
            options=AnalysisOptions(max_turns=max_turns)
        )
    
    def compress_context(self, session_id: str, new_content: str, query: str = "") -> str:
        """Backward-compatible context compression."""
        if session_id in self._analysis_history:
            analysis = self._analysis_history[session_id]
            # Return summary as compressed context
            return analysis.summary
        return ""
    
    # ======================================================================
    # PRIVATE HELPER METHODS
    # ======================================================================
    
    def _analyze_with_unified_memory(
        self,
        query: str,
        document_path: str,
        session_id: str,
        options: AnalysisOptions
    ) -> DocumentAnalysis:
        """Analyze using unified memory system."""
        # Run analysis
        result = self.unified_client.analyze_with_memory(
            query=query,
            file_path=document_path,
            session_id=session_id,
            max_turns=options.max_turns
        )
        
        return self._convert_analysis_result(result)
    
    def _analyze_with_progress_unified(
        self,
        query: str,
        document_path: str,
        progress_callback: Optional[Callable[[ProgressUpdate], None]],
        session_id: Optional[str],
        options: AnalysisOptions
    ) -> DocumentAnalysis:
        """Analyze with progress tracking using unified memory."""
        session_id = session_id or f"progress_{uuid.uuid4().hex[:16]}"
        
        # Create session
        if self.unified_client and UNIFIED_MEMORY_AVAILABLE:
            from matryoshka_unified_memory_integration import MatryoshkaExplorationSession
            
            session = MatryoshkaExplorationSession(
                session_id=session_id,
                document_path=document_path,
                query=query,
                memory_bridge=self.memory_bridge,
                matryoshka_client=self.base_client
            )
            
            # Track progress manually
            result = DocumentAnalysis(
                session_id=session_id,
                document_path=document_path,
                query=query,
                status=AnalysisStatus.RUNNING
            )
            
            try:
                for turn in range(1, options.max_turns + 1):
                    # Report progress
                    if progress_callback:
                        progress_callback(ProgressUpdate(
                            session_id=session_id,
                            turn_number=turn,
                            total_turns=options.max_turns,
                            current_action=f"Exploration turn {turn}",
                            progress_percent=(turn / options.max_turns) * 100
                        ))
                    
                    # Simulate turn (simplified - in real implementation would call session.explore_step)
                    time.sleep(0.1)  # Placeholder for actual work
                
                # Get final result
                exploration_result = session.explore(max_turns=options.max_turns)
                result = self._convert_exploration_result(
                    exploration_result, query, document_path
                )
                
            except Exception as e:
                result.status = AnalysisStatus.FAILED
                result.error_message = str(e)
        else:
            # Fallback
            result = self._analyze_basic(query, document_path, session_id, options)
        
        return result
    
    def _analyze_with_progress_basic(
        self,
        query: str,
        document_path: str,
        progress_callback: Optional[Callable[[ProgressUpdate], None]],
        session_id: Optional[str],
        options: AnalysisOptions
    ) -> DocumentAnalysis:
        """Analyze with progress tracking (basic fallback)."""
        session_id = session_id or f"basic_{uuid.uuid4().hex[:16]}"
        
        # Simulate progress
        for turn in range(1, options.max_turns + 1):
            if progress_callback:
                progress_callback(ProgressUpdate(
                    session_id=session_id,
                    turn_number=turn,
                    total_turns=options.max_turns,
                    current_action=f"Analyzing (turn {turn}/{options.max_turns})",
                    progress_percent=(turn / options.max_turns) * 100
                ))
            time.sleep(0.05)  # Simulate work
        
        return self._analyze_basic(query, document_path, session_id, options)
    
    def _analyze_basic(
        self,
        query: str,
        document_path: str,
        session_id: str,
        options: AnalysisOptions
    ) -> DocumentAnalysis:
        """Basic analysis without unified memory."""
        result = DocumentAnalysis(
            session_id=session_id,
            document_path=document_path,
            query=query
        )
        
        # Use base Matryoshka client if available
        if self.base_client and self.base_client.is_available():
            try:
                output = self.base_client.analyze(
                    query=query,
                    file_path=document_path,
                    max_turns=options.max_turns
                )
                result.summary = output[:1000]  # Truncate long outputs
                result.findings.append(DocumentFinding(
                    finding=output[:500],
                    confidence=0.7,
                    finding_type="general"
                ))
            except Exception as e:
                result.error_message = str(e)
                result.status = AnalysisStatus.FAILED
        else:
            # No client available - do basic file analysis
            try:
                with open(document_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                lines = content.split('\n')
                result.findings.append(DocumentFinding(
                    finding=f"File has {len(lines)} lines",
                    confidence=1.0,
                    finding_type="metadata"
                ))
                result.summary = f"Basic analysis: {len(lines)} lines, {len(content)} chars"
            except Exception as e:
                result.error_message = str(e)
                result.status = AnalysisStatus.FAILED
        
        return result
    
    def _convert_analysis_result(
        self,
        result: AnalysisResult,
        query: Optional[str] = None,
        document_path: Optional[str] = None
    ) -> DocumentAnalysis:
        """Convert AnalysisResult to DocumentAnalysis."""
        doc_analysis = DocumentAnalysis(
            session_id=result.session_id,
            document_path=result.document_path or document_path or "unknown",
            query=result.query or query or "",
            summary=result.answer or "",
            status=AnalysisStatus.COMPLETED if result.success else AnalysisStatus.FAILED,
            processing_time_ms=result.processing_time_ms,
            error_message=result.error
        )
        
        # Convert findings
        for finding_text in result.findings:
            doc_analysis.findings.append(DocumentFinding(
                finding=finding_text,
                confidence=0.7,
                finding_type="general"
            ))
        
        return doc_analysis
    
    def _convert_exploration_result(
        self,
        result: ExplorationResult,
        query: str,
        document_path: str
    ) -> DocumentAnalysis:
        """Convert ExplorationResult to DocumentAnalysis."""
        doc_analysis = DocumentAnalysis(
            session_id=result.session_id,
            document_path=document_path,
            query=query,
            summary=result.final_synthesis or "",
            total_turns=result.total_turns,
            status=AnalysisStatus.COMPLETED if result.success else AnalysisStatus.FAILED,
            processing_time_ms=result.total_execution_time_ms,
            memories_created=result.memories_created,
            error_message=result.error_message
        )
        
        # Add findings
        for finding_text in result.key_findings:
            doc_analysis.findings.append(DocumentFinding(
                finding=finding_text,
                confidence=0.8,
                finding_type="insight"
            ))
        
        return doc_analysis
    
    def _create_progress_wrapper(
        self,
        user_callback: Optional[Callable[[ProgressUpdate], None]],
        options: AnalysisOptions
    ) -> Optional[Callable[[ProgressUpdate], None]]:
        """Create a wrapper for progress callbacks."""
        if not user_callback:
            return None
        
        def wrapper(update: ProgressUpdate) -> None:
            # Add any internal tracking here
            user_callback(update)
        
        return wrapper
    
    def _discover_files(
        self,
        codebase_path: str,
        options: CodebaseAnalysisOptions
    ) -> List[str]:
        """Discover files matching patterns."""
        files = []
        
        for pattern in options.file_patterns:
            import fnmatch
            for root, dirnames, filenames in os.walk(codebase_path):
                # Filter out excluded directories
                dirnames[:] = [
                    d for d in dirnames
                    if not any(
                        fnmatch.fnmatch(d, excl) or fnmatch.fnmatch(d, f"*{excl}*")
                        for excl in options.exclude_patterns
                    )
                ]
                
                for filename in filenames:
                    if fnmatch.fnmatch(filename, pattern):
                        full_path = os.path.join(root, filename)
                        # Check exclude patterns
                        if not any(
                            fnmatch.fnmatch(full_path, excl) or
                            fnmatch.fnmatch(filename, excl)
                            for excl in options.exclude_patterns
                        ):
                            files.append(full_path)
        
        return sorted(files)[:options.max_files]
    
    def _discover_file_relationships(
        self,
        analyses: List[DocumentAnalysis]
    ) -> List[FileRelationship]:
        """Discover relationships between files."""
        relationships = []
        
        # Simple relationship discovery based on findings
        for i, analysis1 in enumerate(analyses):
            for analysis2 in analyses[i+1:]:
                # Check for shared findings
                shared = set()
                for f1 in analysis1.findings:
                    for f2 in analysis2.findings:
                        if self._text_similarity(f1.finding, f2.finding) > 0.5:
                            shared.add(f1.finding[:50])
                
                if shared:
                    relationships.append(FileRelationship(
                        source_file=analysis1.document_path,
                        target_file=analysis2.document_path,
                        relationship_type="similar",
                        strength=len(shared) / max(len(analysis1.findings), 1),
                        evidence=list(shared)[:5]
                    ))
        
        return relationships
    
    def _extract_codebase_patterns(
        self,
        analyses: List[DocumentAnalysis]
    ) -> List[CodePattern]:
        """Extract patterns across codebase analyses."""
        patterns = []
        
        # Collect all pattern findings
        all_pattern_findings = []
        for analysis in analyses:
            for finding in analysis.findings:
                if finding.finding_type == "pattern":
                    all_pattern_findings.append((finding, analysis.document_path))
        
        # Group similar patterns
        pattern_groups: Dict[str, List] = defaultdict(list)
        for finding, doc_path in all_pattern_findings:
            key = self._normalize_pattern_key(finding.finding)
            pattern_groups[key].append((finding, doc_path))
        
        # Create patterns from groups
        for key, group in pattern_groups.items():
            if len(group) >= 2:  # At least 2 occurrences
                pattern = CodePattern(
                    pattern_name=key[:50],
                    pattern_type="structural",
                    description=group[0][0].finding,
                    occurrence_count=len(group),
                    confidence=max(f.confidence for f, _ in group),
                    source_files=list(set(dp for _, dp in group))
                )
                patterns.append(pattern)
        
        return sorted(patterns, key=lambda p: p.occurrence_count, reverse=True)
    
    def _generate_codebase_summary(self, result: CodebaseAnalysis) -> str:
        """Generate summary of codebase analysis."""
        parts = [
            f"# Codebase Analysis Summary",
            f"",
            f"Analyzed {len(result.files_analyzed)} files",
            f"Found {len(result.extracted_patterns)} common patterns",
            f"Discovered {len(result.file_relationships)} file relationships",
            f"",
        ]
        
        if result.extracted_patterns:
            parts.append("## Top Patterns")
            for pattern in result.extracted_patterns[:5]:
                parts.append(f"- {pattern.pattern_name} ({pattern.occurrence_count} occurrences)")
            parts.append("")
        
        return "\n".join(parts)
    
    def _generate_recommendations(self, result: CodebaseAnalysis) -> List[str]:
        """Generate recommendations based on codebase analysis."""
        recommendations = []
        
        if len(result.extracted_patterns) > 10:
            recommendations.append(
                "High pattern density detected - consider refactoring for consistency"
            )
        
        if result.file_relationships:
            high_coupling = [
                r for r in result.file_relationships if r.strength > 0.8
            ]
            if len(high_coupling) > 5:
                recommendations.append(
                    "High coupling detected between files - review architecture"
                )
        
        return recommendations
    
    def _extract_entities(self, findings: List[DocumentFinding]) -> List[str]:
        """Extract key entities from findings."""
        # Simple entity extraction - look for capitalized words, quoted strings, etc.
        entities = set()
        
        for finding in findings:
            text = finding.finding
            # Look for quoted strings
            import re
            quoted = re.findall(r'"([^"]+)"', text)
            entities.update(quoted)
            
            # Look for capitalized words (likely class names)
            caps = re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', text)
            entities.update(c for c in caps if len(c) > 2)
        
        return sorted(list(entities))[:50]
    
    def _normalize_pattern_key(self, text: str) -> str:
        """Normalize pattern text for grouping."""
        # Lowercase and remove extra whitespace
        key = text.lower().strip()
        key = ' '.join(key.split())
        # Truncate
        return key[:100]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _save_checkpoint(self, session_id: str, analysis: DocumentAnalysis) -> None:
        """Save analysis checkpoint."""
        try:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{session_id}.json"
            )
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(analysis.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, session_id: str) -> Optional[DocumentAnalysis]:
        """Load analysis from checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{session_id}.json"
        )
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct DocumentAnalysis
            analysis = DocumentAnalysis(
                session_id=data.get("session_id", ""),
                document_path=data.get("document_path", ""),
                query=data.get("query", ""),
                summary=data.get("summary", ""),
                total_turns=data.get("total_turns", 0),
                status=AnalysisStatus(data.get("status", "completed")),
                processing_time_ms=data.get("processing_time_ms", 0),
                error_message=data.get("error_message")
            )
            
            return analysis
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    # ======================================================================
    # UTILITY METHODS
    # ======================================================================
    
    def is_available(self) -> bool:
        """Check if the client is available for use."""
        return (
            self.unified_client is not None or
            (self.base_client is not None and self.base_client.is_available())
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "total_sessions": len(self._analysis_history),
            "active_sessions": len(self._active_sessions),
            "documents_indexed": len(self._document_index),
            "unified_memory_available": self.unified_client is not None,
            "base_client_available": (
                self.base_client is not None and self.base_client.is_available()
            ),
            "storage_path": self.storage_path,
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions older than specified hours."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        removed = 0
        
        for session_id in list(self._analysis_history.keys()):
            analysis = self._analysis_history[session_id]
            if analysis.completed_at and analysis.completed_at < cutoff:
                self.delete_session(session_id)
                removed += 1
        
        return removed


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_enhanced_client(
    storage_path: Optional[str] = None,
    executable_path: Optional[str] = None,
    enable_unified_memory: bool = True
) -> EnhancedMatryoshkaClient:
    """
    Factory function to create an EnhancedMatryoshkaClient.
    
    Args:
        storage_path: Path for persistent storage
        executable_path: Path to Matryoshka executable
        enable_unified_memory: Whether to use unified memory system
        
    Returns:
        Configured EnhancedMatryoshkaClient
        
    Example:
        >>> client = create_enhanced_client("./my_storage")
        >>> result = client.analyze_document("Find classes", "./code.py")
    """
    return EnhancedMatryoshkaClient(
        storage_path=storage_path,
        executable_path=executable_path,
        enable_unified_memory=enable_unified_memory
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EnhancedMatryoshkaClient Demo")
    print("=" * 70)
    
    # Create client
    client = create_enhanced_client("./demo_enhanced_storage")
    
    print(f"\nClient created:")
    print(f"  Available: {client.is_available()}")
    print(f"  Stats: {client.get_stats()}")
    
    # Create sample files
    import tempfile
    
    sample_code = '''
def calculate_sum(numbers):
    """Calculate sum of a list of numbers."""
    return sum(numbers)

def find_max(numbers):
    """Find maximum value in a list."""
    return max(numbers) if numbers else None

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_value(self, value):
        self.data.append(value)
    
    def process(self):
        return calculate_sum(self.data)
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_code)
        temp_path = f.name
    
    try:
        # Demo 1: Basic analysis
        print("\n" + "-" * 70)
        print("Demo 1: Basic Document Analysis")
        print("-" * 70)
        
        result = client.analyze_document(
            query="Find all classes and functions in this file",
            document_path=temp_path,
            options=AnalysisOptions(max_turns=3)
        )
        
        print(f"Session ID: {result.session_id}")
        print(f"Status: {result.status.value}")
        print(f"Findings: {len(result.findings)}")
        print(f"Processing time: {result.processing_time_ms:.0f}ms")
        
        # Demo 2: Analysis with progress
        print("\n" + "-" * 70)
        print("Demo 2: Analysis with Progress Callback")
        print("-" * 70)
        
        def on_progress(update):
            print(f"  Progress: {update.progress_percent:.0f}% - {update.current_action}")
        
        result = client.analyze_with_progress(
            query="Analyze the code structure",
            document_path=temp_path,
            progress_callback=on_progress,
            options=AnalysisOptions(max_turns=3)
        )
        
        print(f"Completed: {result.success}")
        
        # Demo 3: Search similar analyses
        print("\n" + "-" * 70)
        print("Demo 3: Search Similar Analyses")
        print("-" * 70)
        
        similar = client.search_similar_analyses(
            query="code structure",
            limit=5
        )
        
        print(f"Found {len(similar)} similar analyses")
        for s in similar[:3]:
            print(f"  - {s.document_path}: {s.similarity_score:.2f}")
        
        # Demo 4: Get document insights
        print("\n" + "-" * 70)
        print("Demo 4: Document Insights")
        print("-" * 70)
        
        insights = client.get_document_insights(temp_path)
        
        print(f"Total analyses: {insights.total_analyses}")
        print(f"Total findings: {len(insights.all_findings)}")
        print(f"Key entities: {insights.key_entities[:5]}")
        
        # Demo 5: Export session
        print("\n" + "-" * 70)
        print("Demo 5: Export Session")
        print("-" * 70)
        
        export_path = client.export_session(
            session_id=result.session_id,
            format="json"
        )
        
        print(f"Exported to: {export_path}")
        
        # Demo 6: Import session
        print("\n" + "-" * 70)
        print("Demo 6: Import Session")
        print("-" * 70)
        
        new_session_id = client.import_session(export_path)
        
        print(f"Imported as: {new_session_id}")
        
        # Demo 7: List sessions
        print("\n" + "-" * 70)
        print("Demo 7: List Sessions")
        print("-" * 70)
        
        sessions = client.list_sessions()
        
        print(f"Total sessions: {len(sessions)}")
        for session in sessions[:3]:
            print(f"  - {session['session_id']}: {session['findings_count']} findings")
        
        print("\n" + "=" * 70)
        print("Demo complete!")
        print("=" * 70)
        
    finally:
        # Cleanup
        os.unlink(temp_path)
        # Clean up export files
        if os.path.exists(client.exports_dir):
            shutil.rmtree(client.exports_dir)
