"""
Checkpoint & Resume System for OpenEvolve Gauntlet System

Provides reliable checkpointing and resume capabilities for long-running
problem solving pipelines, enabling crash recovery and fault tolerance.

Key Features:
- State serialization and deserialization
- Checkpoint storage (database or file-based)
- Automatic checkpoint creation at key pipeline stages
- Resume from last checkpoint on failure
- Checkpoint cleanup and retention management
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import asyncio
from pathlib import Path
import hashlib

# **ACTUAL INTEGRATION**: Alerting and knowledge for Checkpoint Manager
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    checkpoint_id: str
    problem_id: str
    timestamp: datetime
    level: int  # Hierarchy level
    stage: str  # decomposition, solving, reassembly, validation
    state_size: int = 0
    compressed: bool = False
    parent_checkpoint_id: Optional[str] = None


@dataclass
class PipelineState:
    """Complete state of the pipeline at a point in time"""
    problem: Dict[str, Any]
    context: Dict[str, Any]
    solutions: Dict[str, Any] = field(default_factory=dict)
    decomposition_tree: Optional[Dict[str, Any]] = None
    execution_status: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateSerializer:
    """
    Serializes and deserializes pipeline state for checkpointing.
    """

    def __init__(self, compression_enabled: bool = False):
        self.compression_enabled = compression_enabled

    async def serialize(self, state: PipelineState) -> bytes:
        """
        Serialize pipeline state to bytes.

        Args:
            state: Pipeline state to serialize

        Returns:
            Serialized state as bytes
        """
        # Convert to dict
        state_dict = {
            'problem': state.problem,
            'context': self._sanitize_context(state.context),
            'solutions': state.solutions,
            'decomposition_tree': state.decomposition_tree,
            'execution_status': state.execution_status,
            'metrics': state.metrics,
            'metadata': state.metadata,
            'serialized_at': datetime.utcnow().isoformat(),
        }

        # Convert to JSON
        json_str = json.dumps(state_dict, default=str, indent=2)

        # Compress if enabled
        if self.compression_enabled:
            try:
                import gzip
                return gzip.compress(json_str.encode('utf-8'))
            except ImportError:
                logger.warning("gzip not available, skipping compression")
                return json_str.encode('utf-8')

        return json_str.encode('utf-8')

    async def deserialize(self, data: bytes) -> PipelineState:
        """
        Deserialize bytes to pipeline state.

        Args:
            data: Serialized state data

        Returns:
            PipelineState object
        """
        # Decompress if needed
        if self.compression_enabled:
            try:
                import gzip
                # Try to decompress
                try:
                    data = gzip.decompress(data)
                except OSError:
                    # Not compressed, continue
                    pass
            except ImportError:
                pass

        # Parse JSON
        state_dict = json.loads(data.decode('utf-8'))

        # Reconstruct state
        return PipelineState(
            problem=state_dict.get('problem', {}),
            context=state_dict.get('context', {}),
            solutions=state_dict.get('solutions', {}),
            decomposition_tree=state_dict.get('decomposition_tree'),
            execution_status=state_dict.get('execution_status', {}),
            metrics=state_dict.get('metrics', {}),
            metadata=state_dict.get('metadata', {}),
        )

    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize context for serialization.

        Removes non-serializable items like database connections,
        file handles, etc.
        """
        sanitized = {}

        for key, value in context.items():
            # Skip functions and classes
            if callable(value):
                continue

            # Skip objects with certain attributes
            if hasattr(value, 'conn') or hasattr(value, 'cursor'):
                continue

            # Include serializable values
            try:
                json.dumps(value)
                sanitized[key] = value
            except (TypeError, ValueError):
                # Not serializable, skip
                logger.debug(f"Skipping non-serializable context key: {key}")

        return sanitized


class CheckpointRepository:
    """
    Repository for storing and retrieving checkpoints.
    """

    def __init__(self, storage_type: str = 'file', storage_path: str = './checkpoints'):
        self.storage_type = storage_type
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache for fast access
        self._cache: Dict[str, tuple[bytes, CheckpointMetadata]] = {}

    async def save(
        self,
        checkpoint_id: str,
        data: bytes,
        metadata: CheckpointMetadata
    ) -> bool:
        """
        Save checkpoint data.

        Args:
            checkpoint_id: Unique checkpoint identifier
            data: Checkpoint data
            metadata: Checkpoint metadata

        Returns:
            True if saved successfully
        """
        try:
            if self.storage_type == 'file':
                return await self._save_file(checkpoint_id, data, metadata)
            elif self.storage_type == 'memory':
                self._cache[checkpoint_id] = (data, metadata)
                return True
            else:
                logger.error(f"Unknown storage type: {self.storage_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    async def load(self, checkpoint_id: str) -> Optional[tuple[bytes, CheckpointMetadata]]:
        """
        Load checkpoint data.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Tuple of (data, metadata) or None if not found
        """
        try:
            if self.storage_type == 'file':
                return await self._load_file(checkpoint_id)
            elif self.storage_type == 'memory':
                return self._cache.get(checkpoint_id)
            else:
                logger.error(f"Unknown storage type: {self.storage_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint"""
        try:
            if self.storage_type == 'file':
                checkpoint_file = self.storage_path / f"{checkpoint_id}.checkpoint"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    return True
                return False
            elif self.storage_type == 'memory':
                if checkpoint_id in self._cache:
                    del self._cache[checkpoint_id]
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False

    async def list_checkpoints(self, problem_id: str = None) -> List[CheckpointMetadata]:
        """List all checkpoints, optionally filtered by problem_id"""
        if self.storage_type == 'file':
            return await self._list_files(problem_id)
        elif self.storage_type == 'memory':
            return [metadata for _, metadata in self._cache.values()
                   if problem_id is None or metadata.problem_id == problem_id]
        return []

    async def cleanup_old_checkpoints(self, problem_id: str, keep_last_n: int = 5) -> int:
        """
        Clean up old checkpoints, keeping only the most recent N.

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = await self.list_checkpoints(problem_id)

        if len(checkpoints) <= keep_last_n:
            return 0

        # Sort by timestamp, oldest first
        checkpoints.sort(key=lambda c: c.timestamp)

        # Delete oldest checkpoints
        to_delete = checkpoints[:-keep_last_n]
        deleted_count = 0

        for checkpoint in to_delete:
            if await self.delete(checkpoint.checkpoint_id):
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old checkpoints for problem {problem_id}")
        return deleted_count

    async def _save_file(
        self,
        checkpoint_id: str,
        data: bytes,
        metadata: CheckpointMetadata
    ) -> bool:
        """Save checkpoint to file"""
        checkpoint_file = self.storage_path / f"{checkpoint_id}.checkpoint"
        meta_file = self.storage_path / f"{checkpoint_id}.meta"

        # Save data
        with open(checkpoint_file, 'wb') as f:
            f.write(data)

        # Save metadata
        with open(meta_file, 'w') as f:
            json.dump({
                'checkpoint_id': metadata.checkpoint_id,
                'problem_id': metadata.problem_id,
                'timestamp': metadata.timestamp.isoformat(),
                'level': metadata.level,
                'stage': metadata.stage,
                'state_size': metadata.state_size,
                'compressed': metadata.compressed,
                'parent_checkpoint_id': metadata.parent_checkpoint_id,
            }, f, indent=2)

        return True

    async def _load_file(self, checkpoint_id: str) -> Optional[tuple[bytes, CheckpointMetadata]]:
        """Load checkpoint from file"""
        checkpoint_file = self.storage_path / f"{checkpoint_id}.checkpoint"
        meta_file = self.storage_path / f"{checkpoint_id}.meta"

        if not checkpoint_file.exists() or not meta_file.exists():
            return None

        # Load data
        with open(checkpoint_file, 'rb') as f:
            data = f.read()

        # Load metadata
        with open(meta_file, 'r') as f:
            meta_dict = json.load(f)

        metadata = CheckpointMetadata(
            checkpoint_id=meta_dict['checkpoint_id'],
            problem_id=meta_dict['problem_id'],
            timestamp=datetime.fromisoformat(meta_dict['timestamp']),
            level=meta_dict['level'],
            stage=meta_dict['stage'],
            state_size=meta_dict.get('state_size', 0),
            compressed=meta_dict.get('compressed', False),
            parent_checkpoint_id=meta_dict.get('parent_checkpoint_id'),
        )

        return (data, metadata)

    async def _list_files(self, problem_id: str = None) -> List[CheckpointMetadata]:
        """List checkpoints from files"""
        checkpoints = []

        for meta_file in self.storage_path.glob("*.meta"):
            try:
                with open(meta_file, 'r') as f:
                    meta_dict = json.load(f)

                if problem_id is not None and meta_dict['problem_id'] != problem_id:
                    continue

                metadata = CheckpointMetadata(
                    checkpoint_id=meta_dict['checkpoint_id'],
                    problem_id=meta_dict['problem_id'],
                    timestamp=datetime.fromisoformat(meta_dict['timestamp']),
                    level=meta_dict['level'],
                    stage=meta_dict['stage'],
                    state_size=meta_dict.get('state_size', 0),
                    compressed=meta_dict.get('compressed', False),
                    parent_checkpoint_id=meta_dict.get('parent_checkpoint_id'),
                )
                checkpoints.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read checkpoint metadata from {meta_file}: {e}")

        return checkpoints


class CheckpointManager:
    """
    Main checkpoint manager for the Gauntlet pipeline.

    Manages checkpoint creation, loading, and cleanup.
    """

    def __init__(
        self,
        repository: CheckpointRepository = None,
        serializer: StateSerializer = None,
        enabled: bool = True,
        auto_cleanup: bool = True
    ):
        self.repository = repository or CheckpointRepository()
        self.serializer = serializer or StateSerializer()
        self.enabled = enabled
        self.auto_cleanup = auto_cleanup
        self.checkpoint_count = 0

    async def create_checkpoint(
        self,
        problem: Dict[str, Any],
        context: Dict[str, Any],
        solutions: Dict[str, Any] = None,
        decomposition_tree: Dict[str, Any] = None,
        execution_status: Dict[str, str] = None,
        metrics: Dict[str, Any] = None,
        level: int = 0,
        stage: str = 'unknown',
        parent_checkpoint_id: str = None
    ) -> Optional[str]:
        """
        Create a checkpoint of the current pipeline state.

        Args:
            problem: Current problem being solved
            context: Execution context
            solutions: Solutions found so far
            decomposition_tree: Current decomposition hierarchy
            execution_status: Status of subproblems
            metrics: Execution metrics
            level: Hierarchy level
            stage: Pipeline stage
            parent_checkpoint_id: Parent checkpoint for hierarchy

        Returns:
            Checkpoint ID if created successfully, None otherwise
        """
        if not self.enabled:
            return None

        try:
            # Create pipeline state
            state = PipelineState(
                problem=problem,
                context=context,
                solutions=solutions or {},
                decomposition_tree=decomposition_tree,
                execution_status=execution_status or {},
                metrics=metrics or {},
                metadata={'checkpoint_count': self.checkpoint_count},
            )

            # Serialize state
            data = await self.serializer.serialize(state)

            # Generate checkpoint ID
            problem_id = problem.get('id', 'unknown')
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
            checkpoint_id = f"{problem_id}_{level}_{stage}_{timestamp}"

            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                problem_id=problem_id,
                timestamp=datetime.utcnow(),
                level=level,
                stage=stage,
                state_size=len(data),
                compressed=self.serializer.compression_enabled,
                parent_checkpoint_id=parent_checkpoint_id,
            )

            # Save checkpoint
            success = await self.repository.save(checkpoint_id, data, metadata)

            if success:
                self.checkpoint_count += 1
                logger.info(f"[OK] Checkpoint created: {checkpoint_id} ({len(data)} bytes)")

                # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful checkpoint
                self._extract_checkpoint_knowledge("create_checkpoint", checkpoint_id, metadata, state)
                self._track_checkpoint_performance("create_checkpoint", True, len(data))

                # Auto-cleanup old checkpoints
                if self.auto_cleanup:
                    await self.repository.cleanup_old_checkpoints(problem_id, keep_last_n=5)

                return checkpoint_id
            else:
                logger.error(f"Failed to save checkpoint: {checkpoint_id}")

                # **ACTUAL INTEGRATION**: Trigger alert, track performance for failed checkpoint
                self._trigger_checkpoint_alerts("create_checkpoint", False, problem_id, checkpoint_id, stage, "Failed to save checkpoint")
                self._track_checkpoint_performance("create_checkpoint", False, len(data))

                return None

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return None

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[PipelineState]:
        """
        Load a checkpoint.

        Args:
            checkpoint_id: Checkpoint to load

        Returns:
            PipelineState if loaded successfully, None otherwise
        """
        if not self.enabled:
            return None

        try:
            result = await self.repository.load(checkpoint_id)

            if result is None:
                logger.warning(f"Checkpoint not found: {checkpoint_id}")

                # **ACTUAL INTEGRATION**: Trigger alert for missing checkpoint
                self._trigger_checkpoint_alerts("resume_from_checkpoint", False, None, checkpoint_id, None, "Checkpoint not found")
                self._track_checkpoint_performance("resume_from_checkpoint", False)

                return None

            data, metadata = result

            # Deserialize state
            import time
            start_time = time.time()
            state = await self.serializer.deserialize(data)
            load_time = time.time() - start_time

            logger.info(f"[OK] Checkpoint loaded: {checkpoint_id} from {metadata.timestamp}")

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful resume
            self._extract_checkpoint_knowledge("resume_from_checkpoint", checkpoint_id, metadata, state)
            self._track_checkpoint_performance("resume_from_checkpoint", True, len(data), load_time)

            return state

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

            # **ACTUAL INTEGRATION**: Trigger alert and track performance for failed resume
            self._trigger_checkpoint_alerts("resume_from_checkpoint", False, None, checkpoint_id, None, str(e))
            self._track_checkpoint_performance("resume_from_checkpoint", False)

            return None

    async def list_checkpoints(self, problem_id: str = None) -> List[CheckpointMetadata]:
        """List available checkpoints"""
        return await self.repository.list_checkpoints(problem_id)

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint"""
        return await self.repository.delete(checkpoint_id)

    async def cleanup_checkpoints(self, problem_id: str, keep_last_n: int = 5) -> int:
        """Clean up old checkpoints"""
        return await self.repository.cleanup_old_checkpoints(problem_id, keep_last_n)

    def generate_checkpoint_id(
        self,
        problem_id: str,
        level: int,
        stage: str
    ) -> str:
        """Generate a unique checkpoint ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        return f"{problem_id}_{level}_{stage}_{timestamp}"

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Checkpoint Manager
    # =========================================================================

    def _trigger_checkpoint_alerts(
        self,
        operation: str,
        success: bool,
        problem_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        stage: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for checkpoint failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            # Alert on checkpoint failures
            if not success:
                alert_manager.create_alert(
                    title=f"Checkpoint Manager Alert: {operation}",
                    description=f"Checkpoint operation '{operation}' failed" +
                                 (f" for problem '{problem_id}'" if problem_id else "") +
                                 (f" at stage '{stage}'" if stage else "") +
                                 (f" (checkpoint: {checkpoint_id})" if checkpoint_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.HIGH.value,
                    source="checkpoint_manager",
                    component="checkpoint_resumption",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Checkpoint alert: {e}")

    def _extract_checkpoint_knowledge(
        self,
        operation: str,
        checkpoint_id: str,
        metadata: CheckpointMetadata,
        state: PipelineState
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract checkpoint knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"checkpoint_{operation}_{checkpoint_id}",
                artifact_type="checkpoint",
                source_component="checkpoint_manager",
                title=f"Checkpoint: {checkpoint_id} ({operation})",
                content={
                    "operation": operation,
                    "checkpoint_id": checkpoint_id,
                    "problem_id": metadata.problem_id,
                    "stage": metadata.stage,
                    "level": metadata.level,
                    "state_size": metadata.state_size,
                    "compressed": metadata.compressed,
                    "num_solutions": len(state.solutions) if state.solutions else 0,
                    "timestamp": datetime.utcnow().isoformat()
                },
                metadata={
                    "parent_checkpoint_id": metadata.parent_checkpoint_id,
                    "execution_status": state.execution_status,
                    "metrics": state.metrics
                },
                tags=["checkpoint", "resumption", operation, metadata.stage]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Checkpoint knowledge for {operation}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Checkpoint knowledge: {e}")
            return False

    def _track_checkpoint_performance(
        self,
        operation: str,
        success: bool,
        state_size: int = 0,
        load_time: float = 0.0
    ):
        """**ACTUAL INTEGRATION**: Track checkpoint performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            # Quality based on success and size efficiency
            quality = 1.0 if success else 0.0
            if success:
                # Penalize very large checkpoints (>10MB)
                if state_size > 10_000_000:
                    quality *= 0.8
                # Penalize very slow loads (>5 seconds)
                if load_time > 5.0:
                    quality *= 0.8
            quality = max(quality, 0.0)

            performance_data = StrategyPerformanceData(
                strategy_name=f"checkpoint_manager_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.utcnow(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "state_size": state_size,
                    "load_time": load_time
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Checkpoint performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track Checkpoint performance: {e}")


def create_checkpoint_manager(
    storage_type: str = 'file',
    storage_path: str = './checkpoints',
    compression: bool = False
) -> CheckpointManager:
    """
    Factory function to create a checkpoint manager.

    Args:
        storage_type: Type of storage ('file' or 'memory')
        storage_path: Path for file-based storage
        compression: Enable compression for checkpoint data

    Returns:
        CheckpointManager instance
    """
    repository = CheckpointRepository(storage_type=storage_type, storage_path=storage_path)
    serializer = StateSerializer(compression_enabled=compression)

    return CheckpointManager(repository=repository, serializer=serializer)
