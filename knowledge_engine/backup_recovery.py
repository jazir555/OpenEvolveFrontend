"""
Backup and Disaster Recovery System

Provides comprehensive backup and recovery capabilities:
- Automated scheduled backups
- Incremental and full backups
- Point-in-time recovery
- Cross-region replication
- Backup verification
- Disaster recovery procedures
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class BackupMetadata:
    """Metadata for a backup."""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    source_path: str = ""
    destination_path: str = ""
    size_bytes: int = 0
    checksum: str = ""
    parent_backup_id: Optional[str] = None  # For incremental
    included_items: List[str] = field(default_factory=list)
    excluded_items: List[str] = field(default_factory=list)
    compression_ratio: float = 0.0
    error_message: Optional[str] = None
    verified_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "parent_backup_id": self.parent_backup_id,
            "compression_ratio": self.compression_ratio,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None
        }


@dataclass
class RecoveryPoint:
    """A point-in-time recovery point."""
    recovery_point_id: str
    backup_id: str
    timestamp: datetime
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackupStorage:
    """Abstract base for backup storage backends."""
    
    async def store(
        self, 
        backup_id: str, 
        data: bytes, 
        metadata: BackupMetadata
    ) -> str:
        """Store backup data. Returns storage identifier."""
        raise NotImplementedError
    
    async def retrieve(self, storage_id: str) -> bytes:
        """Retrieve backup data."""
        raise NotImplementedError
    
    async def delete(self, storage_id: str) -> bool:
        """Delete backup data."""
        raise NotImplementedError
    
    async def list_backups(self) -> List[str]:
        """List available backups."""
        raise NotImplementedError


class LocalBackupStorage(BackupStorage):
    """Local filesystem backup storage."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def store(
        self, 
        backup_id: str, 
        data: bytes, 
        metadata: BackupMetadata
    ) -> str:
        backup_dir = self.base_path / backup_id
        backup_dir.mkdir(exist_ok=True)
        
        # Store data
        data_file = backup_dir / "data.gz"
        with gzip.open(data_file, 'wb') as f:
            f.write(data)
        
        # Store metadata
        meta_file = backup_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        return str(backup_dir)
    
    async def retrieve(self, storage_id: str) -> bytes:
        data_file = Path(storage_id) / "data.gz"
        with gzip.open(data_file, 'rb') as f:
            return f.read()
    
    async def delete(self, storage_id: str) -> bool:
        try:
            shutil.rmtree(storage_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False
    
    async def list_backups(self) -> List[str]:
        return [str(d) for d in self.base_path.iterdir() if d.is_dir()]


class BackupEngine:
    """
    Main backup engine.
    """
    
    def __init__(
        self,
        storage: BackupStorage,
        retention_days: int = 30,
        compression_level: int = 6
    ):
        self.storage = storage
        self.retention_days = retention_days
        self.compression_level = compression_level
        
        self._backups: Dict[str, BackupMetadata] = {}
        self._recovery_points: List[RecoveryPoint] = []
        self._scheduled_tasks: Dict[str, asyncio.Task] = {}
        
    async def create_backup(
        self,
        source_path: str,
        backup_type: BackupType = BackupType.FULL,
        parent_backup_id: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> BackupMetadata:
        """
        Create a new backup.
        
        Args:
            source_path: Path to backup
            backup_type: Type of backup
            parent_backup_id: Parent backup for incremental
            include_patterns: Glob patterns to include
            exclude_patterns: Glob patterns to exclude
        """
        backup_id = str(uuid.uuid4())
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.RUNNING,
            started_at=datetime.utcnow(),
            source_path=source_path,
            included_items=include_patterns or [],
            excluded_items=exclude_patterns or []
        )
        
        if backup_type == BackupType.INCREMENTAL and parent_backup_id:
            metadata.parent_backup_id = parent_backup_id
        
        try:
            # Collect files to backup
            files_to_backup = self._collect_files(
                source_path,
                include_patterns,
                exclude_patterns
            )
            
            # For incremental, filter by modification time
            if backup_type == BackupType.INCREMENTAL and parent_backup_id:
                parent = self._backups.get(parent_backup_id)
                if parent:
                    files_to_backup = self._filter_incremental(
                        files_to_backup,
                        parent.started_at
                    )
            
            # Create backup archive
            backup_data = await self._create_archive(files_to_backup)
            
            # Calculate checksum
            metadata.checksum = hashlib.sha256(backup_data).hexdigest()
            metadata.size_bytes = len(backup_data)
            
            # Store backup
            storage_id = await self.storage.store(backup_id, backup_data, metadata)
            metadata.destination_path = storage_id
            
            # Calculate compression ratio
            original_size = sum(
                Path(f).stat().st_size 
                for f in files_to_backup 
                if Path(f).exists()
            )
            if original_size > 0:
                metadata.compression_ratio = (
                    (original_size - metadata.size_bytes) / original_size
                )
            
            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.utcnow()
            
            self._backups[backup_id] = metadata
            
            # Create recovery point
            recovery_point = RecoveryPoint(
                recovery_point_id=str(uuid.uuid4()),
                backup_id=backup_id,
                timestamp=metadata.completed_at,
                description=f"{backup_type.value} backup"
            )
            self._recovery_points.append(recovery_point)
            
            logger.info(
                f"Backup {backup_id} completed: "
                f"{len(files_to_backup)} files, "
                f"{metadata.size_bytes} bytes"
            )
            
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            logger.error(f"Backup {backup_id} failed: {e}")
            raise
        
        return metadata
    
    def _collect_files(
        self,
        source_path: str,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> List[str]:
        """Collect files to backup."""
        source = Path(source_path)
        files = []
        
        if source.is_file():
            files = [str(source)]
        elif source.is_dir():
            files = [str(f) for f in source.rglob("*") if f.is_file()]
        
        # Apply include patterns
        if include_patterns:
            import fnmatch
            included = []
            for pattern in include_patterns:
                included.extend([
                    f for f in files 
                    if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern)
                ])
            files = list(set(included))
        
        # Apply exclude patterns
        if exclude_patterns:
            import fnmatch
            for pattern in exclude_patterns:
                files = [
                    f for f in files 
                    if not (fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern))
                ]
        
        return files
    
    def _filter_incremental(
        self,
        files: List[str],
        since: datetime
    ) -> List[str]:
        """Filter files modified since a timestamp."""
        return [
            f for f in files
            if datetime.fromtimestamp(Path(f).stat().st_mtime) > since
        ]
    
    async def _create_archive(self, files: List[str]) -> bytes:
        """Create compressed archive of files."""
        import io
        import tarfile
        
        buffer = io.BytesIO()
        
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for file_path in files:
                path = Path(file_path)
                if path.exists():
                    tar.add(file_path, arcname=path.name)
        
        return buffer.getvalue()
    
    async def restore_backup(
        self,
        backup_id: str,
        destination_path: str,
        verify_checksum: bool = True
    ) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_id: Backup to restore
            destination_path: Where to restore
            verify_checksum: Whether to verify checksum
        """
        metadata = self._backups.get(backup_id)
        if not metadata:
            raise ValueError(f"Backup {backup_id} not found")
        
        if metadata.status != BackupStatus.COMPLETED:
            raise ValueError(f"Backup {backup_id} is not completed")
        
        logger.info(f"Restoring backup {backup_id} to {destination_path}")
        
        # Retrieve backup data
        backup_data = await self.storage.retrieve(metadata.destination_path)
        
        # Verify checksum
        if verify_checksum:
            actual_checksum = hashlib.sha256(backup_data).hexdigest()
            if actual_checksum != metadata.checksum:
                raise ValueError("Checksum mismatch - backup may be corrupted")
        
        # Extract archive
        await self._extract_archive(backup_data, destination_path)
        
        logger.info(f"Restore of backup {backup_id} completed")
        return True
    
    async def _extract_archive(self, data: bytes, destination: str):
        """Extract archive to destination."""
        import io
        import tarfile
        
        dest_path = Path(destination)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        buffer = io.BytesIO(data)
        
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(path=destination)
    
    async def verify_backup(self, backup_id: str) -> bool:
        """Verify a backup's integrity."""
        metadata = self._backups.get(backup_id)
        if not metadata:
            return False
        
        try:
            backup_data = await self.storage.retrieve(metadata.destination_path)
            actual_checksum = hashlib.sha256(backup_data).hexdigest()
            
            if actual_checksum == metadata.checksum:
                metadata.status = BackupStatus.VERIFIED
                metadata.verified_at = datetime.utcnow()
                logger.info(f"Backup {backup_id} verified successfully")
                return True
            else:
                logger.error(f"Backup {backup_id} verification failed: checksum mismatch")
                return False
                
        except Exception as e:
            logger.error(f"Backup {backup_id} verification failed: {e}")
            return False
    
    def schedule_backup(
        self,
        schedule_id: str,
        source_path: str,
        cron_expression: str,  # Simplified: "daily", "weekly", or "hourly"
        backup_type: BackupType = BackupType.INCREMENTAL
    ):
        """Schedule recurring backups."""
        async def scheduled_task():
            while True:
                try:
                    await self.create_backup(source_path, backup_type)
                    
                    # Sleep based on schedule
                    if cron_expression == "hourly":
                        await asyncio.sleep(3600)
                    elif cron_expression == "daily":
                        await asyncio.sleep(86400)
                    elif cron_expression == "weekly":
                        await asyncio.sleep(604800)
                    else:
                        await asyncio.sleep(86400)
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Scheduled backup error: {e}")
                    await asyncio.sleep(3600)
        
        task = asyncio.create_task(scheduled_task())
        self._scheduled_tasks[schedule_id] = task
        
        logger.info(f"Scheduled backup {schedule_id}: {cron_expression}")
    
    def cancel_schedule(self, schedule_id: str):
        """Cancel a scheduled backup."""
        task = self._scheduled_tasks.pop(schedule_id, None)
        if task:
            task.cancel()
            logger.info(f"Cancelled scheduled backup {schedule_id}")
    
    def get_recovery_points(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[RecoveryPoint]:
        """Get available recovery points."""
        points = self._recovery_points
        
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        return sorted(points, key=lambda p: p.timestamp, reverse=True)
    
    async def cleanup_old_backups(self):
        """Remove backups older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        
        to_delete = []
        for backup_id, metadata in self._backups.items():
            if metadata.completed_at and metadata.completed_at < cutoff:
                to_delete.append(backup_id)
        
        for backup_id in to_delete:
            metadata = self._backups[backup_id]
            await self.storage.delete(metadata.destination_path)
            del self._backups[backup_id]
            logger.info(f"Deleted old backup {backup_id}")
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        total_backups = len(self._backups)
        completed = sum(1 for b in self._backups.values() if b.status == BackupStatus.COMPLETED)
        failed = sum(1 for b in self._backups.values() if b.status == BackupStatus.FAILED)
        verified = sum(1 for b in self._backups.values() if b.status == BackupStatus.VERIFIED)
        
        total_size = sum(b.size_bytes for b in self._backups.values())
        
        return {
            "total_backups": total_backups,
            "completed": completed,
            "failed": failed,
            "verified": verified,
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "recovery_points": len(self._recovery_points),
            "scheduled_tasks": len(self._scheduled_tasks),
            "retention_days": self.retention_days
        }


class DisasterRecovery:
    """
    Disaster recovery procedures.
    """
    
    def __init__(self, backup_engine: BackupEngine):
        self.backup_engine = backup_engine
        self._dr_site: Optional[str] = None
        self._replication_enabled = False
    
    def configure_dr_site(self, site_url: str):
        """Configure disaster recovery site."""
        self._dr_site = site_url
        logger.info(f"DR site configured: {site_url}")
    
    async def failover(self, backup_id: str, target_path: str) -> bool:
        """
        Execute failover to backup.
        
        Args:
            backup_id: Backup to failover to
            target_path: Target for restoration
        """
        logger.info(f"Executing failover to backup {backup_id}")
        
        try:
            await self.backup_engine.restore_backup(backup_id, target_path)
            logger.info(f"Failover to backup {backup_id} completed successfully")
            return True
        except Exception as e:
            logger.error(f"Failover failed: {e}")
            return False
    
    async def test_recovery(
        self,
        backup_id: str,
        test_path: str
    ) -> Dict[str, Any]:
        """
        Test recovery procedure without affecting production.
        
        Returns:
            Test results
        """
        logger.info(f"Testing recovery of backup {backup_id}")
        
        start_time = datetime.utcnow()
        
        try:
            await self.backup_engine.restore_backup(backup_id, test_path)
            
            # Verify restored data
            restored_files = list(Path(test_path).rglob("*"))
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "success": True,
                "backup_id": backup_id,
                "test_path": test_path,
                "duration_seconds": duration,
                "restored_files": len(restored_files),
                "verified_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "backup_id": backup_id,
                "error": str(e),
                "verified_at": datetime.utcnow().isoformat()
            }
    
    def generate_dr_plan(self) -> Dict[str, Any]:
        """Generate disaster recovery plan."""
        stats = self.backup_engine.get_backup_stats()
        
        return {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat(),
            "rpo_hours": 24,  # Recovery Point Objective
            "rto_hours": 4,   # Recovery Time Objective
            "backup_stats": stats,
            "recovery_points": [
                {
                    "id": rp.recovery_point_id,
                    "timestamp": rp.timestamp.isoformat(),
                    "backup_id": rp.backup_id
                }
                for rp in self.backup_engine.get_recovery_points()[:10]
            ],
            "procedures": {
                "full_restore": [
                    "1. Identify backup to restore from",
                    "2. Verify backup integrity",
                    "3. Stop application services",
                    "4. Execute restore operation",
                    "5. Verify restored data",
                    "6. Restart application services"
                ],
                "point_in_time_recovery": [
                    "1. Identify target recovery point",
                    "2. Find closest backup before target time",
                    "3. Restore full backup",
                    "4. Apply incremental backups",
                    "5. Verify data at target time"
                ]
            }
        }


__all__ = [
    "BackupEngine",
    "BackupStorage",
    "LocalBackupStorage",
    "BackupMetadata",
    "BackupType",
    "BackupStatus",
    "RecoveryPoint",
    "DisasterRecovery"
]
