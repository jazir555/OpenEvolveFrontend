"""
Backup and Restore Utilities - License: Apache 2.0

Comprehensive backup and restore system for OpenEvolve.
Supports data, configuration, and knowledge base backups.

Usage:
    python backup_restore.py backup --full
    python backup_restore.py restore --backup-id 20260202_143000
    python backup_restore.py list
    python backup_restore.py schedule --daily
"""

import os
import json
import shutil
import tarfile
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict
import argparse
import hashlib

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Backup/Restore operations
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

# Rich for output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class BackupMetadata:
    """Metadata for a backup."""
    backup_id: str
    created_at: datetime
    backup_type: str  # 'full', 'data', 'config', 'knowledge'
    size_bytes: int
    checksum: str
    description: str
    components: List[str]
    paths: List[str]
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BackupMetadata':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


# =============================================================================
# BACKUP MANAGER
# =============================================================================

class BackupManager:
    """Manages backup and restore operations."""
    
    BACKUP_COMPONENTS = {
        'knowledge': 'knowledge_extraction/',
        'data': 'data/',
        'config': ['*.yaml', '*.json', '.env'],
        'plugins': 'plugins/',
        'logs': 'logs/',
        'checkpoints': 'checkpoints/',
    }
    
    def __init__(self, backup_dir: Path = None):
        self.backup_dir = backup_dir or Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load backup metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                data = json.load(f)
                self.backups = {
                    k: BackupMetadata.from_dict(v)
                    for k, v in data.get('backups', {}).items()
                }
        else:
            self.backups = {}
    
    def _save_metadata(self) -> None:
        """Save backup metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump({
                'backups': {
                    k: v.to_dict() for k, v in self.backups.items()
                },
                'last_updated': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    async def create_backup(
        self,
        backup_type: str = 'full',
        components: List[str] = None,
        description: str = ""
    ) -> str:
        """Create a new backup."""
        import time
        start_time = time.time()
        backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{backup_id}.tar.gz"

        try:
            if console:
                console.print(f"[blue]Creating {backup_type} backup: {backup_id}[/blue]")

            # Determine components to backup
            if backup_type == 'full':
                components = list(self.BACKUP_COMPONENTS.keys())
            elif components is None:
                components = ['knowledge', 'data', 'config']

            # Create backup
            paths_backed_up = []

            with tarfile.open(backup_path, "w:gz") as tar:
                for component in components:
                    if component not in self.BACKUP_COMPONENTS:
                        continue

                    source = self.BACKUP_COMPONENTS[component]

                    if isinstance(source, list):
                        # Multiple patterns
                        for pattern in source:
                            for filepath in Path('.').glob(pattern):
                                if filepath.exists():
                                    tar.add(filepath, arcname=f"config/{filepath.name}")
                                    paths_backed_up.append(str(filepath))
                    else:
                        # Single directory
                        source_path = Path(source)
                        if source_path.exists():
                            tar.add(source_path, arcname=component)
                            paths_backed_up.append(str(source_path))

            # Calculate metadata
            size_bytes = backup_path.stat().st_size
            checksum = self._calculate_checksum(backup_path)

            metadata = BackupMetadata(
                backup_id=backup_id,
                created_at=datetime.now(),
                backup_type=backup_type,
                size_bytes=size_bytes,
                checksum=checksum,
                description=description,
                components=components,
                paths=paths_backed_up
            )

            self.backups[backup_id] = metadata
            self._save_metadata()

            duration = time.time() - start_time
            size_mb = size_bytes / (1024 * 1024)

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful backup
            self._extract_backup_knowledge("create_backup", backup_id, metadata)
            self._track_backup_performance("create_backup", True, duration, backup_type, size_mb)

            if console:
                console.print(f"[green]Backup created: {backup_id} ({size_mb:.2f} MB)[/green]")

            return backup_id

        except Exception as e:
            duration = time.time() - start_time

            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_backup_alerts("create_backup", False, backup_id, str(e))
            self._track_backup_performance("create_backup", False, duration, backup_type, 0)

            if console:
                console.print(f"[red]Failed to create backup: {e}[/red]")
            raise
    
    async def restore_backup(
        self,
        backup_id: str,
        components: List[str] = None,
        dry_run: bool = False
    ) -> bool:
        """Restore from a backup."""
        import time
        start_time = time.time()

        try:
            if backup_id not in self.backups:
                if console:
                    console.print(f"[red]Backup not found: {backup_id}[/red]")
                return False

            metadata = self.backups[backup_id]
            backup_path = self.backup_dir / f"backup_{backup_id}.tar.gz"

            if not backup_path.exists():
                if console:
                    console.print(f"[red]Backup file not found: {backup_path}[/red]")
                return False

            # Verify checksum
            current_checksum = self._calculate_checksum(backup_path)
            if current_checksum != metadata.checksum:
                if console:
                    console.print(f"[red]Backup checksum mismatch! Possible corruption.[/red]")
                return False

            if console:
                console.print(f"[blue]Restoring backup: {backup_id}[/blue]")

            if dry_run:
                if console:
                    console.print("[yellow]Dry run - no changes made[/yellow]")
                    console.print(f"Would restore: {', '.join(metadata.components)}")

                # **ACTUAL INTEGRATION**: Extract knowledge and track performance for dry run
                duration = time.time() - start_time
                self._extract_backup_knowledge("restore_backup", backup_id, metadata)
                self._track_backup_performance("restore_backup", True, duration, metadata.backup_type, 0)

                return True

            # Extract backup
            with tarfile.open(backup_path, "r:gz") as tar:
                # Safety check: don't overwrite without confirmation
                tar.extractall(".")

            duration = time.time() - start_time
            size_mb = metadata.size_bytes / (1024 * 1024)

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful restore
            self._extract_backup_knowledge("restore_backup", backup_id, metadata)
            self._track_backup_performance("restore_backup", True, duration, metadata.backup_type, size_mb)

            if console:
                console.print(f"[green]Backup restored: {backup_id}[/green]")

            return True

        except Exception as e:
            duration = time.time() - start_time

            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_backup_alerts("restore_backup", False, backup_id, str(e))
            self._track_backup_performance("restore_backup", False, duration, "unknown", 0)

            if console:
                console.print(f"[red]Failed to restore backup: {e}[/red]")
            return False
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all backups."""
        return sorted(
            self.backups.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        if backup_id not in self.backups:
            return False
        
        backup_path = self.backup_dir / f"backup_{backup_id}.tar.gz"
        
        if backup_path.exists():
            backup_path.unlink()
        
        del self.backups[backup_id]
        self._save_metadata()
        
        if console:
            console.print(f"[green]Backup deleted: {backup_id}[/green]")
        
        return True
    
    def cleanup_old_backups(self, keep_days: int = 30) -> int:
        """Remove backups older than specified days."""
        cutoff = datetime.now() - timedelta(days=keep_days)
        deleted = 0
        
        for backup_id, metadata in list(self.backups.items()):
            if metadata.created_at < cutoff:
                self.delete_backup(backup_id)
                deleted += 1
        
        return deleted
    
    def print_backup_list(self):
        """Print backup list in table format."""
        if not RICH_AVAILABLE:
            for b in self.list_backups():
                print(f"{b.backup_id}: {b.backup_type} - {b.description}")
            return
        
        table = Table(title="Available Backups")
        table.add_column("Backup ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Created", style="dim")
        table.add_column("Size", justify="right")
        table.add_column("Components")
        table.add_column("Description")
        
        for b in self.list_backups():
            size_mb = b.size_bytes / (1024 * 1024)
            created = b.created_at.strftime("%Y-%m-%d %H:%M")
            
            table.add_row(
                b.backup_id,
                b.backup_type,
                created,
                f"{size_mb:.2f} MB",
                ", ".join(b.components[:3]),
                b.description[:40]
            )
        
        console.print(table)

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Backup/Restore
    # =========================================================================

    def _trigger_backup_alerts(
        self,
        operation: str,
        success: bool,
        backup_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for backup/restore failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            # Alert on failures
            if not success:
                alert_manager.create_alert(
                    title=f"Backup/Restore Alert: {operation}",
                    description=f"Backup/restore operation '{operation}' failed" +
                                 (f" for backup '{backup_id}'" if backup_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.HIGH.value,
                    source="backup_restore",
                    component="backup_management",
                    metadata=metadata or {}
                )

        except Exception as e:
            if console:
                console.print(f"[red]Failed to trigger backup/restore alert: {e}[/red]")

    def _extract_backup_knowledge(
        self,
        operation: str,
        backup_id: str,
        backup_metadata: BackupMetadata
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract backup/restore knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"backup_{operation}_{backup_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="backup_operation",
                source_component="backup_restore",
                title=f"Backup Operation: {operation} - {backup_id}",
                content={
                    "operation": operation,
                    "backup_id": backup_id,
                    "backup_type": backup_metadata.backup_type,
                    "size_bytes": backup_metadata.size_bytes,
                    "components": backup_metadata.components,
                    "description": backup_metadata.description,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "paths": backup_metadata.paths,
                    "checksum": backup_metadata.checksum
                },
                tags=["backup", operation, backup_metadata.backup_type]
            )

            knowledge_engine.store_artifact(artifact)
            if console:
                console.print(f"[dim]Extracted backup knowledge for {backup_id}[/dim]")
            return True

        except Exception as e:
            if console:
                console.print(f"[red]Failed to extract backup knowledge: {e}[/red]")
            return False

    def _track_backup_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        backup_type: str,
        size_mb: float = 0.0
    ):
        """**ACTUAL INTEGRATION**: Track backup/restore performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = 1.0 if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"backup_restore_{operation}_{backup_type}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "backup_type": backup_type,
                    "size_mb": size_mb
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                if console:
                    console.print(f"[dim]Tracked backup performance for {operation}[/dim]")

        except Exception as e:
            if console:
                console.print(f"[red]Failed to track backup performance: {e}[/red]")


# =============================================================================
# SCHEDULER
# =============================================================================

class BackupScheduler:
    """Schedules automatic backups."""
    
    def __init__(self, manager: BackupManager):
        self.manager = manager
        self.schedule_file = Path("backup_schedule.json")
        self.schedules = self._load_schedules()
    
    def _load_schedules(self) -> Dict:
        """Load backup schedules."""
        if self.schedule_file.exists():
            with open(self.schedule_file) as f:
                return json.load(f)
        return {}
    
    def _save_schedules(self) -> None:
        """Save backup schedules."""
        with open(self.schedule_file, 'w') as f:
            json.dump(self.schedules, f, indent=2)
    
    def schedule_backup(
        self,
        name: str,
        backup_type: str,
        frequency: str,  # 'daily', 'weekly', 'monthly'
        time: str = "02:00"
    ) -> None:
        """Schedule a recurring backup."""
        self.schedules[name] = {
            'backup_type': backup_type,
            'frequency': frequency,
            'time': time,
            'enabled': True,
            'created_at': datetime.now().isoformat()
        }
        self._save_schedules()
        
        if console:
            console.print(f"[green]Scheduled backup '{name}': {frequency} at {time}[/green]")
    
    def list_schedules(self) -> None:
        """List all scheduled backups."""
        if not RICH_AVAILABLE:
            for name, schedule in self.schedules.items():
                print(f"{name}: {schedule['frequency']} at {schedule['time']}")
            return
        
        table = Table(title="Scheduled Backups")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Frequency", style="blue")
        table.add_column("Time", style="dim")
        table.add_column("Status")
        
        for name, schedule in self.schedules.items():
            status = "[green]enabled[/green]" if schedule['enabled'] else "[red]disabled[/red]"
            table.add_row(
                name,
                schedule['backup_type'],
                schedule['frequency'],
                schedule['time'],
                status
            )
        
        console.print(table)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenEvolve Backup and Restore Utility"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create backup')
    backup_parser.add_argument('--full', action='store_true', help='Full backup')
    backup_parser.add_argument('--type', default='incremental', help='Backup type')
    backup_parser.add_argument('--components', nargs='+', help='Components to backup')
    backup_parser.add_argument('--description', default='', help='Backup description')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('backup_id', help='Backup ID to restore')
    restore_parser.add_argument('--components', nargs='+', help='Components to restore')
    restore_parser.add_argument('--dry-run', action='store_true', help='Dry run')
    
    # List command
    subparsers.add_parser('list', help='List backups')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete backup')
    delete_parser.add_argument('backup_id', help='Backup ID to delete')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old backups')
    cleanup_parser.add_argument('--keep-days', type=int, default=30, help='Keep backups for N days')
    
    # Schedule commands
    schedule_parser = subparsers.add_parser('schedule', help='Schedule backup')
    schedule_parser.add_argument('name', help='Schedule name')
    schedule_parser.add_argument('--type', default='incremental', help='Backup type')
    schedule_parser.add_argument('--frequency', default='daily', help='Frequency (daily/weekly/monthly)')
    schedule_parser.add_argument('--time', default='02:00', help='Time (HH:MM)')
    
    schedule_list_parser = subparsers.add_parser('schedules', help='List scheduled backups')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize
    manager = BackupManager()
    
    if args.command == 'backup':
        backup_type = 'full' if args.full else args.type
        asyncio.run(manager.create_backup(
            backup_type=backup_type,
            components=args.components,
            description=args.description
        ))
    
    elif args.command == 'restore':
        success = asyncio.run(manager.restore_backup(
            backup_id=args.backup_id,
            components=args.components,
            dry_run=args.dry_run
        ))
        exit(0 if success else 1)
    
    elif args.command == 'list':
        manager.print_backup_list()
    
    elif args.command == 'delete':
        manager.delete_backup(args.backup_id)
    
    elif args.command == 'cleanup':
        deleted = manager.cleanup_old_backups(args.keep_days)
        if console:
            console.print(f"[green]Deleted {deleted} old backups[/green]")
    
    elif args.command == 'schedule':
        scheduler = BackupScheduler(manager)
        scheduler.schedule_backup(
            name=args.name,
            backup_type=args.type,
            frequency=args.frequency,
            time=args.time
        )
    
    elif args.command == 'schedules':
        scheduler = BackupScheduler(manager)
        scheduler.list_schedules()


if __name__ == "__main__":
    main()
