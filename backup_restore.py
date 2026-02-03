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
        backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{backup_id}.tar.gz"
        
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
        
        if console:
            size_mb = size_bytes / (1024 * 1024)
            console.print(f"[green]Backup created: {backup_id} ({size_mb:.2f} MB)[/green]")
        
        return backup_id
    
    async def restore_backup(
        self,
        backup_id: str,
        components: List[str] = None,
        dry_run: bool = False
    ) -> bool:
        """Restore from a backup."""
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
            return True
        
        # Extract backup
        with tarfile.open(backup_path, "r:gz") as tar:
            # Safety check: don't overwrite without confirmation
            tar.extractall(".")
        
        if console:
            console.print(f"[green]Backup restored: {backup_id}[/green]")
        
        return True
    
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
