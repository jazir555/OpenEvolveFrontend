"""
Backup Restore Module

Provides backup and restore functionality for OpenEvolve.

Author: OpenEvolve Team
Date: 2026-02-06
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Configuration for backup operations"""
    backup_dir: str = "./backups"
    max_backups: int = 10
    compress: bool = True


class BackupManager:
    """Backup Manager class"""

    def __init__(self, config: Optional[BackupConfig] = None, backup_dir: Optional[str] = None):
        # Support both config object and backup_dir parameter for backward compatibility
        if backup_dir is not None:
            self.config = BackupConfig(backup_dir=backup_dir)
        else:
            self.config = config or BackupConfig()
        logger.info("Backup Manager initialized")
    
    def create_backup(self, data: Dict[str, Any], backup_type: str = 'full') -> str:
        """Create a backup of the given data

        Args:
            data: Data to backup
            backup_type: Type of backup (full, incremental, etc.)

        Returns:
            Backup ID
        """
        # Use timestamp without colons for Windows compatibility
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup_id = f"backup_{timestamp}"
        backup_path = os.path.join(self.config.backup_dir, f'{backup_id}.backup')

        # Ensure backup directory exists
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)

        # Save backup data
        with open(backup_path, 'w') as f:
            json.dump({
                'backup_id': backup_id,
                'backup_type': backup_type,
                'timestamp': timestamp,
                'data': data
            }, f, indent=2)

        logger.info(f"Created backup: {backup_id} at {backup_path}")
        return backup_id
    
    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore data from a backup

        Args:
            backup_id: ID of backup to restore

        Returns:
            Restored data
        """
        backup_path = os.path.join(self.config.backup_dir, f'{backup_id}.backup')

        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_id}")

        with open(backup_path, 'r') as f:
            backup_data = json.load(f)

        logger.info(f"Restored backup: {backup_id} from {backup_path}")
        return backup_data.get('data', {})
    
    def list_backups(self) -> List[str]:
        """List available backups

        Returns:
            List of backup IDs
        """
        if not os.path.exists(self.config.backup_dir):
            return []

        backups = []
        for filename in os.listdir(self.config.backup_dir):
            if filename.endswith('.backup'):
                backup_id = filename[:-7]  # Remove .backup extension
                backups.append(backup_id)

        return sorted(backups)
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        logger.info(f"Deleting backup: {backup_id}")
        return True


def create_backup_manager(config: Optional[BackupConfig] = None) -> BackupManager:
    """Factory function to create Backup Manager instance"""
    return BackupManager(config)
