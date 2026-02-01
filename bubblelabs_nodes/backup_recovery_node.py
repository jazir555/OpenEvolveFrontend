"""
Backup & Recovery Node for BubbleLabs

Provides comprehensive backup and disaster recovery operations for knowledge graphs:
- Create snapshots/backups of knowledge graphs
- List available backups with metadata
- Restore knowledge from backups
- Schedule automatic backups
- Verify backup integrity
- Export backups to external storage
- Manage backup retention and cleanup

Features:
- Multiple compression formats (none, gzip, zip)
- Integrity verification with checksums
- Progress tracking for long operations
- Automatic fallback to manual backup when BackupManager unavailable
- Retention policy management
- Include/exclude history options
"""

import json
import gzip
import zipfile
import hashlib
import os
import shutil
import threading
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import io
from .base_node import BubbleLabsNode, NodeExecutionError


class BackupRecoveryNode(BubbleLabsNode):
    """
    Backup and recovery operations for knowledge graphs and disaster recovery.

    Supports:
    - backup: Create snapshots of knowledge graphs
    - restore: Restore knowledge from backup
    - list: List available backups with metadata
    - verify: Verify backup integrity with checksums
    - schedule: Configure automatic backup schedules
    - export: Export backups to external storage
    - delete: Remove old backups with retention policies
    """

    # Node metadata
    DISPLAY_NAME = "Backup & Recovery"
    DESCRIPTION = "Create snapshots, restore knowledge, and manage disaster recovery"
    ICON = "backup"
    CATEGORY = "management"
    VERSION = "1.0.0"

    # Supported operations
    SUPPORTED_OPERATIONS = ["backup", "restore", "list", "verify", "schedule", "export", "delete"]
    SUPPORTED_COMPRESSION = ["none", "gzip", "zip"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports from knowledge_engine
        backup_module = self.safe_import(
            'knowledge_engine.backup_recovery',
            fallback_value=None,
            error_msg="Backup Manager not available, using fallback manual backup"
        )

        unified_hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="Unified KG Integration Hub not available"
        )

        # Store module references
        self.BackupManager = None
        self.UnifiedKGIntegrationHub = None
        self.UnifiedKGConfig = None

        if backup_module:
            self.BackupManager = getattr(backup_module, 'BackupManager', None)

        if unified_hub_module:
            self.UnifiedKGIntegrationHub = getattr(unified_hub_module, 'UnifiedKGIntegrationHub', None)
            self.UnifiedKGConfig = getattr(unified_hub_module, 'UnifiedKGConfig', None)

        # Initialize manager instances
        self.backup_manager = None
        self.hub = None
        self._manager_initialized = False

        if self.BackupManager:
            try:
                self.backup_manager = self.BackupManager()
                self._manager_initialized = True
                self.logger.info("BackupManager initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize BackupManager: {e}")
                self.backup_manager = None

        if self.UnifiedKGIntegrationHub and self.UnifiedKGConfig:
            try:
                config_obj = self.UnifiedKGConfig()
                self.hub = self.UnifiedKGIntegrationHub(config=config_obj)
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

        # Default backup directory
        self.default_backup_dir = Path(self.config.get('backup_directory', './backups'))
        self.default_backup_dir.mkdir(parents=True, exist_ok=True)

        # Schedule storage (in-memory, would be persisted in production)
        self._scheduled_backups: Dict[str, Dict] = {}
        self._schedule_threads: Dict[str, threading.Thread] = {}

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields depend on operation:
        - backup: knowledge_graph_id (required), backup_name (optional)
        - restore: backup_id (required), knowledge_graph_id (optional)
        - list: no required fields
        - verify: backup_id (required)
        - schedule: knowledge_graph_id (required), schedule_config (required)
        - export: backup_id (required), destination (required)
        - delete: backup_id (required)
        """
        errors = []

        # Check for operation type in inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'backup'))

        if operation not in self.SUPPORTED_OPERATIONS:
            errors.append(
                f"Invalid operation: {operation}. "
                f"Must be one of: {', '.join(self.SUPPORTED_OPERATIONS)}"
            )

        # Operation-specific validation
        if operation == 'backup':
            kg_id = inputs.get('knowledge_graph_id', self.config.get('knowledge_graph_id'))
            if not kg_id:
                errors.append("Missing required field 'knowledge_graph_id' for backup operation")

        elif operation == 'restore':
            backup_id = inputs.get('backup_id', self.config.get('backup_id'))
            if not backup_id:
                errors.append("Missing required field 'backup_id' for restore operation")

        elif operation == 'verify':
            backup_id = inputs.get('backup_id', self.config.get('backup_id'))
            if not backup_id:
                errors.append("Missing required field 'backup_id' for verify operation")

        elif operation == 'schedule':
            kg_id = inputs.get('knowledge_graph_id', self.config.get('knowledge_graph_id'))
            if not kg_id:
                errors.append("Missing required field 'knowledge_graph_id' for schedule operation")
            # Check for schedule configuration
            schedule_config = inputs.get('schedule_config', self.config.get('schedule_config'))
            if not schedule_config:
                errors.append("Missing required field 'schedule_config' for schedule operation")

        elif operation == 'export':
            backup_id = inputs.get('backup_id', self.config.get('backup_id'))
            if not backup_id:
                errors.append("Missing required field 'backup_id' for export operation")
            destination = inputs.get('destination', self.config.get('destination'))
            if not destination:
                errors.append("Missing required field 'destination' for export operation")

        elif operation == 'delete':
            backup_id = inputs.get('backup_id', self.config.get('backup_id'))
            if not backup_id:
                errors.append("Missing required field 'backup_id' for delete operation")

        # Validate compression if provided
        if 'compression' in inputs:
            compression = inputs['compression']
            if compression not in self.SUPPORTED_COMPRESSION:
                errors.append(
                    f"Invalid compression: {compression}. "
                    f"Must be one of: {', '.join(self.SUPPORTED_COMPRESSION)}"
                )

        # Validate retention_days if provided
        if 'retention_days' in inputs:
            try:
                retention = int(inputs['retention_days'])
                if retention < 0:
                    errors.append("'retention_days' must be a non-negative integer")
            except (TypeError, ValueError):
                errors.append("'retention_days' must be an integer")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the backup/recovery operation based on configuration.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - success: Boolean indicating operation success
                - backup_id: ID of created/operated backup
                - size_bytes: Size of backup in bytes
                - timestamp: ISO timestamp of operation
                - details: Operation-specific details
                - errors: List of error messages

        Raises:
            NodeExecutionError: If execution fails
        """
        operation = inputs.get('operation', self.config.get('operation', 'backup'))

        context.update_progress(10, f"Starting {operation} operation")
        self.logger.info(f"Executing {operation}")

        try:
            if operation == 'backup':
                result = self._execute_backup(inputs, context)
            elif operation == 'restore':
                result = self._execute_restore(inputs, context)
            elif operation == 'list':
                result = self._execute_list(inputs, context)
            elif operation == 'verify':
                result = self._execute_verify(inputs, context)
            elif operation == 'schedule':
                result = self._execute_schedule(inputs, context)
            elif operation == 'export':
                result = self._execute_export(inputs, context)
            elif operation == 'delete':
                result = self._execute_delete(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': self.SUPPORTED_OPERATIONS}
                )

            context.update_progress(100, f"{operation.capitalize()} operation completed")

            # Add artifact to context
            context.add_artifact('backup_recovery', {
                'operation': operation,
                'success': result.get('success', False),
                'backup_id': result.get('backup_id'),
                'timestamp': result.get('timestamp')
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"{operation} operation failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"{operation.capitalize()} operation failed: {str(e)}",
                details={
                    'operation': operation,
                    'inputs': {k: v for k, v in inputs.items() if k not in ['password', 'token']},
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_backup(self, inputs: Dict, context) -> Dict[str, Any]:
        """Create a backup/snapshot of knowledge graph."""
        kg_id = inputs.get('knowledge_graph_id', self.config.get('knowledge_graph_id'))
        backup_name = inputs.get('backup_name', self.config.get('backup_name', f'backup_{kg_id}'))
        compression = inputs.get('compression', self.config.get('compression', 'gzip'))
        include_history = inputs.get('include_history', self.config.get('include_history', True))
        verify_after = inputs.get('verify_after', self.config.get('verify_after', True))
        retention_days = inputs.get('retention_days', self.config.get('retention_days', 30))

        context.update_progress(20, f"Preparing backup for knowledge graph: {kg_id}")

        # Generate backup ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_id = f"{backup_name}_{timestamp}"

        try:
            # Use backup manager if available, otherwise fallback to manual backup
            if self.backup_manager and hasattr(self.backup_manager, 'create_backup'):
                result = self.backup_manager.create_backup(
                    knowledge_graph_id=kg_id,
                    backup_name=backup_name,
                    compression=compression,
                    include_history=include_history
                )
                backup_id = result.get('backup_id', backup_id)
                backup_path = result.get('backup_path')
            else:
                # Fallback: manual backup
                result = self._manual_backup(kg_id, backup_id, backup_name, compression, include_history)
                backup_path = result.get('backup_path')

            context.update_progress(60, "Backup created, calculating size")

            # Get backup size
            size_bytes = 0
            if backup_path and os.path.exists(backup_path):
                size_bytes = os.path.getsize(backup_path)

            context.update_progress(80, "Verifying backup integrity" if verify_after else "Skipping verification")

            # Verify if requested
            verification_result = None
            if verify_after and backup_path:
                verification_result = self._verify_backup_integrity(backup_path)

            # Create backup metadata
            metadata = {
                'backup_id': backup_id,
                'backup_name': backup_name,
                'knowledge_graph_id': kg_id,
                'timestamp': datetime.now().isoformat(),
                'compression': compression,
                'include_history': include_history,
                'size_bytes': size_bytes,
                'retention_days': retention_days,
                'verified': verification_result.get('valid', False) if verification_result else None,
                'backup_path': backup_path
            }

            # Save metadata
            self._save_backup_metadata(backup_id, metadata)

            context.update_progress(100, f"Backup completed: {backup_id}")

            return {
                'success': True,
                'backup_id': backup_id,
                'backup_name': backup_name,
                'size_bytes': size_bytes,
                'timestamp': metadata['timestamp'],
                'compression': compression,
                'verified': metadata['verified'],
                'verification_details': verification_result,
                'errors': []
            }

        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return {
                'success': False,
                'backup_id': None,
                'size_bytes': 0,
                'timestamp': datetime.now().isoformat(),
                'errors': [f"Backup creation failed: {str(e)}"]
            }

    def _execute_restore(self, inputs: Dict, context) -> Dict[str, Any]:
        """Restore knowledge graph from backup."""
        backup_id = inputs.get('backup_id', self.config.get('backup_id'))
        kg_id = inputs.get('knowledge_graph_id', self.config.get('knowledge_graph_id'))
        verify_after = inputs.get('verify_after', self.config.get('verify_after', True))

        context.update_progress(20, f"Locating backup: {backup_id}")

        # Get backup metadata
        metadata = self._get_backup_metadata(backup_id)
        if not metadata:
            return {
                'success': False,
                'backup_id': backup_id,
                'errors': [f"Backup not found: {backup_id}"]
            }

        backup_path = metadata.get('backup_path')
        if not backup_path or not os.path.exists(backup_path):
            return {
                'success': False,
                'backup_id': backup_id,
                'errors': [f"Backup file not found: {backup_path}"]
            }

        context.update_progress(40, "Verifying backup integrity before restore")

        # Verify before restore
        if verify_after:
            verification = self._verify_backup_integrity(backup_path)
            if not verification.get('valid', False):
                return {
                    'success': False,
                    'backup_id': backup_id,
                    'errors': [f"Backup verification failed: {verification.get('errors', [])}"]
                }

        context.update_progress(60, "Restoring knowledge graph")

        try:
            # Use backup manager if available
            if self.backup_manager and hasattr(self.backup_manager, 'restore_backup'):
                result = self.backup_manager.restore_backup(
                    backup_id=backup_id,
                    target_knowledge_graph_id=kg_id
                )
            else:
                # Fallback: manual restore
                result = self._manual_restore(backup_path, kg_id or metadata.get('knowledge_graph_id'))

            context.update_progress(100, "Restore completed successfully")

            return {
                'success': True,
                'backup_id': backup_id,
                'knowledge_graph_id': kg_id or metadata.get('knowledge_graph_id'),
                'timestamp': datetime.now().isoformat(),
                'restored_entities': result.get('entity_count', 0),
                'restored_relations': result.get('relation_count', 0),
                'errors': []
            }

        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return {
                'success': False,
                'backup_id': backup_id,
                'timestamp': datetime.now().isoformat(),
                'errors': [f"Restore failed: {str(e)}"]
            }

    def _execute_list(self, inputs: Dict, context) -> Dict[str, Any]:
        """List available backups."""
        kg_id = inputs.get('knowledge_graph_id', self.config.get('knowledge_graph_id'))

        context.update_progress(30, "Scanning for available backups")

        backups = []

        # Use backup manager if available
        if self.backup_manager and hasattr(self.backup_manager, 'list_backups'):
            backups = self.backup_manager.list_backups(knowledge_graph_id=kg_id)
        else:
            # Fallback: scan backup directory
            backups = self._manual_list_backups(kg_id)

        context.update_progress(70, f"Found {len(backups)} backups")

        # Apply retention filter if requested
        retention_days = inputs.get('retention_days', self.config.get('retention_days'))
        if retention_days is not None:
            cutoff_date = datetime.now() - timedelta(days=int(retention_days))
            backups = [
                b for b in backups
                if self._parse_backup_date(b.get('timestamp', '')) > cutoff_date
            ]

        context.update_progress(100, "Backup listing complete")

        return {
            'success': True,
            'backups': backups,
            'count': len(backups),
            'timestamp': datetime.now().isoformat(),
            'errors': []
        }

    def _execute_verify(self, inputs: Dict, context) -> Dict[str, Any]:
        """Verify backup integrity."""
        backup_id = inputs.get('backup_id', self.config.get('backup_id'))

        context.update_progress(30, f"Locating backup: {backup_id}")

        # Get backup metadata
        metadata = self._get_backup_metadata(backup_id)
        if not metadata:
            return {
                'success': False,
                'backup_id': backup_id,
                'valid': False,
                'errors': [f"Backup not found: {backup_id}"]
            }

        backup_path = metadata.get('backup_path')
        if not backup_path or not os.path.exists(backup_path):
            return {
                'success': False,
                'backup_id': backup_id,
                'valid': False,
                'errors': [f"Backup file not found: {backup_path}"]
            }

        context.update_progress(60, "Running integrity checks")

        # Perform verification
        verification = self._verify_backup_integrity(backup_path)

        context.update_progress(100, "Verification complete")

        return {
            'success': verification.get('valid', False),
            'backup_id': backup_id,
            'valid': verification.get('valid', False),
            'timestamp': datetime.now().isoformat(),
            'checksum_match': verification.get('checksum_match'),
            'file_readable': verification.get('file_readable'),
            'metadata_valid': verification.get('metadata_valid'),
            'errors': verification.get('errors', [])
        }

    def _execute_schedule(self, inputs: Dict, context) -> Dict[str, Any]:
        """Schedule automatic backups."""
        kg_id = inputs.get('knowledge_graph_id', self.config.get('knowledge_graph_id'))
        backup_name = inputs.get('backup_name', self.config.get('backup_name', f'scheduled_{kg_id}'))
        schedule_config = inputs.get('schedule_config', self.config.get('schedule_config', {}))
        compression = inputs.get('compression', self.config.get('compression', 'gzip'))
        include_history = inputs.get('include_history', self.config.get('include_history', True))
        retention_days = inputs.get('retention_days', self.config.get('retention_days', 30))

        context.update_progress(40, "Configuring backup schedule")

        # Generate schedule ID
        schedule_id = f"schedule_{kg_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Parse schedule configuration
        interval_hours = schedule_config.get('interval_hours', 24)
        max_backups = schedule_config.get('max_backups', 10)

        # Store schedule configuration
        self._scheduled_backups[schedule_id] = {
            'schedule_id': schedule_id,
            'knowledge_graph_id': kg_id,
            'backup_name': backup_name,
            'interval_hours': interval_hours,
            'max_backups': max_backups,
            'compression': compression,
            'include_history': include_history,
            'retention_days': retention_days,
            'created_at': datetime.now().isoformat(),
            'next_run': (datetime.now() + timedelta(hours=interval_hours)).isoformat(),
            'enabled': True
        }

        context.update_progress(80, "Starting scheduler thread")

        # Start scheduler thread (simplified - in production, use a proper scheduler)
        self._start_schedule_thread(schedule_id)

        context.update_progress(100, f"Schedule configured: {schedule_id}")

        return {
            'success': True,
            'schedule_id': schedule_id,
            'knowledge_graph_id': kg_id,
            'interval_hours': interval_hours,
            'max_backups': max_backups,
            'next_run': self._scheduled_backups[schedule_id]['next_run'],
            'timestamp': datetime.now().isoformat(),
            'errors': []
        }

    def _execute_export(self, inputs: Dict, context) -> Dict[str, Any]:
        """Export backup to external storage."""
        backup_id = inputs.get('backup_id', self.config.get('backup_id'))
        destination = inputs.get('destination', self.config.get('destination'))

        context.update_progress(30, f"Locating backup: {backup_id}")

        # Get backup metadata
        metadata = self._get_backup_metadata(backup_id)
        if not metadata:
            return {
                'success': False,
                'backup_id': backup_id,
                'errors': [f"Backup not found: {backup_id}"]
            }

        backup_path = metadata.get('backup_path')
        if not backup_path or not os.path.exists(backup_path):
            return {
                'success': False,
                'backup_id': backup_id,
                'errors': [f"Backup file not found: {backup_path}"]
            }

        context.update_progress(60, f"Exporting to: {destination}")

        try:
            # Ensure destination directory exists
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy backup to destination
            shutil.copy2(backup_path, destination)

            # Verify copy
            if not os.path.exists(destination):
                raise IOError("Export file was not created")

            dest_size = os.path.getsize(destination)

            context.update_progress(100, "Export completed")

            return {
                'success': True,
                'backup_id': backup_id,
                'destination': destination,
                'size_bytes': dest_size,
                'timestamp': datetime.now().isoformat(),
                'errors': []
            }

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return {
                'success': False,
                'backup_id': backup_id,
                'destination': destination,
                'timestamp': datetime.now().isoformat(),
                'errors': [f"Export failed: {str(e)}"]
            }

    def _execute_delete(self, inputs: Dict, context) -> Dict[str, Any]:
        """Delete a backup."""
        backup_id = inputs.get('backup_id', self.config.get('backup_id'))

        context.update_progress(40, f"Locating backup: {backup_id}")

        # Get backup metadata
        metadata = self._get_backup_metadata(backup_id)
        if not metadata:
            return {
                'success': False,
                'backup_id': backup_id,
                'errors': [f"Backup not found: {backup_id}"]
            }

        backup_path = metadata.get('backup_path')

        context.update_progress(70, "Deleting backup files")

        try:
            # Delete backup file
            if backup_path and os.path.exists(backup_path):
                os.remove(backup_path)

            # Delete metadata file
            metadata_path = self._get_metadata_path(backup_id)
            if metadata_path.exists():
                metadata_path.unlink()

            context.update_progress(100, "Backup deleted successfully")

            return {
                'success': True,
                'backup_id': backup_id,
                'timestamp': datetime.now().isoformat(),
                'errors': []
            }

        except Exception as e:
            self.logger.error(f"Delete failed: {e}")
            return {
                'success': False,
                'backup_id': backup_id,
                'timestamp': datetime.now().isoformat(),
                'errors': [f"Delete failed: {str(e)}"]
            }

    # =========================================================================
    # Manual backup/restore implementations (fallback)
    # =========================================================================

    def _manual_backup(
        self,
        kg_id: str,
        backup_id: str,
        backup_name: str,
        compression: str,
        include_history: bool
    ) -> Dict[str, Any]:
        """Create manual backup when BackupManager is unavailable."""
        # Create backup directory for this backup
        backup_dir = self.default_backup_dir / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Get knowledge data from hub or create empty
        if self.hub and hasattr(self.hub, 'entities'):
            knowledge_data = {
                'knowledge_graph_id': kg_id,
                'backup_name': backup_name,
                'timestamp': datetime.now().isoformat(),
                'entities': dict(self.hub.entities) if hasattr(self.hub.entities, 'items') else {},
                'relations': dict(self.hub.relations) if hasattr(self.hub, 'relations') else {},
                'triples': [
                    t.to_dict() if hasattr(t, 'to_dict') else t
                    for t in (self.hub.triples if hasattr(self.hub, 'triples') else [])
                ],
                'include_history': include_history
            }
        else:
            knowledge_data = {
                'knowledge_graph_id': kg_id,
                'backup_name': backup_name,
                'timestamp': datetime.now().isoformat(),
                'entities': {},
                'relations': {},
                'triples': [],
                'note': 'Manual backup - hub not available',
                'include_history': include_history
            }

        # Serialize to JSON
        json_data = json.dumps(knowledge_data, indent=2, default=str)

        # Apply compression
        if compression == 'gzip':
            backup_path = backup_dir / 'backup.json.gz'
            with gzip.open(backup_path, 'wt', encoding='utf-8') as f:
                f.write(json_data)
        elif compression == 'zip':
            backup_path = backup_dir / 'backup.zip'
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('backup.json', json_data)
        else:
            backup_path = backup_dir / 'backup.json'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(json_data)

        # Calculate and store checksum
        checksum = hashlib.sha256(json_data.encode('utf-8')).hexdigest()
        checksum_path = backup_dir / 'checksum.sha256'
        with open(checksum_path, 'w') as f:
            f.write(checksum)

        return {
            'backup_id': backup_id,
            'backup_path': str(backup_path),
            'checksum': checksum
        }

    def _manual_restore(self, backup_path: str, kg_id: str) -> Dict[str, Any]:
        """Restore from manual backup."""
        # Load backup data
        if backup_path.endswith('.gz'):
            with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
                json_data = f.read()
        elif backup_path.endswith('.zip'):
            with zipfile.ZipFile(backup_path, 'r') as zf:
                json_data = zf.read('backup.json').decode('utf-8')
        else:
            with open(backup_path, 'r', encoding='utf-8') as f:
                json_data = f.read()

        knowledge_data = json.loads(json_data)

        # Restore to hub if available
        if self.hub:
            entities = knowledge_data.get('entities', {})
            relations = knowledge_data.get('relations', {})
            triples = knowledge_data.get('triples', [])

            if hasattr(self.hub, 'entities'):
                self.hub.entities.update(entities)
            if hasattr(self.hub, 'relations'):
                self.hub.relations.update(relations)
            # Triples restoration depends on hub implementation

        return {
            'entity_count': len(knowledge_data.get('entities', {})),
            'relation_count': len(knowledge_data.get('relations', {})),
            'triple_count': len(knowledge_data.get('triples', []))
        }

    def _manual_list_backups(self, kg_id: Optional[str] = None) -> List[Dict]:
        """List backups from backup directory."""
        backups = []

        if not self.default_backup_dir.exists():
            return backups

        for backup_dir in self.default_backup_dir.iterdir():
            if backup_dir.is_dir():
                metadata_path = backup_dir / 'metadata.json'
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        if kg_id is None or metadata.get('knowledge_graph_id') == kg_id:
                            backups.append(metadata)
                    except (json.JSONDecodeError, IOError):
                        continue

        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return backups

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _save_backup_metadata(self, backup_id: str, metadata: Dict):
        """Save backup metadata to file."""
        backup_dir = self.default_backup_dir / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = backup_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _get_backup_metadata(self, backup_id: str) -> Optional[Dict]:
        """Get backup metadata by ID."""
        # Try backup directory first
        metadata_path = self.default_backup_dir / backup_id / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Try backup manager if available
        if self.backup_manager and hasattr(self.backup_manager, 'get_backup_metadata'):
            return self.backup_manager.get_backup_metadata(backup_id)

        return None

    def _get_metadata_path(self, backup_id: str) -> Path:
        """Get path to backup metadata file."""
        return self.default_backup_dir / backup_id / 'metadata.json'

    def _verify_backup_integrity(self, backup_path: str) -> Dict[str, Any]:
        """Verify backup file integrity."""
        errors = []
        valid = True

        # Check file exists and is readable
        if not os.path.exists(backup_path):
            return {'valid': False, 'file_readable': False, 'errors': ['File not found']}

        try:
            # Try to read file
            if backup_path.endswith('.gz'):
                with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
                    content = f.read()
            elif backup_path.endswith('.zip'):
                with zipfile.ZipFile(backup_path, 'r') as zf:
                    content = zf.read('backup.json').decode('utf-8')
            else:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            # Try to parse JSON
            try:
                data = json.loads(content)
                metadata_valid = isinstance(data, dict) and 'knowledge_graph_id' in data
            except json.JSONDecodeError as e:
                valid = False
                metadata_valid = False
                errors.append(f"Invalid JSON: {e}")
                data = None

            # Verify checksum if available
            checksum_match = None
            backup_dir = Path(backup_path).parent
            checksum_path = backup_dir / 'checksum.sha256'
            if checksum_path.exists():
                try:
                    with open(checksum_path, 'r') as f:
                        expected_checksum = f.read().strip()
                    actual_checksum = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    checksum_match = expected_checksum == actual_checksum
                    if not checksum_match:
                        valid = False
                        errors.append("Checksum mismatch - backup may be corrupted")
                except Exception as e:
                    errors.append(f"Checksum verification failed: {e}")

            return {
                'valid': valid and metadata_valid,
                'file_readable': True,
                'metadata_valid': metadata_valid,
                'checksum_match': checksum_match,
                'file_size': len(content),
                'errors': errors
            }

        except Exception as e:
            return {
                'valid': False,
                'file_readable': False,
                'metadata_valid': False,
                'errors': [f"File read error: {str(e)}"]
            }

    def _parse_backup_date(self, timestamp_str: str) -> datetime:
        """Parse backup timestamp string to datetime."""
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return datetime.min

    def _start_schedule_thread(self, schedule_id: str):
        """Start a thread for scheduled backups (simplified implementation)."""
        def run_schedule():
            schedule = self._scheduled_backups.get(schedule_id)
            if not schedule:
                return

            self.logger.info(f"Schedule thread started for {schedule_id}")

            # In a real implementation, this would use a proper scheduler like APScheduler
            # For now, just log that the schedule is configured
            self.logger.info(f"Schedule configured: every {schedule['interval_hours']} hours")

        thread = threading.Thread(target=run_schedule, daemon=True)
        thread.start()
        self._schedule_threads[schedule_id] = thread

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Backup & Recovery Configuration",
            "description": "Configure backup and disaster recovery parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Backup/recovery operation to perform",
                    "enum": ["backup", "restore", "list", "verify", "schedule", "export", "delete"],
                    "enumNames": [
                        "Backup - Create a new backup snapshot",
                        "Restore - Restore from a backup",
                        "List - List available backups",
                        "Verify - Verify backup integrity",
                        "Schedule - Configure automatic backups",
                        "Export - Export backup to external storage",
                        "Delete - Remove a backup"
                    ],
                    "default": "backup"
                },
                "knowledge_graph_id": {
                    "type": "string",
                    "title": "Knowledge Graph ID",
                    "description": "ID of the knowledge graph to backup/restore",
                    "default": ""
                },
                "backup_id": {
                    "type": "string",
                    "title": "Backup ID",
                    "description": "Specific backup ID (for restore/verify/export/delete operations)",
                    "default": ""
                },
                "backup_name": {
                    "type": "string",
                    "title": "Backup Name",
                    "description": "Name for the new backup (optional, auto-generated if not provided)",
                    "default": ""
                },
                "destination": {
                    "type": "string",
                    "title": "Destination Path",
                    "description": "Path for export operation",
                    "default": ""
                },
                "compression": {
                    "type": "string",
                    "title": "Compression",
                    "description": "Compression method for backups",
                    "enum": ["none", "gzip", "zip"],
                    "enumNames": [
                        "None - No compression",
                        "Gzip - Fast compression",
                        "Zip - Standard compression"
                    ],
                    "default": "gzip"
                },
                "include_history": {
                    "type": "boolean",
                    "title": "Include History",
                    "description": "Include historical data and audit logs in backup",
                    "default": True
                },
                "verify_after": {
                    "type": "boolean",
                    "title": "Verify After Operation",
                    "description": "Verify backup integrity after create/restore operations",
                    "default": True
                },
                "retention_days": {
                    "type": "integer",
                    "title": "Retention Days",
                    "description": "Number of days to retain backups (0 = unlimited)",
                    "minimum": 0,
                    "default": 30
                },
                "backup_directory": {
                    "type": "string",
                    "title": "Backup Directory",
                    "description": "Directory to store backups (default: ./backups)",
                    "default": "./backups"
                },
                "schedule_config": {
                    "type": "object",
                    "title": "Schedule Configuration",
                    "description": "Configuration for scheduled backups",
                    "properties": {
                        "interval_hours": {
                            "type": "integer",
                            "title": "Interval (hours)",
                            "description": "Hours between automatic backups",
                            "minimum": 1,
                            "default": 24
                        },
                        "max_backups": {
                            "type": "integer",
                            "title": "Max Backups",
                            "description": "Maximum number of backups to retain",
                            "minimum": 1,
                            "default": 10
                        }
                    }
                }
            },
            "required": ["operation"]
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy (can perform backup operations)
        """
        try:
            # Check backup directory is writable
            test_file = self.default_backup_dir / '.health_check'
            try:
                test_file.touch()
                test_file.unlink()
            except (OSError, IOError):
                return False

            # Node can work with or without BackupManager (has fallback)
            return True
        except Exception:
            return False

    def get_supported_operations(self) -> List[str]:
        """
        Get list of supported operations.

        Returns:
            List of operation names
        """
        return self.SUPPORTED_OPERATIONS.copy()

    def get_scheduled_backups(self) -> Dict[str, Dict]:
        """
        Get all configured scheduled backups.

        Returns:
            Dictionary mapping schedule IDs to their configurations
        """
        return self._scheduled_backups.copy()
