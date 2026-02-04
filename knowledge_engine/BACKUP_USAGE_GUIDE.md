# Backup and Recovery System - Quick Usage Guide

## Quick Start

### 1. Local Storage (Simplest)

```python
from knowledge_engine.backup_recovery import create_storage_backend, BackupEngine

# Create local storage backend
storage = create_storage_backend('local', base_path='./backups')

# Create backup engine
engine = BackupEngine(storage=storage, retention_days=30)

# Create a backup
metadata = await engine.create_backup('/path/to/important/data')
print(f"Backup created: {metadata.backup_id}")
```

### 2. Cloud Storage

```python
# S3
storage = create_storage_backend('s3', bucket_name='my-backups')

# Google Cloud
storage = create_storage_backend('gcs', bucket_name='my-backups', project_id='my-project')

# Azure
storage = create_storage_backend('azure', container_name='my-backups')

# Use the storage backend
engine = BackupEngine(storage=storage)
```

## Common Operations

### Create Backup

```python
# Full backup
metadata = await engine.create_backup(
    source_path='/data',
    backup_type=BackupType.FULL
)

# Incremental backup (only changed files)
metadata = await engine.create_backup(
    source_path='/data',
    backup_type=BackupType.INCREMENTAL,
    parent_backup_id='parent-backup-id'
)

# With include/exclude patterns
metadata = await engine.create_backup(
    source_path='/data',
    include_patterns=['*.py', '*.txt'],
    exclude_patterns=['*.tmp', '*.log']
)
```

### Restore Backup

```python
# Restore with checksum verification
success = await engine.restore_backup(
    backup_id='backup-id',
    destination_path='/restore/target',
    verify_checksum=True
)
```

### Verify Backup Integrity

```python
verified = await engine.verify_backup('backup-id')
if verified:
    print("Backup is valid and can be restored")
```

### List Backups and Statistics

```python
# Get statistics
stats = engine.get_backup_stats()
print(f"Total backups: {stats['total_backups']}")
print(f"Total size: {stats['total_size_gb']:.2f} GB")

# Get recovery points
points = engine.get_recovery_points()
for point in points:
    print(f"{point.timestamp}: {point.backup_id}")
```

### Scheduled Backups

```python
# Daily incremental backups
engine.schedule_backup(
    schedule_id='daily',
    source_path='/data',
    cron_expression='daily',
    backup_type=BackupType.INCREMENTAL
)

# Weekly full backups
engine.schedule_backup(
    schedule_id='weekly',
    source_path='/data',
    cron_expression='weekly',
    backup_type=BackupType.FULL
)

# Cancel a schedule
engine.cancel_schedule('daily')
```

### Cleanup Old Backups

```python
# Remove backups older than retention period
await engine.cleanup_old_backups()
```

## Disaster Recovery

```python
from knowledge_engine.backup_recovery import DisasterRecovery

dr = DisasterRecovery(engine)

# Generate DR plan
plan = dr.generate_dr_plan()
print(f"RPO: {plan['rpo_hours']}h, RTO: {plan['rto_hours']}h")

# Test recovery (non-destructive)
result = await dr.test_recovery(
    backup_id='backup-id',
    test_path='/tmp/test-restore'
)
print(f"Test restore: {result['success']}")

# Execute failover
success = await dr.failover(
    backup_id='backup-id',
    target_path='/production/data'
)
```

## Configuration Examples

### AWS S3 with Environment Variables

```bash
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
export AWS_REGION="us-west-2"
```

### S3-Compatible (MinIO, DigitalOcean Spaces)

```python
storage = create_storage_backend(
    's3',
    bucket_name='my-bucket',
    endpoint_url='https://nyc3.digitaloceanspaces.com',
    region='nyc3'
)
```

### Google Cloud with Service Account

```python
storage = create_storage_backend(
    'gcs',
    bucket_name='my-bucket',
    project_id='my-project',
    credentials_path='/path/to/service-account.json'
)
```

### Azure with Connection String

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=myaccount;..."
```

## Error Handling

```python
try:
    metadata = await engine.create_backup('/data')
except ValueError as e:
    print(f"Configuration error: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install boto3  # or google-cloud-storage or azure-storage-blob")
except Exception as e:
    print(f"Backup failed: {e}")
```

## Best Practices

1. **Use Incremental Backups** for large datasets to save space and time
2. **Verify Backups** regularly with `verify_backup()`
3. **Test Recovery** periodically with `test_recovery()`
4. **Set Retention Policies** to automatically clean up old backups
5. **Monitor Logs** for backup failures
6. **Use Cloud Storage** for production (more reliable than local)
7. **Schedule Backups** during off-peak hours

## Troubleshooting

### "Missing dependency" Error

```bash
# For S3
pip install boto3

# For GCS
pip install google-cloud-storage

# For Azure
pip install azure-storage-blob
```

### "Credentials not found" Error

Make sure environment variables are set correctly for your cloud provider.

### "Backup not completed" Error

Check the backup status:
```python
metadata = engine._backups['backup-id']
print(f"Status: {metadata.status}")
print(f"Error: {metadata.error_message}")
```

### Storage Backend Issues

The system logs all operations with structured JSON. Check logs for detailed error messages.

## Testing

```bash
# Run the test suite
python test_backup_system.py

# Test local storage
python -c "
from knowledge_engine.backup_recovery import create_storage_backend
storage = create_storage_backend('local', base_path='./test-backups')
print('Local storage works!')
"

# Test cloud backend (will fail gracefully if credentials missing)
python -c "
from knowledge_engine.backup_recovery import create_storage_backend
try:
    storage = create_storage_backend('s3', bucket_name='test')
    print('S3 backend initialized!')
except Exception as e:
    print(f'Expected: {e}')
"
```

## Full Example

```python
import asyncio
from pathlib import Path
from knowledge_engine.backup_recovery import (
    create_storage_backend,
    BackupEngine,
    BackupType,
    DisasterRecovery
)

async def main():
    # Setup
    storage = create_storage_backend('local', base_path='./backups')
    engine = BackupEngine(storage=storage, retention_days=30)

    # Create test data
    test_dir = Path('./test_data')
    test_dir.mkdir(exist_ok=True)
    (test_dir / 'important.txt').write_text('Critical data')

    # Create full backup
    print("Creating full backup...")
    full_backup = await engine.create_backup(str(test_dir), BackupType.FULL)
    print(f"  ID: {full_backup.backup_id}")
    print(f"  Size: {full_backup.size_bytes} bytes")

    # Verify backup
    print("Verifying backup...")
    verified = await engine.verify_backup(full_backup.backup_id)
    print(f"  Verified: {verified}")

    # Restore backup
    print("Restoring backup...")
    restore_dir = Path('./restored')
    success = await engine.restore_backup(
        full_backup.backup_id,
        str(restore_dir),
        verify_checksum=True
    )
    print(f"  Restored: {success}")

    # Statistics
    stats = engine.get_backup_stats()
    print(f"\nStatistics:")
    print(f"  Total backups: {stats['total_backups']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Verified: {stats['verified']}")

if __name__ == '__main__':
    asyncio.run(main())
```

## Additional Resources

- **Implementation Details**: See `BACKUP_SYSTEM_FIX_REPORT.md`
- **Test Suite**: See `test_backup_system.py`
- **Source Code**: See `knowledge_engine/backup_recovery.py`
