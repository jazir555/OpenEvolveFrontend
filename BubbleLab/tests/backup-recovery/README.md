# BubbleLab Backup & Recovery Testing

## Overview

This document describes the backup and recovery procedures for the BubbleLab system. Regular testing of backup and recovery procedures is **critical** for production readiness.

**Recovery Time Objective (RTO)**: 1 hour
**Recovery Point Objective (RPO)**: 5 minutes

---

## Table of Contents

1. [Database Backup Procedures](#database-backup-procedures)
2. [Database Restore Procedures](#database-restore-procedures)
3. [Configuration Backup](#configuration-backup)
4. [Disaster Recovery Procedures](#disaster-recovery-procedures)
5. [Testing Checklist](#testing-checklist)
6. [Runbooks](#runbooks)

---

## Database Backup Procedures

### PostgreSQL Backup

#### Automatic Backups (Recommended for Production)

**Schedule**: Every 5 minutes (WAL archiving)
**Retention**: 30 days

**Setup**:

```bash
# 1. Configure WAL archiving in postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'
max_wal_senders = 3
wal_keep_size = 1GB

# 2. Create backup directory
mkdir -p /backup/wal
chown -R postgres:postgres /backup

# 3. Configure pgBackRest (recommended)
# See: https://pgbackrest.org/configuration.html
```

**Cron Job**:

```bash
# Full backup daily at 2 AM
0 2 * * * pgbackrest --stanza=bubblelab --type=full backup

# Differential backup every 6 hours
0 */6 * * * pgbackrest --stanza=bubblelab --type=diff backup

# Incremental backup every 5 minutes
*/5 * * * * pgbackrest --stanza=bubblelab --type=incr backup
```

#### Manual Backup

**Full Backup**:

```bash
# Using pg_dump
pg_dump -U postgres -F c -b -v -f "/backup/bubblelab_$(date +%Y%m%d_%H%M%S).backup" bubblelab

# Using pgBackRest
pgbackrest --stanza=bubblelab --type=full backup
```

**Schema Only**:

```bash
pg_dump -U postgres -s bubblelab > "/backup/bubblelab_schema_$(date +%Y%m%d_%H%M%S).sql"
```

**Data Only**:

```bash
pg_dump -U postgres -a bubblelab > "/backup/bubblelab_data_$(date +%Y%m%d_%H%M%S).sql"
```

### SQLite Backup (Development/Staging)

```bash
# Using sqlite3 backup command
sqlite3 bubblelab.db ".backup '/backup/bubblelab_$(date +%Y%m%d_%H%M%S).db'"

# Or copy file (must stop writes first)
cp bubblelab.db "/backup/bubblelab_$(date +%Y%m%d_%H%M%S).db"
```

### Qdrant Backup

**Using Qdrant API**:

```bash
# 1. Create snapshot
curl -X PUT "http://localhost:6333/collections/{collection_name}/snapshots"

# 2. Download snapshot
curl -X GET "http://localhost:6333/collections/{collection_name}/snapshots/{snapshot_name}" \
  --output /backup/qdrant_{collection_name}_$(date +%Y%m%d_%H%M%S).snapshot

# 3. List snapshots
curl -X GET "http://localhost:6333/collections/{collection_name}/snapshots"
```

**Automated Backup Script**:

```bash
#!/bin/bash
# backup-qdrant.sh

COLLECTIONS=$(curl -s http://localhost:6333/collections | jq -r '.result.collections[].name')

for collection in $COLLECTIONS; do
  echo "Backing up collection: $collection"

  # Create snapshot
  curl -X PUT "http://localhost:6333/collections/$collection/snapshots"

  # Get snapshot name
  SNAPSHOT=$(curl -s "http://localhost:6333/collections/$collection/snapshots" | jq -r '.result[0].name')

  # Download snapshot
  curl -X GET "http://localhost:6333/collections/$collection/snapshots/$SNAPSHOT" \
    --output "/backup/qdrant_${collection}_$(date +%Y%m%d_%H%M%S).snapshot"
done

echo "Qdrant backup completed: $(date)"
```

### Elasticsearch Backup

**Using Snapshot Repository**:

```bash
# 1. Register snapshot repository
curl -X PUT "localhost:9200/_snapshot/backup_repo" -H 'Content-Type: application/json' -d'
{
  "type": "fs",
  "settings": {
    "location": "/backup/elasticsearch"
  }
}
'

# 2. Create snapshot
curl -X PUT "localhost:9200/_snapshot/backup_repo/snapshot_$(date +%Y%m%d_%H%M%S)?wait_for_completion=true"

# 3. List snapshots
curl -X GET "localhost:9200/_snapshot/backup_repo/_all"
```

**Automated Backup Script**:

```bash
#!/bin/bash
# backup-elasticsearch.sh

SNAPSHOT_NAME="snapshot_$(date +%Y%m%d_%H%M%S)"

curl -X PUT "localhost:9200/_snapshot/backup_repo/${SNAPSHOT_NAME}?wait_for_completion=true"

echo "Elasticsearch snapshot created: $SNAPSHOT_NAME"
```

### Redis Backup

**Using RDB Snapshots**:

```bash
# Redis automatically saves to .rdb file
# Copy RDB file for backup
cp /var/lib/redis/dump.rdb "/backup/redis_$(date +%Y%m%d_%H%M%S).rdb"
```

**Using AOF (Append Only File)**:

```bash
# Enable AOF in redis.conf
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec

# Copy AOF file for backup
cp /var/lib/redis/appendonly.aof "/backup/redis_$(date +%Y%m%d_%H%M%S).aof"
```

---

## Database Restore Procedures

### PostgreSQL Restore

#### Restore from pg_dump Backup

```bash
# 1. Stop the application
docker-compose stop bubblelab-api

# 2. Drop existing database
psql -U postgres -c "DROP DATABASE IF EXISTS bubblelab"

# 3. Create new database
psql -U postgres -c "CREATE DATABASE bubblelab"

# 4. Restore from backup
pg_restore -U postgres -d bubblelab -v "/backup/bubblelab_20260118_020000.backup"

# 5. Restart application
docker-compose start bubblelab-api

# 6. Verify restore
psql -U postgres -d bubblelab -c "SELECT COUNT(*) FROM users"
```

#### Point-in-Time Recovery (PITR)

```bash
# 1. Restore from base backup
pgbackrest --stanza=bubblelab --delta restore

# 2. Recover to specific time
pgbackrest --stanza=bubblelab --type=time "--target=2026-01-18 14:30:00" delta restore

# 3. Start PostgreSQL
systemctl start postgresql

# 4. Verify recovery
psql -U postgres -d bubblelab -c "SELECT NOW()"
```

### SQLite Restore

```bash
# 1. Stop the application
docker-compose stop bubblelab-api

# 2. Restore from backup
cp /backup/bubblelab_20260118_020000.db bubblelab.db

# 3. Verify integrity
sqlite3 bubblelab.db "PRAGMA integrity_check"

# 4. Restart application
docker-compose start bubblelab-api
```

### Qdrant Restore

```bash
# 1. Upload snapshot to Qdrant
curl -X POST "http://localhost:6333/collections/{collection_name}/snapshots/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@/backup/qdrant_{collection_name}_20260118_020000.snapshot"

# 2. Recover from snapshot
curl -X POST "http://localhost:6333/collections/{collection_name}/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location": "snapshot_name"}'

# 3. Verify recovery
curl -X GET "http://localhost:6333/collections/{collection_name}"
```

### Elasticsearch Restore

```bash
# 1. List available snapshots
curl -X GET "localhost:9200/_snapshot/backup_repo/_all"

# 2. Restore from snapshot
curl -X POST "localhost:9200/_snapshot/backup_repo/snapshot_20260118_020000/_restore"

# 3. Close all indices first (if needed)
curl -X POST "localhost:9200/_all/_close"

# 4. Verify restore
curl -X GET "localhost:9200/_cat/indices?v"
```

### Redis Restore

```bash
# 1. Stop Redis
systemctl stop redis

# 2. Copy backup file
cp /backup/redis_20260118_020000.rdb /var/lib/redis/dump.rdb

# 3. Start Redis
systemctl start redis

# 4. Verify restore
redis-cli DBSIZE
redis-cli GET "some_key"
```

---

## Configuration Backup

### Backup Configuration Files

```bash
#!/bin/bash
# backup-config.sh

BACKUP_DIR="/backup/config/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup environment files
cp -r /bubblelab/config/environments "$BACKUP_DIR/"

# Backup docker-compose files
cp /bubblelab/docker-compose*.yml "$BACKUP_DIR/"

# Backup service discovery
cp /bubblelab/config/service-discovery.yaml "$BACKUP_DIR/"

# Backup workflow registry
cp /bubblelab/config/workflow-registry.yaml "$BACKUP_DIR/"

# Backup credentials template (without actual secrets)
cp /bubblelab/config/credentials-template.yaml "$BACKUP_DIR/"

echo "Configuration backed up to: $BACKUP_DIR"
```

### Restore Configuration

```bash
#!/bin/bash
# restore-config.sh

BACKUP_DIR="/backup/config/20260118_020000"

# Restore configuration files
cp -r "$BACKUP_DIR/environments" /bubblelab/config/
cp "$BACKUP_DIR"/*.yml /bubblelab/config/

# Validate configuration
node /bubblelab/config/validate-config.js --env production --strict

# If validation passes, restart services
docker-compose restart
```

---

## Disaster Recovery Procedures

### Scenario 1: Complete Server Failure

**Recovery Time**: 1-2 hours

**Steps**:

1. **Provision new server** (30 minutes)
   ```bash
   # Use infrastructure as code
   terraform apply
   ```

2. **Install dependencies** (15 minutes)
   ```bash
   # Docker, Docker Compose, etc.
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   ```

3. **Restore configuration** (5 minutes)
   ```bash
   # From config backup
   ./restore-config.sh
   ```

4. **Restore databases** (30 minutes)
   ```bash
   # PostgreSQL
   ./restore-postgresql.sh

   # Qdrant
   ./restore-qdrant.sh

   # Elasticsearch
   ./restore-elasticsearch.sh

   # Redis
   ./restore-redis.sh
   ```

5. **Start services** (10 minutes)
   ```bash
   docker-compose up -d
   ```

6. **Verify restoration** (10 minutes)
   ```bash
   # Health checks
   curl http://localhost:3000/health

   # Check services
   curl http://localhost:6333/healthz  # Qdrant
   curl http://localhost:9200/_cluster/health  # Elasticsearch
   ```

### Scenario 2: Database Corruption

**Recovery Time**: 30 minutes

**Steps**:

1. **Detect corruption** (5 minutes)
   ```bash
   psql -U postgres -d bubblelab -c "SELECT * FROM users LIMIT 1"
   # If error, database is corrupted
   ```

2. **Stop application** (2 minutes)
   ```bash
   docker-compose stop bubblelab-api
   ```

3. **Restore from latest backup** (15 minutes)
   ```bash
   pg_restore -U postgres -d bubblelab -v /backup/bubblelab_latest.backup
   ```

4. **Verify restore** (5 minutes)
   ```bash
   psql -U postgres -d bubblelab -c "SELECT COUNT(*) FROM users"
   ```

5. **Start application** (3 minutes)
   ```bash
   docker-compose start bubblelab-api
   ```

### Scenario 3: Accidental Data Deletion

**Recovery Time**: 20 minutes

**Steps**:

1. **Identify deletion time** (5 minutes)
   ```bash
   # Check audit logs
   grep "DELETE FROM users" /var/log/bubblelab/app.log
   ```

2. **Stop writes** (2 minutes)
   ```bash
   docker-compose stop bubblelab-api
   ```

3. **PITR to before deletion** (10 minutes)
   ```bash
   pgbackrest --stanza=bubblelab --type=time "--target=2026-01-18 14:25:00" delta restore
   ```

4. **Verify restore** (3 minutes)
   ```bash
   psql -U postgres -d bubblelab -c "SELECT COUNT(*) FROM users"
   ```

### Scenario 4: Ransomware Attack

**Recovery Time**: 2-4 hours

**Steps**:

1. **Isolate infected systems** (immediate)
   ```bash
   # Disconnect from network
   # Don't shut down (for forensics)
   ```

2. **Assess damage** (30 minutes)
   ```bash
   # Check which files are encrypted
   find / -name "*.encrypted" -type f
   ```

3. **Rebuild from scratch** (1 hour)
   ```bash
   # Provision clean servers
   # Install dependencies
   ```

4. **Restore from offline backups** (1 hour)
   ```bash
   # Use backups stored offsite
   ./restore-all.sh
   ```

5. **Verify no malware persists** (30 minutes)
   ```bash
   # Run security scan
   clamscan -r /bubblelab
   ```

6. **Change all credentials** (30 minutes)
   ```bash
   # Rotate API keys, passwords, certificates
   ```

---

## Testing Checklist

### Weekly Backup Verification

- [ ] **PostgreSQL backup completed**
  - [ ] Full backup exists
  - [ ] Backup file size reasonable
  - [ ] Backup file not corrupted

- [ ] **Qdrant snapshot created**
  - [ ] Snapshot exists for all collections
  - [ ] Snapshot can be downloaded

- [ ] **Elasticsearch snapshot created**
  - [ ] Snapshot in repository
  - [ ] Snapshot status is SUCCESS

- [ ] **Redis RDB/AOF file exists**
  - [ ] File size reasonable
  - [ ] File not corrupted

- [ ] **Configuration backup exists**
  - [ ] All config files backed up
  - [ ] Backup recent (< 24 hours)

### Monthly Restore Test

- [ ] **PostgreSQL restore tested**
  - [ ] Restore to test environment successful
  - [ ] Data integrity verified
  - [ ] Application connects successfully

- [ ] **Qdrant restore tested**
  - [ ] Snapshot upload successful
  - [ ] Collection recovered
  - [ ] Data verified (point count)

- [ ] **Elasticsearch restore tested**
  - [ ] Snapshot restore successful
  - [ ] Indices recovered
  - [ ] Data verified (document count)

- [ ] **Redis restore tested**
  - [ ] RDB/AOF file loaded
  - [ ] Keys present
  - [ ] Data verified

- [ ] **Configuration restore tested**
  - [ ] Config files restored
  - [ ] Validation passes
  - [ ] Services start successfully

### Quarterly Disaster Recovery Drill

- [ ] **Complete server failure tested**
  - [ ] New server provisioned
  - [ ] All backups restored
  - [ ] Services operational
  - [ ] RTO met (< 1 hour)

- [ ] **Database corruption tested**
  - [ ] Corruption detected
  - [ ] Backup restored
  - [ ] RTO met (< 30 minutes)

- [ ] **Data recovery tested**
  - [ ] PITR tested
  - [ ] Data recovered to point in time
  - [ ] RPO met (< 5 minutes)

- [ ] **Team trained on procedures**
  - [ ] Runbooks reviewed
  - [ ] Team practiced procedures
  - [ ] Improvements documented

---

## Runbooks

### Runbook: PostgreSQL Backup Failure

**Severity**: CRITICAL

**Symptoms**:
- Backup job fails
- No backup file created
- Alert: "PostgreSQL backup failed"

**Diagnosis**:

```bash
# Check backup logs
tail -f /var/log/pgbackrest.log

# Check disk space
df -h /backup

# Check PostgreSQL status
systemctl status postgresql
```

**Resolution**:

1. **Disk space issue**:
   ```bash
   # Clean old backups
   pgbackrest --stanza=bubblelab --type=full --retention-full=2 expire

   # Or add more disk space
   ```

2. **PostgreSQL down**:
   ```bash
   # Restart PostgreSQL
   systemctl restart postgresql
   ```

3. **Permission issue**:
   ```bash
   # Fix permissions
   chown -R postgres:postgres /backup
   chmod 700 /backup
   ```

**Prevention**:
- Monitor disk space
- Set up alerts
- Regular restore tests

### Runbook: Qdrant Snapshot Failure

**Severity**: HIGH

**Symptoms**:
- Snapshot creation fails
- Error: "Failed to create snapshot"

**Diagnosis**:

```bash
# Check Qdrant logs
docker logs qdrant

# Check disk space
df -h /backup

# Check collection status
curl http://localhost:6333/collections/{collection_name}
```

**Resolution**:

1. **Disk space issue**:
   ```bash
   # Clean old snapshots
   curl -X DELETE "http://localhost:6333/collections/{collection_name}/snapshots/{old_snapshot}"
   ```

2. **Collection locked**:
   ```bash
   # Wait for ongoing operations to complete
   # Or restart Qdrant
   docker restart qdrant
   ```

**Prevention**:
- Monitor disk space
- Clean old snapshots regularly
- Set up alerts

### Runbook: Elasticsearch Snapshot Failure

**Severity**: HIGH

**Symptoms**:
- Snapshot creation hangs or fails
- Error: "Snapshot failed"

**Diagnosis**:

```bash
# Check snapshot status
curl -X GET "localhost:9200/_snapshot/backup_repo/_all"

# Check Elasticsearch logs
tail -f /var/log/elasticsearch/elasticsearch.log

# Check disk space
df -h /backup
```

**Resolution**:

1. **Snapshot in progress**:
   ```bash
   # Wait for completion
   # Or cancel and retry
   curl -X DELETE "localhost:9200/_snapshot/backup_repo/{snapshot_name}"
   ```

2. **Repository not registered**:
   ```bash
   # Re-register repository
   curl -X PUT "localhost:9200/_snapshot/backup_repo" -H 'Content-Type: application/json' -d'
   {
     "type": "fs",
     "settings": {
       "location": "/backup/elasticsearch"
     }
   }
   '
   ```

**Prevention**:
- Monitor snapshot status
- Set up alerts
- Clean old snapshots

---

## Metrics and Monitoring

### Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Backup success rate | 100% | < 100% |
| Backup age (latest) | < 5 minutes | > 10 minutes |
| Backup restore time | < 30 minutes | > 1 hour |
| Disk space (backup) | < 80% | > 90% |
| RTO compliance | < 1 hour | > 1 hour |
| RPO compliance | < 5 minutes | > 5 minutes |

### Monitoring Dashboard

Create Grafana dashboard showing:
1. Backup job status
2. Backup file sizes
3. Backup age
4. Disk space usage
5. Restore test results
6. RTO/RPO compliance

### Alerts

Configure alerts for:
- Backup job failure
- Backup too old (> 10 minutes)
- Disk space low (> 90%)
- Restore test failure
- RTO/RPO breach

---

## Documentation

**RTO**: Recovery Time Objective - Time to restore service
**RPO**: Recovery Point Objective - Maximum acceptable data loss

**Current RTO**: 1 hour
**Current RPO**: 5 minutes

**Target RTO**: 30 minutes
**Target RPO**: 1 minute

**Gap**: Need to implement streaming replication for PostgreSQL

---

**Last Updated**: 2026-01-18
**Next Review**: 2026-02-18
**Approved By**: _______________

---

## Appendix: Scripts Directory

Place these scripts in `/bubblelab/scripts/backup-restore/`:

- `backup-postgresql.sh`
- `backup-qdrant.sh`
- `backup-elasticsearch.sh`
- `backup-redis.sh`
- `backup-config.sh`
- `restore-postgresql.sh`
- `restore-qdrant.sh`
- `restore-elasticsearch.sh`
- `restore-redis.sh`
- `restore-config.sh`
- `test-backup.sh` - Runs all backup tests
- `test-restore.sh` - Runs all restore tests
