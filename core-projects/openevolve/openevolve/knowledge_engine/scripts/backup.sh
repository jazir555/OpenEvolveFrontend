#!/bin/bash

# Neo4j Backup Script
# OpenEvolve Knowledge Engine - Phase 1.1.1
#
# This script creates online backups of the Neo4j database
# Usage: ./backup.sh [backup_name]
#
# Requirements:
# - Neo4j must be running
# - Sufficient disk space for backup
# - Write permissions to backup directory

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration from environment variables
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-openevolve2026}"
NEO4J_BACKUP_DIR="${NEO4J_BACKUP_DIR:-/var/lib/neo4j/backups}"
BACKUP_NAME="${1:-neo4j-backup-$(date +%Y%m%d-%H%M%S)}"

echo "================================================"
echo "Neo4j Backup Script - OpenEvolve Knowledge Engine"
echo "================================================"
echo ""

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "INFO" ]; then
        echo -e "${BLUE}ℹ${NC} $message"
    elif [ "$status" = "WARNING" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

# ============================================================================
# PRE-BACKUP CHECKS
# ============================================================================

print_status "INFO" "Starting pre-backup checks..."

# Check if Neo4j is running
print_status "INFO" "Checking if Neo4j is running..."
if ! echo "RETURN 1" | cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" >/dev/null 2>&1; then
    print_status "FAIL" "Cannot connect to Neo4j. Is it running?"
    exit 1
fi
print_status "OK" "Neo4j is running"

# Check if backup directory exists
print_status "INFO" "Checking backup directory..."
if [ ! -d "$NEO4J_BACKUP_DIR" ]; then
    print_status "INFO" "Creating backup directory: $NEO4J_BACKUP_DIR"
    mkdir -p "$NEO4J_BACKUP_DIR"
fi
print_status "OK" "Backup directory is ready"

# Check available disk space
print_status "INFO" "Checking available disk space..."
AVAILABLE_SPACE=$(df -BG "$NEO4J_BACKUP_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
print_status "OK" "Available disk space: ${AVAILABLE_SPACE}GB"

# Estimate database size
print_status "INFO" "Estimating database size..."
DB_SIZE=$(du -sh /data 2>/dev/null | awk '{print $1}' || echo "unknown")
print_status "OK" "Database size: $DB_SIZE"

echo ""

# ============================================================================
# CREATE BACKUP
# ============================================================================

print_status "INFO" "Starting backup process..."
BACKUP_PATH="$NEO4J_BACKUP_DIR/$BACKUP_NAME"

# Create backup using neo4j-admin dump
print_status "INFO" "Using neo4j-admin dump for backup..."

# Note: This requires access to the Neo4j data directory
# For containerized deployments, you may need to use cypher-shell export instead

# Method 1: neo4j-admin dump (preferred for full backups)
if command -v neo4j-admin >/dev/null 2>&1; then
    print_status "INFO" "Using neo4j-admin dump..."
    neo4j-admin dump \
        --database=neo4j \
        --to="$BACKUP_NAME.dump" \
        --force

    if [ $? -eq 0 ]; then
        print_status "OK" "Backup created successfully: $BACKUP_NAME.dump"
    else
        print_status "FAIL" "Backup failed using neo4j-admin"
        exit 1
    fi

# Method 2: Cypher-based export (alternative for containerized deployments)
else
    print_status "INFO" "Using Cypher export (neo4j-admin not available)..."

    # Export all data as Cypher script
    EXPORT_SCRIPT="$BACKUP_PATH.cypher"

    echo "" > "$EXPORT_SCRIPT"
    echo "// Neo4j Backup Export" >> "$EXPORT_SCRIPT"
    echo "// Date: $(date)" >> "$EXPORT_SCRIPT"
    echo "// Database: $NEO4J_URI" >> "$EXPORT_SCRIPT"
    echo "" >> "$EXPORT_SCRIPT"

    # Export nodes
    echo "BEGIN;" >> "$EXPORT_SCRIPT"
    echo "// Exporting nodes..." >> "$EXPORT_SCRIPT"
    cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
        "CALL apoc.export.cypher.all('$EXPORT_SCRIPT', {batchSize: 1000})" \
        >> "$EXPORT_SCRIPT" 2>&1

    if [ $? -eq 0 ]; then
        print_status "OK" "Cypher export created: $EXPORT_SCRIPT"
    else
        print_status "FAIL" "Cypher export failed"
        exit 1
    fi
fi

echo ""

# ============================================================================
# BACKUP VERIFICATION
# ============================================================================

print_status "INFO" "Verifying backup..."

# Check if backup file exists
if [ -f "$BACKUP_NAME.dump" ] || [ -f "$EXPORT_SCRIPT" ]; then
    print_status "OK" "Backup file exists"
else
    print_status "FAIL" "Backup file not found"
    exit 1
fi

# Check backup file size
BACKUP_SIZE=$(du -sh "$BACKUP_NAME.dump" 2>/dev/null || du -sh "$EXPORT_SCRIPT" | awk '{print $1}')
print_status "OK" "Backup size: $BACKUP_SIZE"

echo ""

# ============================================================================
# BACKUP METADATA
# ============================================================================

print_status "INFO" "Creating backup metadata..."

METADATA_FILE="$BACKUP_PATH-metadata.txt"

cat > "$METADATA_FILE" << EOF
Neo4j Backup Metadata
=====================

Backup Name: $BACKUP_NAME
Date: $(date)
Timestamp: $(date +%s)

Neo4j URI: $NEO4J_URI
Database: neo4j

Backup Method: cypher-export
Backup Size: $BACKUP_SIZE

System Information:
- Hostname: $(hostname)
- OS: $(uname -s)
- Architecture: $(uname -m)

Database Statistics:
EOF

# Get database statistics
echo "" >> "$METADATA_FILE"
echo "CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Primitive count') YIELD attributes" | \
    cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" >> "$METADATA_FILE" 2>&1

print_status "OK" "Backup metadata created: $METADATA_FILE"

echo ""

# ============================================================================
# CLEANUP OLD BACKUPS
# ============================================================================

print_status "INFO" "Cleaning up old backups..."

# Keep last 7 days of backups (configurable)
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
find "$NEO4J_BACKUP_DIR" -name "neo4j-backup-*" -type f -mtime +$RETENTION_DAYS -delete

DELETED_COUNT=$(find "$NEO4J_BACKUP_DIR" -name "neo4j-backup-*" -type f -mtime +$RETENTION_DAYS 2>/dev/null | wc -l)
print_status "OK" "Cleaned up $DELETED_COUNT old backups (older than $RETENTION_DAYS days)"

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

print_status "OK" "Backup completed successfully!"
echo ""
echo "Backup Details:"
echo "  - Location: $BACKUP_PATH"
echo "  - Size: $BACKUP_SIZE"
echo "  - Retention: $RETENTION_DAYS days"
echo ""

# List recent backups
print_status "INFO" "Recent backups:"
ls -lht "$NEO4J_BACKUP_DIR" | head -n 6

echo ""
echo "================================================"
echo "Backup completed at: $(date)"
echo "================================================"

exit 0
