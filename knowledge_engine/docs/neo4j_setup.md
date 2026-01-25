# Neo4j Setup and Configuration Guide

**OpenEvolve Knowledge Engine - Phase 1.1.1**

This guide provides comprehensive instructions for deploying and configuring Neo4j 5.26+ for the OpenEvolve Knowledge Engine, including vector indices, temporal capabilities, and production-ready settings.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [Database Initialization](#database-initialization)
6. [Health Monitoring](#health-monitoring)
7. [Backup and Recovery](#backup-and-recovery)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tuning](#performance-tuning)
10. [Security Best Practices](#security-best-practices)

---

## Prerequisites

### System Requirements

**Development Environment:**
- CPU: 2 cores minimum
- RAM: 8GB minimum (4GB heap + 2GB page cache + 2GB OS)
- Disk: 20GB free space
- OS: Linux, macOS, or Windows with WSL2

**Production Environment:**
- CPU: 4+ cores recommended
- RAM: 32GB+ recommended (16GB heap + 14GB page cache + 2GB OS)
- Disk: SSD with 100GB+ free space
- OS: Linux (Ubuntu 20.04+, RHEL 8+)

### Software Requirements

- Docker Engine 20.10+ (for containerized deployment)
- Docker Compose 2.0+
- Python 3.9+ (for client applications)
- cypher-shell (Neo4j command-line tool)

### Verify Prerequisites

```bash
# Check Docker version
docker --version
docker-compose --version

# Check Python version
python --version

# Check cypher-shell (optional)
cypher-shell --version
```

---

## Quick Start

### 1. Clone Repository

```bash
cd /path/to/openevolve-frontend
```

### 2. Start Neo4j

```bash
# Development environment
docker-compose -f docker-compose.neo4j.yml --env-file knowledge_engine/config/neo4j.dev.env up -d

# Production environment
docker-compose -f docker-compose.neo4j.yml --env-file knowledge_engine/config/neo4j.prod.env up -d
```

### 3. Verify Installation

```bash
# Run health check
bash knowledge_engine/scripts/health_check.sh

# Check container status
docker ps | grep neo4j
```

### 4. Initialize Database

```bash
# Run initialization script
cypher-shell -u neo4j -p openevolve2026 < knowledge_engine/scripts/init_neo4j.cypher

# Or using Docker exec
docker exec -it openevolve-neo4j cypher-shell -u neo4j -p openevolve2026 -f /scripts/init_neo4j.cypher
```

### 5. Access Neo4j Browser

Open your browser and navigate to:
```
http://localhost:7474
```

**Default credentials:**
- Username: `neo4j`
- Password: `openevolve2026`

---

## Installation Methods

### Method 1: Docker Compose (Recommended)

**Advantages:**
- Isolated environment
- Easy version management
- Simple upgrades and rollbacks
- Consistent across platforms

**Steps:**

1. **Configure environment variables**

```bash
# For development
cp knowledge_engine/config/neo4j.dev.env knowledge_engine/config/neo4j.local.env
# Edit knowledge_engine/config/neo4j.local.env with your settings

# For production
cp knowledge_engine/config/neo4j.prod.env knowledge_engine/config/neo4j.local.env
# Edit knowledge_engine/config/neo4j.local.env with your settings
```

2. **Create data directories**

```bash
mkdir -p data/neo4j/{data,logs,import,plugins,backups}
chmod -R 777 data/neo4j
```

3. **Start Neo4j**

```bash
docker-compose -f docker-compose.neo4j.yml --env-file knowledge_engine/config/neo4j.local.env up -d
```

4. **View logs**

```bash
docker-compose -f docker-compose.neo4j.yml logs -f neo4j
```

### Method 2: Native Installation

**For Linux:**

```bash
# Import Neo4j GPG key
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -

# Add Neo4j repository
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list

# Install Neo4j
sudo apt-get update
sudo apt-get install -y neo4j

# Configure Neo4j
sudo nano /etc/neo4j/neo4j.conf
```

**For macOS:**

```bash
# Using Homebrew
brew install neo4j

# Start Neo4j
neo4j start
```

---

## Configuration

### Environment Variables

**Development Configuration** (`knowledge_engine/config/neo4j.dev.env`)

```bash
# Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=openevolve2026

# Memory (Development)
NEO4J_server_memory_heap_initial__size=1g
NEO4J_server_memory_heap_max__size=2g
NEO4J_server_memory_pagecache_size=2g

# Logging
NEO4J_LOG_LEVEL=DEBUG
```

**Production Configuration** (`knowledge_engine/config/neo4j.prod.env`)

```bash
# Connection
NEO4J_URI=bolt://openevolve-neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<CHANGE_THIS_PASSWORD>

# Memory (Production)
NEO4J_server_memory_heap_initial__size=8g
NEO4J_server_memory_heap_max__size=16g
NEO4J_server_memory_pagecache_size=14g

# Logging
NEO4J_LOG_LEVEL=INFO

# Security
NEO4J_dbms_security_auth_enabled=true
NEO4J_dbms_connector_bolt_tls__level=REQUIRED
```

### Memory Configuration

**Formula for optimal memory allocation:**

```
Total RAM = Heap + Page Cache + OS Overhead (2GB)

For 32GB RAM:
- Heap: 16GB (50%)
- Page Cache: 14GB (44%)
- OS: 2GB (6%)
```

**Key settings:**

```bash
# Heap size (Java heap for Neo4j)
NEO4J_server_memory_heap_initial__size=<size>
NEO4J_server_memory_heap_max__size=<size>

# Page cache (for frequently accessed data)
NEO4J_server_memory_pagecache_size=<size>
```

### Connection Pooling

**Python client configuration:**

```python
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(
    uri,
    auth=("neo4j", "openevolve2026"),
    max_connection_lifetime=3600,
    max_connection_pool_size=50,
    connection_acquisition_timeout=60
)
```

---

## Database Initialization

### Automated Initialization

```bash
# Using Docker exec
docker exec -it openevolve-neo4j cypher-shell -u neo4j -p openevolve2026 -f /scripts/init_neo4j.cypher

# Using cypher-shell directly
cypher-shell -u neo4j -p openevolve2026 < knowledge_engine/scripts/init_neo4j.cypher
```

### Manual Initialization Steps

**1. Create Constraints**

```cypher
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;
```

**2. Create Vector Indices**

```cypher
CALL db.index.vector.createNodeIndex(
    'entity_embeddings',
    'Entity',
    'embedding',
    {
        dimension: 1536,
        similarityFunction: 'cosine'
    }
);
```

**3. Verify Setup**

```cypher
CALL db.constraints();
CALL db.indexes();
```

### Verify Initialization

```bash
# Run health check
bash knowledge_engine/scripts/health_check.sh

# Check constraints and indexes
cypher-shell -u neo4j -p openevolve2026 "CALL db.constraints()"
cypher-shell -u neo4j -p openevolve2026 "CALL db.indexes()"
```

---

## Health Monitoring

### Health Check Script

```bash
# Run comprehensive health check
bash knowledge_engine/scripts/health_check.sh
```

**Health check includes:**
- HTTP endpoint availability
- Bolt protocol connection
- Database version verification
- Constraint and index verification
- APOC plugin availability
- Memory usage monitoring

### Manual Health Checks

**1. Check Container Status**

```bash
docker ps | grep neo4j
docker logs openevolve-neo4j --tail 50
```

**2. Check Database Connectivity**

```bash
cypher-shell -u neo4j -p openevolve2026 "RETURN 1"
```

**3. Check Database Statistics**

```bash
cypher-shell -u neo4j -p openevolve2026 "CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Primitive count')"
```

**4. Monitor Memory Usage**

```bash
docker stats openevolve-neo4j
```

### Prometheus Metrics

Neo4j exposes metrics on port 2004 (if enabled):

```bash
# Access metrics
curl http://localhost:2004/metrics
```

---

## Backup and Recovery

### Automated Backup

```bash
# Run backup script
bash knowledge_engine/scripts/backup.sh

# Scheduled backup (cron)
0 2 * * * /path/to/knowledge_engine/scripts/backup.sh
```

### Manual Backup

**Using neo4j-admin:**

```bash
# Dump database
neo4j-admin dump --database=neo4j --to=backup.dump

# Restore database
neo4j-admin load --from=backup.dump --database=neo4j --force
```

**Using Cypher export:**

```bash
# Export all data
cypher-shell -u neo4j -p openevolve2026 "CALL apoc.export.cypher.all('backup.cypher', {})"

# Import data
cypher-shell -u neo4j -p openevolve2026 < backup.cypher
```

### Backup Strategy

**Development:**
- Daily backups
- Retain 7 days
- Store locally

**Production:**
- Hourly incremental backups
- Daily full backups
- Retain 30 days
- Store offsite (S3, GCS, Azure Blob)

---

## Troubleshooting

### Common Issues

#### Issue 1: Container Won't Start

**Symptoms:**
- Container exits immediately
- Logs show "Unable to lock JVM memory"

**Solution:**

```bash
# Check logs
docker logs openevolve-neo4j

# Reduce memory allocation in docker-compose.neo4j.yml
# Or increase system swap space
```

#### Issue 2: Connection Refused

**Symptoms:**
- "Connection refused" error
- Cannot connect to Bolt port

**Solution:**

```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Check port availability
netstat -tuln | grep 7687

# Verify firewall rules
sudo ufw allow 7687/tcp
```

#### Issue 3: Out of Memory

**Symptoms:**
- Container crashes
- "Java heap space" error

**Solution:**

```bash
# Increase heap size in docker-compose.neo4j.yml
NEO4J_server_memory_heap_max__size=4g

# Or reduce concurrent queries
NEO4J_dbms_transaction_concurrent_maximum=100
```

#### Issue 4: Vector Index Creation Fails

**Symptoms:**
- "Unknown procedure" error
- Vector index not created

**Solution:**

```bash
# Verify GDS library is installed
docker exec -it openevolve-neo4j ls -la /plugins/

# Install GDS library
docker exec -it openevolve-neo4j /var/lib/neo4j/bin/neo4j-admin server install gds --from http://... --to /plugins/

# Restart Neo4j
docker restart openevolve-neo4j
```

#### Issue 5: Slow Queries

**Symptoms:**
- Queries timeout
- High latency

**Solution:**

```bash
# Enable query logging
NEO4J_server_logs_debug_level=INFO

# Analyze query plan
cypher-shell -u neo4j -p openevolve2026 "EXPLAIN MATCH (n) RETURN n"

# Create appropriate indexes
CREATE INDEX idx_name FOR (n:Entity) ON (n.name);
```

### Diagnostic Commands

```bash
# Check Neo4j logs
docker logs openevolve-neo4j --tail 100

# Check system resources
docker stats openevolve-neo4j

# Check database locks
cypher-shell -u neo4j -p openevolve2026 "SHOW TRANSACTIONS"

# Check active queries
cypher-shell -u neo4j -p openevolve2026 "SHOW QUERIES"
```

---

## Performance Tuning

### Memory Optimization

**For small datasets (< 1M nodes):**
```bash
Heap: 2GB
Page Cache: 2GB
```

**For medium datasets (1-10M nodes):**
```bash
Heap: 4GB
Page Cache: 6GB
```

**For large datasets (> 10M nodes):**
```bash
Heap: 16GB
Page Cache: 14GB
```

### Query Optimization

**1. Use indexes effectively**

```cypher
# Bad - full table scan
MATCH (e:Entity) WHERE e.name = 'Concept' RETURN e;

# Good - uses index
CREATE INDEX idx_entity_name FOR (e:Entity) ON (e.name);
MATCH (e:Entity) WHERE e.name = 'Concept' RETURN e;
```

**2. Use parameterized queries**

```python
# Bad - query compilation overhead
query = f"MATCH (e:Entity {{name: '{name}'}}) RETURN e"

# Good - query plan cached
query = "MATCH (e:Entity {name: $name}) RETURN e"
result = session.run(query, name="Concept")
```

**3. Batch operations**

```python
# Bad - one transaction per node
for node in nodes:
    session.run("CREATE (e:Entity {id: $id})", id=node.id)

# Good - batch operations
with session.begin_transaction() as tx:
    for node in nodes:
        tx.run("CREATE (e:Entity {id: $id})", id=node.id)
    tx.commit()
```

### Connection Pool Tuning

```python
# For read-heavy workloads
driver = GraphDatabase.driver(
    uri,
    auth=(user, password),
    max_connection_pool_size=100,
    connection_acquisition_timeout=30
)

# For write-heavy workloads
driver = GraphDatabase.driver(
    uri,
    auth=(user, password),
    max_connection_pool_size=50,
    max_transaction_retry_time=10
)
```

---

## Security Best Practices

### 1. Change Default Password

```bash
# Change password immediately after first login
cypher-shell -u neo4j -p openevolve2026
> CALL dbms.security.changePassword('new_secure_password');
```

### 2. Enable SSL/TLS

```bash
# In neo4j.conf
dbms.connector.bolt.tls_level=REQUIRED
dbms.connector.https.enabled=true
```

### 3. Restrict Network Access

```bash
# Use firewall rules
sudo ufw allow from 10.0.0.0/8 to any port 7687
sudo ufw deny 7687
```

### 4. Use Environment Variables for Secrets

```bash
# Never hardcode passwords
export NEO4J_PASSWORD=$(openssl rand -base64 32)
docker-compose up -d
```

### 5. Enable Audit Logging

```bash
# In neo4j.conf
dbms.security.log_successful_authentication=true
dbms.security.procedures.unrestricted=apoc.*
```

### 6. Regular Security Updates

```bash
# Update Neo4j regularly
docker-compose pull neo4j
docker-compose up -d
```

---

## Additional Resources

- **Neo4j Documentation**: https://neo4j.com/docs/
- **Cypher Query Language**: https://neo4j.com/docs/cypher-manual/
- **Neo4j Driver Documentation**: https://neo4j.com/docs/python-manual/
- **APOC Library**: https://neo4j.com/docs/labs/apoc/current/
- **GDS Library**: https://neo4j.com/docs/graph-data-science/current/

---

## Support

For issues or questions:
- Check the troubleshooting section above
- Review Neo4j logs: `docker logs openevolve-neo4j`
- Run health check: `bash knowledge_engine/scripts/health_check.sh`
- Create an issue in the project repository

---

**Version:** 1.0.0
**Last Updated:** 2026-01-07
**Maintainer:** OpenEvolve Team
