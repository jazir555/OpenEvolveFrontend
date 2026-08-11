<<<<<<< HEAD
# OpenEvolve Infrastructure Setup Guide

**Phase 0 Foundation - crewai → Vibe-Kanban Migration**

This guide covers setting up the core infrastructure services required for OpenEvolve operations, including the crewai task management system and Vibe-Kanban integration.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Service Details](#service-details)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)
- [Security Considerations](#security-considerations)

---

## Overview

The OpenEvolve infrastructure consists of three core services:

| Service | Purpose | Port |
|---------|---------|------|
| **Qdrant** | Vector database for embeddings and semantic search | 6333 (HTTP), 6334 (gRPC) |
| **PostgreSQL** | Relational database for structured data | 5432 |
| **Redis** | Caching and message broker | 6379 |

Optional management tools:
- **pgAdmin** - PostgreSQL web interface (port 5050)
- **Redis Commander** - Redis web interface (port 8081)

---

## Prerequisites

### Required Software

1. **Docker Desktop** (recommended) or Docker Engine
   - Download: https://www.docker.com/products/docker-desktop/
   - Version: 20.10 or later
   - Ensure Docker has at least 4GB RAM allocated

2. **Docker Compose**
   - Included with Docker Desktop
   - Or install standalone: https://docs.docker.com/compose/install/

3. **Git** (for cloning the repository)
   - Download: https://git-scm.com/downloads

### Verify Installation

```bash
# Check Docker
docker --version
docker compose version

# Check Docker is running
docker ps
```

---

## Quick Start

### 1. Configure Environment

Create the environment file from the template:

```bash
cp .env.infrastructure.example .env.infrastructure
```

**IMPORTANT:** Edit `.env.infrastructure` and change the default passwords:

```bash
# Change these values!
POSTGRES_PASSWORD=your-secure-password-here
PGADMIN_EMAIL=your-email@example.com
PGADMIN_PASSWORD=your-pgadmin-password-here
```

### 2. Start Services

**Linux/macOS:**

```bash
chmod +x scripts/dev-start.sh
./scripts/dev-start.sh
```

**Windows:**

```cmd
scripts\dev-start.bat
```

### 3. Verify Services are Running

Open a browser and test:

- Qdrant Dashboard: http://localhost:6333/dashboard
- PostgreSQL: Connection string `postgresql://openevolve:your-password@localhost:5432/openevolve`
- Redis: `redis://localhost:6379`

### 4. Stop Services

**Linux/macOS:**

```bash
./scripts/dev-stop.sh
```

**Windows:**

```cmd
scripts\dev-stop.bat
```

---

## Service Details

### Qdrant Vector Database

**Purpose:** Stores vector embeddings for semantic search, knowledge graph operations, and crewai task memory.

**Key Features:**
- High-performance vector similarity search
- Support for filtering and hybrid queries
- RESTful API and gRPC
- Built-in dashboard for management

**API Endpoints:**
- HTTP API: http://localhost:6333
- Dashboard: http://localhost:6333/dashboard
- gRPC: localhost:6334

**Common Operations:**

```bash
# Check health
curl http://localhost:6333/health

# List collections
curl http://localhost:6333/collections

# Get collection info
curl http://localhost:6333/collections/your_collection_name
```

**Environment Variables:**
```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional, for authentication
QDRANT_COLLECTION_PREFIX=openevolve  # Prefix for collection names
```

### PostgreSQL Database

**Purpose:** Relational database for user management, tasks, projects, and structured application data.

**Schema Includes:**
- `users` - User accounts and authentication
- `projects` - Project metadata
- `tasks` - crewai task management
- Indexes and triggers for performance

**Connection Information:**
```
Host: localhost
Port: 5432
Database: openevolve
User: openevolve
Password: (from .env.infrastructure)
```

**Connection String Format:**
```
postgresql://openevolve:password@localhost:5432/openevolve
```

**Environment Variables:**
```bash
DATABASE_URL=postgresql://openevolve:password@localhost:5432/openevolve
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=openevolve
POSTGRES_USER=openevolve
POSTGRES_PASSWORD=your-password
```

**Common Operations:**

```bash
# Connect using psql (if installed)
psql -h localhost -U openevolve -d openevolve

# Or use Docker exec
docker exec -it openevolve-postgres psql -U openevolve -d openevolve

# Backup database
docker exec openevolve-postgres pg_dump -U openevolve openevolve > backup.sql

# Restore database
docker exec -i openevolve-postgres psql -U openevolve openevolve < backup.sql
```

### Redis Cache

**Purpose:** Caching layer for LLM responses, session storage, and message queuing.

**Configuration:**
- Max memory: 512MB
- Eviction policy: allkeys-lru (least recently used)
- Persistence: AOF (Append Only File) enabled

**Connection Information:**
```
Host: localhost
Port: 6379
URL: redis://localhost:6379
```

**Environment Variables:**
```bash
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379
```

**Common Operations:**

```bash
# Connect using redis-cli (if installed)
redis-cli -h localhost -p 6379

# Or use Docker exec
docker exec -it openevolve-redis redis-cli

# Test connection
docker exec openevolve-redis redis-cli ping
# Should return: PONG

# Monitor commands in real-time
docker exec openevolve-redis redis-cli MONITOR

# Flush all cache (WARNING: deletes all data)
docker exec openevolve-redis redis-cli FLUSHALL
```

---

## Configuration

### Environment Files

1. **`.env.infrastructure`** - Infrastructure services only
   - Minimal configuration for Docker Compose
   - Database passwords and credentials

2. **`.env.development`** - Full development configuration
   - All service URLs
   - API keys for LLM providers
   - Application settings

3. **`.env.production`** - Production configuration
   - Override defaults for production deployment
   - External service endpoints
   - Security settings

### Docker Compose Profiles

The `docker-compose.infrastructure.yml` file supports profiles:

```bash
# Start core services only
docker-compose -f docker-compose.infrastructure.yml up -d

# Start with management tools
docker-compose -f docker-compose.infrastructure.yml --profile tools up -d
```

**Available Profiles:**
- `tools` - Includes pgAdmin and Redis Commander

### Resource Limits

Default resource allocations (can be adjusted in docker-compose):

| Service | Memory Limit | CPU Limit |
|---------|--------------|-----------|
| Qdrant | 1GB | 1.0 |
| PostgreSQL | 512MB | 0.5 |
| Redis | 256MB | 0.25 |

To modify, edit `docker-compose.infrastructure.yml`:

```yaml
services:
  qdrant:
    deploy:
      resources:
        limits:
          memory: 2G  # Adjust as needed
          cpus: '2.0'
```

---

## Verification

### Health Check Script

```bash
#!/bin/bash
# scripts/verify-infrastructure.sh

echo "Checking infrastructure services..."

# Check PostgreSQL
echo -n "PostgreSQL... "
if pg_isready -h localhost -p 5432 -U openevolve > /dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
fi

# Check Qdrant
echo -n "Qdrant... "
if curl -sf http://localhost:6333/health > /dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
fi

# Check Redis
echo -n "Redis... "
if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
fi
```

### Service Status

```bash
# View all containers
docker ps -a

# View logs for specific service
docker logs openevolve-postgres
docker logs openevolve-qdrant
docker logs openevolve-redis

# Follow logs in real-time
docker logs -f openevolve-postgres
```

### Connection Testing

**PostgreSQL:**

```bash
docker exec -it openevolve-postgres psql -U openevolve -d openevolve -c "SELECT version();"
```

**Qdrant:**

```bash
curl http://localhost:6333/
curl http://localhost:6333/collections
```

**Redis:**

```bash
docker exec openevolve-redis redis-cli INFO
```

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

**Error:** `Bind for 0.0.0.0:5432 failed: port is already allocated`

**Solution:**
```bash
# Find what's using the port
# Linux/macOS:
lsof -i :5432

# Windows:
netstat -ano | findstr :5432

# Either stop the conflicting service or change the port in docker-compose.yml
```

#### 2. Container Won't Start

**Error:** Container exits immediately

**Solution:**
```bash
# Check logs
docker logs openevolve-postgres

# Common causes:
# - Wrong environment variables (check .env.infrastructure)
# - Permission issues on volume mounts
# - Insufficient resources (check Docker RAM allocation)
```

#### 3. Cannot Connect to Database

**Error:** `connection refused` or `authentication failed`

**Solution:**
```bash
# 1. Verify container is running
docker ps | grep postgres

# 2. Check environment variables
cat .env.infrastructure | grep POSTGRES

# 3. Test connection from within container
docker exec -it openevolve-postgres psql -U openevolve -d openevolve

# 4. If authentication failed, reset password
docker exec -it openevolve-postgres psql -U postgres -c "ALTER USER openevolve PASSWORD 'newpassword';"
```

#### 4. Qdrant Shows No Collections

**Note:** This is normal for a fresh installation. Collections are created automatically by applications when needed.

#### 5. Docker Out of Memory

**Error:** `no space left on device` or containers being killed

**Solution:**
```bash
# Clean up unused resources
docker system prune -a

# Increase Docker memory allocation in Docker Desktop settings
# Recommended: 8GB or more

# Check current usage
docker system df
```

### Getting Help

1. Check logs: `docker logs <container-name>`
2. Check service status: `docker ps -a`
3. Review configuration: `cat .env.infrastructure`
4. Check Docker resources: Docker Desktop → Settings → Resources

---

## Maintenance

### Backup Procedures

#### PostgreSQL Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="./backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

docker exec openevolve-postgres pg_dump -U openevolve openevolve > \
    $BACKUP_DIR/openevolve_$TIMESTAMP.sql

# Keep last 7 days
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
```

#### Qdrant Backup

```bash
# Qdrant stores data in Docker volume
# Backup the entire volume
docker run --rm \
    -v openevolve-qdrant-data:/data \
    -v $(pwd)/backups:/backup \
    alpine tar czf /backup/qdrant_backup_$(date +%Y%m%d).tar.gz -C /data .
```

#### Redis Backup

Redis persistence is enabled (AOF mode). Backup the Redis data volume:

```bash
docker run --rm \
    -v openevolve-redis-data:/data \
    -v $(pwd)/backups:/backup \
    alpine tar czf /backup/redis_backup_$(date +%Y%m%d).tar.gz -C /data .
```

### Updates

#### Updating Docker Images

```bash
# Pull latest images
docker-compose -f docker-compose.infrastructure.yml pull

# Recreate containers with new images
docker-compose -f docker-compose.infrastructure.yml up -d

# Or use the startup script
./scripts/dev-stop.sh
./scripts/dev-start.sh
```

#### Database Migrations

For schema changes:

```bash
# Run migrations (if using a migration tool)
# Example with Alembic (Python):
alembic upgrade head

# Example with SQLx (Rust):
sqlx database create
sqlx migrate run
```

### Monitoring

#### Resource Usage

```bash
# Container resource usage
docker stats

# Specific container
docker stats openevolve-postgres openevolve-qdrant openevolve-redis
```

#### Disk Usage

```bash
# Docker disk usage
docker system df

# Volume details
docker volume ls
docker volume inspect openevolve-qdrant-data
```

#### Log Management

```bash
# View logs
docker logs -f --tail 100 openevolve-postgres

# Log rotation is handled by Docker, but you can manually clean
docker system prune -a
```

---

## Security Considerations

### Development vs Production

**Development (current setup):**
- Default passwords (acceptable for local dev)
- No encryption on connections
- Management tools exposed

**Production Requirements:**
- Strong, unique passwords
- SSL/TLS for all connections
- Restrict management tool access
- Use secrets management (Vault, AWS Secrets Manager, etc.)
- Regular security updates
- Network isolation
- Rate limiting
- Authentication for all services

### Hardening Checklist

- [ ] Change all default passwords
- [ ] Enable SSL/TLS for PostgreSQL
- [ ] Restrict Redis to localhost only
- [ ] Enable Qdrant authentication
- [ ] Use Docker secrets instead of environment variables
- [ ] Regular security updates
- [ ] Network segmentation
- [ ] Firewall rules
- [ ] Backup encryption
- [ ] Access logging and monitoring

### Environment Variables

**Never commit:**
- `.env.infrastructure` (or any `.env` files)
- `.env.development`
- `.env.production`
- Any files with real passwords or API keys

**Use version control for:**
- `.env.infrastructure.example`
- `.env.development.example`
- `.env.production.example`

---

## Advanced Usage

### Custom Configuration

#### PostgreSQL Extensions

Add extensions to `scripts/init-db.sql`:

```sql
CREATE EXTENSION IF NOT EXISTS "postgis";  -- Geographic data
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- Cryptographic functions
```

#### Redis Configuration

Modify Redis settings in `docker-compose.infrastructure.yml`:

```yaml
redis:
  command: [
    "redis-server",
    "--appendonly", "yes",
    "--maxmemory", "1gb",  # Increase from 512mb
    "--maxmemory-policy", "allkeys-lru"
  ]
```

#### Qdrant Configuration

Qdrant configuration via environment variables:

```yaml
qdrant:
  environment:
    - QDRANT__LOG_LEVEL=DEBUG
    - QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=32
```

### Networking

#### Custom Network

Services are on the `openevolve-network` bridge network. To connect external services:

```yaml
services:
  your-service:
    networks:
      - openevolve-network
    external_links:
      - openevolve-postgres
      - openevolve-qdrant
      - openevolve-redis

networks:
  openevolve-network:
    external: true
```

#### Service Discovery

Services can reach each other by container name:
- PostgreSQL: `openevolve-postgres:5432`
- Qdrant: `openevolve-qdrant:6333`
- Redis: `openevolve-redis:6379`

---

## Appendix

### File Structure

```
openevolve/
├── docker-compose.infrastructure.yml
├── scripts/
│   ├── init-db.sql
│   ├── dev-start.sh
│   ├── dev-start.bat
│   ├── dev-stop.sh
│   └── dev-stop.bat
├── docs/
│   └── INFRASTRUCTURE_SETUP.md
├── .env.infrastructure.example
└── .env.development
```

### Default Ports Reference

| Service | Internal Port | External Port | Purpose |
|---------|--------------|---------------|---------|
| PostgreSQL | 5432 | 5432 | Database |
| Qdrant HTTP | 6333 | 6333 | Vector DB API |
| Qdrant gRPC | 6334 | 6334 | Vector DB gRPC |
| Redis | 6379 | 6379 | Cache/MQ |
| pgAdmin | 80 | 5050 | DB Management UI |
| Redis Commander | 8081 | 8081 | Redis UI |

### Useful Links

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

### Support

For issues specific to:
- **OpenEvolve:** Check project documentation and GitHub issues
- **Docker:** Docker Desktop documentation
- **Services:** Official documentation for each service

---

**Last Updated:** 2026-01-11
**Version:** 1.0.0
**Phase:** Phase 0 Foundation
=======
# OpenEvolve Infrastructure Setup Guide

**Phase 0 Foundation - crewai → Vibe-Kanban Migration**

This guide covers setting up the core infrastructure services required for OpenEvolve operations, including the crewai task management system and Vibe-Kanban integration.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Service Details](#service-details)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)
- [Security Considerations](#security-considerations)

---

## Overview

The OpenEvolve infrastructure consists of three core services:

| Service | Purpose | Port |
|---------|---------|------|
| **Qdrant** | Vector database for embeddings and semantic search | 6333 (HTTP), 6334 (gRPC) |
| **PostgreSQL** | Relational database for structured data | 5432 |
| **Redis** | Caching and message broker | 6379 |

Optional management tools:
- **pgAdmin** - PostgreSQL web interface (port 5050)
- **Redis Commander** - Redis web interface (port 8081)

---

## Prerequisites

### Required Software

1. **Docker Desktop** (recommended) or Docker Engine
   - Download: https://www.docker.com/products/docker-desktop/
   - Version: 20.10 or later
   - Ensure Docker has at least 4GB RAM allocated

2. **Docker Compose**
   - Included with Docker Desktop
   - Or install standalone: https://docs.docker.com/compose/install/

3. **Git** (for cloning the repository)
   - Download: https://git-scm.com/downloads

### Verify Installation

```bash
# Check Docker
docker --version
docker compose version

# Check Docker is running
docker ps
```

---

## Quick Start

### 1. Configure Environment

Create the environment file from the template:

```bash
cp .env.infrastructure.example .env.infrastructure
```

**IMPORTANT:** Edit `.env.infrastructure` and change the default passwords:

```bash
# Change these values!
POSTGRES_PASSWORD=your-secure-password-here
PGADMIN_EMAIL=your-email@example.com
PGADMIN_PASSWORD=your-pgadmin-password-here
```

### 2. Start Services

**Linux/macOS:**

```bash
chmod +x scripts/dev-start.sh
./scripts/dev-start.sh
```

**Windows:**

```cmd
scripts\dev-start.bat
```

### 3. Verify Services are Running

Open a browser and test:

- Qdrant Dashboard: http://localhost:6333/dashboard
- PostgreSQL: Connection string `postgresql://openevolve:your-password@localhost:5432/openevolve`
- Redis: `redis://localhost:6379`

### 4. Stop Services

**Linux/macOS:**

```bash
./scripts/dev-stop.sh
```

**Windows:**

```cmd
scripts\dev-stop.bat
```

---

## Service Details

### Qdrant Vector Database

**Purpose:** Stores vector embeddings for semantic search, knowledge graph operations, and crewai task memory.

**Key Features:**
- High-performance vector similarity search
- Support for filtering and hybrid queries
- RESTful API and gRPC
- Built-in dashboard for management

**API Endpoints:**
- HTTP API: http://localhost:6333
- Dashboard: http://localhost:6333/dashboard
- gRPC: localhost:6334

**Common Operations:**

```bash
# Check health
curl http://localhost:6333/health

# List collections
curl http://localhost:6333/collections

# Get collection info
curl http://localhost:6333/collections/your_collection_name
```

**Environment Variables:**
```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional, for authentication
QDRANT_COLLECTION_PREFIX=openevolve  # Prefix for collection names
```

### PostgreSQL Database

**Purpose:** Relational database for user management, tasks, projects, and structured application data.

**Schema Includes:**
- `users` - User accounts and authentication
- `projects` - Project metadata
- `tasks` - crewai task management
- Indexes and triggers for performance

**Connection Information:**
```
Host: localhost
Port: 5432
Database: openevolve
User: openevolve
Password: (from .env.infrastructure)
```

**Connection String Format:**
```
postgresql://openevolve:password@localhost:5432/openevolve
```

**Environment Variables:**
```bash
DATABASE_URL=postgresql://openevolve:password@localhost:5432/openevolve
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=openevolve
POSTGRES_USER=openevolve
POSTGRES_PASSWORD=your-password
```

**Common Operations:**

```bash
# Connect using psql (if installed)
psql -h localhost -U openevolve -d openevolve

# Or use Docker exec
docker exec -it openevolve-postgres psql -U openevolve -d openevolve

# Backup database
docker exec openevolve-postgres pg_dump -U openevolve openevolve > backup.sql

# Restore database
docker exec -i openevolve-postgres psql -U openevolve openevolve < backup.sql
```

### Redis Cache

**Purpose:** Caching layer for LLM responses, session storage, and message queuing.

**Configuration:**
- Max memory: 512MB
- Eviction policy: allkeys-lru (least recently used)
- Persistence: AOF (Append Only File) enabled

**Connection Information:**
```
Host: localhost
Port: 6379
URL: redis://localhost:6379
```

**Environment Variables:**
```bash
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379
```

**Common Operations:**

```bash
# Connect using redis-cli (if installed)
redis-cli -h localhost -p 6379

# Or use Docker exec
docker exec -it openevolve-redis redis-cli

# Test connection
docker exec openevolve-redis redis-cli ping
# Should return: PONG

# Monitor commands in real-time
docker exec openevolve-redis redis-cli MONITOR

# Flush all cache (WARNING: deletes all data)
docker exec openevolve-redis redis-cli FLUSHALL
```

---

## Configuration

### Environment Files

1. **`.env.infrastructure`** - Infrastructure services only
   - Minimal configuration for Docker Compose
   - Database passwords and credentials

2. **`.env.development`** - Full development configuration
   - All service URLs
   - API keys for LLM providers
   - Application settings

3. **`.env.production`** - Production configuration
   - Override defaults for production deployment
   - External service endpoints
   - Security settings

### Docker Compose Profiles

The `docker-compose.infrastructure.yml` file supports profiles:

```bash
# Start core services only
docker-compose -f docker-compose.infrastructure.yml up -d

# Start with management tools
docker-compose -f docker-compose.infrastructure.yml --profile tools up -d
```

**Available Profiles:**
- `tools` - Includes pgAdmin and Redis Commander

### Resource Limits

Default resource allocations (can be adjusted in docker-compose):

| Service | Memory Limit | CPU Limit |
|---------|--------------|-----------|
| Qdrant | 1GB | 1.0 |
| PostgreSQL | 512MB | 0.5 |
| Redis | 256MB | 0.25 |

To modify, edit `docker-compose.infrastructure.yml`:

```yaml
services:
  qdrant:
    deploy:
      resources:
        limits:
          memory: 2G  # Adjust as needed
          cpus: '2.0'
```

---

## Verification

### Health Check Script

```bash
#!/bin/bash
# scripts/verify-infrastructure.sh

echo "Checking infrastructure services..."

# Check PostgreSQL
echo -n "PostgreSQL... "
if pg_isready -h localhost -p 5432 -U openevolve > /dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
fi

# Check Qdrant
echo -n "Qdrant... "
if curl -sf http://localhost:6333/health > /dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
fi

# Check Redis
echo -n "Redis... "
if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
fi
```

### Service Status

```bash
# View all containers
docker ps -a

# View logs for specific service
docker logs openevolve-postgres
docker logs openevolve-qdrant
docker logs openevolve-redis

# Follow logs in real-time
docker logs -f openevolve-postgres
```

### Connection Testing

**PostgreSQL:**

```bash
docker exec -it openevolve-postgres psql -U openevolve -d openevolve -c "SELECT version();"
```

**Qdrant:**

```bash
curl http://localhost:6333/
curl http://localhost:6333/collections
```

**Redis:**

```bash
docker exec openevolve-redis redis-cli INFO
```

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

**Error:** `Bind for 0.0.0.0:5432 failed: port is already allocated`

**Solution:**
```bash
# Find what's using the port
# Linux/macOS:
lsof -i :5432

# Windows:
netstat -ano | findstr :5432

# Either stop the conflicting service or change the port in docker-compose.yml
```

#### 2. Container Won't Start

**Error:** Container exits immediately

**Solution:**
```bash
# Check logs
docker logs openevolve-postgres

# Common causes:
# - Wrong environment variables (check .env.infrastructure)
# - Permission issues on volume mounts
# - Insufficient resources (check Docker RAM allocation)
```

#### 3. Cannot Connect to Database

**Error:** `connection refused` or `authentication failed`

**Solution:**
```bash
# 1. Verify container is running
docker ps | grep postgres

# 2. Check environment variables
cat .env.infrastructure | grep POSTGRES

# 3. Test connection from within container
docker exec -it openevolve-postgres psql -U openevolve -d openevolve

# 4. If authentication failed, reset password
docker exec -it openevolve-postgres psql -U postgres -c "ALTER USER openevolve PASSWORD 'newpassword';"
```

#### 4. Qdrant Shows No Collections

**Note:** This is normal for a fresh installation. Collections are created automatically by applications when needed.

#### 5. Docker Out of Memory

**Error:** `no space left on device` or containers being killed

**Solution:**
```bash
# Clean up unused resources
docker system prune -a

# Increase Docker memory allocation in Docker Desktop settings
# Recommended: 8GB or more

# Check current usage
docker system df
```

### Getting Help

1. Check logs: `docker logs <container-name>`
2. Check service status: `docker ps -a`
3. Review configuration: `cat .env.infrastructure`
4. Check Docker resources: Docker Desktop → Settings → Resources

---

## Maintenance

### Backup Procedures

#### PostgreSQL Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="./backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

docker exec openevolve-postgres pg_dump -U openevolve openevolve > \
    $BACKUP_DIR/openevolve_$TIMESTAMP.sql

# Keep last 7 days
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
```

#### Qdrant Backup

```bash
# Qdrant stores data in Docker volume
# Backup the entire volume
docker run --rm \
    -v openevolve-qdrant-data:/data \
    -v $(pwd)/backups:/backup \
    alpine tar czf /backup/qdrant_backup_$(date +%Y%m%d).tar.gz -C /data .
```

#### Redis Backup

Redis persistence is enabled (AOF mode). Backup the Redis data volume:

```bash
docker run --rm \
    -v openevolve-redis-data:/data \
    -v $(pwd)/backups:/backup \
    alpine tar czf /backup/redis_backup_$(date +%Y%m%d).tar.gz -C /data .
```

### Updates

#### Updating Docker Images

```bash
# Pull latest images
docker-compose -f docker-compose.infrastructure.yml pull

# Recreate containers with new images
docker-compose -f docker-compose.infrastructure.yml up -d

# Or use the startup script
./scripts/dev-stop.sh
./scripts/dev-start.sh
```

#### Database Migrations

For schema changes:

```bash
# Run migrations (if using a migration tool)
# Example with Alembic (Python):
alembic upgrade head

# Example with SQLx (Rust):
sqlx database create
sqlx migrate run
```

### Monitoring

#### Resource Usage

```bash
# Container resource usage
docker stats

# Specific container
docker stats openevolve-postgres openevolve-qdrant openevolve-redis
```

#### Disk Usage

```bash
# Docker disk usage
docker system df

# Volume details
docker volume ls
docker volume inspect openevolve-qdrant-data
```

#### Log Management

```bash
# View logs
docker logs -f --tail 100 openevolve-postgres

# Log rotation is handled by Docker, but you can manually clean
docker system prune -a
```

---

## Security Considerations

### Development vs Production

**Development (current setup):**
- Default passwords (acceptable for local dev)
- No encryption on connections
- Management tools exposed

**Production Requirements:**
- Strong, unique passwords
- SSL/TLS for all connections
- Restrict management tool access
- Use secrets management (Vault, AWS Secrets Manager, etc.)
- Regular security updates
- Network isolation
- Rate limiting
- Authentication for all services

### Hardening Checklist

- [ ] Change all default passwords
- [ ] Enable SSL/TLS for PostgreSQL
- [ ] Restrict Redis to localhost only
- [ ] Enable Qdrant authentication
- [ ] Use Docker secrets instead of environment variables
- [ ] Regular security updates
- [ ] Network segmentation
- [ ] Firewall rules
- [ ] Backup encryption
- [ ] Access logging and monitoring

### Environment Variables

**Never commit:**
- `.env.infrastructure` (or any `.env` files)
- `.env.development`
- `.env.production`
- Any files with real passwords or API keys

**Use version control for:**
- `.env.infrastructure.example`
- `.env.development.example`
- `.env.production.example`

---

## Advanced Usage

### Custom Configuration

#### PostgreSQL Extensions

Add extensions to `scripts/init-db.sql`:

```sql
CREATE EXTENSION IF NOT EXISTS "postgis";  -- Geographic data
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- Cryptographic functions
```

#### Redis Configuration

Modify Redis settings in `docker-compose.infrastructure.yml`:

```yaml
redis:
  command: [
    "redis-server",
    "--appendonly", "yes",
    "--maxmemory", "1gb",  # Increase from 512mb
    "--maxmemory-policy", "allkeys-lru"
  ]
```

#### Qdrant Configuration

Qdrant configuration via environment variables:

```yaml
qdrant:
  environment:
    - QDRANT__LOG_LEVEL=DEBUG
    - QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=32
```

### Networking

#### Custom Network

Services are on the `openevolve-network` bridge network. To connect external services:

```yaml
services:
  your-service:
    networks:
      - openevolve-network
    external_links:
      - openevolve-postgres
      - openevolve-qdrant
      - openevolve-redis

networks:
  openevolve-network:
    external: true
```

#### Service Discovery

Services can reach each other by container name:
- PostgreSQL: `openevolve-postgres:5432`
- Qdrant: `openevolve-qdrant:6333`
- Redis: `openevolve-redis:6379`

---

## Appendix

### File Structure

```
openevolve/
├── docker-compose.infrastructure.yml
├── scripts/
│   ├── init-db.sql
│   ├── dev-start.sh
│   ├── dev-start.bat
│   ├── dev-stop.sh
│   └── dev-stop.bat
├── docs/
│   └── INFRASTRUCTURE_SETUP.md
├── .env.infrastructure.example
└── .env.development
```

### Default Ports Reference

| Service | Internal Port | External Port | Purpose |
|---------|--------------|---------------|---------|
| PostgreSQL | 5432 | 5432 | Database |
| Qdrant HTTP | 6333 | 6333 | Vector DB API |
| Qdrant gRPC | 6334 | 6334 | Vector DB gRPC |
| Redis | 6379 | 6379 | Cache/MQ |
| pgAdmin | 80 | 5050 | DB Management UI |
| Redis Commander | 8081 | 8081 | Redis UI |

### Useful Links

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

### Support

For issues specific to:
- **OpenEvolve:** Check project documentation and GitHub issues
- **Docker:** Docker Desktop documentation
- **Services:** Official documentation for each service

---

**Last Updated:** 2026-01-11
**Version:** 1.0.0
**Phase:** Phase 0 Foundation
>>>>>>> 1cb9c5e35 (update)
