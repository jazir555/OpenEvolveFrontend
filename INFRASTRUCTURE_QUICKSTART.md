# Infrastructure Quick Start

**OpenEvolve Phase 0 Foundation - Quick Reference**

## Prerequisites

- Docker Desktop installed and running
- 4GB+ RAM allocated to Docker
- Git (for cloning)

## 30-Second Setup

### 1. Create Environment File

```bash
cp .env.infrastructure.example .env.infrastructure
```

**IMPORTANT:** Edit `.env.infrastructure` and change the default password:
```bash
POSTGRES_PASSWORD=your-secure-password-here
```

### 2. Start Services

**Linux/macOS:**
```bash
./scripts/dev-start.sh
```

**Windows:**
```cmd
scripts\dev-start.bat
```

### 3. Verify

```bash
# Linux/macOS
./scripts/verify-infrastructure.sh

# Windows
scripts\verify-infrastructure.bat
```

## Service Endpoints

| Service | URL | Credentials |
|---------|-----|-------------|
| Qdrant Dashboard | http://localhost:6333/dashboard | None |
| PostgreSQL | localhost:5432 | User: `openevolve`<br>Password: (from .env.infrastructure) |
| Redis | localhost:6379 | None |

## Common Commands

### View Logs
```bash
docker logs -f openevolve-postgres
docker logs -f openevolve-qdrant
docker logs -f openevolve-redis
```

### Stop Services
```bash
# Linux/macOS
./scripts/dev-stop.sh

# Windows
scripts\dev-stop.bat
```

### Start with Management Tools
```bash
# Includes pgAdmin (port 5050) and Redis Commander (port 8081)
./scripts/dev-start.sh --with-tools
```

## Connection Strings

### PostgreSQL
```
postgresql://openevolve:your-password@localhost:5432/openevolve
```

### Qdrant
```
http://localhost:6333
```

### Redis
```
redis://localhost:6379
```

## Troubleshooting

**Port already in use?**
```bash
# Check what's using the port
# Linux/macOS:
lsof -i :5432

# Windows:
netstat -ano | findstr :5432
```

**Container won't start?**
```bash
# Check logs
docker logs openevolve-postgres

# Check environment variables
cat .env.infrastructure
```

**Can't connect to database?**
```bash
# Test connection from within container
docker exec -it openevolve-postgres psql -U openevolve -d openevolve
```

## Next Steps

1. **Full Documentation:** See `docs/INFRASTRUCTURE_SETUP.md`
2. **Configure Application:** Update your application's connection strings
3. **Run Migrations:** If needed, run database migrations
4. **Development:** Start your application services

## Project Structure

```
Frontend/
├── docker-compose.infrastructure.yml  # Infrastructure services
├── scripts/
│   ├── dev-start.sh                   # Start services (Linux/Mac)
│   ├── dev-start.bat                  # Start services (Windows)
│   ├── dev-stop.sh                    # Stop services (Linux/Mac)
│   ├── dev-stop.bat                   # Stop services (Windows)
│   ├── verify-infrastructure.sh       # Verify setup (Linux/Mac)
│   └── verify-infrastructure.bat      # Verify setup (Windows)
├── docs/
│   └── INFRASTRUCTURE_SETUP.md        # Full documentation
├── .env.infrastructure.example        # Configuration template
└── .env.development                   # Full dev environment config
```

## Support

- **Documentation:** `docs/INFRASTRUCTURE_SETUP.md`
- **Issues:** Check Docker logs first
- **Service Docs:** Qdrant, PostgreSQL, Redis official documentation

---

**Quick Links:**
- [Qdrant Dashboard](http://localhost:6333/dashboard)
- [pgAdmin](http://localhost:5050) (if `--with-tools` enabled)
- [Redis Commander](http://localhost:8081) (if `--with-tools` enabled)
