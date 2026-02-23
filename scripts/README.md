# OpenEvolve Deployment Scripts

Complete set of automated deployment and validation scripts for OpenEvolve Frontend.

## Overview

These scripts provide automated setup, deployment, validation, and cleanup for the OpenEvolve Frontend project. They follow the **ZERO TRUST** operating mode - verifying everything before proceeding and handling failures gracefully.

## Scripts

| Script | Unix | Windows | Description |
|--------|------|---------|-------------|
| **Environment Setup** | `setup-env.sh` | `setup-env.cmd` | Create and configure .env file for Docker Compose |
| **Environment Validate** | `validate-env.sh` | `validate-env.cmd` | Validate environment variables are set correctly |
| **Quick Start** | `quick-start.sh` | `quick-start.cmd` | Automated setup and first-time deployment |
| **Deploy** | `deploy.sh` | `deploy.cmd` | Deploy to local or production environment |
| **Validate** | `validate.sh` | `validate.cmd` | Validate service health and configuration |
| **Health Check** | `health-check.sh` | `health-check.cmd` | Quick health status check of all services |
| **Smoke Test** | `smoke-test.sh` | `smoke-test.cmd` | Run post-deployment smoke tests |
| **Cleanup** | `cleanup.sh` | `cleanup.cmd` | Stop services and clean up resources |
| **Deploy All Adapters** | `deploy-all-adapters.sh` | `deploy-all-adapters.cmd` | Build and deploy all adapters |
| **Deploy Single Adapter** | `deploy-adapter.sh` | `deploy-adapter.cmd` | Build and deploy a single adapter |

## Quick Start

### 1. Set Up Environment (Required First Step)

Before running any deployment scripts, you must set up your environment variables.

**Linux / macOS:**
```bash
# Make scripts executable (first time only)
chmod +x scripts/*.sh

# Set up environment file
./scripts/setup-env.sh

# Validate environment
./scripts/validate-env.sh
```

**Windows:**
```cmd
REM Set up environment file
scripts\setup-env.cmd

REM Validate environment
scripts\validate-env.cmd
```

The setup script will prompt you to choose:
1. **Full OpenEvolve Federation** - All adapters and services
2. **LoongFlow Core only** - Just the workflow engine
3. **Minimal development setup** - Basic configuration

After setup, you'll need to edit the `.env` file to add your API keys and credentials:
```bash
# Edit the environment file
nano .env  # or code .env
```

**Required variables to set:**
- `LOONGFLOW_LLM_API_KEY` - OpenAI API key for LoongFlow
- `OPENAI_API_KEY` - OpenAI API key for other services
- `NEO4J_PASSWORD` - Neo4j database password
- Other service-specific credentials

### 2. Run Deployment

Once your environment is configured:

### Linux / macOS

```bash
# Run quick start
./scripts/quick-start.sh
```

### Windows

```cmd
REM Run quick start
scripts\quick-start.cmd
```

## Detailed Usage

### Quick Start Script

Automated setup for first-time deployment.

**Unix:**
```bash
./scripts/quick-start.sh [OPTIONS]
```

**Windows:**
```cmd
scripts\quick-start.cmd [OPTIONS]
```

**Options:**
- `--dry-run` - Preview actions without executing
- `--skip-tests` - Skip running tests
- `--env-file FILE` - Use specific environment file

**What it does:**
1. Checks prerequisites (Node, npm, Docker, Docker Compose)
2. Validates environment variables
3. Installs dependencies
4. Runs tests (type check, lint, unit tests)
5. Builds TypeScript code and Docker images
6. Starts all services with Docker Compose
7. Verifies service health
8. Shows next steps

**Example:**
```bash
./scripts/quick-start.sh --skip-tests
```

---

### Environment Setup Script

Create and configure `.env` file for Docker Compose.

**Unix:**
```bash
./scripts/setup-env.sh
```

**Windows:**
```cmd
scripts\setup-env.cmd
```

**What it does:**
1. Checks if `.env` already exists (prompts to overwrite)
2. Backs up existing `.env` to `.env.backup`
3. Prompts for environment type:
   - Full OpenEvolve Federation (all adapters)
   - LoongFlow Core only
   - Minimal development setup
4. Copies from example file or creates minimal `.env`
5. Displays next steps

**Following CLAUDE.md - Law of Configuration Explicitness:**
- All values must be explicitly configured
- NO magic defaults - services will crash if required values are missing
- All timestamps in UTC (Law of UTC)

**Example output:**
```
================================================
OpenEvolve Environment Setup
================================================

Which environment do you want to set up?
1) Full OpenEvolve Federation (all adapters)
2) LoongFlow Core only
3) Minimal development setup

Choose (1-3): 1

✅ Environment file created: .env

⚠️  IMPORTANT: Review .env and update values before starting services!

Required variables to set:
  - API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
  - Service URLs (if not using defaults)
  - Database credentials (NEO4J_PASSWORD, etc.)
```

**After running setup:**
```bash
# Edit the environment file with your values
nano .env

# Validate the configuration
./scripts/validate-env.sh
```

---

### Environment Validation Script

Validate that required environment variables are set correctly.

**Unix:**
```bash
./scripts/validate-env.sh
```

**Windows:**
```cmd
scripts\validate-env.cmd
```

**What it checks:**

**Infrastructure Configuration:**
- `EVENT_BUS_URL` - Event bus connection URL
- `TZ` - Timezone (should be UTC)

**API Endpoints:**
- `LOONGFLOW_API_URL` - LoongFlow Core API endpoint
- `OPENEVOLVE_API_URL` - OpenEvolve API endpoint
- `BUBBLELAB_API_URL` - BubbleLab API endpoint
- `RAGBITS_API_URL` - RagBits API endpoint

**Service Configuration:**
- `TIMEOUT_MS` - Request timeout (default: 30000)
- `MAX_RETRIES` - Retry attempts (default: 3)
- `LOG_LEVEL` - Logging level (default: INFO)

**LLM Provider Configuration:**
- `LOONGFLOW_LLM_API_KEY` - OpenAI API key (required)
- `OPENAI_API_KEY` - OpenAI API key (optional)
- `ANTHROPIC_API_KEY` - Anthropic API key (optional)
- `GOOGLE_API_KEY` - Google API key (optional)

**Database Configuration:**
- `NEO4J_URI` - Neo4j connection URI
- `NEO4J_USER` - Neo4j username
- `NEO4J_PASSWORD` - Neo4j password

**Development Settings:**
- `DEBUG` - Debug mode (default: false)
- `SKIP_CONTRACT_TESTS` - Skip contract tests (default: false)

**Exit codes:**
- `0` - Validation passed
- `1` - Validation failed (missing required variables)

**Example output:**
```
================================================
OpenEvolve Environment Validation
================================================

🔍 Validating environment variables...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Infrastructure Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ EVENT_BUS_URL
✅ TZ

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API Endpoints
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ LOONGFLOW_API_URL
✅ OPENEVOLVE_API_URL
✅ BUBBLELAB_API_URL
✅ RAGBITS_API_URL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLM Provider Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  PLACEHOLDER: LOONGFLOW_LLM_API_KEY
   OpenAI API key for LoongFlow
   Current value: sk-your-openai-api-key-here

================================================
Validation Summary
================================================
✅ Valid variables: 8
⚠️  Placeholders: 1
❌ Missing required: 0

⚠️  VALIDATION PASSED WITH WARNINGS

Some variables contain placeholder values.
Services may not start correctly without proper values.

Please review and update placeholder values in .env
```

---

### Deploy Script

Deploy to local or production environment.

**Unix:**
```bash
./scripts/deploy.sh [ENVIRONMENT] [OPTIONS]
```

**Windows:**
```cmd
scripts\deploy.cmd [ENVIRONMENT] [OPTIONS]
```

**Environments:**
- `local` - Deploy to local Docker Compose (default)
- `production` - Deploy to production

**Options:**
- `--dry-run` - Preview deployment without executing
- `--skip-smoke-tests` - Skip smoke tests after deployment

**What it does:**
1. Pre-deployment checks (Docker running, environment file)
2. Builds adapters and glue layer
3. Builds Docker images
4. Validates configurations
5. Stops existing services
6. Deploys new services
7. Waits for services to be healthy
8. Runs smoke tests
9. Shows deployment status

**Examples:**
```bash
# Deploy to local
./scripts/deploy.sh local

# Deploy to production
./scripts/deploy.sh production

# Dry run deployment
./scripts/deploy.sh local --dry-run
```

---

### Validate Script

Validate service health and generate report.

**Unix:**
```bash
./scripts/validate.sh [OPTIONS]
```

**Windows:**
```cmd
scripts\validate.cmd
```

**Options:**
- `--report-only` - Generate report without console output

**What it checks:**
1. Service status (running/stopped)
2. Health endpoint responses
3. API connectivity
4. Event bus connection
5. Log analysis (errors/warnings)

**Output:**
- Console: Real-time status updates
- File: Markdown report in `logs/validation-report-YYYYMMDD-HHMMSS.md`

**Exit codes:**
- `0` - All healthy
- `1` - Degraded
- `2` - Unhealthy

**Example:**
```bash
./scripts/validate.sh
```

---

### Health Check Script

Quick health status of all services.

**Unix:**
```bash
./scripts/health-check.sh [OPTIONS]
```

**Windows:**
```cmd
scripts\health-check.cmd [OPTIONS]
```

**Options:**
- `--json` - Output in JSON format
- `--quiet` - Suppress verbose output

**Services checked:**
- `openevolve-app` - Main application (http://localhost:8080/health)
- `openevolve-valkey` - Message broker (tcp://localhost:6379)
- `openevolve-prometheus` - Metrics (http://localhost:9090/-/healthy)
- `openevolve-grafana` - Dashboard (http://localhost:3000/api/health)

**Output formats:**

Table (default):
```
Service                        Status          Response
------------------------------ --------------- --------------------
openevolve-app                 healthy         HTTP 200
openevolve-valkey              healthy         Port open
```

JSON:
```json
{
  "openevolve-app": {"status": "healthy", "response": "HTTP 200"}
}
```

**Exit codes:**
- `0` - All healthy
- `1` - Some services unhealthy

**Examples:**
```bash
# Table format
./scripts/health-check.sh

# JSON format
./scripts/health-check.sh --json

# Quiet output
./scripts/health-check.sh --quiet
```

---

### Smoke Test Script

Run post-deployment smoke tests.

**Unix:**
```bash
./scripts/smoke-test.sh [OPTIONS]
```

**Windows:**
```cmd
scripts\smoke-test.cmd [OPTIONS]
```

**Options:**
- `-e, --environment <env>` - Environment (staging|production|local)
- `-u, --url <url>` - Custom base URL
- `-t, --timeout <seconds>` - Request timeout (default: 30)
- `-v, --verbose` - Enable verbose output
- `-k, --kubectl` - Run Kubernetes-specific tests

**Tests performed:**
1. Health endpoint (`/health`)
2. API readiness (`/api/v1/ready`)
3. Event bus connection (`/api/v1/status/eventbus`)
4. Workflow engine (`/api/v1/workflows/health`)
5. Adapter status (`/api/v1/adapters/status`)
6. Metrics endpoint (`/metrics`)
7. Kubernetes deployment (optional)
8. Kubernetes pods (optional)

**Examples:**
```bash
# Test local environment
./scripts/smoke-test.sh -e local

# Test with custom URL
./scripts/smoke-test.sh -u http://localhost:8080

# Test with Kubernetes checks
./scripts/smoke-test.sh -e production -k

# Verbose output
./scripts/smoke-test.sh -e local -v
```

---

### Cleanup Script

Stop services and clean up resources.

**Unix:**
```bash
./scripts/cleanup.sh [OPTIONS]
```

**Windows:**
```cmd
scripts\cleanup.cmd [OPTIONS]
```

**Options:**
- `--volumes` - Remove Docker volumes (WARNING: Deletes data!)
- `--all` - Remove everything including images and build cache
- `--dry-run` - Preview cleanup without executing

**What it cleans:**
1. Stops all services (`docker compose down`)
2. Removes containers
3. Removes volumes (if `--volumes` specified)
4. Removes build artifacts (node_modules, dist)
5. Cleans old log files
6. Removes Docker images (if `--all` specified)

**Warning:** Using `--volumes` or `--all` will delete data. Use with caution!

**Examples:**
```bash
# Stop services only
./scripts/cleanup.sh

# Stop services and remove volumes
./scripts/cleanup.sh --volumes

# Complete cleanup
./scripts/cleanup.sh --all

# Dry run
./scripts/cleanup.sh --all --dry-run
```

## Adapter Deployment Scripts

### Deploy All Adapters

Universal deployment script that builds and deploys all adapters in `glue/adapters/`.

**Linux/macOS:**
```bash
./scripts/deploy-all-adapters.sh [OPTIONS]
```

**Windows:**
```cmd
scripts\deploy-all-adapters.cmd [OPTIONS]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--push` | Push images to registry after building | `false` |
| `--registry <url>` | Docker registry URL | `localhost:5000` |
| `--tag <tag>` | Image tag | `latest` |
| `--skip-tests` | Skip running contract tests | `false` |
| `--adapter <name>` | Deploy only a specific adapter | All adapters |
| `--dry-run` | Show what would be deployed without building | `false` |
| `-h, --help` | Show help message | - |

**Examples:**
```bash
# Deploy all adapters
./scripts/deploy-all-adapters.sh

# Deploy specific adapter
./scripts/deploy-all-adapters.sh --adapter bubblelab-adapter

# Build and push to registry
./scripts/deploy-all-adapters.sh --push --registry registry.example.com --tag v1.0.0

# Skip tests for faster iteration
./scripts/deploy-all-adapters.sh --skip-tests

# Dry run to see what would be deployed
./scripts/deploy-all-adapters.sh --dry-run
```

**What it does:**
1. Scans `glue/adapters/` for all adapter directories
2. Checks each adapter has a Dockerfile
3. Builds Docker image for each adapter
4. Runs contract tests (if `tests/` directory exists)
5. Optionally pushes images to registry
6. Provides deployment summary

**Supported Adapters:**
The following adapters have Dockerfiles and can be deployed:
- `adaptive_mdap-adapter`
- `bubblelab-adapter`
- `graphiti-adapter`
- `icr-adapter`
- `leanaide-adapter`
- `loongflow-adapter`
- `openevolve-adapter`
- `vectordb-adapter`
- `z3-adapter`

---

### Deploy Single Adapter

Deploy a single adapter by name.

**Linux/macOS:**
```bash
./scripts/deploy-adapter.sh <adapter-name> [OPTIONS]
```

**Windows:**
```cmd
scripts\deploy-adapter.cmd <adapter-name> [OPTIONS]
```

**Arguments:**
- `adapter-name` - Name of the adapter (e.g., `bubblelab-adapter`)

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--push` | Push image to registry after building | `false` |
| `--registry <url>` | Docker registry URL | `localhost:5000` |
| `--tag <tag>` | Image tag | `latest` |
| `--skip-tests` | Skip running contract tests | `false` |
| `--dry-run` | Show what would be deployed without building | `false` |
| `-h, --help` | Show help message | - |

**Examples:**
```bash
# Deploy specific adapter
./scripts/deploy-adapter.sh bubblelab-adapter

# Deploy with custom tag
./scripts/deploy-adapter.sh z3-adapter --tag v1.0.0

# Deploy and push to registry
./scripts/deploy-adapter.sh openevolve-adapter --push --registry registry.example.com

# Deploy without tests (fast iteration)
./scripts/deploy-adapter.sh loongflow-adapter --skip-tests
```

**What it does:**
1. Validates adapter exists and has Dockerfile
2. Builds Docker image
3. Runs contract tests (if available)
4. Optionally pushes to registry
5. Shows deployment status and next steps

### Environment Variables for Adapter Deployment

You can configure adapter deployment using environment variables:

```bash
export DOCKER_REGISTRY=registry.example.com
export IMAGE_TAG=v1.0.0

# Now all deployments use these defaults
./scripts/deploy-all-adapters.sh --push
```

### Running Deployed Adapters

After deployment, run adapters:

```bash
# Run adapter
docker run -p 8080:8080 localhost:5000/bubblelab-adapter:latest

# Run with custom environment
docker run -p 8080:8080 -e LOG_LEVEL=DEBUG localhost:5000/z3-adapter:latest

# View logs
docker logs -f <container-id>

# Execute commands in container
docker exec -it <container-id> /bin/bash
```

### Development Workflow

```bash
# 1. Quick rebuild during development (skip tests)
./scripts/deploy-adapter.sh bubblelab-adapter --skip-tests

# 2. Test locally
docker run -p 8080:8080 localhost:5000/bubblelab-adapter:latest

# 3. When ready, run full tests
./scripts/deploy-adapter.sh bubblelab-adapter

# 4. Deploy to production
./scripts/deploy-adapter.sh bubblelab-adapter --registry registry.example.com --tag v1.0.0 --push
```

### CI/CD Integration

```yaml
# .github/workflows/deploy-adapters.yml
name: Deploy Adapters

on:
  push:
    paths:
      - 'glue/adapters/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to registry
        uses: docker/login-action@v2
        with:
          registry: ${{ secrets.DOCKER_REGISTRY }}
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Deploy all adapters
        run: |
          chmod +x scripts/*.sh
          ./scripts/deploy-all-adapters.sh \
            --registry ${{ secrets.DOCKER_REGISTRY }} \
            --tag ${{ github.sha }} \
            --push
```

## Prerequisites

### Required Software

| Software | Version | Check Command |
|----------|---------|---------------|
| Node.js | >= 18.0.0 | `node -v` |
| npm | >= 9.0.0 | `npm -v` |
| Docker | Latest stable | `docker --version` |
| Docker Compose | v2+ | `docker compose version` |

### Optional but Recommended

| Software | Purpose |
|----------|---------|
| Python | For certain build tools |
| Git | For version control |
| curl | For health checks (pre-installed on most systems) |
| jq | For JSON parsing in smoke tests |

### Installing Prerequisites

#### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Docker Desktop
brew install --cask docker

# Install Node.js
brew install node

# Install jq (optional)
brew install jq
```

#### Ubuntu/Debian

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install jq (optional)
sudo apt-get install -y jq
```

#### Windows

1. Install Docker Desktop: https://www.docker.com/products/docker-desktop
2. Install Node.js: https://nodejs.org/
3. Install Git: https://git-scm.com/download/win

## Environment Variables

### Quick Setup

The easiest way to set up environment variables is to use the setup script:

**Linux / macOS:**
```bash
./scripts/setup-env.sh
```

**Windows:**
```cmd
scripts\setup-env.cmd
```

This will create a `.env` file with all required variables pre-configured with sensible defaults. You'll need to edit the file to add your API keys and credentials.

### Manual Setup

Alternatively, you can manually copy from an example file:

```bash
# Full environment
cp infra/.env.example .env

# Or LoongFlow only
cp infra/.env.loongflow.example .env
```

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SECRET_KEY` | Secret key for encryption | Generate with: `openssl rand -hex 32` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENEVOLVE_LOG_LEVEL` | Logging level | INFO |
| `OPENEVOLVE_EVENT_BUS__HOST` | Event bus host | localhost |
| `OPENEVOLVE_EVENT_BUS__PORT` | Event bus port | 6379 |
| `OPENEVOLVE_TELEMETRY__OTLP_ENDPOINT` | OTLP endpoint | http://localhost:4317 |

See `.env.example` for all available variables.

## Service URLs

After deployment, services are available at:

| Service | URL | Credentials |
|---------|-----|-------------|
| OpenEvolve API | http://localhost:8000 | - |
| GraphQL API | http://localhost:8001 | - |
| Orchestrator/Gateway | http://localhost:8080 | - |
| BubbleLab Dashboard | http://localhost:8501 | - |
| Jaeger Tracing | http://localhost:16686 | - |
| Prometheus Metrics | http://localhost:9090 | - |
| Grafana Dashboard | http://localhost:3000 | admin/admin |

## Troubleshooting

### Docker issues

**Problem:** Docker daemon not running

```bash
# Start Docker Desktop (macOS/Windows)
# Or start Docker service (Linux)
sudo systemctl start docker
```

**Problem:** Permission denied

```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### Port conflicts

**Problem:** Ports already in use

```bash
# Check what's using the port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Stop conflicting services or change ports in .env
```

### Services not starting

**Problem:** Services fail to start

```bash
# Check logs
docker compose logs -f

# Check specific service
docker compose logs openevolve-app

# Restart services
docker compose restart
```

### Health checks failing

**Problem:** Health endpoint not responding

```bash
# Wait longer for services to start
sleep 30

# Check if containers are running
docker compose ps

# Check container health
docker inspect openevolve-app | grep -A 10 Health
```

### Build failures

**Problem:** npm install fails

```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
rm -rf node_modules
npm install --legacy-peer-deps
```

**Problem:** Docker build fails

```bash
# Clear Docker build cache
docker builder prune -af

# Rebuild without cache
docker compose build --no-cache
```

### Out of disk space

**Problem:** No space left on device

```bash
# Clean up Docker resources
docker system prune -a --volumes

# Remove old logs
find logs/ -name "*.log" -mtime +7 -delete
```

## Log Files

Logs are stored in the `logs/` directory:

| Log Type | Pattern | Retention |
|----------|---------|-----------|
| Quick start | `quick-start-YYYYMMDD-HHMMSS.log` | Manual |
| Deploy | `deploy-YYYYMMDD-HHMMSS.log` | Manual |
| Validation | `validation-report-YYYYMMDD-HHMMSS.md` | Manual |
| Docker logs | `docker compose logs` | Container lifecycle |

View logs:
```bash
# View all logs
docker compose logs -f

# View specific service
docker compose logs -f openevolve-app

# View last 100 lines
docker compose logs --tail=100
```

## CI/CD Integration

These scripts can be integrated into CI/CD pipelines:

### GitHub Actions Example

```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          chmod +x scripts/*.sh
          ./scripts/deploy.sh production
```

### GitLab CI Example

```yaml
deploy:
  stage: deploy
  script:
    - chmod +x scripts/*.sh
    - ./scripts/deploy.sh production
  only:
    - main
```

## Best Practices

1. **Always run in dry-run mode first** for production deployments
   ```bash
   ./scripts/deploy.sh production --dry-run
   ```

2. **Keep `.env` file out of version control**
   ```bash
   echo ".env" >> .gitignore
   ```

3. **Use strong `SECRET_KEY` in production**
   ```bash
   openssl rand -hex 32
   ```

4. **Run smoke tests after every deployment**
   ```bash
   ./scripts/deploy.sh local && ./scripts/smoke-test.sh
   ```

5. **Monitor logs during deployment**
   ```bash
   # In one terminal
   ./scripts/deploy.sh production

   # In another terminal
   docker compose logs -f
   ```

6. **Clean up old log files regularly**
   ```bash
   find logs/ -name "*.log" -mtime +30 -delete
   ```

7. **Back up data before cleanup with --volumes**
   ```bash
   docker run --rm -v openevolve_frontend_openevolve_data:/data -v $(pwd)/backup:/backup alpine tar czf /backup/data.tar.gz /data
   ```

## Support

For issues and questions:

1. Check troubleshooting section above
2. Review logs in `logs/` directory
3. Check Docker logs: `docker compose logs`
4. Open an issue on GitHub

## License

Apache 2.0 - See LICENSE file for details

## Contributing

When adding new scripts:

1. Follow existing naming convention
2. Add usage instructions to this README
3. Include `--help` flag
4. Use colored output for readability
5. Support both Unix and Windows
6. Add error handling
7. Log actions to file

Example template:
```bash
#!/bin/bash
# =============================================================================
# OpenEvolve Your Script
# License: Apache 2.0
# Description: What your script does
# Usage: ./your-script.sh [OPTIONS]
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Your script logic here
```
