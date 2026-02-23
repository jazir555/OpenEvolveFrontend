# Environment Setup Quick Reference

## Overview

This guide explains how to set up and validate environment variables for the OpenEvolve Frontend project.

## Prerequisites

Before setting up the environment, ensure you have:

- Docker and Docker Compose installed
- Access to required API keys (OpenAI, Anthropic, etc.)
- Database credentials (Neo4j, etc.)

## Quick Start

### Step 1: Run Setup Script

**Linux / macOS:**
```bash
chmod +x scripts/*.sh
./scripts/setup-env.sh
```

**Windows:**
```cmd
scripts\setup-env.cmd
```

### Step 2: Edit Environment File

The setup script will create a `.env` file in the project root. Edit this file to add your credentials:

**Linux / macOS:**
```bash
nano .env
# or
code .env
```

**Windows:**
```cmd
notepad .env
```

### Step 3: Validate Configuration

**Linux / macOS:**
```bash
./scripts/validate-env.sh
```

**Windows:**
```cmd
scripts\validate-env.cmd
```

## Required Environment Variables

### Infrastructure

| Variable | Description | Example |
|----------|-------------|---------|
| `EVENT_BUS_URL` | Redis event bus URL | `redis://event-bus:6379` |
| `TZ` | Timezone (must be UTC) | `UTC` |

### API Endpoints

| Variable | Description | Example |
|----------|-------------|---------|
| `LOONGFLOW_API_URL` | LoongFlow Core API | `http://loongflow-core:8050` |
| `OPENEVOLVE_API_URL` | OpenEvolve API | `http://openevolve-core:8000` |
| `BUBBLELAB_API_URL` | BubbleLab API | `http://bubblelab-core:8501` |
| `RAGBITS_API_URL` | RagBits API | `http://ragbits-core:8000` |

### LLM Provider

| Variable | Description | Required |
|----------|-------------|----------|
| `LOONGFLOW_LLM_API_KEY` | OpenAI API key for LoongFlow | Yes |
| `OPENAI_API_KEY` | OpenAI API key for other services | Recommended |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `GOOGLE_API_KEY` | Google API key | Optional |

### Database

| Variable | Description | Example |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `bolt://neo4j:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | (your password) |

### Service Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TIMEOUT_MS` | Request timeout (ms) | `30000` |
| `MAX_RETRIES` | Max retry attempts | `3` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | Log format | `json` |
| `DEBUG` | Debug mode | `false` |
| `SKIP_CONTRACT_TESTS` | Skip contract tests | `false` |

## Setup Options

The setup script offers three environment types:

### 1. Full OpenEvolve Federation
- All adapters and services
- Complete configuration
- Recommended for production

### 2. LoongFlow Core Only
- Workflow engine only
- Minimal dependencies
- Good for development

### 3. Minimal Development Setup
- Basic configuration
- Essential variables only
- Fastest setup

## Validation Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Validation passed |
| `1` | Validation failed (missing required variables) |

## Troubleshooting

### Issue: Validation fails with "Missing required variables"

**Solution:**
```bash
# Edit .env file
nano .env

# Add the missing variables
# Example:
LOONGFLOW_LLM_API_KEY=sk-your-actual-api-key-here

# Re-validate
./scripts/validate-env.sh
```

### Issue: Setup script says ".env file already exists"

**Solution:** You can safely overwrite it, or backup first:
```bash
# Backup existing .env
cp .env .env.backup

# Re-run setup
./scripts/setup-env.sh
# Choose 'y' when prompted to overwrite
```

### Issue: Services fail to start after setup

**Solution:**
1. Check validation passed:
   ```bash
   ./scripts/validate-env.sh
   ```

2. Check for placeholder values:
   ```bash
   grep "your-\|changeme\|here" .env
   ```

3. Replace placeholders with actual values

4. Restart services:
   ```bash
   docker-compose -f infra/docker-compose-all-adapters.yml restart
   ```

## Best Practices

1. **Never commit .env to version control**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use different .env files for different environments**
   - `.env` - Local development
   - `.env.production` - Production
   - `.env.staging` - Staging

3. **Validate after changes**
   ```bash
   ./scripts/validate-env.sh
   ```

4. **Keep API keys secure**
   - Use environment variable managers in production
   - Rotate keys regularly
   - Use different keys for different environments

5. **Follow CLAUDE.md principles**
   - Law of Configuration Explicitness: No magic defaults
   - Law of UTC: Always use UTC timezone
   - Services will crash if required variables are missing

## Next Steps

After environment setup:

1. **Start infrastructure:**
   ```bash
   docker-compose -f docker-compose.infrastructure.yml up -d
   ```

2. **Start services:**
   ```bash
   docker-compose -f infra/docker-compose-all-adapters.yml up -d
   ```

3. **Check health:**
   ```bash
   ./scripts/health-check.sh
   ```

4. **Run smoke tests:**
   ```bash
   ./scripts/smoke-test.sh
   ```

## Additional Resources

- [CLAUDE.md](../CLAUDE.md) - Architecture and design principles
- [scripts/README.md](README.md) - Complete scripts documentation
- [infra/.env.example](../infra/.env.example) - Full environment template
- [infra/.env.loongflow.example](../infra/.env.loongflow.example) - LoongFlow template

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Run validation: `./scripts/validate-env.sh`
3. Check logs: `docker-compose logs`
4. Review [scripts/README.md](README.md) for detailed documentation
