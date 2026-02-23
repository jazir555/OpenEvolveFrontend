# Adapter Deployment Scripts - Quick Reference

## Overview

Universal deployment scripts for building and deploying all OpenEvolve adapters.

## Scripts

| Script | Platform | Purpose |
|--------|----------|---------|
| `deploy-all-adapters.sh` | Linux/macOS | Deploy all adapters or specific ones |
| `deploy-all-adapters.cmd` | Windows | Deploy all adapters or specific ones |
| `deploy-adapter.sh` | Linux/macOS | Deploy a single adapter |
| `deploy-adapter.cmd` | Windows | Deploy a single adapter |

## Quick Start

### Deploy All Adapters

```bash
# Linux/macOS
./scripts/deploy-all-adapters.sh

# Windows
scripts\deploy-all-adapters.cmd
```

### Deploy Single Adapter

```bash
# Linux/macOS
./scripts/deploy-adapter.sh bubblelab-adapter

# Windows
scripts\deploy-adapter.cmd bubblelab-adapter
```

## Common Use Cases

### 1. Development - Fast Iteration

Skip tests for quick builds during development:

```bash
./scripts/deploy-adapter.sh my-adapter --skip-tests
```

### 2. Test Before Deploy

Build with tests to verify everything works:

```bash
./scripts/deploy-adapter.sh my-adapter
```

### 3. Deploy to Production

Build, test, and push to registry:

```bash
./scripts/deploy-all-adapters.sh \
  --registry registry.example.com \
  --tag v1.0.0 \
  --push
```

### 4. Deploy Specific Adapter

Build only what you need:

```bash
./scripts/deploy-all-adapters.sh --adapter bubblelab-adapter
```

### 5. Preview Deployment

See what would be deployed without actually building:

```bash
./scripts/deploy-all-adapters.sh --dry-run
```

## Options

### Universal Script (`deploy-all-adapters.sh`)

| Flag | Description |
|------|-------------|
| `--push` | Push images to registry after building |
| `--registry <url>` | Docker registry URL (default: localhost:5000) |
| `--tag <tag>` | Image tag (default: latest) |
| `--skip-tests` | Skip running contract tests |
| `--adapter <name>` | Deploy only a specific adapter |
| `--dry-run` | Show what would be deployed without building |
| `-h, --help` | Show help message |

### Single Adapter Script (`deploy-adapter.sh`)

| Flag | Description |
|------|-------------|
| `--push` | Push image to registry after building |
| `--registry <url>` | Docker registry URL (default: localhost:5000) |
| `--tag <tag>` | Image tag (default: latest) |
| `--skip-tests` | Skip running contract tests |
| `--dry-run` | Show what would be deployed without building |
| `-h, --help` | Show help message |

## Environment Variables

Configure defaults using environment variables:

```bash
export DOCKER_REGISTRY=registry.example.com
export IMAGE_TAG=v1.0.0

./scripts/deploy-all-adapters.sh --push
```

## Supported Adapters

Adapters with Dockerfiles (can be deployed):

- `adaptive_mdap-adapter`
- `bubblelab-adapter`
- `graphiti-adapter`
- `icr-adapter`
- `leanaide-adapter`
- `loongflow-adapter`
- `openevolve-adapter`
- `vectordb-adapter`
- `z3-adapter`

## Running Deployed Adapters

After deployment:

```bash
# Run adapter
docker run -p 8080:8080 localhost:5000/bubblelab-adapter:latest

# Run with custom environment
docker run -p 8080:8080 -e LOG_LEVEL=DEBUG localhost:5000/z3-adapter:latest

# View logs
docker logs -f <container-id>

# Interactive shell
docker exec -it <container-id> /bin/bash
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Deploy adapters
  run: |
    chmod +x scripts/*.sh
    ./scripts/deploy-all-adapters.sh \
      --registry ${{ secrets.DOCKER_REGISTRY }} \
      --tag ${{ github.sha }} \
      --push
```

### GitLab CI

```yaml
deploy:
  script:
    - chmod +x scripts/*.sh
    - ./scripts/deploy-all-adapters.sh --push --tag $CI_COMMIT_SHA
```

## Troubleshooting

### Docker Not Running

```
[ERROR] Docker is not running
```

Start Docker Desktop or your Docker daemon.

### Adapter Not Found

```
[ERROR] Adapter not found: my-adapter
```

Check the adapter name matches a directory in `glue/adapters/`.

### No Dockerfile

```
[WARN] Skipping my-adapter (no Dockerfile)
```

The adapter doesn't have a Dockerfile. Create one first.

### Tests Failed

```
[ERROR] Failed tests for my-adapter
```

Run tests manually to see details:
```bash
cd glue/adapters/my-adapter
docker run -it localhost:5000/my-adapter:latest pytest tests/ -v
```

### Registry Push Failed

```
[ERROR] Failed to push my-adapter
```

Ensure you're authenticated:
```bash
docker login registry.example.com
```

## Architecture

These scripts follow the OpenEvolve Federation Constitution:

- **Zero Trust**: Verify everything before deployment
- **Idempotency**: Safe to run multiple times
- **Runtime Truth**: Execute against live containers
- **Explicit Configuration**: All settings via flags or environment variables
- **UTC**: All operations in UTC timezone

## Contributing

When adding new adapters:

1. Create directory: `glue/adapters/my-adapter/`
2. Add `Dockerfile`
3. Add `tests/` directory with contract tests
4. Deploy: `./scripts/deploy-adapter.sh my-adapter`

No need to modify deployment scripts - they're universal!

## See Also

- [scripts/README.md](README.md) - Full documentation for all scripts
- [CLAUDE.md](../CLAUDE.md) - Federation Constitution
- [glue/adapters/](../glue/adapters/) - Adapter implementations
