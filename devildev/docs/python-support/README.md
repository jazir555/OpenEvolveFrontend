# DevilDev Python Support - Documentation

**Version**: 1.0.0
**Status**: In Progress
**Last Updated**: 2025-01-16

## Overview

DevilDev now supports **Python** as a first-class language alongside TypeScript and JavaScript. This documentation covers the implementation, architecture, and usage of Python support in the DevilDev platform.

## What's Supported

### Python Versions
- **3.9** (minimum)
- **3.10** (with match statements)
- **3.11** (with exception groups)
- **3.12** (with type parameter syntax)
- **3.13** (latest)

### Package Managers
- pip
- poetry
- pipenv
- conda
- venv
- uv (Rust-based, 100x faster)
- pip-tools
- PDM
- Hatch

### Web Frameworks
- FastAPI ⚡
- Flask
- Django
- Falcon
- Starlette
- Quart
- Sanic
- Tornado
- AIOHTTP

### Data Science & ML
- NumPy, Pandas, Matplotlib
- Scikit-learn, TensorFlow, PyTorch
- JAX, MXNet, XGBoost, LightGBM
- Jupyter Notebooks

### Database ORMs
- SQLAlchemy
- Django ORM
- Tortoise ORM (async)
- databases (async)
- SQLModel
- Pony ORM

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     DevilDev Frontend                        │
│                   (Next.js + React)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Python Execution Engine                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Language   │  │   Code       │  │   Circuit    │     │
│  │   Detector   │  │  Sanitizer   │  │   Breaker    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              E2B Sandbox Templates                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     Base     │  │ Data Science │  │      Web     │     │
│  │   Template   │  │   Template   │  │   Template   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Database Schema

```prisma
enum ProjectLanguage {
  TYPESCRIPT
  JAVASCRIPT
  PYTHON
}

model PythonPackage {
  id        String   @id @default(uuid())
  projectId String
  name      String
  version   String?
  project   Project  @relation(fields: [projectId], references: [id])
  @@unique([projectId, name])
}

model DeadLetter {
  id         String   @id @default(uuid())
  operation  String
  userId     String
  projectId  String?
  payload    Json
  error      String?
  retryCount Int      @default(0)
  status     String   @default("PENDING")
}
```

## Quick Start

### 1. Environment Setup

```bash
# Copy environment variables
cp .env.example .env

# Configure Python settings
PYTHON_EXECUTION_TIMEOUT=30000
PYTHON_MEMORY_LIMIT=512
PYTHON_SANDBOX_POOL_MIN=5
PYTHON_SANDBOX_POOL_MAX=20
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Create a Python Project

```typescript
import { createProject } from '@/lib/api/projects';

const project = await createProject({
  name: 'My Python Project',
  language: ProjectLanguage.PYTHON,
  framework: 'FastAPI'
});
```

## Security

### Code Sanitization

All Python code is automatically scanned for:
- Dangerous imports (os, subprocess, sys, shutil, etc.)
- Dangerous functions (eval, exec, compile, __import__, open)
- Infinite loops (while True, recursion without base case)
- Code length limits (< 100KB)
- UTF-8 encoding validation

### Sandboxing

- All code executes in E2B sandboxes
- Network access is controlled
- File system is isolated
- Memory limits enforced
- Execution timeouts enforced

## Monitoring

### Metrics

- Execution success rate
- Average execution time
- Sandbox pool utilization
- Cache hit rate
- Error rate by type

### Error Tracking

- Sentry integration for error tracking
- Dead letter queue for failed operations
- Automatic retry with exponential backoff
- Circuit breaker pattern for fault tolerance

## Documentation Index

- [API Reference](./api-reference.md)
- [Execution Engine](./execution-engine.md)
- [Security Model](./security.md)
- [Testing Guide](./testing.md)
- [Troubleshooting](./troubleshooting.md)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to Python support.

## License

MIT License - see [LICENSE](../LICENSE) for details.
