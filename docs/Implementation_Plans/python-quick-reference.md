<<<<<<< HEAD
# Python Support Implementation - Quick Reference Guide

**Version**: 2.0 Bulletproof
**Date**: 2025-01-16
**Status**: Ready for Implementation

---

## 📋 Document Index

1. **[python-support-implementation-plan.md](./python-support-implementation-plan.md)** - Original v1.0 plan
2. **[python-support-bulletproof-plan-v2.md](./python-support-bulletproof-plan-v2.md)** - Complete v2.0 specification (bulletproof)
3. **[python-todolist-hypergranular.md](./python-todolist-hypergranular.md)** - 487 detailed tasks
4. **This Document** - Quick reference guide

---

## 🎯 Executive Summary

### Goal
Add Python as a first-class language to DevilDev alongside JavaScript/TypeScript, enabling:
- ✅ Python code execution in isolated E2B sandboxes
- ✅ Python project scaffolding and management
- ✅ Automatic language detection
- ✅ PyPI package management
- ✅ Jupyter notebook support
- ✅ Python debugging capabilities

### Non-Negotiable Requirements
| Requirement | Target | Why It Matters |
|------------|--------|----------------|
| Security | Zero vulnerabilities | User code execution is high-risk |
| Uptime | 99.9% | Max 43 min downtime/month |
| Latency | p95 < 500ms | User experience |
| Test Coverage | 90%+ | Code quality |
| Idempotency | 100% operations | Reliability |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────┐
│              CLIENT (Browser/Mobile)            │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│          NEXT.JS APP (API Routes)               │
│  /api/python/execute  /api/python/packages      │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│         LANGUAGE ROUTER (Orchestration)          │
│    Detect → Validate → Route → Monitor          │
└─────┬──────────────────────┬────────────────────┘
      │                      │
      ▼                      ▼
┌──────────┐          ┌──────────────┐
│  Python  │          │  JS/TS       │
│  Adapter │          │  Adapter     │
└─────┬────┘          └──────┬───────┘
      │                      │
      ▼                      ▼
┌──────────┐          ┌──────────────┐
│  Python  │          │  Next.js     │
│  Sandbox │          │  Sandbox     │
│  (E2B)   │          │  (E2B)       │
└──────────┘          └──────────────┘
```

---

## 🔐 Security Model

### Threat Mitigations

| Threat | Mitigation |
|--------|-----------|
| Code Injection | AST sanitization, sandbox isolation |
| Sandbox Escape | E2B containers, resource limits |
| DoS | Rate limiting (10 req/min), timeouts |
| Data Exposure | User isolation, cache hashing |
| Package Attacks | Name validation, version pinning |

### Input Sanitization Rules

**Banned Imports:**
```python
os, subprocess, sys, shutil, pathlib, socket,
http, urllib, ftplib, telnetlib, pickle, shelve, marshal
```

**Banned Functions:**
```python
eval, exec, compile, __import__, open, file, input, raw_input
```

**Limits:**
- Max code length: 100KB
- Execution timeout: 30s (default), 300s (max)
- Memory limit: 512MB (default)

---

## 📊 Data Models

### Core Schema Changes

**Project Model Updates:**
```prisma
model Project {
  language  ProjectLanguage  @default(TYPESCRIPT)
  // ... existing fields
}

enum ProjectLanguage {
  TYPESCRIPT
  JAVASCRIPT
  PYTHON
}
```

**New Models:**
```prisma
model PythonPackage {
  id        String   @id @default(cuid())
  name      String
  version   String?
  projectId String
  createdAt DateTime @default(now())

  project   Project  @relation(fields: [projectId], references: [id])

  @@unique([projectId, name])
  @@index([projectId])
}

model DeadLetter {
  id           String   @id @default(cuid())
  operation    String
  userId       String
  projectId    String
  payload      Json
  error        String
  errorCode    String
  retryCount   Int      @default(0)
  lastAttemptAt DateTime
  createdAt    DateTime @default(now())
  status       String   @default("PENDING")

  @@index([userId, status])
  @@index([createdAt])
}
```

---

## 🚀 API Endpoints

### Execution API

**POST /api/python/execute**
```json
Request:
{
  "code": "print('Hello, World!')",
  "projectId": "clxxx...",
  "fileId": "clxxx...", // optional
  "timeout": 30000 // optional, ms
}

Response:
{
  "executionId": "exec-123",
  "success": true,
  "output": "Hello, World!\n",
  "executionTime": 125,
  "memoryUsage": 52428800,
  "sandboxId": "sb-xxx",
  "cached": false
}
```

### Package Management API

**GET /api/python/packages?projectId=xxx**
```json
Response:
{
  "packages": [
    {
      "id": "clxxx",
      "name": "numpy",
      "version": "1.26.4",
      "projectId": "clxxx"
    }
  ]
}
```

**POST /api/python/packages**
```json
Request:
{
  "packageName": "pandas",
  "version": "2.2.1", // optional
  "projectId": "clxxx"
}

Response:
{
  "success": true,
  "package": { /* package object */ }
}
```

**DELETE /api/python/packages/:id**
```json
Response:
{
  "success": true
}
```

### Sandbox Management API

**POST /api/python/sandbox/create**
```json
Request:
{
  "template": "devil-python-base",
  "projectId": "clxxx"
}

Response:
{
  "sandboxId": "sb-xxx",
  "url": "https://xxx.e2b.dev"
}
```

---

## 🔧 Key Components

### 1. Language Router
**Purpose:** Detect project language and route to appropriate adapter

**Location:** `src/lib/python/language-router.ts`

**Key Methods:**
- `route(request)` - Routes based on project language
- Validates user rate limits
- Checks project ownership
- Validates project state

### 2. Code Sanitizer
**Purpose:** Validate and sanitize Python code before execution

**Location:** `src/lib/python/sanitizer.ts`

**Key Methods:**
- `sanitize(code)` - Returns safe/unsafe with reasons
- `scanAST(tree)` - Detects dangerous patterns
- `detectInfiniteLoops(tree)` - Finds potential infinite loops

### 3. Sandbox Pool Manager
**Purpose:** Manage pool of reusable Python sandboxes

**Location:** `src/lib/python/sandbox-pool.ts`

**Configuration:**
- Min size: 5 sandboxes
- Max size: 20 sandboxes
- Idle timeout: 10 minutes
- Health check interval: 30 seconds

**Key Methods:**
- `acquire()` - Get sandbox from pool
- `release(instance)` - Return sandbox to pool
- `terminate(sandboxId)` - Destroy sandbox

### 4. Python Adapter
**Purpose:** Orchestrate Python code execution

**Location:** `src/lib/python/adapter.ts`

**Key Methods:**
- `execute(code, options)` - Execute Python code
- `installPackage(packageName)` - Install PyPI package
- `getMetrics()` - Get execution metrics

**Features:**
- Circuit breaker for resilience
- Result caching (TTL: 1 hour)
- Resource monitoring
- Automatic retry on transient failures

### 5. Circuit Breaker
**Purpose:** Prevent cascading failures

**Location:** `src/lib/circuit-breaker.ts`

**States:**
- CLOSED - Normal operation
- OPEN - Failing, reject requests
- HALF_OPEN - Testing recovery

**Configuration:**
- Failure threshold: 5
- Reset timeout: 60 seconds

### 6. Rate Limiter
**Purpose:** Prevent API abuse

**Location:** `src/lib/rate-limiter.ts`

**Configuration:**
- Limit: 10 requests per minute per user
- Window: 60 seconds
- Backend: Redis

---

## 📦 E2B Sandbox Templates

### Base Template
**ID:** `devil-python-base`
**Python:** 3.12
**Packages:** numpy, pandas, requests, python-dotenv, ipython
**Size:** ~150MB
**Startup Time:** ~2s

### Data Science Template
**ID:** `devil-python-datascience`
**Python:** 3.12
**Packages:** numpy, pandas, matplotlib, seaborn, scikit-learn, jupyter, scipy, plotly
**Size:** ~500MB
**Startup Time:** ~5s

### Web Framework Template
**ID:** `devil-python-web`
**Python:** 3.12
**Packages:** fastapi, uvicorn, flask, django, pydantic, httpx, websockets
**Size:** ~300MB
**Startup Time:** ~3s

---

## 🧪 Testing Strategy

### Test Coverage Targets
| Component | Coverage Target |
|-----------|----------------|
| Language Router | 95% |
| Code Sanitizer | 100% |
| Sandbox Pool | 90% |
| Python Adapter | 90% |
| API Routes | 85% |
| React Components | 80% |

### Test Types

**Unit Tests:**
- Framework: Jest
- Location: `__tests__/lib/python/`
- Run: `npm test -- __tests__/lib/python/`

**Integration Tests:**
- Framework: Supertest + Jest
- Location: `__tests__/api/python/`
- Run: `npm test -- __tests__/api/python/`

**E2E Tests:**
- Framework: Playwright
- Location: `tests/e2e/python/`
- Run: `npm run test:e2e`

**Load Tests:**
- Framework: k6
- Location: `tests/load/python/`
- Run: `k6 run tests/load/python/`

---

## 📈 Monitoring & Metrics

### Key Metrics

**Execution Metrics:**
```prometheus
python_execution_duration_seconds{status, template}
python_executions_total{status, template}
python_executions_active
```

**Resource Metrics:**
```prometheus
python_memory_usage_bytes{template}
python_sandbox_pool_size{state}
```

**Cache Metrics:**
```prometheus
python_cache_hits_total
python_cache_misses_total
```

### Logging

**Format:** JSON Lines
**Fields:**
- timestamp (ISO-8601 UTC)
- level (info, warn, error)
- msg
- correlation_id
- user_id
- project_id
- execution_id
- error (if applicable)

**Example:**
```json
{
  "timestamp": "2025-01-16T12:34:56.789Z",
  "level": "info",
  "msg": "Python execution completed",
  "correlation_id": "abc-123",
  "user_id": "user_456",
  "project_id": "proj_789",
  "execution_id": "exec_001",
  "success": true,
  "execution_time_ms": 1250
}
```

---

## 🚦 Implementation Phases

### Phase 1: Foundation (Week 1-2) - 127 Tasks
**Goal:** Establish Python runtime infrastructure

**Key Deliverables:**
- ✅ E2B sandbox templates built and deployed
- ✅ Database schema migrated
- ✅ Core libraries implemented (sanitizer, circuit breaker, rate limiter)
- ✅ Type definitions complete
- ✅ Probe scripts passing

**Exit Criteria:**
- [ ] All probe scripts passing
- [ ] Database migrations applied successfully
- [ ] Sandbox templates deployed to E2B
- [ ] Unit tests passing (>90% coverage)

### Phase 2: Core Functionality (Week 3-4) - 152 Tasks
**Goal:** Implement Python code execution and project management

**Key Deliverables:**
- ✅ Sandbox pool manager working
- ✅ Python adapter operational
- ✅ API routes implemented
- ✅ Server actions working
- ✅ Package management functional

**Exit Criteria:**
- [ ] Can execute Python code successfully
- [ ] Can create Python projects
- [ ] Can install PyPI packages
- [ ] API endpoints tested
- [ ] Integration tests passing

### Phase 3: Advanced Features (Week 5-6) - 124 Tasks
**Goal:** Add Python-specific tooling

**Key Deliverables:**
- ✅ React components (editor, package manager, templates)
- ✅ Jupyter notebook support
- ✅ Debugging integration
- ✅ Testing framework integration (pytest)

**Exit Criteria:**
- [ ] Python editor functional
- [ ] Package manager UI working
- [ ] Jupyter notebooks executing
- [ ] Debugging features working
- [ ] E2E tests passing

### Phase 4: Testing & Hardening (Week 7-8) - 84 Tasks
**Goal:** Production-ready

**Key Deliverables:**
- ✅ Comprehensive test suite (90%+ coverage)
- ✅ Performance benchmarks met
- ✅ Security audit passed
- ✅ Documentation complete

**Exit Criteria:**
- [ ] 90%+ test coverage
- [ ] Load tests passing
- [ ] Security scan clean
- [ ] Documentation complete
- [ ] Ready for beta rollout

---

## 🔑 Environment Variables

### Required
```bash
# E2B Configuration
E2B_API_KEY=your_e2b_api_key_here
E2B_PYTHON_TEMPLATE_ID=devil-python-base

# Python Execution Limits
PYTHON_EXECUTION_TIMEOUT=30000
PYTHON_MEMORY_LIMIT=512
PYTHON_SANDBOX_POOL_MIN=5
PYTHON_SANDBOX_POOL_MAX=20

# Redis (for caching & rate limiting)
REDIS_URL=redis://localhost:6379
```

### Optional
```bash
# Feature Flags
PYTHON_ENABLE_JUPYTER=true
PYTHON_ENABLE_DEBUGGING=true

# Monitoring
PYTHON_METRICS_ENABLED=true
PYTHON_TRACING_ENABLED=true
```

---

## 📚 Quick Commands

### Development
```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Run Python-specific tests
npm test -- --testPathPattern=python

# Run Python integration tests
npm test -- __tests__/api/python/

# Run load tests
k6 run tests/load/python/execution.js
```

### Database
```bash
# Generate migration
npx prisma migrate dev --name add_python_support

# Deploy migration
npx prisma migrate deploy

# Reset database (dev only)
npx prisma migrate reset

# Seed test data
npm run seed:python
```

### E2B Templates
```bash
# Build base template
e2b template build devil-python-base

# Push to registry
e2b template push devil-python-base

# Test template
e2b sandbox test devil-python-base

# List templates
e2b template list
```

---

## 🎯 Success Metrics

### Technical Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Execution Success Rate | >99% | - |
| Average Execution Time | <3s | - |
| p95 Latency | <500ms | - |
| Sandbox Startup Time | <5s | - |
| API Response Time | <500ms | - |
| Test Coverage | >90% | - |

### User Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Adoption Rate | 30% of projects | - |
| User Satisfaction (NPS) | >50 | - |
| Packages Installed/Project | >5 | - |
| Error Recovery Rate | >95% | - |

---

## 🚨 Common Issues & Solutions

### Issue: Sandbox Creation Fails
**Symptoms:** 503 errors, "SANDBOX_CREATION_FAILED"
**Causes:**
- E2B API key invalid
- Template not found
- E2B service down
**Solutions:**
1. Verify E2B_API_KEY
2. Check template ID
3. Check E2B status page

### Issue: Execution Timeout
**Symptoms:** 408 errors, "EXECUTION_TIMEOUT"
**Causes:**
- Code takes too long
- Infinite loop
- Resource exhaustion
**Solutions:**
1. Increase timeout limit
2. Check code for infinite loops
3. Optimize code

### Issue: Rate Limited
**Symptoms:** 429 errors, "RATE_LIMITED"
**Causes:**
- Too many requests
- Multiple concurrent sessions
**Solutions:**
1. Wait for reset time
2. Reduce request frequency
3. Upgrade plan for higher limits

### Issue: Package Install Fails
**Symptoms:** "PACKAGE_INSTALL_FAILED"
**Causes:**
- Invalid package name
- Network issues
- Dependency conflicts
**Solutions:**
1. Verify package name on PyPI
2. Check network connectivity
3. Try specific version

---

## 📞 Support & Resources

### Documentation
- [E2B Docs](https://e2b.dev/docs)
- [Next.js Server Actions](https://nextjs.org/docs/app/building-your-application/data-fetching/server-actions)
- [Prisma Docs](https://www.prisma.io/docs)
- [PyPI](https://pypi.org/)

### Internal Resources
- Implementation Plan v1.0: `python-support-implementation-plan.md`
- Bulletproof Spec v2.0: `python-support-bulletproof-plan-v2.md`
- Task List: `python-todolist-hypergranular.md`
- Federation Constitution: `../../CLAUDE.md`

### Team Contacts
- **Tech Lead**: [TBD]
- **Security Review**: [TBD]
- **Database Admin**: [TBD]
- **DevOps**: [TBD]

---

## ✅ Pre-Implementation Checklist

Before starting implementation, verify:

- [ ] E2B account set up with API key
- [ ] Redis instance available (for caching/rate limiting)
- [ ] PostgreSQL database accessible
- [ ] Development environment configured
- [ ] Team members assigned to phases
- [ ] Code review process defined
- [ ] Staging environment ready
- [ ] Monitoring tools configured
- [ ] Alert thresholds defined
- [ ] Rollback plan documented

---

## 🎬 Getting Started

1. **Review Documentation**
   - Read this quick reference
   - Read bulletproof plan v2.0
   - Review task list

2. **Set Up Environment**
   - Clone feature branch
   - Install dependencies
   - Configure .env.local

3. **Start with Phase 1**
   - Pick first task from hypergranular list
   - Complete task
   - Run tests
   - Mark task complete

4. **Track Progress**
   - Update task statuses daily
   - Commit frequently
   - Create PRs for logical groups
   - Request code reviews

5. **Stay Aligned**
   - Daily standups
   - Weekly phase reviews
   - Document decisions
   - Raise blockers early

---

**Good luck! 🚀**

*Remember: This is a bulletproof implementation. Every task, every validation, every test matters. Cut corners now, pay later.*

Last Updated: 2025-01-16
Version: 2.0
=======
# Python Support Implementation - Quick Reference Guide

**Version**: 2.0 Bulletproof
**Date**: 2025-01-16
**Status**: Ready for Implementation

---

## 📋 Document Index

1. **[python-support-implementation-plan.md](./python-support-implementation-plan.md)** - Original v1.0 plan
2. **[python-support-bulletproof-plan-v2.md](./python-support-bulletproof-plan-v2.md)** - Complete v2.0 specification (bulletproof)
3. **[python-todolist-hypergranular.md](./python-todolist-hypergranular.md)** - 487 detailed tasks
4. **This Document** - Quick reference guide

---

## 🎯 Executive Summary

### Goal
Add Python as a first-class language to DevilDev alongside JavaScript/TypeScript, enabling:
- ✅ Python code execution in isolated E2B sandboxes
- ✅ Python project scaffolding and management
- ✅ Automatic language detection
- ✅ PyPI package management
- ✅ Jupyter notebook support
- ✅ Python debugging capabilities

### Non-Negotiable Requirements
| Requirement | Target | Why It Matters |
|------------|--------|----------------|
| Security | Zero vulnerabilities | User code execution is high-risk |
| Uptime | 99.9% | Max 43 min downtime/month |
| Latency | p95 < 500ms | User experience |
| Test Coverage | 90%+ | Code quality |
| Idempotency | 100% operations | Reliability |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────┐
│              CLIENT (Browser/Mobile)            │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│          NEXT.JS APP (API Routes)               │
│  /api/python/execute  /api/python/packages      │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│         LANGUAGE ROUTER (Orchestration)          │
│    Detect → Validate → Route → Monitor          │
└─────┬──────────────────────┬────────────────────┘
      │                      │
      ▼                      ▼
┌──────────┐          ┌──────────────┐
│  Python  │          │  JS/TS       │
│  Adapter │          │  Adapter     │
└─────┬────┘          └──────┬───────┘
      │                      │
      ▼                      ▼
┌──────────┐          ┌──────────────┐
│  Python  │          │  Next.js     │
│  Sandbox │          │  Sandbox     │
│  (E2B)   │          │  (E2B)       │
└──────────┘          └──────────────┘
```

---

## 🔐 Security Model

### Threat Mitigations

| Threat | Mitigation |
|--------|-----------|
| Code Injection | AST sanitization, sandbox isolation |
| Sandbox Escape | E2B containers, resource limits |
| DoS | Rate limiting (10 req/min), timeouts |
| Data Exposure | User isolation, cache hashing |
| Package Attacks | Name validation, version pinning |

### Input Sanitization Rules

**Banned Imports:**
```python
os, subprocess, sys, shutil, pathlib, socket,
http, urllib, ftplib, telnetlib, pickle, shelve, marshal
```

**Banned Functions:**
```python
eval, exec, compile, __import__, open, file, input, raw_input
```

**Limits:**
- Max code length: 100KB
- Execution timeout: 30s (default), 300s (max)
- Memory limit: 512MB (default)

---

## 📊 Data Models

### Core Schema Changes

**Project Model Updates:**
```prisma
model Project {
  language  ProjectLanguage  @default(TYPESCRIPT)
  // ... existing fields
}

enum ProjectLanguage {
  TYPESCRIPT
  JAVASCRIPT
  PYTHON
}
```

**New Models:**
```prisma
model PythonPackage {
  id        String   @id @default(cuid())
  name      String
  version   String?
  projectId String
  createdAt DateTime @default(now())

  project   Project  @relation(fields: [projectId], references: [id])

  @@unique([projectId, name])
  @@index([projectId])
}

model DeadLetter {
  id           String   @id @default(cuid())
  operation    String
  userId       String
  projectId    String
  payload      Json
  error        String
  errorCode    String
  retryCount   Int      @default(0)
  lastAttemptAt DateTime
  createdAt    DateTime @default(now())
  status       String   @default("PENDING")

  @@index([userId, status])
  @@index([createdAt])
}
```

---

## 🚀 API Endpoints

### Execution API

**POST /api/python/execute**
```json
Request:
{
  "code": "print('Hello, World!')",
  "projectId": "clxxx...",
  "fileId": "clxxx...", // optional
  "timeout": 30000 // optional, ms
}

Response:
{
  "executionId": "exec-123",
  "success": true,
  "output": "Hello, World!\n",
  "executionTime": 125,
  "memoryUsage": 52428800,
  "sandboxId": "sb-xxx",
  "cached": false
}
```

### Package Management API

**GET /api/python/packages?projectId=xxx**
```json
Response:
{
  "packages": [
    {
      "id": "clxxx",
      "name": "numpy",
      "version": "1.26.4",
      "projectId": "clxxx"
    }
  ]
}
```

**POST /api/python/packages**
```json
Request:
{
  "packageName": "pandas",
  "version": "2.2.1", // optional
  "projectId": "clxxx"
}

Response:
{
  "success": true,
  "package": { /* package object */ }
}
```

**DELETE /api/python/packages/:id**
```json
Response:
{
  "success": true
}
```

### Sandbox Management API

**POST /api/python/sandbox/create**
```json
Request:
{
  "template": "devil-python-base",
  "projectId": "clxxx"
}

Response:
{
  "sandboxId": "sb-xxx",
  "url": "https://xxx.e2b.dev"
}
```

---

## 🔧 Key Components

### 1. Language Router
**Purpose:** Detect project language and route to appropriate adapter

**Location:** `src/lib/python/language-router.ts`

**Key Methods:**
- `route(request)` - Routes based on project language
- Validates user rate limits
- Checks project ownership
- Validates project state

### 2. Code Sanitizer
**Purpose:** Validate and sanitize Python code before execution

**Location:** `src/lib/python/sanitizer.ts`

**Key Methods:**
- `sanitize(code)` - Returns safe/unsafe with reasons
- `scanAST(tree)` - Detects dangerous patterns
- `detectInfiniteLoops(tree)` - Finds potential infinite loops

### 3. Sandbox Pool Manager
**Purpose:** Manage pool of reusable Python sandboxes

**Location:** `src/lib/python/sandbox-pool.ts`

**Configuration:**
- Min size: 5 sandboxes
- Max size: 20 sandboxes
- Idle timeout: 10 minutes
- Health check interval: 30 seconds

**Key Methods:**
- `acquire()` - Get sandbox from pool
- `release(instance)` - Return sandbox to pool
- `terminate(sandboxId)` - Destroy sandbox

### 4. Python Adapter
**Purpose:** Orchestrate Python code execution

**Location:** `src/lib/python/adapter.ts`

**Key Methods:**
- `execute(code, options)` - Execute Python code
- `installPackage(packageName)` - Install PyPI package
- `getMetrics()` - Get execution metrics

**Features:**
- Circuit breaker for resilience
- Result caching (TTL: 1 hour)
- Resource monitoring
- Automatic retry on transient failures

### 5. Circuit Breaker
**Purpose:** Prevent cascading failures

**Location:** `src/lib/circuit-breaker.ts`

**States:**
- CLOSED - Normal operation
- OPEN - Failing, reject requests
- HALF_OPEN - Testing recovery

**Configuration:**
- Failure threshold: 5
- Reset timeout: 60 seconds

### 6. Rate Limiter
**Purpose:** Prevent API abuse

**Location:** `src/lib/rate-limiter.ts`

**Configuration:**
- Limit: 10 requests per minute per user
- Window: 60 seconds
- Backend: Redis

---

## 📦 E2B Sandbox Templates

### Base Template
**ID:** `devil-python-base`
**Python:** 3.12
**Packages:** numpy, pandas, requests, python-dotenv, ipython
**Size:** ~150MB
**Startup Time:** ~2s

### Data Science Template
**ID:** `devil-python-datascience`
**Python:** 3.12
**Packages:** numpy, pandas, matplotlib, seaborn, scikit-learn, jupyter, scipy, plotly
**Size:** ~500MB
**Startup Time:** ~5s

### Web Framework Template
**ID:** `devil-python-web`
**Python:** 3.12
**Packages:** fastapi, uvicorn, flask, django, pydantic, httpx, websockets
**Size:** ~300MB
**Startup Time:** ~3s

---

## 🧪 Testing Strategy

### Test Coverage Targets
| Component | Coverage Target |
|-----------|----------------|
| Language Router | 95% |
| Code Sanitizer | 100% |
| Sandbox Pool | 90% |
| Python Adapter | 90% |
| API Routes | 85% |
| React Components | 80% |

### Test Types

**Unit Tests:**
- Framework: Jest
- Location: `__tests__/lib/python/`
- Run: `npm test -- __tests__/lib/python/`

**Integration Tests:**
- Framework: Supertest + Jest
- Location: `__tests__/api/python/`
- Run: `npm test -- __tests__/api/python/`

**E2E Tests:**
- Framework: Playwright
- Location: `tests/e2e/python/`
- Run: `npm run test:e2e`

**Load Tests:**
- Framework: k6
- Location: `tests/load/python/`
- Run: `k6 run tests/load/python/`

---

## 📈 Monitoring & Metrics

### Key Metrics

**Execution Metrics:**
```prometheus
python_execution_duration_seconds{status, template}
python_executions_total{status, template}
python_executions_active
```

**Resource Metrics:**
```prometheus
python_memory_usage_bytes{template}
python_sandbox_pool_size{state}
```

**Cache Metrics:**
```prometheus
python_cache_hits_total
python_cache_misses_total
```

### Logging

**Format:** JSON Lines
**Fields:**
- timestamp (ISO-8601 UTC)
- level (info, warn, error)
- msg
- correlation_id
- user_id
- project_id
- execution_id
- error (if applicable)

**Example:**
```json
{
  "timestamp": "2025-01-16T12:34:56.789Z",
  "level": "info",
  "msg": "Python execution completed",
  "correlation_id": "abc-123",
  "user_id": "user_456",
  "project_id": "proj_789",
  "execution_id": "exec_001",
  "success": true,
  "execution_time_ms": 1250
}
```

---

## 🚦 Implementation Phases

### Phase 1: Foundation (Week 1-2) - 127 Tasks
**Goal:** Establish Python runtime infrastructure

**Key Deliverables:**
- ✅ E2B sandbox templates built and deployed
- ✅ Database schema migrated
- ✅ Core libraries implemented (sanitizer, circuit breaker, rate limiter)
- ✅ Type definitions complete
- ✅ Probe scripts passing

**Exit Criteria:**
- [ ] All probe scripts passing
- [ ] Database migrations applied successfully
- [ ] Sandbox templates deployed to E2B
- [ ] Unit tests passing (>90% coverage)

### Phase 2: Core Functionality (Week 3-4) - 152 Tasks
**Goal:** Implement Python code execution and project management

**Key Deliverables:**
- ✅ Sandbox pool manager working
- ✅ Python adapter operational
- ✅ API routes implemented
- ✅ Server actions working
- ✅ Package management functional

**Exit Criteria:**
- [ ] Can execute Python code successfully
- [ ] Can create Python projects
- [ ] Can install PyPI packages
- [ ] API endpoints tested
- [ ] Integration tests passing

### Phase 3: Advanced Features (Week 5-6) - 124 Tasks
**Goal:** Add Python-specific tooling

**Key Deliverables:**
- ✅ React components (editor, package manager, templates)
- ✅ Jupyter notebook support
- ✅ Debugging integration
- ✅ Testing framework integration (pytest)

**Exit Criteria:**
- [ ] Python editor functional
- [ ] Package manager UI working
- [ ] Jupyter notebooks executing
- [ ] Debugging features working
- [ ] E2E tests passing

### Phase 4: Testing & Hardening (Week 7-8) - 84 Tasks
**Goal:** Production-ready

**Key Deliverables:**
- ✅ Comprehensive test suite (90%+ coverage)
- ✅ Performance benchmarks met
- ✅ Security audit passed
- ✅ Documentation complete

**Exit Criteria:**
- [ ] 90%+ test coverage
- [ ] Load tests passing
- [ ] Security scan clean
- [ ] Documentation complete
- [ ] Ready for beta rollout

---

## 🔑 Environment Variables

### Required
```bash
# E2B Configuration
E2B_API_KEY=your_e2b_api_key_here
E2B_PYTHON_TEMPLATE_ID=devil-python-base

# Python Execution Limits
PYTHON_EXECUTION_TIMEOUT=30000
PYTHON_MEMORY_LIMIT=512
PYTHON_SANDBOX_POOL_MIN=5
PYTHON_SANDBOX_POOL_MAX=20

# Redis (for caching & rate limiting)
REDIS_URL=redis://localhost:6379
```

### Optional
```bash
# Feature Flags
PYTHON_ENABLE_JUPYTER=true
PYTHON_ENABLE_DEBUGGING=true

# Monitoring
PYTHON_METRICS_ENABLED=true
PYTHON_TRACING_ENABLED=true
```

---

## 📚 Quick Commands

### Development
```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Run Python-specific tests
npm test -- --testPathPattern=python

# Run Python integration tests
npm test -- __tests__/api/python/

# Run load tests
k6 run tests/load/python/execution.js
```

### Database
```bash
# Generate migration
npx prisma migrate dev --name add_python_support

# Deploy migration
npx prisma migrate deploy

# Reset database (dev only)
npx prisma migrate reset

# Seed test data
npm run seed:python
```

### E2B Templates
```bash
# Build base template
e2b template build devil-python-base

# Push to registry
e2b template push devil-python-base

# Test template
e2b sandbox test devil-python-base

# List templates
e2b template list
```

---

## 🎯 Success Metrics

### Technical Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Execution Success Rate | >99% | - |
| Average Execution Time | <3s | - |
| p95 Latency | <500ms | - |
| Sandbox Startup Time | <5s | - |
| API Response Time | <500ms | - |
| Test Coverage | >90% | - |

### User Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Adoption Rate | 30% of projects | - |
| User Satisfaction (NPS) | >50 | - |
| Packages Installed/Project | >5 | - |
| Error Recovery Rate | >95% | - |

---

## 🚨 Common Issues & Solutions

### Issue: Sandbox Creation Fails
**Symptoms:** 503 errors, "SANDBOX_CREATION_FAILED"
**Causes:**
- E2B API key invalid
- Template not found
- E2B service down
**Solutions:**
1. Verify E2B_API_KEY
2. Check template ID
3. Check E2B status page

### Issue: Execution Timeout
**Symptoms:** 408 errors, "EXECUTION_TIMEOUT"
**Causes:**
- Code takes too long
- Infinite loop
- Resource exhaustion
**Solutions:**
1. Increase timeout limit
2. Check code for infinite loops
3. Optimize code

### Issue: Rate Limited
**Symptoms:** 429 errors, "RATE_LIMITED"
**Causes:**
- Too many requests
- Multiple concurrent sessions
**Solutions:**
1. Wait for reset time
2. Reduce request frequency
3. Upgrade plan for higher limits

### Issue: Package Install Fails
**Symptoms:** "PACKAGE_INSTALL_FAILED"
**Causes:**
- Invalid package name
- Network issues
- Dependency conflicts
**Solutions:**
1. Verify package name on PyPI
2. Check network connectivity
3. Try specific version

---

## 📞 Support & Resources

### Documentation
- [E2B Docs](https://e2b.dev/docs)
- [Next.js Server Actions](https://nextjs.org/docs/app/building-your-application/data-fetching/server-actions)
- [Prisma Docs](https://www.prisma.io/docs)
- [PyPI](https://pypi.org/)

### Internal Resources
- Implementation Plan v1.0: `python-support-implementation-plan.md`
- Bulletproof Spec v2.0: `python-support-bulletproof-plan-v2.md`
- Task List: `python-todolist-hypergranular.md`
- Federation Constitution: `../../CLAUDE.md`

### Team Contacts
- **Tech Lead**: [TBD]
- **Security Review**: [TBD]
- **Database Admin**: [TBD]
- **DevOps**: [TBD]

---

## ✅ Pre-Implementation Checklist

Before starting implementation, verify:

- [ ] E2B account set up with API key
- [ ] Redis instance available (for caching/rate limiting)
- [ ] PostgreSQL database accessible
- [ ] Development environment configured
- [ ] Team members assigned to phases
- [ ] Code review process defined
- [ ] Staging environment ready
- [ ] Monitoring tools configured
- [ ] Alert thresholds defined
- [ ] Rollback plan documented

---

## 🎬 Getting Started

1. **Review Documentation**
   - Read this quick reference
   - Read bulletproof plan v2.0
   - Review task list

2. **Set Up Environment**
   - Clone feature branch
   - Install dependencies
   - Configure .env.local

3. **Start with Phase 1**
   - Pick first task from hypergranular list
   - Complete task
   - Run tests
   - Mark task complete

4. **Track Progress**
   - Update task statuses daily
   - Commit frequently
   - Create PRs for logical groups
   - Request code reviews

5. **Stay Aligned**
   - Daily standups
   - Weekly phase reviews
   - Document decisions
   - Raise blockers early

---

**Good luck! 🚀**

*Remember: This is a bulletproof implementation. Every task, every validation, every test matters. Cut corners now, pay later.*

Last Updated: 2025-01-16
Version: 2.0
>>>>>>> 1cb9c5e35 (update)
