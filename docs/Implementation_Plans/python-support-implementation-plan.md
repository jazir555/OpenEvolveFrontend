<<<<<<< HEAD
# Python Support Implementation Plan for DevilDev

**Project**: DevilDev - Next.js 15 Code Development Platform
**Date**: 2025-01-16
**Status**: Planning Phase
**Priority**: High

---

## Executive Summary

This document outlines a comprehensive strategy to add Python language support to the existing Next.js/React-based DevilDev platform. The implementation follows the **"Air Gap" principle** from the Federation Constitution, maintaining strict separation between the core projects while providing seamless integration through adapters and orchestration layers.

### Current State
- **Framework**: Next.js 15 with App Router
- **Languages Supported**: TypeScript, JavaScript, Node.js
- **Code Execution**: E2B Sandbox (Next.js template: `261f355l6tv1xpmpg8o7`)
- **Architecture**: Server Actions + API Routes + Prisma ORM
- **Authentication**: Clerk
- **Background Jobs**: Inngest

### Goal
Add Python as a first-class language alongside JavaScript/TypeScript, enabling:
- Python code execution in isolated sandboxes
- Python project scaffolding and management
- Language detection and routing
- Python-specific tooling and package management
- Unified developer experience across languages

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Implementation Phases](#2-implementation-phases)
3. [Technical Specifications](#3-technical-specifications)
4. [Integration Points](#4-integration-points)
5. [Data Models & Schemas](#5-data-models--schemas)
6. [API Design](#6-api-design)
7. [Sandbox Configuration](#7-sandbox-configuration)
8. [Testing Strategy](#8-testing-strategy)
9. [Deployment Considerations](#9-deployment-considerations)
10. [Rollout Plan](#10-rollout-plan)

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DevilDev Frontend                        │
│                    (Next.js + React)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Language Router Layer                      │
│  (Detects language from file extensions, project config)    │
└────────┬────────────────────────────────┬───────────────────┘
         │                                │
         ▼                                ▼
┌──────────────────┐          ┌─────────────────────┐
│  JS/TS Adapter   │          │  Python Adapter     │
│  (Existing)      │          │  (New Implementation)│
└────────┬─────────┘          └──────────┬──────────┘
         │                                │
         ▼                                ▼
┌──────────────────┐          ┌─────────────────────┐
│  Next.js Sandbox │          │  Python Sandbox     │
│  (E2B Template)  │          │  (E2B Template)     │
└──────────────────┘          └─────────────────────┘
         │                                │
         └────────────┬───────────────────┘
                      ▼
         ┌────────────────────────┐
         │   Shared Services      │
         │  - Database (Prisma)   │
         │  - Auth (Clerk)        │
         │  - Jobs (Inngest)      │
         └────────────────────────┘
```

### 1.2 Core Principles

Following the **6 Immutable Laws** from CLAUDE.md:

1. **Air Gap**: Python runtime and tooling exist in isolated sandbox, no direct imports
2. **Runtime Truth**: All features validated via probe scripts before implementation
3. **Untouchable DB**: Read-only access through Prisma adapters
4. **Idempotency**: All operations safe to retry
5. **Configuration Explicitness**: All Python runtime config via environment variables
6. **UTC Time**: All timestamps in UTC ISO-8601

### 1.3 Directory Structure

```text
devildev/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   ├── python/                    # NEW: Python-specific API routes
│   │   │   │   ├── execute/route.ts       # Python code execution
│   │   │   │   ├── packages/route.ts      # PyPI package management
│   │   │   │   └── sandbox/route.ts       # Python sandbox management
│   │   │   └── ...
│   │   └── ...
│   ├── actions/
│   │   ├── python/                        # NEW: Python server actions
│   │   │   ├── execute.ts                 # Execute Python code
│   │   │   ├── create-project.ts          # Create Python project
│   │   │   ├── install-package.ts         # Install Python packages
│   │   │   └── analyze-dependencies.ts    # Analyze Python requirements
│   │   └── ...
│   ├── components/
│   │   ├── python/                        # NEW: Python-specific components
│   │   │   ├── PythonEditor.tsx           # Python code editor
│   │   │   ├── PythonPackageManager.tsx   # PyPI integration UI
│   │   │   ├── PythonOutput.tsx           # Python execution output
│   │   │   └── PythonTemplates.tsx        # Python project templates
│   │   └── ...
│   ├── lib/
│   │   ├── python/                        # NEW: Python library functions
│   │   │   ├── sandbox.ts                 # E2B Python sandbox client
│   │   │   ├── executor.ts                # Python code execution logic
│   │   │   ├── parser.ts                  # Python code parsing/analysis
│   │   │   └── templates.ts               # Python project templates
│   │   └── ...
│   ├── types/
│   │   ├── python.ts                      # NEW: Python-specific types
│   │   └── ...
│   └── ...
├── sandbox-templates/
│   ├── python/                            # NEW: Python sandbox templates
│   │   ├── base/                          # Base Python environment
│   │   │   ├── Dockerfile                 # Python runtime config
│   │   │   ├── requirements.txt           # Base dependencies
│   │   │   └── template.json              # E2B template config
│   │   ├── data-science/                  # Data Science template
│   │   │   ├── requirements.txt           # numpy, pandas, matplotlib
│   │   │   └── template.json
│   │   └── web/                           # Web framework template
│   │       ├── requirements.txt           # fastapi, flask, django
│   │       └── template.json
│   └── ...
├── prisma/
│   └── schema.prisma                      # UPDATED: Add Python support
└── ...
```

---

## 2. Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal**: Establish Python runtime infrastructure

**Tasks**:
1. Create Python sandbox template in E2B
2. Implement Python detection logic
3. Create base Python adapter
4. Set up Python-specific Prisma models
5. Create probe scripts for validation

**Deliverables**:
- Python sandbox template deployed
- Language router working
- Database migrations applied
- Probe scripts passing

### Phase 2: Core Functionality (Week 3-4)
**Goal**: Implement Python code execution and project management

**Tasks**:
1. Build Python code executor
2. Create Python project scaffolding
3. Implement package management (PyPI integration)
4. Build Python file templates
5. Create Python editor components

**Deliverables**:
- Working Python code execution
- Project creation for Python
- Package installation working
- UI components for Python

### Phase 3: Advanced Features (Week 5-6)
**Goal**: Add Python-specific tooling and integrations

**Tasks**:
1. Jupyter notebook support
2. Python debugging integration
3. Virtual environment management
4. Type hints and autocomplete
5. Python testing framework integration (pytest)

**Deliverables**:
- Jupyter notebook execution
- Debugging capabilities
- Virtual environment isolation
- Rich editor experience

### Phase 4: Testing & Hardening (Week 7-8)
**Goal**: Production-ready Python support

**Tasks**:
1. Comprehensive test suite
2. Performance optimization
3. Security hardening
4. Error handling improvements
5. Documentation completion

**Deliverables**:
- 90%+ test coverage
- Performance benchmarks
- Security audit passed
- Complete documentation

---

## 3. Technical Specifications

### 3.1 Python Sandbox Template

#### Base Template Configuration

```dockerfile
# sandbox-templates/python/base/Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python package management tools
RUN pip install --no-cache-dir \
    pip \
    setuptools \
    wheel \
    poetry \
    pipenv

# Create workspace directory
WORKDIR /workspace

# Copy base requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set up environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace

# Expose port for web frameworks
EXPOSE 8000

CMD ["python"]
```

```json
// sandbox-templates/python/base/template.json
{
  "name": "devil-python-base",
  "description": "Base Python environment for DevilDev",
  "language": "python",
  "version": "3.12.0",
  "packages": [
    "numpy",
    "pandas",
    "requests",
    "python-dotenv"
  ],
  "capabilities": [
    "code_execution",
    "package_management",
    "file_operations"
  ]
}
```

#### Specialized Templates

**Data Science Template**:
```text
# sandbox-templates/python/data-science/requirements.txt
numpy==1.26.4
pandas==2.2.1
matplotlib==3.8.3
seaborn==0.13.2
scikit-learn==1.4.1
jupyter==1.0.0
ipython==8.22.2
```

**Web Framework Template**:
```text
# sandbox-templates/python/web/requirements.txt
fastapi==0.110.0
uvicorn[standard]==0.27.0
flask==3.0.2
django==5.0.1
pydantic==2.6.3
httpx==0.27.0
```

### 3.2 Language Detection Logic

```typescript
// src/lib/language-detector.ts
export interface LanguageDetectionResult {
  language: 'typescript' | 'javascript' | 'python';
  confidence: number;
  reason: string;
}

export function detectProjectLanguage(
  files: { path: string; content: string }[]
): LanguageDetectionResult {
  const pythonFiles = files.filter(f =>
    f.path.endsWith('.py') ||
    f.path === 'requirements.txt' ||
    f.path === 'pyproject.toml' ||
    f.path === 'Pipfile'
  ).length;

  const jsFiles = files.filter(f =>
    f.path.endsWith('.ts') ||
    f.path.endsWith('.js') ||
    f.path.endsWith('.tsx') ||
    f.path.endsWith('.jsx') ||
    f.path === 'package.json'
  ).length;

  const total = files.length;

  if (pythonFiles > jsFiles) {
    return {
      language: 'python',
      confidence: pythonFiles / total,
      reason: `Found ${pythonFiles} Python files vs ${jsFiles} JS/TS files`
    };
  }

  return {
    language: jsFiles > 0 ? 'typescript' : 'javascript',
    confidence: jsFiles / total,
    reason: `Found ${jsFiles} JS/TS files vs ${pythonFiles} Python files`
  };
}
```

### 3.3 Python Code Executor

```typescript
// src/lib/python/executor.ts
import { E2BClient } from '@e2b/code-interpreter';

export interface PythonExecutionResult {
  success: boolean;
  output: string;
  error?: string;
  executionTime: number;
  memoryUsage?: number;
}

export class PythonExecutor {
  private client: E2BClient;
  private sandboxId: string | null = null;

  constructor(apiKey: string) {
    this.client = new E2BClient(apiKey);
  }

  async execute(
    code: string,
    template: string = 'devil-python-base'
  ): Promise<PythonExecutionResult> {
    const startTime = Date.now();

    try {
      // Get or create sandbox
      if (!this.sandboxId) {
        const sandbox = await this.client.sandbox.create(template);
        this.sandboxId = sandbox.sandboxId;
      }

      // Execute Python code
      const result = await this.client.sandbox.runCode(this.sandboxId, code, {
        language: 'python',
        timeout: 30000, // 30 seconds
      });

      const executionTime = Date.now() - startTime;

      return {
        success: !result.error,
        output: result.stdout || result.stderr,
        error: result.error,
        executionTime,
        memoryUsage: result.stats?.memory,
      };
    } catch (error) {
      return {
        success: false,
        output: '',
        error: error instanceof Error ? error.message : 'Unknown error',
        executionTime: Date.now() - startTime,
      };
    }
  }

  async installPackage(packageName: string): Promise<boolean> {
    if (!this.sandboxId) {
      throw new Error('No active sandbox');
    }

    try {
      await this.client.sandbox.runCode(
        this.sandboxId,
        `import subprocess; subprocess.run(['pip', 'install', '${packageName}'], check=True)`,
        { language: 'python' }
      );
      return true;
    } catch {
      return false;
    }
  }

  async cleanup(): Promise<void> {
    if (this.sandboxId) {
      await this.client.sandbox.kill(this.sandboxId);
      this.sandboxId = null;
    }
  }
}
```

---

## 4. Integration Points

### 4.1 Database Schema Updates

```prisma
// prisma/schema.prisma

// Updated Project model
model Project {
  id            String    @id @default(cuid())
  name          String
  description   String?
  userId        String
  language      ProjectLanguage @default(TYPESCRIPT)
  repositoryId  String?   @unique
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt

  // Relations
  user          User      @relation(fields: [userId], references: [id])
  repository    Repository? @relation(fields: [repositoryId], references: [id])
  files         File[]
  executions    Execution[]

  @@index([userId])
  @@index([repositoryId])
}

// New enum for project languages
enum ProjectLanguage {
  TYPESCRIPT
  JAVASCRIPT
  PYTHON
}

// Updated File model
model File {
  id          String   @id @default(cuid())
  projectId   String
  path        String
  content     String   @db.Text
  language    String   // Auto-detected from extension
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt

  project     Project  @relation(fields: [projectId], references: [id], onDelete: Cascade)

  @@index([projectId])
  @@unique([projectId, path])
}

// Updated Execution model
model Execution {
  id            String        @id @default(cuid())
  projectId     String
  fileId        String?
  language      ProjectLanguage
  code          String        @db.Text
  output        String?       @db.Text
  error         String?       @db.Text
  status        ExecutionStatus @default(PENDING)
  executionTime Int?          // in milliseconds
  memoryUsage   Int?          // in bytes
  createdAt     DateTime      @default(now())

  project       Project       @relation(fields: [projectId], references: [id], onDelete: Cascade)

  @@index([projectId])
  @@index([status])
}

// New PythonPackage model
model PythonPackage {
  id          String   @id @default(cuid())
  name        String
  version     String?
  projectId   String
  createdAt   DateTime @default(now())

  project     Project  @relation(fields: [projectId], references: [id], onDelete: Cascade)

  @@unique([projectId, name])
  @@index([projectId])
}

enum ExecutionStatus {
  PENDING
  RUNNING
  SUCCESS
  FAILED
  TIMEOUT
}
```

### 4.2 API Routes

#### Python Execution Endpoint

```typescript
// src/app/api/python/execute/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs';
import { PythonExecutor } from '@/lib/python/executor';
import { prisma } from '@/lib/prisma';

export async function POST(req: NextRequest) {
  const { userId } = auth();

  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const { code, projectId, fileId } = await req.json();

    // Verify project ownership
    const project = await prisma.project.findFirst({
      where: { id: projectId, userId },
    });

    if (!project || project.language !== 'PYTHON') {
      return NextResponse.json(
        { error: 'Invalid project or language' },
        { status: 400 }
      );
    }

    // Create execution record
    const execution = await prisma.execution.create({
      data: {
        projectId,
        fileId,
        language: 'PYTHON',
        code,
        status: 'RUNNING',
      },
    });

    // Execute code
    const executor = new PythonExecutor(process.env.E2B_API_KEY!);
    const result = await executor.execute(code);

    // Update execution record
    await prisma.execution.update({
      where: { id: execution.id },
      data: {
        output: result.output,
        error: result.error,
        status: result.success ? 'SUCCESS' : 'FAILED',
        executionTime: result.executionTime,
        memoryUsage: result.memoryUsage,
      },
    });

    await executor.cleanup();

    return NextResponse.json({
      executionId: execution.id,
      ...result,
    });
  } catch (error) {
    console.error('Python execution error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
```

#### Python Package Management Endpoint

```typescript
// src/app/api/python/packages/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs';
import { prisma } from '@/lib/prisma';
import { PythonExecutor } from '@/lib/python/executor';

export async function POST(req: NextRequest) {
  const { userId } = auth();

  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const { packageName, version, projectId } = await req.json();

    // Verify project ownership
    const project = await prisma.project.findFirst({
      where: { id: projectId, userId },
    });

    if (!project || project.language !== 'PYTHON') {
      return NextResponse.json(
        { error: 'Invalid project or language' },
        { status: 400 }
      );
    }

    // Install package in sandbox
    const executor = new PythonExecutor(process.env.E2B_API_KEY!);
    const packageSpec = version ? `${packageName}==${version}` : packageName;
    const success = await executor.installPackage(packageSpec);

    if (!success) {
      return NextResponse.json(
        { error: 'Failed to install package' },
        { status: 500 }
      );
    }

    // Record package in database
    const pythonPackage = await prisma.pythonPackage.upsert({
      where: {
        projectId_name: {
          projectId,
          name: packageName,
        },
      },
      create: {
        projectId,
        name: packageName,
        version,
      },
      update: {
        version,
      },
    });

    await executor.cleanup();

    return NextResponse.json({
      success: true,
      package: pythonPackage,
    });
  } catch (error) {
    console.error('Package installation error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
```

### 4.3 Server Actions

```typescript
// src/actions/python/execute.ts
'use server';

import { auth } from '@clerk/nextjs';
import { revalidatePath } from 'next/cache';
import { prisma } from '@/lib/prisma';
import { PythonExecutor } from '@/lib/python/executor';

export async function executePythonCode(formData: FormData) {
  const { userId } = auth();

  if (!userId) {
    throw new Error('Unauthorized');
  }

  const code = formData.get('code') as string;
  const projectId = formData.get('projectId') as string;

  // Verify project
  const project = await prisma.project.findFirst({
    where: { id: projectId, userId },
  });

  if (!project || project.language !== 'PYTHON') {
    throw new Error('Invalid project');
  }

  // Execute code
  const executor = new PythonExecutor(process.env.E2B_API_KEY!);
  const result = await executor.execute(code);
  await executor.cleanup();

  // Save execution
  await prisma.execution.create({
    data: {
      projectId,
      language: 'PYTHON',
      code,
      output: result.output,
      error: result.error,
      status: result.success ? 'SUCCESS' : 'FAILED',
      executionTime: result.executionTime,
    },
  });

  revalidatePath(`/project/${projectId}`);

  return result;
}
```

---

## 5. Data Models & Schemas

### 5.1 TypeScript Types

```typescript
// src/types/python.ts
export interface PythonProjectConfig {
  name: string;
  description?: string;
  template: 'base' | 'data-science' | 'web' | 'ml';
  pythonVersion: '3.10' | '3.11' | '3.12';
  packages: PythonPackage[];
}

export interface PythonPackage {
  name: string;
  version?: string;
  dependencies?: string[];
}

export interface PythonExecutionRequest {
  code: string;
  projectId: string;
  fileId?: string;
  timeout?: number;
  memoryLimit?: number;
}

export interface PythonExecutionResponse {
  executionId: string;
  success: boolean;
  output: string;
  error?: string;
  executionTime: number;
  memoryUsage?: number;
}

export interface PythonTemplate {
  id: string;
  name: string;
  description: string;
  thumbnail?: string;
  files: TemplateFile[];
  packages: PythonPackage[];
}

export interface TemplateFile {
  path: string;
  content: string;
  language: 'python' | 'text' | 'markdown' | 'yaml';
}
```

### 5.2 Zod Schemas

```typescript
// src/lib/schemas/python.ts
import { z } from 'zod';

export const pythonExecutionSchema = z.object({
  code: z.string().min(1, 'Code cannot be empty'),
  projectId: z.string().cuid(),
  fileId: z.string().cuid().optional(),
  timeout: z.number().min(1000).max(300000).optional(),
});

export const pythonPackageSchema = z.object({
  name: z.string().min(1).regex(/^[a-z][a-z0-9_-]*$/),
  version: z.string().optional(),
  projectId: z.string().cuid(),
});

export const pythonProjectSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(500).optional(),
  template: z.enum(['base', 'data-science', 'web', 'ml']),
  pythonVersion: z.enum(['3.10', '3.11', '3.12']),
});
```

---

## 6. API Design

### 6.1 REST API Endpoints

#### Python Execution
```
POST /api/python/execute
Body: { code, projectId, fileId?, timeout? }
Response: { executionId, success, output, error?, executionTime }
```

#### Package Management
```
POST /api/python/packages
Body: { packageName, version?, projectId }
Response: { success, package }

GET /api/python/packages?projectId={id}
Response: { packages: PythonPackage[] }

DELETE /api/python/packages/{packageId}
Response: { success }
```

#### Sandbox Management
```
POST /api/python/sandbox/create
Body: { template, projectId }
Response: { sandboxId, url }

POST /api/python/sandbox/terminate
Body: { sandboxId }
Response: { success }
```

#### Project Templates
```
GET /api/python/templates
Response: { templates: PythonTemplate[] }

POST /api/python/projects
Body: { name, description?, template, pythonVersion }
Response: { project }
```

### 6.2 WebSocket Events (Real-time Output)

```typescript
// Server-side WebSocket handler for streaming output
import { WebSocketServer } from 'ws';

wss.on('connection', (ws, req) => {
  const executionId = new URL(req.url!, `http://${req.headers.host}`)
    .searchParams.get('executionId');

  if (!executionId) {
    ws.close();
    return;
  }

  // Stream execution output
  streamExecutionOutput(executionId, (data) => {
    ws.send(JSON.stringify(data));
  });
});
```

---

## 7. Sandbox Configuration

### 7.1 E2B Template Setup

```bash
# Create Python base template
e2b template create devil-python-base

# Build template with Dockerfile
e2b template build devil-python-base --file sandbox-templates/python/base/Dockerfile

# Push template to E2B registry
e2b template push devil-python-base
```

### 7.2 Template Configuration

```json
{
  "templateID": "devil-python-base",
  "name": "DevilDev Python Base",
  "description": "Base Python 3.12 environment for DevilDev",
  "dockerfile": "sandbox-templates/python/base/Dockerfile",
  "port": 8000,
  "environmentVariables": {
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "/workspace"
  },
  "startCommand": "python",
  "mounts": [
    {
      "source": "/workspace",
      "target": "/workspace"
    }
  ]
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```typescript
// __tests__/lib/python/executor.test.ts
import { PythonExecutor } from '@/lib/python/executor';

describe('PythonExecutor', () => {
  let executor: PythonExecutor;

  beforeEach(() => {
    executor = new PythonExecutor(process.env.E2B_API_KEY!);
  });

  afterEach(async () => {
    await executor.cleanup();
  });

  test('should execute simple Python code', async () => {
    const result = await executor.execute('print("Hello, World!")');
    expect(result.success).toBe(true);
    expect(result.output).toContain('Hello, World!');
  });

  test('should handle Python syntax errors', async () => {
    const result = await executor.execute('print("missing quote)');
    expect(result.success).toBe(false);
    expect(result.error).toBeDefined();
  });

  test('should install Python packages', async () => {
    const success = await executor.installPackage('requests');
    expect(success).toBe(true);
  });
});
```

### 8.2 Integration Tests

```typescript
// __tests__/api/python/execute.test.ts
import { POST } from '@/app/api/python/execute/route';
import { NextRequest } from 'next/server';

describe('/api/python/execute', () => {
  test('should execute Python code', async () => {
    const request = new NextRequest('http://localhost:3000/api/python/execute', {
      method: 'POST',
      body: JSON.stringify({
        code: 'print("Test")',
        projectId: 'test-project-id',
      }),
      headers: {
        Authorization: 'Bearer test-token',
      },
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
  });
});
```

### 8.3 Contract Tests (Phase 2 - Proof of Work)

```bash
# glue/adapters/python/probes/check_api.sh
#!/bin/bash

# Probe script to validate Python sandbox API
echo "Testing Python Sandbox API..."

SANDBOX_ID=$(e2b sandbox create devil-python-base | jq -r '.sandboxId')

if [ -z "$SANDBOX_ID" ]; then
  echo "FAIL: Could not create sandbox"
  exit 1
fi

RESULT=$(e2b sandbox runCode $SANDBOX_ID 'print("Test")' | jq -r '.stdout')

if echo "$RESULT" | grep -q "Test"; then
  echo "PASS: Python execution working"
  e2b sandbox kill $SANDBOX_ID
  exit 0
else
  echo "FAIL: Python execution not working"
  e2b sandbox kill $SANDBOX_ID
  exit 1
fi
```

---

## 9. Deployment Considerations

### 9.1 Environment Variables

```bash
# .env.local - Python-specific configuration
E2B_API_KEY=your_e2b_api_key
E2B_PYTHON_TEMPLATE_ID=devil-python-base
PYTHON_EXECUTION_TIMEOUT=30000
PYTHON_MEMORY_LIMIT=512
PYTHON_MAX_PROCESSES=10
```

### 9.2 Database Migrations

```bash
# Generate migration for Python support
npx prisma migrate dev --name add_python_support

# Deploy migration to production
npx prisma migrate deploy
```

### 9.3 Monitoring & Logging

```typescript
// src/lib/python/logger.ts
import { logger } from '@/lib/logging';

export function logPythonExecution(
  executionId: string,
  userId: string,
  projectId: string,
  result: PythonExecutionResult
) {
  logger.info({
    msg: 'Python execution completed',
    event: 'python_execution',
    execution_id: executionId,
    user_id: userId,
    project_id: projectId,
    success: result.success,
    execution_time: result.executionTime,
    memory_usage: result.memoryUsage,
    timestamp: new Date().toISOString(),
  });
}
```

---

## 10. Rollout Plan

### 10.1 Phase 1: Alpha (Internal Testing)
- **Audience**: Internal team only
- **Scope**: Basic Python execution
- **Duration**: 2 weeks
- **Success Criteria**:
  - Python code executes successfully
  - Package installation works
  - No critical bugs

### 10.2 Phase 2: Beta (Limited Users)
- **Audience**: 10 selected users
- **Scope**: Full feature set
- **Duration**: 3 weeks
- **Success Criteria**:
  - 90%+ success rate for executions
  - Average execution time < 5 seconds
  - Positive user feedback

### 10.3 Phase 3: GA (General Availability)
- **Audience**: All users
- **Scope**: Production-ready
- **Duration**: Ongoing
- **Success Criteria**:
  - 99.9% uptime
  - < 1% error rate
  - Comprehensive documentation

---

## Success Metrics

### Technical Metrics
- **Execution Success Rate**: > 99%
- **Average Execution Time**: < 3 seconds
- **Sandbox Startup Time**: < 5 seconds
- **API Response Time**: < 500ms (p95)

### User Metrics
- **Adoption Rate**: % of projects using Python
- **User Satisfaction**: NPS score > 50
- **Feature Usage**: Packages installed per project
- **Error Recovery**: % of errors successfully handled

### Business Metrics
- **User Retention**: % of users returning after using Python
- **Project Creation**: # of Python projects created
- **Engagement**: Time spent in Python editor
- **Conversion**: % of Python users upgrading to paid plans

---

## Risks & Mitigations

### Risk 1: Sandbox Security Vulnerabilities
- **Mitigation**: Regular security audits, isolated containers, resource limits

### Risk 2: Performance Degradation
- **Mitigation**: Horizontal scaling, caching, connection pooling

### Risk 3: Package Installation Failures
- **Mitigation**: Pre-built templates, fallback packages, error handling

### Risk 4: User Adoption Lower Than Expected
- **Mitigation**: User research, beta testing, iterative improvements

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Create detailed task breakdown** for Phase 1
3. **Set up development environment** for Python testing
4. **Begin E2B template creation**
5. **Implement language detection** logic
6. **Start database schema** migration

---

## Appendix

### A. Python Version Support Matrix

| Version | Status | EOL Date | Support Level |
|---------|--------|----------|---------------|
| 3.12    | ✅ Recommended | 2028-10 | Full |
| 3.11    | ✅ Supported | 2027-10 | Full |
| 3.10    | ⚠️ Deprecated | 2026-10 | Maintenance |

### B. Package Repository Integration

- **PyPI**: Default package repository
- **conda**: Support for conda packages (future)
- **Private repos**: Support for private package repositories (future)

### C. Related Documentation

- [E2B Documentation](https://e2b.dev/docs)
- [Next.js Server Actions](https://nextjs.org/docs/app/building-your-application/data-fetching/server-actions)
- [Prisma Python](https://www.prisma.io/docs/reference/api-reference/prisma-client-python)
- [Clerk Authentication](https://clerk.com/docs)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-16
**Author**: Claude (Distinguished Engineer)
**Reviewers**: [Pending]
=======
# Python Support Implementation Plan for DevilDev

**Project**: DevilDev - Next.js 15 Code Development Platform
**Date**: 2025-01-16
**Status**: Planning Phase
**Priority**: High

---

## Executive Summary

This document outlines a comprehensive strategy to add Python language support to the existing Next.js/React-based DevilDev platform. The implementation follows the **"Air Gap" principle** from the Federation Constitution, maintaining strict separation between the core projects while providing seamless integration through adapters and orchestration layers.

### Current State
- **Framework**: Next.js 15 with App Router
- **Languages Supported**: TypeScript, JavaScript, Node.js
- **Code Execution**: E2B Sandbox (Next.js template: `261f355l6tv1xpmpg8o7`)
- **Architecture**: Server Actions + API Routes + Prisma ORM
- **Authentication**: Clerk
- **Background Jobs**: Inngest

### Goal
Add Python as a first-class language alongside JavaScript/TypeScript, enabling:
- Python code execution in isolated sandboxes
- Python project scaffolding and management
- Language detection and routing
- Python-specific tooling and package management
- Unified developer experience across languages

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Implementation Phases](#2-implementation-phases)
3. [Technical Specifications](#3-technical-specifications)
4. [Integration Points](#4-integration-points)
5. [Data Models & Schemas](#5-data-models--schemas)
6. [API Design](#6-api-design)
7. [Sandbox Configuration](#7-sandbox-configuration)
8. [Testing Strategy](#8-testing-strategy)
9. [Deployment Considerations](#9-deployment-considerations)
10. [Rollout Plan](#10-rollout-plan)

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DevilDev Frontend                        │
│                    (Next.js + React)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Language Router Layer                      │
│  (Detects language from file extensions, project config)    │
└────────┬────────────────────────────────┬───────────────────┘
         │                                │
         ▼                                ▼
┌──────────────────┐          ┌─────────────────────┐
│  JS/TS Adapter   │          │  Python Adapter     │
│  (Existing)      │          │  (New Implementation)│
└────────┬─────────┘          └──────────┬──────────┘
         │                                │
         ▼                                ▼
┌──────────────────┐          ┌─────────────────────┐
│  Next.js Sandbox │          │  Python Sandbox     │
│  (E2B Template)  │          │  (E2B Template)     │
└──────────────────┘          └─────────────────────┘
         │                                │
         └────────────┬───────────────────┘
                      ▼
         ┌────────────────────────┐
         │   Shared Services      │
         │  - Database (Prisma)   │
         │  - Auth (Clerk)        │
         │  - Jobs (Inngest)      │
         └────────────────────────┘
```

### 1.2 Core Principles

Following the **6 Immutable Laws** from CLAUDE.md:

1. **Air Gap**: Python runtime and tooling exist in isolated sandbox, no direct imports
2. **Runtime Truth**: All features validated via probe scripts before implementation
3. **Untouchable DB**: Read-only access through Prisma adapters
4. **Idempotency**: All operations safe to retry
5. **Configuration Explicitness**: All Python runtime config via environment variables
6. **UTC Time**: All timestamps in UTC ISO-8601

### 1.3 Directory Structure

```text
devildev/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   ├── python/                    # NEW: Python-specific API routes
│   │   │   │   ├── execute/route.ts       # Python code execution
│   │   │   │   ├── packages/route.ts      # PyPI package management
│   │   │   │   └── sandbox/route.ts       # Python sandbox management
│   │   │   └── ...
│   │   └── ...
│   ├── actions/
│   │   ├── python/                        # NEW: Python server actions
│   │   │   ├── execute.ts                 # Execute Python code
│   │   │   ├── create-project.ts          # Create Python project
│   │   │   ├── install-package.ts         # Install Python packages
│   │   │   └── analyze-dependencies.ts    # Analyze Python requirements
│   │   └── ...
│   ├── components/
│   │   ├── python/                        # NEW: Python-specific components
│   │   │   ├── PythonEditor.tsx           # Python code editor
│   │   │   ├── PythonPackageManager.tsx   # PyPI integration UI
│   │   │   ├── PythonOutput.tsx           # Python execution output
│   │   │   └── PythonTemplates.tsx        # Python project templates
│   │   └── ...
│   ├── lib/
│   │   ├── python/                        # NEW: Python library functions
│   │   │   ├── sandbox.ts                 # E2B Python sandbox client
│   │   │   ├── executor.ts                # Python code execution logic
│   │   │   ├── parser.ts                  # Python code parsing/analysis
│   │   │   └── templates.ts               # Python project templates
│   │   └── ...
│   ├── types/
│   │   ├── python.ts                      # NEW: Python-specific types
│   │   └── ...
│   └── ...
├── sandbox-templates/
│   ├── python/                            # NEW: Python sandbox templates
│   │   ├── base/                          # Base Python environment
│   │   │   ├── Dockerfile                 # Python runtime config
│   │   │   ├── requirements.txt           # Base dependencies
│   │   │   └── template.json              # E2B template config
│   │   ├── data-science/                  # Data Science template
│   │   │   ├── requirements.txt           # numpy, pandas, matplotlib
│   │   │   └── template.json
│   │   └── web/                           # Web framework template
│   │       ├── requirements.txt           # fastapi, flask, django
│   │       └── template.json
│   └── ...
├── prisma/
│   └── schema.prisma                      # UPDATED: Add Python support
└── ...
```

---

## 2. Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal**: Establish Python runtime infrastructure

**Tasks**:
1. Create Python sandbox template in E2B
2. Implement Python detection logic
3. Create base Python adapter
4. Set up Python-specific Prisma models
5. Create probe scripts for validation

**Deliverables**:
- Python sandbox template deployed
- Language router working
- Database migrations applied
- Probe scripts passing

### Phase 2: Core Functionality (Week 3-4)
**Goal**: Implement Python code execution and project management

**Tasks**:
1. Build Python code executor
2. Create Python project scaffolding
3. Implement package management (PyPI integration)
4. Build Python file templates
5. Create Python editor components

**Deliverables**:
- Working Python code execution
- Project creation for Python
- Package installation working
- UI components for Python

### Phase 3: Advanced Features (Week 5-6)
**Goal**: Add Python-specific tooling and integrations

**Tasks**:
1. Jupyter notebook support
2. Python debugging integration
3. Virtual environment management
4. Type hints and autocomplete
5. Python testing framework integration (pytest)

**Deliverables**:
- Jupyter notebook execution
- Debugging capabilities
- Virtual environment isolation
- Rich editor experience

### Phase 4: Testing & Hardening (Week 7-8)
**Goal**: Production-ready Python support

**Tasks**:
1. Comprehensive test suite
2. Performance optimization
3. Security hardening
4. Error handling improvements
5. Documentation completion

**Deliverables**:
- 90%+ test coverage
- Performance benchmarks
- Security audit passed
- Complete documentation

---

## 3. Technical Specifications

### 3.1 Python Sandbox Template

#### Base Template Configuration

```dockerfile
# sandbox-templates/python/base/Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python package management tools
RUN pip install --no-cache-dir \
    pip \
    setuptools \
    wheel \
    poetry \
    pipenv

# Create workspace directory
WORKDIR /workspace

# Copy base requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set up environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace

# Expose port for web frameworks
EXPOSE 8000

CMD ["python"]
```

```json
// sandbox-templates/python/base/template.json
{
  "name": "devil-python-base",
  "description": "Base Python environment for DevilDev",
  "language": "python",
  "version": "3.12.0",
  "packages": [
    "numpy",
    "pandas",
    "requests",
    "python-dotenv"
  ],
  "capabilities": [
    "code_execution",
    "package_management",
    "file_operations"
  ]
}
```

#### Specialized Templates

**Data Science Template**:
```text
# sandbox-templates/python/data-science/requirements.txt
numpy==1.26.4
pandas==2.2.1
matplotlib==3.8.3
seaborn==0.13.2
scikit-learn==1.4.1
jupyter==1.0.0
ipython==8.22.2
```

**Web Framework Template**:
```text
# sandbox-templates/python/web/requirements.txt
fastapi==0.110.0
uvicorn[standard]==0.27.0
flask==3.0.2
django==5.0.1
pydantic==2.6.3
httpx==0.27.0
```

### 3.2 Language Detection Logic

```typescript
// src/lib/language-detector.ts
export interface LanguageDetectionResult {
  language: 'typescript' | 'javascript' | 'python';
  confidence: number;
  reason: string;
}

export function detectProjectLanguage(
  files: { path: string; content: string }[]
): LanguageDetectionResult {
  const pythonFiles = files.filter(f =>
    f.path.endsWith('.py') ||
    f.path === 'requirements.txt' ||
    f.path === 'pyproject.toml' ||
    f.path === 'Pipfile'
  ).length;

  const jsFiles = files.filter(f =>
    f.path.endsWith('.ts') ||
    f.path.endsWith('.js') ||
    f.path.endsWith('.tsx') ||
    f.path.endsWith('.jsx') ||
    f.path === 'package.json'
  ).length;

  const total = files.length;

  if (pythonFiles > jsFiles) {
    return {
      language: 'python',
      confidence: pythonFiles / total,
      reason: `Found ${pythonFiles} Python files vs ${jsFiles} JS/TS files`
    };
  }

  return {
    language: jsFiles > 0 ? 'typescript' : 'javascript',
    confidence: jsFiles / total,
    reason: `Found ${jsFiles} JS/TS files vs ${pythonFiles} Python files`
  };
}
```

### 3.3 Python Code Executor

```typescript
// src/lib/python/executor.ts
import { E2BClient } from '@e2b/code-interpreter';

export interface PythonExecutionResult {
  success: boolean;
  output: string;
  error?: string;
  executionTime: number;
  memoryUsage?: number;
}

export class PythonExecutor {
  private client: E2BClient;
  private sandboxId: string | null = null;

  constructor(apiKey: string) {
    this.client = new E2BClient(apiKey);
  }

  async execute(
    code: string,
    template: string = 'devil-python-base'
  ): Promise<PythonExecutionResult> {
    const startTime = Date.now();

    try {
      // Get or create sandbox
      if (!this.sandboxId) {
        const sandbox = await this.client.sandbox.create(template);
        this.sandboxId = sandbox.sandboxId;
      }

      // Execute Python code
      const result = await this.client.sandbox.runCode(this.sandboxId, code, {
        language: 'python',
        timeout: 30000, // 30 seconds
      });

      const executionTime = Date.now() - startTime;

      return {
        success: !result.error,
        output: result.stdout || result.stderr,
        error: result.error,
        executionTime,
        memoryUsage: result.stats?.memory,
      };
    } catch (error) {
      return {
        success: false,
        output: '',
        error: error instanceof Error ? error.message : 'Unknown error',
        executionTime: Date.now() - startTime,
      };
    }
  }

  async installPackage(packageName: string): Promise<boolean> {
    if (!this.sandboxId) {
      throw new Error('No active sandbox');
    }

    try {
      await this.client.sandbox.runCode(
        this.sandboxId,
        `import subprocess; subprocess.run(['pip', 'install', '${packageName}'], check=True)`,
        { language: 'python' }
      );
      return true;
    } catch {
      return false;
    }
  }

  async cleanup(): Promise<void> {
    if (this.sandboxId) {
      await this.client.sandbox.kill(this.sandboxId);
      this.sandboxId = null;
    }
  }
}
```

---

## 4. Integration Points

### 4.1 Database Schema Updates

```prisma
// prisma/schema.prisma

// Updated Project model
model Project {
  id            String    @id @default(cuid())
  name          String
  description   String?
  userId        String
  language      ProjectLanguage @default(TYPESCRIPT)
  repositoryId  String?   @unique
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt

  // Relations
  user          User      @relation(fields: [userId], references: [id])
  repository    Repository? @relation(fields: [repositoryId], references: [id])
  files         File[]
  executions    Execution[]

  @@index([userId])
  @@index([repositoryId])
}

// New enum for project languages
enum ProjectLanguage {
  TYPESCRIPT
  JAVASCRIPT
  PYTHON
}

// Updated File model
model File {
  id          String   @id @default(cuid())
  projectId   String
  path        String
  content     String   @db.Text
  language    String   // Auto-detected from extension
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt

  project     Project  @relation(fields: [projectId], references: [id], onDelete: Cascade)

  @@index([projectId])
  @@unique([projectId, path])
}

// Updated Execution model
model Execution {
  id            String        @id @default(cuid())
  projectId     String
  fileId        String?
  language      ProjectLanguage
  code          String        @db.Text
  output        String?       @db.Text
  error         String?       @db.Text
  status        ExecutionStatus @default(PENDING)
  executionTime Int?          // in milliseconds
  memoryUsage   Int?          // in bytes
  createdAt     DateTime      @default(now())

  project       Project       @relation(fields: [projectId], references: [id], onDelete: Cascade)

  @@index([projectId])
  @@index([status])
}

// New PythonPackage model
model PythonPackage {
  id          String   @id @default(cuid())
  name        String
  version     String?
  projectId   String
  createdAt   DateTime @default(now())

  project     Project  @relation(fields: [projectId], references: [id], onDelete: Cascade)

  @@unique([projectId, name])
  @@index([projectId])
}

enum ExecutionStatus {
  PENDING
  RUNNING
  SUCCESS
  FAILED
  TIMEOUT
}
```

### 4.2 API Routes

#### Python Execution Endpoint

```typescript
// src/app/api/python/execute/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs';
import { PythonExecutor } from '@/lib/python/executor';
import { prisma } from '@/lib/prisma';

export async function POST(req: NextRequest) {
  const { userId } = auth();

  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const { code, projectId, fileId } = await req.json();

    // Verify project ownership
    const project = await prisma.project.findFirst({
      where: { id: projectId, userId },
    });

    if (!project || project.language !== 'PYTHON') {
      return NextResponse.json(
        { error: 'Invalid project or language' },
        { status: 400 }
      );
    }

    // Create execution record
    const execution = await prisma.execution.create({
      data: {
        projectId,
        fileId,
        language: 'PYTHON',
        code,
        status: 'RUNNING',
      },
    });

    // Execute code
    const executor = new PythonExecutor(process.env.E2B_API_KEY!);
    const result = await executor.execute(code);

    // Update execution record
    await prisma.execution.update({
      where: { id: execution.id },
      data: {
        output: result.output,
        error: result.error,
        status: result.success ? 'SUCCESS' : 'FAILED',
        executionTime: result.executionTime,
        memoryUsage: result.memoryUsage,
      },
    });

    await executor.cleanup();

    return NextResponse.json({
      executionId: execution.id,
      ...result,
    });
  } catch (error) {
    console.error('Python execution error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
```

#### Python Package Management Endpoint

```typescript
// src/app/api/python/packages/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs';
import { prisma } from '@/lib/prisma';
import { PythonExecutor } from '@/lib/python/executor';

export async function POST(req: NextRequest) {
  const { userId } = auth();

  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const { packageName, version, projectId } = await req.json();

    // Verify project ownership
    const project = await prisma.project.findFirst({
      where: { id: projectId, userId },
    });

    if (!project || project.language !== 'PYTHON') {
      return NextResponse.json(
        { error: 'Invalid project or language' },
        { status: 400 }
      );
    }

    // Install package in sandbox
    const executor = new PythonExecutor(process.env.E2B_API_KEY!);
    const packageSpec = version ? `${packageName}==${version}` : packageName;
    const success = await executor.installPackage(packageSpec);

    if (!success) {
      return NextResponse.json(
        { error: 'Failed to install package' },
        { status: 500 }
      );
    }

    // Record package in database
    const pythonPackage = await prisma.pythonPackage.upsert({
      where: {
        projectId_name: {
          projectId,
          name: packageName,
        },
      },
      create: {
        projectId,
        name: packageName,
        version,
      },
      update: {
        version,
      },
    });

    await executor.cleanup();

    return NextResponse.json({
      success: true,
      package: pythonPackage,
    });
  } catch (error) {
    console.error('Package installation error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
```

### 4.3 Server Actions

```typescript
// src/actions/python/execute.ts
'use server';

import { auth } from '@clerk/nextjs';
import { revalidatePath } from 'next/cache';
import { prisma } from '@/lib/prisma';
import { PythonExecutor } from '@/lib/python/executor';

export async function executePythonCode(formData: FormData) {
  const { userId } = auth();

  if (!userId) {
    throw new Error('Unauthorized');
  }

  const code = formData.get('code') as string;
  const projectId = formData.get('projectId') as string;

  // Verify project
  const project = await prisma.project.findFirst({
    where: { id: projectId, userId },
  });

  if (!project || project.language !== 'PYTHON') {
    throw new Error('Invalid project');
  }

  // Execute code
  const executor = new PythonExecutor(process.env.E2B_API_KEY!);
  const result = await executor.execute(code);
  await executor.cleanup();

  // Save execution
  await prisma.execution.create({
    data: {
      projectId,
      language: 'PYTHON',
      code,
      output: result.output,
      error: result.error,
      status: result.success ? 'SUCCESS' : 'FAILED',
      executionTime: result.executionTime,
    },
  });

  revalidatePath(`/project/${projectId}`);

  return result;
}
```

---

## 5. Data Models & Schemas

### 5.1 TypeScript Types

```typescript
// src/types/python.ts
export interface PythonProjectConfig {
  name: string;
  description?: string;
  template: 'base' | 'data-science' | 'web' | 'ml';
  pythonVersion: '3.10' | '3.11' | '3.12';
  packages: PythonPackage[];
}

export interface PythonPackage {
  name: string;
  version?: string;
  dependencies?: string[];
}

export interface PythonExecutionRequest {
  code: string;
  projectId: string;
  fileId?: string;
  timeout?: number;
  memoryLimit?: number;
}

export interface PythonExecutionResponse {
  executionId: string;
  success: boolean;
  output: string;
  error?: string;
  executionTime: number;
  memoryUsage?: number;
}

export interface PythonTemplate {
  id: string;
  name: string;
  description: string;
  thumbnail?: string;
  files: TemplateFile[];
  packages: PythonPackage[];
}

export interface TemplateFile {
  path: string;
  content: string;
  language: 'python' | 'text' | 'markdown' | 'yaml';
}
```

### 5.2 Zod Schemas

```typescript
// src/lib/schemas/python.ts
import { z } from 'zod';

export const pythonExecutionSchema = z.object({
  code: z.string().min(1, 'Code cannot be empty'),
  projectId: z.string().cuid(),
  fileId: z.string().cuid().optional(),
  timeout: z.number().min(1000).max(300000).optional(),
});

export const pythonPackageSchema = z.object({
  name: z.string().min(1).regex(/^[a-z][a-z0-9_-]*$/),
  version: z.string().optional(),
  projectId: z.string().cuid(),
});

export const pythonProjectSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(500).optional(),
  template: z.enum(['base', 'data-science', 'web', 'ml']),
  pythonVersion: z.enum(['3.10', '3.11', '3.12']),
});
```

---

## 6. API Design

### 6.1 REST API Endpoints

#### Python Execution
```
POST /api/python/execute
Body: { code, projectId, fileId?, timeout? }
Response: { executionId, success, output, error?, executionTime }
```

#### Package Management
```
POST /api/python/packages
Body: { packageName, version?, projectId }
Response: { success, package }

GET /api/python/packages?projectId={id}
Response: { packages: PythonPackage[] }

DELETE /api/python/packages/{packageId}
Response: { success }
```

#### Sandbox Management
```
POST /api/python/sandbox/create
Body: { template, projectId }
Response: { sandboxId, url }

POST /api/python/sandbox/terminate
Body: { sandboxId }
Response: { success }
```

#### Project Templates
```
GET /api/python/templates
Response: { templates: PythonTemplate[] }

POST /api/python/projects
Body: { name, description?, template, pythonVersion }
Response: { project }
```

### 6.2 WebSocket Events (Real-time Output)

```typescript
// Server-side WebSocket handler for streaming output
import { WebSocketServer } from 'ws';

wss.on('connection', (ws, req) => {
  const executionId = new URL(req.url!, `http://${req.headers.host}`)
    .searchParams.get('executionId');

  if (!executionId) {
    ws.close();
    return;
  }

  // Stream execution output
  streamExecutionOutput(executionId, (data) => {
    ws.send(JSON.stringify(data));
  });
});
```

---

## 7. Sandbox Configuration

### 7.1 E2B Template Setup

```bash
# Create Python base template
e2b template create devil-python-base

# Build template with Dockerfile
e2b template build devil-python-base --file sandbox-templates/python/base/Dockerfile

# Push template to E2B registry
e2b template push devil-python-base
```

### 7.2 Template Configuration

```json
{
  "templateID": "devil-python-base",
  "name": "DevilDev Python Base",
  "description": "Base Python 3.12 environment for DevilDev",
  "dockerfile": "sandbox-templates/python/base/Dockerfile",
  "port": 8000,
  "environmentVariables": {
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "/workspace"
  },
  "startCommand": "python",
  "mounts": [
    {
      "source": "/workspace",
      "target": "/workspace"
    }
  ]
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```typescript
// __tests__/lib/python/executor.test.ts
import { PythonExecutor } from '@/lib/python/executor';

describe('PythonExecutor', () => {
  let executor: PythonExecutor;

  beforeEach(() => {
    executor = new PythonExecutor(process.env.E2B_API_KEY!);
  });

  afterEach(async () => {
    await executor.cleanup();
  });

  test('should execute simple Python code', async () => {
    const result = await executor.execute('print("Hello, World!")');
    expect(result.success).toBe(true);
    expect(result.output).toContain('Hello, World!');
  });

  test('should handle Python syntax errors', async () => {
    const result = await executor.execute('print("missing quote)');
    expect(result.success).toBe(false);
    expect(result.error).toBeDefined();
  });

  test('should install Python packages', async () => {
    const success = await executor.installPackage('requests');
    expect(success).toBe(true);
  });
});
```

### 8.2 Integration Tests

```typescript
// __tests__/api/python/execute.test.ts
import { POST } from '@/app/api/python/execute/route';
import { NextRequest } from 'next/server';

describe('/api/python/execute', () => {
  test('should execute Python code', async () => {
    const request = new NextRequest('http://localhost:3000/api/python/execute', {
      method: 'POST',
      body: JSON.stringify({
        code: 'print("Test")',
        projectId: 'test-project-id',
      }),
      headers: {
        Authorization: 'Bearer test-token',
      },
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
  });
});
```

### 8.3 Contract Tests (Phase 2 - Proof of Work)

```bash
# glue/adapters/python/probes/check_api.sh
#!/bin/bash

# Probe script to validate Python sandbox API
echo "Testing Python Sandbox API..."

SANDBOX_ID=$(e2b sandbox create devil-python-base | jq -r '.sandboxId')

if [ -z "$SANDBOX_ID" ]; then
  echo "FAIL: Could not create sandbox"
  exit 1
fi

RESULT=$(e2b sandbox runCode $SANDBOX_ID 'print("Test")' | jq -r '.stdout')

if echo "$RESULT" | grep -q "Test"; then
  echo "PASS: Python execution working"
  e2b sandbox kill $SANDBOX_ID
  exit 0
else
  echo "FAIL: Python execution not working"
  e2b sandbox kill $SANDBOX_ID
  exit 1
fi
```

---

## 9. Deployment Considerations

### 9.1 Environment Variables

```bash
# .env.local - Python-specific configuration
E2B_API_KEY=your_e2b_api_key
E2B_PYTHON_TEMPLATE_ID=devil-python-base
PYTHON_EXECUTION_TIMEOUT=30000
PYTHON_MEMORY_LIMIT=512
PYTHON_MAX_PROCESSES=10
```

### 9.2 Database Migrations

```bash
# Generate migration for Python support
npx prisma migrate dev --name add_python_support

# Deploy migration to production
npx prisma migrate deploy
```

### 9.3 Monitoring & Logging

```typescript
// src/lib/python/logger.ts
import { logger } from '@/lib/logging';

export function logPythonExecution(
  executionId: string,
  userId: string,
  projectId: string,
  result: PythonExecutionResult
) {
  logger.info({
    msg: 'Python execution completed',
    event: 'python_execution',
    execution_id: executionId,
    user_id: userId,
    project_id: projectId,
    success: result.success,
    execution_time: result.executionTime,
    memory_usage: result.memoryUsage,
    timestamp: new Date().toISOString(),
  });
}
```

---

## 10. Rollout Plan

### 10.1 Phase 1: Alpha (Internal Testing)
- **Audience**: Internal team only
- **Scope**: Basic Python execution
- **Duration**: 2 weeks
- **Success Criteria**:
  - Python code executes successfully
  - Package installation works
  - No critical bugs

### 10.2 Phase 2: Beta (Limited Users)
- **Audience**: 10 selected users
- **Scope**: Full feature set
- **Duration**: 3 weeks
- **Success Criteria**:
  - 90%+ success rate for executions
  - Average execution time < 5 seconds
  - Positive user feedback

### 10.3 Phase 3: GA (General Availability)
- **Audience**: All users
- **Scope**: Production-ready
- **Duration**: Ongoing
- **Success Criteria**:
  - 99.9% uptime
  - < 1% error rate
  - Comprehensive documentation

---

## Success Metrics

### Technical Metrics
- **Execution Success Rate**: > 99%
- **Average Execution Time**: < 3 seconds
- **Sandbox Startup Time**: < 5 seconds
- **API Response Time**: < 500ms (p95)

### User Metrics
- **Adoption Rate**: % of projects using Python
- **User Satisfaction**: NPS score > 50
- **Feature Usage**: Packages installed per project
- **Error Recovery**: % of errors successfully handled

### Business Metrics
- **User Retention**: % of users returning after using Python
- **Project Creation**: # of Python projects created
- **Engagement**: Time spent in Python editor
- **Conversion**: % of Python users upgrading to paid plans

---

## Risks & Mitigations

### Risk 1: Sandbox Security Vulnerabilities
- **Mitigation**: Regular security audits, isolated containers, resource limits

### Risk 2: Performance Degradation
- **Mitigation**: Horizontal scaling, caching, connection pooling

### Risk 3: Package Installation Failures
- **Mitigation**: Pre-built templates, fallback packages, error handling

### Risk 4: User Adoption Lower Than Expected
- **Mitigation**: User research, beta testing, iterative improvements

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Create detailed task breakdown** for Phase 1
3. **Set up development environment** for Python testing
4. **Begin E2B template creation**
5. **Implement language detection** logic
6. **Start database schema** migration

---

## Appendix

### A. Python Version Support Matrix

| Version | Status | EOL Date | Support Level |
|---------|--------|----------|---------------|
| 3.12    | ✅ Recommended | 2028-10 | Full |
| 3.11    | ✅ Supported | 2027-10 | Full |
| 3.10    | ⚠️ Deprecated | 2026-10 | Maintenance |

### B. Package Repository Integration

- **PyPI**: Default package repository
- **conda**: Support for conda packages (future)
- **Private repos**: Support for private package repositories (future)

### C. Related Documentation

- [E2B Documentation](https://e2b.dev/docs)
- [Next.js Server Actions](https://nextjs.org/docs/app/building-your-application/data-fetching/server-actions)
- [Prisma Python](https://www.prisma.io/docs/reference/api-reference/prisma-client-python)
- [Clerk Authentication](https://clerk.com/docs)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-16
**Author**: Claude (Distinguished Engineer)
**Reviewers**: [Pending]
>>>>>>> 1cb9c5e35 (update)
