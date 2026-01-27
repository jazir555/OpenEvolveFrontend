# 🚀 BubbleLab Automation Guide for OpenEvolve

**Complete End-to-End Development Automation with Node-Based Workflows**

---

## 📋 Table of Contents

1. [Introduction](#1-introduction)
2. [Quick Start](#2-quick-start)
3. [Foundational Workflows](#3-foundational-workflows)
4. [Development Automation](#4-development-automation)
5. [CI/CD Integration](#5-cicd-integration)
6. [Monitoring & Alerting](#6-monitoring--alerting)
7. [Deployment Automation](#7-deployment-automation)
8. [Advanced Automation](#8-advanced-automation)
9. [Best Practices](#9-best-practices)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Introduction

### 1.1 What is BubbleLab?

**BubbleLab** is a TypeScript-native workflow automation platform built around a **node-based visual builder** that compiles workflows into clean, production-ready TypeScript code. Unlike traditional workflow tools that lock users into proprietary JSON formats, BubbleLab provides full code ownership, exportability, and type safety.

**Key Benefits for OpenEvolve:**
- 🎯 **Type-Safe Workflows**: Full TypeScript with Zod validation
- 🔧 **65+ Pre-built Bubbles**: Service integrations, tools, and workflows
- 🎨 **Visual Builder**: ReactFlow-based node editor with drag-and-drop
- 🤖 **AI-Powered Generation**: Two-phase AI (Coffee + Boba) for automatic code generation
- 📊 **Full Observability**: Structured logging with timing, costs, metrics
- 🔄 **Multiple Triggers**: Webhooks, cron schedules, Slack events, and more
- 💻 **Code Ownership**: Export clean, production-ready TypeScript
- 🔒 **Secure**: Credential management with encryption and access controls

### 1.2 Why BubbleLab for OpenEvolve?

The OpenEvolve project has:
- **100+ integrated components** requiring coordination
- **272 configurable parameters** across environments
- **Multiple development stages** (dev, staging, production)
- **Complex testing requirements** (2000+ unit tests, 500+ integration tests)
- **Distributed architecture** with anti-corruption layers

**BubbleLab fills critical automation gaps:**

| Gap | Current State | BubbleLab Solution |
|-----|---------------|-------------------|
| Monitoring | Manual health checks | Automated monitoring with alerts |
| Test Scheduling | Manual execution | Scheduled test runs with reporting |
| Deployment Coordination | GitHub Actions only | Orchestrate multi-stage deployments |
| Log Analysis | Manual inspection | Automated log parsing and alerting |
| Dependency Updates | Manual tracking | Automated monitoring with PR creation |
| Backup Validation | Basic scripts | Automated verification with testing |
| AI Integration | Manual setup | Pre-built AI agent bubbles |

### 1.3 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    BubbleLab Platform                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Visual       │  │ AI Code      │  │ TypeScript   │      │
│  │ Builder      │  │ Generator    │  │ Runtime      │      │
│  │ (ReactFlow)  │  │ (Coffee/Boba)│  │ (BubbleRun)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Development  │   │ CI/CD        │   │ Operations   │
│ Automation   │   │ Integration  │   │ Monitoring   │
└──────────────┘   └──────────────┘   └──────────────┘
```

### 1.4 Bubble System Overview

BubbleLab uses three types of bubbles (nodes):

**1. Service Bubbles** - External API integrations
- `AIAgentBubble` - OpenAI, Anthropic, Google Gemini
- `HttpBubble` - REST API calls
- `SlackBubble` - Slack integration
- `PostgreSQLBubble` - Database operations
- `GmailBubble`, `GoogleSheetsBubble`, `AirtableBubble`, etc.

**2. Tool Bubbles** - Utility functions
- `WebSearchTool`, `WebScrapeTool`, `WebCrawlTool`
- `RedditScrapeTool`, `InstagramTool`, `LinkedInTool`
- `GoogleMapsTool`, `YouTubeTool`, `TwitterTool`

**3. Workflow Bubbles** - Pre-built patterns
- `DatabaseAnalyzerWorkflow`
- `SlackNotifierWorkflow`
- `PDFFormOperationsWorkflow`

---

## 2. Quick Start

### 2.1 Installation & Setup

BubbleLab is already included in the OpenEvolve repository at `BubbleLab/`.

#### Start BubbleLab Services

```bash
# Navigate to BubbleLab directory
cd BubbleLab

# Install dependencies
pnpm install

# Start development services
pnpm dev

# This starts:
# - bubble-studio (UI): http://localhost:3000
# - bubblelab-api (Backend): http://localhost:3001
```

#### Access BubbleLab Studio

Open your browser and navigate to: `http://localhost:3000`

**First Time Setup:**
1. Sign up or log in (via Clerk authentication)
2. Configure credentials (Settings → Credentials)
3. Start creating workflows!

### 2.2 Your First BubbleFlow

Let's create a simple health check workflow using the visual builder:

#### Option A: Visual Builder (Recommended)

1. **Create New Flow**
   - Click "New Flow" in BubbleLab Studio
   - Name: "Health Check Monitor"

2. **Add Bubbles**
   - Drag `HttpBubble` node to canvas
   - Configure:
     - URL: `http://qdrant:6333/health`
     - Method: `GET`
   - Add another `HttpBubble` for PostgreSQL
   - Configure:
     - URL: `http://postgres:5432`
     - Method: `GET`

3. **Add AI Agent for Analysis**
   - Drag `AIAgentBubble` to canvas
   - Configure:
     - Model: `google/gemini-2.5-flash`
     - System Prompt: "Analyze health check results"
     - Message: Connect health check outputs

4. **Add Slack Notification**
   - Drag `SlackBubble` to canvas
   - Configure:
     - Webhook URL: Your Slack webhook
     - Message: Connect AI agent output

5. **Connect Bubbles**
   - Draw edges from HTTP bubbles → AI Agent → Slack
   - Click "Save"

6. **Activate Flow**
   - Toggle "Active" switch
   - Your workflow is now live!

#### Option B: Code-First (TypeScript)

Create a new BubbleFlow file:

```typescript
// health-check-workflow.ts
import { BubbleFlow, AIAgentBubble, HttpBubble, SlackBubble, type WebhookEvent } from '@bubblelab/bubble-core';

export interface HealthCheckOutput {
  qdrantStatus: string;
  postgresStatus: string;
  analysis: string;
  notified: boolean;
}

export class HealthCheckWorkflow extends BubbleFlow<'webhook/http'> {
  async handle(payload: WebhookEvent): Promise<HealthCheckOutput> {
    // Check Qdrant
    const qdrant = new HttpBubble({
      url: 'http://qdrant:6333/health',
      method: 'GET',
      timeout: 5000,
    });
    const qdrantResult = await qdrant.action();

    // Check PostgreSQL
    const postgres = new HttpBubble({
      url: 'http://postgres:5432',
      method: 'GET',
      timeout: 5000,
    });
    const postgresResult = await postgres.action();

    // Analyze with AI
    const agent = new AIAgentBubble({
      model: { model: 'google/gemini-2.5-flash' },
      systemPrompt: 'Analyze these health check results and report any issues',
      message: `Qdrant: ${qdrantResult.status}\nPostgreSQL: ${postgresResult.status}`,
    });
    const analysis = await agent.action();

    // Send Slack notification if issues
    const hasIssues = qdrantResult.status !== 200 || postgresResult.status !== 200;
    let notified = false;

    if (hasIssues) {
      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `⚠️ Health Check Alert\n\n${analysis.data.response}`,
      });
      await slack.action();
      notified = true;
    }

    return {
      qdrantStatus: qdrantResult.status === 200 ? 'healthy' : 'unhealthy',
      postgresStatus: postgresResult.status === 200 ? 'healthy' : 'unhealthy',
      analysis: analysis.data.response,
      notified,
    };
  }
}
```

**Deploy via API:**

```bash
curl -X POST "http://localhost:3001/bubble-flow" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "name": "Health Check Monitor",
    "description": "Monitors OpenEvolve services",
    "code": "$(cat health-check-workflow.ts)",
    "eventType": "webhook/http"
  }'
```

### 2.3 Using AI to Generate Workflows

BubbleLab's AI can generate workflows from natural language:

#### Two-Phase AI Generation

**Phase 1: Coffee (Planning)**

```bash
curl -X POST "http://localhost:3001/ai/coffee" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "Create a workflow that monitors OpenEvolve services and sends alerts to Slack"
  }'
```

**Response:**
```json
{
  "questions": [],
  "contextRequests": [],
  "plan": {
    "steps": [
      "Health check Qdrant via HTTP",
      "Health check PostgreSQL via HTTP",
      "Analyze results with AI agent",
      "Send Slack notification if issues found"
    ]
  }
}
```

**Phase 2: Boba (Implementation)**

```bash
curl -X POST "http://localhost:3001/ai/boba" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "Create a workflow that monitors OpenEvolve services and sends alerts to Slack"
  }'
```

**Response:**
```json
{
  "code": "import { BubbleFlow, HttpBubble, AIAgentBubble, SlackBubble }...",
  "explanation": "This workflow checks Qdrant and PostgreSQL health...",
  "validation": { "valid": true }
}
```

---

## 3. Foundational Workflows

### 3.1 Health Check Monitor

**Purpose:** Continuously monitor all OpenEvolve services and alert on failures.

**Trigger:** Webhook (can be called by external cron) or scheduled

**BubbleFlow Implementation:**

```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  type WebhookEvent
} from '@bubblelab/bubble-core';

export class HealthCheckMonitor extends BubbleFlow<'webhook/http'> {
  readonly name = 'Health Check Monitor';
  readonly description = 'Monitors OpenEvolve services and alerts on failures';

  async handle(payload: WebhookEvent): Promise<{
    healthy: boolean;
    services: Record<string, { status: number; healthy: boolean }>;
    analysis?: string;
  }> {
    // Define services to check
    const services = {
      qdrant: 'http://qdrant:6333/health',
      postgres: 'http://postgres:5432',
      redis: 'http://redis:6379/ping',
      api: 'http://openevolve-api:8000/health',
    };

    const results: Record<string, { status: number; healthy: boolean }> = {};

    // Check each service
    for (const [name, url] of Object.entries(services)) {
      try {
        const http = new HttpBubble({
          url,
          method: 'GET',
          timeout: 5000,
        });
        const result = await http.action();
        results[name] = {
          status: result.status,
          healthy: result.status === 200,
        };
      } catch (error) {
        results[name] = { status: 0, healthy: false };
      }
    }

    // Determine overall health
    const healthy = Object.values(results).every(r => r.healthy);

    // Analyze failures with AI
    let analysis;
    if (!healthy) {
      const agent = new AIAgentBubble({
        model: { model: 'google/gemini-2.5-flash' },
        systemPrompt: 'Analyze these health check results and provide a concise summary',
        message: JSON.stringify(results, null, 2),
      });
      const agentResult = await agent.action();
      analysis = agentResult.data.response;

      // Send alert to Slack
      const failedServices = Object.entries(results)
        .filter(([_, r]) => !r.healthy)
        .map(([name]) => name);

      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `⚠️ Health Check Failed\n\nServices: ${failedServices.join(', ')}\n\nAnalysis:\n${analysis}`,
      });
      await slack.action();
    }

    return { healthy, services, analysis };
  }
}
```

**Deploy:**

```typescript
// Using API
const response = await fetch('http://localhost:3001/bubble-flow', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  },
  body: JSON.stringify({
    name: 'Health Check Monitor',
    description: 'Monitors all OpenEvolve services',
    eventType: 'webhook/http',
    code: healthCheckCode,
  }),
});
```

**Trigger via Cron:**

```typescript
// Set up external cron to call webhook
// curl -X POST http://localhost:3001/webhook/USER_ID/PATH/health-check
```

### 3.2 Automated Test Runner

**Purpose:** Schedule and execute test suites, collect results, generate reports.

**Trigger:** Scheduled (cron)

```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  GmailBubble,
  type CronEvent
} from '@bubblelab/bubble-core';

export class AutomatedTestRunner extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '0 2 * * *'; // Daily at 2 AM
  readonly name = 'Automated Test Runner';
  readonly description = 'Runs test suites and generates reports';

  async handle(payload: CronEvent): Promise<{
    total: number;
    passed: number;
    failed: number;
    reportUrl?: string;
  }> {
    // Trigger tests via API
    const testApi = new HttpBubble({
      url: 'http://openevolve-api:8000/api/tests/run',
      method: 'POST',
      body: {
        type: 'full',
        environment: 'production',
      },
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const testResult = await testApi.action();

    // Parse results
    const results = JSON.parse(testResult.body);

    // Analyze with AI
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Analyze test results and provide a summary. Highlight any failures or regressions.',
      message: `Test Results:\n${JSON.stringify(results, null, 2)}`,
    });

    const analysis = await agent.action();

    // Generate HTML report
    const reportAgent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Generate an HTML test report from these results',
      message: JSON.stringify(results),
    });

    const report = await reportAgent.action();

    // Send report via email
    if (results.failed > 0) {
      const email = new GmailBubble({
        to: 'team@openevolve.com',
        subject: `Test Results: ${results.passed}/${results.total} passed`,
        body: `
          <h2>Test Execution Summary</h2>
          <p>Total: ${results.total}</p>
          <p>Passed: ${results.passed}</p>
          <p>Failed: ${results.failed}</p>

          <h3>AI Analysis</h3>
          <p>${analysis.data.response}</p>

          <h3>Detailed Report</h3>
          ${report.data.response}
        `,
      });

      await email.action();
    }

    return results;
  }
}
```

### 3.3 Infrastructure Orchestrator

**Purpose:** Automate infrastructure startup with health verification and service dependencies.

```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  type WebhookEvent
} from '@bubblelab/bubble-core';

export class InfrastructureOrchestrator extends BubbleFlow<'webhook/http'> {
  readonly name = 'Infrastructure Orchestrator';
  readonly description = 'Orchestrates service startup with health checks';

  async handle(payload: WebhookEvent & {
    action: 'start' | 'stop' | 'restart';
  }): Promise<{
    success: boolean;
    services: Record<string, string>;
  }> {
    const { action } = payload;

    // Start services in order
    const services = ['qdrant', 'postgres', 'redis', 'openevolve-api'];
    const results: Record<string, string> = {};

    for (const service of services) {
      // Start service (via Docker API or systemctl)
      const startService = new HttpBubble({
        url: `http://docker-engine/services/${service}/${action}`,
        method: 'POST',
      });

      await startService.action();

      // Wait for health check
      let healthy = false;
      let attempts = 0;

      while (!healthy && attempts < 30) {
        try {
          await new Promise(resolve => setTimeout(resolve, 2000));

          const healthCheck = new HttpBubble({
            url: `http://${service}:health`,
            method: 'GET',
            timeout: 3000,
          });

          const result = await healthCheck.action();
          healthy = result.status === 200;
        } catch (error) {
          attempts++;
        }
      }

      results[service] = healthy ? 'healthy' : 'unhealthy';

      if (!healthy) {
        // Send alert
        const slack = new SlackBubble({
          webhookUrl: process.env.SLACK_WEBHOOK_URL!,
          message: `❌ Failed to start ${service}`,
        });
        await slack.action();

        return { success: false, services: results };
      }
    }

    // All services started successfully
    const slack = new SlackBubble({
      webhookUrl: process.env.SLACK_WEBHOOK_URL!,
      message: `✅ Infrastructure ready\n\nServices: ${Object.keys(results).join(', ')}`,
    });
    await slack.action();

    return { success: true, services: results };
  }
}
```

---

## 4. Development Automation

### 4.1 Dependency Update Monitor

**Purpose:** Automatically monitor dependencies for updates and vulnerabilities.

```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  GmailBubble,
  type CronEvent
} from '@bubblelab/bubble-core';

export class DependencyUpdateMonitor extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '0 9 * * *'; // Daily at 9 AM
  readonly name = 'Dependency Update Monitor';

  async handle(payload: CronEvent): Promise<{
    critical: number;
    high: number;
    prCreated: boolean;
  }> {
    // Check Python dependencies
    const pipAudit = new HttpBubble({
      url: 'http://openevolve-api:8000/api/audit/python',
      method: 'GET',
    });

    const pythonResult = await pipAudit.action();
    const pythonVulns = JSON.parse(pythonResult.body);

    // Check Node.js dependencies
    const npmAudit = new HttpBubble({
      url: 'http://openevolve-api:8000/api/audit/npm',
      method: 'GET',
    });

    const npmResult = await npmAudit.action();
    const npmVulns = JSON.parse(npmResult.body);

    // Count critical vulnerabilities
    const criticalVulns = [
      ...pythonVulns.vulnerabilities.filter((v: any) => v.severity === 'critical'),
      ...npmVulns.vulnerabilities.filter((v: any) => v.severity === 'critical'),
    ];

    const highVulns = [
      ...pythonVulns.vulnerabilities.filter((v: any) => v.severity === 'high'),
      ...npmVulns.vulnerabilities.filter((v: any) => v.severity === 'high'),
    ];

    // If critical vulnerabilities found, create PR and alert
    let prCreated = false;

    if (criticalVulns.length > 0) {
      // Analyze with AI
      const agent = new AIAgentBubble({
        model: { model: 'openai/gpt-4' },
        systemPrompt: 'Analyze these security vulnerabilities and recommend fixes',
        message: JSON.stringify(criticalVulns, null, 2),
      });

      const analysis = await agent.action();

      // Create GitHub PR
      const github = new HttpBubble({
        url: `https://api.github.com/repos/openevolve/frontend/pulls`,
        method: 'POST',
        headers: {
          'Authorization': `token ${process.env.GITHUB_TOKEN}`,
          'Accept': 'application/vnd.github.v3+json',
        },
        body: {
          title: `Security: Fix ${criticalVulns.length} critical vulnerabilities`,
          body: analysis.data.response,
          head: 'automated/security-fix',
          base: 'main',
        },
      });

      await github.action();
      prCreated = true;

      // Send alerts
      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `🚨 Critical security vulnerabilities detected!\n\nCount: ${criticalVulns.length}\n\nPR created for review.`,
      });
      await slack.action();

      const email = new GmailBubble({
        to: 'security@openevolve.com',
        subject: '🚨 Critical Security Vulnerabilities',
        body: analysis.data.response,
      });
      await email.action();
    }

    return {
      critical: criticalVulns.length,
      high: highVulns.length,
      prCreated,
    };
  }
}
```

### 4.2 Code Quality Gatekeeper

**Purpose:** Enforce code quality standards before commits and in CI/CD.

```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  type WebhookEvent
} from '@bubblelab/bubble-core';

export class CodeQualityGatekeeper extends BubbleFlow<'webhook/http'> {
  readonly name = 'Code Quality Gatekeeper';

  async handle(payload: WebhookEvent & {
    files: string[];
    branch: string;
  }): Promise<{
    passed: boolean;
    issues: Array<{
      file: string;
      issue: string;
      severity: string;
    }>;
  }> {
    const { files, branch } = payload;
    const allIssues = [];

    // Run Black (Python formatter)
    const pythonFiles = files.filter(f => f.endsWith('.py'));
    if (pythonFiles.length > 0) {
      const black = new HttpBubble({
        url: 'http://openevolve-api:8000/api/lint/black',
        method: 'POST',
        body: { files: pythonFiles },
      });

      const blackResult = await black.action();
      const blackIssues = JSON.parse(blackResult.body);

      allIssues.push(...blackIssues.map((issue: any) => ({
        file: issue.file,
        issue: issue.message,
        severity: 'error',
      })));
    }

    // Run ESLint (JavaScript linter)
    const jsFiles = files.filter(f => f.match(/\.(js|jsx|ts|tsx)$/));
    if (jsFiles.length > 0) {
      const eslint = new HttpBubble({
        url: 'http://openevolve-api:8000/api/lint/eslint',
        method: 'POST',
        body: { files: jsFiles },
      });

      const eslintResult = await eslint.action();
      const eslintIssues = JSON.parse(eslintResult.body);

      allIssues.push(...eslintIssues.map((issue: any) => ({
        file: issue.filePath,
        issue: issue.message,
        severity: issue.severity,
      })));
    }

    // Run Bandit (security linter)
    if (pythonFiles.length > 0) {
      const bandit = new HttpBubble({
        url: 'http://openevolve-api:8000/api/lint/bandit',
        method: 'POST',
        body: { files: pythonFiles },
      });

      const banditResult = await bandit.action();
      const banditIssues = JSON.parse(banditResult.body);

      allIssues.push(...banditIssues.results.map((issue: any) => ({
        file: issue.filename,
        issue: issue.issue_text,
        severity: issue.issue_severity,
      })));
    }

    const passed = allIssues.length === 0;

    // If issues found, provide fixes via AI
    if (!passed) {
      const agent = new AIAgentBubble({
        model: { model: 'openai/gpt-4' },
        systemPrompt: 'Analyze these code quality issues and provide fix suggestions',
        message: JSON.stringify(allIssues, null, 2),
      });

      const fixes = await agent.action();

      // Send notification
      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `❌ Code Quality Check Failed\n\nBranch: ${branch}\nIssues: ${allIssues.length}\n\nFixes:\n${fixes.data.response}`,
      });
      await slack.action();
    }

    return { passed, issues: allIssues };
  }
}
```

### 4.3 Documentation Generator

**Purpose:** Automatically generate and update documentation from code.

```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  type WebhookEvent
} from '@bubblelab/bubble-core';

export class DocumentationGenerator extends BubbleFlow<'webhook/http'> {
  readonly name = 'Documentation Generator';

  async handle(payload: WebhookEvent & {
    commitHash: string;
    changedFiles: string[];
  }): Promise<{
    updated: boolean;
    docsGenerated: string[];
  }> {
    const { commitHash, changedFiles } = payload;

    // Extract API documentation from code
    const apiDoc = new HttpBubble({
      url: `http://openevolve-api:8000/docs/openapi.json`,
      method: 'GET',
    });

    const apiSpec = await apiDoc.action();

    // Generate documentation with AI
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Generate comprehensive API documentation from this OpenAPI spec',
      message: JSON.stringify(apiSpec.body, null, 2),
    });

    const generatedDocs = await agent.action();

    // Update documentation files
    const updateDocs = new HttpBubble({
      url: 'https://api.github.com/repos/openevolve/frontend/contents/docs/api',
      method: 'PUT',
      headers: {
        'Authorization': `token ${process.env.GITHUB_TOKEN}`,
        'Accept': 'application/vnd.github.v3+json',
      },
      body: {
        message: `docs: auto-update API documentation\n\nCommit: ${commitHash}`,
        content: Buffer.from(generatedDocs.data.response).toString('base64'),
        sha: '', // Current file SHA
      },
    });

    await updateDocs.action();

    return {
      updated: true,
      docsGenerated: ['api.md', 'openapi.json'],
    };
  }
}
```

---

## 5. CI/CD Integration

### 5.1 GitHub Actions Coordinator

```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  type WebhookEvent
} from '@bubblelab/bubble-core';

export class GitHubActionsCoordinator extends BubbleFlow<'webhook/http'> {
  readonly name = 'GitHub Actions Coordinator';

  async handle(payload: WebhookEvent & {
    eventType: 'push' | 'pull_request';
    branch: string;
    commitHash: string;
  }): Promise<{
    triggered: boolean;
    workflow: string;
  }> {
    const { eventType, branch, commitHash } = payload;

    // Route based on event type
    if (eventType === 'push' && branch === 'main') {
      // Trigger release workflow
      const release = new HttpBubble({
        url: `https://api.github.com/repos/openevolve/frontend/actions/workflows/deploy.yml/dispatches`,
        method: 'POST',
        headers: {
          'Authorization': `token ${process.env.GITHUB_TOKEN}`,
          'Accept': 'application/vnd.github.v3+json',
        },
        body: {
          ref: 'main',
          inputs: {
            commit_hash: commitHash,
            triggered_by: 'bubblelab',
          },
        },
      });

      await release.action();

      return { triggered: true, workflow: 'deploy.yml' };
    }

    if (eventType === 'pull_request') {
      // Trigger test workflow
      const test = new HttpBubble({
        url: `https://api.github.com/repos/openevolve/frontend/actions/workflows/test.yml/dispatches`,
        method: 'POST',
        headers: {
          'Authorization': `token ${process.env.GITHUB_TOKEN}`,
          'Accept': 'application/vnd.github.v3+json',
        },
        body: {
          ref: branch,
          inputs: {
            pr_number: payload.pr_number,
            triggered_by: 'bubblelab',
          },
        },
      });

      await test.action();

      return { triggered: true, workflow: 'test.yml' };
    }

    return { triggered: false, workflow: 'none' };
  }
}
```

### 5.2 Deployment Pipeline Orchestrator

```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  type WebhookEvent
} from '@bubblelab/bubble-core';

export class DeploymentPipelineOrchestrator extends BubbleFlow<'webhook/http'> {
  readonly name = 'Deployment Pipeline Orchestrator';

  async handle(payload: WebhookEvent & {
    environment: 'staging' | 'production';
    tag: string;
  }): Promise<{
    success: boolean;
    stage: string;
  }> {
    const { environment, tag } = payload;

    // Deploy to staging
    const deployStaging = new HttpBubble({
      url: `http://openevolve-api:8000/api/deploy/${environment}`,
      method: 'POST',
      body: { tag },
    });

    const stagingResult = await deployStaging.action();
    const stagingData = JSON.parse(stagingResult.body);

    if (!stagingData.success) {
      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `❌ Deployment to ${environment} failed`,
      });
      await slack.action();

      return { success: false, stage: 'staging' };
    }

    // Run smoke tests
    const smokeTests = new HttpBubble({
      url: `http://${environment}.openevolve.com/api/smoke-tests`,
      method: 'POST',
    });

    const testResult = await smokeTests.action();
    const testData = JSON.parse(testResult.body);

    if (!testData.passed) {
      // Rollback
      const rollback = new HttpBubble({
        url: `http://openevolve-api:8000/api/deploy/${environment}/rollback`,
        method: 'POST',
      });

      await rollback.action();

      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `❌ Smoke tests failed. Rolled back ${environment}.`,
      });
      await slack.action();

      return { success: false, stage: 'smoke-tests' };
    }

    // If staging, promote to production
    if (environment === 'staging') {
      const agent = new AIAgentBubble({
        model: { model: 'openai/gpt-4' },
        systemPrompt: 'Analyze these test results and recommend if production deployment is safe',
        message: JSON.stringify(testData, null, 2),
      });

      const analysis = await agent.action();

      // Trigger production deployment
      const deployProd = new HttpBubble({
        url: 'http://openevolve-api:8000/api/deploy/production',
        method: 'POST',
        body: { tag },
      });

      await deployProd.action();
    }

    const slack = new SlackBubble({
      webhookUrl: process.env.SLACK_WEBHOOK_URL!,
      message: `✅ Deployment to ${environment} successful!`,
    });
    await slack.action();

    return { success: true, stage: 'complete' };
  }
}
```

---

## 6. Monitoring & Alerting

### 6.1 Log Aggregation & Analysis

```typescript
import {
  BubbleFlow,
  PostgreSQLBubble,
  AIAgentBubble,
  SlackBubble,
  type CronEvent
} from '@bubblelab/bubble-core';

export class LogAggregationAnalyzer extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '* * * * *'; // Every minute
  readonly name = 'Log Aggregation & Analyzer';

  async handle(payload: CronEvent): Promise<{
    errorsFound: number;
    anomalies: number;
  }> {
    // Collect logs from services
    const postgres = new PostgreSQLBubble({
      connectionString: process.env.DATABASE_URL,
      query: `
        SELECT
          service,
          level,
          message,
          timestamp
        FROM logs
        WHERE timestamp > NOW() - INTERVAL '1 minute'
        ORDER BY timestamp DESC
      `,
    });

    const logs = await postgres.action();

    // Analyze for errors and anomalies
    const errors = logs.data.filter((log: any) => log.level === 'error');

    // Use AI to detect anomalies
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Analyze these log entries and detect anomalies, error spikes, or unusual patterns',
      message: JSON.stringify(logs.data.slice(0, 100), null, 2), // Last 100 logs
    });

    const analysis = await agent.action();
    const anomalyCount = (analysis.data.response.match(/anomaly/gi) || []).length;

    // Alert on critical issues
    if (errors.length > 10 || anomalyCount > 0) {
      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `⚠️ Log Analysis Alert\n\nErrors: ${errors.length}\nAnomalies: ${anomalyCount}\n\nAnalysis:\n${analysis.data.response}`,
      });
      await slack.action();
    }

    return {
      errorsFound: errors.length,
      anomalies: anomalyCount,
    };
  }
}
```

### 6.2 Performance Regression Detector

```typescript
import {
  BubbleFlow,
  HttpBubble,
  PostgreSQLBubble,
  AIAgentBubble,
  SlackBubble,
  type WebhookEvent
} from '@bubblelab/bubble-core';

export class PerformanceRegressionDetector extends BubbleFlow<'webhook/http'> {
  readonly name = 'Performance Regression Detector';

  async handle(payload: WebhookEvent & {
    commitHash: string;
  }): Promise<{
    regression: boolean;
    metrics: Record<string, number>;
  }> {
    // Run performance benchmarks
    const benchmarks = new HttpBubble({
      url: 'http://openevolve-api:8000/api/benchmarks/run',
      method: 'POST',
      body: {
        tests: ['api-response-time', 'db-query-time', 'memory-usage'],
      },
    });

    const benchmarkResults = await benchmarks.action();
    const metrics = JSON.parse(benchmarkResults.body);

    // Get baseline from database
    const postgres = new PostgreSQLBubble({
      connectionString: process.env.DATABASE_URL,
      query: `
        SELECT
          metric_name,
          avg_value as baseline
        FROM performance_baselines
        ORDER BY timestamp DESC
        LIMIT 10
      `,
    });

    const baselineResults = await postgres.action();

    // Compare with baseline
    const comparisons = Object.entries(metrics).map(([metric, value]) => {
      const baseline = baselineResults.data.find((b: any) => b.metric_name === metric);
      if (!baseline) return null;

      const percentChange = ((value as number - baseline.baseline) / baseline.baseline) * 100;

      return {
        metric,
        current: value,
        baseline: baseline.baseline,
        percentChange,
      };
    }).filter(Boolean);

    // Detect regressions (>10% slower)
    const regressions = comparisons.filter((c: any) => c.percentChange > 10);

    if (regressions.length > 0) {
      // Create GitHub issue
      const github = new HttpBubble({
        url: 'https://api.github.com/repos/openevolve/frontend/issues',
        method: 'POST',
        headers: {
          'Authorization': `token ${process.env.GITHUB_TOKEN}`,
          'Accept': 'application/vnd.github.v3+json',
        },
        body: {
          title: 'Performance Regression Detected',
          body: `The following metrics show degradation:\n\n${JSON.stringify(regressions, null, 2)}`,
          labels: ['performance', 'regression'],
        },
      });

      await github.action();

      // Send alert
      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `📉 Performance Regression Detected!\n\n${regressions.map((r: any) => `${r.metric}: +${r.percentChange.toFixed(1)}%`).join('\n')}`,
      });
      await slack.action();
    }

    return {
      regression: regressions.length > 0,
      metrics,
    };
  }
}
```

---

## 7. Deployment Automation

### 7.1 Backup Validator

```typescript
import {
  BubbleFlow,
  PostgreSQLBubble,
  HttpBubble,
  SlackBubble,
  type CronEvent
} from '@bubblelab/bubble-core';

export class BackupValidator extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '0 3 * * *'; // Daily at 3 AM
  readonly name = 'Backup Validator';

  async handle(payload: CronEvent): Promise<{
    valid: boolean;
    restoreTested: boolean;
  }> {
    // Get latest backup
    const listBackups = new HttpBubble({
      url: 'http://openevolve-api:8000/api/backups',
      method: 'GET',
    });

    const backupList = await listBackups.action();
    const latestBackup = JSON.parse(backupList.body)[0];

    // Verify backup integrity
    const verify = new HttpBubble({
      url: `http://openevolve-api:8000/api/backups/${latestBackup.id}/verify`,
      method: 'POST',
    });

    const verifyResult = await verify.action();
    const verification = JSON.parse(verifyResult.body);

    if (!verification.valid) {
      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `❌ Backup verification failed!\n\nBackup: ${latestBackup.id}`,
      });
      await slack.action();

      return { valid: false, restoreTested: false };
    }

    // Test restore to staging database
    const restore = new HttpBubble({
      url: `http://openevolve-api:8000/api/backups/${latestBackup.id}/restore`,
      method: 'POST',
      body: {
        targetDatabase: 'openevolve_test_restore',
      },
    });

    await restore.action();

    // Validate data
    const postgres = new PostgreSQLBubble({
      connectionString: process.env.TEST_DATABASE_URL,
      query: `
        SELECT
          schemaname,
          tablename,
          n_live_tup AS row_count
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
      `,
    });

    const validation = await postgres.action();

    // Compare row counts with production
    const prodPostgres = new PostgreSQLBubble({
      connectionString: process.env.DATABASE_URL,
      query: postgres.query,
    });

    const prodData = await prodPostgres.action();

    const rowCountsMatch = validation.data.length === prodData.data.length;

    // Cleanup test database
    const cleanup = new HttpBubble({
      url: 'http://openevolve-api:8000/api/databases/openevolve_test_restore',
      method: 'DELETE',
    });

    await cleanup.action();

    if (rowCountCountsMatch) {
      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `✅ Backup validation successful\n\nBackup: ${latestBackup.id}`,
      });
      await slack.action();
    }

    return {
      valid: verification.valid,
      restoreTested: rowCountsMatch,
    };
  }
}
```

---

## 8. Advanced Automation

### 8.1 Knowledge Base Sync

```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  type WebhookEvent
} from '@bubblelab/bubble-core';

export class KnowledgeBaseSync extends BubbleFlow<'webhook/http'> {
  readonly name = 'Knowledge Base Sync';

  async handle(payload: WebhookEvent & {
    changedDocuments: string[];
  }): Promise<{
    synced: number;
    embeddingsGenerated: number;
  }> {
    let syncedCount = 0;
    let embeddingsCount = 0;

    for (const docPath of payload.changedDocuments) {
      // Extract content
      const extract = new HttpBubble({
        url: 'http://openevolve-api:8000/api/documents/extract',
        method: 'POST',
        body: { path: docPath },
      });

      const extractResult = await extract.action();
      const content = JSON.parse(extractResult.body).content;

      // Generate embeddings
      const embed = new HttpBubble({
        url: 'http://openevolve-api:8000/api/embeddings/generate',
        method: 'POST',
        body: { text: content },
      });

      const embedResult = await embed.action();
      const embeddings = JSON.parse(embedResultResult.body);

      // Store in Qdrant
      const store = new HttpBubble({
        url: 'http://qdrant:6333/collections/openevolve_docs/points',
        method: 'PUT',
        body: {
          points: [{
            id: hash(docPath),
            vector: embeddings.vector,
            payload: {
              document_path: docPath,
              content: content,
              updated_at: new Date().toISOString(),
            },
          }],
        },
      });

      await store.action();
      syncedCount++;
      embeddingsCount++;
    }

    return {
      synced: syncedCount,
      embeddingsGenerated: embeddingsCount,
    };
  }
}
```

### 8.2 Security Compliance Monitor

```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  GmailBubble,
  type CronEvent
} from '@bubblelab/bubble-core';

export class SecurityComplianceMonitor extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '0 9 * * 1'; // Weekly on Monday at 9 AM
  readonly name = 'Security Compliance Monitor';

  async handle(payload: CronEvent): Promise<{
    compliant: boolean;
    issues: Array<{
      severity: string;
      issue: string;
    }>;
  }> {
    // Scan for vulnerabilities
    const vulnScan = new HttpBubble({
      url: 'http://openevolve-api:8000/api/security/scan',
      method: 'GET',
    });

    const scanResult = await vulnScan.action();
    const vulnerabilities = JSON.parse(scanResult.body);

    // Check license compliance
    const licenseCheck = new HttpBubble({
      url: 'http://openevolve-api:8000/api/licenses/check',
      method: 'GET',
    });

    const licenseResult = await licenseCheck.action();
    const licenses = JSON.parse(licenseResult.body);

    // Audit access logs
    const audit = new HttpBubble({
      url: 'http://openevolve-api:8000/api/audit/recent',
      method: 'GET',
    });

    const auditResult = await audit.action();
    const auditLogs = JSON.parse(auditResult.body);

    // Analyze with AI
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Analyze these security scan results and provide compliance assessment',
      message: JSON.stringify({
        vulnerabilities,
        licenses,
        auditLogs,
      }, null, 2),
    });

    const analysis = await agent.action();

    const criticalIssues = vulnerabilities.filter((v: any) => v.severity === 'critical');
    const compliant = criticalIssues.length === 0;

    if (!compliant) {
      // Send report
      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `🔒 Security Compliance Report\n\nCompliant: ${compliant ? '✅' : '❌'}\n\nCritical Issues: ${criticalIssues.length}\n\nFull Report:\n${analysis.data.response}`,
      });
      await slack.action();

      const email = new GmailBubble({
        to: 'security@openevolve.com',
        subject: 'Security Compliance Report',
        body: analysis.data.response,
      });
      await email.action();
    }

    return {
      compliant,
      issues: criticalIssues.map((v: any) => ({
        severity: v.severity,
        issue: v.description,
      })),
    };
  }
}
```

---

## 9. Best Practices

### 9.1 Workflow Design Principles

**1. Type Safety**
- Use TypeScript for all BubbleFlow definitions
- Leverage Zod schemas for validation
- Define clear input/output interfaces

**2. Error Handling**
- Always wrap bubble actions in try-catch
- Implement retry logic for transient failures
- Use AI agents for intelligent error recovery

**3. Observability**
- Add structured logging throughout
- Track metrics (timing, costs, token usage)
- Store execution history for debugging

**4. Modularity**
- Create reusable workflow bubbles
- Compose complex workflows from simple ones
- Follow single responsibility principle

### 9.2 Bubble Usage Best Practices

**Service Bubbles:**
- Test credentials before execution
- Use appropriate timeouts
- Handle rate limiting
- Cache responses when appropriate

**AI Agent Bubbles:**
- Provide clear system prompts
- Use appropriate models for tasks
- Track token usage and costs
- Validate AI outputs

**Tool Bubbles:**
- Combine with AI agents for best results
- Handle tool failures gracefully
- Validate tool outputs

### 9.3 Performance Optimization

**1. Parallel Execution**
```typescript
// Execute multiple bubbles in parallel
const [result1, result2, result3] = await Promise.all([
  bubble1.action(),
  bubble2.action(),
  bubble3.action(),
]);
```

**2. Caching**
```typescript
// Cache bubble results
const cacheKey = `bubble:${bubbleName}:${JSON.stringify(params)}`;
let result = await cache.get(cacheKey);

if (!result) {
  result = await bubble.action();
  await cache.set(cacheKey, result, 300); // 5 minutes
}
```

**3. Batch Operations**
```typescript
// Batch multiple operations
const batch = new HttpBubble({
  url: 'http://api.example.com/batch',
  method: 'POST',
  body: {
    operations: items.map(item => ({ action: 'create', data: item })),
  },
});
```

### 9.4 Security Best Practices

**1. Credential Management**
- Never hardcode credentials
- Use BubbleLab's credential system
- Rotate credentials regularly
- Test credentials before use

**2. Data Validation**
- Validate all inputs with Zod schemas
- Sanitize outputs before sending
- Encrypt sensitive data

**3. Access Control**
- Use authentication for webhook triggers
- Implement authorization checks
- Audit all workflow executions

---

## 10. Troubleshooting

### 10.1 Common Issues

**Issue 1: Bubble Not Found**
```
Error: Bubble 'http' not found in factory registry
```
**Solution:**
- Ensure bubble is imported from `@bubblelab/bubble-core`
- Check bubble name spelling
- Verify bubble is registered in factory

**Issue 2: Credential Errors**
```
Error: Invalid credentials for OPENAI_CRED
```
**Solution:**
- Verify credential is configured in BubbleLab
- Test credential in Settings → Credentials
- Check credential permissions

**Issue 3: Parsing Errors**
```
Error: Failed to parse BubbleFlow code
```
**Solution:**
- Validate TypeScript syntax
- Ensure all imports are correct
- Check for missing dependencies

### 10.2 Debugging Techniques

**1. Enable Debug Logging**
```typescript
const logger = new BubbleLogger('MyFlow', {
  minLevel: LogLevel.DEBUG,
  enableTiming: true,
});

logger.debug('Input:', payload);
logger.info('Processing...');
logger.warn('Unexpected value');
logger.error('Failed:', error);
```

**2. Inspect Bubble Parameters**
```typescript
// View parsed bubble parameters
const decomposition = generateDisplayedBubbleParameters(bubbleParameters);
console.log(JSON.stringify(decomposition, null, 2));
```

**3. Test Bubbles Individually**
```typescript
// Test bubble in isolation
const bubble = new HttpBubble({ url: 'http://example.com' });
const result = await bubble.action();
console.log(result);
```

### 10.3 Getting Help

- **BubbleLab Docs**: `BubbleLab/docs/`
- **API Documentation**: http://localhost:3001/docs
- **Community**: Slack channel
- **GitHub Issues**: https://github.com/bubblelab/issues

---

## Conclusion

BubbleLab provides a powerful, type-safe, and visually intuitive platform for automating the entire OpenEvolve development lifecycle. With its 65+ pre-built bubbles, AI-powered code generation, and full observability, teams can quickly create sophisticated automation workflows.

**Key Takeaways:**
1. Start with the visual builder for quick prototypes
2. Use code-first approach for complex workflows
3. Leverage AI generation (Coffee + Boba) for rapid development
4. Monitor everything with structured logging
5. Iterate continuously based on execution data

**Expected ROI:**
- **50-70% reduction** in manual operational tasks
- **Faster development** with AI assistance
- **Type-safe workflows** with full code ownership
- **Better visibility** into automation performance
- **Easier maintenance** with visual + code interfaces

Happy automating with BubbleLab! 🚀

---

## Appendix A: Complete Bubble Catalog

This appendix provides a comprehensive reference for all **51 bubbles** available in BubbleLab, organized by type and use case.

### Service Bubbles (21)

Service bubbles connect to external APIs and services. They handle authentication, data transformation, and error recovery.

#### 1. **ai-agent** `AIAgentBubble`
**Purpose**: Multi-model AI agent supporting OpenAI, Anthropic, Google, and local models

**Key Features:**
- Multi-model support (GPT-4, Claude, Gemini, LLaMA, etc.)
- Streaming responses
- Tool/function calling
- Custom system prompts
- Reasoning effort configuration
- Backup model fallback

**Example:**
```typescript
const agent = new AIAgentBubble({
  model: { model: 'anthropic/claude-sonnet-4-20250514' },
  systemPrompt: 'You are a DevOps expert analyzing infrastructure logs...',
  message: 'Analyze these logs for anomalies',
  reasoningEffort: 'high',  // Enable extended reasoning
  backupModel: 'google/gemini-2.5-flash',  // Fallback on error
  tools: ['web-search-tool', 'sql-query-tool']
});

const result = await agent.action();
// result.data.response contains AI response
```

**Credential Types:** `openai_api_key`, `anthropic_api_key`, `google_api_key`

**Use Cases:**
- Log analysis and anomaly detection
- Automated code review
- Natural language queries to databases
- Report generation
- Decision support systems

---

#### 2. **postgresql** `PostgreSQLBubble`
**Purpose**: Direct PostgreSQL database operations with connection pooling

**Key Features:**
- Parameterized queries (SQL injection safe)
- Connection pooling
- Transaction support
- Batch operations
- Query result streaming

**Example:**
```typescript
const db = new PostgreSQLBubble({
  query: 'SELECT * FROM users WHERE created_at > $1',
  params: ['2025-01-01'],
  connectionPool: { max: 10, idleTimeoutMillis: 30000 }
});

const result = await db.action();
// result.data.rows contains query results
// result.data.rowCount contains affected rows
```

**Credential Types:** `postgres_connection_string`

**Use Cases:**
- Health check queries
- Data validation
- Migration verification
- Analytics queries
- Cross-database synchronization

---

#### 3. **slack** `SlackBubble`
**Purpose**: Slack workspace integration for messaging and bot interactions

**Key Features:**
- Send messages to channels
- Direct messages
- File uploads
- Thread replies
- Block kit formatting
- Bot mention triggers

**Example:**
```typescript
const slack = new SlackBubble({
  channel: '#ops-alerts',
  text: 'Database backup completed successfully',
  blocks: [
    {
      type: 'section',
      text: { type: 'mrkdwn', text: '*Backup Summary*\n✅ PostgreSQL: 2.3GB\n✅ Redis: 150MB' }
    }
  ]
});

await slack.action();
```

**Credential Types:** `slack_bot_token`, `slack_signing_secret`

**Use Cases:**
- Alert notifications
- Deployment status updates
- Daily summaries
- Interactive bot commands
- Team collaboration

---

#### 4. **http** `HttpBubble`
**Purpose**: Generic HTTP/HTTPS request handler for REST APIs

**Key Features:**
- All HTTP methods (GET, POST, PUT, DELETE, PATCH)
- Custom headers
- Request/response transformation
- Retry logic with exponential backoff
- Timeout handling
- Basic auth, OAuth bearer tokens

**Example:**
```typescript
const http = new HttpBubble({
  url: 'https://api.github.com/repos/openevolve/frontend',
  method: 'GET',
  headers: {
    'Accept': 'application/vnd.github.v3+json',
    'User-Agent': 'OpenEvolve-BubbleLab'
  },
  timeout: 5000,
  retryConfig: { maxRetries: 3, backoffMultiplier: 2 }
});

const result = await http.action();
// result.data contains response body
```

**Use Cases:**
- Third-party API integration
- Webhook callbacks
- Health check endpoints
- Data fetching
- Microservice communication

---

#### 5. **apify** `ApifyBubble`
**Purpose**: Web scraping and automation using Apify actors

**Key Features:**
- 1000+ pre-built actors
- Dataset management
- Scheduled runs
- Proxy rotation
- CAPTCHA solving

**Example:**
```typescript
const scraper = new ApifyBubble({
  actorId: 'apify/web-scraper',
  input: {
    url: 'https://example.com',
    selectors: { title: 'h1', description: '.desc' }
  }
});

const result = await scraper.action();
// result.data.items contains scraped data
```

**Credential Types:** `apify_api_token`

**Use Cases:**
- Competitive price monitoring
- Content aggregation
- Lead generation
- Social media scraping
- SEO monitoring

---

#### 6. **github** `GithubBubble`
**Purpose**: GitHub repository and workflow automation

**Key Features:**
- Repository operations
- Issue/PR management
- Workflow triggering
- Release management
- Status checks

**Example:**
```typescript
const github = new GithubBubble({
  owner: 'openevolve',
  repo: 'frontend',
  action: 'create_issue',
  title: 'Automated issue from BubbleLab',
  body: 'This issue was created automatically',
  labels: ['automation', 'bug']
});

await github.action();
```

**Credential Types:** `github_pat`

**Use Cases:**
- Automated issue creation from alerts
- PR status updates
- Release automation
- Repository metrics
- Security vulnerability tracking

---

#### 7. **google-drive** `GoogleDriveBubble`
**Purpose**: Google Drive file and folder operations

**Key Features:**
- File upload/download
- Folder creation
- Search operations
- Permission management
- Team drive support

**Example:**
```typescript
const drive = new GoogleDriveBubble({
  operation: 'upload',
  fileName: 'report.pdf',
  folderId: '1a2b3c4d...',
  fileData: base64EncodedData
});

await drive.action();
```

**Credential Types:** `google_oauth_token`, `google_service_account`

**Use Cases:**
- Automated report archiving
- Backup synchronization
- Document sharing
- Folder organization

---

#### 8. **gmail** `GmailBubble`
**Purpose**: Gmail email operations

**Key Features:**
- Send emails
- Search/filter
- Label management
- Thread operations
- Attachment handling

**Example:**
```typescript
const gmail = new GmailBubble({
  to: 'team@openevolve.com',
  subject: 'Daily Build Report',
  body: 'Build passed all tests',
  attachments: ['report.pdf']
});

await gmail.action();
```

**Credential Types:** `google_oauth_token`, `google_service_account`

**Use Cases:**
- Automated notifications
- Report delivery
- Email parsing
- Newsletter management

---

#### 9. **google-sheets** `GoogleSheetsBubble`
**Purpose**: Google Sheets spreadsheet operations

**Key Features:**
- Read/write cells
- Batch operations
- Sheet creation
- Formula application
- Pivot tables

**Example:**
```typescript
const sheets = new GoogleSheetsBubble({
  spreadsheetId: '1AbCdEf...',
  range: 'Sheet1!A1:D10',
  operation: 'update',
  values: [['Name', 'Score'], ['Alice', 95]]
});

await sheets.action();
```

**Credential Types:** `google_oauth_token`, `google_service_account`

**Use Cases:**
- Data collection forms
- Report generation
- Analytics dashboards
- Collaborative editing

---

#### 10. **google-calendar** `GoogleCalendarBubble`
**Purpose**: Google Calendar event management

**Key Features:**
- Create/update/delete events
- Recurring events
- Meeting scheduling
- Reminders
- Calendar sharing

**Example:**
```typescript
const calendar = new GoogleCalendarBubble({
  summary: 'Team Standup',
  start: { dateTime: '2025-01-18T10:00:00Z' },
  end: { dateTime: '2025-01-18T10:30:00Z' },
  attendees: ['alice@example.com', 'bob@example.com'],
  recurrence: ['RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR']
});

await calendar.action();
```

**Credential Types:** `google_oauth_token`, `google_service_account`

**Use Cases:**
- Meeting scheduling
- Release planning
- Reminder automation
- Resource booking

---

#### 11. **notion** `NotionBubble`
**Purpose**: Notion workspace integration

**Key Features:**
- Page creation/editing
- Database operations
- Block manipulation
- Comment handling
- Template usage

**Example:**
```typescript
const notion = new NotionBubble({
  operation: 'create_page',
  parent: { database_id: 'xyz789' },
  properties: {
    Name: { title: [{ text: { content: 'New Task' } }] },
    Status: { select: { name: 'In Progress' } }
  }
});

await notion.action();
```

**Credential Types:** `notion_api_key`

**Use Cases:**
- Documentation automation
- Task management
- Knowledge base updates
- Meeting notes

---

#### 12. **airtable** `AirtableBubble`
**Purpose**: Airtable database operations

**Key Features:**
- CRUD operations
- Bulk record updates
- View filtering
- Attachment handling
- Form submissions

**Example:**
```typescript
const airtable = new AirtableBubble({
  baseId: 'app123',
  tableId: 'tbl456',
  operation: 'create',
  record: {
    Name: 'John Doe',
    Email: 'john@example.com',
    Status: 'Active'
  }
});

await airtable.action();
```

**Credential Types:** `airtable_pat`

**Use Cases:**
- CRM operations
- Project tracking
- Content management
- Survey responses

---

#### 13. **firecrawl** `FirecrawlBubble`
**Purpose:** Advanced web crawling with LLM extraction

**Key Features:**
- Full-page crawling
- LLM-powered extraction
- Sitemap parsing
- JavaScript rendering
- Rate limiting

**Example:**
```typescript
const firecrawl = new FirecrawlBubble({
  url: 'https://openevolve.com/docs',
  mode: 'crawl',
  extractorOptions: {
    mode: 'llm-extraction',
    prompt: 'Extract all API endpoints and their descriptions'
  }
});

const result = await firecrawl.action();
```

**Credential Types:** `firecrawl_api_key`

**Use Cases:**
- Documentation aggregation
- Competitive intelligence
- Content migration
- API discovery

---

#### 14. **eleven-labs** `ElevenLabsBubble`
**Purpose**: Text-to-speech and voice generation

**Key Features:**
- Multi-language voices
- Voice cloning
- SSML support
- Audio streaming
- Custom pronunciation

**Example:**
```typescript
const tts = new ElevenLabsBubble({
  text: 'Welcome to OpenEvolve automated alerts',
  voiceId: 'xyz123',
  modelId: 'eleven_multilingual_v2',
  outputFormat: 'mp3_44100_128'
});

const result = await tts.action();
// result.data.audioUrl contains generated audio
```

**Credential Types:** `eleven_labs_api_key`

**Use Cases:**
- Alert announcements
- Accessibility features
- Content narration
- Voice response systems

---

#### 15. **resend** `ResendBubble`
**Purpose**: Transactional email service

**Key Features:**
- HTML email templates
- Attachments
- Batch sending
- Bounce handling
- Analytics

**Example:**
```typescript
const email = new ResendBubble({
  from: 'noreply@openevolve.com',
  to: 'user@example.com',
  subject: 'Password Reset',
  html: '<h1>Reset Your Password</h1><p>Click here...</p>'
});

await email.action();
```

**Credential Types:** `resend_api_key`

**Use Cases:**
- Password reset emails
- Verification codes
- Transaction notifications
- Marketing campaigns

---

#### 16. **followupboss** `FollowUpBossBubble`
**Purpose**: CRM integration for real estate

**Key Features:**
- Lead management
- Contact updates
- Task automation
- Deal tracking
- Activity logging

**Example:**
```typescript
const fub = new FollowUpBossBubble({
  action: 'create_lead',
  data: {
    name: 'Jane Smith',
    email: 'jane@example.com',
    phone: '+1234567890',
    source: 'Website Form'
  }
});

await fub.action();
```

**Credential Types:** `followupboss_api_key`

**Use Cases:**
- Lead capture automation
- Follow-up reminders
- Activity synchronization
- Reporting

---

#### 17. **telegram** `TelegramBubble`
**Purpose**: Telegram bot integration

**Key Features:**
- Send messages
- Inline keyboards
- File sharing
- Group management
- Webhook setup

**Example:**
```typescript
const telegram = new TelegramBubble({
  chatId: '@openevolve_updates',
  text: 'Deployment successful!',
  parseMode: 'Markdown'
});

await telegram.action();
```

**Credential Types:** `telegram_bot_token`

**Use Cases:**
- Broadcast announcements
- User notifications
- Interactive bots
- Group moderation

---

#### 18. **storage** `StorageBubble`
**Purpose**: Generic file storage abstraction

**Key Features:**
- Multi-provider support (S3, GCS, Azure)
- File upload/download
- Bucket management
- Presigned URLs
- Metadata handling

**Example:**
```typescript
const storage = new StorageBubble({
  provider: 's3',
  operation: 'upload',
  bucket: 'openevolve-backups',
  key: 'backups/db-2025-01-17.sql',
  body: fileData
});

await storage.action();
```

**Credential Types:** `aws_credentials`, `gcp_credentials`, `azure_credentials`

**Use Cases:**
- Backup storage
- File serving
- Log archival
- Asset management

---

#### 19. **agi-inc** `AGIIncBubble`
**Purpose**: AGI Inc. platform integration

**Example:**
```typescript
const agi = new AGIIncBubble({
  // AGI Inc. specific configuration
});
```

---

#### 20. **insforge-db** `InsForgeDbBubble`
**Purpose**: InsForge database operations

**Example:**
```typescript
const insforge = new InsForgeDbBubble({
  // InsForge specific configuration
});
```

---

#### 21. **hello-world** `HelloWorldBubble`
**Purpose**: Simple test bubble for verification

**Example:**
```typescript
const hello = new HelloWorldBubble({ name: 'World' });
await hello.action();
// Returns: { success: true, data: { message: 'Hello, World!' } }
```

---

### Tool Bubbles (18)

Tool bubbles perform specific actions and utilities, often used within AI agent workflows.

#### 1. **web-search-tool** `WebSearchTool`
**Purpose**: Web search with multiple providers (Google, Bing, DuckDuckGo)

**Example:**
```typescript
const search = new WebSearchTool({
  query: 'BubbleLab workflow automation',
  numResults: 10,
  provider: 'google'
});

const results = await search.action();
```

---

#### 2. **web-scrape-tool** `WebScrapeTool`
**Purpose**: Extract content from single web pages

**Example:**
```typescript
const scraper = new WebScrapeTool({
  url: 'https://openevolve.com/docs',
  selectors: {
    title: 'h1',
    content: '.content',
    links: 'a[href]'
  }
});

const data = await scraper.action();
```

---

#### 3. **web-crawl-tool** `WebCrawlTool`
**Purpose**: Multi-page web crawling with depth control

**Example:**
```typescript
const crawler = new WebCrawlTool({
  startUrl: 'https://openevolve.com/docs',
  maxDepth: 2,
  followLinks: true,
  excludePatterns: ['/admin', '/login']
});

const pages = await crawler.action();
```

---

#### 4. **web-extract-tool** `WebExtractTool`
**Purpose**: LLM-powered web content extraction

**Example:**
```typescript
const extractor = new WebExtractTool({
  url: 'https://example.com/article',
  extractionPrompt: 'Extract the main article, author, and publication date'
});

const extracted = await extractor.action();
```

---

#### 5. **research-agent-tool** `ResearchAgentTool`
**Purpose**: Multi-step research with web search and synthesis

**Example:**
```typescript
const researcher = new ResearchAgentTool({
  query: 'Latest trends in workflow automation 2025',
  depth: 3,
  summarize: true
});

const report = await researcher.action();
```

---

#### 6. **reddit-scrape-tool** `RedditScrapeTool`
**Purpose**: Reddit data extraction (posts, comments, metrics)

**Example:**
```typescript
const reddit = new RedditScrapeTool({
  subreddit: 'devops',
  sort: 'hot',
  limit: 25,
  filter: ['score', 'title', 'selftext']
});

const posts = await reddit.action();
```

---

#### 7. **instagram-tool** `InstagramTool`
**Purpose**: Instagram API integration

**Example:**
```typescript
const insta = new InstagramTool({
  action: 'get_media',
  userId: '123456789',
  limit: 20
});
```

---

#### 8. **linkedin-tool** `LinkedInTool`
**Purpose**: LinkedIn profile and company data

**Example:**
```typescript
const linkedin = new LinkedInTool({
  action: 'get_profile',
  profileUrl: 'https://linkedin.com/in/user'
});
```

---

#### 9. **tiktok-tool** `TikTokTool`
**Purpose**: TikTok video and user data

**Example:**
```typescript
const tiktok = new TikTokTool({
  action: 'get_trending',
  region: 'US',
  limit: 10
});
```

---

#### 10. **twitter-tool** `TwitterTool`
**Purpose**: X/Twitter operations

**Example:**
```typescript
const twitter = new TwitterTool({
  action: 'search',
  query: '#openevolve',
  count: 50
});
```

---

#### 11. **youtube-tool** `YouTubeTool`
**Purpose**: YouTube video and channel data

**Example:**
```typescript
const youtube = new YouTubeTool({
  action: 'search',
  query: 'workflow automation tutorial',
  maxResults: 10
});
```

---

#### 12. **google-maps-tool** `GoogleMapsTool`
**Purpose**: Maps, geocoding, and places data

**Example:**
```typescript
const maps = new GoogleMapsTool({
  action: 'geocode',
  address: '1600 Amphitheatre Parkway, Mountain View, CA'
});

const location = await maps.action();
```

---

#### 13. **sql-query-tool** `SQLQueryTool`
**Purpose**: Execute SQL queries with AI assistance

**Example:**
```typescript
const sql = new SQLQueryTool({
  databaseType: 'postgresql',
  naturalLanguageQuery: 'Show top 10 users by registration count this month'
});

const results = await sql.action();
```

---

#### 14. **code-edit-tool** `CodeEditTool`
**Purpose**: AI-powered code modification and refactoring

**Example:**
```typescript
const editor = new CodeEditTool({
  filePath: './src/utils.ts',
  instruction: 'Add error handling and TypeScript types',
  context: 'This utility function is used for data validation'
});

const result = await editor.action();
```

---

#### 15. **bubbleflow-validation-tool** `BubbleFlowValidationTool`
**Purpose**: Validate BubbleFlow workflows

**Example:**
```typescript
const validator = new BubbleFlowValidationTool({
  workflowCode: myWorkflowCode,
  checkTypes: true,
  checkDependencies: true
});

const report = await validator.action();
```

---

#### 16. **chart-js-tool** `ChartJSTool`
**Purpose**: Generate Chart.js visualizations

**Example:**
```typescript
const chart = new ChartJSTool({
  type: 'line',
  data: {
    labels: ['Jan', 'Feb', 'Mar'],
    datasets: [{
      label: 'Deployments',
      data: [10, 15, 20]
    }]
  }
});

const chartConfig = await chart.action();
```

---

#### 17. **list-bubbles-tool** `ListBubblesTool`
**Purpose**: List available bubbles

**Example:**
```typescript
const lister = new ListBubblesTool({});
const bubbles = await lister.action();
```

---

#### 18. **get-bubble-details-tool** `GetBubbleDetailsTool`
**Purpose**: Get detailed information about a bubble

**Example:**
```typescript
const details = new GetBubbleDetailsTool({
  bubbleName: 'slack'
});

const info = await details.action();
```

---

### Workflow Bubbles (12)

Workflow bubbles are pre-built composite patterns combining multiple bubbles.

#### 1. **database-analyzer** `DatabaseAnalyzerWorkflowBubble`
**Purpose**: Analyze database schema and health

**Example:**
```typescript
const analyzer = new DatabaseAnalyzerWorkflowBubble({
  connectionString: 'postgresql://...',
  checks: ['schema', 'performance', 'security']
});

const report = await analyzer.action();
```

---

#### 2. **slack-notifier** `SlackNotifierWorkflowBubble`
**Purpose**: Send formatted Slack notifications

**Example:**
```typescript
const notifier = new SlackNotifierWorkflowBubble({
  channel: '#alerts',
  message: 'Production database health check failed',
  severity: 'critical',
  metadata: { region: 'us-east-1', database: 'postgres-prod' }
});

await notifier.action();
```

---

#### 3. **slack-data-assistant** `SlackDataAssistantWorkflowBubble`
**Purpose**: Interactive Slack data queries

**Example:**
```typescript
const assistant = new SlackDataAssistantWorkflowBubble({
  question: 'How many users registered this week?',
  context: 'query from PostgreSQL database'
});

const answer = await assistant.action();
```

---

#### 4. **slack-formatter-agent** `SlackFormatterAgentBubble`
**Purpose**: Format messages for Slack consumption

**Example:**
```typescript
const formatter = new SlackFormatterAgentBubble({
  data: { metrics: { cpu: 85, memory: 72 } },
  format: 'table'
});

const formatted = await formatter.action();
```

---

#### 5. **pdf-form-operations** `PDFFormOperationsWorkflowBubble`
**Purpose**: Fill and manipulate PDF forms

**Example:**
```typescript
const pdf = new PDFFormOperationsWorkflowBubble({
  template: 'form.pdf',
  fields: { name: 'John Doe', date: '2025-01-17' }
});

await pdf.action();
```

---

#### 6. **pdf-ocr-workflow** `PDFOcrWorkflowBubble`
**Purpose**: OCR text extraction from PDFs

**Example:**
```typescript
const ocr = new PDFOcrWorkflowBubble({
  pdfFile: './scanned.pdf',
  language: 'eng'
});

const text = await ocr.action();
```

---

#### 7. **generate-document-workflow** `GenerateDocumentWorkflowBubble`
**Purpose**: Generate documents from templates

**Example:**
```typescript
const generator = new GenerateDocumentWorkflowBubble({
  template: 'report.html',
  data: { title: 'Q4 Report', metrics: [...] },
  outputFormat: 'pdf'
});

const doc = await generator.action();
```

---

#### 8. **parse-document-workflow** `ParseDocumentWorkflowBubble`
**Purpose**: Parse structured data from documents

**Example:**
```typescript
const parser = new ParseDocumentWorkflowBubble({
  document: './invoice.pdf',
  extractionSchema: {
    invoiceNumber: 'string',
    total: 'number',
    date: 'date'
  }
});

const data = await parser.action();
```

---

## Bubble Type Summary

| Type | Count | Purpose |
|------|-------|---------|
| **Service Bubbles** | 21 | External service integrations |
| **Tool Bubbles** | 18 | Utility functions and actions |
| **Workflow Bubbles** | 12 | Pre-built composite patterns |
| **Total** | **51** | Complete automation platform |

---

## Credential Types Reference

Bubbles require credentials for external service authentication. Configure these in BubbleLab Settings → Credentials.

| Credential Type | Used By |
|----------------|----------|
| `openai_api_key` | ai-agent |
| `anthropic_api_key` | ai-agent |
| `google_api_key` | ai-agent, google-* |
| `slack_bot_token` | slack, slack-* |
| `slack_signing_secret` | slack |
| `postgres_connection_string` | postgresql |
| `github_pat` | github |
| `notion_api_key` | notion |
| `airtable_pat` | airtable |
| `apify_api_token` | apify |
| `firecrawl_api_key` | firecrawl |
| `eleven_labs_api_key` | eleven-labs |
| `resend_api_key` | resend |
| `followupboss_api_key` | followupboss |
| `telegram_bot_token` | telegram |
| `aws_credentials` | storage (S3) |
| `gcp_credentials` | storage (GCS) |
| `azure_credentials` | storage (Azure) |

---

## Quick Bubble Selection Guide

**For Database Operations:**
- `postgresql` - Direct queries
- `sql-query-tool` - Natural language queries

**For Messaging:**
- `slack` - Slack messages
- `telegram` - Telegram messages
- `resend` - Email sending
- `gmail` - Gmail operations

**For Web Operations:**
- `http` - Generic HTTP requests
- `web-scrape-tool` - Single page scraping
- `web-crawl-tool` - Multi-page crawling
- `apify` - Advanced scraping
- `firecrawl` - LLM extraction

**For AI/ML:**
- `ai-agent` - Multi-model AI
- `research-agent-tool` - Multi-step research
- `code-edit-tool` - Code refactoring

**For Data Management:**
- `google-sheets` - Spreadsheets
- `notion` - Documentation
- `airtable` - Databases
- `google-drive` - File storage

---

**End of Appendix A**
