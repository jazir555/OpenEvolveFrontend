# BubbleLab Workflow Templates - Quick Reference

Production-ready workflow templates for OpenEvolve development automation.

## Directory Structure

```
BubbleLab/templates/
├── infrastructure/        # Infrastructure Automation (8 workflows)
├── development/           # Development Workflow (7 workflows)
└── llm-operations/        # LLM Operations (5 workflows)
```

---

## Infrastructure Automation (8 Workflows)

### 1. Container Health Monitor
**File**: `infrastructure/container-health-monitor.ts`
**Schedule**: Every 5 minutes
**Purpose**: Monitors Docker container health and auto-heals unhealthy containers

**Credentials**:
- `DOCKER_HOST`: Docker daemon socket
- `SLACK_WEBHOOK_URL`: For alerts (optional)

**Key Features**:
- Lists all containers
- Checks health status
- Auto-restarts unhealthy containers
- Sends alerts for critical issues

---

### 2. Log Aggregation & Analyzer
**File**: `infrastructure/log-aggregation-analyzer.ts`
**Schedule**: Every minute
**Purpose**: Aggregates logs and detects anomalies using AI

**Credentials**:
- `POSTGRES_CONNECTION_STRING`: Database connection
- `SLACK_WEBHOOK_URL`: For alerts (optional)

**Key Features**:
- Collects logs from database
- Categorizes by severity
- AI-powered anomaly detection
- Stores anomalies for analysis

---

### 3. Database Backup Validator
**File**: `infrastructure/database-backup-validator.ts`
**Schedule**: Daily at 3 AM
**Purpose**: Automated backup with integrity validation

**Credentials**:
- `POSTGRES_CONNECTION_STRING`: Database connection
- `AWS_CREDENTIALS` or `GCP_CREDENTIALS`: Storage backup
- `SLACK_WEBHOOK_URL`: For notifications (optional)

**Key Features**:
- Creates compressed backups
- Uploads to cloud storage
- Validates backup integrity
- Tests restore to staging

---

### 4. Service Deployment Automation
**File**: `infrastructure/service-deployment-automation.ts`
**Event**: Webhook
**Purpose**: Automated deployment with health checks and rollback

**Credentials**:
- `KUBERNETES_CONFIG`: Cluster configuration
- `DOCKER_REGISTRY`: Container registry
- `SLACK_WEBHOOK_URL`: For notifications (optional)

**Key Features**:
- Pre-deployment health checks
- Rolling updates
- Post-deployment validation
- Automatic rollback on failure

---

### 5. Resource Scaling Automation
**File**: `infrastructure/resource-scaling-automation.ts`
**Schedule**: Every 10 minutes
**Purpose**: Auto-scales resources based on metrics

**Credentials**:
- `KUBERNETES_CONFIG`: Cluster configuration
- `PROMETHEUS_URL`: Metrics endpoint
- `SLACK_WEBHOOK_URL`: For notifications (optional)

**Key Features**:
- Monitors CPU/memory usage
- Scales up/down based on thresholds
- Respects min/max replica limits
- Sends scaling notifications

---

### 6. Service Dependency Scanner
**File**: `infrastructure/service-dependency-scanner.ts`
**Schedule**: Daily at 4 AM
**Purpose**: Maps service dependencies across infrastructure

**Credentials**:
- `KUBERNETES_CONFIG`: Cluster configuration
- `PROMETHEUS_URL`: Service metrics
- `POSTGRES_CONNECTION_STRING`: Storage

**Key Features**:
- Discovers all services
- Analyzes service-to-service communication
- Identifies critical paths
- AI-powered dependency analysis

---

### 7. Distributed Tracing Analyzer
**File**: `infrastructure/distributed-tracing-analyzer.ts`
**Schedule**: Every 15 minutes
**Purpose**: Analyzes traces for performance bottlenecks

**Credentials**:
- `JAEGER_API`: Tracing API endpoint
- `POSTGRES_CONNECTION_STRING`: Storage
- `SLACK_WEBHOOK_URL`: For alerts (optional)

**Key Features**:
- Queries Jaeger for traces
- Identifies slow/error traces
- Calculates P95 latencies
- AI-powered recommendations

---

## Development Workflow (7 Workflows)

### 1. Code Review Automation
**File**: `development/code-review-automation.ts`
**Event**: Webhook (PR opened/updated)
**Purpose**: Automated code review with AI analysis

**Credentials**:
- `GITHUB_PAT`: GitHub API access
- `OPENAI_API_KEY`: For code analysis
- `SLACK_WEBHOOK_URL`: For notifications (optional)

**Key Features**:
- Fetches PR diff
- AI-powered code analysis
- Posts review comments
- Adds labels based on issues

---

### 2. Test Execution Reporter
**File**: `development/test-execution-reporter.ts`
**Schedule**: Daily at 2 AM
**Purpose**: Executes tests and generates reports

**Credentials**:
- `GITHUB_PAT`: GitHub API access
- `POSTGRES_CONNECTION_STRING`: Test results database
- `SLACK_WEBHOOK_URL`: For notifications (optional)
- `GMAIL_CRED`: For email reports (optional)

**Key Features**:
- Runs multiple test suites
- Generates HTML reports
- AI-powered analysis
- Sends notifications on failures

---

### 3. Dependency Update Automation
**File**: `development/dependency-update-automation.ts`
**Schedule**: Weekly on Monday at 9 AM
**Purpose**: Monitors dependencies and creates update PRs

**Credentials**:
- `GITHUB_PAT`: GitHub API access
- `SLACK_WEBHOOK_URL`: For notifications (optional)

**Key Features**:
- Checks Python (PyPI) and Node (npm) packages
- Identifies security updates
- Creates automated PRs
- Prioritizes critical updates

---

### 4. Documentation Generator
**File**: `development/documentation-generator.ts`
**Event**: Webhook
**Purpose**: Auto-generates documentation from code

**Credentials**:
- `GITHUB_PAT`: GitHub API access
- `OPENAI_API_KEY`: For AI documentation generation

**Key Features**:
- Generates API documentation
- Updates README
- Creates changelogs
- Generates architecture docs

---

### 5. Deployment Pipeline Orchestrator
**File**: `development/deployment-pipeline-orchestrator.ts`
**Event**: Webhook
**Purpose**: Multi-stage deployment pipeline

**Credentials**:
- `KUBERNETES_CONFIG`: Cluster configuration
- `GITHUB_PAT`: GitHub API access
- `SLACK_WEBHOOK_URL`: For notifications

**Key Features**:
- Build verification
- Staging deployment
- Automated testing
- Production deployment with rollback

---

### 6. Automated Changelog Generator
**File**: `development/automated-changelog-generator.ts`
**Event**: Webhook
**Purpose**: Generates changelogs from commit history

**Credentials**:
- `GITHUB_PAT`: GitHub API access
- `OPENAI_API_KEY`: For intelligent generation

**Key Features**:
- Analyzes commit messages
- Categorizes changes
- Formats markdown changelog
- Lists contributors

---

### 7. Security Vulnerability Scanner
**File**: `development/security-vulnerability-scanner.ts`
**Schedule**: Daily at 6 AM
**Purpose**: Scans code for security vulnerabilities

**Credentials**:
- `GITHUB_PAT`: GitHub API access
- `SLACK_WEBHOOK_URL`: For alerts
- `GMAIL_CRED`: For email reports (optional)

**Key Features**:
- Scans Python dependencies
- Scans Node.js dependencies
- Code security analysis
- Calculates security score

---

## LLM Operations (5 Workflows)

### 1. Prompt Testing Validator
**File**: `llm-operations/prompt-testing-validator.ts`
**Event**: Webhook
**Purpose**: Tests and validates prompts across models

**Credentials**:
- `OPENAI_API_KEY`: For GPT models
- `ANTHROPIC_API_KEY`: For Claude models
- `GOOGLE_API_KEY`: For Gemini models
- `POSTGRES_CONNECTION_STRING`: To store results

**Key Features**:
- Tests prompts across multiple models
- AI-powered evaluation
- Token usage tracking
- Quality scoring

---

### 2. Model Performance Benchmark
**File**: `llm-operations/model-performance-benchmark.ts`
**Schedule**: Weekly on Sunday at 3 AM
**Purpose**: Benchmarks model performance across tasks

**Credentials**:
- `OPENAI_API_KEY`: For GPT models
- `ANTHROPIC_API_KEY`: For Claude models
- `GOOGLE_API_KEY`: For Gemini models
- `POSTGRES_CONNECTION_STRING`: To store results

**Key Features**:
- Tests multiple models
- Various task categories
- Performance metrics
- Cost analysis

---

### 3. Token Usage Monitor
**File**: `llm-operations/token-usage-monitor.ts`
**Schedule**: Every hour
**Purpose**: Monitors and analyzes token usage and costs

**Credentials**:
- `POSTGRES_CONNECTION_STRING`: Database with logs
- `SLACK_WEBHOOK_URL`: For alerts (optional)

**Key Features**:
- Tracks token usage
- Cost monitoring
- AI-powered recommendations
- Projection analysis

---

### 4. AI Response Quality Assessor
**File**: `llm-operations/ai-response-quality-assessor.ts`
**Event**: Webhook
**Purpose**: Assesses and tracks AI response quality

**Credentials**:
- `OPENAI_API_KEY`: For GPT-4 evaluation
- `POSTGRES_CONNECTION_STRING`: To store metrics

**Key Features**:
- Multi-dimensional quality scoring
- Trend analysis
- Issue identification
- Improvement recommendations

---

### 5. Multi-Model Comparison Tester
**File**: `llm-operations/multi-model-comparison-tester.ts`
**Event**: Webhook
**Purpose**: Compares outputs from multiple models

**Credentials**:
- `OPENAI_API_KEY`: For GPT models
- `ANTHROPIC_API_KEY`: For Claude models
- `GOOGLE_API_KEY`: For Gemini models
- `POSTGRES_CONNECTION_STRING`: To store comparisons

**Key Features**:
- Tests same prompt on multiple models
- Performance comparison
- Quality analysis
- Use case recommendations

---

### 6. Prompt Optimizer
**File**: `llm-operations/prompt-optimizer.ts`
**Event**: Webhook
**Purpose**: Optimizes prompts for better performance

**Credentials**:
- `OPENAI_API_KEY`: For prompt testing
- `ANTHROPIC_API_KEY`: For prompt testing
- `POSTGRES_CONNECTION_STRING`: To store optimized prompts

**Key Features**:
- Iterative prompt optimization
- Quality metrics evaluation
- Cost analysis
- AI-powered improvements

---

## Deployment Instructions

### 1. Configure Credentials
Add required credentials to BubbleLab:
1. Navigate to Credentials page
2. Add each required credential (see workflow files)
3. Store securely with encryption enabled

### 2. Deploy Workflows
Using Python SDK:
```python
from bubblelab_manager import BubbleLabWorkflowManager

manager = BubbleLabWorkflowManager('bubblelab-config.yaml')
results = manager.deploy_from_directory('./BubbleLab/templates')
```

Using CLI:
```bash
python scripts/bubblelab-manager.py deploy ./BubbleLab/templates
```

### 3. Activate Webhooks
For webhook-triggered workflows:
```bash
curl -X POST "http://localhost:3001/bubble-flow/{flow_id}/activate" \
  -H "Authorization: Bearer $TOKEN"
```

### 4. Configure Schedules
For scheduled workflows, schedules are predefined:
- Edit workflow file to change schedule
- Redeploy after modification

---

## Environment Variables

Required for all workflows:
```bash
# BubbleLab
BUBBLELAB_URL=http://localhost:3001
BUBBLELAB_API_KEY=your_api_key

# Infrastructure
DOCKER_HOST=http://docker:2375
KUBERNETES_API=https://kubernetes.default.svc
KUBERNETES_TOKEN=your_token
PROMETHEUS_URL=http://prometheus:9090
JAEGER_API=http://jaeger:16686

# Databases
POSTGRES_CONNECTION_STRING=postgresql://user:pass@host:5432/db
POSTGRES_HOST=postgres:5432

# Git
GITHUB_PAT=your_github_pat

# AI Models
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
GMAIL_CRED=your_gmail_credentials

# Storage
AWS_CREDENTIALS=your_aws_creds
GCP_CREDENTIALS=your_gcp_creds
BACKUP_BUCKET=openevolve-backups
```

---

## Quick Start Example

Deploy Container Health Monitor:
```python
from bubblelab_client import BubbleLabClient

client = BubbleLabClient(
    base_url='http://localhost:3001',
    api_key='your_api_key'
)

# Read workflow file
with open('BubbleLab/templates/infrastructure/container-health-monitor.ts', 'r') as f:
    code = f.read()

# Create workflow
flow = client.create_flow(
    name='Container Health Monitor',
    code=code,
    description='Monitors container health and auto-heals',
    eventType='schedule/cron'
)

print(f"Created workflow: {flow['id']}")
```

---

## Monitoring & Debugging

### View Execution History
```bash
curl "http://localhost:3001/bubble-flow/{flow_id}/executions" \
  -H "Authorization: Bearer $TOKEN"
```

### Stream Execution (Real-time)
```bash
curl -X POST "http://localhost:3001/bubble-flow/{flow_id}/execute-stream" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": {}}'
```

### Check Logs
Logs are available in:
- BubbleLab dashboard
- Database logs table
- External logging systems (via integrations)

---

## Best Practices

1. **Credentials**: Always use encrypted credentials
2. **Schedules**: Stagger scheduled workflows to avoid overlap
3. **Monitoring**: Set up alerts for failed executions
4. **Testing**: Test workflows in development first
5. **Backups**: Backup workflow definitions regularly
6. **Version Control**: Keep workflow files in Git
7. **Documentation**: Document custom workflows

---

## Support & Resources

- **BubbleLab Documentation**: See `docs/BUBBLELAB_AUTOMATION_GUIDE.md`
- **API Reference**: See `docs/BUBBLELAB_SCRIPTING_GUIDE.md`
- **Templates**: Located in `BubbleLab/templates/`
- **Examples**: See `docs/examples/`

---

## Changelog

### Version 1.0.0 (2025-01-17)
- Initial release of 20 production workflows
- Infrastructure automation (8 workflows)
- Development workflow (7 workflows)
- LLM operations (5 workflows)

---

*Generated for OpenEvolve Development Automation*
