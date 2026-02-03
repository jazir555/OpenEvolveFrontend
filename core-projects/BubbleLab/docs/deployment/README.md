# BubbleLab Production-Ready Workflow Examples

A comprehensive collection of 24 production-ready workflow examples covering infrastructure automation, development automation, and LLM operations.

## 📁 Directory Structure

```
examples/
├── infrastructure-automation/    # 8 workflows for infrastructure management
├── development-automation/       # 8 workflows for DevOps and development
├── llm-operations/               # 8 workflows for LLM management
└── README.md                     # This file
```

## 🏗️ Infrastructure Automation (8 Workflows)

Automate your infrastructure operations with these production-ready workflows.

### 1. Container Auto-Healing
**File:** `infrastructure-automation/container-autohealing.ts`

Detects and heals unhealthy containers automatically based on health checks, CPU, and memory thresholds.

**Key Features:**
- Real-time container health monitoring
- Automatic restart on failure
- CPU and memory threshold-based scaling
- Root cause analysis with AI
- Slack notifications

**Use Case:** Production environment reliability - automatically restart or replace containers that are failing health checks.

**Trigger:** Scheduled (every 5 minutes) or webhook

---

### 2. Log Anomaly Detection
**File:** `infrastructure-automation/log-anomaly-detection.ts`

ML-based detection of anomalies in application and system logs with intelligent alerting.

**Key Features:**
- Error rate spike detection
- Security event identification
- Performance issue detection
- Resource exhaustion alerts
- AI-powered log analysis
- Comprehensive reporting

**Use Case:** Security monitoring, performance issue detection, and operational intelligence.

**Trigger:** Scheduled (every 15-30 minutes) or webhook

---

### 3. Database Backup Scheduled
**File:** `infrastructure-automation/database-backup-scheduled.ts`

Automated database backups with retention policy, validation, and compression.

**Key Features:**
- Support for PostgreSQL, MySQL, MongoDB
- Automated backup scheduling
- Gzip compression
- Backup validation
- Retention policy enforcement
- Multi-destination storage (S3, Google Drive)

**Use Case:** Critical data protection - scheduled backups of production databases.

**Trigger:** Scheduled (daily at 2 AM UTC)

---

### 4. Service Scaling Automation
**File:** `infrastructure-automation/service-scaling-automation.ts`

Auto-scale services based on CPU, memory, and custom metrics with cooldown periods.

**Key Features:**
- CPU and memory-based scaling
- Configurable min/max instances
- Cooldown period enforcement
- Kubernetes and Docker support
- Cost-aware scaling

**Use Case:** Dynamic infrastructure scaling - automatically add/remove instances based on load.

**Trigger:** Scheduled (every 2-5 minutes) or webhook

---

### 5. Certificate Renewal
**File:** `infrastructure-automation/certificate-renewal.ts`

Automated SSL/TLS certificate renewal before expiration with DNS challenge support.

**Key Features:**
- Let's Encrypt integration
- DNS challenge support (Cloudflare, Route53)
- Expiration monitoring
- Automatic renewal
- Security notifications

**Use Case:** Security compliance - prevent certificate expiration and service disruption.

**Trigger:** Scheduled (daily at 2 AM UTC)

---

### 6. Health Check Dashboard
**File:** `infrastructure-automation/health-check-dashboard.ts`

Aggregate and monitor health status of all services with unified dashboard.

**Key Features:**
- Multi-service health monitoring
- Response time tracking
- Uptime calculation
- Alert generation
- Historical data storage

**Use Case:** Operations dashboard - unified view of system health across all services.

**Trigger:** Scheduled (every 1-5 minutes)

---

### 7. Resource Cleanup
**File:** `infrastructure-automation/resource-cleanup.ts`

Automated cleanup of unused resources to reduce costs (containers, volumes, images).

**Key Features:**
- Docker container cleanup
- Volume cleanup
- Unused image removal
- Age-based retention
- Dry-run mode
- Space freed reporting

**Use Case:** Cost optimization - remove unused containers, volumes, and temporary files.

**Trigger:** Scheduled (daily at 3 AM UTC)

---

### 8. Incident Response
**File:** `infrastructure-automation/incident-response.ts`

Automated response to infrastructure incidents with predefined playbooks.

**Key Features:**
- Automated incident detection
- Predefined response playbooks
- Service restart automation
- Scaling actions
- Jira ticket creation
- Team notifications

**Use Case:** Incident management - automatically respond to common infrastructure issues.

**Trigger:** Webhook from monitoring/alerting system

---

## 💻 Development Automation (8 Workflows)

Streamline your development process with these CI/CD automation workflows.

### 1. Pull Request Automation
**File:** `development-automation/pr-automation.ts`

Automated PR review, testing, and validation with AI-powered code review.

**Key Features:**
- Automated test execution
- Code quality checks (ESLint, Prettier)
- AI-powered code review
- PR comments generation
- Approval workflows

**Use Case:** Development efficiency - automatically review and test PRs.

**Trigger:** Webhook from GitHub/GitLab on PR creation/update

---

### 2. Dependency Update
**File:** `development-automation/dependency-update.ts`

Automated dependency updates with security and compatibility checks.

**Key Features:**
- Outdated dependency detection
- Security vulnerability scanning
- Automated testing
- PR creation for updates
- Auto-merge for patch versions

**Use Case:** Security and maintenance - keep dependencies up to date safely.

**Trigger:** Scheduled (daily)

---

### 3. Deployment Pipeline
**File:** `development-automation/deployment-pipeline.ts`

Full CI/CD orchestration from commit to production with multiple stages.

**Key Features:**
- Automated testing
- Docker image building
- Kubernetes deployment
- Smoke tests
- Rollback support
- Stage gates

**Use Case:** Development operations - automated deployment pipeline with stages.

**Trigger:** Webhook on commit or manual

---

### 4. Code Quality Check
**File:** `development-automation/code-quality-check.ts`

Automated code quality analysis and reporting with AI suggestions.

**Key Features:**
- ESLint integration
- Prettier formatting checks
- TypeScript validation
- AI-powered analysis
- Quality scoring
- Improvement suggestions

**Use Case:** Code quality enforcement - analyze code for maintainability and issues.

**Trigger:** Webhook on PR or scheduled

---

### 5. Documentation Generator
**File:** `development-automation/documentation-generator.ts`

Auto-generate documentation from code and comments using AI.

**Key Features:**
- API documentation generation
- README generation
- Code examples
- Markdown output
- Google Drive storage

**Use Case:** Documentation maintenance - keep docs in sync with code.

**Trigger:** Scheduled (weekly) or manual

---

### 6. Test Orchestration
**File:** `development-automation/test-orchestration.ts`

Run test suites on schedule or trigger with comprehensive reporting.

**Key Features:**
- Unit test execution
- Integration testing
- E2E testing
- Coverage reporting
- Result storage
- Slack notifications

**Use Case:** Quality assurance - automated test execution and reporting.

**Trigger:** Scheduled (every 30 minutes) or webhook

---

### 7. Release Automation
**File:** `development-automation/release-automation.ts`

Automated release process with versioning and changelog generation.

**Key Features:**
- Automatic version bumping
- Changelog generation
- Git tagging
- GitHub release creation
- npm publishing
- Release notes

**Use Case:** Release management - automate version bumping, changelog, and publishing.

**Trigger:** Manual or scheduled

---

### 8. Branch Cleanup
**File:** `development-automation/branch-cleanup.ts`

Clean up old Git branches after merge and remove stale branches.

**Key Features:**
- Merged branch detection
- Stale branch identification
- Protected branch patterns
- Safe deletion
- Cleanup reporting

**Use Case:** Repository maintenance - keep repository clean and organized.

**Trigger:** Scheduled (weekly)

---

## 🤖 LLM Operations (8 Workflows)

Optimize your LLM operations with these monitoring and optimization workflows.

### 1. Prompt Testing Suite
**File:** `llm-operations/prompt-testing-suite.ts`

Test prompts across multiple models for performance and quality comparison.

**Key Features:**
- Multi-model testing
- Quality scoring
- Latency measurement
- Token usage tracking
- Result storage
- Performance comparison

**Use Case:** LLM optimization - compare prompt performance across different models.

**Trigger:** Scheduled (daily/weekly) or manual

---

### 2. Model Benchmarking
**File:** `llm-operations/model-benchmarking.ts`

Compare model performance on standardized benchmarks with cost analysis.

**Key Features:**
- Standardized benchmarks
- Multiple task types
- Performance scoring
- Cost calculation
- Leaderboard generation

**Use Case:** Model selection - evaluate and compare different LLMs.

**Trigger:** Scheduled (weekly) or manual

---

### 3. Token Usage Monitor
**File:** `llm-operations/token-usage-monitor.ts`

Track and alert on token usage across all LLM operations for cost management.

**Key Features:**
- Real-time usage tracking
- Cost calculation
- Threshold alerts
- By-model breakdown
- Historical data
- Trend analysis

**Use Case:** Cost management - monitor and optimize LLM API costs.

**Trigger:** Scheduled (hourly/daily)

---

### 4. AI Quality Assessment
**File:** `llm-operations/ai-quality-assessment.ts`

Evaluate AI response quality using multiple metrics and categories.

**Key Features:**
- Multi-category evaluation
- Quality scoring
- Issue identification
- Feedback generation
- Trend tracking

**Use Case:** Quality assurance - ensure AI responses meet quality standards.

**Trigger:** On-demand or scheduled

---

### 5. Model Failover
**File:** `llm-operations/model-failover.ts`

Automatic model fallback when primary model fails or degrades.

**Key Features:**
- Health monitoring
- Automatic fallback
- Priority-based selection
- Timeout handling
- Error recovery
- Notifications

**Use Case:** Reliability - ensure continuous LLM service availability.

**Trigger:** Real-time on LLM calls

---

### 6. Prompt Optimization
**File:** `llm-operations/prompt-optimization.ts`

Optimize prompts for better performance and cost efficiency using AI.

**Key Features:**
- AI-powered optimization
- Token reduction
- Quality validation
- A/B testing
- Improvement tracking

**Use Case:** Efficiency - improve prompt quality while reducing token usage.

**Trigger:** Manual or scheduled (weekly)

---

### 7. Cost Optimization
**File:** `llm-operations/cost-optimization.ts`

Minimize AI costs through smart model selection and caching recommendations.

**Key Features:**
- Cost analysis
- Model recommendations
- Caching strategies
- Savings calculation
- ROI tracking

**Use Case:** Cost management - reduce LLM API spend while maintaining quality.

**Trigger:** Scheduled (weekly/monthly)

---

### 8. Multi-Model Ensemble
**File:** `llm-operations/multi-model-ensemble.ts`

Combine multiple model outputs for improved quality and reliability.

**Key Features:**
- Multiple model execution
- Response aggregation (vote, merge, best)
- Quality scoring
- Confidence calculation
- Ensemble methods

**Use Case:** Quality enhancement - use ensemble techniques to improve responses.

**Trigger:** On-demand for critical operations

---

## 🚀 Getting Started

### Prerequisites

1. **BubbleLab Setup**: Ensure BubbleLab is properly installed and configured
2. **Credentials**: Configure required credentials for each workflow type:
   - Database credentials
   - Cloud provider credentials (AWS, GCP)
   - LLM API keys (OpenAI, Anthropic, Google)
   - Notification credentials (Slack, Email)
3. **Environment Variables**: Set up required environment variables (see individual workflow files)

### Installation

1. Clone the BubbleLab repository
2. Navigate to the examples directory:
   ```bash
   cd BubbleLab/examples
   ```

3. Browse workflows by category:
   ```bash
   ls infrastructure-automation/
   ls development-automation/
   ls llm-operations/
   ```

### Running Workflows

#### Option 1: Using BubbleLab UI

1. Open BubbleLab Studio
2. Navigate to the "Workflows" section
3. Click "Import Workflow"
4. Select the desired workflow file
5. Configure credentials and parameters
6. Save and run

#### Option 2: Using CLI

```bash
# Import a workflow
bubblelab import workflow ./infrastructure-automation/container-autohealing.ts

# Run a workflow
bubblelab run container-autohealing

# Schedule a workflow
bubblelab schedule container-autohealing --cron "*/5 * * * *"
```

#### Option 3: Using Webhook

Each workflow can be triggered via webhook:

```bash
curl -X POST https://your-bubblelab-instance.com/webhook/container-autohealing \
  -H "Content-Type: application/json" \
  -d '{
    "containerId": "abc123",
    "forceHeal": false,
    "notify": true
  }'
```

## 📋 Workflow Configuration

Each workflow includes:

- **Setup Instructions**: Detailed setup steps in file comments
- **Required Credentials**: List of needed credential types
- **Trigger Options**: How to trigger the workflow
- **Example Payloads**: Sample webhook payloads
- **Monitoring Configuration**: Alerting and monitoring setup
- **Error Handling**: Retry logic and failure management

### Example Configuration

```typescript
/**
 * Workflow: Container Auto-Healing
 *
 * Setup:
 * 1. Configure Docker API endpoint
 * 2. Set monitoring thresholds (CPU: 80%, Memory: 85%)
 * 3. Configure Slack notifications
 *
 * Required Credentials:
 * - docker-host: Docker daemon socket
 * - slack: For notifications (optional)
 *
 * Trigger: Scheduled (every 5 minutes)
 */
```

## 🔧 Customization

All workflows are fully customizable. Key areas to modify:

1. **Thresholds**: Adjust alert and action thresholds
2. **Endpoints**: Configure your specific service endpoints
3. **Credentials**: Use your own credential names
4. **Notifications**: Customize notification channels and messages
5. **Logic**: Modify core workflow logic as needed

## 📊 Monitoring and Logging

Each workflow includes:

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Performance Metrics**: Execution time, success rates
- **Error Tracking**: Detailed error information
- **Custom Metrics**: Domain-specific metrics

### Monitoring Setup

```typescript
// Enable monitoring
this.logger?.info('Starting workflow execution');
this.logger?.error('Error occurred', { error: err.message });
```

## 🔐 Security Best Practices

1. **Credential Management**: Never hardcode credentials
2. **Secrets Rotation**: Regularly rotate API keys
3. **Access Control**: Implement proper RBAC
4. **Audit Logging**: Track all workflow executions
5. **Data Encryption**: Encrypt sensitive data at rest

## 📈 Performance Optimization

1. **Batching**: Process multiple items in batches
2. **Caching**: Cache frequently accessed data
3. **Parallel Processing**: Use parallel execution where possible
4. **Timeouts**: Set appropriate timeouts for all operations
5. **Retry Logic**: Implement exponential backoff

## 🤝 Contributing

Contributions are welcome! To add new examples:

1. Follow the existing structure
2. Include comprehensive documentation
3. Add example payloads
4. Test thoroughly before submitting
5. Update this README

## 📝 License

These examples are part of the BubbleLab project. See the main LICENSE file for details.

## 🆘 Support

For issues or questions:
- Check the workflow-specific documentation in file comments
- Review the BubbleLab main documentation
- Open an issue on GitHub
- Join our Slack community

## 🎯 Summary

- **Total Workflows**: 24 production-ready examples
- **Categories**: 3 (Infrastructure, Development, LLM)
- **Lines of Code**: ~8,000+ lines of production code
- **Coverage**: 50+ real-world automation scenarios

Each workflow is:
- ✅ Production-ready
- ✅ Fully documented
- ✅ Immediately deployable
- ✅ Error-handled
- ✅ Optimized for performance
- ✅ Secure by design

Start automating with BubbleLab today! 🚀
