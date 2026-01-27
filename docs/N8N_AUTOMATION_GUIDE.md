# 🚀 n8n Automation Guide for OpenEvolve

**Complete End-to-End Development Automation Strategy**

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

### 1.1 What is n8n?

**n8n** (pronounced "n-eight-n") is an open-source workflow automation tool that enables you to connect anything to anything. It's designed for technical users who want to automate tasks without writing custom integration code.

**Key Benefits for OpenEvolve:**
- 🔗 **400+ Integrations**: GitHub, Slack, PostgreSQL, Redis, HTTP APIs, and more
- 🔄 **Event-Driven**: React to GitHub webhooks, schedules, manual triggers
- 🐍 **Python Support**: Write custom code nodes in Python
- 🎯 **Visual Workflow Builder**: Drag-and-drop interface for complex logic
- 🐳 **Self-Hosted**: Deploy on your own infrastructure (Docker)
- 🔒 **Secure**: Credential management, access controls, audit logs

### 1.2 Why n8n for OpenEvolve?

The OpenEvolve project has:
- **100+ integrated components** requiring coordination
- **272 configurable parameters** across environments
- **Multiple development stages** (dev, staging, production)
- **Complex testing requirements** (2000+ unit tests, 500+ integration tests)
- **Distributed architecture** with anti-corruption layers

**n8n fills critical automation gaps:**

| Gap | Current State | n8n Solution |
|-----|---------------|--------------|
| Monitoring | Manual health checks | Automated monitoring with alerts |
| Test Scheduling | Manual execution | Scheduled test runs with reporting |
| Deployment Coordination | GitHub Actions only | Orchestrate multi-stage deployments |
| Log Analysis | Manual inspection | Automated log parsing and alerting |
| Dependency Updates | Manual tracking | Automated monitoring with PR creation |
| Backup Validation | Basic scripts | Automated verification with testing |

### 1.3 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     n8n Workflow Engine                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Webhook      │  │ Cron         │  │ Manual       │      │
│  │ Triggers     │  │ Schedules    │  │ Triggers     │      │
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

---

## 2. Quick Start

### 2.1 Installation & Setup

#### Option A: Docker Compose (Recommended)

Create `docker-compose.n8n.yml`:

```yaml
version: '3.8'

services:
  n8n:
    image: n8nio/n8n:latest
    container_name: openevolve-n8n
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=your_secure_password
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - WEBHOOK_URL=http://localhost:5678/
      - GENERIC_TIMEZONE=UTC
      - TZ=UTC
      # PostgreSQL for workflow state
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=n8n
      - DB_POSTGRESDB_PASSWORD=n8n_password
      # Redis for job queue
      - QUEUE_BULL_REDIS_HOST=redis
      - QUEUE_BULL_REDIS_PORT=6379
    volumes:
      - n8n_data:/home/node/.n8n
      - ./n8n/workflows:/workflows
    networks:
      - openevolve-network
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:14-alpine
    container_name: n8n-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=n8n
      - POSTGRES_PASSWORD=n8n_password
      - POSTGRES_DB=n8n
    volumes:
      - n8n_postgres_data:/var/lib/postgresql/data
    networks:
      - openevolve-network

  redis:
    image: redis:7-alpine
    container_name: n8n-redis
    restart: unless-stopped
    networks:
      - openevolve-network

volumes:
  n8n_data:
  n8n_postgres_data:

networks:
  openevolve-network:
    external: true
```

**Start n8n:**

```bash
docker-compose -f docker-compose.n8n.yml up -d
```

**Access n8n:**
- URL: `http://localhost:5678`
- Credentials: `admin / your_secure_password`

#### Option B: npm Installation

```bash
npm install n8n -g
n8n start
```

### 2.2 Initial Configuration

1. **Create Credentials:**
   - GitHub Personal Access Token
   - Slack Bot Token
   - SMTP credentials for email
   - PostgreSQL connection details
   - Qdrant API endpoint

2. **Configure Webhooks:**
   - Set up webhook URL: `https://your-n8n-domain.com/webhook`
   - Configure GitHub webhook to send events to n8n
   - Test webhook delivery

3. **Set Up Notifications:**
   - Configure Slack workspace integration
   - Set up email notification templates
   - Test notification delivery

### 2.3 Your First Workflow

Create a simple "Health Check Monitor" workflow:

```json
{
  "name": "OpenEvolve Health Check Monitor",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [{"field": "minutes", "minutesInterval": 5}]
        }
      },
      "name": "Every 5 Minutes",
      "type": "n8n-nodes-base.scheduleTrigger",
      "typeVersion": 1,
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "http://qdrant:6333/health",
        "options": {}
      },
      "name": "Check Qdrant",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [450, 200]
    },
    {
      "parameters": {
        "url": "http://postgres:5432",
        "options": {}
      },
      "name": "Check PostgreSQL",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [450, 400]
    },
    {
      "parameters": {
        "url": "http://redis:6379/health",
        "options": {}
      },
      "name": "Check Redis",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [450, 600]
    },
    {
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{$json.status}}",
              "operation": "notEquals",
              "value2": "200"
            }
          ]
        }
      },
      "name": "Check for Failures",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [650, 400]
    },
    {
      "parameters": {
        "channel": "#openevolve-alerts",
        "text": "=⚠️ Health Check Failed: {{$node[\"Check Qdrant\"].json.service_name}} - Status: {{$json.status}}"
      },
      "name": "Send Alert to Slack",
      "type": "n8n-nodes-base.slack",
      "typeVersion": 2,
      "position": [850, 200]
    }
  ],
  "connections": {
    "Every 5 Minutes": {
      "main": [[{"node": "Check Qdrant"}, {"node": "Check PostgreSQL"}, {"node": "Check Redis"}]]
    },
    "Check Qdrant": {
      "main": [["Check for Failures"]]
    },
    "Check PostgreSQL": {
      "main": [["Check for Failures"]]
    },
    "Check Redis": {
      "main": [["Check for Failures"]]
    },
    "Check for Failures": {
      "main": [[null], [{"node": "Send Alert to Slack"}]]
    }
  }
}
```

---

## 3. Foundational Workflows

### 3.1 Health Check Monitor

**Purpose:** Continuously monitor all OpenEvolve services and alert on failures.

**Trigger:** Every 5 minutes (Cron)

**Workflow Steps:**

```
┌─────────────────┐
│ Cron Trigger    │
│ (Every 5 min)   │
└────────┬────────┘
         │
         ├─────────────────────────────────────────┐
         │                                         │
         ▼                                         ▼
┌─────────────────┐                     ┌─────────────────┐
│ Check Qdrant    │                     │ Check PostgreSQL│
│ HTTP Request    │                     │ TCP Request     │
└────────┬────────┘                     └────────┬────────┘
         │                                       │
         └──────────────┬────────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Merge Results   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ IF Any Failed?  │
              └────┬───────┬───┘
                   │       │
          No       │       │  Yes
         ┌─────────┘       └──────────┐
         │                            │
         ▼                            ▼
   ┌───────────┐             ┌─────────────────┐
   │ End       │             │ Slack Alert     │
   │ (Log OK)  │             │ + Email Alert   │
   └───────────┘             └─────────────────┘
```

**Implementation Details:**

**Node 1: Cron Trigger**
```json
{
  "cronExpression": "*/5 * * * *"
}
```

**Node 2: Check Qdrant**
```json
{
  "method": "GET",
  "url": "http://qdrant:6333/health",
  "timeout": 5000
}
```

**Node 3: Check PostgreSQL**
```json
{
  "method": "GET",
  "url": "http://postgres:5432",
  "timeout": 5000
}
```

**Node 4: Check Redis**
```json
{
  "method": "GET",
  "url": "http://redis:6379/ping",
  "timeout": 5000
}
```

**Node 5: Merge Results**
```javascript
// Code Node
const results = $input.all();
const failed = results.filter(r => r.json.status !== 200);

return [{
  json: {
    timestamp: new Date().toISOString(),
    total: results.length,
    passed: results.length - failed.length,
    failed: failed.length,
    failures: failed.map(f => ({
      service: f.json.service,
      status: f.json.status,
      error: f.json.error
    }))
  }
}];
```

**Node 6: Conditional Check**
```json
{
  "conditions": {
    "number": [
      {
        "value1": "={{$json.failed}}",
        "operation": "larger",
        "value2": 0
      }
    ]
  }
}
```

**Node 7: Slack Alert**
```json
{
  "channel": "#openevolve-alerts",
  "text": "=⚠️ Health Check Failed\n\nTimestamp: {{$json.timestamp}}\nFailed Services: {{$json.failed}}\n\nFailures:\n{{JSON.stringify($json.failures, null, 2)}}",
  "username": "OpenEvolve Monitor",
  "iconEmoji": ":warning:"
}
```

**Enhancements:**
- Add response time monitoring
- Track historical uptime in PostgreSQL
- Create dashboard visualization
- Implement escalation (Slack → SMS → Email)

### 3.2 Automated Test Runner

**Purpose:** Schedule and execute test suites, collect results, generate reports.

**Trigger:** Schedule (Nightly) + Webhook (On PR)

**Workflow Steps:**

```
┌─────────────────┐
│ Trigger         │
│ (Schedule/Webhook)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Set Test Config │
│ (Test type, env)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SSH to Runner   │
│ or HTTP Request │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run pytest      │
│ with coverage   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Parse Results   │
│ (JUnit XML)     │
└────────┬────────┘
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
┌─────────────────┐   ┌─────────────────┐
│ Store in DB     │   │ Generate Report │
└────────┬────────┘   └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
          ┌─────────────────┐
          │ IF Failed?      │
          └────┬───────┬────┘
               │       │
       No      │       │  Yes
      ┌────────┘       └─────────┐
      │                          │
      ▼                          ▼
┌───────────┐           ┌─────────────────┐
│ Log OK    │           │ Slack + Email   │
│           │           │ Alert + Report  │
└───────────┘           └─────────────────┘
```

**Implementation Details:**

**Node 1: Webhook Trigger**
```json
{
  "path": "test-runner",
  "responseMode": "responseNode",
  "options": {}
}
```

**Node 2: Set Test Configuration**
```javascript
// Code Node
const webhookData = $input.first().json;

return [{
  json: {
    test_type: webhookData.test_type || 'full',
    environment: webhookData.environment || 'development',
    branch: webhookData.branch || 'main',
    pr_number: webhookData.pr_number || null,
    test_suites: [
      'unit',
      'integration',
      'e2e'
    ]
  }
}];
```

**Node 3: Execute Tests via SSH**
```json
{
  "command": "=cd /openevolve && make test-{{$json.test_type}} ENV={{$json.environment}}"
}
```

**Alternative: Execute via HTTP API**
```json
{
  "method": "POST",
  "url": "http://openevolve-api:8000/api/tests/run",
  "body": {
    "type": "={{$json.test_type}}",
    "environment": "={{$json.environment}}"
  }
}
```

**Node 4: Parse JUnit Results**
```python
# Code Node (Python)
import xml.etree.ElementTree as ET
import json

# Get JUnit XML from input
junit_xml = $input.first().json.junit_output

# Parse XML
root = ET.fromstring(junit_xml)

# Extract test results
tests = []
for test_case in root.findall('.//testcase'):
    test = {
        'name': test_case.get('name'),
        'classname': test_case.get('classname'),
        'time': float(test_case.get('time', 0)),
        'status': 'passed'
    }

    failure = test_case.find('failure')
    if failure is not None:
        test['status'] = 'failed'
        test['message'] = failure.get('message')
        test['text'] = failure.text

    tests.append(test)

# Calculate statistics
total = len(tests)
passed = len([t for t in tests if t['status'] == 'passed'])
failed = total - passed
duration = sum(t['time'] for t in tests)

return [{
  'json': {
    'total_tests': total,
    'passed_tests': passed,
    'failed_tests': failed,
    'duration': duration,
    'success_rate': round((passed / total) * 100, 2) if total > 0 else 0,
    'tests': tests
  }
}]
```

**Node 5: Store in PostgreSQL**
```json
{
  "operation": "insert",
  "table": "test_results",
  "columns": [
    "timestamp",
    "test_type",
    "environment",
    "branch",
    "total_tests",
    "passed_tests",
    "failed_tests",
    "duration",
    "success_rate"
  ],
  "values": "={{[$json.timestamp, $json.test_type, $json.environment, $json.branch, $json.total_tests, $json.passed_tests, $json.failed_tests, $json.duration, $json.success_rate]}}"
}
```

**Node 6: Generate HTML Report**
```python
# Code Node
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>OpenEvolve Test Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Type:</strong> {test_type}</p>
        <p><strong>Environment:</strong> {environment}</p>
        <p><strong>Branch:</strong> {branch}</p>
        <p><strong>Total Tests:</strong> {total_tests}</p>
        <p class="passed"><strong>Passed:</strong> {passed_tests}</p>
        <p class="failed"><strong>Failed:</strong> {failed_tests}</p>
        <p><strong>Success Rate:</strong> {success_rate}%</p>
        <p><strong>Duration:</strong> {duration}s</p>
    </div>
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Status</th>
            <th>Duration</th>
        </tr>
        {test_rows}
    </table>
</body>
</html>
"""

test_results = $input.first().json
test_rows = ""

for test in test_results['tests']:
    status_class = "passed" if test['status'] == 'passed' else "failed"
    test_rows += f"""
    <tr>
        <td>{test['name']}</td>
        <td class="{status_class}">{test['status']}</td>
        <td>{test['time']}s</td>
    </tr>
    """

html_report = html_template.format(
    timestamp=test_results['timestamp'],
    test_type=test_results['test_type'],
    environment=test_results['environment'],
    branch=test_results['branch'],
    total_tests=test_results['total_tests'],
    passed_tests=test_results['passed_tests'],
    failed_tests=test_results['failed_tests'],
    success_rate=test_results['success_rate'],
    duration=test_results['duration'],
    test_rows=test_rows
)

return [{'json': {'html_report': html_report}}]
```

**Node 7: Send Report**
```json
{
  "to": "team@openevolve.com",
  "subject": "=Test Results: {{$json.test_type}} ({{$json.passed_tests}}/{{$json.total_tests}} passed)",
  "html": "={{$json.html_report}}"
}
```

### 3.3 Infrastructure Orchestrator

**Purpose:** Automate infrastructure startup with health verification and service dependencies.

**Trigger:** Manual (Webhook) + Schedule (Morning startup)

**Workflow Steps:**

```
┌─────────────────┐
│ Trigger         │
│ (Manual/Time)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check Current   │
│ State           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Already Running?│
└────┬───────┬────┘
     │       │
 Yes │       │ No
     │       │
     ▼       ▼
┌───────────┐ ┌─────────────────┐
│ Exit      │ │ Start Services  │
│ (Log OK)  │ │ In Order        │
└───────────┘ └────────┬────────┘
                      │
                      ▼
              ┌─────────────────┐
              │ Start Qdrant    │
              │ Wait & Verify   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Start PostgreSQL │
              │ Wait & Verify   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Start Redis     │
              │ Wait & Verify   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Start OpenEvolve│
              │ API & UI        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Final Health    │
              │ Check           │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Send Ready      │
              │ Notification    │
              └─────────────────┘
```

**Implementation Details:**

**Node 1: Manual Trigger**
```json
{
  "webhookId": "infra-orchestrator"
}
```

**Node 2: Check Current State**
```bash
# Code Node (Shell)
docker-compose ps --services --filter "status=running"
```

**Node 3: Conditional Check**
```javascript
const runningServices = $input.first().json.stdout.trim().split('\n');
const requiredServices = ['qdrant', 'postgres', 'redis', 'openevolve-api'];
const allRunning = requiredServices.every(svc => runningServices.includes(svc));

return [{
  json: {
    all_running: allRunning,
    running_services: runningServices
  }
}];
```

**Node 4: Start Services**
```bash
# Code Node (Shell)
cd /openevolve
docker-compose -f docker-compose.infrastructure.yml up -d
```

**Node 5-7: Start Individual Services with Wait**
```bash
# Code Node (Shell)
# Start Qdrant
docker-compose up -d qdrant

# Wait for healthy status
timeout=60
until docker-compose ps | grep qdrant | grep -q "healthy"; do
  sleep 2
  timeout=$((timeout - 2))
  if [ $timeout -le 0 ]; then
    echo "Qdrant failed to start"
    exit 1
  fi
done

echo "Qdrant is ready"
```

**Node 8: Final Health Check**
```python
# Code Node (Python)
import requests
import json

services = {
    'Qdrant': 'http://qdrant:6333/health',
    'PostgreSQL': 'http://postgres:5432',
    'Redis': 'http://redis:6379/ping',
    'OpenEvolve API': 'http://openevolve-api:8000/health'
}

results = {}
for name, url in services.items():
    try:
        response = requests.get(url, timeout=5)
        results[name] = {
            'status': 'healthy' if response.status_code == 200 else 'unhealthy',
            'response_time': response.elapsed.total_seconds()
        }
    except Exception as e:
        results[name] = {
            'status': 'unhealthy',
            'error': str(e)
        }

all_healthy = all(r['status'] == 'healthy' for r in results.values())

return [{
  'json': {
    'all_healthy': all_healthy,
    'services': results,
    'timestamp': datetime.utcnow().isoformat()
  }
}]
```

**Node 9: Send Notification**
```json
{
  "channel": "#openevolve-status",
  "text": "=✅ OpenEvolve Infrastructure Ready\n\nTimestamp: {{$json.timestamp}}\n\nServices:\n{{JSON.stringify($json.services, null, 2)}}\n\nAccess URLs:\n- API: http://localhost:8000\n- UI: http://localhost:8501\n- Qdrant Dashboard: http://localhost:6333/dashboard",
  "username": "OpenEvolve Infrastructure",
  "iconEmoji": ":white_check_mark:"
}
```

---

## 4. Development Automation

### 4.1 Dependency Update Monitor

**Purpose:** Automatically monitor dependencies for updates and vulnerabilities.

**Trigger:** Daily (Cron)

**Workflow Steps:**

```
┌─────────────────┐
│ Daily Trigger   │
└────────┬────────┘
         │
         ├──────────────────────────┐
         │                          │
         ▼                          ▼
┌─────────────────┐      ┌─────────────────┐
│ Check Python    │      │ Check Node.js   │
│ Dependencies    │      │ Dependencies    │
│ (pip-audit)     │      │ (npm audit)     │
└────────┬────────┘      └────────┬────────┘
         │                         │
         └───────────┬─────────────┘
                     │
                     ▼
           ┌─────────────────┐
           │ Merge Results   │
           └────────┬────────┘
                    │
                    ▼
          ┌─────────────────┐
          │ Critical Vulns? │
          └────┬───────┬────┘
               │       │
       No      │       │  Yes
      ┌────────┘       └─────────┐
      │                          │
      ▼                          ▼
┌───────────┐           ┌─────────────────┐
│ Log & Send│           │ Immediate Alert │
│ Weekly    │           │ + Create PR     │
│ Report    │           │ to Update       │
└───────────┘           └─────────────────┘
```

**Implementation Details:**

**Node 1: Cron Trigger**
```json
{
  "cronExpression": "0 9 * * *"
}
```

**Node 2: Check Python Dependencies**
```bash
# Code Node (Shell)
pip-audit --format json --output /tmp/python-audit.json
cat /tmp/python-audit.json
```

**Node 3: Check Node.js Dependencies**
```bash
# Code Node (Shell)
npm audit --json > /tmp/npm-audit.json
cat /tmp/npm-audit.json
```

**Node 4: Merge & Analyze**
```python
# Code Node (Python)
import json

python_audit = json.loads($input.first().json.stdout)
npm_audit = json.loads($input.all()[1].json.stdout)

# Extract vulnerabilities
python_vulns = python_audit.get('vulnerabilities', [])
npm_vulns = npm_audit.get('vulnerabilities', {})

# Count by severity
def count_by_severity(vulns):
    return {
        'critical': len([v for v in vulns if v.get('severity') == 'critical']),
        'high': len([v for v in vulns if v.get('severity') == 'high']),
        'medium': len([v for v in vulns if v.get('severity') == 'medium']),
        'low': len([v for v in vulns if v.get('severity') == 'low'])
    }

python_counts = count_by_severity(python_vulns)
npm_counts = {
    'critical': npm_audit.get('metadata', {}).get('vulnerabilities', {}).get('critical', 0),
    'high': npm_audit.get('metadata', {}).get('vulnerabilities', {}).get('high', 0),
    'medium': npm_audit.get('metadata', {}).get('vulnerabilities', {}).get('medium', 0),
    'low': npm_audit.get('metadata', {}).get('vulnerabilities', {}).get('low', 0)
}

total_critical = python_counts['critical'] + npm_counts['critical']
total_high = python_counts['high'] + npm_counts['high']

return [{
  'json': {
    'has_critical': total_critical > 0,
    'has_high': total_high > 0,
    'python_vulns': python_vulns,
    'npm_vulns': npm_vulns,
    'summary': {
      'python': python_counts,
      'npm': npm_counts
    }
  }
}]
```

**Node 5: Conditional Check**
```json
{
  "conditions": {
    "boolean": [
      {
        "value1": "={{$json.has_critical}}",
        "operation": "true"
      }
    ]
  }
}
```

**Node 6: Create GitHub PR (If Critical)**
```python
# Code Node (Python)
import requests

vulnerabilities = $input.first().json.python_vulns
critical_vulns = [v for v in vulnerabilities if v.get('severity') == 'critical']

# Create PR description
pr_body = f"""## Critical Security Vulnerabilities Detected

This PR addresses {len(critical_vulns)} critical vulnerabilities:

"""

for vuln in critical_vulns:
    pr_body += f"""
### {vuln.get('name', 'Unknown')}

- **Affected Versions**: {vuln.get('affected_versions', 'N/A')}
- **Fixed Versions**: {vuln.get('fixed_versions', 'N/A')}
- **CVE**: {vuln.get('CVE', 'N/A')}
- **Advisory**: {vuln.get('advisory', 'N/A')}

"""

# Create PR via GitHub API
response = requests.post(
    'https://api.github.com/repos/your-org/openevolve/pulls',
    headers={
        'Authorization': f'token {$credentials.github_token}',
        'Accept': 'application/vnd.github.v3+json'
    },
    json={
        'title': f'Security: Fix {len(critical_vulns)} critical vulnerabilities',
        'body': pr_body,
        'head': 'automated/security-fix',
        'base': 'main'
    }
)

return [{'json': {'pr_url': response.json().get('html_url')}}]
```

**Node 7: Send Immediate Alert**
```json
{
  "channel": "#openevolve-security",
  "text": "=🚨 Critical Security Vulnerabilities Detected!\n\nPython: {{$json.summary.python.critical}} critical\nNode.js: {{$json.summary.npm.critical}} critical\n\nPR Created: {{$json.pr_url}}\n\nAction Required: Review and merge immediately.",
  "username": "Security Monitor",
  "iconEmoji": ":rotating_light:"
}
```

### 4.2 Code Quality Gatekeeper

**Purpose:** Enforce code quality standards before commits and in CI/CD.

**Trigger:** Pre-commit hook + Nightly

**Workflow Steps:**

```
┌─────────────────┐
│ Trigger         │
│ (Pre-commit)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Get Changed     │
│ Files           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run Python      │
│ Linters         │
│ (Black, Flake8) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run Node.js     │
│ Linters         │
│ (ESLint, Prettier)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run Type Check  │
│ (MyPy, tsc)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run Security    │
│ Scan (Bandit)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Any Issues?     │
└────┬───────┬────┘
     │       │
  No │       │ Yes
     │       │
     ▼       ▼
┌───────────┐ ┌─────────────────┐
│ Allow     │ │ Block Commit    │
│ Commit    │ │ + Show Fixes    │
└───────────┘ └─────────────────┘
```

**Implementation Details:**

**Node 1: Pre-commit Hook Trigger**
```python
# Code Node (Python)
import subprocess
import os

# Get staged files
result = subprocess.run(
    ['git', 'diff', '--cached', '--name-only'],
    capture_output=True,
    text=True
)

staged_files = result.stdout.strip().split('\n')
python_files = [f for f in staged_files if f.endswith('.py')]
js_files = [f for f in staged_files if any(f.endswith(ext) for ext in ['.js', '.jsx', '.ts', '.tsx'])]

return [{
  'json': {
    'python_files': python_files,
    'js_files': js_files,
    'has_changes': bool(python_files or js_files)
  }
}]
```

**Node 2: Run Black (Python Formatter)**
```bash
# Code Node (Shell)
if [ -n "{{$json.python_files}}" ]; then
  black --check --diff {{' '.join($json.python_files)}}
else
  echo "No Python files to check"
fi
```

**Node 3: Run Flake8 (Python Linter)**
```bash
# Code Node (Shell)
if [ -n "{{$json.python_files}}" ]; then
  flake8 {{' '.join($json.python_files)}} --format=json --output-file=/tmp/flake8.json
  cat /tmp/flake8.json
else
  echo '{"[]": []}'
fi
```

**Node 4: Run ESLint (JavaScript Linter)**
```bash
# Code Node (Shell)
if [ -n "{{$json.js_files}}" ]; then
  eslint {{' '.join($json.js_files)}} --format=json
else
  echo '[]'
fi
```

**Node 5: Run MyPy (Type Checker)**
```bash
# Code Node (Shell)
if [ -n "{{$json.python_files}}" ]; then
  mypy {{' '.join($json.python_files)}} --json-report /tmp/mypy.json
  cat /tmp/mypy.json
else
  echo '{}'
fi
```

**Node 6: Run Bandit (Security Linter)**
```bash
# Code Node (Shell)
if [ -n "{{$json.python_files}}" ]; then
  bandit -r {{' '.join($json.python_files)}} -f json
else
  echo '{"results": []}'
fi
```

**Node 7: Aggregate Results**
```python
# Code Node (Python)
import json

# Collect all results
black_result = $input.all()[0].json
flake8_result = json.loads($input.all()[1].json.stdout)
eslint_result = json.loads($input.all()[2].json.stdout)
mypy_result = json.loads($input.all()[3].json.stdout)
bandit_result = json.loads($input.all()[4].json.stdout)

issues = []

# Parse Black result
if black_result.get('error'):
    issues.append({
        'tool': 'black',
        'severity': 'error',
        'message': black_result['error'],
        'fix': 'Run: black <files>'
    })

# Parse Flake8 result
for file_path, file_issues in flake8_result.items():
    for issue in file_issues:
        issues.append({
            'tool': 'flake8',
            'file': file_path,
            'line': issue.get('line'),
            'column': issue.get('column'),
            'severity': 'warning',
            'code': issue.get('code'),
            'message': issue.get('text')
        })

# Parse ESLint result
for result in eslint_result:
    for message in result.get('messages', []):
        if message.get('severity') <= 2:  # Error or Warning
            issues.append({
                'tool': 'eslint',
                'file': result.get('filePath'),
                'line': message.get('line'),
                'column': message.get('column'),
                'severity': 'error' if message.get('severity') == 1 else 'warning',
                'rule': message.get('ruleId'),
                'message': message.get('message'),
                'fix': message.get('fix')
            })

# Parse MyPy result
if 'error' in mypy_result:
    issues.append({
        'tool': 'mypy',
        'severity': 'error',
        'message': mypy_result['error']
    })

# Parse Bandit result
for issue in bandit_result.get('results', []):
    issues.append({
        'tool': 'bandit',
        'file': issue.get('filename'),
        'line': issue.get('line_number'),
        'severity': issue.get('issue_severity'),
        'confidence': issue.get('issue_confidence'),
        'code': issue.get('test_id'),
        'message': issue.get('issue_text')
    })

return [{
  'json': {
    'has_issues': len(issues) > 0,
    'total_issues': len(issues),
    'issues': issues
  }
}]
```

**Node 8: Conditional Check**
```json
{
  "conditions": {
    "boolean": [
      {
        "value1": "={{$json.has_issues}}",
        "operation": "true"
      }
    ]
  }
}
```

**Node 9: Block Commit with Fixes**
```python
# Code Node (Python)
issues = $input.first().json.issues

message = """## ❌ Code Quality Check Failed

Your commit has been blocked due to {count} issue(s):

""".format(count=len(issues))

# Group by severity
errors = [i for i in issues if i.get('severity') == 'error']
warnings = [i for i in issues if i.get('severity') == 'warning']

if errors:
    message += f"### Errors ({len(errors)})\n\n"
    for issue in errors[:10]:  # Show first 10
        message += f"- **{issue.get('tool', 'Unknown')}**: {issue.get('message', 'No message')}\n"
        if 'file' in issue:
            message += f"  File: {issue['file']}:{issue.get('line', '?')}\n"
        if 'fix' in issue:
            message += f"  Fix: {issue['fix']}\n"
    message += "\n"

if warnings:
    message += f"### Warnings ({len(warnings)})\n\n"
    for issue in warnings[:10]:
        message += f"- **{issue.get('tool', 'Unknown')}**: {issue.get('message', 'No message')}\n"

message += """

## Automatic Fixes

Run the following commands to fix most issues:

```bash
# Python formatting
black .

# Python linting
# (Manual fixes required)

# JavaScript/TypeScript formatting
npm run format

# JavaScript/TypeScript linting
npm run lint -- --fix
```

"""

print(message)
exit(1)  # Block commit
```

### 4.3 Documentation Generator

**Purpose:** Automatically generate and update documentation from code.

**Trigger:** On merge to main + Daily

**Workflow Steps:**

```
┌─────────────────┐
│ Trigger         │
│ (On Merge)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract Code    │
│ Changes         │
└────────┬────────┘
         │
         ├────────────────────────────┐
         │                            │
         ▼                            ▼
┌─────────────────┐        ┌─────────────────┐
│ Generate API    │        │ Generate        │
│ Documentation   │        │ Changelog       │
│ (Docstrings,    │        │ (Git Log)       │
│  OpenAPI)       │        │                 │
└────────┬────────┘        └────────┬────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Update Docs     │
              │ Branch          │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Create PR       │
              │ for Review      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Notify Team     │
              └─────────────────┘
```

**Implementation Details:**

**Node 1: GitHub Webhook Trigger**
```json
{
  "path": "docs-generator",
  "responseMode": "responseNode",
  "options": {}
}
```

**Node 2: Extract Changed Files**
```python
# Code Node (Python)
import os
import subprocess

# Get changed Python files
result = subprocess.run(
    ['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'],
    capture_output=True,
    text=True
)

changed_files = result.stdout.strip().split('\n')
python_files = [f for f in changed_files if f.endswith('.py') and os.path.exists(f)]

return [{
  'json': {
    'changed_files': changed_files,
    'python_files': python_files,
    'commit_hash': os.environ.get('GITHUB_SHA', 'unknown'),
    'commit_message': os.environ.get('GITHUB_COMMIT_MESSAGE', '')
  }
}]
```

**Node 3: Extract Docstrings**
```python
# Code Node (Python)
import ast
import inspect

python_files = $input.first().json.python_files

api_docs = {}

for file_path in python_files:
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read(), filename=file_path)

    file_docs = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                file_docs.append({
                    'name': node.name,
                    'type': node.__class__.__name__,
                    'line': node.lineno,
                    'docstring': docstring
                })

    api_docs[file_path] = file_docs

return [{'json': {'api_docs': api_docs}}]
```

**Node 4: Generate OpenAPI Spec**
```python
# Code Node (Python)
import json

# Extract FastAPI routes
from openevolve_api import app

openapi_spec = app.openapi()

return [{'json': {'openapi_spec': openapi_spec}}]
```

**Node 5: Generate Changelog**
```bash
# Code Node (Shell)
# Generate changelog since last tag
git log $(git describe --tags --abbrev=0 HEAD~1)..HEAD --pretty=format:"- %s (%h)" > /tmp/changelog.md
cat /tmp/changelog.md
```

**Node 6: Update Documentation Files**
```python
# Code Node (Python)
import os

api_docs = $input.first().json.api_docs
openapi_spec = $input.all()[1].json.openapi_spec
changelog = $input.all()[2].json.stdout

# Create API documentation
api_doc_content = """# OpenEvolve API Documentation

Generated automatically from code.

## OpenAPI Specification

```json
{openapi}
```

## Function Documentation

"""

for file_path, functions in api_docs.items():
    api_doc_content += f"### {file_path}\n\n"

    for func in functions:
        api_doc_content += f"#### {func['name']} ({func['type']})\n\n"
        api_doc_content += f"**Line:** {func['line']}\n\n"
        api_doc_content += f"{func['docstring']}\n\n"

# Write to docs branch
os.system('git checkout docs')
os.makedirs('docs/api', exist_ok=True)

with open('docs/api/auto-generated.md', 'w') as f:
    f.write(api_doc_content)

with open('CHANGELOG.md', 'a') as f:
    f.write(f"\n## {datetime.date.today()}\n\n")
    f.write(changelog)
    f.write("\n")

# Commit changes
os.system('git add docs/api/auto-generated.md CHANGELOG.md')
os.system('git commit -m "docs: auto-generate API documentation and changelog"')

return [{'json': {'docs_updated': True}}]
```

**Node 7: Create Pull Request**
```python
# Code Node (Python)
import requests

response = requests.post(
    'https://api.github.com/repos/your-org/openevolve/pulls',
    headers={
        'Authorization': f'token {$credentials.github_token}',
        'Accept': 'application/vnd.github.v3+json'
    },
    json={
        'title': 'docs: automated documentation update',
        'body': 'Automatically generated documentation including:\n- API documentation\n- OpenAPI specification\n- Changelog\n\nPlease review and merge.',
        'head': 'docs',
        'base': 'main'
    }
)

return [{'json': {'pr_url': response.json().get('html_url')}}]
```

**Node 8: Notify Team**
```json
{
  "channel": "#openevolve-docs",
  "text": "=📚 Documentation Updated\n\nPR Created: {{$json.pr_url}}\n\nChanges:\n- API documentation regenerated\n- OpenAPI specification updated\n- Changelog updated\n\nPlease review and merge.",
  "username": "Docs Bot",
  "iconEmoji": ":book:"
}
```

---

## 5. CI/CD Integration

### 5.1 GitHub Actions Coordinator

**Purpose:** Orchestrate GitHub Actions workflows from n8n for complex scenarios.

**Trigger:** GitHub Webhook (Push, PR, Release)

**Workflow Steps:**

```
┌─────────────────┐
│ GitHub Webhook  │
│ (Push/PR)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Parse Event     │
│ Determine Type  │
└────────┬────────┘
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────┐
│ On Push to Main │          │ On PR           │
└────────┬────────┘          └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐          ┌─────────────────┐
│ Trigger Release │          │ Trigger Tests   │
│ Workflow        │          │ + Review Apps   │
└────────┬────────┘          └────────┬────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Monitor Progress│
              │ (Poll GitHub)   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ On Complete     │
              │ Notify Status   │
              └─────────────────┘
```

**Implementation Details:**

**Node 1: GitHub Webhook Trigger**
```json
{
  "path": "github-coordinator",
  "responseMode": "responseNode",
  "options": {}
}
```

**Node 2: Parse Event**
```python
# Code Node (Python)
import json

event_data = $input.first().json
event_type = event_data.get('event_type')

if event_type == 'push':
    branch = event_data.get('ref', '').replace('refs/heads/', '')
    commit_hash = event_data.get('after')
    committer = event_data.get('pusher', {}).get('name')
    commit_message = event_data.get('head_commit', {}).get('message')

    return [{
      'json': {
        'event_type': 'push',
        'branch': branch,
        'commit_hash': commit_hash,
        'committer': committer,
        'commit_message': commit_message,
        'is_main': branch == 'main',
        'is_release': branch.startswith('release/')
      }
    }]

elif event_type == 'pull_request':
    action = event_data.get('action')
    pr_number = event_data.get('number')
    pr_title = event_data.get('pull_request', {}).get('title')
    source_branch = event_data.get('pull_request', {}).get('head', {}).get('ref')
    target_branch = event_data.get('pull_request', {}).get('base', {}).get('ref')
    author = event_data.get('pull_request', {}).get('user', {}).get('login')

    return [{
      'json': {
        'event_type': 'pr',
        'action': action,
        'pr_number': pr_number,
        'pr_title': pr_title,
        'source_branch': source_branch,
        'target_branch': target_branch,
        'author': author
      }
    }]
```

**Node 3: Route by Event Type**
```json
{
  "conditions": {
    "string": [
      {
        "value1": "={{$json.event_type}}",
        "operation": "equals",
        "value2": "push"
      }
    ]
  }
}
```

**Node 4: Trigger Release Workflow (Push to Main)**
```python
# Code Node (Python)
import requests

commit_hash = $input.first().json.commit_hash

# Trigger GitHub Actions workflow
response = requests.post(
    f'https://api.github.com/repos/your-org/openevolve/actions/workflows/deploy.yml/dispatches',
    headers={
        'Authorization': f'token {$credentials.github_token}',
        'Accept': 'application/vnd.github.v3+json'
    },
    json={
        'ref': 'main',
        'inputs': {
            'commit_hash': commit_hash,
            'triggered_by': 'n8n-coordinator'
        }
    }
)

return [{'json': {'workflow_triggered': True, 'commit_hash': commit_hash}}]
```

**Node 5: Trigger Test Workflow (PR)**
```python
# Code Node (Python)
import requests

pr_number = $input.first().json.pr_number
source_branch = $input.first().json.source_branch

# Trigger test workflow
response = requests.post(
    f'https://api.github.com/repos/your-org/openevolve/actions/workflows/test.yml/dispatches',
    headers={
        'Authorization': f'token {$credentials.github_token}',
        'Accept': 'application/vnd.github.v3+json'
    },
    json={
        'ref': source_branch,
        'inputs': {
            'pr_number': str(pr_number),
            'triggered_by': 'n8n-coordinator'
        }
    }
)

return [{'json': {'workflow_triggered': True, 'pr_number': pr_number}}]
```

**Node 6: Monitor Workflow Progress**
```python
# Code Node (Python)
import requests
import time

commit_hash = $input.first().json.commit_hash

# Poll GitHub Actions for workflow run status
max_attempts = 60
attempt = 0

while attempt < max_attempts:
    response = requests.get(
        f'https://api.github.com/repos/your-org/openevolve/actions/runs?head_sha={commit_hash}',
        headers={
            'Authorization': f'token {$credentials.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
    )

    runs = response.json().get('workflow_runs', [])

    if runs:
        run = runs[0]
        status = run.get('status')
        conclusion = run.get('conclusion')

        if status == 'completed':
            return [{
              'json': {
                'status': 'completed',
                'conclusion': conclusion,
                'run_url': run.get('html_url'),
                'commit_hash': commit_hash
              }
            }]

    attempt += 1
    time.sleep(10)

return [{'json': {'status': 'timeout', 'commit_hash': commit_hash}}]
```

**Node 7: Notify Status**
```python
# Code Node (Python)
status = $input.first().json

if status['conclusion'] == 'success':
    message = f"""✅ Deployment Successful!

Commit: {status['commit_hash']}
Run: {status['run_url']}

Environment: Production
Status: Live
"""
else:
    message = f"""❌ Deployment Failed!

Commit: {status['commit_hash']}
Run: {status['run_url']}

Conclusion: {status['conclusion']}

Action Required: Investigate and fix.
"""

# Send to Slack
print(message)
```

### 5.2 Deployment Pipeline Orchestrator

**Purpose:** Coordinate multi-stage deployments with health checks and rollback capability.

**Trigger:** Manual + On merge to main

**Workflow Steps:**

```
┌─────────────────┐
│ Trigger         │
│ (Manual/Auto)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Get Latest      │
│ Release Tag     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Deploy to       │
│ Staging         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run Smoke Tests │
│ on Staging      │
└────────┬────────┘
         │
         ├──────────────────────────┐
         │                          │
         ▼                          ▼
┌─────────────────┐        ┌─────────────────┐
│ Tests Pass?     │        │ Tests Fail?     │
└────┬───────┬────┘        └────────┬────────┘
     │       │                     │
  Yes │       │ No               Yes │
     │       │                     │
     ▼       ▼                     ▼
┌───────────┐ ┌───────────┐   ┌─────────────────┐
│ Deploy to │ │ Rollback  │   │ Notify Failure  │
│ Production│ │ Staging   │   │ + Create Issue  │
└─────┬─────┘ └───────────┘   └─────────────────┘
      │
      ▼
┌─────────────────┐
│ Run Smoke Tests │
│ on Production   │
└────────┬────────┘
         │
         ├──────────────────────────┐
         │                          │
         ▼                          ▼
┌─────────────────┐        ┌─────────────────┐
│ Tests Pass?     │        │ Tests Fail?     │
└────┬───────┬────┘        └────────┬────────┘
     │       │                     │
  Yes │       │ No               Yes │
     │       │                     │
     ▼       ▼                     ▼
┌───────────┐ ┌───────────┐   ┌─────────────────┐
│ Notify    │ │ Rollback  │   │ Rollback +      │
│ Success   │ │ Production│   │ Incident Alert  │
└───────────┘ └───────────┘   └─────────────────┘
```

**Implementation Details:**

**Node 1: Manual Trigger**
```json
{
  "webhookId": "deployment-orchestrator"
}
```

**Node 2: Get Latest Release**
```python
# Code Node (Python)
import requests

response = requests.get(
    'https://api.github.com/repos/your-org/openevolve/releases/latest',
    headers={
        'Authorization': f'token {$credentials.github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
)

release = response.json()
tag_name = release.get('tag_name')
commit_hash = release.get('target_commitish')

return [{
  'json': {
    'tag_name': tag_name,
    'commit_hash': commit_hash,
    'release_notes': release.get('body')
  }
}]
```

**Node 3: Deploy to Staging**
```bash
# Code Node (Shell)
#!/bin/bash
set -e

TAG="{{$json.tag_name}}"
ENVIRONMENT="staging"

echo "Deploying $TAG to $ENVIRONMENT..."

# Pull latest images
docker-compose -f docker-compose.$ENVIRONMENT.yml pull

# Start new containers
docker-compose -f docker-compose.$ENVIRONMENT.yml up -d

# Wait for health checks
sleep 30

echo "Deployment to $ENVIRONMENT complete"
echo "URL: http://staging.openevolve.com"
```

**Node 4: Run Smoke Tests (Staging)**
```python
# Code Node (Python)
import requests
import json

staging_url = 'http://staging.openevolve.com'

tests = [
    {
        'name': 'Health Check',
        'url': f'{staging_url}/api/health',
        'expected_status': 200
    },
    {
        'name': 'API Documentation',
        'url': f'{staging_url}/docs',
        'expected_status': 200
    },
    {
        'name': 'Qdrant Connection',
        'url': f'{staging_url}/api/qdrant/health',
        'expected_status': 200
    },
    {
        'name': 'PostgreSQL Connection',
        'url': f'{staging_url}/api/db/health',
        'expected_status': 200
    }
]

results = []
all_passed = True

for test in tests:
    try:
        response = requests.get(test['url'], timeout=10)
        passed = response.status_code == test['expected_status']

        if not passed:
            all_passed = False

        results.append({
            'name': test['name'],
            'passed': passed,
            'status': response.status_code,
            'response_time': response.elapsed.total_seconds()
        })
    except Exception as e:
        all_passed = False
        results.append({
            'name': test['name'],
            'passed': False,
            'error': str(e)
        })

return [{
  'json': {
    'all_passed': all_passed,
    'tests': results,
    'environment': 'staging'
  }
}]
```

**Node 5: Conditional Check**
```json
{
  "conditions": {
    "boolean": [
      {
        "value1": "={{$json.all_passed}}",
        "operation": "true"
      }
    ]
  }
}
```

**Node 6: Deploy to Production**
```bash
# Code Node (Shell)
#!/bin/bash
set -e

TAG="{{$json.tag_name}}"
ENVIRONMENT="production"

echo "Deploying $TAG to $ENVIRONMENT..."

# Blue-green deployment
CURRENT="blue"
NEW="green"

# Determine which is active
if docker-compose -f docker-compose.$ENVIRONMENT.yml ps | grep -q "blue"; then
    CURRENT="blue"
    NEW="green"
else
    CURRENT="green"
    NEW="blue"
fi

# Deploy to inactive environment
export DEPLOYMENT_COLOR=$NEW
docker-compose -f docker-compose.$ENVIRONMENT.yml up -d

# Wait for health checks
sleep 30

# Run smoke tests on new deployment
python scripts/smoke_tests.py --environment $ENVIRONMENT --color $NEW

if [ $? -eq 0 ]; then
    # Switch traffic to new deployment
    export DEPLOYMENT_COLOR=$NEW
    docker-compose -f docker-compose.$ENVIRONMENT.yml up -d

    # Scale down old deployment
    export DEPLOYMENT_COLOR=$CURRENT
    docker-compose -f docker-compose.$ENVIRONMENT.yml stop

    echo "Deployment to $ENVIRONMENT successful"
else
    # Rollback
    export DEPLOYMENT_COLOR=$NEW
    docker-compose -f docker-compose.$ENVIRONMENT.yml down

    echo "Deployment failed, rolled back"
    exit 1
fi
```

**Node 7: Rollback (Production)**
```bash
# Code Node (Shell)
#!/bin/bash

ENVIRONMENT="production"
FAILED_DEPLOYMENT="{{$json.deployment_color}}"

echo "Rolling back $FAILED_DEPLOYMENT..."

# Stop failed deployment
export DEPLOYMENT_COLOR=$FAILED_DEPLOYMENT
docker-compose -f docker-compose.$ENVIRONMENT.yml down

# Ensure active deployment is running
if [ "$FAILED_DEPLOYMENT" = "blue" ]; then
    export DEPLOYMENT_COLOR="green"
else
    export DEPLOYMENT_COLOR="blue"
fi

docker-compose -f docker-compose.$ENVIRONMENT.yml up -d

echo "Rollback complete"
```

**Node 8: Notify Success**
```json
{
  "channel": "#openevolve-deployments",
  "text": "=✅ Deployment Successful!\n\nRelease: {{$json.tag_name}}\nCommit: {{$json.commit_hash}}\nEnvironment: Production\n\nSmoke Tests: All Passed ({{$json.tests.length}} tests)\n\nRelease Notes:\n{{$json.release_notes}}",
  "username": "Deploy Bot",
  "iconEmoji": ":rocket:"
}
```

**Node 9: Notify Failure (Incident Alert)**
```json
{
  "channel": "#openevolve-incidents",
  "text": "=🚨 Deployment Failed!\n\nRelease: {{$json.tag_name}}\nEnvironment: Production\n\nFailed Tests:\n{{JSON.stringify($json.tests, null, 2)}}\n\nAction Required: Rollback initiated. Investigate immediately.",
  "username": "Deploy Bot",
  "iconEmoji": ":rotating_light:"
}
```

---

## 6. Monitoring & Alerting

### 6.1 Log Aggregation & Analysis

**Purpose:** Collect, parse, and analyze logs from all services for errors and anomalies.

**Trigger:** Continuous (Cron every minute)

**Workflow Steps:**

```
┌─────────────────┐
│ Cron Trigger    │
│ (Every 1 min)   │
└────────┬────────┘
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────┐
│ Collect Logs    │          │ Collect Metrics │
│ (Docker logs)   │          │ (API calls)     │
└────────┬────────┘          └────────┬────────┘
         │                              │
         └──────────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Parse & Filter  │
              │ Extract Errors  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Detect Anomalies│
              │ (Error spikes,  │
              │  Patterns)      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Store in        │
              │ PostgreSQL      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Alert on        │
              │ Critical Errors │
              └─────────────────┘
```

**Implementation Details:**

**Node 1: Cron Trigger**
```json
{
  "cronExpression": "* * * * *"
}
```

**Node 2: Collect Docker Logs**
```bash
# Code Node (Shell)
#!/bin/bash

# Get logs from all services in the last minute
services=("qdrant" "postgres" "redis" "openevolve-api" "bubble-studio")

for service in "${services[@]}"; do
  echo "=== $service ==="
  docker logs --since 1m "$service" 2>&1
  echo ""
done
```

**Node 3: Parse Logs**
```python
# Code Node (Python)
import re
import json
from datetime import datetime, timedelta

logs = $input.first().json.stdout
service_pattern = r'=== (.+?) ==='
timestamp_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
error_patterns = [
    r'ERROR',
    r'CRITICAL',
    r'Exception',
    r'Traceback',
    r'failed',
    r'error'
]

# Parse logs by service
parsed_logs = []
current_service = None

for line in logs.split('\n'):
    service_match = re.search(service_pattern, line)
    if service_match:
        current_service = service_match.group(1)
        continue

    if current_service:
        # Check if line contains errors
        is_error = any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns)

        parsed_logs.append({
            'service': current_service,
            'message': line,
            'is_error': is_error,
            'timestamp': datetime.utcnow().isoformat()
        })

# Filter errors only
errors = [log for log in parsed_logs if log['is_error']]

return [{
  'json': {
    'total_logs': len(parsed_logs),
    'error_count': len(errors),
    'errors': errors
  }
}]
```

**Node 4: Detect Anomalies**
```python
# Code Node (Python)
from collections import defaultdict

errors = $input.first().json.errors

# Group errors by service and type
service_errors = defaultdict(lambda: defaultdict(int))
error_patterns = defaultdict(int)

for error in errors:
    service = error['service']
    message = error['message']

    # Count by service
    service_errors[service]['total'] += 1

    # Extract error type
    error_type = message.split(':')[0] if ':' in message else 'Unknown'
    service_errors[service][error_type] += 1
    error_patterns[error_type] += 1

# Detect anomalies (error spikes)
# Compare with baseline from PostgreSQL
# For now, simple threshold-based detection
threshold = 10  # More than 10 errors per minute is anomalous

anomalies = []
for service, counts in service_errors.items():
    if counts['total'] > threshold:
        anomalies.append({
            'service': service,
            'error_count': counts['total'],
            'severity': 'high' if counts['total'] > 50 else 'medium',
            'top_errors': sorted(
                [(k, v) for k, v in counts.items() if k != 'total'],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        })

return [{
  'json': {
    'has_anomalies': len(anomalies) > 0,
    'anomaly_count': len(anomalies),
    'anomalies': anomalies
  }
}]
```

**Node 5: Store in PostgreSQL**
```json
{
  "operation": "insert",
  "table": "log_metrics",
  "columns": [
    "timestamp",
    "service",
    "log_level",
    "error_count",
    "message"
  ],
  "values": "={{[new Date().toISOString(), $json.service, $json.level, $json.count, $json.message]}}"
}
```

**Node 6: Conditional Alert**
```json
{
  "conditions": {
    "boolean": [
      {
        "value1": "={{$json.has_anomalies}}",
        "operation": "true"
      }
    ]
  }
}
```

**Node 7: Send Alert**
```python
# Code Node (Python)
anomalies = $input.first().json.anomalies

message = "⚠️ Log Anomaly Detected\n\n"

for anomaly in anomalies:
    message += f"""
Service: {anomaly['service']}
Error Count: {anomaly['error_count']}
Severity: {anomaly['severity']}

Top Errors:
"""
    for error_type, count in anomaly['top_errors']:
        message += f"  - {error_type}: {count}\n"

    message += "\n"

# Send to Slack
print(message)
```

### 6.2 Performance Regression Detector

**Purpose:** Monitor performance metrics and detect regressions over time.

**Trigger:** Post-deployment + Weekly

**Workflow Steps:**

```
┌─────────────────┐
│ Trigger         │
│ (Post-deploy)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run Benchmarks  │
│ (Load tests)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Collect Metrics │
│ (Response times,│
│  Throughput)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Compare with    │
│ Baseline        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Calculate Delta │
│ (Percent change)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Regression?     │
│ (>10% slower)   │
└────┬───────┬────┘
     │       │
  No │       │ Yes
     │       │
     ▼       ▼
┌───────────┐ ┌─────────────────┐
│ Log OK    │ │ Alert & Create  │
│           │ │ Performance Issue│
└───────────┘ └─────────────────┘
```

**Implementation Details:**

**Node 1: Trigger**
```json
{
  "webhookId": "performance-detector"
}
```

**Node 2: Run Benchmarks**
```python
# Code Node (Python)
import subprocess
import json

# Run Locust load tests
result = subprocess.run(
    [
        'locust',
        '-f', 'tests/performance/locustfile.py',
        '--headless',
        '-u', '100',  # 100 users
        '-r', '10',   # 10 users per second
        '-t', '60s',  # Run for 1 minute
        '--host', 'http://openevolve-api:8000',
        '--html', '/tmp/locust-report.html',
        '--json',
        '--outfile', '/tmp/locust-results.json'
    ],
    capture_output=True,
    text=True
)

# Load results
with open('/tmp/locust-results.json', 'r') as f:
    benchmark_data = json.load(f)

# Extract metrics
metrics = {
    'request_count': benchmark_data.get('stats', {}).get('num_requests'),
    'failure_count': benchmark_data.get('stats', {}).get('num_failures'),
    'median_response_time': benchmark_data.get('stats', {}).get('median_response_time'),
    'average_response_time': benchmark_data.get('stats', {}).get('avg_response_time'),
    'min_response_time': benchmark_data.get('stats', {}).get('min_response_time'),
    'max_response_time': benchmark_data.get('stats', {}).get('max_response_time'),
    'rps': benchmark_data.get('stats', {}).get('total_rps'),
    'failures_per_second': benchmark_data.get('stats', {}).get('failures_per_second')
}

return [{'json': {'metrics': metrics}}]
```

**Node 3: Get Baseline**
```json
{
  "operation": "executeQuery",
  "query": "SELECT * FROM performance_baselines ORDER BY timestamp DESC LIMIT 1",
  "options": {}
}
```

**Node 4: Compare Metrics**
```python
# Code Node (Python)
current_metrics = $input.first().json.metrics
baseline = $input.all()[1].json[0].baseline_metrics

# Calculate percentage change
delta = {}
for key in current_metrics:
    if key in baseline and baseline[key] > 0:
        delta[key] = {
            'current': current_metrics[key],
            'baseline': baseline[key],
            'percent_change': round(((current_metrics[key] - baseline[key]) / baseline[key]) * 100, 2)
        }

# Check for regressions (>10% slower)
regressions = []
for metric, values in delta.items():
    if values['percent_change'] > 10 and 'response_time' in metric:
        regressions.append({
            'metric': metric,
            'baseline': values['baseline'],
            'current': values['current'],
            'percent_change': values['percent_change']
        })

return [{
  'json': {
    'has_regression': len(regressions) > 0,
    'regressions': regressions,
    'delta': delta
  }
}]
```

**Node 5: Conditional Alert**
```json
{
  "conditions": {
    "boolean": [
      {
        "value1": "={{$json.has_regression}}",
        "operation": "true"
      }
    ]
  }
}
```

**Node 6: Alert & Create Issue**
```python
# Code Node (Python)
regressions = $input.first().json.regressions

# Create GitHub issue
import requests

issue_title = "Performance Regression Detected"

issue_body = f"""## Performance Regression Detected

The following metrics show significant degradation:

"""

for regression in regressions:
    issue_body += f"""
### {regression['metric']}

- **Baseline**: {regression['baseline']}
- **Current**: {regression['current']}
- **Change**: +{regression['percent_change']}%

"""

issue_body += """

## Action Required

1. Investigate the code changes that may have caused this regression
2. Profile the application to identify bottlenecks
3. Optimize or revert the changes
4. Re-run benchmarks to verify improvement
"""

response = requests.post(
    'https://api.github.com/repos/your-org/openevolve/issues',
    headers={
        'Authorization': f'token {$credentials.github_token}',
        'Accept': 'application/vnd.github.v3+json'
    },
    json={
        'title': issue_title,
        'body': issue_body,
        'labels': ['performance', 'regression', 'priority: high']
    }
)

issue_url = response.json().get('html_url')

# Send alert to Slack
message = f"""📉 Performance Regression Detected!

{len(regressions)} metric(s) showing degradation:

Issue created: {issue_url}

Please investigate immediately.
"""

print(message)
```

---

## 7. Deployment Automation

### 7.1 Backup Validator

**Purpose:** Automatically verify backup integrity and test restore procedures.

**Trigger:** Post-backup + Daily

**Workflow Steps:**

```
┌─────────────────┐
│ Trigger         │
│ (Post-backup)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ List Recent     │
│ Backups         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Verify Backup   │
│ Integrity       │
│ (Checksums)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Test Restore    │
│ (To Staging DB) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validate Data   │
│ (Row counts,    │
│  Sample queries)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cleanup Test DB │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Report Results  │
└─────────────────┘
```

**Implementation Details:**

**Node 1: Trigger**
```json
{
  "webhookId": "backup-validator"
}
```

**Node 2: List Recent Backups**
```bash
# Code Node (Shell)
ls -lth /backups/ | head -n 10
```

**Node 3: Verify Backup Integrity**
```python
# Code Node (Python)
import hashlib
import subprocess

backup_file = $input.first().json.backup_file

# Calculate checksum
def calculate_checksum(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

checksum = calculate_checksum(backup_file)

# Compare with stored checksum
# (Assuming checksums are stored in a file)
stored_checksum = None
with open(f'{backup_file}.sha256', 'r') as f:
    stored_checksum = f.read().strip()

valid = checksum == stored_checksum

return [{
  'json': {
    'backup_file': backup_file,
    'checksum_valid': valid,
    'checksum': checksum
  }
}]
```

**Node 4: Test Restore**
```bash
# Code Node (Shell)
#!/bin/bash
set -e

BACKUP_FILE="{{$json.backup_file}}"
TEST_DB="openevolve_test_restore"

echo "Restoring backup to test database..."

# Drop test database if exists
psql -U postgres -c "DROP DATABASE IF EXISTS $TEST_DB;"

# Create test database
psql -U postgres -c "CREATE DATABASE $TEST_DB;"

# Restore backup
pg_restore -U postgres -d $TEST_DB "$BACKUP_FILE"

echo "Restore complete"

# Validate row counts
psql -U postgres -d $TEST_DB -c "
SELECT
    schemaname,
    tablename,
    n_live_tup AS row_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
"
```

**Node 5: Validate Data**
```python
# Code Node (Python)
import psycopg2
from psycopg2 import sql

# Connect to test database
conn = psycopg2.connect(
    host='postgres',
    database='openevolve_test_restore',
    user='postgres',
    password='{$credentials.postgres_password}'
)

cur = conn.cursor()

# Get row counts for all tables
cur.execute("""
SELECT
    schemaname,
    tablename,
    n_live_tup AS row_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
""")

row_counts = cur.fetchall()

# Compare with production
# (Assuming we have baseline counts)
baseline = {
    'users': 1000,
    'workflows': 500,
    'executions': 10000
}

validation_results = []
for schema, table, count in row_counts:
    expected = baseline.get(table)
    if expected:
        match = count == expected
        validation_results.append({
            'table': table,
            'expected': expected,
            'actual': count,
            'match': match
        })

all_valid = all(r['match'] for r in validation_results)

cur.close()
conn.close()

return [{
  'json': {
    'all_valid': all_valid,
    'validation_results': validation_results
  }
}]
```

**Node 6: Cleanup**
```bash
# Code Node (Shell)
psql -U postgres -c "DROP DATABASE IF EXISTS openevolve_test_restore;"
echo "Test database cleaned up"
```

**Node 7: Report Results**
```json
{
  "channel": "#openevolve-backups",
  "text": "=✅ Backup Validation Complete\n\nBackup: {{$json.backup_file}}\nChecksum: {{$json.checksum_valid ? 'Valid' : 'Invalid'}}\nData Validation: {{$json.all_valid ? 'Passed' : 'Failed'}}\n\nValidation Results:\n{{JSON.stringify($json.validation_results, null, 2)}}",
  "username": "Backup Validator",
  "iconEmoji": ":database:"
}
```

---

## 8. Advanced Automation

### 8.1 Knowledge Base Sync

**Purpose:** Automatically extract knowledge and update vector embeddings.

**Trigger:** On content change + Hourly

**Workflow Steps:**

```
┌─────────────────┐
│ Trigger         │
│ (Content change)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Detect Changed  │
│ Documents       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract Content │
│ & Metadata      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate        │
│ Embeddings      │
│ (OpenAI API)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Update Qdrant   │
│ (Upsert vectors)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validate Search │
│ Quality         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Report Metrics  │
└─────────────────┘
```

**Implementation Details:**

**Node 1: Detect Changed Documents**
```python
# Code Node (Python)
import subprocess
import os

# Get changed markdown and text files since last sync
result = subprocess.run(
    ['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'],
    capture_output=True,
    text=True
)

changed_files = [
    f for f in result.stdout.strip().split('\n')
    if f.endswith(('.md', '.txt', '.rst')) and os.path.exists(f)
]

return [{
  'json': {
    'changed_files': changed_files,
    'count': len(changed_files)
  }
}]
```

**Node 2: Extract Content**
```python
# Code Node (Python)
from pathlib import Path

changed_files = $input.first().json.changed_files

documents = []
for file_path in changed_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    documents.append({
        'file_path': file_path,
        'content': content,
        'size': len(content)
    })

return [{'json': {'documents': documents}}]
```

**Node 3: Generate Embeddings**
```python
# Code Node (Python)
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

openai.api_key = '{$credentials.openai_api_key}'

documents = $input.first().json.documents

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=60))
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

results = []
for doc in documents:
    try:
        # Split content into chunks (token limit is 8191)
        chunk_size = 4000
        chunks = [doc['content'][i:i+chunk_size] for i in range(0, len(doc['content']), chunk_size)]

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            results.append({
                'file_path': doc['file_path'],
                'chunk_index': i,
                'content': chunk,
                'embedding': embedding
            })
    except Exception as e:
        print(f"Error processing {doc['file_path']}: {e}")

return [{'json': {'embeddings': results}}]
```

**Node 4: Update Qdrant**
```python
# Code Node (Python)
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(url="http://qdrant:6333")
collection_name = "openevolve_docs"

embeddings = $input.first().json.embeddings

points = []
for emb in embeddings:
    points.append(PointStruct(
        id=hash(f"{emb['file_path']}_{emb['chunk_index']}"),
        vector=emb['embedding'],
        payload={
            'file_path': emb['file_path'],
            'chunk_index': emb['chunk_index'],
            'content': emb['content']
        }
    ))

# Upsert points
client.upsert(
    collection_name=collection_name,
    points=points
)

return [{
  'json': {
    'upserted_count': len(points)
  }
}]
```

**Node 5: Validate Search Quality**
```python
# Code Node (Python)
from qdrant_client import QdrantClient

client = QdrantClient(url="http://qdrant:6333")

# Test search
test_queries = [
    "How to use Hephaestus",
    "MAKER integration guide",
    "Testing procedures"
]

results = []
for query in test_queries:
    # Generate embedding for query
    import openai
    query_embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )['data'][0]['embedding']

    # Search
    search_result = client.search(
        collection_name="openevolve_docs",
        query_vector=query_embedding,
        limit=3
    )

    results.append({
        'query': query,
        'results': len(search_result),
        'top_score': search_result[0].score if search_result else 0
    })

# Calculate average relevance score
avg_score = sum(r['top_score'] for r in results) / len(results)

return [{
  'json': {
    'avg_relevance_score': avg_score,
    'test_results': results,
    'quality_threshold_met': avg_score > 0.7
  }
}]
```

**Node 6: Report Metrics**
```json
{
  "channel": "#openevolve-knowledge",
  "text": "=🔄 Knowledge Base Sync Complete\n\nDocuments Processed: {{$json.count}}\nEmbeddings Generated: {{$json.upserted_count}}\nAverage Relevance Score: {{$json.avg_relevance_score}}\nQuality: {{$json.quality_threshold_met ? '✅ Good' : '⚠️ Needs Improvement'}}",
  "username": "Knowledge Sync",
  "iconEmoji": ":books:"
}
```

### 8.2 Security Compliance Monitor

**Purpose:** Continuously monitor security posture and generate compliance reports.

**Trigger:** Weekly + On-demand

**Workflow Steps:**

```
┌─────────────────┐
│ Trigger         │
│ (Weekly)        │
└────────┬────────┘
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────┐
│ Scan Vulnerabilities│      │ Check License   │
│ (Trivy, Bandit) │          │ Compliance      │
└────────┬────────┘          └────────┬────────┘
         │                              │
         ├──────────────────────────────┤
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────┐
│ Audit Access    │          │ Review Secrets  │
│ Logs            │          │ Management      │
└────────┬────────┘          └────────┬────────┘
         │                              │
         └──────────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Aggregate       │
              │ Findings        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Generate Report │
              │ (HTML, PDF)     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Send Report     │
              │ to Team         │
              └─────────────────┘
```

**Implementation Details:**

**Node 1: Trigger**
```json
{
  "cronExpression": "0 9 * * 1"
}
```

**Node 2: Scan Vulnerabilities**
```bash
# Code Node (Shell)
# Run Trivy for container images
trivy image openevolve-api:latest --format json > /tmp/trivy-results.json

# Run Bandit for Python security
bandit -r . -f json -o /tmp/bandit-results.json

# Combine results
echo '{
  "trivy": '$(cat /tmp/trivy-results.json)',
  "bandit": '$(cat /tmp/bandit-results.json)'
}' > /tmp/security-scan-results.json

cat /tmp/security-scan-results.json
```

**Node 3: Check License Compliance**
```python
# Code Node (Python)
import subprocess
import json

# Get all Python dependencies
result = subprocess.run(
    ['pip', 'list', '--format=json'],
    capture_output=True,
    text=True
)

packages = json.loads(result.stdout)

# Check licenses (using a simple allowlist)
allowed_licenses = [
    'MIT',
    'Apache-2.0',
    'BSD-3-Clause',
    'BSD-2-Clause',
    'ISC',
    'Python-2.0'
]

compliance_issues = []

for package in packages:
    name = package['name']
    version = package['version']

    # Get license from PyPI
    try:
        import requests
        response = requests.get(f'https://pypi.org/pypi/{name}/{version}/json')
        data = response.json()
        license_type = data.get('info', {}).get('license', 'Unknown')

        if license_type not in allowed_licenses:
            compliance_issues.append({
                'package': name,
                'version': version,
                'license': license_type,
                'issue': 'License not in allowlist'
            })
    except Exception as e:
        compliance_issues.append({
            'package': name,
            'version': version,
            'issue': f'Failed to check license: {str(e)}'
        })

return [{
  'json': {
    'total_packages': len(packages),
    'compliance_issues': compliance_issues,
    'compliant': len(compliance_issues) == 0
  }
}]
```

**Node 4: Aggregate Findings**
```python
# Code Node (Python)
import json

security_scan = json.loads($input.first().json.stdout)
license_check = $input.all()[1].json

# Extract vulnerabilities from Trivy
trivy_vulns = security_scan.get('trivy', {}).get('Results', [])
critical_vulns = []
high_vulns = []

for result in trivy_vulns:
    for vuln in result.get('Vulnerabilities', []):
        severity = vuln.get('Severity')
        if severity == 'CRITICAL':
            critical_vulns.append(vuln)
        elif severity == 'HIGH':
            high_vulns.append(vuln)

# Extract security issues from Bandit
bandit_results = security_scan.get('bandit', {}).get('results', [])

aggregated = {
    'critical_vulnerabilities': len(critical_vulns),
    'high_vulnerabilities': len(high_vulns),
    'security_issues': len(bandit_results),
    'license_compliance_issues': len(license_check['compliance_issues']),
    'overall_score': 100 - (
        len(critical_vulns) * 25 +
        len(high_vulns) * 10 +
        len(bandit_results) * 5 +
        len(license_check['compliance_issues']) * 2
    )
}

return [{'json': {'findings': aggregated, 'details': security_scan}}]
```

**Node 5: Generate HTML Report**
```python
# Code Node (Python)
findings = $input.first().json.findings

html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Security Compliance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .score {{
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
        }}
        .good {{ background: #4CAF50; color: white; }}
        .warning {{ background: #ff9800; color: white; }}
        .critical {{ background: #f44336; color: white; }}
        .section {{ margin: 20px 0; padding: 20px; background: #f4f4f4; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>OpenEvolve Security Compliance Report</h1>
    <p>Generated: {datetime.date.today()}</p>

    <div class="score {'good' if findings['overall_score'] >= 80 else 'warning' if findings['overall_score'] >= 50 else 'critical'}">
        Security Score: {findings['overall_score']}/100
    </div>

    <div class="section">
        <h2>Vulnerability Summary</h2>
        <div class="metric">
            <h3>Critical</h3>
            <p>{findings['critical_vulnerabilities']}</p>
        </div>
        <div class="metric">
            <h3>High</h3>
            <p>{findings['high_vulnerabilities']}</p>
        </div>
        <div class="metric">
            <h3>Security Issues</h3>
            <p>{findings['security_issues']}</p>
        </div>
        <div class="metric">
            <h3>License Issues</h3>
            <p>{findings['license_compliance_issues']}</p>
        </div>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
"""

if findings['critical_vulnerabilities'] > 0:
    html += "<p>🔴 <strong>Immediate Action Required:</strong> Address critical vulnerabilities immediately.</p>\n"

if findings['high_vulnerabilities'] > 5:
    html += "<p>🟠 <strong>High Priority:</strong> Address high-severity vulnerabilities within the week.</p>\n"

if findings['license_compliance_issues'] > 0:
    html += "<p>⚠️ <strong>License Compliance:</strong> Review and resolve licensing issues.</p>\n"

html += """
    </div>
</body>
</html>
"""

return [{'json': {'html_report': html}}]
```

**Node 6: Send Report**
```json
{
  "to": "security@openevolve.com",
  "subject": "=Security Compliance Report - Score: {{$json.findings.overall_score}}/100",
  "html": "={{$json.html_report}}"
}
```

---

## 9. Best Practices

### 9.1 Workflow Design Principles

**1. Idempotency**
- Every workflow should be safe to run multiple times
- Use "check before create" patterns
- Implement UPSERT logic instead of INSERT

**2. Error Handling**
- Always wrap external API calls in try-catch
- Implement retry logic with exponential backoff
- Use error handling switches to route failures

**3. State Management**
- Store workflow state in PostgreSQL for complex workflows
- Use Redis for caching intermediate results
- Implement timeouts for all operations

**4. Logging**
- Log all workflow executions with timestamps
- Include correlation IDs for tracking
- Store both success and failure logs

### 9.2 Security Best Practices

**1. Credential Management**
- Never hardcode credentials in workflows
- Use n8n's built-in credential vault
- Rotate credentials regularly

**2. Access Control**
- Implement role-based access control (RBAC)
- Restrict webhook URLs
- Use API tokens instead of passwords

**3. Audit Logging**
- Log all workflow executions
- Track who triggered workflows
- Monitor for suspicious activity

### 9.3 Performance Optimization

**1. Parallel Execution**
- Use n8n's split-in-batches node for parallel processing
- Run independent operations simultaneously
- Aggregate results at the end

**2. Caching**
- Cache frequently accessed data
- Use Redis for temporary storage
- Implement cache invalidation strategies

**3. Resource Management**
- Set appropriate timeouts for all operations
- Limit batch sizes to avoid memory issues
- Monitor n8n resource usage

### 9.4 Maintenance

**1. Version Control**
- Export workflows to JSON
- Commit workflow definitions to Git
- Tag workflow versions

**2. Documentation**
- Document each workflow's purpose
- Include trigger conditions
- List expected inputs and outputs

**3. Regular Reviews**
- Review workflows monthly
- Remove unused workflows
- Optimize slow workflows

---

## 10. Troubleshooting

### 10.1 Common Issues

**Issue 1: Webhooks Not Receiving Data**
- Verify webhook URL is publicly accessible
- Check GitHub webhook configuration
- Test webhook with curl: `curl -X POST https://your-n8n.com/webhook/test`

**Issue 2: Timeouts**
- Increase timeout settings in HTTP Request nodes
- Check network connectivity
- Verify external services are responsive

**Issue 3: Credential Errors**
- Verify credentials are correctly configured
- Check credential permissions
- Rotate expired credentials

**Issue 4: Memory Issues**
- Reduce batch sizes
- Implement pagination
- Monitor n8n memory usage

### 10.2 Debugging Techniques

**1. Enable Debug Logging**
```bash
# Set environment variable
export N8N_LOG_LEVEL=debug
docker-compose restart n8n
```

**2. Test Nodes Individually**
- Use "Execute Node" button
- Check output at each step
- Verify data flow

**3. Inspect Workflow State**
- Add "Set" nodes to log intermediate values
- Use "Code" nodes to debug data structures
- Print JSON.stringify($json) for inspection

### 10.3 Getting Help

- **n8n Community**: https://community.n8n.io
- **n8n Documentation**: https://docs.n8n.io
- **OpenEvolve Docs**: ./docs/
- **Slack**: #openevolve-n8n

---

## Conclusion

This guide provides a comprehensive foundation for automating the OpenEvolve development lifecycle with n8n. Start with the foundational workflows (Health Check Monitor, Test Runner, Infrastructure Orchestrator) and gradually implement more complex automation as your team becomes comfortable with the system.

**Key Takeaways:**
1. Start small, scale fast
2. Monitor everything
3. Fail gracefully
4. Document thoroughly
5. Iterate continuously

**Expected ROI:**
- 50-70% reduction in manual tasks
- Faster failure detection
- Improved reliability
- Better visibility
- Happier team

Happy automating! 🚀
