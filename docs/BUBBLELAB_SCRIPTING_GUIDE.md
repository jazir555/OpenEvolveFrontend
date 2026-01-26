# 🔧 BubbleLab Scripting & Automation Guide

**Programmatic Workflow Creation and Management for OpenEvolve**

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [BubbleLab REST API](#2-bubblelab-rest-api)
3. [Python Automation SDK](#3-python-automation-sdk)
4. [Node.js Automation SDK](#4-nodejs-automation-sdk)
5. [Workflow Templates System](#5-workflow-templates-system)
6. [CI/CD Integration](#6-cicd-integration)
7. [Advanced Automation Patterns](#7-advanced-automation-patterns)
8. [Complete Automation Examples](#8-complete-automation-examples)

---

## 1. Overview

### 1.1 Why Script BubbleLab Workflows?

**Benefits:**
- ✅ **Version Control**: Store workflow definitions in Git
- ✅ **Automated Deployment**: Deploy workflows via CI/CD
- ✅ **Consistency**: Ensure identical configurations across environments
- ✅ **Scalability**: Create hundreds of workflows programmatically
- ✅ **Testing**: Integrate workflow creation into tests
- ✅ **Code Ownership**: Export clean TypeScript code

### 1.2 Available Methods

| Method | Best For | Complexity | Language |
|--------|----------|------------|----------|
| **REST API** | Dynamic workflow creation | Medium | Any HTTP client |
| **Python SDK** | Complex automation & scripting | Medium | Python |
| **Node.js SDK** | Native BubbleLab integration | High | JavaScript/TypeScript |
| **AI Generation** | Natural language to workflows | Low | N/A |

---

## 2. BubbleLab REST API

### 2.1 API Authentication

BubbleLab uses JWT tokens for authentication. Obtain tokens via Clerk authentication.

**Get Token:**
```bash
# Login and get token
curl -X POST "http://localhost:3001/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@openevolve.com",
    "password": "password"
  }'

# Use token in subsequent requests
export TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### 2.2 Core API Endpoints

#### BubbleFlow Management

**List All Flows:**
```bash
curl -X GET "http://localhost:3001/bubble-flow" \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "bubbleFlows": [
    {
      "id": 123,
      "name": "Health Check Monitor",
      "description": "Monitors OpenEvolve services",
      "eventType": "webhook/http",
      "isActive": true,
      "cronSchedule": null,
      "executionCount": 1523,
      "lastExecutionAt": "2025-01-17T10:30:00Z",
      "bubbles": [
        {
          "bubbleName": "http",
          "className": "HttpBubble",
          "variableId": 1,
          "variableName": "healthCheck"
        },
        {
          "bubbleName": "slack",
          "className": "SlackBubble",
          "variableId": 2,
          "variableName": "notifier"
        }
      ]
    }
  ],
  "userMonthlyUsage": { "count": 4521 }
}
```

**Create New Flow:**
```bash
curl -X POST "http://localhost:3001/bubble-flow" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Workflow",
    "description": "Automated workflow",
    "eventType": "webhook/http",
    "code": "import { BubbleFlow, HttpBubble }..."
  }'
```

**Response:**
```json
{
  "id": 124,
  "name": "My Workflow",
  "eventType": "webhook/http",
  "webhookPath": "abc123xyz",
  "displayedBubbleParameters": {
    "1": {
      "variableId": 1,
      "bubbleName": "http",
      "className": "HttpBubble",
      "parameters": [...]
    }
  },
  "flowDecomposition": {
    "displayedParameters": [...],
    "dependencies": {...},
    "metadata": {...}
  },
  "requiredCredentials": {
    "SLACK_CRED": true,
    "OPENAI_CRED": false
  }
}
```

**Get Flow Details:**
```bash
curl -X GET "http://localhost:3001/bubble-flow/124" \
  -H "Authorization: Bearer $TOKEN"
```

**Execute Flow:**
```bash
# Simple execution
curl -X POST "http://localhost:3001/bubble-flow/124/execute" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello"}}'

# Streaming execution (SSE)
curl -X POST "http://localhost:3001/bubble-flow/124/execute-stream" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello"}}'
```

**Streaming Response Format:**
```
data: {"type":"bubble_start","data":{"bubbleName":"http","variableId":1}}

data: {"type":"bubble_complete","data":{"bubbleName":"http","output":{...}}}

data: {"type":"complete","data":{"output":{...},"executionId":"..."}}
```

**Update Flow:**
```bash
curl -X PUT "http://localhost:3001/bubble-flow/124" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Workflow Name",
    "code": "import { BubbleFlow }..."
  }'
```

**Delete Flow:**
```bash
curl -X DELETE "http://localhost:3001/bubble-flow/124" \
  -H "Authorization: Bearer $TOKEN"
```

**Activate/Deactivate Webhook:**
```bash
# Activate
curl -X POST "http://localhost:3001/bubble-flow/124/activate" \
  -H "Authorization: Bearer $TOKEN"

# Deactivate
curl -X POST "http://localhost:3001/bubble-flow/124/deactivate" \
  -H "Authorization: Bearer $TOKEN"
```

#### Webhook Triggers

**Trigger Flow via Webhook:**
```bash
# Generic webhook
curl -X POST "http://localhost:3001/webhook/USER_ID/WEBHOOK_PATH" \
  -H "Content-Type: application/json" \
  -d '{"data": "value"}'

# With authentication
curl -X POST "http://localhost:3001/webhook/test" \
  -H "Authorization: Bearer WEBHOOK_TOKEN" \
  -d '{"data": "value"}'
```

#### Template Endpoints

**Get Available Templates:**
```bash
curl -X GET "http://localhost:3001/bubble-flow-templates" \
  -H "Authorization: Bearer $TOKEN"
```

**Create Flow from Template:**
```bash
curl -X POST "http://localhost:3001/bubbleflow-template/data-analyst" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Data Analyst",
    "description": "Analyzes database",
    "parameters": {
      "connectionString": "postgresql://...",
      "query": "SELECT * FROM users"
    }
  }'
```

#### AI Generation Endpoints

**Planning Phase (Coffee):**
```bash
curl -X POST "http://localhost:3001/ai/coffee" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a workflow that monitors services and sends Slack alerts",
    "conversationHistory": []
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
  },
  "messages": [...]
}
```

**Implementation Phase (Boba):**
```bash
curl -X POST "http://localhost:3001/ai/boba" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a workflow that monitors services and sends Slack alerts",
    "messages": [],
    "credentials": {
      "SLACK_CRED": "xoxb-...",
      "OPENAI_CRED": "sk-..."
    }
  }'
```

**Response:**
```json
{
  "code": "import { BubbleFlow, HttpBubble, AIAgentBubble, SlackBubble }...",
  "explanation": "This workflow checks Qdrant and PostgreSQL health...",
  "validation": {
    "valid": true,
    "errors": []
  }
}
```

---

## 3. Python Automation SDK

### 3.1 Installation

```bash
pip install requests pyyaml
```

### 3.2 Basic Python Client

Create `bubblelab_client.py`:

```python
#!/usr/bin/env python3
"""
BubbleLab Python Client
Programmatic workflow management
"""

import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    WEBHOOK = 'webhook/http'
    SCHEDULE = 'schedule/cron'
    SLACK_MENTION = 'slack/bot_mentioned'


@dataclass
class BubbleFlow:
    id: int
    name: str
    description: str
    event_type: str
    is_active: bool
    webhook_path: Optional[str]
    cron_schedule: Optional[str]
    execution_count: int


class BubbleLabClient:
    """Python client for BubbleLab API"""

    def __init__(
        self,
        base_url: str = "http://localhost:3001",
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}' if api_key else ''
        })

    def list_flows(self) -> List[BubbleFlow]:
        """List all BubbleFlows"""
        response = self.session.get(f"{self.base_url}/bubble-flow")
        response.raise_for_status()
        data = response.json()

        return [
            BubbleFlow(
                id=f['id'],
                name=f['name'],
                description=f.get('description', ''),
                event_type=f['eventType'],
                is_active=f.get('isActive', False),
                webhook_path=f.get('webhookPath'),
                cron_schedule=f.get('cronSchedule'),
                execution_count=f.get('executionCount', 0)
            )
            for f in data['bubbleFlows']
        ]

    def get_flow(self, flow_id: int) -> Dict:
        """Get flow details"""
        response = self.session.get(f"{self.base_url}/bubble-flow/{flow_id}")
        response.raise_for_status()
        return response.json()

    def create_flow(
        self,
        name: str,
        code: str,
        description: str = "",
        event_type: str = "webhook/http"
    ) -> Dict:
        """Create new BubbleFlow"""
        response = self.session.post(
            f"{self.base_url}/bubble-flow",
            json={
                "name": name,
                "description": description,
                "eventType": event_type,
                "code": code
            }
        )
        response.raise_for_status()
        return response.json()

    def update_flow(
        self,
        flow_id: int,
        name: Optional[str] = None,
        code: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict:
        """Update existing BubbleFlow"""
        data = {}
        if name is not None:
            data['name'] = name
        if code is not None:
            data['code'] = code
        if description is not None:
            data['description'] = description

        response = self.session.put(
            f"{self.base_url}/bubble-flow/{flow_id}",
            json=data
        )
        response.raise_for_status()
        return response.json()

    def delete_flow(self, flow_id: int) -> bool:
        """Delete BubbleFlow"""
        response = self.session.delete(f"{self.base_url}/bubble-flow/{flow_id}")
        response.raise_for_status()
        return True

    def execute_flow(
        self,
        flow_id: int,
        input_data: Dict,
        stream: bool = False
    ) -> Dict:
        """Execute BubbleFlow"""
        endpoint = f"{self.base_url}/bubble-flow/{flow_id}/execute"
        if stream:
            endpoint += "-stream"

        response = self.session.post(endpoint, json={"input": input_data})
        response.raise_for_status()
        return response.json()

    def activate_flow(self, flow_id: int) -> bool:
        """Activate BubbleFlow webhook"""
        response = self.session.post(f"{self.base_url}/bubble-flow/{flow_id}/activate")
        response.raise_for_status()
        return True

    def deactivate_flow(self, flow_id: int) -> bool:
        """Deactivate BubbleFlow webhook"""
        response = self.session.post(f"{self.base_url}/bubble-flow/{flow_id}/deactivate")
        response.raise_for_status()
        return True

    def trigger_webhook(
        self,
        user_id: str,
        webhook_path: str,
        data: Dict
    ) -> Dict:
        """Trigger flow via webhook"""
        response = requests.post(
            f"{self.base_url}/webhook/{user_id}/{webhook_path}",
            json=data
        )
        response.raise_for_status()
        return response.json()

    def get_templates(self) -> List[Dict]:
        """Get available templates"""
        response = self.session.get(f"{self.base_url}/bubble-flow-templates")
        response.raise_for_status()
        return response.json()['templates']

    def create_from_template(
        self,
        template_name: str,
        name: str,
        parameters: Dict
    ) -> Dict:
        """Create flow from template"""
        response = self.session.post(
            f"{self.base_url}/bubbleflow-template/{template_name}",
            json={
                "name": name,
                "parameters": parameters
            }
        )
        response.raise_for_status()
        return response.json()

    def generate_workflow_ai(
        self,
        prompt: str,
        credentials: Optional[Dict] = None
    ) -> Dict:
        """Generate workflow using AI (Boba)"""
        response = self.session.post(
            f"{self.base_url}/ai/boba",
            json={
                "prompt": prompt,
                "credentials": credentials or {}
            }
        )
        response.raise_for_status()
        return response.json()

    def plan_workflow_ai(
        self,
        prompt: str,
        conversation_history: Optional[List] = None
    ) -> Dict:
        """Plan workflow using AI (Coffee)"""
        response = self.session.post(
            f"{self.base_url}/ai/coffee",
            json={
                "prompt": prompt,
                "conversationHistory": conversation_history or []
            }
        )
        response.raise_for_status()
        return response.json()
```

### 3.3 Advanced Python Manager

Create `bubblelab_manager.py`:

```python
#!/usr/bin/env python3
"""
Advanced BubbleLab Workflow Manager
Supports batch operations, validation, and deployment
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from bubblelab_client import BubbleLabClient, EventType


class BubbleLabWorkflowManager:
    """Advanced workflow management"""

    def __init__(self, config_file: str = 'bubblelab-config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.client = BubbleLabClient(
            base_url=self.config['base_url'],
            api_key=self.config.get('api_key')
        )

    def deploy_from_directory(
        self,
        workflows_dir: str,
        activate: bool = True
    ) -> Dict[str, Any]:
        """Deploy all workflows from directory"""
        workflow_dir = Path(workflows_dir)
        results = {
            'deployed': [],
            'failed': [],
            'skipped': []
        }

        for workflow_file in workflow_dir.glob('*.ts'):
            workflow_name = workflow_file.stem

            try:
                with open(workflow_file, 'r') as f:
                    code = f.read()

                # Check if workflow already exists
                existing_flows = self.client.list_flows()
                existing = next(
                    (f for f in existing_flows if f.name == workflow_name),
                    None
                )

                if existing:
                    # Update existing
                    self.client.update_flow(
                        existing.id,
                        code=code
                    )
                    results['deployed'].append(workflow_name)
                else:
                    # Create new
                    flow = self.client.create_flow(
                        name=workflow_name,
                        code=code,
                        description=f"Auto-generated from {workflow_file.name}"
                    )
                    results['deployed'].append(workflow_name)

                    # Activate if requested
                    if activate:
                        self.client.activate_flow(flow['id'])

            except Exception as e:
                results['failed'].append({
                    'workflow': workflow_name,
                    'error': str(e)
                })

        return results

    def export_all_workflows(self, output_dir: str) -> List[str]:
        """Export all workflows to directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        flows = self.client.list_flows()
        exported_files = []

        for flow in flows:
            flow_details = self.client.get_flow(flow.id)
            workflow_name = flow.name.replace(' ', '-').lower()
            output_file = output_path / f'{workflow_name}.ts'

            with open(output_file, 'w') as f:
                f.write(flow_details['code'])

            exported_files.append(str(output_file))

        return exported_files

    def backup_workflows(self, backup_dir: str = './backups') -> str:
        """Create timestamped backup"""
        from datetime import datetime

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = Path(backup_dir) / timestamp
        backup_path.mkdir(parents=True, exist_ok=True)

        files = self.export_all_workflows(str(backup_path))

        # Create metadata
        metadata = {
            'timestamp': timestamp,
            'workflow_count': len(files),
            'backup_path': str(backup_path)
        }

        with open(backup_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return str(backup_path)

    def generate_and_deploy(
        self,
        prompt: str,
        name: str,
        activate: bool = True
    ) -> Dict:
        """Generate workflow with AI and deploy"""
        # Generate using AI
        generated = self.client.generate_workflow_ai(prompt)

        if not generated['validation']['valid']:
            raise ValueError(f"Generated code is invalid: {generated['validation']['errors']}")

        # Deploy
        flow = self.client.create_flow(
            name=name,
            code=generated['code'],
            description=generated['explanation']
        )

        if activate:
            self.client.activate_flow(flow['id'])

        return {
            'flow_id': flow['id'],
            'explanation': generated['explanation'],
            'webhook_path': flow.get('webhookPath')
        }

    def sync_environment(self, environment: str) -> Dict:
        """Sync workflows for specific environment"""
        env_config = self.config['environments'][environment]

        workflows_dir = Path(env_config['workflows_dir'])
        results = self.deploy_from_directory(str(workflows_dir))

        # Print summary
        print(f"\n{'='*60}")
        print(f"Deployment Summary for {environment}")
        print(f"{'='*60}")
        print(f"✅ Deployed: {len(results['deployed'])}")
        print(f"❌ Failed: {len(results['failed'])}")

        if results['deployed']:
            print(f"\nDeployed Workflows:")
            for name in results['deployed']:
                print(f"  ✅ {name}")

        if results['failed']:
            print(f"\nFailed Workflows:")
            for failure in results['failed']:
                print(f"  ❌ {failure['workflow']}: {failure['error']}")

        return results


def main():
    """CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description='BubbleLab Workflow Manager')
    parser.add_argument('command', choices=['list', 'deploy', 'export', 'backup', 'generate', 'sync'],
                       help='Command to execute')
    parser.add_argument('--environment', '-e', default='development',
                       help='Target environment')
    parser.add_argument('--config', '-c', default='bubblelab-config.yaml',
                       help='Configuration file')
    parser.add_argument('--directory', '-d', help='Workflow directory')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--prompt', '-p', help='AI generation prompt')
    parser.add_argument('--name', '-n', help='Workflow name')

    args = parser.parse_args()

    manager = BubbleLabWorkflowManager(args.config)

    if args.command == 'list':
        flows = manager.client.list_flows()
        print("BubbleLab Workflows:")
        for flow in flows:
            status = "▶️" if flow.is_active else "⏸️"
            print(f"  {status} {flow.name} ({flow.event_type})")
            print(f"     Executions: {flow.execution_count}")

    elif args.command == 'deploy':
        results = manager.deploy_from_directory(args.directory or './workflows')
        print(f"\n✅ Deployed: {len(results['deployed'])}")
        print(f"❌ Failed: {len(results['failed'])}")

    elif args.command == 'export':
        output_dir = args.output or './exported-workflows'
        files = manager.export_all_workflows(output_dir)
        print(f"✅ Exported {len(files)} workflows to {output_dir}")

    elif args.command == 'backup':
        backup_path = manager.backup_workflows()
        print(f"✅ Backup created at {backup_path}")

    elif args.command == 'generate':
        if not args.prompt or not args.name:
            print("❌ --prompt and --name required for generate command")
            exit(1)

        result = manager.generate_and_deploy(args.prompt, args.name)
        print(f"✅ Generated and deployed: {result['flow_id']}")
        print(f"📝 {result['explanation']}")
        if result.get('webhook_path'):
            print(f"🔗 Webhook: {result['webhook_path']}")

    elif args.command == 'sync':
        results = manager.sync_environment(args.environment)


if __name__ == '__main__':
    main()
```

---

## 4. Node.js Automation SDK

### 4.1 Installation

```bash
npm install axios
# or
bun install axios
```

### 4.2 Node.js Client

Create `bubblelab-client.js`:

```javascript
/**
 * BubbleLab Node.js Client
 */

const axios = require('axios');

class BubbleLabClient {
  constructor({ baseUrl = 'http://localhost:3001', apiKey = null } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
    this.client = axios.create({
      baseURL: this.baseUrl,
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey && { 'Authorization': `Bearer ${apiKey}` })
      }
    });
  }

  async listFlows() {
    const response = await this.client.get('/bubble-flow');
    return response.data.bubbleFlows;
  }

  async getFlow(flowId) {
    const response = await this.client.get(`/bubble-flow/${flowId}`);
    return response.data;
  }

  async createFlow({ name, code, description = '', eventType = 'webhook/http' }) {
    const response = await this.client.post('/bubble-flow', {
      name,
      description,
      eventType,
      code
    });
    return response.data;
  }

  async updateFlow(flowId, { name, code, description }) {
    const data = {};
    if (name !== undefined) data.name = name;
    if (code !== undefined) data.code = code;
    if (description !== undefined) data.description = description;

    const response = await this.client.put(`/bubble-flow/${flowId}`, data);
    return response.data;
  }

  async deleteFlow(flowId) {
    await this.client.delete(`/bubble-flow/${flowId}`);
    return true;
  }

  async executeFlow(flowId, inputData, stream = false) {
    const endpoint = stream
      ? `/bubble-flow/${flowId}/execute-stream`
      : `/bubble-flow/${flowId}/execute`;

    const response = await this.client.post(endpoint, { input: inputData });
    return response.data;
  }

  async activateFlow(flowId) {
    await this.client.post(`/bubble-flow/${flowId}/activate`);
    return true;
  }

  async deactivateFlow(flowId) {
    await this.client.post(`/bubble-flow/${flowId}/deactivate`);
    return true;
  }

  async triggerWebhook(userId, webhookPath, data) {
    const response = await axios.post(
      `${this.baseUrl}/webhook/${userId}/${webhookPath}`,
      data
    );
    return response.data;
  }

  async getTemplates() {
    const response = await this.client.get('/bubble-flow-templates');
    return response.data.templates;
  }

  async createFromTemplate(templateName, { name, parameters = {} }) {
    const response = await this.client.post(`/bubbleflow-template/${templateName}`, {
      name,
      parameters
    });
    return response.data;
  }

  async generateWorkflowAI(prompt, credentials = {}) {
    const response = await this.client.post('/ai/boba', {
      prompt,
      credentials
    });
    return response.data;
  }

  async planWorkflowAI(prompt, conversationHistory = []) {
    const response = await this.client.post('/ai/coffee', {
      prompt,
      conversationHistory
    });
    return response.data;
  }
}

module.exports = BubbleLabClient;
```

### 4.3 Advanced Node.js Manager

Create `bubblelab-manager.js`:

```javascript
#!/usr/bin/env node
/**
 * BubbleLab Workflow Manager
 * Advanced automation and deployment
 */

const fs = require('fs').promises;
const path = require('path');
const { BubbleLabClient } = require('./bubblelab-client.js');

class BubbleLabWorkflowManager {
  constructor(configFile = 'bubblelab-config.yaml') {
    // Load config (would need a YAML parser)
    this.config = require('./bubblelab.config.js');
    this.client = new BubbleLabClient({
      baseUrl: this.config.base_url,
      apiKey: this.config.api_key
    });
  }

  async deployFromDirectory(workflowsDir, activate = true) {
    const dir = path.resolve(workflowsDir);
    const files = await fs.readdir(dir);
    const workflowFiles = files.filter(f => f.endsWith('.ts'));

    const results = {
      deployed: [],
      failed: []
    };

    for (const file of workflowFiles) {
      const workflowName = path.basename(file, '.ts');
      const filePath = path.join(dir, file);

      try {
        const code = await fs.readFile(filePath, 'utf8');

        // Check if exists
        const flows = await this.client.listFlows();
        const existing = flows.find(f => f.name === workflowName);

        if (existing) {
          await this.client.updateFlow(existing.id, { code });
          results.deployed.push(workflowName);
        } else {
          const flow = await this.client.createFlow({
            name: workflowName,
            code,
            description: `Auto-generated from ${file}`
          });

          if (activate) {
            await this.client.activateFlow(flow.id);
          }

          results.deployed.push(workflowName);
        }

        console.log(`✅ Deployed: ${workflowName}`);
      } catch (error) {
        results.failed.push({ workflow: workflowName, error: error.message });
        console.error(`❌ Failed: ${workflowName} - ${error.message}`);
      }
    }

    return results;
  }

  async exportAllWorkflows(outputDir) {
    await fs.mkdir(outputDir, { recursive: true });

    const flows = await this.client.listFlows();
    const exported = [];

    for (const flow of flows) {
      const fileName = `${flow.name.replace(/[^a-z0-9]/gi, '-').toLowerCase()}.ts`;
      const filePath = path.join(outputDir, fileName);

      const flowDetails = await this.client.getFlow(flow.id);
      await fs.writeFile(filePath, flowDetails.code);

      exported.push(filePath);
    }

    return exported;
  }

  async backupWorkflows(backupDir = './backups') {
    const date = new Date().toISOString().replace(/[:.]/g, '-');
    const backupPath = path.join(backupDir, date);

    const files = await this.exportAllWorkflows(backupPath);

    const metadata = {
      timestamp: new Date().toISOString(),
      workflowCount: files.length,
      backupPath
    };

    await fs.writeFile(
      path.join(backupPath, 'metadata.json'),
      JSON.stringify(metadata, null, 2)
    );

    return backupPath;
  }

  async generateAndDeploy(prompt, name, activate = true) {
    // Generate using AI
    const generated = await this.client.generateWorkflowAI(prompt);

    if (!generated.validation.valid) {
      throw new Error(`Invalid code: ${generated.validation.errors.join(', ')}`);
    }

    // Deploy
    const flow = await this.client.createFlow({
      name,
      code: generated.code,
      description: generated.explanation
    });

    if (activate) {
      await this.client.activateFlow(flow.id);
    }

    return {
      flowId: flow.id,
      explanation: generated.explanation,
      webhookPath: flow.webhookPath
    };
  }
}

// CLI interface
async function main() {
  const command = process.argv[2];
  const manager = new BubbleLabWorkflowManager();

  switch (command) {
    case 'list':
      const flows = await manager.client.listFlows();
      console.log('BubbleLab Workflows:');
      flows.forEach(f => {
        const status = f.isActive ? '▶️' : '⏸️';
        console.log(`  ${status} ${f.name} (${f.eventType})`);
      });
      break;

    case 'deploy':
      const results = await manager.deployFromDirectory(process.argv[3] || './workflows');
      console.log(`\n✅ Deployed: ${results.deployed.length}`);
      console.log(`❌ Failed: ${results.failed.length}`);
      break;

    case 'export':
      const outputDir = process.argv[3] || './exported-workflows';
      const files = await manager.exportAllWorkflows(outputDir);
      console.log(`✅ Exported ${files.length} workflows to ${outputDir}`);
      break;

    case 'backup':
      const backupPath = await manager.backupWorkflows();
      console.log(`✅ Backup created at ${backupPath}`);
      break;

    case 'generate':
      const prompt = process.argv[3];
      const name = process.argv[4];
      if (!prompt || !name) {
        console.error('❌ Prompt and name required');
        process.exit(1);
      }

      const result = await manager.generateAndDeploy(prompt, name);
      console.log(`✅ Generated and deployed: ${result.flowId}`);
      console.log(`📝 ${result.explanation}`);
      if (result.webhookPath) {
        console.log(`🔗 Webhook: ${result.webhookPath}`);
      }
      break;

    default:
      console.log(`
Usage: node bubblelab-manager.js <command>

Commands:
  list                    List all workflows
  deploy [directory]      Deploy workflows from directory
  export [output]         Export all workflows
  backup                  Create timestamped backup
  generate <prompt> <name> Generate workflow with AI

Examples:
  node bubblelab-manager.js list
  node bubblelab-manager.js deploy ./workflows
  node bubblelab-manager.js generate "Monitor services" "Health Check"
      `);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = BubbleLabWorkflowManager;
```

---

## 5. Workflow Templates System

### 5.1 Template Configuration

Create `bubblelab-config.yaml`:

```yaml
# BubbleLab Configuration
base_url: "http://localhost:3001"
api_key: "${BUBBLELAB_API_KEY}"  # Use environment variable

# Directory Structure
workflows_dir: "./bubblelab-workflows"
templates_dir: "./bubblelab-templates"
exports_dir: "./bubblelab-exports"

# Environment-specific Configuration
environments:
  development:
    workflows_dir: "./bubblelab-workflows/dev"
    api_url: "http://localhost:8000"
    qdrant_url: "http://qdrant:6333"
    postgres_url: "http://postgres:5432"
    redis_url: "http://redis:6379"
    slack_channel: "#openevolve-dev"

  production:
    workflows_dir: "./bubblelab-workflows/prod"
    api_url: "https://api.openevolve.com"
    qdrant_url: "https://qdrant.openevolve.com"
    postgres_url: "postgresql://prod-db:5432"
    redis_url: "redis://prod-redis:6379"
    slack_channel: "#openevolve-alerts"

# Workflow Registry
workflows:
  - name: "Health Check Monitor"
    file: "health-check-monitor.ts"
    enabled: true
    priority: "critical"
    event_type: "webhook/http"

  - name: "Automated Test Runner"
    file: "automated-test-runner.ts"
    enabled: true
    priority: "high"
    event_type: "schedule/cron"
    schedule: "0 2 * * *"

  - name: "Infrastructure Orchestrator"
    file: "infrastructure-orchestrator.ts"
    enabled: true
    priority: "high"
    event_type: "webhook/http"

# Deployment Settings
deployment:
  activate_on_deploy: true
  backup_before_update: true
  backup_dir: "./bubblelab-backups"

# Notifications
notifications:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
  email:
    enabled: true
    smtp_host: "${SMTP_HOST}"
    from_address: "bubblelab@openevolve.com"
```

### 5.2 Workflow Templates

Create templates in `bubblelab-templates/`:

**Template: health-check.j2.ts`
```typescript
import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  SlackBubble,
  type WebhookEvent
} from '@bubblelab/bubble-core';

export class {{ workflow_name }} extends BubbleFlow<'webhook/http'> {
  readonly name = '{{ workflow_name }}';
  readonly description = '{{ description }}';

  async handle(payload: WebhookEvent): Promise<{
    healthy: boolean;
    services: Record<string, { status: number; healthy: boolean }>;
  }> {
    const services = {
      qdrant: '{{ qdrant_url }}/health',
      postgres: '{{ postgres_url }}/health',
      redis: '{{ redis_url }}/ping',
      api: '{{ api_url }}/health',
    };

    const results: Record<string, { status: number; healthy: boolean }> = {};

    for (const [name, url] of Object.entries(services)) {
      const http = new HttpBubble({ url, method: 'GET', timeout: 5000 });
      const result = await http.action();
      results[name] = {
        status: result.status,
        healthy: result.status === 200,
      };
    }

    const healthy = Object.values(results).every(r => r.healthy);

    if (!healthy) {
      const agent = new AIAgentBubble({
        model: { model: 'google/gemini-2.5-flash' },
        systemPrompt: 'Analyze health check results',
        message: JSON.stringify(results, null, 2),
      });

      const analysis = await agent.action();

      const slack = new SlackBubble({
        webhookUrl: process.env.SLACK_WEBHOOK_URL!,
        message: `⚠️ Health Check Failed\n\n${analysis.data.response}`,
      });

      await slack.action();
    }

    return { healthy, services };
  }
}
```

**Template Generator Script:**

Create `generate-workflows.py`:

```python
#!/usr/bin/env python3
"""
Generate BubbleLab workflows from templates
"""

import os
import json
from pathlib import Path
from jinja2 import Template
import yaml


def generate_workflow(template_file: str, config: dict, environment: str = 'development'):
    """Generate workflow from template"""
    env_config = config['environments'][environment]

    with open(template_file, 'r') as f:
        template_content = f.read()

    template = Template(template_content)
    rendered = template.render(
        **env_config,
        workflow_name=Path(template_file).stem.replace('-', ' ').title(),
        description="Auto-generated workflow"
    )

    return rendered


def generate_all_workflows(environment: str = 'development'):
    """Generate all workflows from templates"""
    with open('bubblelab-config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    templates_dir = Path(config['templates_dir'])
    output_dir = Path(config['workflows_dir']) / environment
    output_dir.mkdir(parents=True, exist_ok=True)

    for template_file in templates_dir.glob('*.j2.ts'):
        workflow_name = template_file.stem.replace('.j2', '')
        output_file = output_dir / f'{workflow_name}.ts'

        rendered = generate_workflow(str(template_file), config, environment)

        with open(output_file, 'w') as f:
            f.write(rendered)

        print(f"✅ Generated: {output_file.name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate BubbleLab workflows')
    parser.add_argument('--environment', '-e', default='development')
    args = parser.parse_args()

    generate_all_workflows(args.environment)
```

---

## 6. CI/CD Integration

### 6.1 GitHub Actions Workflow

Create `.github/workflows/deploy-bubblelab.yml`:

```yaml
name: Deploy BubbleLab Workflows

on:
  push:
    branches:
      - main
    paths:
      - 'bubblelab-workflows/**'
      - 'bubblelab-templates/**'
  pull_request:
    branches:
      - main
    paths:
      - 'bubblelab-workflows/**'

env:
  BUBBLELAB_URL: ${{ secrets.BUBBLELAB_URL }}
  BUBBLELAB_API_KEY: ${{ secrets.BUBBLELAB_API_KEY }}

jobs:
  validate:
    name: Validate Workflows
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install requests pyyaml jinja2

      - name: Validate workflows
        run: |
          python scripts/bubblelab-manager.py validate

  deploy-development:
    name: Deploy to Development
    needs: validate
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    environment: development
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install requests pyyaml jinja2

      - name: Deploy workflows
        env:
          BUBBLELAB_API_KEY: ${{ secrets.BUBBLELAB_DEV_API_KEY }}
        run: |
          python scripts/bubblelab-manager.py sync --environment development

      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Development BubbleLab workflows deployed
            PR: #${{ github.event.pull_request.number }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
        if: always()

  deploy-production:
    name: Deploy to Production
    needs: validate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    environment:
      name: production
      url: ${{ secrets.BUBBLELAB_URL }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install requests pyyaml jinja2

      - name: Create backup
        env:
          BUBBLELAB_API_KEY: ${{ secrets.BUBBLELAB_API_KEY }}
        run: |
          python scripts/bubblelab-manager.py backup --output ./backups/pre-deploy

      - name: Deploy workflows
        env:
          BUBBLELAB_API_KEY: ${{ secrets.BUBBLELAB_API_KEY }}
        run: |
          python scripts/bubblelab-manager.py sync --environment production

      - name: Upload backup artifacts
        uses: actions/upload-artifact@v3
        with:
          name: bubblelab-backup-${{ github.sha }}
          path: ./backups/pre-deploy
          retention-days: 30

      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Production BubbleLab workflows deployed
            Commit: ${{ github.sha }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
        if: always()
```

### 6.2 Makefile Integration

Add to `Makefile`:

```makefile
# BubbleLab Workflow Management
BUBBLELAB_URL ?= http://localhost:3001
WORKFLOWS_DIR := ./bubblelab-workflows

.PHONY: bubblelab-validate bubblelab-deploy bubblelab-export bubblelab-backup bubblelist

bubblelab-validate:
	@echo "🔍 Validating BubbleLab workflows..."
	python scripts/bubblelab-manager.py validate

bubblelab-deploy: bubblelab-validate
	@echo "🚀 Deploying BubbleLab workflows..."
	BUBBLELAB_URL=$(BUBBLELAB_URL) python scripts/bubblelab-manager.py sync

bubblelab-export:
	@echo "📦 Exporting BubbleLab workflows..."
	BUBBLELAB_URL=$(BUBBLELAB_URL) python scripts/bubblelab-manager.py export --output ./exports

bubblelab-backup:
	@echo "💾 Creating BubbleLab backup..."
	BUBBLELAB_URL=$(BUBBLELAB_URL) python scripts/bubblelab-manager.py backup

bubblelab-generate:
	@echo "🔨 Generating workflows from templates..."
	python scripts/generate-workflows.py --environment production

bubblelab-list:
	@echo "📊 BubbleLab workflows..."
	@curl -s "$(BUBBLELAB_URL)/bubble-flow" \
		-H "Authorization: Bearer $(BUBBLELAB_API_KEY)" \
		| jq '.bubbleFlows[] | {name, eventType, isActive: .isActive, executions: .executionCount}'
```

---

## 7. Advanced Automation Patterns

### 7.1 Batch Workflow Generation

```python
#!/usr/bin/env python3
"""
Generate multiple workflows from specification
"""

import yaml
from bubblelab_manager import BubbleLabWorkflowManager


def generate_workflows_from_spec(spec_file: str):
    """Generate workflows from YAML specification"""
    with open(spec_file, 'r') as f:
        spec = yaml.safe_load(f)

    manager = BubbleLabWorkflowManager()

    for workflow_spec in spec['workflows']:
        prompt = workflow_spec['prompt']
        name = workflow_spec['name']

        print(f"Generating: {name}")

        try:
            result = manager.generate_and_deploy(
                prompt=prompt,
                name=name,
                activate=workflow_spec.get('activate', True)
            )

            print(f"✅ Generated: {name}")
            print(f"   ID: {result['flow_id']}")
            print(f"   Webhook: {result.get('webhook_path', 'N/A')}")

        except Exception as e:
            print(f"❌ Failed: {name} - {e}")


# Specification file: workflows-spec.yaml
"""
workflows:
  - name: "Health Check Monitor"
    prompt: "Create a workflow that checks Qdrant, PostgreSQL, Redis health and sends Slack alerts on failure"
    activate: true

  - name: "Daily Test Runner"
    prompt: "Create a scheduled workflow that runs tests at 2 AM daily and emails the report"
    activate: true

  - name: "Log Analyzer"
    prompt: "Create a workflow that collects logs every minute, analyzes them with AI, and alerts on anomalies"
    activate: true
"""
```

### 7.2 Workflow Testing

```python
#!/usr/bin/env python3
"""
Test BubbleLab workflows
"""

import asyncio
from bubblelab_client import BubbleLabClient


async def test_workflow(flow_id: int, test_cases: list):
    """Test workflow with multiple test cases"""
    client = BubbleLabClient()

    print(f"Testing workflow {flow_id}...\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test case {i}: {test_case['name']}")

        try:
            result = client.execute_flow(flow_id, test_case['input'])

            if result.get('output'):
                print(f"✅ Passed")
                print(f"   Output: {result['output']}")
            else:
                print(f"❌ Failed: No output")

        except Exception as e:
            print(f"❌ Failed: {e}")

        print()


# Test specification
test_cases = [
    {
        'name': 'Valid health check',
        'input': {'services': ['qdrant', 'postgres']}
    },
    {
        'name': 'Invalid service',
        'input': {'services': ['nonexistent']}
    }
]

# Run tests
asyncio.run(test_workflow(123, test_cases))
```

### 7.3 Workflow Monitoring

```python
#!/usr/bin/env python3
"""
Monitor workflow executions
"""

import time
from bubblelab_client import BubbleLabClient
from datetime import datetime, timedelta


def monitor_workflow_executions(flow_id: int, duration_minutes: int = 60):
    """Monitor workflow executions for specified duration"""
    client = BubbleLabClient()

    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    execution_count = 0
    failure_count = 0

    print(f"Monitoring workflow {flow_id} for {duration_minutes} minutes...\n")

    while datetime.now() < end_time:
        # Get execution history
        flow = client.get_flow(flow_id)
        current_executions = flow.get('executionCount', 0)

        if current_executions > execution_count:
            new_executions = current_executions - execution_count
            execution_count = current_executions

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Executions: +{new_executions}")

        time.sleep(60)  # Check every minute

    print(f"\nMonitoring complete.")
    print(f"Total executions: {execution_count}")
    print(f"Failures: {failure_count}")
```

---

## 8. Complete Automation Examples

### 8.1 Complete Setup Script

Create `scripts/setup-bubblelab.py`:

```python
#!/usr/bin/env python3
"""
Complete BubbleLab Setup Script
Interactive configuration and deployment
"""

import os
import sys
from getpass import getpass
from bubblelab_manager import BubbleLabWorkflowManager


def interactive_setup():
    """Interactive setup wizard"""
    print("="*60)
    print("BubbleLab Setup Wizard")
    print("="*60)

    # Get BubbleLab URL
    bubblelab_url = input("BubbleLab URL [http://localhost:3001]: ") or "http://localhost:3001"

    # Get API key
    api_key = getpass("API Key: ")

    # Create config
    config = {
        'base_url': bubblelab_url,
        'api_key': api_key,
        'environments': {
            'development': {
                'workflows_dir': './bubblelab-workflows/dev',
                'api_url': 'http://localhost:8000',
                'qdrant_url': 'http://qdrant:6333',
                'postgres_url': 'http://postgres:5432',
                'redis_url': 'http://redis:6379'
            }
        }
    }

    # Save config
    import yaml
    with open('bubblelab-config.yaml', 'w') as f:
        yaml.dump(config, f)

    print("\n✅ Configuration saved to bubblelab-config.yaml")

    # Initialize manager
    manager = BubbleLabWorkflowManager()

    # Test connection
    print("\n🔍 Testing connection...")
    try:
        flows = manager.client.list_flows()
        print(f"✅ Connected! Found {len(flows)} existing workflows")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    # Ask what to do
    print("\nWhat would you like to do?")
    print("1. Generate workflows from templates")
    print("2. Deploy existing workflows")
    print("3. Generate workflow with AI")
    print("4. List existing workflows")

    choice = input("Choice [1-4]: ")

    if choice == '1':
        print("\n🔨 Generating workflows from templates...")
        os.system('python scripts/generate-workflows.py')

    elif choice == '2':
        workflows_dir = input("Workflows directory [./bubblelab-workflows]: ") or "./bubblelab-workflows"
        print(f"\n🚀 Deploying workflows from {workflows_dir}...")
        results = manager.deploy_from_directory(workflows_dir)
        print(f"\n✅ Deployed: {len(results['deployed'])}")
        print(f"❌ Failed: {len(results['failed'])}")

    elif choice == '3':
        prompt = input("Describe the workflow: ")
        name = input("Workflow name: ")

        print(f"\n🤖 Generating workflow '{name}' with AI...")
        result = manager.generate_and_deploy(prompt, name)

        print(f"\n✅ Generated and deployed!")
        print(f"   ID: {result['flow_id']}")
        print(f"   Webhook: {result.get('webhook_path', 'N/A')}")
        print(f"\n   {result['explanation']}")

    elif choice == '4':
        flows = manager.client.list_flows()
        print(f"\n📊 Existing Workflows ({len(flows)}):")
        for flow in flows:
            status = "▶️" if flow.is_active else "⏸️"
            print(f"  {status} {flow.name} ({flow.event_type})")

    print("\n✨ Setup complete!")


if __name__ == '__main__':
    interactive_setup()
```

### 8.2 Daily Automation Runner

Create `scripts/daily-automation.py`:

```python
#!/usr/bin/env python3
"""
Daily automation tasks
"""

import schedule
import time
from bubblelab_manager import BubbleLabWorkflowManager
from datetime import datetime


def morning_checks():
    """Run morning health checks"""
    print(f"[{datetime.now()}] Running morning checks...")

    manager = BubbleLabWorkflowManager()

    # Trigger health check workflow
    flows = manager.client.list_flows()
    health_check = next((f for f in flows if 'health' in f.name.lower()), None)

    if health_check:
        result = manager.client.execute_flow(health_check.id, {'check': 'all'})
        print(f"Health check result: {result}")


def weekly_report():
    """Generate weekly report"""
    print(f"[{datetime.now()}] Generating weekly report...")

    manager = BubbleLabWorkflowManager()

    # Trigger report generation
    flows = manager.client.list_flows()
    report = next((f for f in flows if 'report' in f.name.lower()), None)

    if report:
        result = manager.client.execute_flow(report.id, {'period': 'week'})
        print(f"Weekly report: {result}")


def backup_all():
    """Backup all workflows"""
    print(f"[{datetime.now()}] Creating backup...")

    manager = BubbleLabWorkflowManager()
    backup_path = manager.backup_workflows()
    print(f"Backup created: {backup_path}")


# Schedule tasks
schedule.every().day.at("09:00").do(morning_checks)
schedule.every().monday.at("09:00").do(weekly_report)
schedule.every().day.at("03:00").do(backup_all)

print("Daily automation runner started...")
print("Press Ctrl+C to stop")

try:
    while True:
        schedule.run_pending()
        time.sleep(60)
except KeyboardInterrupt:
    print("\nStopped.")
```

---

## Summary

This guide provides comprehensive tools for automating BubbleLab workflows:

### Quick Reference

| Task | Command |
|------|---------|
| List workflows | `python bubblelab-manager.py list` |
| Deploy workflows | `python bubblelab-manager.py deploy ./workflows` |
| Export workflows | `python bubblelab-manager.py export` |
| Create backup | `python bubblelab-manager.py backup` |
| Generate with AI | `python bubblelab-manager.py generate "prompt" "name"` |
| Sync environment | `python bubblelab-manager.py sync --environment prod` |

### Key Benefits

✅ **Version Control** - All workflows in Git
✅ **Automated Deployment** - CI/CD integration
✅ **Environment Parity** - Same workflows across environments
✅ **AI-Powered** - Generate workflows from natural language
✅ **Full Observability** - Track executions and performance

Happy automating with BubbleLab! 🚀

---

## Appendix A: Complete API Reference

This appendix provides comprehensive documentation for all BubbleLab REST API endpoints, including previously undocumented credential management, execution history, streaming, validation, and subscription endpoints.

### Base URL

```
Production: https://api.bubblelab.io
Development: http://localhost:3001
```

### Authentication

All API requests require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.bubblelab.io/bubble-flow
```

---

## 1. Workflow Management Endpoints

### 1.1 Create Workflow

```http
POST /bubble-flow
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "name": "My Workflow",
  "description": "Automated health checks",
  "eventType": "webhook/http",
  "code": "export class MyWorkflow extends BubbleFlow<'webhook/http'> { ... }"
}

Response 201:
{
  "id": 123,
  "name": "My Workflow",
  "description": "Automated health checks",
  "eventType": "webhook/http",
  "active": false,
  "webhookUrl": "https://api.bubblelab.io/webhook/{userId}/{webhookId}",
  "createdAt": "2025-01-17T10:00:00Z",
  "updatedAt": "2025-01-17T10:00:00Z"
}
```

### 1.2 List All Workflows

```http
GET /bubble-flow
Authorization: Bearer {api_key}

Query Parameters:
  - limit: number (default: 50)
  - offset: number (default: 0)
  - active: boolean (filter by active status)
  - eventType: string (filter by event type)

Response 200:
{
  "flows": [
    {
      "id": 123,
      "name": "My Workflow",
      "description": "Automated health checks",
      "eventType": "webhook/http",
      "active": true,
      "executionsCount": 1523,
      "lastExecutionAt": "2025-01-17T10:30:00Z",
      "createdAt": "2025-01-17T10:00:00Z"
    }
  ],
  "total": 5,
  "limit": 50,
  "offset": 0
}
```

### 1.3 Get Workflow Details

```http
GET /bubble-flow/{flow_id}
Authorization: Bearer {api_key}

Response 200:
{
  "id": 123,
  "name": "My Workflow",
  "description": "Automated health checks",
  "eventType": "webhook/http",
  "active": true,
  "code": "export class MyWorkflow extends BubbleFlow<'webhook/http'> { ... }",
  "webhookUrl": "https://api.bubblelab.io/webhook/{userId}/{webhookId}",
  "executionsCount": 1523,
  "createdAt": "2025-01-17T10:00:00Z",
  "updatedAt": "2025-01-17T10:30:00Z"
}
```

### 1.4 Update Workflow

```http
PUT /bubble-flow/{flow_id}
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "name": "Updated Workflow Name",
  "description": "Updated description",
  "code": "export class MyWorkflow extends BubbleFlow<'webhook/http'> { ... }"
}

Response 200:
{
  "id": 123,
  "name": "Updated Workflow Name",
  "description": "Updated description",
  "eventType": "webhook/http",
  "active": true,
  "updatedAt": "2025-01-17T11:00:00Z"
}
```

### 1.5 Delete Workflow

```http
DELETE /bubble-flow/{flow_id}
Authorization: Bearer {api_key}

Response 204: No Content
```

### 1.6 Activate/Deactivate Workflow

```http
POST /bubble-flow/{flow_id}/activate
Authorization: Bearer {api_key}

Response 200:
{
  "id": 123,
  "active": true,
  "activatedAt": "2025-01-17T11:00:00Z"
}

POST /bubble-flow/{flow_id}/deactivate
Authorization: Bearer {api_key}

Response 200:
{
  "id": 123,
  "active": false,
  "deactivatedAt": "2025-01-17T11:00:00Z"
}
```

---

## 2. Execution Endpoints

### 2.1 Execute Workflow (Standard)

```http
POST /bubble-flow/{flow_id}/execute
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "input": {
    "message": "Test input"
  }
}

Response 200:
{
  "executionId": "exec_abc123",
  "status": "success",
  "output": {
    "message": "Processed successfully"
  },
  "duration": 1523,
  "startedAt": "2025-01-17T11:00:00Z",
  "completedAt": "2025-01-17T11:00:01.523Z"
}

Response 500: (Execution Failed)
{
  "executionId": "exec_abc123",
  "status": "failed",
  "error": "Error message",
  "duration": 500
}
```

### 2.2 Execute Workflow (Streaming) **NEW**

```http
POST /bubble-flow/{flow_id}/execute-stream
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "input": {
    "message": "Test input"
  }
}

Response: Server-Sent Events (SSE) Stream

data: {"type":"started","executionId":"exec_abc123","timestamp":"2025-01-17T11:00:00Z"}

data: {"type":"progress","step":"ai-agent","message":"Generating response...","timestamp":"2025-01-17T11:00:00.500Z"}

data: {"type":"progress","step":"slack","message":"Sending notification...","timestamp":"2025-01-17T11:00:01.000Z"}

data: {"type":"completed","executionId":"exec_abc123","status":"success","output":{"message":"Processed"},"duration":1523,"timestamp":"2025-01-17T11:00:01.523Z"}
```

**Python Client for Streaming:**

```python
import requests
import json

def execute_workflow_streaming(flow_id: int, input_data: dict):
    response = requests.post(
        f'{BASE_URL}/bubble-flow/{flow_id}/execute-stream',
        headers={'Authorization': f'Bearer {API_KEY}'},
        json={'input': input_data},
        stream=True
    )

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                event = json.loads(line[6:])
                print(f"[{event['type']}]", event)

# Usage
execute_workflow_streaming(123, {'message': 'Test'})
```

### 2.3 Get Execution History

```http
GET /bubble-flow/{flow_id}/executions
Authorization: Bearer {api_key}

Query Parameters:
  - limit: number (default: 50, max: 1000)
  - offset: number (default: 0)
  - status: 'success' | 'failed' | 'running' (filter by status)
  - startDate: ISO8601 timestamp
  - endDate: ISO8601 timestamp

Response 200:
{
  "executions": [
    {
      "executionId": "exec_abc123",
      "status": "success",
      "input": {"message": "Test input"},
      "output": {"result": "Processed"},
      "duration": 1523,
      "startedAt": "2025-01-17T11:00:00Z",
      "completedAt": "2025-01-17T11:00:01.523Z",
      "error": null
    },
    {
      "executionId": "exec_def456",
      "status": "failed",
      "input": {"message": "Bad input"},
      "output": null,
      "duration": 500,
      "startedAt": "2025-01-17T10:55:00Z",
      "completedAt": "2025-01-17T10:55:00.500Z",
      "error": "Validation failed: Invalid input format"
    }
  ],
  "total": 1523,
  "limit": 50,
  "offset": 0
}
```

### 2.4 Get Execution Details

```http
GET /bubble-flow/{flow_id}/executions/{execution_id}
Authorization: Bearer {api_key}

Response 200:
{
  "executionId": "exec_abc123",
  "flowId": 123,
  "flowName": "My Workflow",
  "status": "success",
  "input": {
    "message": "Test input"
  },
  "output": {
    "result": "Processed successfully",
    "metrics": {
      "queries": 5,
      "apisCalled": 2
    }
  },
  "steps": [
    {
      "name": "validate-input",
      "status": "success",
      "duration": 50,
      "startedAt": "2025-01-17T11:00:00.000Z",
      "completedAt": "2025-01-17T11:00:00.050Z"
    },
    {
      "name": "ai-agent",
      "status": "success",
      "duration": 1420,
      "startedAt": "2025-01-17T11:00:00.050Z",
      "completedAt": "2025-01-17T11:00:01.470Z"
    },
    {
      "name": "slack",
      "status": "success",
      "duration": 53,
      "startedAt": "2025-01-17T11:00:01.470Z",
      "completedAt": "2025-01-17T11:00:01.523Z"
    }
  ],
  "duration": 1523,
  "startedAt": "2025-01-17T11:00:00Z",
  "completedAt": "2025-01-17T11:00:01.523Z",
  "error": null
}
```

---

## 3. Validation Endpoints

### 3.1 Validate Workflow Code (Without Creating) **NEW**

```http
POST /bubble-flow/validate
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "code": "export class MyWorkflow extends BubbleFlow<'webhook/http'> { ... }",
  "eventType": "webhook/http"
}

Response 200: (Validation Passed)
{
  "valid": true,
  "errors": [],
  "warnings": [],
  "suggestions": []
}

Response 200: (Validation Failed)
{
  "valid": false,
  "errors": [
    {
      "line": 15,
      "column": 5,
      "message": "Type 'string' is not assignable to type 'number'",
      "severity": "error",
      "code": "TS2322"
    }
  ],
  "warnings": [
    {
      "line": 20,
      "column": 10,
      "message": "Unused variable 'tempData'",
      "severity": "warning",
      "code": "TS6133"
    }
  ],
  "suggestions": [
    {
      "message": "Consider using async/await instead of Promise chains",
      "line": 25
    }
  ]
}
```

**Python Client for Validation:**

```python
def validate_workflow_code(code: str, event_type: str = 'webhook/http'):
    response = requests.post(
        f'{BASE_URL}/bubble-flow/validate',
        headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
        json={'code': code, 'eventType': event_type}
    )
    return response.json()

# Usage
with open('my-workflow.ts', 'r') as f:
    code = f.read()

result = validate_workflow_code(code)
if result['valid']:
    print("✅ Workflow code is valid!")
else:
    print("❌ Validation errors:")
    for error in result['errors']:
        print(f"  Line {error['line']}: {error['message']}")
```

---

## 4. Context Flow Endpoints

### 4.1 Execute Context Flow **NEW**

```http
POST /context-flow/execute
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "code": "const result = await processData(input); return result;",
  "input": {
    "data": "Sample data"
  },
  "context": {
    "userId": "user_123",
    "sessionId": "session_abc"
  }
}

Response 200:
{
  "success": true,
  "output": {
    "result": "Processed data"
  },
  "duration": 123,
  "startedAt": "2025-01-17T11:00:00Z",
  "completedAt": "2025-01-17T11:00:00.123Z"
}
```

---

## 5. Credential Management Endpoints (52 Endpoints) **NEW**

### 5.1 List All Credentials

```http
GET /credentials
Authorization: Bearer {api_key}

Query Parameters:
  - type: string (filter by credential type)
  - search: string (search in name/description)

Response 200:
{
  "credentials": [
    {
      "id": "cred_abc123",
      "name": "Production Slack Bot",
      "type": "slack_bot_token",
      "description": "Main production Slack bot token",
      "createdAt": "2025-01-17T10:00:00Z",
      "lastUsedAt": "2025-01-17T11:00:00Z",
      "isEncrypted": true
    },
    {
      "id": "cred_def456",
      "name": "OpenAI API Key",
      "type": "openai_api_key",
      "description": "OpenAI GPT-4 access",
      "createdAt": "2025-01-16T10:00:00Z",
      "lastUsedAt": "2025-01-17T10:55:00Z",
      "isEncrypted": true
    }
  ],
  "total": 15
}
```

### 5.2 Create Credential

```http
POST /credentials
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "name": "Production Database",
  "type": "postgres_connection_string",
  "value": "postgresql://user:pass@host:5432/db",
  "description": "Production PostgreSQL database"
}

Response 201:
{
  "id": "cred_xyz789",
  "name": "Production Database",
  "type": "postgres_connection_string",
  "description": "Production PostgreSQL database",
  "createdAt": "2025-01-17T12:00:00Z",
  "isEncrypted": true
}
```

### 5.3 Get Credential Details

```http
GET /credentials/{credential_id}
Authorization: Bearer {api_key}

Response 200:
{
  "id": "cred_abc123",
  "name": "Production Slack Bot",
  "type": "slack_bot_token",
  "description": "Main production Slack bot token",
  "value": "xoxb-...",  // Only returned if explicitly requested
  "createdAt": "2025-01-17T10:00:00Z",
  "updatedAt": "2025-01-17T10:00:00Z",
  "lastUsedAt": "2025-01-17T11:00:00Z",
  "isEncrypted": true,
  "usageCount": 1523
}
```

### 5.4 Update Credential

```http
PUT /credentials/{credential_id}
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "name": "Updated Name",
  "description": "Updated description",
  "value": "new-value-if-needed"
}

Response 200:
{
  "id": "cred_abc123",
  "name": "Updated Name",
  "type": "slack_bot_token",
  "description": "Updated description",
  "updatedAt": "2025-01-17T12:00:00Z"
}
```

### 5.5 Delete Credential

```http
DELETE /credentials/{credential_id}
Authorization: Bearer {api_key}

Response 204: No Content
```

### 5.6 Validate Credential

```http
GET /credentials/{credential_id}/validate
Authorization: Bearer {api_key}

Response 200:
{
  "valid": true,
  "message": "Credential is valid",
  "testedAt": "2025-01-17T12:00:00Z"
}

Response 200: (Invalid)
{
  "valid": false,
  "message": "Authentication failed: Invalid token",
  "testedAt": "2025-01-17T12:00:00Z"
}
```

**Alias Endpoint:**

```http
POST /credentials/{credential_id}/test
Authorization: Bearer {api_key}

Response: Same as GET /validate
```

### 5.7 Get Credential Usage

```http
GET /credentials/{credential_id}/usage
Authorization: Bearer {api_key}

Response 200:
{
  "credentialId": "cred_abc123",
  "usage": [
    {
      "flowId": 123,
      "flowName": "Health Check Monitor",
      "usedIn": "slack bubble"
    },
    {
      "flowId": 124,
      "flowName": "Daily Report",
      "usedIn": "slack bubble"
    }
  ],
  "totalUsage": 2
}
```

### 5.8 Get Credential Audit Log **NEW**

```http
GET /credentials/{credential_id}/audit-log
Authorization: Bearer {api_key}

Query Parameters:
  - limit: number (default: 100)
  - offset: number (default: 0)
  - startDate: ISO8601 timestamp
  - endDate: ISO8601 timestamp

Response 200:
{
  "auditLog": [
    {
      "action": "created",
      "userId": "user_123",
      "timestamp": "2025-01-17T10:00:00Z",
      "ipAddress": "192.168.1.100",
      "userAgent": "BubbleLab CLI v1.0.0"
    },
    {
      "action": "accessed",
      "userId": "user_123",
      "timestamp": "2025-01-17T10:05:00Z",
      "ipAddress": "192.168.1.100",
      "userAgent": "BubbleLab Dashboard"
    },
    {
      "action": "updated",
      "userId": "user_456",
      "timestamp": "2025-01-17T11:00:00Z",
      "ipAddress": "192.168.1.101",
      "changes": {
        "description": {"old": "Old desc", "new": "New desc"}
      }
    }
  ],
  "total": 1523
}
```

### 5.9 Rotate Credential **NEW**

```http
POST /credentials/{credential_id}/rotate
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "newValue": "xoxb-new-token-value"
}

Response 200:
{
  "id": "cred_abc123",
  "name": "Production Slack Bot",
  "rotatedAt": "2025-01-17T12:00:00Z",
  "rotatedBy": "user_123",
  "previousValueHash": "a1b2c3d4..."
}
```

### 5.10 Credential Encryption Status **NEW**

```http
GET /credentials/encryption-status
Authorization: Bearer {api_key}

Response 200:
{
  "encryptionEnabled": true,
  "algorithm": "AES-256-GCM",
  "keyRotationEnabled": true,
  "lastKeyRotation": "2025-01-01T00:00:00Z",
  "credentialsCount": {
    "encrypted": 15,
    "unencrypted": 0
  }
}
```

### 5.11 Update Encryption **NEW**

```http
PUT /credentials/{credential_id}/encryption
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "encrypt": true
}

Response 200:
{
  "id": "cred_abc123",
  "isEncrypted": true,
  "encryptedAt": "2025-01-17T12:00:00Z"
}
```

---

## 6. OAuth Management Endpoints (23 Endpoints) **NEW**

### 6.1 List OAuth Providers

```http
GET /credentials/oauth/providers
Authorization: Bearer {api_key}

Response 200:
{
  "providers": [
    {
      "id": "google",
      "name": "Google",
      "scopes": ["openid", "email", "profile"],
      "authUrl": "https://accounts.google.com/o/oauth2/v2/auth",
      "tokenUrl": "https://oauth2.googleapis.com/token"
    },
    {
      "id": "slack",
      "name": "Slack",
      "scopes": ["chat:write", "channels:read"],
      "authUrl": "https://slack.com/oauth/v2/authorize",
      "tokenUrl": "https://slack.com/api/oauth.v2.access"
    }
  ]
}
```

### 6.2 Start OAuth Flow

```http
GET /credentials/{credential_id}/oauth/authorize
Authorization: Bearer {api_key}

Query Parameters:
  - redirect_uri: string (optional, uses default if not provided)
  - state: string (optional, for CSRF protection)

Response 200:
{
  "authorizeUrl": "https://accounts.google.com/o/oauth2/v2/auth?client_id=...&redirect_uri=...&state=...",
  "state": "random_state_string"
}
```

### 6.3 OAuth Callback

```http
GET /credentials/{credential_id}/oauth/callback
Authorization: Bearer {api_key}

Query Parameters:
  - code: string (authorization code from OAuth provider)
  - state: string (must match authorize state)

Response 200:
{
  "success": true,
  "credentialId": "cred_abc123",
  "accessToken": "ya29....",
  "refreshToken": "refresh_token_value",
  "expiresAt": "2025-01-17T13:00:00Z",
  "tokenType": "Bearer"
}
```

### 6.4 Refresh OAuth Token **NEW**

```http
POST /credentials/{credential_id}/oauth/refresh
Authorization: Bearer {api_key}

Response 200:
{
  "success": true,
  "accessToken": "ya29.new_access_token",
  "refreshToken": "new_refresh_token",
  "expiresAt": "2025-01-17T14:00:00Z"
}
```

### 6.5 Revoke OAuth Access **NEW**

```http
POST /credentials/{credential_id}/oauth/revoke
Authorization: Bearer {api_key}

Response 200:
{
  "success": true,
  "revokedAt": "2025-01-17T12:00:00Z"
}
```

### 6.6 Check OAuth Token Status **NEW**

```http
GET /credentials/{credential_id}/oauth/status
Authorization: Bearer {api_key}

Response 200:
{
  "valid": true,
  "expiresAt": "2025-01-17T13:00:00Z",
  "tokenType": "Bearer",
  "scopes": ["openid", "email", "profile"],
  "expiresIn": 3600
}
```

---

## 7. Subscription & Usage Endpoints (8 Endpoints) **NEW**

### 7.1 Get Current Subscription

```http
GET /subscription
Authorization: Bearer {api_key}

Response 200:
{
  "plan": "pro",
  "status": "active",
  "startedAt": "2025-01-01T00:00:00Z",
  "renewsAt": "2025-02-01T00:00:00Z",
  "limits": {
    "workflows": 100,
    "executionsPerMonth": 100000,
    "storageGB": 10
  }
}
```

### 7.2 Get Usage Statistics

```http
GET /subscription/usage
Authorization: Bearer {api_key}

Query Parameters:
  - period: 'day' | 'week' | 'month' | 'year' (default: 'month')

Response 200:
{
  "period": "month",
  "startDate": "2025-01-01T00:00:00Z",
  "endDate": "2025-01-31T23:59:59Z",
  "usage": {
    "executions": 15234,
    "executionTimeSeconds": 4523,
    "storageUsedGB": 2.3,
    "apiCalls": 15234
  },
  "limits": {
    "executions": 100000,
    "executionTimeSeconds": 30000,
    "storageGB": 10
  },
  "remaining": {
    "executions": 84766,
    "executionTimeSeconds": 25477,
    "storageGB": 7.7
  }
}
```

### 7.3 Get Plan Limits

```http
GET /subscription/limits
Authorization: Bearer {api_key}

Response 200:
{
  "currentPlan": "pro",
  "limits": {
    "workflows": 100,
    "executionsPerMonth": 100000,
    "executionTimeSeconds": 30000,
    "storageGB": 10,
    "teamMembers": 10,
    "apiRateLimit": 1000
  },
  "usage": {
    "workflows": 25,
    "executionsPerMonth": 15234,
    "executionTimeSeconds": 4523,
    "storageGB": 2.3,
    "teamMembers": 5
  }
}
```

### 7.4 Upgrade Plan **NEW**

```http
POST /subscription/upgrade
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "plan": "enterprise",
  "billingCycle": "monthly"
}

Response 200:
{
  "success": true,
  "newPlan": "enterprise",
  "effectiveAt": "2025-01-17T12:00:00Z",
  "nextBillingDate": "2025-02-17T00:00:00Z",
  "amount": 99.00
}
```

### 7.5 Get Billing History **NEW**

```http
GET /subscription/invoices
Authorization: Bearer {api_key}

Query Parameters:
  - limit: number (default: 12)
  - offset: number (default: 0)

Response 200:
{
  "invoices": [
    {
      "id": "inv_abc123",
      "date": "2025-01-01T00:00:00Z",
      "amount": 49.00,
      "currency": "USD",
      "status": "paid",
      "downloadUrl": "https://api.bubblelab.io/invoices/inv_abc123.pdf"
    },
    {
      "id": "inv_def456",
      "date": "2024-12-01T00:00:00Z",
      "amount": 49.00,
      "currency": "USD",
      "status": "paid",
      "downloadUrl": "https://api.bubblelab.io/invoices/inv_def456.pdf"
    }
  ]
}
```

---

## 8. Template System Endpoints (6 Endpoints) **NEW**

### 8.1 List Workflow Templates

```http
GET /templates
Authorization: Bearer {api_key}

Query Parameters:
  - category: string (filter by category)
  - eventType: string (filter by event type)

Response 200:
{
  "templates": [
    {
      "id": "tpl_health_check",
      "name": "Health Check Monitor",
      "description": "Monitor service health and send alerts",
      "category": "monitoring",
      "eventType": "schedule/cron",
      "preview": "export class HealthCheckMonitor extends BubbleFlow<'schedule/cron'> { ... }",
      "usageCount": 1523
    },
    {
      "id": "tpl_slack_notifier",
      "name": "Slack Notification",
      "description": "Send formatted Slack notifications",
      "category": "messaging",
      "eventType": "webhook/http",
      "preview": "export class SlackNotifier extends BubbleFlow<'webhook/http'> { ... }",
      "usageCount": 2341
    }
  ]
}
```

### 8.2 Get Template Details

```http
GET /templates/{template_id}
Authorization: Bearer {api_key}

Response 200:
{
  "id": "tpl_health_check",
  "name": "Health Check Monitor",
  "description": "Monitor service health and send alerts",
  "category": "monitoring",
  "eventType": "schedule/cron",
  "code": "export class HealthCheckMonitor extends BubbleFlow<'schedule/cron'> { ... }",
  "configuration": {
    "services": ["qdrant", "postgres", "redis"],
    "alertChannel": "#ops-alerts",
    "cronSchedule": "*/5 * * * *"
  },
  "usageCount": 1523,
  "createdAt": "2025-01-01T00:00:00Z"
}
```

### 8.3 Create Workflow from Template

```http
POST /templates/{template_id}/instantiate
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "name": "Production Health Check",
  "configuration": {
    "services": ["postgres", "redis"],
    "alertChannel": "#prod-alerts",
    "cronSchedule": "*/10 * * * *"
  }
}

Response 201:
{
  "id": 125,
  "name": "Production Health Check",
  "description": "Monitor service health and send alerts",
  "eventType": "schedule/cron",
  "code": "export class HealthCheckMonitor extends BubbleFlow<'schedule/cron'> { ... }",
  "active": false,
  "createdAt": "2025-01-17T12:00:00Z"
}
```

### 8.4 Create Custom Template **NEW**

```http
POST /templates
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "name": "Custom Data Pipeline",
  "description": "ETL workflow for data processing",
  "category": "data",
  "eventType": "schedule/cron",
  "code": "export class DataPipeline extends BubbleFlow<'schedule/cron'> { ... }",
  "configuration": {
    "source": "postgresql",
    "destination": "google-sheets",
    "schedule": "0 2 * * *"
  },
  "isPublic": false
}

Response 201:
{
  "id": "tpl_custom_123",
  "name": "Custom Data Pipeline",
  "description": "ETL workflow for data processing",
  "category": "data",
  "eventType": "schedule/cron",
  "isPublic": false,
  "createdAt": "2025-01-17T12:00:00Z"
}
```

### 8.5 Export Credential Configuration **NEW**

```http
GET /credentials/{credential_id}/export
Authorization: Bearer {api_key}

Response 200:
{
  "credential": {
    "id": "cred_abc123",
    "name": "Production Slack",
    "type": "slack_bot_token",
    "description": "Production Slack bot token"
  },
  "exportedAt": "2025-01-17T12:00:00Z",
  "format": "json"
}
```

### 8.6 Import Credential Configuration **NEW**

```http
POST /credentials/import
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "credential": {
    "name": "Staging Slack",
    "type": "slack_bot_token",
    "description": "Staging Slack bot token",
    "value": "xoxb-staging-token"
  },
  "skipValidation": false
}

Response 201:
{
  "id": "cred_imported_123",
  "name": "Staging Slack",
  "type": "slack_bot_token",
  "description": "Staging Slack bot token",
  "importedAt": "2025-01-17T12:00:00Z",
  "valid": true
}
```

---

## 9. Error Responses

All endpoints may return these standard error responses:

### 401 Unauthorized

```json
{
  "error": "Unauthorized",
  "message": "Invalid or missing API key",
  "code": "ERR_401"
}
```

### 403 Forbidden

```json
{
  "error": "Forbidden",
  "message": "You don't have permission to access this resource",
  "code": "ERR_403"
}
```

### 404 Not Found

```json
{
  "error": "Not Found",
  "message": "Workflow with ID 999 not found",
  "code": "ERR_404"
}
```

### 429 Rate Limit Exceeded

```json
{
  "error": "Rate Limit Exceeded",
  "message": "Too many requests. Limit: 1000 per minute",
  "code": "ERR_429",
  "retryAfter": 30
}
```

### 500 Internal Server Error

```json
{
  "error": "Internal Server Error",
  "message": "An unexpected error occurred",
  "code": "ERR_500",
  "requestId": "req_abc123"
}
```

---

## 10. Rate Limiting

API rate limits are applied per account:

| Plan | Requests/Minute | Requests/Hour |
|------|-----------------|---------------|
| Free | 60 | 1000 |
| Pro | 1000 | 10000 |
| Enterprise | 10000 | 100000 |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 952
X-RateLimit-Reset: 1705489200
```

---

## 11. Webhooks

### 11.1 Webhook Security

Webhooks can be secured with signature verification:

```python
import hmac
import hashlib

def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected_signature = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(
        f'sha256={expected_signature}',
        signature
    )

# Usage
signature = request.headers.get('X-BubbleLab-Signature')
payload = request.body

if verify_webhook_signature(payload, signature, WEBHOOK_SECRET):
    # Process webhook
    pass
```

### 11.2 Retry Policy

Webhooks are retried with exponential backoff:
- Attempt 1: Immediate
- Attempt 2: 1 minute later
- Attempt 3: 5 minutes later
- Attempt 4: 30 minutes later
- Attempt 5: 2 hours later

After 5 failed attempts, the webhook is marked as failed.

---

**End of Appendix A**
