# 🔧 n8n Workflow Scripting & Automation Guide

**Programmatic Workflow Creation for OpenEvolve**

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Method 1: n8n CLI](#2-method-1-n8n-cli)
3. [Method 2: n8n REST API](#3-method-2-n8n-rest-api)
4. [Method 3: Python Automation Script](#4-method-3-python-automation-script)
5. [Method 4: Node.js Automation Script](#5-method-4-nodejs-automation-script)
6. [Workflow Registry System](#6-workflow-registry-system)
7. [CI/CD Integration](#7-cicd-integration)
8. [Complete Workflow Templates](#8-complete-workflow-templates)

---

## 1. Overview

### 1.1 Why Script Workflow Creation?

**Benefits:**
- ✅ **Version Control**: Store workflow definitions in Git
- ✅ **Automated Deployment**: Deploy workflows via CI/CD
- ✅ **Consistency**: Ensure identical configurations across environments
- ✅ **Scalability**: Create hundreds of workflows programmatically
- ✅ **Documentation**: Self-documenting workflow code
- ✅ **Testing**: Integrate workflow creation into tests

### 1.2 Available Methods

| Method | Best For | Complexity | Language |
|--------|----------|------------|----------|
| **n8n CLI** | Quick imports/deployments | Low | Shell/CLI |
| **n8n REST API** | Dynamic workflow creation | Medium | Any HTTP client |
| **Python SDK** | Complex automation | Medium | Python |
| **Node.js SDK** | Native n8n integration | High | JavaScript/TypeScript |

---

## 2. Method 1: n8n CLI

### 2.1 Installation

```bash
# Install n8n globally
npm install n8n -g

# Or use npx (no installation required)
npx n8n
```

### 2.2 Export Existing Workflows

```bash
# Export all workflows
n8n export:workflow --all --output=./workflows/

# Export specific workflow by ID
n8n export:workflow --id=workflow_id --output=./workflows/

# Export to stdout
n8n export:workflow --all
```

### 2.3 Import Workflows

```bash
# Import single workflow
n8n import:workflow --input=./workflows/health-check-monitor.json

# Import all workflows from directory
n8n import:workflow --input=./workflows/

# Import without activating (dry run)
n8n import:workflow --input=./workflows/health-check-monitor.json --activate=false
```

### 2.4 Workflow Deployment Script

Create `scripts/deploy-workflows.sh`:

```bash
#!/bin/bash
set -e

# Configuration
N8N_URL="${N8N_URL:-http://localhost:5678}"
N8N_API_KEY="${N8N_API_KEY}"
WORKFLOWS_DIR="./n8n/workflows"

echo "🚀 Deploying n8n workflows to $N8N_URL..."

# Check if n8n is running
if ! curl -s "$N8N_URL/healthz" > /dev/null; then
    echo "❌ Error: n8n is not running at $N8N_URL"
    exit 1
fi

# Create workflow directory if it doesn't exist
mkdir -p "$WORKFLOWS_DIR"

# Deploy each workflow
for workflow_file in "$WORKFLOWS_DIR"/*.json; do
    if [ -f "$workflow_file" ]; then
        workflow_name=$(basename "$workflow_file" .json)
        echo "📦 Deploying: $workflow_name"

        # Import workflow via API
        curl -X POST "$N8N_URL/rest/workflows/import" \
            -H "Content-Type: application/json" \
            -d @"$workflow_file" \
            -H "Authorization: Bearer $N8N_API_KEY"

        echo "✅ Deployed: $workflow_name"
    fi
done

echo "✨ All workflows deployed successfully!"
```

Make it executable:

```bash
chmod +x scripts/deploy-workflows.sh
```

### 2.5 Usage

```bash
# Set environment variables
export N8N_URL="http://localhost:5678"
export N8N_API_KEY="your-api-key"

# Run deployment
./scripts/deploy-workflows.sh
```

---

## 3. Method 2: n8n REST API

### 3.1 API Authentication

First, create an API key in n8n:
1. Go to **Settings** → **API**
2. Click **Create API Key**
3. Copy the key

### 3.2 API Endpoints

```bash
# Base URL
N8N_API="http://localhost:5678/rest"

# List all workflows
curl "$N8N_API/workflows" \
    -H "Authorization: Bearer YOUR_API_KEY"

# Get specific workflow
curl "$N8N_API/workflows/WORKFLOW_ID" \
    -H "Authorization: Bearer YOUR_API_KEY"

# Create new workflow
curl -X POST "$N8N_API/workflows" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -d @workflow.json

# Update workflow
curl -X PATCH "$N8N_API/workflows/WORKFLOW_ID" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -d @workflow.json

# Delete workflow
curl -X DELETE "$N8N_API/workflows/WORKFLOW_ID" \
    -H "Authorization: Bearer YOUR_API_KEY"

# Activate workflow
curl -X POST "$N8N_API/workflows/WORKFLOW_ID/activate" \
    -H "Authorization: Bearer YOUR_API_KEY"

# Deactivate workflow
curl -X POST "$N8N_API/workflows/WORKFLOW_ID/deactivate" \
    -H "Authorization: Bearer YOUR_API_KEY"
```

### 3.3 Complete API Script

Create `scripts/n8n-api-manager.py`:

```python
#!/usr/bin/env python3
"""
n8n Workflow Manager via REST API
"""

import requests
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class N8nAPIManager:
    """Manage n8n workflows via REST API"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def list_workflows(self) -> List[Dict]:
        """List all workflows"""
        response = requests.get(
            f'{self.base_url}/workflows',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()['data']

    def get_workflow(self, workflow_id: str) -> Dict:
        """Get specific workflow by ID"""
        response = requests.get(
            f'{self.base_url}/workflows/{workflow_id}',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()['data']

    def create_workflow(self, workflow_data: Dict) -> Dict:
        """Create new workflow"""
        response = requests.post(
            f'{self.base_url}/workflows',
            headers=self.headers,
            json=workflow_data
        )
        response.raise_for_status()
        return response.json()['data']

    def update_workflow(self, workflow_id: str, workflow_data: Dict) -> Dict:
        """Update existing workflow"""
        response = requests.patch(
            f'{self.base_url}/workflows/{workflow_id}',
            headers=self.headers,
            json=workflow_data
        )
        response.raise_for_status()
        return response.json()['data']

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow"""
        response = requests.delete(
            f'{self.base_url}/workflows/{workflow_id}',
            headers=self.headers
        )
        response.raise_for_status()
        return True

    def activate_workflow(self, workflow_id: str) -> Dict:
        """Activate workflow"""
        response = requests.post(
            f'{self.base_url}/workflows/{workflow_id}/activate',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()['data']

    def deactivate_workflow(self, workflow_id: str) -> Dict:
        """Deactivate workflow"""
        response = requests.post(
            f'{self.base_url}/workflows/{workflow_id}/deactivate',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()['data']

    def import_workflow(self, workflow_file: str) -> Dict:
        """Import workflow from file"""
        with open(workflow_file, 'r') as f:
            workflow_data = json.load(f)
        return self.create_workflow(workflow_data)

    def deploy_workflow(self, workflow_file: str, activate: bool = True) -> Dict:
        """Deploy workflow from file"""
        with open(workflow_file, 'r') as f:
            workflow_data = json.load(f)

        # Check if workflow already exists
        workflows = self.list_workflows()
        existing = next(
            (w for w in workflows if w['name'] == workflow_data['name']),
            None
        )

        if existing:
            print(f"📝 Updating existing workflow: {workflow_data['name']}")
            workflow_data['id'] = existing['id']
            result = self.update_workflow(existing['id'], workflow_data)
        else:
            print(f"✨ Creating new workflow: {workflow_data['name']}")
            result = self.create_workflow(workflow_data)

        # Activate if requested
        if activate and result.get('id'):
            print(f"▶️  Activating workflow: {workflow_data['name']}")
            self.activate_workflow(result['id'])

        return result

    def deploy_directory(self, directory: str, activate: bool = True) -> List[Dict]:
        """Deploy all workflows from directory"""
        workflow_dir = Path(directory)
        results = []

        for workflow_file in workflow_dir.glob('*.json'):
            print(f"\n📦 Processing: {workflow_file.name}")
            try:
                result = self.deploy_workflow(str(workflow_file), activate)
                results.append({
                    'file': workflow_file.name,
                    'success': True,
                    'workflow': result
                })
            except Exception as e:
                print(f"❌ Error deploying {workflow_file.name}: {e}")
                results.append({
                    'file': workflow_file.name,
                    'success': False,
                    'error': str(e)
                })

        return results


def main():
    """Main deployment script"""
    # Configuration
    N8N_URL = os.getenv('N8N_URL', 'http://localhost:5678')
    N8N_API_KEY = os.getenv('N8N_API_KEY')
    WORKFLOWS_DIR = os.getenv('WORKFLOWS_DIR', './n8n/workflows')

    if not N8N_API_KEY:
        print("❌ Error: N8N_API_KEY environment variable not set")
        exit(1)

    # Initialize manager
    manager = N8nAPIManager(N8N_URL, N8N_API_KEY)

    # Deploy workflows
    print(f"🚀 Deploying workflows from {WORKFLOWS_DIR}...")
    results = manager.deploy_directory(WORKFLOWS_DIR)

    # Print summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\n✨ Deployment complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    if failed > 0:
        exit(1)


if __name__ == '__main__':
    main()
```

**Usage:**

```bash
# Set environment variables
export N8N_URL="http://localhost:5678"
export N8N_API_KEY="your-api-key-here"
export WORKFLOWS_DIR="./n8n/workflows"

# Run deployment
python scripts/n8n-api-manager.py
```

---

## 4. Method 3: Python Automation Script

### 4.1 Advanced Python Manager

Create `scripts/n8n-workflow-manager.py` with advanced features:

```python
#!/usr/bin/env python3
"""
Advanced n8n Workflow Manager
Supports templating, validation, and bulk operations
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from jinja2 import Template
import yaml


class N8nWorkflowManager:
    """Advanced workflow manager with templating support"""

    def __init__(self, config_file: str = 'n8n-config.yaml'):
        self.config = self._load_config(config_file)
        self.api_url = self.config['n8n_url']
        self.api_key = self.config['api_key']
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Fallback to environment variables
            return {
                'n8n_url': os.getenv('N8N_URL', 'http://localhost:5678'),
                'api_key': os.getenv('N8N_API_KEY'),
                'workflows_dir': os.getenv('WORKFLOWS_DIR', './n8n/workflows'),
                'templates_dir': os.getenv('TEMPLATES_DIR', './n8n/templates'),
                'environments': {
                    'development': {
                        'api_url': 'http://localhost:8000',
                        'slack_channel': '#dev-alerts'
                    },
                    'production': {
                        'api_url': 'https://api.openevolve.com',
                        'slack_channel': '#prod-alerts'
                    }
                }
            }

    def render_template(self, template_file: str, environment: str = 'development') -> Dict:
        """Render workflow template with environment variables"""
        template_path = Path(self.config['templates_dir']) / template_file

        with open(template_path, 'r') as f:
            template_content = f.read()

        # Load environment config
        env_config = self.config['environments'][environment]

        # Render template
        template = Template(template_content)
        rendered = template.render(**env_config)

        return json.loads(rendered)

    def validate_workflow(self, workflow_data: Dict) -> tuple[bool, List[str]]:
        """Validate workflow structure"""
        errors = []

        # Required fields
        required_fields = ['name', 'nodes', 'connections']
        for field in required_fields:
            if field not in workflow_data:
                errors.append(f"Missing required field: {field}")

        # Validate nodes
        if 'nodes' in workflow_data:
            for node in workflow_data['nodes']:
                if 'name' not in node:
                    errors.append(f"Node missing 'name' field")
                if 'type' not in node:
                    errors.append(f"Node '{node.get('name', 'Unknown')}' missing 'type' field")

        # Validate connections
        if 'connections' in workflow_data:
            for node_name, connections in workflow_data['connections'].items():
                if not isinstance(connections, dict):
                    errors.append(f"Invalid connections for node: {node_name}")

        return len(errors) == 0, errors

    def create_workflow_from_template(self, template_name: str, environment: str = 'development') -> Dict:
        """Create workflow from template"""
        # Render template
        workflow_data = self.render_template(template_name, environment)

        # Validate
        is_valid, errors = self.validate_workflow(workflow_data)
        if not is_valid:
            raise ValueError(f"Invalid workflow: {errors}")

        # Create via API
        response = requests.post(
            f'{self.api_url}/rest/workflows',
            headers=self.headers,
            json=workflow_data
        )
        response.raise_for_status()
        return response.json()['data']

    def bulk_deploy(self, environment: str = 'development') -> Dict[str, Any]:
        """Deploy all workflows for an environment"""
        workflows_dir = Path(self.config['workflows_dir'])
        results = {
            'successful': [],
            'failed': [],
            'skipped': []
        }

        for workflow_file in workflows_dir.glob('*.json'):
            workflow_name = workflow_file.stem

            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = json.load(f)

                # Validate
                is_valid, errors = self.validate_workflow(workflow_data)
                if not is_valid:
                    results['failed'].append({
                        'workflow': workflow_name,
                        'errors': errors
                    })
                    continue

                # Check if exists
                existing_workflows = self._list_workflows()
                existing = next(
                    (w for w in existing_workflows if w['name'] == workflow_data['name']),
                    None
                )

                if existing:
                    # Update
                    workflow_data['id'] = existing['id']
                    response = requests.patch(
                        f'{self.api_url}/rest/workflows/{existing["id"]}',
                        headers=self.headers,
                        json=workflow_data
                    )
                else:
                    # Create
                    response = requests.post(
                        f'{self.api_url}/rest/workflows',
                        headers=self.headers,
                        json=workflow_data
                    )

                response.raise_for_status()
                results['successful'].append(workflow_name)

                # Activate
                workflow_id = response.json()['data']['id']
                self._activate_workflow(workflow_id)

            except Exception as e:
                results['failed'].append({
                    'workflow': workflow_name,
                    'error': str(e)
                })

        return results

    def _list_workflows(self) -> List[Dict]:
        """List all workflows"""
        response = requests.get(
            f'{self.api_url}/rest/workflows',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()['data']

    def _activate_workflow(self, workflow_id: str) -> bool:
        """Activate workflow"""
        response = requests.post(
            f'{self.api_url}/rest/workflows/{workflow_id}/activate',
            headers=self.headers
        )
        response.raise_for_status()
        return True

    def export_all_workflows(self, output_dir: str) -> List[str]:
        """Export all workflows to directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        workflows = self._list_workflows()
        exported_files = []

        for workflow in workflows:
            workflow_name = workflow['name'].replace(' ', '-').lower()
            output_file = output_path / f'{workflow_name}.json'

            with open(output_file, 'w') as f:
                json.dump(workflow, f, indent=2)

            exported_files.append(str(output_file))

        return exported_files

    def backup_workflows(self, backup_dir: str = './backups') -> str:
        """Create timestamped backup of all workflows"""
        from datetime import datetime

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = Path(backup_dir) / timestamp
        backup_path.mkdir(parents=True, exist_ok=True)

        self.export_all_workflows(str(backup_path))

        # Create metadata file
        metadata = {
            'timestamp': timestamp,
            'workflow_count': len(self._list_workflows()),
            'backup_path': str(backup_path)
        }

        with open(backup_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return str(backup_path)


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='n8n Workflow Manager')
    parser.add_argument('command', choices=['deploy', 'export', 'backup', 'validate'],
                       help='Command to execute')
    parser.add_argument('--environment', '-e', default='development',
                       help='Environment (development/production)')
    parser.add_argument('--config', '-c', default='n8n-config.yaml',
                       help='Configuration file')
    parser.add_argument('--output', '-o', help='Output directory')

    args = parser.parse_args()

    # Initialize manager
    manager = N8nWorkflowManager(args.config)

    # Execute command
    if args.command == 'deploy':
        print(f"🚀 Deploying workflows for {args.environment}...")
        results = manager.bulk_deploy(args.environment)

        print(f"\n✅ Successful: {len(results['successful'])}")
        print(f"❌ Failed: {len(results['failed'])}")

        if results['failed']:
            print("\nFailed workflows:")
            for failure in results['failed']:
                print(f"  - {failure['workflow']}: {failure.get('error', failure.get('errors'))}")

    elif args.command == 'export':
        output_dir = args.output or './exported-workflows'
        print(f"📦 Exporting workflows to {output_dir}...")
        files = manager.export_all_workflows(output_dir)
        print(f"✅ Exported {len(files)} workflows")

    elif args.command == 'backup':
        print("💾 Creating backup...")
        backup_path = manager.backup_workflows()
        print(f"✅ Backup created at {backup_path}")

    elif args.command == 'validate':
        print("🔍 Validating workflows...")
        workflows_dir = Path(manager.config['workflows_dir'])

        all_valid = True
        for workflow_file in workflows_dir.glob('*.json'):
            with open(workflow_file, 'r') as f:
                workflow_data = json.load(f)

            is_valid, errors = manager.validate_workflow(workflow_data)

            if is_valid:
                print(f"✅ {workflow_file.name}")
            else:
                print(f"❌ {workflow_file.name}")
                for error in errors:
                    print(f"   - {error}")
                all_valid = False

        if not all_valid:
            exit(1)


if __name__ == '__main__':
    main()
```

**Usage:**

```bash
# Deploy workflows
python scripts/n8n-workflow-manager.py deploy --environment production

# Export all workflows
python scripts/n8n-workflow-manager.py export --output ./backups/latest

# Create backup
python scripts/n8n-workflow-manager.py backup

# Validate workflows
python scripts/n8n-workflow-manager.py validate
```

---

## 5. Method 4: Node.js Automation Script

### 5.1 Node.js Workflow Manager

Create `scripts/n8n-manager.js`:

```javascript
/**
 * n8n Workflow Manager - Node.js Implementation
 */

const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');

class N8nWorkflowManager {
  constructor(config = {}) {
    this.baseURL = config.baseURL || process.env.N8N_URL || 'http://localhost:5678';
    this.apiKey = config.apiKey || process.env.N8N_API_KEY;
    this.workflowsDir = config.workflowsDir || './n8n/workflows';

    this.client = axios.create({
      baseURL: `${this.baseURL}/rest`,
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      }
    });
  }

  async listWorkflows() {
    const response = await this.client.get('/workflows');
    return response.data.data;
  }

  async getWorkflow(workflowId) {
    const response = await this.client.get(`/workflows/${workflowId}`);
    return response.data.data;
  }

  async createWorkflow(workflowData) {
    const response = await this.client.post('/workflows', workflowData);
    return response.data.data;
  }

  async updateWorkflow(workflowId, workflowData) {
    const response = await this.client.patch(`/workflows/${workflowId}`, workflowData);
    return response.data.data;
  }

  async deleteWorkflow(workflowId) {
    await this.client.delete(`/workflows/${workflowId}`);
    return true;
  }

  async activateWorkflow(workflowId) {
    const response = await this.client.post(`/workflows/${workflowId}/activate`);
    return response.data.data;
  }

  async deactivateWorkflow(workflowId) {
    const response = await this.client.post(`/workflows/${workflowId}/deactivate`);
    return response.data.data;
  }

  async deployWorkflow(filePath, activate = true) {
    const content = await fs.readFile(filePath, 'utf8');
    const workflowData = JSON.parse(content);

    console.log(`📦 Deploying: ${workflowData.name}`);

    // Check if workflow exists
    const workflows = await this.listWorkflows();
    const existing = workflows.find(w => w.name === workflowData.name);

    let result;
    if (existing) {
      console.log(`📝 Updating existing workflow`);
      workflowData.id = existing.id;
      result = await this.updateWorkflow(existing.id, workflowData);
    } else {
      console.log(`✨ Creating new workflow`);
      result = await this.createWorkflow(workflowData);
    }

    if (activate && result.id) {
      console.log(`▶️  Activating workflow`);
      await this.activateWorkflow(result.id);
    }

    return result;
  }

  async deployAllWorkflows() {
    const files = await fs.readdir(this.workflowsDir);
    const workflowFiles = files.filter(f => f.endsWith('.json'));

    console.log(`🚀 Deploying ${workflowFiles.length} workflows...\n`);

    const results = {
      successful: [],
      failed: []
    };

    for (const file of workflowFiles) {
      const filePath = path.join(this.workflowsDir, file);
      try {
        await this.deployWorkflow(filePath);
        results.successful.push(file);
        console.log(`✅ ${file}\n`);
      } catch (error) {
        results.failed.push({ file, error: error.message });
        console.error(`❌ ${file}: ${error.message}\n`);
      }
    }

    return results;
  }

  async exportAllWorkflows(outputDir) {
    await fs.mkdir(outputDir, { recursive: true });

    const workflows = await this.listWorkflows();
    const exported = [];

    for (const workflow of workflows) {
      const fileName = `${workflow.name.replace(/[^a-z0-9]/gi, '-').toLowerCase()}.json`;
      const filePath = path.join(outputDir, fileName);

      await fs.writeFile(filePath, JSON.stringify(workflow, null, 2));
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
}

// CLI Interface
async function main() {
  const command = process.argv[2];
  const manager = new N8nWorkflowManager();

  switch (command) {
    case 'deploy':
      console.log('🚀 Deploying workflows...\n');
      const results = await manager.deployAllWorkflows();
      console.log(`\n✅ Successful: ${results.successful.length}`);
      console.log(`❌ Failed: ${results.failed.length}`);
      break;

    case 'export':
      const outputDir = process.argv[3] || './exported-workflows';
      console.log(`📦 Exporting workflows to ${outputDir}...`);
      const exported = await manager.exportAllWorkflows(outputDir);
      console.log(`✅ Exported ${exported.length} workflows`);
      break;

    case 'backup':
      console.log('💾 Creating backup...');
      const backupPath = await manager.backupWorkflows();
      console.log(`✅ Backup created at ${backupPath}`);
      break;

    default:
      console.log(`
Usage: node n8n-manager.js <command>

Commands:
  deploy     Deploy all workflows
  export     Export all workflows
  backup     Create timestamped backup

Examples:
  node n8n-manager.js deploy
  node n8n-manager.js export ./backups/latest
  node n8n-manager.js backup
      `);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = N8nWorkflowManager;
```

**Usage:**

```bash
# Install dependencies
npm install axios

# Deploy workflows
node scripts/n8n-manager.js deploy

# Export workflows
node scripts/n8n-manager.js export ./backups/latest

# Create backup
node scripts/n8n-manager.js backup
```

---

## 6. Workflow Registry System

### 6.1 Configuration File

Create `n8n-config.yaml`:

```yaml
# n8n Configuration
n8n_url: "http://localhost:5678"
api_key: "${N8N_API_KEY}"  # Use environment variable

# Directory Structure
workflows_dir: "./n8n/workflows"
templates_dir: "./n8n/templates"
credentials_dir: "./n8n/credentials"

# Environment-specific Configuration
environments:
  development:
    api_url: "http://localhost:8000"
    qdrant_url: "http://qdrant:6333"
    postgres_url: "http://postgres:5432"
    redis_url: "http://redis:6379"
    slack_channel: "#openevolve-dev"
    log_level: "debug"

  staging:
    api_url: "https://staging.openevolve.com"
    qdrant_url: "https://staging.qdrant.openevolve.com"
    postgres_url: "postgresql://staging-db:5432"
    redis_url: "redis://staging-redis:6379"
    slack_channel: "#openevolve-staging"
    log_level: "info"

  production:
    api_url: "https://api.openevolve.com"
    qdrant_url: "https://qdrant.openevolve.com"
    postgres_url: "postgresql://prod-db:5432"
    redis_url: "redis://prod-redis:6379"
    slack_channel: "#openevolve-alerts"
    log_level: "warn"

# Workflow Registry
workflows:
  - name: "Health Check Monitor"
    file: "health-check-monitor.json"
    enabled: true
    schedule: "*/5 * * * *"
    priority: "critical"

  - name: "Automated Test Runner"
    file: "automated-test-runner.json"
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    priority: "high"

  - name: "Infrastructure Orchestrator"
    file: "infrastructure-orchestrator.json"
    enabled: true
    trigger: "manual"
    priority: "high"

  - name: "Dependency Update Monitor"
    file: "dependency-update-monitor.json"
    enabled: true
    schedule: "0 9 * * *"  # Daily at 9 AM
    priority: "medium"

  - name: "Log Aggregation & Analysis"
    file: "log-aggregation.json"
    enabled: true
    schedule: "* * * * *"  # Every minute
    priority: "high"

# Deployment Settings
deployment:
  activate_on_deploy: true
  validate_before_deploy: true
  backup_before_update: true
  backup_dir: "./n8n/backups"

# Notifications
notifications:
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
  email:
    enabled: true
    smtp_host: "${SMTP_HOST}"
    smtp_port: 587
    from_address: "n8n@openevolve.com"
```

### 6.2 Workflow Registry Script

Create `scripts/workflow-registry.py`:

```python
#!/usr/bin/env python3
"""
Workflow Registry Manager
Manages workflow deployments based on configuration
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List
from n8n_workflow_manager import N8nWorkflowManager


class WorkflowRegistry:
    """Manage workflow deployments from registry"""

    def __init__(self, config_file: str = 'n8n-config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.manager = N8nWorkflowManager(config_file)

    def list_registered_workflows(self) -> List[Dict]:
        """List all workflows in registry"""
        return self.config.get('workflows', [])

    def get_enabled_workflows(self) -> List[Dict]:
        """Get only enabled workflows"""
        return [w for w in self.list_registered_workflows() if w.get('enabled', True)]

    def deploy_by_priority(self, environment: str = 'development') -> Dict[str, List]:
        """Deploy workflows in priority order"""
        priorities = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}

        workflows = self.get_enabled_workflows()
        workflows.sort(key=lambda w: priorities.get(w.get('priority', 'low'), 3))

        results = {
            'deployed': [],
            'failed': []
        }

        for workflow in workflows:
            print(f"\n📦 Deploying: {workflow['name']} (Priority: {workflow.get('priority', 'low')})")

            try:
                # Backup if enabled
                if self.config['deployment'].get('backup_before_update'):
                    print("💾 Creating backup...")
                    self.manager.backup_workflows()

                # Deploy workflow
                workflow_file = Path(self.config['workflows_dir']) / workflow['file']
                self.manager.deploy_workflow(str(workflow_file), activate=True)

                results['deployed'].append(workflow['name'])
                print(f"✅ Deployed: {workflow['name']}")

            except Exception as e:
                results['failed'].append({
                    'workflow': workflow['name'],
                    'error': str(e)
                })
                print(f"❌ Failed: {workflow['name']} - {e}")

        return results

    def sync_environment(self, environment: str) -> Dict:
        """Sync all workflows for an environment"""
        print(f"🔄 Syncing workflows for {environment} environment...\n")

        # Load environment variables
        env_config = self.config['environments'][environment]

        # Deploy workflows
        results = self.deploy_by_priority(environment)

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

    parser = argparse.ArgumentParser(description='Workflow Registry Manager')
    parser.add_argument('command', choices=['list', 'deploy', 'sync'],
                       help='Command to execute')
    parser.add_argument('--environment', '-e', default='development',
                       help='Target environment')
    parser.add_argument('--config', '-c', default='n8n-config.yaml',
                       help='Configuration file')

    args = parser.parse_args()

    registry = WorkflowRegistry(args.config)

    if args.command == 'list':
        workflows = registry.list_registered_workflows()
        print("Registered Workflows:")
        for workflow in workflows:
            status = "✅" if workflow.get('enabled', True) else "❌"
            print(f"  {status} {workflow['name']} ({workflow.get('priority', 'low')})")

    elif args.command == 'deploy':
        results = registry.deploy_by_priority(args.environment)

    elif args.command == 'sync':
        results = registry.sync_environment(args.environment)


if __name__ == '__main__':
    main()
```

**Usage:**

```bash
# List registered workflows
python scripts/workflow-registry.py list

# Deploy to development
python scripts/workflow-registry.py deploy --environment development

# Sync production environment
python scripts/workflow-registry.py sync --environment production
```

---

## 7. CI/CD Integration

### 7.1 GitHub Actions Workflow

Create `.github/workflows/deploy-n8n-workflows.yml`:

```yaml
name: Deploy n8n Workflows

on:
  push:
    branches:
      - main
    paths:
      - 'n8n/workflows/**'
      - 'n8n/templates/**'
      - 'n8n-config.yaml'
  pull_request:
    branches:
      - main
    paths:
      - 'n8n/workflows/**'
      - 'n8n/templates/**'

env:
  N8N_URL: ${{ secrets.N8N_URL }}
  N8N_API_KEY: ${{ secrets.N8N_API_KEY }}
  WORKFLOWS_DIR: './n8n/workflows'

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
          pip install pyyaml requests jinja2

      - name: Validate workflows
        run: |
          python scripts/n8n-workflow-manager.py validate

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
          pip install pyyaml requests jinja2

      - name: Deploy workflows
        env:
          N8N_API_KEY: ${{ secrets.N8N_DEV_API_KEY }}
        run: |
          python scripts/workflow-registry.py sync --environment development

      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Development workflows deployed
            PR: #${{ github.event.pull_request.number }}
            Author: ${{ github.actor }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
        if: always()

  deploy-production:
    name: Deploy to Production
    needs: validate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    environment:
      name: production
      url: ${{ secrets.N8N_URL }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pyyaml requests jinja2

      - name: Create backup
        env:
          N8N_API_KEY: ${{ secrets.N8N_API_KEY }}
        run: |
          python scripts/n8n-workflow-manager.py backup --output ./backups/pre-deploy

      - name: Deploy workflows
        env:
          N8N_API_KEY: ${{ secrets.N8N_API_KEY }}
        run: |
          python scripts/workflow-registry.py sync --environment production

      - name: Upload backup artifacts
        uses: actions/upload-artifact@v3
        with:
          name: n8n-backup-${{ github.sha }}
          path: ./backups/pre-deploy
          retention-days: 30

      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Production workflows deployed
            Commit: ${{ github.sha }}
            Author: ${{ github.actor }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
        if: always()
```

### 7.2 Makefile Integration

Add to `Makefile`:

```makefile
# n8n Workflow Management
N8N_URL ?= http://localhost:5678
WORKFLOWS_DIR := ./n8n/workflows
TEMPLATES_DIR := ./n8n/templates

.PHONY: n8n-validate n8n-deploy n8n-export n8n-backup n8n-status

n8n-validate:
	@echo "🔍 Validating workflows..."
	python scripts/n8n-workflow-manager.py validate

n8n-deploy: n8n-validate
	@echo "🚀 Deploying workflows..."
	N8N_URL=$(N8N_URL) python scripts/workflow-registry.py sync

n8n-export:
	@echo "📦 Exporting workflows..."
	N8N_URL=$(N8N_URL) python scripts/n8n-workflow-manager.py export --output ./backups/latest

n8n-backup:
	@echo "💾 Creating backup..."
	N8N_URL=$(N8N_URL) python scripts/n8n-workflow-manager.py backup

n8n-status:
	@echo "📊 Workflow Status..."
	@curl -s "$(N8N_URL)/rest/workflows" \
		-H "Authorization: Bearer $(N8N_API_KEY)" \
		| jq '.data[] | {name: .name, active: .active, id: .id}'
```

**Usage:**

```bash
# Validate workflows
make n8n-validate

# Deploy workflows
make n8n-deploy

# Export workflows
make n8n-export

# Create backup
make n8n-backup

# Check status
make n8n-status
```

---

## 8. Complete Workflow Templates

### 8.1 Template Structure

Create directory structure:

```
n8n/
├── workflows/           # Production workflows (JSON)
│   ├── health-check-monitor.json
│   ├── automated-test-runner.json
│   └── ...
├── templates/          # Jinja2 templates (JSON.j2)
│   ├── health-check-monitor.json.j2
│   └── ...
├── credentials/        # Credential references (not actual values)
│   └── credentials.json
└── backups/           # Auto-generated backups
```

### 8.2 Template Example

Create `n8n/templates/health-check-monitor.json.j2`:

```json
{
  "name": "Health Check Monitor - {{ environment | title }}",
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
        "url": "{{ qdrant_url }}/health",
        "options": {}
      },
      "name": "Check Qdrant",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [450, 200]
    },
    {
      "parameters": {
        "url": "{{ postgres_url }}/health",
        "options": {}
      },
      "name": "Check PostgreSQL",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [450, 400]
    },
    {
      "parameters": {
        "channel": "{{ slack_channel }}",
        "text": "⚠️ Health Check Failed in {{ environment | title }}",
        "username": "OpenEvolve Monitor",
        "iconEmoji": ":warning:"
      },
      "name": "Send Alert to Slack",
      "type": "n8n-nodes-base.slack",
      "typeVersion": 2,
      "position": [850, 200]
    }
  ],
  "connections": {
    "Every 5 Minutes": {
      "main": [[{"node": "Check Qdrant"}, {"node": "Check PostgreSQL"}]]
    },
    "Check Qdrant": {
      "main": [["Send Alert to Slack"]]
    },
    "Check PostgreSQL": {
      "main": [["Send Alert to Slack"]]
    }
  },
  "settings": {
    "executionOrder": "v1"
  },
  "staticData": null,
  "tags": [
    {
      "createdAt": "2024-01-01T00:00:00.000Z",
      "updatedAt": "2024-01-01T00:00:00.000Z",
      "id": "1",
      "name": "{{ environment | title }}"
    }
  ],
  "pinData": {}
}
```

### 8.3 Batch Workflow Generator

Create `scripts/generate-workflows.py`:

```python
#!/usr/bin/env python3
"""
Generate all workflow files from templates
"""

import os
import json
from pathlib import Path
from jinja2 import Template
import yaml


def generate_workflows(environment: str = 'development'):
    """Generate workflow files from templates"""

    # Load config
    with open('n8n-config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env_config = config['environments'][environment]
    templates_dir = Path(config['templates_dir'])
    output_dir = Path(config['workflows_dir'])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each template
    for template_file in templates_dir.glob('*.json.j2'):
        with open(template_file, 'r') as f:
            template_content = f.read()

        # Render template
        template = Template(template_content)
        rendered = template.render(**env_config)

        # Parse JSON to validate
        workflow_data = json.loads(rendered)

        # Write to output file
        output_file = output_dir / f"{template_file.stem}"
        with open(output_file, 'w') as f:
            json.dump(workflow_data, f, indent=2)

        print(f"✅ Generated: {output_file.name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate workflows from templates')
    parser.add_argument('--environment', '-e', default='development',
                       help='Target environment')

    args = parser.parse_args()

    print(f"🔨 Generating workflows for {args.environment}...")
    generate_workflows(args.environment)
    print("✨ Done!")
```

**Usage:**

```bash
# Generate development workflows
python scripts/generate-workflows.py --environment development

# Generate production workflows
python scripts/generate-workflows.py --environment production
```

---

## Summary

This guide provides **4 complete methods** for scripting n8n workflow creation:

### Quick Reference

| Method | Complexity | Use Case | Command |
|--------|------------|----------|---------|
| **n8n CLI** | ⭐ | Quick imports | `n8n import:workflow --input=file.json` |
| **REST API** | ⭐⭐ | Automated scripts | `curl -X POST /rest/workflows -d @file.json` |
| **Python** | ⭐⭐⭐ | Full automation | `python n8n-workflow-manager.py deploy` |
| **Node.js** | ⭐⭐⭐ | Native integration | `node n8n-manager.js deploy` |

### Recommended Setup for OpenEvolve

```bash
# 1. Set up configuration
cp n8n-config.yaml.example n8n-config.yaml
# Edit n8n-config.yaml with your settings

# 2. Generate workflows from templates
python scripts/generate-workflows.py --environment production

# 3. Validate workflows
python scripts/n8n-workflow-manager.py validate

# 4. Deploy workflows
python scripts/workflow-registry.py sync --environment production

# 5. Set up CI/CD
# Add GitHub Actions workflow
# Push to trigger deployment
```

### Key Benefits

✅ **Version Control** - All workflows in Git
✅ **Environment Parity** - Same workflows across environments
✅ **Automated Deployment** - CI/CD integration
✅ **Template-Based** - DRY principle with Jinja2
✅ **Backup & Restore** - Automated backups
✅ **Validation** - Pre-deployment checks

Happy automating! 🚀
