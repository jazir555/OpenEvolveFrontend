# 🚀 BubbleLab Automation CLI

**Complete End-to-End Automation for BubbleLab Workflows**

---

## ⚡ Quick Start

```bash
# 1. Setup (First time only)
python bubblelab_automation.py setup

# 2. List all workflows
python bubblelab_automation.py list

# 3. Generate workflow with AI
python bubblelab_automation.py generate \
  --prompt "Monitor OpenEvolve services and send Slack alerts on failure" \
  --name "Health Check Monitor"
```

---

## 📋 Installation

### Requirements

- Python 3.10+
- BubbleLab running locally or accessible
- API key from BubbleLab

### Install Dependencies

```bash
pip install requests pyyaml
```

---

## 🎯 Commands

### 1. Setup

Interactive setup wizard to configure the CLI.

```bash
python bubblelab_automation.py setup
```

**Prompts you for:**
- BubbleLab URL (default: `http://localhost:3001`)
- API key (from BubbleLab Settings → API)

**Creates:** `bubblelab-config.yaml`

---

### 2. List Workflows

Display all BubbleLab workflows with details.

```bash
python bubblelab_automation.py list
```

**Output:**
```
📊 BubbleLab Workflows (5 total):

1. ▶️ Health Check Monitor
   Type: webhook/http
   Executions: 1523
   Last Run: 2025-01-17T10:30:00Z
   ID: 123

2. ⏸️ Daily Test Runner
   Type: schedule/cron
   Executions: 45
   Last Run: 2025-01-17T02:00:00Z
   ID: 124
...
```

---

### 3. Create Workflow

Create a new workflow from a TypeScript file.

```bash
python bubblelab_automation.py create \
  --file my-workflow.ts \
  --name "My Workflow" \
  --description "Automated workflow"
```

**Options:**
- `--file`: Workflow file (.ts) - **Required**
- `--name`: Workflow name
- `--description`: Workflow description
- `--activate`: Activate after creation (default: true)
- `--no-activate`: Do not activate

---

### 4. Deploy Workflows

Deploy all workflows from a directory.

```bash
python bubblelab_automation.py deploy \
  --directory ./bubblelab-workflows
```

**Options:**
- `--directory`, `-d`: Workflows directory (default: `./bubblelab-workflows`)
- `--no-activate`: Do not activate workflows after deployment

**Process:**
1. Reads all `.ts` files from directory
2. Checks if workflow already exists
3. Creates new or updates existing workflows
4. Optionally activates workflows

---

### 5. Generate Workflow (AI)

Generate a workflow using AI from natural language.

```bash
python bubblelab_automation.py generate \
  --prompt "Monitor Qdrant, PostgreSQL, Redis health and send Slack alerts" \
  --name "Health Check Monitor"
```

**Options:**
- `--prompt`, `-p`: Workflow description - **Required**
- `--name`, `-n`: Workflow name - **Required**
- `--no-activate`: Do not activate after generation

**Process:**
1. Sends prompt to BubbleLab AI (Boba)
2. Generates TypeScript code
3. Validates code
4. Creates workflow
5. Optionally activates

**Output:**
```
🤖 Generating workflow: Health Check Monitor
   Prompt: Monitor Qdrant, PostgreSQL, Redis health...
✅ Code generated successfully
✅ Workflow created: ID 125
▶️ Workflow activated

============================================================
Workflow Generated Successfully!
============================================================
ID: 125
Name: Health Check Monitor

🔗 Webhook URL:
   http://localhost:3001/webhook/USER_ID/abc123xyz

📝 Explanation:
   This workflow checks Qdrant and PostgreSQL health...
```

---

### 6. Export Workflows

Export all workflows to TypeScript files.

```bash
python bubblelab_automation.py export \
  --output ./bubblelab-exports
```

**Options:**
- `--output`, `-o`: Output directory (default: `./bubblelab-exports`)

**Process:**
1. Fetches all workflows
2. Exports code to `.ts` files
3. Saves to output directory

---

### 7. Backup Workflows

Create a timestamped backup of all workflows.

```bash
python bubblelab_automation.py backup \
  --directory ./bubblelab-backups
```

**Options:**
- `--directory`, `-d`: Backup directory (default: `./bubblelab-backups`)

**Creates:**
```
bubblelab-backups/
  20250117_143000/
    health-check-monitor.ts
    automated-test-runner.ts
    infrastructure-orchestrator.ts
    ...
    metadata.json
```

**Metadata includes:**
- Timestamp
- Workflow count
- Backup path
- Creation time

---

### 8. Monitor Workflow

Monitor workflow executions in real-time.

```bash
python bubblelab_automation.py monitor \
  --flow-name "Health Check Monitor" \
  --duration 60
```

**Options:**
- `--flow-id`: Flow ID (integer)
- `--flow-name`: Flow name (string)
- `--duration`: Duration in minutes (default: 60)

**Output:**
```
🔍 Monitoring: Health Check Monitor
   ID: 123
   Duration: 60 minutes
   Press Ctrl+C to stop

[10:30:15] ✨ New executions: +1
[10:35:22] ✨ New executions: +1
...

⏹️  Monitoring stopped

📊 Final Stats:
   New Executions: 3
```

---

### 9. Sync Environment

Deploy workflows for a specific environment.

```bash
python bubblelab_automation.py sync \
  --environment production
```

**Options:**
- `--environment`, `-e`: Target environment (default: `development`)

**Uses configuration from `bubblelab-config.yaml`:**
```yaml
environments:
  production:
    workflows_dir: ./bubblelab-workflows/prod
    api_url: https://api.openevolve.com
    qdrant_url: https://qdrant.openevolve.com
    ...
```

---

### 10. Status

Show system status and configuration.

```bash
python bubblelab_automation.py status
```

**Output:**
```
============================================================
BubbleLab System Status
============================================================

🔌 Connection:
   URL: http://localhost:3001
   Status: ✅ Connected
   Workflows: 5
   Active: 3
   Total Executions: 2341

📋 Templates: 18

⚙️  Configuration:
   Workflows Dir: ./bubblelab-workflows
   Exports Dir: ./bubblelab-exports
   Backups Dir: ./bubblelab-backups

🌍 Environments: ['development', 'production']

============================================================
```

---

## 📁 Configuration File

The CLI uses `bubblelab-config.yaml` for configuration:

```yaml
# BubbleLab Configuration
base_url: "http://localhost:3001"
api_key: "your-api-key-here"

# Directory Structure
workflows_dir: "./bubblelab-workflows"
templates_dir: "./bubblelab-templates"
exports_dir: "./bubblelab-exports"
backups_dir: "./bubblelab-backups"

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
```

---

## 🔄 Common Workflows

### Initial Setup

```bash
# 1. Run setup wizard
python bubblelab_automation.py setup

# 2. Check status
python bubblelab_automation.py status

# 3. List existing workflows
python bubblelab_automation.py list
```

### Daily Operations

```bash
# Deploy new workflows
python bubblelab_automation.py deploy

# Generate new workflow with AI
python bubblelab_automation.py generate \
  --prompt "Your prompt here" \
  --name "Workflow Name"

# Monitor specific workflow
python bubblelab_automation.py monitor \
  --flow-name "Workflow Name" \
  --duration 60
```

### Deployment

```bash
# Backup before deployment
python bubblelab_automation.py backup

# Deploy to development
python bubblelab_automation.py sync --environment development

# Deploy to production
python bubblelab_automation.py sync --environment production

# Verify deployment
python bubblelab_automation.py list
```

### Maintenance

```bash
# Create backup
python bubblelab_automation.py backup

# Export all workflows
python bubblelab_automation.py export

# Check system status
python bubblelab_automation.py status
```

---

## 🤖 AI Generation Tips

### Effective Prompts

**Good prompts:**
- "Monitor Qdrant, PostgreSQL, Redis health every 5 minutes and send Slack alerts on failure"
- "Scrape Reddit r/worldnews hourly, summarize with AI, and email top stories"
- "Run tests at 2 AM daily, analyze results, and create GitHub issue if failures detected"

**Be specific about:**
- **Services** to interact with (HTTP, PostgreSQL, Slack, etc.)
- **Triggers** (webhook, schedule, cron)
- **Actions** (monitor, scrape, analyze, notify)
- **Frequency** (every X minutes, hourly, daily)

### Example Prompts

```bash
# Health monitoring
python bubblelab_automation.py generate \
  --prompt "Check OpenEvolve API health every minute, log errors to PostgreSQL, and send Slack alert if 5+ failures in 10 minutes" \
  --name "API Health Monitor"

# Data pipeline
python bubblelab_automation.py generate \
  --prompt "Fetch data from external API every hour, transform with AI, store in PostgreSQL, and notify on failure" \
  --name "Data Pipeline"

# Scheduled reports
python bubblelab_automation.py generate \
  --prompt "Generate daily summary of logs at 9 AM, analyze with AI for anomalies, and email report" \
  --name "Daily Log Report"
```

---

## 🔧 Troubleshooting

### Connection Issues

**Error:** `API Error (401): Unauthorized`

**Solution:**
1. Verify API key in `bubblelab-config.yaml`
2. Get new API key from BubbleLab Settings → API
3. Run `python bubblelab_automation.py setup` again

### Invalid Workflow Code

**Error:** `Generated code is invalid`

**Solution:**
1. Try a more specific prompt
2. Break down complex workflow into smaller steps
3. Use templates as starting point

### File Not Found

**Error:** `Directory not found`

**Solution:**
1. Create workflows directory: `mkdir bubblelab-workflows`
2. Add TypeScript workflow files
3. Run deploy command again

---

## 📚 Related Documentation

- [BubbleLab Automation Guide](./BUBBLELAB_AUTOMATION_GUIDE.md) - Comprehensive workflow guide
- [BubbleLab Scripting Guide](./BUBBLELAB_SCRIPTING_GUIDE.md) - Advanced scripting
- [BubbleLab Docs](../BubbleLab/docs/) - Official BubbleLab documentation

---

## 🚀 Getting Help

**Issues:**
- Check configuration: `python bubblelab_automation.py status`
- Verify BubbleLab is running
- Check API key validity

**Commands:**
```bash
# Show help
python bubblelab_automation.py --help

# Show command help
python bubblelab_automation.py generate --help

# Check status
python bubblelab_automation.py status
```

---

## ✨ Features

- ✅ **Easy Setup** - Interactive wizard for configuration
- ✅ **AI Generation** - Create workflows from natural language
- ✅ **Batch Operations** - Deploy/export/backup multiple workflows
- ✅ **Environment Management** - Separate configs for dev/prod
- ✅ **Monitoring** - Real-time execution monitoring
- ✅ **Safe Operations** - Backup before deployment
- ✅ **Type Safety** - Full TypeScript support
- ✅ **Version Control** - Export workflows for Git

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Setup | `python bubblelab_automation.py setup` |
| List workflows | `python bubblelab_automation.py list` |
| Create workflow | `python bubblelab_automation.py create --file file.ts` |
| Deploy workflows | `python bubblelab_automation.py deploy` |
| Generate with AI | `python bubblelab_automation.py generate --prompt "..." --name "..."` |
| Export workflows | `python bubblelab_automation.py export` |
| Backup workflows | `python bubblelab_automation.py backup` |
| Monitor workflow | `python bubblelab_automation.py monitor --flow-name "..."` |
| Sync environment | `python bubblelab_automation.py sync --environment prod` |
| Check status | `python bubblelab_automation.py status` |

---

Happy automating! 🚀
