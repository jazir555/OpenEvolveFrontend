# Sovereign-Grade Problem Decomposition System - API Documentation

## Overview

The Sovereign-Grade Problem Decomposition System provides a comprehensive framework for breaking down complex problems into manageable components using AI-powered analysis, multi-team validation, and systematic orchestration.

## Base URL

```
https://api.sovereign-decomposition.com/v1
```

## Authentication

This API uses API keys for authentication. Include your API key in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

## Rate Limiting

All API endpoints are subject to rate limiting. The default limit is 1,000 requests per minute per API key.

## Response Format

All responses are in JSON format with the following structure:

```json
{
  "success": true,
  "data": {},
  "message": "Optional message",
  "error": "Optional error message (only present when success is false)",
  "request_id": "Unique request identifier",
  "timestamp": "ISO 8601 timestamp"
}
```

---

## Endpoints

### Problem Management

#### Create Problem
```
POST /problems
```

Create a new problem for decomposition.

**Request Body:**
```json
{
  "title": "string",
  "description": "string",
  "problem_type": "string",
  "domain_context": {
    "domain": "string",
    "subdomain": "string",
    "related_domains": ["string"],
    "domain_knowledge": {}
  },
  "constraints": [
    {
      "id": "string",
      "description": "string",
      "type": "string",
      "severity": "string"
    }
  ],
  "success_criteria": [
    {
      "id": "string",
      "description": "string",
      "metric": "string",
      "threshold": 0.0,
      "validation_method": "string"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "string",
    "title": "string",
    "created_at": "ISO 8601 timestamp"
  },
  "request_id": "string",
  "timestamp": "ISO 8601 timestamp"
}
```

#### Get Problem
```
GET /problems/{problem_id}
```

Get details of a specific problem.

**Parameters:**
- `problem_id` (required): Unique identifier of the problem

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "string",
    "title": "string",
    "description": "string",
    "problem_type": "string",
    "domain_context": {},
    "complexity_score": {},
    "constraints": [],
    "success_criteria": [],
    "created_at": "ISO 8601 timestamp",
    "updated_at": "ISO 8601 timestamp"
  }
}
```

#### List Problems
```
GET /problems
```

List all problems with optional filtering.

**Query Parameters:**
- `problem_type`: Filter by problem type
- `limit`: Maximum number of results (default: 20, max: 100)
- `offset`: Number of results to skip (default: 0)
- `status`: Filter by status

**Response:**
```json
{
  "success": true,
  "data": {
    "problems": [
      {
        "id": "string",
        "title": "string",
        "problem_type": "string",
        "status": "string",
        "created_at": "ISO 8601 timestamp"
      }
    ],
    "total_count": 0,
    "limit": 0,
    "offset": 0
  }
}
```

### Decomposition Management

#### Create Decomposition Plan
```
POST /decompositions
```

Create a new decomposition plan for a problem.

**Request Body:**
```json
{
  "problem_id": "string",
  "strategy": "string",
  "sub_problems": [
    {
      "id": "string",
      "title": "string",
      "description": "string",
      "type": "string",
      "dependencies": ["string"],
      "success_criteria": []
    }
  ],
  "dependency_graph": {}
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "string",
    "problem_id": "string",
    "strategy": "string",
    "status": "string"
  }
}
```

#### Get Decomposition Plan
```
GET /decompositions/{plan_id}
```

Get details of a specific decomposition plan.

**Parameters:**
- `plan_id` (required): Unique identifier of the decomposition plan

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "string",
    "problem_id": "string",
    "strategy": "string",
    "sub_problems": [],
    "dependency_graph": {},
    "quality_scores": {},
    "status": "string",
    "created_at": "ISO 8601 timestamp"
  }
}
```

### Solution Management

#### Create Solution Attempt
```
POST /solutions
```

Create a solution attempt for a sub-problem.

**Request Body:**
```json
{
  "sub_problem_id": "string",
  "approach": "string",
  "solution_content": "string",
  "team_id": "string",
  "confidence_score": 0.0
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "string",
    "sub_problem_id": "string",
    "status": "string"
  }
}
```

#### Validate Solution
```
POST /solutions/{solution_id}/validate
```

Validate a solution attempt using the validation gauntlet.

**Parameters:**
- `solution_id` (required): Unique identifier of the solution attempt

**Request Body:**
```json
{
  "validation_type": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "solution_id": "string",
    "is_approved": true,
    "score": 0.0,
    "feedback": "string",
    "validation_results": []
  }
}
```

### Team Coordination

#### Assign to Team
```
POST /assignments
```

Assign a task to a specific team for validation or refinement.

**Request Body:**
```json
{
  "task_id": "string",
  "team": "string",
  "priority": 0,
  "due_hours": 0
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "string",
    "task_id": "string",
    "team": "string",
    "status": "string",
    "assigned_at": "ISO 8601 timestamp"
  }
}
```

#### Get Team Assignment
```
GET /assignments/{assignment_id}
```

Get details of a specific team assignment.

**Parameters:**
- `assignment_id` (required): Unique identifier of the assignment

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "string",
    "task_id": "string",
    "team": "string",
    "status": "string",
    "assigned_at": "ISO 8601 timestamp",
    "due_date": "ISO 8601 timestamp"
  }
}
```

### Gauntlet System

#### Run Gauntlet
```
POST /gauntlets/{gauntlet_name}/run
```

Run a specific gauntlet for validation.

**Parameters:**
- `gauntlet_name` (required): Name of the gauntlet to run

**Request Body:**
```json
{
  "content": "string",
  "context": {}
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_approved": true,
    "score": 0.0,
    "feedback": "string",
    "report": {}
  }
}
```

### Analytics and Monitoring

#### Get System Metrics
```
GET /analytics/metrics
```

Get current system metrics and performance indicators.

**Response:**
```json
{
  "success": true,
  "data": {
    "active_workflows": 0,
    "completed_workflows": 0,
    "active_problems": 0,
    "system_health": "string",
    "cpu_usage": 0.0,
    "memory_usage": 0.0,
    "uptime": 0
  }
}
```

#### Get Workflow Status
```
GET /analytics/workflows/{workflow_id}
```

Get detailed status of a specific workflow.

**Parameters:**
- `workflow_id` (required): Unique identifier of the workflow

**Response:**
```json
{
  "success": true,
  "data": {
    "workflow_id": "string",
    "status": "string",
    "progress": 0.0,
    "steps_completed": [],
    "steps_remaining": [],
    "start_time": "ISO 8601 timestamp",
    "estimated_completion": "ISO 8601 timestamp"
  }
}
```

### API Key Management

#### Create API Key
```
POST /auth/api-keys
```

Create a new API key for programmatic access.

**Request Body:**
```json
{
  "name": "string",
  "permissions": ["string"],
  "expires_in_days": 0
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "api_key": "string",
    "key_id": "string",
    "name": "string",
    "created_at": "ISO 8601 timestamp"
  }
}
```

#### Revoke API Key
```
DELETE /auth/api-keys/{key_id}
```

Revoke an existing API key.

**Parameters:**
- `key_id` (required): Unique identifier of the API key

**Response:**
```json
{
  "success": true,
  "message": "API key revoked successfully"
}
```

## Error Codes

The API uses standard HTTP status codes with additional details in the response body:

| Status Code | Error Code | Meaning |
|-------------|------------|---------|
| 400 | `bad_request` | Invalid request format or parameters |
| 401 | `unauthorized` | Missing or invalid API key |
| 403 | `forbidden` | Insufficient permissions |
| 404 | `not_found` | Resource not found |
| 429 | `rate_limit_exceeded` | Rate limit exceeded |
| 500 | `server_error` | Internal server error |
| 503 | `service_unavailable` | Service temporarily unavailable |

## Usage Examples

### Creating a New Problem

```bash
curl -X POST https://api.sovereign-decomposition.com/v1/problems \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Optimize Database Query Performance",
    "description": "Improve the performance of slow database queries in the user management system",
    "problem_type": "OPTIMIZATION",
    "domain_context": {
      "domain": "software_engineering",
      "subdomain": "database_optimization"
    },
    "constraints": [
      {
        "description": "Must not break existing functionality",
        "type": "quality",
        "severity": "hard"
      }
    ],
    "success_criteria": [
      {
        "description": "Reduce query time by 50%",
        "metric": "query_execution_time",
        "threshold": 0.5
      }
    ]
  }'
```

### Creating a Decomposition Plan

```bash
curl -X POST https://api.sovereign-decomposition.com/v1/decompositions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_id": "prob_abc123",
    "strategy": "complexity",
    "sub_problems": [
      {
        "id": "sp_1",
        "title": "Analyze Slow Queries",
        "description": "Identify the slowest database queries",
        "type": "ANALYSIS"
      },
      {
        "id": "sp_2",
        "title": "Design Index Strategy", 
        "description": "Design optimal indexing strategy for identified queries",
        "type": "DESIGN",
        "dependencies": ["sp_1"]
      }
    ]
  }'
```

## SDKs and Libraries

We provide official SDKs for various programming languages:

- [Python SDK](#)
- [JavaScript/Node.js SDK](#)
- [Java SDK](#)
- [Go SDK](#)

## Webhook Notifications

The API can send webhook notifications for important events. To configure webhooks:

1. Set your webhook endpoint in the dashboard
2. Subscribe to specific event types
3. Receive JSON payloads with event details

**Supported Event Types:**
- `problem.created`
- `decomposition.completed`
- `solution.validated`
- `workflow.status_changed`
- `team.assignment.created`

## Support

For API support, contact our developer team at [api-support@sovereigndecomposition.com](mailto:api-support@sovereigndecomposition.com) or use our [developer portal](https://developers.sovereigndecomposition.com).

---

## LeanAide Integration APIs

### Overview

LeanAide provides formal mathematical verification and Lean 4 theorem proving capabilities integrated into the OpenEvolve workflow.

### LeanAide Client API

#### LeanAideClient

The main client for interacting with LeanAide server.

```python
from leanaide_client import LeanAideClient, LeanAideConfig
import asyncio

async def main():
    # Create client
    config = LeanAideConfig(
        host="localhost",
        port=7654,
        timeout=6000.0,
        max_connections=100
    )
    client = LeanAideClient(config)

    # Check health
    is_healthy = await client.health_check()
```

#### Task Methods

**translate_thm**
```
POST /translate_thm
```

Translate a natural-language theorem into Lean and elaborate its type.

**Request:**
```json
{
  "task": "translate_thm",
  "theorem_text": "There are infinitely many prime numbers"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "lean_code": "theorem infinitely_many_primes : Infinite {p : Nat | Prime p}",
    "elaborated_type": "Prop"
  }
}
```

**translate_thm_detailed**
```
POST /translate_thm_detailed
```

Translate a theorem with optional name and produce Lean declaration.

**Request:**
```json
{
  "task": "translate_thm_detailed",
  "theorem_text": "There are infinitely many prime numbers",
  "theorem_name": "infinitely_many_primes"
}
```

**translate_def**
```
POST /translate_def
```

Translate a natural-language definition into Lean code.

**prove_for_formalization**
```
POST /prove_for_formalization
```

Generate a detailed proof or proof sketch for a theorem.

**Request:**
```json
{
  "task": "prove_for_formalization",
  "theorem_text": "There are infinitely many prime numbers",
  "theorem_code": "theorem infinitely_many_primes : Infinite {p : Nat | Prime p}",
  "theorem_statement": "Infinitely many primes exist"
}
```

**elaborate**
```
POST /elaborate
```

Elaborate Lean code and collect results, logs, and unsolved goals.

**math_query**
```
POST /math_query
```

Answer a math question in natural language.

**Request:**
```json
{
  "task": "math_query",
  "query": "What is the fundamental theorem of algebra?",
  "n": 3,
  "history": []
}
```

### LeanAide CrewAI Bridge API

#### LeanAideCrewAIBridge

Bridge between LeanAide and CrewAI workflow phases.

```python
from leanaide_crewai_bridge import LeanAideCrewAIBridge, LeanAideConfig
import asyncio

async def main():
    config = LeanAideConfig(
        host="localhost",
        port=7654,
        enable_tickets=True,
        ticket_base_url="http://localhost:8000"
    )

    bridge = LeanAideCrewAIBridge(config)

    # Run full workflow
    result = await bridge.execute_full_workflow(
        "Prove that there are infinitely many prime numbers"
    )
```

#### Phase Methods

**execute_phase_1_analysis**
```
POST /leanaide/phase1/analyze
```

Analyze mathematical content in problems.

**Request:**
```json
{
  "problem_statement": "Prove there are infinitely many primes",
  "context": {}
}
```

**Response:**
```json
{
  "success": true,
  "phase": "phase_1_analysis",
  "ticket_id": "LEANAIDE-000001",
  "metadata": {
    "has_mathematical_content": true,
    "domain": "number_theory",
    "num_components": 3,
    "average_complexity": 0.6
  }
}
```

**execute_phase_2_translate**
```
POST /leanaide/phase2/translate
```

Translate natural language math to Lean 4.

**execute_phase_3_verify**
```
POST /leanaide/phase3/verify
```

Verify solutions using Lean 4 elaboration.

**execute_phase_4_proof_check**
```
POST /leanaide/phase4/proof_check
```

Check proof validity and completeness.

**execute_phase_5_formal_verification**
```
POST /leanaide/phase5/formal_verification
```

Final formal verification.

**execute_phase_6_knowledge_extraction**
```
POST /leanaide/phase6/knowledge_extraction
```

Extract verified theorems for knowledge base.

### LeanAide MCP Tools

Model Context Protocol tools for agent integration.

#### Available MCP Tools

**leanaide_translate_theorem**
- Translate natural language theorems to Lean 4
- Parameters: `theorem_text`, `theorem_name` (optional)

**leanaide_prove_theorem**
- Generate proofs for theorems
- Parameters: `theorem_text`, `theorem_code`, `theorem_statement`

**leanaide_verify_code**
- Verify Lean code correctness
- Parameters: `lean_code`, `timeout` (optional)

**leanaide_math_query**
- Math Q&A with conversation history
- Parameters: `query`, `n` (number of answers), `history`

**leanaide_generate_docs**
- Generate documentation for Lean code
- Parameters: `name`, `code`, `type` (theorem/definition)

**leanaide_extract_components**
- Extract mathematical components from text
- Parameters: `text`, `component_types`

**leanaide_batch_translate**
- Batch translate multiple theorems
- Parameters: `theorems` (list)

#### MCP Tool Usage Example

```python
from leanaide_mcp_tools import leanaide_translate_theorem

# Translate theorem
result = leanaide_translate_theorem(
    theorem_text="There are infinitely many prime numbers"
)

# Result contains:
# {
#     "success": true,
#     "lean_code": "...",
#     "elaborated_type": "..."
# }
```

### LeanAide Configuration

#### Environment Variables

```bash
LEANAIDE_HOST=localhost
LEANAIDE_PORT=7654
LEANAIDE_TIMEOUT=120
```

#### Configuration Object

```python
from leanaide_crewai_bridge import LeanAideConfig

config = LeanAideConfig(
    # Server configuration
    host="localhost",
    port=7654,
    api_endpoint="/api/v1/translate",

    # Execution settings
    default_timeout=300,
    max_concurrent_requests=5,
    execution_mode="synchronous",

    # Verification settings
    enable_verification=True,
    enable_caching=True,
    cache_ttl_seconds=3600,

    # Lean 4 settings
    lean_workspace="./lean_workspace",
    lean_library_path="./lean_libraries",

    # CrewAI ticket settings
    enable_tickets=True,
    ticket_base_url="http://localhost:8000"
)
```

### Mathematical Domains

Supported mathematical domains for classification:

- `ALGEBRA` - Group theory, ring theory, field theory
- `ANALYSIS` - Limits, derivatives, integrals
- `TOPOLOGY` - Topological spaces, metrics
- `NUMBER_THEORY` - Primes, divisibility, modular arithmetic
- `COMBINATORICS` - Permutations, combinations, graphs
- `GEOMETRY` - Triangles, circles, polygons
- `LOGIC` - Propositional, predicate logic
- `SET_THEORY` - Sets, cardinality, infinity
- `GENERAL` - General or mixed mathematical content

### Error Handling

#### Error Codes

- `LEAN_001` - Server not available
- `LEAN_002` - Translation timeout
- `LEAN_003` - Invalid Lean code
- `LEAN_004` - Verification failed
- `LEAN_005` - Proof incomplete
- `LEAN_006` - Unknown mathematical domain
- `LEAN_007` - Invalid task type

#### Error Response Format

```json
{
  "success": false,
  "error_code": "LEAN_002",
  "error": "Translation timeout exceeded",
  "details": {
    "task": "translate_thm",
    "timeout": 300
  }
}
```

---

## Changelog

### v1.1.0
- Added LeanAide integration APIs
- Added formal mathematical verification endpoints
- Added MCP tools for LeanAide
- Added mathematical domain classification
- Added knowledge extraction endpoints

### v1.0.0
- Initial API release
- Problem management endpoints
- Decomposition planning
- Solution validation
- Team coordination
- Analytics and monitoring