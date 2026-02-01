# OpenEvolve-Knowledge Engine API Integration Specification

## Document Information
- **Version**: 1.0
- **Date**: February 1, 2026
- **Status**: Draft
- **Authors**: OpenEvolve Team

## Table of Contents
1. [Overview](#overview)
2. [API Architecture](#api-architecture)
3. [Authentication & Authorization](#authentication--authorization)
4. [Core Endpoints](#core-endpoints)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Security](#security)
9. [Testing](#testing)

## Overview

### Purpose
This document specifies the API integration between OpenEvolve and the Knowledge Engine. The API enables OpenEvolve to access knowledge services, submit artifacts for processing, and receive insights during the evolutionary process.

### Goals
- Define RESTful API endpoints for knowledge operations
- Establish secure communication protocols
- Provide consistent error handling
- Enable efficient data exchange between systems

### Non-Goals
- Defining internal system architectures
- Specifying implementation details of individual services
- Covering UI or client-side concerns

## API Architecture

### Protocol
- **Transport**: HTTPS/TLS
- **Format**: JSON over HTTP
- **Style**: RESTful with HATEOAS principles
- **Versioning**: URI versioning (e.g., `/api/v1/`)

### Base URL
```
https://knowledge-engine.openevolve.org/api/v1
```

### Supported HTTP Methods
- `GET` - Retrieve resources
- `POST` - Create resources or trigger operations
- `PUT` - Update entire resources
- `PATCH` - Partial resource updates
- `DELETE` - Remove resources

## Authentication & Authorization

### Authentication Method
- **Type**: API Key with JWT tokens
- **Header**: `Authorization: Bearer {token}`
- **Token Format**: JWT with RS256 signing

### API Key Management
````
POST /api/v1/auth/api-keys
Content-Type: application/json

{
  "name": "openevolve-integration",
  "permissions": ["read:knowledge", "write:artifacts", "read:insights"],
  "expires_in_days": 365
}
````

### Token Refresh
````
POST /api/v1/auth/refresh
Content-Type: application/json
Authorization: Bearer {valid_token}

{
  "refresh_token": "refresh_token_value"
}
````

## Core Endpoints

### Project Management

#### Register Project
````
POST /api/v1/projects
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Optimization Project Alpha",
  "description": "Code optimization experiment",
  "domain": "machine_learning",
  "metadata": {
    "language": "python",
    "complexity": "high",
    "deadline": "2026-06-01"
  }
}

HTTP/1.1 201 Created
{
  "project_id": "proj_abc123",
  "name": "Optimization Project Alpha",
  "created_at": "2026-02-01T10:00:00Z",
  "knowledge_graph_id": "kg_proj_abc123"
}
````

#### Get Project
````
GET /api/v1/projects/{project_id}
Authorization: Bearer {token}

HTTP/1.1 200 OK
{
  "project_id": "proj_abc123",
  "name": "Optimization Project Alpha",
  "description": "Code optimization experiment",
  "stage": "in_progress",
  "created_at": "2026-02-01T10:00:00Z",
  "updated_at": "2026-02-01T12:00:00Z",
  "metadata": {
    "language": "python",
    "complexity": "high"
  },
  "stats": {
    "artifacts_processed": 45,
    "insights_generated": 12,
    "last_activity": "2026-02-01T12:00:00Z"
  }
}
````

### Knowledge Operations

#### Submit Artifact for Processing
````
POST /api/v1/projects/{project_id}/artifacts
Content-Type: multipart/form-data
Authorization: Bearer {token}

FormData:
- artifact_type: "code_change"
- content: <file_content>
- metadata: {"author": "evolution_agent", "iteration": 42}

HTTP/1.1 202 Accepted
{
  "artifact_id": "art_123xyz",
  "status": "processing",
  "estimated_completion": "2026-02-01T10:05:00Z",
  "processing_job_id": "job_456def"
}
````

#### Get Processed Insights
````
GET /api/v1/projects/{project_id}/artifacts/{artifact_id}/insights
Authorization: Bearer {token}

HTTP/1.1 200 OK
{
  "artifact_id": "art_123xyz",
  "status": "completed",
  "insights": [
    {
      "type": "performance_optimization",
      "confidence": 0.85,
      "description": "Suggested algorithm change reduces complexity from O(n²) to O(n log n)",
      "recommendation": "Replace bubble sort with merge sort",
      "code_snippet": "def optimized_sort(arr):\n    return sorted(arr)"
    },
    {
      "type": "memory_efficiency",
      "confidence": 0.72,
      "description": "Memory allocation pattern could be improved",
      "recommendation": "Use generator expressions instead of list comprehensions"
    }
  ],
  "processing_time_ms": 1245,
  "extracted_entities": ["algorithm", "complexity", "sorting", "merge_sort"],
  "relations": [
    {"subject": "bubble_sort", "relation": "less_efficient_than", "object": "merge_sort"}
  ]
}
````

### Knowledge Graph Operations

#### Query Knowledge Graph
````
POST /api/v1/projects/{project_id}/knowledge/query
Content-Type: application/json
Authorization: Bearer {token}

{
  "query": "Find all optimization techniques for sorting algorithms",
  "filters": {
    "domain": "algorithms",
    "relevance_score": {"$gte": 0.7}
  },
  "limit": 10
}

HTTP/1.1 200 OK
{
  "results": [
    {
      "id": "tech_001",
      "type": "optimization_technique",
      "name": "Divide and Conquer",
      "description": "Split problem into smaller subproblems",
      "related_algorithms": ["merge_sort", "quick_sort"],
      "confidence": 0.92,
      "source": "algorithm_textbook"
    }
  ],
  "total_results": 1,
  "query_time_ms": 45
}
````

#### Add Knowledge Triple
````
POST /api/v1/projects/{project_id}/knowledge/triples
Content-Type: application/json
Authorization: Bearer {token}

{
  "subject": "merge_sort",
  "predicate": "has_complexity",
  "object": "O(n log n)",
  "confidence": 0.95,
  "metadata": {
    "source": "evolution_process",
    "iteration": 42
  }
}

HTTP/1.1 201 Created
{
  "triple_id": "triple_xyz789",
  "inserted": true,
  "duplicate": false
}
````

### Real-time Updates

#### WebSocket Connection
````
GET wss://knowledge-engine.openevolve.org/ws/v1/projects/{project_id}/updates
Sec-WebSocket-Key: {key}
Sec-WebSocket-Version: 13
Authorization: Bearer {token}

# Client sends subscription message
{
  "action": "subscribe",
  "channels": ["artifacts", "insights", "project_updates"]
}

# Server sends updates
{
  "channel": "insights",
  "event": "new_insight",
  "data": {
    "artifact_id": "art_123xyz",
    "insight": {
      "type": "performance_optimization",
      "description": "New optimization discovered"
    },
    "timestamp": "2026-02-01T10:30:00Z"
  }
}
````

### Batch Operations

#### Submit Multiple Artifacts
````
POST /api/v1/projects/{project_id}/artifacts/batch
Content-Type: application/json
Authorization: Bearer {token}

{
  "artifacts": [
    {
      "artifact_type": "code_change",
      "content": "def func1(): ...",
      "metadata": {"iteration": 1}
    },
    {
      "artifact_type": "code_change", 
      "content": "def func2(): ...",
      "metadata": {"iteration": 2}
    }
  ],
  "options": {
    "process_sequentially": false,
    "priority": "normal"
  }
}

HTTP/1.1 202 Accepted
{
  "batch_id": "batch_abc123",
  "total_artifacts": 2,
  "status": "processing",
  "estimated_completion": "2026-02-01T10:15:00Z"
}
````

## Data Models

### Project Context
````
{
  "project_id": "string (required)",
  "name": "string (required, max 255 chars)",
  "description": "string (optional, max 1000 chars)",
  "stage": "enum (initialized|planning|in_progress|review|completed|archived)",
  "domain": "string (optional)",
  "created_at": "ISO 8601 datetime",
  "updated_at": "ISO 8601 datetime",
  "metadata": "object (optional, max 10 key-value pairs)",
  "stats": {
    "artifacts_processed": "integer",
    "insights_generated": "integer",
    "last_activity": "ISO 8601 datetime"
  }
}
````

### Knowledge Artifact
````
{
  "artifact_id": "string (required)",
  "project_id": "string (required)",
  "artifact_type": "enum (code_change|performance_metric|error_log|evaluation_result)",
  "content": "string or file reference",
  "metadata": {
    "author": "string",
    "iteration": "integer",
    "language": "string",
    "complexity": "string"
  },
  "created_at": "ISO 8601 datetime",
  "status": "enum (submitted|processing|completed|failed)"
}
````

### Insight
````
{
  "insight_id": "string (required)",
  "artifact_id": "string (required)",
  "type": "enum (performance_optimization|memory_efficiency|algorithm_improvement|bug_fix|security_vulnerability)",
  "confidence": "float (0.0 - 1.0)",
  "description": "string (required)",
  "recommendation": "string (optional)",
  "code_snippet": "string (optional)",
  "related_entities": ["string"],
  "relations": [
    {
      "subject": "string",
      "predicate": "string", 
      "object": "string"
    }
  ],
  "generated_at": "ISO 8601 datetime"
}
````

### Knowledge Triple
````
{
  "triple_id": "string (required)",
  "subject": "string (required)",
  "predicate": "string (required)",
  "object": "string (required)",
  "confidence": "float (0.0 - 1.0)",
  "project_id": "string (required)",
  "source": "string (required)",
  "created_at": "ISO 8601 datetime",
  "metadata": "object (optional)"
}
````

## Error Handling

### Standard Error Response
````
{
  "error": {
    "code": "string (e.g., INVALID_INPUT, RESOURCE_NOT_FOUND)",
    "message": "human readable error message",
    "details": "object with specific error details (optional)",
    "timestamp": "ISO 8601 datetime",
    "request_id": "string (for debugging)"
  }
}
````

### HTTP Status Codes
- `200 OK` - Request successful
- `201 Created` - Resource created
- `202 Accepted` - Request accepted for processing
- `400 Bad Request` - Invalid request format
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource doesn't exist
- `422 Unprocessable Entity` - Valid request, invalid semantics
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

### Common Error Codes
- `INVALID_INPUT` - Request data doesn't match schema
- `RESOURCE_NOT_FOUND` - Referenced resource doesn't exist
- `INSUFFICIENT_PERMISSIONS` - Insufficient access rights
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `PROCESSING_ERROR` - Error during background processing
- `AUTHENTICATION_FAILED` - Invalid credentials

## Rate Limiting

### Limits
- **Authenticated requests**: 1000 requests/hour per API key
- **Unauthenticated requests**: 100 requests/hour per IP
- **File uploads**: 10 files/hour per project
- **Concurrent processing**: 5 artifacts simultaneously per project

### Headers
````
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1643734800  # Unix timestamp
````

### Exceeding Limits
When rate limit is exceeded:
- Status: `429 Too Many Requests`
- Body: Standard error response with `RATE_LIMIT_EXCEEDED` code
- Retry-After header with suggested wait time

## Security

### Transport Security
- All APIs require HTTPS
- TLS 1.2 minimum
- Certificate pinning recommended for clients

### Data Protection
- API keys must be transmitted via Authorization header
- Sensitive data encrypted at rest
- Audit logs for all API access

### Input Validation
- All inputs validated against schemas
- Sanitized to prevent injection attacks
- File uploads scanned for malware

### Access Control
- Role-based permissions
- Project isolation
- API key scope restrictions

## Testing

### Test Endpoints
- **Sandbox environment**: `https://sandbox.knowledge-engine.openevolve.org/api/v1`
- **Health check**: `GET /api/v1/health`
- **Version info**: `GET /api/v1/version`

### Health Check Response
````
GET /api/v1/health

HTTP/1.1 200 OK
{
  "status": "healthy",
  "version": "1.2.3",
  "timestamp": "2026-02-01T10:00:00Z",
  "dependencies": {
    "database": "connected",
    "message_queue": "connected",
    "storage": "available"
  }
}
````

### SDK Examples
```python
# Python SDK example
from openevolve_kg_client import KnowledgeEngineClient

client = KnowledgeEngineClient(
    api_key="your_api_key",
    base_url="https://knowledge-engine.openevolve.org/api/v1"
)

# Register project
project = await client.register_project({
    "name": "My Project",
    "domain": "optimization"
})

# Submit artifact
artifact = await client.submit_artifact(
    project_id=project.id,
    artifact_type="code_change",
    content="def optimized_function(): ..."
)

# Get insights
insights = await client.get_insights(artifact.id)
```

## Appendix

### Glossary
- **Artifact**: Code, data, or other content submitted for knowledge processing
- **Insight**: Processed knowledge or recommendations derived from artifacts
- **Triple**: Subject-Predicate-Object relationship in the knowledge graph
- **Project**: Logical grouping of related knowledge operations

### Change Log
- **v1.0** - Initial specification