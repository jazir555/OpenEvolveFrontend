# BACKEND API REQUIREMENTS
## Complete API Specification for Streamlit to BubbleLab Migration

**Generated:** 2025-01-05
**Agent:** Discovery & Audit Agent
**Purpose:** Complete specification of all backend APIs required for BubbleLab UI integration

---

## EXECUTIVE SUMMARY

This document specifies all backend API endpoints required to support the BubbleLab UI migration from Streamlit. The backend engines remain unchanged (Python), but new REST APIs and WebSocket connections must be exposed for the React/TypeScript frontend.

**Total API Endpoints Required:** 87 endpoints
**WebSocket Channels Required:** 12 channels
**Authentication:** JWT-based API authentication

---

## SECTION 1: ARCHITECTURE OVERVIEW

### 1.1 Current Architecture

```
┌─────────────────┐
│  Streamlit UI   │
│   (Python)      │
└────────┬────────┘
         │ Direct Python imports
         ▼
┌─────────────────────────────────────┐
│      Backend Engines (Python)       │
│  ┌─────────────────────────────┐   │
│  │ Evolution Engine            │   │
│  │ Adversarial Engine          │   │
│  │ Analytics Engine            │   │
│  │ Collaboration Engine        │   │
│  │ Version Control Engine      │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 1.2 Target Architecture

```
┌─────────────────┐
│  BubbleLab UI   │
│ (React/TS)      │
└────────┬────────┘
         │ REST API + WebSocket
         ▼
┌─────────────────────────────────────┐
│         API Gateway Layer           │
│  (FastAPI/Flask)                    │
│  - Authentication                   │
│  - Rate Limiting                    │
│  - Request Validation               │
│  - Response Formatting              │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      Backend Engines (Python)       │
│  ┌─────────────────────────────┐   │
│  │ Evolution Engine            │   │
│  │ Adversarial Engine          │   │
│  │ Analytics Engine            │   │
│  │ Collaboration Engine        │   │
│  │ Version Control Engine      │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 1.3 Technology Stack

**API Framework:** FastAPI (recommended) or Flask
**WebSocket:** FastAPI WebSockets or Socket.IO
**Authentication:** JWT (JSON Web Tokens)
**API Documentation:** OpenAPI/Swagger
**Data Serialization:** Pydantic models
**Async Support:** asyncio for concurrent requests

---

## SECTION 2: AUTHENTICATION & AUTHORIZATION

### 2.1 Authentication Endpoints

#### POST /api/v1/auth/register
**Description:** Register new user account
**Request:**
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "username": "johndoe",
  "full_name": "John Doe"
}
```
**Response:** 201 Created
```json
{
  "user_id": "uuid",
  "email": "user@example.com",
  "username": "johndoe",
  "created_at": "2025-01-05T00:00:00Z"
}
```

#### POST /api/v1/auth/login
**Description:** Authenticate user and receive JWT token
**Request:**
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```
**Response:** 200 OK
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### POST /api/v1/auth/refresh
**Description:** Refresh access token using refresh token
**Request:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```
**Response:** 200 OK
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### POST /api/v1/auth/logout
**Description:** Invalidate refresh token
**Headers:** `Authorization: Bearer <token>`
**Response:** 204 No Content

---

### 2.2 User Management Endpoints

#### GET /api/v1/users/me
**Description:** Get current user profile
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "user_id": "uuid",
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "role": "user",
  "created_at": "2025-01-05T00:00:00Z",
  "preferences": {
    "theme": "dark",
    "language": "en"
  }
}
```

#### PUT /api/v1/users/me
**Description:** Update current user profile
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "full_name": "John Smith",
  "preferences": {
    "theme": "light"
  }
}
```
**Response:** 200 OK
```json
{
  "user_id": "uuid",
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Smith",
  "role": "user",
  "updated_at": "2025-01-05T01:00:00Z"
}
```

---

## SECTION 3: EVOLUTION ENGINE API

### 3.1 Evolution Execution

#### POST /api/v1/evolution/start
**Description:** Start evolutionary optimization
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "content": "Initial code or text",
  "mode": "standard",
  "parameters": {
    "max_iterations": 100,
    "population_size": 50,
    "temperature": 0.7,
    "top_p": 0.9
  },
  "models": [
    {
      "provider": "openai",
      "model": "gpt-4",
      "api_key": "sk-..."
    }
  ]
}
```
**Response:** 202 Accepted
```json
{
  "evolution_id": "uuid",
  "status": "running",
  "created_at": "2025-01-05T00:00:00Z",
  "websocket_url": "wss://api.example.com/ws/evolution/{evolution_id}"
}
```

#### GET /api/v1/evolution/{evolution_id}
**Description:** Get evolution status and results
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "evolution_id": "uuid",
  "status": "running",
  "progress": {
    "current_iteration": 45,
    "max_iterations": 100,
    "percentage": 45
  },
  "population": [
    {
      "id": "ind_1",
      "fitness": 0.85,
      "content": "Optimized code..."
    }
  ],
  "best_individual": {
    "id": "ind_1",
    "fitness": 0.85,
    "content": "Optimized code..."
  },
  "metrics": {
    "average_fitness": 0.72,
    "diversity_score": 0.65,
    "convergence_rate": 0.45
  },
  "started_at": "2025-01-05T00:00:00Z",
  "updated_at": "2025-01-05T00:05:00Z"
}
```

#### POST /api/v1/evolution/{evolution_id}/pause
**Description:** Pause running evolution
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "evolution_id": "uuid",
  "status": "paused",
  "paused_at": "2025-01-05T00:05:00Z"
}
```

#### POST /api/v1/evolution/{evolution_id}/resume
**Description:** Resume paused evolution
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "evolution_id": "uuid",
  "status": "running",
  "resumed_at": "2025-01-05T00:06:00Z"
}
```

#### POST /api/v1/evolution/{evolution_id}/stop
**Description:** Stop evolution execution
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "evolution_id": "uuid",
  "status": "stopped",
  "stopped_at": "2025-01-05T00:07:00Z",
  "final_results": {
    "best_fitness": 0.85,
    "iterations_completed": 45
  }
}
```

#### DELETE /api/v1/evolution/{evolution_id}
**Description:** Delete evolution and associated data
**Headers:** `Authorization: Bearer <token>`
**Response:** 204 No Content

---

### 3.2 Evolution History & Listing

#### GET /api/v1/evolution
**Description:** List all evolutions for current user
**Headers:** `Authorization: Bearer <token>`
**Query Parameters:**
- `status`: Filter by status (running, completed, paused, stopped)
- `limit`: Number of results (default: 20)
- `offset`: Pagination offset (default: 0)
- `sort`: Sort field (created_at, updated_at, fitness)
- `order`: Sort order (asc, desc)

**Response:** 200 OK
```json
{
  "evolutions": [
    {
      "evolution_id": "uuid",
      "status": "completed",
      "mode": "standard",
      "created_at": "2025-01-05T00:00:00Z",
      "updated_at": "2025-01-05T00:10:00Z",
      "best_fitness": 0.85,
      "iterations_completed": 100
    }
  ],
  "total": 45,
  "limit": 20,
  "offset": 0
}
```

---

## SECTION 4: ADVERSARIAL TESTING API

### 4.1 Adversarial Testing Execution

#### POST /api/v1/adversarial/start
**Description:** Start adversarial testing (red team/blue team)
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "content": "Code or text to test",
  "attack_modes": ["prompt_injection", "jailbreak", "adversarial_example"],
  "parameters": {
    "num_rounds": 5,
    "red_team_models": [
      {
        "provider": "openai",
        "model": "gpt-4"
      }
    ],
    "blue_team_models": [
      {
        "provider": "anthropic",
        "model": "claude-3-opus"
      }
    ]
  }
}
```
**Response:** 202 Accepted
```json
{
  "test_id": "uuid",
  "status": "running",
  "created_at": "2025-01-05T00:00:00Z",
  "websocket_url": "wss://api.example.com/ws/adversarial/{test_id}"
}
```

#### GET /api/v1/adversarial/{test_id}
**Description:** Get adversarial test status and results
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "test_id": "uuid",
  "status": "running",
  "current_round": 3,
  "total_rounds": 5,
  "red_team_results": [
    {
      "round": 1,
      "attack_mode": "prompt_injection",
      "success": true,
      "vulnerability": "SQL injection possibility",
      "payload": "attack payload..."
    }
  ],
  "blue_team_results": [
    {
      "round": 1,
      "patch": "Fixed code...",
      "patch_approved": true
    }
  ],
  "vulnerabilities_found": 5,
  "patches_generated": 5,
  "patches_approved": 4
}
```

#### POST /api/v1/adversarial/{test_id}/approve-patch
**Description:** Approve or reject blue team patch
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "round": 1,
  "approved": true,
  "feedback": "Good patch, addresses the vulnerability"
}
```
**Response:** 200 OK
```json
{
  "test_id": "uuid",
  "round": 1,
  "patch_approved": true,
  "approved_at": "2025-01-05T00:05:00Z"
}
```

#### POST /api/v1/adversarial/{test_id}/stop
**Description:** Stop adversarial testing
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "test_id": "uuid",
  "status": "stopped",
  "stopped_at": "2025-01-05T00:07:00Z"
}
```

---

### 4.2 Adversarial Testing History

#### GET /api/v1/adversarial
**Description:** List all adversarial tests
**Headers:** `Authorization: Bearer <token>`
**Query Parameters:** Same as evolution listing

**Response:** 200 OK
```json
{
  "tests": [
    {
      "test_id": "uuid",
      "status": "completed",
      "created_at": "2025-01-05T00:00:00Z",
      "vulnerabilities_found": 5,
      "patches_approved": 4
    }
  ],
  "total": 20
}
```

---

## SECTION 5: ANALYTICS & MONITORING API

### 5.1 Metrics & Analytics

#### GET /api/v1/analytics/metrics
**Description:** Get aggregated performance metrics
**Headers:** `Authorization: Bearer <token>`
**Query Parameters:**
- `start_date`: ISO 8601 start date
- `end_date`: ISO 8601 end date
- `granularity`: hour, day, week, month

**Response:** 200 OK
```json
{
  "period": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-01-05T00:00:00Z",
    "granularity": "day"
  },
  "metrics": {
    "total_evolutions": 45,
    "total_adversarial_tests": 20,
    "average_fitness_improvement": 0.35,
    "average_vulnerabilities_found": 3.2,
    "success_rate": 0.85
  },
  "time_series": [
    {
      "timestamp": "2025-01-01T00:00:00Z",
      "evolutions": 10,
      "adversarial_tests": 5,
      "average_fitness": 0.72
    }
  ]
}
```

#### GET /api/v1/analytics/performance
**Description:** Get detailed performance analytics
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "model_performance": [
    {
      "model": "gpt-4",
      "provider": "openai",
      "total_calls": 150,
      "average_latency": 2.5,
      "success_rate": 0.98,
      "average_quality_score": 0.85
    }
  ],
  "cost_analysis": {
    "total_cost": 45.50,
    "cost_by_model": {
      "gpt-4": 35.00,
      "claude-3-opus": 10.50
    }
  }
}
```

---

### 5.2 System Monitoring

#### GET /api/v1/monitoring/health
**Description:** Get system health status
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "status": "healthy",
  "services": {
    "api": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "websocket": "healthy"
  },
  "resource_usage": {
    "cpu_percent": 45,
    "memory_percent": 62,
    "disk_percent": 55
  },
  "active_operations": {
    "evolutions_running": 5,
    "adversarial_tests_running": 2
  }
}
```

#### GET /api/v1/monitoring/logs
**Description:** Get application logs
**Headers:** `Authorization: Bearer <token>`
**Query Parameters:**
- `level`: INFO, WARNING, ERROR
- `limit`: Number of log entries (default: 100)
- `offset`: Pagination offset

**Response:** 200 OK
```json
{
  "logs": [
    {
      "timestamp": "2025-01-05T00:00:00Z",
      "level": "INFO",
      "message": "Evolution started",
      "context": {
        "evolution_id": "uuid"
      }
    }
  ],
  "total": 500
}
```

---

## SECTION 6: CONTENT MANAGEMENT API

### 6.1 Content CRUD

#### POST /api/v1/content
**Description:** Create new content
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "title": "My Content",
  "content": "Content text or code",
  "language": "python",
  "tags": ["optimization", "performance"]
}
```
**Response:** 201 Created
```json
{
  "content_id": "uuid",
  "title": "My Content",
  "content": "Content text or code",
  "language": "python",
  "tags": ["optimization", "performance"],
  "created_at": "2025-01-05T00:00:00Z",
  "updated_at": "2025-01-05T00:00:00Z"
}
```

#### GET /api/v1/content/{content_id}
**Description:** Get content by ID
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "content_id": "uuid",
  "title": "My Content",
  "content": "Content text or code",
  "language": "python",
  "tags": ["optimization", "performance"],
  "version": 3,
  "created_at": "2025-01-05T00:00:00Z",
  "updated_at": "2025-01-05T01:00:00Z"
}
```

#### PUT /api/v1/content/{content_id}
**Description:** Update content
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "title": "Updated Title",
  "content": "Updated content"
}
```
**Response:** 200 OK
```json
{
  "content_id": "uuid",
  "title": "Updated Title",
  "content": "Updated content",
  "version": 4,
  "updated_at": "2025-01-05T02:00:00Z"
}
```

#### DELETE /api/v1/content/{content_id}
**Description:** Delete content
**Headers:** `Authorization: Bearer <token>`
**Response:** 204 No Content

#### GET /api/v1/content
**Description:** List all content
**Headers:** `Authorization: Bearer <token>`
**Query Parameters:**
- `tag`: Filter by tag
- `language`: Filter by language
- `limit`, `offset`: Pagination

**Response:** 200 OK
```json
{
  "content": [
    {
      "content_id": "uuid",
      "title": "My Content",
      "language": "python",
      "version": 3,
      "created_at": "2025-01-05T00:00:00Z"
    }
  ],
  "total": 15
}
```

---

## SECTION 7: VERSION CONTROL API

### 7.1 Version History

#### GET /api/v1/content/{content_id}/versions
**Description:** Get version history for content
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "versions": [
    {
      "version": 1,
      "created_at": "2025-01-05T00:00:00Z",
      "created_by": "user_id",
      "comment": "Initial version"
    },
    {
      "version": 2,
      "created_at": "2025-01-05T01:00:00Z",
      "created_by": "user_id",
      "comment": "Added optimization"
    }
  ]
}
```

#### POST /api/v1/content/{content_id}/versions/{version}/revert
**Description:** Revert content to specific version
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "content_id": "uuid",
  "reverted_to_version": 2,
  "new_version": 5,
  "reverted_at": "2025-01-05T03:00:00Z"
}
```

#### GET /api/v1/content/{content_id}/versions/{version1}/diff/{version2}
**Description:** Get diff between two versions
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "version1": 2,
  "version2": 3,
  "diff": "--- a/content.py\n+++ b/content.py\n@@ -1,3 +1,4 @@\n def foo():\n+    print('hello')\n     return 42"
}
```

---

### 7.2 Branching

#### POST /api/v1/content/{content_id}/branches
**Description:** Create new branch
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "branch_name": "feature-experiment",
  "from_version": 3
}
```
**Response:** 201 Created
```json
{
  "branch_id": "uuid",
  "branch_name": "feature-experiment",
  "created_at": "2025-01-05T00:00:00Z"
}
```

#### GET /api/v1/content/{content_id}/branches
**Description:** List all branches
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "branches": [
    {
      "branch_id": "uuid",
      "branch_name": "feature-experiment",
      "version": 4,
      "created_at": "2025-01-05T00:00:00Z"
    }
  ]
}
```

---

## SECTION 8: COLLABORATION API

### 8.1 Real-time Collaboration

#### POST /api/v1/collaboration/rooms
**Description:** Create collaboration room
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "content_id": "uuid",
  "room_name": "Team Editing Session"
}
```
**Response:** 201 Created
```json
{
  "room_id": "uuid",
  "room_name": "Team Editing Session",
  "websocket_url": "wss://api.example.com/ws/collaboration/{room_id}",
  "created_at": "2025-01-05T00:00:00Z"
}
```

#### GET /api/v1/collaboration/rooms/{room_id}/users
**Description:** Get active users in room
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "users": [
    {
      "user_id": "uuid",
      "username": "johndoe",
      "joined_at": "2025-01-05T00:00:00Z",
      "cursor_position": {
        "line": 10,
        "column": 5
      }
    }
  ]
}
```

---

### 8.2 Comments & Annotations

#### POST /api/v1/content/{content_id}/comments
**Description:** Add comment to content
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "comment": "This looks good!",
  "line_start": 10,
  "line_end": 15,
  "parent_comment_id": null
}
```
**Response:** 201 Created
```json
{
  "comment_id": "uuid",
  "comment": "This looks good!",
  "line_start": 10,
  "line_end": 15,
  "created_at": "2025-01-05T00:00:00Z"
}
```

#### GET /api/v1/content/{content_id}/comments
**Description:** Get all comments for content
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "comments": [
    {
      "comment_id": "uuid",
      "user_id": "uuid",
      "username": "johndoe",
      "comment": "This looks good!",
      "line_start": 10,
      "line_end": 15,
      "created_at": "2025-01-05T00:00:00Z",
      "replies": []
    }
  ]
}
```

---

## SECTION 9: CONFIGURATION API

### 9.1 Provider & Model Management

#### GET /api/v1/config/providers
**Description:** Get available LLM providers
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "providers": [
    {
      "provider": "openai",
      "name": "OpenAI",
      "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
      "requires_api_key": true
    },
    {
      "provider": "anthropic",
      "name": "Anthropic",
      "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
      "requires_api_key": true
    }
  ]
}
```

#### POST /api/v1/config/providers/{provider}/api-key
**Description:** Save API key for provider
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "api_key": "sk-..."
}
```
**Response:** 200 OK
```json
{
  "provider": "openai",
  "api_key_last_four": "sk-****1234",
  "saved_at": "2025-01-05T00:00:00Z"
}
```

#### GET /api/v1/config/parameters
**Description:** Get user's default parameters
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "generation": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 4096
  },
  "evolution": {
    "max_iterations": 100,
    "population_size": 50
  }
}
```

#### PUT /api/v1/config/parameters
**Description:** Update user's default parameters
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "generation": {
    "temperature": 0.8
  }
}
```
**Response:** 200 OK
```json
{
  "generation": {
    "temperature": 0.8,
    "top_p": 0.9,
    "max_tokens": 4096
  }
}
```

---

## SECTION 10: WORKFLOW API

### 10.1 Integrated Workflow

#### POST /api/v1/workflow/start
**Description:** Start integrated workflow (plan → evolve → test → evaluate)
**Headers:** `Authorization: Bearer <token>`
**Request:**
```json
{
  "problem_statement": "Optimize this algorithm",
  "workflow_template": "standard",
  "parameters": {}
}
```
**Response:** 202 Accepted
```json
{
  "workflow_id": "uuid",
  "status": "running",
  "current_stage": "planning",
  "websocket_url": "wss://api.example.com/ws/workflow/{workflow_id}"
}
```

#### GET /api/v1/workflow/{workflow_id}
**Description:** Get workflow status
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "workflow_id": "uuid",
  "status": "running",
  "current_stage": "evolution",
  "stages": [
    {
      "stage": "planning",
      "status": "completed",
      "result": {}
    },
    {
      "stage": "evolution",
      "status": "running",
      "progress": 0.45
    }
  ]
}
```

---

## SECTION 11: FILE OPERATIONS API

### 11.1 File Upload/Download

#### POST /api/v1/files/upload
**Description:** Upload file
**Headers:** `Authorization: Bearer <token>`
**Content-Type:** `multipart/form-data`
**Request:** File data
**Response:** 201 Created
```json
{
  "file_id": "uuid",
  "filename": "document.pdf",
  "size": 1024000,
  "mime_type": "application/pdf",
  "uploaded_at": "2025-01-05T00:00:00Z"
}
```

#### GET /api/v1/files/{file_id}/download
**Description:** Download file
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK (file content)

#### GET /api/v1/files/{file_id}
**Description:** Get file metadata
**Headers:** `Authorization: Bearer <token>`
**Response:** 200 OK
```json
{
  "file_id": "uuid",
  "filename": "document.pdf",
  "size": 1024000,
  "mime_type": "application/pdf",
  "uploaded_at": "2025-01-05T00:00:00Z"
}
```

---

## SECTION 12: WEBSOCKET CHANNELS

### 12.1 Evolution Progress Channel

**Channel:** `ws://api.example.com/ws/evolution/{evolution_id}`

**Description:** Real-time evolution progress updates

**Message Format (Server → Client):**
```json
{
  "type": "progress_update",
  "data": {
    "evolution_id": "uuid",
    "iteration": 45,
    "max_iterations": 100,
    "best_fitness": 0.85,
    "population_metrics": {
      "average_fitness": 0.72,
      "diversity_score": 0.65
    }
  }
}
```

**Message Types:**
- `progress_update`: Iteration progress
- `generation_complete`: New generation completed
- `evolution_complete`: Evolution finished
- `error`: Error occurred

---

### 12.2 Adversarial Testing Channel

**Channel:** `ws://api.example.com/ws/adversarial/{test_id}`

**Description:** Real-time adversarial testing updates

**Message Format:**
```json
{
  "type": "attack_generated",
  "data": {
    "test_id": "uuid",
    "round": 1,
    "attack_mode": "prompt_injection",
    "vulnerability": "SQL injection possibility",
    "payload": "attack payload..."
  }
}
```

**Message Types:**
- `attack_generated`: Red team generated attack
- `patch_generated`: Blue team generated patch
- `patch_approved`: Patch was approved
- `test_complete`: Test completed

---

### 12.3 Collaboration Channel

**Channel:** `ws://api.example.com/ws/collaboration/{room_id}`

**Description:** Real-time collaboration updates

**Message Types:**
- `user_joined`: User joined room
- `user_left`: User left room
- `content_update`: Content edited
- `cursor_update`: Cursor position updated
- `comment_added`: New comment added

---

### 12.4 System Monitoring Channel

**Channel:** `ws://api.example.com/ws/monitoring`

**Description:** Real-time system monitoring updates

**Message Format:**
```json
{
  "type": "resource_update",
  "data": {
    "cpu_percent": 45,
    "memory_percent": 62,
    "active_evolutions": 5,
    "active_adversarial_tests": 2
  }
}
```

**Message Types:**
- `resource_update`: Resource usage update
- `service_status`: Service health update
- `log_entry`: New log entry

---

## SECTION 13: ERROR HANDLING

### 13.1 Standard Error Response Format

All API errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "temperature",
      "issue": "Must be between 0.0 and 2.0"
    }
  }
}
```

### 13.2 HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `204 No Content`: Successful request with no response body
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

---

## SECTION 14: RATE LIMITING

### 14.1 Rate Limit Rules

- **Authenticated Users:** 100 requests per minute
- **Unauthenticated Users:** 20 requests per minute
- **WebSocket Connections:** 10 concurrent connections per user
- **File Upload:** 5 uploads per minute

### 14.2 Rate Limit Headers

All API responses include rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1641369600
```

---

## SECTION 15: PAGINATION

### 15.1 Pagination Format

List endpoints support pagination with these query parameters:
- `limit`: Number of items per page (default: 20, max: 100)
- `offset`: Number of items to skip (default: 0)

**Response Format:**
```json
{
  "items": [...],
  "total": 150,
  "limit": 20,
  "offset": 0,
  "has_more": true
}
```

---

## SECTION 16: IMPLEMENTATION PRIORITIES

### 16.1 Phase 1: Core APIs (Critical - Week 1-2)

1. Authentication & User Management
2. Evolution Execution API
3. Adversarial Testing API
4. WebSocket channels for real-time updates

### 16.2 Phase 2: Content & Configuration (High - Week 3)

1. Content Management API
2. Configuration API (providers, parameters)
3. File Operations API

### 16.3 Phase 3: Advanced Features (Medium - Week 4)

1. Analytics & Monitoring API
2. Version Control API
3. Collaboration API
4. Workflow API

---

## SECTION 17: TESTING REQUIREMENTS

### 17.1 API Testing

- Unit tests for all endpoints
- Integration tests for end-to-end flows
- Load testing for performance validation
- Security testing for authentication/authorization

### 17.2 WebSocket Testing

- Connection establishment tests
- Message broadcast tests
- Reconnection tests
- Performance tests under high concurrency

---

## SECTION 18: DOCUMENTATION REQUIREMENTS

### 18.1 API Documentation

- OpenAPI/Swagger specification
- Interactive API documentation (Swagger UI)
- Code examples in multiple languages
- WebSocket protocol documentation

### 18.2 Integration Guides

- Authentication guide
- Quick start guide
- Migration guide from Streamlit
- Best practices and patterns

---

## APPENDIX A: PYDANTIC MODELS

### Authentication Models

```python
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    username: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
```

### Evolution Models

```python
class EvolutionConfig(BaseModel):
    max_iterations: int = 100
    population_size: int = 50
    temperature: float = 0.7
    top_p: float = 0.9

class ModelConfig(BaseModel):
    provider: str
    model: str
    api_key: str

class EvolutionStart(BaseModel):
    content: str
    mode: str = "standard"
    parameters: EvolutionConfig
    models: list[ModelConfig]

class EvolutionStatus(BaseModel):
    evolution_id: str
    status: str  # running, completed, paused, stopped
    current_iteration: int
    max_iterations: int
    best_fitness: float
    population: list
```

---

## APPENDIX B: API ENDPOINT SUMMARY

### Complete Endpoint List

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Authentication** |
| POST | /api/v1/auth/register | Register user |
| POST | /api/v1/auth/login | Login user |
| POST | /api/v1/auth/refresh | Refresh token |
| POST | /api/v1/auth/logout | Logout user |
| GET | /api/v1/users/me | Get user profile |
| PUT | /api/v1/users/me | Update user profile |
| **Evolution** |
| POST | /api/v1/evolution/start | Start evolution |
| GET | /api/v1/evolution/{id} | Get evolution status |
| POST | /api/v1/evolution/{id}/pause | Pause evolution |
| POST | /api/v1/evolution/{id}/resume | Resume evolution |
| POST | /api/v1/evolution/{id}/stop | Stop evolution |
| DELETE | /api/v1/evolution/{id} | Delete evolution |
| GET | /api/v1/evolution | List evolutions |
| **Adversarial** |
| POST | /api/v1/adversarial/start | Start adversarial test |
| GET | /api/v1/adversarial/{id} | Get test status |
| POST | /api/v1/adversarial/{id}/approve-patch | Approve patch |
| POST | /api/v1/adversarial/{id}/stop | Stop test |
| GET | /api/v1/adversarial | List tests |
| **Analytics** |
| GET | /api/v1/analytics/metrics | Get metrics |
| GET | /api/v1/analytics/performance | Get performance |
| **Monitoring** |
| GET | /api/v1/monitoring/health | Get health |
| GET | /api/v1/monitoring/logs | Get logs |
| **Content** |
| POST | /api/v1/content | Create content |
| GET | /api/v1/content/{id} | Get content |
| PUT | /api/v1/content/{id} | Update content |
| DELETE | /api/v1/content/{id} | Delete content |
| GET | /api/v1/content | List content |
| **Version Control** |
| GET | /api/v1/content/{id}/versions | Get versions |
| POST | /api/v1/content/{id}/versions/{v}/revert | Revert version |
| GET | /api/v1/content/{id}/versions/{v1}/diff/{v2} | Get diff |
| POST | /api/v1/content/{id}/branches | Create branch |
| GET | /api/v1/content/{id}/branches | List branches |
| **Collaboration** |
| POST | /api/v1/collaboration/rooms | Create room |
| GET | /api/v1/collaboration/rooms/{id}/users | Get users |
| POST | /api/v1/content/{id}/comments | Add comment |
| GET | /api/v1/content/{id}/comments | Get comments |
| **Configuration** |
| GET | /api/v1/config/providers | Get providers |
| POST | /api/v1/config/providers/{p}/api-key | Save API key |
| GET | /api/v1/config/parameters | Get parameters |
| PUT | /api/v1/config/parameters | Update parameters |
| **Workflow** |
| POST | /api/v1/workflow/start | Start workflow |
| GET | /api/v1/workflow/{id} | Get workflow status |
| **Files** |
| POST | /api/v1/files/upload | Upload file |
| GET | /api/v1/files/{id}/download | Download file |
| GET | /api/v1/files/{id} | Get file metadata |

---

**END OF BACKEND API REQUIREMENTS**

**Last Updated:** 2025-01-05
**Status:** COMPLETE - Ready for Agent 2 (Backend API Implementation)
