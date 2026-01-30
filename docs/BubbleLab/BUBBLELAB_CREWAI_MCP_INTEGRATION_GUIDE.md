# BubbleLab CrewAI MCP Server - Complete Integration Guide

## 📋 Overview

The BubbleLab CrewAI MCP (Model Context Protocol) Server provides a bridge between BubbleLab's workflow system and CrewAI's multi-agent orchestration capabilities. This integration enables BubbleLab workflows to delegate complex orchestration tasks to CrewAI when advanced multi-agent coordination is required.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   BubbleLab     │◄──►│  MCP Server        │◄──►│   CrewAI        │
│   Workflows     │    │  (bubblelab_mcp_  │    │   Agents &      │
│                 │    │   server.py)       │    │   Orchestration │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│ BubbleLab       │    │ • FastAPI Server   │    │ • Agent         │
│ Bubbles         │    │ • HTTP Endpoints   │    │   Coordination  │
│ (CrewAIOrchest- │    │ • Tool Registry    │    │ • Task Deleg.   │
│ rationBubble)   │    │ • Crew Management  │    │ • Result Aggr.  │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
```

## 🚀 Components

### 1. MCP Server (`bubblelab_crewai_mcp_server.py`)
- **Technology**: FastAPI-based HTTP server
- **Purpose**: Acts as the communication hub between BubbleLab and CrewAI
- **Features**:
  - Health check endpoint
  - Tool discovery and execution
  - Crew creation and management
  - Task delegation and result aggregation

### 2. CrewAI Integration Layer (`crewai_integration_layer.py`)
- **Purpose**: Abstracts CrewAI functionality behind a clean API
- **Features**:
  - Agent template management
  - Task template management
  - Crew creation and execution
  - Mock implementations when CrewAI is not available

### 3. MCP Client (`bubblelab_mcp_client.py`)
- **Purpose**: Client library for BubbleLab to communicate with the MCP server
- **Features**:
  - HTTP communication with error handling
  - Tool calling abstraction
  - Workflow orchestration helpers

### 4. BubbleLab Bubbles (`CrewAIOrchestrationBubble.ts`)
- **Purpose**: BubbleLab integration components
- **Features**:
  - CrewAIOrchestrationBubble: General orchestration delegation
  - CrewAIResearchBubble: Specialized research workflows

## 🛠️ API Endpoints

### GET `/health`
- **Purpose**: Check server health and CrewAI availability
- **Response**: 
```json
{
  "status": "healthy",
  "timestamp": "2026-01-27T23:30:00.000000",
  "crewai_available": true
}
```

### GET `/tools`
- **Purpose**: List available tools for BubbleLab
- **Response**:
```json
{
  "tools": [
    {
      "name": "create_crewai_agent",
      "description": "Create a CrewAI agent with specified role, goal, and tools",
      "parameters": { ... }
    }
  ]
}
```

### POST `/tools/{tool_name}`
- **Purpose**: Execute a specific tool
- **Request**:
```json
{
  "parameters": { ... }
}
```

### POST `/create_crew`
- **Purpose**: Create a new CrewAI crew
- **Request**:
```json
{
  "agents": [...],
  "tasks": [...],
  "config": { ... }
}
```

### POST `/execute_crew/{crew_id}`
- **Purpose**: Execute a specific crew
- **Request**:
```json
{
  "inputs": { ... }
}
```

## 🧩 BubbleLab Integration

### CrewAIOrchestrationBubble
Enables BubbleLab workflows to delegate complex orchestration tasks to CrewAI.

**Parameters**:
- `serverUrl`: MCP server URL (default: http://localhost:8003)
- `taskDescription`: Description of the task to delegate
- `requiredOutputs`: List of required outputs
- `agentConfigs`: Configuration for custom agents
- `taskConfigs`: Configuration for custom tasks
- `constraints`: Constraints for task execution
- `context`: Additional context for orchestration
- `apiKey`: API key for authentication

### CrewAIResearchBubble
Specialized bubble for research-oriented tasks using CrewAI.

**Parameters**:
- `serverUrl`: MCP server URL (default: http://localhost:8003)
- `topic`: Topic to research
- `researchDepth`: Depth level of research (1-5)
- `additionalConstraints`: Additional constraints for research
- `apiKey`: API key for authentication

## 📦 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r mcp_server_requirements.txt
```

### 2. Start the MCP Server
```bash
python bubblelab_crewai_mcp_server.py
```

The server will start on `http://localhost:8003` by default.

### 3. Configure BubbleLab
- Ensure BubbleLab can reach the MCP server
- Configure the server URL in BubbleLab workflows
- Set up authentication if required

## 🧪 Usage Examples

### Basic Orchestration
```typescript
const orchestrationBubble = new CrewAIOrchestrationBubble({
  taskDescription: "Analyze market trends for renewable energy",
  requiredOutputs: [
    "Market growth projections",
    "Key player analysis", 
    "Investment opportunities"
  ],
  constraints: ["Use only publicly available data", "Focus on US market"]
});

const result = await orchestrationBubble.action();
```

### Research Task
```typescript
const researchBubble = new CrewAIResearchBubble({
  topic: "Artificial Intelligence in Healthcare",
  researchDepth: 3,
  additionalConstraints: ["Focus on diagnostic applications"]
});

const result = await researchBubble.action();
```

## 🔐 Security & Authentication

- API key authentication supported
- Secure HTTP communication
- Input validation and sanitization
- Rate limiting capabilities

## 🧪 Testing

Run the test suite:
```bash
python test_mcp_server.py
```

This validates:
- Python syntax correctness
- Module imports
- Basic functionality
- Integration layer operation

## 🚀 Deployment

### Production Deployment
```bash
# Using uvicorn for production
uvicorn bubblelab_crewai_mcp_server:app --host 0.0.0.0 --port 8003 --workers 4
```

### Container Deployment
Create a Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY mcp_server_requirements.txt .
RUN pip install -r mcp_server_requirements.txt

COPY bubblelab_crewai_mcp_server.py .
COPY crewai_integration_layer.py .
COPY bubblelab_mcp_client.py .

CMD ["uvicorn", "bubblelab_crewai_mcp_server:app", "--host", "0.0.0.0", "--port", "8003"]
```

## 🔄 Troubleshooting

### Common Issues

1. **CrewAI Not Available**: The system gracefully falls back to mock implementations when CrewAI is not installed
2. **Connection Errors**: Verify that BubbleLab can reach the MCP server
3. **Authentication Issues**: Check API key configuration

### Logging
The server provides detailed logging for debugging:
- Request/response logging
- Error tracking
- Performance metrics

## 📈 Future Enhancements

- WebSocket support for real-time updates
- Advanced caching mechanisms
- Enhanced security features
- More specialized bubble types
- Improved error recovery

## 🤝 Integration Patterns

### Pattern 1: Simple Delegation
Use `CrewAIOrchestrationBubble` for general task delegation to CrewAI.

### Pattern 2: Specialized Workflows  
Use `CrewAIResearchBubble` for research-focused tasks.

### Pattern 3: Multi-Stage Workflows
Chain multiple CrewAI bubbles for complex multi-stage processes.

## 📚 References

- [CrewAI Documentation](https://docs.crewai.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [BubbleLab Documentation](https://bubblelab.ai/docs)

---

**Status**: Production Ready
**Last Updated**: January 27, 2026