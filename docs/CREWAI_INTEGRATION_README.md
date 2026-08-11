# CrewAI Integration in OpenEvolve

This document provides a comprehensive overview of the CrewAI integration within the OpenEvolve system.

## Overview

The CrewAI integration provides a complete multi-agent workflow orchestration system that replaces the previous crewai-based system with MIT-licensed components. The integration includes:

- **CrewAI Workflow Execution**: Multi-agent task execution with role specialization
- **State Management**: Persistent workflow state with versioning and rollback
- **Zero-Error Orchestration**: Error detection, correction, and recovery
- **ACE Learning Integration**: Continuous improvement through experience
- **API Integration**: RESTful endpoints for workflow management
- **Monitoring**: Real-time workflow tracking and metrics

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Layer     │────│  CrewAI Hub      │────│  CrewAI Core    │
│                 │    │                  │    │                 │
│ - REST API      │    │ - Unified Flow   │    │ - Agents        │
│ - WebSocket     │    │ - State Manager  │    │ - Tasks         │
│ - Authentication│    │ - Client/Monitor │    │ - Crews         │
└─────────────────┘    │ - Integration    │    └─────────────────┘
                       │ - Delegation Mgr │          │
                       └──────────────────┘          │
                              │                      │
                       ┌──────────────────┐          │
                       │  Supporting      │◄─────────┘
                       │  Components      │
                       │                  │
                       │ - State Storage  │
                       │ - Zero-Error WF  │
                       │ - ACE Bridge     │
                       │ - Monitoring     │
                       └──────────────────┘
```

## Key Components

### 1. CrewAI Hub (`crewai_hub.py`)
The central integration point that ties all CrewAI components together. Provides:
- Unified interface to all CrewAI functionality
- Consistent error handling
- Resource management
- Component lifecycle management

### 2. State Management (`crewai_state_management.py`)
Handles workflow persistence with:
- JSON-based state storage
- Versioning and rollback capabilities
- State transition validation
- Export/import functionality

### 3. Zero-Error Workflow (`crewai_zero_error_workflow.py`)
Provides error-resistant execution with:
- Automatic error detection
- Correction strategies
- Retry mechanisms
- Detailed error reporting

### 4. ACE Bridge (`ace_crewai_bridge.py`)
Integrates learning capabilities with:
- Six-phase workflow integration
- Skill acquisition and retention
- Continuous improvement
- Experience-based optimization

### 5. Unified Flow (`crewai_unified_flow.py`)
Provides execution method routing with:
- Seven execution methods (Traditional, ROMA, ROMA-MDAP-MAKER, etc.)
- Intelligent method selection
- Fallback implementations
- Event-driven architecture

### 6. API Routes (`crewai_api_routes.py`)
RESTful endpoints for:
- Workflow execution
- State management
- Monitoring and metrics
- Task delegation

## Execution Methods

The system supports seven execution methods:

1. **Traditional**: Standard AI-assisted decomposition
2. **ROMA**: Recursive meta-agent decomposition
3. **ROMA-MDAP-MAKER**: Recursive decomposition + Zero-error voting (NEW)
4. **Claudiomiro**: Autonomous development CLI
5. **DataPizza**: Multi-agent coordination
6. **Hybrid**: ROMA + Decomposition Workflow teams
7. **Auto**: Intelligent selection based on problem analysis

## API Endpoints

### Execute Workflow
```
POST /crewai/execute
```
Execute a multi-agent workflow with specified configuration.

### List Workflows
```
GET /crewai/workflows
```
List all workflows, optionally filtered by status.

### Get Workflow
```
GET /crewai/workflows/{workflow_id}
```
Get detailed state of a specific workflow.

### Get Metrics
```
GET /crewai/workflows/{workflow_id}/metrics
```
Get comprehensive metrics for a workflow.

### Get Tickets
```
GET /crewai/workflows/{workflow_id}/tickets
```
Get ticket-like entries for workflow sub-tasks.

### Sync Delegations
```
POST /crewai/sync
```
Sync all delegated tasks with the CrewAI system.

### Health Check
```
GET /crewai/status
```
Get status of all CrewAI components.

## Usage Examples

### Basic Workflow Execution
```python
from crewai_hub import execute_crewai_task
from crewai_state_management import ExecutionMethod

# Execute a workflow with default settings
result = await execute_crewai_task(
    problem_statement="How can we improve user engagement?",
    execution_method=ExecutionMethod.ROMA_MDAP_MAKER
)
```

### Advanced Configuration
```python
from crewai_hub import CrewAIHub

# Initialize with custom configuration
hub = CrewAIHub(
    model="gpt-4o-mini",
    state_storage_dir="./my_workflow_states",
    enable_learning=True,
    enable_zero_error=True
)

# Define agents
agents_config = [
    {
        "role": "Research Analyst",
        "goal": "Analyze market trends",
        "backstory": "Expert market analyst",
        "allow_delegation": False
    }
]

# Define tasks
tasks_config = [
    {
        "description": "Analyze current market trends for AI tools",
        "expected_output": "Detailed market analysis report"
    }
]

# Execute workflow
result = await hub.execute_workflow(
    problem_statement="What are the current trends in AI tools?",
    agents_config=agents_config,
    tasks_config=tasks_config,
    execution_method=ExecutionMethod.DATAPIZZA
)
```

### Using the API
```bash
# Execute a workflow
curl -X POST http://localhost:8000/crewai/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "problem_statement": "Design a scalable web application",
    "execution_method": "roma_mdap_maker",
    "enable_learning": true,
    "enable_zero_error": true
  }'

# List workflows
curl -X GET http://localhost:8000/crewai/workflows \
  -H "Authorization: Bearer YOUR_API_KEY"

# Get workflow status
curl -X GET http://localhost:8000/crewai/workflows/WORKFLOW_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Testing

Run the comprehensive test suite:
```bash
python -m pytest test_crewai_integration_complete.py
# or
python test_crewai_integration_complete.py
```

## Configuration

### Environment Variables
- `CREWAI_STATE_DIR`: Directory for state storage (default: "./crewai_states")
- `CREWAI_DEFAULT_MODEL`: Default LLM model (default: "gpt-4o-mini")

### Settings
The system can be configured programmatically with:
- Model selection
- State storage directory
- Learning enablement
- Zero-error execution
- Persistence settings

## Error Handling

The system implements comprehensive error handling:
- Automatic error detection and classification
- Built-in correction strategies
- Graceful degradation
- Detailed error reporting
- Fallback implementations for all components

## Performance Considerations

- Workflow execution time depends on LLM response times
- State persistence adds overhead but provides reliability
- Concurrent workflows may compete for LLM resources
- Large workflows may require significant memory
- Use appropriate execution methods for your use case

## Security Considerations

- All state data is stored locally
- No external services for state management
- Input validation is performed on all data
- Error messages don't expose sensitive information
- API endpoints require authentication

## Migration Notes

This integration replaces the previous crewai-based system with:
- MIT-licensed CrewAI instead of AGPL-licensed crewai
- Local state management instead of database-backed storage
- Improved error handling and recovery
- Better integration with ACE learning system

## Future Enhancements

Planned improvements include:
- Advanced workflow visualization
- More sophisticated error correction strategies
- Enhanced monitoring dashboards
- Integration with additional AI models
- Workflow templating and reuse
- Real-time collaboration features

## Troubleshooting

### Common Issues
1. "CrewAI not available" - Install the crewai package
2. "State directory not writable" - Check permissions on state_storage_dir
3. "Workflow execution failed" - Check LLM connectivity and credentials
4. "Memory issues" - Reduce the number of concurrent workflows

### Debugging
Enable debug logging to troubleshoot issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To contribute to the CrewAI integration:
1. Follow the existing code patterns and conventions
2. Write comprehensive tests for new functionality
3. Update documentation as needed
4. Ensure backward compatibility where possible
5. Submit pull requests with clear descriptions

## License

The CrewAI integration is released under the MIT License, replacing the previous AGPL-licensed components.