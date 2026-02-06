"""
CrewAI Integration Documentation
===============================

This document describes the complete CrewAI integration in the OpenEvolve system.

Overview
--------

The CrewAI integration provides a comprehensive workflow orchestration system that combines:
- CrewAI for multi-agent task execution
- State management for workflow persistence
- Zero-error workflow orchestration
- ACE learning for continuous improvement
- Monitoring and reporting capabilities

Architecture
------------

The integration consists of several key components:

1. CrewAIIntegration - Main integration class that ties everything together
2. StateManager - Handles workflow state persistence
3. ZeroErrorWorkflow - Provides error-resistant workflow execution
4. ACECrewAIWorkflowBridge - Integrates learning capabilities
5. CrewAIDelegationManager - Manages task delegation to CrewAI

Key Features
------------

1. Multi-Agent Workflows: Create and execute complex workflows with multiple specialized agents
2. State Persistence: Automatically save and restore workflow states
3. Zero-Error Execution: Built-in error detection, correction, and recovery
4. Learning Integration: Continuous improvement through ACE learning system
5. Monitoring: Real-time tracking of workflow progress and status
6. Delegation: Ability to delegate tasks to CrewAI workflows
7. Scalability: Support for concurrent workflow execution

Usage Examples
--------------

Basic Workflow Execution:

```python
from crewai_integration_complete import execute_crewai_workflow

agents_config = [
    {
        "role": "Research Analyst",
        "goal": "Analyze market trends",
        "backstory": "Expert market analyst",
        "allow_delegation": False
    }
]

tasks_config = [
    {
        "description": "Analyze current market trends for AI tools",
        "expected_output": "Detailed market analysis report"
    }
]

result = await execute_crewai_workflow(
    problem_statement="What are the current trends in AI tools?",
    agents_config=agents_config,
    tasks_config=tasks_config
)
```

Advanced Workflow with State Management:

```python
from crewai_integration_complete import CrewAIIntegration

integration = CrewAIIntegration(
    model="gpt-4o-mini",
    state_storage_dir="./workflow_states",
    enable_learning=True,
    enable_zero_error=True
)

result = await integration.create_and_execute_workflow(
    problem_statement="Solve complex business problem",
    agents_config=agents_config,
    tasks_config=tasks_config,
    execution_method=ExecutionMethod.ROMA_MDAP_MAKER
)
```

Configuration Options
--------------------

The CrewAI integration supports various configuration options:

- model: LLM model to use (default: "gpt-4o-mini")
- state_storage_dir: Directory for state persistence (default: "./crewai_states")
- enable_learning: Whether to enable ACE learning (default: True)
- enable_zero_error: Whether to use zero-error orchestration (default: True)

State Management
---------------

The system automatically manages workflow states with:

- Automatic state persistence
- Versioning and rollback capabilities
- Snapshot creation and restoration
- Export/import functionality
- Cleanup of old states

Error Handling
-------------

The zero-error workflow system provides:

- Automatic error detection
- Built-in correction strategies
- Retry mechanisms with exponential backoff
- Graceful degradation
- Detailed error reporting

Learning Integration
------------------

The ACE bridge provides:

- Continuous learning from workflow execution
- Skill acquisition and retention
- Performance improvement over time
- Knowledge transfer between workflows

Monitoring and Reporting
-----------------------

The system provides:

- Real-time workflow status tracking
- Progress monitoring
- Performance metrics
- Error reports
- Resource utilization tracking

Best Practices
-------------

1. Always use meaningful problem statements
2. Design agents with clear, specific roles
3. Define tasks with clear expected outputs
4. Monitor workflow execution regularly
5. Use appropriate execution methods for your use case
6. Regularly clean up old workflow states
7. Implement proper error handling in your applications

Troubleshooting
--------------

Common issues and solutions:

1. "CrewAI not available" - Install the crewai package
2. "State directory not writable" - Check permissions on state_storage_dir
3. "Workflow execution failed" - Check LLM connectivity and credentials
4. "Memory issues" - Reduce the number of concurrent workflows

Migration Notes
--------------

This integration replaces the previous Hephaestus-based system with:
- MIT-licensed CrewAI instead of AGPL-licensed Hephaestus
- Local state management instead of database-backed storage
- Improved error handling and recovery
- Better integration with ACE learning system

Performance Considerations
-------------------------

- Workflow execution time depends on LLM response times
- State persistence adds overhead but provides reliability
- Concurrent workflows may compete for LLM resources
- Large workflows may require significant memory

Security Considerations
---------------------

- All state data is stored locally
- No external services for state management
- Input validation is performed on all data
- Error messages don't expose sensitive information

Future Enhancements
------------------

Planned improvements include:
- Advanced workflow visualization
- More sophisticated error correction strategies
- Enhanced monitoring dashboards
- Integration with additional AI models
- Workflow templating and reuse
"""

# This file serves as documentation for the CrewAI integration
# Actual implementation is in crewai_integration_complete.py