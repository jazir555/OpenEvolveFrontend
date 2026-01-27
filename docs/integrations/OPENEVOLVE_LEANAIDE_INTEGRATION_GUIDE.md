# OpenEvolve-LeanAIDE Integration Guide

## Comprehensive Guide to Autoformalization of OpenEvolve Evolved Workflows

This guide provides a complete overview of the OpenEvolve-LeanAIDE integration system, which enables automatic formalization of mathematical problems within OpenEvolve's decomposition and evolution workflows.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Integration Components](#integration-components)
4. [Workflow Autoformalization Pipeline](#workflow-autoformalization-pipeline)
5. [Integration with OpenEvolve Stages](#integration-with-openevolve-stages)
6. [Configuration and Setup](#configuration-and-setup)
7. [Usage Examples](#usage-examples)
8. [Advanced Features](#advanced-features)
9. [Error Handling and Fallbacks](#error-handling-and-fallbacks)
10. [Monitoring and Reporting](#monitoring-and-reporting)
11. [Troubleshooting](#troubleshooting)

## Introduction

The OpenEvolve-LeanAIDE integration system provides a powerful bridge between OpenEvolve's AI-driven decomposition and evolution workflows and LeanAIDE's formal verification capabilities. This integration enables:

- **Automatic Detection**: Identifies mathematical problems within workflows
- **Autoformalization**: Converts natural language mathematical statements to formal Lean 4 code
- **Formal Verification**: Verifies solutions using Lean 4 theorem prover
- **Confidence-Based Decision Making**: Uses confidence scores to guide workflow decisions
- **Comprehensive Reporting**: Provides detailed autoformalization reports

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        OpenEvolve-LeanAIDE Integration System                    │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌───────────────────┐  │
│  │  OpenEvolve         │    │  LeanAIDE           │    │  Integration     │  │
│  │  Workflow Engine    │    │  Autoformalization  │    │  Bridge          │  │
│  │                     │    │  Engine             │    │                   │  │
│  └─────────────────────┘    └─────────────────────┘    └───────────────────┘  │
│              ▲                          ▲                          ▲                  │
│              │                          │                          │                  │
│              ▼                          ▼                          ▼                  │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌───────────────────┐  │
│  │  Enhanced           │    │  Autoformalization  │    │  Autoformalization │  │
│  │  Workflow Stages    │    │  Strategies         │    │  Results          │  │
│  │                     │    │                     │    │                   │  │
│  └─────────────────────┘    └─────────────────────┘    └───────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Integration Components

### 1. OpenEvolveLeanAideBridge

The core bridge component that connects OpenEvolve workflows with LeanAIDE autoformalization:

- **Autoformalization Engine**: Manages LeanAIDE client and autoformalization strategies
- **Mathematical Detection**: Identifies mathematical problems and their domains
- **Strategy Recommendation**: Suggests optimal autoformalization strategies
- **Caching System**: Caches autoformalization results for performance
- **Error Handling**: Comprehensive error handling and fallback mechanisms

### 2. OpenEvolveLeanAideIntegrationSystem

The main integration system that extends OpenEvolve workflows:

- **Enhanced Workflow Stages**: Modified decomposition, evolution, and verification stages
- **Event Monitoring**: Hooks for workflow events (decomposition complete, solution generated, etc.)
- **Reporting System**: Comprehensive autoformalization reporting
- **Workflow Engine Integration**: Seamless integration with existing workflow engines

### 3. EnhancedWorkflowState

Extended workflow state with LeanAIDE support:

- **Autoformalization Results**: Stores results from each workflow stage
- **Mathematical Domains**: Tracks detected mathematical domains
- **Formal Verification Results**: Stores formal verification outcomes
- **Configuration**: Controls autoformalization behavior

## Workflow Autoformalization Pipeline

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                     Workflow Autoformalization Pipeline                          │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────────────────────────┐  │
│  │  Problem    │    │  Mathematical│    │  Strategy Selection              │  │
│  │  Input      │───▶│  Detection   │───▶│  (Domain + Stage Analysis)       │  │
│  └─────────────┘    └─────────────┘    └───────────────────────────────────┘  │
│              ▲                          ▲                          ▲                  │
│              │                          │                          │                  │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────────────────────────┐  │
│  │  Cache      │    │  Auto-      │    │  LeanAIDE Autoformalization       │  │
│  │  Check      │◀───┤  formalize  │◀───┤  (MDAP/MAKER/Hybrid/Adaptive)    │  │
│  └─────────────┘    └─────────────┘    └───────────────────────────────────┘  │
│              ▲                          ▲                          ▲                  │
│              │                          │                          │                  │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────────────────────────┐  │
│  │  Result     │    │  Formal     │    │  Formal Verification              │  │
│  │  Caching    │◀───┤  Verification│◀───┤  (Lean 4 Theorem Prover)         │  │
│  └─────────────┘    └─────────────┘    └───────────────────────────────────┘  │
│              ▲                          ▲                          ▲                  │
│              │                          │                          │                  │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────────────────────────┐  │
│  │  Workflow   │    │  Confidence │    │  Decision Making                  │  │
│  │  State      │◀───┤  Analysis   │◀───┤  (Accept/Reject/Refine)           │  │
│  │  Update     │    │             │    │                               │  │
│  └─────────────┘    └─────────────┘    └───────────────────────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Integration with OpenEvolve Stages

### 1. Decomposition Stage

**Enhanced Process:**
- Standard decomposition creates subproblems
- LeanAIDE autoformalization analyzes each subproblem
- Mathematical subproblems get formalized versions
- Formal verification methods are added to metadata

**Code Example:**
```python
# Enhanced decomposition with autoformalization
decomposition_plan = await integration_system.enhanced_decompose_problem(
    workflow_state, 
    "Prove that for all n, the sum of first n odd numbers is n²"
)
```

### 2. Solution Generation Stage

**Enhanced Process:**
- Standard evolution generates solution attempts
- LeanAIDE autoformalization creates formal versions
- Formalized solutions are verified using Lean 4
- Verification results are stored in metadata

**Code Example:**
```python
# Enhanced solution generation with autoformalization
solution_attempt = await integration_system.enhanced_generate_solution_attempt(
    workflow_state, 
    subproblem
)
```

### 3. Verification Stage

**Enhanced Process:**
- Standard verification checks solution quality
- LeanAIDE formal verification validates mathematical correctness
- Hybrid verification combines both approaches
- Confidence scores are updated based on formal verification

**Code Example:**
```python
# Enhanced verification with formal verification
verification_report = await integration_system.enhanced_verify_solution_attempt(
    workflow_state, 
    solution_attempt, 
    subproblem
)
```

## Configuration and Setup

### Basic Configuration

```python
from openevolve_leanaide_integration_system import get_openevolve_leanaide_integration
from workflow_engine import WorkflowEngine

# Initialize the integration system
integration_system = get_openevolve_leanaide_integration()

# Or with an existing workflow engine
workflow_engine = WorkflowEngine()
integration_system = get_openevolve_leanaide_integration(workflow_engine)
```

### Advanced Configuration

```python
from openevolve_leanaide_bridge import OpenEvolveLeanAideConfig
from leanaide_client import LeanAideConfig

# Create custom configuration
leanaide_config = LeanAideConfig(
    host="leanaide-server.example.com",
    port=7654,
    timeout=300.0
)

bridge_config = OpenEvolveLeanAideConfig(
    autoformalization_enabled=True,
    auto_detect_math_problems=True,
    default_strategy=AutoformalizationStrategy.HYBRID,
    min_confidence_for_verification=0.85,
    enable_caching=True,
    cache_ttl_seconds=7200,  # 2 hours
    leanaide_config=leanaide_config
)

# Initialize with custom configuration
integration_system = get_openevolve_leanaide_integration()
integration_system.leanaide_bridge = OpenEvolveLeanAideBridge(bridge_config)
```

## Usage Examples

### Basic Workflow Integration

```python
async def run_math_workflow(problem_statement: str):
    # Initialize integration system
    integration_system = get_openevolve_leanaide_integration()
    
    # Create enhanced workflow state
    workflow_state = integration_system.create_enhanced_workflow_state(problem_statement)
    
    # Run enhanced workflow
    result = await integration_system.run_enhanced_workflow(problem_statement)
    
    return result

# Example usage
result = await run_math_workflow("Prove that 2 + 2 = 4")
```

### Stage-by-Stage Integration

```python
async def run_stage_by_stage_workflow(problem_statement: str):
    integration_system = get_openevolve_leanaide_integration()
    workflow_state = integration_system.create_enhanced_workflow_state(problem_statement)
    
    # Stage 1: Decomposition
    decomposition_plan = await integration_system.enhanced_decompose_problem(
        workflow_state, problem_statement
    )
    
    # Stage 2: Process subproblems
    for subproblem in decomposition_plan.subproblems:
        solution_attempt = await integration_system.enhanced_generate_solution_attempt(
            workflow_state, subproblem
        )
        
        verification_report = await integration_system.enhanced_verify_solution_attempt(
            workflow_state, solution_attempt, subproblem
        )
        
    # Stage 3: Generate report
    report = await integration_system.create_comprehensive_autoformalization_report(workflow_state)
    
    return {
        'decomposition_plan': decomposition_plan,
        'solution_attempts': workflow_state.solution_attempts,
        'verification_reports': workflow_state.verification_reports,
        'autoformalization_report': report
    }
```

### Integration with Existing Workflow Engine

```python
# Integrate with existing workflow engine
workflow_engine = WorkflowEngine()
integration_system = get_openevolve_leanaide_integration(workflow_engine)

# This automatically enhances the workflow engine methods:
# - workflow_engine.decompose_problem (now with autoformalization)
# - workflow_engine.generate_solution_attempt (now with autoformalization)
# - workflow_engine.verify_solution_attempt (now with formal verification)

# Use the enhanced workflow engine normally
workflow_state = WorkflowState(problem_statement="Prove that...", ...)
decomposition_plan = await workflow_engine.decompose_problem(workflow_state, problem_statement)
```

## Advanced Features

### Strategy Recommendation

```python
# Get strategy recommendation based on problem and stage
problem = "Prove by induction that the sum of first n natural numbers is n(n+1)/2"
stage = "decomposition"

recommended_strategy = integration_system.get_autoformalization_strategy_recommendation(
    problem, stage
)

print(f"Recommended strategy: {recommended_strategy}")
```

### Event Monitoring

```python
# Create monitoring hooks
hooks = integration_system.create_workflow_monitoring_hooks()

# Use hooks in your workflow
async def on_decomposition_complete(workflow_state, decomposition_plan):
    if 'on_decomposition_complete' in hooks:
        enhanced_plan = await hooks['on_decomposition_complete'](workflow_state, decomposition_plan)
        return enhanced_plan
    return decomposition_plan
```

### Confidence-Based Decision Making

```python
# Use confidence scores for decision making
async def make_verification_decision(verification_report):
    leanaide_verification = verification_report.leanaide_verification
    
    if leanaide_verification and leanaide_verification['success']:
        confidence = leanaide_verification['confidence_score']
        
        if confidence >= 0.9:
            return "ACCEPT_WITH_HIGH_CONFIDENCE"
        elif confidence >= 0.7:
            return "ACCEPT_WITH_REVIEW"
        else:
            return "REQUIRE_REFINEMENT"
    
    return "USE_STANDARD_VERIFICATION"
```

## Error Handling and Fallbacks

### Graceful Fallback Mechanism

The system provides comprehensive error handling:

```python
# Example of fallback behavior
try:
    result = await integration_system.autoformalize_and_verify_workflow(
        workflow_state, 
        problem_statement
    )
    
    if result['status'] == 'completed':
        # Use formal verification results
        pass
    elif result['status'] == 'autoformalization_failed':
        # Fallback to standard workflow
        standard_result = await standard_workflow(problem_statement)
        pass
    elif result['status'] == 'partial':
        # Partial success - use what's available
        pass
        
except Exception as e:
    logger.error(f"Integration error: {e}")
    # Fallback to completely standard workflow
```

### Configuration-Based Fallbacks

```python
# Configure fallback behavior
config = OpenEvolveLeanAideConfig(
    fallback_to_standard_workflow=True,  # Enable fallback
    hybrid_verification_enabled=True,     # Use hybrid verification when possible
    min_confidence_for_verification=0.7   # Minimum confidence threshold
)
```

## Monitoring and Reporting

### Comprehensive Autoformalization Report

```python
# Generate comprehensive report
report = await integration_system.create_comprehensive_autoformalization_report(workflow_state)

# Report structure
{
    'workflow_id': 'workflow_1234567890',
    'original_problem': 'Prove that...',
    'stages_processed': ['decomposition', 'solution_attempt', 'verification'],
    'successful_stages': ['decomposition', 'solution_attempt'],
    'failed_stages': ['verification'],
    'overall_success': False,
    'highest_confidence': 0.85,
    'mathematical_domains': ['algebra', 'number_theory'],
    'strategies_used': ['HYBRID', 'MAKER'],
    'execution_times': {
        'decomposition': 12.45,
        'solution_attempt': 8.72,
        'verification': 15.33
    },
    'errors': ['Verification failed: timeout exceeded'],
    'warnings': ['Low confidence in subproblem 3']
}
```

### Real-time Monitoring

```python
# Monitor workflow progress
hooks = integration_system.create_workflow_monitoring_hooks()

# Track events
workflow_events = []

async def track_event(event_name, *args):
    if event_name in hooks:
        result = await hooks[event_name](*args)
        workflow_events.append({
            'event': event_name,
            'timestamp': time.time(),
            'result': result
        })
        return result
    return args[0] if args else None
```

## Troubleshooting

### Common Issues and Solutions

**Issue: LeanAIDE service not available**
- **Cause**: LeanAIDE server is not running or misconfigured
- **Solution**: Check server configuration and connection

**Issue: Autoformalization fails for mathematical problems**
- **Cause**: Problem complexity exceeds current capabilities
- **Solution**: Try different strategies or simplify the problem

**Issue: Low confidence scores**
- **Cause**: Problem is at the edge of system capabilities
- **Solution**: Use hybrid verification or manual review

**Issue: Performance issues**
- **Cause**: Complex problems require significant computation
- **Solution**: Enable caching, adjust timeouts, or use simpler strategies

### Debugging Tips

```python
# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Check system availability
print(f"OpenEvolve available: {OPENEVOLVE_AVAILABLE}")
print(f"LeanAIDE bridge available: {LEANAIDE_BRIDGE_AVAILABLE}")
print(f"LeanAIDE autoformalization available: {LEANAIDE_AUTOFORMALIZATION_AVAILABLE}")

# Test basic functionality
problem = "Prove that 1 + 1 = 2"
is_math = integration_system.leanaide_bridge.is_mathematical_problem(problem)
domain = integration_system.leanaide_bridge.detect_mathematical_domain(problem)

print(f"Is mathematical: {is_math}")
print(f"Domain: {domain}")
```

## Best Practices

### 1. Start with Simple Problems
Begin with straightforward mathematical problems to validate the integration before moving to complex ones.

### 2. Use Adaptive Strategy
The adaptive strategy automatically selects the best approach based on problem characteristics.

### 3. Monitor Confidence Scores
Use confidence scores to guide decision making and identify problems that need manual review.

### 4. Enable Caching
Caching significantly improves performance for repeated problems or similar subproblems.

### 5. Configure Appropriate Timeouts
Set timeouts based on problem complexity and available resources.

### 6. Use Hybrid Verification
Combine formal verification with standard verification for robust results.

### 7. Monitor and Analyze Reports
Regularly review autoformalization reports to identify patterns and improvement opportunities.

## Integration Benefits

1. **Rigorous Verification**: Mathematical problems are formally verified using Lean 4
2. **Automated Formalization**: Natural language problems are automatically converted to formal code
3. **Confidence-Based Decisions**: Workflow decisions are guided by formal verification confidence
4. **Comprehensive Reporting**: Detailed reports provide insights into the formalization process
5. **Seamless Integration**: Works with existing OpenEvolve workflows without major modifications
6. **Performance Optimization**: Caching and strategy selection optimize resource usage

## Future Enhancements

The OpenEvolve-LeanAIDE integration system is designed for extensibility. Future enhancements may include:

- **Domain-Specific Optimization**: Specialized handling for different mathematical domains
- **Interactive Formalization**: Human-in-the-loop formalization for complex problems
- **Automated Theorem Proving**: Integration with automated theorem provers
- **Performance Benchmarking**: Comprehensive performance monitoring and optimization
- **Advanced Caching**: Machine learning-based cache optimization
- **Multi-Language Support**: Support for additional formalization targets

## Conclusion

The OpenEvolve-LeanAIDE integration system provides a powerful framework for autoformalization of mathematical problems within OpenEvolve workflows. By combining OpenEvolve's AI-driven decomposition and evolution capabilities with LeanAIDE's formal verification, the system enables rigorous, automated mathematical problem solving with comprehensive verification and reporting.

This integration represents a significant advancement in automated mathematical reasoning, providing a robust foundation for formal verification of AI-generated mathematical solutions.