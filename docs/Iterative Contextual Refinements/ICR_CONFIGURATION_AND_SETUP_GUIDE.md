# ICR Configuration and Setup Guide

**Version:** 1.0  
**Date:** 2026-02-02  
**Status:** Production Ready  
**Integration Status:** 95%+ Complete

---

## Table of Contents

1. [Overview](#1-overview)
2. [Features and Capabilities](#2-features-and-capabilities)
3. [Installation and Setup](#3-installation-and-setup)
4. [Configuration Options](#4-configuration-options)
5. [VLM Analysis Setup](#5-vlm-analysis-setup)
6. [API Endpoint Documentation](#6-api-endpoint-documentation)
7. [Usage Examples](#7-usage-examples)
8. [Best Practices](#8-best-practices)
9. [Troubleshooting](#9-troubleshooting)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Overview

### What is ICR?

**Iterative Contextual Refinements (ICR)** is a system-wide capability that enables continuous improvement of decomposition plans, solutions, and validation processes through contextual feedback loops. ICR creates a closed-loop system where all components learn and improve from accumulated execution experience.

### Key Innovation

```
Traditional Systems: One-shot generation → Validation → Output
ICR-Enhanced Systems: Generation → Validation → Refinement → Re-validation → Output
```

### Expected Benefits

| Benefit | Improvement |
|---------|-------------|
| Decomposition Quality | 15-25% improvement in quality scores |
| False Positive Reduction | 30-40% reduction in validation false positives |
| Resource Efficiency | 20-30% improvement in allocation efficiency |
| Learning | Continuous learning without retraining overhead |
| Self-Healing | Workflows adapt to failure patterns automatically |

### Integration Status

ICR integration is at **95%+ completion** across the codebase with comprehensive coverage of:

- ✅ Core refinement orchestration
- ✅ Blue/Red/Gold team integration
- ✅ Entanglement matrix for dependency tracking
- ✅ Meta-cognitive repair loops
- ✅ Knowledge graph linkage (ADR, Skillbook 2.0)
- ✅ Digital Twin Sandbox (Z3) integration
- ✅ API contract self-healing
- ✅ Agent fatigue monitoring
- ✅ RobustnessCoordinator integration
- ✅ BubbleLab nodes integration
- ✅ ROMA components integration
- ✅ Vision-Augmented UI heatmapping
- ✅ Multi-modal insight synthesis
- ✅ Auto-refine UI and configuration
- ✅ Reward calibration UI

---

## 2. Features and Capabilities

### 2.1 Core Features

#### Pattern Learning
- Stores execution patterns across multiple dimensions
- Learns from successful and failed operations
- Tracks pass rates and quality scores by context

#### Adaptive Thresholds
- Automatically adjusts quality thresholds based on historical patterns
- Predicts operation success/failure probability
- Enables self-tuning quality gates

#### Cross-Component Integration
- Unified pattern storage across all components
- Shared learning context
- Coordinated refinements across the system

#### Vision-Augmented Analysis
- VLM-powered UI heatmap interpretation
- Friction detection in user interfaces
- Layout and interaction pattern analysis

### 2.2 Component-Specific Features

#### QualityGateEngine
- Adaptive quality thresholds
- Content type and quality level pattern tracking
- Metric-specific pattern learning

#### SGDWorkflowOrchestrator
- Workflow pattern learning
- Team configuration optimization
- Gauntlet configuration refinement

#### RobustnessCoordinator
- Robustness pattern storage
- Operation outcome prediction
- Adaptive threshold adjustments

#### BubbleLab Nodes
- Execution, verification, routing, and research patterns
- Operation history tracking
- Node-specific adaptive thresholds

#### ROMA Components
- Atomizer: Atomization patterns and atom count tracking
- Executor: Execution patterns and tool usage tracking
- Planner: Planning patterns and complexity tracking
- Verifier: Verification patterns and goal type tracking
- Aggregator: Aggregation patterns and subtask count tracking

---

## 3. Installation and Setup

### 3.1 Prerequisites

- Python 3.9+
- FastAPI (for API endpoints)
- Pydantic v2 (for schemas)
- Required dependencies in your project's `requirements.txt`

### 3.2 Installation Steps

#### Step 1: Verify Dependencies

Ensure all required packages are installed:

```bash
pip install fastapi uvicorn pydantic
```

#### Step 2: Verify ICR Integration Files

The following files should exist in your codebase:

```
├── quality_gate_engine.py          # QualityGateEngine with ICR
├── sgd_workflow_orchestrator.py    # Workflow orchestrator with ICR
├── robustness_integration.py       # RobustnessCoordinator with ICR
├── bubblelabs_nodes/
│   ├── base_node.py               # Base node with ICR
│   ├── assembly_node.py           # Assembly node with ICR
│   ├── gauntlet_node.py           # Gauntlet node with ICR
│   └── verification_node.py       # Verification node with ICR
├── ROMA/src/roma_dspy/core/modules/
│   ├── atomizer.py                # Atomizer with ICR
│   ├── executor.py                # Executor with ICR
│   ├── planner.py                 # Planner with ICR
│   ├── verifier.py                # Verifier with ICR
│   └── aggregator.py              # Aggregator with ICR
├── vision_language_monitor.py      # VLM analysis support
├── api_server.py                   # API endpoints
└── api/gateway/models/
    └── icr_schemas.py             # Pydantic schemas
```

#### Step 3: Configure Environment Variables

Set up the required environment variables (see [Configuration Options](#4-configuration-options)):

```bash
# Enable ICR (default: true)
export ICR_ENABLED=true

# Enable VLM analysis (optional)
export ICR_VLM_ENABLED=true

# VLM Provider Configuration
export ICR_VLM_PROVIDER=openai
export ICR_VLM_MODEL=gpt-4o
export ICR_VLM_API_KEY=your_api_key_here
```

#### Step 4: Start the API Server

```bash
python api_server.py
```

The API server will be available at `http://localhost:8000` by default.

### 3.3 Verification

Verify ICR is working correctly:

```python
# Test ICR initialization
from quality_gate_engine import QualityGateEngine

engine = QualityGateEngine(enable_icr=True)
assert engine.enable_icr == True
assert engine.icr_pattern_store is not None
print("ICR initialized successfully!")
```

---

## 4. Configuration Options

### 4.1 Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ICR_ENABLED` | bool | `true` | Enable/disable ICR globally |
| `ICR_VLM_ENABLED` | bool | `false` | Enable VLM analysis for heatmaps |
| `ICR_VLM_PROVIDER` | string | `openai` | VLM provider (openai, anthropic, google, azure, mock) |
| `ICR_VLM_MODEL` | string | `gpt-4o` | Model name for VLM analysis |
| `ICR_VLM_API_KEY` | string | - | API key for VLM provider |
| `ICR_VLM_TEMPERATURE` | float | `0.2` | Temperature for VLM (0.0-1.0) |
| `ICR_VLM_MAX_TOKENS` | int | `1024` | Max tokens for VLM response |
| `ICR_VLM_BASE_URL` | string | - | Custom base URL for VLM API |
| `ICR_VLM_CACHE_ENABLED` | bool | `true` | Enable VLM response caching |
| `ICR_VLM_CACHE_TTL` | int | `3600` | Cache TTL in seconds |
| `ICR_VLM_TIMEOUT` | int | `30` | VLM request timeout in seconds |
| `ICR_API_BASE_URL` | string | - | Base URL for ICR API endpoints |

### 4.2 Code-Level Configuration

#### QualityGateEngine

```python
from quality_gate_engine import QualityGateEngine

engine = QualityGateEngine(
    threshold_manager=threshold_manager,
    enable_icr=True,  # Enable ICR
    icr_pattern_store=None,  # Use default pattern store
    max_patterns=500,  # Max patterns per type
    min_pattern_count=5,  # Min patterns for learning
    confidence_threshold=0.7  # Confidence threshold for refinements
)
```

#### SGDWorkflowOrchestrator

```python
from sgd_workflow_orchestrator import SGDWorkflowOrchestrator

orchestrator = SGDWorkflowOrchestrator(
    openevolve_api_base="http://localhost:8000",
    enable_icr=True,  # Enable ICR
    max_workflows=100,  # Max workflows in history
    max_patterns=500  # Max patterns per type
)
```

#### RobustnessCoordinator

```python
from robustness_integration import RobustnessCoordinator, RobustnessConfig

config = RobustnessConfig(
    enable_icr=True,  # Enable ICR
    max_patterns=500,
    min_pattern_count=5,
    confidence_threshold=0.7
)

coordinator = RobustnessCoordinator(config=config)
```

#### BubbleLab Nodes

```python
from bubblelabs_nodes.base_node import BubbleLabsNode

node_config = {
    'enable_icr': True,  # Enable ICR
    'max_patterns': 500,
    'min_pattern_count': 5,
    'confidence_threshold': 0.7
}

node = BubbleLabsNode(config=node_config)
```

#### ROMA Components

```python
from ROMA.src.roma_dspy.core.modules.planner import Planner

planner = Planner(
    model_name="gpt-4",
    enable_icr=True,  # Enable ICR
    icr_pattern_store=None,  # Use default pattern store
    max_patterns=500
)
```

### 4.3 Pattern Storage Limits

| Component | Default Max Patterns | Default History Size |
|-----------|---------------------|---------------------|
| QualityGateEngine | 500 | 500 |
| SGDWorkflowOrchestrator | 500 | 100 workflows |
| RobustnessCoordinator | 500 | 500 |
| BubbleLab Nodes | 500 | 500 |
| ROMA Modules | 500 | 500 |

---

## 5. VLM Analysis Setup

### 5.1 Supported VLM Providers

| Provider | Environment Variable | Default Model |
|----------|---------------------|---------------|
| OpenAI | `ICR_VLM_PROVIDER=openai` | `gpt-4o` |
| Anthropic | `ICR_VLM_PROVIDER=anthropic` | `claude-3-5-sonnet-20241022` |
| Google | `ICR_VLM_PROVIDER=google` | `gemini-pro-vision` |
| Azure | `ICR_VLM_PROVIDER=azure` | Configured via base URL |
| Mock | `ICR_VLM_PROVIDER=mock` | Mock responses |

### 5.2 OpenAI Setup

```bash
# Set environment variables
export ICR_VLM_ENABLED=true
export ICR_VLM_PROVIDER=openai
export ICR_VLM_MODEL=gpt-4o
export ICR_VLM_API_KEY=sk-your-openai-api-key
export ICR_VLM_TEMPERATURE=0.2
export ICR_VLM_MAX_TOKENS=1024
```

### 5.3 Anthropic Setup

```bash
# Set environment variables
export ICR_VLM_ENABLED=true
export ICR_VLM_PROVIDER=anthropic
export ICR_VLM_MODEL=claude-3-5-sonnet-20241022
export ICR_VLM_API_KEY=sk-ant-your-anthropic-api-key
export ICR_VLM_TEMPERATURE=0.2
export ICR_VLM_MAX_TOKENS=1024
```

### 5.4 Google Setup

```bash
# Set environment variables
export ICR_VLM_ENABLED=true
export ICR_VLM_PROVIDER=google
export ICR_VLM_MODEL=gemini-pro-vision
export ICR_VLM_API_KEY=your-google-api-key
export ICR_VLM_TEMPERATURE=0.2
export ICR_VLM_MAX_TOKENS=1024
```

### 5.5 Azure Setup

```bash
# Set environment variables
export ICR_VLM_ENABLED=true
export ICR_VLM_PROVIDER=azure
export ICR_VLM_MODEL=gpt-4-vision-preview
export ICR_VLM_API_KEY=your-azure-api-key
export ICR_VLM_BASE_URL=https://your-resource.openai.azure.com
export ICR_VLM_TEMPERATURE=0.2
export ICR_VLM_MAX_TOKENS=1024
```

### 5.6 VLM Analysis Types

| Analysis Type | Description |
|---------------|-------------|
| `layout_analysis` | Analyzes UI layout structure |
| `interaction_patterns` | Identifies interaction patterns |
| `friction_detection` | Detects user friction points |
| `heatmap_interpretation` | Interprets heatmap overlays |
| `comprehensive` | All of the above |

---

## 6. API Endpoint Documentation

### 6.1 ICR Events

#### Emit Refinement Needed Event

```http
POST /icr/events/refinement-needed
Content-Type: application/json

{
  "source": "quality_gate",
  "content_type": "code",
  "quality_level": "standard",
  "reason": "Threshold not met",
  "metrics": {
    "overall_score": 0.65,
    "threshold": 0.70
  },
  "context": {
    "problem_id": "123",
    "workflow_id": "456"
  }
}
```

**Response:**
```json
{
  "queued": true
}
```

#### Get Refinement Needed Events

```http
GET /icr/events/refinement-needed?limit=5
```

**Response:**
```json
[
  {
    "source": "quality_gate",
    "content_type": "code",
    "reason": "Threshold not met",
    "timestamp": "2026-02-02T00:00:00Z"
  }
]
```

### 6.2 Reward Calibration

#### Queue Reward Calibration Request

```http
POST /icr/reward-calibration/request
Content-Type: application/json

{
  "request_id": "req-123",
  "source": "blue_team",
  "workflow_id": "456",
  "options": [
    {"id": "A", "description": "Option A"},
    {"id": "B", "description": "Option B"}
  ],
  "context": {
    "problem_type": "design",
    "complexity": 7
  }
}
```

**Response:**
```json
{
  "queued": true,
  "request_id": "req-123"
}
```

#### Get Next Reward Calibration

```http
GET /icr/reward-calibration/next
```

**Response:**
```json
{
  "request_id": "req-123",
  "options": [
    {"id": "A", "description": "Option A"},
    {"id": "B", "description": "Option B"}
  ]
}
```

#### Submit Reward Calibration Response

```http
POST /icr/reward-calibration/respond
Content-Type: application/json

{
  "request_id": "req-123",
  "choice": "A",
  "confidence": 0.85,
  "reasoning": "Option A provides better coverage"
}
```

**Response:**
```json
{
  "received": true,
  "request_id": "req-123"
}
```

#### Get Reward Calibration Response

```http
GET /icr/reward-calibration/response/{request_id}
```

**Response:**
```json
{
  "request_id": "req-123",
  "choice": "A",
  "confidence": 0.85,
  "reasoning": "Option A provides better coverage",
  "timestamp": "2026-02-02T00:00:00Z"
}
```

### 6.3 Heatmap Snapshots

#### Submit Heatmap Snapshot

```http
POST /icr/heatmap/snapshot
Content-Type: application/json

{
  "snapshot_id": "snap-123",
  "workflow_id": "456",
  "dom_snapshot": "base64-encoded-dom",
  "heatmap_overlay": "base64-encoded-heatmap",
  "timestamp": "2026-02-02T00:00:00Z",
  "request_vlm_analysis": true,
  "analysis_type": "comprehensive"
}
```

**Response:**
```json
{
  "received": true,
  "snapshot_id": "snap-123",
  "vlm_analysis_requested": true
}
```

### 6.4 VLM Configuration

#### Get VLM Configuration

```http
GET /icr/vlm/config
```

**Response:**
```json
{
  "provider": "openai",
  "model": "gpt-4o",
  "temperature": 0.2,
  "max_tokens": 1024,
  "available": true,
  "enabled": true,
  "configured": true
}
```

---

## 7. Usage Examples

### 7.1 QualityGateEngine

#### Basic Usage

```python
from quality_gate_engine import QualityGateEngine
from threshold_manager import ThresholdManager

# Initialize threshold manager
threshold_manager = ThresholdManager()

# Create engine with ICR enabled
engine = QualityGateEngine(
    threshold_manager=threshold_manager,
    enable_icr=True
)

# Assess quality
assessments = engine.assess_quality(
    content="def hello():\n    return 'world'",
    content_type="code",
    quality_level="standard",
    complexity_score=5
)

# Get ICR statistics
stats = engine.get_icr_statistics()
print(f"ICR patterns stored: {stats['total_patterns']}")
print(f"Learning confidence: {stats['confidence']}")
```

#### Pattern Learning

```python
# Store a pattern for learning
engine.store_icr_pattern(
    assessments=assessments,
    report=report,
    solution_context={
        "content_type": "code",
        "quality_level": "standard",
        "complexity_score": 5
    }
)

# Apply adaptive threshold
adaptive_threshold = engine.adapt_threshold(
    threshold=0.70,
    content_type="code",
    quality_level="standard",
    complexity_score=5
)
```

### 7.2 SGDWorkflowOrchestrator

#### Basic Usage

```python
from sgd_workflow_orchestrator import SGDWorkflowOrchestrator

# Create orchestrator with ICR enabled
orchestrator = SGDWorkflowOrchestrator(
    openevolve_api_base="http://localhost:8000",
    enable_icr=True
)

# Execute workflow
result = orchestrator.execute_workflow(
    problem_statement="Design a scalable microservices architecture",
    team_config={
        'content_analyzer_team': 'expert_analysis_team',
        'planner_team': 'expert_planning_team',
        'solver_team': 'expert_solver_team'
    },
    gauntlet_config={
        'sub_problem_red_gauntlet': 'adaptive',
        'final_red_gauntlet': 'adaptive'
    }
)

# Get ICR statistics
stats = orchestrator.get_icr_statistics()
print(f"Workflows completed: {stats['workflows_completed']}")
print(f"Patterns learned: {stats['total_patterns']}")
```

### 7.3 RobustnessCoordinator

#### Basic Usage

```python
from robustness_integration import RobustnessCoordinator, RobustnessConfig

# Create configuration with ICR enabled
config = RobustnessConfig(
    enable_icr=True,
    max_patterns=500
)

# Create coordinator
coordinator = RobustnessCoordinator(config=config)

# Execute operation with ICR tracking
result = coordinator.execute_operation(
    operation="verification",
    context={
        "content_type": "code",
        "complexity": 7
    }
)

# Get predicted success probability
prediction = coordinator.predict_outcome(
    operation="verification",
    context={
        "content_type": "code",
        "complexity": 7
    }
)
print(f"Predicted success probability: {prediction['success_probability']}")
```

### 7.4 BubbleLab Nodes

#### Base Node

```python
from bubblelabs_nodes.base_node import BubbleLabsNode

# Create node with ICR enabled
node_config = {
    'enable_icr': True,
    'node_id': 'assembly-001'
}

node = BubbleLabsNode(config=node_config)

# Execute node operation
result = node.execute(
    input_data={"subproblems": [...]},
    context={"workflow_id": "456"}
)

# Get ICR statistics
stats = node.get_icr_statistics()
print(f"Execution patterns: {len(stats['execution_patterns'])}")
```

#### Assembly Node

```python
from bubblelabs_nodes.assembly_node import AssemblyNode

# Create assembly node with ICR
node = AssemblyNode(
    config={'enable_icr': True},
    node_id='assembly-001'
)

# Assemble solution
result = node.assemble(
    subproblem_solutions=[...],
    original_problem="..."
)

# ICR statistics included in result
if 'icr_statistics' in result:
    print(f"Assembly patterns: {result['icr_statistics']['assembly_patterns']}")
```

### 7.5 ROMA Components

#### Planner Module

```python
from ROMA.src.roma_dspy.core.modules.planner import Planner

# Create planner with ICR enabled
planner = Planner(
    model_name="gpt-4",
    enable_icr=True
)

# Plan task
result = planner.forward(
    goal="Implement a REST API for user management",
    context={"complexity": 7}
)

# Get ICR statistics
stats = planner.get_icr_statistics()
print(f"Planning patterns: {len(stats['planning_patterns'])}")
print(f"Average planning time: {stats['average_planning_time']}")
```

#### Executor Module

```python
from ROMA.src.roma_dspy.core.modules.executor import Executor

# Create executor with ICR enabled
executor = Executor(
    model_name="gpt-4",
    enable_icr=True
)

# Execute task
result = executor.forward(
    goal="Create a user model with validation",
    context={"language": "python"}
)

# Get ICR statistics
stats = executor.get_icr_statistics()
print(f"Execution patterns: {len(stats['execution_patterns'])}")
print(f"Tool usage patterns: {stats['tool_usage_patterns']}")
```

---

## 8. Best Practices

### 8.1 Configuration Best Practices

#### Enable ICR Gradually

```python
# Start with ICR disabled to establish baseline
engine = QualityGateEngine(enable_icr=False)

# Collect baseline metrics for a period

# Enable ICR after baseline established
engine = QualityGateEngine(enable_icr=True)
```

#### Set Appropriate Pattern Limits

```python
# For high-volume systems, increase pattern limits
engine = QualityGateEngine(
    enable_icr=True,
    max_patterns=1000,  # Increased from default 500
    min_pattern_count=10  # Higher threshold for learning
)
```

#### Configure Confidence Thresholds

```python
# Conservative approach for critical systems
engine = QualityGateEngine(
    enable_icr=True,
    confidence_threshold=0.9  # High confidence required
)

# Aggressive approach for experimentation
engine = QualityGateEngine(
    enable_icr=True,
    confidence_threshold=0.6  # Lower threshold for more refinements
)
```

### 8.2 VLM Analysis Best Practices

#### Use Mock Provider for Testing

```bash
# Use mock provider during development/testing
export ICR_VLM_PROVIDER=mock
```

#### Enable Caching for Production

```bash
# Enable caching to reduce API costs
export ICR_VLM_CACHE_ENABLED=true
export ICR_VLM_CACHE_TTL=3600  # 1 hour cache
```

#### Choose Appropriate Model

```bash
# For quick analysis (faster, cheaper)
export ICR_VLM_MODEL=gpt-4o-mini

# For detailed analysis (slower, more expensive)
export ICR_VLM_MODEL=gpt-4o
```

### 8.3 Pattern Management

#### Regular Pattern Cleanup

```python
# Clear old patterns periodically
engine.clear_icr_patterns()
```

#### Export Patterns for Analysis

```python
# Get all patterns for analysis
patterns = engine.get_all_patterns()

# Export to file
import json
with open('icr_patterns.json', 'w') as f:
    json.dump(patterns, f, indent=2)
```

#### Monitor Pattern Growth

```python
# Check pattern statistics
stats = engine.get_icr_statistics()

# Alert if patterns exceed limit
if stats['total_patterns'] > 450:  # 90% of default limit
    print("Warning: Approaching pattern storage limit")
```

### 8.4 Performance Optimization

#### Disable ICR for Non-Critical Paths

```python
# Disable ICR for quick, non-critical operations
result = engine.assess_quality(
    content="...",
    content_type="code",
    quality_level="standard",
    store_pattern=False  # Don't store pattern for this assessment
)
```

#### Use Asynchronous Pattern Storage

```python
# Store patterns asynchronously to avoid blocking
import asyncio

async def store_pattern_async(engine, pattern):
    await asyncio.to_thread(engine.store_icr_pattern, pattern)
```

---

## 9. Troubleshooting

### 9.1 Common Issues

#### Issue: ICR Not Learning Patterns

**Symptoms:**
- Pattern counts remain at zero
- No adaptive threshold adjustments
- ICR statistics show no learning

**Solutions:**

1. **Check ICR is enabled:**
```python
engine = QualityGateEngine(enable_icr=True)
assert engine.enable_icr == True
```

2. **Verify pattern storage is called:**
```python
# Ensure store_pattern=True when calling assessments
assessments = engine.assess_quality(
    content="...",
    content_type="code",
    quality_level="standard",
    store_pattern=True  # Must be True
)
```

3. **Check minimum pattern count:**
```python
# Learning may not start until min_pattern_count is reached
engine = QualityGateEngine(
    enable_icr=True,
    min_pattern_count=5  # Default
)
```

#### Issue: VLM Analysis Not Working

**Symptoms:**
- VLM analysis returns null
- Heatmap snapshots not analyzed
- VLM configuration shows "configured": false

**Solutions:**

1. **Check VLM is enabled:**
```bash
export ICR_VLM_ENABLED=true
```

2. **Verify API key is set:**
```bash
export ICR_VLM_API_KEY=your_api_key_here
```

3. **Check VLM configuration:**
```python
from vision_language_monitor import VisionLanguageMonitor

monitor = VisionLanguageMonitor()
config = monitor.get_config()
print(f"VLM configured: {monitor.is_configured()}")
```

4. **Use mock provider for testing:**
```bash
export ICR_VLM_PROVIDER=mock
```

#### Issue: Pattern Storage Limit Exceeded

**Symptoms:**
- Old patterns are being dropped
- Pattern counts not increasing
- Inconsistent learning behavior

**Solutions:**

1. **Increase pattern limit:**
```python
engine = QualityGateEngine(
    enable_icr=True,
    max_patterns=1000  # Increase from default 500
)
```

2. **Implement pattern pruning:**
```python
# Clear old patterns periodically
engine.clear_icr_patterns()
```

3. **Export and analyze patterns:**
```python
# Export patterns before clearing
patterns = engine.get_all_patterns()
# Analyze which patterns are most valuable
# Keep only valuable patterns
```

#### Issue: API Endpoints Not Responding

**Symptoms:**
- API calls return 404
- ICR events not being queued
- Reward calibration not working

**Solutions:**

1. **Check API server is running:**
```bash
python api_server.py
```

2. **Verify ICR endpoints are registered:**
```python
# Check api_server.py for ICR endpoint definitions
# Look for @app.post("/icr/...") decorators
```

3. **Check CORS configuration:**
```python
# Ensure CORS allows your origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 9.2 Debug Mode

#### Enable Debug Logging

```python
import logging

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logging.getLogger('quality_gate_engine').setLevel(logging.DEBUG)
```

#### Enable ICR Debug Mode

```python
# Add debug parameter to enable verbose ICR logging
engine = QualityGateEngine(
    enable_icr=True,
    debug=True  # Enable debug logging
)
```

### 9.3 Getting Help

If you encounter issues not covered here:

1. Check the [ICR Integration Summary](ICR_INTEGRATION_SUMMARY_2026-02-02.md)
2. Review the [Master Integration Guide](ITERATIVE_CONTEXTUAL_REFINEMENTS_MASTER_GUIDE.md)
3. Examine test files for usage examples:
   - `test_quality_gate_icr_integration.py`
   - `test_sgd_workflow_icr_integration.py`

---

## 10. Configuration Reference

### 10.1 ICR Pattern Types

| Pattern Type | Description | Used By |
|--------------|-------------|---------|
| `content_type` | Patterns by content type | QualityGateEngine |
| `quality_level` | Patterns by quality level | QualityGateEngine |
| `metric` | Patterns by individual metrics | QualityGateEngine |
| `workflow` | Workflow execution patterns | SGDWorkflowOrchestrator |
| `problem_type` | Patterns by problem type | SGDWorkflowOrchestrator |
| `complexity` | Patterns by complexity score | All components |
| `team_config` | Patterns by team configuration | SGDWorkflowOrchestrator |
| `gauntlet_config` | Patterns by gauntlet configuration | SGDWorkflowOrchestrator |
| `execution` | Code execution patterns | BubbleLab Nodes, ROMA Executor |
| `verification` | Verification patterns | BubbleLab Nodes, ROMA Verifier |
| `planning` | Planning patterns | ROMA Planner |
| `atomization` | Atomization patterns | ROMA Atomizer |
| `routing` | Routing patterns | BubbleLab Nodes |
| `research` | Research patterns | BubbleLab Nodes |

### 10.2 ICR Refinement Types

| Refinement Type | Description |
|-----------------|-------------|
| `threshold_adjustment` | Adjust quality thresholds |
| `parameter_tuning` | Tune algorithm parameters |
| `strategy_change` | Change execution strategy |
| `config_update` | Update component configuration |
| `prompt_refactor` | Refactor LLM prompts |
| `team_reconfiguration` | Reconfigure team assignments |
| `gauntlet_reconfiguration` | Reconfigure gauntlet rules |

### 10.3 ICR Status Values

| Status | Description |
|--------|-------------|
| `enabled` | ICR is active and learning |
| `disabled` | ICR is disabled |
| `learning` | Actively collecting patterns |
| `predicting` | Using patterns for predictions |
| `refining` | Applying refinements |

### 10.4 Quality Thresholds

| Quality Level | Default Threshold | Description |
|---------------|-------------------|-------------|
| `standard` | 0.70 | Standard quality requirement |
| `high` | 0.80 | High quality requirement |
| `premium` | 0.90 | Premium quality requirement |

### 10.5 Complexity Scores

| Score | Description |
|-------|-------------|
| 1-3 | Low complexity |
| 4-6 | Medium complexity |
| 7-10 | High complexity |

---

## Appendix A: Quick Reference

### Enable ICR

```python
# Component level
engine = QualityGateEngine(enable_icr=True)

# Environment level
export ICR_ENABLED=true
```

### Enable VLM Analysis

```bash
export ICR_VLM_ENABLED=true
export ICR_VLM_PROVIDER=openai
export ICR_VLM_API_KEY=your_key
```

### Get ICR Statistics

```python
stats = engine.get_icr_statistics()
print(stats)
```

### Clear Patterns

```python
engine.clear_icr_patterns()
```

### Export Patterns

```python
patterns = engine.get_all_patterns()
```

---

## Appendix B: Example Configuration Files

### .env File

```bash
# ICR Configuration
ICR_ENABLED=true
ICR_VLM_ENABLED=true

# VLM Provider Configuration
ICR_VLM_PROVIDER=openai
ICR_VLM_MODEL=gpt-4o
ICR_VLM_API_KEY=sk-your-api-key-here
ICR_VLM_TEMPERATURE=0.2
ICR_VLM_MAX_TOKENS=1024

# VLM Cache Configuration
ICR_VLM_CACHE_ENABLED=true
ICR_VLM_CACHE_TTL=3600
ICR_VLM_TIMEOUT=30

# API Configuration
ICR_API_BASE_URL=http://localhost:8000
```

### config.py File

```python
"""ICR Configuration"""

# ICR Settings
ICR_ENABLED = True
ICR_VLM_ENABLED = True

# Pattern Storage
MAX_PATTERNS = 500
MIN_PATTERN_COUNT = 5
CONFIDENCE_THRESHOLD = 0.7

# VLM Configuration
VLM_PROVIDER = "openai"
VLM_MODEL = "gpt-4o"
VLM_TEMPERATURE = 0.2
VLM_MAX_TOKENS = 1024
VLM_CACHE_ENABLED = True
VLM_CACHE_TTL = 3600
VLM_TIMEOUT = 30

# Quality Thresholds
QUALITY_THRESHOLDS = {
    "standard": 0.70,
    "high": 0.80,
    "premium": 0.90
}
```

---

**Document End**

For additional information, see:
- [ICR Integration Summary](ICR_INTEGRATION_SUMMARY_2026-02-02.md)
- [Master Integration Guide](ITERATIVE_CONTEXTUAL_REFINEMENTS_MASTER_GUIDE.md)
- [ICR Implementation Status](ICR_IMPLEMENTATION_STATUS.md)
