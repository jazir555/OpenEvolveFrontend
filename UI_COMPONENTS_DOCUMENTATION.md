# UI Components Documentation

## Overview

This document provides comprehensive documentation for the missing UI components that have been implemented for the OpenEvolve Decomposition Workflow system.

## Table of Contents

1. [Dependency Visualization](#dependency-visualization)
2. [Analytics Dashboard](#analytics-dashboard)
3. [Knowledge Base Interface](#knowledge-base-interface)
4. [Auto-Approval Configuration](#auto-approval-configuration)
5. [Batch Operations](#batch-operations)
6. [Real-time Monitoring](#real-time-monitoring)
7. [Workflow Templates](#workflow-templates)
8. [Shared Utilities](#shared-utilities)

---

## Dependency Visualization

### Purpose
Visualizes sub-problem dependencies in an interactive graph to help users understand workflow structure and identify issues.

### Usage

```python
from ui_components import render_dependency_graph

# Render dependency graph
render_dependency_graph(
    sub_problems=sub_problems_list,
    show_critical_path=True,
    highlight_circular=True
)
```

### Features

- **Interactive Graph**: Hover over nodes to see details, click to navigate
- **Circular Dependency Detection**: Automatically highlights circular dependencies in red
- **Critical Path Analysis**: Shows the longest path through the workflow
- **Status Indicators**: Color-coded nodes based on completion status
  - Green: Completed
  - Orange: In Progress / Critical Path
  - Gray: Pending
  - Red: Failed / Circular Dependency

### Integration

The dependency visualization is integrated into the Manual Review Panel via an expander:

```python
with st.expander("📊 Dependency Visualization", expanded=False):
    render_dependency_visualization(decomposition_plan)
```

---

## Analytics Dashboard

### Purpose
Provides comprehensive analytics about workflow performance, team effectiveness, and solution quality.

### Usage

```python
from ui_components import render_analytics_dashboard

# Render analytics dashboard
render_analytics_dashboard(
    workflow_history=workflow_history,
    time_range=("2025-01-01", "2025-10-21")
)
```

### Features

#### Overview Tab
- Total workflows executed
- Success rate percentage
- Average duration
- Total OpenEvolve API cost

#### Workflow Performance Tab
- Success rate trends over time
- Execution duration distribution
- OpenEvolve API usage charts
- Resource usage metrics

#### Team Performance Tab
- Success rate comparison by team
- Quality score comparison
- Resource efficiency metrics
- OpenEvolve solution quality per team

#### Gauntlet Effectiveness Tab
- Flaw detection rate by gauntlet
- Verification accuracy metrics
- False positive rate tracking

#### Solution Quality Tab
- Quality score trends over time
- Quality distribution histograms
- OpenEvolve fitness evolution charts

#### Knowledge Base Tab
- Total artifacts count
- Usage frequency statistics
- Effectiveness scores
- Artifact type distribution

#### Custom Reports Tab
- Metric selection interface
- Export to CSV, JSON, or Excel
- Downloadable reports

### OpenEvolve Integration

The analytics dashboard tracks OpenEvolve-specific metrics:
- API calls and tokens used
- Evolution iterations
- Cost tracking
- Solution quality scores

---

## Knowledge Base Interface

### Purpose
Browse, search, and manage knowledge artifacts extracted from workflow executions.

### Usage

```python
from ui_components import render_knowledge_base_interface
from knowledge_manager import KnowledgeManager

km = KnowledgeManager()
render_knowledge_base_interface(km)
```

### Features

#### Browse Artifacts Tab
- Searchable artifact list with pagination
- Filter by type (pattern, solution, error, best_practice)
- Detailed artifact view with metadata
- CRUD operations (Create, Read, Update, Delete)
- Export individual artifacts as JSON

#### Knowledge Graph Tab
- Interactive network visualization
- Shows relationships between artifacts
- Color-coded by artifact type
- Node size indicates usage count

#### Learning Configuration Tab
- Configure knowledge extraction options
- Set usage policies
- Define minimum effectiveness thresholds
- Control auto-application of learned knowledge

### Artifact Structure

```python
{
    "id": "unique_id",
    "type": "pattern|solution|error|best_practice",
    "content": "artifact content",
    "domain": "domain name",
    "problem_type": "problem type",
    "usage_count": 0,
    "effectiveness_score": 0.0,
    "related_artifacts": []
}
```

---

## Auto-Approval Configuration

### Purpose
Configure automated approval rules for decomposition plans to reduce manual review overhead.

### Usage

```python
from ui_components import render_auto_approval_config

render_auto_approval_config()
```

### Features

#### Status Control
- Enable/disable auto-approval with a toggle
- Immediate configuration updates

#### Rule Management
- Create, edit, and delete rules
- Set rule priority (higher = evaluated first)
- Enable/disable individual rules

#### Rule Builder
- Visual rule builder interface
- Multiple condition support
- Logical operators (AND/OR)
- Field options:
  - Complexity
  - Confidence
  - Domain
  - Number of sub-problems
  - Team type

#### Rule Testing
- Test rules against sample plans
- Preview which plans would be auto-approved
- Validation before deployment

#### Audit Log
- Complete history of auto-approved decisions
- Filter by action and rule
- Timestamp and details for each decision

### Rule Structure

```python
{
    "name": "Simple Problems",
    "priority": 10,
    "action": "approve|reject|escalate",
    "enabled": True,
    "conditions": [
        {
            "field": "complexity",
            "operator": "<",
            "value": "3",
            "logical_op": "AND"
        }
    ]
}
```

---

## Batch Operations

### Purpose
Perform actions on multiple sub-problems simultaneously for efficient workflow management.

### Usage

Batch operations are integrated into the Manual Review Panel:

```python
with st.expander("🔄 Batch Operations", expanded=False):
    from batch_operations import render_batch_operations_ui
    sub_problems = render_batch_operations_ui(sub_problems)
```

### Features

#### Selection Management
- Multi-select checkboxes
- Filter by complexity, team status, gauntlet status
- Select all / deselect all
- Selection statistics

#### Batch Operations
- Assign solver team to multiple sub-problems
- Assign red gauntlet to multiple sub-problems
- Assign gold gauntlet to multiple sub-problems
- Update evolution mode
- Update complexity score
- Update content type

#### Preview and Confirmation
- Preview changes before applying
- Show before/after states
- Confirmation dialog
- Success/failure summary

#### Rollback
- Automatic state preservation
- One-click rollback
- Rollback within 5-second target

---

## Real-time Monitoring

### Purpose
Monitor workflow execution with detailed metrics and interactive controls.

### Usage

```python
from ui_components import render_enhanced_monitoring

render_enhanced_monitoring(
    workflow_state=current_workflow_state,
    resource_monitor=resource_monitor_instance
)
```

### Features

#### Status and Controls
- Current workflow status indicator
- Pause/Resume buttons
- Stop button with confirmation
- Manual refresh

#### Resource Usage Tab
- CPU and memory usage with progress bars
- OpenEvolve API metrics:
  - API calls
  - Tokens used
  - Cost
  - Evolution iterations
- API call limit tracking
- Resource usage charts over time

#### Performance Tab
- Execution progress bar
- Elapsed time
- Throughput (tasks/min)
- OpenEvolve evolution progress:
  - Generations
  - Fitness improvement
  - Convergence rate
- Performance timeline charts

#### Solution Quality Tab
- Average, min, max quality scores
- Quality distribution histogram
- OpenEvolve fitness evolution chart

#### Alerts Tab
- Real-time alert notifications
- Severity levels (error, warning, info)
- Filter by severity
- Clear all alerts
- Alert details expansion

#### Logs Tab
- Scrollable execution logs
- Filter by log level
- Search functionality
- Timestamp display
- Download logs as text file

---

## Workflow Templates

### Purpose
Save and reuse workflow configurations for common patterns.

### Usage

```python
from ui_components import render_workflow_templates
from template_manager import TemplateManager

tm = TemplateManager()
render_workflow_templates(tm, current_config)
```

### Features

#### Current Configuration
- View current workflow configuration
- Save as template button
- Load template button

#### Template Library
- Browse all saved templates
- Search by name, description, or tags
- Sort by name, date, or usage count
- Template cards with metadata
- Load, export, and delete operations

#### Save Template
- Template name and description
- Automatic OpenEvolve settings capture:
  - Model configuration
  - Evolution mode
  - Temperature
  - Max iterations
- Tag support for categorization
- Configuration preview

#### Load Template
- Template selection dropdown
- Template details display
- Validation before loading
- Usage count tracking

#### Import/Export
- Upload template JSON files
- Export individual templates
- Export all templates as ZIP
- Template format validation

### Template Structure

```json
{
  "id": "template_id",
  "name": "Simple Decomposition",
  "description": "Basic workflow for simple problems",
  "version": "1.0",
  "config": {
    "max_depth": 3,
    "teams": ["reasoning", "coding"],
    "gauntlets": ["basic_validation"],
    "auto_approval": true,
    "resource_limits": {
      "max_api_calls": 500,
      "timeout": 3600
    },
    "openevolve": {
      "model": "gpt-4",
      "evolution_mode": "standard",
      "temperature": 0.7,
      "max_iterations": 10
    }
  },
  "usage_count": 0,
  "created_at": "2025-10-21T12:00:00",
  "updated_at": "2025-10-21T12:00:00",
  "tags": ["simple", "fast"]
}
```

---

## Shared Utilities

### Purpose
Common utility functions used across all UI components.

### Location
`ui_utils.py`

### Key Functions

#### Error Handling

```python
from ui_utils import with_error_handling, safe_execute

@with_error_handling
def my_function():
    # Function code
    pass

result = safe_execute(
    lambda: risky_operation(),
    fallback=default_value,
    error_message="Operation failed"
)
```

#### Data Validation

```python
from ui_utils import validate_and_default, validate_required_fields

data = validate_and_default(
    data=user_input,
    validator=lambda x: x > 0,
    default_factory=lambda: 0
)

is_valid = validate_required_fields(
    data={"name": "test"},
    required_fields=["name", "email"]
)
```

#### Chart Rendering

```python
from ui_utils import render_chart_with_fallback

render_chart_with_fallback(
    chart_func=lambda data: st.plotly_chart(create_chart(data)),
    data=chart_data,
    fallback_func=lambda data: st.dataframe(data)
)
```

#### OpenEvolve Integration

```python
from ui_utils import display_openevolve_metrics, format_openevolve_config

display_openevolve_metrics({
    "api_calls": 150,
    "tokens": 50000,
    "cost": 1.50,
    "evolution_iterations": 10
})

config_str = format_openevolve_config(openevolve_config)
```

#### Session State Management

```python
from ui_utils import get_or_init_state, update_state, clear_state

value = get_or_init_state("my_key", lambda: [])
update_state("my_key", new_value)
clear_state("my_key")
```

#### Formatting

```python
from ui_utils import format_duration, format_number, format_percentage

duration_str = format_duration(3665)  # "1h 1m"
number_str = format_number(1500000)   # "1.50M"
percent_str = format_percentage(0.85) # "85.0%"
```

---

## Configuration

### UI Configuration
Location: `ui_config.py`

```python
UI_CONFIG = {
    "analytics": {
        "default_time_range": "7d",
        "max_data_points": 1000,
        "refresh_interval": 60
    },
    "knowledge_base": {
        "page_size": 20,
        "max_search_results": 100
    },
    "monitoring": {
        "update_interval": 2,
        "alert_threshold": 0.85
    }
}
```

### Feature Flags

```python
FEATURE_FLAGS = {
    "analytics_dashboard": True,
    "knowledge_base": True,
    "dependency_viz": True,
    "auto_approval_ui": True,
    "batch_operations": True,
    "enhanced_monitoring": True,
    "workflow_templates": True
}
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when importing UI components

**Solution**: Ensure all required files are in the same directory:
- `ui_components.py`
- `ui_utils.py`
- `ui_models.py`
- `ui_config.py`
- `template_manager.py`

#### 2. Session State Issues

**Problem**: Session state not persisting between reruns

**Solution**: Use `get_or_init_state()` utility:
```python
from ui_utils import get_or_init_state
value = get_or_init_state("key", lambda: default_value)
```

#### 3. Chart Rendering Errors

**Problem**: Charts not displaying or throwing errors

**Solution**: Use `render_chart_with_fallback()`:
```python
from ui_utils import render_chart_with_fallback
render_chart_with_fallback(chart_func, data, fallback_func)
```

#### 4. OpenEvolve Integration Issues

**Problem**: OpenEvolve metrics not displaying

**Solution**: Ensure workflow history includes OpenEvolve fields:
```python
workflow_data = {
    "openevolve_api_calls": 0,
    "openevolve_tokens": 0,
    "openevolve_cost": 0.0,
    "evolution_iterations": 0
}
```

---

## Performance Optimization

### Caching

Use Streamlit's caching for expensive operations:

```python
@st.cache_data(ttl=300)
def get_analytics_data(time_range):
    return compute_analytics(time_range)
```

### Lazy Loading

Load data only when needed:

```python
if st.session_state.get("show_analytics", False):
    render_analytics_dashboard(workflow_history)
```

### Pagination

Implement pagination for large datasets:

```python
page_size = 20
page = st.number_input("Page", min_value=1, max_value=total_pages)
start_idx = (page - 1) * page_size
end_idx = start_idx + page_size
display_items = items[start_idx:end_idx]
```

---

## Best Practices

1. **Always use error handling**: Wrap UI functions with `@with_error_handling`
2. **Validate user input**: Use validation utilities before processing
3. **Provide feedback**: Show success/error messages for user actions
4. **Use session state wisely**: Initialize with `get_or_init_state()`
5. **Optimize performance**: Cache expensive operations
6. **Follow naming conventions**: Use descriptive function names
7. **Document your code**: Add docstrings to all functions
8. **Test thoroughly**: Test with various data scenarios

---

## Version History

### Version 1.0.0 (2025-10-21)
- Initial implementation of all 7 missing UI components
- Dependency Visualization
- Analytics Dashboard
- Knowledge Base Interface
- Auto-Approval Configuration
- Batch Operations (already existed, documented)
- Real-time Monitoring Enhancements
- Workflow Templates
- Shared utilities and configuration
- Code deduplication analysis
- Comprehensive documentation

---

## Support

For issues or questions:
1. Check this documentation
2. Review the troubleshooting section
3. Check the code comments in source files
4. Review the design document in `.kiro/specs/missing-ui-components/design.md`

---

## License

This implementation is part of the OpenEvolve Decomposition Workflow system.
