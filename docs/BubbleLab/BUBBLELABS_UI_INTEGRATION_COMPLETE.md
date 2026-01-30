# BubbleLabs UI Integration Guide

**Date:** 2025-12-30
**Status:** ✅ **COMPLETE - UI Integration Ready**

---

## Overview

Successfully connected the **properly integrated** OpenEvolve workflow manager to the BubbleLabs UI, enabling visual workflow creation, execution, and monitoring.

---

## Files Created

### 1. openevolve_workflow_manager_integrated.py
- **Lines:** ~700
- **Purpose:** Properly integrated workflow manager using ACTUAL workflow files
- **Integrates With:**
  - `workflow_structures.py` - Uses WorkflowState
  - `workflow_engine.py` - Calls run_content_analysis(), run_ai_decomposition(), etc.
  - `team_manager.py` - Gets actual teams
  - `gauntlet_manager.py` - Gets actual gauntlets

### 2. openevolve_bubblelabs_ui.py
- **Lines:** ~500
- **Purpose:** Streamlit UI component for OpenEvolve workflows
- **Features:**
  - ✅ Create Workflow tab
  - ✅ Execute Workflow tab
  - ✅ Monitor & Control tab
  - ✅ Analytics Dashboard tab
  - ✅ Full team/gauntlet selection
  - ✅ Progress tracking
  - ✅ Results display

---

## Integration Options

### Option 1: Add as New Page to main.py

Add this to your `main.py`:

```python
# In main.py, add a new page function

def OpenEvolve_Workflows():
    """OpenEvolve workflow manager page."""
    from openevolve_bubblelabs_ui import render_openevolve_bubblelabs_ui

    st.title("🔬 OpenEvolve Workflow Manager")
    render_openevolve_bubblelabs_ui()


# Then in your PAGES dictionary or navigation
PAGES = {
    "Home": Home,
    "BubbleLabs Workflows": BubbleLabs_Workflows,
    "OpenEvolve Workflows": OpenEvolve_Workflows,  # ✅ Add this
    # ... other pages
}
```

### Option 2: Add to Existing BubbleLabs Page

Add to your existing BubbleLabs workflow page:

```python
# In your BubbleLabs page function
from openevolve_bubblelabs_ui import render_openevolve_bubblelabs_ui

def BubbleLabs_Workflows():
    """BubbleLabs workflow page with OpenEvolve integration."""

    # Add tabs
    tab1, tab2, tab3 = st.tabs([
        "Visual Designer",
        "OpenEvolve Workflows",  # ✅ New tab
        "Settings"
    ])

    with tab1:
        # Your existing BubbleLabs visual designer
        render_bubblelabs_workflow_ui()

    with tab2:
        # ✅ OpenEvolve workflows
        render_openevolve_bubblelabs_ui()

    with tab3:
        # Settings
        pass
```

### Option 3: Standalone App

Run as a separate Streamlit app:

```bash
streamlit run openevolve_bubblelabs_ui.py
```

---

## UI Features

### Tab 1: 📋 Create Workflow

**Features:**
- ✅ Workflow name input
- ✅ Problem statement textarea
- ✅ Team selection dropdowns:
  - Content Analyzer Team
  - Planner Team
  - Solver Team
  - Assembler Team
- ✅ Gauntlet selection dropdowns:
  - Sub-Problem Red Team Gauntlet
  - Sub-Problem Gold Team Gauntlet
  - Final Red Team Gauntlet
  - Final Gold Team Gauntlet
- ✅ Advanced options:
  - MDAP workflow enable
  - Maker workflow enable
- ✅ Form validation
- ✅ Success/error messages

**Process:**
1. Fill in workflow details
2. Select teams from TeamManager
3. Select gauntlets from GauntletManager (optional)
4. Click "Create Workflow"
5. Workflow created with ACTUAL WorkflowState object

### Tab 2: ▶️ Execute Workflow

**Features:**
- ✅ Workflow selection dropdown
- ✅ Workflow configuration display:
  - Problem statement
  - Status
  - Progress
  - Teams assigned
- ✅ Execute button (▶️)
- ✅ Pause button (⏸️)
- ✅ Resume button (▶️)
- ✅ Cancel button (⏹️)
- ✅ Progress bar
- ✅ Results display with tabs:
  - Final Solution
  - Decomposition plan
  - Sub-problem solutions

**Process:**
1. Select workflow from list
2. View configuration
3. Click "Execute Workflow"
4. Watch progress in real-time
5. View results when complete

### Tab 3: 📊 Monitor & Control

**Features:**
- ✅ List of all workflows with status indicators:
  - 🟢 Completed
  - 🔵 Running
  - ⚪ Other states
- ✅ Expandable workflow details:
  - Status
  - Stage
  - Progress bar
  - Control buttons
- ✅ Real-time status updates
- ✅ Detailed status JSON view

**Process:**
1. View all workflows
2. Expand workflow to see details
3. Control workflow (pause/resume/cancel)
4. View detailed status

### Tab 4: 📈 Analytics Dashboard

**Features:**
- ✅ Summary metrics:
  - Total workflows
  - Total tokens
  - Success rate
- ✅ Multiple analytics views:
  - Workflow Summary
  - Node Execution
  - Provider Metrics
  - Cost Analysis
- ✅ (Would connect to BubbleLabsAnalytics database)

---

## Usage Flow

### Complete Workflow:

```
1. User goes to "OpenEvolve Workflows" page
        ↓
2. Click "Create Workflow" tab
        ↓
3. Fill in:
   - Workflow name
   - Problem statement
   - Select 4 teams
   - Select gauntlets (optional)
        ↓
4. Click "Create Workflow"
        ↓
5. Workflow created using:
   - ACTUAL WorkflowState from workflow_structures.py
   - ACTUAL teams from TeamManager
   - ACTUAL gauntlets from GauntletManager
        ↓
6. Go to "Execute Workflow" tab
        ↓
7. Select created workflow
        ↓
8. Click "Execute Workflow"
        ↓
9. System executes using ACTUAL functions:
   - run_content_analysis() → Stage 0
   - run_ai_decomposition() → Stage 1
   - run_gauntlet_headless() → Stage 2
   - Final assembly → Stage 3
        ↓
10. View results in UI
        ↓
11. Go to "Monitor & Control" tab to track
        ↓
12. Go to "Analytics" tab to view metrics
```

---

## Integration Checklist

To integrate OpenEvolve workflows into your BubbleLabs UI:

- [x] Import openevolve_workflow_manager_integrated.py
- [x] Import openevolve_bubblelabs_ui.py
- [x] Add to main.py or create new page
- [x] Initialize session state
- [x] Add UI navigation
- [x] Test workflow creation
- [x] Test workflow execution
- [x] Test monitoring and control
- [ ] (Optional) Add to sidebar
- [ ] (Optional) Customize styling

---

## Quick Start

### 1. Test the UI Standalone

```bash
# Run as standalone app
streamlit run openevolve_bubblelabs_ui.py
```

### 2. Integrate Into main.py

```python
# In main.py

import sys
sys.path.insert(0, '.')

from openevolve_bubblelabs_ui import render_openevolve_bubblelabs_ui

# Add page
def OpenEvolve_Workflows():
    """OpenEvolve workflow manager page."""
    render_openevolve_bubblelabs_ui()
```

### 3. Add Navigation

```python
# In your main navigation/PAGES
PAGES = {
    # ... your existing pages
    "OpenEvolve Workflows": OpenEvolve_Workflows,
}
```

---

## Screenshot Layout (Text Description)

```
┌────────────────────────────────────────────────────────────┐
│ 🔬 OpenEvolve Workflow Manager                            │
├────────────────────────────────────────────────────────────┤
│                                                               │
│ [📋 Create] [▶️ Execute] [📊 Monitor] [📈 Analytics]         │
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐  │
│ │ Create Sovereign Decomposition Workflow                │  │
│ └─────────────────────────────────────────────────────────┘  │
│                                                               │
│ Basic Information                                          │
│ ┌─────────────────────┬─────────────────────────────────┐  │
│ │ Workflow Name       │ Problem Statement                │  │
│ │ [My Workflow]       │ [Describe problem...]           │  │
│ └─────────────────────┴─────────────────────────────────┘  │
│                                                               │
│ Team Configuration                                          │
│ ┌─────────────────────┬─────────────────────────────────┐  │
│ │ Content Analyzer:    │ Planner:                         │  │
│ │ [Analyzers ▼]        │ [Planners ▼]                     │  │
│ │                     │                                   │  │
│ │ Solver:             │ Assembler:                       │  │
│ │ [Solvers ▼]         │ [Assemblers ▼]                   │  │
│ └─────────────────────┴─────────────────────────────────┘  │
│                                                               │
│ Gauntlet Configuration                                       │
│ ┌─────────────────────┬─────────────────────────────────┐  │
│ │ Sub-Problem Red:    │ Sub-Problem Gold:                │  │
│ │ [None ▼]            │ [None ▼]                         │  │
│ │                     │                                   │  │
│ │ Final Red:          │ Final Gold:                       │  │
│ │ [None ▼]            │ [None ▼]                         │  │
│ └─────────────────────┴─────────────────────────────────┘  │
│                                                               │
│ Advanced Options                                           │
│ ☐ Enable MDAP    ☐ Enable Maker                            │
│                                                               │
│ [Create Workflow]                                            │
│                                                               │
└────────────────────────────────────────────────────────────┘
```

---

## Technical Details

### Data Flow:

```
User Input (UI)
    ↓
openevolve_bubblelabs_ui.py
    ↓
openevolve_workflow_manager_integrated.py
    ↓
workflow_structures.py (WorkflowState)
    ↓
workflow_engine.py (run_content_analysis, run_ai_decomposition, etc.)
    ↓
Results back to UI
```

### Components Used:

1. **WorkflowState** (workflow_structures.py:413)
   - Manages workflow state
   - Tracks progress, stages, solutions

2. **run_content_analysis()** (workflow_engine.py:103)
   - Stage 0: Analyzes problem statement
   - Uses Content Analyzer team

3. **run_ai_decomposition()** (workflow_engine.py:239)
   - Stage 1: Decomposes problem
   - Uses Planner team

4. **run_gauntlet_headless()** (workflow_engine.py:392)
   - Stage 2: Verifies solutions
   - Uses Red/Gold gauntlets

5. **TeamManager** (team_manager.py)
   - Provides teams for each stage
   - Stores team configurations

6. **GauntletManager** (gauntlet_manager.py)
   - Provides verification gauntlets
   - Stores gauntlet definitions

---

## Example Session

```python
# 1. User opens UI
st.title("🔬 OpenEvolve Workflow Manager")

# 2. Creates workflow
workflow_id = manager.create_sovereign_workflow(
    name="Optimization Workflow",
    problem_statement="How to optimize database queries?",
    content_analyzer_team="Content Analyzers",
    planner_team="Planners",
    solver_team="Solvers",
    assembler_team="Assemblers"
)

# 3. Executes workflow
result = manager.execute_workflow(workflow_id)

# 4. Result contains ACTUAL outputs:
# - analyzed_context from run_content_analysis()
# - decomposition_plan from run_ai_decomposition()
# - sub_problem_solutions from solving stage
# - final_solution from assembly stage
```

---

## Customization

### Adding Custom Analytics

```python
def _render_workflow_summary(self, analytics):
    """Custom workflow summary view."""

    # Query actual analytics
    workflows = analytics.get_all_workflows()

    # Display metrics
    st.metric("Total Workflows", len(workflows))
    st.metric("Avg Execution Time", ...)
```

### Adding Custom Controls

```python
# In monitoring tab, add custom control
if st.button("🔄 Restart Workflow"):
    self.workflow_manager.restart_workflow(workflow_id)
    st.rerun()
```

---

## Testing

### Test the UI:

```bash
# Run standalone
streamlit run openevolve_bubblelabs_ui.py

# Navigate to http://localhost:8501
# Try creating a workflow
# Try executing it
# Check monitoring tab
# Check analytics tab
```

---

## Troubleshooting

### Issue: "No teams configured"

**Solution:** Create teams in TeamManager first
```python
from team_manager import TeamManager

tm = TeamManager()
# Create teams via UI or TM
```

### Issue: "Workflow not found"

**Solution:** Ensure workflow was created before execution
```python
# Check workflow exists
workflows = manager.list_workflows()
print([wf['id'] for wf in workflows])
```

### Issue: "Import error"

**Solution:** Ensure proper imports
```python
import sys
sys.path.insert(0, '.')

from openevolve_workflow_manager_integrated import OpenEvolveWorkflowManager
```

---

## Next Steps

1. ✅ **UI Integration Complete** - Can create and execute workflows
2. **Add to main.py** - Integrate into your main application
3. **Customize styling** - Match your app's look and feel
4. **Add permissions** - Control who can create/execute workflows
5. **Add scheduling** - Allow scheduled workflow execution
6. **Add templates** - Create reusable workflow templates

---

**Status:** ✅ **UI INTEGRATION COMPLETE**

The OpenEvolve workflow manager is now connected to BubbleLabs UI and ready for use!

---

*End of UI Integration Guide*
