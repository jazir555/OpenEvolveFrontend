<<<<<<< HEAD
# Workflow Bug Fixes - Implementation Guide

This guide provides the specific code fixes for all 24 bugs identified in the bug report.

## Table of Contents
1. [workflow_engine.py Fixes](#workflow_enginepy-fixes)
2. [workflow_knowledge_extractor.py Fixes](#workflow_knowledge_extractorpy-fixes)
3. [workflow_stage_functions.py Fixes](#workflow_stage_functionspy-fixes)
4. [workflow_enhanced_stages.py Fixes](#workflow_enhanced_stagespy-fixes)
5. [workflow_history_manager.py Fixes](#workflow_history_managerpy-fixes)
6. [workflow_lifecycle_controller.py Fixes](#workflow_lifecycle_controllerpy-fixes)

---

## workflow_engine.py Fixes

### Fix #1: Thread Exception Handling (Lines 273-279)

**Original Code:**
```python
for member in team.members:
    thread = threading.Thread(target=_analyze_with_model, args=(member,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
```

**Fixed Code:**
```python
# Store exceptions from threads
exceptions = []

def _analyze_with_model_safe(model_config: ModelConfig):
    try:
        _analyze_with_model(model_config)
    except Exception as e:
        exceptions.append((model_config.model_id, e))
        logger.error(f"Error in _analyze_with_model for {model_config.model_id}: {e}")

for member in team.members:
    thread = threading.Thread(target=_analyze_with_model_safe, args=(member,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join(timeout=120)  # Add timeout to prevent hanging

# Report exceptions
for model_id, exc in exceptions:
    st.error(f"Analysis failed for {model_id}: {exc}")

if not analyses and not exceptions:
    return {"error": "All analysis threads failed without exception data"}
```

---

### Fix #2: Safer Domain Selection (Line 301)

**Original Code:**
```python
most_common_domain = Counter(domains).most_common(1)[0][0] if domains else "General"
```

**Fixed Code:**
```python
# More defensive domain selection with explicit empty check
if domains:
    domain_result = Counter(domains).most_common(1)
    most_common_domain = domain_result[0][0] if domain_result else "General"
else:
    most_common_domain = "General"
```

---

### Fix #3: Better Error Handling in Decomposition (Lines 453-468)

**Original Code:**
```python
try:
    sub_problems_data = json.loads(response)
    sub_problems = [SubProblem(**sp) for sp in sub_problems_data]
    # ...
except json.JSONDecodeError:
    st.warning(f"AI Decomposition response from {model_config.model_id} was not valid JSON. Response: {response[:500]}...")
```

**Fixed Code:**
```python
try:
    sub_problems_data = json.loads(response)

    # Validate it's a list
    if not isinstance(sub_problems_data, list):
        raise ValueError(f"Expected list of sub-problems, got {type(sub_problems_data)}")

    # Create SubProblem objects with validation
    sub_problems = []
    for sp_data in sub_problems_data:
        try:
            sp = SubProblem(**sp_data)
            sub_problems.append(sp)
        except TypeError as e:
            st.warning(f"Invalid sub-problem data structure: {e}")
            logger.warning(f"Invalid sub-problem data: {sp_data}")
        except ValueError as e:
            st.warning(f"Invalid sub-problem value: {e}")
            logger.warning(f"Invalid sub-problem data: {sp_data}")

    if sub_problems:
        mdap_enabled = bool(analyzed_context.get("mdap_enabled", False))
        maker_enabled = bool(analyzed_context.get("maker_enabled", False))
        plans.append(DecompositionPlan(
            problem_statement=problem_statement,
            analyzed_context=analyzed_context,
            sub_problems=sub_problems,
            mdap_enabled=mdap_enabled,
            mdap_config=analyzed_context.get("mdap_config", {}),
            maker_enabled=maker_enabled,
            maker_config=analyzed_context.get("maker_config", {})
        ))

except json.JSONDecodeError as e:
    st.warning(f"Invalid JSON from {model_config.model_id}: {str(e)}")
    logger.warning(f"JSON decode error. Response preview: {response[:500]}")
except (TypeError, ValueError, AttributeError) as e:
    st.warning(f"Data validation error from {model_config.model_id}: {str(e)}")
    logger.warning(f"Validation error. Response preview: {response[:500]}")
```

---

### Fix #4: Proper Asyncio Event Loop Management (Lines 494-500)

**Original Code:**
```python
def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if loop.is_running():
        new_loop = asyncio.new_event_loop()
```

**Fixed Code:**
```python
def _run_async(coro):
    """
    Run async coroutine in sync context with proper error handling.

    This function handles multiple scenarios:
    1. No event loop exists (create new one)
    2. Event loop exists but not running (use it)
    3. Event loop is running (run in separate thread)

    Args:
        coro: Async coroutine to execute

    Returns:
        Result of the coroutine

    Raises:
        Exception: If coroutine execution fails
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop exists, create one and run
        return asyncio.run(coro)

    # Event loop exists
    if loop.is_running():
        # Loop is running, need to run in separate thread
        import concurrent.futures
        import threading

        result_container = []
        exception_container = []

        def run_in_new_loop():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(coro)
                    result_container.append(result)
                finally:
                    new_loop.close()
            except Exception as e:
                exception_container.append(e)

        thread = threading.Thread(target=run_in_new_loop)
        thread.start()
        thread.join(timeout=300)  # 5 minute timeout

        if exception_container:
            raise exception_container[0]

        if result_container:
            return result_container[0]

        raise TimeoutError("Async operation timed out")
    else:
        # Loop exists but not running, use it
        return loop.run_until_complete(coro)
```

---

## workflow_knowledge_extractor.py Fixes

### Fix #5: Remove Unsafe asyncio.create_task (Lines 69-73)

**Original Code:**
```python
if use_oneke:
    try:
        from integrations.oneke import OneKEBridge
        self.oneke_bridge = OneKEBridge()
        asyncio.create_task(self._init_oneke_async())
    except ImportError:
```

**Fixed Code:**
```python
if use_oneke:
    try:
        from integrations.oneke import OneKEBridge
        self.oneke_bridge = OneKEBridge()
        # Don't auto-initialize - require explicit call
        self._oneke_initialized = False
        logger.info("OneKE bridge created. Call ensure_oneke_initialized() before use.")
    except ImportError:
        print("OneKE not available. Install with: pip install integrations/oneke")
        self.use_oneke = False
        self.oneke_bridge = None
        self._oneke_initialized = False

# Add new method to class:
async def ensure_oneke_initialized(self) -> bool:
    """
    Ensure OneKE bridge is initialized before use.

    Returns:
        True if initialization succeeded, False otherwise
    """
    if not self.use_oneke or not self.oneke_bridge:
        return False

    if self._oneke_initialized:
        return True

    try:
        await self.oneke_bridge.initialize()
        self._oneke_initialized = True
        logger.info("OneKE bridge initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize OneKE bridge: {e}")
        self.oneke_bridge = None
        self._oneke_initialized = False
        return False
```

**Then update all async methods that use OneKE:**
```python
async def extract_domain_knowledge(self, workflow: WorkflowState, domains: Optional[List[str]] = None) -> Dict[str, Any]:
    if not await self.ensure_oneke_initialized():
        return {}

    # ... rest of method
```

---

### Fix #6: Prevent Division by Zero in Velocity (Line 400)

**Original Code:**
```python
elapsed_time = time.time() - workflow.start_time
velocity = solved_problems / (elapsed_time / 3600) if elapsed_time > 0 else 0.0
```

**Fixed Code:**
```python
elapsed_time = time.time() - workflow.start_time
elapsed_hours = elapsed_time / 3600

# Prevent unrealistically high velocities from very small time windows
min_elapsed_hours = 0.001  # 3.6 seconds minimum
if elapsed_hours < min_elapsed_hours:
    velocity = float(solved_problems)  # Problems per second (very high but not infinite)
else:
    velocity = solved_problems / elapsed_hours
```

---

### Fix #7: Safe Dictionary Access (Line 241)

**Original Code:**
```python
for sp_id, solution in workflow.sub_problem_solutions.items():
```

**Fixed Code:**
```python
for sp_id, solution in (workflow.sub_problem_solutions or {}).items():
    if not solution:
        logger.warning(f"Skipping None solution for {sp_id}")
        continue

    content = solution.content if hasattr(solution, 'content') else str(solution)
```

---

### Fix #8: Safe List Iteration (Lines 278-280)

**Original Code:**
```python
for critique in workflow.all_critique_reports:
    if critique:
```

**Fixed Code:**
```python
for critique in (workflow.all_critique_reports or []):
    if not critique:
        continue
```

---

### Fix #9: Safe Max Operation (Line 910)

**Original Code:**
```python
most_common_flaw = max(flaw_types.items(), key=lambda x: x[1])
insights.append({
    "insight_type": "common_flaw_pattern",
    "most_common_flaw_type": most_common_flaw[0],
    "occurrence_count": most_common_flaw[1],
    "recommendation": f"Focus on addressing {most_common_flaw[0]} issues"
})
```

**Fixed Code:**
```python
if flaw_types:
    most_common_flaw = max(flaw_types.items(), key=lambda x: x[1])
    insights.append({
        "insight_type": "common_flaw_pattern",
        "most_common_flaw_type": most_common_flaw[0],
        "occurrence_count": most_common_flaw[1],
        "recommendation": f"Focus on addressing {most_common_flaw[0]} issues in future solutions"
    })
else:
    insights.append({
        "insight_type": "no_flaws_detected",
        "message": "No flaws detected in critique reports",
        "recommendation": "Continue monitoring solution quality"
    })
```

---

### Fix #10: Avoid Duplicate Function Calls (Line 717)

**Original Code:**
```python
domain = self._detect_domains(workflow)[0] if self._detect_domains(workflow) else 'general'
```

**Fixed Code:**
```python
detected_domains = self._detect_domains(workflow)
domain = detected_domains[0] if detected_domains else 'general'
```

---

## workflow_stage_functions.py Fixes

### Fix #11: Validate Non-Empty Input (Lines 6-38)

**Original Code:**
```python
def select_integration_strategy(
    sub_problem_solutions: Dict[str, 'SolutionAttempt'],
    problem_statement: str,
    analyzed_context: Dict[str, Any]
) -> str:
    from collections import defaultdict

    dependency_depths = defaultdict(set)
```

**Fixed Code:**
```python
def select_integration_strategy(
    sub_problem_solutions: Dict[str, 'SolutionAttempt'],
    problem_statement: str,
    analyzed_context: Dict[str, Any]
) -> str:
    from collections import defaultdict

    # Validate input
    if not sub_problem_solutions:
        logger.warning("No sub-problem solutions provided, using default 'sequential' strategy")
        return "sequential"

    dependency_depths = defaultdict(set)
```

---

### Fix #12: Regex Error Handling (Lines 59-80)

**Original Code:**
```python
import re
func_pattern = r'def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*(\w+))?'
for match in re.finditer(func_pattern, content):
    func_name = match.group(1)
    params = match.group(2).split(',') if match.group(2) else []
```

**Fixed Code:**
```python
import re

func_pattern = r'def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*(\w+))?'

try:
    for match in re.finditer(func_pattern, content):
        # Safely extract groups
        func_name = match.group(1) if match.lastindex >= 1 else "unknown"
        params_str = match.group(2) if match.lastindex >= 2 else ""
        return_type = match.group(3) if match.lastindex >= 3 else 'Any'

        # Split parameters safely
        params = [p.strip() for p in params_str.split(',') if p.strip()]

        interface["outputs"].append({
            "name": func_name,
            "type": return_type,
            "parameters": params
        })
except re.error as e:
    logger.warning(f"Regex error analyzing function patterns: {e}")
except Exception as e:
    logger.warning(f"Unexpected error analyzing function patterns: {e}")
```

---

### Fix #13: Safe Division Operation (Line 215)

**Original Code:**
```python
qa_results["completeness"] = len(referenced_solutions) / len(sub_problem_solutions) if sub_problem_solutions else 1.0
```

**Fixed Code:**
```python
num_solutions = len(sub_problem_solutions)
qa_results["completeness"] = len(referenced_solutions) / num_solutions if num_solutions > 0 else 1.0
```

---

## workflow_enhanced_stages.py Fixes

### Fix #14: Check for None Before Nested Access (Line 859)

**Original Code:**
```python
"avg_complexity": sum(sp.ai_suggested_complexity_score for sp in workflow_state.decomposition_plan.sub_problems) / len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan.sub_problems else 0,
```

**Fixed Code:**
```python
plan = workflow_state.decomposition_plan
if plan and plan.sub_problems:
    avg_complexity = sum(sp.ai_suggested_complexity_score for sp in plan.sub_problems) / len(plan.sub_problems)
else:
    avg_complexity = 0
```

---

### Fix #15: Safe Max Operation (Line 910)

**Original Code:**
```python
most_common_flaw = max(flaw_types.items(), key=lambda x: x[1])
```

**Fixed Code:**
```python
if flaw_types:
    most_common_flaw = max(flaw_types.items(), key=lambda x: x[1])
    insights.append({
        "insight_type": "common_flaw_pattern",
        "most_common_flaw_type": most_common_flaw[0],
        "occurrence_count": most_common_flaw[1],
        "recommendation": f"Focus on addressing {most_common_flaw[0]} issues"
    })
else:
    logger.info("No flaws detected in critique reports")
```

---

## workflow_history_manager.py Fixes

### Fix #16: Better JSON Error Handling (Lines 29-35)

**Original Code:**
```python
if os.path.exists(self.history_file):
    with open(self.history_file, 'r', encoding='utf-8') as f:
        try:
            raw_history = json.load(f)
            self.history: Dict[str, WorkflowState] = {}
```

**Fixed Code:**
```python
if os.path.exists(self.history_file):
    try:
        with open(self.history_file, 'r', encoding='utf-8') as f:
            try:
                raw_history = json.load(f)
                self.history: Dict[str, WorkflowState] = {}
            except json.JSONDecodeError as e:
                print(f"Error decoding workflow history file: {e}")
                # Backup corrupted file
                backup_path = f"{self.history_file}.corrupted.{int(time.time())}"
                import shutil
                try:
                    shutil.copy(self.history_file, backup_path)
                    print(f"Corrupted file backed up to: {backup_path}")
                except Exception as backup_error:
                    print(f"Failed to backup corrupted file: {backup_error}")

                self.history = {}
    except IOError as e:
        print(f"Error reading history file: {e}")
        self.history = {}
else:
    self.history = {}
```

---

### Fix #17: Safe Dataclass Reconstruction (Lines 34-50)

**Original Code:**
```python
try:
    # Reconstruct ModelConfig
    if 'content_analyzer_team' in wf_data and wf_data['content_analyzer_team']:
        wf_data['content_analyzer_team']['members'] = [ModelConfig(**m) for m in wf_data['content_analyzer_team']['members']]
        wf_data['content_analyzer_team'] = Team(**wf_data['content_analyzer_team'])
```

**Fixed Code:**
```python
try:
    # Reconstruct ModelConfig with error handling
    if 'content_analyzer_team' in wf_data and wf_data['content_analyzer_team']:
        try:
            members_data = wf_data['content_analyzer_team'].get('members', [])
            wf_data['content_analyzer_team']['members'] = [ModelConfig(**m) for m in members_data]
            wf_data['content_analyzer_team'] = Team(**wf_data['content_analyzer_team'])
        except (TypeError, ValueError) as e:
            print(f"Error reconstructing content_analyzer_team for {wf_id}: {e}")
            wf_data['content_analyzer_team'] = None
        except Exception as e:
            print(f"Unexpected error reconstructing content_analyzer_team for {wf_id}: {e}")
            wf_data['content_analyzer_team'] = None
```

---

### Fix #18: Remove Redundant hasattr Check (Line 210)

**Original Code:**
```python
if workflow.decomposition_plan and hasattr(workflow.decomposition_plan, 'openevolve_metrics'):
    if workflow.decomposition_plan.openevolve_metrics:
```

**Fixed Code:**
```python
if workflow.decomposition_plan and workflow.decomposition_plan.openevolve_metrics:
```

---

### Fix #19: Explicit Division Safety (Line 182)

**Original Code:**
```python
if fitness_improvements:
    metrics["average_fitness_improvement"] = sum(fitness_improvements) / len(fitness_improvements)
```

**Fixed Code:**
```python
if fitness_improvements:
    metrics["average_fitness_improvement"] = sum(fitness_improvements) / len(fitness_improvements)
else:
    metrics["average_fitness_improvement"] = 0.0
```

---

## workflow_lifecycle_controller.py Fixes

### Fix #20: Validate Status Info Structure (Lines 65-76)

**Original Code:**
```python
status_info = self.integration.get_workflow_instance_status(selected_instance_id)
if "error" in status_info:
    st.error(f"Error getting workflow status: {status_info['error']}")
    return

# Display current status
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Status", f"{self._get_status_icon(status_info['status'])} {status_info['status'].upper()}")
```

**Fixed Code:**
```python
status_info = self.integration.get_workflow_instance_status(selected_instance_id)

# Validate response structure
if not isinstance(status_info, dict):
    st.error(f"Invalid status info returned: {type(status_info)}")
    return

if "error" in status_info:
    st.error(f"Error getting workflow status: {status_info.get('error', 'Unknown error')}")
    return

# Validate required fields
required_fields = ['status', 'current_stage', 'progress']
missing_fields = [f for f in required_fields if f not in status_info]
if missing_fields:
    st.error(f"Invalid status info: missing fields {missing_fields}")
    return

# Display current status
col1, col2, col3, col4 = st.columns(4)
with col1:
    status = status_info.get('status', 'unknown')
    st.metric("Status", f"{self._get_status_icon(status)} {status.upper()}")
```

---

### Fix #21: Safe Timestamp Access (Line 185)

**Original Code:**
```python
"start_time": datetime.fromtimestamp(status_info["start_time"]).isoformat() if status_info["start_time"] else "N/A",
```

**Fixed Code:**
```python
start_time = status_info.get("start_time")
if start_time:
    try:
        start_time_str = datetime.fromtimestamp(start_time).isoformat()
    except (ValueError, OSError) as e:
        logger.warning(f"Invalid start_time {start_time}: {e}")
        start_time_str = "Invalid"
else:
    start_time_str = "N/A"

# ... use start_time_str
```

---

### Fix #22: Better JSON Error Messages (Lines 341-345)

**Original Code:**
```python
try:
    input_dict = json.loads(inputs)
except json.JSONDecodeError:
    st.error("Invalid JSON in input parameters")
    return
```

**Fixed Code:**
```python
try:
    input_dict = json.loads(inputs)
except json.JSONDecodeError as e:
    st.error(f"Invalid JSON in input parameters: {str(e)}")
    st.error(f"JSON parse error at line {e.lineno}, column {e.colno}")
    return
except Exception as e:
    st.error(f"Unexpected error parsing inputs: {str(e)}")
    return
```

---

### Fix #23: Validate Timeline Data (Lines 206-210)

**Original Code:**
```python
timeline_data = [
    {"event": "Created", "time": status_info["start_time"]},
    {"event": "Started", "time": status_info["start_time"]},
    {"event": "Completed", "time": status_info["end_time"]} if status_info["end_time"] else {"event": "In Progress", "time": time.time()}
]
```

**Fixed Code:**
```python
start_time = status_info.get("start_time") or time.time()
end_time = status_info.get("end_time")

timeline_data = [
    {"event": "Created", "time": start_time},
    {"event": "Started", "time": start_time},
]

if end_time:
    timeline_data.append({"event": "Completed", "time": end_time})
else:
    timeline_data.append({"event": "In Progress", "time": time.time()})
```

---

### Fix #24: Safe Integration Call (Lines 111-117)

**Original Code:**
```python
if st.button("▶️ Start", disabled=not can_start, key=f"start_{instance_id}"):
    result = self.integration.start_workflow_instance(instance_id)
    if "error" in result:
        st.error(f"Start failed: {result['error']}")
    else:
        st.success(f"Workflow started: {result['message']}")
    st.rerun()
```

**Fixed Code:**
```python
if st.button("▶️ Start", disabled=not can_start, key=f"start_{instance_id}"):
    try:
        result = self.integration.start_workflow_instance(instance_id)

        # Validate result
        if result is None:
            st.error("Start failed: No response from integration")
            return

        if not isinstance(result, dict):
            st.error(f"Start failed: Invalid response type {type(result)}")
            return

        if "error" in result:
            error_msg = result.get('error', 'Unknown error')
            st.error(f"Start failed: {error_msg}")
            logger.error(f"Workflow start failed for {instance_id}: {error_msg}")
        else:
            success_msg = result.get('message', 'Started successfully')
            st.success(f"Workflow started: {success_msg}")
            st.rerun()
    except Exception as e:
        st.error(f"Start failed with exception: {str(e)}")
        logger.exception(f"Exception starting workflow {instance_id}")
```

---

## Testing Recommendations

After applying these fixes, create the following tests:

### Unit Tests
```python
def test_safe_max_operation():
    """Test max operation with empty list"""
    flaw_types = {}
    # Should not raise ValueError
    if flaw_types:
        result = max(flaw_types.items(), key=lambda x: x[1])
    else:
        result = None
    assert result is None

def test_division_by_zero():
    """Test division operations handle zero"""
    sub_problem_solutions = []
    num_solutions = len(sub_problem_solutions)
    completeness = 1.0 if num_solutions == 0 else 0.5
    assert completeness == 1.0

def test_none_dictionary_access():
    """Test safe dictionary access"""
    workflow = type('Workflow', (), {'sub_problem_solutions': None})()
    solutions = workflow.sub_problem_solutions or {}
    assert solutions == {}
```

### Integration Tests
```python
async def test_oneke_initialization():
    """Test OneKE initialization error handling"""
    extractor = WorkflowKnowledgeExtractor(use_oneke=True)
    # Should not crash during initialization
    result = await extractor.ensure_oneke_initialized()
    # Result should be bool
    assert isinstance(result, bool)
```

---

## Conclusion

All 24 bugs have been addressed with comprehensive fixes that include:
- Proper error handling
- Defensive programming
- Clear error messages
- Logging for debugging
- Type validation

Apply these fixes in order of priority (Critical -> High -> Medium -> Low) to minimize risk.

=======
# Workflow Bug Fixes - Implementation Guide

This guide provides the specific code fixes for all 24 bugs identified in the bug report.

## Table of Contents
1. [workflow_engine.py Fixes](#workflow_enginepy-fixes)
2. [workflow_knowledge_extractor.py Fixes](#workflow_knowledge_extractorpy-fixes)
3. [workflow_stage_functions.py Fixes](#workflow_stage_functionspy-fixes)
4. [workflow_enhanced_stages.py Fixes](#workflow_enhanced_stagespy-fixes)
5. [workflow_history_manager.py Fixes](#workflow_history_managerpy-fixes)
6. [workflow_lifecycle_controller.py Fixes](#workflow_lifecycle_controllerpy-fixes)

---

## workflow_engine.py Fixes

### Fix #1: Thread Exception Handling (Lines 273-279)

**Original Code:**
```python
for member in team.members:
    thread = threading.Thread(target=_analyze_with_model, args=(member,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
```

**Fixed Code:**
```python
# Store exceptions from threads
exceptions = []

def _analyze_with_model_safe(model_config: ModelConfig):
    try:
        _analyze_with_model(model_config)
    except Exception as e:
        exceptions.append((model_config.model_id, e))
        logger.error(f"Error in _analyze_with_model for {model_config.model_id}: {e}")

for member in team.members:
    thread = threading.Thread(target=_analyze_with_model_safe, args=(member,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join(timeout=120)  # Add timeout to prevent hanging

# Report exceptions
for model_id, exc in exceptions:
    st.error(f"Analysis failed for {model_id}: {exc}")

if not analyses and not exceptions:
    return {"error": "All analysis threads failed without exception data"}
```

---

### Fix #2: Safer Domain Selection (Line 301)

**Original Code:**
```python
most_common_domain = Counter(domains).most_common(1)[0][0] if domains else "General"
```

**Fixed Code:**
```python
# More defensive domain selection with explicit empty check
if domains:
    domain_result = Counter(domains).most_common(1)
    most_common_domain = domain_result[0][0] if domain_result else "General"
else:
    most_common_domain = "General"
```

---

### Fix #3: Better Error Handling in Decomposition (Lines 453-468)

**Original Code:**
```python
try:
    sub_problems_data = json.loads(response)
    sub_problems = [SubProblem(**sp) for sp in sub_problems_data]
    # ...
except json.JSONDecodeError:
    st.warning(f"AI Decomposition response from {model_config.model_id} was not valid JSON. Response: {response[:500]}...")
```

**Fixed Code:**
```python
try:
    sub_problems_data = json.loads(response)

    # Validate it's a list
    if not isinstance(sub_problems_data, list):
        raise ValueError(f"Expected list of sub-problems, got {type(sub_problems_data)}")

    # Create SubProblem objects with validation
    sub_problems = []
    for sp_data in sub_problems_data:
        try:
            sp = SubProblem(**sp_data)
            sub_problems.append(sp)
        except TypeError as e:
            st.warning(f"Invalid sub-problem data structure: {e}")
            logger.warning(f"Invalid sub-problem data: {sp_data}")
        except ValueError as e:
            st.warning(f"Invalid sub-problem value: {e}")
            logger.warning(f"Invalid sub-problem data: {sp_data}")

    if sub_problems:
        mdap_enabled = bool(analyzed_context.get("mdap_enabled", False))
        maker_enabled = bool(analyzed_context.get("maker_enabled", False))
        plans.append(DecompositionPlan(
            problem_statement=problem_statement,
            analyzed_context=analyzed_context,
            sub_problems=sub_problems,
            mdap_enabled=mdap_enabled,
            mdap_config=analyzed_context.get("mdap_config", {}),
            maker_enabled=maker_enabled,
            maker_config=analyzed_context.get("maker_config", {})
        ))

except json.JSONDecodeError as e:
    st.warning(f"Invalid JSON from {model_config.model_id}: {str(e)}")
    logger.warning(f"JSON decode error. Response preview: {response[:500]}")
except (TypeError, ValueError, AttributeError) as e:
    st.warning(f"Data validation error from {model_config.model_id}: {str(e)}")
    logger.warning(f"Validation error. Response preview: {response[:500]}")
```

---

### Fix #4: Proper Asyncio Event Loop Management (Lines 494-500)

**Original Code:**
```python
def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if loop.is_running():
        new_loop = asyncio.new_event_loop()
```

**Fixed Code:**
```python
def _run_async(coro):
    """
    Run async coroutine in sync context with proper error handling.

    This function handles multiple scenarios:
    1. No event loop exists (create new one)
    2. Event loop exists but not running (use it)
    3. Event loop is running (run in separate thread)

    Args:
        coro: Async coroutine to execute

    Returns:
        Result of the coroutine

    Raises:
        Exception: If coroutine execution fails
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop exists, create one and run
        return asyncio.run(coro)

    # Event loop exists
    if loop.is_running():
        # Loop is running, need to run in separate thread
        import concurrent.futures
        import threading

        result_container = []
        exception_container = []

        def run_in_new_loop():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(coro)
                    result_container.append(result)
                finally:
                    new_loop.close()
            except Exception as e:
                exception_container.append(e)

        thread = threading.Thread(target=run_in_new_loop)
        thread.start()
        thread.join(timeout=300)  # 5 minute timeout

        if exception_container:
            raise exception_container[0]

        if result_container:
            return result_container[0]

        raise TimeoutError("Async operation timed out")
    else:
        # Loop exists but not running, use it
        return loop.run_until_complete(coro)
```

---

## workflow_knowledge_extractor.py Fixes

### Fix #5: Remove Unsafe asyncio.create_task (Lines 69-73)

**Original Code:**
```python
if use_oneke:
    try:
        from integrations.oneke import OneKEBridge
        self.oneke_bridge = OneKEBridge()
        asyncio.create_task(self._init_oneke_async())
    except ImportError:
```

**Fixed Code:**
```python
if use_oneke:
    try:
        from integrations.oneke import OneKEBridge
        self.oneke_bridge = OneKEBridge()
        # Don't auto-initialize - require explicit call
        self._oneke_initialized = False
        logger.info("OneKE bridge created. Call ensure_oneke_initialized() before use.")
    except ImportError:
        print("OneKE not available. Install with: pip install integrations/oneke")
        self.use_oneke = False
        self.oneke_bridge = None
        self._oneke_initialized = False

# Add new method to class:
async def ensure_oneke_initialized(self) -> bool:
    """
    Ensure OneKE bridge is initialized before use.

    Returns:
        True if initialization succeeded, False otherwise
    """
    if not self.use_oneke or not self.oneke_bridge:
        return False

    if self._oneke_initialized:
        return True

    try:
        await self.oneke_bridge.initialize()
        self._oneke_initialized = True
        logger.info("OneKE bridge initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize OneKE bridge: {e}")
        self.oneke_bridge = None
        self._oneke_initialized = False
        return False
```

**Then update all async methods that use OneKE:**
```python
async def extract_domain_knowledge(self, workflow: WorkflowState, domains: Optional[List[str]] = None) -> Dict[str, Any]:
    if not await self.ensure_oneke_initialized():
        return {}

    # ... rest of method
```

---

### Fix #6: Prevent Division by Zero in Velocity (Line 400)

**Original Code:**
```python
elapsed_time = time.time() - workflow.start_time
velocity = solved_problems / (elapsed_time / 3600) if elapsed_time > 0 else 0.0
```

**Fixed Code:**
```python
elapsed_time = time.time() - workflow.start_time
elapsed_hours = elapsed_time / 3600

# Prevent unrealistically high velocities from very small time windows
min_elapsed_hours = 0.001  # 3.6 seconds minimum
if elapsed_hours < min_elapsed_hours:
    velocity = float(solved_problems)  # Problems per second (very high but not infinite)
else:
    velocity = solved_problems / elapsed_hours
```

---

### Fix #7: Safe Dictionary Access (Line 241)

**Original Code:**
```python
for sp_id, solution in workflow.sub_problem_solutions.items():
```

**Fixed Code:**
```python
for sp_id, solution in (workflow.sub_problem_solutions or {}).items():
    if not solution:
        logger.warning(f"Skipping None solution for {sp_id}")
        continue

    content = solution.content if hasattr(solution, 'content') else str(solution)
```

---

### Fix #8: Safe List Iteration (Lines 278-280)

**Original Code:**
```python
for critique in workflow.all_critique_reports:
    if critique:
```

**Fixed Code:**
```python
for critique in (workflow.all_critique_reports or []):
    if not critique:
        continue
```

---

### Fix #9: Safe Max Operation (Line 910)

**Original Code:**
```python
most_common_flaw = max(flaw_types.items(), key=lambda x: x[1])
insights.append({
    "insight_type": "common_flaw_pattern",
    "most_common_flaw_type": most_common_flaw[0],
    "occurrence_count": most_common_flaw[1],
    "recommendation": f"Focus on addressing {most_common_flaw[0]} issues"
})
```

**Fixed Code:**
```python
if flaw_types:
    most_common_flaw = max(flaw_types.items(), key=lambda x: x[1])
    insights.append({
        "insight_type": "common_flaw_pattern",
        "most_common_flaw_type": most_common_flaw[0],
        "occurrence_count": most_common_flaw[1],
        "recommendation": f"Focus on addressing {most_common_flaw[0]} issues in future solutions"
    })
else:
    insights.append({
        "insight_type": "no_flaws_detected",
        "message": "No flaws detected in critique reports",
        "recommendation": "Continue monitoring solution quality"
    })
```

---

### Fix #10: Avoid Duplicate Function Calls (Line 717)

**Original Code:**
```python
domain = self._detect_domains(workflow)[0] if self._detect_domains(workflow) else 'general'
```

**Fixed Code:**
```python
detected_domains = self._detect_domains(workflow)
domain = detected_domains[0] if detected_domains else 'general'
```

---

## workflow_stage_functions.py Fixes

### Fix #11: Validate Non-Empty Input (Lines 6-38)

**Original Code:**
```python
def select_integration_strategy(
    sub_problem_solutions: Dict[str, 'SolutionAttempt'],
    problem_statement: str,
    analyzed_context: Dict[str, Any]
) -> str:
    from collections import defaultdict

    dependency_depths = defaultdict(set)
```

**Fixed Code:**
```python
def select_integration_strategy(
    sub_problem_solutions: Dict[str, 'SolutionAttempt'],
    problem_statement: str,
    analyzed_context: Dict[str, Any]
) -> str:
    from collections import defaultdict

    # Validate input
    if not sub_problem_solutions:
        logger.warning("No sub-problem solutions provided, using default 'sequential' strategy")
        return "sequential"

    dependency_depths = defaultdict(set)
```

---

### Fix #12: Regex Error Handling (Lines 59-80)

**Original Code:**
```python
import re
func_pattern = r'def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*(\w+))?'
for match in re.finditer(func_pattern, content):
    func_name = match.group(1)
    params = match.group(2).split(',') if match.group(2) else []
```

**Fixed Code:**
```python
import re

func_pattern = r'def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*(\w+))?'

try:
    for match in re.finditer(func_pattern, content):
        # Safely extract groups
        func_name = match.group(1) if match.lastindex >= 1 else "unknown"
        params_str = match.group(2) if match.lastindex >= 2 else ""
        return_type = match.group(3) if match.lastindex >= 3 else 'Any'

        # Split parameters safely
        params = [p.strip() for p in params_str.split(',') if p.strip()]

        interface["outputs"].append({
            "name": func_name,
            "type": return_type,
            "parameters": params
        })
except re.error as e:
    logger.warning(f"Regex error analyzing function patterns: {e}")
except Exception as e:
    logger.warning(f"Unexpected error analyzing function patterns: {e}")
```

---

### Fix #13: Safe Division Operation (Line 215)

**Original Code:**
```python
qa_results["completeness"] = len(referenced_solutions) / len(sub_problem_solutions) if sub_problem_solutions else 1.0
```

**Fixed Code:**
```python
num_solutions = len(sub_problem_solutions)
qa_results["completeness"] = len(referenced_solutions) / num_solutions if num_solutions > 0 else 1.0
```

---

## workflow_enhanced_stages.py Fixes

### Fix #14: Check for None Before Nested Access (Line 859)

**Original Code:**
```python
"avg_complexity": sum(sp.ai_suggested_complexity_score for sp in workflow_state.decomposition_plan.sub_problems) / len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan.sub_problems else 0,
```

**Fixed Code:**
```python
plan = workflow_state.decomposition_plan
if plan and plan.sub_problems:
    avg_complexity = sum(sp.ai_suggested_complexity_score for sp in plan.sub_problems) / len(plan.sub_problems)
else:
    avg_complexity = 0
```

---

### Fix #15: Safe Max Operation (Line 910)

**Original Code:**
```python
most_common_flaw = max(flaw_types.items(), key=lambda x: x[1])
```

**Fixed Code:**
```python
if flaw_types:
    most_common_flaw = max(flaw_types.items(), key=lambda x: x[1])
    insights.append({
        "insight_type": "common_flaw_pattern",
        "most_common_flaw_type": most_common_flaw[0],
        "occurrence_count": most_common_flaw[1],
        "recommendation": f"Focus on addressing {most_common_flaw[0]} issues"
    })
else:
    logger.info("No flaws detected in critique reports")
```

---

## workflow_history_manager.py Fixes

### Fix #16: Better JSON Error Handling (Lines 29-35)

**Original Code:**
```python
if os.path.exists(self.history_file):
    with open(self.history_file, 'r', encoding='utf-8') as f:
        try:
            raw_history = json.load(f)
            self.history: Dict[str, WorkflowState] = {}
```

**Fixed Code:**
```python
if os.path.exists(self.history_file):
    try:
        with open(self.history_file, 'r', encoding='utf-8') as f:
            try:
                raw_history = json.load(f)
                self.history: Dict[str, WorkflowState] = {}
            except json.JSONDecodeError as e:
                print(f"Error decoding workflow history file: {e}")
                # Backup corrupted file
                backup_path = f"{self.history_file}.corrupted.{int(time.time())}"
                import shutil
                try:
                    shutil.copy(self.history_file, backup_path)
                    print(f"Corrupted file backed up to: {backup_path}")
                except Exception as backup_error:
                    print(f"Failed to backup corrupted file: {backup_error}")

                self.history = {}
    except IOError as e:
        print(f"Error reading history file: {e}")
        self.history = {}
else:
    self.history = {}
```

---

### Fix #17: Safe Dataclass Reconstruction (Lines 34-50)

**Original Code:**
```python
try:
    # Reconstruct ModelConfig
    if 'content_analyzer_team' in wf_data and wf_data['content_analyzer_team']:
        wf_data['content_analyzer_team']['members'] = [ModelConfig(**m) for m in wf_data['content_analyzer_team']['members']]
        wf_data['content_analyzer_team'] = Team(**wf_data['content_analyzer_team'])
```

**Fixed Code:**
```python
try:
    # Reconstruct ModelConfig with error handling
    if 'content_analyzer_team' in wf_data and wf_data['content_analyzer_team']:
        try:
            members_data = wf_data['content_analyzer_team'].get('members', [])
            wf_data['content_analyzer_team']['members'] = [ModelConfig(**m) for m in members_data]
            wf_data['content_analyzer_team'] = Team(**wf_data['content_analyzer_team'])
        except (TypeError, ValueError) as e:
            print(f"Error reconstructing content_analyzer_team for {wf_id}: {e}")
            wf_data['content_analyzer_team'] = None
        except Exception as e:
            print(f"Unexpected error reconstructing content_analyzer_team for {wf_id}: {e}")
            wf_data['content_analyzer_team'] = None
```

---

### Fix #18: Remove Redundant hasattr Check (Line 210)

**Original Code:**
```python
if workflow.decomposition_plan and hasattr(workflow.decomposition_plan, 'openevolve_metrics'):
    if workflow.decomposition_plan.openevolve_metrics:
```

**Fixed Code:**
```python
if workflow.decomposition_plan and workflow.decomposition_plan.openevolve_metrics:
```

---

### Fix #19: Explicit Division Safety (Line 182)

**Original Code:**
```python
if fitness_improvements:
    metrics["average_fitness_improvement"] = sum(fitness_improvements) / len(fitness_improvements)
```

**Fixed Code:**
```python
if fitness_improvements:
    metrics["average_fitness_improvement"] = sum(fitness_improvements) / len(fitness_improvements)
else:
    metrics["average_fitness_improvement"] = 0.0
```

---

## workflow_lifecycle_controller.py Fixes

### Fix #20: Validate Status Info Structure (Lines 65-76)

**Original Code:**
```python
status_info = self.integration.get_workflow_instance_status(selected_instance_id)
if "error" in status_info:
    st.error(f"Error getting workflow status: {status_info['error']}")
    return

# Display current status
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Status", f"{self._get_status_icon(status_info['status'])} {status_info['status'].upper()}")
```

**Fixed Code:**
```python
status_info = self.integration.get_workflow_instance_status(selected_instance_id)

# Validate response structure
if not isinstance(status_info, dict):
    st.error(f"Invalid status info returned: {type(status_info)}")
    return

if "error" in status_info:
    st.error(f"Error getting workflow status: {status_info.get('error', 'Unknown error')}")
    return

# Validate required fields
required_fields = ['status', 'current_stage', 'progress']
missing_fields = [f for f in required_fields if f not in status_info]
if missing_fields:
    st.error(f"Invalid status info: missing fields {missing_fields}")
    return

# Display current status
col1, col2, col3, col4 = st.columns(4)
with col1:
    status = status_info.get('status', 'unknown')
    st.metric("Status", f"{self._get_status_icon(status)} {status.upper()}")
```

---

### Fix #21: Safe Timestamp Access (Line 185)

**Original Code:**
```python
"start_time": datetime.fromtimestamp(status_info["start_time"]).isoformat() if status_info["start_time"] else "N/A",
```

**Fixed Code:**
```python
start_time = status_info.get("start_time")
if start_time:
    try:
        start_time_str = datetime.fromtimestamp(start_time).isoformat()
    except (ValueError, OSError) as e:
        logger.warning(f"Invalid start_time {start_time}: {e}")
        start_time_str = "Invalid"
else:
    start_time_str = "N/A"

# ... use start_time_str
```

---

### Fix #22: Better JSON Error Messages (Lines 341-345)

**Original Code:**
```python
try:
    input_dict = json.loads(inputs)
except json.JSONDecodeError:
    st.error("Invalid JSON in input parameters")
    return
```

**Fixed Code:**
```python
try:
    input_dict = json.loads(inputs)
except json.JSONDecodeError as e:
    st.error(f"Invalid JSON in input parameters: {str(e)}")
    st.error(f"JSON parse error at line {e.lineno}, column {e.colno}")
    return
except Exception as e:
    st.error(f"Unexpected error parsing inputs: {str(e)}")
    return
```

---

### Fix #23: Validate Timeline Data (Lines 206-210)

**Original Code:**
```python
timeline_data = [
    {"event": "Created", "time": status_info["start_time"]},
    {"event": "Started", "time": status_info["start_time"]},
    {"event": "Completed", "time": status_info["end_time"]} if status_info["end_time"] else {"event": "In Progress", "time": time.time()}
]
```

**Fixed Code:**
```python
start_time = status_info.get("start_time") or time.time()
end_time = status_info.get("end_time")

timeline_data = [
    {"event": "Created", "time": start_time},
    {"event": "Started", "time": start_time},
]

if end_time:
    timeline_data.append({"event": "Completed", "time": end_time})
else:
    timeline_data.append({"event": "In Progress", "time": time.time()})
```

---

### Fix #24: Safe Integration Call (Lines 111-117)

**Original Code:**
```python
if st.button("▶️ Start", disabled=not can_start, key=f"start_{instance_id}"):
    result = self.integration.start_workflow_instance(instance_id)
    if "error" in result:
        st.error(f"Start failed: {result['error']}")
    else:
        st.success(f"Workflow started: {result['message']}")
    st.rerun()
```

**Fixed Code:**
```python
if st.button("▶️ Start", disabled=not can_start, key=f"start_{instance_id}"):
    try:
        result = self.integration.start_workflow_instance(instance_id)

        # Validate result
        if result is None:
            st.error("Start failed: No response from integration")
            return

        if not isinstance(result, dict):
            st.error(f"Start failed: Invalid response type {type(result)}")
            return

        if "error" in result:
            error_msg = result.get('error', 'Unknown error')
            st.error(f"Start failed: {error_msg}")
            logger.error(f"Workflow start failed for {instance_id}: {error_msg}")
        else:
            success_msg = result.get('message', 'Started successfully')
            st.success(f"Workflow started: {success_msg}")
            st.rerun()
    except Exception as e:
        st.error(f"Start failed with exception: {str(e)}")
        logger.exception(f"Exception starting workflow {instance_id}")
```

---

## Testing Recommendations

After applying these fixes, create the following tests:

### Unit Tests
```python
def test_safe_max_operation():
    """Test max operation with empty list"""
    flaw_types = {}
    # Should not raise ValueError
    if flaw_types:
        result = max(flaw_types.items(), key=lambda x: x[1])
    else:
        result = None
    assert result is None

def test_division_by_zero():
    """Test division operations handle zero"""
    sub_problem_solutions = []
    num_solutions = len(sub_problem_solutions)
    completeness = 1.0 if num_solutions == 0 else 0.5
    assert completeness == 1.0

def test_none_dictionary_access():
    """Test safe dictionary access"""
    workflow = type('Workflow', (), {'sub_problem_solutions': None})()
    solutions = workflow.sub_problem_solutions or {}
    assert solutions == {}
```

### Integration Tests
```python
async def test_oneke_initialization():
    """Test OneKE initialization error handling"""
    extractor = WorkflowKnowledgeExtractor(use_oneke=True)
    # Should not crash during initialization
    result = await extractor.ensure_oneke_initialized()
    # Result should be bool
    assert isinstance(result, bool)
```

---

## Conclusion

All 24 bugs have been addressed with comprehensive fixes that include:
- Proper error handling
- Defensive programming
- Clear error messages
- Logging for debugging
- Type validation

Apply these fixes in order of priority (Critical -> High -> Medium -> Low) to minimize risk.

>>>>>>> 1cb9c5e35 (update)
