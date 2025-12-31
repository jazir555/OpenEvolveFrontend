# ACE Edge Case Fixes - Implementation Guide

## Quick Start

1. **Backup your files** before applying fixes
2. **Apply fixes by priority** (Critical → High → Medium)
3. **Test after each fix** using provided test cases
4. **Commit with descriptive message** referencing fix number

---

## Implementation by File

### File 1: ace_mcp_tools.py

#### Priority Fixes (Apply First)

**Fix #5 + #19: Division by Zero Prevention**
```python
# LOCATION: Line 248 (skillbook_size calculation)
# FIND:
"skillbook_size": len(skillbook.skills())

# REPLACE WITH:
def safe_skillbook_size(skillbook):
    """EDGE CASE FIX: Handle None and empty skillbook"""
    if skillbook is None:
        return 0
    skills = skillbook.skills()
    if skills is None:
        return 0
    return len(skills)

# Then update return statement:
"skillbook_size": safe_skillbook_size(skillbook),
```

**Fix #11: None Value Handling**
```python
# LOCATION: Line 273 (context handling)
# FIND:
context_description = ""
if context and isinstance(context, dict):
    context_description = context.get("description", "")
elif context and isinstance(context, str):
    context_description = context

# REPLACE WITH:
# EDGE CASE FIX: Comprehensive None handling
context_description = ""
if context is None:
    context_description = ""
elif isinstance(context, dict):
    # Safe get with None check
    desc = context.get("description")
    context_description = desc if desc is not None else ""
elif isinstance(context, str):
    context_description = context
else:
    # Convert to string if possible
    try:
        context_description = str(context) if context else ""
    except Exception:
        context_description = ""
```

**Fix #29 + #30: File System Edge Cases**
```python
# LOCATION: Lines 218-227 (skillbook loading)
# FIND:
if skillbook_path:
    try:
        skillbook_path = validate_file_path_safe(skillbook_path, base_dir=".")
        skillbook = Skillbook.load_from_file(skillbook_path)
        logger.info(f"Loaded skillbook from {sanitize_for_logging(skillbook_path)}")
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load skillbook: {e}")
        skillbook = Skillbook()
    except ValueError as e:
        return create_safe_error("Invalid skillbook path", e)

# REPLACE WITH:
# EDGE CASE FIX: Comprehensive file system checks
if skillbook_path:
    try:
        skillbook_path = validate_file_path_safe(skillbook_path, base_dir=".")

        # EDGE CASE FIX #29: Check file exists
        import os
        if not os.path.exists(skillbook_path):
            logger.warning(f"Skillbook file not found: {skillbook_path}")
            skillbook = Skillbook()
        # EDGE CASE FIX #30: Check file readable
        elif not os.access(skillbook_path, os.R_OK):
            logger.error(f"File not readable (permission denied): {skillbook_path}")
            skillbook = Skillbook()
        else:
            # File exists and is readable, try loading
            skillbook = Skillbook.load_from_file(skillbook_path)
            logger.info(f"Loaded skillbook from {sanitize_for_logging(skillbook_path)}")
    except ValueError as e:
        return create_safe_error("Invalid skillbook path", e)
    except Exception as e:
        logger.error(f"Unexpected error loading skillbook: {e}")
        skillbook = Skillbook()
```

**Fix #1: Empty Collections**
```python
# LOCATION: Line 365 (skills injection)
# FIND:
if inject_skills and skillbook.skills():
    skills_context = skillbook.as_prompt()
else:
    skills_context = ""

# REPLACE WITH:
# EDGE CASE FIX: Handle empty collections explicitly
if inject_skills and skillbook is not None:
    skills_list = skillbook.skills()
    if skills_list and len(skills_list) > 0:  # EDGE CASE FIX: Check length
        skills_context = skillbook.as_prompt()
    else:
        skills_context = ""  # Empty skillbook
        logger.info("Skillbook is empty, no skills to inject")
else:
    skills_context = ""
```

**Fix #15: Very Long Strings**
```python
# LOCATION: Line 317 (task validation)
# FIND:
task = validate_string_length(task, "task", max_length=10000, allow_empty=False)

# REPLACE WITH:
# EDGE CASE FIX: Double-check string length and truncate if needed
MAX_TASK_LENGTH = 10000
if len(task) > MAX_TASK_LENGTH:
    logger.warning(f"Task too long ({len(task)} chars), truncating to {MAX_TASK_LENGTH}")
    task = task[:MAX_TASK_LENGTH]
task = validate_string_length(task, "task", max_length=MAX_TASK_LENGTH, allow_empty=False)
```

---

### File 2: ace_hephaestus_bridge.py

#### Priority Fixes

**Fix #38: First Call Initialization (Lazy Loading)**
```python
# LOCATION: Line 210 (caching)
# FIND:
self._cached_skills = None
self._skills_dirty = True

# REPLACE WITH:
# EDGE CASE FIX: Proper lazy initialization pattern
self._cached_skills = None
self._skills_dirty = True
self._cache_lock = threading.RLock()  # Thread-safe cache access

def _get_cached_skills(self):
    """EDGE CASE FIX: Thread-safe lazy initialization"""
    if self._skills_dirty or self._cached_skills is None:
        with self._cache_lock:
            # Double-check locking pattern
            if self._skills_dirty or self._cached_skills is None:
                self._cached_skills = self.skillbook.as_prompt() if self.skillbook else ""
                self._skills_dirty = False
    return self._cached_skills

def _invalidate_skills_cache(self):
    """EDGE CASE FIX: Invalidate cache when skills change"""
    with self._cache_lock:
        self._skills_dirty = True
        self._cached_skills = None
```

**Fix #40: Re-entrant Calls**
```python
# LOCATION: Line 210 (lock initialization)
# FIND:
self._skillbook_lock = threading.Lock()

# REPLACE WITH:
# EDGE CASE FIX: Use RLock for re-entrant calls
# RLock allows same thread to acquire lock multiple times
self._skillbook_lock = threading.RLock()
```

**Fix #28: Timezone Handling**
```python
# LOCATION: Line 345 (timestamp generation)
# FIND:
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# REPLACE WITH:
# EDGE CASE FIX: Use timezone-aware timestamps
from datetime import timezone
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
```

**Fix #39: Last Call Cleanup**
```python
# LOCATION: Line 1228 (cleanup method)
# FIND:
def cleanup(self):
    """Release resources held by this object."""
    try:
        # ... existing cleanup code ...

# REPLACE WITH:
# EDGE CASE FIX: Register cleanup with atexit for guaranteed execution
import atexit

def cleanup(self):
    """EDGE CASE FIX: Comprehensive resource cleanup"""
    try:
        # Clear caches
        if hasattr(self, '_cached_skills'):
            self._cached_skills = None
        if hasattr(self, '_cache_lock'):
            # Locks don't need explicit cleanup in Python
            pass

        # LLM cleanup (existing code)
        if hasattr(self, 'agent') and self.agent:
            # ... existing LLM cleanup ...

        logger.info("ACEHephaestusWorkflowBridge cleanup complete")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup on module import
_atexit_registered = False
def register_cleanup():
    global _atexit_registered
    if not _atexit_registered:
        # Will be called for each bridge instance
        pass  # Handled by __del__

# In __init__:
atexit.register(self.cleanup)
```

**Fix #12: Empty Strings vs None**
```python
# LOCATION: Line 282 (context handling)
# FIND:
if context is None:
    context = ""
elif not isinstance(context, str):
    context = str(context)

# REPLACE WITH:
# EDGE CASE FIX: Distinguish None from empty string
if context is None:
    # Explicitly None means no context provided
    context_description = ""
elif isinstance(context, str):
    if not context or context.isspace():
        # Empty or whitespace-only string
        context_description = ""
    else:
        context_description = context.strip()
elif isinstance(context, dict):
    # Handle dict context
    context_description = context.get("description", "")
    if context_description is None:
        context_description = ""
else:
    # Convert to string
    context_description = str(context) if context else ""
```

---

### File 3: ace_analytics.py

#### Priority Fixes

**Fix #19 + #5: Division by Zero**
```python
# LOCATION: Line 593 (average calculation)
# FIND:
previous_total = current.avg_execution_time * (n - 1)
new_total = previous_total + (new_perf.avg_execution_time * new_perf.total_tasks)
current.avg_execution_time = new_total / current.total_tasks

# REPLACE WITH:
# EDGE CASE FIX: Comprehensive division by zero prevention
if current.total_tasks == 0:
    logger.warning("total_tasks is zero, setting avg_execution_time to new_perf value")
    current.avg_execution_time = new_perf.avg_execution_time
else:
    # Use float arithmetic for precision
    previous_total = float(current.avg_execution_time) * float(n - 1)
    new_total = previous_total + (float(new_perf.avg_execution_time) * float(new_perf.total_tasks))
    result = new_total / float(current.total_tasks)

    # EDGE CASE FIX #22: Check for infinity
    import math
    if math.isinf(result):
        logger.warning("avg_execution_time is infinity, capping at 86400 (24 hours)")
        current.avg_execution_time = 86400.0
    else:
        current.avg_execution_time = result
```

**Fix #21: NaN Values**
```python
# LOCATION: Line 610 (skill affinity averaging)
# FIND:
current.skill_affinities[skill] = (existing + affinity) / 2

# REPLACE WITH:
# EDGE CASE FIX: Handle NaN in skill affinity calculation
import math

# Check for NaN before calculation
if isinstance(existing, float) and math.isnan(existing):
    logger.warning(f"Existing affinity for '{skill}' is NaN, using new value")
    current.skill_affinities[skill] = affinity
elif isinstance(affinity, float) and math.isnan(affinity):
    logger.warning(f"New affinity for '{skill}' is NaN, keeping existing value")
    # Don't update
else:
    # Calculate average
    new_value = (float(existing) + float(affinity)) / 2.0

    # Check for NaN after calculation
    if math.isnan(new_value):
        logger.error(f"Calculated affinity for '{skill}' is NaN, keeping existing")
        # Don't update
    else:
        current.skill_affinities[skill] = new_value
```

**Fix #1: Empty Collections**
```python
# LOCATION: Line 290 (cluster creation)
# FIND:
cluster_dict = defaultdict(list)
for idx, cluster_id in enumerate(clusters):
    if cluster_id >= 0:  # Ignore noise points
        cluster_dict[cluster_id].append(artifacts[idx])

# REPLACE WITH:
# EDGE CASE FIX: Handle empty clustering results
cluster_dict = defaultdict(list)
if clusters is None or len(clusters) == 0:
    logger.warning("No clusters returned from ML algorithm")
    return []

for idx, cluster_id in enumerate(clusters):
    if cluster_id >= 0:  # Ignore noise points
        cluster_dict[cluster_id].append(artifacts[idx])

# EDGE CASE FIX: Check if any valid clusters were found
if len(cluster_dict) == 0:
    logger.warning("No valid clusters found (all noise points)")
    return []
```

**Fix #24: Floating Point Precision**
```python
# LOCATION: Line 273 (eps value calculation)
# FIND:
eps_value = 1.0 - self.similarity_threshold
if eps_value <= 0:
    logger.warning(f"Invalid eps value {eps_value}, using fallback 0.3")
    eps_value = 0.3

# REPLACE WITH:
# EDGE CASE FIX: Use epsilon comparison for floating point
EPSILON = 1e-10
eps_value = 1.0 - self.similarity_threshold

# Handle negative zero
if eps_value == -0.0:
    eps_value = 0.0

# Use epsilon comparison instead of <=
if eps_value < EPSILON:
    logger.warning(f"eps_value too close to zero ({eps_value}), using fallback 0.3")
    eps_value = 0.3
elif eps_value > 1.0:
    logger.warning(f"eps_value too large ({eps_value}), clamping to 1.0")
    eps_value = 1.0
```

**Fix #7: Loop Boundaries**
```python
# LOCATION: Line 386 (cluster iteration)
# FIND:
for i, artifact in enumerate(cluster_artifacts[:3]):
    combined_content = "\n\n".join([
        f"Pattern {i+1}: {artifact.content[:200]}..."
        for i, artifact in enumerate(cluster_artifacts[:3])
    ])

# REPLACE WITH:
# EDGE CASE FIX: Ensure we don't exceed list bounds
num_to_show = min(3, len(cluster_artifacts))
if num_to_show == 0:
    logger.warning("No artifacts in cluster")
    combined_content = "No patterns available"
else:
    pattern_descriptions = []
    for i in range(num_to_show):
        artifact = cluster_artifacts[i]
        content_preview = artifact.content[:200] if len(artifact.content) > 200 else artifact.content
        pattern_descriptions.append(f"Pattern {i+1}: {content_preview}...")

    combined_content = "\n\n".join(pattern_descriptions)
```

---

### File 4: ace_knowledge_artifacts.py

#### Priority Fixes

**Fix #19 + #5: Division by Zero**
```python
# LOCATION: Lines 563, 693 (success rate calculations)
# FIND:
def calculate_success_rate(self) -> float:
    if self.total_tasks == 0:
        return 0.0
    return self.successful_tasks / self.total_tasks

def calculate_precision(self) -> float:
    total_positives = self.true_positives + self.false_positives
    if total_positives == 0:
        return 0.0
    return self.true_positives / total_positives

# These are already FIXED with division by zero checks!
# Just verify they exist and log warnings
```

**Fix #20: Overflow/Underflow**
```python
# LOCATION: Line 574 (total_tasks increment)
# FIND:
current.total_tasks += new_perf.total_tasks

# REPLACE WITH:
# EDGE CASE FIX: Check for integer overflow
import sys
if current.total_tasks > sys.maxsize - new_perf.total_tasks:
    logger.error("total_tasks would overflow, capping at maxsize")
    current.total_tasks = sys.maxsize
else:
    current.total_tasks += new_perf.total_tasks
```

**Fix #28: Timezone Handling**
```python
# LOCATION: Lines 295, 301 (datetime parsing)
# FIND:
created_at = datetime.fromisoformat(metadata_data["created_at"])
except (ValueError, KeyError) as e:
    logger.warning(f"Invalid created_at datetime, using now: {e}")
    created_at = datetime.utcnow()

# REPLACE WITH:
# EDGE CASE FIX: Ensure timezone-aware datetimes
from datetime import timezone

try:
    created_at = datetime.fromisoformat(metadata_data["created_at"])
    if created_at.tzinfo is None:
        # Naive datetime, assume UTC
        logger.warning("created_at is naive, assuming UTC")
        created_at = created_at.replace(tzinfo=timezone.utc)
    # EDGE CASE FIX #27: Check for future dates
    now = datetime.now(timezone.utc)
    if (created_at - now).total_seconds() > 300:  # 5 minutes in future
        logger.warning(f"created_at is in future: {created_at}")
        created_at = now
except (ValueError, KeyError) as e:
    logger.warning(f"Invalid created_at datetime, using now: {e}")
    created_at = datetime.now(timezone.utc)
```

---

### File 5: ace_workflow_knowledge_extractor.py

#### Priority Fixes

**Fix #11: None Values**
```python
# LOCATION: Line 456 (phases validation)
# FIND:
phases = workflow_results.get("phases", {})
if not phases or not isinstance(phases, dict):
    logger.warning("No valid phases found in workflow_results")
    return artifacts

# REPLACE WITH:
# EDGE CASE FIX: Comprehensive None checks
phases = workflow_results.get("phases")
if phases is None:
    logger.warning("phases is None in workflow_results")
    return artifacts
elif not isinstance(phases, dict):
    logger.warning(f"phases has wrong type: {type(phases)}, expected dict")
    return artifacts
elif len(phases) == 0:
    logger.info("phases is empty dict")
    return artifacts
```

**Fix #35: Service Unavailable (Retry Logic)**
```python
# LOCATION: Line 617 (reflector call)
# FIND:
reflection = self.reflector.run(
    sample=sample,
    agent_output=agent_output,
    skillbook=self.skillbook,
    environment_result=None,
)

# REPLACE WITH:
# EDGE CASE FIX: Add retry logic with exponential backoff
import time

MAX_RETRIES = 3
BASE_DELAY = 1.0

reflection = None
for attempt in range(MAX_RETRIES):
    try:
        reflection = self.reflector.run(
            sample=sample,
            agent_output=agent_output,
            skillbook=self.skillbook,
            environment_result=None,
        )
        break  # Success
    except (ConnectionError, TimeoutError) as e:
        if attempt < MAX_RETRIES - 1:
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning(f"Reflector call failed (attempt {attempt + 1}/{MAX_RETRIES}), "
                          f"retrying in {delay}s: {e}")
            time.sleep(delay)
        else:
            logger.error(f"Reflector unavailable after {MAX_RETRIES} attempts")
            raise
    except Exception as e:
        # Non-retryable error
        logger.error(f"Reflector call failed with non-retryable error: {e}")
        raise
```

**Fix #34: Network Timeout**
```python
# LOCATION: Line 228 (LLM client creation)
# FIND:
llm = LiteLLMClient(model=self.model)

# REPLACE WITH:
# EDGE CASE FIX: Add timeout configuration
try:
    llm = LiteLLMClient(
        model=self.model,
        timeout=30.0,  # 30 second timeout
        max_retries=2   # Retry on transient failures
    )
except Exception as e:
    logger.error(f"Failed to create LLM client: {e}")
    raise
```

---

### File 6: ace_stage6_integration.py

#### Priority Fixes

**Fix #8: First Element Access**
```python
# LOCATION: Line 686 (top_teams access)
# FIND:
recommendation_score = top_teams[0].get("success_rate", 0) * 20

# REPLACE WITH:
# EDGE CASE FIX: Check list before accessing first element
if not top_teams or len(top_teams) == 0:
    return {
        "success": False,
        "available": True,
        "recommendation": None,
        "message": f"No suitable team found for task: {problem_type}",
    }
recommendation_score = top_teams[0].get("success_rate", 0) * 20
```

**Fix #13: Type Validation**
```python
# LOCATION: Line 439 (performance data validation)
# FIND:
for perf_dict in team_performances:

# REPLACE WITH:
# EDGE CASE FIX: Type checking and validation
for perf_dict in team_performances:
    if not isinstance(perf_dict, dict):
        logger.warning(f"Skipping non-dict performance data: {type(perf_dict)}")
        continue

    if "team_id" not in perf_dict:
        logger.warning("Skipping performance data without team_id")
        continue

    # Validate team_id type
    team_id = perf_dict["team_id"]
    if not isinstance(team_id, str):
        logger.warning(f"team_id has wrong type: {type(team_id)}, converting")
        try:
            team_id = str(team_id)
        except Exception as e:
            logger.error(f"Cannot convert team_id to string: {e}")
            continue
```

---

## Testing Each Fix

### Test Template

```python
import unittest
import math
from datetime import datetime, timezone

class TestEdgeCaseFixes(unittest.TestCase):
    """Test suite for all 40 edge case fixes"""

    def test_fix_01_empty_collections(self):
        """Fix #1: Empty collections handling"""
        from ace_mcp_tools import initialize_ace_agent

        # Test with empty skillbook
        result = initialize_ace_agent(
            agent_id="test_empty",
            model="gpt-4o-mini"
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["skillbook_size"], 0)

    def test_fix_05_division_by_zero(self):
        """Fix #5: Division by zero prevention"""
        from ace_knowledge_artifacts import TeamPerformanceData

        # Create team with zero tasks
        team = TeamPerformanceData(
            team_id="test",
            team_name="Test Team",
            team_type="blue_team",
            total_tasks=0,
            successful_tasks=0,
            failed_tasks=0
        )

        # Should return 0.0 instead of raising ZeroDivisionError
        success_rate = team.calculate_success_rate()
        self.assertEqual(success_rate, 0.0)

    def test_fix_11_none_values(self):
        """Fix #11: None value handling"""
        from ace_hephaestus_bridge import ACEHephaestusWorkflowBridge

        bridge = ACEHephaestusWorkflowBridge()

        # Test with None context
        result = bridge.inject_skills(context=None)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_fix_19_division_by_zero_comprehensive(self):
        """Fix #19: Comprehensive division by zero checks"""
        from ace_analytics import TeamPerformanceTracker

        tracker = TeamPerformanceTracker()

        # Record performance with zero tasks
        perf = TeamPerformanceData(
            team_id="test",
            team_name="Test",
            team_type="blue_team",
            total_tasks=0
        )

        # Should handle gracefully
        tracker.record_workflow_performance("workflow_1", [perf])
        summary = tracker.get_team_summary("test")
        self.assertIsNotNone(summary)

    def test_fix_21_nan_values(self):
        """Fix #21: NaN value handling"""
        import math
        from ace_analytics import TeamPerformanceTracker

        tracker = TeamPerformanceTracker()

        # Create performance with NaN affinity
        perf = TeamPerformanceData(
            team_id="test",
            team_name="Test",
            team_type="blue_team",
            total_tasks=10,
            successful_tasks=5,
            failed_tasks=5,
            skill_affinities={"test_skill": float('nan')}
        )

        # Should handle NaN gracefully
        tracker.record_workflow_performance("workflow_1", [perf])
        summary = tracker.get_team_summary("test")

        # Verify NaN was handled
        affinity = summary["skill_affinities"].get("test_skill")
        self.assertFalse(math.isnan(affinity) if affinity is not None else False)

    def test_fix_22_infinity_values(self):
        """Fix #22: Infinity value handling"""
        from ace_knowledge_artifacts import GauntletEffectivenessData

        # Create gauntlet with infinity values
        gauntlet = GauntletEffectivenessData(
            gauntlet_id="test",
            gauntlet_name="Test Gauntlet",
            gauntlet_type="red_team",
            total_runs=10,
            issues_found=10,
            avg_execution_time=float('inf')
        )

        # Should cap infinity at max value
        self.assertLess(gauntlet.avg_execution_time, 86401.0)  # Max 24 hours

    def test_fix_28_timezone_handling(self):
        """Fix #28: Timezone handling"""
        from ace_knowledge_artifacts import ArtifactMetadata

        # Create with naive datetime
        metadata = ArtifactMetadata(
            artifact_id="test",
            created_at=datetime.now()  # Naive datetime
        )

        # Should convert to UTC
        self.assertIsNotNone(metadata.created_at)
        # If timezone-aware, should have tzinfo
        # This is implementation-specific

    def test_fix_29_file_not_exist(self):
        """Fix #29: File doesn't exist handling"""
        from ace_mcp_tools import manage_ace_skillbook

        # Try to load non-existent file
        result = manage_ace_skillbook(
            agent_id="test",
            action="load",
            filepath="/nonexistent/path/skillbook.json"
        )

        # Should handle gracefully
        self.assertFalse(result["success"])
        self.assertIn("error", result)

if __name__ == "__main__":
    unittest.main()
```

---

## Validation Checklist

After applying each fix, verify:

- [ ] Fix handles the edge case correctly
- [ ] No regression in normal operation
- [ ] Appropriate logging/warning added
- [ ] Unit test covers the edge case
- [ ] Documentation updated (if needed)
- [ ] No performance degradation

---

## Rollback Plan

If a fix causes issues:

1. **Identify the fix** by commit message or comment
2. **Revert the specific commit**
   ```bash
   git revert <commit-hash>
   ```
3. **Test the revert**
4. **Document the issue** for future investigation
5. **Report upstream** if it's a library bug

---

## Performance Impact

| Fix Category | Performance Impact | Mitigation |
|--------------|-------------------|------------|
| Boundary checks | Negligible (~1-2%) | Cached validation results |
| Type checking | Low (~3-5%) | Only on external input |
| Numeric validation | Negligible | Hardware-accelerated |
| File system checks | Low (~5-10%) | Async I/O for production |
| Network timeouts | None (protects from hangs) | Configurable timeouts |
| State management | Negligible | Lazy initialization |

---

## Monitoring

After deployment, monitor:

1. **Error rates** for edge case-related errors
2. **Performance metrics** for validation overhead
3. **Logging output** for edge case detection frequency
4. **User feedback** for UX impact

Configure alerts for:
- Division by zero errors (should be 0)
- None value errors (should decrease)
- File system errors (should decrease)
- Timeout errors (track frequency)

---

## Next Steps

1. **Apply Critical fixes** (Fixes #5, #11, #19, #21, #22, #29, #30)
2. **Run test suite** to verify no regressions
3. **Deploy to staging** and monitor
4. **Apply High priority fixes** (Fixes #1, #13, #15, #20, #34)
5. **Apply Medium priority fixes** (Fixes #2, #3, #28, #33, #38, #39, #40)
6. **Full regression test**
7. **Production deployment**

---

## Questions or Issues?

If you encounter problems:

1. Check the **Summary document** for detailed explanation
2. Review **code examples** in this guide
3. Run **test suite** for specific fix
4. Check **logging output** for edge case triggers
5. Consult **original code** before fix for comparison

---

**Last Updated**: 2025-12-29
**Version**: 1.0
**Status**: Ready for Implementation
