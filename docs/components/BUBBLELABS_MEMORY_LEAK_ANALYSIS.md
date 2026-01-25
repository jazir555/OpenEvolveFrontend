# BubbleLabs Integration - Memory & Resource Leak Analysis Report

**Date:** 2025-12-29
**Analyst:** Claude Code
**Scope:** Comprehensive memory and resource leak detection for BubbleLabs integration system

---

## Executive Summary

This report provides a comprehensive analysis of potential memory and resource leaks in the BubbleLabs integration system. The analysis covered 5 core files:

1. `bubblelabs_hephaestus_bridge.py` (755 lines)
2. `bubblelabs_mcp_tools.py` (804 lines)
3. `bubblelabs_analytics.py` (813 lines)
4. `bubblelabs_integration.py` (286 lines)
5. `bubblelabs_security.py` (707 lines)

**Overall Assessment:** **MODERATE RISK** - Several potential memory leaks identified, primarily related to unbounded data structures and missing cleanup mechanisms.

---

## 1. CRITICAL MEMORY LEAKS (Unbounded Growth)

### 1.1 bubblelabs_hephaestus_bridge.py

#### CRITICAL: `self.mappings` Dictionary - No Eviction Policy
**Location:** Line 111
**Severity:** CRITICAL
**Impact:** Grows without bound over time

```python
# Track workflow-to-ticket mappings
self.mappings: Dict[str, WorkflowTicketMapping] = {}
```

**Issue:**
- Dictionary grows indefinitely as new workflows are created
- No eviction policy for old/completed workflows
- Each entry contains WorkflowTicketMapping objects with timestamps
- Memory usage: O(n) where n = total workflows ever created

**Evidence:**
- Line 181: `self.mappings[workflow_definition.id] = mapping` - Always adds, never removes
- No cleanup method or expiration mechanism

**Recommended Fix:**
- Implement LRU cache with max size (e.g., 1000 entries)
- Add TTL-based expiration (e.g., 24 hours)
- Clean up mappings when workflows complete
- Add `cleanup_old_mappings()` method called periodically

```python
from functools import lru_cache
from collections import OrderedDict

class BubbleLabsHephaestusBridge:
    def __init__(self, max_mappings: int = 1000, mapping_ttl: int = 86400):
        self.max_mappings = max_mappings
        self.mapping_ttl = mapping_ttl  # 24 hours
        self.mappings: OrderedDict = OrderedDict()  # For LRU eviction

    def _evict_old_mappings(self):
        """Remove mappings older than TTL or exceeding max size."""
        now = time.time()
        to_remove = [
            wid for wid, m in self.mappings.items()
            if now - m.created_at > self.mapping_ttl
        ]
        for wid in to_remove:
            del self.mappings[wid]

        # Also evict oldest if over max size
        while len(self.mappings) > self.max_mappings:
            self.mappings.popitem(last=False)  # Remove oldest (LRU)
```

---

#### CRITICAL: `self.instance_to_definition_map` - Unbounded Cache
**Location:** Line 115
**Severity:** CRITICAL
**Impact:** Grows with every workflow instance

```python
# Instance to definition ID reverse mapping cache (fixes issue #3)
self.instance_to_definition_map: Dict[str, str] = {}
```

**Issue:**
- Cache grows indefinitely (one entry per instance)
- Line 696: `self.instance_to_definition_map[instance_id] = instance.definition_id` - Always adds
- Only updated in `_update_instance_cache()` (line 571), but never prunes
- Old instances never removed

**Evidence:**
- Line 588: `self.instance_to_definition_map = new_cache` - Replaces entire cache but only with CURRENT instances
- However, if BubbleLabsIntegration keeps old instances, this cache grows unbounded

**Recommended Fix:**
- Limit cache size (e.g., max 10,000 entries)
- Add TTL for instance mappings (e.g., 7 days)
- Prune entries for completed/cancelled instances

```python
def _update_instance_cache(self) -> None:
    """Update the instance-to-definition mapping cache with size limits."""
    try:
        instances = self.bubblelabs.list_workflow_instances()

        with self.lock:
            # Rebuild cache with current instances
            new_cache: Dict[str, str] = {}
            for instance in instances:
                # Only cache active instances, not completed ones
                if instance.status in ["running", "pending", "paused"]:
                    new_cache[instance.id] = instance.definition_id

            # Enforce max size
            if len(new_cache) > 10000:
                # Keep only most recent 10,000
                new_cache = dict(list(new_cache.items())[-10000:])

            # Update cache (atomic replacement)
            self.instance_to_definition_map = new_cache

            logger.debug(f"Updated instance cache with {len(new_cache)} entries")

    except Exception as e:
        logger.warning(f"Error updating instance cache: {e}")
```

---

### 1.2 bubblelabs_mcp_tools.py

#### CRITICAL: `_MCP_TOOLS` Registry - No Removal Mechanism
**Location:** Line 123
**Severity:** MEDIUM
**Impact:** Low impact (typically only registered once at startup)

```python
_MCP_TOOLS: Dict[str, Callable] = {}
```

**Issue:**
- Tools are registered but never unregistered
- Low impact since tools are typically registered once at module load
- However, if tools are dynamically registered/unregistered, this could leak

**Recommended Fix:**
- Add `unregister_mcp_tool(name: str)` method
- Document that tools should only be registered at startup

---

#### MEDIUM: Singleton Instances - Never Cleaned Up
**Location:** Lines 64-66
**Severity:** MEDIUM
**Impact:** Singleton instances persist for process lifetime

```python
_shared_bubblelabs_integration = None
_shared_api_instance = None
```

**Issue:**
- Singletons are created but never explicitly destroyed
- They hold references to BubbleLabsIntegration and OpenEvolveBubbleLabsIntegration
- These objects contain their own unbounded collections

**Recommended Fix:**
- Add cleanup method to reset singletons (for testing)
- Implement weak references if appropriate
- Document singleton lifecycle

```python
def reset_singletons():
    """Reset singleton instances (for testing)."""
    global _shared_bubblelabs_integration, _shared_api_instance
    _shared_bubblelabs_integration = None
    _shared_api_instance = None
```

---

### 1.3 bubblelabs_analytics.py

#### CRITICAL: Database Tables - Unbounded Growth
**Location:** Database tables (lines 225-271)
**Severity:** CRITICAL
**Impact:** Database grows without bound

```sql
CREATE TABLE IF NOT EXISTS workflows (...)
CREATE TABLE IF NOT EXISTS node_metrics (...)
CREATE TABLE IF NOT EXISTS provider_metrics (...)
```

**Issue:**
- Tables accumulate data forever
- No automatic cleanup/archival of old records
- `node_metrics` table can grow very large (one row per node execution)
- No partitioning or TTL mechanism

**Evidence:**
- Line 402: `INSERT INTO node_metrics ...` - Always inserts
- No corresponding DELETE or archival mechanism

**Recommended Fix:**
- Implement data retention policy (e.g., keep 90 days)
- Add scheduled cleanup job
- Consider partitioning by date
- Add `cleanup_old_data()` method

```python
def cleanup_old_data(self, retention_days: int = 90):
    """Clean up analytics data older than retention period."""
    try:
        cutoff_time = time.time() - (retention_days * 86400)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Delete old workflows
            cursor.execute("""
                DELETE FROM workflows
                WHERE created_at < ?
            """, (cutoff_time,))

            # Delete old node metrics
            cursor.execute("""
                DELETE FROM node_metrics
                WHERE timestamp < ?
            """, (cutoff_time,))

            # Delete old provider metrics
            cursor.execute("""
                DELETE FROM provider_metrics
                WHERE timestamp < ?
            """, (cutoff_time,))

            conn.commit()

            logger.info(f"Cleaned up analytics data older than {retention_days} days")

    except Exception as e:
        logger.error(f"Error cleaning up old data: {e}")
```

---

#### MEDIUM: Connection Pool - Bounded but May Leak Connections
**Location:** Line 142
**Severity:** MEDIUM
**Impact:** Connection exhaustion under error conditions

```python
self._connection_pool: List[sqlite3.Connection] = []
```

**Issue:**
- Connection pool has max size (line 143: `self._pool_size = pool_size`)
- However, if exceptions occur in `get_connection()`, connections may not be returned
- Lines 189-195: Exception handling may not properly return connection to pool

**Evidence:**
- Lines 183-187: Connection only returned to pool if `len(self._connection_pool) < self._pool_size`
- If pool is full, connection is closed (good!)
- But if exception occurs before line 186, connection is neither returned nor closed

**Recommended Fix:**
- Ensure all paths return connection to pool or close it
- Use try/finally more carefully

```python
@contextmanager
def get_connection(self):
    """Context manager for database connections with improved error handling."""
    conn = None
    try:
        # Try to get connection from pool
        with self._pool_lock:
            if self._connection_pool:
                conn = self._connection_pool.pop()
                logger.debug(f"Reusing connection from pool (pool size: {len(self._connection_pool)})")

        # Create new connection if pool was empty
        if conn is None:
            conn = sqlite3.connect(self.db_path)

        yield conn

        # Return connection to pool on success
        with self._pool_lock:
            if len(self._connection_pool) < self._pool_size:
                self._connection_pool.append(conn)
                conn = None  # Mark as returned to pool

    except Exception as e:
        logger.error(f"Database connection error: {e}")
        # Don't return connection to pool if error occurred
        conn = None  # Ensure it gets closed in finally
        raise

    finally:
        # Close connection if not returned to pool
        if conn is not None:
            try:
                conn.close()
            except:
                pass
```

---

### 1.4 bubblelabs_integration.py

#### CRITICAL: `self.workflow_instances` - No Eviction
**Location:** Line 77
**Severity:** CRITICAL
**Impact:** Accumulates all workflow instances forever

```python
self.workflow_instances: Dict[str, BubbleWorkflowInstance] = {}
```

**Issue:**
- Stores all workflow instances ever created
- Line 195: `self.workflow_definitions[workflow_id] = definition` - Always adds
- No cleanup of completed/cancelled instances
- Each instance contains potentially large `data` dict

**Evidence:**
- No method to remove old instances
- No expiration mechanism
- Lists all instances without filtering (line 209)

**Recommended Fix:**
- Add TTL for completed instances (e.g., 7 days)
- Limit max instances stored
- Implement archival for old instances

```python
def cleanup_old_instances(self, retention_days: int = 7):
    """Remove old workflow instances."""
    cutoff_time = time.time() - (retention_days * 86400)

    with self._instances_lock:
        to_remove = [
            iid for iid, inst in self.workflow_instances.items()
            if inst.status in ["completed", "failed", "cancelled"]
            and inst.updated_at < cutoff_time
        ]

        for iid in to_remove:
            del self.workflow_instances[iid]

        logger.info(f"Cleaned up {len(to_remove)} old workflow instances")
```

---

#### CRITICAL: `self.workflow_definitions` - No Eviction
**Location:** Line 78
**Severity:** MEDIUM
**Impact:** Lower impact than instances (definitions are smaller)

```python
self.workflow_definitions: Dict[str, BubbleWorkflowDefinition] = {}
```

**Issue:**
- Similar to instances, grows without bound
- Definitions contain nodes/edges lists which can be large

**Recommended Fix:**
- Same as instances - TTL-based cleanup

---

#### HIGH: `self.running_threads` - Threads May Not Be Cleaned Up
**Location:** Line 79
**Severity:** HIGH
**Impact:** Thread leakage and resource exhaustion

```python
self.running_threads: Dict[str, threading.Thread] = {}
```

**Issue:**
- Lines 252-266: Thread is removed from dict on cancel, but not necessarily stopped
- Line 266: `self.running_threads.pop(instance_id, None)` - Removes from dict
- However, no `.join()` to wait for thread to actually finish
- Threads may continue running in background

**Evidence:**
- Lines 254-263: Sets cancel_event/stop_event but doesn't wait for thread to stop
- No verification that thread actually stopped

**Recommended Fix:**
- Add `.join()` with timeout after setting stop event
- Verify thread stopped before removing from dict

```python
elif action == "cancel":
    instance.status = "cancelled"
    instance.updated_at = time.time()

    # Stop the running thread if it exists
    with self._threads_lock:
        if instance_id in self.running_threads:
            thread = self.running_threads.get(instance_id)
            if hasattr(thread, "cancel_event"):
                thread.cancel_event.set()
            if hasattr(thread, "stop_event"):
                thread.stop_event.set()
            instance.data["cancel_requested"] = True

            # FIX: Wait for thread to stop
            if thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Thread for {instance_id} did not stop within 5s")

            # Now remove from dict
            self.running_threads.pop(instance_id, None)

    return {"message": "Workflow cancelled", "status": instance.status}
```

---

### 1.5 bubblelabs_security.py

#### MEDIUM: `self.api_keys` - No Eviction
**Location:** Line 290
**Severity:** MEDIUM
**Impact:** Low (typically small number of API keys)

```python
self.api_keys: Dict[str, SecurityContext] = {}
```

**Issue:**
- API keys accumulate
- No revocation mechanism for old keys
- SecurityContext objects contain sets of permissions

**Recommended Fix:**
- Add `revoke_api_key()` method
- Implement key expiration

---

#### HIGH: `self.sessions` - No Eviction
**Location:** Line 291
**Severity:** HIGH
**Impact:** Session data accumulates (potentially large)

```python
self.sessions: Dict[str, SecurityContext] = {}
```

**Issue:**
- Sessions stored forever
- No session timeout/expiry
- Each session has SecurityContext with permissions set

**Recommended Fix:**
- Implement session expiration (e.g., 24 hour timeout)
- Add cleanup job

```python
def cleanup_expired_sessions(self, timeout_seconds: int = 86400):
    """Remove expired sessions."""
    cutoff_time = time.time() - timeout_seconds

    # Need to add created_at to SecurityContext first
    with self.lock:
        to_remove = []
        for session_id, context in self.sessions.items():
            if hasattr(context, 'created_at') and context.created_at < cutoff_time:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self.sessions[session_id]

        logger.info(f"Cleaned up {len(to_remove)} expired sessions")
```

---

#### HIGH: `self.tokens` (CSRF) - Partial Cleanup
**Location:** Line 382
**Severity:** MEDIUM
**Impact:** Tokens accumulate until used or expire

```python
self.tokens: Dict[str, Dict[str, Any]] = {}
```

**Issue:**
- Tokens removed on validation (line 431) if expired
- But unused tokens never cleaned up
- TTL check only happens on validation, not proactively

**Evidence:**
- Line 430: `if time.time() - token_data["created_at"] > 3600:` - Only checked during validation
- No background cleanup job

**Recommended Fix:**
- Add periodic cleanup of expired tokens
- Or use TTL-based cache (e.g., `cachetools.TTLCache`)

---

#### HIGH: `self.buckets` (RateLimiter) - No Eviction
**Location:** Line 463
**Severity:** MEDIUM
**Impact:** Accumulates one entry per unique user/session

```python
self.buckets: Dict[str, Dict[str, Any]] = {}
```

**Issue:**
- Each user gets a bucket that persists forever
- No cleanup of inactive users
- Could accumulate entries for all historical users

**Recommended Fix:**
- Implement idle timeout (e.g., remove buckets inactive for 1 hour)
- Add periodic cleanup

```python
def cleanup_inactive_buckets(self, idle_seconds: int = 3600):
    """Remove buckets for inactive users."""
    cutoff_time = time.time() - idle_seconds

    with self.lock:
        to_remove = [
            user_id for user_id, bucket in self.buckets.items()
            if bucket["last_update"] < cutoff_time
        ]

        for user_id in to_remove:
            del self.buckets[user_id]

        logger.debug(f"Cleaned up {len(to_remove)} inactive rate limit buckets")
```

---

## 2. THREAD LEAKS

### 2.1 Background Sync Thread - May Not Stop Properly

**File:** `bubblelabs_hephaestus_bridge.py`
**Location:** Lines 376-439
**Severity:** MEDIUM
**Status:** PARTIALLY FIXED

**Issue:**
- `stop_background_sync()` method (line 408) uses Event.wait() properly (line 429)
- However, thread is daemon (line 395: `daemon=True`), so it will be killed on exit
- Timeout is 10 seconds (line 429) - may not be enough for cleanup
- No verification that thread actually stopped

**Evidence:**
- Line 429: `self.sync_thread.join(timeout=timeout)` - Good!
- Line 431-432: Checks if thread is still alive and logs error - Good!
- But doesn't raise exception or retry

**Recommended Fix:**
- Increase default timeout to 30 seconds
- Add retry mechanism
- Consider making it non-daemon and ensuring proper shutdown

---

### 2.2 UI Workflow Thread - No Cleanup

**File:** `bubblelabs_ui_component.py`
**Location:** Line 709
**Severity:** HIGH

```python
thread = threading.Thread(target=run_workflow)
```

**Issue:**
- Thread created but never joined
- No cleanup mechanism
- If workflow is cancelled, thread may continue running

**Recommended Fix:**
- Store thread reference
- Implement stop event
- Join thread with timeout

---

## 3. FILE HANDLE LEAKS

### 3.1 File Operations - Generally Safe

**Files Checked:**
- `bubblelabs_analytics.py` lines 638, 642
- `bubblelabs_typescript_export.py` line 225

**Status:** ✓ NO LEAKS DETECTED

**Evidence:**
- All file operations use `with open()` context manager
- Proper exception handling in place
- Files are automatically closed

**Example:**
```python
# Line 638 - Good!
with open(output_path, 'w') as f:
    json.dump(summary, f, indent=2)
```

---

## 4. DATABASE CONNECTION LEAKS

### 4.1 Analytics Database - Generally Safe with Minor Issues

**File:** `bubblelabs_analytics.py`
**Severity:** MEDIUM
**Status:** PARTIALLY FIXED

**Good Practices:**
- Lines 152-195: Context manager `get_connection()` implemented
- Line 197-211: `close_all_connections()` method exists
- Lines 221, 335, 392, 455, 497, 576: All database operations use context manager

**Issues:**
- See section 1.3 for connection pool edge cases
- No automatic cleanup on object destruction
- User must remember to call `close_all_connections()`

**Recommended Fix:**
- Implement `__del__` method to close connections
- Add atexit handler
- Document cleanup requirement

```python
def __del__(self):
    """Cleanup on object destruction."""
    try:
        self.close_all_connections()
    except:
        pass  # Ignore errors during cleanup
```

---

## 5. EVENT LISTENER LEAKS

**Status:** ✓ NO EVENT LISTENERS DETECTED

**Analysis:**
- No observable/callback pattern detected in codebase
- No event emitters or pub/sub systems
- This category is not applicable

---

## 6. CIRCULAR REFERENCES

**Status:** ✓ NO CRITICAL CIRCULAR REFERENCES DETECTED

**Analysis:**
- Most objects use simple parent-child relationships
- No mutual references between objects
- Thread-safe locks use RLock/Lock which don't create cycles
- Potential minor cycle: Bridge → BubbleLabsIntegration → (holds references to workflows)
- However, Python's GC can handle most simple cycles
- No `__del__` methods that would prevent GC

**Recommendation:**
- Use weak references for parent pointers if applicable
- Monitor with `gc.get_count()` periodically

---

## 7. SUMMARY OF FINDINGS

### Critical Leaks (Unbounded Growth)

| File | Line | Issue | Severity | Impact |
|------|------|-------|----------|--------|
| `bubblelabs_hephaestus_bridge.py` | 111 | `self.mappings` - no eviction | CRITICAL | Grows with every workflow |
| `bubblelabs_hephaestus_bridge.py` | 115 | `self.instance_to_definition_map` - no eviction | CRITICAL | Grows with every instance |
| `bubblelabs_analytics.py` | 225-271 | Database tables - no cleanup | CRITICAL | Unbounded DB growth |
| `bubblelabs_integration.py` | 77 | `self.workflow_instances` - no eviction | CRITICAL | Grows with every instance |
| `bubblelabs_integration.py` | 78 | `self.workflow_definitions` - no eviction | MEDIUM | Grows with every definition |
| `bubblelabs_integration.py` | 79 | `self.running_threads` - threads not joined | HIGH | Thread leakage |
| `bubblelabs_security.py` | 291 | `self.sessions` - no eviction | HIGH | Accumulates all sessions |
| `bubblelabs_security.py` | 382 | `self.tokens` (CSRF) - partial cleanup | MEDIUM | Unused tokens not cleaned |
| `bubblelabs_security.py` | 463 | `self.buckets` (rate limit) - no eviction | MEDIUM | Accumulates per user |

### Thread Leaks

| File | Line | Issue | Severity | Impact |
|------|------|-------|----------|--------|
| `bubblelabs_hephaestus_bridge.py` | 376-439 | Background sync thread - partial cleanup | MEDIUM | May not stop in time |
| `bubblelabs_ui_component.py` | 709 | UI workflow thread - no cleanup | HIGH | Thread never joined |

### Resource Leaks

| File | Line | Issue | Severity | Impact |
|------|------|-------|----------|--------|
| `bubblelabs_analytics.py` | 183-195 | Connection pool - may leak on error | MEDIUM | Connection exhaustion |

### No Issues Found

- ✓ File handles - all use context managers
- ✓ Event listeners - not applicable
- ✓ Circular references - none detected
- ✓ Database connections - mostly safe with context managers

---

## 8. PRIORITIZED RECOMMENDATIONS

### Priority 1 (Immediate Action Required)

1. **Add eviction policy to `bubblelabs_hephaestus_bridge.py`**
   - Implement LRU cache for `self.mappings` (max 1000 entries)
   - Implement TTL-based expiration (24 hours)
   - Add cleanup method

2. **Add eviction policy to `bubblelabs_integration.py`**
   - Cleanup old workflow instances (TTL: 7 days)
   - Cleanup old workflow definitions (TTL: 30 days)
   - Add `.join()` for threads on cancel

3. **Implement database cleanup in `bubblelabs_analytics.py`**
   - Add data retention policy (90 days)
   - Implement scheduled cleanup job
   - Add `cleanup_old_data()` method

### Priority 2 (High Importance)

4. **Add session cleanup to `bubblelabs_security.py`**
   - Implement session expiration (24 hours)
   - Add periodic cleanup job

5. **Fix thread cleanup in `bubblelabs_integration.py`**
   - Add `.join()` with timeout after cancel
   - Verify thread stopped before removing from dict

6. **Fix UI workflow thread in `bubblelabs_ui_component.py`**
   - Store thread reference
   - Implement stop event
   - Join thread with timeout

### Priority 3 (Best Practices)

7. **Add CSRF token cleanup to `bubblelabs_security.py`**
   - Implement periodic cleanup of expired tokens
   - Consider using TTL cache

8. **Add rate limiter cleanup to `bubblelabs_security.py`**
   - Implement idle timeout (1 hour)
   - Add periodic cleanup

9. **Improve connection pool error handling in `bubblelabs_analytics.py`**
   - Ensure all error paths return connection to pool or close it
   - Add `__del__` method for cleanup

10. **Add cleanup methods to MCP tools singletons**
    - Implement `reset_singletons()` for testing
    - Document singleton lifecycle

---

## 9. TESTING RECOMMENDATIONS

A comprehensive memory leak test script has been created at:
**`test_memory_leaks.py`**

### Test Coverage

The test script includes:

1. **BubbleLabs-Hephaestus Bridge Test**
   - Creates 100 workflows
   - Tests background sync thread lifecycle
   - Verifies mapping cleanup

2. **MCP Tools Test**
   - Verifies singleton pattern
   - Checks tool registry

3. **Analytics Test**
   - Tracks 100 workflows
   - Tests connection pool behavior
   - Verifies database growth

4. **Integration Test**
   - Creates 100 workflow definitions
   - Checks unbounded collections

5. **Security Test**
   - Creates 100 sessions
   - Generates 100 CSRF tokens
   - Tests rate limiter buckets

6. **Database Connection Leak Test**
   - Performs 50 operations
   - Verifies connection pool bounds
   - Checks connection closure

### Running the Tests

```bash
# Install dependencies
pip install psutil matplotlib

# Run tests
python test_memory_leaks.py
```

### Expected Output

The test will:
- Track memory usage at each checkpoint
- Report memory growth in MB
- Detect leaks by comparing growth before/after cleanup
- Identify specific data structure leaks
- Provide final assessment (PASS/FAIL)

---

## 10. MONITORING RECOMMENDATIONS

To detect memory leaks in production:

### 10.1 Add Memory Metrics

```python
import psutil
import gc

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    logger.info(f"Memory - RSS: {mem_info.rss / 1024 / 1024:.2f} MB, "
                f"VMS: {mem_info.vms / 1024 / 1024:.2f} MB")

    # Log object counts
    logger.info(f"GC objects: {len(gc.get_objects())}")
```

### 10.2 Add Size Metrics

```python
def get_collection_sizes():
    """Log sizes of unbounded collections."""
    logger.info(f"mappings: {len(bridge.mappings)}")
    logger.info(f"instances: {len(integration.workflow_instances)}")
    logger.info(f"definitions: {len(integration.workflow_definitions)}")
    logger.info(f"sessions: {len(auth_manager.sessions)}")
```

### 10.3 Alerting

Set up alerts for:
- Memory usage > 1 GB
- Collection size > 10,000 entries
- Database size > 10 GB
- Thread count > 50

---

## 11. CONCLUSION

The BubbleLabs integration system has **several potential memory leaks** that should be addressed:

**Critical Issues (5):**
1. Unbounded mappings in hephaestus_bridge
2. Unbounded instance cache in hephaestus_bridge
3. Unbounded database tables in analytics
4. Unbounded workflow instances in integration
5. Unbounded workflow definitions in integration

**High Issues (4):**
1. Thread leaks in integration
2. Session accumulation in security
3. CSRF token accumulation in security
4. Rate limiter bucket accumulation in security

**Medium Issues (3):**
1. Connection pool edge cases in analytics
2. API key accumulation in security
3. Singleton lifecycle in MCP tools

**Recommended Action Plan:**
1. Run `test_memory_leaks.py` to confirm issues
2. Implement Priority 1 fixes immediately
3. Add monitoring to detect leaks in production
4. Schedule regular cleanup jobs
5. Re-test after fixes

**Estimated Effort:**
- Priority 1 fixes: 2-3 days
- Priority 2 fixes: 1-2 days
- Priority 3 fixes: 1 day
- Testing & validation: 1 day

**Total Estimated Effort:** 5-7 days

---

## 12. APPENDIX: Static Analysis Results

### 12.1 Unbounded Data Structures Found

| File | Variable | Type | Initial Size | Growth Rate |
|------|----------|------|--------------|-------------|
| `bubblelabs_hephaestus_bridge.py` | `self.mappings` | Dict | 0 | 1 per workflow |
| `bubblelabs_hephaestus_bridge.py` | `self.instance_to_definition_map` | Dict | 0 | 1 per instance |
| `bubblelabs_integration.py` | `self.workflow_instances` | Dict | 0 | 1 per instance |
| `bubblelabs_integration.py` | `self.workflow_definitions` | Dict | 0 | 1 per definition |
| `bubblelabs_integration.py` | `self.running_threads` | Dict | 0 | 1 per running workflow |
| `bubblelabs_security.py` | `self.api_keys` | Dict | 1 | 1 per API key |
| `bubblelabs_security.py` | `self.sessions` | Dict | 0 | 1 per session |
| `bubblelabs_security.py` | `self.tokens` | Dict | 0 | 1 per CSRF token |
| `bubblelabs_security.py` | `self.buckets` | Dict | 0 | 1 per unique user |
| `bubblelabs_analytics.py` | `self._connection_pool` | List | 0 | Max `pool_size` |
| `bubblelabs_mcp_tools.py` | `_MCP_TOOLS` | Dict | 0 | 1 per tool (static) |

### 12.2 Threads Found

| File | Line | Type | Daemon? | Cleanup? |
|------|------|------|---------|----------|
| `bubblelabs_hephaestus_bridge.py` | 395 | Thread | Yes | Partial |
| `bubblelabs_ui_component.py` | 709 | Thread | No | No |

### 12.3 File Operations Found

| File | Line | Operation | Context Manager? | Safe? |
|------|------|-----------|------------------|-------|
| `bubblelabs_analytics.py` | 638 | open(write) | Yes (with) | ✓ |
| `bubblelabs_analytics.py` | 642 | open(write) | Yes (with) | ✓ |
| `bubblelabs_typescript_export.py` | 225 | open(write) | Yes (with) | ✓ |

### 12.4 Database Operations Found

| File | Operation | Context Manager? | Safe? |
|------|-----------|------------------|-------|
| `bubblelabs_analytics.py` | sqlite3.connect() | Yes (get_connection()) | ✓ |
| `bubblelabs_analytics.py` | cursor.execute() | Yes (with) | ✓ |

---

**End of Report**
