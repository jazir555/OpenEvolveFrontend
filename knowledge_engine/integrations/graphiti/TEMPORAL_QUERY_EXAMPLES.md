# Temporal Query Examples

**Comprehensive Examples for Graphiti Temporal Knowledge Graph Queries**

## Table of Contents
1. [Basic Queries](#basic-queries)
2. [Temporal Filter Types](#temporal-filter-types)
3. [Workflow Artifact Queries](#workflow-artifact-queries)
4. [Advanced Temporal Patterns](#advanced-temporal-patterns)
5. [Performance Optimization](#performance-optimization)
6. [Real-World Use Cases](#real-world-use-cases)

---

## Basic Queries

### Example 1: Simple Current Knowledge Query

Get all currently valid knowledge about a topic.

```python
from datetime import datetime
from knowledge_engine.integrations.graphiti.temporal_bridge import (
    GraphitiTemporalBridge,
    TemporalFilter,
)

# Initialize bridge
bridge = GraphitiTemporalBridge()
await bridge.initialize()

# Query for current knowledge
results = await bridge.search_temporal(
    query="machine learning algorithms",
    filter_type=TemporalFilter.CURRENT,
    max_results=10,
)

# Process results
for edge in results["edges"]:
    print(f"Fact: {edge['fact']}")
    print(f"Source: {edge['source']} -> {edge['relation']} -> {edge['target']}")
    print(f"Score: {edge['score']}")
    print()
```

### Example 2: Get Knowledge Valid at Specific Time

Query knowledge as it was at a point in time.

```python
from datetime import datetime

# Query knowledge as it was on January 1, 2024
point_in_time = datetime(2024, 1, 1)

results = await bridge.search_temporal(
    query="database architecture",
    filter_type=TemporalFilter.TIME_RANGE,
    start_time=point_in_time - timedelta(hours=1),
    end_time=point_in_time + timedelta(hours=1),
    max_results=20,
)

print(f"Knowledge valid around {point_in_time}:")
for edge in results["edges"]:
    print(f"- {edge['fact']}")
```

### Example 3: Get Knowledge from Time Range

Query knowledge within a specific time window.

```python
from datetime import datetime, timedelta

# Get knowledge from the last week
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=7)

results = await bridge.search_temporal(
    query="API updates",
    filter_type=TemporalFilter.TIME_RANGE,
    start_time=start_time,
    end_time=end_time,
    max_results=50,
)

print(f"Knowledge from last 7 days: {len(results['edges'])} results")
```

---

## Temporal Filter Types

### CURRENT Filter

Get knowledge that is valid right now.

```python
# Only returns edges where:
# - valid_at <= now
# - invalid_at is NULL or invalid_at > now

results = await bridge.search_temporal(
    query="current product pricing",
    filter_type=TemporalFilter.CURRENT,
    max_results=10,
)

for edge in results["edges"]:
    # These facts are currently valid
    print(f"Valid: {edge['fact']}")
```

### TIME_RANGE Filter

Get knowledge valid within a time range.

```python
# Get knowledge from Q4 2023
start_time = datetime(2023, 10, 1)
end_time = datetime(2024, 1, 1)

results = await bridge.search_temporal(
    query="quarterly revenue",
    filter_type=TemporalFilter.TIME_RANGE,
    start_time=start_time,
    end_time=end_time,
    max_results=20,
)

# Returns edges where knowledge was valid at any point in Q4
for edge in results["edges"]:
    created_at = datetime.fromisoformat(edge['created_at'])
    expired_at = datetime.fromisoformat(edge['expired_at']) if edge['expired_at'] else None

    print(f"Valid from {created_at} to {expired_at or 'present'}")
```

### POINT_IN_TIME Filter

Reconstruct knowledge state at a specific moment.

```python
# What did we know about the competitor on June 1, 2024?
snapshot_time = datetime(2024, 6, 1)

# Narrow window around the point
results = await bridge.search_temporal(
    query="competitor product features",
    filter_type=TemporalFilter.TIME_RANGE,
    start_time=snapshot_time - timedelta(minutes=5),
    end_time=snapshot_time + timedelta(minutes=5),
    max_results=30,
)

print(f"Knowledge state on {snapshot_time}:")
for edge in results["edges"]:
    print(f"- {edge['fact']}")
```

### ALL_TIME Filter

Get all knowledge regardless of time.

```python
# Get all historical data
results = await bridge.search_temporal(
    query="product evolution",
    filter_type=TemporalFilter.ALL_TIME,
    max_results=100,
)

print(f"Total knowledge over all time: {len(results['edges'])} facts")
```

---

## Workflow Artifact Queries

### Example 1: Track Workflow Execution

Track a workflow through its lifecycle.

```python
from knowledge_engine.integrations.graphiti.temporal_bridge import (
    WorkflowState,
)

# Start workflow
artifact = await bridge.track_workflow_artifact(
    workflow_id="etl-pipeline-001",
    workflow_name="Customer Data ETL",
    state=WorkflowState.IN_PROGRESS,
    started_at=datetime.utcnow(),
    metadata={"source": "Salesforce", "target": "Snowflake"},
)

print(f"Workflow started: {artifact.artifact_id}")

# ... workflow executes ...

# Complete workflow
artifact = await bridge.track_workflow_artifact(
    workflow_id="etl-pipeline-001",
    workflow_name="Customer Data ETL",
    state=WorkflowState.COMPLETED,
    completed_at=datetime.utcnow(),
    metadata={
        "records_processed": 50000,
        "duration_seconds": 3600,
    },
)

print(f"Workflow completed: {artifact.episode_uuid}")
```

### Example 2: Query Workflow State History

Get the history of a workflow's state changes.

```python
workflow_id = "etl-pipeline-001"

# Query current state
current_state = await bridge.query_workflow_state_at_time(
    workflow_id=workflow_id,
    timestamp=datetime.utcnow(),
)

print(f"Current state: {current_state.state.value}")

# Query state 1 hour ago
past_time = datetime.utcnow() - timedelta(hours=1)
past_state = await bridge.query_workflow_state_at_time(
    workflow_id=workflow_id,
    timestamp=past_time,
)

print(f"State 1 hour ago: {past_state.state.value if past_state else 'Unknown'}")
```

### Example 3: Get Workflow Timeline

Get a chronological timeline of workflow events.

```python
# Get events for the last 24 hours
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=1)

timeline = await bridge.get_workflow_timeline(
    workflow_id="etl-pipeline-001",
    start_time=start_time,
    end_time=end_time,
)

print("Workflow Timeline:")
for event in sorted(timeline, key=lambda x: x['timestamp']):
    timestamp = datetime.fromisoformat(event['timestamp'])
    print(f"  {timestamp.strftime('%H:%M:%S')}: {event['event_type']}")
```

### Example 4: Analyze Workflow Performance

Query completed workflows for performance analysis.

```python
# Search for completed ETL workflows
results = await bridge.search_temporal(
    query="workflow:etl-pipeline state:completed",
    filter_type=TemporalFilter.TIME_RANGE,
    start_time=datetime.utcnow() - timedelta(days=30),
    max_results=100,
)

# Analyze performance
durations = []
for edge in results["edges"]:
    metadata = edge.get('metadata', {})
    if 'duration_seconds' in metadata:
        durations.append(metadata['duration_seconds'])

if durations:
    avg_duration = sum(durations) / len(durations)
    print(f"Average workflow duration: {avg_duration:.0f}s")
    print(f"Min: {min(durations):.0f}s, Max: {max(durations):.0f}s")
```

---

## Advanced Temporal Patterns

### Example 1: Temporal Join Queries

Find relationships that were valid at the same time.

```python
async def find_concurrent_relationships(
    entity: str,
    time1: datetime,
    time2: datetime,
):
    """Find relationships for entity at two different times."""
    results1 = await bridge.search_temporal(
        query=entity,
        filter_type=TemporalFilter.TIME_RANGE,
        start_time=time1 - timedelta(hours=1),
        end_time=time1 + timedelta(hours=1),
        max_results=50,
    )

    results2 = await bridge.search_temporal(
        query=entity,
        filter_type=TemporalFilter.TIME_RANGE,
        start_time=time2 - timedelta(hours=1),
        end_time=time2 + timedelta(hours=1),
        max_results=50,
    )

    # Compare the two snapshots
    edges1 = {e['fact'] for e in results1['edges']}
    edges2 = {e['fact'] for e in results2['edges']}

    added = edges2 - edges1
    removed = edges1 - edges2

    print(f"Relationships added: {len(added)}")
    print(f"Relationships removed: {len(removed)}")

    return added, removed

# Usage
time1 = datetime(2024, 1, 1)
time2 = datetime(2024, 6, 1)
await find_concurrent_relationships("ProductA", time1, time2)
```

### Example 2: Temporal Aggregation

Aggregate knowledge over time periods.

```python
from collections import defaultdict

async def count_entities_by_period(
    query: str,
    start_time: datetime,
    end_time: datetime,
    period_days: int = 7,
):
    """Count entities mentioned per time period."""
    period_start = start_time
    entity_counts = defaultdict(int)

    while period_start < end_time:
        period_end = min(period_start + timedelta(days=period_days), end_time)

        results = await bridge.search_temporal(
            query=query,
            filter_type=TemporalFilter.TIME_RANGE,
            start_time=period_start,
            end_time=period_end,
            max_results=100,
        )

        # Count unique entities
        entities = set()
        for edge in results['edges']:
            entities.add(edge.get('source'))
            entities.add(edge.get('target'))

        entity_counts[period_start.strftime('%Y-%m-%d')] = len(entities)

        period_start = period_end

    return entity_counts

# Usage
counts = await count_entities_by_period(
    query="product launches",
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31),
    period_days=30,  # Monthly counts
)

for period, count in counts.items():
    print(f"{period}: {count} entities")
```

### Example 3: Temporal Path Finding

Find paths between entities at specific times.

```python
async def find_temporal_path(
    source: str,
    target: str,
    timestamp: datetime,
    max_hops: int = 3,
):
    """Find a path from source to target at a given time."""
    visited = set()
    queue = [(source, 0)]
    path = []

    while queue and len(path) < max_hops:
        current, hops = queue.pop(0)

        if current in visited:
            continue
        visited.add(current)

        path.append(current)

        if current == target:
            return path

        # Find neighbors at the given time
        results = await bridge.search_temporal(
            query=current,
            filter_type=TemporalFilter.TIME_RANGE,
            start_time=timestamp - timedelta(hours=1),
            end_time=timestamp + timedelta(hours=1),
            max_results=50,
        )

        for edge in results['edges']:
            if edge['source'] == current and edge['target'] not in visited:
                queue.append((edge['target'], hops + 1))

    return None  # No path found

# Usage
path = await find_temporal_path(
    source="CompanyA",
    target="CompanyC",
    timestamp=datetime(2024, 6, 1),
    max_hops=4,
)

if path:
    print(f"Path found: {' -> '.join(path)}")
```

### Example 4: Temporal Diff

Compare knowledge states between two times.

```python
async def temporal_diff(
    query: str,
    time1: datetime,
    time2: datetime,
):
    """Compare knowledge at two different times."""
    results1 = await bridge.search_temporal(
        query=query,
        filter_type=TemporalFilter.TIME_RANGE,
        start_time=time1 - timedelta(hours=1),
        end_time=time1 + timedelta(hours=1),
        max_results=100,
    )

    results2 = await bridge.search_temporal(
        query=query,
        filter_type=TemporalFilter.TIME_RANGE,
        start_time=time2 - timedelta(hours=1),
        end_time=time2 + timedelta(hours=1),
        max_results=100,
    )

    facts1 = {e['fact']: e for e in results1['edges']}
    facts2 = {e['fact']: e for e in results2['edges']}

    added = set(facts2.keys()) - set(facts1.keys())
    removed = set(facts1.keys()) - set(facts2.keys())
    unchanged = set(facts1.keys()) & set(facts2.keys())

    print(f"Added: {len(added)}")
    print(f"Removed: {len(removed)}")
    print(f"Unchanged: {len(unchanged)}")

    return {
        'added': [facts2[f] for f in added],
        'removed': [facts1[f] for f in removed],
        'unchanged': list(unchanged),
    }

# Usage
diff = await temporal_diff(
    query="product features",
    time1=datetime(2024, 1, 1),
    time2=datetime(2024, 6, 1),
)

print(f"\nNew features:")
for fact in diff['added'][:5]:
    print(f"  + {fact['fact']}")
```

---

## Performance Optimization

### Example 1: Use Specific Time Ranges

Narrow time ranges for faster queries.

```python
# BAD: Query all time
results = await bridge.search_temporal(
    query="product updates",
    filter_type=TemporalFilter.ALL_TIME,
    max_results=1000,  # Too many results
)

# GOOD: Query specific range
results = await bridge.search_temporal(
    query="product updates",
    filter_type=TemporalFilter.TIME_RANGE,
    start_time=datetime.utcnow() - timedelta(days=7),  # Last week only
    max_results=50,  # Reasonable limit
)
```

### Example 2: Batch Queries

Process multiple queries concurrently.

```python
import asyncio

async def batch_temporal_queries(queries: List[str]):
    """Execute multiple queries concurrently."""
    tasks = [
        bridge.search_temporal(
            query=q,
            filter_type=TemporalFilter.CURRENT,
            max_results=10,
        )
        for q in queries
    ]

    results = await asyncio.gather(*tasks)
    return results

# Usage
queries = [
    "product pricing",
    "product features",
    "product availability",
]

all_results = await batch_temporal_queries(queries)

for query, results in zip(queries, all_results):
    print(f"{query}: {len(results['edges'])} results")
```

### Example 3: Cache Frequently Accessed Data

Cache current state queries.

```python
from functools import lru_cache
from datetime import datetime, timedelta

class TemporalQueryCache:
    def __init__(self, bridge: GraphitiTemporalBridge):
        self.bridge = bridge
        self._cache = {}
        self._cache_time = {}
        self._ttl = timedelta(minutes=5)

    async def get_current_knowledge(self, query: str):
        """Get current knowledge with caching."""
        now = datetime.utcnow()

        # Check cache
        if query in self._cache:
            cache_time = self._cache_time[query]
            if now - cache_time < self._ttl:
                return self._cache[query]

        # Cache miss or expired
        results = await self.bridge.search_temporal(
            query=query,
            filter_type=TemporalFilter.CURRENT,
            max_results=10,
        )

        # Update cache
        self._cache[query] = results
        self._cache_time[query] = now

        return results

# Usage
cache = TemporalQueryCache(bridge)
results = await cache.get_current_knowledge("product pricing")
```

---

## Real-World Use Cases

### Use Case 1: Knowledge Base Versioning

Track how documentation evolved over time.

```python
async def get_documentation_state(
    feature: str,
    as_of_date: datetime,
):
    """Get documentation for a feature as of a specific date."""
    results = await bridge.search_temporal(
        query=f"documentation {feature}",
        filter_type=TemporalFilter.TIME_RANGE,
        start_time=as_of_date - timedelta(days=30),
        end_time=as_of_date,
        max_results=20,
    )

    print(f"Documentation for '{feature}' as of {as_of_date}:")
    for edge in results['edges']:
        print(f"- {edge['fact']}")

# Usage
await get_documentation_state(
    feature="API authentication",
    as_of_date=datetime(2024, 3, 15),
)
```

### Use Case 2: Compliance Auditing

Query historical system states for compliance.

```python
async def compliance_audit(
    system: str,
    audit_period_start: datetime,
    audit_period_end: datetime,
):
    """Audit system state for compliance."""
    results = await bridge.search_temporal(
        query=f"{system} configuration",
        filter_type=TemporalFilter.TIME_RANGE,
        start_time=audit_period_start,
        end_time=audit_period_end,
        max_results=100,
    )

    print(f"Compliance audit for {system}:")
    print(f"Period: {audit_period_start} to {audit_period_end}")
    print(f"Configuration changes found: {len(results['edges'])}")

    for edge in results['edges']:
        created_at = datetime.fromisoformat(edge['created_at'])
        print(f"  [{created_at}] {edge['fact']}")

# Usage
await compliance_audit(
    system="payment_gateway",
    audit_period_start=datetime(2024, 1, 1),
    audit_period_end=datetime(2024, 3, 31),
)
```

### Use Case 3: Product Evolution Tracking

Track how a product changed over time.

```python
async def track_product_evolution(
    product: str,
    months: int = 12,
):
    """Track product features over time."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30 * months)

    # Get all knowledge about the product
    results = await bridge.search_temporal(
        query=product,
        filter_type=TemporalFilter.TIME_RANGE,
        start_time=start_time,
        end_time=end_time,
        max_results=200,
    )

    # Group by month
    monthly_changes = {}
    for edge in results['edges']:
        created_at = datetime.fromisoformat(edge['created_at'])
        month_key = created_at.strftime('%Y-%m')

        if month_key not in monthly_changes:
            monthly_changes[month_key] = []

        monthly_changes[month_key].append(edge['fact'])

    # Print timeline
    print(f"Evolution of {product} over {months} months:")
    for month in sorted(monthly_changes.keys()):
        print(f"\n{month}: {len(monthly_changes[month])} changes")
        for fact in monthly_changes[month][:3]:  # Top 3
            print(f"  - {fact}")

# Usage
await track_product_evolution("ProductA", months=12)
```

### Use Case 4: Incident Investigation

Reconstruct system state during an incident.

```python
async def investigate_incident(
    incident_time: datetime,
    affected_system: str,
    window_minutes: int = 30,
):
    """Investigate what was known during an incident."""
    window_start = incident_time - timedelta(minutes=window_minutes)
    window_end = incident_time + timedelta(minutes=window_minutes)

    # Get system state before, during, and after
    before = await bridge.search_temporal(
        query=f"{affected_system} status",
        filter_type=TemporalFilter.TIME_RANGE,
        start_time=window_start,
        end_time=incident_time,
        max_results=50,
    )

    during = await bridge.search_temporal(
        query=f"{affected_system} status",
        filter_type=TemporalFilter.TIME_RANGE,
        start_time=incident_time - timedelta(minutes=5),
        end_time=incident_time + timedelta(minutes=5),
        max_results=50,
    )

    after = await bridge.search_temporal(
        query=f"{affected_system} status",
        filter_type=TemporalFilter.TIME_RANGE,
        start_time=incident_time,
        end_time=window_end,
        max_results=50,
    )

    print(f"Incident Investigation: {affected_system}")
    print(f"Incident time: {incident_time}")
    print(f"\nBEFORE ({len(before['edges'])} facts):")
    for edge in before['edges'][:5]:
        print(f"  {edge['fact']}")

    print(f"\nDURING ({len(during['edges'])} facts):")
    for edge in during['edges'][:5]:
        print(f"  {edge['fact']}")

    print(f"\nAFTER ({len(after['edges'])} facts):")
    for edge in after['edges'][:5]:
        print(f"  {edge['fact']}")

# Usage
await investigate_incident(
    incident_time=datetime(2024, 6, 15, 14, 30),
    affected_system="payment_api",
    window_minutes=60,
)
```

---

**Last Updated:** 2026-01-08
**Version:** 1.0.0
