# Traceability Matrix Guide for Gauntlet System

This guide explains how to use the traceability system to maintain a complete audit trail of all changes in the Gauntlet system.

## Table of Contents

1. [Overview](#overview)
2. [Change Tracking](#change-tracking)
3. [Audit Trail](#audit-trail)
4. [Querying Changes](#querying-changes)
5. [Reporting](#reporting)
6. [Best Practices](#best-practices)

---

## Overview

The traceability system provides:

- **Complete change tracking**: Every change is logged
- **Full audit trail**: From problem creation to solution
- **Query capabilities**: Filter by problem, team, type, time
- **Diff generation**: Before/after comparisons
- **Compliance support**: Meet regulatory requirements

### Quick Start

```python
from bubblelabs_nodes import get_change_tracker

tracker = get_change_tracker()

# Track a change
tracker.track_change(
    problem_id='problem_123',
    team='blue_team_1',
    change_type='solution_update',
    description='Improved algorithm efficiency',
    before=old_solution,
    after=new_solution
)

# Query changes
changes = tracker.get_changes_for_problem('problem_123')
```

---

## Change Tracking

### Trackable Change Types

1. **solution_update**: Solution modified
2. **validation**: Validation result
3. **decomposition**: Problem decomposed
4. **reassembly**: Solutions combined
5. **gauntlet_pass**: Passed a gauntlet round
6. **gauntlet_fail**: Failed a gauntlet round
7. **cache_hit**: Solution retrieved from cache
8. **cache_miss**: Solution not in cache

### Tracking Examples

```python
from bubblelabs_nodes import get_change_tracker

tracker = get_change_tracker()

# Track solution update
change = tracker.track_change(
    problem_id='problem_123',
    team='blue_team_1',
    change_type='solution_update',
    description='Optimized algorithm',
    before={'algorithm': 'v1', 'score': 0.75},
    after={'algorithm': 'v2', 'score': 0.85}
)

# Track validation
change = tracker.track_change(
    problem_id='problem_123',
    team='gold_team',
    change_type='validation',
    description='Validated solution',
    before={'validated': False},
    after={'validated': True}
)
```

---

## Audit Trail

### Getting Audit Trail

```python
# Get complete audit trail for a problem
audit_trail = tracker.get_timeline('problem_123')

for change in audit_trail:
    print(f"{change.timestamp} - {change.team} - {change.change_type}")
    print(f"  {change.description}")
```

### Complete Audit Trail Example

```
2026-01-23 10:00:00 - blue_team_1 - solution_update
  Improved algorithm efficiency

2026-01-23 10:01:00 - red_team - fuzzing
  Fuzzed solution with 1000 iterations

2026-01-23 10:02:00 - gold_team - validation
  Validated solution, score: 0.85

2026-01-23 10:03:00 - blue_team_1 - gauntlet_pass
  Passed gauntlet round 2

2026-01-23 10:04:00 - system - reassembly
    Reassembled into final solution
```

---

## Querying Changes

### Query by Problem

```python
changes = tracker.get_changes_for_problem('problem_123')

for change in changes:
    print(f"{change.team}: {change.change_type}")
```

### Query by Team

```python
# Get all changes by a team
blue_changes = tracker.get_changes_by_team('blue_team_1')

# Count changes by team
change_counts = tracker.get_change_counts_by_team()
```

### Query by Type

```python
# Get all solution updates
updates = tracker.get_changes_by_type('solution_update')

# Get all gauntlet passes
passes = tracker.get_changes_by_type('gauntlet_pass')
```

### Query by Time Range

```python
from datetime import datetime, timedelta

# Get last hour of changes
end = datetime.utcnow()
start = end - timedelta(hours=1)

changes = tracker.get_changes_by_time_range(start, end)
```

### Complex Queries

```python
# Get failed gauntlet attempts by blue team
changes = tracker.get_changes(
    problem_id='problem_123',
    team='blue_team_1',
    change_type='gauntlet_fail'
)

# Get all validation changes in last day
day_ago = datetime.utcnow() - timedelta(days=1)
validations = tracker.get_changes_by_time_range(
    day_ago,
    datetime.utcnow()
)
validations = [c for c in validations if c.change_type == 'validation']
```

---

## Reporting

### Generate Traceability Report

```python
matrix = TraceabilityMatrix()

# Add all changes
for change in all_changes:
    matrix.add_change(change)

# Generate report
report = matrix.generate_report()

print(f"Total changes: {report['total_changes']}")
print(f"Teams involved: {report['teams']}")
print(f"Change types: {report['change_types']}")
```

### Export to CSV

```python
import csv

changes = tracker.get_all_changes()

with open('traceability.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # Write header
    writer.writerow([
        'timestamp', 'problem_id', 'team', 'change_type',
        'description', 'before_hash', 'after_hash'
    ])

    # Write rows
    for change in changes:
        writer.writerow([
            change.timestamp,
            change.problem_id,
            change.team,
            change.change_type,
            change.description,
            change.before_hash,
            change.after_hash
        ])
```

### Export to JSON

```python
import json

changes = tracker.get_all_changes()

with open('traceability.json', 'w') as f:
    json.dump([
        {
            'timestamp': c.timestamp.isoformat(),
            'problem_id': c.problem_id,
            'team': c.team,
            'change_type': c.change_type,
            'description': c.description,
            'before': c.before,
            'after': c.after,
        }
        for c in changes
    ], f, indent=2)
```

---

## Best Practices

### 1. Track All Modifications

```python
# Always track changes
tracker.track_change(
    problem_id=problem['id'],
    team=current_team,
    change_type='modification_type',
    description='Clear description of change',
    before=before_state,
    after=after_state
)
```

### 2. Use Descriptive Descriptions

```python
# Good: Descriptive
description = "Fixed null pointer dereference in validation"

# Bad: Vague
description = "Bug fix"
```

### 3. Include Context

```python
tracker.track_change(
    problem_id='problem_123',
    team='blue_team_1',
    change_type='solution_update',
    description='Optimized algorithm after profiling',
    before={'algorithm': 'v1', 'score': 0.75},
    after={'algorithm': 'v2', 'score': 0.85},
    metadata={
        'reason': 'Performance issue',
        'profiling_data': {...}
    }
)
```

### 4. Regular Exports

```python
# Export traceability data regularly
import asyncio
from datetime import timedelta

async def periodic_export():
    while True:
        # Export daily
        changes = tracker.get_changes_by_time_range(
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow()
        )

        # Save to backup
        backup_file = f"traceability_{datetime.utcnow():%Y%m%d}.json"
        with open(backup_file, 'w') as f:
            json.dump(changes, f)

        # Wait until next export
        await asyncio.sleep(24 * 3600)

asyncio.create_task(periodiodic_export())
```

### 5. Validate Audit Trail

```python
def validate_audit_trail(problem_id: str) -> bool:
    """Validate audit trail is complete"""
    changes = tracker.get_changes_for_problem(problem_id)

    # Check for required changes
    required_types = ['decomposition', 'validation']
    present_types = {c.change_type for c in changes}

    return required_types.issubset(present_types)
```

---

## Summary

Traceability in Gauntlet provides:
- ✅ Complete change tracking
- ✅ Full audit trail generation
- ✅ Flexible querying capabilities
- ✅ Multiple export formats
- ✅ Compliance support

For more information:
- `bubblelabs_nodes/traceability.py` - Traceability implementation
