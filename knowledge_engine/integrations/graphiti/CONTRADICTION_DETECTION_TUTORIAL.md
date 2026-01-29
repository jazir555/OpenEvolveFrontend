# Contradiction Detection Tutorial

**Complete Guide to Detecting, Resolving, and Managing Contradictions in Graphiti**

## Table of Contents
1. [Introduction](#introduction)
2. [Detecting Contradictions](#detecting-contradictions)
3. [Resolution Strategies](#resolution-strategies)
4. [Automated Reporting](#automated-reporting)
5. [Knowledge Pruning](#knowledge-pruning)
6. [Monitoring and Alerts](#monitoring-and-alerts)
7. [Best Practices](#best-practices)
8. [Advanced Patterns](#advanced-patterns)

---

## Introduction

### What are Contradictions?

Contradictions occur when the knowledge graph contains conflicting information about the same entity or relationship. For example:

```
Fact 1: "ProductA costs $99"
Fact 2: "ProductA costs $149"
```

Without contradiction detection, these conflicts can propagate through the system and lead to incorrect decisions.

### Why Detect Contradictions?

1. **Data Quality:** Ensure knowledge graph consistency
2. **Decision Making:** Prevent conflicting information from affecting decisions
3. **Trust:** Maintain trust in the knowledge base
4. **Compliance:** Meet regulatory requirements for data accuracy
5. **Automation:** Enable safe automated knowledge evolution

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Graph                            │
│                  (with Contradictions)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Contradiction Detector                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Detect     │→ │   Resolve    │→ │    Report    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Actions & Monitoring                            │
│  • Prune Knowledge  • Generate Alerts  • Track Trends       │
└─────────────────────────────────────────────────────────────┘
```

---

## Detecting Contradictions

### Basic Detection

Detect contradictions for a specific entity:

```python
from datetime import datetime, timedelta
from knowledge_engine.integrations.graphiti.contradiction_detector import (
    GraphitiContradictionDetector,
)

# Initialize detector
detector = GraphitiContradictionDetector()
detector.set_bridge(bridge)

# Detect contradictions for an entity
contradictions = await detector.detect_contradictions(
    entity_name="ProductA",
    time_range=(
        datetime.utcnow() - timedelta(days=7),
        datetime.utcnow(),
    ),
)

# Process results
for contradiction in contradictions:
    print(f"Severity: {contradiction.severity.value}")
    print(f"Confidence: {contradiction.confidence:.2f}")
    print(f"Contradictions found: {len(contradiction.contradictions)}")

    for item in contradiction.contradictions:
        print(f"  - {item}")
```

### Understanding Contradiction Types

**Direct Contradictions:**
```
Fact 1: "ProductA status is ACTIVE"
Fact 2: "ProductA status is INACTIVE"
```

**Temporal Contradictions:**
```
Fact 1: "ProductA price is $99 (valid: 2024-01-01 to 2024-06-01)"
Fact 2: "ProductA price is $149 (valid: 2024-03-01 to present)"
# Overlap period creates contradiction
```

**Relation Contradictions:**
```
Fact 1: "ProductA REQUIRES ComponentX"
Fact 2: "ProductA EXCLUDES ComponentX"
```

### Batch Detection

Detect contradictions across multiple entities:

```python
async def detect_batch_contradictions(
    entities: List[str],
    detector: GraphitiContradictionDetector,
):
    """Detect contradictions for multiple entities."""
    all_contradictions = []

    for entity in entities:
        try:
            contradictions = await detector.detect_contradictions(
                entity_name=entity,
                time_range=(
                    datetime.utcnow() - timedelta(days=1),
                    datetime.utcnow(),
                ),
            )
            all_contradictions.extend(contradictions)

        except Exception as e:
            print(f"Error detecting for {entity}: {e}")

    return all_contradictions

# Usage
entities = ["ProductA", "ProductB", "ProductC"]
contradictions = await detect_batch_contradictions(entities, detector)

print(f"Found {len(contradictions)} contradictions")
```

### Filtering by Severity

Focus on high-priority contradictions:

```python
from knowledge_engine.integrations.graphiti.contradiction_detector import (
    ContradictionSeverity,
)

contradictions = await detector.detect_contradictions(
    entity_name="ProductA",
)

# Filter by severity
critical = [c for c in contradictions if c.severity == ContradictionSeverity.CRITICAL]
high = [c for c in contradictions if c.severity == ContradictionSeverity.HIGH]

print(f"Critical: {len(critical)}")
print(f"High: {len(high)}")

# Resolve critical first
for contradiction in critical:
    print(f"Critical: {contradiction.entity_name}")
```

---

## Resolution Strategies

### Strategy Overview

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `KEEP_NEWEST` | Keep most recent knowledge | Recency is most important |
| `KEEP_OLDEST` | Keep oldest knowledge | Historical accuracy matters |
| `KEEP_HIGHEST_CONFIDENCE` | Keep knowledge with highest confidence | Quality scores available |
| `MERGE` | Merge conflicting knowledge | Both have partial truth |
| `FLAG_FOR_REVIEW` | Flag for human review | Unsure of correct resolution |
| `DELETE_ALL` | Remove all contradictory knowledge | Data is unreliable |

### Example 1: Keep Newest

```python
from knowledge_engine.integrations.graphiti.contradiction_detector import (
    ResolutionAction,
)

# Resolve by keeping newest
contradiction = contradictions[0]

success = await detector.resolve_contradiction(
    contradiction_id=contradiction.contradiction_id,
    action=ResolutionAction.KEEP_NEWEST,
    resolution_notes="Newer pricing data is more accurate",
)

if success:
    print("Resolved by keeping newest knowledge")
```

### Example 2: Keep Highest Confidence

```python
# Resolve by keeping highest confidence
success = await detector.resolve_contradiction(
    contradiction_id=contradiction.contradiction_id,
    action=ResolutionAction.KEEP_HIGHEST_CONFIDENCE,
    resolution_notes="Source with higher confidence is more reliable",
)

if success:
    print("Resolved by keeping highest confidence knowledge")
```

### Example 3: Flag for Review

```python
# Flag for human review
success = await detector.resolve_contradiction(
    contradiction_id=contradiction.contradiction_id,
    action=ResolutionAction.FLAG_FOR_REVIEW,
    resolution_notes="Requires manual review - conflicting data from equal sources",
)

if success:
    print("Flagged for human review")

# Get flagged contradictions
alerts = await detector.get_contradiction_alerts(
    severity_threshold=ContradictionSeverity.MEDIUM,
    unresolved_only=True,
)

for alert in alerts:
    print(f"Review needed: {alert['entity']}")
```

### Example 4: Merge Strategy

```python
# Merge contradictory knowledge
success = await detector.resolve_contradiction(
    contradiction_id=contradiction.contradiction_id,
    action=ResolutionAction.MERGE,
    resolution_notes="Merged pricing data - both sources partially correct",
)

if success:
    print("Merged contradictory knowledge")
    # Note: Merge implementation depends on LLM-based merging
```

### Example 5: Batch Resolution

Resolve multiple contradictions automatically:

```python
async def auto_resolve_contradictions(
    contradictions: List,
    strategy: ResolutionAction = ResolutionAction.KEEP_NEWEST,
):
    """Automatically resolve contradictions using a strategy."""
    resolved = 0
    failed = 0

    for contradiction in contradictions:
        # Only auto-resolve if confidence is high
        if contradiction.confidence < 0.8:
            print(f"Skipping low-confidence contradiction: {contradiction.entity_name}")
            continue

        try:
            success = await detector.resolve_contradiction(
                contradiction_id=contradiction.contradiction_id,
                action=strategy,
                resolution_notes=f"Auto-resolved using {strategy.value} strategy",
            )

            if success:
                resolved += 1
            else:
                failed += 1

        except Exception as e:
            print(f"Failed to resolve {contradiction.entity_name}: {e}")
            failed += 1

    print(f"Resolved: {resolved}, Failed: {failed}")
    return resolved, failed

# Usage
high_priority = [c for c in contradictions if c.severity.value >= "high"]
await auto_resolve_contradictions(
    high_priority,
    strategy=ResolutionAction.KEEP_NEWEST,
)
```

---

## Automated Reporting

### Generate Daily Reports

```python
async def generate_daily_report():
    """Generate a daily contradiction report."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)

    report = await detector.generate_contradiction_report(
        time_range=(start_time, end_time),
        include_resolved=True,
    )

    # Print summary
    print(f"Daily Contradiction Report")
    print(f"Period: {start_time} to {end_time}")
    print(f"\nSummary:")
    print(f"  Total: {report.summary['total']}")
    print(f"  By Severity:")
    for severity, count in report.summary['by_severity'].items():
        print(f"    {severity}: {count}")
    print(f"  Resolved: {report.summary['resolved']}")
    print(f"  Unresolved: {report.summary['unresolved']}")

    return report

# Usage
report = await generate_daily_report()
```

### Generate Custom Reports

```python
async def generate_custom_report(
    days: int = 7,
    min_severity: str = "medium",
):
    """Generate custom contradiction report."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    report = await detector.generate_contradiction_report(
        time_range=(start_time, end_time),
        include_resolved=False,
    )

    # Filter by severity
    from knowledge_engine.integrations.graphiti.contradiction_detector import (
        ContradictionSeverity,
    )

    min_severity_enum = ContradictionSeverity(min_severity)
    filtered = [
        c for c in report.contradictions
        if c.severity.value >= min_severity
    ]

    print(f"Custom Report ({days} days, severity >= {min_severity})")
    print(f"Total contradictions: {len(filtered)}")

    for contradiction in filtered[:10]:
        print(f"\n{contradiction.entity_name} ({contradiction.severity.value})")
        print(f"  Confidence: {contradiction.confidence:.2f}")
        print(f"  Contradictions: {len(contradiction.contradictions)}")

    return report

# Usage
await generate_custom_report(days=30, min_severity="high")
```

### Export Report to File

```python
import json

async def export_report_to_file(
    report,
    output_path: str,
):
    """Export contradiction report to JSON file."""
    report_data = report.to_dict()

    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"Report exported to {output_path}")

# Usage
report = await detector.generate_contradiction_report()
await export_report_to_file(report, "contradiction_report_2024_01_08.json")
```

### Trend Analysis

```python
async def analyze_trends(days: int = 30):
    """Analyze contradiction trends over time."""
    reports = []

    for day in range(days):
        end_time = datetime.utcnow() - timedelta(days=day)
        start_time = end_time - timedelta(days=1)

        report = await detector.generate_contradiction_report(
            time_range=(start_time, end_time),
        )
        reports.append(report)

    # Analyze trends
    daily_counts = [r.summary['total'] for r in reports]
    avg_daily = sum(daily_counts) / len(daily_counts)

    print(f"Contradiction Trends ({days} days)")
    print(f"Average daily: {avg_daily:.1f}")
    print(f"Max daily: {max(daily_counts)}")
    print(f"Min daily: {min(daily_counts)}")

    # Detect increasing trend
    recent_avg = sum(daily_counts[-7:]) / 7
    older_avg = sum(daily_counts[:-7]) / (len(daily_counts) - 7)

    if recent_avg > older_avg * 1.2:
        print("WARNING: Contradictions are increasing!")

# Usage
await analyze_trends(days=30)
```

---

## Knowledge Pruning

### Automatic Pruning

Automatically remove critical contradictions:

```python
from knowledge_engine.integrations.graphiti.contradiction_detector import (
    ContradictionSeverity,
)

# Prune critical contradictions
pruned_count = await detector.prune_contradicted_knowledge(
    entity_name="ProductA",
    severity_threshold=ContradictionSeverity.CRITICAL,
)

print(f"Pruned {pruned_count} critical contradictions")
```

### Selective Pruning

Prune specific types of contradictions:

```python
async def prune_by_confidence(
    detector: GraphitiContradictionDetector,
    min_confidence: float = 0.9,
):
    """Prune only high-confidence contradictions."""
    # Get all contradictions
    all_alerts = await detector.get_contradiction_alerts(
        severity_threshold=ContradictionSeverity.HIGH,
        unresolved_only=True,
    )

    # Filter by confidence
    high_conf = [a for a in all_alerts if a.get('confidence', 0) >= min_confidence]

    pruned = 0
    for alert in high_conf:
        count = await detector.prune_contradicted_knowledge(
            entity_name=alert['entity'],
            severity_threshold=ContradictionSeverity.HIGH,
        )
        pruned += count

    print(f"Pruned {pruned} high-confidence contradictions")
    return pruned

# Usage
await prune_by_confidence(detector, min_confidence=0.9)
```

### Safe Pruning with Rollback

Implement safe pruning with audit trail:

```python
async def safe_prune_with_backup(
    detector: GraphitiContradictionDetector,
    backup_file: str,
):
    """Safely prune contradictions with backup."""
    import json

    # 1. Backup current state
    report = await detector.generate_contradiction_report(
        include_resolved=True,
    )

    with open(backup_file, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"Backup saved to {backup_file}")

    # 2. Prune critical contradictions
    pruned = await detector.prune_contradicted_knowledge(
        severity_threshold=ContradictionSeverity.CRITICAL,
    )

    print(f"Pruned {pruned} contradictions")

    # 3. Verify results
    new_report = await detector.generate_contradiction_report()
    remaining = new_report.summary['unresolved']

    print(f"Remaining unresolved: {remaining}")

    # 4. Return backup path for potential rollback
    return backup_file

# Usage
backup = await safe_prune_with_backup(
    detector,
    backup_file="contradiction_backup_2024_01_08.json",
)

# If needed, implement restore from backup
```

---

## Monitoring and Alerts

### Real-Time Alerts

Set up real-time contradiction monitoring:

```python
import asyncio

async def monitor_contradictions(
    detector: GraphitiContradictionDetector,
    interval_seconds: int = 300,  # 5 minutes
):
    """Continuously monitor for new contradictions."""
    while True:
        try:
            # Get high-severity alerts
            alerts = await detector.get_contradiction_alerts(
                severity_threshold=ContradictionSeverity.HIGH,
                unresolved_only=True,
            )

            if alerts:
                print(f"\n⚠️  ALERT: {len(alerts)} high-severity contradictions detected!")

                for alert in alerts:
                    print(f"  Entity: {alert['entity']}")
                    print(f"  Severity: {alert['severity']}")
                    print(f"  Detected: {alert['detected_at']}")

                    # Send to monitoring system
                    await send_to_monitoring(alert)

            # Wait before next check
            await asyncio.sleep(interval_seconds)

        except Exception as e:
            print(f"Error in monitoring: {e}")
            await asyncio.sleep(60)  # Wait before retry

# Usage
asyncio.create_task(monitor_contradictions(detector))
```

### Integration with Monitoring Systems

Send alerts to external monitoring:

```python
async def send_to_monitoring(alert: dict):
    """Send alert to monitoring system."""
    import aiohttp

    webhook_url = "https://your-monitoring-system.com/webhook"

    payload = {
        "alert_type": "contradiction_detected",
        "severity": alert['severity'],
        "entity": alert['entity'],
        "timestamp": alert['detected_at'],
        "source": "graphiti_contradiction_detector",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    print("Alert sent to monitoring")
                else:
                    print(f"Failed to send alert: {response.status}")
    except Exception as e:
        print(f"Error sending alert: {e}")
```

### Metrics Collection

Collect contradiction metrics:

```python
async def collect_metrics(
    detector: GraphitiContradictionDetector,
):
    """Collect contradiction metrics for monitoring."""
    report = await detector.generate_contradiction_report()

    metrics = {
        "contradictions_total": report.summary['total'],
        "contradictions_by_severity": report.summary['by_severity'],
        "contradictions_resolved": report.summary['resolved'],
        "contradictions_unresolved": report.summary['unresolved'],
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Send to metrics system (Prometheus, CloudWatch, etc.)
    print(f"Metrics: {metrics}")

    return metrics

# Usage in a monitoring loop
while True:
    metrics = await collect_metrics(detector)
    await asyncio.sleep(60)  # Every minute
```

---

## Best Practices

### 1. Set Appropriate Thresholds

```python
# For production systems
config = GraphitiConfig(
    contradiction_enabled=True,
    contradiction_threshold=0.7,  # Balance precision/recall
)

# For testing/debugging
config = GraphitiConfig(
    contradiction_enabled=True,
    contradiction_threshold=0.5,  # Catch more potential issues
)
```

### 2. Use Human Review for Complex Cases

```python
# For high-impact decisions
if contradiction.severity == ContradictionSeverity.CRITICAL:
    # Flag for review instead of auto-resolving
    await detector.resolve_contradiction(
        contradiction_id=contradiction.contradiction_id,
        action=ResolutionAction.FLAG_FOR_REVIEW,
    )
```

### 3. Maintain Audit Trail

```python
# Always include resolution notes
await detector.resolve_contradiction(
    contradiction_id=contradiction.contradiction_id,
    action=action,
    resolution_notes=f"Resolved by {resolver_name} at {datetime.utcnow()}",
)
```

### 4. Regularly Clean Up Cache

```python
# Clear resolved contradictions older than 30 days
cleared = await detector.clear_resolved_from_cache(
    older_than_days=30,
)

print(f"Cleared {cleared} old contradictions from cache")
```

### 5. Monitor Trends

```python
# Track contradiction rates over time
if current_rate > baseline_rate * 1.5:
    print("WARNING: Contradiction rate spike detected!")
    # Investigate root cause
```

---

## Advanced Patterns

### Pattern 1: Contradiction Prevention

Prevent contradictions by checking before adding knowledge:

```python
async def safe_add_knowledge(
    bridge: GraphitiTemporalBridge,
    detector: GraphitiContradictionDetector,
    entity_name: str,
    fact: str,
):
    """Add knowledge only if it doesn't create contradictions."""
    # Check for existing contradictions
    contradictions = await detector.detect_contradictions(
        entity_name=entity_name,
    )

    if contradictions:
        print(f"Warning: {len(contradictions)} contradictions exist for {entity_name}")
        # Decide whether to proceed

    # Add the knowledge
    await bridge.add_episode(
        name=f"Knowledge: {entity_name}",
        episode_body=fact,
        reference_time=datetime.utcnow(),
    )

    # Re-check for new contradictions
    new_contradictions = await detector.detect_contradictions(
        entity_name=entity_name,
    )

    if len(new_contradictions) > len(contradictions):
        print("New contradictions created!")
        # Rollback or flag for review
```

### Pattern 2: Progressive Resolution

Resolve contradictions progressively by severity:

```python
async def progressive_resolution(detector: GraphitiContradictionDetector):
    """Resolve contradictions from lowest to highest severity."""
    severity_order = [
        ContradictionSeverity.LOW,
        ContradictionSeverity.MEDIUM,
        ContradictionSeverity.HIGH,
        ContradictionSeverity.CRITICAL,
    ]

    for severity in severity_order:
        print(f"Resolving {severity.value} contradictions...")

        alerts = await detector.get_contradiction_alerts(
            severity_threshold=severity,
            unresolved_only=True,
        )

        for alert in alerts:
            # Auto-resolve low severity, flag high severity
            if severity in [ContradictionSeverity.LOW, ContradictionSeverity.MEDIUM]:
                await detector.resolve_contradiction(
                    contradiction_id=alert['contradiction_id'],
                    action=ResolutionAction.KEEP_NEWEST,
                )
            else:
                await detector.resolve_contradiction(
                    contradiction_id=alert['contradiction_id'],
                    action=ResolutionAction.FLAG_FOR_REVIEW,
                )
```

### Pattern 3: Source-Based Resolution

Resolve based on data source reliability:

```python
async def resolve_by_source(
    detector: GraphitiContradictionDetector,
    source_reliability: dict,
):
    """Resolve contradictions based on source reliability."""
    contradictions = await detector.detect_contradictions(
        entity_name="ProductA",
    )

    for contradiction in contradictions:
        # Get sources from contradictions
        sources = extract_sources_from_contradiction(contradiction)

        # Find most reliable source
        most_reliable = max(sources, key=lambda s: source_reliability.get(s, 0))

        # Keep knowledge from most reliable source
        await resolve_contradiction_by_source(
            detector,
            contradiction.contradiction_id,
            keep_source=most_reliable,
        )
```

---

**Last Updated:** 2026-01-08
**Version:** 1.0.0
