# Phase 5: UI/CLI Integration

## Overview

Phase 5 provides user interface enhancements, CLI tools, monitoring dashboards, and interactive knowledge exploration for the RAGBits integration.

## Components

### 1. CLI Tools (`ragbits_integration.ui_cli.cli`)

Command-line interface for RAGBits operations.

**Features:**
- Knowledge extraction from artifacts
- Artifact scoring with detailed output
- Historical comparison
- Evaluation dashboard generation
- Knowledge base exploration
- System statistics
- Gauntlet validation
- Trend analysis

**Usage:**

```bash
# Extract knowledge
ragbits extract --file solution.md --type solution --use-llm

# Score an artifact
ragbits score --artifact art_123 --details

# Compare with historical
ragbits compare --artifact art_123 --type solution --lookback-days 30

# Generate dashboard
ragbits dashboard --workflow workflow_123 --type workflow --output dashboard.html

# Explore knowledge base
ragbits explore --query "authentication patterns" --search-type hybrid --limit 10

# Show statistics
ragbits stats --type storage

# Validate with gauntlet
ragbits validate --artifact art_123 --requirements "req1" "req2"

# Analyze trends
ragbits trend --type solution --days 30 --category quality
```

**Output Formats:**
- `--output json`: JSON format
- `--output text`: Formatted text (default)
- `--output table`: Table format

---

### 2. Review Interface (`ragbits_integration.ui_cli.interfaces`)

Enhanced review interface with collaborative features.

**Features:**
- Inline commenting with threading
- Version comparison and diffing
- Approval/rejection tracking
- Review metrics and analytics
- Multi-format report export (Markdown, JSON, HTML)

**Usage:**

```python
from ragbits_integration.ui_cli.interfaces import ReviewInterface, ReviewStatus, CommentType

# Create review interface
review = ReviewInterface(storage_manager, knowledge_retriever)

# Create review session
session = await review.create_review_session(
    artifact_id="art_123",
    artifact_content=content,
    artifact_type="solution",
    reviewers=["alice", "bob", "charlie"]
)

# Add comments
comment = await review.add_comment(
    review_id=session.review_id,
    author="alice",
    content="Consider adding error handling",
    comment_type=CommentType.SUGGESTION,
    section="Error Handling",
    line_number=42
)

# Reply to comment
reply = await review.add_comment(
    review_id=session.review_id,
    author="bob",
    content="Good point, I'll add it",
    parent_comment_id=comment.comment_id
)

# Resolve comment
await review.resolve_comment(
    review_id=session.review_id,
    comment_id=comment.comment_id,
    resolver="alice"
)

# Submit decision
decision = await review.submit_decision(
    review_id=session.review_id,
    status=ReviewStatus.APPROVED,
    reviewer="alice",
    summary="Approved with minor conditions",
    conditions=["Add error handling", "Add unit tests"],
    approved_sections=["Authentication", "Authorization"],
    rejected_sections=[]
)

# Get summary
summary = await review.get_review_summary(session.review_id)
print(f"Total comments: {summary['total_comments']}")
print(f"Unresolved: {summary['unresolved_comments']}")

# Export report
report = await review.export_review_report(
    review_id=session.review_id,
    format="markdown",
    output_path="review_report.md"
)

# Compare versions
diff = await review.compare_versions(
    artifact_id="art_123",
    version_a="v1",
    version_b="v2"
)
print(f"Added {len(diff.added_lines)}, removed {len(diff.removed_lines)} lines")
```

**Review Status:**
- `PENDING`: Review not started
- `IN_PROGRESS`: Review in progress
- `APPROVED`: All reviewers approved
- `REJECTED`: Review rejected
- `NEEDS_REVISION`: Changes requested

**Comment Types:**
- `SUGGESTION`: Improvement suggestion
- `ISSUE`: Problem identified
- `QUESTION`: Clarification needed
- `APPROVAL`: Approval expressed
- `GENERAL`: General comment

---

### 3. Monitoring Dashboard (`ragbits_integration.ui_cli.monitoring`)

Real-time monitoring dashboard for system health and metrics.

**Features:**
- Metric collection and visualization
- Alert generation and notification
- System health monitoring
- Performance tracking
- Resource usage monitoring
- HTML dashboard export

**Tracked Metrics:**

| Metric Name | Type | Description |
|-------------|------|-------------|
| `artifacts_stored_total` | Counter | Total artifacts stored |
| `artifacts_stored_rate` | Gauge | Artifacts per minute |
| `queries_total` | Counter | Total queries |
| `queries_rate` | Gauge | Queries per minute |
| `query_latency_ms` | Histogram | Query latency |
| `extraction_time_ms` | Histogram | Extraction time |
| `vector_index_size` | Gauge | Indexed document count |
| `cache_hit_rate` | Gauge | Cache hit percentage |
| `active_review_sessions` | Gauge | Active reviews |
| `llm_requests_total` | Counter | Total LLM requests |
| `llm_latency_ms` | Histogram | LLM request latency |
| `storage_used_mb` | Gauge | Storage used |
| `memory_usage_mb` | Gauge | Memory usage |

**Usage:**

```python
from ragbits_integration.ui_cli.monitoring import (
    MonitoringDashboard,
    MetricType,
    AlertSeverity
)

# Create dashboard
dashboard = MonitoringDashboard(storage_manager)

# Record metrics
dashboard.record_metric("artifacts_stored_total", 150.0)
dashboard.record_metric("queries_total", 1200.0)
dashboard.record_metric("query_latency_ms", 245.0, labels={"endpoint": "/search"})

# Define alert
dashboard.define_alert(
    alert_id="high_query_latency",
    name="High Query Latency",
    description="Query latency exceeds 1 second",
    metric_name="query_latency_ms",
    condition="> 1000",
    severity=AlertSeverity.WARNING
)

# Get current metric value
metric = dashboard.get_metric("query_latency_ms")
print(f"Current latency: {metric.get_current_value()} ms")
print(f"Average (5min): {metric.get_average(duration_minutes=5)} ms")

# Check active alerts
active_alerts = dashboard.get_active_alerts()
for alert in active_alerts:
    print(f"ALERT: {alert.name} - {alert.description}")

# Update system health
health = await dashboard.update_system_health()
print(f"System status: {health.status}")
print(f"Components: {health.components}")
print(f"Issues: {health.issues}")

# Generate HTML dashboard
html = await dashboard.generate_dashboard_html(
    duration_minutes=60,
    include_alerts=True
)
with open("dashboard.html", "w") as f:
    f.write(html)

# Export metrics as JSON
json_str = dashboard.export_metrics_json(duration_minutes=60)
```

**System Health Status:**
- `healthy`: All components operating normally
- `degraded`: Some components experiencing issues
- `unhealthy`: Critical problems detected

**Alert Severity:**
- `INFO`: Informational
- `WARNING`: Warning condition
- `ERROR`: Error condition
- `CRITICAL`: Critical failure

---

### 4. Knowledge Explorer (`ragbits_integration.ui_cli.exploration`)

Interactive knowledge exploration with advanced search and filtering.

**Features:**
- Multi-strategy search (semantic, keyword, hybrid, exact)
- Advanced filtering and faceting
- Knowledge graph visualization
- Entity relationship exploration
- Search history tracking
- Multi-format export (JSON, Markdown, CSV)

**Usage:**

```python
from ragbits_integration.ui_cli.exploration import (
    KnowledgeExplorer,
    SearchStrategy,
    SortOrder,
    EntityType,
    SearchFilter
)

# Create explorer
explorer = KnowledgeExplorer(storage_manager, rag_engine)

# Basic search
results, metadata = await explorer.search(
    query="authentication patterns",
    strategy=SearchStrategy.HYBRID,
    limit=10
)
print(f"Found {metadata['total_count']} results")

# Advanced search with filters
filter = SearchFilter(
    entity_types=[EntityType.SOLUTION_PATTERN, EntityType.BEST_PRACTICE],
    min_quality_score=0.7,
    tags=["security", "authentication"],
    artifact_types=["solution"],
    stage="stage_3"
)

results, metadata = await explorer.search(
    query="JWT authentication",
    strategy=SearchStrategy.SEMANTIC,
    filters=filter,
    limit=20,
    offset=0,
    sort_by=SortOrder.QUALITY_DESC
)

# Process results
for result in results:
    print(f"[{result.entity_type.value}] {result.entity_id}")
    print(f"Relevance: {result.relevance_score:.2f}")
    print(f"Quality: {result.quality_score:.2f}")
    print(f"Content: {result.content[:200]}...")
    print("Highlights:")
    for highlight in result.highlights:
        print(f"  - {highlight}")
    print()

# Get facets for filtering
facets = await explorer.get_facets()
print("Entity types:")
for entity_type, count in facets["entity_type"].items():
    print(f"  {entity_type}: {count}")

# Get entity details
details = await explorer.get_entity_details("entity_123")

# Find similar entities
similar = await explorer.get_similar_entities(
    entity_id="entity_123",
    limit=5
)

# Get knowledge graph
graph = await explorer.get_knowledge_graph(
    center_entity_id="entity_123",
    max_depth=2,
    max_nodes=50
)
print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

# Export results
json_str = explorer.export_search_results(
    results,
    metadata,
    format="json",
    output_path="search_results.json"
)

md_str = explorer.export_search_results(
    results,
    metadata,
    format="markdown",
    output_path="search_results.md"
)

csv_str = explorer.export_search_results(
    results,
    metadata,
    format="csv",
    output_path="search_results.csv"
)

# View search history
history = explorer.get_search_history()
for h in history:
    print(f"{h['timestamp']}: {h['query']} ({h['strategy']})")
```

**Search Strategies:**
- `SEMANTIC`: Vector-based semantic search
- `KEYWORD`: Full-text keyword search
- `HYBRID`: Combined semantic + keyword
- `EXACT`: Exact phrase matching

**Sort Orders:**
- `RELEVANCE`: By relevance score (default)
- `DATE_NEWEST`: Newest first
- `DATE_OLDEST`: Oldest first
- `QUALITY_ASC`: Quality ascending
- `QUALITY_DESC`: Quality descending

---

## Integration Examples

### Complete Workflow: Artifact Review

```python
from ragbits_integration.ui_cli import (
    ReviewInterface,
    MonitoringDashboard,
    KnowledgeExplorer
)

# Setup
review = ReviewInterface(storage_manager, retriever)
dashboard = MonitoringDashboard(storage_manager)
explorer = KnowledgeExplorer(storage_manager, rag_engine)

# 1. Create review session
session = await review.create_review_session(
    artifact_id="art_123",
    artifact_content=content,
    artifact_type="solution",
    reviewers=["alice", "bob"]
)

# 2. Find similar solutions for context
similar, _ = await explorer.search(
    query="authentication system",
    limit=5
)

# 3. Record metrics
dashboard.record_metric("active_review_sessions", 1.0)

# 4. Reviewers add comments
await review.add_comment(
    review_id=session.review_id,
    author="alice",
    content="Consider adding MFA",
    comment_type=CommentType.SUGGESTION
)

# 5. Submit decision
await review.submit_decision(
    review_id=session.review_id,
    status=ReviewStatus.APPROVED,
    reviewer="alice",
    summary="Approved with MFA requirement"
)

# 6. Update metrics
dashboard.record_metric("active_review_sessions", 0.0)

# 7. Generate reports
report = await review.export_review_report(
    review_id=session.review_id,
    format="html",
    output_path="review.html"
)

html = await dashboard.generate_dashboard_html()
```

### Monitoring and Alerting

```python
from ragbits_integration.ui_cli.monitoring import MonitoringDashboard, AlertSeverity

dashboard = MonitoringDashboard()

# Setup alerts for key metrics
dashboard.define_alert(
    alert_id="slow_queries",
    name="Slow Queries",
    description="Query latency > 1s",
    metric_name="query_latency_ms",
    condition="> 1000",
    severity=AlertSeverity.WARNING
)

dashboard.define_alert(
    alert_id="low_cache_hit",
    name="Low Cache Hit Rate",
    description="Cache hit rate < 50%",
    metric_name="cache_hit_rate",
    condition="< 50",
    severity=AlertSeverity.ERROR
)

dashboard.define_alert(
    alert_id="high_memory",
    name="High Memory Usage",
    description="Memory > 4GB",
    metric_name="memory_usage_mb",
    condition="> 4096",
    severity=AlertSeverity.CRITICAL
)

# Record metrics (in a loop, typically)
for i in range(100):
    latency = measure_query_latency()
    dashboard.record_metric("query_latency_ms", latency)

    # Alert will trigger automatically if condition met
```

---

## Testing

```bash
# Run all Phase 5 tests
python -m pytest ragbits_integration/ui_cli/tests/test_phase5_ui_cli.py -v

# Run specific test class
python -m pytest ragbits_integration/ui_cli/tests/test_phase5_ui_cli.py::TestReviewInterface -v

# Run with coverage
python -m pytest --cov=ragbits_integration.ui_cli ragbits_integration/ui_cli/tests/ -v
```

---

## Architecture

```
ui_cli/
├── cli/
│   └── ragbits_cli.py          # Command-line interface
├── interfaces/
│   └── review_interface.py     # Review interface
├── monitoring/
│   └── dashboard.py            # Monitoring dashboard
├── exploration/
│   └── knowledge_explorer.py   # Knowledge explorer
└── tests/
    └── test_phase5_ui_cli.py   # Comprehensive tests
```

---

## Dependencies

Phase 5 components depend on earlier phases:

- **Phase 1**: Intermediary storage and document search
- **Phase 2**: Agent coordination (for review context)
- **Phase 3**: Evaluation framework (for scoring)
- **Phase 4**: Enhanced knowledge base (for exploration)

---

## API Reference

See individual component docstrings for detailed API documentation:

```python
from ragbits_integration.ui_cli.cli import RAGBitsCLI
from ragbits_integration.ui_cli.interfaces import ReviewInterface
from ragbits_integration.ui_cli.monitoring import MonitoringDashboard
from ragbits_integration.ui_cli.exploration import KnowledgeExplorer

# Help
help(RAGBitsCLI)
help(ReviewInterface)
help(MonitoringDashboard)
help(KnowledgeExplorer)
```

---

## Status

✅ **COMPLETE**

All Phase 5 components implemented and tested:
- ✅ CLI tools with 8 commands
- ✅ Review interface with collaborative features
- ✅ Monitoring dashboard with alerts
- ✅ Knowledge explorer with multi-strategy search
- ✅ Comprehensive test coverage
