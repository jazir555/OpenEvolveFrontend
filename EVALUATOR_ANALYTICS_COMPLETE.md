# Evaluator Analytics and Reporting - Complete Guide

## Overview

The Evaluator Analytics and Reporting system provides comprehensive performance tracking, bias detection, and multi-format reporting capabilities for the OpenEvolve Evaluator Team. This system enables data-driven decision making and continuous improvement of evaluation quality.

## Table of Contents

1. [Architecture](#architecture)
2. [Components](#components)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Features](#core-features)
6. [API Reference](#api-reference)
7. [Integration Guide](#integration-guide)
8. [Testing](#testing)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Evaluator Analytics                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ EvaluationRecord │────────▶│ EvaluatorMetrics │         │
│  └──────────────────┘         └──────────────────┘         │
│           │                            │                    │
│           ▼                            ▼                    │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ BiasDetector     │         │EvaluatorAnalytics│         │
│  └──────────────────┘         └──────────────────┘         │
│           │                            │                    │
│           ▼                            ▼                    │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ EvaluationReporter│────────▶│  Multi-Format   │         │
│  └──────────────────┘         │     Export      │         │
│                               └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Collection**: EvaluationRecord instances capture each evaluation
2. **Processing**: EvaluatorAnalytics calculates metrics and detects biases
3. **Analysis**: Statistical analysis identifies trends and patterns
4. **Reporting**: EvaluationReporter generates formatted reports
5. **Export**: Reports exported in multiple formats (JSON, CSV, HTML, PDF, Markdown)

## Components

### 1. EvaluationRecord

Data structure representing a single evaluation.

**Attributes**:
- `evaluator_id`: Unique identifier for evaluator
- `evaluation_id`: Unique identifier for evaluation
- `stage`: EvaluationStage enum value
- `timestamp`: datetime of evaluation
- `score`: Numerical score (0-10 typical)
- `confidence`: Evaluator confidence level (0-1)
- `time_taken`: Evaluation duration in seconds
- `criteria_scores`: Dict of criterion-specific scores
- `feedback`: Optional text feedback
- `metadata`: Additional evaluation metadata

**Example**:
```python
from evaluator_analytics import EvaluationRecord, EvaluationStage
from datetime import datetime

record = EvaluationRecord(
    evaluator_id="eval_001",
    evaluation_id="solution_123",
    stage=EvaluationStage.SOLUTION_GENERATION,
    timestamp=datetime.now(),
    score=8.5,
    confidence=0.92,
    time_taken=145.3,
    criteria_scores={
        "quality": 8.0,
        "completeness": 9.0,
        "innovation": 8.5
    },
    feedback="Strong solution with minor issues",
    metadata={"domain": "machine_learning"}
)
```

### 2. EvaluatorMetrics

Comprehensive metrics for an evaluator's performance.

**Key Metrics**:
- `total_evaluations`: Total number of evaluations performed
- `average_score`: Mean score across all evaluations
- `average_confidence`: Mean confidence level
- `average_time`: Mean evaluation duration
- `accuracy`: Evaluation accuracy score
- `consistency_score`: Score consistency (0-1)
- `reliability_score`: Inter-rater reliability (0-1)
- `bias_scores`: Dict of detected biases
- `stage_performance`: Per-stage performance metrics
- `evaluation_frequency`: Evaluations per day

**Example**:
```python
metrics = analytics.get_evaluator_metrics("eval_001")
print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"Consistency: {metrics.consistency_score:.2%}")
print(f"Reliability: {metrics.reliability_score:.2%}")
```

### 3. BiasDetector

Statistical bias detection using multiple algorithms.

**Bias Types Detected**:
- **Leniency Bias**: Consistently higher scores than team
- **Severity Bias**: Consistently lower scores than team
- **Central Tendency**: Excessive clustering around middle scores
- **Temporal Bias**: Time-based patterns in scoring
- **Halo Effect**: Overall impression bias
- **Confirmation Bias**: Confirming preexisting beliefs
- **Subject Matter Bias**: Domain-specific biases

**Methods**:
```python
# Detect specific bias
has_bias, severity, description = bias_detector.detect_leniency_bias(
    evaluator_scores=[8.5, 9.0, 8.8, 9.2],
    team_scores=[7.0, 7.5, 6.8, 7.2]
)

# Generate complete bias profile
profiles = bias_detector.generate_bias_profile(
    evaluator_id="eval_001",
    records=evaluator_records,
    all_records=team_records
)
```

### 4. EvaluatorAnalytics

Main analytics engine for tracking and analysis.

**Key Features**:
- Real-time metrics calculation
- Individual and team analytics
- Trend analysis
- Bias detection
- Performance comparison
- Quality scoring

**Example Usage**:
```python
from evaluator_analytics import EvaluatorAnalytics

# Initialize
analytics = EvaluatorAnalytics(knowledge_base=kb)

# Add evaluation record
analytics.add_evaluation_record(record)

# Get metrics
metrics = analytics.get_evaluator_metrics("eval_001")

# Analyze trends
trends = analytics.analyze_performance_trends("eval_001")

# Compare evaluators
comparison = analytics.compare_evaluators(["eval_001", "eval_002"])

# Detect biases
biases = analytics.detect_biases("eval_001")

# Generate quality report
quality_report = analytics.generate_quality_report("eval_001")
```

### 5. EvaluationReporter

Multi-format report generation and export.

**Report Types**:
- `INDIVIDUAL_PERFORMANCE`: Single evaluator detailed report
- `TEAM_OVERVIEW`: Team-wide metrics and rankings
- `BIAS_ANALYSIS`: Comprehensive bias analysis
- `TREND_ANALYSIS`: Performance trends over time
- `COMPARISON`: Side-by-side evaluator comparison
- `QUALITY_GATE`: Quality threshold assessment
- `CUSTOM`: Custom report configuration

**Export Formats**:
- JSON: Machine-readable format
- CSV: Spreadsheet-compatible format
- HTML: Interactive web format with styling
- PDF: Professional document format (requires additional dependencies)
- Markdown: Documentation format

**Example**:
```python
from evaluator_reporter import EvaluationReporter, ReportConfig, ReportType, ReportFormat

# Create reporter
reporter = EvaluationReporter(analytics)

# Configure report
config = ReportConfig(
    report_type=ReportType.INDIVIDUAL_PERFORMANCE,
    format=ReportFormat.HTML,
    evaluator_ids=["eval_001"],
    include_charts=True,
    include_recommendations=True,
    include_biases=True,
    include_trends=True
)

# Generate report
report = reporter.generate_report(config)

# Export
html_report = reporter.export_report(report, ReportFormat.HTML, "report.html")
```

## Installation

### Requirements

```
numpy>=1.21.0
scipy>=1.7.0
python-dateutil>=2.8.0
```

### Optional Dependencies

For PDF export:
```
weasyprint>=52.0  # or
pdfkit>=1.0.0
```

For enhanced visualizations:
```
matplotlib>=3.5.0
plotly>=5.0.0
```

### Setup

```bash
# Install core dependencies
pip install numpy scipy python-dateutil

# Install optional dependencies
pip install weasyprint matplotlib plotly

# Copy files to your project
cp evaluator_analytics.py /path/to/project/
cp evaluator_reporter.py /path/to/project/
cp test_evaluator_analytics.py /path/to/project/
```

## Quick Start

### Basic Usage

```python
from evaluator_analytics import EvaluatorAnalytics, EvaluationRecord, EvaluationStage
from evaluator_reporter import EvaluationReporter, ReportConfig, ReportType, ReportFormat
from datetime import datetime

# 1. Initialize analytics
analytics = EvaluatorAnalytics()

# 2. Add evaluation records
record = EvaluationRecord(
    evaluator_id="eval_001",
    evaluation_id="sol_123",
    stage=EvaluationStage.SOLUTION_GENERATION,
    timestamp=datetime.now(),
    score=8.5,
    confidence=0.9,
    time_taken=120.0,
    criteria_scores={"quality": 8.0, "completeness": 9.0}
)
analytics.add_evaluation_record(record)

# 3. Get metrics
metrics = analytics.get_evaluator_metrics("eval_001")
print(f"Evaluations: {metrics.total_evaluations}")
print(f"Average Score: {metrics.average_score:.2f}")

# 4. Generate report
reporter = EvaluationReporter(analytics)
config = ReportConfig(
    report_type=ReportType.INDIVIDUAL_PERFORMANCE,
    format=ReportFormat.HTML,
    evaluator_ids=["eval_001"]
)
report = reporter.generate_report(config)

# 5. Export report
reporter.export_report(report, ReportFormat.HTML, "eval_001_report.html")
```

### Integration with Existing Workflow

```python
# In your evaluation workflow
async def evaluate_solution(evaluator_id, solution_id):
    # ... perform evaluation ...

    # Record evaluation
    record = EvaluationRecord(
        evaluator_id=evaluator_id,
        evaluation_id=solution_id,
        stage=EvaluationStage.SOLUTION_GENERATION,
        timestamp=datetime.now(),
        score=score,
        confidence=confidence,
        time_taken=time.time() - start_time,
        criteria_scores=criteria_scores,
        feedback=feedback
    )

    # Add to analytics
    analytics.add_evaluation_record(record)

    # Check for biases periodically
    if evaluator_needs_bias_check(evaluator_id):
        biases = analytics.detect_biases(evaluator_id)
        if biases:
            notify_evaluator_biases(evaluator_id, biases)
```

## Core Features

### 1. Performance Tracking

**Real-time Metrics**:
- Automatic metric calculation on each evaluation
- Individual and team-level tracking
- Stage-specific performance analysis

**Example**:
```python
# Get real-time metrics
metrics = analytics.get_evaluator_metrics("eval_001")

# Monitor specific metrics
if metrics.accuracy < 0.75:
    schedule_training("eval_001", "accuracy_improvement")
```

### 2. Bias Detection and Mitigation

**Statistical Detection**:
- Uses t-tests, ANOVA, and other statistical methods
- Configurable confidence thresholds
- Multiple bias type detection

**Bias Profiles**:
```python
biases = analytics.detect_biases("eval_001")

for bias in biases:
    print(f"Bias Type: {bias.bias_type.value}")
    print(f"Severity: {bias.severity:.2f}")
    print(f"Description: {bias.description}")
    print(f"Mitigation Suggestions:")
    for suggestion in bias.mitigation_suggestions:
        print(f"  - {suggestion}")
```

### 3. Trend Analysis

**Performance Trends**:
```python
trends = analytics.analyze_performance_trends("eval_001")

print(f"Score Trend: {trends['score_trend']}")
print(f"Slope: {trends['score_slope']:.4f}")

# Interpret trends
if trends['score_trend'] == 'improving':
    print("Evaluator shows consistent improvement")
elif trends['score_trend'] == 'declining':
    print("Evaluator may need support")
```

### 4. Quality Scoring

**Comprehensive Quality Assessment**:
```python
quality_report = analytics.generate_quality_report("eval_001")

quality_score = quality_report['overall_quality_score']

if quality_score >= 0.9:
    quality_level = "Excellent"
elif quality_score >= 0.75:
    quality_level = "Good"
elif quality_score >= 0.6:
    quality_level = "Acceptable"
else:
    quality_level = "Needs Improvement"

print(f"Quality Level: {quality_level} ({quality_score:.2%})")
```

### 5. Comparison Reports

**Evaluator Comparison**:
```python
comparison = analytics.compare_evaluators(
    ["eval_001", "eval_002", "eval_003"]
)

# View rankings
for metric, ranking in comparison['ranking'].items():
    print(f"\n{metric}:")
    for rank, (eval_id, value) in enumerate(ranking, 1):
        print(f"  {rank}. {eval_id}: {value:.2f}")
```

### 6. Multi-Format Export

**Export to Different Formats**:
```python
# Generate report once
report = reporter.generate_report(config)

# Export to multiple formats
reporter.export_report(report, ReportFormat.JSON, "report.json")
reporter.export_report(report, ReportFormat.CSV, "report.csv")
reporter.export_report(report, ReportFormat.HTML, "report.html")
reporter.export_report(report, ReportFormat.MARKDOWN, "report.md")
```

## API Reference

### EvaluatorAnalytics

#### Methods

**add_evaluation_record(record: EvaluationRecord) -> None**
Add an evaluation record to analytics.

**get_evaluator_metrics(evaluator_id: str) -> Optional[EvaluatorMetrics]**
Get metrics for a specific evaluator.

**get_team_metrics() -> Dict[str, Any]**
Get team-level metrics.

**calculate_consistency_score(records: List[EvaluationRecord]) -> float**
Calculate consistency score (0-1).

**calculate_reliability_score(evaluator_id: str, records: List[EvaluationRecord]) -> float**
Calculate inter-rater reliability score (0-1).

**analyze_performance_trends(evaluator_id: str, window_size: int = 10) -> Dict[str, Any]**
Analyze performance trends over time.

**compare_evaluators(evaluator_ids: List[str]) -> Dict[str, Any]**
Compare multiple evaluators.

**detect_biases(evaluator_id: str) -> List[BiasProfile]**
Detect biases for an evaluator.

**get_top_performers(metric: str = "accuracy", top_n: int = 5) -> List[Tuple[str, float]]**
Get top performers by metric.

**get_stage_performance(stage: EvaluationStage) -> Dict[str, Any]**
Get performance for specific stage.

**generate_quality_report(evaluator_id: str) -> Dict[str, Any]**
Generate comprehensive quality report.

**export_analytics() -> Dict[str, Any]**
Export all analytics data.

**load_analytics(data: Dict[str, Any]) -> None**
Load analytics from export.

### EvaluationReporter

#### Methods

**generate_report(config: ReportConfig) -> Dict[str, Any]**
Generate report based on configuration.

**export_report(report: Dict[str, Any], format: ReportFormat, output_path: Optional[str] = None) -> str**
Export report in specified format.

**get_report_summary(report: Dict[str, Any]) -> Dict[str, Any]**
Get summary of report.

**schedule_report(config: ReportConfig, schedule: str) -> str**
Schedule periodic report generation.

**clear_cache() -> None**
Clear report cache.

### ReportConfig

#### Parameters

**report_type: ReportType**
Type of report to generate.

**format: ReportFormat**
Output format for report.

**evaluator_ids: Optional[List[str]]**
List of evaluator IDs to include.

**stage: Optional[EvaluationStage]**
Filter by evaluation stage.

**start_date: Optional[datetime]**
Filter by start date.

**end_date: Optional[datetime]**
Filter by end date.

**include_charts: bool = True**
Include charts in report.

**include_recommendations: bool = True**
Include recommendations in report.

**include_biases: bool = True**
Include bias analysis in report.

**include_trends: bool = True**
Include trend analysis in report.

**comparison_baseline: Optional[str]**
Baseline evaluator for comparison.

**metrics: List[str]**
List of metrics to include.

## Integration Guide

### With Evaluator Team Coordinator

```python
# In evaluator_team_coordinator.py
from evaluator_analytics import EvaluatorAnalytics, EvaluationRecord, EvaluationStage

class EvaluatorTeamCoordinator:
    def __init__(self):
        self.analytics = EvaluatorAnalytics()

    async def assign_evaluation(self, evaluator_id, task):
        # Assign task to evaluator
        result = await self.perform_evaluation(evaluator_id, task)

        # Record evaluation
        record = EvaluationRecord(
            evaluator_id=evaluator_id,
            evaluation_id=task.id,
            stage=EvaluationStage.SOLUTION_GENERATION,
            timestamp=datetime.now(),
            score=result.score,
            confidence=result.confidence,
            time_taken=result.duration,
            criteria_scores=result.criteria_scores,
            feedback=result.feedback
        )

        self.analytics.add_evaluation_record(record)

        return result

    def get_evaluator_performance(self, evaluator_id):
        return self.analytics.get_evaluator_metrics(evaluator_id)

    def detect_biases(self):
        for eval_id in self.get_active_evaluators():
            biases = self.analytics.detect_biases(eval_id)
            if biases:
                self.handle_biases(eval_id, biases)
```

### With Quality Gate Engine

```python
# In quality_gate_engine.py
from evaluator_analytics import EvaluatorAnalytics

class QualityGateEngine:
    def __init__(self, analytics: EvaluatorAnalytics):
        self.analytics = analytics

    def evaluate_quality(self, evaluator_id: str) -> bool:
        metrics = self.analytics.get_evaluator_metrics(evaluator_id)

        # Define quality thresholds
        if metrics.accuracy < 0.7:
            return False
        if metrics.consistency_score < 0.7:
            return False
        if metrics.reliability_score < 0.7:
            return False

        return True

    def generate_quality_report(self, evaluator_id: str):
        return self.analytics.generate_quality_report(evaluator_id)
```

### With Knowledge Base

```python
# In knowledge_base.py
class KnowledgeBase:
    def store_analytics(self, analytics: EvaluatorAnalytics):
        """Persist analytics data"""
        data = analytics.export_analytics()
        self.store("evaluator_analytics", data)

    def load_analytics(self) -> EvaluatorAnalytics:
        """Load analytics data"""
        data = self.retrieve("evaluator_analytics")
        analytics = EvaluatorAnalytics(self)
        analytics.load_analytics(data)
        return analytics
```

## Testing

### Running Tests

```bash
# Run all tests
pytest test_evaluator_analytics.py -v

# Run specific test class
pytest test_evaluator_analytics.py::TestEvaluatorAnalytics -v

# Run specific test
pytest test_evaluator_analytics.py::TestEvaluatorAnalytics::test_add_evaluation_record -v

# Run with coverage
pytest test_evaluator_analytics.py --cov=evaluator_analytics --cov=evaluator_reporter
```

### Test Coverage

The test suite includes:

- **25+ tests** covering all major functionality
- **Unit tests** for each component
- **Integration tests** for end-to-end workflows
- **Edge case tests** for error handling
- **Performance tests** for large datasets

Expected pass rate: **90%+**

### Example Test

```python
def test_bias_detection_workflow():
    """Test complete bias detection and reporting workflow"""
    analytics = EvaluatorAnalytics()

    # Add biased evaluator data
    # ... (add records)

    # Detect biases
    biases = analytics.detect_biases("biased_evaluator")
    assert len(biases) > 0

    # Generate bias report
    reporter = EvaluationReporter(analytics)
    config = ReportConfig(
        report_type=ReportType.BIAS_ANALYSIS,
        format=ReportFormat.HTML
    )
    report = reporter.generate_report(config)

    # Export
    html = reporter.export_report(report, ReportFormat.HTML)
    assert "Bias Analysis" in html
```

## Best Practices

### 1. Regular Data Collection

```python
# Collect data on every evaluation
async def evaluate_with_tracking(evaluator_id, task):
    start_time = time.time()

    # Perform evaluation
    result = await perform_evaluation(evaluator_id, task)

    # Record metrics
    record = EvaluationRecord(
        evaluator_id=evaluator_id,
        evaluation_id=task.id,
        stage=task.stage,
        timestamp=datetime.now(),
        score=result.score,
        confidence=result.confidence,
        time_taken=time.time() - start_time,
        criteria_scores=result.criteria_scores,
        feedback=result.feedback
    )

    analytics.add_evaluation_record(record)
    return result
```

### 2. Periodic Bias Checks

```python
# Check for biases weekly
async def weekly_bias_check():
    for eval_id in get_all_evaluators():
        biases = analytics.detect_biases(eval_id)

        if biases:
            high_severity = [b for b in biases if b.severity > 0.7]

            if high_severity:
                notify_management(eval_id, high_severity)
                suggest_mitigation(eval_id, high_severity)
```

### 3. Performance-Based Assignment

```python
# Assign evaluators based on performance
def select_best_evaluator(task):
    candidates = get_available_evaluators()

    # Get metrics for candidates
    metrics = {
        eval_id: analytics.get_evaluator_metrics(eval_id)
        for eval_id in candidates
    }

    # Select based on accuracy and availability
    best_evaluator = max(
        candidates,
        key=lambda e: (metrics[e].accuracy, metrics[e].evaluation_frequency)
    )

    return best_evaluator
```

### 4. Automated Reporting

```python
# Generate weekly performance reports
async def generate_weekly_reports():
    reporter = EvaluationReporter(analytics)

    for eval_id in get_all_evaluators():
        config = ReportConfig(
            report_type=ReportType.INDIVIDUAL_PERFORMANCE,
            format=ReportFormat.HTML,
            evaluator_ids=[eval_id],
            include_recommendations=True
        )

        report = reporter.generate_report(config)

        # Save report
        filename = f"reports/{eval_id}_weekly_{datetime.now().strftime('%Y%m%d')}.html"
        reporter.export_report(report, ReportFormat.HTML, filename)

        # Email to evaluator
        email_report(eval_id, filename)
```

### 5. Quality Gates

```python
# Implement quality gates for critical evaluations
def quality_gate_check(evaluator_id, task):
    metrics = analytics.get_evaluator_metrics(evaluator_id)

    # Check if evaluator meets quality standards
    if task.criticality == "high":
        min_accuracy = 0.85
        min_reliability = 0.85
    else:
        min_accuracy = 0.70
        min_reliability = 0.70

    if metrics.accuracy < min_accuracy:
        return False, f"Accuracy below threshold ({metrics.accuracy:.2%} < {min_accuracy:.2%})"

    if metrics.reliability_score < min_reliability:
        return False, f"Reliability below threshold ({metrics.reliability_score:.2%} < {min_reliability:.2%})"

    return True, "Quality gate passed"
```

## Troubleshooting

### Common Issues

**Issue 1: Insufficient Data for Bias Detection**

```python
# Problem: Not enough evaluations
biases = analytics.detect_biases("eval_001")
# Returns: [] (empty list)

# Solution: Wait for more data or adjust threshold
# Minimum requirements:
# - 5 evaluations for basic bias detection
# - 10 evaluations for temporal bias
# - 20 evaluations for comprehensive analysis
```

**Issue 2: High False Positive Rate**

```python
# Problem: Too many bias detections

# Solution: Adjust confidence threshold
bias_detector = BiasDetector(confidence_threshold=0.01)  # More strict
```

**Issue 3: Memory Usage with Large Datasets**

```python
# Problem: High memory usage with many records

# Solution 1: Periodic export and clear
analytics.export_analytics_to_file("backup.json")
analytics.evaluation_records.clear()

# Solution 2: Use database-backed analytics
analytics = EvaluatorAnalytics(knowledge_base=db_kb)
```

**Issue 4: Slow Report Generation**

```python
# Problem: Reports take too long to generate

# Solution: Use caching
config = ReportConfig(...)
report = reporter.generate_report(config)  # First time: slow
report = reporter.generate_report(config)  # Subsequent: fast (cached)

# Clear cache when needed
reporter.clear_cache()
```

**Issue 5: Export Format Not Working**

```python
# Problem: PDF export fails

# Solution: Install optional dependencies
pip install weasyprint  # or pdfkit

# Or use HTML export instead
reporter.export_report(report, ReportFormat.HTML, "report.html")
```

### Performance Optimization

```python
# For large datasets
import asyncio

async def batch_add_records(records):
    """Add records in batches for better performance"""
    batch_size = 100

    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        for record in batch:
            analytics.add_evaluation_record(record)

        # Allow event loop to process
        await asyncio.sleep(0)

# Use with
await batch_add_records(large_record_list)
```

## Advanced Usage

### Custom Bias Detection

```python
class CustomBiasDetector(BiasDetector):
    def detect_custom_bias(self, records):
        """Implement custom bias detection logic"""
        # Your custom logic here
        pass

# Use custom detector
analytics = EvaluatorAnalytics()
analytics.bias_detector = CustomBiasDetector()
```

### Custom Report Sections

```python
# Extend reporter for custom sections
class CustomReporter(EvaluationReporter):
    def _generate_custom_section(self, evaluator_id):
        """Add custom report section"""
        # Your custom section logic
        return ReportSection(
            title="Custom Metrics",
            content=custom_data
        )
```

### Scheduled Reports

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

# Schedule weekly reports
@scheduler.scheduled_job('cron', day_of_week='mon', hour=9)
async def weekly_report_job():
    reporter = EvaluationReporter(analytics)

    for eval_id in get_all_evaluators():
        config = ReportConfig(
            report_type=ReportType.INDIVIDUAL_PERFORMANCE,
            format=ReportFormat.HTML,
            evaluator_ids=[eval_id]
        )

        report = reporter.generate_report(config)
        # Send report...

scheduler.start()
```

## Conclusion

The Evaluator Analytics and Reporting system provides a comprehensive solution for tracking, analyzing, and reporting on evaluator performance. With robust bias detection, multi-format reporting, and seamless integration capabilities, it enables data-driven continuous improvement of evaluation quality.

For additional support or questions, refer to the test suite for usage examples or consult the API reference above.

---

**Document Version**: 1.0
**Last Updated**: 2025-01-04
**Author**: OpenEvolve Team
