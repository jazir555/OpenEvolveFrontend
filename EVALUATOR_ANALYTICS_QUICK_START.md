# Evaluator Analytics - Quick Start Guide

## Installation

```bash
# Install dependencies
pip install numpy scipy python-dateutil

# For PDF export (optional)
pip install weasyprint
```

## Basic Usage (5 minutes)

### 1. Initialize and Add Data

```python
from evaluator_analytics import EvaluatorAnalytics, EvaluationRecord, EvaluationStage
from datetime import datetime

analytics = EvaluatorAnalytics()

# Record an evaluation
record = EvaluationRecord(
    evaluator_id="eval_001",
    evaluation_id="solution_123",
    stage=EvaluationStage.SOLUTION_GENERATION,
    timestamp=datetime.now(),
    score=8.5,
    confidence=0.9,
    time_taken=120.0,
    criteria_scores={"quality": 8.0, "completeness": 9.0}
)
analytics.add_evaluation_record(record)
```

### 2. Get Metrics

```python
# Individual metrics
metrics = analytics.get_evaluator_metrics("eval_001")
print(f"Score: {metrics.average_score:.2f}")
print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"Consistency: {metrics.consistency_score:.2%}")

# Team metrics
team = analytics.get_team_metrics()
print(f"Total Evaluations: {team['total_evaluations']}")
```

### 3. Detect Biases

```python
biases = analytics.detect_biases("eval_001")
for bias in biases:
    print(f"{bias.bias_type.value}: {bias.severity:.2f}")
    for suggestion in bias.mitigation_suggestions:
        print(f"  - {suggestion}")
```

### 4. Generate Reports

```python
from evaluator_reporter import EvaluationReporter, ReportConfig, ReportType, ReportFormat

reporter = EvaluationReporter(analytics)

# Individual performance report
config = ReportConfig(
    report_type=ReportType.INDIVIDUAL_PERFORMANCE,
    format=ReportFormat.HTML,
    evaluator_ids=["eval_001"]
)
report = reporter.generate_report(config)
reporter.export_report(report, ReportFormat.HTML, "report.html")
```

## Common Tasks

### Compare Evaluators

```python
comparison = analytics.compare_evaluators(["eval_001", "eval_002", "eval_003"])
for metric, ranking in comparison['ranking'].items():
    print(f"\n{metric}:")
    for rank, (eval_id, value) in enumerate(ranking, 1):
        print(f"  {rank}. {eval_id}: {value:.2f}")
```

### Analyze Trends

```python
trends = analytics.analyze_performance_trends("eval_001")
print(f"Score Trend: {trends['score_trend']}")
print(f"Time Efficiency: {trends['time_trend']}")
```

### Generate Quality Report

```python
quality = analytics.generate_quality_report("eval_001")
print(f"Quality Score: {quality['overall_quality_score']:.2%}")

# Quality components
for component, value in quality['quality_components'].items():
    print(f"  {component}: {value:.2%}")
```

### Export All Formats

```python
# Generate once
report = reporter.generate_report(config)

# Export to multiple formats
reporter.export_report(report, ReportFormat.JSON, "report.json")
reporter.export_report(report, ReportFormat.CSV, "report.csv")
reporter.export_report(report, ReportFormat.HTML, "report.html")
reporter.export_report(report, ReportFormat.MARKDOWN, "report.md")
```

## Report Types

### Individual Performance
```python
config = ReportConfig(
    report_type=ReportType.INDIVIDUAL_PERFORMANCE,
    evaluator_ids=["eval_001"]
)
```

### Team Overview
```python
config = ReportConfig(
    report_type=ReportType.TEAM_OVERVIEW,
    evaluator_ids=["eval_001", "eval_002", "eval_003"]
)
```

### Bias Analysis
```python
config = ReportConfig(
    report_type=ReportType.BIAS_ANALYSIS,
    evaluator_ids=["eval_001"]
)
```

### Trend Analysis
```python
config = ReportConfig(
    report_type=ReportType.TREND_ANALYSIS,
    evaluator_ids=["eval_001"]
)
```

### Comparison
```python
config = ReportConfig(
    report_type=ReportType.COMPARISON,
    evaluator_ids=["eval_001", "eval_002"]
)
```

### Quality Gate
```python
config = ReportConfig(
    report_type=ReportType.QUALITY_GATE,
    evaluator_ids=["eval_001", "eval_002"]
)
```

## Integration Examples

### With Evaluator Team Coordinator

```python
class EvaluatorTeamCoordinator:
    def __init__(self):
        self.analytics = EvaluatorAnalytics()

    async def assign_evaluation(self, evaluator_id, task):
        result = await self.perform_evaluation(evaluator_id, task)

        # Track the evaluation
        record = EvaluationRecord(
            evaluator_id=evaluator_id,
            evaluation_id=task.id,
            stage=task.stage,
            timestamp=datetime.now(),
            score=result.score,
            confidence=result.confidence,
            time_taken=result.duration,
            criteria_scores=result.criteria
        )
        self.analytics.add_evaluation_record(record)

        return result
```

### With Quality Gate

```python
def check_evaluator_quality(evaluator_id: str) -> bool:
    metrics = analytics.get_evaluator_metrics(evaluator_id)

    # Quality thresholds
    if metrics.accuracy < 0.7:
        return False
    if metrics.reliability_score < 0.7:
        return False
    if metrics.consistency_score < 0.7:
        return False

    return True
```

### Automated Weekly Reports

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('cron', day_of_week='mon', hour=9)
async def weekly_reports():
    for eval_id in get_all_evaluators():
        config = ReportConfig(
            report_type=ReportType.INDIVIDUAL_PERFORMANCE,
            format=ReportFormat.HTML,
            evaluator_ids=[eval_id]
        )
        report = reporter.generate_report(config)
        reporter.export_report(report, ReportFormat.HTML,
                              f"reports/{eval_id}_weekly.html")

scheduler.start()
```

## Key Metrics Reference

### EvaluatorMetrics
- `total_evaluations`: Number of evaluations performed
- `average_score`: Mean score (0-10 typical)
- `average_confidence`: Mean confidence (0-1)
- `average_time`: Mean duration in seconds
- `accuracy`: Evaluation accuracy (0-1)
- `consistency_score`: Score consistency (0-1)
- `reliability_score`: Inter-rater reliability (0-1)
- `evaluation_frequency`: Evaluations per day

### Bias Types
- `LENIENCY`: Consistently higher scores
- `SEVERITY`: Consistently lower scores
- `CENTRAL_TENDENCY`: Clustering around middle
- `TEMPORAL`: Time-based patterns
- `HALO_EFFECT`: Overall impression bias
- `CONFIRMATION`: Preexisting beliefs
- `SUBJECT_MATTER`: Domain-specific bias

## Quality Score Components

The overall quality score is calculated from:
- **Accuracy**: 30% weight
- **Consistency**: 20% weight
- **Reliability**: 20% weight
- **Confidence**: 10% weight
- **Bias-free**: 20% weight

Quality Levels:
- **Excellent**: 90%+
- **Good**: 75-90%
- **Acceptable**: 60-75%
- **Needs Improvement**: <60%

## Testing

```bash
# Run all tests
pytest test_evaluator_analytics.py -v

# Run specific test class
pytest test_evaluator_analytics.py::TestEvaluatorAnalytics -v

# Run with coverage
pytest test_evaluator_analytics.py --cov=evaluator_analytics --cov=evaluator_reporter
```

## Troubleshooting

### Issue: Insufficient Data for Bias Detection
**Solution**: Wait for at least 5 evaluations per evaluator

### Issue: High Memory Usage
**Solution**: Periodically export and clear old data
```python
analytics.export_analytics_to_file("backup.json")
analytics.evaluation_records.clear()
```

### Issue: Slow Report Generation
**Solution**: Reports are cached for 1 hour
```python
# First call: slow
report = reporter.generate_report(config)
# Subsequent calls: fast (cached)
report = reporter.generate_report(config)
```

## Best Practices

1. **Record Every Evaluation**: Track all evaluations for comprehensive analytics
2. **Regular Bias Checks**: Run bias detection weekly
3. **Performance-Based Assignment**: Use metrics to select best evaluators
4. **Automated Reports**: Schedule weekly/monthly performance reports
5. **Quality Gates**: Implement threshold-based approval
6. **Data Backup**: Regularly export analytics data

## Support

For detailed documentation, see:
- `EVALUATOR_ANALYTICS_COMPLETE.md` - Complete guide
- `EVALUATOR_ANALYTICS_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `test_evaluator_analytics.py` - Usage examples

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Add evaluation | `analytics.add_evaluation_record(record)` | Track evaluation |
| Get metrics | `analytics.get_evaluator_metrics(id)` | Performance data |
| Detect biases | `analytics.detect_biases(id)` | Bias analysis |
| Compare evaluators | `analytics.compare_evaluators(ids)` | Side-by-side |
| Analyze trends | `analytics.analyze_performance_trends(id)` | Over time |
| Quality report | `analytics.generate_quality_report(id)` | Quality score |
| Generate report | `reporter.generate_report(config)` | Create report |
| Export report | `reporter.export_report(report, format)` | Save file |

---

**Version**: 1.0
**Date**: 2025-01-04
**Status**: Production Ready
