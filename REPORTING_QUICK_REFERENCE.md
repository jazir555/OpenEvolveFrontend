# Reporting System Quick Reference

## Installation

```bash
# Full installation
pip install -r reporting_requirements.txt

# Minimum (HTML, JSON, Markdown only)
# No pip install needed - uses Python stdlib only!

# PDF support
pip install reportlab

# Excel support
pip install openpyxl

# Template system
pip install jinja2
```

## Basic Usage

### Generate a Report

```python
from report_generator import generate_quick_report

# HTML report
report_path = generate_quick_report(plan, format='html')

# PDF report
report_path = generate_quick_report(plan, format='pdf')

# Excel report
report_path = generate_quick_report(plan, format='xlsx')
```

### Custom Configuration

```python
from report_generator import ReportGenerator

generator = ReportGenerator({
    'output_dir': './reports',
    'include_charts': True,
    'branding': {
        'primary_color': '#2C3E50'
    }
})

report_path = generator.generate_report(
    plan,
    output_format='html',
    template_name='custom_template'
)
```

## Template System

### Use Built-in Template

```python
from report_templates import TemplateManager, ReportContextBuilder

manager = TemplateManager()
builder = ReportContextBuilder()

context = builder.build_executive_summary_context(
    plan,
    insights=["Great progress!"],
    recommendations=["Focus on testing"]
)

rendered = manager.render_template(
    'executive_summary',
    context,
    use_builtin=True
)
```

### Create Custom Template

```python
# Create template
custom_template = """
# {{ title }}

Progress: {{ completion }}%
"""

manager.create_custom_template('my_template', custom_template)

# Use it
rendered = manager.render_template('my_template', {'title': 'Project', 'completion': 75})
```

## Scheduled Reports

### Daily Report

```python
from scheduled_reports import ScheduledReportManager, create_daily_report

manager = ScheduledReportManager({
    'smtp': {
        'host': 'smtp.gmail.com',
        'port': 587,
        'username': 'your@gmail.com',
        'password': 'app-password'
    }
})

daily = create_daily_report(
    'daily-summary',
    'executive_summary',
    ['manager@company.com'],
    format='pdf'
)

manager.add_schedule(daily)

# Start scheduler
def get_plans(filters):
    return engine.get_all_plans()

manager.start(get_plans)
```

### Weekly Report

```python
from scheduled_reports import create_weekly_report

weekly = create_weekly_report(
    'weekly-detailed',
    'detailed_report',
    ['team@company.com'],
    format='html'
)
```

### Custom Schedule (Cron)

```python
from scheduled_reports import ScheduleConfig

# Every Monday at 9 AM
custom = ScheduleConfig(
    schedule_id='monday-report',
    report_type='progress',
    schedule_expression='0 9 * * 1',
    recipients=['team@company.com']
)
```

## Common Tasks

### Generate All Formats

```python
formats = ['pdf', 'html', 'md', 'xlsx', 'json']
for fmt in formats:
    try:
        path = generator.generate_report(plan, output_format=fmt)
        print(f"{fmt}: {path}")
    except Exception as e:
        print(f"{fmt}: Failed - {e}")
```

### Executive Summary

```python
summary_path = generator.generate_executive_summary(
    plan,
    output_format='pdf'
)
```

### Aggregated Report

```python
plans = [plan1, plan2, plan3]

agg_path = generator.generate_aggregated_report(
    plans,
    output_format='html',
    group_by='strategy'
)
```

### Custom Branding

```python
generator = ReportGenerator({
    'branding': {
        'primary_color': '#0066CC',
        'secondary_color': '#004499',
        'logo': './company-logo.png',
        'company_name': 'Acme Corporation'
    }
})
```

## Template Filters

```python
# In templates
{{ value|percentage }}        # 75.5%
{{ value|date }}              # 2025-01-03
{{ value|number }}            # 75.50
{{ status|status_badge }}     # HTML badge
{{ progress|progress_bar }}   # ASCII bar
{{ text|truncate(50) }}       # Truncate text
{{ text|word_wrap(80) }}      # Wrap text
```

## Schedule Management

```python
# List schedules
schedules = manager.list_schedules()

# Enable/disable
manager.disable_schedule('daily-001')
manager.enable_schedule('daily-001')

# Get status
status = manager.get_schedule_status('daily-001')

# Remove
manager.remove_schedule('daily-001')

# Cleanup old reports
manager.cleanup_old_reports()
```

## Testing

```bash
# Run tests
pytest test_reporting_system.py -v

# With coverage
pytest test_reporting_system.py --cov=. --cov-report=html
```

## Troubleshooting

### PDF Generation Fails

```bash
pip install --upgrade reportlab
# On Linux: sudo apt-get install texlive-fonts-recommended
```

### Excel File Corrupted

```bash
pip install --upgrade openpyxl
```

### Template Not Found

```python
# Check available templates
templates = manager.list_templates()
print(templates)
```

### Schedule Not Running

```python
# Check if enabled
status = manager.get_schedule_status('schedule-id')
print(f"Enabled: {status['enabled']}")

# Manually trigger
manager.run_schedule('schedule-id', data_source)
```

## File Structure

```
report_generator.py       # Main report generator
report_templates.py       # Template system
scheduled_reports.py      # Scheduling system
test_reporting_system.py  # Test suite (35+ tests)
reporting_demo.py         # Demo script
reporting_requirements.txt # Dependencies
REPORTING_SYSTEM_COMPLETE.md    # Full documentation
REPORTING_QUICK_REFERENCE.md    # This file
```

## Quick Demo

```bash
# Run demo
python reporting_demo.py
```

## Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| PDF | .pdf | Formal reports, archiving |
| HTML | .html | Interactive dashboards |
| Markdown | .md | Documentation, wikis |
| Excel | .xlsx | Data analysis |
| JSON | .json | API integration |

## Template Types

- `executive_summary` - Concise overview
- `detailed_report` - Full breakdown
- `progress_report` - Status tracking
- `quality_report` - Quality assessment
- `comparison_report` - Multiple plans

## Schedule Expressions

- `daily` - Every day at 9 AM
- `weekly` - Every week at 9 AM
- `monthly` - First day of month
- `every N hours` - Every N hours
- `every N days` - Every N days
- `0 9 * * 1` - Cron (9 AM Monday)

## More Information

See `REPORTING_SYSTEM_COMPLETE.md` for full documentation.
