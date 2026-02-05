"""
OpenEvolve Reporting System - Demo and Examples

This script demonstrates the key features of the reporting system including:
- Basic report generation in multiple formats
- Custom templates
- Scheduled reports
- Report aggregation
- Executive summaries
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import reporting components
try:
    from report_generator import ReportGenerator, generate_quick_report
    from report_templates import TemplateManager, ReportContextBuilder
    from scheduled_reports import (
        ScheduledReportManager, ScheduleConfig,
        create_daily_report, create_weekly_report
    )
    REPORTING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import reporting modules: {e}")
    REPORTING_AVAILABLE = False


def create_sample_plan():
    """Create a sample decomposition plan for demonstration."""
    return {
        'plan_id': f'demo-plan-{datetime.now().strftime("%Y%m%d")}',
        'problem': {
            'title': 'Build E-Commerce Platform',
            'description': 'Develop a full-featured e-commerce platform with product catalog, shopping cart, payment processing, and order management.',
            'type': 'software_development',
            'priority': 'high'
        },
        'strategy': 'semantic',
        'created_at': datetime.now().isoformat(),
        'sub_problems': [
            {
                'id': 'sp-001',
                'title': 'Design Database Schema',
                'description': 'Create normalized database schema for products, users, orders, and payments.',
                'status': 'completed',
                'type': 'database',
                'complexity': 'medium',
                'dependencies': [],
                'quality_score': 88.0,
                'success_criteria': [
                    'Schema normalized to 3NF',
                    'Proper indexing strategy',
                    'Foreign key relationships defined'
                ]
            },
            {
                'id': 'sp-002',
                'title': 'Implement User Authentication',
                'description': 'Build secure user authentication with OAuth support and JWT tokens.',
                'status': 'completed',
                'type': 'security',
                'complexity': 'high',
                'dependencies': ['sp-001'],
                'quality_score': 85.0,
                'success_criteria': [
                    'OAuth 2.0 integration',
                    'JWT token management',
                    'Secure password hashing'
                ]
            },
            {
                'id': 'sp-003',
                'title': 'Develop Product Catalog API',
                'description': 'Create REST API for product CRUD operations with search and filtering.',
                'status': 'in_progress',
                'type': 'backend',
                'complexity': 'high',
                'dependencies': ['sp-001'],
                'quality_score': 75.0,
                'success_criteria': [
                    'RESTful endpoints',
                    'Search functionality',
                    'Category filtering',
                    'API documentation'
                ]
            },
            {
                'id': 'sp-004',
                'title': 'Build Shopping Cart',
                'description': 'Implement shopping cart functionality with session management.',
                'status': 'pending',
                'type': 'frontend',
                'complexity': 'medium',
                'dependencies': ['sp-003'],
                'quality_score': 70.0,
                'success_criteria': [
                    'Add/remove items',
                    'Quantity management',
                    'Session persistence',
                    'Cart calculations'
                ]
            },
            {
                'id': 'sp-005',
                'title': 'Integrate Payment Gateway',
                'description': 'Integrate Stripe payment processing with secure card handling.',
                'status': 'pending',
                'type': 'integration',
                'complexity': 'high',
                'dependencies': ['sp-004'],
                'quality_score': 72.0,
                'success_criteria': [
                    'Stripe integration',
                    'Secure card handling',
                    'Payment confirmation',
                    'Error handling'
                ]
            },
            {
                'id': 'sp-006',
                'title': 'Create Admin Dashboard',
                'description': 'Build admin interface for order and product management.',
                'status': 'pending',
                'type': 'frontend',
                'complexity': 'medium',
                'dependencies': ['sp-003'],
                'quality_score': 68.0,
                'success_criteria': [
                    'Product management',
                    'Order management',
                    'Analytics dashboard',
                    'User management'
                ]
            }
        ],
        'quality_scores': {
            'overall_score': 76.3,
            'completeness_score': 80.0,
            'consistency_score': 75.0,
            'feasibility_score': 78.0,
            'dependency_score': 85.0,
            'balance_score': 70.0,
            'meets_thresholds': True,
            'critical_issues': [],
            'improvement_recommendations': [
                'Consider breaking down payment integration into smaller tasks',
                'Add more detailed testing criteria',
                'Review dependency balance for frontend tasks'
            ]
        },
        'dependency_graph': {
            'total_edges': 5,
            'critical_path': ['sp-001', 'sp-003', 'sp-004', 'sp-005']
        },
        'metadata': {
            'plan_id': 'demo-plan-001',
            'strategy': 'semantic',
            'total_sub_problems': 6,
            'completion_percentage': 33.3
        }
    }


def demo_basic_reports():
    """Demonstrate basic report generation in multiple formats."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Report Generation")
    print("="*70)

    if not REPORTING_AVAILABLE:
        print("Skipping: Reporting modules not available")
        return

    # Create sample plan
    plan = create_sample_plan()

    # Initialize generator
    generator = ReportGenerator({
        'output_dir': './demo_reports',
        'include_charts': True,
        'branding': {
            'primary_color': '#2C3E50',
            'secondary_color': '#34495E',
            'company_name': 'Demo Corp'
        }
    })

    print("\nGenerating reports in multiple formats...")

    formats = ['html', 'json', 'md']
    generated = []

    for fmt in formats:
        try:
            output_path = generator.generate_report(
                plan,
                output_format=fmt,
                include_sections=['overview', 'sub_problems', 'quality_assessment']
            )
            generated.append(output_path)
            print(f"  [OK] {fmt.upper()}: {output_path}")
        except (IOError, ValueError, TypeError) as e:
            print(f"  [FAIL] {fmt.upper()}: Failed - {e}")

    print(f"\nSuccessfully generated {len(generated)} reports")
    return generated


def demo_executive_summary():
    """Demonstrate executive summary generation."""
    print("\n" + "="*70)
    print("DEMO 2: Executive Summary")
    print("="*70)

    if not REPORTING_AVAILABLE:
        print("Skipping: Reporting modules not available")
        return

    plan = create_sample_plan()
    generator = ReportGenerator({'output_dir': './demo_reports'})

    try:
        output_path = generator.generate_executive_summary(
            plan,
            output_format='html'
        )
        print(f"Executive summary generated: {output_path}")
        print("\nKey Insights:")
        print("  - Project is in early stages with 33% completion")
        print("  - 2 of 6 sub-problems completed")
        print("  - Critical path has 4 dependent tasks")
        print("  - Overall quality score: 76.3/100")
    except (IOError, ValueError, TypeError) as e:
        print(f"Failed to generate executive summary: {e}")


def demo_custom_template():
    """Demonstrate custom template creation."""
    print("\n" + "="*70)
    print("DEMO 3: Custom Templates")
    print("="*70)

    if not REPORTING_AVAILABLE:
        print("Skipping: Reporting modules not available")
        return

    plan = create_sample_plan()

    # Create template manager
    template_manager = TemplateManager()
    context_builder = ReportContextBuilder()

    # Build context
    context = context_builder.build_executive_summary_context(
        plan,
        insights=[
            "Database design completed successfully",
            "Authentication system implemented and tested",
            "Payment integration needs security review"
        ],
        recommendations=[
            "Focus on completing product catalog API",
            "Start shopping cart implementation soon",
            "Schedule security audit for payment processing"
        ]
    )

    # Create custom template
    custom_template = """
# 🎯 Project Status Report

**Project:** {{ problem.title }}
**Generated:** {{ now() }}
**Status:** 🔄 In Progress

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| Total Tasks | {{ key_metrics.total_sub_problems }} |
| Completed | {{ key_metrics.completed }} |
| In Progress | {{ key_metrics.in_progress }} |
| Pending | {{ key_metrics.pending }} |
| Completion | {{ key_metrics.completion_percentage }}% |

## 💡 Key Insights

{% for insight in insights %}
{{ loop.index }}. {{ insight }}
{% endfor %}

## 📋 Recommendations

{% for recommendation in recommendations %}
{{ loop.index }}. {{ recommendation }}
{% endfor %}

## 📈 Quality Metrics

| Dimension | Score | Visual |
|-----------|-------|--------|
{% for dimension, score in quality_summary.items() %}
| {{ dimension.replace('_', ' ').title() }} | {{ score }} | {{ '█' * (score // 10) }}{{ '░' * (10 - score // 10) }} |
{% endfor %}

---

*Generated by OpenEvolve Reporting System*
"""

    # Save and render custom template
    try:
        template_manager.create_custom_template(
            'status_report',
            custom_template,
            save_to_file=False
        )

        rendered = template_manager.render_template(
            'status_report',
            context
        )

        # Save rendered report
        output_path = Path('./demo_reports/custom_status_report.md')
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(rendered)

        print(f"Custom template report generated: {output_path}")
        print("\nTemplate features demonstrated:")
        print("  - Custom formatting with emojis")
        print("  - Dynamic data insertion")
        print("  - Conditional loops")
        print("  - Visual progress indicators")
    except (IOError, ValueError, TypeError) as e:
        print(f"Failed to create custom template: {e}")


def demo_scheduled_reports():
    """Demonstrate scheduled report configuration."""
    print("\n" + "="*70)
    print("DEMO 4: Scheduled Reports")
    print("="*70)

    if not REPORTING_AVAILABLE:
        print("Skipping: Reporting modules not available")
        return

    # Create temporary manager for demo
    import tempfile
    temp_dir = tempfile.mkdtemp()

    manager = ScheduledReportManager({
        'smtp': {
            'host': 'smtp.example.com',
            'port': 587,
            'username': 'reports@example.com',
            'password': 'password',
            'from': 'noreply@example.com'
        }
    }, storage_path=temp_dir)

    # Create different schedule types
    schedules = [
        create_daily_report(
            'daily-summary',
            'executive_summary',
            ['manager@example.com'],
            format='html'
        ),
        create_weekly_report(
            'weekly-detailed',
            'detailed_report',
            ['team@example.com'],
            format='pdf'
        ),
        ScheduleConfig(
            schedule_id='biweekly-progress',
            report_type='progress_report',
            schedule_expression='every 14 days',
            recipients=['stakeholders@example.com'],
            format='xlsx'
        )
    ]

    print("\nConfigured schedules:")
    for schedule in schedules:
        manager.add_schedule(schedule)
        next_run = manager.calculate_next_run(schedule)
        print(f"\n  [SCHEDULE] {schedule.schedule_id}")
        print(f"     Type: {schedule.report_type}")
        print(f"     Schedule: {schedule.schedule_expression}")
        print(f"     Format: {schedule.format}")
        print(f"     Recipients: {', '.join(schedule.recipients)}")
        print(f"     Next run: {next_run}")

    # Show schedule status
    print("\nSchedule Status:")
    for schedule_id in manager.schedules.keys():
        status = manager.get_schedule_status(schedule_id)
        print(f"\n  {schedule_id}:")
        print(f"    Enabled: {status['enabled']}")
        print(f"    Run count: {status['run_count']}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_aggregated_reports():
    """Demonstrate report aggregation."""
    print("\n" + "="*70)
    print("DEMO 5: Aggregated Reports")
    print("="*70)

    if not REPORTING_AVAILABLE:
        print("Skipping: Reporting modules not available")
        return

    # Create multiple plans
    plans = []
    for i in range(3):
        plan = create_sample_plan()
        plan['plan_id'] = f'aggregated-plan-{i:03d}'
        plan['problem']['title'] = f'Project Phase {i+1}'
        plan['metadata']['completion_percentage'] = (i + 1) * 25
        plans.append(plan)

    generator = ReportGenerator({'output_dir': './demo_reports'})

    try:
        output_path = generator.generate_aggregated_report(
            plans,
            output_format='html',
            group_by='date',
            output_path='./demo_reports/aggregated_report.html'
        )
        print(f"Aggregated report generated: {output_path}")
        print("\nAggregation summary:")
        print(f"  * Total plans: {len(plans)}")
        print(f"  * Grouped by: date")
        print(f"  * Average quality: {sum(p['quality_scores']['overall_score'] for p in plans) / len(plans):.1f}")
    except (IOError, ValueError, TypeError) as e:
        print(f"Failed to generate aggregated report: {e}")


def demo_quick_functions():
    """Demonstrate convenience functions."""
    print("\n" + "="*70)
    print("DEMO 6: Quick Report Functions")
    print("="*70)

    if not REPORTING_AVAILABLE:
        print("Skipping: Reporting modules not available")
        return

    plan = create_sample_plan()

    # Quick report generation
    try:
        output_path = generate_quick_report(
            plan,
            format='html',
            output_dir='./demo_reports'
        )
        print(f"Quick report generated: {output_path}")
    except (IOError, ValueError, TypeError) as e:
        print(f"Failed to generate quick report: {e}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("OpenEvolve Reporting System - Feature Demonstration")
    print("="*70)
    print("\nThis demo showcases the key features of the reporting system:")
    print("  1. Basic report generation in multiple formats")
    print("  2. Executive summary with insights")
    print("  3. Custom template creation")
    print("  4. Scheduled report configuration")
    print("  5. Aggregated reports across multiple plans")
    print("  6. Quick convenience functions")

    if not REPORTING_AVAILABLE:
        print("\n[WARN]  WARNING: Reporting modules not available")
        print("Please ensure all dependencies are installed:")
        print("  pip install reportlab openpyxl jinja2")
        return

    try:
        # Run demos
        demo_basic_reports()
        demo_executive_summary()
        demo_custom_template()
        demo_scheduled_reports()
        demo_aggregated_reports()
        demo_quick_functions()

        # Summary
        print("\n" + "="*70)
        print("Demo Complete!")
        print("="*70)
        print("\n📁 Generated reports saved to: ./demo_reports/")
        print("\nNext steps:")
        print("  1. Review the generated reports")
        print("  2. Customize templates for your needs")
        print("  3. Set up scheduled reports for automation")
        print("  4. Integrate with your decomposition pipeline")
        print("\nFor more information, see REPORTING_SYSTEM_COMPLETE.md")

    except (IOError, ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n[FAIL] Error: {e}")


if __name__ == '__main__':
    main()
