#!/usr/bin/env python3
"""
Example: UI Dashboard Generation

This example demonstrates how to generate interactive visualizations
and dashboard data for the BubbleLab UI.

Usage:
    cd examples
    python example_ui_dashboard.py
"""

import os
import sys
import json
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set environment variables
os.environ.setdefault("ADAPTIVE_MDAP_TIMEOUT_MS", "5000")

from src import get_advanced_bubblelab_ui


def main():
    """Demonstrate UI dashboard generation."""
    print("=" * 70)
    print("  EXAMPLE: UI Dashboard Generation")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now(timezone.utc).isoformat()}\n")

    # Get advanced UI integration
    ui = get_advanced_bubblelab_ui()

    # Phase 1: Analyze complexity for UI
    print("Phase 1: Complexity Analysis for UI")
    print("-" * 70)

    result = ui.analyze_complexity_for_ui(
        problem_description="Build real-time analytics dashboard with WebSocket streaming",
        domain="analytics",
        depth=3
    )

    print(f"\nProblem ID: {result.problem_id}")
    print(f"Overall Complexity: {result.overall_complexity:.3f}")
    print(f"Strategy: {result.strategy.value}")
    print(f"Execution Time: {result.execution_time_ms:.0f}ms")

    print("\nComplexity Breakdown:")
    for dimension, score in result.complexity_dimensions.items():
        print(f"  {dimension}: {score:.3f}")

    # Phase 2: Create complexity radar chart
    print("\n" + "=" * 70)
    print("Phase 2: Complexity Radar Chart")
    print("=" * 70)

    radar_chart = ui.create_complexity_radar_chart(
        analysis_id=result.problem_id,
        include_recommendations=True
    )

    if radar_chart:
        print(f"\nChart Type: {radar_chart.chart_type.value}")
        print(f"Chart Title: {radar_chart.title}")
        print(f"Labels: {', '.join(radar_chart.data['labels'])}")

        print("\nDataset:")
        for i, (label, value) in enumerate(zip(
            radar_chart.data['labels'],
            radar_chart.data['datasets'][0]['data']
        )):
            print(f"  {label}: {value:.3f}")

        if radar_chart.recommendations:
            print("\nRecommendations:")
            for rec in radar_chart.recommendations[:3]:
                print(f"  - {rec}")

    # Phase 3: Create MAKER voting chart
    print("\n" + "=" * 70)
    print("Phase 3: MAKER Voting Chart")
    print("=" * 70)

    voting_chart = ui.create_maker_voting_chart(
        workflow_id="example_workflow",
        decision_point="architecture_selection"
    )

    if voting_chart:
        print(f"\nChart Type: {voting_chart.chart_type.value}")
        print(f"Title: {voting_chart.title}")
        print(f"Total Votes: {voting_chart.data.get('total_votes', 0)}")

        print("\nVote Distribution:")
        for option, votes in voting_chart.data.get('vote_distribution', {}).items():
            percentage = (votes / voting_chart.data.get('total_votes', 1)) * 100
            print(f"  {option}: {votes} votes ({percentage:.1f}%)")

        if voting_chart.data.get('consensus_reached'):
            print("\n[INFO] Consensus reached!")
        else:
            print("\n[INFO] No consensus - additional voting rounds needed")

    # Phase 4: Create workflow timeline
    print("\n" + "=" * 70)
    print("Phase 4: Workflow Timeline Visualization")
    print("=" * 70)

    timeline = ui.create_workflow_timeline(
        workflow_id="example_workflow"
    )

    if timeline:
        print(f"\nTimeline Type: {timeline.chart_type.value}")
        print(f"Stages: {len(timeline.stages)}")
        print(f"Total Duration: {timeline.total_duration_ms:.0f}ms")

        print("\nStages:")
        for stage in timeline.stages:
            duration = stage.get('duration_ms', 0)
            status = stage.get('status', 'unknown')
            print(f"  [{status}] {stage['stage']}: {duration:.0f}ms")

    # Phase 5: Create ICR insights dashboard
    print("\n" + "=" * 70)
    print("Phase 5: ICR Insights Dashboard")
    print("=" * 70)

    icr_dashboard = ui.create_icr_insights_dashboard()

    if icr_dashboard:
        print(f"\nDashboard Type: {icr_dashboard.chart_type.value}")
        print(f"Title: {icr_dashboard.title}")

        if 'pattern_types' in icr_dashboard.data:
            print("\nPattern Types Tracked:")
            for ptype, count in icr_dashboard.data['pattern_types'].items():
                print(f"  {ptype}: {count} patterns")

        if 'confidence_distribution' in icr_dashboard.data:
            print("\nConfidence Distribution:")
            for bucket, count in icr_dashboard.data['confidence_distribution'].items():
                print(f"  {bucket}: {count} predictions")

    # Phase 6: Create adapter health dashboard
    print("\n" + "=" * 70)
    print("Phase 6: Adapter Health Dashboard")
    print("=" * 70)

    health_dashboard = ui.create_adapter_health_dashboard()

    print(f"\nOverall Health: {health_dashboard['health']['overall_status']}")

    print("\nComponent Status:")
    for component, status in health_dashboard['health']['components'].items():
        print(f"  {component}: {status['status']}")

    if health_dashboard.get('alerts'):
        print(f"\nActive Alerts: {len(health_dashboard['alerts'])}")
        for alert in health_dashboard['alerts'][:3]:
            print(f"  [{alert['severity']}] {alert['message']}")

    # Phase 7: Export dashboard report
    print("\n" + "=" * 70)
    print("Phase 7: Export Dashboard Report")
    print("=" * 70)

    # Export as JSON
    report_json = ui.export_report(
        workflow_id="example_dashboard",
        format="json"
    )

    print(f"\nJSON Report Size: {len(report_json)} characters")

    # Export as Markdown
    report_md = ui.export_report(
        workflow_id="example_dashboard",
        format="markdown"
    )

    print(f"Markdown Report Size: {len(report_md)} characters")

    # Save to file
    output_path = "/tmp/dashboard_report.md"
    with open(output_path, 'w') as f:
        f.write(report_md)

    print(f"Saved to: {output_path}")

    # Display first few lines
    lines = report_md.split('\n')[:10]
    print("\nPreview (first 10 lines):")
    print("-" * 70)
    for line in lines:
        print(line)

    print("\n" + "=" * 70)
    print("  EXAMPLE COMPLETE")
    print("=" * 70)
    print(f"\nEnd Time: {datetime.now(timezone.utc).isoformat()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
