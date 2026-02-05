"""
Compliance Monitor Quickstart Example

This script demonstrates how to use the Continuous Compliance Monitoring Agent.

Author: AI Architecture Team
Date: 2026-01-30
"""

import asyncio
import json
from pathlib import Path

from openevolve.agents.compliance_monitor import (
    ComplianceMonitor,
    AlertSeverity,
    run_compliance_monitor
)


async def example_1_basic_monitoring():
    """Example 1: Basic compliance monitoring"""
    print("\n" + "="*60)
    print("Example 1: Basic Compliance Monitoring")
    print("="*60 + "\n")

    # Create compliance monitor
    monitor = ComplianceMonitor(
        checkpoint_dir="./checkpoints/compliance_demo",
        scan_interval_seconds=60,  # Scan every minute for demo
        alert_threshold=AlertSeverity.MEDIUM,
        use_formal_verification=False,  # Disable for faster demo
        log_level="INFO"
    )

    # Run one monitoring cycle
    print("Running monitoring cycle...")
    await monitor._monitoring_cycle()

    # Get status
    status = monitor.get_status()
    print(f"\nMonitor Status:")
    print(f"  Running: {status['running']}")
    print(f"  Current Phase: {status['current_phase']}")
    print(f"  Last Scan: {status['last_scan']}")

    # Get compliance report
    report = await monitor.get_compliance_report()
    print(f"\nCompliance Report:")
    print(f"  Regulatory Version: {report['regulatory_version']}")
    print(f"  Total Rules: {report['metrics'].get('total_rules', 0)}")
    print(f"  Active Alerts: {report['metrics'].get('active_alerts', 0)}")

    print("\n[OK] Example 1 complete\n")


async def example_2_regulatory_update():
    """Example 2: Force regulatory update"""
    print("\n" + "="*60)
    print("Example 2: Force Regulatory Update")
    print("="*60 + "\n")

    # Create monitor
    monitor = ComplianceMonitor(
        checkpoint_dir="./checkpoints/compliance_demo",
        use_formal_verification=False,
        log_level="WARNING"
    )

    # Simulate regulatory changes
    regulatory_changes = [
        {
            'title': 'New SEC Rule 10b-5 Amendment',
            'description': 'Updated requirements for insider trading detection',
            'url': 'https://www.sec.gov/rules/final/2025/34-12345.pdf',
            'published_date': '2025-01-15T10:00:00Z',
            'change_type': 'amendment',
            'affected_areas': ['trading', 'reporting']
        },
        {
            'title': 'FINRA Crypto Asset Guidance',
            'description': 'New guidance for cryptocurrency trading oversight',
            'url': 'https://www.finra.org/rules-guidance/guidance/crypto',
            'published_date': '2025-01-14T14:30:00Z',
            'change_type': 'guidance',
            'affected_areas': ['crypto', 'trading']
        }
    ]

    print(f"Forcing update with {len(regulatory_changes)} regulatory changes...")

    # Force update
    await monitor.force_update(regulatory_changes)

    # Check results
    if monitor.state.last_update_time:
        print(f"[OK] Rules updated at {monitor.state.last_update_time}")
        print(f"  Current rules: {len(monitor.state.current_rules)}")

    print("\n[OK] Example 2 complete\n")


async def example_3_alert_generation():
    """Example 3: Alert generation and handling"""
    print("\n" + "="*60)
    print("Example 3: Alert Generation and Handling")
    print("="*60 + "\n")

    from openevolve.agents.compliance import ComplianceAlerter

    # Create alerter
    alerter = ComplianceAlerter(
        threshold=AlertSeverity.LOW,  # Catch all alerts for demo
        enable_fatigue_prevention=False  # Disable for demo
    )

    # Create test violations
    violations = [
        {
            'title': 'Critical Trading Violation',
            'message': 'Detected suspicious trading pattern matching insider trading',
            'type': 'insider_trading',
            'risk_score': 95,
            'source': 'trading_monitor'
        },
        {
            'title': 'Reporting Deadline Missed',
            'message': 'Form 10-K not filed within deadline',
            'type': 'reporting',
            'risk_score': 75,
            'source': 'filing_monitor'
        },
        {
            'title': 'Minor Data Issue',
            'message': 'Minor discrepancy in trade data',
            'type': 'data_quality',
            'risk_score': 25,
            'source': 'data_validator'
        }
    ]

    print(f"Generating alerts for {len(violations)} violations...\n")

    alerts_generated = []
    for violation in violations:
        alert = await alerter.generate_alert(violation)
        if alert:
            alerts_generated.append(alert)
            print(f"[OK] Alert Generated: {alert.alert_id}")
            print(f"  Severity: {alert.severity.value}")
            print(f"  Title: {alert.title}")
            print(f"  Status: {alert.status.value}\n")

    # Get statistics
    stats = alerter.get_alert_statistics()
    print(f"Alert Statistics:")
    print(f"  Total: {stats['total']}")
    print(f"  By Severity: {stats['by_severity']}")
    print(f"  False Positive Rate: {stats['false_positive_rate']:.1%}")

    print("\n[OK] Example 3 complete\n")


async def example_4_edge_case_discovery():
    """Example 4: Edge case discovery"""
    print("\n" + "="*60)
    print("Example 4: Edge Case Discovery")
    print("="*60 + "\n")

    from openevolve.agents.compliance import EdgeCaseDiscovery

    # Create edge case discovery
    discovery = EdgeCaseDiscovery(
        max_adversarial_iterations=5,  # Small number for demo
        max_fuzz_iterations=10
    )

    # Sample rules
    rules = {
        'rule_001': {
            'name': 'Large Trade Reporting',
            'description': 'Report trades exceeding $10,000',
            'logic': 'if trade_amount > 10000 then require_report()'
        },
        'rule_002': {
            'name': 'Insider Trading Detection',
            'description': 'Flag trades based on non-public information',
            'logic': 'if (material_non_public_info and trade_executed) then flag_insider_trading()'
        }
    }

    print(f"Discovering edge cases for {len(rules)} rules...\n")

    # Discover edge cases (simplified for demo - without full LoongFlow)
    # In production, this would use adversarial testing
    from openevolve.agents.compliance.edge_discovery import EdgeCase, EdgeCaseType

    # Simulate discovering edge cases
    edge_cases = [
        EdgeCase(
            case_id='edge_001',
            case_type=EdgeCaseType.BOUNDARY,
            description='Trade at exactly $10,000 threshold',
            scenario={'trade_amount': 10000.00},
            expected_behavior='Should require reporting',
            severity='medium',
            affected_rules=['rule_001']
        ),
        EdgeCase(
            case_id='edge_002',
            case_type=EdgeCaseType.COMBINATORIAL,
            description='Multiple trades just below threshold',
            scenario={'trades': [9999, 9999, 9999]},
            expected_behavior='Should detect structuring',
            severity='high',
            affected_rules=['rule_001']
        )
    ]

    discovery.edge_cases.extend(edge_cases)

    print(f"[OK] Discovered {len(edge_cases)} edge cases:")
    for case in edge_cases:
        print(f"  - {case.case_id}: {case.description}")
        print(f"    Type: {case.case_type.value}")
        print(f"    Severity: {case.severity}\n")

    # Analyze coverage
    report = await discovery.analyze_coverage(rules)
    print(f"Coverage Analysis:")
    print(f"  Total Rules: {report.total_rules}")
    print(f"  Coverage: {report.coverage_percentage:.1f}%")
    print(f"  Edge Cases Found: {report.edge_cases_found}")

    print("\n[OK] Example 4 complete\n")


async def example_5_compliance_verification():
    """Example 5: Compliance verification"""
    print("\n" + "="*60)
    print("Example 5: Compliance Verification")
    print("="*60 + "\n")

    from openevolve.agencies.compliance import ComplianceVerifier, ProofType

    # Create verifier (without formal methods for faster demo)
    verifier = ComplianceVerifier(
        use_formal_methods=False,  # Use logical verification
        timeout_seconds=10
    )

    # Sample rules
    rules = {
        'rule_001': {
            'name': 'Trade Authorization',
            'description': 'All trades must be authorized',
            'logic': 'if not authorized then reject_trade()'
        }
    }

    print(f"Verifying {len(rules)} compliance rules...\n")

    # Verify rules
    results = await verifier.verify_rules(
        rules=rules,
        proof_types=[ProofType.CONSISTENCY, ProofType.COMPLETENESS]
    )

    print(f"Verification Results:")
    for result in results:
        status = "[OK] PASSED" if result.success else "[FAIL] FAILED"
        print(f"  {result.proof_type.value.upper()}: {status}")
        print(f"    Method: {result.method.value}")
        print(f"    Confidence: {result.confidence:.1%}")
        if result.proof:
            print(f"    Proof: {result.proof[:100]}...")
        print()

    print("\n[OK] Example 5 complete\n")


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Compliance Monitor Quickstart Examples")
    print("="*60)

    # Create checkpoint directory
    Path("./checkpoints/compliance_demo").mkdir(parents=True, exist_ok=True)

    try:
        # Run examples
        await example_1_basic_monitoring()
        await example_2_regulatory_update()
        await example_3_alert_generation()
        await example_4_edge_case_discovery()
        await example_5_compliance_verification()

        print("\n" + "="*60)
        print("All Examples Complete!")
        print("="*60 + "\n")

        print("Next Steps:")
        print("1. Review the generated checkpoints in ./checkpoints/compliance_demo/")
        print("2. Customize regulatory sources for your domain")
        print("3. Define your compliance rules")
        print("4. Enable formal verification for production use")
        print("5. Integrate with your monitoring systems")
        print("\nFor more information, see: docs/agents/compliance_monitor.md\n")

    except Exception as e:
        print(f"\n[FAIL] Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
