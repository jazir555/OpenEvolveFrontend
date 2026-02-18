#!/usr/bin/env python3
"""
Unified Entry Point for Adaptive MDAP/MAKER Adapter Integration

This script provides a single entry point for all integration capabilities,
demonstrating the complete functionality of the enhanced adapter.

Usage:
    python unified_entry.py --help
"""

import argparse
import asyncio
import json
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set required environment variables
os.environ.setdefault("ADAPTIVE_MDAP_TIMEOUT_MS", "5000")


class UnifiedAdapterInterface:
    """
    Unified interface for all adapter capabilities.

    Provides simple methods for common operations across all integrated systems.
    """

    def __init__(self):
        """Initialize unified interface with all components."""
        print("Initializing Adaptive MDAP/MAKER Adapter (v2.0)...")

        # Core adapters
        from src import get_adapter, get_maker_adapter
        self.mdap = get_adapter()
        self.maker = get_maker_adapter()

        # Integration components
        from src import get_integration_manager
        self.manager = get_integration_manager()

        # Advanced components
        try:
            from src import get_advanced_openevolve_integration
            self.advanced_openevolve = get_advanced_openevolve_integration()
        except Exception as e:
            print(f"Warning: Advanced OpenEvolve integration not available: {e}")
            self.advanced_openevolve = None

        try:
            from src import get_advanced_bubblelab_ui
            self.advanced_ui = get_advanced_bubblelab_ui()
        except Exception as e:
            print(f"Warning: Advanced UI not available: {e}")
            self.advanced_ui = None

        try:
            from src import get_advanced_gauntlet_integration
            self.advanced_gauntlet = get_advanced_gauntlet_integration()
        except Exception as e:
            print(f"Warning: Advanced gauntlet not available: {e}")
            self.advanced_gauntlet = None

        try:
            from src import get_advanced_icr_integration
            self.advanced_icr = get_advanced_icr_integration()
        except Exception as e:
            print(f"Warning: Advanced ICR not available: {e}")
            self.advanced_icr = None

        try:
            from src import get_async_adapter
            self.async_adapter = get_async_adapter()
        except Exception as e:
            print(f"Warning: Async adapter not available: {e}")
            self.async_adapter = None

        try:
            from src import get_unified_system_monitor
            self.system_monitor = get_unified_system_monitor()
        except Exception as e:
            print(f"Warning: System monitor not available: {e}")
            self.system_monitor = None

        print("[OK] Initialization complete\n")

    def analyze(self, problem: str, domain: str = "general") -> Dict[str, Any]:
        """
        Analyze problem complexity (basic).

        Args:
            problem: Problem description
            domain: Problem domain

        Returns:
            Analysis results
        """
        import time
        from src import CanonicalSubProblem

        subproblem = CanonicalSubProblem(
            id=f"analysis_{int(time.time() * 1000)}",
            description=problem,
            domain=domain,
            depth=1
        )

        response = self.mdap.analyze_complexity(subproblem)

        return {
            "status": response.status.value,
            "complexity": response.complexity_score.overall_score if response.complexity_score else 0,
            "strategy": response.strategy.value if response.strategy else None,
            "execution_time_ms": response.execution_time_ms
        }

    def analyze_advanced(self, problem: str, workflow_type: str = "evolution") -> Dict[str, Any]:
        """
        Analyze with full decomposition (advanced).

        Args:
            problem: Problem description
            workflow_type: Type of workflow

        Returns:
            Advanced analysis with decomposition
        """
        if not self.advanced_openevolve:
            return {"error": "Advanced OpenEvolve integration not available"}

        # Basic complexity analysis
        basic_analysis = self.manager.analyze_workflow(
            workflow_id="unified_analysis",
            problem_statement=problem,
            workflow_type=workflow_type
        )

        # Advanced decomposition
        decomposition = self.advanced_openevolve.decompose_problem(
            workflow_id="unified_analysis",
            problem_statement=problem,
            workflow_type=workflow_type,
            max_depth=3
        )

        # Team selection
        team_selection = self.advanced_openevolve.select_teams_for_stage(
            workflow_id="unified_analysis",
            stage="planning",
            workflow_type=workflow_type,
            complexity_score=basic_analysis.overall_complexity
        )

        # Resource optimization
        optimization = self.advanced_openevolve.optimize_resources(
            workflow_id="unified_analysis",
            stage="execution",
            complexity_score=basic_analysis.overall_complexity,
            estimated_duration_ms=basic_analysis.estimated_duration_ms
        )

        return {
            "basic_analysis": {
                "complexity": basic_analysis.overall_complexity,
                "strategy": basic_analysis.recommended_strategy,
                "estimated_duration_ms": basic_analysis.estimated_duration_ms
            },
            "decomposition": {
                "sub_problems": len(decomposition.sub_problems),
                "strategy": decomposition.decomposition_strategy,
                "parallelization": decomposition.recommended_parallelization
            },
            "team_selection": {
                "teams": team_selection.recommended_teams,
                "estimated_cost": team_selection.estimated_cost
            },
            "resource_optimization": {
                "cpu": optimization.cpu_allocation,
                "memory_mb": optimization.memory_allocation_mb,
                "timeout_ms": optimization.timeout_ms,
                "cost_savings": optimization.estimated_cost_savings
            }
        }

    def verify(self, solution: Any, complexity: float = 0.5) -> Dict[str, Any]:
        """
        Run solution through gauntlet pipeline.

        Args:
            solution: Solution to verify
            complexity: Solution complexity (0-1)

        Returns:
            Verification results
        """
        if not self.advanced_gauntlet:
            return {"error": "Advanced gauntlet not available"}

        from src import GauntletType

        pipeline = self.advanced_gauntlet.create_gauntlet_pipeline(
            complexity_score=complexity,
            base_gauntlet_type=GauntletType.ADVERSARIAL,
            include_cross_validation=True
        )

        result = self.advanced_gauntlet.execute_pipeline(
            pipeline=pipeline,
            solution=solution
        )

        return {
            "total_gauntlets": result.total_gauntlets,
            "passed_gauntlets": result.passed_gauntlets,
            "overall_pass": result.overall_pass,
            "aggregate_score": result.aggregate_score,
            "execution_time_ms": result.execution_time_ms
        }

    def get_ui_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive UI dashboard data.

        Returns:
            Dashboard data with all visualizations
        """
        dashboard = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": None,
            "charts": {},
            "alerts": []
        }

        # Health status
        health = self.manager.get_health_status()
        dashboard["health"] = {
            "overall": health.overall_status.value,
            "mdap": health.mdap_adapter_status,
            "maker": health.maker_adapter_status
        }

        # Advanced UI components
        if self.advanced_ui:
            # Health dashboard with alerts
            health_dashboard = self.advanced_ui.create_adapter_health_dashboard()
            dashboard["health"] = health_dashboard["health"]
            dashboard["alerts"] = health_dashboard["alerts"]

            # ICR insights
            icr_chart = self.advanced_ui.create_icr_insights_dashboard()
            dashboard["charts"]["icr_insights"] = icr_chart.data

        # System monitoring
        if self.system_monitor:
            system_health = self.system_monitor.get_overall_health()
            dashboard["system_health"] = system_health

        return dashboard

    def learn(self, pattern_type: str, passed: bool, context: Dict[str, Any]) -> str:
        """
        Store pattern for ICR learning.

        Args:
            pattern_type: Type of pattern
            passed: Whether operation passed
            context: Context information

        Returns:
            Pattern ID
        """
        if not self.advanced_icr:
            return "ICR not available"

        from src import ICRPatternType

        pattern_type_enum = ICRPatternType[pattern_type.upper()]

        pattern_id = self.advanced_icr.store_pattern_advanced(
            pattern_type=pattern_type_enum,
            passed=passed,
            context=context,
            metrics={"timestamp": datetime.now(timezone.utc).isoformat()}
        )

        return pattern_id

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all components.

        Returns:
            Status dictionary
        """
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "adapter": {
                "mdap": self.mdap.health_check(),
                "maker": self.maker.health_check()
            },
            "advanced_components": {},
            "systems": {}
        }

        # Advanced components status
        if self.advanced_openevolve:
            status["advanced_components"]["openevolve"] = "available"
        if self.advanced_ui:
            status["advanced_components"]["ui"] = "available"
        if self.advanced_gauntlet:
            status["advanced_components"]["gauntlet"] = "available"
        if self.advanced_icr:
            status["advanced_components"]["icr"] = "available"
        if self.async_adapter:
            status["advanced_components"]["async"] = "available"

        # Additional systems status
        if self.system_monitor:
            system_health = self.system_monitor.get_overall_health()
            status["systems"] = system_health

        return status


def main():
    """CLI interface for unified entry point."""
    parser = argparse.ArgumentParser(
        description="Adaptive MDAP/MAKER Adapter - Unified Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze problem complexity')
    analyze_parser.add_argument('--problem', required=True, help='Problem description')
    analyze_parser.add_argument('--domain', default='general', help='Problem domain')
    analyze_parser.add_argument('--advanced', action='store_true', help='Use advanced analysis with decomposition')
    analyze_parser.add_argument('--workflow-type', default='evolution', help='Workflow type for advanced analysis')
    analyze_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Run solution through gauntlet')
    verify_parser.add_argument('--solution', required=True, help='Solution to verify')
    verify_parser.add_argument('--complexity', type=float, default=0.5, help='Solution complexity (0-1)')
    verify_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Get UI dashboard data')
    dashboard_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # Learn command
    learn_parser = subparsers.add_parser('learn', help='Store pattern for learning')
    learn_parser.add_argument('--pattern-type', required=True, help='Pattern type')
    learn_parser.add_argument('--passed', type=bool, required=True, help='Whether operation passed')
    learn_parser.add_argument('--context', required=True, help='Context as JSON string')

    # Status command
    status_parser = subparsers.add_parser('status', help='Get system status')
    status_parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize unified interface
    interface = UnifiedAdapterInterface()

    # Execute command
    if args.command == 'analyze':
        if args.advanced:
            result = interface.analyze_advanced(args.problem, args.workflow_type)
        else:
            result = interface.analyze(args.problem, args.domain)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Analysis Result:")
            print(f"  Status: {result.get('status', 'unknown')}")
            if 'complexity' in result:
                print(f"  Complexity: {result['complexity']:.3f}")
            if 'strategy' in result:
                print(f"  Strategy: {result['strategy']}")

    elif args.command == 'verify':
        result = interface.verify(args.solution, args.complexity)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Verification Result:")
            print(f"  Total Gauntlets: {result.get('total_gauntlets', 0)}")
            print(f"  Passed: {result.get('passed_gauntlets', 0)}")
            print(f"  Overall Pass: {result.get('overall_pass', False)}")
            print(f"  Score: {result.get('aggregate_score', 0):.3f}")

    elif args.command == 'dashboard':
        dashboard = interface.get_ui_dashboard()

        if args.json:
            print(json.dumps(dashboard, indent=2))
        else:
            print(f"Dashboard Status:")
            print(f"  Overall: {dashboard['health'].get('overall', 'unknown')}")
            print(f"  MDAP: {dashboard['health'].get('mdap', 'unknown')}")
            print(f"  MAKER: {dashboard['health'].get('maker', 'unknown')}")
            print(f"  Alerts: {len(dashboard.get('alerts', []))}")

    elif args.command == 'learn':
        context = json.loads(args.context)
        pattern_id = interface.learn(args.pattern_type, args.passed, context)
        print(f"Pattern stored: {pattern_id}")

    elif args.command == 'status':
        status = interface.get_status()

        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            print(f"System Status:")
            print(f"  MDAP Adapter: {status['adapter']['mdap'].get('status', 'unknown')}")
            print(f"  MAKER Adapter: {status['adapter']['maker'].get('status', 'unknown')}")
            print(f"  Advanced Components: {len(status.get('advanced_components', {}))}")
            print(f"  Additional Systems: {status.get('systems', {}).get('available_systems', 0)}/{status.get('systems', {}).get('total_systems', 0)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
