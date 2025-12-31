#!/usr/bin/env python
"""
RAGBits CLI Tool

Command-line interface for RAGBits integration operations.
Provides access to knowledge extraction, evaluation, scoring, and analysis.
"""

import argparse
import asyncio
import sys
import json
from typing import Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RAGBitsCLI:
    """
    Command-line interface for RAGBits operations.

    Usage:
        cli = RAGBitsCLI()
        cli.run()
    """

    def __init__(self):
        """Initialize CLI"""
        self.parser = self._create_parser()
        self.commands = self._get_commands()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            prog="ragbits",
            description="RAGBits Integration CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Extract knowledge from artifact
  ragbits extract --file solution.md --type solution

  # Score an artifact
  ragbits score --artifact art_123

  # Compare with historical
  ragbits compare --artifact art_123 --type solution

  # Generate evaluation dashboard
  ragbits dashboard --workflow workflow_123

  # Explore knowledge base
  ragbits explore --query "authentication patterns"

  # Show statistics
  ragbits stats
            """
        )

        # Global options
        parser.add_argument(
            "--config",
            type=str,
            help="Path to configuration file"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )
        parser.add_argument(
            "--output", "-o",
            type=str,
            choices=["json", "text", "table"],
            default="text",
            help="Output format (default: text)"
        )

        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Extract command
        extract_parser = subparsers.add_parser(
            "extract",
            help="Extract knowledge from artifact"
        )
        extract_parser.add_argument("--file", required=True, help="File to extract from")
        extract_parser.add_argument("--type", default="solution", help="Artifact type")
        extract_parser.add_argument("--use-llm", action="store_true", help="Use LLM for extraction")
        extract_parser.add_argument("--output-file", "-o", help="Output file for results")

        # Score command
        score_parser = subparsers.add_parser(
            "score",
            help="Score an artifact"
        )
        score_parser.add_argument("--artifact", required=True, help="Artifact ID")
        score_parser.add_argument("--type", default="solution", help="Artifact type")
        score_parser.add_argument("--details", action="store_true", help="Show detailed scores")

        # Compare command
        compare_parser = subparsers.add_parser(
            "compare",
            help="Compare with historical data"
        )
        compare_parser.add_argument("--artifact", required=True, help="Artifact ID")
        compare_parser.add_argument("--type", required=True, help="Artifact type")
        compare_parser.add_argument("--lookback-days", type=int, default=30, help="Days to look back")

        # Dashboard command
        dashboard_parser = subparsers.add_parser(
            "dashboard",
            help="Generate evaluation dashboard"
        )
        dashboard_parser.add_argument("--workflow", help="Workflow ID")
        dashboard_parser.add_argument("--subproblem", help="Sub-problem ID")
        dashboard_parser.add_argument("--type", choices=["workflow", "subproblem", "trend"], help="Dashboard type")
        dashboard_parser.add_argument("--output", "-o", help="Output HTML file")

        # Explore command
        explore_parser = subparsers.add_parser(
            "explore",
            help="Explore knowledge base"
        )
        explore_parser.add_argument("--query", "-q", required=True, help="Search query")
        explore_parser.add_argument("--limit", type=int, default=10, help="Max results")
        explore_parser.add_argument("--search-type", choices=["semantic", "keyword", "hybrid"], default="hybrid")

        # Stats command
        stats_parser = subparsers.add_parser(
            "stats",
            help="Show system statistics"
        )
        stats_parser.add_argument("--type", choices=["extraction", "scoring", "storage"], help="Statistics type")

        # Validate command
        validate_parser = subparsers.add_parser(
            "validate",
            help="Run gauntlet validation"
        )
        validate_parser.add_argument("--artifact", required=True, help="Artifact ID")
        validate_parser.add_argument("--requirements", nargs="+", help="Requirements list")
        validate_parser.add_argument("--tests", nargs="+", help="Specific tests to run")

        # Analyze trends
        trend_parser = subparsers.add_parser(
            "trend",
            help="Analyze trends"
        )
        trend_parser.add_argument("--type", required=True, help="Artifact type")
        trend_parser.add_argument("--days", type=int, default=30, help="Number of days")
        trend_parser.add_argument("--category", help="Metric category")

        return parser

    def _get_commands(self) -> dict:
        """Get command handlers"""
        return {
            "extract": self.cmd_extract,
            "score": self.cmd_score,
            "compare": self.cmd_compare,
            "dashboard": self.cmd_dashboard,
            "explore": self.cmd_explore,
            "stats": self.cmd_stats,
            "validate": self.cmd_validate,
            "trend": self.cmd_trend
        }

    def run(self, args: List[str] = None):
        """Run CLI with given arguments"""
        if args is None:
            args = sys.argv[1:]

        parsed = self.parser.parse_args(args)

        # Setup logging
        if parsed.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # Execute command
        handler = self.commands.get(parsed.command)

        if handler is None:
            self.parser.print_help()
            sys.exit(1)

        # Run command
        try:
            result = asyncio.run(handler(parsed))

            # Format output
            if parsed.output == "json":
                print(json.dumps(result, indent=2))
            elif parsed.output == "table":
                self._print_table(result)
            else:
                self._print_text(result)

        except Exception as e:
            logger.error(f"Command failed: {e}")
            if parsed.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    async def cmd_extract(self, args):
        """Extract knowledge from artifact"""
        from ragbits_integration.knowledge_base import KnowledgeExtractor

        # Read file
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {args.file}")

        content = file_path.read_text()

        print(f"Extracting knowledge from {args.file}...")

        # Initialize extractor
        extractor = KnowledgeExtractor()

        # Extract
        result = await extractor.extract_from_artifact(
            artifact_id=file_path.stem,
            content=content,
            artifact_type=args.type,
            use_llm=args.use_llm
        )

        output = {
            "command": "extract",
            "artifact_id": result.artifact_id,
            "entities_extracted": len(result.entities),
            "processing_time_ms": result.processing_time_ms,
            "summary": result.extraction_summary,
            "entities": [
                {
                    "type": e.entity_type.value,
                    "content": e.content[:100],
                    "confidence": f"{e.confidence:.2f}",
                    "tags": list(e.tags)
                }
                for e in result.entities[:10]  # First 10
            ]
        }

        # Save to file if requested
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.write_text(json.dumps(output, indent=2))
            print(f"Results saved to {args.output_file}")

        return output

    async def cmd_score(self, args):
        """Score an artifact"""
        from ragbits_integration.evaluation import MetricsAnalyzer
        from ragbits_integration.evaluation.metrics import EvaluationMetricsCollector

        print(f"Scoring artifact {args.artifact}...")

        # Initialize
        collector = EvaluationMetricsCollector()
        analyzer = MetricsAnalyzer(collector)

        # Analyze
        report = await analyzer.analyze_artifact(args.artifact)

        if report is None:
            return {"error": f"No metrics found for artifact {args.artifact}"}

        output = {
            "command": "score",
            "artifact_id": args.artifact,
            "overall_score": f"{report.overall_score:.2f}",
            "verdict": self._determine_verdict(report.overall_score),
            "category_scores": {
                cs.category.value: f"{cs.score:.2f}"
                for cs in report.category_scores
            }
        }

        if args.details:
            output["detailed_scores"] = {
                cs.category.value: {
                    "score": f"{cs.score:.2f}",
                    "weight": cs.weight,
                    "issues": cs.issues,
                    "strengths": cs.strengths
                }
                for cs in report.category_scores
            }
            output["recommendations"] = report.recommendations
            output["critical_issues"] = report.critical_issues

        return output

    async def cmd_compare(self, args):
        """Compare artifact with historical data"""
        from ragbits_integration.evaluation import (
            HistoricalComparator,
            MetricsAnalyzer,
            EvaluationMetricsCollector
        )

        print(f"Comparing {args.artifact} with historical data...")

        # Initialize
        collector = EvaluationMetricsCollector()
        analyzer = MetricsAnalyzer(collector)
        comparator = HistoricalComparator(collector, analyzer)

        # Compare
        report = await comparator.compare_with_historical(
            artifact_id=args.artifact,
            artifact_type=args.type,
            lookback_days=args.lookback_days
        )

        if report is None:
            return {"error": "No historical data available for comparison"}

        output = {
            "command": "compare",
            "artifact_id": args.artifact,
            "current_score": f"{report.current_score:.2f}",
            "percentile_rank": f"{report.percentile_rank:.1f}%",
            "summary": report._generate_summary(),
            "historical_count": len(report.historical_scores)
        }

        return output

    async def cmd_dashboard(self, args):
        """Generate evaluation dashboard"""
        from ragbits_integration.evaluation import EvaluationDashboard
        from ragbits_integration.evaluation.metrics import EvaluationMetricsCollector
        from ragbits_integration.evaluation.metrics import MetricsAnalyzer
        from ragbits_integration.evaluation.gauntlets import EnhancedGauntletValidator
        from ragbits_integration.evaluation.comparison import HistoricalComparator

        print("Generating evaluation dashboard...")

        # Initialize components
        collector = EvaluationMetricsCollector()
        analyzer = MetricsAnalyzer(collector)
        validator = EnhancedGauntletValidator(collector)
        comparator = HistoricalComparator(collector, analyzer)
        dashboard = EvaluationDashboard(collector, analyzer, validator, comparator)

        # Generate appropriate dashboard
        if args.type == "workflow" and args.workflow:
            report = await dashboard.generate_workflow_dashboard(
                workflow_id=args.workflow
            )
        elif args.type == "subproblem" and args.subproblem:
            report = await dashboard.generate_subproblem_dashboard(
                sub_problem_id=args.subproblem
            )
        elif args.type == "trend":
            report = await dashboard.generate_trend_dashboard(
                artifact_type="solution",
                days=30
            )
        else:
            return {"error": "Invalid dashboard type or missing ID"}

        # Generate HTML
        html = report.to_html()

        # Save to file
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(html)
            print(f"Dashboard saved to {args.output}")
        else:
            # Print to stdout
            print(html)

        return {
            "command": "dashboard",
            "type": args.type,
            "report_id": report.report_id,
            "metric_cards": len(report.metric_cards),
            "charts": len(report.charts),
            "tables": len(report.tables),
            "summary": report.summary
        }

    async def cmd_explore(self, args):
        """Explore knowledge base"""
        from ragbits_integration.knowledge_base import AdvancedRAGEngine

        print(f"Exploring knowledge base for: {args.query}")

        # Initialize
        rag_engine = AdvancedRAGEngine()

        # Query
        result = await rag_engine.query(
            query_text=args.query,
            search_type=args.search_type,
            top_k=args.limit
        )

        output = {
            "command": "explore",
            "query": args.query,
            "search_type": args.search_type,
            "results_count": len(result.ranked_documents),
            "retrieval_time_ms": result.retrieval_time_ms,
            "documents": []
        }

        for doc in result.ranked_documents:
            output["documents"].append({
                "id": doc.get("id", "unknown"),
                "score": f"{doc.get('score', 0):.2f}",
                "content": doc.get("content", "")[:200]
            })

        # Show query expansions if available
        if result.query_expansion:
            output["expanded_queries"] = result.query_expansion

        return output

    async def cmd_stats(self, args):
        """Show system statistics"""
        from ragbits_integration.knowledge_base.extraction import KnowledgeExtractor
        from ragbits_integration.evaluation import EvaluationMetricsCollector

        output = {"command": "stats", "statistics": {}}

        if args.type == "extraction":
            # Would need a shared extractor instance
            output["statistics"] = {
                "note": "Extraction statistics require persistent instance"
            }
        elif args.type == "scoring":
            collector = EvaluationMetricsCollector()
            output["statistics"] = collector.get_statistics()
        elif args.type == "storage":
            collector = EvaluationMetricsCollector()
            output["statistics"] = {
                "total_artifacts": len(collector.metrics_store),
                "total_metrics": sum(
                    len(ms.metrics) for ms in collector.metrics_store.values()
                )
            }
        else:
            # All statistics
            collector = EvaluationMetricsCollector()
            output["statistics"] = {
                "artifacts_stored": len(collector.metrics_store),
                "metrics_collected": sum(
                    len(ms.metrics) for ms in collector.metrics_store.values()
                ),
                "metric_sets_available": len(collector.metrics_store)
            }

        return output

    async def cmd_validate(self, args):
        """Run gauntlet validation"""
        from ragbits_integration.evaluation.gauntlets import EnhancedGauntletValidator
        from ragbits_integration.evaluation.metrics import EvaluationMetricsCollector

        # Read artifact content (for demo, assume it's stored)
        # In practice, would load from storage
        print(f"Running gauntlet validation for {args.artifact}...")

        # Initialize
        collector = EvaluationMetricsCollector()
        validator = EnhancedGauntletValidator(collector)

        # Read artifact content
        # For now, use placeholder
        solution_text = "Solution content would be loaded here"

        # Get requirements
        requirements = args.requirements or []

        # Run validation
        result = await validator.validate_solution(
            artifact_id=args.artifact,
            solution_text=solution_text,
            requirements=requirements
        )

        score = result.multi_dimensional_score

        output = {
            "command": "validate",
            "artifact_id": args.artifact,
            "verdict": score.get_verdict(),
            "overall_score": f"{score.overall_score:.2f}",
            "dimension_scores": {
                "functionality": f"{score.functionality:.1f}",
                "performance": f"{score.performance:.1f}",
                "security": f"{score.security:.1f}",
                "reliability": f"{score.reliability:.1f}",
                "completeness": f"{score.completeness:.1f}",
                "efficiency": f"{score.efficiency:.1f}",
                "maintainability": f"{score.maintainability:.1f}",
                "scalability": f"{score.scalability:.1f}"
            },
            "tests_passed": score.tests_passed,
            "tests_failed": score.tests_failed,
            "tests_total": score.tests_total,
            "critical_dimensions": score.critical_dimensions
        }

        return output

    async def cmd_trend(self, args):
        """Analyze trends"""
        from ragbits_integration.evaluation import HistoricalComparator, MetricsAnalyzer, EvaluationMetricsCollector

        print(f"Analyzing trends for {args.type} over {args.days} days...")

        # Initialize
        collector = EvaluationMetricsCollector()
        analyzer = MetricsAnalyzer(collector)
        comparator = HistoricalComparator(collector, analyzer)

        # Analyze trend
        trend = await comparator.analyze_trends(
            artifact_type=args.type,
            metric_category=args.category,
            window_size=50
        )

        if "error" in trend:
            return {"error": trend["error"]}

        output = {
            "command": "trend",
            "artifact_type": args.type,
            "metric_category": args.category or "overall",
            "trend_direction": trend["trend"]["direction"],
            "slope": trend["trend"]["slope"],
            "start_score": trend["trend"]["start_score"],
            "end_score": trend["trend"]["end_score"],
            "change": trend["trend"]["change"],
            "data_points": trend["data_points"]
        }

        return output

    def _determine_verdict(self, score: float) -> str:
        """Determine verdict from score"""
        if score >= 8.0:
            return "EXCELLENT"
        elif score >= 6.5:
            return "GOOD"
        elif score >= 5.0:
            return "ACCEPTABLE"
        else:
            return "POOR"

    def _print_table(self, data: dict):
        """Print data as table"""
        if "error" in data:
            print(f"Error: {data['error']}")
            return

        for key, value in data.items():
            if isinstance(value, (list, dict)):
                print(f"{key}:")
                print(f"  {value}")
            else:
                print(f"{key}: {value}")

    def _print_text(self, data: dict):
        """Print data as formatted text"""
        if "error" in data:
            print(f"❌ Error: {data['error']}")
            return

        print("=" * 70)
        print(f"Command: {data.get('command', 'unknown')}")
        print("=" * 70)

        for key, value in data.items():
            if key == "command":
                continue

            if isinstance(value, dict):
                print(f"\n{key.replace('_', ' ').title()}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            elif isinstance(value, list) and value:
                print(f"\n{key.replace('_', ' ').title()}:")
                for i, item in enumerate(value[:5], 1):
                    print(f"  {i}. {item}")
                if len(value) > 5:
                    print(f"  ... and {len(value) - 5} more")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")

        print("\n" + "=" * 70)


def main():
    """Main entry point"""
    cli = RAGBitsCLI()
    cli.run()


if __name__ == "__main__":
    main()
