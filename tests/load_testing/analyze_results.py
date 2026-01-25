"""
Load Test Result Analyzer

Analyzes load test results and generates comprehensive reports with
performance insights, bottleneck identification, and capacity planning.

Usage:
    from analyze_results import LoadTestAnalyzer

    analyzer = LoadTestAnalyzer("load_test_results.json")
    analyzer.generate_report("report.txt")
"""

import json
import statistics
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LoadTestAnalyzer:
    """
    Analyze load test results and generate comprehensive reports.

    Analysis capabilities:
    - Performance trends across tests
    - Bottleneck identification
    - Capacity planning recommendations
    - Comparative analysis
    """

    def __init__(self, results_path: str):
        """
        Initialize analyzer with results file.

        Args:
            results_path: Path to load test results JSON file
        """
        self.results_path = Path(results_path)
        self.results = self._load_results()
        self.tests = self.results.get("tests", [])

    def _load_results(self) -> Dict:
        """
        Load results from JSON file.

        Returns:
            Dictionary with test results
        """
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        with open(self.results_path, 'r') as f:
            return json.load(f)

    def analyze_throughput(self) -> Dict:
        """
        Analyze throughput across all tests.

        Returns:
            Dictionary with throughput analysis
        """
        throughputs = []

        for test in self.tests:
            metrics = test.get("metrics", {})
            if "throughput_ops_per_sec" in metrics:
                throughputs.append({
                    "test_name": test["test_name"],
                    "throughput": metrics["throughput_ops_per_sec"],
                    "passed": test["passed"]
                })

        if not throughputs:
            return {"error": "No throughput data found"}

        avg_throughput = statistics.mean([t["throughput"] for t in throughputs])
        max_throughput = max([t["throughput"] for t in throughputs])
        min_throughput = min([t["throughput"] for t in throughputs])

        return {
            "average_throughput": avg_throughput,
            "max_throughput": max_throughput,
            "min_throughput": min_throughput,
            "throughput_by_test": throughputs,
            "total_throughput_tests": len(throughputs)
        }

    def analyze_error_rates(self) -> Dict:
        """
        Analyze error rates across all tests.

        Returns:
            Dictionary with error rate analysis
        """
        error_rates = []

        for test in self.tests:
            metrics = test.get("metrics", {})
            if "error_rate" in metrics:
                error_rates.append({
                    "test_name": test["test_name"],
                    "error_rate": metrics["error_rate"],
                    "passed": test["passed"],
                    "total_operations": metrics.get("total_operations", 0),
                    "errors": int(metrics.get("error_rate", 0) * metrics.get("total_operations", 0))
                })

        if not error_rates:
            return {"error": "No error rate data found"}

        avg_error_rate = statistics.mean([e["error_rate"] for e in error_rates])
        max_error_rate = max([e["error_rate"] for e in error_rates])

        return {
            "average_error_rate": avg_error_rate,
            "max_error_rate": max_error_rate,
            "error_rates_by_test": error_rates,
            "tests_with_errors": sum(1 for e in error_rates if e["error_rate"] > 0),
            "total_errors": sum(e["errors"] for e in error_rates)
        }

    def analyze_response_times(self) -> Dict:
        """
        Analyze response time distributions.

        Returns:
            Dictionary with response time analysis
        """
        response_times = []

        for test in self.tests:
            metrics = test.get("metrics", {})

            # Check for spike test data
            if "baseline_response_time" in metrics and "spike_response_time" in metrics:
                response_times.append({
                    "test_name": test["test_name"],
                    "baseline": metrics["baseline_response_time"],
                    "spike": metrics["spike_response_time"],
                    "degradation": metrics.get("response_time_degradation", 0)
                })

        if not response_times:
            return {"error": "No response time data found"}

        avg_baseline = statistics.mean([r["baseline"] for r in response_times])
        avg_spike = statistics.mean([r["spike"] for r in response_times])
        avg_degradation = statistics.mean([r["degradation"] for r in response_times])

        return {
            "average_baseline_response": avg_baseline,
            "average_spike_response": avg_spike,
            "average_degradation": avg_degradation,
            "response_times_by_test": response_times,
            "significant_degradation": sum(
                1 for r in response_times if r["degradation"] > 0.3
            )
        }

    def identify_bottlenecks(self) -> List[str]:
        """
        Identify system bottlenecks based on test results.

        Returns:
            List of bottleneck descriptions with severity
        """
        bottlenecks = []

        # Check for high error rates
        error_analysis = self.analyze_error_rates()
        if "error_rates_by_test" in error_analysis:
            for test in error_analysis["error_rates_by_test"]:
                if test["error_rate"] > 0.05:  # 5% threshold
                    bottlenecks.append({
                        "severity": "HIGH" if test["error_rate"] > 0.1 else "MEDIUM",
                        "test": test["test_name"],
                        "issue": f"High error rate: {test['error_rate']:.2%}",
                        "recommendation": "Investigate error handling and retry logic"
                    })

        # Check for response time degradation
        response_analysis = self.analyze_response_times()
        if "response_times_by_test" in response_analysis:
            for test in response_analysis["response_times_by_test"]:
                if test["degradation"] > 0.5:  # 50% degradation
                    bottlenecks.append({
                        "severity": "HIGH",
                        "test": test["test_name"],
                        "issue": f"Severe response time degradation: {test['degradation']:.1%}",
                        "recommendation": "Scale resources or optimize query performance"
                    })

        # Check for memory issues
        for test in self.tests:
            metrics = test.get("metrics", {})
            if "memory_growth_gb" in metrics:
                if metrics["memory_growth_gb"] > 0.5:  # 500 MB
                    bottlenecks.append({
                        "severity": "HIGH" if metrics["memory_growth_gb"] > 1.0 else "MEDIUM",
                        "test": test["test_name"],
                        "issue": f"Memory growth: {metrics['memory_growth_gb']:.3f} GB",
                        "recommendation": "Investigate potential memory leaks"
                    })

        # Check for performance degradation over time
        for test in self.tests:
            metrics = test.get("metrics", {})
            if "performance_degradation" in metrics:
                if metrics["performance_degradation"] > 0.2:  # 20% degradation
                    bottlenecks.append({
                        "severity": "MEDIUM",
                        "test": test["test_name"],
                        "issue": f"Performance degradation: {metrics['performance_degradation']:.1%}",
                        "recommendation": "Check for resource exhaustion or caching issues"
                    })

        return bottlenecks

    def estimate_capacity(
        self,
        target_response_time: float = 1.0
    ) -> Dict:
        """
        Estimate system capacity for target response time.

        Args:
            target_response_time: Target response time in seconds

        Returns:
            Dictionary with capacity estimates
        """
        # Find tests with concurrent user data
        user_tests = []
        for test in self.tests:
            metrics = test.get("metrics", {})
            if "concurrent_users" in metrics and "throughput_ops_per_sec" in metrics:
                user_tests.append({
                    "test_name": test["test_name"],
                    "users": metrics["concurrent_users"],
                    "throughput": metrics["throughput_ops_per_sec"]
                })

        if not user_tests:
            return {"error": "Insufficient data for capacity estimation"}

        # Estimate max users based on throughput
        max_throughput_test = max(user_tests, key=lambda x: x["throughput"])

        # Simple linear extrapolation (conservative estimate)
        estimated_max_users = int(max_throughput_test["users"] * 1.5)
        estimated_max_rps = int(max_throughput_test["throughput"] * 1.5)

        # Determine scaling strategy
        scaling_recommendation = "HORIZONTAL"
        if estimated_max_users < 100:
            scaling_recommendation = "SINGLE_INSTANCE"
        elif estimated_max_users < 500:
            scaling_recommendation = "HORIZONTAL"

        return {
            "estimated_max_concurrent_users": estimated_max_users,
            "estimated_max_requests_per_second": estimated_max_rps,
            "target_response_time": target_response_time,
            "scaling_recommendation": scaling_recommendation,
            "baseline_test": max_throughput_test,
            "confidence": "MEDIUM" if len(user_tests) >= 3 else "LOW"
        }

    def generate_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on test results.

        Returns:
            List of recommendations with priority
        """
        recommendations = []
        bottlenecks = self.identify_bottlenecks()

        # Analyze overall test health
        total_tests = len(self.tests)
        passed_tests = sum(1 for t in self.tests if t["passed"])
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0

        # Overall health recommendations
        if pass_rate < 0.5:
            recommendations.append({
                "priority": "CRITICAL",
                "category": "System Health",
                "recommendation": f"Only {pass_rate:.1%} of tests passed. Major system issues detected.",
                "action": "Review and fix critical failures before production deployment"
            })
        elif pass_rate < 0.8:
            recommendations.append({
                "priority": "HIGH",
                "category": "System Health",
                "recommendation": f"Pass rate is {pass_rate:.1%}. Several tests failed.",
                "action": "Address failing tests before scaling"
            })

        # Bottleneck-based recommendations
        for bottleneck in bottlenecks:
            recommendations.append({
                "priority": bottleneck["severity"],
                "category": "Performance",
                "recommendation": bottleneck["issue"],
                "action": bottleneck["recommendation"]
            })

        # Throughput recommendations
        throughput_analysis = self.analyze_throughput()
        if "average_throughput" in throughput_analysis:
            avg_throughput = throughput_analysis["average_throughput"]
            if avg_throughput < 50:
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Throughput",
                    "recommendation": f"Low throughput: {avg_throughput:.1f} ops/sec",
                    "action": "Consider horizontal scaling or query optimization"
                })

        # Capacity recommendations
        capacity = self.estimate_capacity()
        if "estimated_max_concurrent_users" in capacity:
            max_users = capacity["estimated_max_concurrent_users"]
            if max_users < 100:
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Capacity Planning",
                    "recommendation": f"Estimated capacity: {max_users} concurrent users",
                    "action": "Plan for horizontal scaling if higher loads are expected"
                })

        # Caching recommendations
        has_read_heavy = any(t["test_name"] == "read_heavy" for t in self.tests)
        if has_read_heavy:
            recommendations.append({
                "priority": "LOW",
                "category": "Optimization",
                "recommendation": "Read-heavy workload detected",
                "action": "Implement caching layer for frequently accessed data"
            })

        return recommendations

    def generate_report(self, output_path: str):
        """
        Generate comprehensive load test report.

        Args:
            output_path: Path to save report
        """
        report_lines = []

        # Header
        report_lines.append("="*70)
        report_lines.append("KNOWLEDGE GRAPH LOAD TEST REPORT")
        report_lines.append("="*70)
        report_lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        report_lines.append(f"Results File: {self.results_path}")
        report_lines.append("")

        # Executive Summary
        report_lines.append("-"*70)
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-"*70)
        report_lines.append("")

        total_tests = len(self.tests)
        passed_tests = sum(1 for t in self.tests if t["passed"])
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0

        report_lines.append(f"Total Tests: {total_tests}")
        report_lines.append(f"Passed: {passed_tests}")
        report_lines.append(f"Failed: {total_tests - passed_tests}")
        report_lines.append(f"Pass Rate: {pass_rate:.1%}")

        # Overall status
        if pass_rate == 1.0:
            status = "EXCELLENT - All tests passed"
        elif pass_rate >= 0.8:
            status = "GOOD - Most tests passed"
        elif pass_rate >= 0.5:
            status = "FAIR - Many tests failed"
        else:
            status = "POOR - Most tests failed"

        report_lines.append(f"Overall Status: {status}")
        report_lines.append("")

        # Test Results
        report_lines.append("-"*70)
        report_lines.append("TEST RESULTS")
        report_lines.append("-"*70)
        report_lines.append("")

        for test in self.tests:
            status = "✓ PASSED" if test["passed"] else "✗ FAILED"
            report_lines.append(f"{status}: {test['test_name']}")
            report_lines.append(f"  Timestamp: {test['timestamp']}")

            metrics = test.get("metrics", {})
            if "throughput_ops_per_sec" in metrics:
                report_lines.append(f"  Throughput: {metrics['throughput_ops_per_sec']:.2f} ops/sec")
            if "error_rate" in metrics:
                report_lines.append(f"  Error Rate: {metrics['error_rate']:.2%}")
            if "concurrent_users" in metrics:
                report_lines.append(f"  Concurrent Users: {metrics['concurrent_users']}")
            if "duration_seconds" in metrics:
                report_lines.append(f"  Duration: {metrics['duration_seconds']:.1f}s")

            if test["errors"]:
                report_lines.append("  Errors:")
                for error in test["errors"]:
                    report_lines.append(f"    - {error}")

            if test["warnings"]:
                report_lines.append("  Warnings:")
                for warning in test["warnings"]:
                    report_lines.append(f"    - {warning}")

            report_lines.append("")

        # Performance Analysis
        report_lines.append("-"*70)
        report_lines.append("PERFORMANCE ANALYSIS")
        report_lines.append("-"*70)
        report_lines.append("")

        # Throughput
        throughput_analysis = self.analyze_throughput()
        if "average_throughput" in throughput_analysis:
            report_lines.append("Throughput:")
            report_lines.append(f"  Average: {throughput_analysis['average_throughput']:.2f} ops/sec")
            report_lines.append(f"  Maximum: {throughput_analysis['max_throughput']:.2f} ops/sec")
            report_lines.append(f"  Minimum: {throughput_analysis['min_throughput']:.2f} ops/sec")
            report_lines.append("")

        # Error Rates
        error_analysis = self.analyze_error_rates()
        if "average_error_rate" in error_analysis:
            report_lines.append("Error Rates:")
            report_lines.append(f"  Average: {error_analysis['average_error_rate']:.2%}")
            report_lines.append(f"  Maximum: {error_analysis['max_error_rate']:.2%}")
            report_lines.append(f"  Total Errors: {error_analysis['total_errors']}")
            report_lines.append("")

        # Response Times
        response_analysis = self.analyze_response_times()
        if "average_baseline_response" in response_analysis:
            report_lines.append("Response Times:")
            report_lines.append(f"  Baseline: {response_analysis['average_baseline_response']:.3f}s")
            report_lines.append(f"  Under Load: {response_analysis['average_spike_response']:.3f}s")
            report_lines.append(f"  Degradation: {response_analysis['average_degradation']:.1%}")
            report_lines.append("")

        # Bottlenecks
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            report_lines.append("-"*70)
            report_lines.append("IDENTIFIED BOTTLENECKS")
            report_lines.append("-"*70)
            report_lines.append("")

            for bottleneck in bottlenecks:
                report_lines.append(f"[{bottleneck['severity']}] {bottleneck['test']}")
                report_lines.append(f"  Issue: {bottleneck['issue']}")
                report_lines.append(f"  Recommendation: {bottleneck['recommendation']}")
                report_lines.append("")

        # Recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            report_lines.append("-"*70)
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-"*70)
            report_lines.append("")

            for rec in recommendations:
                report_lines.append(f"[{rec['priority']}] {rec['category']}")
                report_lines.append(f"  {rec['recommendation']}")
                report_lines.append(f"  Action: {rec['action']}")
                report_lines.append("")

        # Capacity Planning
        capacity = self.estimate_capacity()
        if "estimated_max_concurrent_users" in capacity:
            report_lines.append("-"*70)
            report_lines.append("CAPACITY PLANNING")
            report_lines.append("-"*70)
            report_lines.append("")
            report_lines.append(f"Estimated Max Concurrent Users: {capacity['estimated_max_concurrent_users']}")
            report_lines.append(f"Estimated Max Requests/Second: {capacity['estimated_max_requests_per_second']}")
            report_lines.append(f"Scaling Strategy: {capacity['scaling_recommendation']}")
            report_lines.append(f"Confidence Level: {capacity['confidence']}")
            report_lines.append("")

        # Footer
        report_lines.append("="*70)
        report_lines.append("END OF REPORT")
        report_lines.append("="*70)

        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Report generated: {output_path}")

        # Print summary to console
        print("\n" + "="*70)
        print("LOAD TEST ANALYSIS COMPLETE")
        print("="*70)
        print(f"Report saved to: {output_path}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Bottlenecks Identified: {len(bottlenecks)}")
        print(f"Recommendations: {len(recommendations)}")
        print("="*70)
