"""
Load Test Report Generator

Generates HTML and PDF reports from load test results.

Usage:
    python generate_report.py --input load_test_results.json --output report.html
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def generate_html_report(results_path: str, output_path: str):
    """
    Generate HTML report from test results.

    Args:
        results_path: Path to results JSON file
        output_path: Path to save HTML report
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)

    tests = data.get("tests", [])
    timestamp = data.get("timestamp", datetime.utcnow().isoformat())

    # Calculate summary
    total_tests = len(tests)
    passed_tests = sum(1 for t in tests if t["passed"])
    failed_tests = total_tests - passed_tests
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Load Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}

        .header .timestamp {{
            opacity: 0.9;
            margin-top: 10px;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }}

        .summary-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}

        .summary-card.pass .value {{ color: #10b981; }}
        .summary-card.fail .value {{ color: #ef4444; }}

        .test-section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .test-section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}

        .test-result {{
            background: #f9fafb;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            border-left: 4px solid #10b981;
        }}

        .test-result.failed {{
            border-left-color: #ef4444;
        }}

        .test-result .status {{
            font-weight: bold;
            margin-bottom: 10px;
        }}

        .test-result.passed .status {{ color: #10b981; }}
        .test-result.failed .status {{ color: #ef4444; }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}

        .metric {{
            background: white;
            padding: 10px;
            border-radius: 4px;
        }}

        .metric-label {{
            font-size: 0.85em;
            color: #666;
        }}

        .metric-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }}

        .errors {{
            margin-top: 10px;
            padding: 10px;
            background: #fef2f2;
            border-radius: 4px;
            border-left: 3px solid #ef4444;
        }}

        .warnings {{
            margin-top: 10px;
            padding: 10px;
            background: #fffbeb;
            border-radius: 4px;
            border-left: 3px solid #f59e0b;
        }}

        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Load Test Report</h1>
        <div class="timestamp">Generated: {timestamp}</div>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>Total Tests</h3>
            <div class="value">{total_tests}</div>
        </div>
        <div class="summary-card pass">
            <h3>Passed</h3>
            <div class="value">{passed_tests}</div>
        </div>
        <div class="summary-card fail">
            <h3>Failed</h3>
            <div class="value">{failed_tests}</div>
        </div>
        <div class="summary-card">
            <h3>Pass Rate</h3>
            <div class="value">{pass_rate:.1f}%</div>
        </div>
    </div>

    <div class="test-section">
        <h2>Test Results</h2>
"""

    # Add test results
    for test in tests:
        status_class = "passed" if test["passed"] else "failed"
        status_text = "✓ PASSED" if test["passed"] else "✗ FAILED"

        metrics = test.get("metrics", {})

        html += f"""
        <div class="test-result {status_class}">
            <div class="status">{status_text}: {test['test_name']}</div>
            <div class="metrics">
"""

        # Add key metrics
        if "throughput_ops_per_sec" in metrics:
            html += f"""
                <div class="metric">
                    <div class="metric-label">Throughput</div>
                    <div class="metric-value">{metrics['throughput_ops_per_sec']:.2f} ops/s</div>
                </div>
"""

        if "error_rate" in metrics:
            html += f"""
                <div class="metric">
                    <div class="metric-label">Error Rate</div>
                    <div class="metric-value">{metrics['error_rate']:.2%}</div>
                </div>
"""

        if "concurrent_users" in metrics:
            html += f"""
                <div class="metric">
                    <div class="metric-label">Concurrent Users</div>
                    <div class="metric-value">{metrics['concurrent_users']}</div>
                </div>
"""

        if "duration_seconds" in metrics:
            html += f"""
                <div class="metric">
                    <div class="metric-label">Duration</div>
                    <div class="metric-value">{metrics['duration_seconds']:.1f}s</div>
                </div>
"""

        html += """
            </div>
"""

        # Add errors if any
        if test.get("errors"):
            html += """
            <div class="errors">
                <strong>Errors:</strong>
"""
            for error in test["errors"]:
                html += f"                <div>• {error}</div>\n"
            html += "            </div>\n"

        # Add warnings if any
        if test.get("warnings"):
            html += """
            <div class="warnings">
                <strong>Warnings:</strong>
"""
            for warning in test["warnings"]:
                html += f"                <div>• {warning}</div>\n"
            html += "            </div>\n"

        html += "        </div>\n"

    # Footer
    html += """
    </div>

    <div class="footer">
        <p>Knowledge Graph Load Testing Framework v1.0.0</p>
        <p>Generated automatically from test results</p>
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html)

    logger.info(f"HTML report generated: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate HTML report from load test results"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to load test results JSON file"
    )
    parser.add_argument(
        "--output",
        default="load_test_report.html",
        help="Path to save HTML report (default: load_test_report.html)"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Generate report
    try:
        generate_html_report(args.input, args.output)
        print(f"\n✓ Report generated successfully: {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Failed to generate report: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
