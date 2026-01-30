"""
Conflict Detector Usage Examples

This file provides comprehensive examples of how to use the conflict_detector module
for detecting and resolving conflicts in sub-solutions.

Author: OpenEvolve AI System
Version: 1.0.0
"""

from conflict_detector import (
    ConflictDetector,
    ConflictReporter,
    Conflict,
    ConflictType,
    ConflictSeverity,
    detect_conflicts,
    analyze_naming_conflicts,
    analyze_logic_conflicts,
    analyze_dependency_conflicts,
    assess_conflict_severity,
    propose_resolution
)


def example_1_basic_usage():
    """
    Example 1: Basic conflict detection

    Demonstrates simple usage of the conflict detector with two solutions
    that have naming conflicts.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Conflict Detection")
    print("=" * 80)

    # Define two solutions with conflicting function names
    solution_1 = """
def process_data(data):
    return data.upper()

def validate_input(input_data):
    return len(input_data) > 0
"""

    solution_2 = """
def process_data(data):
    return data.lower()

def validate_input(input_data):
    return input_data is not None
"""

    # Detect conflicts using convenience function
    conflicts = detect_conflicts(
        sub_solutions=[solution_1, solution_2],
        metadata=[{'id': 'solution_1'}, {'id': 'solution_2'}]
    )

    print(f"\nTotal conflicts detected: {len(conflicts)}")

    for i, conflict in enumerate(conflicts, 1):
        print(f"\nConflict {i}:")
        print(f"  Type: {conflict.conflict_type.value}")
        print(f"  Severity: {conflict.severity.value}")
        print(f"  Description: {conflict.description}")
        print(f"  Affected Solutions: {', '.join(conflict.affected_solutions)}")
        print(f"  Confidence: {conflict.confidence:.2f}")


def example_2_naming_conflicts():
    """
    Example 2: Detecting naming conflicts

    Focuses on naming-specific conflicts including duplicates,
    shadowing, and inconsistent naming.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Naming Conflict Detection")
    print("=" * 80)

    solutions = [
        """
class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self, item):
        return item * 2
""",
        """
class DataProcessor:
    def __init__(self):
        self.data = {}

    def process(self, item):
        return item + 1
""",
        """
def helper():
    return "result"

list = [1, 2, 3]  # Shadows builtin
"""
    ]

    # Analyze only naming conflicts
    conflicts = analyze_naming_conflicts(solutions)

    print(f"\nNaming conflicts found: {len(conflicts)}")

    for conflict in conflicts:
        print(f"\n{conflict.severity.value} - {conflict.description}")
        if 'builtin' in conflict.description.lower():
            print("  ⚠️  Warning: Builtin shadowing detected!")


def example_3_logic_conflicts():
    """
    Example 3: Detecting logic conflicts

    Demonstrates detection of contradictory logic, incompatible
    control flow, and state management issues.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Logic Conflict Detection")
    print("=" * 80)

    solutions = [
        """
def enable_feature():
    config['feature_enabled'] = True
    return True

def verify_positive(value):
    assert value > 0, "Value must be positive"
    return True
""",
        """
def disable_feature():
    config['feature_enabled'] = False
    return False

def verify_negative(value):
    assert value < 0, "Value must be negative"
    return False
"""
    ]

    # Analyze logic conflicts
    conflicts = analyze_logic_conflicts(solutions)

    print(f"\nLogic conflicts found: {len(conflicts)}")

    for conflict in conflicts:
        print(f"\n{conflict.conflict_type.value}: {conflict.description}")
        print(f"Severity: {conflict.severity.value}")

        # Show resolution proposal
        if conflict.suggested_resolution:
            resolution = conflict.suggested_resolution
            print(f"\nProposed Resolution:")
            print(f"  Strategy: {resolution.get('strategy', 'N/A')}")
            if 'options' in resolution:
                print("  Options:")
                for option in resolution['options']:
                    print(f"    - {option}")


def example_4_dependency_conflicts():
    """
    Example 4: Detecting dependency conflicts

    Shows detection of API incompatibilities, circular dependencies,
    and import conflicts.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Dependency Conflict Detection")
    print("=" * 80)

    solutions = [
        """
import threading
import time

def process_work():
    def worker():
        time.sleep(1)
        print("Work done")

    thread = threading.Thread(target=worker)
    thread.start()
    return thread
""",
        """
import asyncio
import time

async def process_work():
    await asyncio.sleep(1)
    print("Work done")
    return True
""",
        """
# Simulated circular dependency
from solution_2 import helper

def another_helper():
    return helper()
"""
    ]

    # Analyze dependency conflicts
    conflicts = analyze_dependency_conflicts(solutions)

    print(f"\nDependency conflicts found: {len(conflicts)}")

    for conflict in conflicts:
        print(f"\n{conflict.severity.value} - {conflict.description}")

        if conflict.metadata:
            print(f"Details: {conflict.metadata}")


def example_5_comprehensive_analysis():
    """
    Example 5: Comprehensive conflict analysis

    Performs a complete analysis using the ConflictDetector class
    and generates detailed reports.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Comprehensive Analysis")
    print("=" * 80)

    # Real-world scenario: Multiple approaches to data processing
    solutions = [
        # Solution 1: JSON-based synchronous approach
        """
import json
import os
from typing import List, Dict

class JSONDataProcessor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def process_data(self, data: List[Dict]) -> List[Dict]:
        return [self._transform(item) for item in data]

    def _transform(self, item: Dict) -> Dict:
        item['processed'] = True
        return item
""",

        # Solution 2: Async approach with different naming
        """
import asyncio
import aiofiles
from typing import List, Dict

class AsyncDataProcessor:
    def __init__(self, config_path: str):
        self.config_path = config_path

    async def process_data(self, data: List[Dict]) -> List[Dict]:
        tasks = [self._transform_async(item) for item in data]
        return await asyncio.gather(*tasks)

    async def _transform_async(self, item: Dict) -> Dict:
        await asyncio.sleep(0.1)
        item['async_processed'] = True
        return item
""",

        # Solution 3: Threading-based approach
        """
import threading
import queue
from typing import List, Dict

class ThreadedProcessor:
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads
        self.queue = queue.Queue()

    def process_data(self, data: List[Dict]) -> List[Dict]:
        threads = []
        for item in data:
            t = threading.Thread(target=self._process_item, args=(item,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return data

    def _process_item(self, item: Dict):
        item['thread_processed'] = True
"""
    ]

    metadata = [
        {'id': 'json_processor', 'author': 'Alice', 'version': '1.0'},
        {'id': 'async_processor', 'author': 'Bob', 'version': '2.0'},
        {'id': 'threaded_processor', 'author': 'Charlie', 'version': '1.5'}
    ]

    # Create detector and analyze
    detector = ConflictDetector(strict_mode=False)
    conflicts = detector.detect_conflicts(solutions, metadata)

    print(f"\nAnalyzing {len(solutions)} solutions...")
    print(f"Total conflicts detected: {len(conflicts)}")

    # Group by severity
    by_severity = {}
    for conflict in conflicts:
        severity = conflict.severity.value
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(conflict)

    print("\nBreakdown by severity:")
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = len(by_severity.get(severity, []))
        if count > 0:
            print(f"  {severity}: {count} conflicts")

    # Show detailed analysis
    print("\n" + "-" * 80)
    print("DETAILED CONFLICT ANALYSIS")
    print("-" * 80)

    for i, conflict in enumerate(conflicts[:5], 1):  # Show first 5
        print(f"\n{i}. {conflict.conflict_type.value} [{conflict.severity.value}]")
        print(f"   {conflict.description}")
        print(f"   Solutions: {', '.join(conflict.affected_solutions)}")

        # Show resolution
        resolution = propose_resolution(conflict)
        print(f"   Resolution: {resolution.get('strategy', 'N/A')}")
        if 'implementation_steps' in resolution:
            print("   Steps:")
            for step in resolution['implementation_steps']:
                print(f"     {step}")


def example_6_generating_reports():
    """
    Example 6: Generating reports in different formats

    Demonstrates text, JSON, and Markdown report generation.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Report Generation")
    print("=" * 80)

    # Create sample conflicts
    solutions = [
        "def process(): pass",
        "def process(): pass",
        "import threading\npass",
        "import asyncio\npass"
    ]

    conflicts = detect_conflicts(
        solutions,
        [{'id': f's{i}'} for i in range(len(solutions))]
    )

    # Generate different report formats
    print("\n1. TEXT REPORT:")
    print("-" * 80)
    text_report = ConflictReporter.generate_report(conflicts, 'text')
    print(text_report[:500] + "..." if len(text_report) > 500 else text_report)

    print("\n2. JSON REPORT:")
    print("-" * 80)
    json_report = ConflictReporter.generate_report(conflicts, 'json')
    print(json_report[:300] + "..." if len(json_report) > 300 else json_report)

    print("\n3. MARKDOWN REPORT:")
    print("-" * 80)
    markdown_report = ConflictReporter.generate_report(conflicts, 'markdown')
    print(markdown_report[:500] + "..." if len(markdown_report) > 500 else markdown_report)

    # Save reports to files (uncomment to use)
    # with open('conflict_report.txt', 'w') as f:
    #     f.write(text_report)
    #
    # with open('conflict_report.json', 'w') as f:
    #     f.write(json_report)
    #
    # with open('conflict_report.md', 'w') as f:
    #     f.write(markdown_report)


def example_7_custom_conflict_analysis():
    """
    Example 7: Custom conflict analysis workflow

    Shows how to build a custom workflow using the detector's
    individual components.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Custom Analysis Workflow")
    print("=" * 80)

    # Step 1: Analyze solutions individually
    detector = ConflictDetector()

    solution_a = """
class DataHandler:
    def process(self, data):
        return self._transform(data)

    def _transform(self, data):
        return data * 2
"""

    solution_b = """
class DataHandler:
    def process(self, data):
        return self._validate(data)

    def _validate(self, data):
        assert data is not None
        return data
"""

    print("\nStep 1: Analyzing solutions individually...")
    analysis_a = detector._analyze_solution(solution_a, 'handler_a')
    analysis_b = detector._analyze_solution(solution_b, 'handler_b')

    print(f"  Handler A: {len(analysis_a.names_defined)} names defined")
    print(f"  Handler B: {len(analysis_b.names_defined)} names defined")

    # Step 2: Detect conflicts
    print("\nStep 2: Detecting conflicts...")
    conflicts = detector.detect_conflicts(
        [solution_a, solution_b],
        [{'id': 'handler_a'}, {'id': 'handler_b'}]
    )

    # Step 3: Filter by severity
    print("\nStep 3: Filtering by severity...")
    critical_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.CRITICAL]
    high_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.HIGH]

    print(f"  Critical: {len(critical_conflicts)}")
    print(f"  High: {len(high_conflicts)}")

    # Step 4: Generate resolution plans
    print("\nStep 4: Generating resolution plans...")
    for conflict in high_conflicts:
        resolution = propose_resolution(conflict)
        print(f"\n  Conflict: {conflict.description[:60]}...")
        print(f"  Strategy: {resolution.get('strategy', 'N/A')}")

    # Step 5: Create action items
    print("\nStep 5: Creating action items...")
    action_items = []
    for conflict in conflicts:
        if conflict.conflict_type == ConflictType.NAMING_CONFLICT:
            action_items.append({
                'priority': conflict.severity.value,
                'action': 'Rename conflicting identifiers',
                'solutions': conflict.affected_solutions
            })

    print(f"\n  Total action items: {len(action_items)}")
    for item in action_items[:3]:
        print(f"    [{item['priority']}] {item['action']}")


def example_8_real_world_scenario():
    """
    Example 8: Real-world scenario - API integration

    Simulates a real scenario where multiple developers propose
    different solutions for API integration.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Real-World Scenario - API Integration")
    print("=" * 80)

    # Developer 1: Synchronous requests
    dev1_solution = """
import requests
from typing import Dict, Any

class APIClient:
    BASE_URL = "https://api.example.com"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def fetch_data(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def post_data(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
"""

    # Developer 2: Async/aiohttp
    dev2_solution = """
import aiohttp
import asyncio
from typing import Dict, Any

class APIClient:
    BASE_URL = "https://api.example.com"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def fetch_data(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()

    async def post_data(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                return await response.json()
"""

    # Developer 3: HTTPX with sync/async support
    dev3_solution = """
import httpx
from typing import Dict, Any

class APIClient:
    BASE_URL = "https://api.example.com"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.Client()

    async def fetch_data_async(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    def fetch_data(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.client.get(url)
        response.raise_for_status()
        return response.json()
"""

    solutions = [dev1_solution, dev2_solution, dev3_solution]
    metadata = [
        {'id': 'requests_client', 'developer': 'Alice', 'approach': 'sync'},
        {'id': 'aiohttp_client', 'developer': 'Bob', 'approach': 'async'},
        {'id': 'httpx_client', 'developer': 'Charlie', 'approach': 'hybrid'}
    ]

    print("\nScenario: Three developers propose different API client implementations")
    print("-" * 80)

    # Detect conflicts
    detector = ConflictDetector()
    conflicts = detector.detect_conflicts(solutions, metadata)

    print(f"\nAnalysis complete. Found {len(conflicts)} potential issues.")

    # Categorize conflicts
    categories = {
        'Naming': [c for c in conflicts if c.conflict_type == ConflictType.NAMING_CONFLICT],
        'Logic': [c for c in conflicts if c.conflict_type == ConflictType.LOGIC_CONFLICT],
        'Dependency': [c for c in conflicts if c.conflict_type == ConflictType.DEPENDENCY_CONFLICT]
    }

    print("\nConflict Breakdown:")
    for category, items in categories.items():
        print(f"  {category}: {len(items)}")

    # Show recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if categories['Dependency']:
        print("\n⚠️  Dependency Conflicts Detected:")
        print("   The solutions use incompatible HTTP libraries (requests, aiohttp, httpx).")
        print("\n   Recommended Actions:")
        print("   1. Choose one library to standardize on")
        print("   2. If async support is needed, use httpx (supports both sync/async)")
        print("   3. If using aiohttp, separate sync and async code paths")

    if categories['Naming']:
        print("\n⚠️  Naming Conflicts Detected:")
        print("   All solutions define classes with the same name 'APIClient'.")
        print("\n   Recommended Actions:")
        print("   1. Use descriptive names: RequestsAPIClient, AsyncAPIClient, HybridAPIClient")
        print("   2. Or use namespace/package structure")
        print("   3. Or consolidate into single implementation with strategy pattern")


def example_9_edge_cases():
    """
    Example 9: Handling edge cases

    Demonstrates how the detector handles unusual or edge cases.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Edge Cases")
    print("=" * 80)

    edge_cases = [
        ("Empty solution", ""),
        ("Syntax error", "def foo(\n"),
        ("Unicode names", "def café(): return 'coffee'"),
        ("Deeply nested", """
class Outer:
    class Middle:
        class Inner:
            def method(self):
                pass
"""),
        ("Many imports", """
import os
import sys
import json
import re
import time
import datetime
import collections
""")
    ]

    detector = ConflictDetector()

    for name, code in edge_cases:
        print(f"\nTesting: {name}")
        try:
            conflicts = detector.detect_conflicts([code], [{'id': name}])
            print(f"  ✓ Handled successfully, found {len(conflicts)} conflicts")
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("CONFLICT DETECTOR - USAGE EXAMPLES")
    print("=" * 80)
    print("\nThis file contains 9 comprehensive examples of using the conflict_detector module.")
    print("Each example demonstrates different features and use cases.")

    # Run examples (comment out those you don't want to see)
    example_1_basic_usage()
    example_2_naming_conflicts()
    example_3_logic_conflicts()
    example_4_dependency_conflicts()
    example_5_comprehensive_analysis()
    example_6_generating_reports()
    example_7_custom_conflict_analysis()
    example_8_real_world_scenario()
    example_9_edge_cases()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nFor more information, see the module documentation and unit tests.")
    print("\nKey takeaways:")
    print("  • Use detect_conflicts() for quick analysis")
    print("  • Use ConflictDetector class for detailed control")
    print("  • Use ConflictReporter to generate formatted reports")
    print("  • Filter conflicts by severity to prioritize fixes")
    print("  • Use propose_resolution() to get actionable recommendations")
    print("\n")


if __name__ == '__main__':
    main()
