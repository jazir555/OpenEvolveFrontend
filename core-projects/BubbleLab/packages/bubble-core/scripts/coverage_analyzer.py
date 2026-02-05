#!/usr/bin/env python3
"""
Coverage Gap Analyzer

Identifies files without tests and generates comprehensive test plans
for achieving 100% code coverage.
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

class CoverageAnalyzer:
    def __init__(self, src_dir: str):
        self.src_dir = Path(src_dir)
        self.source_files = []
        self.test_files = []
        self.files_without_tests = []
        self.coverage_gaps = defaultdict(lambda: {
            'missing_branches': [],
            'missing_functions': [],
            'complexity_score': 0
        })

    def find_files(self) -> None:
        """Find all source and test files"""
        print("🔍 Scanning for source and test files...")

        for f in self.src_dir.rglob("*.ts"):
            if "test" in f.name or "spec" in f.name:
                self.test_files.append(f)
            elif f.name.endswith(".ts") and not f.name.endswith(".d.ts"):
                self.source_files.append(f)

        print(f"[OK] Found {len(self.source_files)} source files")
        print(f"[OK] Found {len(self.test_files)} test files")

    def identify_missing_tests(self) -> None:
        """Identify source files without corresponding test files"""
        print("\n🔍 Identifying files without tests...")

        for source_file in self.source_files:
            # Try to find a test file
            test_name = source_file.with_suffix(".test.ts")
            alt_test_name = source_file.with_suffix(".spec.ts")
            alt_test_name2 = source_file.parent / f"{source_file.stem}.test.ts"

            if not (test_name.exists() or alt_test_name.exists() or alt_test_name2.exists()):
                self.files_without_tests.append(source_file)

        print(f"[OK] Found {len(self.files_without_tests)} files without tests")

    def analyze_complexity(self, file_path: Path) -> int:
        """
        Analyze file complexity based on:
        - Number of functions
        - Number of branches
        - Nesting depth
        - Number of try/catch blocks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple complexity metrics (since we can't parse TypeScript with ast)
            complexity = 0

            # Count functions
            functions = re.findall(r'\b(function\s+\w+|=>\s*{|async\s+\w+\(|const\s+\w+\s*=\s*\([^)]*\)\s*=>)', content)
            complexity += len(functions) * 2

            # Count branches (if statements)
            branches = re.findall(r'\bif\s*\(', content)
            complexity += len(branches) * 2

            # Count try/catch blocks
            try_catch = re.findall(r'\btry\s*{', content)
            complexity += len(try_catch) * 3

            # Count switches
            switches = re.findall(r'\bswitch\s*\(', content)
            complexity += len(switches) * 3

            return complexity
        except Exception as e:
            print(f"  [WARN]  Error analyzing {file_path}: {e}")
            return 0

    def prioritize_files(self) -> List[Tuple[Path, int]]:
        """
        Prioritize files by:
        1. Complexity (higher priority)
        2. Whether it's a service/tool bubble (higher priority)
        3. Core functionality (higher priority)
        """
        print("\n🔍 Prioritizing files for testing...")

        priorities = []
        for file_path in self.files_without_tests:
            complexity = self.analyze_complexity(file_path)

            # Boost priority for certain file types
            path_str = str(file_path)
            boost = 0

            if 'service-bubble' in path_str:
                boost += 50
            if 'tool-bubble' in path_str:
                boost += 40
            if 'bubble-flow' in path_str:
                boost += 30
            if file_path.name == 'index.ts':
                boost += 20

            priority = complexity + boost
            priorities.append((file_path, priority))

        # Sort by priority (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)

        print(f"[OK] Prioritized {len(priorities)} files")
        return priorities

    def generate_test_template(self, source_file: Path) -> str:
        """Generate a comprehensive test template for a source file"""
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract class/function names
            classes = re.findall(r'export\s+class\s+(\w+)', content)
            functions = re.findall(r'export\s+(?:async\s+)?function\s+(\w+)', content)
            constants = re.findall(r'export\s+const\s+(\w+)', content)

            # Determine test type
            test_type = "Unit"
            if 'service-bubble' in str(source_file):
                test_type = "Service Bubble"
            elif 'tool-bubble' in str(source_file):
                test_type = "Tool Bubble"

            template = f"""/**
 * Comprehensive Test Suite for {source_file.name}
 *
 * Auto-generated by Coverage Analyzer
 * Test Type: {test_type}
 *
 * Coverage Areas:
 * - All exported functions and classes
 * - Error handling paths
 * - Edge cases and boundary conditions
 * - Input validation
 * - Integration scenarios
 */

import {{ describe, it, expect, beforeEach, afterEach, vi }} from 'vitest';
import {{ }} from './{source_file.stem}';

describe('{source_file.stem.replace('-', ' ').title()}', () => {{
  beforeEach(() => {{
    // Setup test environment
    vi.clearAllMocks();
  }});

  afterEach(() => {{
    // Cleanup
    vi.restoreAllMocks();
  }});

"""

            # Generate tests for classes
            for cls in classes:
                template += f"""
  describe('{cls}', () => {{
    it('should instantiate correctly', () => {{
      // TODO: Implement test
      expect(true).toBe(true);
    }});

    it('should handle initialization errors gracefully', async () => {{
      // TODO: Test error handling
      expect(true).toBe(true);
    }});
  }});
"""

            # Generate tests for functions
            for func in functions:
                template += f"""
  describe('{func}', () => {{
    it('should execute successfully with valid inputs', async () => {{
      // TODO: Implement test
      expect(true).toBe(true);
    }});

    it('should handle null/undefined inputs', async () => {{
      // TODO: Test null handling
      expect(true).toBe(true);
    }});

    it('should handle empty collections', async () => {{
      // TODO: Test empty array/object
      expect(true).toBe(true);
    }});

    it('should handle boundary values', async () => {{
      // TODO: Test min/max values
      expect(true).toBe(true);
    }});

    it('should propagate errors correctly', async () => {{
      // TODO: Test error propagation
      expect(true).toBe(true);
    }});
  }});
"""

            # Add edge cases section
            template += """
  describe('Edge Cases', () => {
    it('should handle concurrent operations', async () => {
      // TODO: Test concurrency
      expect(true).toBe(true);
    });

    it('should handle timeout scenarios', async () => {
      // TODO: Test timeouts
      expect(true).toBe(true);
    });

    it('should cleanup resources on error', async () => {
      // TODO: Test cleanup
      expect(true).toBe(true);
    });
  });

  describe('Performance', () => {
    it('should handle large datasets efficiently', async () => {
      // TODO: Test performance
      expect(true).toBe(true);
    });
  });
});
"""
            return template
        except Exception as e:
            print(f"  [WARN]  Error generating template for {source_file}: {e}")
            return ""

    def generate_report(self) -> Dict:
        """Generate comprehensive coverage report"""
        print("\n📊 Generating coverage report...")

        report = {
            'summary': {
                'total_source_files': len(self.source_files),
                'total_test_files': len(self.test_files),
                'files_without_tests': len(self.files_without_tests),
                'coverage_percentage': round((len(self.test_files) / len(self.source_files)) * 100, 2)
            },
            'prioritized_files': [],
            'by_directory': defaultdict(lambda: {'with_tests': 0, 'without_tests': 0})
        }

        # Get prioritized files
        priorities = self.prioritize_files()
        report['prioritized_files'] = [
            {'file': str(f), 'priority': p}
            for f, p in priorities[:50]  # Top 50
        ]

        # Group by directory
        for f in self.source_files:
            parent = str(f.parent)
            has_test = (
                f.with_suffix(".test.ts").exists() or
                f.with_suffix(".spec.ts").exists()
            )
            if has_test:
                report['by_directory'][parent]['with_tests'] += 1
            else:
                report['by_directory'][parent]['without_tests'] += 1

        print(f"[OK] Generated report")
        return report

    def run(self) -> Dict:
        """Run the complete analysis"""
        print("🚀 Starting Coverage Gap Analysis\n")
        print("="*70)

        self.find_files()
        self.identify_missing_tests()

        priorities = self.prioritize_files()

        print("\n📋 Top 20 Files Requiring Tests (by priority):")
        print("="*70)
        for i, (file_path, priority) in enumerate(priorities[:20], 1):
            rel_path = file_path.relative_to(self.src_dir)
            print(f"{i:2}. {rel_path}")
            print(f"    Priority Score: {priority}")

        report = self.generate_report()

        print("\n" + "="*70)
        print(f"📊 SUMMARY")
        print("="*70)
        print(f"Total Source Files:  {report['summary']['total_source_files']}")
        print(f"Total Test Files:    {report['summary']['total_test_files']}")
        print(f"Files Without Tests: {report['summary']['files_without_tests']}")
        print(f"Test Coverage:       {report['summary']['coverage_percentage']}%")

        # Save report
        report_path = self.src_dir / "coverage_gap_report.json"
        with open(report_path, 'w') as f:
            # Convert defaultdict to dict for JSON serialization
            report_copy = dict(report)
            report_copy['by_directory'] = dict(report['by_directory'])
            json.dump(report_copy, f, indent=2)

        print(f"\n[OK] Full report saved to: {report_path}")

        return report


def main():
    """Main entry point"""
    import sys
    import io

    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    src_dir = Path(__file__).parent.parent / "src"

    analyzer = CoverageAnalyzer(src_dir)
    report = analyzer.run()

    print("\n[OK] Analysis complete!")
    print("\nNext steps:")
    print("1. Review coverage_gap_report.json")
    print("2. Prioritize files based on complexity and importance")
    print("3. Generate tests using the template generator")
    print("4. Run coverage reports to track progress")


if __name__ == "__main__":
    main()
