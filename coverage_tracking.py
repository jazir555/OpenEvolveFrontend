"""
Sovereign-Grade Problem Decomposition System - Code Coverage Tracking
Implements comprehensive code coverage tracking and reporting using pytest-cov.
"""


import os
import subprocess
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import coverage


class CoverageReporter:
    """Generates code coverage reports for the system."""
    
    def __init__(self, project_root: str = ".", source_dir: str = ".", report_dir: str = "coverage_reports"):
        """
        Initialize coverage reporter.
        
        Args:
            project_root: Root directory of the project
            source_dir: Directory containing source code to analyze
            report_dir: Directory to store coverage reports
        """
        self.project_root = Path(project_root)
        self.source_dir = Path(source_dir)
        self.report_dir = Path(report_dir)
        self.logger = logging.getLogger(__name__)
        
        # Set up logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.report_dir.mkdir(exist_ok=True)
        self.coverage_data = None
    
    def run_coverage_analysis(self, test_command: str = "python -m pytest test_suite.py", 
                           source_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run coverage analysis on the codebase.
        
        Args:
            test_command: Command to run tests for coverage analysis
            source_paths: List of source paths to include in coverage (defaults to all Python files)
            
        Returns:
            Dictionary containing coverage results
        """
        self.logger.info("Starting coverage analysis...")
        
        start_time = datetime.now()
        
        if source_paths is None:
            source_paths = [str(self.source_dir)]
        
        try:
            # Create coverage object
            cov = coverage.Coverage(
                source=source_paths,
                omit=[
                    '*/tests/*',
                    '*/test_*.py',
                    '*/conftest.py',
                    '*/venv/*',
                    '*/env/*',
                    '*/__pycache__/*',
                    '*/migrations/*'
                ]
            )
            
            # Start coverage measurement
            cov.start()
            
            # Run tests to gather coverage data
            self.logger.info(f"Running tests: {test_command}")
            # SECURITY FIX: Split command to avoid shell=True for security
            import shlex
            cmd_parts = shlex.split(test_command) if isinstance(test_command, str) else test_command
            result = subprocess.run(
                cmd_parts,
                shell=False,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Test command failed: {result.stderr}")
                # Continue with coverage analysis even if tests fail
            
            # Stop coverage measurement
            cov.stop()
            cov.save()
            
            # Calculate coverage stats
            total_coverage = cov.report()
            
            # Generate HTML report
            html_report_dir = self.report_dir / "html"
            cov.html_report(directory=str(html_report_dir))
            
            # Generate XML report (Cobertura format)
            xml_report_path = self.report_dir / "coverage.xml"
            cov.xml_report(outfile=str(xml_report_path))
            
            # Generate JSON report
            json_report_path = self.report_dir / "coverage.json"
            self._generate_json_report(cov, str(json_report_path))
            
            end_time = datetime.now()
            
            results = {
                'success': True,
                'total_coverage': total_coverage,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': (end_time - start_time).total_seconds(),
                'report_paths': {
                    'html': str(html_report_dir),
                    'xml': str(xml_report_path),
                    'json': str(json_report_path)
                },
                'details': self._get_detailed_coverage_info(cov)
            }
            
            self.coverage_data = results
            self.logger.info(f"Coverage analysis completed. Total coverage: {total_coverage:.2f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during coverage analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_coverage': 0.0,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration': (datetime.now() - start_time).total_seconds(),
                'report_paths': {},
                'details': {}
            }
    
    def _generate_json_report(self, cov: coverage.Coverage, output_path: str) -> bool:
        """Generate JSON coverage report."""
        try:
            # Get coverage data
            data = cov.get_data()
            
            # Create a JSON representation of the coverage data
            coverage_json = {
                'meta': {
                    'timestamp': datetime.now().isoformat(),
                    'covered_lines': data.lines(),
                    'num_files': len(data.measured_files())
                },
                'files': {}
            }
            
            for filename in data.measured_files():
                coverage_stats = data.summary(covered_lines=data.lines(filename))
                coverage_json['files'][filename] = {
                    'summary': coverage_stats,
                    'lines': {
                        line_num: status  # status: 1 for covered, 0 for not covered
                        for line_num, status in enumerate(
                            [1 if line in data.lines(filename) else 0 
                             for line in range(1, max(data.lines(filename) or [0]) + 1)], 
                            1
                        )
                    }
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coverage_json, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {e}")
            return False
    
    def _get_detailed_coverage_info(self, cov: coverage.Coverage) -> Dict[str, Any]:
        """Get detailed coverage information."""
        try:
            # Get formatted coverage data
            data = cov.get_data()
            files = data.measured_files()
            
            detailed_info = {
                'summary': {
                    'num_files': len(files),
                    'total_lines': 0,
                    'covered_lines': 0,
                    'missing_lines': 0
                },
                'files': [],
                'coverage_by_file': {}
            }
            
            total_lines = 0
            total_covered = 0
            
            for filename in files:
                lines = data.lines(filename)
                summary = data.summary(covered_lines=lines)
                
                covered_lines = len(lines)
                all_statements = data.lines(filename) if hasattr(data, 'lines') else []
                
                if all_statements:
                    all_lines_count = max(all_statements) if all_statements else 0
                    missing_lines = all_lines_count - covered_lines
                else:
                    all_lines_count = covered_lines
                    missing_lines = 0
                
                file_coverage = (covered_lines / all_lines_count * 100) if all_lines_count > 0 else 0
                
                file_info = {
                    'filename': filename,
                    'total_lines': all_lines_count,
                    'covered_lines': covered_lines,
                    'missing_lines': missing_lines,
                    'coverage_percent': file_coverage
                }
                
                detailed_info['files'].append(file_info)
                detailed_info['coverage_by_file'][filename] = file_coverage
                
                total_lines += all_lines_count
                total_covered += covered_lines
            
            if total_lines > 0:
                detailed_info['summary']['total_lines'] = total_lines
                detailed_info['summary']['covered_lines'] = total_covered
                detailed_info['summary']['missing_lines'] = total_lines - total_covered
            
            return detailed_info
            
        except Exception as e:
            self.logger.error(f"Error getting detailed coverage info: {e}")
            return {}
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """
        Get the latest coverage report.
        
        Returns:
            Dictionary containing coverage report data
        """
        if self.coverage_data:
            return self.coverage_data
        else:
            return {
                'success': False,
                'error': 'No coverage analysis has been run yet',
                'total_coverage': 0.0,
                'report_paths': {},
                'details': {}
            }
    
    def check_coverage_threshold(self, threshold: float = 80.0) -> bool:
        """
        Check if coverage meets the specified threshold.
        
        Args:
            threshold: Minimum coverage percentage required
            
        Returns:
            True if coverage meets threshold, False otherwise
        """
        report = self.get_coverage_report()
        
        if not report['success']:
            self.logger.error("No successful coverage report available")
            return False
        
        current_coverage = report['total_coverage']
        meets_threshold = current_coverage >= threshold
        
        self.logger.info(f"Coverage: {current_coverage:.2f}% - Threshold: {threshold}% - Meets: {meets_threshold}")
        
        return meets_threshold
    
    def generate_coverage_badge(self, output_path: str = "coverage_badge.svg", 
                              threshold: float = 80.0) -> bool:
        """
        Generate a coverage badge SVG.
        
        Args:
            output_path: Path to save the badge SVG
            threshold: Coverage threshold for color coding
            
        Returns:
            True if badge was generated successfully
        """
        report = self.get_coverage_report()
        
        if not report['success']:
            self.logger.error("Cannot generate badge: no coverage report available")
            return False
        
        coverage = report['total_coverage']
        
        # Determine color based on coverage
        if coverage >= threshold:
            color = "#4c1"  # Green
        elif coverage >= threshold * 0.7:  # 70% of threshold
            color = "#dfb317"  # Yellow
        else:
            color = "#e05d44"  # Red
        
        # Create SVG badge
        badge_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="102" height="20">
  <linearGradient id="a" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <rect rx="3" width="102" height="20" fill="#555"/>
  <rect rx="3" x="37" width="65" height="20" fill="{color}"/>
  <path fill="{color}" d="M37 0h4v20h-4z"/>
  <rect rx="3" width="102" height="20" fill="url(#a)"/>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="19" y="15" fill="#010101" fill-opacity=".3">coverage</text>
    <text x="19" y="14">coverage</text>
    <text x="68" y="15" fill="#010101" fill-opacity=".3">{coverage:.1f}%</text>
    <text x="68" y="14">{coverage:.1f}%</text>
  </g>
</svg>"""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(badge_svg)
            
            self.logger.info(f"Coverage badge generated: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating coverage badge: {e}")
            return False


class CoverageEnforcer:
    """Enforces coverage requirements for the codebase."""
    
    def __init__(self, coverage_reporter: CoverageReporter, required_coverage: float = 80.0):
        """
        Initialize coverage enforcer.
        
        Args:
            coverage_reporter: CoverageReporter instance
            required_coverage: Required minimum coverage percentage
        """
        self.coverage_reporter = coverage_reporter
        self.required_coverage = required_coverage
        self.logger = logging.getLogger(__name__)
    
    def enforce_coverage(self, test_command: str = "python -m pytest test_suite.py") -> Dict[str, Any]:
        """
        Enforce coverage requirements by running analysis and checking threshold.
        
        Args:
            test_command: Command to run tests for coverage analysis
            
        Returns:
            Dictionary with enforcement results
        """
        self.logger.info(f"Enforcing coverage requirement: {self.required_coverage}%")
        
        # Run coverage analysis
        analysis_result = self.coverage_reporter.run_coverage_analysis(test_command)
        
        if not analysis_result['success']:
            return {
                'success': False,
                'enforcement_passed': False,
                'error': analysis_result.get('error', 'Unknown error'),
                'coverage': 0.0,
                'required': self.required_coverage
            }
        
        # Check if coverage meets requirements
        meets_threshold = self.coverage_reporter.check_coverage_threshold(self.required_coverage)
        
        result = {
            'success': True,
            'enforcement_passed': meets_threshold,
            'coverage': analysis_result['total_coverage'],
            'required': self.required_coverage,
            'analysis_result': analysis_result
        }
        
        if meets_threshold:
            self.logger.info(f"[OK] Coverage requirement met: {analysis_result['total_coverage']:.2f}% >= {self.required_coverage}%")
        else:
            self.logger.error(f"[FAIL] Coverage requirement not met: {analysis_result['total_coverage']:.2f}% < {self.required_coverage}%")
        
        return result
    
    def get_coverage_gaps(self) -> List[Dict[str, Any]]:
        """
        Identify files with low coverage that need attention.
        
        Returns:
            List of files with coverage below threshold (default 50%)
        """
        report = self.coverage_reporter.get_coverage_report()
        
        if not report['success']:
            return []
        
        gaps = []
        threshold = 50.0  # Files below 50% coverage need attention
        
        for file_info in report['details']['files']:
            if file_info['coverage_percent'] < threshold:
                gaps.append({
                    'filename': file_info['filename'],
                    'coverage_percent': file_info['coverage_percent'],
                    'total_lines': file_info['total_lines'],
                    'covered_lines': file_info['covered_lines'],
                    'missing_lines': file_info['missing_lines']
                })
        
        # Sort by coverage percentage (lowest first)
        gaps.sort(key=lambda x: x['coverage_percent'])
        
        return gaps


def setup_coverage_config() -> Dict[str, Any]:
    """
    Create coverage configuration file.
    
    Returns:
        Configuration dictionary
    """
    config = {
        'run': {
            'source': ['.'],
            'omit': [
                '*/tests/*',
                '*/test_*.py',
                '*/conftest.py',
                '*/venv/*',
                '*/env/*',
                '*/__pycache__/*',
                '*/migrations/*',
                '*/setup.py',
                '*/docs/*'
            ],
            'data_file': '.coverage',
            'plugins': []
        },
        'report': {
            'exclude_lines': [
                'pragma: no cover',
                'def __repr__',
                'raise AssertionError',
                'raise NotImplementedError',
                'if __name__ == .__main__.:'
            ],
            'show_missing': True,
            'skip_covered': False
        },
        'html': {
            'directory': 'htmlcov',
            'title': 'Sovereign Decomposition System - Coverage Report'
        },
        'xml': {
            'output': 'coverage.xml'
        }
    }
    
    return config


def create_coverage_config_file():
    """Create .coveragerc configuration file."""
    config = setup_coverage_config()
    
    config_content = "[run]\n"
    config_content += f"source = {config['run']['source'][0]}\n"
    config_content += f"omit = {','.join(config['run']['omit'])}\n"
    config_content += f"data_file = {config['run']['data_file']}\n\n"
    
    config_content += "[report]\n"
    config_content += f"exclude_lines = {','.join(config['report']['exclude_lines'])}\n"
    config_content += f"show_missing = {config['report']['show_missing']}\n"
    config_content += f"skip_covered = {config['report']['skip_covered']}\n\n"
    
    config_content += "[html]\n"
    config_content += f"directory = {config['html']['directory']}\n"
    config_content += f"title = {config['html']['title']}\n\n"
    
    config_content += "[xml]\n"
    config_content += f"output = {config['xml']['output']}\n"
    
    with open('.coveragerc', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("Created .coveragerc configuration file")


def run_coverage_report():
    """Run coverage analysis and generate reports."""
    # Create coverage reporter
    reporter = CoverageReporter(source_dir=".", report_dir="coverage_reports")
    
    # Run analysis
    result = reporter.run_coverage_analysis(
        test_command="python -m pytest test_suite.py -v",
        source_paths=["."]
    )
    
    if result['success']:
        print(f"[OK] Coverage analysis completed: {result['total_coverage']:.2f}%")
        
        # Check threshold
        meets_threshold = reporter.check_coverage_threshold(80.0)
        print(f"[OK] Meets 80% threshold: {meets_threshold}")
        
        # Generate badge
        badge_success = reporter.generate_coverage_badge("docs/coverage_badge.svg")
        print(f"[OK] Badge generated: {badge_success}")
        
        # Create enforcer and get gaps
        enforcer = CoverageEnforcer(reporter)
        gaps = enforcer.get_coverage_gaps()
        
        if gaps:
            print(f"\n[WARN]  Files with low coverage (<50%):")
            for gap in gaps[:5]:  # Show top 5 gaps
                print(f"  - {gap['filename']}: {gap['coverage_percent']:.1f}% "
                      f"({gap['covered_lines']}/{gap['total_lines']} lines)")
        else:
            print("\n[OK] All files meet minimum coverage requirements!")
    else:
        print(f"[FAIL] Coverage analysis failed: {result.get('error', 'Unknown error')}")
    
    return result


def integration_test():
    """Run integration test for coverage functionality."""
    print("Running coverage tracking integration test...")
    
    # Set up coverage
    create_coverage_config_file()
    
    # Test basic functionality
    reporter = CoverageReporter()
    enforcer = CoverageEnforcer(reporter, required_coverage=50.0)  # Lower threshold for testing
    
    # Create a simple test file to ensure we have some code to measure
    test_code = '''
def add_numbers(a, b):
    """Add two numbers together."""
    return a + b

def multiply_numbers(a, b):
    """Multiply two numbers together."""
    return a * b

def divide_numbers(a, b):
    """Divide two numbers, with error handling."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
    
    with open('test_module.py', 'w') as f:
        f.write(test_code)
    
    # Create a simple test for our test module
    test_content = '''
from test_module import add_numbers, multiply_numbers, divide_numbers

def test_addition():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0

def test_multiplication():
    assert multiply_numbers(3, 4) == 12
    assert multiply_numbers(-2, 3) == -6

# Skip the division test to create a coverage gap
# def test_division():
#     assert divide_numbers(10, 2) == 5
#     assert divide_numbers(7, 1) == 7
'''
    
    with open('test_test_module.py', 'w') as f:
        f.write(test_content)
    
    try:
        # Run coverage analysis
        result = enforcer.enforce_coverage("python -m pytest test_test_module.py -v")
        
        print(f"Coverage enforcement result: {result['enforcement_passed']}")
        print(f"Actual coverage: {result['coverage']:.2f}%")
        print(f"Required: {result['required']}%")
        
        # Get coverage gaps
        gaps = enforcer.get_coverage_gaps()
        print(f"Coverage gaps found: {len(gaps)}")
        
        if gaps:
            print("Low-coverage files:")
            for gap in gaps:
                print(f"  - {gap['filename']}: {gap['coverage_percent']:.1f}%")
        
        # Clean up test files
        os.unlink('test_module.py')
        os.unlink('test_test_module.py')
        
        return result['enforcement_passed']
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        # Clean up test files even if test fails
        for file in ['test_module.py', 'test_test_module.py']:
            if os.path.exists(file):
                os.unlink(file)
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "setup":
            create_coverage_config_file()
        elif sys.argv[1] == "run":
            run_coverage_report()
        elif sys.argv[1] == "test":
            success = integration_test()
            sys.exit(0 if success else 1)
        else:
            print("Usage: python coverage_tracking.py [setup|run|test]")
    else:
        # Default: run coverage report
        run_coverage_report()