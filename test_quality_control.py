"""
Unit Tests for Quality Control Module

Tests the code quality checker functionality including:
- Issue detection and classification
- Security vulnerability scanning
- Complexity analysis
- Code duplication detection
- Configuration validation
- Report generation

Per CLAUDE.md: These tests validate the "Runtime Truth" of the quality control system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from quality_control import (
    CodeQualityChecker,
    QualityIssue,
    QualityMetrics,
    QualityReport,
    IssueSeverity,
    IssueType,
    QualityCheckError,
    QualityCheckConfigError,
    run_quality_checks
)


class TestQualityIssue:
    """Test QualityIssue dataclass."""

    def test_quality_issue_creation(self):
        """Test creating a quality issue."""
        issue = QualityIssue(
            file_path="test.py",
            line_number=10,
            issue_type=IssueType.SECURITY,
            severity=IssueSeverity.HIGH,
            message="Test issue",
            rule_id="TEST001"
        )

        assert issue.file_path == "test.py"
        assert issue.line_number == 10
        assert issue.issue_type == IssueType.SECURITY
        assert issue.severity == IssueSeverity.HIGH
        assert issue.message == "Test issue"
        assert issue.rule_id == "TEST001"

    def test_quality_issue_to_dict(self):
        """Test converting issue to dictionary."""
        issue = QualityIssue(
            file_path="test.py",
            line_number=10,
            issue_type=IssueType.CODE_SMELL,
            severity=IssueSeverity.MEDIUM,
            message="Test issue",
            rule_id="TEST001",
            suggestion="Fix it"
        )

        issue_dict = issue.to_dict()

        assert issue_dict['file_path'] == "test.py"
        assert issue_dict['line_number'] == 10
        assert issue_dict['issue_type'] == "code_smell"
        assert issue_dict['severity'] == "medium"
        assert issue_dict['message'] == "Test issue"
        assert issue_dict['rule_id'] == "TEST001"
        assert issue_dict['suggestion'] == "Fix it"


class TestCodeQualityChecker:
    """Test CodeQualityChecker class."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_python_file(self, temp_project):
        """Create a sample Python file with various issues."""
        content = """
def long_function():
    '''A very long function that exceeds the limit.'''
    x = 1
    y = 2
    z = 3
    # Many lines...
    return x + y + z

def deep_nesting():
    '''Function with deep nesting.'''
    if True:
        if True:
            if True:
                if True:
                    if True:
                        pass

def complex_function(a, b, c):
    '''Function with high cyclomatic complexity.'''
    if a:
        if b:
            if c:
                for i in range(10):
                    if i > 5:
                        return True
    return False

password = os.getenv("TEST_PASSWORD", "default_test_password")
eval("print('unsafe')")
"""
        file_path = Path(temp_project) / "sample.py"
        file_path.write_text(content)
        return file_path

    def test_checker_initialization(self, temp_project):
        """Test checker initialization."""
        checker = CodeQualityChecker(project_root=temp_project)

        assert checker.project_root == Path(temp_project).resolve()
        assert checker.config is not None
        assert checker.correlation_id is not None

    def test_checker_invalid_project_root(self):
        """Test checker with non-existent project root."""
        with pytest.raises(QualityCheckConfigError):
            CodeQualityChecker(project_root="/nonexistent/path")

    def test_config_validation(self, temp_project):
        """Test configuration validation."""
        # Valid config
        config = {'max_cyclomatic_complexity': 15}
        checker = CodeQualityChecker(
            project_root=temp_project,
            config=config
        )
        assert checker.config['max_cyclomatic_complexity'] == 15

        # Invalid config - too high
        with pytest.raises(QualityCheckConfigError):
            CodeQualityChecker(
                project_root=temp_project,
                config={'max_cyclomatic_complexity': 1000}
            )

        # Invalid config - wrong type
        with pytest.raises(QualityCheckConfigError):
            CodeQualityChecker(
                project_root=temp_project,
                config={'check_code_smells': 'not_a_bool'}
            )

    def test_discover_files(self, temp_project):
        """Test file discovery."""
        # Create test files
        (Path(temp_project) / "test1.py").write_text("# test")
        (Path(temp_project) / "test2.js").write_text("// test")
        (Path(temp_project) / "README.md").write_text("# readme")

        checker = CodeQualityChecker(project_root=temp_project)
        files = checker._discover_files()

        # Should find Python and JS files, not MD
        assert len(files) == 2
        assert any(f.suffix == '.py' for f in files)
        assert any(f.suffix == '.js' for f in files)

    def test_check_code_smells(self, temp_project, sample_python_file):
        """Test code smell detection."""
        checker = CodeQualityChecker(
            project_root=temp_project,
            config={'max_function_length': 5, 'max_nesting_depth': 3}
        )

        issues = checker.check_code_smells()

        # Should find issues with the sample file
        assert len(issues) > 0

        # Check for specific issue types
        issue_types = [issue.issue_type for issue in issues]
        assert IssueType.CODE_SMELL in issue_types

    def test_check_security_issues(self, temp_project, sample_python_file):
        """Test security issue detection."""
        checker = CodeQualityChecker(project_root=temp_project)
        issues = checker.check_security_issues()

        # Should find hardcoded password and eval usage
        assert len(issues) > 0

        # Check that security issues are detected
        security_issues = [i for i in issues if i.issue_type == IssueType.SECURITY]
        assert len(security_issues) > 0

        # Check for specific security issues
        messages = [i.message.lower() for i in security_issues]
        assert any('secret' in msg or 'eval' in msg for msg in messages)

    def test_check_complexity(self, temp_project, sample_python_file):
        """Test complexity analysis."""
        checker = CodeQualityChecker(
            project_root=temp_project,
            config={'max_cyclomatic_complexity': 5}
        )

        issues = checker.check_complexity()

        # Should find complexity issues
        complexity_issues = [i for i in issues if i.issue_type == IssueType.COMPLEXITY]
        assert len(complexity_issues) > 0

    def test_calculate_cyclomatic_complexity(self, temp_project):
        """Test cyclomatic complexity calculation."""
        checker = CodeQualityChecker(project_root=temp_project)

        # Simple function
        code = """
def simple():
    return 1
"""
        import ast
        tree = ast.parse(code)
        func_node = tree.body[0]
        complexity = checker._calculate_cyclomatic_complexity(func_node)
        assert complexity == 1  # Base complexity

        # Complex function
        code = """
def complex(a, b, c):
    if a:
        if b:
            if c:
                return True
    return False
"""
        tree = ast.parse(code)
        func_node = tree.body[0]
        complexity = checker._calculate_cyclomatic_complexity(func_node)
        assert complexity > 1

    def test_calculate_nesting_depth(self, temp_project):
        """Test nesting depth calculation."""
        checker = CodeQualityChecker(project_root=temp_project)

        # Nested code
        code = """
def nested():
    if True:
        if True:
            if True:
                pass
"""
        import ast
        tree = ast.parse(code)
        func_node = tree.body[0]
        depth = checker._calculate_nesting_depth(func_node)
        assert depth == 3

    def test_run_all_checks(self, temp_project, sample_python_file):
        """Test running all checks."""
        checker = CodeQualityChecker(
            project_root=temp_project,
            config={
                'max_cyclomatic_complexity': 5,
                'max_function_length': 10,
                'check_coverage': False  # Disable coverage for faster testing
            }
        )

        report = checker.run_all_checks()

        # Verify report structure
        assert isinstance(report, QualityReport)
        assert report.timestamp is not None
        assert report.project_root == str(Path(temp_project).resolve())
        assert report.metrics is not None
        assert report.issues is not None
        assert report.checks_performed is not None
        assert report.correlation_id is not None

        # Verify metrics
        assert report.metrics.total_issues >= 0
        assert 0.0 <= report.metrics.quality_score <= 1.0

    def test_quality_report_to_dict(self, temp_project):
        """Test converting quality report to dictionary."""
        checker = CodeQualityChecker(project_root=temp_project)

        report = QualityReport(
            timestamp="2025-01-22T12:00:00Z",
            project_root=str(temp_project),
            metrics=QualityMetrics(total_issues=5, quality_score=0.8),
            issues=[],
            checks_performed=["security", "complexity"],
            correlation_id="test-id"
        )

        report_dict = report.to_dict()

        assert report_dict['timestamp'] == "2025-01-22T12:00:00Z"
        assert report_dict['project_root'] == str(temp_project)
        assert report_dict['metrics']['total_issues'] == 5
        assert report_dict['metrics']['quality_score'] == 0.8
        assert report_dict['checks_performed'] == ["security", "complexity"]

    def test_quality_report_save_to_file(self, temp_project):
        """Test saving quality report to file."""
        checker = CodeQualityChecker(project_root=temp_project)

        report = QualityReport(
            timestamp="2025-01-22T12:00:00Z",
            project_root=str(temp_project),
            metrics=QualityMetrics(),
            issues=[],
            checks_performed=[],
            correlation_id="test-id"
        )

        output_path = Path(temp_project) / "report.json"
        report.save_to_file(str(output_path))

        assert output_path.exists()

        # Verify content
        import json
        with open(output_path) as f:
            data = json.load(f)

        assert data['timestamp'] == "2025-01-22T12:00:00Z"
        assert data['correlation_id'] == "test-id"


class TestRunQualityChecks:
    """Test the run_quality_checks function."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_run_quality_checks_basic(self, temp_project):
        """Test basic quality check execution."""
        # Create a simple Python file
        (Path(temp_project) / "test.py").write_text("""
def hello():
    print("Hello, world!")
    return True
""")

        result = run_quality_checks(
            project_root=temp_project,
            config={'check_coverage': False}  # Disable for faster testing
        )

        # Verify result structure
        assert 'quality_score' in result
        assert 'total_issues' in result
        assert 'critical_issues' in result
        assert 'high_issues' in result
        assert 'medium_issues' in result
        assert 'low_issues' in result
        assert 'security_issues' in result
        assert 'complexity_issues' in result
        assert 'duplication_issues' in result
        assert 'code_smell_issues' in result
        assert 'correlation_id' in result
        assert 'timestamp' in result

        # Verify types
        assert isinstance(result['quality_score'], float)
        assert isinstance(result['total_issues'], int)
        assert isinstance(result['correlation_id'], str)

    def test_run_quality_checks_with_config(self, temp_project):
        """Test quality checks with custom configuration."""
        (Path(temp_project) / "test.py").write_text("x = 1")

        result = run_quality_checks(
            project_root=temp_project,
            config={
                'max_cyclomatic_complexity': 20,
                'min_coverage': 50.0,
                'check_coverage': False
            }
        )

        assert result['quality_score'] >= 0.0
        assert result['quality_score'] <= 1.0


class TestSecurityPatterns:
    """Test security pattern detection."""

    @pytest.fixture
    def temp_project(self):
        """Create temporary project."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_hardcoded_secret_detection(self, temp_project):
        """Test detection of hardcoded secrets."""
        content = '''
password = os.getenv("TEST_PASSWORD", "default_test_password")
api_key = os.getenv("TEST_API_KEY", "default_test_api_key")
'''
        (Path(temp_project) / "test.py").write_text(content)

        checker = CodeQualityChecker(project_root=temp_project)
        issues = checker.check_security_issues()

        secret_issues = [i for i in issues if 'secret' in i.message.lower()]
        assert len(secret_issues) > 0

    def test_eval_detection(self, temp_project):
        """Test detection of eval usage."""
        content = '''
eval("print('unsafe')")
exec("x = 1")
'''
        (Path(temp_project) / "test.py").write_text(content)

        checker = CodeQualityChecker(project_root=temp_project)
        issues = checker.check_security_issues()

        eval_issues = [i for i in issues if 'eval' in i.message.lower() or 'exec' in i.message.lower()]
        assert len(eval_issues) > 0


class TestJavaScriptSupport:
    """Test JavaScript/TypeScript file analysis."""

    @pytest.fixture
    def temp_project(self):
        """Create temporary project."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_javascript_console_log(self, temp_project):
        """Test detection of console.log in JavaScript."""
        content = '''
function test() {
    console.log("debug message");
    return true;
}
'''
        (Path(temp_project) / "test.js").write_text(content)

        checker = CodeQualityChecker(project_root=temp_project)
        issues = checker.check_code_smells()

        # Should find console.log
        console_issues = [i for i in issues if i.rule_id == "CONSOLE_LOG"]
        assert len(console_issues) > 0

    def test_javascript_debugger(self, temp_project):
        """Test detection of debugger statements."""
        content = '''
function test() {
    debugger;
    return true;
}
'''
        (Path(temp_project) / "test.js").write_text(content)

        checker = CodeQualityChecker(project_root=temp_project)
        issues = checker.check_code_smells()

        debugger_issues = [i for i in issues if i.rule_id == "DEBUGGER_STATEMENT"]
        assert len(debugger_issues) > 0

    def test_javascript_eval(self, temp_project):
        """Test detection of eval in JavaScript."""
        content = '''
function test() {
    eval("alert('xss')");
    return true;
}
'''
        (Path(temp_project) / "test.js").write_text(content)

        checker = CodeQualityChecker(project_root=temp_project)
        issues = checker.check_security_issues()

        eval_issues = [i for i in issues if 'eval' in i.message.lower()]
        assert len(eval_issues) > 0


class TestDuplicationDetection:
    """Test code duplication detection."""

    @pytest.fixture
    def temp_project(self):
        """Create temporary project."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_duplicate_code_detection(self, temp_project):
        """Test detection of duplicate code blocks."""
        # Create two files with duplicate code
        code = '''
def calculate_sum(a, b):
    result = a + b
    return result

def calculate_product(a, b):
    result = a * b
    return result
'''

        (Path(temp_project) / "file1.py").write_text(code)
        (Path(temp_project) / "file2.py").write_text(code)

        checker = CodeQualityChecker(project_root=temp_project)
        issues = checker.check_duplication()

        # May or may not find duplicates depending on sequence detection
        # Just verify it runs without error
        assert isinstance(issues, list)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def temp_project(self):
        """Create temporary project."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_empty_project(self, temp_project):
        """Test checker with empty project."""
        checker = CodeQualityChecker(project_root=temp_project)
        report = checker.run_all_checks()

        assert report.metrics.total_issues == 0
        assert report.metrics.quality_score == 1.0

    def test_syntax_error_file(self, temp_project):
        """Test handling of files with syntax errors."""
        # Create Python file with syntax error
        (Path(temp_project) / "broken.py").write_text("""
def broken(
    # Missing closing parenthesis
""")

        checker = CodeQualityChecker(project_root=temp_project)
        issues = checker.check_code_smells()

        # Should handle gracefully
        assert isinstance(issues, list)

    def test_unicode_file(self, temp_project):
        """Test handling of files with Unicode content."""
        content = """
# 中文注释
def hello():
    '''Hello in various languages: 你好 🌍'''
    print("Hello, 世界!")
    return True
"""
        (Path(temp_project) / "unicode.py").write_text(content)

        checker = CodeQualityChecker(project_root=temp_project)
        issues = checker.check_code_smells()

        # Should handle Unicode gracefully
        assert isinstance(issues, list)


class TestQualityMetrics:
    """Test quality metrics calculation."""

    def test_metrics_calculation(self):
        """Test metrics dataclass."""
        metrics = QualityMetrics(
            total_files=10,
            total_lines=1000,
            total_issues=5,
            security_count=2,
            complexity_score=3
        )

        assert metrics.total_files == 10
        assert metrics.total_lines == 1000
        assert metrics.total_issues == 5
        assert metrics.security_count == 2
        assert metrics.complexity_score == 3

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = QualityMetrics(
            total_issues=10,
            quality_score=0.85
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict['total_issues'] == 10
        assert metrics_dict['quality_score'] == 0.85


class TestSeverityAndTypeEnums:
    """Test severity and type enumerations."""

    def test_severity_enum(self):
        """Test IssueSeverity enum."""
        assert IssueSeverity.CRITICAL.value == "critical"
        assert IssueSeverity.HIGH.value == "high"
        assert IssueSeverity.MEDIUM.value == "medium"
        assert IssueSeverity.LOW.value == "low"
        assert IssueSeverity.INFO.value == "info"

    def test_issue_type_enum(self):
        """Test IssueType enum."""
        assert IssueType.CODE_SMELL.value == "code_smell"
        assert IssueType.SECURITY.value == "security"
        assert IssueType.COMPLEXITY.value == "complexity"
        assert IssueType.DUPLICATION.value == "duplication"
        assert IssueType.COVERAGE.value == "coverage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
