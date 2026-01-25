"""
Unit tests for the Load Testing Framework

Run with: pytest test_load_tests.py -v
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.load_testing.kg_load_tests import KnowledgeGraphLoadTest, LoadTestResult
from tests.load_testing.analyze_results import LoadTestAnalyzer


@pytest.fixture
def mock_engine():
    """Create a mock knowledge engine."""
    engine = Mock()

    engine.search = AsyncMock(return_value={"results": []})
    engine.add_knowledge = AsyncMock(return_value={"id": "test_id"})
    engine.get_graph_stats = AsyncMock(return_value={"nodes": 100, "edges": 200})

    return engine


@pytest.fixture
def load_test(mock_engine):
    """Create a load test instance."""
    return KnowledgeGraphLoadTest(mock_engine)


class TestLoadTestResult:
    """Tests for LoadTestResult dataclass."""

    def test_create_result(self):
        """Test creating a load test result."""
        result = LoadTestResult(
            test_name="test_read_heavy",
            metrics={"throughput": 100},
            passed=True
        )

        assert result.test_name == "test_read_heavy"
        assert result.metrics["throughput"] == 100
        assert result.passed is True
        assert result.timestamp is not None
        assert result.errors == []
        assert result.warnings == []

    def test_result_with_errors(self):
        """Test result with errors."""
        result = LoadTestResult(
            test_name="test_spike",
            metrics={},
            passed=False,
            errors=["High error rate", "Slow response"]
        )

        assert result.passed is False
        assert len(result.errors) == 2

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = LoadTestResult(
            test_name="test_write",
            metrics={"operations": 500},
            passed=True
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["test_name"] == "test_write"
        assert result_dict["metrics"]["operations"] == 500
        assert result_dict["passed"] is True
        assert "timestamp" in result_dict


class TestKnowledgeGraphLoadTest:
    """Tests for KnowledgeGraphLoadTest."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_engine):
        """Test load tester initialization."""
        load_test = KnowledgeGraphLoadTest(mock_engine)

        assert load_test.engine is mock_engine
        assert load_test.metrics["success_count"] == 0
        assert load_test.metrics["error_count"] == 0
        assert load_test.test_results == []

    @pytest.mark.asyncio
    async def test_read_heavy_test_basic(self, load_test):
        """Test basic read-heavy workload."""
        with patch('random.random', return_value=0.9):  # Force reads
            with patch('asyncio.sleep'):  # Skip sleep
                result = await load_test.run_read_heavy_test(
                    num_users=5,
                    spawn_rate=5,
                    test_duration=10
                )

        assert result.test_name == "read_heavy"
        assert "throughput_ops_per_sec" in result.metrics
        assert "error_rate" in result.metrics
        assert "concurrent_users" in result.metrics

    @pytest.mark.asyncio
    async def test_write_heavy_test_basic(self, load_test):
        """Test basic write-heavy workload."""
        with patch('random.random', return_value=0.7):  # Force writes
            with patch('asyncio.sleep'):
                result = await load_test.run_write_heavy_test(
                    num_users=3,
                    spawn_rate=3,
                    test_duration=10
                )

        assert result.test_name == "write_heavy"
        assert "throughput_ops_per_sec" in result.metrics
        assert result.metrics["concurrent_users"] == 3

    @pytest.mark.asyncio
    async def test_spike_test_basic(self, load_test):
        """Test basic spike test."""
        with patch('asyncio.sleep'):
            result = await load_test.run_spike_test(
                base_users=2,
                spike_users=5,
                spike_duration=10
            )

        assert result.test_name == "spike_test"
        assert "baseline_response_time" in result.metrics
        assert "spike_response_time" in result.metrics

    @pytest.mark.asyncio
    async def test_endurance_test_basic(self, load_test):
        """Test basic endurance test."""
        with patch('asyncio.sleep'):
            result = await load_test.run_endurance_test(
                num_users=3,
                test_duration=15  # Short duration for testing
            )

        assert result.test_name == "endurance"
        assert "memory_growth_gb" in result.metrics
        assert "performance_degradation" in result.metrics

    @pytest.mark.asyncio
    async def test_save_and_load_results(self, load_test, mock_engine):
        """Test saving and loading results."""
        # Run a test
        with patch('asyncio.sleep'):
            result = await load_test.run_read_heavy_test(
                num_users=2,
                spawn_rate=2,
                test_duration=5
            )

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            load_test.save_results(temp_path)

            # Verify file exists and is valid JSON
            assert Path(temp_path).exists()

            with open(temp_path, 'r') as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "tests" in data
            assert len(data["tests"]) > 0
            assert data["tests"][0]["test_name"] == "read_heavy"

        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)

    def test_get_summary(self, load_test):
        """Test getting test summary."""
        # Add some mock results
        load_test.test_results = [
            LoadTestResult("test1", {}, passed=True),
            LoadTestResult("test2", {}, passed=True),
            LoadTestResult("test3", {}, passed=False)
        ]

        summary = load_test.get_summary()

        assert summary["total_tests"] == 3
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 2/3

    @pytest.mark.asyncio
    async def test_config_validation(self, load_test):
        """Test configuration validation."""
        config = {
            "target_throughput": 1000,  # Very high target
            "max_error_rate": 0.001     # Very strict error threshold
        }

        with patch('asyncio.sleep'):
            result = await load_test.run_read_heavy_test(
                num_users=2,
                spawn_rate=2,
                test_duration=5,
                config=config
            )

        # Should fail due to high targets
        assert result.passed is False
        assert len(result.errors) > 0


class TestLoadTestAnalyzer:
    """Tests for LoadTestAnalyzer."""

    @pytest.fixture
    def sample_results_file(self):
        """Create a sample results file."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "tests": [
                {
                    "test_name": "read_heavy",
                    "passed": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": {
                        "throughput_ops_per_sec": 150.5,
                        "error_rate": 0.005,
                        "concurrent_users": 100,
                        "duration_seconds": 60
                    },
                    "errors": [],
                    "warnings": []
                },
                {
                    "test_name": "spike_test",
                    "passed": False,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": {
                        "baseline_response_time": 0.1,
                        "spike_response_time": 0.3,
                        "response_time_degradation": 2.0,
                        "spike_users": 100
                    },
                    "errors": ["Response time degradation exceeded threshold"],
                    "warnings": []
                }
            ]
        }

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(results, f)
            temp_path = f.name

        yield temp_path

        # Clean up
        Path(temp_path).unlink(missing_ok=True)

    def test_initialization(self, sample_results_file):
        """Test analyzer initialization."""
        analyzer = LoadTestAnalyzer(sample_results_file)

        assert analyzer.results_path.exists()
        assert len(analyzer.tests) == 2

    def test_analyze_throughput(self, sample_results_file):
        """Test throughput analysis."""
        analyzer = LoadTestAnalyzer(sample_results_file)
        throughput = analyzer.analyze_throughput()

        assert "average_throughput" in throughput
        assert throughput["average_throughput"] == 150.5
        assert throughput["max_throughput"] == 150.5
        assert throughput["total_throughput_tests"] == 1

    def test_analyze_error_rates(self, sample_results_file):
        """Test error rate analysis."""
        analyzer = LoadTestAnalyzer(sample_results_file)
        errors = analyzer.analyze_error_rates()

        assert "average_error_rate" in errors
        assert errors["average_error_rate"] == 0.005
        assert errors["total_errors"] == 0

    def test_analyze_response_times(self, sample_results_file):
        """Test response time analysis."""
        analyzer = LoadTestAnalyzer(sample_results_file)
        response_times = analyzer.analyze_response_times()

        assert "average_baseline_response" in response_times
        assert response_times["average_baseline_response"] == 0.1
        assert response_times["average_spike_response"] == 0.3
        assert response_times["average_degradation"] == 2.0

    def test_identify_bottlenecks(self, sample_results_file):
        """Test bottleneck identification."""
        analyzer = LoadTestAnalyzer(sample_results_file)
        bottlenecks = analyzer.identify_bottlenecks()

        assert len(bottlenecks) > 0

        # Check for spike test bottleneck
        spike_bottlenecks = [b for b in bottlenecks if b["test"] == "spike_test"]
        assert len(spike_bottlenecks) > 0
        assert spike_bottlenecks[0]["severity"] == "HIGH"

    def test_estimate_capacity(self, sample_results_file):
        """Test capacity estimation."""
        analyzer = LoadTestAnalyzer(sample_results_file)
        capacity = analyzer.estimate_capacity(target_response_time=1.0)

        assert "estimated_max_concurrent_users" in capacity
        assert "estimated_max_requests_per_second" in capacity
        assert "scaling_recommendation" in capacity
        assert capacity["scaling_recommendation"] in ["SINGLE_INSTANCE", "HORIZONTAL"]

    def test_generate_recommendations(self, sample_results_file):
        """Test recommendation generation."""
        analyzer = LoadTestAnalyzer(sample_results_file)
        recommendations = analyzer.generate_recommendations()

        assert len(recommendations) > 0

        # All recommendations should have required fields
        for rec in recommendations:
            assert "priority" in rec
            assert "category" in rec
            assert "recommendation" in rec
            assert "action" in rec

    def test_generate_report(self, sample_results_file):
        """Test report generation."""
        analyzer = LoadTestAnalyzer(sample_results_file)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            report_path = f.name

        try:
            analyzer.generate_report(report_path)

            # Verify report exists
            assert Path(report_path).exists()

            # Check report content
            with open(report_path, 'r') as f:
                content = f.read()

            assert "LOAD TEST REPORT" in content
            assert "read_heavy" in content
            assert "spike_test" in content

        finally:
            Path(report_path).unlink(missing_ok=True)

    def test_missing_file(self):
        """Test handling of missing results file."""
        with pytest.raises(FileNotFoundError):
            LoadTestAnalyzer("nonexistent_file.json")


class TestIntegration:
    """Integration tests for the complete workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_engine):
        """Test complete workflow from test to analysis."""
        # Run tests
        load_test = KnowledgeGraphLoadTest(mock_engine)

        with patch('asyncio.sleep'):
            await load_test.run_read_heavy_test(2, 2, 5)
            await load_test.run_spike_test(2, 5, 5)

        # Save results
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            results_path = f.name

        try:
            load_test.save_results(results_path)

            # Analyze results
            analyzer = LoadTestAnalyzer(results_path)

            # Verify analysis
            summary = load_test.get_summary()
            assert summary["total_tests"] == 2

            # Generate report
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                report_path = f.name

            try:
                analyzer.generate_report(report_path)
                assert Path(report_path).exists()

            finally:
                Path(report_path).unlink(missing_ok=True)

        finally:
            Path(results_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
