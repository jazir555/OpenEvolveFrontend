"""
Comprehensive Unit Tests for Analytics Manager

Tests the analytics management system including:
- Metrics collection
- Performance tracking
- Data aggregation
- Report generation
- Dashboard integration

Author: OpenEvolve QA Team
Date: 2026-02-05
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAnalyticsManager:
    """Test AnalyticsManager functionality"""

    @pytest.fixture
    def analytics_manager(self, tmp_path):
        """Create analytics manager for testing"""
        from analytics_manager import AnalyticsManager
        return AnalyticsManager(
            storage_path=str(tmp_path / "analytics"),
            retention_days=30
        )

    def test_analytics_manager_creation(self, analytics_manager):
        """Test AnalyticsManager initialization"""
        from analytics_manager import AnalyticsManager
        
        manager = AnalyticsManager(
            storage_path="/tmp/analytics",
            retention_days=7
        )
        assert manager.storage_path == "/tmp/analytics"
        assert manager.retention_days == 7

    def test_record_metric(self, analytics_manager):
        """Test recording a metric"""
        analytics_manager.record_metric(
            name="cpu_usage",
            value=75.5,
            tags={"host": "server1"}
        )
        
        assert analytics_manager.get_metric("cpu_usage") is not None

    def test_get_metric(self, analytics_manager):
        """Test retrieving a metric"""
        analytics_manager.record_metric("test_metric", 100)
        
        metric = analytics_manager.get_metric("test_metric")
        
        assert metric is not None
        assert metric.value == 100

    def test_list_metrics(self, analytics_manager):
        """Test listing all metrics"""
        analytics_manager.record_metric("metric1", 10)
        analytics_manager.record_metric("metric2", 20)
        
        metrics = analytics_manager.list_metrics()
        
        assert len(metrics) >= 2


class TestMetricsCollection:
    """Test metrics collection"""

    def test_counter_metric(self):
        """Test counter metric"""
        from analytics_manager import Counter
        
        counter = Counter("requests")
        counter.increment()
        counter.increment()
        
        assert counter.value == 2

    def test_gauge_metric(self):
        """Test gauge metric"""
        from analytics_manager import Gauge
        
        gauge = Gauge("temperature")
        gauge.set(25.5)
        gauge.set(26.0)
        
        assert gauge.value == 26.0

    def test_histogram_metric(self):
        """Test histogram metric"""
        from analytics_manager import Histogram
        
        hist = Histogram("response_time", buckets=[10, 50, 100])
        hist.observe(25)
        hist.observe(75)
        hist.observe(150)
        
        assert hist.count == 3


class TestPerformanceTracking:
    """Test performance tracking"""

    def test_timer(self):
        """Test performance timer"""
        from analytics_manager import Timer
        
        timer = Timer("operation_duration")
        timer.start()
        # Simulate work
        import time
        time.sleep(0.01)
        duration = timer.stop()
        
        assert duration >= 0.01

    def test_track_execution(self):
        """Test execution tracking decorator"""
        from analytics_manager import track_execution
        
        @track_execution("test_function")
        def test_func():
            return "result"
        
        result = test_func()
        
        assert result == "result"


class TestDataAggregation:
    """Test data aggregation"""

    def test_aggregate_values(self):
        """Test aggregating values"""
        from analytics_manager import Aggregator
        
        aggregator = Aggregator()
        
        values = [10, 20, 30, 40, 50]
        
        stats = aggregator.aggregate(values)
        
        assert stats["mean"] == 30
        assert stats["sum"] == 150

    def test_time_based_aggregation(self):
        """Test time-based aggregation"""
        from analytics_manager import TimeAggregator
        
        aggregator = TimeAggregator(
            interval_seconds=60,
            aggregation_method="avg"
        )
        
        # Add data points
        aggregator.add(datetime.now(), 10)
        aggregator.add(datetime.now(), 20)
        
        result = aggregator.get_aggregated()
        
        assert result is not None


class TestReportGeneration:
    """Test report generation"""

    def test_generate_summary_report(self):
        """Test summary report generation"""
        from analytics_manager import ReportGenerator
        
        generator = ReportGenerator()
        
        report = generator.generate_summary(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        assert report is not None

    def test_generate_performance_report(self):
        """Test performance report"""
        from analytics_manager import ReportGenerator
        
        generator = ReportGenerator()
        
        report = generator.generate_performance_report(
            metrics=["cpu", "memory"],
            period="24h"
        )
        
        assert report is not None

    def test_export_report(self):
        """Test report export"""
        from analytics_manager import ReportGenerator, ReportFormat
        
        generator = ReportGenerator()
        
        report_data = {
            "title": "Test Report",
            "metrics": {"cpu": 50}
        }
        
        exported = generator.export(report_data, format=ReportFormat.JSON)
        
        assert exported is not None


class TestDashboardIntegration:
    """Test dashboard integration"""

    def test_dashboard_config(self):
        """Test dashboard configuration"""
        from analytics_manager import DashboardConfig
        
        config = DashboardConfig(
            refresh_interval=30,
            charts=["cpu", "memory", "network"]
        )
        
        assert config.refresh_interval == 30

    def test_get_dashboard_data(self):
        """Test dashboard data retrieval"""
        from analytics_manager import DashboardDataProvider
        
        provider = DashboardDataProvider()
        
        data = provider.get_dashboard_data()
        
        assert isinstance(data, dict)


class TestAnalyticsStorage:
    """Test analytics data storage"""

    def test_save_to_file(self, tmp_path):
        """Test saving analytics to file"""
        from analytics_manager import AnalyticsStorage
        
        storage = AnalyticsStorage(str(tmp_path))
        
        storage.save("test_metric", {"value": 100})
        
        assert (tmp_path / "test_metric.json").exists()

    def test_load_from_file(self, tmp_path):
        """Test loading analytics from file"""
        from analytics_manager import AnalyticsStorage
        
        storage = AnalyticsStorage(str(tmp_path))
        
        # Save first
        storage.save("load_test", {"data": "test"})
        
        # Load
        loaded = storage.load("load_test")
        
        assert loaded["data"] == "test"


class TestAnalyticsConfig:
    """Test analytics configuration"""

    def test_config_creation(self):
        """Test AnalyticsConfig"""
        from analytics_manager import AnalyticsConfig
        
        config = AnalyticsConfig(
            enabled=True,
            sample_rate=1.0,
            buffer_size=1000
        )
        
        assert config.enabled == True
        assert config.sample_rate == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
