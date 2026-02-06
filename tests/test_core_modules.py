"""
Test Suite for OpenEvolve Core Modules

Tests for:
- workflow_engine.py
- config.py
- parameter_manager.py
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import tempfile
import os


class TestConfigModule(unittest.TestCase):
    """Test configuration module functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_config_dataclass_creation(self):
        """Test Config dataclass can be created."""
        try:
            from config import Config
            config = Config()
            self.assertIsNotNone(config)
        except ImportError as e:
            self.skipTest(f"Config module not available: {e}")
    
    def test_config_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        config_data = {
            'server': {
                'host': '0.0.0.0',
                'port': 8000
            },
            'database': {
                'path': 'test.db'
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        try:
            from config_loader import load_config
            config = load_config(self.config_file)
            self.assertIsNotNone(config)
        except ImportError:
            self.skipTest("config_loader module not available")
    
    def test_config_validation(self):
        """Test configuration validation."""
        try:
            from config import ConfigValidator
            validator = ConfigValidator()
            self.assertIsNotNone(validator)
        except ImportError:
            self.skipTest("ConfigValidator not available")


class TestParameterManager(unittest.TestCase):
    """Test parameter management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_parameter_manager_creation(self):
        """Test ParameterManager can be created."""
        try:
            from parameter_manager import ParameterManager
            manager = ParameterManager()
            self.assertIsNotNone(manager)
        except ImportError:
            self.skipTest("parameter_manager module not available")
    
    def test_parameter_get_set(self):
        """Test parameter get and set operations."""
        try:
            from parameter_manager import ParameterManager
            manager = ParameterManager()
            
            # Test setting and getting a parameter
            manager.set('test.param', 'value')
            result = manager.get('test.param')
            self.assertEqual(result, 'value')
        except ImportError:
            self.skipTest("parameter_manager module not available")
    
    def test_parameter_defaults(self):
        """Test default parameter values."""
        try:
            from parameter_manager import ParameterManager
            manager = ParameterManager()
            
            # Get a non-existent parameter with default
            result = manager.get('nonexistent', default='default_value')
            self.assertEqual(result, 'default_value')
        except ImportError:
            self.skipTest("parameter_manager module not available")
    
    def test_parameter_export(self):
        """Test parameter export functionality."""
        try:
            from parameter_manager import ParameterManager
            manager = ParameterManager()
            manager.set('export.test', 'value')
            
            export_path = os.path.join(self.temp_dir, 'params.json')
            manager.export_to_json(export_path)
            
            self.assertTrue(os.path.exists(export_path))
        except ImportError:
            self.skipTest("parameter_manager module not available")


class TestWorkflowEngine(unittest.TestCase):
    """Test workflow engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_workflow_engine_creation(self):
        """Test WorkflowEngine can be created."""
        try:
            from workflow_engine import WorkflowEngine
            engine = WorkflowEngine()
            self.assertIsNotNone(engine)
        except ImportError:
            self.skipTest("workflow_engine module not available")
    
    def test_workflow_definition(self):
        """Test workflow definition creation."""
        try:
            from workflow_engine import WorkflowDefinition, WorkflowStage
            
            definition = WorkflowDefinition(
                name='test_workflow',
                stages=[
                    WorkflowStage(name='stage1', type='decomposition'),
                    WorkflowStage(name='stage2', type='evolution')
                ]
            )
            
            self.assertEqual(definition.name, 'test_workflow')
            self.assertEqual(len(definition.stages), 2)
        except ImportError:
            self.skipTest("WorkflowDefinition not available")
    
    def test_workflow_execution(self):
        """Test workflow execution."""
        try:
            from workflow_engine import WorkflowEngine
            
            engine = WorkflowEngine()
            result = engine.execute_workflow('test_workflow')
            self.assertIsNotNone(result)
        except ImportError:
            self.skipTest("workflow_engine module not available")
    
    def test_workflow_state_persistence(self):
        """Test workflow state persistence."""
        try:
            from workflow_engine import WorkflowState, WorkflowStateSerializer
            
            state = WorkflowState(
                workflow_id='test_id',
                current_stage='stage1',
                completed_stages=['stage0']
            )
            
            state_path = os.path.join(self.temp_dir, 'state.json')
            WorkflowStateSerializer.save(state, state_path)
            loaded_state = WorkflowStateSerializer.load(state_path)
            
            self.assertEqual(state.workflow_id, loaded_state.workflow_id)
        except ImportError:
            self.skipTest("WorkflowStateSerializer not available")


class TestAlertingSystem(unittest.TestCase):
    """Test alerting system functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_alert_creation(self):
        """Test alert creation."""
        try:
            from alerting_system import Alert, AlertSeverity, AlertType
            
            alert = Alert(
                alert_id='alert_001',
                severity=AlertSeverity.HIGH,
                type=AlertType.SYSTEM,
                message='Test alert message',
                timestamp=datetime.now()
            )
            
            self.assertEqual(alert.alert_id, 'alert_001')
            self.assertEqual(alert.severity, AlertSeverity.HIGH)
        except ImportError:
            self.skipTest("alerting_system module not available")
    
    def test_alert_manager(self):
        """Test alert manager functionality."""
        try:
            from alerting_system import AlertManager
            
            manager = AlertManager()
            self.assertIsNotNone(manager)
        except ImportError:
            self.skipTest("AlertManager not available")
    
    def test_alert_dispatch(self):
        """Test alert dispatch."""
        try:
            from alerting_system import AlertDispatcher
            
            dispatcher = AlertDispatcher()
            result = dispatcher.dispatch_alert(
                severity='HIGH',
                message='Test message',
                channel='email'
            )
            self.assertIsNotNone(result)
        except ImportError:
            self.skipTest("AlertDispatcher not available")
    
    def test_alert_persistence(self):
        """Test alert persistence."""
        try:
            from alerting_system import AlertStore
            
            store = AlertStore(db_path=os.path.join(self.temp_dir, 'alerts.db'))
            store.save_alert({
                'alert_id': 'test_alert',
                'message': 'Test message'
            })
            
            alerts = store.get_all_alerts()
            self.assertEqual(len(alerts), 1)
        except ImportError:
            self.skipTest("AlertStore not available")


class TestAnalyticsManager(unittest.TestCase):
    """Test analytics manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_analytics_manager_creation(self):
        """Test AnalyticsManager can be created."""
        try:
            from analytics_manager import AnalyticsManager
            manager = AnalyticsManager()
            self.assertIsNotNone(manager)
        except ImportError:
            self.skipTest("analytics_manager module not available")
    
    def test_track_event(self):
        """Test event tracking."""
        try:
            from analytics_manager import EventTracker
            
            tracker = EventTracker()
            tracker.track_event(
                event_type='test_event',
                properties={'key': 'value'}
            )
            
            events = tracker.get_events()
            self.assertGreaterEqual(len(events), 1)
        except ImportError:
            self.skipTest("EventTracker not available")
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        try:
            from analytics_manager import MetricsCollector
            
            collector = MetricsCollector()
            collector.record_metric('cpu_usage', 75.5)
            collector.record_metric('memory_usage', 1024)
            
            metrics = collector.get_metrics()
            self.assertIn('cpu_usage', metrics)
        except ImportError:
            self.skipTest("MetricsCollector not available")
    
    def test_report_generation(self):
        """Test analytics report generation."""
        try:
            from analytics_manager import AnalyticsReporter
            
            reporter = AnalyticsReporter()
            report = reporter.generate_report(
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
            
            self.assertIsNotNone(report)
        except ImportError:
            self.skipTest("AnalyticsReporter not available")


class TestBackupRestore(unittest.TestCase):
    """Test backup and restore functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backup_dir = os.path.join(self.temp_dir, 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_backup_creation(self):
        """Test backup creation."""
        try:
            from backup_restore import BackupManager
            
            manager = BackupManager(backup_dir=self.backup_dir)
            backup_id = manager.create_backup(
                data={'test': 'data'},
                backup_type='full'
            )
            
            self.assertIsNotNone(backup_id)
            self.assertTrue(os.path.exists(
                os.path.join(self.backup_dir, f'{backup_id}.backup')
            ))
        except ImportError:
            self.skipTest("backup_restore module not available")
    
    def test_backup_restore(self):
        """Test backup restoration."""
        try:
            from backup_restore import BackupManager
            
            manager = BackupManager(backup_dir=self.backup_dir)
            
            # Create backup
            backup_id = manager.create_backup(
                data={'key': 'original_value'},
                backup_type='full'
            )
            
            # Restore backup
            restored_data = manager.restore_backup(backup_id)
            self.assertEqual(restored_data['key'], 'original_value')
        except ImportError:
            self.skipTest("backup_restore module not available")
    
    def test_backup_list(self):
        """Test backup listing."""
        try:
            from backup_restore import BackupManager
            
            manager = BackupManager(backup_dir=self.backup_dir)
            
            # Create multiple backups
            manager.create_backup(data={'id': 1}, backup_type='full')
            manager.create_backup(data={'id': 2}, backup_type='full')
            
            backups = manager.list_backups()
            self.assertEqual(len(backups), 2)
        except ImportError:
            self.skipTest("backup_restore module not available")


class TestBatchOperations(unittest.TestCase):
    """Test batch operations functionality."""
    
    def test_batch_processor(self):
        """Test batch processing."""
        try:
            from batch_operations import BatchProcessor
            
            processor = BatchProcessor(batch_size=10)
            results = []
            
            def process_item(item):
                return item * 2
            
            items = [1, 2, 3, 4, 5]
            results = processor.process_batch(items, process_item)
            
            self.assertEqual(results, [2, 4, 6, 8, 10])
        except ImportError:
            self.skipTest("batch_operations module not available")
    
    def test_batch_parallel(self):
        """Test parallel batch processing."""
        try:
            from batch_operations import ParallelBatchProcessor
            
            processor = ParallelBatchProcessor(
                max_workers=4,
                batch_size=5
            )
            
            items = range(20)
            results = list(processor.process(items))
            
            self.assertEqual(len(results), 20)
        except ImportError:
            self.skipTest("ParallelBatchProcessor not available")
    
    def test_batch_error_handling(self):
        """Test batch error handling."""
        try:
            from batch_operations import BatchErrorHandler
            
            handler = BatchErrorHandler()
            
            errors = [
                Exception("Error 1"),
                Exception("Error 2")
            ]
            
            handled = handler.handle_errors(errors)
            self.assertEqual(len(handled), 2)
        except ImportError:
            self.skipTest("BatchErrorHandler not available")


if __name__ == '__main__':
    unittest.main()
