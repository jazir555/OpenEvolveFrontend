"""
Test Suite for UI and API Components

Tests for:
- BubbleLab UI components
- API endpoints
- WebSocket handling
- Frontend utilities
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import json
import tempfile
import os
from typing import Dict, Any, List
from datetime import datetime, timedelta


class TestBubbleLabUI(unittest.TestCase):
    """Test BubbleLab UI components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_ui_components(self):
        """Test UI component factory."""
        try:
            from bubblelabs_ui_component import UIComponentFactory
            factory = UIComponentFactory()
            self.assertIsNotNone(factory)
        except ImportError:
            self.skipTest("UI components not available")
    
    def test_workflow_visualizer(self):
        """Test workflow visualization component."""
        try:
            from bubblelabs_ui_component import WorkflowVisualizer
            
            visualizer = WorkflowVisualizer()
            config = visualizer.generate_config(
                workflow={'stages': ['decompose', 'solve', 'assemble']}
            )
            
            self.assertIsNotNone(config)
        except ImportError:
            self.skipTest("WorkflowVisualizer not available")
    
    def test_metrics_display(self):
        """Test metrics display component."""
        try:
            from bubblelabs_ui_component import MetricsDisplay
            
            display = MetricsDisplay()
            html = display.render(
                metrics={'accuracy': 0.95, 'time': 120}
            )
            
            self.assertIsNotNone(html)
        except ImportError:
            self.skipTest("MetricsDisplay not available")
    
    def test_code_editor(self):
        """Test code editor component."""
        try:
            from bubblelabs_ui_component import CodeEditor
            
            editor = CodeEditor()
            config = editor.get_config(
                language='python',
                theme='dark'
            )
            
            self.assertIsNotNone(config)
        except ImportError:
            self.skipTest("CodeEditor not available")
    
    def test_progress_indicator(self):
        """Test progress indicator component."""
        try:
            from bubblelabs_ui_component import ProgressIndicator
            
            indicator = ProgressIndicator()
            html = indicator.render(
                current=50,
                total=100,
                label='Processing'
            )
            
            self.assertIsNotNone(html)
        except ImportError:
            self.skipTest("ProgressIndicator not available")
    
    def test_sidebar_navigation(self):
        """Test sidebar navigation component."""
        try:
            from bubblelabs_ui_component import SidebarNavigation
            
            nav = SidebarNavigation()
            config = nav.generate_config(
                items=[
                    {'label': 'Home', 'path': '/'},
                    {'label': 'Analysis', 'path': '/analysis'}
                ]
            )
            
            self.assertIsNotNone(config)
        except ImportError:
            self.skipTest("SidebarNavigation not available")


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoint functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_endpoint_registration(self):
        """Test endpoint registration."""
        try:
            from api_server import EndpointRegistry
            
            registry = EndpointRegistry()
            registry.register(
                path='/api/v1/analyze',
                methods=['POST'],
                handler='analyze_handler'
            )
            
            endpoints = registry.list_endpoints()
            self.assertIn('/api/v1/analyze', endpoints)
        except ImportError:
            self.skipTest("api_server module not available")
    
    def test_request_parsing(self):
        """Test request parsing."""
        try:
            from api_server import RequestParser
            
            parser = RequestParser()
            result = parser.parse(
                method='POST',
                path='/api/v1/analyze',
                body={'problem': 'test problem'}
            )
            
            self.assertIn('parsed', result)
        except ImportError:
            self.skipTest("RequestParser not available")
    
    def test_response_formatting(self):
        """Test response formatting."""
        try:
            from api_server import ResponseFormatter
            
            formatter = ResponseFormatter()
            response = formatter.success(
                data={'result': 'success'},
                message='Operation completed'
            )
            
            self.assertEqual(response['status'], 200)
        except ImportError:
            self.skipTest("ResponseFormatter not available")
    
    def test_error_handling(self):
        """Test API error handling."""
        try:
            from api_server import ErrorHandler
            
            handler = ErrorHandler()
            response = handler.handle(
                error_code=404,
                message='Resource not found'
            )
            
            self.assertEqual(response['status'], 404)
        except ImportError:
            self.skipTest("ErrorHandler not available")
    
    def test_middleware_chain(self):
        """Test middleware chain."""
        try:
            from api_server import MiddlewareChain
            
            chain = MiddlewareChain()
            chain.add_middleware('auth')
            chain.add_middleware('logging')
            
            self.assertEqual(len(chain.get_chain()), 2)
        except ImportError:
            self.skipTest("MiddlewareChain not available")
    
    def test_rate_limiting_endpoint(self):
        """Test endpoint rate limiting."""
        try:
            from api_server import EndpointRateLimiter
            
            limiter = EndpointRateLimiter()
            allowed = limiter.allow_request(
                endpoint='/api/v1/analyze',
                client_id='client-1'
            )
            
            self.assertTrue(allowed)
        except ImportError:
            self.skipTest("EndpointRateLimiter not available")


class TestWebSocketHandling(unittest.TestCase):
    """Test WebSocket handling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_websocket_manager(self):
        """Test WebSocket manager creation."""
        try:
            from websocket_manager import WebSocketManager
            manager = WebSocketManager()
            self.assertIsNotNone(manager)
        except ImportError:
            self.skipTest("websocket_manager module not available")
    
    def test_connection_handling(self):
        """Test WebSocket connection handling."""
        try:
            from websocket_manager import ConnectionHandler
            
            handler = ConnectionHandler()
            conn_id = handler.accept_connection(
                client_id='client-123',
                metadata={'agent': 'test'}
            )
            
            self.assertIsNotNone(conn_id)
        except ImportError:
            self.skipTest("ConnectionHandler not available")
    
    def test_message_sending(self):
        """Test WebSocket message sending."""
        try:
            from websocket_manager import MessageSender
            
            sender = MessageSender()
            sent = sender.send(
                connection_id='conn-123',
                message={'type': 'update', 'data': 'test'}
            )
            
            self.assertTrue(sent)
        except ImportError:
            self.skipTest("MessageSender not available")
    
    def test_broadcast_functionality(self):
        """Test broadcast functionality."""
        try:
            from websocket_manager import BroadcastManager
            
            broadcast = BroadcastManager()
            count = broadcast.broadcast(
                message={'type': 'announcement'},
                filter_connections={'room': 'updates'}
            )
            
            self.assertGreater(count, 0)
        except ImportError:
            self.skipTest("BroadcastManager not available")
    
    def test_heartbeat_handling(self):
        """Test heartbeat handling."""
        try:
            from websocket_manager import HeartbeatHandler
            
            handler = HeartbeatHandler()
            result = handler.process_heartbeat(
                connection_id='conn-123'
            )
            
            self.assertTrue(result)
        except ImportError:
            self.skipTest("HeartbeatHandler not available")
    
    def test_disconnection_handling(self):
        """Test disconnection handling."""
        try:
            from websocket_manager import DisconnectionHandler
            
            handler = DisconnectionHandler()
            handler.handle_disconnect(
                connection_id='conn-123',
                reason='client_closed'
            )
            
            active = handler.get_active_connections()
            self.assertNotIn('conn-123', active)
        except ImportError:
            self.skipTest("DisconnectionHandler not available")


class TestFrontendUtilities(unittest.TestCase):
    """Test frontend utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_json_helpers(self):
        """Test JSON helper functions."""
        try:
            from frontend_utils import JSONHelpers
            
            helpers = JSONHelpers()
            parsed = helpers.safe_parse('{"key": "value"}')
            
            self.assertEqual(parsed['key'], 'value')
        except ImportError:
            self.skipTest("JSONHelpers not available")
    
    def test_date_helpers(self):
        """Test date helper functions."""
        try:
            from frontend_utils import DateHelpers
            
            helpers = DateHelpers()
            formatted = helpers.format_date(
                datetime.now(),
                format='YYYY-MM-DD'
            )
            
            self.assertIsNotNone(formatted)
        except ImportError:
            self.skipTest("DateHelpers not available")
    
    def test_validation_helpers(self):
        """Test validation helper functions."""
        try:
            from frontend_utils import ValidationHelpers
            
            helpers = ValidationHelpers()
            is_valid = helpers.is_valid_email('test@example.com')
            
            self.assertTrue(is_valid)
        except ImportError:
            self.skipTest("ValidationHelpers not available")
    
    def test_storage_helpers(self):
        """Test browser storage helpers."""
        try:
            from frontend_utils import StorageHelpers
            
            helpers = StorageHelpers()
            helpers.set_item('test_key', 'test_value')
            value = helpers.get_item('test_key')
            
            self.assertEqual(value, 'test_value')
        except ImportError:
            self.skipTest("StorageHelpers not available")
    
    def test_url_helpers(self):
        """Test URL helper functions."""
        try:
            from frontend_utils import URLHelpers
            
            helpers = URLHelpers()
            params = helpers.parse_params('?key=value&foo=bar')
            
            self.assertEqual(params['key'], 'value')
        except ImportError:
            self.skipTest("URLHelpers not available")
    
    def test_debounce_helpers(self):
        """Test debounce functionality."""
        try:
            from frontend_utils import DebounceHelper
            
            helper = DebounceHelper(wait=100)
            result = helper.create_wrapper(lambda: None)
            
            self.assertTrue(callable(result))
        except ImportError:
            self.skipTest("DebounceHelper not available")


class TestNotificationSystem(unittest.TestCase):
    """Test notification system functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_notification_manager(self):
        """Test NotificationManager creation."""
        try:
            from notification_system import NotificationManager
            manager = NotificationManager()
            self.assertIsNotNone(manager)
        except ImportError:
            self.skipTest("notification_system module not available")
    
    def test_notification_creation(self):
        """Test notification creation."""
        try:
            from notification_system import NotificationCreator
            
            creator = NotificationCreator()
            notification = creator.create(
                type='success',
                title='Task Completed',
                message='Your task has been completed successfully'
            )
            
            self.assertEqual(notification['type'], 'success')
        except ImportError:
            self.skipTest("NotificationCreator not available")
    
    def test_notification_display(self):
        """Test notification display."""
        try:
            from notification_system import NotificationDisplay
            
            display = NotificationDisplay()
            html = display.render(
                notification={'type': 'info', 'message': 'Hello'}
            )
            
            self.assertIsNotNone(html)
        except ImportError:
            self.skipTest("NotificationDisplay not available")
    
    def test_notification_queue(self):
        """Test notification queue."""
        try:
            from notification_system import NotificationQueue
            
            queue = NotificationQueue()
            queue.enqueue({'message': 'First'})
            queue.enqueue({'message': 'Second'})
            
            self.assertEqual(queue.get_queue_size(), 2)
        except ImportError:
            self.skipTest("NotificationQueue not available")
    
    def test_toast_notifications(self):
        """Test toast notification system."""
        try:
            from notification_system import ToastManager
            
            manager = ToastManager()
            toast_id = manager.show(
                message='Operation completed',
                duration=3000
            )
            
            self.assertIsNotNone(toast_id)
        except ImportError:
            self.skipTest("ToastManager not available")


class TestProgressTracking(unittest.TestCase):
    """Test progress tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_progress_tracker(self):
        """Test ProgressTracker creation."""
        try:
            from progress_tracking import ProgressTracker
            tracker = ProgressTracker()
            self.assertIsNotNone(tracker)
        except ImportError:
            self.skipTest("progress_tracking module not available")
    
    def test_task_progress(self):
        """Test task progress tracking."""
        try:
            from progress_tracking import TaskProgress
            
            progress = TaskProgress()
            progress.start('task-1', 'Analyzing problem')
            progress.update('task-1', current=50, total=100)
            
            status = progress.get_status('task-1')
            self.assertEqual(status['current'], 50)
        except ImportError:
            self.skipTest("TaskProgress not available")
    
    def test_progress_visualization(self):
        """Test progress visualization."""
        try:
            from progress_tracking import ProgressVisualizer
            
            visualizer = ProgressVisualizer()
            html = visualizer.render_bar(
                current=75,
                total=100,
                label='Processing'
            )
            
            self.assertIsNotNone(html)
        except ImportError:
            self.skipTest("ProgressVisualizer not available")
    
    def test_multiprogress(self):
        """Test multi-progress tracking."""
        try:
            from progress_tracking import MultiProgress
            
            multi = MultiProgress()
            multi.create_task('task-a', 'Task A')
            multi.create_task('task-b', 'Task B')
            
            tasks = multi.get_all_tasks()
            self.assertEqual(len(tasks), 2)
        except ImportError:
            self.skipTest("MultiProgress not available")
    
    def test_progress_callbacks(self):
        """Test progress callbacks."""
        try:
            from progress_tracking import ProgressCallbacks
            
            callbacks = ProgressCallbacks()
            called = [False]
            
            def on_complete():
                called[0] = True
            
            callbacks.register('complete', on_complete)
            callbacks.trigger('complete')
            
            self.assertTrue(called[0])
        except ImportError:
            self.skipTest("ProgressCallbacks not available")


class TestFormHandling(unittest.TestCase):
    """Test form handling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_form_builder(self):
        """Test FormBuilder creation."""
        try:
            from form_handling import FormBuilder
            builder = FormBuilder()
            self.assertIsNotNone(builder)
        except ImportError:
            self.skipTest("form_handling module not available")
    
    def test_field_creation(self):
        """Test form field creation."""
        try:
            from form_handling import FieldFactory
            
            factory = FieldFactory()
            field = factory.create(
                type='text',
                name='username',
                label='Username',
                required=True
            )
            
            self.assertEqual(field['type'], 'text')
        except ImportError:
            self.skipTest("FieldFactory not available")
    
    def test_form_validation(self):
        """Test form validation."""
        try:
            from form_handling import FormValidator
            
            validator = FormValidator()
            result = validator.validate(
                form_data={'username': 'test', 'email': 'test@example.com'},
                rules={'username': {'required': True}, 'email': {'type': 'email'}}
            )
            
            self.assertTrue(result.valid)
        except ImportError:
            self.skipTest("FormValidator not available")
    
    def test_form_rendering(self):
        """Test form rendering."""
        try:
            from form_handling import FormRenderer
            
            renderer = FormRenderer()
            html = renderer.render(
                fields=[{'type': 'text', 'name': 'test'}]
            )
            
            self.assertIsNotNone(html)
        except ImportError:
            self.skipTest("FormRenderer not available")
    
    def test_form_submission(self):
        """Test form submission handling."""
        try:
            from form_handling import FormSubmitter
            
            submitter = FormSubmitter()
            result = submitter.submit(
                form_id='contact-form',
                data={'name': 'John', 'message': 'Hello'}
            )
            
            self.assertTrue(result.success)
        except ImportError:
            self.skipTest("FormSubmitter not available")


if __name__ == '__main__':
    unittest.main()
