import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from knowledge_engine.global_context_manager import GlobalContextManager

class TestGlobalContextManagement(unittest.TestCase):
    
    def setUp(self):
        # Reset singleton instance for testing if necessary, 
        # but here we can just initialize it
        self.gcm = GlobalContextManager()
        self.gcm.context_threshold_chars = 100 # Low threshold for testing
        
    @patch('knowledge_engine.global_context_manager.StatefulMatryoshkaClient')
    def test_manage_no_compression(self, MockClient):
        # Setup
        mock_client_instance = MockClient.return_value
        mock_client_instance.is_available.return_value = True
        self.gcm.client = mock_client_instance
        
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Short message"}
        ]
        
        # Execute
        result = self.gcm.manage("session1", messages)
        
        # Verify
        self.assertEqual(len(result), 2)
        self.assertEqual(result, messages)
        mock_client_instance.distill_history.assert_not_called()

    @patch('knowledge_engine.global_context_manager.StatefulMatryoshkaClient')
    def test_manage_with_compression(self, MockClient):
        # Setup
        mock_client_instance = MockClient.return_value
        mock_client_instance.is_available.return_value = True
        mock_client_instance.distill_history.return_value = "Distilled summary"
        self.gcm.client = mock_client_instance
        
        # Exceed threshold (100) or force compress
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Middle message 1"},
            {"role": "user", "content": "Middle message 2"},
            {"role": "user", "content": "Latest message 1"},
            {"role": "user", "content": "Latest message 2"}
        ]
        
        # Execute
        result = self.gcm.manage("session2", messages, force_compress=True)
        
        # Verify
        self.assertGreater(len(messages), len(result)) # 5 -> 4 (system, summary, latest2)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "system")
        self.assertIn("Distilled summary", result[1]["content"])
        self.assertEqual(result[2]["content"], "Latest message 1")
        
        mock_client_instance.distill_history.assert_called_once()

if __name__ == '__main__':
    unittest.main()
