import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from knowledge_engine.context_manager import ContextManager

class TestContextManager(unittest.TestCase):
    
    def setUp(self):
        self.cm = ContextManager()
        
    @patch('knowledge_engine.context_manager.os.path.exists')
    @patch('knowledge_engine.context_manager.os.path.getsize')
    @patch('knowledge_engine.context_manager.MatryoshkaClient')
    def test_process_large_document(self, MockClient, mock_getsize, mock_exists):
        # Setup
        mock_exists.return_value = True
        # 15 MB
        mock_getsize.return_value = 15 * 1024 * 1024 
        
        # Mock Matryoshka client instance
        mock_client_instance = MockClient.return_value
        mock_client_instance.is_available.return_value = True
        mock_client_instance.analyze.return_value = "Matryoshka Analysis"
        
        # Inject mock client into cm
        self.cm.matryoshka = mock_client_instance
        
        # Execute
        result = self.cm.process_document("query", "/path/to/large.txt")
        
        # Verify
        self.assertEqual(result, "Matryoshka Analysis")
        mock_client_instance.analyze.assert_called_once_with("query", "/path/to/large.txt")
        
    @patch('knowledge_engine.context_manager.os.path.exists')
    @patch('knowledge_engine.context_manager.os.path.getsize')
    def test_process_small_document(self, mock_getsize, mock_exists):
        # Setup
        mock_exists.return_value = True
        # 1 MB
        mock_getsize.return_value = 1 * 1024 * 1024 
        
        # Mock open to return content (need to patch open in context_manager module)
        with patch('builtins.open', unittest.mock.mock_open(read_data="Small Content")):
            # Execute
            result = self.cm.process_document("query", "/path/to/small.txt")
            
        # Verify
        self.assertTrue("Truncated" in result or "Content" in result)
        # Should not call matryoshka.analyze (we haven't mocked it here but it shouldn't be called)
        
    @patch('knowledge_engine.context_manager.os.path.exists')
    def test_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            self.cm.process_document("query", "nonexistent.txt")

    @patch('knowledge_engine.context_manager.MatryoshkaClient')
    def test_process_input_url(self, MockClient):
        # Mock Matryoshka client instance
        mock_client_instance = MockClient.return_value
        mock_client_instance.is_available.return_value = True
        mock_client_instance.analyze_url.return_value = "URL Analysis Result"
        
        # Inject mock client into cm
        self.cm.matryoshka = mock_client_instance
        
        # Execute
        result = self.cm.process_input("query", "http://example.com/big.txt", input_type='url')
        
        # Verify
        self.assertEqual(result, "URL Analysis Result")
        mock_client_instance.analyze_url.assert_called_once_with("query", "http://example.com/big.txt")

if __name__ == '__main__':
    unittest.main()
