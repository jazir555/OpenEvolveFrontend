import unittest
import os
from unittest.mock import patch, MagicMock
from glue.adapters.matryoshka_adapter import MatryoshkaClient

class TestMatryoshkaAdapter(unittest.TestCase):
    
    def setUp(self):
        self.client = MatryoshkaClient()
        # Mock executable path existence
        self.client.executable_path = "/mock/path/to/index.js"
        
    @patch('glue.adapters.matryoshka_adapter.os.path.exists')
    @patch('glue.adapters.matryoshka_adapter.subprocess.run')
    def test_analyze_command_construction(self, mock_run, mock_exists):
        # Setup
        mock_exists.return_value = True # File exists
        mock_run.return_value = MagicMock(returncode=0, stdout="Analysis Result")
        
        # Execute
        result = self.client.analyze(
            query="Summarize", 
            file_path="/path/to/doc.txt",
            max_turns=5,
            model="gpt-4"
        )
        
        # Verify
        self.assertEqual(result, "Analysis Result")
        
        # Check command arguments
        args, kwargs = mock_run.call_args
        cmd = args[0]
        
        self.assertEqual(cmd[0], "node")
        self.assertEqual(cmd[1], "/mock/path/to/index.js")
        self.assertEqual(cmd[2], "Summarize")
        self.assertEqual(cmd[3], "/path/to/doc.txt")
        
        # Check options
        self.assertIn("--max-turns", cmd)
        idx = cmd.index("--max-turns")
        self.assertEqual(cmd[idx+1], "5")
        
        self.assertIn("--model", cmd)
        idx = cmd.index("--model")
        self.assertEqual(cmd[idx+1], "gpt-4")
        
    @patch('glue.adapters.matryoshka_adapter.os.path.exists')
    def test_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            self.client.analyze("query", "nonexistent.txt")

    @patch('glue.adapters.matryoshka_adapter.tempfile.NamedTemporaryFile')
    @patch('glue.adapters.matryoshka_adapter.os.path.exists') # For cleanup check
    @patch('glue.adapters.matryoshka_adapter.os.unlink')      # For cleanup
    def test_analyze_text(self, mock_unlink, mock_exists, mock_temp):
        # Setup mock temp file
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/mock_file.txt"
        mock_temp.return_value.__enter__.return_value = mock_tmp
        
        # Mock analyze (internal call)
        with patch.object(self.client, 'analyze', return_value="Text Result") as mock_analyze:
            mock_exists.return_value = True # File exists for cleanup
            
            # Execute
            result = self.client.analyze_text("Summarize", "Some content")
            
            # Verify
            self.assertEqual(result, "Text Result")
            mock_tmp.write.assert_called_with("Some content")
            mock_analyze.assert_called_with("Summarize", "/tmp/mock_file.txt")
            mock_unlink.assert_called_with("/tmp/mock_file.txt")

    @patch('glue.adapters.matryoshka_adapter.urllib.request.urlopen')
    @patch('glue.adapters.matryoshka_adapter.tempfile.NamedTemporaryFile')
    @patch('glue.adapters.matryoshka_adapter.os.path.exists')
    @patch('glue.adapters.matryoshka_adapter.os.unlink')
    def test_analyze_url(self, mock_unlink, mock_exists, mock_temp, mock_urlopen):
        # Setup mock temp file
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/url_file.txt"
        mock_temp.return_value.__enter__.return_value = mock_tmp
        
        # Setup mock url response
        mock_response = MagicMock()
        mock_response.read.return_value = b"URL Content"
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Mock analyze
        with patch.object(self.client, 'analyze', return_value="URL Result") as mock_analyze:
            # Need to mock open() for writing the downloaded content
            with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
                mock_exists.return_value = True
                
                # Execute
                result = self.client.analyze_url("Extract", "http://example.com")
                
                # Verify
                self.assertEqual(result, "URL Result")
                mock_file().write.assert_called_with(b"URL Content")
                mock_analyze.assert_called_with("Extract", "/tmp/url_file.txt")
                mock_unlink.assert_called_with("/tmp/url_file.txt")

if __name__ == '__main__':
    unittest.main()
