"""
Matryoshka Adapter

This module provides a Python interface to the Matryoshka Recursive Language Model (RLM) system.
It wraps the Node.js CLI to allow analyzing large documents from within the Python codebase.
"""

import os
import sys
import subprocess
import json
import logging
import tempfile
import urllib.request
from typing import Dict, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class MatryoshkaClient:
    """
    Client for interacting with the Matryoshka RLM CLI.
    """
    
    def __init__(self, executable_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the Matryoshka client.
        
        Args:
            executable_path: Path to the compiled Matryoshka index.js file. 
                             Defaults to core-projects/Matryoshka/dist/index.js
            config_path: Path to a config.json file for Matryoshka.
        """
        self.root_dir = self._find_project_root()
        
        if executable_path:
            self.executable_path = executable_path
        else:
            self.executable_path = os.path.join(
                self.root_dir, 
                "core-projects", 
                "Matryoshka", 
                "dist", 
                "index.js"
            )
            
        self.config_path = config_path
        self._verify_installation()
        
    def _find_project_root(self) -> str:
        """Find the project root directory."""
        # Assume we are in glue/adapters/, so root is ../../
        # Adjust logic if file moves
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up until we find core-projects or root markers
        root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        return root

    def _verify_installation(self):
        """Verify that the Matryoshka executable exists."""
        if not os.path.exists(self.executable_path):
            logger.warning(f"Matryoshka executable not found at {self.executable_path}. Please build it first.")
            # We don't raise error here to allow import even if not built, but usage will fail.

    def analyze(self, 
                query: str, 
                file_path: str, 
                max_turns: int = 10,
                timeout_ms: int = 30000,
                model: Optional[str] = None,
                verbose: bool = False) -> str:
        """
        Analyze a document using Matryoshka.
        
        Args:
            query: The question or task to perform.
            file_path: Path to the document file.
            max_turns: Maximum exploration turns (default 10).
            timeout_ms: Timeout per turn in ms (default 30000).
            model: Optional model override.
            verbose: Enable verbose logging.
            
        Returns:
            The analysis result as a string.
            
        Raises:
            FileNotFoundError: If the document file doesn't exist.
            RuntimeError: If Matryoshka execution fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
            
        cmd = [
            "node", 
            self.executable_path,
            query,
            file_path,
            "--max-turns", str(max_turns),
            "--timeout", str(timeout_ms)
        ]
        
        if self.config_path:
            cmd.extend(["--config", self.config_path])
            
        if model:
            cmd.extend(["--model", model])
            
        if verbose:
            cmd.append("--verbose")
            
        logger.info(f"Running Matryoshka: {' '.join(cmd)}")
        
        try:
            # Set CWD to project root so relative paths work if needed
            result = subprocess.run(
                cmd,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=False # We handle return code manually
            )
            
            if result.returncode != 0:
                error_msg = f"Matryoshka failed (exit code {result.returncode}):\n{result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            return result.stdout.strip()
            
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to execute Matryoshka subprocess: {e}")
            raise RuntimeError(f"Subprocess execution failed: {e}")

    def analyze_text(self, 
                     query: str, 
                     text: str, 
                     **kwargs) -> str:
        """
        Analyze raw text using Matryoshka.
        Writes text to a temporary file and analyzes it.
        """
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as tmp:
            tmp.write(text)
            tmp_path = tmp.name
            
        try:
            return self.analyze(query, tmp_path, **kwargs)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def analyze_url(self, 
                    query: str, 
                    url: str, 
                    **kwargs) -> str:
        """
        Analyze content from a URL using Matryoshka.
        Downloads content to a temporary file and analyzes it.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            tmp_path = tmp.name
            
        try:
            with urllib.request.urlopen(url) as response, open(tmp_path, 'wb') as out_file:
                out_file.write(response.read())
                
            return self.analyze(query, tmp_path, **kwargs)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise RuntimeError(f"Failed to download or analyze URL {url}: {e}")
        finally:
            # Note: analyze() calls the process. If analyze fails, we still want to cleanup.
            # However, if analyze succeeds, we also want cleanup.
            # analyze_text handles its own cleanup. Here we do it manually.
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def is_available(self) -> bool:
        """Check if Matryoshka is installed and runnable."""
        return os.path.exists(self.executable_path)

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    client = MatryoshkaClient()
    if client.is_available():
        print(f"Matryoshka found at {client.executable_path}")
    else:
        print("Matryoshka not found")
