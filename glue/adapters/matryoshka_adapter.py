"""
Matryoshka Adapter

This module provides a Python interface to the Matryoshka Recursive Language Model (RLM) system.
It wraps the Node.js CLI to allow analyzing large documents from within the Python codebase.

Enhanced with unified memory system integration for persistent, searchable analysis history
and cross-session learning capabilities.
"""

import os
import sys
import subprocess
import json
import logging
import tempfile
import urllib.request
from typing import Dict, Any, Optional, Union, List

# Unified memory system integration
try:
    from matryoshka_unified_memory_integration import (
        MatryoshkaMemoryBridge,
        UnifiedMatryoshkaClient,
        create_unified_matryoshka_client
    )
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError:
    UNIFIED_MEMORY_AVAILABLE = False

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


class UnifiedMemoryMatryoshkaClient(MatryoshkaClient):
    """
    Matryoshka client with full unified memory system integration.
    
    Features:
    - 4-layer indexing of exploration steps
    - Always-true state management
    - Hybrid retrieval for context
    - Cross-session learning
    """
    
    def __init__(self, *args, memory_storage_path: Optional[str] = None, **kwargs):
        """
        Initialize the unified memory Matryoshka client.
        
        Args:
            *args: Positional arguments passed to MatryoshkaClient
            memory_storage_path: Path to the memory database file.
                                 Defaults to "./matryoshka_memory.db"
            **kwargs: Keyword arguments passed to MatryoshkaClient
        """
        super().__init__(*args, **kwargs)
        
        if not UNIFIED_MEMORY_AVAILABLE:
            raise ImportError(
                "Unified memory system not available. "
                "Please ensure matryoshka_unified_memory_integration.py is in the Python path."
            )
        
        self.unified_client = create_unified_matryoshka_client(
            storage_path=memory_storage_path or "./matryoshka_memory.db"
        )
        
    def analyze_with_unified_memory(self, query: str, file_path: str, **kwargs):
        """
        Analyze using full unified memory system.
        
        Args:
            query: The question or task to perform
            file_path: Path to the document file
            **kwargs: Additional arguments passed to the unified client
            
        Returns:
            Analysis result with memory integration
        """
        return self.unified_client.analyze_with_memory(query, file_path, **kwargs)
        
    def continue_analysis(self, session_id: str, follow_up_query: str, **kwargs):
        """
        Continue a previous analysis with memory.
        
        Args:
            session_id: Session ID from a previous analysis
            follow_up_query: Follow-up question or task
            **kwargs: Additional arguments passed to the unified client
            
        Returns:
            Continued analysis result
        """
        return self.unified_client.continue_analysis(session_id, follow_up_query, **kwargs)
        
    def search_past_analyses(self, query: str, limit: int = 10):
        """
        Search insights across all past Matryoshka analyses.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of relevant past analyses
        """
        return self.unified_client.search_across_sessions(query, limit)


class StatefulMatryoshkaClient(MatryoshkaClient):
    """
    Extends MatryoshkaClient to manage stateful context sessions.
    Useful for solving 'context rot' in long-running LLM interactions.
    
    Now with optional unified memory system integration for persistent,
    searchable session storage and cross-session learning.
    """
    
    def __init__(self, *args, use_unified_memory: bool = False, memory_storage_path: Optional[str] = None, **kwargs):
        """
        Initialize the stateful Matryoshka client.
        
        Args:
            *args: Positional arguments passed to MatryoshkaClient
            use_unified_memory: Whether to use the unified memory system.
                               Defaults to False for backward compatibility.
            memory_storage_path: Path to the memory database (when using unified memory)
            **kwargs: Keyword arguments passed to MatryoshkaClient
        """
        super().__init__(*args, **kwargs)
        self.use_unified_memory = use_unified_memory
        
        if use_unified_memory:
            if not UNIFIED_MEMORY_AVAILABLE:
                logger.warning(
                    "Unified memory requested but not available. "
                    "Falling back to simple dict storage. "
                    "Please ensure matryoshka_unified_memory_integration.py is in the Python path."
                )
                self.use_unified_memory = False
                self.sessions: Dict[str, Dict[str, Any]] = {}
            else:
                self.unified_client = create_unified_matryoshka_client(memory_storage_path)
                self.sessions = {}  # Unified memory handles session storage
        else:
            self.sessions: Dict[str, Dict[str, Any]] = {}  # Legacy simple dict storage

    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "summary": "",
                "tokens": 0,
                "last_updated": None
            }
        return self.sessions[session_id]

    def compress_context(self, session_id: str, new_content: str, query: str = "Summarize the key information and state transitions.") -> str:
        """
        Use Matryoshka to compress current context session.
        """
        session = self.get_or_create_session(session_id)
        
        # Combine existing summary with new content
        full_text = f"EXISTING_SUMMARY: {session['summary']}\n\nNEW_CONTENT: {new_content}"
        
        # Run Matryoshka analysis to distill the context
        compressed = self.analyze_text(query, full_text)
        
        # Update session
        session["summary"] = compressed
        session["last_updated"] = os.getpid() # Simplified marker
        
        return compressed

    def distill_history(self, session_id: str, messages: List[Dict[str, str]]) -> str:
        """
        Distill a conversation history into a concise representation.
        """
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.analyze_text("Identify critical entities, decisions, and unanswered questions from this history.", history_text)
    
    def search_session_history(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search across all session history (requires unified memory).
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching historical analyses
        """
        if self.use_unified_memory and UNIFIED_MEMORY_AVAILABLE:
            return self.unified_client.search_across_sessions(query, limit)
        else:
            logger.warning("Unified memory not enabled. Session history search not available.")
            return []
    
    def continue_session(self, session_id: str, follow_up_query: str, file_path: str, **kwargs) -> str:
        """
        Continue a previous session with a follow-up query.
        
        When unified memory is enabled, this uses the memory system for
        context retrieval. Otherwise, it falls back to simple session state.
        
        Args:
            session_id: Session identifier
            follow_up_query: Follow-up question or task
            file_path: Path to the document file
            **kwargs: Additional arguments for analysis
            
        Returns:
            Analysis result
        """
        if self.use_unified_memory and UNIFIED_MEMORY_AVAILABLE:
            return self.unified_client.continue_analysis(session_id, follow_up_query, **kwargs)
        else:
            # Legacy fallback: use existing session summary
            session = self.get_or_create_session(session_id)
            context = f"Previous summary: {session.get('summary', 'None')}\n\nFollow-up: {follow_up_query}"
            return self.analyze(context, file_path, **kwargs)


def create_matryoshka_client(
    client_type: str = "basic",
    memory_storage_path: Optional[str] = None,
    **kwargs
) -> MatryoshkaClient:
    """
    Factory function for creating Matryoshka clients.
    
    This factory provides a convenient way to create different types of
    Matryoshka clients based on your needs.
    
    Args:
        client_type: Type of client to create
            - "basic": Standard MatryoshkaClient
            - "stateful": Simple stateful client with in-memory sessions
            - "unified": Full unified memory system with persistent storage
            - "stateful_unified": Stateful client with unified memory enabled
        memory_storage_path: Path to the memory database file
                             (used for "unified" and "stateful_unified" types)
        **kwargs: Additional arguments passed to the client constructor
        
    Returns:
        Configured Matryoshka client instance
        
    Raises:
        ValueError: If an unknown client_type is specified
        ImportError: If unified memory is requested but not available
        
    Examples:
        >>> # Basic client
        >>> client = create_matryoshka_client("basic")
        
        >>> # Stateful client with unified memory
        >>> client = create_matryoshka_client(
        ...     "stateful_unified",
        ...     memory_storage_path="./my_memory.db"
        ... )
        
        >>> # Full unified memory client
        >>> client = create_matryoshka_client("unified")
    """
    client_type = client_type.lower()
    
    if client_type == "basic":
        return MatryoshkaClient(**kwargs)
    
    elif client_type == "stateful":
        return StatefulMatryoshkaClient(**kwargs)
    
    elif client_type == "unified":
        if not UNIFIED_MEMORY_AVAILABLE:
            raise ImportError(
                "Unified memory system not available. "
                "Please ensure matryoshka_unified_memory_integration.py is in the Python path."
            )
        return UnifiedMemoryMatryoshkaClient(
            memory_storage_path=memory_storage_path,
            **kwargs
        )
    
    elif client_type == "stateful_unified":
        return StatefulMatryoshkaClient(
            use_unified_memory=True,
            memory_storage_path=memory_storage_path,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown client_type: '{client_type}'. "
            f"Valid options are: 'basic', 'stateful', 'unified', 'stateful_unified'"
        )


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    client = MatryoshkaClient()
    if client.is_available():
        print(f"Matryoshka found at {client.executable_path}")
    else:
        print("Matryoshka not found")
