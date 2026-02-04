"""
Context Manager for Layer 5: Context Management

This module handles the processing of documents for the Deterministic Pipeline.
It routes large documents (>10MB) to Matryoshka and smaller documents to standard RAG (DSPy).
"""

import os
import logging
from typing import Optional, Any
from glue.adapters.matryoshka_adapter import MatryoshkaClient

# Try to import dspy, handle if not configured
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages context extraction from documents of varying sizes.
    Layer 5 of the Deterministic Pipeline.
    """

    def __init__(self):
        """Initialize the ContextManager."""
        self.matryoshka = MatryoshkaClient()
        self.threshold_mb = 10.0 # Threshold in MB to switch to Matryoshka
        self.enabled = os.getenv("MATRYOSHKA_ENABLED", "true").lower() == "true"

    def process_input(self, query: str, input_data: str, input_type: str = 'file') -> str:
        """
        Process general input (file, text, or URL).

        Args:
            query: Analysis query.
            input_data: File path, text content, or URL.
            input_type: One of 'file', 'text', 'url'.

        Returns:
            Analysis result.
        """
        if input_type == 'file':
            return self.process_document(query, input_data)
        
        elif input_type == 'text':
            # Check length to decide strategy (e.g. > 100KB -> Matryoshka)
            size_mb = len(input_data.encode('utf-8')) / (1024 * 1024)
            if size_mb > self.threshold_mb and self.enabled:
                logger.info(f"Text input size ({size_mb:.2f} MB) exceeds threshold. Using Matryoshka.")
                if self.matryoshka.is_available():
                     return self.matryoshka.analyze_text(query, input_data)
                return "Matryoshka unavailable for large text."
            else:
                 # Small text: Just return it or use RAG if implemented
                 return f"Analysis of text:\n{input_data[:1000]}..." # Placeholder
                 
        elif input_type == 'url':
             # For URL, we always use Matryoshka as we don't know size easily without fetching
             if self.enabled:
                logger.info(f"Processing URL {input_data} with Matryoshka.")
                if self.matryoshka.is_available():
                    return self.matryoshka.analyze_url(query, input_data)
                return "Matryoshka unavailable for URL analysis."
             else:
                return "Matryoshka is disabled; URL analysis requires Matryoshka."
             
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

    def process_document(self, query: str, document_path: str, size_mb: Optional[float] = None) -> str:
        """
        Process a document and return the relevant context or analysis.

        Args:
            query: The query or task to perform on the document.
            document_path: Path to the document file.
            size_mb: Optional size in MB. If not provided, it will be calculated.

        Returns:
            The analysis string or context.
        """
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")

        if size_mb is None:
            size_bytes = os.path.getsize(document_path)
            size_mb = size_bytes / (1024 * 1024)

        logger.info(f"Processing document {document_path} ({size_mb:.2f} MB) with query: {query}")

        if size_mb > self.threshold_mb and self.enabled:
            logger.info("Document size exceeds threshold. Using Matryoshka.")
            if not self.matryoshka.is_available():
                logger.warning("Matryoshka is not available. Falling back to standard RAG, but performance may degrade.")
                return self._standard_rag(query, document_path)
            
            try:
                return self.matryoshka.analyze(query, document_path)
            except Exception as e:
                logger.error(f"Matryoshka analysis failed: {e}. Falling back to standard RAG.")
                return self._standard_rag(query, document_path)
        else:
            if not self.enabled and size_mb > self.threshold_mb:
                logger.info("Large document detected but Matryoshka is disabled. Using Standard RAG.")
            else:
                logger.info("Document size within standard limits. Using Standard RAG.")
            return self._standard_rag(query, document_path)

    def _standard_rag(self, query: str, document_path: str) -> str:
        """
        Standard RAG implementation using DSPy or simple fallback.
        """
        if DSPY_AVAILABLE:
            try:
                # Placeholder for DSPy retrieval logic
                # In a real scenario, we'd ingest the doc into a vector DB or use a temporary index
                # For now, assuming dspy.Retrieve works if configured, but here we might just read the file 
                # if it's small enough to fit in context, or use a basic sliding window.
                
                # Simple fallback: Read the file and return content if it's small enough (e.g. < 100KB)
                # otherwise, we really need a retriever.
                file_size = os.path.getsize(document_path)
                if file_size < 100 * 1024: # 100KB
                    with open(document_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    return f"Context from {document_path}:\n{content}"
                
                # If larger, we would ideally use DSPy Retrieve.
                # Since we don't have a configured retriever index for this specific file on the fly, 
                # we return a placeholder message or simple truncation.
                logger.warning("Standard RAG not fully implemented for ad-hoc files. Returning truncated content.")
                with open(document_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(10000) # First 10KB
                return f"Context from {document_path} (Truncated):\n{content}..."
                
            except Exception as e:
                logger.error(f"Standard RAG failed: {e}")
                return f"Error reading document: {e}"
        else:
            # Fallback if DSPy not available
            try:
                with open(document_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(10000)
                return f"Context from {document_path} (Truncated, DSPy unavailable):\n{content}..."
            except Exception as e:
                return f"Error reading document: {e}"
