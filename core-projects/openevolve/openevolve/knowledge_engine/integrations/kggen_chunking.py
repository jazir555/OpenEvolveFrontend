"""
Advanced Document Chunking for KG-Gen Pipeline

This module provides intelligent document chunking capabilities optimized for
knowledge graph extraction, including sentence boundary preservation and
overlap handling.
"""

import logging
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.
    """
    text: str
    chunk_id: int
    start_pos: int
    end_pos: int
    metadata: Optional[dict] = None

    def __len__(self):
        return len(self.text)


class DocumentChunker:
    """
    Intelligent document chunking for kg-gen pipeline.

    Features:
    - NLTK sentence tokenization
    - Configurable chunk sizes
    - Word-level fallback
    - Overlap preservation
    """

    def __init__(self, chunk_size: int = 5000, overlap: int = 200):
        """
        Initialize document chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Overlap size between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._sentence_tokenizer = None

        # Try to import NLTK for better sentence tokenization
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            self._sentence_tokenizer = nltk.sent_tokenize
            logger.info("NLTK sentence tokenizer loaded")
        except ImportError:
            logger.warning("NLTK not available, using fallback sentence splitting")
            self._sentence_tokenizer = self._fallback_sentence_tokenize

    def chunk_document(self, text: str) -> List[Chunk]:
        """
        Split document into intelligent chunks.

        Args:
            text: Input document text

        Returns:
            List of Chunk objects
        """
        return self.chunk_with_preservation(text, preserve_sentences=True)

    def chunk_with_preservation(
        self,
        text: str,
        preserve_sentences: bool = True
    ) -> List[Chunk]:
        """
        Chunk while preserving sentence boundaries.

        Args:
            text: Input document text
            preserve_sentences: Whether to preserve sentence boundaries

        Returns:
            List of Chunk objects
        """
        if preserve_sentences:
            return self._chunk_by_sentences(text)
        else:
            return self._chunk_by_size(text)

    def _chunk_by_sentences(self, text: str) -> List[Chunk]:
        """
        Chunk text while preserving sentence boundaries.

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        # Tokenize into sentences
        sentences = self._sentence_tokenizer(text)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        position = 0

        for sentence in sentences:
            sentence_size = len(sentence) + 1  # +1 for space

            # Check if adding this sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_pos=position,
                    end_pos=position + len(chunk_text),
                    metadata={
                        'sentence_count': len(current_chunk),
                        'method': 'sentence_preservation'
                    }
                ))

                # Start new chunk with overlap
                chunk_id += 1
                position += len(chunk_text) - self.overlap

                # Keep last few sentences for overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_size = sum(len(s) + 1 for s in overlap_sentences)

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                start_pos=position,
                end_pos=position + len(chunk_text),
                metadata={
                    'sentence_count': len(current_chunk),
                    'method': 'sentence_preservation'
                }
            ))

        logger.info(f"Chunked into {len(chunks)} chunks using sentence preservation")
        return chunks

    def _chunk_by_size(self, text: str) -> List[Chunk]:
        """
        Chunk text by size without sentence preservation.

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        chunks = []
        chunk_id = 0
        position = 0

        while position < len(text):
            end_pos = min(position + self.chunk_size, len(text))
            chunk_text = text[position:end_pos]

            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                start_pos=position,
                end_pos=end_pos,
                metadata={
                    'method': 'size_based'
                }
            ))

            chunk_id += 1
            position = end_pos - self.overlap

        logger.info(f"Chunked into {len(chunks)} chunks using size-based splitting")
        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        Get last few sentences for overlap.

        Args:
            sentences: List of sentences in current chunk

        Returns:
            List of sentences to include in overlap
        """
        overlap_chars = 0
        overlap_sentences = []

        # Take sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            if overlap_chars + len(sentence) > self.overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_chars += len(sentence) + 1

        return overlap_sentences

    def _fallback_sentence_tokenize(self, text: str) -> List[str]:
        """
        Fallback sentence tokenization using regex.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def chunk_by_paragraphs(
        self,
        text: str,
        max_paragraphs_per_chunk: int = 10
    ) -> List[Chunk]:
        """
        Chunk text by paragraphs.

        Args:
            text: Input text
            max_paragraphs_per_chunk: Maximum paragraphs per chunk

        Returns:
            List of chunks
        """
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        chunk_id = 0
        position = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            current_chunk.append(para)

            if len(current_chunk) >= max_paragraphs_per_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_pos=position,
                    end_pos=position + len(chunk_text),
                    metadata={
                        'paragraph_count': len(current_chunk),
                        'method': 'paragraph_based'
                    }
                ))

                chunk_id += 1
                position += len(chunk_text)
                current_chunk = []

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                start_pos=position,
                end_pos=position + len(chunk_text),
                metadata={
                    'paragraph_count': len(current_chunk),
                    'method': 'paragraph_based'
                }
            ))

        logger.info(f"Chunked into {len(chunks)} chunks using paragraph-based splitting")
        return chunks

    def chunk_by_semantic_units(
        self,
        text: str,
        unit_markers: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Chunk text by semantic units (e.g., sections, subsections).

        Args:
            text: Input text
            unit_markers: List of markers that delimit semantic units

        Returns:
            List of chunks
        """
        if unit_markers is None:
            unit_markers = [
                r'^#+\s',  # Markdown headers
                r'^Chapter\s+\d+',
                r'^Section\s+\d+',
                r'^\d+\.\s',
            ]

        # Combine markers into pattern
        pattern = '|'.join(f'({marker})' for marker in unit_markers)
        sections = re.split(pattern, text, flags=re.MULTILINE)

        chunks = []
        chunk_id = 0
        position = 0

        for section in sections:
            section = section.strip()
            if not section or len(section) < 100:  # Skip very short sections
                continue

            # If section is too large, subdivide it
            if len(section) > self.chunk_size:
                sub_chunks = self._chunk_by_size(section)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = chunk_id
                    sub_chunk.metadata = sub_chunk.metadata or {}
                    sub_chunk.metadata['method'] = 'semantic_subdivision'
                    chunks.append(sub_chunk)
                    chunk_id += 1
            else:
                chunks.append(Chunk(
                    text=section,
                    chunk_id=chunk_id,
                    start_pos=position,
                    end_pos=position + len(section),
                    metadata={
                        'method': 'semantic_unit'
                    }
                ))
                chunk_id += 1

            position += len(section)

        logger.info(f"Chunked into {len(chunks)} chunks using semantic units")
        return chunks

    def get_chunk_statistics(self, chunks: List[Chunk]) -> dict:
        """
        Get statistics about chunks.

        Args:
            chunks: List of chunks

        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_length': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0
            }

        lengths = [len(chunk) for chunk in chunks]

        return {
            'total_chunks': len(chunks),
            'total_length': sum(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'methods': {
                chunk.metadata.get('method', 'unknown')
                for chunk in chunks
                if chunk.metadata
            }
        }
