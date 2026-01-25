"""
Document Chunking Module - Production Grade

Part of KG-Gen Sprint 2 Integration.

Following CLAUDE.md Principles:
- AIR GAP: Independent implementation, no imports from kg-gen source
- IDEMPOTENCY: Chunking is deterministic and reproducible
- CONFIGURATION EXPLICITNESS: All config via init parameters
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    A chunk of document text.
    """
    text: str
    chunk_index: int
    start_pos: int
    end_pos: int


class DocumentChunker:
    """
    Split documents into overlapping chunks for processing.

    LAW OF IDEMPOTENCY: Same input always produces same chunks.
    """

    def __init__(
        self,
        chunk_size: int = 5000,
        overlap: int = 200,
        separator: str = "\n\n"
    ):
        """
        Initialize document chunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            separator: Preferred separator for splitting
        """
        if chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {chunk_size}")
        if overlap < 0:
            raise ValueError(f"Invalid overlap: {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"Overlap {overlap} must be less than chunk_size {chunk_size}")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator

        logger.info(
            "DocumentChunker initialized",
            extra={"chunk_size": chunk_size, "overlap": overlap}
        )

    def chunk_document(self, text: str) -> List[DocumentChunk]:
        """
        Split document into chunks.

        Args:
            text: Document text

        Returns:
            List of DocumentChunk objects
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # If not at end, try to break at separator
            if end < len(text):
                # Look for separator near end
                separator_pos = text.rfind(self.separator, start, end)

                if separator_pos > start + (self.chunk_size // 2):
                    # Found good separator
                    end = separator_pos + len(self.separator)

            # Extract chunk
            chunk_text = text[start:end]

            # Create chunk object
            chunk = DocumentChunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_pos=start,
                end_pos=end
            )

            chunks.append(chunk)

            # Move start position with overlap
            start = end - self.overlap
            chunk_index += 1

            # Avoid infinite loop
            if start <= 0:
                start = end

        logger.info(
            f"Document chunked into {len(chunks)} chunks",
            extra={"original_length": len(text), "chunk_count": len(chunks)}
        )

        return chunks

    def chunk_by_separator(
        self,
        text: str,
        separator: str = "\n\n",
        max_chunks: Optional[int] = None
    ) -> List[DocumentChunk]:
        """
        Chunk document by separator (paragraphs, sections, etc.).

        Args:
            text: Document text
            separator: Separator string
            max_chunks: Maximum chunks to create

        Returns:
            List of DocumentChunk objects
        """
        segments = text.split(separator)

        chunks = []
        current_chunk = ""
        start_pos = 0
        chunk_index = 0

        for i, segment in enumerate(segments):
            # Check if adding segment would exceed chunk size
            if len(current_chunk) + len(segment) + len(separator) <= self.chunk_size:
                # Add to current chunk
                if current_chunk:
                    current_chunk += separator + segment
                else:
                    current_chunk = segment
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunk = DocumentChunk(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        start_pos=start_pos,
                        end_pos=start_pos + len(current_chunk)
                    )
                    chunks.append(chunk)

                    start_pos += len(current_chunk) + len(separator)
                    chunk_index += 1

                    # Check max chunks
                    if max_chunks and len(chunks) >= max_chunks:
                        break

                # Start new chunk
                current_chunk = segment

        # Add final chunk
        if current_chunk and (not max_chunks or len(chunks) < max_chunks):
            chunk = DocumentChunk(
                text=current_chunk,
                chunk_index=chunk_index,
                start_pos=start_pos,
                end_pos=start_pos + len(current_chunk)
            )
            chunks.append(chunk)

        logger.info(
            f"Document chunked into {len(chunks)} chunks by separator",
            extra={"separator": separator, "chunk_count": len(chunks)}
        )

        return chunks
