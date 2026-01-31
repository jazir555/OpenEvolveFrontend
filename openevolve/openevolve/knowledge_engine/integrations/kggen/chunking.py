"""
Document Chunking Module for KG-Gen Pipeline

This module provides functionality for splitting large documents into smaller chunks
for processing, with support for overlap and various splitting strategies.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: str
    metadata: dict


class DocumentChunker:
    """
    Document chunking utility for splitting large documents into manageable pieces.
    
    Supports various chunking strategies and overlap handling.
    """
    
    def __init__(
        self,
        chunk_size: int = 5000,
        overlap: int = 200,
        split_on: str = "\n\n",
        min_chunk_size: int = 100
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            split_on: Character sequence to split on (if applicable)
            min_chunk_size: Minimum size for a valid chunk
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.split_on = split_on
        self.min_chunk_size = min_chunk_size
        
        logger.info({
            "msg": "DocumentChunker initialized",
            "chunk_size": chunk_size,
            "overlap": overlap,
            "split_on": split_on,
            "min_chunk_size": min_chunk_size,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def chunk_document(self, document: str) -> List[DocumentChunk]:
        """
        Split a document into chunks.
        
        Args:
            document: The document text to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting document chunking",
            "document_length": len(document),
            "chunk_size": self.chunk_size,
            "timestamp": start_time.isoformat()
        })
        
        chunks = []
        doc_len = len(document)
        start_idx = 0
        chunk_id_counter = 0
        
        while start_idx < doc_len:
            # Determine the end position for this chunk
            end_idx = start_idx + self.chunk_size
            
            # If we're near the end, adjust to avoid going past the document
            if end_idx >= doc_len:
                end_idx = doc_len
            else:
                # Try to find a good breaking point near the end of the chunk
                # Look for the specified split_on sequence
                search_start = end_idx - self.overlap if self.overlap > 0 else end_idx
                search_end = min(end_idx + 100, doc_len)  # Look ahead a bit to find good break
                
                # Find the last occurrence of split_on within the range
                if self.split_on:
                    break_point = -1
                    for i in range(search_end, search_start, -1):
                        if document[max(start_idx, i-len(self.split_on)):i].endswith(self.split_on):
                            break_point = i
                            break
                    
                    if break_point != -1:
                        end_idx = break_point + len(self.split_on)
            
            # Extract the chunk
            chunk_text = document[start_idx:end_idx]
            
            # Create chunk if it meets minimum size requirements
            if len(chunk_text) >= self.min_chunk_size:
                chunk = DocumentChunk(
                    text=chunk_text,
                    start_pos=start_idx,
                    end_pos=end_idx,
                    chunk_id=f"chunk_{chunk_id_counter}",
                    metadata={
                        "position": chunk_id_counter,
                        "original_start": start_idx,
                        "original_end": end_idx,
                        "length": len(chunk_text)
                    }
                )
                chunks.append(chunk)
                chunk_id_counter += 1
            
            # Move to the next chunk position, accounting for overlap
            if end_idx >= doc_len:
                # At the end of the document
                break
            else:
                # Move start position by chunk_size minus overlap
                start_idx = end_idx - self.overlap
                # Ensure we don't get stuck in an infinite loop
                if start_idx <= start_idx:  # Previous start_idx
                    start_idx += 1
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.info({
            "msg": "Document chunking completed",
            "chunks_created": len(chunks),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return chunks
    
    def chunk_by_sentences(self, document: str, max_sentences: int = 10) -> List[DocumentChunk]:
        """
        Split a document into chunks based on sentences.
        
        Args:
            document: The document text to chunk
            max_sentences: Maximum number of sentences per chunk
            
        Returns:
            List of DocumentChunk objects
        """
        import re
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting sentence-based chunking",
            "document_length": len(document),
            "max_sentences": max_sentences,
            "timestamp": start_time.isoformat()
        })
        
        # Split document into sentences
        # This is a simplified sentence splitter - in practice, you'd use a more robust NLP library
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, document)
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_start = 0
        chunk_id_counter = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)
            
            # If we've reached the max sentences or this is the last sentence
            if len(current_chunk_sentences) >= max_sentences or i == len(sentences) - 1:
                # Create a chunk from the accumulated sentences
                chunk_text = '. '.join(current_chunk_sentences) + '.'
                
                # Calculate the position in the original document
                chunk_start_pos = document.find(current_chunk_sentences[0], current_chunk_start)
                if chunk_start_pos == -1:
                    chunk_start_pos = current_chunk_start
                chunk_end_pos = chunk_start_pos + len(chunk_text)
                
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        start_pos=chunk_start_pos,
                        end_pos=chunk_end_pos,
                        chunk_id=f"sentence_chunk_{chunk_id_counter}",
                        metadata={
                            "position": chunk_id_counter,
                            "sentence_count": len(current_chunk_sentences),
                            "original_start": chunk_start_pos,
                            "original_end": chunk_end_pos,
                            "length": len(chunk_text)
                        }
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
                
                # Reset for the next chunk
                current_chunk_sentences = []
                current_chunk_start = chunk_end_pos
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.info({
            "msg": "Sentence-based chunking completed",
            "chunks_created": len(chunks),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return chunks
    
    def chunk_by_paragraphs(self, document: str, max_paragraphs: int = 3) -> List[DocumentChunk]:
        """
        Split a document into chunks based on paragraphs.
        
        Args:
            document: The document text to chunk
            max_paragraphs: Maximum number of paragraphs per chunk
            
        Returns:
            List of DocumentChunk objects
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting paragraph-based chunking",
            "document_length": len(document),
            "max_paragraphs": max_paragraphs,
            "timestamp": start_time.isoformat()
        })
        
        # Split document into paragraphs
        paragraphs = document.split('\n\n')
        
        chunks = []
        current_chunk_paragraphs = []
        current_chunk_start = 0
        chunk_id_counter = 0
        
        for i, paragraph in enumerate(paragraphs):
            current_chunk_paragraphs.append(paragraph)
            
            # If we've reached the max paragraphs or this is the last paragraph
            if len(current_chunk_paragraphs) >= max_paragraphs or i == len(paragraphs) - 1:
                # Create a chunk from the accumulated paragraphs
                chunk_text = '\n\n'.join(current_chunk_paragraphs)
                
                # Calculate the position in the original document
                chunk_start_pos = document.find(current_chunk_paragraphs[0], current_chunk_start)
                if chunk_start_pos == -1:
                    chunk_start_pos = current_chunk_start
                chunk_end_pos = chunk_start_pos + len(chunk_text)
                
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        start_pos=chunk_start_pos,
                        end_pos=chunk_end_pos,
                        chunk_id=f"paragraph_chunk_{chunk_id_counter}",
                        metadata={
                            "position": chunk_id_counter,
                            "paragraph_count": len(current_chunk_paragraphs),
                            "original_start": chunk_start_pos,
                            "original_end": chunk_end_pos,
                            "length": len(chunk_text)
                        }
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
                
                # Reset for the next chunk
                current_chunk_paragraphs = []
                current_chunk_start = chunk_end_pos
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.info({
            "msg": "Paragraph-based chunking completed",
            "chunks_created": len(chunks),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return chunks
    
    def merge_chunks(self, chunks: List[DocumentChunk], max_chunk_size: int = 10000) -> List[DocumentChunk]:
        """
        Merge small chunks together to reach a minimum size.
        
        Args:
            chunks: List of DocumentChunk objects to merge
            max_chunk_size: Maximum size for merged chunks
            
        Returns:
            List of potentially merged DocumentChunk objects
        """
        if not chunks:
            return []
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting chunk merging",
            "initial_chunks": len(chunks),
            "max_chunk_size": max_chunk_size,
            "timestamp": start_time.isoformat()
        })
        
        merged_chunks = []
        current_merged_text = ""
        current_metadata = {"merged_chunks": []}
        chunk_id_counter = 0
        
        for chunk in chunks:
            # Check if adding this chunk would exceed the max size
            if (len(current_merged_text) + len(chunk.text) > max_chunk_size and 
                current_merged_text):  # If we already have content
                # Save the current merged chunk
                merged_chunk = DocumentChunk(
                    text=current_merged_text.strip(),
                    start_pos=min(current_metadata.get("original_start", [])),
                    end_pos=max(current_metadata.get("original_end", [])),
                    chunk_id=f"merged_chunk_{chunk_id_counter}",
                    metadata=current_metadata
                )
                merged_chunks.append(merged_chunk)
                
                # Start a new merged chunk with the current chunk
                current_merged_text = chunk.text
                current_metadata = {
                    "merged_chunks": [chunk.chunk_id],
                    "original_start": [chunk.start_pos],
                    "original_end": [chunk.end_pos],
                    "length": len(chunk.text)
                }
                chunk_id_counter += 1
            else:
                # Add to the current merged chunk
                if current_merged_text:
                    current_merged_text += "\n\n" + chunk.text
                else:
                    current_merged_text = chunk.text
                
                # Update metadata
                current_metadata["merged_chunks"].append(chunk.chunk_id)
                current_metadata.setdefault("original_start", []).append(chunk.start_pos)
                current_metadata.setdefault("original_end", []).append(chunk.end_pos)
                current_metadata["length"] = len(current_merged_text)
        
        # Add the final merged chunk if it has content
        if current_merged_text:
            merged_chunk = DocumentChunk(
                text=current_merged_text.strip(),
                start_pos=min(current_metadata.get("original_start", [])),
                end_pos=max(current_metadata.get("original_end", [])),
                chunk_id=f"merged_chunk_{chunk_id_counter}",
                metadata=current_metadata
            )
            merged_chunks.append(merged_chunk)
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.info({
            "msg": "Chunk merging completed",
            "initial_chunks": len(chunks),
            "final_chunks": len(merged_chunks),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return merged_chunks