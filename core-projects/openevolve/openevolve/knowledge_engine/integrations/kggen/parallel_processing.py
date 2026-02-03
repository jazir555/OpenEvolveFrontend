"""
Parallel Processing Module for KG-Gen Pipeline

This module provides functionality for processing document chunks in parallel
to improve extraction performance.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Any, Callable, Optional, Awaitable
from dataclasses import dataclass
import concurrent.futures


logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Represents the result of a processing operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    chunk_id: Optional[str] = None
    processing_time_ms: float = 0.0


class ParallelChunkProcessor:
    """
    Parallel processing utility for handling document chunks concurrently.
    
    Manages a pool of workers to process chunks in parallel.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the parallel chunk processor.
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        
        logger.info({
            "msg": "ParallelChunkProcessor initialized",
            "max_workers": max_workers,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def process_chunks_parallel(
        self,
        chunks: List[Any],
        processing_fn: Callable[[Any], Awaitable[Any]],
        correlation_id: Optional[str] = None
    ) -> List[ProcessingResult]:
        """
        Process chunks in parallel using the provided processing function.
        
        Args:
            chunks: List of chunks to process
            processing_fn: Async function to process each chunk
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of ProcessingResult objects
        """
        correlation_id = correlation_id or f"parallel_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting parallel chunk processing",
            "chunk_count": len(chunks),
            "max_workers": self.max_workers,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        # Create a semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(item, idx):
            async with semaphore:
                try:
                    start_chunk = datetime.now(timezone.utc)
                    result_data = await processing_fn(item)
                    processing_time = (datetime.now(timezone.utc) - start_chunk).total_seconds() * 1000
                    
                    return ProcessingResult(
                        success=True,
                        data=result_data,
                        chunk_id=getattr(item, 'chunk_id', f'chunk_{idx}'),
                        processing_time_ms=processing_time
                    )
                except Exception as e:
                    processing_time = (datetime.now(timezone.utc) - start_chunk).total_seconds() * 1000
                    
                    logger.error({
                        "msg": "Chunk processing failed",
                        "chunk_id": getattr(item, 'chunk_id', f'chunk_{idx}'),
                        "error": str(e),
                        "correlation_id": correlation_id
                    })
                    
                    return ProcessingResult(
                        success=False,
                        error=str(e),
                        chunk_id=getattr(item, 'chunk_id', f'chunk_{idx}'),
                        processing_time_ms=processing_time
                    )
        
        # Create tasks for all chunks
        tasks = [process_with_semaphore(chunk, i) for i, chunk in enumerate(chunks)]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any unexpected exceptions in the gather operation
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error({
                    "msg": "Unexpected exception during parallel processing",
                    "chunk_index": i,
                    "error": str(result),
                    "correlation_id": correlation_id
                })
                processed_results.append(ProcessingResult(
                    success=False,
                    error=str(result),
                    chunk_id=getattr(chunks[i], 'chunk_id', f'chunk_{i}'),
                    processing_time_ms=0.0
                ))
            else:
                processed_results.append(result)
        
        total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        successful_count = sum(1 for r in processed_results if r.success)
        
        logger.info({
            "msg": "Parallel chunk processing completed",
            "correlation_id": correlation_id,
            "total_chunks": len(chunks),
            "successful_chunks": successful_count,
            "failed_chunks": len(chunks) - successful_count,
            "total_processing_time_ms": total_processing_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return processed_results
    
    async def process_batches_parallel(
        self,
        batches: List[List[Any]],
        processing_fn: Callable[[List[Any]], Awaitable[Any]],
        correlation_id: Optional[str] = None
    ) -> List[ProcessingResult]:
        """
        Process batches of chunks in parallel.
        
        Args:
            batches: List of batches (each batch is a list of chunks)
            processing_fn: Async function to process each batch
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of ProcessingResult objects
        """
        correlation_id = correlation_id or f"batch_parallel_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting parallel batch processing",
            "batch_count": len(batches),
            "max_workers": self.max_workers,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        # Create a semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_batch_with_semaphore(batch, idx):
            async with semaphore:
                try:
                    start_batch = datetime.now(timezone.utc)
                    result_data = await processing_fn(batch)
                    processing_time = (datetime.now(timezone.utc) - start_batch).total_seconds() * 1000
                    
                    return ProcessingResult(
                        success=True,
                        data=result_data,
                        chunk_id=f"batch_{idx}",
                        processing_time_ms=processing_time
                    )
                except Exception as e:
                    processing_time = (datetime.now(timezone.utc) - start_batch).total_seconds() * 1000
                    
                    logger.error({
                        "msg": "Batch processing failed",
                        "batch_id": f"batch_{idx}",
                        "error": str(e),
                        "correlation_id": correlation_id
                    })
                    
                    return ProcessingResult(
                        success=False,
                        error=str(e),
                        chunk_id=f"batch_{idx}",
                        processing_time_ms=processing_time
                    )
        
        # Create tasks for all batches
        tasks = [process_batch_with_semaphore(batch, i) for i, batch in enumerate(batches)]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any unexpected exceptions in the gather operation
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error({
                    "msg": "Unexpected exception during batch parallel processing",
                    "batch_index": i,
                    "error": str(result),
                    "correlation_id": correlation_id
                })
                processed_results.append(ProcessingResult(
                    success=False,
                    error=str(result),
                    chunk_id=f"batch_{i}",
                    processing_time_ms=0.0
                ))
            else:
                processed_results.append(result)
        
        total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        successful_count = sum(1 for r in processed_results if r.success)
        
        logger.info({
            "msg": "Parallel batch processing completed",
            "correlation_id": correlation_id,
            "total_batches": len(batches),
            "successful_batches": successful_count,
            "failed_batches": len(batches) - successful_count,
            "total_processing_time_ms": total_processing_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return processed_results
    
    async def map_reduce_parallel(
        self,
        items: List[Any],
        map_fn: Callable[[Any], Awaitable[Any]],
        reduce_fn: Callable[[List[Any]], Any],
        correlation_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Perform a map-reduce operation in parallel.
        
        Args:
            items: List of items to process
            map_fn: Async function to transform each item
            reduce_fn: Function to combine results
            correlation_id: Correlation ID for tracking
            
        Returns:
            ProcessingResult with reduced data
        """
        correlation_id = correlation_id or f"mapreduce_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting parallel map-reduce operation",
            "item_count": len(items),
            "max_workers": self.max_workers,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # First, process all items in parallel using the map function
            map_results = await self.process_chunks_parallel(
                chunks=items,
                processing_fn=map_fn,
                correlation_id=f"{correlation_id}_map"
            )
            
            # Check if all mappings were successful
            successful_maps = [r.data for r in map_results if r.success]
            failed_count = len(map_results) - len(successful_maps)
            
            if failed_count > 0:
                logger.warning({
                    "msg": "Some map operations failed",
                    "failed_count": failed_count,
                    "successful_count": len(successful_maps),
                    "correlation_id": correlation_id
                })
            
            # Apply the reduce function to combine results
            reduced_result = reduce_fn(successful_maps)
            
            total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Parallel map-reduce completed",
                "correlation_id": correlation_id,
                "item_count": len(items),
                "successful_mappings": len(successful_maps),
                "failed_mappings": failed_count,
                "total_processing_time_ms": total_processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return ProcessingResult(
                success=True,
                data=reduced_result,
                processing_time_ms=total_processing_time
            )
            
        except Exception as e:
            total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Parallel map-reduce failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "total_processing_time_ms": total_processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=total_processing_time
            )
    
    def get_worker_stats(self) -> dict:
        """
        Get statistics about worker utilization.
        
        Returns:
            Dictionary with worker statistics
        """
        return {
            "max_workers": self.max_workers,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }