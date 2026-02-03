"""
Parallel Processing for KG-Gen Pipeline

This module provides parallel chunk processing capabilities using
ThreadPoolExecutor with progress tracking and error handling.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Callable, Optional, Any, Dict
from dataclasses import dataclass

from .kggen_chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """
    Result of processing a single chunk.
    """
    chunk_id: int
    success: bool
    result: Any
    error: Optional[str] = None
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'success': self.success,
            'error': self.error,
            'processing_time': self.processing_time,
            'timestamp': datetime.now().isoformat()
        }


@dataclass
class BatchProgress:
    """
    Progress information for batch processing.
    """
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    start_time: datetime
    current_chunk_id: Optional[int] = None

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.completed_chunks / self.total_chunks) * 100

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_chunks': self.total_chunks,
            'completed_chunks': self.completed_chunks,
            'failed_chunks': self.failed_chunks,
            'completion_percentage': self.completion_percentage,
            'elapsed_time': self.elapsed_time,
            'current_chunk_id': self.current_chunk_id
        }


class ParallelChunkProcessor:
    """
    Process multiple chunks in parallel using ThreadPoolExecutor.
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize parallel processor.

        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"ParallelChunkProcessor initialized with {max_workers} workers")

    async def process_chunks_parallel(
        self,
        chunks: List[Chunk],
        processor_func: Callable,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Process chunks in parallel and aggregate results.

        Args:
            chunks: List of chunks to process
            processor_func: Function to process each chunk (can be async or sync)
            timeout: Optional timeout for each chunk in seconds

        Returns:
            List of results in same order as input chunks
        """
        if not chunks:
            logger.warning("No chunks to process")
            return []

        logger.info(f"Processing {len(chunks)} chunks with {self.max_workers} workers")
        start_time = datetime.now()

        # Process chunks
        results = [None] * len(chunks)
        futures = {}

        # Submit all tasks
        for i, chunk in enumerate(chunks):
            future = self.executor.submit(
                self._process_single_chunk,
                chunk,
                processor_func,
                timeout
            )
            futures[future] = i

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures.keys()):
            chunk_idx = futures[future]
            try:
                result = future.result()
                results[chunk_idx] = result
                completed += 1

                if completed % 10 == 0 or completed == len(chunks):
                    logger.info(f"Completed {completed}/{len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {e}")
                results[chunk_idx] = None

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Completed processing {len(chunks)} chunks in {elapsed:.2f}s "
            f"({elapsed/len(chunks):.2f}s per chunk)"
        )

        return results

    def _process_single_chunk(
        self,
        chunk: Chunk,
        processor_func: Callable,
        timeout: Optional[float]
    ) -> Any:
        """
        Process a single chunk.

        Args:
            chunk: Chunk to process
            processor_func: Processing function
            timeout: Optional timeout

        Returns:
            Processing result
        """
        import time

        start_time = time.time()

        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(processor_func):
                # Run async function in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        asyncio.wait_for(
                            processor_func(chunk),
                            timeout=timeout
                        )
                    )
                finally:
                    loop.close()
            else:
                # Run sync function directly
                if timeout:
                    import signal

                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Processing timeout after {timeout}s")

                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))
                    try:
                        result = processor_func(chunk)
                    finally:
                        signal.alarm(0)
                else:
                    result = processor_func(chunk)

            processing_time = time.time() - start_time
            logger.debug(
                f"Chunk {chunk.chunk_id} processed in {processing_time:.2f}s"
            )

            return result

        except TimeoutError as e:
            logger.error(f"Chunk {chunk.chunk_id} timed out: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
            raise

    async def process_with_progress(
        self,
        chunks: List[Chunk],
        processor_func: Callable,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        log_interval: float = 10.0
    ) -> List[Any]:
        """
        Process chunks with real-time progress tracking.

        Args:
            chunks: List of chunks to process
            processor_func: Function to process each chunk
            progress_callback: Optional callback for progress updates
            log_interval: Log progress every N seconds

        Returns:
            List of results
        """
        if not chunks:
            return []

        logger.info(f"Processing {len(chunks)} chunks with progress tracking")

        progress = BatchProgress(
            total_chunks=len(chunks),
            completed_chunks=0,
            failed_chunks=0,
            start_time=datetime.now()
        )

        results = [None] * len(chunks)
        futures = {}

        # Submit all tasks
        for i, chunk in enumerate(chunks):
            future = self.executor.submit(
                self._process_single_chunk,
                chunk,
                processor_func,
                timeout=None
            )
            futures[future] = (i, chunk.chunk_id)

        # Track last log time
        last_log_time = datetime.now()

        # Collect results
        for future in as_completed(futures.keys()):
            chunk_idx, chunk_id = futures[future]

            try:
                result = future.result()
                results[chunk_idx] = result
                progress.completed_chunks += 1
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {e}")
                results[chunk_idx] = None
                progress.failed_chunks += 1

            progress.current_chunk_id = chunk_id

            # Log progress
            elapsed = (datetime.now() - last_log_time).total_seconds()
            if elapsed >= log_interval or progress.completed_chunks == progress.total_chunks:
                logger.info(
                    f"Progress: {progress.completed_chunks}/{progress.total_chunks} "
                    f"({progress.completion_percentage:.1f}%) - "
                    f"{progress.failed_chunks} failed"
                )

                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback(progress)
                    except Exception as e:
                        logger.error(f"Error in progress callback: {e}")

                last_log_time = datetime.now()

        logger.info(
            f"Processing complete: {progress.completed_chunks} succeeded, "
            f"{progress.failed_chunks} failed in {progress.elapsed_time:.2f}s"
        )

        return results

    async def process_batches(
        self,
        chunks: List[Chunk],
        processor_func: Callable,
        batch_size: int = 10
    ) -> List[Any]:
        """
        Process chunks in batches to control memory usage.

        Args:
            chunks: List of chunks to process
            processor_func: Function to process each chunk
            batch_size: Number of chunks to process in each batch

        Returns:
            List of results
        """
        if not chunks:
            return []

        logger.info(
            f"Processing {len(chunks)} chunks in batches of {batch_size}"
        )

        results = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            batch = chunks[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_idx + 1}/{total_batches} "
                f"({len(batch)} chunks)"
            )

            batch_results = await self.process_chunks_parallel(
                batch,
                processor_func
            )

            results.extend(batch_results)

        return results

    async def process_with_retry(
        self,
        chunks: List[Chunk],
        processor_func: Callable,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> List[Any]:
        """
        Process chunks with automatic retry on failure.

        Args:
            chunks: List of chunks to process
            processor_func: Function to process each chunk
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            List of results
        """
        logger.info(
            f"Processing {len(chunks)} chunks with retry (max {max_retries} attempts)"
        )

        results = []
        retry_counts = {}

        for i, chunk in enumerate(chunks):
            retry_count = 0
            last_error = None

            while retry_count <= max_retries:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._process_single_chunk,
                        chunk,
                        processor_func,
                        timeout=None
                    )

                    results.append(result)
                    break

                except Exception as e:
                    last_error = e
                    retry_count += 1

                    if retry_count <= max_retries:
                        logger.warning(
                            f"Chunk {chunk.chunk_id} failed (attempt {retry_count}/{max_retries}): {e}. "
                            f"Retrying in {retry_delay}s..."
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Chunk {chunk.chunk_id} failed after {max_retries} retries: {e}"
                        )
                        results.append(None)
                        retry_counts[chunk.chunk_id] = retry_count

        if retry_counts:
            logger.warning(
                f"{len(retry_counts)} chunks required retries: "
                f"{sum(retry_counts.values())} total retry attempts"
            )

        return results

    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)
        logger.info("ParallelChunkProcessor shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)
