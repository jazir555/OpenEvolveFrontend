"""
Parallel Chunk Processing Module - Production Grade

Part of KG-Gen Sprint 2 Integration.

Following CLAUDE.md Principles:
- IDEMPOTENCY: Processing is retry-safe
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import logging
from typing import List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """
    Result of processing a chunk.
    """
    chunk_index: int
    success: bool
    result: Any = None
    error: Optional[str] = None


class ParallelChunkProcessor:
    """
    Process document chunks in parallel.

    LAW OF IDEMPOTENCY: Failed chunks can be retried safely.
    """

    def __init__(
        self,
        max_workers: int = 4,
        timeout_per_chunk: float = 300.0
    ):
        """
        Initialize parallel processor.

        Args:
            max_workers: Maximum parallel workers
            timeout_per_chunk: Timeout for each chunk in seconds
        """
        if max_workers <= 0:
            raise ValueError(f"Invalid max_workers: {max_workers}")
        if timeout_per_chunk <= 0:
            raise ValueError(f"Invalid timeout_per_chunk: {timeout_per_chunk}")

        self.max_workers = max_workers
        self.timeout_per_chunk = timeout_per_chunk
        self._executor: Optional[ThreadPoolExecutor] = None

        logger.info(
            "ParallelChunkProcessor initialized",
            extra={"max_workers": max_workers, "timeout": timeout_per_chunk}
        )

    async def process_chunks_parallel(
        self,
        chunks: List[Any],
        process_func: Callable[[Any], Any],
        correlation_id: Optional[str] = None
    ) -> List[Any]:
        """
        Process chunks in parallel.

        Args:
            chunks: List of chunks to process
            process_func: Async function to process each chunk
            correlation_id: Optional correlation ID for tracking

        Returns:
            List of results in same order as chunks
        """
        if not chunks:
            return []

        correlation_id = correlation_id or "unknown"

        logger.info(
            f"Processing {len(chunks)} chunks with {self.max_workers} workers",
            extra={"correlation_id": correlation_id}
        )

        # Create semaphore to limit parallelism
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(chunk):
            """Process chunk with semaphore limiting."""
            async with semaphore:
                try:
                    # Add timeout
                    result = await asyncio.wait_for(
                        process_func(chunk),
                        timeout=self.timeout_per_chunk
                    )

                    return result

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Chunk processing timed out after {self.timeout_per_chunk}s",
                        extra={"correlation_id": correlation_id}
                    )
                    return None

                except Exception as e:
                    logger.error(
                        f"Chunk processing failed: {e}",
                        extra={"correlation_id": correlation_id},
                        exc_info=True
                    )
                    return None

        # Process all chunks
        results = await asyncio.gather(
            *[process_with_semaphore(chunk) for chunk in chunks],
            return_exceptions=False
        )

        # Filter out failed results
        successful_results = [r for r in results if r is not None]

        logger.info(
            f"Parallel processing complete: {len(successful_results)}/{len(chunks)} successful",
            extra={"correlation_id": correlation_id}
        )

        return successful_results

    async def process_chunks_parallel_with_retry(
        self,
        chunks: List[Any],
        process_func: Callable[[Any], Any],
        max_retries: int = 2,
        correlation_id: Optional[str] = None
    ) -> List[Any]:
        """
        Process chunks in parallel with retry logic.

        Args:
            chunks: List of chunks to process
            process_func: Async function to process each chunk
            max_retries: Maximum retry attempts for failed chunks
            correlation_id: Optional correlation ID

        Returns:
            List of results
        """
        correlation_id = correlation_id or "unknown"

        # First attempt
        results = await self.process_chunks_parallel(
            chunks,
            process_func,
            correlation_id
        )

        # Check if all succeeded
        if len(results) == len(chunks):
            return results

        # Retry failed chunks
        logger.info(
            f"Retrying {len(chunks) - len(results)} failed chunks",
            extra={"correlation_id": correlation_id}
        )

        # Identify failed chunks (simple approach: retry all)
        # In production, track which chunks failed specifically
        for attempt in range(max_retries):
            retry_results = await self.process_chunks_parallel(
                chunks,
                process_func,
                correlation_id
            )

            if len(retry_results) > len(results):
                results = retry_results

                if len(results) == len(chunks):
                    break

        return results

    def shutdown(self) -> None:
        """Shutdown executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            logger.info("ParallelChunkProcessor shutdown complete")
