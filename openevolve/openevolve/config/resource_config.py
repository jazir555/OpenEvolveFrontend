"""
Resource-Aware Configuration

This module provides automatic resource detection and configuration adjustment
based on available system resources (CPU, memory, GPU).
"""

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

from ..unified.config import UnifiedEvolutionConfig


logger = logging.getLogger(__name__)


@dataclass
class ResourceInfo:
    """Information about available system resources"""
    cpu_count: int
    cpu_usage: float  # Percentage
    memory_total_gb: float
    memory_available_gb: float
    memory_usage_percent: float
    gpu_available: bool
    gpu_count: int
    gpu_memory_total: float  # Total GPU memory in GB
    gpu_memory_available: float  # Available GPU memory in GB
    disk_space_gb: float


@dataclass
class ResourceLimits:
    """User-specified resource limits"""
    max_memory_mb: Optional[int] = None
    max_cpu_cores: Optional[float] = None
    max_time_seconds: Optional[int] = None
    require_gpu: bool = False


class ResourceAwareConfigurator:
    """
    Detect resources and auto-adjust configuration

    Features:
    - Automatic CPU, memory, GPU detection
    - Configuration adjustment based on resources
    - User-specified limits enforcement
    - Dynamic scaling during execution
    - Resource monitoring
    """

    def __init__(self):
        """Initialize resource-aware configurator"""
        self.detected_resources: Optional[ResourceInfo] = None
        self.resource_history: list = []

    def detect_resources(self) -> ResourceInfo:
        """
        Detect available system resources

        Returns:
            ResourceInfo with detected resources
        """
        try:
            import psutil

            # CPU detection
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=0.1)

            # Memory detection
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024 ** 3)
            memory_available_gb = memory.available / (1024 ** 3)
            memory_usage_percent = memory.percent

            # Disk detection
            disk = psutil.disk_usage('.')
            disk_space_gb = disk.free / (1024 ** 3)

            # GPU detection
            gpu_available, gpu_count, gpu_mem_total, gpu_mem_avail = self._detect_gpu()

            resources = ResourceInfo(
                cpu_count=cpu_count or 4,
                cpu_usage=cpu_usage,
                memory_total_gb=memory_total_gb,
                memory_available_gb=memory_available_gb,
                memory_usage_percent=memory_usage_percent,
                gpu_available=gpu_available,
                gpu_count=gpu_count,
                gpu_memory_total=gpu_mem_total,
                gpu_memory_available=gpu_mem_avail,
                disk_space_gb=disk_space_gb
            )

            self.detected_resources = resources
            self.resource_history.append({
                "timestamp": None,  # Would add datetime
                "resources": resources
            })

            logger.info(
                f"Detected resources: {cpu_count} CPUs, "
                f"{memory_available_gb:.1f}GB RAM available, "
                f"{gpu_count} GPUs"
            )

            return resources

        except ImportError:
            logger.warning("psutil not available, using minimal resource detection")
            return self._minimal_resource_detection()

    def _detect_gpu(self) -> tuple:
        """
        Detect GPU availability and memory

        Returns:
            Tuple of (available, count, total_memory_gb, available_memory_gb)
        """
        gpu_available = False
        gpu_count = 0
        gpu_mem_total = 0.0
        gpu_mem_avail = 0.0

        # Try PyTorch
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                # Get memory for first GPU
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                gpu_mem_avail = (torch.cuda.get_device_properties(0).total_memory -
                                torch.cuda.memory_allocated(0)) / (1024 ** 3)
                return gpu_available, gpu_count, gpu_mem_total, gpu_mem_avail
        except ImportError:
            pass

        # Try TensorFlow
        try:
            import tensorflow as tf
            gpu_devices = tf.config.list_physical_devices('GPU')
            gpu_available = len(gpu_devices) > 0
            gpu_count = len(gpu_devices)
            # TensorFlow doesn't easily expose memory info without allocation
            if gpu_available:
                gpu_mem_total = 8.0  # Conservative estimate
                gpu_mem_avail = 8.0
            return gpu_available, gpu_count, gpu_mem_total, gpu_mem_avail
        except ImportError:
            pass

        # Try CUDA directly
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            if gpu_count > 0:
                gpu_available = True
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_total = mem_info.total / (1024 ** 3)
                gpu_mem_avail = mem_info.free / (1024 ** 3)
            return gpu_available, gpu_count, gpu_mem_total, gpu_mem_avail
        except Exception:
            pass

        return False, 0, 0.0, 0.0

    def _minimal_resource_detection(self) -> ResourceInfo:
        """Fallback resource detection without psutil"""
        import os
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        gpu_available, gpu_count, _, _ = self._detect_gpu()

        return ResourceInfo(
            cpu_count=cpu_count,
            cpu_usage=0.0,
            memory_total_gb=16.0,  # Assume 16GB
            memory_available_gb=8.0,
            memory_usage_percent=50.0,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_memory_total=8.0 if gpu_available else 0.0,
            gpu_memory_available=8.0 if gpu_available else 0.0,
            disk_space_gb=100.0  # Assume 100GB
        )

    def adjust_config_for_resources(
        self,
        base_config: UnifiedEvolutionConfig,
        limits: Optional[ResourceLimits] = None
    ) -> UnifiedEvolutionConfig:
        """
        Adjust configuration based on available resources

        Args:
            base_config: Base configuration
            limits: Optional resource limits (user-specified)

        Returns:
            Resource-adjusted configuration
        """
        if self.detected_resources is None:
            self.detect_resources()

        import copy
        config = copy.deepcopy(base_config)
        resources = self.detected_resources

        # Adjust for CPU
        config = self._adjust_for_cpu(config, resources, limits)

        # Adjust for memory
        config = self._adjust_for_memory(config, resources, limits)

        # Adjust for GPU
        config = self._adjust_for_gpu(config, resources, limits)

        # Apply user limits if specified
        if limits:
            config = self._apply_user_limits(config, limits)

        logger.info(f"Adjusted configuration for detected resources")
        return config

    def _adjust_for_cpu(
        self,
        config: UnifiedEvolutionConfig,
        resources: ResourceInfo,
        limits: Optional[ResourceLimits]
    ) -> UnifiedEvolutionConfig:
        """Adjust configuration based on CPU resources"""
        # Leave one core free for system
        available_cores = max(1, resources.cpu_count - 1)

        # Adjust concurrency
        config.common.concurrency = min(
            config.common.concurrency,
            available_cores
        )

        # Adjust parallel evaluations
        config.evaluator.parallel_evaluations = min(
            config.evaluator.parallel_evaluations,
            available_cores
        )

        # Adjust database parallelism
        config.database.num_islands = min(
            config.database.num_islands,
            available_cores
        )

        return config

    def _adjust_for_memory(
        self,
        config: UnifiedEvolutionConfig,
        resources: ResourceInfo,
        limits: Optional[ResourceLimits]
    ) -> UnifiedEvolutionConfig:
        """Adjust configuration based on memory resources"""
        # Estimate memory per individual
        memory_per_individual_mb = self._estimate_memory_per_individual(config)

        # Calculate max population based on available memory
        available_memory_mb = resources.memory_available_gb * 1024
        safe_memory_mb = available_memory_mb * 0.8  # Use 80% max

        max_population = int(safe_memory_mb / memory_per_individual_mb)

        # Adjust population size
        config.database.population_size = min(
            config.database.population_size,
            max_population
        )

        # Adjust archive size
        config.database.elite_archive_size = min(
            config.database.elite_archive_size,
            max_population // 10
        )

        # Adjust concurrency based on memory
        # Ensure each concurrent worker has enough memory
        memory_per_worker = safe_memory_mb / config.common.concurrency
        if memory_per_worker < 1000:  # Less than 1GB per worker
            # Reduce concurrency
            config.common.concurrency = max(
                1,
                int(safe_memory_mb / 1000)
            )

        return config

    def _adjust_for_gpu(
        self,
        config: UnifiedEvolutionConfig,
        resources: ResourceInfo,
        limits: Optional[ResourceLimits]
    ) -> UnifiedEvolutionConfig:
        """Adjust configuration based on GPU resources"""
        if not resources.gpu_available:
            # GPU not available, ensure CPU-only settings
            logger.info("No GPU detected, using CPU configuration")
            # Could disable GPU-specific features here
            return config

        # GPU available - optimize for GPU usage
        logger.info(f"GPU detected: {resources.gpu_count} GPUs")

        # Increase batch sizes for GPU
        if hasattr(config.evaluator, 'batch_size'):
            # Use larger batches with GPU
            config.evaluator.batch_size *= resources.gpu_count

        # Adjust concurrency for GPU workloads
        # GPUs work better with higher concurrency
        config.common.concurrency = min(
            config.common.concurrency * resources.gpu_count,
            resources.cpu_count
        )

        return config

    def _apply_user_limits(
        self,
        config: UnifiedEvolutionConfig,
        limits: ResourceLimits
    ) -> UnifiedEvolutionConfig:
        """Apply user-specified resource limits"""
        if limits.max_memory_mb:
            # Adjust population to fit in memory limit
            memory_per_individual = self._estimate_memory_per_individual(config)
            max_population = (limits.max_memory_mb * 0.8) / memory_per_individual

            config.database.population_size = min(
                config.database.population_size,
                int(max_population)
            )

        if limits.max_cpu_cores:
            config.common.concurrency = min(
                config.common.concurrency,
                int(limits.max_cpu_cores)
            )
            config.evaluator.parallel_evaluations = min(
                config.evaluator.parallel_evaluations,
                int(limits.max_cpu_cores)
            )

        if limits.max_time_seconds:
            config.common.max_iterations = self._estimate_iterations_for_time(
                config,
                limits.max_time_seconds
            )

        if limits.require_gpu and not self.detected_resources.gpu_available:
            logger.warning("GPU required but not available. Configuration may fail.")

        return config

    def _estimate_memory_per_individual(
        self,
        config: UnifiedEvolutionConfig
    ) -> float:
        """
        Estimate memory usage per individual in MB

        Args:
            config: Current configuration

        Returns:
            Estimated memory per individual in MB
        """
        # Base estimate
        base_mb = 10.0

        # Add for code storage
        if config.openevolve:
            code_mb = config.openevolve.max_code_length / (1024 * 1024)
            base_mb += code_mb

        # Add for artifacts
        if config.evaluator.enable_artifacts:
            artifact_mb = config.evaluator.max_artifact_storage / (1024 * 1024)
            base_mb += artifact_mb

        # Add for prompts
        if config.openevolve and config.openevolve.include_artifacts:
            base_mb += config.openevolve.max_artifact_bytes / (1024 * 1024)

        return base_mb

    def _estimate_iterations_for_time(
        self,
        config: UnifiedEvolutionConfig,
        max_time_seconds: int
    ) -> int:
        """
        Estimate max iterations that fit in time budget

        Args:
            config: Current configuration
            max_time_seconds: Maximum time available

        Returns:
            Estimated max iterations
        """
        # Conservative estimate: 1 iteration per population evaluation
        # Adjust based on concurrency and evaluation time

        # Assume average evaluation time of 5 seconds
        avg_eval_time = 5.0

        # With concurrency, effective time per iteration
        effective_time_per_iter = avg_eval_time / max(config.common.concurrency, 1)

        # Account for population size
        time_per_full_iteration = effective_time_per_iter * config.database.population_size

        # Estimate iterations (leave 20% buffer)
        estimated_iterations = int((max_time_seconds * 0.8) / time_per_full_iteration)

        # Ensure at least 1 iteration
        return max(1, estimated_iterations)

    def get_resource_recommendations(
        self,
        config: UnifiedEvolutionConfig
    ) -> Dict[str, Any]:
        """
        Get recommendations for optimal resource usage

        Args:
            config: Current configuration

        Returns:
            Dictionary with recommendations
        """
        if self.detected_resources is None:
            self.detect_resources()

        resources = self.detected_resources
        recommendations = {}

        # CPU recommendations
        cpu_utilization = config.common.concurrency / resources.cpu_count
        if cpu_utilization < 0.5:
            recommendations["cpu"] = (
                f"Consider increasing concurrency to {int(resources.cpu_count * 0.8)} "
                f"for better CPU utilization"
            )
        elif cpu_utilization > 1.0:
            recommendations["cpu"] = (
                f"Consider reducing concurrency to {int(resources.cpu_count * 0.8)} "
                f"to avoid CPU over-subscription"
            )

        # Memory recommendations
        estimated_memory_mb = (
            self._estimate_memory_per_individual(config) *
            config.database.population_size
        )
        available_memory_mb = resources.memory_available_gb * 1024

        if estimated_memory_mb > available_memory_mb * 0.9:
            recommendations["memory"] = (
                f"Population size may exceed available memory. "
                f"Consider reducing to {int((available_memory_mb * 0.8) / self._estimate_memory_per_individual(config))}"
            )

        # GPU recommendations
        if resources.gpu_available:
            if config.common.concurrency < resources.gpu_count * 2:
                recommendations["gpu"] = (
                    f"Consider increasing concurrency to {resources.gpu_count * 2} "
                    f"to better utilize GPU resources"
                )
        else:
            if config.evaluator.parallel_evaluations > resources.cpu_count:
                recommendations["gpu"] = (
                    f"GPU not available. Consider reducing parallel evaluations "
                    f"to {resources.cpu_count}"
                )

        return recommendations

    def monitor_resources(self) -> Dict[str, float]:
        """
        Get current resource usage

        Returns:
            Dictionary with current usage metrics
        """
        if self.detected_resources is None:
            self.detect_resources()

        # Re-detect to get current usage
        try:
            import psutil

            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('.').percent,
            }
        except ImportError:
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0,
            }
