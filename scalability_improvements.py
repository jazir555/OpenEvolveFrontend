"""
Sovereign-Grade Problem Decomposition System - Scalability Improvements
Implements distributed processing, load balancing, and resource management.
"""



import asyncio
import concurrent.futures
import threading
import multiprocessing
import queue
import time
import json
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from contextlib import contextmanager
import psutil
from collections import OrderedDict

# Optional imports for distributed processing
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None

import uuid


logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    """Represents a unit of work to be processed"""
    id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 5  # 1-10, where 10 is highest priority
    created_at: datetime = None
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ResourceMonitor:
    """Monitor system resources and performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self, interval: float = 5.0):
        """Start resource monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                args=(interval,),
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self._monitoring:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            
            # Check for alerts
            self._check_alerts(metrics)
            
            # Keep only recent metrics (last 1000 entries)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            time.sleep(interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available': memory.available,
            'memory_used': memory.used,
            'disk_percent': (disk.used / disk.total) * 100,
            'network_bytes_sent': network.bytes_sent,
            'network_bytes_recv': network.bytes_recv,
            'process_count': len(psutil.pids()),
            'active_threads': threading.active_count()
        }
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for resource usage alerts"""
        alerts = []
        
        if metrics['cpu_percent'] > 90:
            alerts.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
        
        if metrics['memory_percent'] > 90:
            alerts.append(f"High memory usage: {metrics['memory_percent']:.1f}%")
        
        if metrics['disk_percent'] > 95:
            alerts.append(f"High disk usage: {metrics['disk_percent']:.1f}%")
        
        for alert in alerts:
            logger.warning(alert)
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'alert': alert
            })
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        return self._collect_metrics()
    
    def get_historical_metrics(self, minutes: int = 10) -> List[Dict[str, Any]]:
        """Get historical metrics for the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]
    
    def get_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get alerts for the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']) > cutoff
        ]


class DistributedQueue:
    """Queue system for distributing work across multiple workers"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.queue_name = "sovereign_work_queue"
        self.result_queue = "sovereign_result_queue"
    
    def add_work_item(self, work_item: WorkItem) -> bool:
        """Add a work item to the queue"""
        try:
            # Serialize work item
            work_data = {
                'id': work_item.id,
                'task_type': work_item.task_type,
                'payload': json.dumps(work_item.payload),
                'priority': work_item.priority,
                'created_at': work_item.created_at.isoformat(),
                'status': work_item.status
            }
            
            # Add to Redis queue with priority-based scoring
            score = -(work_item.priority * 1000000) - int(work_item.created_at.timestamp())
            self.redis_client.zadd(self.queue_name, {json.dumps(work_data): score})
            
            logger.info(f"Added work item {work_item.id} to queue with priority {work_item.priority}")
            return True
        except Exception as e:
            logger.error(f"Failed to add work item to queue: {e}")
            return False
    
    def get_work_item(self, timeout: int = 5) -> Optional[WorkItem]:
        """Get a work item from the queue"""
        try:
            # Get the highest priority item (lowest score)
            result = self.redis_client.bzpopmin(self.queue_name, timeout=timeout)
            if result:
                work_data_str, score = result
                work_data = json.loads(work_data_str)
                
                return WorkItem(
                    id=work_data['id'],
                    task_type=work_data['task_type'],
                    payload=json.loads(work_data['payload']),
                    priority=work_data['priority'],
                    created_at=datetime.fromisoformat(work_data['created_at']),
                    status=work_data['status']
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get work item from queue: {e}")
            return None
    
    def complete_work_item(self, work_item: WorkItem) -> bool:
        """Complete a work item and store result"""
        try:
            result_data = {
                'id': work_item.id,
                'status': work_item.status,
                'result': json.dumps(work_item.result) if work_item.result else None,
                'error': work_item.error,
                'completed_at': datetime.now().isoformat()
            }
            
            # Add to results queue
            self.redis_client.lpush(self.result_queue, json.dumps(result_data))
            
            logger.info(f"Completed work item {work_item.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to complete work item: {e}")
            return False
    
    def get_completed_results(self) -> List[Dict[str, Any]]:
        """Get completed results from the queue"""
        try:
            results = []
            while True:
                result = self.redis_client.rpop(self.result_queue)
                if not result:
                    break
                results.append(json.loads(result))
            return results
        except Exception as e:
            logger.error(f"Failed to get completed results: {e}")
            return []


class LoadBalancer:
    """Load balancing for distributing work across multiple nodes"""
    
    def __init__(self):
        self.nodes = {}  # {node_id: node_info}
        self.node_stats = {}  # {node_id: stats}
        self._lock = threading.Lock()
    
    def register_node(self, node_id: str, address: str, capabilities: Dict[str, Any] = None):
        """Register a processing node"""
        with self._lock:
            self.nodes[node_id] = {
                'id': node_id,
                'address': address,
                'capabilities': capabilities or {},
                'last_heartbeat': datetime.now(),
                'active_tasks': 0,
                'total_tasks': 0
            }
            self.node_stats[node_id] = {
                'cpu_usage': 0,
                'memory_usage': 0,
                'tasks_completed': 0,
                'tasks_failed': 0
            }
            logger.info(f"Registered node {node_id} at {address}")
    
    def heartbeat(self, node_id: str):
        """Update node heartbeat"""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id]['last_heartbeat'] = datetime.now()
    
    def get_available_node(self, task_type: str = None) -> Optional[str]:
        """Get the best available node for a task"""
        with self._lock:
            # Filter active nodes (heartbeats within last 30 seconds)
            active_nodes = {
                node_id: info for node_id, info in self.nodes.items()
                if datetime.now() - info['last_heartbeat'] < timedelta(seconds=30)
            }
            
            if not active_nodes:
                return None
            
            # Find node with lowest active task count
            best_node = min(
                active_nodes.keys(),
                key=lambda n: active_nodes[n]['active_tasks']
            )
            
            # Increment active tasks count
            self.nodes[best_node]['active_tasks'] += 1
            return best_node
    
    def report_task_completion(self, node_id: str, success: bool):
        """Report task completion to update node stats"""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id]['active_tasks'] -= 1
                self.nodes[node_id]['total_tasks'] += 1
                
                if node_id in self.node_stats:
                    if success:
                        self.node_stats[node_id]['tasks_completed'] += 1
                    else:
                        self.node_stats[node_id]['tasks_failed'] += 1
    
    def get_node_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        with self._lock:
            return {
                'total_nodes': len(self.nodes),
                'active_nodes': len([n for n, info in self.nodes.items()
                                   if datetime.now() - info['last_heartbeat'] < timedelta(seconds=30)]),
                'nodes': dict(self.nodes),
                'stats': dict(self.node_stats)
            }


class WorkflowQueue:
    """Queue system for managing high-volume workflows"""
    
    def __init__(self, max_concurrent: int = 10, task_handlers: Optional[Dict[str, Callable]] = None):
        self.max_concurrent = max_concurrent
        self.queue = queue.PriorityQueue()
        self.active_workers = 0
        self.workers = []
        self._running = False
        self._lock = threading.Lock()
        self.task_handlers = task_handlers or {}
    
    def start(self):
        """Start the workflow queue"""
        self._running = True
        for i in range(self.max_concurrent):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started workflow queue with {self.max_concurrent} workers")
    
    def stop(self):
        """Stop the workflow queue"""
        self._running = False
        for worker in self.workers:
            worker.join()
        logger.info("Stopped workflow queue")
    
    def submit_work_item(self, work_item: WorkItem):
        """Submit a work item to the queue"""
        # Priority queue uses negative priority for highest priority first
        priority = -(work_item.priority * 10000) - int(time.time() * 1000)
        self.queue.put((priority, work_item))
        logger.info(f"Submitted work item {work_item.id} with priority {work_item.priority}")

    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type"""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def _worker_loop(self):
        """Main worker loop"""
        while self._running:
            try:
                priority, work_item = self.queue.get(timeout=1)
                
                with self._lock:
                    self.active_workers += 1
                    active = self.active_workers
                
                logger.info(f"Worker processing task {work_item.id}, {active} active workers")
                
                # Process the work item
                result = self._process_work_item(work_item)
                
                with self._lock:
                    self.active_workers -= 1
                
                # Mark task as complete
                self.queue.task_done()
                
                logger.info(f"Completed task {work_item.id}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
                continue
    
    def _process_work_item(self, work_item: WorkItem) -> Any:
        """Process a work item - override this method in subclasses"""
        try:
            handler = self.task_handlers.get(work_item.task_type)
            if handler:
                result = handler(work_item.payload)
            elif "handler" in work_item.payload and callable(work_item.payload["handler"]):
                result = work_item.payload["handler"](work_item.payload)
            else:
                result = {
                    "status": "completed",
                    "message": f"No handler registered for task type '{work_item.task_type}'"
                }
            
            work_item.status = "completed"
            work_item.result = result
            return result
        except Exception as e:
            work_item.status = "failed"
            work_item.error = str(e)
            logger.error(f"Failed to process work item {work_item.id}: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            return {
                'pending_tasks': self.queue.qsize(),
                'active_workers': self.active_workers,
                'max_concurrent': self.max_concurrent
            }


class MemoryOptimizer:
    """Optimize memory usage for large workflows and datasets"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        # Optimization: Use OrderedDict for O(1) LRU management instead of list.remove()
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
    
    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get item from memory cache"""
        with self._lock:
            # Optimization: Use dict.get() and move_to_end()
            value = self.cache.get(key)
            if value is not None:
                self.cache.move_to_end(key)
                return value
            return None
    
    def put_in_cache(self, key: str, value: Any, size_estimate: int = 1) -> bool:
        """Put item in memory cache with size check"""
        with self._lock:
            # Check if key already exists
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = value
                return True

            # Check memory usage
            if self._estimate_memory_usage() + size_estimate > self.max_memory_mb:
                # Remove LRU items until memory is available
                while (self._estimate_memory_usage() + size_estimate > self.max_memory_mb 
                       and self.cache):
                    self.cache.popitem(last=False)
            
            # Check if we have space now
            if self._estimate_memory_usage() + size_estimate <= self.max_memory_mb:
                self.cache[key] = value
                return True
            else:
                logger.warning(f"Memory cache full, could not add item: {key}")
                return False
    
    def _estimate_memory_usage(self) -> int:
        """Estimate current memory usage in MB"""
        # This is a very rough estimate - in practice, use sys.getsizeof for more accuracy
        return len(self.cache)  # Estimate 1MB per item
    
    def clear_cache(self):
        """Clear the memory cache"""
        with self._lock:
            self.cache.clear()
            self.lru_order.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'cache_size': len(self.cache),
                'max_cache_size': self.max_memory_mb,
                'lru_order': self.lru_order[-10:]  # Last 10 accessed
            }


class DistributedProcessor:
    """Processor for distributing work across multiple nodes/processes"""
    
    def __init__(self, use_multiprocessing: bool = True):
        self.use_multiprocessing = use_multiprocessing
        self._executor = None
        self._initialize_executor()
    
    def _initialize_executor(self):
        """Initialize the execution engine"""
        if self.use_multiprocessing:
            # Use process pool for CPU-intensive tasks
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=multiprocessing.cpu_count()
            )
        else:
            # Use thread pool for I/O-bound tasks
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 1) + 4)
            )
    
    def submit_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task for processing"""
        if kwargs:
            # For kwargs, we need to use a wrapper function
            def task_wrapper():
                return func(*args, **kwargs)
            return self._executor.submit(task_wrapper)
        else:
            return self._executor.submit(func, *args)
    
    def submit_batch(self, tasks: List[tuple]) -> List[concurrent.futures.Future]:
        """Submit multiple tasks for processing
        Each task is a tuple of (function, args, kwargs)
        """
        futures = []
        for func, args, kwargs in tasks:
            futures.append(self.submit_task(func, *args, **kwargs))
        return futures
    
    def wait_for_completion(self, futures: List[concurrent.futures.Future], 
                          timeout: Optional[float] = None) -> List[Any]:
        """Wait for all futures to complete and return results"""
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append(None)
        return results
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor"""
        if self._executor:
            self._executor.shutdown(wait=wait)


class ScalabilityManager:
    """Main scalability management class"""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.distributed_queue = DistributedQueue()
        self.load_balancer = LoadBalancer()
        self.task_handlers = {}
        self.workflow_queue = WorkflowQueue(task_handlers=self.task_handlers)
        self.memory_optimizer = MemoryOptimizer()
        self.distributed_processor = DistributedProcessor()
        self._initialized = False
    
    def initialize(self):
        """Initialize all scalability components"""
        if not self._initialized:
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            # Start workflow queue
            self.workflow_queue.start()
            
            self._initialized = True
            logger.info("Scalability manager initialized")
    
    def submit_workflow_task(self, task_type: str, payload: Dict[str, Any], 
                           priority: int = 5) -> str:
        """Submit a task to the scalable workflow system"""
        task_id = str(uuid.uuid4())
        
        work_item = WorkItem(
            id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority
        )
        
        # Add to workflow queue
        self.workflow_queue.submit_work_item(work_item)
        
        logger.info(f"Submitted workflow task {task_id} of type {task_type}")
        return task_id
    
    def register_worker_node(self, node_id: str, address: str, 
                           capabilities: Dict[str, Any] = None):
        """Register a worker node for load balancing"""
        self.load_balancer.register_node(node_id, address, capabilities)
    
    def get_scalability_stats(self) -> Dict[str, Any]:
        """Get overall scalability statistics"""
        return {
            'resource_stats': self.resource_monitor.get_current_metrics(),
            'queue_stats': self.workflow_queue.get_queue_stats(),
            'load_balancer_stats': self.load_balancer.get_node_stats(),
            'memory_stats': self.memory_optimizer.get_cache_stats()
        }
    
    def optimize_memory(self):
        """Run memory optimization"""
        # Clear least recently used items from memory cache if needed
        with self.memory_optimizer._lock:
            while (self.memory_optimizer._estimate_memory_usage() > self.memory_optimizer.max_memory_mb
                   and self.memory_optimizer.cache):
                self.memory_optimizer.cache.popitem(last=False)
        logger.info("Memory optimization completed")

    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a task handler for workflow processing"""
        self.task_handlers[task_type] = handler
        self.workflow_queue.register_handler(task_type, handler)
    
    def shutdown(self):
        """Shutdown all scalability components"""
        if self._initialized:
            self.resource_monitor.stop_monitoring()
            self.workflow_queue.stop()
            self.distributed_processor.shutdown()
            self._initialized = False
            logger.info("Scalability manager shutdown")


# Global scalability manager instance
_scalability_manager = None


def get_scalability_manager() -> ScalabilityManager:
    """Get the scalability manager instance"""
    global _scalability_manager
    if _scalability_manager is None:
        _scalability_manager = ScalabilityManager()
        _scalability_manager.initialize()
    return _scalability_manager


def submit_workflow_task(task_type: str, payload: Dict[str, Any], priority: int = 5) -> str:
    """Submit a workflow task"""
    return get_scalability_manager().submit_workflow_task(task_type, payload, priority)


def register_worker_node(node_id: str, address: str, capabilities: Dict[str, Any] = None):
    """Register a worker node"""
    get_scalability_manager().register_worker_node(node_id, address, capabilities)


def get_scalability_stats() -> Dict[str, Any]:
    """Get scalability statistics"""
    return get_scalability_manager().get_scalability_stats()


# Example usage
if __name__ == "__main__":
    # Initialize scalability manager
    scale_manager = get_scalability_manager()
    
    # Register some worker nodes
    register_worker_node("node_1", "localhost:8001", {"cpu": 4, "memory": 16})
    register_worker_node("node_2", "localhost:8002", {"cpu": 8, "memory": 32})
    
    # Submit some workflow tasks
    task_id_1 = submit_workflow_task(
        "problem_analysis", 
        {"problem_id": "prob_123", "content": "Analyze this problem..."},
        priority=8
    )
    
    task_id_2 = submit_workflow_task(
        "solution_generation",
        {"sub_problem_id": "sub_456", "requirements": "Generate solution..."},
        priority=6
    )
    
    print(f"Submitted tasks: {task_id_1}, {task_id_2}")
    
    # Get scalability statistics
    stats = get_scalability_stats()
    print(f"Current scalability stats: {json.dumps(stats, indent=2)[:200]}...")
    
    # Simulate some work and then check stats again
    time.sleep(2)
    
    final_stats = get_scalability_stats()
    print(f"Final stats: {final_stats['queue_stats']}")
    
    print("Scalability improvements implemented successfully!")
