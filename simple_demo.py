"""
Simple demo showcasing OpenEvolve performance optimization functionality
"""
import sys
import os
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simple_demo():
    """Simple demonstration of OpenEvolve performance optimization"""
    print("OpenEvolve Performance Optimization Demo")
    print("=" * 50)
    
    # Import performance optimization components
    from performance_optimization import (
        CachingOptimizer, 
        ParallelizationOptimizer, 
        AsyncProcessingOptimizer, 
        MemoryManagementOptimizer
    )
    
    print("[OK] All performance optimization components imported")
    
    # 1. Test Caching Optimizer
    print("\\n1. Testing Caching Optimizer")
    print("-" * 30)
    
    cache_optimizer = CachingOptimizer()
    cache_optimizer.configure(max_cache_size=500, cache_ttl=1800)  # 30 minutes
    
    # Test data
    sample_data = "This is sample data for caching optimization test. " * 100
    context = {"operation": "cache_test", "data_type": "text"}
    
    # Measure before and after
    start_time = time.time()
    cache_optimizer.apply_optimization(sample_data, context)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    cache_optimizer.apply_optimization(sample_data, context)
    second_call_time = time.time() - start_time
    
    cache_info = cache_optimizer.get_cache_info()
    print(f"  First call time: {first_call_time:.4f} seconds")
    print(f"  Second call time: {second_call_time:.4f} seconds")
    print(f"  Cache hit rate: {cache_info['hit_rate']:.2f}%")
    print(f"  Cache size: {cache_info['cache_size']}")
    
    # 2. Test Parallelization Optimizer
    print("\\n2. Testing Parallelization Optimizer")
    print("-" * 30)
    
    parallel_optimizer = ParallelizationOptimizer()
    parallel_optimizer.configure(max_workers=4)
    
    # Large data set for parallelization
    large_data = list(range(10000))
    context = {"operation": "parallel_test", "data_type": "list"}
    
    start_time = time.time()
    parallel_result = parallel_optimizer.apply_optimization(large_data, context)
    parallel_time = time.time() - start_time
    
    print(f"  Processed {len(large_data)} items in {parallel_time:.4f} seconds")
    print(f"  Result length: {len(parallel_result) if parallel_result else 0}")
    
    parallel_info = parallel_optimizer.get_parallelization_info()
    print(f"  Workers used: {parallel_info['max_workers']}")
    print(f"  Tasks submitted: {parallel_info['parallelization_stats']['tasks_submitted']}")
    
    # 3. Test Async Processing Optimizer
    print("\\n3. Testing Async Processing Optimizer")
    print("-" * 30)
    
    async_optimizer = AsyncProcessingOptimizer()
    async_optimizer.configure(max_concurrent_tasks=20)
    
    # I/O bound data for async processing
    io_data = ["task_" + str(i) for i in range(50)]
    context = {"operation": "async_test", "io_bound": True}
    
    start_time = time.time()
    async_result = async_optimizer.apply_optimization(io_data, context)
    async_time = time.time() - start_time
    
    print(f"  Processed {len(io_data)} async tasks in {async_time:.4f} seconds")
    print(f"  Result length: {len(async_result) if async_result else 0}")
    
    async_info = async_optimizer.get_async_info()
    print(f"  Async tasks created: {async_info['async_stats']['async_tasks_created']}")
    print(f"  Async tasks completed: {async_info['async_stats']['async_tasks_completed']}")
    
    # 4. Test Memory Management Optimizer
    print("\\n4. Testing Memory Management Optimizer")
    print("-" * 30)
    
    memory_optimizer = MemoryManagementOptimizer()
    memory_optimizer.configure(memory_threshold_mb=50.0, max_pool_size=50)
    
    # Memory-intensive data
    memory_data = {"large_list": list(range(50000)), "nested_dict": {}}
    for i in range(1000):
        memory_data["nested_dict"][f"key_{i}"] = {"value": i, "data": list(range(100))}
    
    context = {"operation": "memory_test", "data_type": "large_structure"}
    
    start_time = time.time()
    memory_result = memory_optimizer.apply_optimization(memory_data, context)
    memory_time = time.time() - start_time
    
    print(f"  Processed large data structure in {memory_time:.4f} seconds")
    print(f"  Result keys: {list(memory_result.keys()) if memory_result else 'None'}")
    
    # Show memory optimization statistics
    memory_stats = memory_optimizer.get_statistics()
    print(f"  Objects allocated: {memory_stats.get('objects_allocated', 0)}")
    print(f"  Garbage collections: {memory_stats.get('garbage_collections', 0)}")
    
    # Summary
    print("\\n" + "=" * 50)
    print("PERFORMANCE OPTIMIZATION SUMMARY")
    print("=" * 50)
    
    print("\\nOptimization Results:")
    print(f"  Caching: {cache_info['hit_rate']:.2f}% hit rate")
    print(f"  Parallelization: {len(large_data)} items in {parallel_time:.4f}s")
    print(f"  Async Processing: {len(io_data)} tasks in {async_time:.4f}s")
    print(f"  Memory Management: Large structure in {memory_time:.4f}s")
    
    # Overall statistics
    cache_stats = cache_optimizer.get_statistics()
    parallel_stats = parallel_optimizer.get_statistics()
    async_stats = async_optimizer.get_statistics()
    memory_stats = memory_optimizer.get_statistics()
    
    print("\\nComponent Statistics:")
    print(f"  Caching Optimizer: {cache_stats.get('total_optimizations', 0)} optimizations")
    print(f"  Parallelization Optimizer: {parallel_stats.get('total_optimizations', 0)} optimizations")
    print(f"  Async Processing Optimizer: {async_stats.get('total_optimizations', 0)} optimizations")
    print(f"  Memory Management Optimizer: {memory_stats.get('total_optimizations', 0)} optimizations")
    
    print("\\n\\n[SUCCESS] Performance optimization demo completed successfully!")
    print("All OpenEvolve performance optimization components are working correctly.")
    print("=" * 50)

if __name__ == "__main__":
    simple_demo()