"""Test enhanced distributed processing features."""

import time
from distributed_processing import (
    DistributedCoordinator,
    WorkerNode,
    SyncManager,
    DistributedWorkflowExecutor,
    WorkerStatus
)
from workflow_structures import SubProblem, DecompositionPlan, WorkflowState, SolutionAttempt


def test_sync_manager():
    """Test synchronization manager."""
    print("Testing SyncManager...")
    sync_mgr = SyncManager()
    
    # Test state updates
    sync_mgr.update_shared_state("key1", "value1")
    sync_mgr.update_shared_state("key2", 42)
    
    assert sync_mgr.get_shared_state("key1") == "value1"
    assert sync_mgr.get_shared_state("key2") == 42
    assert sync_mgr.version_counter == 2
    
    # Test get all state
    all_state = sync_mgr.get_all_state()
    assert len(all_state) == 2
    
    # Test clear
    sync_mgr.clear_state()
    assert len(sync_mgr.get_all_state()) == 0
    assert sync_mgr.version_counter == 0
    
    print("[OK] SyncManager tests passed")


def test_worker_node():
    """Test worker node."""
    print("\nTesting WorkerNode...")
    worker = WorkerNode("test_worker_1")
    
    assert worker.info.worker_id == "test_worker_1"
    assert worker.info.status == WorkerStatus.IDLE
    assert worker.info.tasks_completed == 0
    
    # Test status reporting
    status = worker.report_status()
    assert status.worker_id == "test_worker_1"
    
    # Test shutdown
    worker.shutdown()
    assert worker.info.status == WorkerStatus.SHUTDOWN
    
    print("[OK] WorkerNode tests passed")


def test_distributed_coordinator():
    """Test distributed coordinator."""
    print("\nTesting DistributedCoordinator...")
    coordinator = DistributedCoordinator(max_workers=2)
    
    # Check workers initialized
    assert len(coordinator.workers) == 2
    assert "worker_0" in coordinator.workers
    assert "worker_1" in coordinator.workers
    
    # Test worker status
    worker_status = coordinator.get_worker_status()
    assert len(worker_status) == 2
    
    # Test sub-problem distribution
    sub_problems = [
        SubProblem(id="sp1", description="Test 1", dependencies=[]),
        SubProblem(id="sp2", description="Test 2", dependencies=["sp1"])
    ]
    dependencies = {"sp1": [], "sp2": ["sp1"]}
    coordinator.distribute_sub_problems(sub_problems, dependencies)
    
    assert len(coordinator.tasks) == 2
    assert "task_sp1" in coordinator.tasks
    assert "task_sp2" in coordinator.tasks
    
    # Test shutdown
    coordinator.shutdown()
    
    print("[OK] DistributedCoordinator tests passed")


def test_worker_failure_handling():
    """Test worker failure detection and handling."""
    print("\nTesting worker failure handling...")
    coordinator = DistributedCoordinator(max_workers=2)
    
    # Simulate a worker failure
    worker_id = "worker_0"
    coordinator.workers[worker_id].status = WorkerStatus.BUSY
    coordinator.workers[worker_id].last_heartbeat = time.time() - 100  # Old heartbeat
    
    # Create a task for this worker
    from distributed_processing import TaskInfo
    task = TaskInfo(
        task_id="task_1",
        sub_problem_id="sp1",
        worker_id=worker_id,
        status="running"
    )
    coordinator.tasks["task_1"] = task
    
    # Handle failure
    coordinator.handle_worker_failure(worker_id)
    
    # Check worker marked as failed
    assert coordinator.workers[worker_id].status == WorkerStatus.FAILED
    
    # Check task reassigned
    assert coordinator.tasks["task_1"].status == "pending"
    assert coordinator.tasks["task_1"].worker_id is None
    assert coordinator.tasks["task_1"].retry_count == 1
    
    coordinator.shutdown()
    print("[OK] Worker failure handling tests passed")


def test_distributed_executor_stats():
    """Test distributed executor statistics."""
    print("\nTesting DistributedWorkflowExecutor statistics...")
    executor = DistributedWorkflowExecutor(max_workers=2)
    
    # Get statistics
    stats = executor.get_execution_statistics()
    assert "max_workers" in stats
    assert "worker_status" in stats
    assert "sync_state_version" in stats
    assert stats["max_workers"] == 2
    
    # Get worker status
    worker_status = executor.get_worker_status()
    assert len(worker_status) == 2
    
    executor.shutdown()
    print("[OK] DistributedWorkflowExecutor statistics tests passed")


if __name__ == "__main__":
    print("Running enhanced distributed processing tests...\n")
    
    test_sync_manager()
    test_worker_node()
    test_distributed_coordinator()
    test_worker_failure_handling()
    test_distributed_executor_stats()
    
    print("\n" + "="*50)
    print("All enhanced distributed processing tests passed!")
    print("="*50)
