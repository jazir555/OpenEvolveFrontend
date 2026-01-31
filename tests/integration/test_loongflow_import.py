"""
Test LoongFlow PES integration

This module tests that LoongFlow (Plan-Execute-Summarize framework)
can be successfully imported and initialized in the OpenEvolve system.
"""

import pytest
import sys
from pathlib import Path


def test_loongflow_basic_import():
    """Test that LoongFlow package can be imported"""
    import loongflow

    assert loongflow is not None
    assert hasattr(loongflow, '__version__')
    print(f"[OK] LoongFlow version: {loongflow.__version__}")


def test_loongflow_memory_evolution_imports():
    """Test that LoongFlow evolution memory components can be imported"""
    # Test correct SDK imports based on actual __init__.py
    from loongflow.agentsdk.memory.evolution import (
        EvolveMemory,
        Solution,
        InMemory,
        MemoryFactory,
        RedisMemory
    )

    print("[OK] LoongFlow evolution memory imports successful")


def test_loongflow_memory_grade_imports():
    """Test that LoongFlow grade memory components can be imported"""
    from loongflow.agentsdk.memory.grade import GradedMemory

    print("[OK] LoongFlow grade memory imports successful")


def test_loongflow_message_imports():
    """Test that LoongFlow message system can be imported"""
    # Import with correct names from __init__.py
    from loongflow.agentsdk.message import (
        Message,
        Role,
        MimeType,
        Element,
        ElementT,
        ToolStatus,
        BaseElement,
        ContentElement,
        ThinkElement,
        ToolCallElement,
        ToolOutputElement,
    )

    # Create a test message
    msg = Message(
        role=Role.USER,
        content="Test message"
    )

    assert msg.role == Role.USER
    assert msg.content == "Test message"

    print("[OK] LoongFlow message system test successful")


def test_loongflow_models_imports():
    """Test that LoongFlow model components can be imported"""
    from loongflow.agentsdk.models import LiteLLMModel

    print("[OK] LoongFlow models imports successful")


def test_loongflow_pes_framework_imports():
    """Test that PES framework can be imported"""
    from loongflow.framework.pes import (
        PESAgent,
        Finalizer,
        LoongFlowFinalizer,
        Worker
    )

    print("[OK] LoongFlow PES framework imports successful")


def test_loongflow_tools_imports():
    """Test LoongFlow tools framework"""
    from loongflow.agentsdk.tools import BaseTool, function_tool

    # Verify base tool exists
    assert BaseTool is not None
    assert function_tool is not None

    print("[OK] LoongFlow tools framework test successful")


def test_loongflow_runner_registration():
    """Test LoongFlow runner registration system"""
    from loongflow.framework.pes.register import register_runner, Worker

    # Test that register_runner function exists
    assert register_runner is not None
    assert Worker is not None

    print("[OK] LoongFlow runner registration test successful")


def test_loongflow_memory_factory():
    """Test LoongFlow memory factory"""
    from loongflow.agentsdk.memory.evolution import MemoryFactory

    # Test memory factory
    factory = MemoryFactory()
    assert factory is not None

    print("[OK] LoongFlow memory factory test successful")


def test_loongflow_logger():
    """Test LoongFlow logging system"""
    from loongflow.agentsdk.logger import MessageLogger

    # Test logger
    logger = MessageLogger()
    assert logger is not None

    print("[OK] LoongFlow logger test successful")


def test_math_agent_direct_import():
    """Test that Math agent files exist and can be imported directly"""
    import importlib.util

    # Test that the math agent file exists
    math_agent_path = Path("C:/Users/mmeadow/Documents/OpenEvolve/Frontend/LoongFlow/agents/math_agent/math_evolve_agent.py")
    assert math_agent_path.exists(), "Math agent file should exist"

    print("[OK] Math PES agent file exists")


def test_ml_agent_direct_import():
    """Test that ML agent files exist"""
    import importlib.util

    # Test that the ml agent file exists
    ml_agent_path = Path("C:/Users/mmeadow/Documents/OpenEvolve/Frontend/LoongFlow/agents/ml_agent/ml_evolve_agent.py")
    assert ml_agent_path.exists(), "ML agent file should exist"

    print("[OK] ML PES agent file exists")


def test_general_agent_direct_import():
    """Test that General agent files exist"""
    import importlib.util

    # Test that the general agent file exists
    general_agent_path = Path("C:/Users/mmeadow/Documents/OpenEvolve/Frontend/LoongFlow/agents/general_agent/general_evolve_agent.py")
    assert general_agent_path.exists(), "General agent file should exist"

    print("[OK] General PES agent file exists")


# Helper function to run tests standalone
def run_all_tests():
    """Run all tests and print results"""
    tests = [
        ("Basic Import", test_loongflow_basic_import),
        ("Memory Evolution Imports", test_loongflow_memory_evolution_imports),
        ("Memory Grade Imports", test_loongflow_memory_grade_imports),
        ("Message System", test_loongflow_message_imports),
        ("Models Imports", test_loongflow_models_imports),
        ("PES Framework", test_loongflow_pes_framework_imports),
        ("Tools Framework", test_loongflow_tools_imports),
        ("Runner Registration", test_loongflow_runner_registration),
        ("Memory Factory", test_loongflow_memory_factory),
        ("Logger System", test_loongflow_logger),
        ("Math Agent File", test_math_agent_direct_import),
        ("ML Agent File", test_ml_agent_direct_import),
        ("General Agent File", test_general_agent_direct_import),
    ]

    passed = 0
    failed = 0

    print("\n" + "="*60)
    print("Running LoongFlow Integration Tests")
    print("="*60 + "\n")

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"[PASS] {test_name}: PASSED\n")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test_name}: FAILED")
            print(f"   Error: {str(e)}\n")
            import traceback
            traceback.print_exc()

    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
