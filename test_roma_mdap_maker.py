"""
ROMA-MDAP-MAKER Test Suite

Comprehensive test suite for ROMA-MDAP-MAKER integration.

Run with: python test_roma_mdap_maker.py
"""

import logging
import sys
import time
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ROMAMDAPMakerTestSuite:
    """Comprehensive test suite for ROMA-MDAP-MAKER integration"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def test(self, name: str, func):
        """Run a test function"""
        try:
            logger.info(f"\n[TEST] {name}")
            result = func()
            if result:
                self.passed += 1
                logger.info(f"[PASS] {name}")
                self.tests.append({"name": name, "status": "pass"})
            else:
                self.failed += 1
                logger.warning(f"[FAIL] {name} - Returned False")
                self.tests.append({"name": name, "status": "fail", "reason": "Returned False"})
        except Exception as e:
            self.failed += 1
            logger.error(f"[ERROR] {name} - {e}")
            self.tests.append({"name": name, "status": "error", "reason": str(e)})

    def summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        total = self.passed + self.failed
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {self.passed / total * 100:.1f}%")
        print("=" * 80)
        return self.failed == 0


# =============================================================================
# TEST 1: Import Tests
# =============================================================================

def test_import_engine():
    """Test that engine module imports correctly"""
    try:
        from roma_mdap_maker_engine import (
            ROMAMDAPMakerEngine,
            ROMAMDAPMakerConfig,
            ROMARedFlagger,
            HierarchicalVotingStrategy,
            AdaptiveKSelector,
            create_roma_mdap_maker_config,
            get_roma_mdap_maker_status,
            ROMA_AVAILABLE,
            MDAP_AVAILABLE,
        )
        return True
    except ImportError as e:
        logger.error(f"Failed to import engine: {e}")
        return False


def test_import_mcp_tools():
    """Test that MCP tools import correctly"""
    try:
        from roma_mdap_maker_mcp_tools import (
            solve_with_roma_mdap_maker,
            solve_subproblem_with_roma_mdap_maker,
            get_roma_mdap_maker_status,
            analyze_problem_with_roma_mdap,
            verify_solution_with_roma_mdap,
            create_roma_mdap_maker_config_tool,
            get_roma_mdap_maker_metrics,
            list_mcp_tools,
        )
        return True
    except ImportError as e:
        logger.error(f"Failed to import MCP tools: {e}")
        return False


def test_import_bridge():
    """Test that crewai bridge imports correctly"""
    try:
        from roma_mdap_maker_crewai_bridge import (
            execute_phase_1_setup,
            execute_phase_2_solve,
            execute_phase_3_critique,
            execute_phase_4_verify,
            execute_phase_5_reassemble,
            execute_phase_6_final_validation,
            execute_full_workflow,
            get_romamdapmaker_bridge_status,
            PHASE_FUNCTIONS,
        )
        return True
    except ImportError as e:
        logger.error(f"Failed to import bridge: {e}")
        return False


# =============================================================================
# TEST 2: Configuration Tests
# =============================================================================

def test_config_creation():
    """Test configuration creation"""
    try:
        from roma_mdap_maker_engine import create_roma_mdap_maker_config

        config = create_roma_mdap_maker_config(
            roma_max_depth_analysis=3,
            roma_max_depth_solving=2,
            mdap_k_ahead=3,
            enable_adaptive_k=True,
        )

        assert config.roma_max_depth_analysis == 3
        assert config.roma_max_depth_solving == 2
        assert config.mdap_k_ahead == 3
        assert config.enable_adaptive_k == True

        logger.info("Configuration created with correct parameters")
        return True
    except Exception as e:
        logger.error(f"Config creation failed: {e}")
        return False


def test_config_defaults():
    """Test configuration defaults"""
    try:
        from roma_mdap_maker_engine import ROMAMDAPMakerConfig

        config = ROMAMDAPMakerConfig()

        assert config.mdap_enabled == True
        assert config.mdap_k_ahead == 3
        assert config.mdap_enable_red_flagging == True
        assert config.apply_maker_to_roma_atomic == True
        assert config.enable_hierarchical_voting == True
        assert config.enable_adaptive_k == True

        logger.info("All defaults are correct")
        return True
    except Exception as e:
        logger.error(f"Config defaults test failed: {e}")
        return False


# =============================================================================
# TEST 3: Status Tests
# =============================================================================

def test_engine_status():
    """Test engine status function"""
    try:
        from roma_mdap_maker_engine import get_roma_mdap_maker_status

        status = get_roma_mdap_maker_status()

        assert "roma_available" in status
        assert "mdap_available" in status
        assert "roma_mdap_maker_available" in status
        assert "total_execution_methods" in status
        assert "execution_methods" in status

        assert status["total_execution_methods"] == 7
        assert "roma_mdap_maker" in status["execution_methods"]

        logger.info(f"Engine status: {status['roma_mdap_maker_available']}")
        return True
    except Exception as e:
        logger.error(f"Engine status test failed: {e}")
        return False


def test_bridge_status():
    """Test bridge status function"""
    try:
        from roma_mdap_maker_crewai_bridge import get_romamdapmaker_bridge_status

        status = get_romamdapmaker_bridge_status()

        assert "bridge_available" in status
        assert "roma_available" in status
        assert "mdap_available" in status
        assert "phases_supported" in status

        assert len(status["phases_supported"]) == 6

        logger.info(f"Bridge status: {status['bridge_available']}, phases: {len(status['phases_supported'])}")
        return True
    except Exception as e:
        logger.error(f"Bridge status test failed: {e}")
        return False


# =============================================================================
# TEST 4: MCP Tools Tests
# =============================================================================

def test_mcp_tool_count():
    """Test that all 7 MCP tools are registered"""
    try:
        from roma_mdap_maker_mcp_tools import list_mcp_tools

        tools = list_mcp_tools()

        assert len(tools) == 7
        assert "solve_with_roma_mdap_maker" in tools
        assert "solve_subproblem_with_roma_mdap_maker" in tools
        assert "get_roma_mdap_maker_status" in tools
        assert "analyze_problem_with_roma_mdap" in tools
        assert "verify_solution_with_roma_mdap" in tools
        assert "create_roma_mdap_maker_config" in tools
        assert "get_roma_mdap_maker_metrics" in tools

        logger.info(f"All 7 MCP tools registered: {tools}")
        return True
    except Exception as e:
        logger.error(f"MCP tool count test failed: {e}")
        return False


# =============================================================================
# TEST 5: Routing Logic Tests
# =============================================================================

def test_routing_explicit_roma_mdap_maker():
    """Test explicit roma_mdap_maker selection"""
    try:
        from decomposition_mcp_tools import _determine_execution_method

        result = _determine_execution_method(
            "roma_mdap_maker",
            False, False, False, False,  # use_claudiomiro, use_datapizza, use_roma, use_hybrid
            True,  # use_roma_mdap_maker
            "test-id",
            "Solve this problem"
        )

        assert result == "roma_mdap_maker"
        logger.info(f"Explicit selection: {result}")
        return True
    except Exception as e:
        logger.error(f"Explicit routing test failed: {e}")
        return False


def test_routing_auto_critical():
    """Test auto-selection for critical tasks"""
    try:
        from decomposition_mcp_tools import _determine_execution_method

        critical_keywords = [
            "Design critical zero-error system",
            "Build flawless component",
            "Create perfect solution",
            "Mission-critical application",
            "Safety-critical system",
            "High-reliability requirement"
        ]

        all_passed = True
        for description in critical_keywords:
            result = _determine_execution_method(
                "auto",
                False, False, False, False,
                True,
                "test-id",
                description
            )
            if result != "roma_mdap_maker":
                logger.error(f"Failed for: {description}")
                all_passed = False

        if all_passed:
            logger.info(f"All {len(critical_keywords)} critical keywords auto-selected correctly")
        return all_passed
    except Exception as e:
        logger.error(f"Auto-routing test failed: {e}")
        return False


def test_routing_auto_normal():
    """Test auto-selection fallback for normal tasks"""
    try:
        from decomposition_mcp_tools import _determine_execution_method

        result = _determine_execution_method(
            "auto",
            False, False, False, False,
            True,
            "test-id",
            "Build a simple web page"
        )

        # Should fall back to traditional for normal tasks
        assert result in ["traditional", "roma"]
        logger.info(f"Normal task routing: {result}")
        return True
    except Exception as e:
        logger.error(f"Normal routing test failed: {e}")
        return False


# =============================================================================
# TEST 6: Decomposition Integration Tests
# =============================================================================

def test_decomposition_status():
    """Test decomposition workflow status includes ROMA-MDAP-MAKER"""
    try:
        from decomposition_mcp_tools import get_decomposition_status

        status = get_decomposition_status()

        assert status["total_execution_methods"] == 7
        assert "roma_mdap_maker" in status["execution_methods"]
        assert status["roma_mdap_maker_available"] == True

        logger.info("Decomposition integration: OK")
        return True
    except Exception as e:
        logger.error(f"Decomposition status test failed: {e}")
        return False


def test_unified_bridge_status():
    """Test unified bridge status includes ROMA-MDAP-MAKER"""
    try:
        from crewai_unified_bridge import get_unified_bridge_status

        status = get_unified_bridge_status()

        assert status["total_execution_methods"] == 7
        assert "roma_mdap_maker" in status["execution_methods"]
        assert status["roma_mdap_maker_bridge_available"] == True

        logger.info("Unified bridge integration: OK")
        return True
    except Exception as e:
        logger.error(f"Unified bridge status test failed: {e}")
        return False


# =============================================================================
# TEST 7: Phase Functions Tests
# =============================================================================

def test_phase_functions_exist():
    """Test that all 6 phase functions exist"""
    try:
        from roma_mdap_maker_crewai_bridge import PHASE_FUNCTIONS

        assert len(PHASE_FUNCTIONS) == 6

        for phase_num in [1, 2, 3, 4, 5, 6]:
            assert phase_num in PHASE_FUNCTIONS
            assert PHASE_FUNCTIONS[phase_num].__name__.startswith("execute_phase_")

        logger.info(f"All 6 phase functions exist: {list(PHASE_FUNCTIONS.keys())}")
        return True
    except Exception as e:
        logger.error(f"Phase functions test failed: {e}")
        return False


# =============================================================================
# TEST 8: Red-Flagger Tests
# =============================================================================

def test_red_flagger_cycle_detection():
    """Test cycle detection in red-flagger"""
    try:
        from roma_mdap_maker_engine import ROMARedFlagger, create_roma_mdap_maker_config

        config = create_roma_mdap_maker_config(roma_max_depth_analysis=3)
        flagger = ROMARedFlagger(config)

        # Create DAG with cycle
        dag_with_cycle = {
            "task1": {"children": ["task2"]},
            "task2": {"children": ["task3"]},
            "task3": {"children": ["task1"]},  # Cycle
        }

        flags = flagger.check_roma_decomposition_red_flags(dag_with_cycle)

        # Should detect cycle
        assert any("cycle" in flag.lower() for flag in flags)
        logger.info(f"Cycle detection: {flags}")
        return True
    except Exception as e:
        logger.error(f"Cycle detection test failed: {e}")
        return False


def test_red_flagger_depth_detection():
    """Test depth detection in red-flagger"""
    try:
        from roma_mdap_maker_engine import ROMARedFlagger, create_roma_mdap_maker_config

        config = create_roma_mdap_maker_config(roma_max_depth_analysis=3)
        flagger = ROMARedFlagger(config)

        # Create deep DAG
        dag_deep = {
            "task1": {"children": ["task2"], "description": "A"},
            "task2": {"children": ["task3"], "description": "B"},
            "task3": {"children": ["task4"], "description": "C"},
            "task4": {"children": ["task5"], "description": "D"},
            "task5": {"children": [], "description": "E"},
        }

        depth = flagger._calculate_depth(dag_deep)
        flags = flagger.check_roma_decomposition_red_flags(dag_deep)

        logger.info(f"Depth: {depth}, flags: {flags}")
        return True
    except Exception as e:
        logger.error(f"Depth detection test failed: {e}")
        return False


# =============================================================================
# TEST 9: Adaptive K Selector Tests
# =============================================================================

def test_adaptive_k_simple():
    """Test adaptive k selection for simple task"""
    try:
        from roma_mdap_maker_engine import AdaptiveKSelector, create_roma_mdap_maker_config

        config = create_roma_mdap_maker_config(mdap_k_ahead=3)
        selector = AdaptiveKSelector(config)

        task = {"description": "Simple task", "dependencies": []}
        k = selector.select_k_for_roma_task(task, depth=0, base_k=3)

        # Simple task should get k around 2-3
        assert 2 <= k <= 3
        logger.info(f"Simple task k: {k}")
        return True
    except Exception as e:
        logger.error(f"Adaptive k simple test failed: {e}")
        return False


def test_adaptive_k_deep():
    """Test adaptive k selection for deep task"""
    try:
        from roma_mdap_maker_engine import AdaptiveKSelector, create_roma_mdap_maker_config

        config = create_roma_mdap_maker_config(mdap_k_ahead=3)
        selector = AdaptiveKSelector(config)

        task = {"description": "Deep task", "dependencies": []}
        k = selector.select_k_for_roma_task(task, depth=4, base_k=3)

        # Deep task should get higher k
        assert k >= 3
        logger.info(f"Deep task k: {k}")
        return True
    except Exception as e:
        logger.error(f"Adaptive k deep test failed: {e}")
        return False


# =============================================================================
# TEST 10: Integration Tests
# =============================================================================

def test_end_to_end_integration():
    """Test complete integration chain"""
    try:
        # Import all components
        from roma_mdap_maker_engine import get_roma_mdap_maker_status
        from roma_mdap_maker_mcp_tools import list_mcp_tools
        from roma_mdap_maker_crewai_bridge import get_romamdapmaker_bridge_status
        from decomposition_mcp_tools import get_decomposition_status
        from crewai_unified_bridge import get_unified_bridge_status

        # Check all statuses
        engine_status = get_roma_mdap_maker_status()
        bridge_status = get_romamdapmaker_bridge_status()
        decomp_status = get_decomposition_status()
        unified_status = get_unified_bridge_status()

        # Verify integration
        assert engine_status["total_execution_methods"] == 7
        assert bridge_status["bridge_available"] == True
        assert decomp_status["total_execution_methods"] == 7
        assert unified_status["total_execution_methods"] == 7

        tools = list_mcp_tools()
        assert len(tools) == 7

        logger.info("End-to-end integration: SUCCESS")
        logger.info(f"  Engine: {engine_status['roma_mdap_maker_available']}")
        logger.info(f"  Bridge: {bridge_status['bridge_available']}")
        logger.info(f"  Decomposition: {decomp_status['roma_mdap_maker_available']}")
        logger.info(f"  Unified: {unified_status['roma_mdap_maker_bridge_available']}")
        logger.info(f"  MCP Tools: {len(tools)}")

        return True
    except Exception as e:
        logger.error(f"End-to-end integration test failed: {e}")
        return False


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ROMA-MDAP-MAKER COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    suite = ROMAMDAPMakerTestSuite()

    # Import Tests
    print("\n" + "-" * 80)
    print("IMPORT TESTS")
    print("-" * 80)
    suite.test("Import engine module", test_import_engine)
    suite.test("Import MCP tools", test_import_mcp_tools)
    suite.test("Import bridge", test_import_bridge)

    # Configuration Tests
    print("\n" + "-" * 80)
    print("CONFIGURATION TESTS")
    print("-" * 80)
    suite.test("Config creation", test_config_creation)
    suite.test("Config defaults", test_config_defaults)

    # Status Tests
    print("\n" + "-" * 80)
    print("STATUS TESTS")
    print("-" * 80)
    suite.test("Engine status", test_engine_status)
    suite.test("Bridge status", test_bridge_status)

    # MCP Tools Tests
    print("\n" + "-" * 80)
    print("MCP TOOLS TESTS")
    print("-" * 80)
    suite.test("MCP tool count", test_mcp_tool_count)

    # Routing Tests
    print("\n" + "-" * 80)
    print("ROUTING TESTS")
    print("-" * 80)
    suite.test("Explicit roma_mdap_maker routing", test_routing_explicit_roma_mdap_maker)
    suite.test("Auto-routing for critical tasks", test_routing_auto_critical)
    suite.test("Auto-routing for normal tasks", test_routing_auto_normal)

    # Integration Tests
    print("\n" + "-" * 80)
    print("INTEGRATION TESTS")
    print("-" * 80)
    suite.test("Decomposition integration", test_decomposition_status)
    suite.test("Unified bridge integration", test_unified_bridge_status)

    # Phase Functions Tests
    print("\n" + "-" * 80)
    print("PHASE FUNCTIONS TESTS")
    print("-" * 80)
    suite.test("Phase functions exist", test_phase_functions_exist)

    # Red-Flagger Tests
    print("\n" + "-" * 80)
    print("RED-FLAGGER TESTS")
    print("-" * 80)
    suite.test("Cycle detection", test_red_flagger_cycle_detection)
    suite.test("Depth detection", test_red_flagger_depth_detection)

    # Adaptive K Tests
    print("\n" + "-" * 80)
    print("ADAPTIVE K TESTS")
    print("-" * 80)
    suite.test("Adaptive k for simple task", test_adaptive_k_simple)
    suite.test("Adaptive k for deep task", test_adaptive_k_deep)

    # End-to-End Test
    print("\n" + "-" * 80)
    print("END-TO-END TEST")
    print("-" * 80)
    suite.test("Complete integration chain", test_end_to_end_integration)

    # Print summary
    success = suite.summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
