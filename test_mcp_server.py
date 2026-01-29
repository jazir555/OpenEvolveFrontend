#!/usr/bin/env python3
"""
Test script for BubbleLab CrewAI MCP Server functionality
"""

import asyncio
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_mcp_server_syntax():
    """Test that the MCP server has valid Python syntax."""
    print("Testing MCP server syntax...")

    try:
        import py_compile
        py_compile.compile('bubblelab_crewai_mcp_server.py', doraise=True)
        print("MCP server has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"Syntax error in MCP server: {e}")
        return False
    except Exception as e:
        print(f"Error compiling MCP server: {e}")
        return False


def test_crewai_integration_syntax():
    """Test that the CrewAI integration layer has valid Python syntax."""
    print("\nTesting CrewAI integration layer syntax...")

    try:
        import py_compile
        py_compile.compile('crewai_integration_layer.py', doraise=True)
        print("CrewAI integration layer has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"Syntax error in CrewAI integration layer: {e}")
        return False
    except Exception as e:
        print(f"Error compiling CrewAI integration layer: {e}")
        return False


def test_client_syntax():
    """Test that the MCP client has valid Python syntax."""
    print("\nTesting MCP client syntax...")

    try:
        import py_compile
        py_compile.compile('bubblelab_mcp_client.py', doraise=True)
        print("MCP client has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"Syntax error in MCP client: {e}")
        return False
    except Exception as e:
        print(f"Error compiling MCP client: {e}")
        return False


def test_imports():
    """Test that all modules can be imported."""
    print("\nTesting module imports...")

    try:
        from bubblelab_crewai_mcp_server import mcp_server
        print("MCP server module imported successfully")
    except Exception as e:
        print(f"Error importing MCP server: {e}")
        return False

    try:
        from crewai_integration_layer import get_crewai_service
        service = get_crewai_service()
        print("CrewAI integration layer imported successfully")
    except Exception as e:
        print(f"Error importing CrewAI integration layer: {e}")
        return False

    try:
        from bubblelab_mcp_client import BubbleLabMCPClient
        print("MCP client imported successfully")
    except Exception as e:
        print(f"Error importing MCP client: {e}")
        return False

    return True


def test_crewai_service():
    """Test the CrewAI service functionality."""
    print("\nTesting CrewAI service functionality...")

    try:
        from crewai_integration_layer import get_crewai_service
        service = get_crewai_service()

        # Test getting templates
        templates = asyncio.run(service.get_available_templates())
        print(f"Got available templates: {list(templates.keys())}")

        # Test creating a mock agent (since CrewAI might not be available)
        agent = asyncio.run(service.create_agent_from_template("researcher"))
        print(f"Created agent: {agent['role']} with ID: {agent['id']}")

        return True
    except Exception as e:
        print(f"Error testing CrewAI service: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Starting BubbleLab CrewAI MCP Server Tests\n")

    tests = [
        ("MCP Server Syntax", test_mcp_server_syntax),
        ("CrewAI Integration Syntax", test_crewai_integration_syntax),
        ("MCP Client Syntax", test_client_syntax),
        ("Module Imports", test_imports),
        ("CrewAI Service", test_crewai_service),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        result = test_func()
        results.append((test_name, result))

    print(f"\nTest Results:")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False

    if all_passed:
        print(f"\nAll tests passed! MCP server is ready for use.")
        print("\nTo start the server, run: python bubblelab_crewai_mcp_server.py")
        return 0
    else:
        print(f"\nSome tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())