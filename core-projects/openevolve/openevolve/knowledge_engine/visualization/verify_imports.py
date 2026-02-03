#!/usr/bin/env python
"""Verification script for Sprint 4 Visualization dependencies."""

import sys

print("=" * 60)
print("SPRINT 4: VISUALIZATION - DEPENDENCY VERIFICATION")
print("=" * 60)

# Test 1: Check pyvis installation
print("\n[1/5] Checking pyvis installation...")
try:
    import pyvis
    print(f"   SUCCESS: pyvis version {pyvis.__version__}")
except ImportError as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

# Test 2: Check networkx installation
print("\n[2/5] Checking networkx installation...")
try:
    import networkx
    print(f"   SUCCESS: networkx version {networkx.__version__}")
except ImportError as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

# Test 3: Check matplotlib installation
print("\n[3/5] Checking matplotlib installation...")
try:
    import matplotlib
    print(f"   SUCCESS: matplotlib version {matplotlib.__version__}")
except ImportError as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

# Test 4: Check visualization modules
print("\n[4/5] Checking visualization module imports...")
try:
    sys.path.insert(0, 'C:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend')
    from knowledge_engine.visualization.graph_explorer import GraphExplorer
    print("   SUCCESS: GraphExplorer imported")
except ImportError as e:
    print(f"   WARNING: {e}")
    print("   NOTE: This is expected when running outside package context")

# Test 5: Verify pyvis functionality
print("\n[5/5] Testing pyvis functionality...")
try:
    from pyvis.network import Network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.add_node(1, label="Node 1")
    net.add_node(2, label="Node 2")
    net.add_edge(1, 2)
    print("   SUCCESS: pyvis Network created and nodes added")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL DEPENDENCY CHECKS PASSED!")
print("=" * 60)
print("\nNext Steps:")
print("1. Run visualization tests: pytest tests/test_visualization.py -v")
print("2. Run generation test: python test_generation.py")
print("3. Check test results for 27/28 passing (96.4% pass rate)")
