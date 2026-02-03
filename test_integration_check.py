import sys
sys.path.insert(0, 'C:/Users/mmeadow/Documents/OpenEvolve/Frontend')

# Test basic import
try:
    import dspy_integration
    print(f"SUCCESS: dspy_integration imported, DSPY_AVAILABLE={dspy_integration.DSPY_AVAILABLE}")
except Exception as e:
    print(f"ERROR importing dspy_integration: {e}")

try:
    import robust_z3_leanaide_integration
    print("SUCCESS: robust_z3_leanaide_integration imported")
except Exception as e:
    print(f"ERROR importing robust_z3_leanaide_integration: {e}")

try:
    import z3_leanaide_bridge
    print("SUCCESS: z3_leanaide_bridge imported")
except Exception as e:
    print(f"ERROR importing z3_leanaide_bridge: {e}")