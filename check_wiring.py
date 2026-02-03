#!/usr/bin/env python3
"""Check Adaptive MDAP wiring in key files."""

# Check workflow_engine.py
with open('workflow_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()
print('workflow_engine.py:')
print(f'  ADAPTIVE_MDAP_AVAILABLE: {"ADAPTIVE_MDAP_AVAILABLE" in content}')
print(f'  get_adaptive_workflow: {"get_adaptive_workflow" in content}')
print(f'  get_adaptive_mdap_status: {"get_adaptive_mdap_status" in content}')
print()

# Check evolution.py
with open('evolution.py', 'r', encoding='utf-8') as f:
    content = f.read()
print('evolution.py:')
print(f'  ADAPTIVE_MDAP_AVAILABLE: {"ADAPTIVE_MDAP_AVAILABLE" in content}')
print(f'  enable_adaptive_mdap: {"enable_adaptive_mdap" in content}')
print()

# Check openevolve_orchestrator.py
with open('openevolve_orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()
print('openevolve_orchestrator.py:')
print(f'  ADAPTIVE_MDAP_AVAILABLE: {"ADAPTIVE_MDAP_AVAILABLE" in content}')
print(f'  adaptive_mdap_config: {"adaptive_mdap_config" in content}')
print()

# Check sidebar.py
with open('sidebar.py', 'r', encoding='utf-8') as f:
    content = f.read()
print('sidebar.py:')
print(f'  enable_adaptive_mdap: {"enable_adaptive_mdap" in content}')
print(f'  adaptive_profile: {"adaptive_profile" in content}')
print()

# Check api_server.py
with open('api_server.py', 'r', encoding='utf-8') as f:
    content = f.read()
print('api_server.py:')
print(f'  /adaptive-mdap/: {"/adaptive-mdap/" in content}')
print()

# Check app.py
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()
print('app.py:')
print(f'  TaskComplexityClassifier: {"TaskComplexityClassifier" in content}')
