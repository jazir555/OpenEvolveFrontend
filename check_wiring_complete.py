#!/usr/bin/env python3
"""Complete check for Adaptive MDAP wiring in all files."""

print("=" * 60)
print("ADAPTIVE MDAP WIRING VERIFICATION")
print("=" * 60)

checks = []

# 1. workflow_engine.py
with open('workflow_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n1. workflow_engine.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
checks.append(('get_adaptive_workflow', 'get_adaptive_workflow' in content))
checks.append(('get_adaptive_mdap_status', 'get_adaptive_mdap_status' in content))
for name, result in checks[-3:]:
    print(f"   {name}: {result}")

# 2. evolution.py
with open('evolution.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n2. evolution.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
checks.append(('enable_adaptive_mdap', 'enable_adaptive_mdap' in content))
for name, result in checks[-2:]:
    print(f"   {name}: {result}")

# 3. openevolve_orchestrator.py
with open('openevolve_orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n3. openevolve_orchestrator.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 4. sidebar.py
with open('sidebar.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n4. sidebar.py:")
checks.append(('enable_adaptive_mdap', 'enable_adaptive_mdap' in content))
checks.append(('adaptive_profile', 'adaptive_profile' in content))
for name, result in checks[-2:]:
    print(f"   {name}: {result}")

# 5. api_server.py
with open('api_server.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n5. api_server.py:")
checks.append(('/adaptive-mdap/', '/adaptive-mdap/' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 6. app.py
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n6. app.py:")
checks.append(('TaskComplexityClassifier', 'TaskComplexityClassifier' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 7. openevolve_cli.py
with open('openevolve_cli.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n7. openevolve_cli.py:")
checks.append(('adaptive command', 'def adaptive():' in content))
checks.append(('classify command', 'def classify(' in content))
checks.append(('allocate command', 'def allocate(' in content))
for name, result in checks[-3:]:
    print(f"   {name}: {result}")

# 8. red_team.py
with open('red_team.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n8. red_team.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 9. blue_team.py
with open('blue_team.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n9. blue_team.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 10. demo_mdap_maker.py
with open('demo_mdap_maker.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n10. demo_mdap_maker.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 11. config_loader.py
with open('config_loader.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n11. config_loader.py:")
checks.append(('AdaptiveMDAPConfig', 'AdaptiveMDAPConfig' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 12. team_assignment_engine.py
with open('team_assignment_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n12. team_assignment_engine.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
checks.append(('compute_subproblem_complexity', 'compute_subproblem_complexity' in content))
for name, result in checks[-2:]:
    print(f"   {name}: {result}")

# 13. gauntlet_manager.py
with open('gauntlet_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n13. gauntlet_manager.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
checks.append(('create_adaptive_gauntlet', 'create_adaptive_gauntlet' in content))
for name, result in checks[-2:]:
    print(f"   {name}: {result}")

# 14. quality_assessment.py
with open('quality_assessment.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n14. quality_assessment.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
checks.append(('assess_quality_with_complexity', 'assess_quality_with_complexity' in content))
for name, result in checks[-2:]:
    print(f"   {name}: {result}")

# 15. monitoring_system.py
with open('monitoring_system.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n15. monitoring_system.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
checks.append(('record_adaptive_classification', 'record_adaptive_classification' in content))
for name, result in checks[-2:]:
    print(f"   {name}: {result}")

# 16. parameter_manager.py
with open('parameter_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n16. parameter_manager.py:")
checks.append(('adaptive_mdap params', 'adaptive_mdap_profile' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 17. alerting_system.py
with open('alerting_system.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n17. alerting_system.py:")
checks.append(('adaptive alerts', 'create_adaptive_classification_alert' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 18. reporting_system.py
with open('reporting_system.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n18. reporting_system.py:")
checks.append(('adaptive report', 'generate_adaptive_mdap_report' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 19. c2c_cache_manager.py (NEW)
with open('c2c_cache_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n19. c2c_cache_manager.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 20. performance_optimization.py (NEW)
with open('performance_optimization.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n20. performance_optimization.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 21. distributed_processing.py (NEW)
with open('distributed_processing.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n21. distributed_processing.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 22. plugin_system.py (NEW)
with open('plugin_system.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n22. plugin_system.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 23. content_analyzer.py (NEW)
with open('content_analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n23. content_analyzer.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 24. dependency_analyzer.py (NEW)
with open('dependency_analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n24. dependency_analyzer.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 25. solution_manager.py (NEW)
with open('solution_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n25. solution_manager.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 26. verification_engine.py (NEW)
with open('verification_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n26. verification_engine.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 27. bubblelabs_integration.py (NEW)
with open('bubblelabs_integration.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n27. bubblelabs_integration.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 28. roma_openevolve_integration.py (NEW)
with open('roma_openevolve_integration.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n28. roma_openevolve_integration.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 29. crewai_integration.py (NEW)
with open('crewai_integration.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n29. crewai_integration.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 30. z3_leanaide_bridge.py (NEW)
with open('z3_leanaide_bridge.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n30. z3_leanaide_bridge.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# Summary
print("\n" + "=" * 60)
passed = sum(1 for _, r in checks if r)
total = len(checks)
print(f"VERIFICATION COMPLETE - {passed}/{total} Integration Points")
print("=" * 60)
