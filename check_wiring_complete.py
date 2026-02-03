#!/usr/bin/env python3
"""Complete check for Adaptive MDAP wiring in all 52 files."""

print("=" * 60)
print("ADAPTIVE MDAP WIRING VERIFICATION - 52 INTEGRATION POINTS")
print("=" * 60)

checks = []

# Core Integration Points (12)
print("\n--- CORE INTEGRATION POINTS (12) ---")

# 1. workflow_engine.py
with open('workflow_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n1. workflow_engine.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
checks.append(('get_adaptive_workflow', 'get_adaptive_workflow' in content))
for name, result in checks[-2:]:
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
for name, result in checks[-1:]:
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
for name, result in checks[-2:]:
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
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# Workflow & UI Integration Points (12)
print("\n--- WORKFLOW & UI INTEGRATION POINTS (12) ---")

# 13. openevolve_structures.py
with open('openevolve_structures.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n13. openevolve_structures.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 14. openevolve_visualization.py
with open('openevolve_visualization.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n14. openevolve_visualization.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 15. openevolve_crewai_bridge.py
with open('openevolve_crewai_bridge.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n15. openevolve_crewai_bridge.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 16. openevolve_crewai_delegation.py
with open('openevolve_crewai_delegation.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n16. openevolve_crewai_delegation.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 17. openevolve_decomposition_adapter.py
with open('openevolve_decomposition_adapter.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n17. openevolve_decomposition_adapter.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 18. openevolve_imports.py
with open('openevolve_imports.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n18. openevolve_imports.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 19. openevolve_leanaide_bridge.py
with open('openevolve_leanaide_bridge.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n19. openevolve_leanaide_bridge.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 20. openevolve_validation.py
with open('openevolve_validation.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n20. openevolve_validation.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 21. openevolve_workflow_manager_integrated.py
with open('openevolve_workflow_manager_integrated.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n21. openevolve_workflow_manager_integrated.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 22. openevolve_maker_integration.py
with open('openevolve_maker_integration.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n22. openevolve_maker_integration.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 23. openevolve_leanaide_integration_system.py
with open('openevolve_leanaide_integration_system.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n23. openevolve_leanaide_integration_system.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 24. openevolve_leanaide_workflow_integration.py
with open('openevolve_leanaide_workflow_integration.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n24. openevolve_leanaide_workflow_integration.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# Support System Integration Points (16)
print("\n--- SUPPORT SYSTEM INTEGRATION POINTS (16) ---")

# 25. template_manager.py
with open('template_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n25. template_manager.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 26. session_manager.py
with open('session_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n26. session_manager.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 27. knowledge_base.py
with open('knowledge_base.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n27. knowledge_base.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 28. conflict_detector.py
with open('conflict_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n28. conflict_detector.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 29. sovereign_reliability.py
with open('sovereign_reliability.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n29. sovereign_reliability.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 30. z3_result_cache.py
with open('z3_result_cache.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n30. z3_result_cache.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 31. sovereign_database.py
with open('sovereign_database.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n31. sovereign_database.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 32. validation_manager.py
with open('validation_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n32. validation_manager.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 33. alerting_system.py
with open('alerting_system.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n33. alerting_system.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 34. monitoring.py
with open('monitoring.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n34. monitoring.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 35. reporting_system.py
with open('reporting_system.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n35. reporting_system.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 36. quality_assessment.py
with open('quality_assessment.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n36. quality_assessment.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 37. quality_gate_engine.py
with open('quality_gate_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n37. quality_gate_engine.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 38. quality_tracker.py
with open('quality_tracker.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n38. quality_tracker.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 39. analytics.py
with open('analytics.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n39. analytics.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 40. analytics_manager.py
with open('analytics_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n40. analytics_manager.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# Advanced System Integration Points (12)
print("\n--- ADVANCED SYSTEM INTEGRATION POINTS (12) ---")

# 41. decomposition_engine.py
with open('decomposition_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n41. decomposition_engine.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 42. decomposition_strategy.py
with open('decomposition_strategy.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n42. decomposition_strategy.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 43. verification_engine.py
with open('verification_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n43. verification_engine.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 44. solution_assembler.py
with open('solution_assembler.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n44. solution_assembler.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 45. solution_manager.py
with open('solution_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n45. solution_manager.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 46. problem_analyzer.py
with open('problem_analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n46. problem_analyzer.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 47. problem_classifier.py
with open('problem_classifier.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n47. problem_classifier.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 48. maker_engine.py
with open('maker_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n48. maker_engine.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 49. maker_workflow_integration.py
with open('maker_workflow_integration.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n49. maker_workflow_integration.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 50. maker_integration_bridge.py
with open('maker_integration_bridge.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n50. maker_integration_bridge.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 51. integrated_workflow.py
with open('integrated_workflow.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n51. integrated_workflow.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# 52. workflow_structures.py
with open('workflow_structures.py', 'r', encoding='utf-8') as f:
    content = f.read()
print("\n52. workflow_structures.py:")
checks.append(('ADAPTIVE_MDAP_AVAILABLE', 'ADAPTIVE_MDAP_AVAILABLE' in content))
for name, result in checks[-1:]:
    print(f"   {name}: {result}")

# Summary
print("\n" + "=" * 60)
passed = sum(1 for _, r in checks if r)
total = len(checks)
print(f"VERIFICATION COMPLETE - {passed}/{total} Integration Points")
print("=" * 60)

if passed == total:
    print("\n[PASS] ALL 52 INTEGRATION POINTS VERIFIED!")
else:
    print(f"\n[FAIL] {total - passed} integration points failed verification")
    for name, result in checks:
        if not result:
            print(f"   - {name}")
