# Complete File Inventory with Risk Assessment

## Legend
- **CRITICAL:** Core system files - highest priority, require careful testing
- **HIGH:** Multiple patterns or complex session state usage
- **MEDIUM:** Direct parameter definitions
- **LOW:** Test files or simple import changes

## CRITICAL RISK (4 files)

1. **adversarial.py**
   - Patterns: old_imports, param_manager, direct_params
   - [CORE FILE]

2. **evolution.py**
   - Patterns: old_imports, param_manager, direct_params
   - [CORE FILE]

3. **base_configuration.py**
   - Patterns: param_manager, direct_params
   - [CORE FILE]

4. **unified_configuration.py**
   - Patterns: param_manager, direct_params
   - [CORE FILE]

## HIGH RISK (1 files)

1. **integrated_workflow.py**
   - Patterns: old_imports, session_state, direct_params

## MEDIUM RISK (75 files)

1. **openevolve_bubblelabs_api.py**
   - Patterns: old_imports, param_manager, direct_params

2. **c2c_mcp_tools.py**
   - Patterns: direct_params, mcp_tools

3. **claudiomiro_mcp_tools.py**
   - Patterns: direct_params, mcp_tools

4. **roma_mdap_maker_mcp_tools.py**
   - Patterns: direct_params, mcp_tools

5. **adversarial_maker_integration.py**
   - Patterns: direct_params

6. **adversarial_mdap_mcts.py**
   - Patterns: direct_params

7. **adversarial_testing.py**
   - Patterns: direct_params

8. **analytics_monitoring_dashboard.py**
   - Patterns: direct_params

9. **blue_team.py**
   - Patterns: direct_params

10. **bubblelabs_evolution_controls.py**
   - Patterns: direct_params

11. **bubblelabs_leanaide_integration.py**
   - Patterns: direct_params

12. **config.py**
   - Patterns: direct_params

13. **config_loader.py**
   - Patterns: direct_params

14. **content_analyzer.py**
   - Patterns: direct_params

15. **deep-research-agent\src\agents.py**
   - Patterns: direct_params

16. **evaluator_team.py**
   - Patterns: direct_params

17. **evolution_maker_integration.py**
   - Patterns: direct_params

18. **evolutionary_optimization.py**
   - Patterns: direct_params

19. **evolve_sop.py**
   - Patterns: direct_params

20. **gauntlet_manager.py**
   - Patterns: direct_params

21. **generic_maker_integration.py**
   - Patterns: direct_params

22. **graphiti\graphiti_core\llm_client\config.py**
   - Patterns: direct_params

23. **hybrid_maker_config.py**
   - Patterns: direct_params

24. **hybrid_maker_integration.py**
   - Patterns: direct_params

25. **hybrid_maker_workflow.py**
   - Patterns: direct_params

26. **hybrid_mcts_framework.py**
   - Patterns: direct_params

27. **integrations\base\experimentation_interface.py**
   - Patterns: direct_params

28. **integrations\curie\adapter.py**
   - Patterns: direct_params

29. **integrations\stage3.py**
   - Patterns: direct_params

30. **invention_planner_integrations.py**
   - Patterns: direct_params

31. **kg-gen\experiments\MINE\_1_evaluation.py**
   - Patterns: direct_params

32. **kg-gen\src\kg_gen\kg_gen.py**
   - Patterns: direct_params

33. **knowledge_engine\engine.py**
   - Patterns: direct_params

34. **leanaide_config.py**
   - Patterns: direct_params

35. **leanaide_evolution.py**
   - Patterns: direct_params

36. **leanaide_evolution_mdap.py**
   - Patterns: direct_params

37. **leanaide_evolution_mdap_workflow.py**
   - Patterns: direct_params

38. **leanaide_evolutionary_workflow.py**
   - Patterns: direct_params

39. **leanaide_hybrid_maker_enhanced.py**
   - Patterns: direct_params

40. **leanaide_hybrid_strategies.py**
   - Patterns: direct_params

41. **leanaide_maker.py**
   - Patterns: direct_params

42. **leanaide_mcts.py**
   - Patterns: direct_params

43. **leanaide_mcts_mdap.py**
   - Patterns: direct_params

44. **leanaide_mcts_mdap_complete.py**
   - Patterns: direct_params

45. **leanaide_mdap.py**
   - Patterns: direct_params

46. **leanaide_selfplay.py**
   - Patterns: direct_params

47. **llm_cache.py**
   - Patterns: direct_params

48. **llm_caching.py**
   - Patterns: direct_params

49. **llm_utils.py**
   - Patterns: direct_params

50. **mcts_coevolution.py**
   - Patterns: direct_params

51. **mcts_coevolution_mdap.py**
   - Patterns: direct_params

52. **mcts_evolutionary_nodes.py**
   - Patterns: direct_params

53. **mcts_evolutionary_nodes_mdap.py**
   - Patterns: direct_params

54. **mcts_evolved_policies.py**
   - Patterns: direct_params

55. **mcts_evolved_policies_mdap.py**
   - Patterns: direct_params

56. **mdap_maker_mcts_unified.py**
   - Patterns: direct_params

57. **metrics_collector.py**
   - Patterns: direct_params

58. **model_orchestration.py**
   - Patterns: direct_params

59. **openevolve\openevolve\config.py**
   - Patterns: direct_params

60. **openevolve_api.py**
   - Patterns: direct_params

61. **openevolve_integration.py**
   - Patterns: direct_params

62. **openevolve_structures.py**
   - Patterns: direct_params

63. **phase2\imech\algorithms\weisfeiler_lehman.py**
   - Patterns: direct_params

64. **phase2\imech\transfer\repair.py**
   - Patterns: direct_params

65. **phase3\mcts_search.py**
   - Patterns: direct_params

66. **psv_selfplay.py**
   - Patterns: direct_params

67. **ragbits_integration\config.py**
   - Patterns: direct_params

68. **red_team.py**
   - Patterns: direct_params

69. **roma_mdap_maker_engine.py**
   - Patterns: direct_params

70. **sop_generator.py**
   - Patterns: direct_params

71. **sop_integrated_system.py**
   - Patterns: direct_params

72. **sovereign_refinement_comprehensive.py**
   - Patterns: direct_params

73. **ui_models.py**
   - Patterns: direct_params

74. **workflow_engine.py**
   - Patterns: direct_params

75. **workflow_structures.py**
   - Patterns: direct_params

## LOW RISK (64 files)

1. **comprehensive_functional_tests.py**
   - Patterns: old_imports, param_manager

2. **mainlayout.py**
   - Patterns: old_imports, session_state

3. **sidebar.py**
   - Patterns: param_manager, session_state

4. **test_adversarial_comprehensive.py**
   - Patterns: old_imports, param_manager
   - [Test file]

5. **test_adversarial_evolution_complete.py**
   - Patterns: old_imports, param_manager
   - [Test file]

6. **test_adversarial_simple.py**
   - Patterns: param_manager, direct_params
   - [Test file]

7. **test_evolution_adversarial_basic.py**
   - Patterns: old_imports, param_manager
   - [Test file]

8. **test_evolution_comprehensive.py**
   - Patterns: old_imports, param_manager
   - [Test file]

9. **ace_mcp_tools.py**
   - Patterns: mcp_tools

10. **ace_mcp_tools_FIXED.py**
   - Patterns: mcp_tools

11. **ace_stage6_integration.py**
   - Patterns: mcp_tools

12. **adversarial_adapter.py**
   - Patterns: old_imports

13. **api_contract_fixes.py**
   - Patterns: mcp_tools

14. **apply_code_quality_fixes.py**
   - Patterns: mcp_tools

15. **apply_phase4_validation.py**
   - Patterns: mcp_tools

16. **bubblelabs_integration_tests.py**
   - Patterns: session_state

17. **bubblelabs_leanaide_integration_patch.py**
   - Patterns: session_state

18. **bubblelabs_maker_integration.py**
   - Patterns: session_state

19. **bubblelabs_mcp_tools.py**
   - Patterns: mcp_tools

20. **bubblelabs_ui_component.py**
   - Patterns: session_state

21. **configuration_system.py**
   - Patterns: session_state

22. **datapizza_mcp_tools.py**
   - Patterns: mcp_tools

23. **decomposition_mcp_tools.py**
   - Patterns: mcp_tools

24. **demo_adversarial_maker.py**
   - Patterns: old_imports

25. **demo_evolution_maker.py**
   - Patterns: old_imports

26. **evolution_adapter.py**
   - Patterns: old_imports

27. **export_import_manager.py**
   - Patterns: session_state

28. **leanaide_mcp_tools.py**
   - Patterns: mcp_tools

29. **main.py**
   - Patterns: session_state

30. **monitoring_dashboard.py**
   - Patterns: session_state

31. **openevolve_client.py**
   - Patterns: param_manager

32. **openevolve_imports.py**
   - Patterns: old_imports

33. **openevolve_mcp_tools.py**
   - Patterns: mcp_tools

34. **openevolve_orchestrator.py**
   - Patterns: session_state

35. **openevolve_workflow_manager_integrated.py**
   - Patterns: param_manager

36. **parameter_sync_manager.py**
   - Patterns: session_state

37. **roma_mcp_tools.py**
   - Patterns: mcp_tools

38. **session_defaults.py**
   - Patterns: session_state

39. **session_state_classes.py**
   - Patterns: session_state

40. **session_utils.py**
   - Patterns: session_state

41. **state.py**
   - Patterns: session_state

42. **steer_mcp_tools.py**
   - Patterns: mcp_tools

43. **suggestions.py**
   - Patterns: old_imports

44. **test_ace_thread_safety.py**
   - Patterns: mcp_tools
   - [Test file]

45. **test_critical_blockers_resolved.py**
   - Patterns: old_imports
   - [Test file]

46. **test_error_handling.py**
   - Patterns: old_imports
   - [Test file]

47. **test_integration_openevolve.py**
   - Patterns: param_manager
   - [Test file]

48. **test_leanaide_evolution_mdap.py**
   - Patterns: old_imports
   - [Test file]

49. **test_missing_dependencies.py**
   - Patterns: old_imports
   - [Test file]

50. **test_openevolve_integration.py**
   - Patterns: param_manager
   - [Test file]

51. **test_phase1_team_integration.py**
   - Patterns: old_imports
   - [Test file]

52. **test_session_state_removal.py**
   - Patterns: old_imports
   - [Test file]

53. **test_sidebar_parameter_integration.py**
   - Patterns: param_manager
   - [Test file]

54. **test_team_system_working.py**
   - Patterns: old_imports
   - [Test file]

55. **test_ultimate_integration.py**
   - Patterns: old_imports
   - [Test file]

56. **tests\integration\test_sovereign_workflow.py**
   - Patterns: session_state
   - [Test file]

57. **tests\test_enhanced_adversarial.py**
   - Patterns: old_imports
   - [Test file]

58. **tests\test_integrated_functionality.py**
   - Patterns: session_state
   - [Test file]

59. **tests\test_integration.py**
   - Patterns: old_imports
   - [Test file]

60. **ui_components.py**
   - Patterns: session_state

61. **ui_utils.py**
   - Patterns: session_state

62. **validate_adversarial_maker_integration.py**
   - Patterns: old_imports

63. **validate_evolution_maker_integration.py**
   - Patterns: old_imports

64. **verify_fix.py**
   - Patterns: old_imports

## Summary Statistics
- Total files needing migration: 144
- Core files: 4
- Test files: 21
- Application files: 119
