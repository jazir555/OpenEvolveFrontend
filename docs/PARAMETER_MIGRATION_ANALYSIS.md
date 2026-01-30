# OpenEvolve Parameter Migration Analysis Report

## Executive Summary
- Total files requiring migration: **129**
- Quick wins (1-2 lines): **19**
- Medium changes (5-10 lines): **30**
- Complex refactors (20+ lines): **80**

## Pattern Breakdown

### old_evolution_import
- **Difficulty:** EASY
- **Files affected:** 23
- **Action:** Replace with openevolve_imports

### old_adversarial_import
- **Difficulty:** EASY
- **Files affected:** 13
- **Action:** Replace with openevolve_imports

### parameter_manager_import
- **Difficulty:** MEDIUM
- **Files affected:** 17
- **Action:** Replace with unified config system

### session_state_usage
- **Difficulty:** MEDIUM
- **Files affected:** 21
- **Action:** Refactor to use ParameterManager directly

### config_dict_usage
- **Difficulty:** EASY
- **Files affected:** 0
- **Action:** Replace with Configuration objects

### direct_params
- **Difficulty:** HARD
- **Files affected:** 80
- **Action:** Migrate to unified configuration

## Quick Wins - Easy Migrations
*Files requiring only simple import statement replacements*

1. **demo_evolution_maker.py**
   - Patterns: old_evolution_import

2. **evolution_adapter.py**
   - Patterns: old_evolution_import

3. **openevolve_imports.py**
   - Patterns: old_evolution_import

4. **suggestions.py**
   - Patterns: old_evolution_import

5. **test_critical_blockers_resolved.py**
   - Patterns: old_evolution_import, old_adversarial_import

6. **test_error_handling.py**
   - Patterns: old_evolution_import

7. **test_leanaide_evolution_mdap.py**
   - Patterns: old_evolution_import

8. **test_missing_dependencies.py**
   - Patterns: old_evolution_import

9. **test_phase1_team_integration.py**
   - Patterns: old_evolution_import, old_adversarial_import

10. **test_session_state_removal.py**
   - Patterns: old_evolution_import, old_adversarial_import

11. **test_team_system_working.py**
   - Patterns: old_evolution_import

12. **test_ultimate_integration.py**
   - Patterns: old_evolution_import, old_adversarial_import

13. **validate_evolution_maker_integration.py**
   - Patterns: old_evolution_import

14. **verify_fix.py**
   - Patterns: old_evolution_import

15. **tests\test_integration.py**
   - Patterns: old_evolution_import, old_adversarial_import

16. **adversarial_adapter.py**
   - Patterns: old_adversarial_import

17. **demo_adversarial_maker.py**
   - Patterns: old_adversarial_import

18. **validate_adversarial_maker_integration.py**
   - Patterns: old_adversarial_import

19. **tests\test_enhanced_adversarial.py**
   - Patterns: old_adversarial_import

## Medium Complexity Migrations
*Files requiring moderate refactoring*

1. **comprehensive_functional_tests.py**
   - Patterns: old_evolution_import, parameter_manager_import

2. **mainlayout.py**
   - Patterns: session_state_usage, old_evolution_import, old_adversarial_import

3. **test_adversarial_evolution_complete.py**
   - Patterns: old_evolution_import, parameter_manager_import

4. **test_evolution_adversarial_basic.py**
   - Patterns: old_evolution_import, parameter_manager_import

5. **test_evolution_comprehensive.py**
   - Patterns: old_evolution_import, parameter_manager_import

6. **test_adversarial_comprehensive.py**
   - Patterns: parameter_manager_import, old_adversarial_import

7. **openevolve_client.py**
   - Patterns: parameter_manager_import

8. **openevolve_workflow_manager_integrated.py**
   - Patterns: parameter_manager_import

9. **sidebar.py**
   - Patterns: session_state_usage, parameter_manager_import

10. **test_integration_openevolve.py**
   - Patterns: parameter_manager_import

11. **test_openevolve_integration.py**
   - Patterns: parameter_manager_import

12. **test_sidebar_parameter_integration.py**
   - Patterns: parameter_manager_import

13. **bubblelabs_integration_tests.py**
   - Patterns: session_state_usage

14. **bubblelabs_leanaide_integration_patch.py**
   - Patterns: session_state_usage

15. **bubblelabs_maker_integration.py**
   - Patterns: session_state_usage

16. **bubblelabs_ui_component.py**
   - Patterns: session_state_usage

17. **configuration_system.py**
   - Patterns: session_state_usage

18. **export_import_manager.py**
   - Patterns: session_state_usage

19. **main.py**
   - Patterns: session_state_usage

20. **monitoring_dashboard.py**
   - Patterns: session_state_usage

*... and 10 more medium complexity files*

## Complex Refactors
*Files requiring significant restructuring*

1. **adversarial.py**
   - Patterns: old_evolution_import, direct_params, parameter_manager_import

2. **integrated_workflow.py**
   - Patterns: session_state_usage, old_evolution_import, direct_params

3. **openevolve_bubblelabs_api.py**
   - Patterns: old_evolution_import, direct_params, parameter_manager_import, old_adversarial_import

4. **evolution.py**
   - Patterns: direct_params, parameter_manager_import, old_adversarial_import

5. **base_configuration.py**
   - Patterns: direct_params, parameter_manager_import

6. **test_adversarial_simple.py**
   - Patterns: direct_params, parameter_manager_import

7. **unified_configuration.py**
   - Patterns: direct_params, parameter_manager_import

8. **adversarial_maker_integration.py**
   - Patterns: direct_params

9. **adversarial_mdap_mcts.py**
   - Patterns: direct_params

10. **adversarial_testing.py**
   - Patterns: direct_params

11. **analytics_monitoring_dashboard.py**
   - Patterns: direct_params

12. **blue_team.py**
   - Patterns: direct_params

13. **bubblelabs_evolution_controls.py**
   - Patterns: direct_params

14. **bubblelabs_leanaide_integration.py**
   - Patterns: direct_params

15. **c2c_mcp_tools.py**
   - Patterns: direct_params

16. **claudiomiro_mcp_tools.py**
   - Patterns: direct_params

17. **config.py**
   - Patterns: direct_params

18. **config_loader.py**
   - Patterns: direct_params

19. **content_analyzer.py**
   - Patterns: direct_params

20. **evaluator_team.py**
   - Patterns: direct_params

*... and 60 more complex files*

