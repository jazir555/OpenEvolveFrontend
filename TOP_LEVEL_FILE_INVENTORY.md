# Top-Level File Inventory

> Generated: 2026-08-10 | Original loose files: **1,708** → Now organized: **31** at root
> Purpose: categorize every loose file at the top level of the repo so it can be reviewed, moved, or cleaned up.

## REORGANIZATION COMPLETE

All 1,677 movable files have been organized into labeled folders using `git mv` (history preserved).

### Final Root Structure (31 files)

```
root/
├── .coverage, .env, .env.example, .env.schema
├── .gitignore, .gitlab-ci.yml, .mcp.json
├── CLAUDE.md, README.md, LICENSE
├── Dockerfile, docker-compose*.yml, Makefile, Jenkinsfile
├── package.json, package-lock.json, tsconfig.json, jest.config.js
├── pyproject.toml, pytest.ini, pytest_leanaide.ini
├── requirements.txt, requirements_*.txt
├── mcp.json, mcp_agent.secrets.yaml
└── TOP_LEVEL_FILE_INVENTORY.md (this file)
```

### New Folder Structure

| Folder | Files | Purpose |
|--------|-------|---------|
| `engines/` | 526 | Core platform modules organized by subsystem |
| `integrations/` | 329 | External system integrations by domain |
| `tests/` | 387 | Test files, test runners, test infrastructure |
| `docs/` | 2,047 | Documentation, guides, reports, status docs |
| `scripts/` | 177 | Dev utilities, fix/scan/audit tools, setup scripts |
| `examples/` | 109 | Demos, examples, e2e planners |
| `data/` | 128 | Databases, JSON state, test artifacts |
| `reports/` | 98 | Text reports, status summaries |
| `logs/` | 34 | Log files |
| `archive/` | 65 | Backups, junk files, platform shims |
| `config/` | 12 | YAML/JSON configs |

### engines/ Subfolders

| Subfolder | Files | Purpose |
|-----------|-------|---------|
| `other/` | 230 | Uncategorized engine modules |
| `orchestration/` | 34 | Workflow orchestration, dependencies, batch ops |
| `knowledge/` | 48 | Knowledge graphs, memory, retrieval, caching |
| `observability/` | 29 | Analytics, monitoring, performance, reporting |
| `decomposition/` | 26 | Problem decomposition, recomposition engines |
| `gauntlets/` | 19 | Gauntlet system, evaluation, benchmarking |
| `teams/` | 17 | Red/blue team, evaluator, team coordination |
| `plugins/` | 13 | Plugin system, hybrid plugins |
| `mcts_mdap/` | 12 | MCTS & MDAP evolution engines |
| `workflow/` | 11 | Workflow engine, stages, persistence |
| `domain/` | 11 | Physics, chemistry, finance validators |
| `quality/` | 10 | Quality gates, assessment, metrics |
| `ui/` | 10 | UI components, visualization |
| `security/` | 8 | Auth, RBAC, security framework |
| `sop/` | 8 | SOP generation, templates |
| `e2e_invention/` | 8 | End-to-end invention planners |
| `config/` | 7 | Configuration modules |
| `solutions/` | 6 | Solution assembly, pattern mining |
| `strategies/` | 4 | Strategy templates, builders |
| `deploy/` | 4 | CI/CD, deployment |
| `adaptive/` | 3 | Adaptive decomposition |
| `reliability/` | 3 | Fallback, self-healing |
| `alerting/` | 2 | Notification systems |
| `validation/` | 3 | Input parsing, validation |

### integrations/ Subfolders

| Subfolder | Files | Purpose |
|-----------|-------|---------|
| `other/` | 31 | MCP, steering, misc integrations |
| `z3/` | 31 | Z3 prover, formal verification |
| `openevolve/` | 25 | OpenEvolve platform integration |
| `bubblelabs/` | 22 | BubbleLabs UI & plugin |
| `leanaide/` | 42 | LeanAide & Lean 4 |
| `oneke/` | 18 | OneKE knowledge extraction |
| `crewai/` | 17 | CrewAI orchestration |
| `roma/` | 16 | ROMA decomposition |
| `sovereign/` | 15 | Sovereign system |
| `base/` | 9 | Base integration classes |
| `cognitive_hydraulics/` | 9 | Cognitive hydraulics |
| `neuromancer/` | 8 | Neuromancer integration |
| `icr/` | 6 | ICR integration |
| `dts/` | 6 | DTS integration |
| `guardrails/` | 6 | Guardrails MCP |
| `causal_learn/` | 4 | Causal learning |
| `curie/` | 4 | Curie integration |
| `global_chem/` | 4 | Chemistry |
| `graphiti/` | 4 | Graphiti KG |
| `lmql/` | 4 | LMQL adapter |
| `outlines/` | 4 | Outlines |
| `pygraphistry/` | 4 | PyGraphistry viz |
| `uqtestfuns/` | 4 | UQ test functions |
| `deepke/` | 3 | DeepKE NER |
| `bug_fixes/` | 7 | Bug fix integrations |
| `ai_knowledge_graph/` | 1 | AI KG |

---

## Previous Detailed File Lists (pre-reorganization)

---

## 1. Application Entry Points & Root Configuration (intended at root)

Core app launchers and project-level config that legitimately belong at the root.

- `.coverage`
- `.env`
- `.env.example`
- `.env.schema`
- `.eslintrc.json`
- `.gitignore`
- `.gitlab-ci.yml`
- `.mcp.json`
- `AGENTS.md`
- `app.py`
- `api_server.py`
- `config.yaml`
- `docker-compose.infrastructure.yml`
- `docker-compose.loongflow-core.yml`
- `docker-compose.neo4j.yml`
- `docker-compose.yml`
- `Dockerfile`
- `Jenkinsfile`
- `LICENSE`
- `Makefile`
- `main.py`
- `mcp.json`
- `mcp_agent.secrets.yaml`
- `package.json`
- `package-lock.json`
- `pyproject.toml`
- `pytest.ini`
- `pytest_leanaide.ini`
- `requirements.txt`
- `requirements_integration.txt`
- `requirements_optional.txt`
- `requirements_with_testing.txt`
- `tsconfig.json`
- `jest.config.js`
- `conftest.py`
- `README.md`
- `CLAUDE.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `setup.bat`
- `dev-evolution.cmd`
- `start-p3.bat`
- `mcp_server_requirements.txt`
- `reporting_requirements.txt`
- `ragbits_server_requirements.txt`
- `api_bridge_requirements.txt`
- `leanaide_client_requirements.txt`

---

## 2. Core Application Modules (the actual platform)

Main runtime modules for the evolution/decomposition/workflow engine and UI.

- `evolution.py`
- `evolution.py.backup_batch4` (backup)
- `evolutionary_optimization.py`
- `evolution_adapter.py`
- `evolution_maker_integration.py`
- `evolution_workflow_templates.py`
- `evolution_z3_fitness.py`
- `evolution_adversarial_examples.py`
- `adversarial.py`
- `adversarial_advanced.py`
- `adversarial_maker_integration.py`
- `adversarial_mdap_mcts.py`
- `adversarial_testing.py`
- `adversarial_unified.py`
- `mcts_coevolution.py`
- `mcts_coevolution_mdap.py`
- `mcts_evolutionary_nodes.py`
- `mcts_evolved_policies.py`
- `mcts_evolved_policies_mdap.py`
- `mdap_engine.py`
- `mdap_enhanced_client.py`
- `mdap_memory_bridge.py`
- `mdap_coevolution_examples.py`
- `mdap_maker_associative_integration.py`
- `mdap_maker_complete.py`
- `mdap_maker_gauntlet_integration.py`
- `mdap_maker_matryoshka_integration.py`
- `mdap_maker_mcts_unified.py`
- `adaptive_mdap.py`
- `adaptive_mdap_pes_integration.py`
- `adaptive_mdap_pes_demo.py`
- `adaptive_strategy_integration.py`
- `adaptive_strategy_selector.py`
- `adaptive_decomposition_integration.py`
- `adaptive_gauntlet_system.py`
- `adaptive_learner.py`
- `hybrid_mcts_framework.py`
- `hybrid_strategy_orchestrator.py`
- `hybrid_advanced_plugins.py`
- `hybrid_maker_config.py`
- `hybrid_maker_config_example.py`
- `hybrid_maker_integration.py`
- `hybrid_maker_workflow.py`
- `hybrid_config.py`
- `hybrid_error_handling.py`
- `hybrid_performance.py`
- `hybrid_types.py`
- `generic_maker_integration.py`
- `maker_engine.py`
- `maker_workflow_integration.py`
- `psv_selfplay.py`
- `problem_analyzer.py`
- `problem_analyzer.py.backup2` (backup)
- `problem_classifier.py`
- `problem_decomposition.py`
- `problem_fractal_pipeline.py`
- `problem_recomposition.py`
- `decomposition_engine.py`
- `decomposition_engine_adaptive_enhancement.py`
- `decomposition_engine_lean_enhanced.py`
- `decomposition_strategy.py`
- `decomposition_mcp_tools.py`
- `decomposition_mdap_integration.py`
- `decomposition_recomposition_integration.py`
- `decomposition_z3_validator.py`
- `decomposition_crewai_bridge.py`
- `decomposition_crewai_tools.py`
- `decomposition_dashboard.py`
- `decomposition_config_lean.yaml`
- `comprehensive_decomposition_engine.py`
- `enhanced_decomposition_engine.py`
- `persistent_decomposition_engine.py`
- `universal_decomposition_engine.py`
- `universal_problem_solver.py`
- `universal_recomposition_engine.py`
- `universal_alerting_integration.py`
- `associative_recomposition.py`
- `comprehensive_recomposition_engine.py`
- `enhanced_recomposition_engine.py`
- `verified_recomposition.py`
- `semantic_decomposition.py`
- `spatial_decomposition.py`
- `sub_problem_solver.py`
- `solution_assembler.py`
- `solution_cache.py`
- `solution_manager.py`
- `solution_orchestration.py`
- `solution_pattern_miner.py`
- `solution_validation_pipeline.py`
- `strategies.py`
- `strategy_performance_tracker.py`
- `strategy_templates.py`
- `custom_strategy_builder.py`
- `success_criteria.py`
- `suggestions.py`
- `semantic_analyzer.py`
- `semantic_analyzer.py.backup2` (backup)
- `automated_proof_engine.py`
- `verification_engine.py`
- `verification_methods.py`
- `verification_result.py`
- `verification.py`
- `symbolic_constraint_engine.py`
- `algorithmic_verification.py`
- `truth_package_generator.py`
- `steer_context_engine.py`
- `steer_crewai_bridge.py`
- `steer_mcp_tools.py`
- `steer_rules.yaml`
- `sop_component_system.py`
- `sop_generator.py`
- `sop_generator_enhanced.py`
- `sop_generator_real.py`
- `sop_generator_research_quest.py`
- `sop_integrated_system.py`
- `sop_templates.py`
- `evolve_sop.py`
- `evolve_sop_facets.py`
- `uncertainty_propagation.py`
- `uncertainty_propagation_enhanced.py`
- `uncertainty_propagation_real.py`
- `science_domain_patterns.py` (see scientific_domain_patterns.py)
- `scientific_domain_patterns.py`
- `domain_configurations.py`
- `domain_optimization_manager.py`
- `tripartite_production.py`
- `scalability_improvements.py`
- `parallel_processing.py`
- `distributed_processing.py`
- `distributed_z3_solver_pool.py`
- `thread_safety_utils.py`
- `batch_operations.py`
- `advanced_features.py`
- `advanced_validation_workflows.py`
- `future_enhancements.py`
- `execution_sandbox.py`
- `execution_types.py`
- `robustness_integration.py`
- `self_healing_mechanism.py`
- `reliability.py`
- `reliability_config.py`
- `fallback_handler.py`
- `graceful_degradation_*` (docs only, see docs section)

---

## 3. Workflow & Orchestration System

- `workflow_engine.py`
- `workflow_enhanced_stages.py`
- `workflow_history_manager.py`
- `workflow_knowledge_extractor.py`
- `workflow_lifecycle_controller.py`
- `workflow_orchestrator.py`
- `workflow_persistence.py`
- `workflow_stage_functions.py`
- `workflow_stage_z3.py`
- `workflow_state_manager.py`
- `workflow_structures.py`
- `workflow_templates.py`
- `workflow_visualization.py`
- `workflow_adapter.py`
- `workflow_automation.py`
- `integrated_workflow.py`
- `integrated_workflow.py.backup_batch2c` (backup)
- `enhanced_stages_integration.py`
- `master_integration_system.py`
- `master_test_runner.py`
- `service_orchestrator.py`
- `model_orchestration.py`
- `learning_loop_manager.py`
- `task_runner.py` (n/a)
- `tasks.py`
- `event_bus.py`
- `collaboration.py`
- `collaboration_manager.py`
- `conflict_detector.py`
- `conflict_detector_examples.py`
- `process_optimization.py`
- `export_import_manager.py`
- `integrations.py`
- `integration_config.py`
- `master_integration_system.py` (see above)
- `ultimate_integration_bridge.py`
- `working_integration_bridge.py`
- `integration_and_performance_tests.py` (test)
- `system_integration_validation.py` (test)

---

## 4. Team System (Red / Blue / Gold, Gauntlet, Evaluation)

- `red_team.py`
- `red_team_coordinator.py`
- `red_team_feedback_system.py`
- `blue_team.py`
- `blue_team_performance_integration.py`
- `blue_team_performance_tracker.py`
- `blue_team_solver_engine.py`
- `blue_team_tools.py`
- `blue_team_utilities.py`
- `blue_team_z3_validator.py`
- `evaluator_team.py`
- `evaluator_team_coordinator.py`
- `evaluator_analytics.py`
- `evaluator_config.py`
- `evaluator_reporter.py`
- `evaluator_uploader.py`
- `team_assignment_engine.py`
- `team_base.py`
- `team_manager.py`
- `team_performance_tracker.py`
- `critique_aggregator.py`
- `critique_aggregator_examples.py`
- `gauntlet_manager.py`
- `gauntlet_orchestrator.py`
- `gauntlet_system.py`
- `gauntlet_server.py`
- `gauntlet_types.py`
- `gauntlet_structures.py`
- `gauntlet_config.py`
- `gauntlet_metrics.py`
- `gauntlet_solver.py`
- `gauntlet_benchmarks.py`
- `gauntlet_evaluator.py`
- `gauntlet_test_data.py`
- `gauntlet_effectiveness_analyzer.py`
- `gauntlet_decomposition_integration.py`
- `gauntlet_integration.py`
- `gauntlet_pipeline_checkpointed.py`
- `formal_gauntlet_system.py`
- `enhanced_gauntlet_manager.py`
- `dynamic_gauntlet_adaptation.py`
- `predictive_gauntlet_executor.py`
- `sovereign_gauntlets.py`
- `multi_round_testing.py`
- `quality_assessment.py`
- `quality_assurance.py`
- `quality_calculator.py`
- `quality_metrics.py`
- `quality_tracker.py`
- `quality_control_examples.py`
- `quality_enhancement.py`
- `quality_enhancer.py`
- `quality_gate_leanaide_verifier.py`
- `quality_gate_z3_verifier.py`
- `enhanced_quality_methods.py`
- `sovereign_quality_assessment.py`
- `sovereign_refinement.py`
- `sovereign_validation.py`
- `sovereign_team_coordination.py`
- `sovereign_solution_orchestration.py`
- `sovereign_persistence.py`
- `sovereign_integration.py`
- `sovereign_reliability.py`
- `sovereign_performance.py`
- `sovereign_performance_optimization.py`
- `sovereign_database.py`
- `sovereign_data_models.py`
- `sovereign_knowledge_manager.py`
- `sovereign_knowledge_manager.py.backup2` (backup)
- `sovereign_problem_analyzer.py`
- `sovereign_decomposition_strategy.py`
- `sovereign_decomposition_crewai_integration.py`
- `sovereign_sidebar_integration.py`
- `sovereign_ui.py`
- `sovereign_ui_components.py`
- `sovereign_decomposition.db` (data)

---

## 5. Knowledge, Memory & Retrieval Systems

- `knowledge_base.py`
- `knowledge_base_ui.py`
- `knowledge_manager.py`
- `knowledge_storage.py`
- `knowledge_extractor.py`
- `knowledge_artifact_extractor.py`
- `knowledge_context_assembler.py`
- `knowledge_engine_hierarchical_integration.py`
- `knowledge_engine_icr_integration.py`
- `knowledge_engine_orchestrator.py`
- `knowledge_graph.py`
- `knowledge_graph_index.py`
- `knowledge_graph_index.db` (data)
- `knowledge_graph_reasoning_integration.py`
- `knowledge_graph_visualizer.py`
- `knowledge_graph_visualizer_pygraphistry.py`
- `knowledge_graph_z3_connector.py`
- `knowledge_hash_index.py`
- `knowledge_hash_index.db` (data)
- `knowledge_hierarchical_index.py`
- `knowledge_hierarchical_index.db` (data)
- `knowledge_hybrid_retrieval.py`
- `knowledge_indexing_config.py`
- `knowledge_lifecycle_manager.py`
- `knowledge_performance_integration.py`
- `knowledge_semantic_index.py`
- `knowledge_state_manager.py`
- `knowledge_unified_memory_system.py`
- `knowledge_working_memory.py`
- `knowledge_artifacts.db` (data)
- `unified_knowledge_extraction.py`
- `unified_kg.py`
- `unified_kg_integration_hub.py`
- `unified_knowledge_platform.py`
- `external_knowledge_integration.py`
- `stage6_knowledge_extraction.py`
- `knowledge_engine_icr_integration.py` (see above)
- `chronicle_memory.py`
- `chronicle_memory_z3_integration.py`
- `memory_agent.py`
- `matryoshka_enhanced_client.py`
- `matryoshka_execution_engine.py`
- `matryoshka_unified_memory_integration.py`
- `unified_matryoshka_architecture.py`
- `ml_pattern_clustering.py`
- `ground_truth_store.py`
- `ground_truth_store.json` (data)
- `vector_search.py`
- `vector_store.py`
- `langchain_chroma_integration.py`
- `c2c_cache_manager.py`
- `c2c_mcp_tools.py`
- `c2c_usage_examples.py`
- `llm_cache.py`
- `llm_caching.py`
- `llm_cache.db` (data)
- `unified_caching.py`
- `z3_result_cache.py`
- `ragbits_server.py`
- `ragbits_document_processor_example.py`
- `valkey_integration.py`
- `external_knowledge_integration.py` (see above)
- `physics_knowledge_engine.py`
- `smart_contract_domain_knowledge.py`
- `enhanced_knowledge_core.py`
- `knowledge_engine_import_issues.txt` (report)
- `DUPLICATE_KNOWLEDGE_ENGINE_RESOLUTION.md` (docs)
- `DUPLICATE_MERGE_COMPLETE_REPORT.md` (docs)

---

## 6. OpenEvolve Integration Layer

- `openevolve_analytics.py`
- `openevolve_api.py`
- `openevolve_bubblelabs_api.py`
- `openevolve_bubblelabs_plugin.py`
- `openevolve_bubblelabs_ui.py`
- `openevolve_cli.py`
- `openevolve_client.py`
- `openevolve_crewai_adapter.py`
- `openevolve_crewai_bridge.py`
- `openevolve_crewai_delegation.py`
- `openevolve_dashboard.py`
- `openevolve_decomposition_adapter.py`
- `openevolve_enhanced_decomposition_integration.py`
- `openevolve_evolution_integration.py`
- `openevolve_imports.py`
- `openevolve_integration.py`
- `openevolve_integration_library.py`
- `openevolve_integrations.py`
- `openevolve_knowledge_integration.py`
- `openevolve_leanaide_bridge.py`
- `openevolve_leanaide_integration_system.py`
- `openevolve_leanaide_workflow_integration.py`
- `openevolve_maker_integration.py`
- `openevolve_mcp_tools.py`
- `openevolve_orchestrator.py`
- `openevolve_pes_integration.py`
- `openevolve_structures.py`
- `openevolve_unified_math_service.py`
- `openevolve_validation.py`
- `openevolve_visualization.py`
- `openevolve_workflow_manager.py`
- `openevolve_workflow_manager_integrated.py`
- `openevolve_workflow_mcp_tools.py`
- `openevolve_metrics.db` (data)

---

## 7. LeanAide / Lean 4 Integration

- `lean4_integration.py`
- `lean4_integration_enhanced.py`
- `lean4_true_100_integration.py`
- `lean_bootstrap.py`
- `lean_type_theory.py`
- `leanaide.py`
- `leanaide_adversarial.py`
- `leanaide_api_routes.py`
- `leanaide_bubblelab_integration.py`
- `leanaide_client.py`
- `leanaide_config.py`
- `leanaide_config.example.yaml`
- `leanaide_continuous_math.py`
- `leanaide_continuous_mcp.py`
- `leanaide_crewai_bridge.py`
- `leanaide_decomposition_integration.py`
- `leanaide_evolution.py`
- `leanaide_evolution_mdap.py`
- `leanaide_evolution_mdap_workflow.py`
- `leanaide_hybrid_maker_enhanced.py`
- `leanaide_hybrid_strategies.py`
- `leanaide_integration.py`
- `leanaide_integration_complete.py`
- `leanaide_knowledge_extraction.py`
- `leanaide_maker.py`
- `leanaide_mcp_tools.py`
- `leanaide_mcts.py`
- `leanaide_mcts_mdap.py`
- `leanaide_mcts_mdap_complete.py`
- `leanaide_mcts_strategies.py`
- `leanaide_mdap.py`
- `leanaide_mdap_demo.py`
- `leanaide_mdap_workflow.py`
- `leanaide_pes_benchmark.py`
- `leanaide_pes_handler.py`
- `leanaide_predictive_flagging.py`
- `leanaide_production_connector.py`
- `leanaide_proof_checker.py`
- `leanaide_proof_integration.py`
- `leanaide_real_connector.py`
- `leanaide_redflagging_system.py`
- `leanaide_rese_workflow.py`
- `leanaide_selfplay.py`
- `leanaide_sop_integration.py`
- `leanaide_strategies.py`
- `leanaide_systems.py`
- `leanaide_web3_status.py`
- `leanaide_workflow_integration.py`
- `mathlib4_integration.py`
- `activate_lean_integration.py`
- `setup_lean4.py`
- `setup_lean4_enhanced.py`
- `zero_touch_lean_setup.py`
- `audit_lean_files.py`
- `leanaide_files.txt` (report)
- `examples_leanaide_selfplay.py` (demo)

---

## 8. Z3 / Formal Verification Integration

- `z3_api.py`
- `z3_api_server.py`
- `z3_auto_extraction.py`
- `z3_bubblelabs_advanced_ui.py`
- `z3_canonicalizer.py`
- `z3_cav_nlp_integration.py`
- `z3_cli.py`
- `z3_config.yaml`
- `z3_config_manager.py`
- `z3_crewai_bridge.py`
- `z3_database_models.py`
- `z3_enhanced_knowledge.py`
- `z3_knowledge_complete.py`
- `z3_knowledge_extraction.py`
- `z3_knowledge_integration.py`
- `z3_knowledge.db` (data)
- `z3_leanaide_bridge.py`
- `z3_leanaide_bubblelabs_ui.py`
- `z3_leanaide_bubbles.py`
- `z3_leanaide_openevolve_integration.py`
- `z3_mcp_tools.py`
- `z3_performance_monitor.py`
- `z3_reliability_checker.py`
- `z3_result_cache.py`
- `z3_semantic_synthesis.py`
- `z3_solver_connector.py`
- `z3_solver_pool.py`
- `z3_to_lean_integration.py`
- `z3_to_lean_invention_integration.py`
- `z3_validated_ir.py`
- `z3prover_advanced.py`
- `z3prover_integration.py`
- `advanced_nl_to_z3_converter.py`
- `expand_z3_verification.py`
- `deploy_z3_service.py`
- `z3_cache.db` (data)
- `enhanced_z3_to_lean_integration.py`
- `robust_z3_leanaide_integration.py`
- `definitive_ssv_insolvency_proof.py`
- `definitive_ssv_proof.py`
- `web3_formal_evidence.py`
- `web3_validator_tool.py`
- `SSV_DEFINITIVE_PROOF.json` (data)
- `SSV_FORMAL_PROOF_CERTIFICATE.json` (data)

---

## 9. BubbleLabs / BubbleLab UI Integration

- `bubblelab-auto-setup.py`
- `bubblelab-auto-setup-v1-backup.py`
- `bubblelab-auto-setup-v2.py`
- `bubblelab-auto-setup-v3.py`
- `bubblelab_crewai_mcp_server.py`
- `bubblelab_mcp_client.py`
- `bubblelabs_analytics.py`
- `bubblelabs_analytics.db` (data)
- `bubblelabs_automation.py`
- `bubblelabs_crewai_bridge.py`
- `bubblelabs_evolution_controls.py`
- `bubblelabs_evolution_integration.py`
- `bubblelabs_evolution_ui_patch.py`
- `bubblelabs_extended_integration.py`
- `bubblelabs_gauntlet_bubbles.py`
- `bubblelabs_integration.py`
- `bubblelabs_integration_tests.py`
- `bubblelabs_knowledge_integration.py`
- `bubblelabs_leanaide_diagram.py`
- `bubblelabs_leanaide_examples.py`
- `bubblelabs_leanaide_integration.py`
- `bubblelabs_leanaide_integration_patch.py`
- `bubblelabs_leanaide_ui.py`
- `bubblelabs_maker_integration.py`
- `bubblelabs_mcp_tools.py`
- `bubblelabs_mcp_tools_security_patch.py`
- `bubblelabs_node_completion.py`
- `bubblelabs_plugin_system.py`
- `bubblelabs_ragbits_bubbles.py`
- `bubblelabs_security.py`
- `bubblelabs_ui_component.py`
- `bubblelabs_validation.py`
- `start_bubblelabs_integration.py`
- `live_web_interface.py`
- `bubble_inventory.json` (data)
- `BUBBLELABS_TEST_RESULTS.txt` (report)
- `BUBBLELABS_TEST_SUMMARY.txt` (report)
- `BUBBLE_INVENTORY_DELIVERABLES.txt` (report)
- `message_display.py`

---

## 10. ROMA Integration

- `roma_associative_integration.py`
- `roma_config.py`
- `roma_config_helper.py`
- `roma_crewai_bridge.py`
- `roma_crewai_tools.py`
- `roma_decomposition_advanced.py`
- `roma_decomposition_basic.py`
- `roma_decomposition_comparison.py`
- `roma_decomposition_hybrid.py`
- `roma_entity_kg.py`
- `roma_entity_kg_integration.py`
- `roma_integration.py`
- `roma_matryoshka_adapter.py`
- `roma_matryoshka_integration.py`
- `roma_mcp_tools.py`
- `roma_mdap_maker_associative_integration.py`
- `roma_mdap_maker_config.py`
- `roma_mdap_maker_crewai_bridge.py`
- `roma_mdap_maker_crewai_tools.py`
- `roma_mdap_maker_engine.py`
- `roma_mdap_maker_mcp_tools.py`
- `roma_mdap_maker_reliability_ssot.py`
- `roma_openevolve_integration.py`
- `roma_recomposition_config.py`
- `roma_reliability_ssot.py`
- `roma_types.py`
- `complete_roma_mdap_maker_integration.py`
- `demonstrate_roma_improvements.py` (demo)
- `demo_roma_mdap_maker.py` (demo)

---

## 11. CrewAI Integration

- `crewai_api_routes.py`
- `crewai_client.py`
- `crewai_config_fix.py`
- `crewai_enhanced_decomposition_bridge.py`
- `crewai_hub.py`
- `crewai_integration.py`
- `crewai_integration_complete.py`
- `crewai_integration_layer.py`
- `crewai_mdap_integrator.py`
- `crewai_mdap_maker_engine.py`
- `crewai_research_core.py`
- `crewai_research_enhanced.py`
- `crewai_research_external.py`
- `crewai_research_templates.py`
- `crewai_research_tools.py`
- `crewai_state_management.py`
- `crewai_unified_bridge.py`
- `crewai_unified_flow.py`
- `crewai_zero_error_workflow.py`
- `fixed_crewai_bridges.py`
- `icr_crewai_integration.py`
- `icr_gap_analysis.py`
- `icr_integration.py`
- `example_crewai_delegation.py` (demo)
- `demo_crewai_research_features.py` (demo)
- `CREWAI_REMOVAL_ROADMAP.md` (docs)

---

## 12. Other External System Integrations

- `datapizza_api_server.py`
- `datapizza_config.py`
- `datapizza_crewai_bridge.py`
- `datapizza_mcp_tools.py`
- `claudiomiro_config.py`
- `claudiomiro_crewai_bridge.py`
- `claudiomiro_mcp_tools.py`
- `cav_nlp_integration.py`
- `dspy_integration.py`
- `dts_integration.py`
- `lmql_adapter.py`
- `lmql_mcp_tools.py`
- `deepke.py`
- `causal_learn_integration.py`
- `neuralkg_integration.py`
- `math_api_complete.py`
- `math_knowledge_cli.py`
- `math_knowledge_config.py`
- `math_mcp_tools.py`
- `continuous_math_detector.py`
- `enhanced_math_detector.py`
- `complete_continuous_math.py`
- `openevolve_unified_math_service.py`
- `unified_math_service.py`
- `unified_math_bridge_complete.py`
- `unified_math_knowledge_bridge.py`
- `unified_evolution_api.py`
- `unified_evolution_integration.py`
- `unified_manager.py`
- `unified_mcp_gateway.py`
- `migrate_to_unified_mcp.py`
- `mcp_bridge.py`
- `mcp_gateway.py`
- `mcp_gateway_integration.py`
- `mcp_import_fix.py`
- `mcp_server.py`
- `sop_integration.py` (n/a)
- `providercatalogue.py`
- `providers.py`
- `model_orchestration.py` (see section 3)
- `github_config.py`
- `global_chem.py`
- `smart_contract_exploit_solver.py`
- `smart_contract_logic_analyzer.py`
- `read_pdf.py`
- `async_pattern_verification.py`
- `langchain_chroma_integration.py` (see section 5)
- `setup_deepke.py`
- `setup_oneke.py`
- `setup_integration.py`
- `setup_ace_dependencies.py`
- `install_optionals_for_100.py`
- `initialize_integrations.py`
- `add_integration_flags.py`
- `refactor_evolution_config.py`
- `migrate_adversarial.py`
- `migrate_phase2_remaining.py`
- `migrate_tests_batch4.py`
- `migration_report.py`
- `migrations.py`
- `backup_restore.py`
- `plugin_registry.py`
- `plugin_system.py`
- `template_manager.py`
- `add_integration_flags.py` (see above)
- `unified_configuration.py`
- `configuration_manager.py`
- `configuration_system.py`
- `config_data.py`
- `config_loader.py`
- `config_provider.py`
- `config_validation.py`
- `config.py` (core config)

---

## 13. API / Server / Web Endpoints

- `api_bridge.py`
- `api_contract_fixes.py`
- `api_gateway.py`
- `api_key_manager.py`
- `api_keys.py`
- `api_routes.py`
- `api_keys.db` (data)
- `secure_api.py`
- `health_checks.py`
- `health_endpoint.py`
- `auth_system.py`
- `rate_limiting.py`
- `rbac_enhanced.py`
- `input_parser.py`
- `input_sanitizer.py`
- `input_validation.py`
- `websocket_manager.py`
- `webhook_manager.py`
- `frontend_health_check.py`
- `frontend_utils.py`
- `system_health.py`
- `live_web_interface.py` (see section 9)
- `start_rese_health_apis.py`
- `deploy_z3_service.py` (see section 8)

---

## 14. UI Components & Layout

- `ui_components.py`
- `ui_components_additional.py`
- `ui_config.py`
- `ui_models.py`
- `ui_shim.py`
- `ui_utils.py`
- `mainlayout.py`
- `layout.py`
- `sidebar.py`
- `sessionstate.py`
- `session_defaults.py`
- `session_manager.py`
- `session_state_classes.py`
- `session_utils.py`
- `state.py`
- `styles.css`
- `dashboard_ui_components.py`
- `advanced_visualization.py`
- `progress_visualizer.py`
- `interactive_visualizer.py`
- `form_handling.py`
- `notification_system.py`
- `notifications.py`
- `suggestions.py` (see section 2)
- `message_display.py` (see section 9)

---

## 15. Analytics, Monitoring, Telemetry & Performance

- `analytics.py`
- `analytics_dashboard.py`
- `analytics_data.py`
- `analytics_manager.py`
- `analytics_monitoring_dashboard.py`
- `analytics_z3_connector.py`
- `ace_analytics.py`
- `metrics_collector.py`
- `metrics.db` (data)
- `monitoring.py`
- `monitoring_dashboard.py`
- `monitoring_system.py`
- `advanced_sgd_monitoring.py`
- `telemetry.py`
- `tracing.py`
- `log_streaming.py`
- `logging_util.py`
- `logger_utils.py` (n/a)
- `performance_metrics_tracker.py`
- `performance_optimization.py`
- `performance_optimizations.py`
- `performance_profiler.py`
- `performance_utils.py`
- `performance_improvements.py` (n/a)
- `benchmark_improvements.py`
- `benchmark_integrations.py`
- `benchmark_knowledge_artifact_generation.py`
- `benchmark_knowledge_artifacts_extended.py`
- `benchmark_ultra_comprehensive_artifacts.py`
- `scheduled_reports.py`
- `report_generator.py`
- `report_templates.py`
- `report_templates.json` (data)
- `reporting_demo.py`
- `reporting_system.py`
- `integrated_reporting.py`
- `coverage_tracking.py`
- `strategy_performance_tracker.py` (see section 2)
- `progress_tracking.py`
- `todo_tracker.py`
- `vision_language_monitor.py`
- `openevolve.log` (log)
- `model_performance.json` (data)
- `user_preferences.json` (data)
- `alerts.json` (data)
- `teams.json` (data)
- `gauntlets.json` (data)
- `performance_*` docs (see docs section)

---

## 16. Security System

- `security_framework.py`
- `security_helpers.py`
- `security_layer.py`
- `security_utils.py`
- `security_verification.py`
- `bubblelabs_security.py`
- `ace_security_utils.py`
- `real_audit_logging_tests.py` (test)
- `real_rate_limiting_tests.py` (test)
- `real_security_headers_tests.py` (test)
- `real_security_tests.py` (test)
- `real_sql_injection_tests.py` (test)
- `real_xss_prevention_tests.py` (test)
- `security_tests.py` (test)
- `security_test_suite.py` (test)
- `security_integration_tests.py` (test)
- `security_performance_tests.py` (test)
- `comprehensive_security_test_coverage.py` (test)
- `run_security_tests.py` (test runner)
- `run_real_security_tests.py` (test runner)
- `run_security_true_100_tests.py` (test runner)
- `run_security_fixes.bat`
- `rbac_enhanced_tests.py` (test)
- `security_analysis_20260120_231444.json` (report data)
- `security_analysis_20260120_231708.json` (report data)
- `security_fix_log_20260120_231039.log` (log)
- `security_fix_log_20260120_231703.log` (log)
- `test.local_cert.pem` (test cert)
- `test.local_key.pem` (test key)

---

## 17. Tests & Test Infrastructure (Python)

- `additional_unit_tests.py`
- `additional_verification_test.py`
- `advanced_system_unit_tests.py`
- `advanced_unit_tests_comprehensive.py`
- `categorize_tests.py`
- `clean_final_verification_test.py`
- `comprehensive_final_test.py`
- `comprehensive_functional_tests.py`
- `comprehensive_import_test.py`
- `comprehensive_integration_test.py`
- `comprehensive_openevolve_test.py`
- `comprehensive_system_test.py`
- `comprehensive_test_suite.py`
- `comprehensive_validation_tests.py`
- `comprehensive_verification_report.py`
- `edge_case_tests.py`
- `extended_unit_tests.py`
- `extra_comprehensive_tests.py`
- `federation_smoke_test.py`
- `final_api_test.py`
- `final_comprehensive_import_test.py`
- `final_import_test.py`
- `final_integration_verification.py`
- `gauntlet_tests.py`
- `generate_bubble_tests.py`
- `integration_test.py`
- `massive_import_test.py`
- `prove_it_works.py`
- `quick_integration_test.py`
- `quick_pygraphistry_test.py`
- `quick_test.py`
- `quick_test_integration.py`
- `quick_true_100_check.py`
- `quick_verify.py`
- `regression_test_4_files.py`
- `simple_check.py`
- `simple_demo.py`
- `simple_dspy_check.py`
- `simple_dspy_test.py`
- `simple_test.py`
- `simple_test_clean.py`
- `simple_verification_test.py`
- `simple_verify_implementation.py`
- `system_test.py`
- `test_adapter_integration.js` (JS)
- `test_api_keys.py`
- `test_business_logic.py`
- `test_comprehensive_final.py`
- `test_concurrent_evolutions.py`
- `test_crewai_integration.py`
- `test_crewai_integration_complete.py`
- `test_deepseek_complete.py`
- `test_deepseek_gauntlet.py`
- `test_deepseek_multiagent.py`
- `test_deepseek_roles.py`
- `test_e2e_simple.js` (JS)
- `test_enhancement_3_advanced_nl.py`
- `test_enhancement_4_distributed_z3.py`
- `test_formalization_levels_final.py`
- `test_gap_fixes_comprehensive.py`
- `test_icr_e2e.ts` (TS)
- `test_icr_integration_comprehensive.py`
- `test_import_fixes_phase1.py`
- `test_imports_batch2.py`
- `test_imports_batch2_final.py`
- `test_imports_batch2_robust.py`
- `test_imports_check.py`
- `test_leanaide_mcts_mdap.py`
- `test_new_zai.py`
- `test_phase2_evolution.py`
- `test_roma_business_logic.py`
- `test_sanitize.js` (JS)
- `test_valkey_persistence.py`
- `test_z3_lean_invention_integration.py`
- `test_z3_lean_invention_planner_integration.py`
- `test_z3_lean_quick.py`
- `test_z3_live_proof.py`
- `test_zai.py`
- `test_zai_litellm.py`
- `test_zhipu_all.py`
- `test_zhipu_direct.py`
- `test_zhipu_final.py`
- `test_zhipu_formats.py`
- `test_zhipu_models.py`
- `test_zhipu_sdk.py`
- `test_zhipu_simple.py`
- `test-core-components.js` (JS)
- `test-functionality.js` (JS)
- `test-functionality.ts` (TS)
- `test-reliability-fixes.ts` (TS)
- `test-workflows.js` (JS)
- `testing_framework.py`
- `thorough_integration_test.py`
- `thorough_verification.py`
- `ultimate_comprehensive_tests.py`
- `ultimate_import_verification.py`
- `ultimate_validation.py`
- `ultra_comprehensive_tests.py`
- `comprehensive_phase1_verification.py` (verifier)
- `bubblelabs_integration_tests.py`
- `quick_syntax_fix.py`
- `syntax_checker.py`
- `detailed_syntax_check.py`
- `generate_certificate.py`

---

## 18. Test Runners & Harnesses

- `run_agents_config_tests.py`
- `run_all_ace_tests.py`
- `run_all_batch2_tests.py`
- `run_all_gauntlet_tests.py`
- `run_all_tests.py`
- `run_batch2_validation.bat`
- `run_evolution_mdap_tests.py`
- `run_evolutionary_tests.py`
- `run_full_rese_e2e_pipeline.py`
- `run_gauntlet_tests.py`
- `run_import_test_batch3.py`
- `run_import_test_batch4.py`
- `run_integration_tests.py`
- `run_leanaide_tests.py`
- `run_mcts_mdap_tests.py`
- `run_mcts_tests.py`
- `run_mdap_tests.py`
- `run_real_security_tests.py`
- `run_rese_tests.py`
- `run_security_tests.py`
- `run_security_true_100_tests.py`
- `run_tests.py`
- `run_top_level_fixes.bat`
- `RUN_ALL_VALIDATION_TESTS.sh`
- `run_security_fixes.bat`
- `master_test_runner.py` (see section 3)

---

## 19. Validation & Verification Scripts (one-shot)

- `validate_adversarial_maker_integration.py`
- `validate_all_fixes.py`
- `validate_end_to_end_invention.py`
- `validate_enhanced_adversarial.py`
- `validate_evolution_maker_integration.py`
- `validate_generic_maker_integration.py`
- `validate_hybrid_maker_integration.py`
- `validate_imports.py`
- `validate_integration.py`
- `validate_leanaide_tests.py`
- `validate_maker_integration.py`
- `validate_performance.py`
- `validate_phase1_complete.py`
- `validate_production_ready.py`
- `validate_ragbits_integration.py`
- `validate_sop_components.py`
- `validate_sop_generator.py`
- `validate_sop_integrated.py`
- `validate_task_15.py`
- `validate_task_16.py`
- `validation.py`
- `validation_manager.py`
- `verify_additional_math_bubbles.py`
- `verify_all_lean_wiring.py`
- `verify_bubblelabs_integration.py`
- `verify_causal_learn_final.py`
- `verify_complete_integration.py`
- `verify_crewai_changes.py`
- `verify_dts_imports.py`
- `verify_dts_integration.py`
- `verify_final_imports.py`
- `verify_gauntlet_system_complete.py`
- `verify_gauntlet_wiring.py`
- `verify_global_dspy_integration.py`
- `verify_icr_integration.js` (JS)
- `verify_integration.py`
- `verify_integrations.py`
- `verify_knowledge_engine.py`
- `verify_knowledge_extraction.py`
- `verify_lean_integration.py`
- `verify_lean_wiring.py`
- `verify_leanaide_integration.py`
- `verify_leanaide_true_100.py`
- `verify_math_bubbles.py`
- `verify_mcp.py`
- `verify_mdap_maker_gauntlet_integration.py`
- `verify_rese_health_apis.py`
- `verify_roma_fix.py`
- `verify_ssv_insolvency.py`
- `verify_stubs.py`
- `verify_system_wiring.py`
- `verify_true_100_integration.py`
- `verify_true_100_knowledge_extraction.py`
- `verify_web3_integration.py`
- `true_100_verification.py`
- `final_import_verification.py`
- `final_import_verification_fast.py`
- `ultimate_import_verification.py`

---

## 20. Demos & Examples

- `comprehensive_demo.py`
- `demo_adversarial_maker.py`
- `demo_app.py`
- `demo_crewai_research_features.py`
- `demo_database_cleanup.py`
- `demo_e2e_invention_enhanced.py`
- `demo_end_to_end_invention.py`
- `demo_enhanced_adversarial.py`
- `demo_enhanced_decomposition_recomposition.py`
- `demo_evolution_maker.py`
- `demo_evolution_mdap.py`
- `demo_evolutionary_tests.py`
- `demo_generic_maker.py`
- `demo_hierarchical_indexing.py`
- `demo_hybrid_maker.py`
- `demo_hybrid_mcts.py`
- `demo_integration.py`
- `demo_knowledge_extraction_ml.py`
- `demo_leanaide_autoformalization_mdap_maker.py`
- `demo_leanaide_client.py`
- `demo_leanaide_config.py`
- `demo_leanaide_redflagging.py`
- `demo_maker_complete.py`
- `demo_matryoshka_auto.py`
- `demo_matryoshka_unified_memory.py`
- `demo_mcts.py`
- `demo_mcts_mdap.py`
- `demo_mdap_maker.py`
- `demo_mdap_maker_matryoshka.py`
- `demo_mdap_maker_mcts_unified.py`
- `demo_openevolve_bubblelabs.py`
- `demo_openevolve_integration.py`
- `demo_openevolve_pes_integration.py`
- `demo_pes_workflow.py`
- `demo_pes_workflow_universal.py`
- `demo_problem_classifier.py`
- `demo_quality_calculator.py`
- `demo_reliability_system.py`
- `demo_roma_mdap_maker.py`
- `demo_sop_components.py`
- `demo_sop_generator.py`
- `demo_sop_integrated.py`
- `demo_team_assignment.py`
- `demo_ui_integration.py`
- `demo_unified_memory_system.py`
- `demo_web3_audit.py`
- `demo_z3_leanaide_integration.py`
- `demonstrate_benchmark_scoring.py`
- `demonstrate_roma_improvements.py`
- `demonstrate_scoring_simple.py`
- `example_decomposition_integration.py`
- `example_enhanced_decomposition.py`
- `example_integration_usage.py`
- `examples_subproblem_classifier.py`
- `example_crewai_delegation.py`
- `create_sample_report.py`
- `evolve_sop.py` (see section 2)
- `quality_control_examples.py`
- `conflict_detector_examples.py`
- `evolution_adversarial_examples.py`
- `mdap_coevolution_examples.py`
- `end_to_end_invention_planner.py`
- `end_to_end_invention_planner_agent2.py`
- `e2e_invention_planner_enhanced.py`
- `e2e_invention_planner_real.py`
- `e2e_invention_validation.py`
- `invention_planner_integrations.py`
- `invention_planner_integration_helpers.py`
- `invention_planner_structures.py`
- `autonomous_research_quest.py`

---

## 21. Fix / Scan / Audit / Refactor Utility Scripts (one-shot tooling)

- `analyze_imports.py`
- `analyze_ke_imports.py`
- `analyze_openevolve_integration.py`
- `analyze_problem_analyzer.py`
- `apply_ace_phase4_fixes.py`
- `apply_ace_security_fixes.py`
- `apply_api_consistency_fixes.py`
- `apply_code_quality_fixes.py`
- `apply_component_alerting.py`
- `apply_phase4_validation.py`
- `assess_decomposition.py`
- `brutal_audit.py`
- `bug_scanner.py`
- `bug_sweep.js` (JS)
- `compare_before_after.py`
- `compare_icr_versions.py`
- `compare_parameter_managers.py`
- `compare_parameter_managers_simple.py`
- `compare_phase1_phase2.py`
- `compare_simple_ascii.py`
- `comprehensive_edge_case_analysis.py`
- `comprehensive_gap_audit.py`
- `comprehensive_syntax_fixer.py`
- `comprehensive_workflow_auditor.py`
- `data_consistency_verification.py`
- `debug_class.py`
- `debug_source.py`
- `debug_test.py`
- `debug_test_wrapper.py`
- `deduplication_analysis.py`
- `dependency_analyzer.py`
- `dependency_builder.py`
- `dependency_decomposition.py`
- `dependency_manager.py`
- `dependency_visualizer.py`
- `detailed_audit.py`
- `evaluate_scripts.py` (n/a)
- `find_duplicate_files.py`
- `fix_demo.py`
- `fix_demo_mcts.py`
- `fix_demo_mcts_final.py`
- `fix_domain_imports.py`
- `fix_high_severity.py`
- `fix_import_issues.py`
- `fix_leanaide.py`
- `fix_logger_calls.py`
- `fix_manual_security_issues.py`
- `fix_mcts.py`
- `fix_medium_severity.py`
- `fix_non_security_issues.py`
- `fix_subprocess_shell.py`
- `fix_syntax_errors.py`
- `fix_unicode_characters.py`
- `gap_analysis_report_2026.py`
- `identify_test_errors.py`
- `scan_batch_3.py`
- `scan_batch_8.py`
- `scan_import_errors_batch2.py`
- `scan_import_errors_batch5.py`
- `scan_imports_batch_10.py`
- `scan_imports_batch_11_16.py`
- `scan_imports_batch_7.py`
- `scan_imports_batch4.py`
- `scan_imports_batch6.py`
- `scan_top_level_only.py`
- `show_end.py`
- `.tmp_dump_env.py`

---

## 22. Documentation (.md files)

- `100_COMPLETE.md`
- `ADAPTIVE_MDAP_194_INTEGRATIONS_COMPLETE.md`
- `ADAPTIVE_MDAP_40_POINT_INTEGRATION.md`
- `ADAPTIVE_MDAP_52_INTEGRATIONS_COMPLETE.md`
- `ADAPTIVE_MDAP_COMPLETE_INTEGRATION.md`
- `ADAPTIVE_MDAP_FINAL_SUMMARY.md`
- `ADAPTIVE_MDAP_INTEGRATION_GUIDE.md`
- `ADAPTIVE_MDAP_PES_INTEGRATION_DESIGN.md`
- `ADAPTIVE_MDAP_PES_INTEGRATION_SUMMARY.md`
- `ADAPTIVE_MDAP_QUICK_START.md`
- `ADAPTIVE_MDAP_README.md`
- `ADAPTIVE_MDAP_TROUBLESHOOTING.md`
- `ADAPTIVE_MDAP_WIRING_COMPLETE.md`
- `ADAPTIVE_MDAP_WIRING_SUMMARY.md`
- `ALL_IMPORT_FIXES_COMPLETE_REPORT.md`
- `ALL_IMPORT_TESTS_COMPLETE_REPORT.md`
- `API.md`
- `API_GATEWAY_SPEC.md`
- `API_REFERENCE.md`
- `API_CONTRACT_FIX_REPORT.txt` (see reports)
- `ARCHITECTURE.md`
- `ARCHITECTURE_ISSUE.md`
- `BENCHMARK_VALIDATION_COMPLETE.md`
- `BENCHMARKING_GUIDE.md`
- `BRIDGE_COMPONENTS_TO_PRESERVE.md`
- `BRUTAL_VERIFICATION_REPORT.md`
- `BUBBLELAB_PYGRAPHISTRY_INTEGRATION_COMPLETE.md`
- `BUBBLELABS_INTEGRATION_ROADMAP.md`
- `BUBBLELABS_INTEGRATION_ROADMAP_UPDATED.md`
- `BUILD_REPORT.md`
- `BUNDLELAB_FIXES_SUMMARY.md`
- `CAUSAL_LEARN_COMPLETE_INTEGRATION_REVIEW.md`
- `CAUSAL_LEARN_FINAL_REVIEW.md`
- `CAUSAL_LEARN_INTEGRATION_FINAL_SUMMARY.md`
- `CAUSAL_LEARN_INTEGRATION_STATUS.md`
- `CAUSAL_LEARN_OPTIONALITY_VERIFICATION.md`
- `CAV_NLP_COMPLETE_WIRING_REPORT.md`
- `CAV_NLP_FINAL_WIRING_MANIFEST.md`
- `CAV_NLP_INTEGRATION_REPORT.md`
- `CAV_NLP_INTEGRATION_REVIEW_REPORT.md`
- `CAV_NLP_INTEGRATION_STRATEGY.md`
- `CAV_NLP_INTEGRATION_SUMMARY.md`
- `CAV_NLP_WIRING_100_PERCENT_COMPLETE.md`
- `CAV_NLP_WIRING_ABSOLUTELY_COMPLETE.md`
- `CAV_NLP_WIRING_COMPLETE.md`
- `CAV_NLP_WIRING_FINAL.md`
- `CAV_NLP_WIRING_FINAL_COMPLETE.md`
- `CAV_NLP_WIRING_ULTRA_COMPLETE.md`
- `CI_CD.md`
- `CI_CD_QUICKSTART.md`
- `CIRCULAR_IMPORT_FIX.md`
- `COMPLETE_DSPY_DTS_INTEGRATION_SUMMARY.md`
- `COMPLETE_DSPY_INTEGRATION_SUMMARY.md`
- `COMPLETE_IMPLEMENTATION_SUMMARY.md`
- `COMPLETE_INTEGRATION_SUMMARY.md`
- `COMPLETION_SUMMARY.md`
- `COMPREHENSIVE_CODEBASE_SECURITY_ANALYSIS.md`
- `COMPREHENSIVE_DSPY_INTEGRATION_COMPLETE.md`
- `COMPREHENSIVE_GAP_ANALYSIS.md`
- `COMPREHENSIVE_INTEGRATION_SUMMARY.md`
- `COMPREHENSIVE_SECURITY_AND_QUALITY_ANALYSIS.md`
- `CRITICAL_HIGH_FIXES_REPORT.md`
- `CREWAI_ENHANCEMENT_ROADMAP.md` (CrewAI_Enhancement_Roadmap.md)
- `CrewAI_Enhancement_Roadmap.md`
- `CREWAI_INTEGRATION_DOCS.md`
- `CREWAI_INTEGRATION_README.md`
- `CREWAI_REMOVAL_ROADMAP.md`
- `CREWAI_RESEARCH_GAP_ANALYSIS.md`
- `CREWAI_RESEARCH_IMPLEMENTATION_COMPLETE.md`
- `CREWAI_RESEARCH_ROADMAP.md`
- `CREWAI_RESEARCH_TRUE_100_COMPLETE.md`
- `CREWAI_ROADMAP_GAP_ANALYSIS_REPORT.md`
- `DECOMPOSITION_100_PERCENT_FIX_SUMMARY.md`
- `DEPLOYMENT_READY.md`
- `DEVELOPER_GUIDE.md`
- `DEVELOPMENT.md`
- `DISTRIBUTED_COORDINATION_CONSENSUS_SPEC.md`
- `DOCUMENTATION_COMPLETENESS_REPORT.md`
- `DOCUMENTATION_GAP_ANALYSIS.md`
- `DOCUMENTATION_INDEX.md`
- `DSPY_INTEGRATION_COMPLETE.md`
- `DUPLICATE_KNOWLEDGE_ENGINE_RESOLUTION.md`
- `DUPLICATE_MERGE_COMPLETE_REPORT.md`
- `E2E_COMPLETE.md`
- `E2E_IMPLEMENTATION_COMPLETE_SUMMARY.md`
- `E2E_INVENTION_GAP_ANALYSIS.md`
- `E2E_INVENTION_PLANNER_TRUE_100_PERCENT.md`
- `E2E_VERIFICATION_REPORT.md`
- `END_TO_END_TEST_REPORT.md`
- `ENHANCED_MATH_KNOWLEDGE_INTEGRATION_COMPLETE.md`
- `ENHANCED_Z3_TO_LEAN_IMPROVEMENTS.md`
- `ENHANCEMENTS_3_4_COMPLETE.md`
- `ENVIRONMENT_SETUP_GUIDE.md`
- `EVOLUTION_ALGORITHM_ENHANCEMENT_SPEC.md`
- `EVERYTHING_FIXED_FINAL_REPORT.md`
- `FAILED_TESTS_FIX_SUMMARY.md`
- `FEDERATION_STRUCTURE.md`
- `FINAL_CAV_NLP_VERIFICATION_REPORT.md`
- `FINAL_COMPLETE_FIXES_REPORT.md`
- `FINAL_COMPLETION_PERCENTAGES.md`
- `FINAL_DUPLICATE_MERGE_SUMMARY.md`
- `FINAL_GAP_ANALYSIS.md`
- `FINAL_IMPLEMENTATION_SUMMARY.md`
- `FINAL_IMPORT_TEST_REPORT.md`
- `FINAL_INTEGRATION_SUMMARY.md`
- `FINAL_SECURITY_FIXES_SUMMARY.md`
- `FINAL_SUMMARY.md`
- `FINAL_TEST_FIXES_REPORT.md`
- `FINAL_TEST_REPORT.md`
- `FINAL_VERIFICATION_REPORT.md`
- `FIXES_APPLIED.md`
- `FUNCTIONALITY_TEST_REPORT.md`
- `GAP_12_FIX_COMPLETE.md`
- `GAP_ANALYSIS.md`
- `GAP_ANALYSIS_MATH_KNOWLEDGE.md`
- `GAP_ANALYSIS_REPORT.md`
- `GAP_ANALYSIS_REPORT_2026_02_17.md`
- `GAP_FIX_SUMMARY.md`
- `GAPS_FILLED_COMPREHENSIVE_REPORT.md`
- `GAUNTLET_GAP_ANALYSIS.md`
- `GAUNTLET_IMPLEMENTATION_COMPLETE.md`
- `GAUNTLET_SELECTION_GUIDE.md`
- `GAUNTLET_SYSTEM_100_PERCENT_COMPLETE.md`
- `GAUNTLET_SYSTEM_COMPLETION_REPORT.md`
- `GAUNTLET_SYSTEM_FINAL_IMPLEMENTATION_COMPLETE.md`
- `GAUNTLET_SYSTEM_TRUE_100_COMPLETE.md`
- `GAUNTLET_TESTS_SUMMARY.md`
- `GRACEFUL_DEGRADATION_QUICK_START.md`
- `GRACEFUL_DEGRADATION_REPORT.md`
- `GRAND_TOTAL_IMPORT_FIXES_REPORT.md`
- `HONEST_INTEGRATION_ASSESSMENT.md`
- `HYBRID_PES_README.md`
- `IMPLEMENTATION_COMPLETE.md`
- `IMPORT_FIXES_COMPLETE.md`
- `IMPORT_FIXES_COMPLETE_FINAL.md`
- `IMPORT_FIXES_FINAL_COMPLETE.md`
- `IMPORT_FIXES_FINAL_REPORT.md`
- `IMPORT_FIXES_FINAL_SUMMARY.md`
- `IMPORT_FIXES_SUMMARY.md`
- `IMPORT_FIXES_ULTIMATE_SUMMARY.md`
- `INSURANCE_TEST_FIXES_SUMMARY.md`
- `INTEGRATION_COMPLETE.md`
- `INTEGRATION_GAP_ANALYSIS_REPORT.md`
- `INTEGRATION_GUIDE.md`
- `INTEGRATION_IMPLEMENTATION_COMPLETE.md`
- `INTEGRATION_IMPLEMENTATION_SUMMARY.md`
- `INTEGRATION_PROGRESS_REPORT.md`
- `INTEGRATION_STATUS.md`
- `INTEGRATION_SUMMARY.md`
- `INTEGRATION_TESTS_GROUP_A_FINAL_REPORT.md`
- `INTEGRATION_TESTS_GROUP_A_SUMMARY.md`
- `integration_validity_report.md`
- `KNOWLEDGE_ARTIFACTS_COLLECTION_SUMMARY.md`
- `KNOWLEDGE_ENGINE_100_PERCENT_COMPLETE.md`
- `KNOWLEDGE_ENGINE_COMPLETION_STATUS_REPORT.md`
- `KNOWLEDGE_ENGINE_GAPS_COMPLETE.md`
- `KNOWLEDGE_ENGINE_INTEGRATION_TEST_FIXES.md`
- `KNOWLEDGE_EXTRACTION_FIX_STATUS.md`
- `KNOWLEDGE_EXTRACTION_GAP_ANALYSIS.md`
- `KNOWLEDGE_EXTRACTION_TRUE_100_COMPLETE.md`
- `KNOWLEDGE_GRAPH_QUERY_ANALYTICS_SPEC.md`
- `KNOWLEDGE_GRAPH_STORAGE_INDEXING_SPEC.md`
- `LEAN_4_INTEGRATION_WIRING_REPORT.md`
- `LEAN_4_WIRING_COMPLETE_FIXES_REPORT.md`
- `LEAN_4_WIRING_COMPLETION_REPORT_FINAL.md`
- `LEAN_4_WIRING_FINAL_REPORT.md`
- `LEAN_4_WIRING_THOROUGH_PASS_COMPLETE.md`
- `LEAN_INTEGRATION_COMPLETION_REPORT.md`
- `LEAN_INTEGRATION_OPPORTUNITIES_REPORT.md`
- `LEAN_TEST_FIXES_REPORT.md`
- `LEANAIDE_CAV_NLP_INTEGRATION_ANALYSIS.md`
- `LEANAIDE_FINAL_REPORT.md`
- `LEANAIDE_GAP_ANALYSIS.md`
- `LEANAIDE_IMPLEMENTATION_COMPLETE.md`
- `LEANAIDE_INTEGRATION_COMPLETE.md`
- `LEANAIDE_MIGRATION_PLAN.md`
- `LEANAIDE_SETUP.md`
- `LEANAIDE_TRUE_100_COMPLETE.md`
- `LEANAIDE_TRUE_100_SUMMARY.md`
- `LEANAIDE_USER_GUIDE.md`
- `LOONGFLOW_COST_OPTIMIZATION_EXTRACTION.md`
- `LOONGFLOW_INTEGRATION_REPORT.md`
- `LOONGFLOW_ULTIMATE_REPORT.md`
- `MCP_MIGRATION_REPORT.md`
- `MCP_SERVER_103_TOOLS.md`
- `MCP_SERVER_151_TOOLS.md`
- `MCP_SERVER_200_TOOLS.md`
- `MCP_SERVER_255_TOOLS.md`
- `MCP_SERVER_308_TOOLS.md`
- `MDAP_MAKER_GAUNTLET_INTEGRATION_COMPLETE.md`
- `MDAP_MAKER_IMPLEMENTATION_ANALYSIS.md`
- `OPENEVOLVE_BUBBLELAB_INTEGRATION_COMPLETE.md`
- `OPENEVOLVE_COMPREHENSIVE_ASSESSMENT_FINAL.md`
- `OPENEVOLVE_INTEGRATION_STATUS.md`
- `OPENEVOLVE_LOONGFLOW_INTEGRATION_ARCHITECTURE.md`
- `OPENTFLOW_UNIFIED_UPDATE.md`
- `PERFORMANCE_OPTIMIZATION_SUMMARY.txt` (reports)
- `PES_ENHANCED_COMPLETE_INTEGRATION_SUMMARY.md`
- `PES_ENHANCED_CONFIG_INTEGRATION.md`
- `PES_ENHANCED_INTEGRATION_FINAL_REPORT.md`
- `PES_ENHANCED_INTEGRATION_REPORT.md`
- `PES_ENHANCED_INTEGRATION_SUMMARY.md`
- `PES_ENHANCED_SECOND_GAP_ANALYSIS_FIXES.md`
- `PESS_GAP_ANALYSIS_REPORT.md`
- `PHASE2_COMPLETE.md`
- `PHASE2_PROGRESS.md`
- `PHASE2_SESSION_REPORT.md`
- `PHASE2B_TOP5_INTEGRATION_TESTS_COMPLETE.md`
- `PHASE2C_REMAINING_INTEGRATION_TESTS_COMPLETE.md`
- `PHI2_DEBIASING_SUMMARY.md`
- `PHYSICS_VALIDATION_COMPLETE.md`
- `PROBE_FIXES_SUMMARY.md`
- `PROBE_VERIFICATION_REPORT.md`
- `PROJECT_COMPLETE.md`
- `PROJECT_IMPLEMENTATION_STATUS_REPORT.md`
- `QUICK_START.md`
- `QUICK_START_GAUNTLET.md`
- `QUICK_START_GUIDE.md`
- `QUICK_START_PES_ENHANCED.md`
- `RAGBITS_BUBBLELAB_INTEGRATION_COMPLETE.md`
- `RAGBITS_INTEGRATION_COMPLETE_100_PERCENT.md`
- `ragbits_integration_summary.md`
- `README_INTEGRATION.md`
- `REAL_SECURITY_TESTS_NEEDED.md`
- `REAL_SECURITY_TESTS_SUMMARY.md`
- `REMAINING_GAPS_IDENTIFIED.md`
- `REORGANIZATION_COMPLETE.md`
- `REORGANIZATION_STATUS.md`
- `RESE_FINAL_INTEGRATION_TEST_SUMMARY.md`
- `RESE_HEALTH_APIS_SUMMARY.md`
- `RESE_HEALTH_APIS_TEST_REPORT.md`
- `RESE_IMPLEMENTATION_GAP_ANALYSIS.md`
- `RESE_IMPLEMENTATION_ROADMAP.md`
- `RESE_PIPELINE_VERIFICATION_REPORT.md`
- `RESE_PROBE_DELIVERY_REPORT.md`
- `RESE_PROBE_SUMMARY.md`
- `RESE_STATUS_DASHBOARD.md`
- `RESE_TEST_DELIVERABLES.md`
- `RESE_TEST_DELIVERABLES_SUMMARY.md`
- `RESE_TEST_EXECUTION_CHECKLIST.md`
- `RESE_TEST_SUMMARY.md`
- `RESE_Z3_INTEGRATION_COMPLETE_REPORT.md`
- `RESE_Z3_INTEGRATION_PLAN.md`
- `RESOURCE_MANAGEMENT_SPEC.md`
- `ROBUSTNESS_LAYER_GUIDE.md`
- `ROMA_100_PERCENT_COMPLETE.md`
- `ROMA_ALL_ISSUES_FIXED.md`
- `ROMA_COMPLETE_STATUS.md`
- `ROMA_COMPLETE_SUMMARY.md`
- `ROMA_COMPREHENSIVE_ANALYSIS.md`
- `ROMA_FINAL_100_PERCENT_COMPLETE.md`
- `ROMA_FINAL_COMPLETION_REPORT.md`
- `ROMA_FINAL_FIX_REPORT.md`
- `ROMA_FINAL_IMPLEMENTATION_REPORT.md`
- `ROMA_FINAL_REPORT.md`
- `ROMA_FINAL_STATUS_REPORT.md`
- `ROMA_IMPLEMENTATION_SUMMARY.md`
- `ROMA_INTEGRATION_COMPLETION_REPORT.md`
- `ROMA_INTEGRATION_FINAL_SUMMARY.md`
- `ROMA_KG_INTEGRATION_SUMMARY.md`
- `ROMA_QUICK_REFERENCE.md`
- `ROMA_TEST_FIXES_COMPLETE.md`
- `ROMA_WIRING_VERIFICATION_REPORT.md`
- `SECURITY_ARCHITECTURE.md`
- `SECURITY_FIXES_SUMMARY.md`
- `SECURITY_GAP_ANALYSIS.md`
- `SECURITY_IMPLEMENTATION_COMPLETE.md`
- `SECURITY_IMPLEMENTATION_SUMMARY.md`
- `SECURITY_TEST_DEADLOCK_FIX.md`
- `SECURITY_TEST_GAP_SUMMARY.md`
- `SECURITY_TESTS_COMPLETE.md`
- `SECURITY_TESTS_SUMMARY.md`
- `SECURITY_TRUE_100_COMPLETE.md`
- `SECURITY_TRUE_100_VERIFICATION.md`
- `SECURITY_WORKFLOW_EVOLUTION_TEST_FIXES.md`
- `SESSION_COMPLETE.md`
- `SESSION_COMPLETE_100_PERCENT.md`
- `SKILL.md`
- `SSV_INSOLVENCY_VULNERABILITY.md`
- `STAGE6_COMPLETION_REPORT.md`
- `STAGE6_IMPLEMENTATION_COMPLETE.md`
- `STUB_INTEGRATIONS_LIST.md`
- `strategy_system_comparison.md`
- `SYSTEM_INTEGRATION_TESTS_COMPLETE.md`
- `TEST_CONFIGURATION_FIXES_SUMMARY.md`
- `TEST_COVERAGE_FINAL_REPORT.md`
- `TEST_EXECUTION_GUIDE.md`
- `TEST_FIXES_ANALYSIS.md`
- `TEST_FIXES_SUMMARY.md`
- `TEST_PROGRESS_SUMMARY.md`
- `TEST_STATUS_SUMMARY.md`
- `TESTING_FRAMEWORK_GAP_ANALYSIS.md`
- `TESTING_FRAMEWORK_GAP_FIXES_COMPLETE.md`
- `TESTING_FRAMEWORK_TRUE_100_COMPLETE.md`
- `TROUBLESHOOTING.md`
- `TRUE_100_COMPLETE.md`
- `TRUE_100_COMPLETION_SUMMARY.md`
- `TRUE_100_FIXES_SUMMARY.md`
- `TRUE_100_PERCENT_IMPORT_SUCCESS.md`
- `TRUE_100_VERIFICATION_REPORT.md`
- `TRUE_96_7_PERCENT_COMPLETE.md`
- `ULTIMATE_IMPORT_VERIFICATION_REPORT.md`
- `UNIFIED_MATH_SERVICE_WIRING_PLAN.md`
- `VERIFICATION_COMPLETE.md`
- `Z3_BUG_FIXES_APPLIED.md`
- `Z3_CAV_NLP_INTEGRATION_COMPLETE.md`
- `Z3_GAP_ANALYSIS.md`
- `Z3_IMPLEMENTATION_COMPLETE.md`
- `Z3_INTEGRATION_100_PERCENT_COMPLETE.md`
- `Z3_INTEGRATION_COMPLETION_ASSESSMENT.md`
- `Z3_INTEGRATION_FINAL_GUIDE.md`
- `Z3_INTEGRATION_README.md`
- `Z3_KNOWLEDGE_INTEGRATION_SUMMARY.md`
- `Z3_LEAN_100_PERCENT_COMPLETE.md`
- `Z3_LEAN_ADDITIONAL_GAP_FIXES.md`
- `Z3_LEAN_ALL_GAP_FIXES_COMPLETE.md`
- `Z3_LEAN_GAP_FIXES_COMPLETE.md`
- `Z3_LEAN_INVENTION_INTEGRATION_COMPLETE.md`
- `Z3_LEAN_INVENTION_PLANNER_INTEGRATION.md`
- `Z3_LEANAIDE_INTEGRATION_COMPLETE.md`
- `Z3_LEANAIDE_INTEGRATION_README.md`
- `Z3_LEANAIDE_LEAN_INTEGRATION_REVIEW_COMPLETE.md`
- `Z3_PHASE1_IMPLEMENTATION_SUMMARY.md`
- `Z3_TO_LEAN_INTEGRATION_COMPLETE.md`
- `Z3_TO_LEAN_INVENTION_FINAL_SUMMARY.md`
- `Z3_TRUE_100_PERCENT_COMPLETE.md`

---

## 23. Status Reports / Summaries / Logs (.txt, .log, .html)

- `adaptive_mdap_wiring_report.txt`
- `API_CONTRACT_FIX_REPORT.txt`
- `backend_stderr.log`
- `backend_stdout.log`
- `BRUTAL_FUNCTIONALITY_AUDIT_REPORT.txt`
- `BRUTAL_SECURITY_VERIFICATION_REPORT.txt`
- `BRUTAL_VERIFICATION_REPORT.txt`
- `BUBBLELABS_TEST_RESULTS.txt`
- `BUBBLELABS_TEST_SUMMARY.txt`
- `BUBBLE_INVENTORY_DELIVERABLES.txt`
- `bug_scan_results.txt`
- `comprehensive_verification_report.txt`
- `CREWAI_RESEARCH_COMPLETION_SUMMARY.txt`
- `docs_todo.txt`
- `E2E_COMPLETION_SUMMARY.txt`
- `E2E_EXECUTIVE_SUMMARY.txt`
- `E2EInventionEngineRESEframework.txt`
- `e2e_execution_output.log`
- `e2e_final_complete.log`
- `e2e_validation.log`
- `e2e_validation_report_20260101_010632.txt`
- `e2e_validation_report_20260101_010721.txt`
- `e2e_validation_report_20260101_010745.txt`
- `e2e_validation_report_20260101_010954.txt`
- `e2e_validation_report_20260101_011146.txt`
- `EDGE_CASE_FIXES_SUMMARY.txt`
- `error.log`
- `failed_files.txt`
- `FILE_STRUCTURE.txt`
- `final_complete_test_run.txt`
- `FINAL_COMPLETION_REPORT.txt`
- `FINAL_FIX_VERIFICATION_REPORT.txt`
- `FINAL_REPORT.txt`
- `FINAL_SECURITY_SCAN_REPORT.txt`
- `FINAL_TASK_COMPLETION_REPORT.txt`
- `final_test_counts.txt`
- `final_test_results.txt`
- `final_test_summary.txt`
- `FIX_SUMMARY.txt`
- `FIXED_TEST_RESULTS.log`
- `FIXES_SUMMARY.txt`
- `full_pipeline_execution.log`
- `full_test_output.txt`
- `full_test_results.txt`
- `GAP_AUDIT_OUTPUT.txt`
- `GAP_AUDIT_REPORT.txt`
- `gauntlet_test_results.txt`
- `gemini_progress.txt`
- `HELPER_DEPENDENCIES_REPORT.txt`
- `import_error.log`
- `import_test_after_fixes.log`
- `import_test_final_output.log`
- `import_test_final_results.log`
- `import_test_full_output.log`
- `import_test_output.log`
- `import_test_robust_output.log`
- `import_test_simple_output.log`
- `IMPORT_VALIDATION_SUMMARY.txt`
- `INDEPENDENT_AUDIT_REPORT.txt`
- `indexer.log`
- `knowledge_engine_import_issues.txt`
- `LEAN_VERIFICATION_INTEGRATION_REPORT.txt`
- `LEANAIDE_TRUE_100_VERIFICATION.txt`
- `mcts_mdap_test_results.log`
- `MDAP_MAKER_MCTS_SUCCESS.txt`
- `MEMORY_LEAK_FIX_SUMMARY.txt`
- `MISSION_ACCOMPLISHED.txt`
- `NON_SECURITY_FIXES_LOG.txt`
- `openevolve.log`
- `pdf_extract.txt`
- `PERFORMANCE_OPTIMIZATION_SUMMARY.txt`
- `PES_ENHANCED_SECOND_GAP_FIXES_SUMMARY.txt`
- `PHASE_1.1.1_FILE_STRUCTURE.txt`
- `PHASE_2_SUMMARY.txt`
- `PHASE_5_FINAL_SUMMARY.txt`
- `PHASE6_SUMMARY.txt`
- `PHPDOC_COVERAGE_REPORT.txt`
- `pip_install_log.txt`
- `project_status_generation.log`
- `pytest_output.log`
- `QUICK_AUDIT_FACTS.txt`
- `remaining_failures.txt`
- `request_for_help.txt`
- `RESE_Z3_INTEGRATION_FILES.txt`
- `resume_session.txt`
- `security_fix_log_20260120_231039.log`
- `security_fix_log_20260120_231703.log`
- `SECURITY_STATUS.txt`
- `selfplay.txt`
- `setup_output.txt`
- `syntax_fix_log_20260120_233122.log`
- `test_ascii_debug.txt`
- `test_execution_report.txt`
- `test_failures.txt`
- `test_knowledge_artifacts.log`
- `test_output.log`
- `test_output.txt`
- `test_output2.txt`
- `TEST_OUTPUTS_SUMMARY.txt`
- `test_report.txt`
- `test_reports.html`
- `test_results.log`
- `test_results.txt`
- `test_results_output.txt`
- `test_roma_output.txt`
- `test_roma_output2.txt`
- `test_run_output.txt`
- `test_run_output2.txt`
- `test_run_results.txt`
- `TEST_SUCCESS_SUMMARY.txt`
- `test_summary.txt`
- `test_summary_final.txt`
- `test_summary_full.txt`
- `test_summary_only.txt`
- `test_trace.log`
- `test_tree_ascii.txt`
- `test_tree_output.dot`
- `test_tree_output.html`
- `todo_remaining.txt`
- `top_level_fix_log_20260120_232631.log`
- `top_level_fix_log_20260120_235638.log`
- `TRUE_98_3_PERCENT_CERTIFICATE.txt`
- `ULTIMATE_SUCCESS_REPORT.txt`
- `VALIDATION_RESULTS.txt`
- `VALIDATION_SUMMARY.txt`
- `validation_output.txt`
- `validator_agent.txt`
- `VERIFICATION_SUMMARY.txt`
- `verify_output.txt`
- `SOP.txt`
- `E2E_EXECUTIVE_SUMMARY.txt` (see above)
- `INTEGRATION_VALIDITY_SUMMARY.txt`
- `all_test_results.txt`
- `BUBBLELABS_TEST_SUMMARY.txt` (see above)
- `FIXED_TEST_RESULTS.log` (see above)
- `full_pipeline_execution_report.json` (data)

---

## 24. Data & State Files (.db, .json data, .pkl, .yaml configs)

- `alerts.json`
- `all_fixes_final_summary.json`
- `api_keys.db`
- `audit_logs.db`
- `bubble_inventory.json`
- `bubblelabs_analytics.db`
- `critical_import_errors.json`
- `data_consistency_report.json`
- `data_consistency_report_sample.json`
- `deploy_config_development.json`
- `deploy_config_production.json`
- `deploy_config_staging.json`
- `e2e_validation_report_20260101_010632.json`
- `e2e_validation_report_20260101_010721.json`
- `e2e_validation_report_20260101_010745.json`
- `e2e_validation_report_20260101_010954.json`
- `e2e_validation_report_20260101_011146.json`
- `EDGE_CASE_DETAILED_REPORT.json`
- `END_TO_END_TEST_RESULTS.json`
- `evaluator_coordinator_state.pkl`
- `example_workflow_001_export.json`
- `FINAL_100_PERCENT_VERIFICATION.json`
- `FINAL_INTEGRATION_TEST_RESULTS.json`
- `final_scan.json`
- `fixes_100_percent_phase1.json`
- `fixes_100_percent_phase2.json`
- `fixes_100_percent_phase3.json`
- `fixes_100_percent_phase4.json`
- `fixes_adaptive_mdap.json`
- `fixes_additional_batch1.json`
- `fixes_additional_batch2.json`
- `fixes_additional_batch3.json`
- `fixes_additional_batch4.json`
- `fixes_applied_final.json`
- `fixes_bubblelabs_examples.json`
- `fixes_cav_nlp.json`
- `fixes_dspy_batch1.json`
- `fixes_dspy_batch2.json`
- `fixes_dspy_batch3.json`
- `fixes_dspy_batch4.json`
- `fixes_remaining.json`
- `fixes_remaining_final.json`
- `fixes_root_tests.json`
- `fixes_syntax_batch1.json`
- `fixes_syntax_batch1_curie.json`
- `fixes_toplevel.json`
- `FIX_TRACKING_DATABASE.json`
- `full_pipeline_execution_report.json`
- `gauntlets.json`
- `ground_truth_store.json`
- `import_errors_batch_1.json`
- `import_errors_batch_10.json`
- `import_errors_batch_11_16.json`
- `import_errors_batch_17_22.json`
- `import_errors_batch_2.json`
- `import_errors_batch_23_28.json`
- `import_errors_batch_29_32.json`
- `import_errors_batch_3.json`
- `import_errors_batch_4.json`
- `import_errors_batch_5.json`
- `import_errors_batch_6.json`
- `import_errors_batch_7.json`
- `import_errors_batch_8.json`
- `import_errors_batch_9.json`
- `import_fix_report.json`
- `import_fixes_report.json`
- `import_fixes_summary.json`
- `IMPORT_FIXES_FINAL_ROUND.json`
- `import_test_batch1.json`
- `import_test_batch10.json`
- `import_test_batch2.json`
- `import_test_batch3.json`
- `import_test_batch4.json`
- `import_test_batch5.json`
- `import_test_batch6.json`
- `import_test_batch7.json`
- `import_test_batch8.json`
- `import_test_batch9.json`
- `import_test_results_20260108_232738.json`
- `import_test_results_20260108_233137.json`
- `integration_test_results.json`
- `knowledge_artifacts_collection.json`
- `knowledge_artifacts_complete_collection.json`
- `knowledge_artifacts.db`
- `knowledge_graph_index.db`
- `knowledge_hash_index.db`
- `knowledge_hierarchical_index.db`
- `llm_cache.db`
- `mcp_agent.secrets.yaml`
- `metrics.db`
- `model_performance.json`
- `openevolve_metrics.db`
- `parameter_settings.json`
- `report_templates.json`
- `rese_import_analysis.json`
- `RESE_PIPELINE_DEMO_RESULTS.json`
- `RESE_PIPELINE_VERIFICATION_REPORT.json`
- `security_analysis_20260120_231444.json`
- `security_analysis_20260120_231708.json`
- `sovereign_decomposition.db`
- `SSV_DEFINITIVE_PROOF.json`
- `SSV_FORMAL_PROOF_CERTIFICATE.json`
- `static_analysis_report.json`
- `steer_rules.yaml`
- `syntax_errors_batch1.json`
- `syntax_errors_root.json`
- `syntax_errors_specific_dirs.json`
- `teams.json`
- `temp_detailed_issues.json`
- `temp_import_errors.json`
- `temp_issues.json`
- `TEST_COVERAGE_SUMMARY.json`
- `test_fk_verification.db`
- `TRUE_100_PERCENT_IMPORT_REPORT.json`
- `TRUE_100_VERIFICATION_REPORT.json`
- `user_preferences.json`
- `verification_report.json`
- `verification_results.json`
- `verification_results_28_files.json`
- `z3_cache.db`
- `z3_constraint_hardening_results.json`
- `z3_config.yaml`
- `z3_knowledge.db`
- `config.yaml`
- `decomposition_config_lean.yaml`
- `leanaide_config.example.yaml`

---

## 25. Frontend / Web Assets (JS, TS, CSS, HTML, certs)

- `bug_sweep.js`
- `jest.config.js` (see section 1)
- `styles.css`
- `test_adapter_integration.js`
- `test-core-components.js`
- `test_e2e_simple.js`
- `test-functionality.js`
- `test-functionality.ts`
- `test_icr_e2e.ts`
- `test-reliability-fixes.ts`
- `test_sanitize.js`
- `test-workflows.js`
- `test_tree_output.dot`
- `test_tree_output.html`
- `test_reports.html`
- `verify_icr_integration.js`
- `test.local_cert.pem`
- `test.local_key.pem`
- `tsconfig.json` (see section 1)

---

## 26. Junk / Stray Artifacts (recommend deletion)

- `=0.1.0` (0 bytes, junk)
- `=1.0.0` (junk)
- `=1.78.0` (0 bytes, junk)
- `=2.0.0` (0 bytes, junk)
- `=8.0.0` (0 bytes, junk)
- `projectdependenciesdocumentation` (1 byte, junk)
- `rustup-init.exe` (13.5 MB installer, misplaced)
- `selfplay.pdf` (2.9 MB, misplaced)
- `all_files.txt`
- `all_python_files.txt` (1.7 MB)
- `ASTOR.py` shim (fcntl.py/astor.py look like platform shims - verify before deleting)
- `fcntl.py` (platform shim - verify before deleting)
- `astor.py` (platform shim - verify before deleting)
- `evolution.py.backup_batch4` (backup, move to backups/)
- `integrated_workflow.py.backup_batch2c` (backup, move to backups/)
- `problem_analyzer.py.backup2` (backup, move to backups/)
- `semantic_analyzer.py.backup2` (backup, move to backups/)
- `sovereign_knowledge_manager.py.backup2` (backup, move to backups/)
- `bubblelab-auto-setup-v1-backup.py` (backup, move to backups/)
- `.tmp_dump_env.py` (scratch)
- `projectdependenciesdocumentation` (see above)

---

## Summary

| # | Category | Approx. Files |
|---|----------|---------------|
| 1 | App entry points & root config | ~48 |
| 2 | Core application modules | ~150 |
| 3 | Workflow & orchestration | ~35 |
| 4 | Team / gauntlet / quality / sovereign | ~90 |
| 5 | Knowledge, memory & retrieval | ~60 |
| 6 | OpenEvolve integration layer | ~35 |
| 7 | LeanAide / Lean 4 | ~70 |
| 8 | Z3 / formal verification | ~50 |
| 9 | BubbleLabs / BubbleLab UI | ~40 |
| 10 | ROMA | ~30 |
| 11 | CrewAI | ~35 |
| 12 | Other external integrations | ~45 |
| 13 | API / server / endpoints | ~25 |
| 14 | UI components & layout | ~30 |
| 15 | Analytics / monitoring / performance | ~50 |
| 16 | Security system | ~35 |
| 17 | Tests (Python) | ~140 |
| 18 | Test runners & harnesses | ~25 |
| 19 | Validation & verification scripts | ~60 |
| 20 | Demos & examples | ~75 |
| 21 | Fix / scan / audit scripts | ~80 |
| 22 | Documentation (.md) | ~250 |
| 23 | Reports / logs (.txt/.log/.html) | ~120 |
| 24 | Data & state files (.db/.json/.pkl) | ~120 |
| 25 | Frontend / web assets | ~18 |
| 26 | Junk / strays | ~17 |

**Total: 1,708**

Notes:
- Duplicate/near-duplicate reports (e.g., 8+ `ROMA_*REPORT`, 6+ `CAV_NLP_WIRING_*`, 8+ `LEAN_*WIRING*`, 5+ `MCP_SERVER_*TOOLS.md`) indicate repeated verification passes; most are candidates for archival to `docs/archive/`.
- ~20 files have no meaningful code (0-200 bytes: `unified_kg.py`, `unified_manager.py`, `unified_math_service.py`, `workflow_adapter.py`, etc.) - likely stubs or failed writes; verify before deletion.
- Backups (`.backup*`, `-v1-backup`, `*.backup2`) can be consolidated under `backups/`.
