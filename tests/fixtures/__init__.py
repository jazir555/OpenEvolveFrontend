"""
Evolutionary Test Data Fixtures Package

Provides realistic mock data and test fixtures for Knowledge Engine
integration testing with OpenEvolve and LoongFlow.

Main Module:
    evolution_test_data.py

Available Fixtures:
    - sample_loongflow_result: Successful PES run
    - sample_openevolve_result: Quality-Diversity run
    - mock_knowledge_engine: Mock KE instance
    - domain_specific_problems: All 6 domain definitions
    - temporal_artifacts: Time-series artifacts

Available Generators:
    - get_loongflow_success_result()
    - get_openevolve_qd_result()
    - get_domain_problem(domain)
    - get_temporal_artifacts(n)
    - generate_random_pes_result()
    - generate_multi_run_history(n)

Usage:
    from tests.fixtures.evolution_test_data import get_loongflow_success_result

Documentation:
    - docs/knowledge_engine/KNOWLEDGE_ENGINE_TESTING.md

Copyright 2026 OpenEvolve
Licensed under Apache License 2.0
"""

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"
