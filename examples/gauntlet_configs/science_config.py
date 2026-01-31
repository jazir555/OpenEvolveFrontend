"""
Science Domain Gauntlet Configuration
=====================================

Moderate configuration for scientific and engineering problems.
Balances rigor with exploration of novel approaches.

Use Cases:
- Experimental design
- Data analysis pipelines
- Simulation optimization
- Research methodology

Author: OpenEvolve Gauntlet System
Date: 2026-01-30
"""

from openevolve.gauntlets.three_round_orchestrator import ThreeRoundConfig

# Science Domain Configuration
SCIENCE_CONFIG = ThreeRoundConfig(
    # Round 1: LoongFlow AI - Moderate threshold
    round1_config={
        'llm_config': {
            'model': 'claude-3-5-sonnet-20241022',
            'api_key': '',
            'url': 'http://localhost:8001',
            'temperature': 0.4,  # Moderate temperature for scientific reasoning
            'max_tokens': 4096
        },
        'timeout': 60,
        'domain': 'science'
    },
    round1_weight=0.2,
    round1_threshold=0.5,  # Moderate threshold
    round1_enabled=True,

    # Round 2: Red Team - Moderate adversarial testing
    round2_config={
        'attack_vectors': [
            'edge_case_scenarios',
            'outlier_sensitivity',
            'parameter_variations',
            'noise_resistance'
        ],
        'attack_intensity': 'moderate',
        'timeout': 120
    },
    round2_weight=0.3,
    round2_threshold=0.6,  # Moderate threshold
    round2_enabled=True,

    # Round 3: Gold Team - Consensus with peer review style
    round3_config={
        'evaluators': [
            'domain_expert',
            'methodology_reviewer',
            'statistical_analyst'
        ],
        'consensus_threshold': 0.75,
        'formal_verification': False,
        'timeout': 240
    },
    round3_weight=0.5,
    round3_threshold=0.7,  # Good solutions pass
    round3_enabled=True,

    # Global settings
    enable_early_termination=True,
    enable_parallel_execution=False,
    aggregate_artifacts=True,
    generate_detailed_report=True
)

# Experimental Design Configuration
EXPERIMENTAL_DESIGN_CONFIG = ThreeRoundConfig(
    round1_config={
        'llm_config': {
            'model': 'claude-3-5-sonnet-20241022',
            'temperature': 0.3
        },
        'timeout': 90
    },
    round1_weight=0.2,
    round1_threshold=0.6,
    round1_enabled=True,

    round2_config={
        'attack_vectors': [
            'sample_size_adequacy',
            'control_validity',
            'confounding_variables',
            'measurement_error'
        ],
        'attack_intensity': 'moderate'
    },
    round2_weight=0.3,
    round2_threshold=0.65,
    round2_enabled=True,

    round3_config={
        'evaluators': [
            'experimental_designer',
            'statistician',
            'domain_researcher'
        ],
        'consensus_threshold': 0.8
    },
    round3_weight=0.5,
    round3_threshold=0.75,
    round3_enabled=True,
    enable_early_termination=True
)

# Data Analysis Configuration
DATA_ANALYSIS_CONFIG = ThreeRoundConfig(
    round1_config={
        'llm_config': {
            'model': 'claude-3-5-sonnet-20241022',
            'temperature': 0.3
        },
        'timeout': 60
    },
    round1_weight=0.2,
    round1_threshold=0.5,
    round1_enabled=True,

    round2_config={
        'attack_vectors': [
            'missing_data_handling',
            'outlier_influence',
            'assumption_violations',
            'data_leakage'
        ],
        'attack_intensity': 'moderate'
    },
    round2_weight=0.3,
    round2_threshold=0.6,
    round2_enabled=True,

    round3_config={
        'evaluators': [
            'data_scientist',
            'statistical_analyst',
            'peer_reviewer'
        ],
        'consensus_threshold': 0.75
    },
    round3_weight=0.5,
    round3_threshold=0.7,
    round3_enabled=True,
    enable_early_termination=True
)


def get_science_config(sub_domain: str = 'general') -> ThreeRoundConfig:
    """
    Get science domain configuration for sub-domain.

    Args:
        sub_domain: Sub-domain (general, experimental_design, data_analysis)

    Returns:
        ThreeRoundConfig for sub-domain
    """
    configs = {
        'general': SCIENCE_CONFIG,
        'experimental_design': EXPERIMENTAL_DESIGN_CONFIG,
        'data_analysis': DATA_ANALYSIS_CONFIG
    }

    return configs.get(sub_domain.lower(), SCIENCE_CONFIG)


# Example usage
if __name__ == "__main__":
    from openevolve.gauntlets.three_round_orchestrator import ThreeRoundGauntletOrchestrator

    # Get configuration
    config = get_science_config('experimental_design')

    # Create orchestrator
    orchestrator = ThreeRoundGauntletOrchestrator(config=config)

    print("Science Gauntlet Configuration Loaded")
    print(f"Round 1 Threshold: {config.round1_threshold}")
    print(f"Round 2 Threshold: {config.round2_threshold}")
    print(f"Round 3 Threshold: {config.round3_threshold}")
