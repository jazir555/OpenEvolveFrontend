"""
Web Development Domain Gauntlet Configuration
=============================================

Lenient configuration for web development problems.
Focuses on functionality and user experience over formal rigor.

Use Cases:
- Frontend development
- Backend API design
- Full-stack applications
- UI/UX optimization

Author: OpenEvolve Gauntlet System
Date: 2026-01-30
"""

from openevolve.gauntlets.three_round_orchestrator import ThreeRoundConfig

# Web Domain Configuration
WEB_CONFIG = ThreeRoundConfig(
    # Round 1: LoongFlow AI - Low threshold, encourage creativity
    round1_config={
        'llm_config': {
            'model': 'claude-3-5-sonnet-20241022',
            'api_key': '',
            'url': 'http://localhost:8001',
            'temperature': 0.7,  # Higher temperature for creative solutions
            'max_tokens': 4096
        },
        'timeout': 45,
        'domain': 'web'
    },
    round1_weight=0.2,
    round1_threshold=0.3,  # Low threshold - most solutions proceed
    round1_enabled=True,

    # Round 2: Red Team - Light adversarial testing
    round2_config={
        'attack_vectors': [
            'edge_cases',
            'user_errors',
            'performance_bottlenecks',
            'accessibility_issues'
        ],
        'attack_intensity': 'low',
        'timeout': 60
    },
    round2_weight=0.3,
    round2_threshold=0.5,  # Moderate threshold
    round2_enabled=True,

    # Round 3: Gold Team - User experience focus
    round3_config={
        'evaluators': [
            'ux_designer',
            'frontend_engineer',
            'backend_engineer'
        ],
        'consensus_threshold': 0.6,
        'formal_verification': False,
        'timeout': 120
    },
    round3_weight=0.5,
    round3_threshold=0.6,  # Accessible threshold
    round3_enabled=True,

    # Global settings
    enable_early_termination=False,  # Run all rounds for feedback
    enable_parallel_execution=False,
    aggregate_artifacts=True,
    generate_detailed_report=True
)

# Frontend Configuration
FRONTEND_CONFIG = ThreeRoundConfig(
    round1_config={
        'llm_config': {
            'model': 'claude-3-5-sonnet-20241022',
            'temperature': 0.7
        },
        'timeout': 45
    },
    round1_weight=0.2,
    round1_threshold=0.3,
    round1_enabled=True,

    round2_config={
        'attack_vectors': [
            'responsive_design_break',
            'browser_compatibility',
            'accessibility_violations',
            'performance_degradation'
        ],
        'attack_intensity': 'low'
    },
    round2_weight=0.3,
    round2_threshold=0.5,
    round2_enabled=True,

    round3_config={
        'evaluators': [
            'ux_designer',
            'frontend_expert',
            'accessibility_specialist'
        ],
        'consensus_threshold': 0.6
    },
    round3_weight=0.5,
    round3_threshold=0.6,
    round3_enabled=True,
    enable_early_termination=False
)

# Backend Configuration
BACKEND_CONFIG = ThreeRoundConfig(
    round1_config={
        'llm_config': {
            'model': 'claude-3-5-sonnet-20241022',
            'temperature': 0.5
        },
        'timeout': 60
    },
    round1_weight=0.2,
    round1_threshold=0.4,
    round1_enabled=True,

    round2_config={
        'attack_vectors': [
            'sql_injection',
            'rate_limiting',
            'error_handling',
            'scalability_issues'
        ],
        'attack_intensity': 'moderate'
    },
    round2_weight=0.3,
    round2_threshold=0.6,
    round2_enabled=True,

    round3_config={
        'evaluators': [
            'backend_architect',
            'security_engineer',
            'devops_engineer'
        ],
        'consensus_threshold': 0.7
    },
    round3_weight=0.5,
    round3_threshold=0.65,
    round3_enabled=True,
    enable_early_termination=True
)


def get_web_config(sub_domain: str = 'general') -> ThreeRoundConfig:
    """
    Get web domain configuration for sub-domain.

    Args:
        sub_domain: Sub-domain (general, frontend, backend)

    Returns:
        ThreeRoundConfig for sub-domain
    """
    configs = {
        'general': WEB_CONFIG,
        'frontend': FRONTEND_CONFIG,
        'backend': BACKEND_CONFIG
    }

    return configs.get(sub_domain.lower(), WEB_CONFIG)


# Example usage
if __name__ == "__main__":
    from openevolve.gauntlets.three_round_orchestrator import ThreeRoundGauntletOrchestrator

    # Get configuration
    config = get_web_config('frontend')

    # Create orchestrator
    orchestrator = ThreeRoundGauntletOrchestrator(config=config)

    print("Web Gauntlet Configuration Loaded")
    print(f"Round 1 Threshold: {config.round1_threshold}")
    print(f"Round 2 Threshold: {config.round2_threshold}")
    print(f"Round 3 Threshold: {config.round3_threshold}")
