"""
Finance Domain Gauntlet Configuration
======================================

High-stakes configuration for financial optimization problems.
Strict thresholds ensure only robust, reliable solutions pass.

Use Cases:
- Portfolio optimization
- Trading strategy development
- Risk management
- Algorithmic trading

Author: OpenEvolve Gauntlet System
Date: 2026-01-30
"""

from openevolve.gauntlets.three_round_orchestrator import ThreeRoundConfig

# Finance Domain Configuration
FINANCE_CONFIG = ThreeRoundConfig(
    # Round 1: LoongFlow AI - High threshold to filter quickly
    round1_config={
        'llm_config': {
            'model': 'claude-3-5-sonnet-20241022',
            'api_key': '',  # Set via environment variable
            'url': 'http://localhost:8001',
            'temperature': 0.2,  # Lower temperature for consistent evaluation
            'max_tokens': 4096
        },
        'timeout': 90,  # Allow more time for financial analysis
        'domain': 'finance'
    },
    round1_weight=0.2,
    round1_threshold=0.7,  # High threshold - only promising solutions continue
    round1_enabled=True,

    # Round 2: Red Team - Aggressive adversarial testing
    round2_config={
        'attack_vectors': [
            'market_crash_scenario',
            'liquidity_crisis',
            'extreme_volatility',
            'black_swan_event',
            'data_corruption'
        ],
        'attack_intensity': 'high',
        'timeout': 180
    },
    round2_weight=0.3,
    round2_threshold=0.8,  # Very high threshold - must be robust
    round2_enabled=True,

    # Round 3: Gold Team - Strict consensus required
    round3_config={
        'evaluators': [
            'financial_analyst',
            'risk_manager',
            'quantitative_researcher',
            'regulatory_compliance'
        ],
        'consensus_threshold': 0.85,
        'formal_verification': False,  # Lean 4 not typically used in finance
        'timeout': 300
    },
    round3_weight=0.5,
    round3_threshold=0.9,  # Extremely high - only exceptional solutions pass
    round3_enabled=True,

    # Global settings
    enable_early_termination=True,  # Stop early if fails - save compute resources
    enable_parallel_execution=False,  # Sequential for accurate timing
    aggregate_artifacts=True,
    generate_detailed_report=True
)

# Trading Sub-Domain Configuration
TRADING_CONFIG = ThreeRoundConfig(
    round1_config={
        'llm_config': {
            'model': 'claude-3-5-sonnet-20241022',
            'temperature': 0.2
        },
        'timeout': 60
    },
    round1_weight=0.2,
    round1_threshold=0.75,
    round1_enabled=True,

    round2_config={
        'attack_vectors': [
            'slippage_impact',
            'transaction_costs',
            'overfitting_test',
            'regime_change'
        ],
        'attack_intensity': 'high'
    },
    round2_weight=0.3,
    round2_threshold=0.85,
    round2_enabled=True,

    round3_config={
        'evaluators': [
            'trading_system_architect',
            'backtesting_specialist',
            'market_microstructure_expert'
        ],
        'consensus_threshold': 0.9
    },
    round3_weight=0.5,
    round3_threshold=0.9,
    round3_enabled=True,
    enable_early_termination=True
)

# Risk Management Configuration
RISK_CONFIG = ThreeRoundConfig(
    round1_config={
        'llm_config': {
            'model': 'claude-3-5-sonnet-20241022',
            'temperature': 0.1  # Very low for risk assessment
        },
        'timeout': 90
    },
    round1_weight=0.2,
    round1_threshold=0.8,
    round1_enabled=True,

    round2_config={
        'attack_vectors': [
            'correlation_breakdown',
            'tail_risk',
            'concentration_risk',
            'leverage_impact'
        ],
        'attack_intensity': 'extreme'
    },
    round2_weight=0.3,
    round2_threshold=0.85,
    round2_enabled=True,

    round3_config={
        'evaluators': [
            'chief_risk_officer',
            'risk_analyst',
            'stress_tester',
            'compliance_officer'
        ],
        'consensus_threshold': 0.95
    },
    round3_weight=0.5,
    round3_threshold=0.95,  # Nearly perfect required
    round3_enabled=True,
    enable_early_termination=True
)


def get_finance_config(sub_domain: str = 'general') -> ThreeRoundConfig:
    """
    Get finance domain configuration for sub-domain.

    Args:
        sub_domain: Sub-domain (general, trading, risk)

    Returns:
        ThreeRoundConfig for sub-domain
    """
    configs = {
        'general': FINANCE_CONFIG,
        'trading': TRADING_CONFIG,
        'risk': RISK_CONFIG
    }

    return configs.get(sub_domain.lower(), FINANCE_CONFIG)


# Example usage
if __name__ == "__main__":
    from openevolve.gauntlets.three_round_orchestrator import ThreeRoundGauntletOrchestrator

    # Get configuration
    config = get_finance_config('trading')

    # Create orchestrator
    orchestrator = ThreeRoundGauntletOrchestrator(config=config)

    print("Finance Gauntlet Configuration Loaded")
    print(f"Round 1 Threshold: {config.round1_threshold}")
    print(f"Round 2 Threshold: {config.round2_threshold}")
    print(f"Round 3 Threshold: {config.round3_threshold}")
