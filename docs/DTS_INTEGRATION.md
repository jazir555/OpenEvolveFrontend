# DTS (Dialogue Tree Search) Integration Guide

## Overview
This document describes the integration of Dialogue Tree Search (DTS) capabilities throughout the OpenEvolve codebase. DTS is a conversational strategy exploration system that uses parallel beam search with multi-judge scoring to optimize conversations and decision-making processes.

## Integrated Components

### 1. DTS Integration Module (`dts_integration.py`)
The main integration module provides a bridge between OpenEvolve and the DTS engine. It includes:

- **DTSIntegrationConfig**: Configuration class for DTS parameters
- **DTSIntegration**: Main integration class with methods for:
  - `adversarial_dialogue()`: Run adversarial dialogue between attacker and defender
  - `multi_judge_scoring()`: Multi-judge evaluation system
  - `generate_strategies()`: Strategy exploration using DTS
  - Fallback implementations when DTS is unavailable

### 2. Red Team Integration (`red_team.py`)
Enhanced adversarial testing capabilities:
- `run_adversarial_dialogue_with_dts()`: Uses DTS for enhanced adversarial dialogue
- Multi-judge scoring for vulnerability assessments
- Falls back to standard red team methods when DTS unavailable

### 3. Blue Team Integration (`blue_team.py`)
Enhanced fix generation capabilities:
- `generate_fixes_with_dts()`: Uses DTS for exploring fix strategies
- Strategy exploration for addressing identified issues
- Falls back to standard blue team methods when DTS unavailable

### 4. Quality Assessment Integration (`quality_assessment.py`)
Enhanced evaluation capabilities:
- `assess_with_dts_multi_judge()`: Multi-judge scoring for quality assessment
- Comparative and absolute scoring modes
- Falls back to standard assessment when DTS unavailable

### 5. Evolution Integration (`evolution.py`)
Enhanced strategy exploration for evolution processes:
- `run_evolution_with_dts_strategy_exploration()`: Uses DTS for evolution strategy exploration
- Integrates with DSPy for enhanced programmatic prompting
- Falls back to standard evolution when DTS unavailable

### 6. Evaluator Team Integration (`evaluator_team.py`)
Enhanced evaluation capabilities:
- `evaluate_with_dts_multi_judge()`: Multi-judge scoring using DTS for comprehensive evaluation
- Uses comparative and absolute scoring modes
- Falls back to standard evaluation when DTS unavailable

### 7. Gauntlet System Integration (`sovereign_gauntlets.py`)
Enhanced gauntlet validation with strategy exploration:
- `run_gauntlet_with_dts_strategy_exploration()`: Uses DTS for enhanced strategy exploration during gauntlet runs
- Explores multiple validation strategies to find the most effective approach
- Falls back to standard gauntlet execution when DTS unavailable

## Configuration

### DTS Configuration Options
The DTS integration can be configured using the `DTSIntegrationConfig` class with the following parameters:

- `enabled`: Whether DTS integration is enabled (default: True)
- `max_rounds`: Maximum rounds for DTS conversations (default: 3)
- `init_branches`: Initial branches for tree search (default: 6)
- `turns_per_branch`: Turns per branch (default: 5)
- `user_intents_per_branch`: User intents per branch (default: 3)
- `scoring_mode`: Scoring mode ("comparative" or "absolute", default: "comparative")
- `prune_threshold`: Pruning threshold for low-scoring branches (default: 6.5)
- `deep_research`: Enable deep research integration (default: False)
- `user_variability`: Enable user variability (default: False)
- `max_concurrency`: Maximum concurrent operations (default: 16)

### API Configuration
DTS requires API keys for LLM access:
- `llm_api_key`: API key for LLM provider
- `llm_base_url`: Base URL for LLM API
- `llm_model`: Model identifier (default: "minimax/minimax-m2.1")

## Fallback Mechanisms

All DTS integrations include robust fallback mechanisms:
- When DTS is unavailable (due to missing API keys or other issues)
- When DTS operations fail
- During initialization errors
- The system gracefully degrades to standard OpenEvolve methods

## Usage Examples

### Using DTS-Enhanced Red Team
```python
from red_team import RedTeam
from dts_integration import DTSIntegrationConfig

# Create red team with DTS integration
red_team = RedTeam()
result = red_team.run_adversarial_dialogue_with_dts(
    content="Your content here",
    content_type="code",
    attacker_persona="security_auditor",
    defender_persona="code_defender"
)
```

### Using DTS-Enhanced Quality Assessment
```python
from quality_assessment import QualityAssessmentEngine
from dts_integration import DTSIntegrationConfig

# Assess content with DTS multi-judge scoring
engine = QualityAssessmentEngine()
result = engine.assess_with_dts_multi_judge(
    content="Your content here",
    content_type="documentation",
    use_comparative_scoring=True,
    judge_count=5
)
```

## Error Handling and Resilience

The DTS integration is designed to be resilient:
- All DTS operations are wrapped in try-catch blocks
- Proper error logging is implemented
- Clear fallback pathways exist for all operations
- Configuration validation prevents runtime errors

## Dependencies

- DTS backend system
- DSPy for enhanced prompting (optional)
- Appropriate LLM API keys for DTS operations
- Standard OpenEvolve dependencies

## Testing

The integration includes comprehensive testing:
- `test_dts_integration.py` - Main test suite
- Tests for all integrated components
- Fallback mechanism validation
- Compatibility checks

## Performance Considerations

- DTS operations may increase latency
- API costs may increase with DTS usage
- Concurrency limits are configurable
- Results caching can be implemented for repeated operations

## Troubleshooting

Common issues and solutions:
- **DTS not available**: Verify API keys and network connectivity
- **Rate limiting**: Adjust concurrency and retry settings
- **Poor quality results**: Tune DTS parameters and prompts
- **Fallback activation**: Check DTS availability and configuration

## Future Enhancements

Planned improvements:
- Enhanced caching mechanisms
- Additional DSPy integration points
- Real-time performance monitoring
- Advanced parameter tuning capabilities