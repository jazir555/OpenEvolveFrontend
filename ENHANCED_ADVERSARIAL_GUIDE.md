# Enhanced Adversarial Testing - Integration Guide

## Overview

This guide explains how to use the enhanced adversarial testing system with all advanced features enabled.

## New Features

### 1. AI-Driven Attack Generation
- **LLM-Guided Attacks**: Use GPT-4 to generate sophisticated, context-aware attacks
- **Mutation-Based Attacks**: Build on successful attack patterns
- **Semantic Attacks**: Target logical meaning rather than just syntax
- **Adaptive Attacks**: Adjust strategies based on defense responses

### 2. Adaptive Defense Mechanisms
- **Dynamic Adaptation**: Automatically adjust defenses based on threat level
- **Ensemble Defenses**: Combine multiple defense strategies
- **Formal Verification Integration**: Use LeanAide for mathematical proofs
- **Real-time Strategy Adjustment**: Adapt every N iterations

### 3. Explainability Framework
- **Basic Level**: High-level summaries
- **Detailed Level**: Step-by-step reasoning
- **Full Level**: Complete trace with internal states
- **User-Friendly Explanations**: Non-technical descriptions

### 4. Continuous Learning
- **Offline Learning**: Learn from historical data
- **Online Learning**: Learn in real-time during testing
- **Experience Buffer**: Store past encounters
- **Pattern Recognition**: Identify successful strategies

### 5. Ensemble Attack System
- **Diverse Strategies**: Combine multiple attack types
- **Voting Mechanisms**: Majority, weighted, or soft voting
- **Weight Rebalancing**: Dynamically adjust strategy weights
- **Performance Tracking**: Monitor ensemble effectiveness

### 6. Advanced Analytics
- **Attack Pattern Tracking**: Monitor which attacks work best
- **Defense Effectiveness**: Track defense success rates
- **Adaptation Metrics**: Measure improvement over time
- **Comprehensive Reports**: JSON, HTML, or PDF output

## Quick Start

### Basic Usage

```python
from adversarial_advanced import (
    EnhancedAdversarialEngine,
    AdvancedAdversarialConfig,
    create_enhanced_config,
    quick_enhanced_test
)

# Option 1: Quick test with defaults
result = quick_enhanced_test(
    content="def my_function(): ...",
    content_type="code_python",
    theorem="Function correctness"
)

print(f"Robustness: {result['final_robustness']:.2%}")
print(f"Duration: {result['duration']:.2f}s")
```

### Advanced Configuration

```python
# Create custom configuration
config = create_enhanced_config(
    enable_advanced_features=True,
    # AI-Driven Attacks
    enable_llm_attacks=True,
    llm_attack_model="gpt-4",
    llm_attack_temperature=0.8,

    # Adaptive Defense
    enable_adaptive_defense=True,
    defense_adaptation_rate=0.1,
    defense_diversity_threshold=0.3,

    # Explainability
    explainability_level="detailed",  # basic, detailed, full
    explain_to_user=True,
    include_internal_states=False,

    # Learning
    learning_mode="online",  # offline, online, hybrid
    learning_rate=0.01,
    experience_buffer_size=1000,

    # Ensemble
    enable_ensemble=True,
    ensemble_size=5,
    voting_strategy="weighted",  # majority, weighted, soft

    # Analytics
    enable_advanced_analytics=True,
    generate_reports=True,
    report_format="json"
)

# Create engine with custom config
engine = EnhancedAdversarialEngine(config)

# Run testing
result = await engine.enhanced_adversarial_test(
    content=your_code,
    content_type="code_python",
    theorem="Your theorem statement",
    max_iterations=10
)
```

## Integration with Existing System

### Option 1: Replace Existing Adversarial Testing

```python
# Old way
from adversarial import run_comprehensive_adversarial_testing
result = run_comprehensive_adversarial_testing(content, content_type, config)

# New way (drop-in replacement)
from adversarial_advanced import EnhancedAdversarialEngine, AdvancedAdversarialConfig

config = AdvancedAdversarialConfig()
engine = EnhancedAdversarialEngine(config)
result = await engine.enhanced_adversarial_test(
    content=content,
    content_type=content_type,
    theorem=theorem,
    max_iterations=config.adversarial_rounds
)
```

### Option 2: Gradual Migration

```python
from adversarial_advanced import AdaptiveDefenseSystem, ContinuousLearningSystem

# Add to existing system
from adversarial import RedTeam, BlueTeam

class EnhancedRedTeam(RedTeam):
    def __init__(self, team_size, config):
        super().__init__(team_size, config)
        # Add learning system
        self.learning = ContinuousLearningSystem(config)

    async def generate_attacks(self, proof, theorem):
        # Get recommendation from learning system
        strategy, confidence = self.learning.recommend_attack_strategy(
            {"proof": proof, "theorem": theorem},
            list(AdvancedAttackStrategy)
        )

        # Use strategy to generate attacks
        attacks = await super().generate_attacks(proof, theorem)

        # Record experiences
        for attack in attacks:
            self.learning.record_experience(
                attack=attack,
                defense=None,
                outcome={"success": attack.success}
            )

        return attacks
```

## Result Structure

```python
{
    "success": True,
    "content": "original content",
    "final_content": "improved content",
    "content_type": "code_python",
    "theorem": "theorem statement",
    "iterations_completed": 10,
    "final_robustness": 0.87,  # 0.0 to 1.0
    "duration": 45.23,  # seconds

    # Attack and defense history
    "attacks": [
        {
            "attack_id": "...",
            "attack_strategy": "edges",
            "success": true,
            "severity": 0.75,
            "description": "...",
            "confidence": 0.85
        },
        # ... more attacks
    ],

    "defenses": [
        {
            "defense_id": "...",
            "defense_strategy": "reinforce",
            "attack_blocked": true,
            "effectiveness": 0.85,
            "improved_proof": "...",
            "description": "...",
            "confidence": 0.90
        },
        # ... more defenses
    ],

    # Explanations
    "explanations": [
        {
            "timestamp": "2025-01-07T12:00:00",
            "decision": "attack_selection",
            "strategy": "llm_guided",
            "level": "detailed",
            "reasoning": "LLM-guided attacks provide sophisticated...",
            "user_friendly": "Using AI to find vulnerabilities...",
            "confidence": 0.82
        },
        # ... more explanations
    ],

    # Adaptations
    "adaptations": [
        {
            "adaptation_recommended": True,
            "threat_level": 0.65,
            "most_common_attacks": [("edges", 5), ("assumptions", 3)],
            "recommended_defenses": [
                "robustness_testing",
                "constrained_generation",
                "adaptive_defense"
            ],
            "adaptation_strength": 0.72
        },
        # ... more adaptations
    ],

    # Metrics
    "metrics": {
        "total_attacks": 25,
        "successful_attacks": 12,
        "total_defenses": 12,
        "successful_defenses": 10,
        "adaptations_performed": 3,
        "attack_success_rate": 0.48,
        "defense_success_rate": 0.83,
        "explanations_generated": 50,
        "learning_improvements": 5
    },

    # Learning insights
    "learning_insights": {
        "total_experiences": 100,
        "overall_success_rate": 0.52,
        "most_successful_attacks": [
            ("semantic_attack", 0.78),
            ("llm_guided", 0.72),
            ("adaptive_attack", 0.68)
        ],
        "most_effective_defenses": [
            ("formal_verification", 0.92),
            ("lean_integration", 0.88),
            ("semantic_validation", 0.85)
        ],
        "attack_defense_correlations": {
            "edges": {
                "robustness_testing": 0.85,
                "constrained_generation": 0.78
            },
            "assumptions": {
                "semantic_validation": 0.91,
                "formal_verification": 0.88
            }
        }
    },

    # Explainability summary
    "explainability_summary": {
        "total_explanations": 50,
        "attack_selections": 25,
        "defense_selections": 25
    }
}
```

## Feature Configuration

### Enable/Disable Features

```python
# Minimal configuration (fast)
config = create_enhanced_config(
    enable_advanced_features=False,
    enable_llm_attacks=False,
    enable_adaptive_defense=False,
    explainability_level="basic",
    learning_mode="offline",
    enable_ensemble=False
)

# Maximum configuration (thorough)
config = create_enhanced_config(
    enable_advanced_features=True,
    enable_llm_attacks=True,
    enable_adaptive_defense=True,
    explainability_level="full",
    learning_mode="hybrid",
    enable_ensemble=True,
    ensemble_size=7,
    enable_advanced_analytics=True,
    enable_realtime_adaptation=True,
    enable_zero_knowledge_proofs=True,  # Experimental
    enable_formal_verification=True,
    enable_symbolic_execution=True
)
```

### Performance Tuning

```python
# For speed (quick testing)
config = create_enhanced_config(
    max_iterations=3,
    ensemble_size=3,
    explainability_level="basic",
    enable_caching=True,
    cache_size=10000
)

# For quality (thorough testing)
config = create_enhanced_config(
    max_iterations=20,
    ensemble_size=7,
    explainability_level="full",
    enable_adaptive_defense=True,
    adaptation_interval=5,
    learning_mode="hybrid",
    experience_buffer_size=5000
)
```

## Use Cases

### 1. Security Testing for Code

```python
code = """
def process_payment(card_number, amount):
    # Process payment
    return gateway.charge(card_number, amount)
"""

result = quick_enhanced_test(
    content=code,
    content_type="code_python",
    theorem="Secure payment processing"
)

# Check vulnerabilities
for attack in result['attacks']:
    if attack['success']:
        print(f"Vulnerability: {attack['description']}")
        print(f"Severity: {attack['severity']}")
        print(f"Recommendation: See defense in results")
```

### 2. Document Review

```python
document = """
# Security Policy
All employees must use strong passwords.
Passwords should be changed monthly.
"""

result = quick_enhanced_test(
    content=document,
    content_type="document_general",
    theorem="Comprehensive security policy"
)

# Get improvements
print(f"Original robustness: Low")
print(f"Final robustness: {result['final_robustness']:.2%}")
print(f"Improved document:\n{result['final_content']}")
```

### 3. API Specification Testing

```python
api_spec = """
POST /api/users
{
  "username": string,
  "password": string,
  "email": string
}
No authentication required
"""

result = quick_enhanced_test(
    content=api_spec,
    content_type="api_spec",
    theorem="Secure user creation API"
)

# Analyze results
print("Security issues found:")
for attack in result['attacks']:
    if attack['success']:
        print(f"  - {attack['description']} (severity: {attack['severity']})")
```

## Best Practices

### 1. Start Simple, Then Scale Up

```python
# Step 1: Basic testing (fast)
result = quick_enhanced_test(content, content_type, theorem)

# Step 2: Enable specific features
config = create_enhanced_config(
    enable_llm_attacks=True,
    enable_adaptive_defense=True
)
engine = EnhancedAdversarialEngine(config)
result = await engine.enhanced_adversarial_test(...)

# Step 3: Full featured testing
config = create_enhanced_config(enable_advanced_features=True)
```

### 2. Use Learning System for Continuous Improvement

```python
# First run - establishes baseline
result1 = quick_enhanced_test(content, content_type, theorem)

# Second run - learns from first run
result2 = quick_enhanced_test(content, content_type, theorem)

# Check learning insights
print("Learning insights:")
print(result2['learning_insights']['most_successful_attacks'])
print(result2['learning_insights']['most_effective_defenses'])
```

### 3. Leverage Explainability for Understanding

```python
config = create_enhanced_config(
    explainability_level="detailed",
    explain_to_user=True
)

result = await engine.enhanced_adversarial_test(...)

# Review explanations
for explanation in result['explanations']:
    print(f"Decision: {explanation['decision']}")
    print(f"Reasoning: {explanation['reasoning']}")
    print(f"User-friendly: {explanation['user_friendly']}")
    print()
```

### 4. Monitor Adaptations

```python
config = create_enhanced_config(
    enable_realtime_adaptation=True,
    adaptation_interval=5,
    enable_advanced_analytics=True
)

result = await engine.enhanced_adversarial_test(...)

# Review adaptations
print(f"Adaptations performed: {len(result['adaptations'])}")
for adaptation in result['adaptations']:
    print(f"  Threat level: {adaptation['threat_level']:.2%}")
    print(f"  Recommended defenses: {adaptation['recommended_defenses']}")
```

## Performance Considerations

### Speed vs Quality Trade-off

| Feature | Speed Impact | Quality Impact |
|---------|-------------|----------------|
| LLM Attacks | High | High |
| Ensemble Attacks | Medium | High |
| Adaptive Defense | Low | Medium |
| Explainability | Low-Medium | Low |
| Learning System | Low | Medium |
| Advanced Analytics | Low | Low |

### Recommended Settings

**Fast Testing (Development)**
```python
config = create_enhanced_config(
    max_iterations=3,
    enable_llm_attacks=False,  # Skip for speed
    ensemble_size=3,
    explainability_level="basic",
    learning_mode="offline"
)
```

**Balanced Testing (CI/CD)**
```python
config = create_enhanced_config(
    max_iterations=7,
    enable_llm_attacks=True,
    ensemble_size=5,
    explainability_level="detailed",
    learning_mode="online"
)
```

**Thorough Testing (Security Audit)**
```python
config = create_enhanced_config(
    max_iterations=15,
    enable_llm_attacks=True,
    ensemble_size=7,
    explainability_level="full",
    learning_mode="hybrid",
    enable_all_advanced_features=True
)
```

## Troubleshooting

### Issue: Slow Performance

**Solution**: Reduce ensemble size or disable LLM attacks
```python
config = create_enhanced_config(
    ensemble_size=3,
    enable_llm_attacks=False
)
```

### Issue: Low Robustness Scores

**Solution**: Enable more features and increase iterations
```python
config = create_enhanced_config(
    max_iterations=20,
    enable_ensemble=True,
    enable_adaptive_defense=True,
    learning_mode="hybrid"
)
```

### Issue: Too Many False Positives

**Solution**: Adjust severity thresholds
```python
# In your custom attack generation
attack.severity = max(0.5, calculated_severity)  # Minimum 0.5
```

## Migration Checklist

- [ ] Import new modules
- [ ] Replace run_comprehensive_adversarial_testing with EnhancedAdversarialEngine
- [ ] Configure advanced features based on needs
- [ ] Update result processing (new structure)
- [ ] Enable/disable features appropriately
- [ ] Test with sample content
- [ ] Review explainability output
- [ ] Monitor learning insights
- [ ] Adjust performance settings
- [ ] Deploy to production

## API Reference

See `adversarial_advanced.py` for complete API documentation.

Key classes:
- `EnhancedAdversarialEngine`: Main testing engine
- `AdvancedAdversarialConfig`: Configuration
- `ExplainabilityFramework`: Decision explanations
- `ContinuousLearningSystem`: Learning from experience
- `AdaptiveDefenseSystem`: Dynamic defense adjustment
- `EnsembleAttackSystem`: Combined attack strategies

## Support

For issues or questions:
1. Check the examples in `demo_enhanced_adversarial.py`
2. Review the code documentation in `adversarial_advanced.py`
3. Examine the integration tests
4. Check logs for detailed error messages
