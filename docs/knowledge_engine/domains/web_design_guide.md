# Web Design Domain Guide

**Version:** 1.0
**Last Updated:** January 30, 2026

---

## Domain Overview

### What Problems Does This Domain Solve?

- **Landing Page Optimization** - Layout, copy, CTAs
- **UX Optimization** - Navigation, user flow, accessibility
- **Conversion Rate Optimization** - Forms, checkout, sign-ups
- **A/B Testing** - Variant generation, multivariate testing
- **Performance Optimization** - Load time, responsiveness

### Unique Challenges

1. **Very Fast Evaluations** - Can test 100s of variants per hour
2. **Human Perception** - Subjective quality metrics
3. **Business Impact** - Direct revenue impact
4. **Multi-Modal** - Text, images, layout, color
5. **Real-Time Adaptation** - Need quick iterations

### Why Evolutionary Optimization?

Traditional methods (A/B testing) are slow and test few variants. Evolutionary methods:
- Test many variants in parallel
- Combine best elements from multiple variants
- Adapt to user behavior in real-time
- Find unexpected winning combinations

---

## Recommended Approach

### Best System: OpenEvolve

**Why?**
- Standard GA works well for cheap evaluations
- Can explore many variants quickly
- Easy to parallelize

### Best Mode: Standard GA

**Why Standard?**
- Evaluations are very fast (milliseconds)
- Can afford large population
- Simple crossover/mutation works well
- No need for complex planning

---

## Configuration

```python
from openevolve.unified import UnifiedEvolutionConfig

web_config = UnifiedEvolutionConfig(
    domain="web_design",
    evolution_mode="standard",
    max_evaluations=500,  # Can afford many

    # Objectives
    objectives=["conversion_rate", "bounce_rate", "time_on_page"],

    # Large population for diversity
    population_size=200,

    # Fast iterations
    max_iterations=50,
    convergence_threshold=0.01
)
```

---

## Examples

### Example 1: Landing Page Optimization

```python
problem = """
Optimize landing page for conversions.

Elements to optimize:
- Headline: 5 variants
- Hero image: 3 variants
- CTA button: 4 variants
- Layout: 3 variants
- Color scheme: 5 variants

Objectives:
- Maximize conversion rate
- Minimize bounce rate
- Maximize time on page
"""

result = await evolve(
    problem=problem,
    domain="web_design",
    max_evaluations=500,
    objectives=["conversion_rate", "bounce_rate", "time_on_page"]
)

print(f"Best conversion rate: {result['objectives']['conversion_rate']:.2%}")
print(f"Bounce rate: {result['objectives']['bounce_rate']:.2%}")
print(f"Time on page: {result['objectives']['time_on_page']:.1f}s")
print(f"Variants tested: {result['evaluations']}")
```

---

## Best Practices

### 1. Use Real User Data

```python
# Connect to analytics
def evaluate_variant(variant):
    # Deploy to subset of users
    deploy_to_production(variant, traffic_fraction=0.01)

    # Wait for statistically significant data
    wait_for_significance(min_samples=1000)

    # Get metrics from analytics
    metrics = analytics.get_metrics(variant_id=variant['id'])
    return metrics['conversion_rate']
```

### 2. Statistical Significance

```python
# Ensure enough data
config = UnifiedEvolutionConfig(
    min_samples_per_variant=1000,
    confidence_level=0.95,
    significance_threshold=0.05
)
```

### 3. Multi-Armed Bandit

```python
# Use bandit for faster learning
result = await evolve(
    problem=problem,
    domain="web_design",
    selection_method="ucb",  # Upper confidence bound
    exploration_rate=0.1
)
```

---

**End of Web Design Domain Guide**
