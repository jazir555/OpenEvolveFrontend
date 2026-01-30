# BubbleLabs Evolution & Adversarial - Quick Reference

## Quick Start Commands

```bash
# Run the integration
streamlit run bubblelabs_evolution_integration.py

# Run with specific port
streamlit run bubblelabs_evolution_integration.py --server.port 8502
```

## Common Workflows

### 1. Optimize Code (Standard)

```
1. Go to "Evolution Workflows" tab
2. Select "Standard Evolution"
3. Paste your code
4. Use preset "balanced"
5. Click "Start Evolution"
```

### 2. Optimize Code (MAKER Zero-Error)

```
1. Go to "Evolution Workflows" tab
2. Select "MAKER Voting Evolution"
3. Paste your code
4. Set voting_threshold = 3
5. Click "Start Evolution"
```

### 3. Security Audit

```
1. Go to "Adversarial Testing" tab
2. Select "maker_full" mode
3. Paste code to audit
4. Set 5 rounds, team size 3
5. Click "Start Adversarial Testing"
```

### 4. Prompt Refinement

```
1. Go to "Evolution Workflows" tab
2. Select "MDAP Decomposition Evolution"
3. Enter your prompt
4. Enable decomposition_depth = 3
5. Click "Start Evolution"
```

## Parameter Cheat Sheet

### Evolution Parameters

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| population_size | 2-500 | 20 | Number of individuals |
| max_generations | 1-1000 | 100 | Max iterations |
| mutation_rate | 0.0-1.0 | 0.1 | Mutation probability |
| crossover_rate | 0.0-1.0 | 0.7 | Crossover probability |
| tournament_size | 2-20 | 3 | Tournament selection size |
| elitism_count | 0-20 | 2 | Top individuals to preserve |

### MAKER Parameters

| Parameter | Range | Recommended | Purpose |
|-----------|-------|-------------|---------|
| enable_maker_voting | bool | True (for critical) | Enable voting |
| voting_threshold (k) | 1-10 | 3 | Consensus level |
| num_candidates (N) | 1-20 | 5 (N ≥ 2k-1) | Voting candidates |
| adaptive_voting | bool | True | Dynamic adjustment |

### Adversarial Parameters

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| adversarial_rounds | 1-20 | 5 | Testing rounds |
| red_team_size | 1-10 | 3 | Red team agents |
| blue_team_size | 1-10 | 3 | Blue team agents |
| attack_strength | 0.0-1.0 | 0.5 | Attack intensity |
| coevolution | bool | False | Coevolution mode |

## Preset Configurations

### Fast Exploration
```python
{
    "population_size": 10,
    "max_generations": 30,
    "mutation_rate": 0.15,
    "crossover_rate": 0.8
}
```
**Use for:** Quick tests, prototyping

### Balanced
```python
{
    "population_size": 20,
    "max_generations": 100,
    "mutation_rate": 0.1,
    "crossover_rate": 0.7
}
```
**Use for:** Standard evolution, most cases

### Deep Search
```python
{
    "population_size": 50,
    "max_generations": 200,
    "mutation_rate": 0.05,
    "crossover_rate": 0.6
}
```
**Use for:** Complex problems, thorough optimization

### MAKER Voting
```python
{
    "population_size": 25,
    "max_generations": 120,
    "enable_maker_voting": True,
    "voting_threshold": 3,
    "adaptive_voting": True
}
```
**Use for:** Zero-error requirements, critical systems

## Fitness Function Templates

### Code Quality
```python
def code_quality_fitness(code: str) -> float:
    score = 0.0
    if '"""' in code: score += 0.2  # Docstring
    if ': ' in code: score += 0.3  # Type hints
    if 'try:' in code: score += 0.3  # Error handling
    if 50 <= len(code) <= 1000: score += 0.2  # Length
    return min(score, 1.0)
```

### Conciseness
```python
def conciseness_fitness(text: str) -> float:
    # Prefer shorter texts that retain information
    word_count = len(text.split())
    if 20 <= word_count <= 100:
        return 1.0
    elif word_count < 20:
        return 0.5
    else:
        return max(0, 1.0 - (word_count - 100) / 500)
```

### Performance
```python
def performance_fitness(code: str) -> float:
    import time
    start = time.time()
    try:
        exec(code, {'__name__': '__main__'})
        elapsed = time.time() - start
        return max(0, 1.0 - elapsed / 10.0)  # Faster is better
    except:
        return 0.0
```

## Troubleshooting Quick Fixes

### Problem: No convergence
**Fix:**
- Increase mutation_rate to 0.15
- Reduce population_size to 15
- Enable diversity_maintenance

### Problem: Too slow
**Fix:**
- Reduce population_size to 10
- Decrease max_generations to 30
- Disable MAKER voting

### Problem: Poor quality results
**Fix:**
- Enable MAKER voting with k=3
- Increase population_size to 30
- Add custom fitness function

### Problem: Out of memory
**Fix:**
- Reduce population_size
- Clear task history
- Disable fitness_history tracking

## Keyboard Shortcuts (Streamlit)

| Action | Shortcut |
|--------|----------|
| Rerun app | R |
| Clear cache | C |
| Navigate | Tab |

## File Structure

```
Frontend/
├── bubblelabs_evolution_integration.py      # Main integration
├── bubblelabs_evolution_controls.py         # UI controls
├── evolution_workflow_templates.py          # Templates
├── BUBBLELABS_EVOLUTION_INTEGRATION_GUIDE.md  # Full docs
├── BUBBLELABS_EVOLUTION_QUICK_REFERENCE.md     # This file
├── evolution.py                             # Evolution engine
├── adversarial.py                           # Adversarial engine
├── evolution_maker_integration.py           # MAKER integration
└── adversarial_maker_integration.py         # Adversarial MAKER
```

## Common Use Cases

### Code Refactoring
**Template:** Code Optimization
**Mode:** MAKER Voting
**Generations:** 100
**Population:** 25

### Prompt Engineering
**Template:** Prompt Refinement
**Mode:** MDAP Decomposition
**Generations:** 75
**Population:** 20

### Security Testing
**Template:** Security Audit
**Mode:** MAKER Full
**Rounds:** 5
**Teams:** 5 red, 3 blue

### System Hardening
**Template:** Coevolution Hardening
**Mode:** Coevolution
**Rounds:** 10
**Teams:** 4 red, 4 blue

## Metric Interpretation

### Fitness Metrics
- **0.0-0.3**: Poor
- **0.3-0.6**: Fair
- **0.6-0.8**: Good
- **0.8-0.9**: Excellent
- **0.9-1.0**: Outstanding

### Diversity Metrics
- **0.0-0.1**: Low diversity (converged)
- **0.1-0.3**: Moderate diversity
- **0.3-0.6**: Good diversity
- **0.6-1.0**: High diversity (exploratory)

### Convergence Indicators
- **Plateau**: No improvement for 10+ generations
- **Divergence**: Fitness decreasing
- **Converged**: Diversity < 0.1 and stable fitness

## API Quick Reference

### Create Evolution Task
```python
integration = BubbleLabsEvolutionIntegration()
task = integration._create_evolution_task(
    task_id="my_evo",
    initial_content="my code",
    content_type="code",
    evolution_type="maker_voting",
    config={...}
)
```

### Create Adversarial Task
```python
task = integration._create_adversarial_task(
    task_id="my_adv",
    target_content="my code",
    content_type="code",
    adversarial_mode="maker_full",
    config={...}
)
```

### Render Dashboard
```python
integration.render_evolution_dashboard()
```

## Performance Benchmarks

### Small Population (10)
- Generations: 30
- Time: ~1-2 minutes
- Use for: Quick tests

### Medium Population (20)
- Generations: 100
- Time: ~5-10 minutes
- Use for: Standard evolution

### Large Population (50)
- Generations: 200
- Time: ~20-30 minutes
- Use for: Thorough search

### MAKER Voting Overhead
- Additional time: ~2-3x
- Trade-off: Zero-error guarantees
- Use for: Critical tasks

## Tips & Tricks

1. **Start with presets** - Don't tune from scratch
2. **Monitor first 10 generations** - Catch issues early
3. **Save good configurations** - Use template system
4. **Compare runs** - Use analytics dashboard
5. **Use MAKER for critical** - Worth the time cost
6. **Enable diversity** - Prevents premature convergence
7. **Check fitness function** - Most common source of issues
8. **Review history** - Learn from past runs

## Resources

### Documentation
- Full Guide: `BUBBLELABS_EVOLUTION_INTEGRATION_GUIDE.md`
- Evolution: `evolution.py`
- Adversarial: `adversarial.py`

### Templates
- Code Optimization
- Prompt Refinement
- Security Audit
- System Hardening

### Examples
- See GUIDE.md for detailed examples
- Template system has working examples
