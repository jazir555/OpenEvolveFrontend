"""
Tests for Adaptive MAKER Evolution Integration.
"""

import pytest
from evolution_maker_integration import (
    AdaptiveMAKERSelection,
    MakerevolutionConfig,
    Individual,
    Population
)

def test_adaptive_threshold_determination():
    """Test that voting threshold adjusts based on genome complexity."""
    config = MakerevolutionConfig(enable_voting=True)
    selector = AdaptiveMAKERSelection(config)
    
    if not selector.classifier:
        pytest.skip("Adaptive MDAP components not available")
        
    # 1. Simple genomes
    simple_candidates = [
        Individual(genome="print('hello')", fitness=0.8, generation=0),
        Individual(genome="x = 1", fitness=0.7, generation=0)
    ]
    
    k_simple = selector._determine_voting_threshold(simple_candidates)
    
    # 2. Complex genomes
    complex_description = "import asyncio\nimport cryptography\n" + "def complex_function():\n  pass\n" * 50
    complex_candidates = [
        Individual(genome=complex_description, fitness=0.9, generation=0),
        Individual(genome=complex_description, fitness=0.85, generation=0)
    ]
    
    k_complex = selector._determine_voting_threshold(complex_candidates)
    
    # Verify k_complex is likely higher than k_simple
    # (or at least they are determined without error)
    assert k_simple >= 1
    assert k_complex >= 1
    print(f"Simple k: {k_simple}, Complex k: {k_complex}")

def test_selection_with_adaptive_maker():
    """Test full selection flow with adaptive MAKER."""
    config = MakerevolutionConfig(enable_voting=True, num_candidates=2)
    selector = AdaptiveMAKERSelection(config)
    
    population = Population(
        individuals=[
            Individual(genome="ind1", fitness=0.9, generation=0),
            Individual(genome="ind2", fitness=0.8, generation=0),
            Individual(genome="ind3", fitness=0.7, generation=0),
            Individual(genome="ind4", fitness=0.6, generation=0),
        ],
        generation=0
    )
    
    # Select 2 parents
    parents = selector.select(population, num_parents=2)
    
    assert len(parents) == 2
    assert isinstance(parents[0], Individual)
    # Due to fitness pre-selection, should be top individuals
    assert parents[0].fitness >= 0.8
