"""Offline unit tests for the NEAT neuroevolution engine.

No LLM or network access required. Verifies that:
- A XOR-style problem is solved / fitness improves over generations.
- Mutation and crossover produce valid, evaluable networks.
- Speciation splits a population into multiple species.
- The ``neat_selection`` contract wrapper mirrors nsga3/novelty_search.
"""

import math

import numpy as np
import pytest

from openevolve.neat import (
    NEAT,
    ConnectionGene,
    Genome,
    InnovationManager,
    NodeGene,
    neat_selection,
    run_neat,
)
from openevolve.selection import select_mo


XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_TARGETS = [0.0, 1.0, 1.0, 0.0]


def xor_fitness(genome: Genome) -> float:
    error = 0.0
    for (x1, x2), y in zip(XOR_INPUTS, XOR_TARGETS):
        out = genome.forward([x1, x2])[0]
        error += (out - y) ** 2
    return 1.0 / (1.0 + error)


def test_genome_forward_minimal_network():
    neat = NEAT(n_inputs=2, n_outputs=1, random_state=0)
    genome = neat.create_minimal_genome()
    out = genome.forward([0.0, 1.0])
    assert len(out) == 1
    assert math.isfinite(out[0])
    assert -0.01 <= out[0] <= 1.01


def test_innovation_manager_stable_numbers():
    mgr = InnovationManager()
    a = mgr.get_innovation(0, 3)
    b = mgr.get_innovation(1, 3)
    assert a != b
    assert mgr.get_innovation(0, 3) == a


def test_mutation_produces_valid_network():
    neat = NEAT(n_inputs=2, n_outputs=1, random_state=1)
    genome = neat.create_minimal_genome()
    child = neat.mutate(genome)
    for (i, o), conn in child.connections.items():
        assert conn.in_node != conn.out_node
        assert child.nodes[i].depth < child.nodes[o].depth
        assert math.isfinite(conn.weight)
    out = child.forward([1.0, 0.0])
    assert len(out) == 1 and math.isfinite(out[0])


def test_add_node_mutation_increases_connections():
    neat = NEAT(n_inputs=2, n_outputs=1, random_state=2)
    genome = neat.create_minimal_genome()
    before = len(genome.connections)
    neat.add_node_mutation(genome)
    assert len(genome.connections) == before + 1
    out = genome.forward([0.0, 0.0])
    assert math.isfinite(out[0])


def test_add_connection_mutation_increases_connections():
    neat = NEAT(n_inputs=2, n_outputs=1, random_state=3)
    genome = neat.create_minimal_genome()
    before = len(genome.connections)
    neat.add_connection_mutation(genome)
    assert len(genome.connections) >= before
    out = genome.forward([1.0, 1.0])
    assert math.isfinite(out[0])


def test_crossover_produces_valid_network():
    neat = NEAT(n_inputs=2, n_outputs=1, random_state=4)
    p1 = neat.mutate(neat.create_minimal_genome())
    p2 = neat.mutate(neat.create_minimal_genome())
    child = neat.crossover(p1, p2)
    for (i, o), conn in child.connections.items():
        assert child.nodes[i].depth < child.nodes[o].depth
    assert len(child.forward([0.0, 1.0])) == 1


def test_speciation_splits_population():
    neat = NEAT(
        n_inputs=2, n_outputs=1, population_size=40, random_state=5,
        compatibility_threshold=0.5,
    )
    pop = neat.initialize_population()
    for g in pop:
        for _ in range(3):
            g = neat.mutate(g)
    species = neat.speciate(pop)
    assert len(species) > 1


def test_evolve_improves_xor_fitness():
    neat = NEAT(
        n_inputs=2, n_outputs=1, population_size=80, random_state=7,
        add_connection_prob=0.2, add_node_prob=0.1, mutate_weight_prob=1.0,
    )
    best, history = run_neat(xor_fitness, n_inputs=2, n_outputs=1,
                             population_size=80, generations=80,
                             random_state=7)
    assert history[-1] >= history[0]
    assert best is not None
    assert xor_fitness(best) > 0.6


def test_neat_selection_contract():
    rng = np.random.RandomState(0)
    objectives = rng.rand(60, 4)
    selected = neat_selection(objectives, population_size=20, random_state=0)
    assert len(selected) == 20
    assert len(set(selected)) == len(selected)


def test_select_mo_routes_neat():
    rng = np.random.RandomState(1)
    objectives = rng.rand(50, 3)
    selected = select_mo(objectives, population_size=25, method="neat", random_state=0)
    assert len(selected) == 25


def test_moconfig_accepts_neat():
    from openevolve.unified.config import MOConfig
    cfg = MOConfig(selection_method="neuroevolution")
    assert cfg.selection_method == "neat"
    cfg2 = MOConfig(selection_method="neat")
    assert cfg2.selection_method == "neat"
