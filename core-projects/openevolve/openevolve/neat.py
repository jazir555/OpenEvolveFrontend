"""NEAT: NeuroEvolution of Augmenting Topologies.

A compact, dependency-free implementation of the NEAT algorithm (Stanley &
Miikkulainen, 2002) for fixed-topology-free neuroevolution. Genomes are directed
feed-forward graphs of nodes (input / bias / hidden / output) and weighted,
innovation-numbered connections. Evolution proceeds via speciation by genomic
distance, explicit fitness sharing, and per-species reproduction with mutation
(add node / add connection / perturb or reset weights) and crossover.

The module also exposes ``neat_selection`` which mirrors the
``select_mo`` selection contract (objectives matrix -> selected indices) by
treating each individual's feature vector as a genome and applying NEAT-style
speciation, so it slots into OpenEvolve's multi-objective selection path the
same way ``nsga3`` and ``novelty_search`` do.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple


class InnovationManager:
    """Maps (in_node, out_node) pairs to stable innovation numbers."""

    def __init__(self) -> None:
        self._forward: Dict[Tuple[int, int], int] = {}
        self._next = 0

    def get_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self._forward:
            self._forward[key] = self._next
            self._next += 1
        return self._forward[key]


class NodeGene:
    """A single network node."""

    def __init__(self, node_id: int, node_type: str, depth: float = 0.0) -> None:
        self.id = node_id
        self.type = node_type
        self.depth = depth

    def copy(self) -> "NodeGene":
        return NodeGene(self.id, self.type, self.depth)

    def __repr__(self) -> str:
        return f"Node({self.id}:{self.type})"


class ConnectionGene:
    """A weighted, innovation-numbered connection between two nodes."""

    def __init__(
        self,
        in_node: int,
        out_node: int,
        weight: float,
        enabled: bool,
        innovation: int,
    ) -> None:
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

    def copy(self) -> "ConnectionGene":
        return ConnectionGene(
            self.in_node,
            self.out_node,
            self.weight,
            self.enabled,
            self.innovation,
        )

    def __repr__(self) -> str:
        flag = "+" if self.enabled else "-"
        return (
            f"Conn({self.in_node}->{self.out_node} "
            f"w={self.weight:.3f}{flag}#{self.innovation})"
        )


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _tanh(x: float) -> float:
    return math.tanh(x)


class Genome:
    """A feed-forward neural network genome (nodes + connections)."""

    def __init__(self) -> None:
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[Tuple[int, int], ConnectionGene] = {}
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0

    def copy(self) -> "Genome":
        g = Genome()
        g.nodes = {nid: node.copy() for nid, node in self.nodes.items()}
        g.connections = {
            k: conn.copy() for k, conn in self.connections.items()
        }
        g.fitness = self.fitness
        g.adjusted_fitness = self.adjusted_fitness
        return g

    def add_node(self, node: NodeGene) -> None:
        self.nodes[node.id] = node

    def add_connection(self, conn: ConnectionGene) -> None:
        self.connections[(conn.in_node, conn.out_node)] = conn

    def get_connection(self, in_node: int, out_node: int) -> Optional[ConnectionGene]:
        return self.connections.get((in_node, out_node))

    def forward(
        self,
        inputs: Sequence[float],
        hidden_activation: Callable[[float], float] = _tanh,
        output_activation: Callable[[float], float] = _sigmoid,
    ) -> List[float]:
        """Evaluate the genome on a single input vector (feed-forward)."""
        values: Dict[int, float] = {}
        inputs = list(inputs)

        input_nodes = [n for n in self.nodes.values() if n.type == "input"]
        input_nodes.sort(key=lambda node: node.id)
        for i, node in enumerate(input_nodes):
            values[node.id] = float(inputs[i]) if i < len(inputs) else 0.0
        for node in self.nodes.values():
            if node.type == "bias":
                values[node.id] = 1.0

        others = [
            node
            for node in self.nodes.values()
            if node.type not in ("input", "bias")
        ]
        others.sort(key=lambda node: (node.depth, node.id))
        for node in others:
            s = 0.0
            for conn in self.connections.values():
                if conn.enabled and conn.out_node == node.id:
                    s += values.get(conn.in_node, 0.0) * conn.weight
            act = hidden_activation if node.type != "output" else output_activation
            values[node.id] = act(s)

        output_nodes = [n for n in self.nodes.values() if n.type == "output"]
        output_nodes.sort(key=lambda node: node.id)
        return [values[n.id] for n in output_nodes]

    def distance(self, other: "Genome", neat: "NEAT") -> float:
        """Genomic compatibility distance (excess + disjoint + weight diff)."""
        c1 = neat.excess_coefficient
        c2 = neat.disjoint_coefficient
        c3 = neat.weight_coefficient

        a = self.connections
        b = other.connections
        all_keys = set(a.keys()) | set(b.keys())
        max_innov = 0
        for k in all_keys:
            max_innov = max(max_innov, a[k].innovation if k in a else b[k].innovation)

        matching = 0
        disjoint = 0
        excess = 0
        weight_diff = 0.0
        for k in all_keys:
            in_a = k in a
            in_b = k in b
            if in_a and in_b:
                matching += 1
                weight_diff += abs(a[k].weight - b[k].weight)
            else:
                innov = a[k].innovation if in_a else b[k].innovation
                if innov > max_innov:
                    excess += 1
                else:
                    disjoint += 1

        n = max(len(a), len(b), 1)
        wdiff = weight_diff / max(matching, 1)
        return (c1 * excess + c2 * disjoint) / n + c3 * wdiff


class Species:
    """A species: a representative genome plus its member genomes."""

    def __init__(self, species_id: int, representative: Genome) -> None:
        self.id = species_id
        self.representative = representative
        self.members: List[Genome] = []
        self.target_size = 0
        self.stagnation = 0
        self.best_fitness = float("-inf")

    def average_fitness(self) -> float:
        if not self.members:
            return 0.0
        return sum(m.fitness for m in self.members) / len(self.members)


class NEAT:
    """NeuroEvolution of Augmenting Topologies engine."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        population_size: int = 150,
        compatibility_threshold: float = 3.0,
        excess_coefficient: float = 1.0,
        disjoint_coefficient: float = 1.0,
        weight_coefficient: float = 0.4,
        elitism: int = 2,
        survival_threshold: float = 0.2,
        mutate_weight_prob: float = 0.9,
        perturb_prob: float = 0.75,
        weight_reset_prob: float = 0.1,
        add_connection_prob: float = 0.05,
        add_node_prob: float = 0.03,
        weight_std_dev: float = 0.5,
        max_stagnation: int = 15,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.population_size = population_size
        self.compatibility_threshold = compatibility_threshold
        self.excess_coefficient = excess_coefficient
        self.disjoint_coefficient = disjoint_coefficient
        self.weight_coefficient = weight_coefficient
        self.elitism = elitism
        self.survival_threshold = survival_threshold
        self.mutate_weight_prob = mutate_weight_prob
        self.perturb_prob = perturb_prob
        self.weight_reset_prob = weight_reset_prob
        self.add_connection_prob = add_connection_prob
        self.add_node_prob = add_node_prob
        self.weight_std_dev = weight_std_dev
        self.max_stagnation = max_stagnation
        self.rng = random.Random(random_state)
        self.innovations = InnovationManager()
        self.node_counter = 0
        self.species: List[Species] = []
        self.species_counter = 0
        self.best_genome: Optional[Genome] = None
        self.best_fitness = float("-inf")
        self.history: List[float] = []

    def _new_node_id(self) -> int:
        nid = self.node_counter
        self.node_counter += 1
        return nid

    def create_minimal_genome(self) -> Genome:
        g = Genome()
        bias_id = self._new_node_id()
        g.add_node(NodeGene(bias_id, "bias", depth=0.0))
        input_ids = []
        for i in range(self.n_inputs):
            nid = self._new_node_id()
            g.add_node(NodeGene(nid, "input", depth=0.0))
            input_ids.append(nid)
        output_ids = []
        for i in range(self.n_outputs):
            nid = self._new_node_id()
            g.add_node(NodeGene(nid, "output", depth=1e9))
            output_ids.append(nid)
        for oid in output_ids:
            conn = ConnectionGene(
                bias_id,
                oid,
                self.rng.gauss(0.0, 1.0),
                True,
                self.innovations.get_innovation(bias_id, oid),
            )
            g.add_connection(conn)
        return g

    def initialize_population(self) -> List[Genome]:
        return [self.create_minimal_genome() for _ in range(self.population_size)]

    def _valid_connections(self, genome: Genome) -> List[Tuple[int, int]]:
        candidates = []
        for a in genome.nodes.values():
            for b in genome.nodes.values():
                if a.id == b.id:
                    continue
                if b.type == "input" or b.type == "bias":
                    continue
                if a.type == "output":
                    continue
                if not (a.depth < b.depth):
                    continue
                if genome.get_connection(a.id, b.id) is not None:
                    continue
                candidates.append((a.id, b.id))
        return candidates

    def add_connection_mutation(self, genome: Genome) -> None:
        candidates = self._valid_connections(genome)
        if not candidates:
            return
        a, b = self.rng.choice(candidates)
        conn = ConnectionGene(
            a,
            b,
            self.rng.gauss(0.0, 1.0),
            True,
            self.innovations.get_innovation(a, b),
        )
        genome.add_connection(conn)

    def add_node_mutation(self, genome: Genome) -> None:
        enabled = [c for c in genome.connections.values() if c.enabled]
        if not enabled:
            return
        old = self.rng.choice(enabled)
        new_id = self._new_node_id()
        new_depth = (genome.nodes[old.in_node].depth + genome.nodes[old.out_node].depth) / 2.0
        genome.add_node(NodeGene(new_id, "hidden", depth=new_depth))

        c1 = ConnectionGene(
            old.in_node,
            new_id,
            1.0,
            True,
            self.innovations.get_innovation(old.in_node, new_id),
        )
        c2 = ConnectionGene(
            new_id,
            old.out_node,
            old.weight,
            True,
            self.innovations.get_innovation(new_id, old.out_node),
        )
        del genome.connections[(old.in_node, old.out_node)]
        genome.add_connection(c1)
        genome.add_connection(c2)

    def mutate_weights(self, genome: Genome) -> None:
        for conn in genome.connections.values():
            if self.rng.random() < self.weight_reset_prob:
                conn.weight = self.rng.gauss(0.0, 1.0)
            elif self.rng.random() < self.perturb_prob:
                conn.weight += self.rng.gauss(0.0, self.weight_std_dev)

    def mutate(self, genome: Genome) -> Genome:
        child = genome.copy()
        if self.rng.random() < self.mutate_weight_prob:
            self.mutate_weights(child)
        if self.rng.random() < self.add_connection_prob:
            self.add_connection_mutation(child)
        if self.rng.random() < self.add_node_prob:
            self.add_node_mutation(child)
        return child

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1
        child = Genome()
        for nid, node in parent1.nodes.items():
            child.add_node(node.copy())
        for key, conn1 in parent1.connections.items():
            conn2 = parent2.get_connection(*key)
            if conn2 is not None:
                chosen = conn1 if self.rng.random() < 0.5 else conn2
                inherited = chosen.copy()
                if not conn1.enabled or not conn2.enabled:
                    inherited.enabled = self.rng.random() > 0.25
                child.add_connection(inherited)
            else:
                child.add_connection(conn1.copy())
        return child

    def speciate(self, population: List[Genome]) -> List[Species]:
        if not self.species:
            self.species = []
        new_species: List[Species] = []
        for genome in population:
            placed = False
            for sp in self.species:
                if genome.distance(sp.representative, self) < self.compatibility_threshold:
                    sp.members.append(genome)
                    placed = True
                    break
            if not placed:
                rep = genome.copy()
                sp = Species(self.species_counter, rep)
                self.species_counter += 1
                sp.members.append(genome)
                new_species.append(sp)
        kept = [sp for sp in self.species if sp.members]
        kept.extend(new_species)
        self.species = kept
        return self.species

    def _adjust_species_sizes(self, population: List[Genome]) -> None:
        total_adj = sum(m.adjusted_fitness for m in population)
        if total_adj <= 0:
            size = len(population) // max(len(self.species), 1)
            for sp in self.species:
                sp.target_size = size
            return
        for sp in self.species:
            sp.target_size = int(
                round(sp.target_size if hasattr(sp, "target_size") else 0)
            )
        remaining = self.population_size
        for sp in sorted(self.species, key=lambda s: s.average_fitness(), reverse=True):
            share = sum(m.adjusted_fitness for m in sp.members) / total_adj
            sp.target_size = max(0, int(round(share * self.population_size)))
            remaining -= sp.target_size

    def _select_parent(self, species: Species) -> Genome:
        members = sorted(species.members, key=lambda m: m.fitness, reverse=True)
        cutoff = max(1, int(math.ceil(self.survival_threshold * len(members))))
        pool = members[:cutoff]
        return self.rng.choice(pool)

    def _reproduce_species(self, species: Species) -> List[Genome]:
        offspring: List[Genome] = []
        members = sorted(species.members, key=lambda m: m.fitness, reverse=True)
        elites = members[: self.elitism]
        for e in elites:
            offspring.append(e.copy())
        while len(offspring) < species.target_size:
            if len(species.members) == 1:
                child = self.mutate(species.members[0])
            else:
                p1 = self._select_parent(species)
                p2 = self._select_parent(species)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
            offspring.append(child)
        return offspring[: max(species.target_size, self.elitism)]

    def evaluate(self, population: List[Genome], fitness_fn: Callable[[Genome], float]) -> None:
        for genome in population:
            genome.fitness = fitness_fn(genome)
        gen_best = max(population, key=lambda m: m.fitness).fitness
        if gen_best > self.best_fitness:
            self.best_fitness = gen_best
            self.best_genome = max(population, key=lambda m: m.fitness).copy()
        self.history.append(self.best_fitness)

    def evolve(
        self,
        fitness_fn: Callable[[Genome], float],
        generations: int = 100,
        verbose: bool = False,
    ) -> Genome:
        population = self.initialize_population()
        for gen in range(generations):
            self.evaluate(population, fitness_fn)

            self.speciate(population)
            for sp in self.species:
                for m in sp.members:
                    m.adjusted_fitness = m.fitness / max(len(sp.members), 1)
                if sp.average_fitness() >= sp.best_fitness:
                    sp.best_fitness = sp.average_fitness()
                    sp.stagnation = 0
                else:
                    sp.stagnation += 1

            self.species = [
                sp for sp in self.species
                if sp.stagnation < self.max_stagnation and sp.target_size > 0
                or sp in self.species[:1]
            ]
            if not self.species:
                self.species = [Species(self.species_counter, population[0].copy())]
                self.species_counter += 1

            self._adjust_species_sizes(population)
            next_pop: List[Genome] = []
            for sp in self.species:
                next_pop.extend(self._reproduce_species(sp))
            while len(next_pop) < self.population_size:
                next_pop.append(self.create_minimal_genome())
            population = next_pop[: self.population_size]

            if verbose:
                print(
                    f"gen {gen}: best={self.best_fitness:.4f} "
                    f"species={len(self.species)}"
                )

        return self.best_genome if self.best_genome is not None else population[0]


def run_neat(
    fitness_fn: Callable[[Genome], float],
    n_inputs: int,
    n_outputs: int,
    population_size: int = 150,
    generations: int = 100,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Genome, List[float]]:
    """Convenience entry point for a single NEAT run.

    Returns the best genome found and the per-generation best-fitness history.
    """
    engine = NEAT(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        population_size=population_size,
        random_state=random_state,
    )
    best = engine.evolve(fitness_fn, generations=generations, verbose=verbose)
    return best, engine.history


def neat_selection(
    objectives: "object",
    population_size: int,
    compatibility_threshold: float = 0.5,
    random_state: Optional[int] = None,
) -> List[int]:
    """``select_mo``-style wrapper: speciate feature vectors and select diverse,
    high-fitness individuals.

    ``objectives`` is a 2D array (rows = individuals, columns = features). Each
    row is treated as a genome; compatibility is Euclidean distance. Returns the
    indices selected for the next generation (best-per-species plus filler).
    """
    import numpy as np

    matrix = np.asarray(objectives, dtype=float)
    n = matrix.shape[0]
    if n == 0:
        return []
    rng = random.Random(random_state) if isinstance(random_state, int) else random.Random(random_state)

    def dist(i: int, j: int) -> float:
        return float(np.linalg.norm(matrix[i] - matrix[j]))

    species: List[List[int]] = []
    for idx in range(n):
        placed = False
        for sp in species:
            if dist(idx, sp[0]) < compatibility_threshold:
                sp.append(idx)
                placed = True
                break
        if not placed:
            species.append([idx])

    selected: List[int] = []
    for sp in species:
        sp_sorted = sorted(sp, key=lambda i: float(matrix[i].sum()), reverse=True)
        elite = sp_sorted[: max(1, len(sp_sorted) // 4)]
        selected.extend(elite)

    filler = sorted(
        range(n), key=lambda i: float(matrix[i].sum()), reverse=True
    )
    for i in filler:
        if len(selected) >= population_size:
            break
        if i not in selected:
            selected.append(i)
    return selected[:population_size]
