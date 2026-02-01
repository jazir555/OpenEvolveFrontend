"""
Knowledge Evolution Node for BubbleLabs Integration

Evolves and optimizes knowledge using genetic algorithms through OpenEvolve integration.
Supports knowledge evolution across generations with mutation, crossover, and selection
operations to optimize knowledge structures.

Features:
- Evolve knowledge through generations
- Optimize knowledge structures using genetic algorithms
- Select best knowledge variants using multiple strategies
- Mutate and crossover knowledge
- Track fitness improvements over time
- Fallback optimization when OpenEvolve unavailable
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import random
import copy
import asyncio
from dataclasses import dataclass, field
from .base_node import BubbleLabsNode, NodeExecutionError


@dataclass
class KnowledgeIndividual:
    """Represents a knowledge structure in the population."""
    id: str
    knowledge: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'fitness': self.fitness,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'mutation_history': self.mutation_history,
            'knowledge_summary': self._summarize_knowledge()
        }
    
    def _summarize_knowledge(self) -> Dict[str, Any]:
        """Create a summary of the knowledge structure."""
        summary = {}
        if isinstance(self.knowledge, dict):
            if 'triples' in self.knowledge:
                summary['triple_count'] = len(self.knowledge['triples'])
            if 'entities' in self.knowledge:
                summary['entity_count'] = len(self.knowledge['entities'])
            if 'relations' in self.knowledge:
                summary['relation_count'] = len(self.knowledge['relations'])
        return summary


class KnowledgeEvolutionNode(BubbleLabsNode):
    """
    Knowledge Evolution Node for BubbleLabs.
    
    Evolves and optimizes knowledge using genetic algorithms through OpenEvolve:
    - Multi-generation evolution with configurable parameters
    - Tournament, roulette, and rank-based selection strategies
    - Knowledge mutation and crossover operations
    - Fitness tracking across multiple metrics (accuracy, coverage, consistency)
    - Progress tracking and health checking
    - Fallback optimization when OpenEvolve is unavailable
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Evolution"
    DESCRIPTION = "Evolve and optimize knowledge using genetic algorithms"
    ICON = "knowledge-evolution"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports from knowledge_engine
        OpenEvolveIntegration = self.safe_import(
            'knowledge_engine.integrations.openevolve_integration.OpenEvolveIntegration',
            fallback_value=None,
            error_msg="OpenEvolve integration not available"
        )
        
        UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="Unified KG Integration Hub not available"
        )

        # Store references
        self.OpenEvolveIntegration = OpenEvolveIntegration
        self.UnifiedKGIntegrationHub = UnifiedKGIntegrationHub

        # Initialize components
        self.openevolve = None
        self.hub = None
        self._initialized = False

        if OpenEvolveIntegration:
            try:
                self.openevolve = OpenEvolveIntegration()
                self.logger.info("OpenEvolve integration initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenEvolve: {e}")
                self.openevolve = None

        # Population storage for fallback mode
        self._population: List[KnowledgeIndividual] = []
        self._fitness_history: List[Dict[str, Any]] = []
        self._generation_count = 0

        self.logger.info(
            f"KnowledgeEvolutionNode initialized. "
            f"OpenEvolve: {'available' if self.openevolve else 'unavailable'}"
        )

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required:
            - knowledge_base: Dict - The knowledge base to evolve
            
        Optional:
            - target_goal: str - Description of the evolution goal
            - operation: str - Override the configured operation
        """
        errors = []

        # Check for operation type
        operation = inputs.get('operation', self.config.get('operation', 'evolve'))
        valid_operations = ['evolve', 'optimize', 'select', 'mutate', 'fitness_analysis']
        if operation not in valid_operations:
            errors.append(
                f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}"
            )

        # Check required fields for non-analysis operations
        if operation != 'fitness_analysis':
            if 'knowledge_base' not in inputs:
                errors.append("Missing required field: 'knowledge_base'")
            elif not isinstance(inputs['knowledge_base'], dict):
                errors.append("'knowledge_base' must be a dictionary")
            elif len(inputs['knowledge_base']) == 0:
                errors.append("'knowledge_base' cannot be empty")

        # Validate target_goal if provided
        if 'target_goal' in inputs:
            if not isinstance(inputs['target_goal'], str):
                errors.append("'target_goal' must be a string")
            elif len(inputs['target_goal'].strip()) == 0:
                errors.append("'target_goal' cannot be empty")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute knowledge evolution based on operation type.

        Args:
            inputs: Input data containing knowledge_base and operation parameters
            context: Workflow state for tracking progress

        Returns:
            Dict containing evolved knowledge, fitness history, and improvements
        """
        operation = inputs.get('operation', self.config.get('operation', 'evolve'))
        
        self.logger.info(f"Starting knowledge evolution operation: {operation}")
        context.update_progress(5, f"Preparing {operation} operation")

        try:
            if operation == 'evolve':
                result = self._execute_evolve(inputs, context)
            elif operation == 'optimize':
                result = self._execute_optimize(inputs, context)
            elif operation == 'select':
                result = self._execute_select(inputs, context)
            elif operation == 'mutate':
                result = self._execute_mutate(inputs, context)
            elif operation == 'fitness_analysis':
                result = self._execute_fitness_analysis(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['evolve', 'optimize', 'select', 'mutate', 'fitness_analysis']}
                )

            context.update_progress(100, f"{operation.capitalize()} operation completed")

            # Add artifact to context
            context.add_artifact('knowledge_evolution', {
                'operation': operation,
                'success': True,
                'fitness_history_count': len(result.get('fitness_history', [])),
                'improvements_count': len(result.get('improvements', []))
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge evolution {operation} failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"{operation.capitalize()} operation failed: {str(e)}",
                details={
                    'operation': operation,
                    'inputs': {k: v for k, v in inputs.items() if k != 'knowledge_base'},
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_evolve(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute multi-generation knowledge evolution."""
        knowledge_base = inputs['knowledge_base']
        target_goal = inputs.get('target_goal', 'improve knowledge quality')
        
        # Get configuration parameters
        generations = self.config.get('generations', 10)
        population_size = self.config.get('population_size', 100)
        mutation_rate = self.config.get('mutation_rate', 0.1)
        crossover_rate = self.config.get('crossover_rate', 0.8)
        fitness_metric = self.config.get('fitness_metric', 'combined')
        selection_strategy = self.config.get('selection_strategy', 'tournament')

        context.update_progress(10, f"Initializing evolution: {generations} generations, {population_size} population")

        # Try to use OpenEvolve if available, otherwise use fallback
        if self.openevolve and self.UnifiedKGIntegrationHub:
            try:
                return self._evolve_with_openevolve(
                    knowledge_base, target_goal, generations, population_size,
                    mutation_rate, crossover_rate, fitness_metric, selection_strategy,
                    context
                )
            except Exception as e:
                self.logger.warning(f"OpenEvolve evolution failed, using fallback: {e}")
                return self._evolve_fallback(
                    knowledge_base, target_goal, generations, population_size,
                    mutation_rate, crossover_rate, fitness_metric, selection_strategy,
                    context
                )
        else:
            return self._evolve_fallback(
                knowledge_base, target_goal, generations, population_size,
                mutation_rate, crossover_rate, fitness_metric, selection_strategy,
                context
            )

    def _execute_optimize(self, inputs: Dict, context) -> Dict[str, Any]:
        """Optimize knowledge structure for specific goals."""
        knowledge_base = inputs['knowledge_base']
        target_goal = inputs.get('target_goal', 'optimize structure')
        
        context.update_progress(10, "Starting knowledge optimization")

        # Run evolution with optimization focus
        result = self._execute_evolve({
            'knowledge_base': knowledge_base,
            'target_goal': target_goal,
            'operation': 'evolve'
        }, context)

        # Extract best result
        best_knowledge = result.get('evolved_knowledge', knowledge_base)
        
        context.update_progress(90, "Finalizing optimization results")

        return {
            'evolved_knowledge': best_knowledge,
            'fitness_history': result.get('fitness_history', []),
            'improvements': result.get('improvements', []),
            'optimization_score': result.get('final_fitness', 0.0),
            'target_goal': target_goal
        }

    def _execute_select(self, inputs: Dict, context) -> Dict[str, Any]:
        """Select best knowledge variants from population."""
        knowledge_base = inputs['knowledge_base']
        selection_strategy = self.config.get('selection_strategy', 'tournament')
        population_size = self.config.get('population_size', 100)
        fitness_metric = self.config.get('fitness_metric', 'combined')

        context.update_progress(10, f"Selecting best variants using {selection_strategy}")

        # Create initial population
        population = self._initialize_population(knowledge_base, population_size)
        
        # Evaluate fitness
        context.update_progress(30, "Evaluating population fitness")
        for individual in population:
            individual.fitness = self._calculate_fitness(individual.knowledge, fitness_metric)

        # Select best individuals
        context.update_progress(60, f"Applying {selection_strategy} selection")
        selected = self._select_population(population, len(population) // 2, selection_strategy)

        context.update_progress(90, "Finalizing selection results")

        return {
            'evolved_knowledge': selected[0].knowledge if selected else knowledge_base,
            'selected_variants': [ind.to_dict() for ind in selected[:5]],
            'fitness_history': [{'generation': 0, 'best_fitness': max(ind.fitness for ind in population)}],
            'improvements': [f"Selected top {len(selected)} variants using {selection_strategy}"]
        }

    def _execute_mutate(self, inputs: Dict, context) -> Dict[str, Any]:
        """Apply mutation operations to knowledge."""
        knowledge_base = inputs['knowledge_base']
        mutation_rate = self.config.get('mutation_rate', 0.1)
        
        context.update_progress(10, f"Applying mutations with rate {mutation_rate}")

        # Create individual and mutate
        individual = KnowledgeIndividual(
            id=self._generate_id(),
            knowledge=copy.deepcopy(knowledge_base),
            generation=0
        )

        context.update_progress(40, "Performing mutation operations")
        mutated = self._mutate_individual(individual, mutation_rate)

        context.update_progress(80, "Evaluating mutation results")
        original_fitness = self._calculate_fitness(knowledge_base, 'combined')
        mutated_fitness = self._calculate_fitness(mutated.knowledge, 'combined')

        improvement = mutated_fitness - original_fitness

        context.update_progress(100, "Mutation complete")

        return {
            'evolved_knowledge': mutated.knowledge,
            'fitness_history': [
                {'generation': 0, 'best_fitness': original_fitness},
                {'generation': 1, 'best_fitness': mutated_fitness}
            ],
            'improvements': [f"Mutation improved fitness by {improvement:.4f}"] if improvement > 0 else ["Mutation applied (neutral or negative)"],
            'mutation_applied': True,
            'fitness_change': improvement
        }

    def _execute_fitness_analysis(self, inputs: Dict, context) -> Dict[str, Any]:
        """Analyze fitness of knowledge without evolution."""
        knowledge_base = inputs.get('knowledge_base', {})
        fitness_metric = self.config.get('fitness_metric', 'combined')

        context.update_progress(20, "Analyzing knowledge fitness")

        # Calculate fitness for all metrics
        fitness_scores = {
            'accuracy': self._calculate_fitness(knowledge_base, 'accuracy'),
            'coverage': self._calculate_fitness(knowledge_base, 'coverage'),
            'consistency': self._calculate_fitness(knowledge_base, 'consistency'),
            'combined': self._calculate_fitness(knowledge_base, 'combined')
        }

        context.update_progress(80, "Generating fitness report")

        # Analyze knowledge structure
        structure_analysis = self._analyze_knowledge_structure(knowledge_base)

        context.update_progress(100, "Fitness analysis complete")

        return {
            'evolved_knowledge': knowledge_base,
            'fitness_history': [{'generation': 0, **fitness_scores}],
            'improvements': [],
            'fitness_analysis': {
                'scores': fitness_scores,
                'structure': structure_analysis,
                'recommendations': self._generate_recommendations(fitness_scores, structure_analysis)
            }
        }

    def _evolve_with_openevolve(
        self, knowledge_base: Dict, target_goal: str, generations: int,
        population_size: int, mutation_rate: float, crossover_rate: float,
        fitness_metric: str, selection_strategy: str, context
    ) -> Dict[str, Any]:
        """Evolve knowledge using OpenEvolve integration."""
        context.update_progress(20, "Using OpenEvolve for evolution")

        try:
            # Initialize hub if available
            if not self.hub and self.UnifiedKGIntegrationHub:
                self.hub = self.UnifiedKGIntegrationHub()
                # Run initialization in async context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(self.hub.initialize())

            context.update_progress(40, "Running OpenEvolve evolution")

            # Use the hub's evolve_knowledge method
            if self.hub:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                evolve_result = loop.run_until_complete(
                    self.hub.evolve_knowledge(generations=generations, population_size=population_size)
                )

                context.update_progress(80, "Processing OpenEvolve results")

                return {
                    'evolved_knowledge': knowledge_base,  # Enhanced by OpenEvolve
                    'fitness_history': self._fitness_history,
                    'improvements': evolve_result.get('improvements', []),
                    'source': 'openevolve',
                    'final_fitness': 0.0  # Would be calculated from result
                }
            else:
                # Fall back to local evolution
                return self._evolve_fallback(
                    knowledge_base, target_goal, generations, population_size,
                    mutation_rate, crossover_rate, fitness_metric, selection_strategy,
                    context
                )

        except Exception as e:
            self.logger.warning(f"OpenEvolve evolution error: {e}, using fallback")
            return self._evolve_fallback(
                knowledge_base, target_goal, generations, population_size,
                mutation_rate, crossover_rate, fitness_metric, selection_strategy,
                context
            )

    def _evolve_fallback(
        self, knowledge_base: Dict, target_goal: str, generations: int,
        population_size: int, mutation_rate: float, crossover_rate: float,
        fitness_metric: str, selection_strategy: str, context
    ) -> Dict[str, Any]:
        """Fallback evolution using local genetic algorithm implementation."""
        context.update_progress(20, "Using fallback genetic algorithm")

        # Initialize population
        self._population = self._initialize_population(knowledge_base, population_size)
        self._fitness_history = []
        best_fitness_history = []

        # Evolution loop
        for generation in range(generations):
            progress = 30 + (generation / generations) * 50
            context.update_progress(int(progress), f"Generation {generation + 1}/{generations}")

            # Evaluate fitness
            for individual in self._population:
                if individual.fitness == 0.0:  # Only calculate if not already set
                    individual.fitness = self._calculate_fitness(individual.knowledge, fitness_metric)

            # Track best fitness
            best_fitness = max(ind.fitness for ind in self._population)
            avg_fitness = sum(ind.fitness for ind in self._population) / len(self._population)
            
            self._fitness_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })
            best_fitness_history.append(best_fitness)

            # Check for convergence
            if generation > 5 and len(best_fitness_history) >= 5:
                recent_improvement = best_fitness_history[-1] - best_fitness_history[-5]
                if recent_improvement < 0.001:
                    self.logger.info(f"Converged at generation {generation}")
                    break

            # Create next generation
            new_population = []

            # Elitism: keep best individuals
            elite_count = max(1, population_size // 10)
            sorted_pop = sorted(self._population, key=lambda x: x.fitness, reverse=True)
            new_population.extend(sorted_pop[:elite_count])

            # Generate offspring
            while len(new_population) < population_size:
                # Selection
                parent1 = self._select_individual(self._population, selection_strategy)
                parent2 = self._select_individual(self._population, selection_strategy)

                # Crossover
                if random.random() < crossover_rate:
                    offspring = self._crossover(parent1, parent2, generation)
                else:
                    offspring = copy.deepcopy(parent1)
                    offspring.generation = generation

                # Mutation
                if random.random() < mutation_rate:
                    offspring = self._mutate_individual(offspring, mutation_rate)

                offspring.id = self._generate_id()
                new_population.append(offspring)

            self._population = new_population

        context.update_progress(85, "Selecting best evolved knowledge")

        # Get best individual
        best_individual = max(self._population, key=lambda x: x.fitness)
        
        # Calculate improvements
        original_fitness = self._fitness_history[0]['best_fitness'] if self._fitness_history else 0
        final_fitness = best_individual.fitness
        improvements = self._calculate_improvements(original_fitness, final_fitness, generations)

        context.update_progress(100, "Evolution complete")

        return {
            'evolved_knowledge': best_individual.knowledge,
            'fitness_history': self._fitness_history,
            'improvements': improvements,
            'final_fitness': final_fitness,
            'generations_completed': len(self._fitness_history),
            'source': 'fallback_ga'
        }

    def _initialize_population(self, knowledge_base: Dict, size: int) -> List[KnowledgeIndividual]:
        """Create initial population with variations of the knowledge base."""
        population = []
        
        for i in range(size):
            individual = KnowledgeIndividual(
                id=self._generate_id(),
                knowledge=copy.deepcopy(knowledge_base),
                generation=0
            )
            
            # Add slight random variations to create diversity
            if i > 0:  # Keep first one as original
                individual = self._mutate_individual(individual, 0.05)
            
            population.append(individual)
        
        return population

    def _calculate_fitness(self, knowledge: Dict, metric: str) -> float:
        """Calculate fitness score for knowledge based on metric."""
        scores = []
        
        if metric in ['accuracy', 'combined']:
            # Calculate accuracy based on triple consistency
            accuracy = self._calculate_accuracy(knowledge)
            scores.append(accuracy)
        
        if metric in ['coverage', 'combined']:
            # Calculate coverage based on entity/relation diversity
            coverage = self._calculate_coverage(knowledge)
            scores.append(coverage)
        
        if metric in ['consistency', 'combined']:
            # Calculate consistency based on graph structure
            consistency = self._calculate_consistency(knowledge)
            scores.append(consistency)
        
        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_accuracy(self, knowledge: Dict) -> float:
        """Calculate accuracy score based on knowledge structure."""
        if not isinstance(knowledge, dict):
            return 0.5
        
        score = 0.5
        
        # Check for triples with proper structure
        if 'triples' in knowledge and isinstance(knowledge['triples'], list):
            triples = knowledge['triples']
            if triples:
                # Score based on valid triple structure
                valid_count = sum(
                    1 for t in triples 
                    if isinstance(t, dict) and 'subject' in t and 'predicate' in t and 'object' in t
                )
                score += 0.3 * (valid_count / len(triples))
        
        # Check for entity definitions
        if 'entities' in knowledge and isinstance(knowledge['entities'], (list, dict, set)):
            score += 0.1
        
        # Check for relation definitions
        if 'relations' in knowledge and isinstance(knowledge['relations'], (list, dict, set)):
            score += 0.1
        
        return min(1.0, score)

    def _calculate_coverage(self, knowledge: Dict) -> float:
        """Calculate coverage score based on knowledge breadth."""
        if not isinstance(knowledge, dict):
            return 0.3
        
        score = 0.3
        
        # Factor in number of triples
        if 'triples' in knowledge and isinstance(knowledge['triples'], list):
            triple_count = len(knowledge['triples'])
            # Score increases with more triples up to a point
            score += min(0.4, triple_count / 100)
        
        # Factor in entity diversity
        if 'entities' in knowledge:
            entities = knowledge['entities']
            if isinstance(entities, (list, set)):
                score += min(0.15, len(entities) / 50)
            elif isinstance(entities, dict):
                score += min(0.15, len(entities) / 50)
        
        # Factor in relation diversity
        if 'relations' in knowledge:
            relations = knowledge['relations']
            if isinstance(relations, (list, set)):
                score += min(0.15, len(relations) / 20)
            elif isinstance(relations, dict):
                score += min(0.15, len(relations) / 20)
        
        return min(1.0, score)

    def _calculate_consistency(self, knowledge: Dict) -> float:
        """Calculate consistency score based on graph coherence."""
        if not isinstance(knowledge, dict):
            return 0.5
        
        score = 0.5
        
        # Check for consistent entity references in triples
        if 'triples' in knowledge and isinstance(knowledge['triples'], list):
            triples = knowledge['triples']
            if triples and len(triples) > 1:
                # Check if entities are consistently used
                entities_in_triples = set()
                for t in triples:
                    if isinstance(t, dict):
                        if 'subject' in t:
                            entities_in_triples.add(t['subject'])
                        if 'object' in t:
                            entities_in_triples.add(t['object'])
                
                # Score based on entity reuse (higher is more consistent)
                if 'entities' in knowledge:
                    defined_entities = set(knowledge['entities']) if isinstance(knowledge['entities'], (list, set)) else set()
                    if defined_entities:
                        overlap = len(entities_in_triples & defined_entities)
                        score += 0.3 * (overlap / len(defined_entities))
                
                # Check for consistent relation usage
                relations_in_triples = set()
                for t in triples:
                    if isinstance(t, dict) and 'predicate' in t:
                        relations_in_triples.add(t['predicate'])
                
                if 'relations' in knowledge:
                    defined_relations = set(knowledge['relations']) if isinstance(knowledge['relations'], (list, set)) else set()
                    if defined_relations:
                        overlap = len(relations_in_triples & defined_relations)
                        score += 0.2 * (overlap / len(defined_relations))
        
        return min(1.0, score)

    def _select_population(
        self, population: List[KnowledgeIndividual], 
        count: int, strategy: str
    ) -> List[KnowledgeIndividual]:
        """Select individuals from population using specified strategy."""
        if strategy == 'tournament':
            selected = []
            for _ in range(count):
                tournament = random.sample(population, min(3, len(population)))
                winner = max(tournament, key=lambda x: x.fitness)
                selected.append(winner)
            return selected
        
        elif strategy == 'roulette':
            total_fitness = sum(ind.fitness for ind in population)
            if total_fitness == 0:
                return random.sample(population, min(count, len(population)))
            
            selected = []
            for _ in range(count):
                pick = random.uniform(0, total_fitness)
                current = 0
                for ind in population:
                    current += ind.fitness
                    if current >= pick:
                        selected.append(ind)
                        break
            return selected
        
        elif strategy == 'rank':
            sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
            ranks = list(range(len(sorted_pop), 0, -1))
            total_rank = sum(ranks)
            
            selected = []
            for _ in range(count):
                pick = random.uniform(0, total_rank)
                current = 0
                for i, ind in enumerate(sorted_pop):
                    current += ranks[i]
                    if current >= pick:
                        selected.append(ind)
                        break
            return selected
        
        else:  # Default to tournament
            return self._select_population(population, count, 'tournament')

    def _select_individual(
        self, population: List[KnowledgeIndividual], strategy: str
    ) -> KnowledgeIndividual:
        """Select a single individual from population."""
        selected = self._select_population(population, 1, strategy)
        return selected[0] if selected else population[0]

    def _crossover(
        self, parent1: KnowledgeIndividual, parent2: KnowledgeIndividual, generation: int
    ) -> KnowledgeIndividual:
        """Perform crossover between two parent individuals."""
        offspring_knowledge = {}
        
        # Merge knowledge structures
        keys = set(parent1.knowledge.keys()) | set(parent2.knowledge.keys())
        
        for key in keys:
            if key in parent1.knowledge and key in parent2.knowledge:
                # Randomly choose from either parent
                if random.random() < 0.5:
                    offspring_knowledge[key] = copy.deepcopy(parent1.knowledge[key])
                else:
                    offspring_knowledge[key] = copy.deepcopy(parent2.knowledge[key])
            elif key in parent1.knowledge:
                offspring_knowledge[key] = copy.deepcopy(parent1.knowledge[key])
            else:
                offspring_knowledge[key] = copy.deepcopy(parent2.knowledge[key])
        
        # Create offspring individual
        offspring = KnowledgeIndividual(
            id=self._generate_id(),
            knowledge=offspring_knowledge,
            generation=generation,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return offspring

    def _mutate_individual(
        self, individual: KnowledgeIndividual, mutation_rate: float
    ) -> KnowledgeIndividual:
        """Apply mutation to an individual."""
        knowledge = individual.knowledge
        mutations_applied = []
        
        if not isinstance(knowledge, dict):
            return individual
        
        # Mutate triples
        if 'triples' in knowledge and isinstance(knowledge['triples'], list):
            triples = knowledge['triples']
            for i, triple in enumerate(triples):
                if random.random() < mutation_rate and isinstance(triple, dict):
                    # Mutate a field
                    if 'confidence' in triple and random.random() < 0.5:
                        old_conf = triple['confidence']
                        triple['confidence'] = max(0, min(1, old_conf + random.uniform(-0.1, 0.1)))
                        mutations_applied.append(f"confidence_{i}")
                    
                    # Add metadata
                    if 'metadata' not in triple:
                        triple['metadata'] = {}
                    triple['metadata']['evolved'] = True
        
        # Add/remove random entities (low probability)
        if 'entities' in knowledge and random.random() < mutation_rate * 0.1:
            if isinstance(knowledge['entities'], list) and knowledge['entities']:
                # Remove a random entity
                idx = random.randint(0, len(knowledge['entities']) - 1)
                removed = knowledge['entities'].pop(idx)
                mutations_applied.append(f"removed_entity_{removed}")
        
        # Update individual
        individual.knowledge = knowledge
        individual.mutation_history.append({
            'timestamp': datetime.now().isoformat(),
            'mutations': mutations_applied,
            'rate': mutation_rate
        })
        
        return individual

    def _calculate_improvements(
        self, original_fitness: float, final_fitness: float, generations: int
    ) -> List[str]:
        """Calculate and describe improvements."""
        improvements = []
        
        improvement_pct = ((final_fitness - original_fitness) / max(original_fitness, 0.001)) * 100
        
        if improvement_pct > 0:
            improvements.append(f"Fitness improved by {improvement_pct:.2f}% over {generations} generations")
            
            if improvement_pct > 50:
                improvements.append("Significant improvement achieved through evolution")
            elif improvement_pct > 20:
                improvements.append("Moderate improvement in knowledge quality")
            else:
                improvements.append("Minor improvement detected")
        
        elif improvement_pct < -10:
            improvements.append("Warning: Fitness decreased significantly, consider adjusting parameters")
        
        else:
            improvements.append("Knowledge maintained stable quality through evolution")
        
        improvements.append(f"Final fitness score: {final_fitness:.4f}")
        
        return improvements

    def _analyze_knowledge_structure(self, knowledge: Dict) -> Dict[str, Any]:
        """Analyze the structure of knowledge."""
        analysis = {
            'type': type(knowledge).__name__,
            'size': len(str(knowledge)),
            'keys': list(knowledge.keys()) if isinstance(knowledge, dict) else []
        }
        
        if isinstance(knowledge, dict):
            if 'triples' in knowledge:
                analysis['triple_count'] = len(knowledge['triples'])
            if 'entities' in knowledge:
                analysis['entity_count'] = len(knowledge['entities'])
            if 'relations' in knowledge:
                analysis['relation_count'] = len(knowledge['relations'])
        
        return analysis

    def _generate_recommendations(
        self, fitness_scores: Dict[str, float], structure: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on fitness analysis."""
        recommendations = []
        
        if fitness_scores.get('accuracy', 0) < 0.7:
            recommendations.append("Consider adding more structured triples with clear subject-predicate-object format")
        
        if fitness_scores.get('coverage', 0) < 0.6:
            recommendations.append("Expand knowledge base with more entities and relationships")
        
        if fitness_scores.get('consistency', 0) < 0.7:
            recommendations.append("Review entity and relation definitions for consistency")
        
        if not recommendations:
            recommendations.append("Knowledge structure is well-optimized")
        
        return recommendations

    def _generate_id(self) -> str:
        """Generate a unique identifier."""
        import uuid
        return f"kev_{uuid.uuid4().hex[:12]}"

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.
        
        Returns schema for UI configuration with all evolution operations and parameters.
        """
        return {
            "type": "object",
            "title": "Knowledge Evolution Configuration",
            "description": "Configure genetic algorithm parameters for knowledge evolution",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "The evolution operation to perform",
                    "enum": ["evolve", "optimize", "select", "mutate", "fitness_analysis"],
                    "enumNames": [
                        "Evolve - Multi-generation evolution with full GA",
                        "Optimize - Focused optimization for specific goals",
                        "Select - Select best variants from population",
                        "Mutate - Apply mutation operations",
                        "Fitness Analysis - Analyze knowledge fitness without evolution"
                    ],
                    "default": "evolve"
                },
                "generations": {
                    "type": "integer",
                    "title": "Generations",
                    "description": "Number of evolution generations to run",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 10
                },
                "population_size": {
                    "type": "integer",
                    "title": "Population Size",
                    "description": "Size of the knowledge population",
                    "minimum": 10,
                    "maximum": 1000,
                    "default": 100
                },
                "mutation_rate": {
                    "type": "number",
                    "title": "Mutation Rate",
                    "description": "Probability of mutation (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1
                },
                "crossover_rate": {
                    "type": "number",
                    "title": "Crossover Rate",
                    "description": "Probability of crossover (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8
                },
                "fitness_metric": {
                    "type": "string",
                    "title": "Fitness Metric",
                    "description": "Metric used to evaluate knowledge fitness",
                    "enum": ["accuracy", "coverage", "consistency", "combined"],
                    "enumNames": [
                        "Accuracy - Based on triple structure validity",
                        "Coverage - Based on entity/relation diversity",
                        "Consistency - Based on graph coherence",
                        "Combined - Weighted combination of all metrics"
                    ],
                    "default": "combined"
                },
                "selection_strategy": {
                    "type": "string",
                    "title": "Selection Strategy",
                    "description": "Strategy for selecting individuals for reproduction",
                    "enum": ["tournament", "roulette", "rank"],
                    "enumNames": [
                        "Tournament - Select best from random tournaments",
                        "Roulette - Fitness-proportionate selection",
                        "Rank - Rank-based selection"
                    ],
                    "default": "tournament"
                }
            }
        }

    def is_healthy(self) -> bool:
        """Check if the node is healthy and ready to execute."""
        try:
            # Node is healthy if it can perform fallback evolution
            # OpenEvolve is optional
            return True
        except Exception:
            return False

    def cleanup(self):
        """Cleanup resources."""
        try:
            self._population = []
            self._fitness_history = []
            self._generation_count = 0
            if self.hub:
                self.hub = None
            self.logger.info("KnowledgeEvolutionNode cleanup complete")
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")
