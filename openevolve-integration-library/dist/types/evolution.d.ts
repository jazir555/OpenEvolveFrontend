import { ExecutionConfig } from './common';
export interface EvolutionInputs {
    operation: 'evolution' | 'adversarial' | 'coevolution';
    config: EvolutionConfig | AdversarialConfig | CoevolutionConfig;
    execConfig?: ExecutionConfig;
}
export interface EvolutionConfig {
    problem: string;
    populationSize: number;
    generations: number;
    mutationRate: number;
    crossoverRate: number;
    selectionMethod: 'tournament' | 'roulette' | 'rank' | 'steady-state';
    elitism?: number;
    fitnessFunction: string;
    representation: 'binary' | 'real' | 'permutation' | 'tree';
}
export interface AdversarialConfig {
    problem: string;
    populationSize: number;
    generations: number;
    mutationRate: number;
    adversarialCount: number;
    attackStrategy: 'fgsm' | 'pgd' | 'cw' | 'custom';
}
export interface CoevolutionConfig {
    problem: string;
    populationSizes: number[];
    generations: number;
    mutationRates: number[];
    numPopulations: number;
    interactionPattern: 'competitive' | 'cooperative' | 'mixed';
    fitnessSharing?: boolean;
}
export interface EvolutionResult {
    executionId: string;
    bestSolution: any;
    bestFitness: number;
    fitnessHistory: number[];
    generationStats: GenerationStats[];
    metadata: EvolutionMetadata;
}
export interface AdversarialResult {
    executionId: string;
    bestSolution: any;
    bestFitness: number;
    adversarialExamples: AdversarialExample[];
}
export interface CoevolutionResult {
    executionId: string;
    bestSolutions: any[];
    fitnessHistory: number[][];
    populationStats: PopulationStats[];
    metadata: EvolutionMetadata;
}
export interface GenerationStats {
    generation: number;
    bestFitness: number;
    averageFitness: number;
    worstFitness: number;
    diversity?: number;
    convergence?: number;
}
export interface PopulationStats {
    populationId: number;
    generation: number;
    stats: GenerationStats;
}
export interface AdversarialExample {
    original: any;
    adversarial: any;
    perturbation: any;
    success: boolean;
}
export interface MetricsSnapshot {
    generation: number;
    metrics: Record<string, number>;
}
export interface EvolutionMetadata {
    executionTime: number;
    generationsCompleted: number;
    converged: boolean;
    convergenceGeneration?: number;
    timestamp: string;
    algorithm: string;
    configHash: string;
}
export interface EvolutionProgress {
    executionId: string;
    currentGeneration: number;
    totalGenerations: number;
    bestFitness: number;
    populationStats: GenerationStats;
    status: 'running' | 'paused' | 'completed' | 'failed';
    estimatedTimeRemaining?: number;
}
//# sourceMappingURL=evolution.d.ts.map