import { default as React } from 'react';
export interface EvolutionConfig {
    populationSize: number;
    generations: number;
    elitismCount: number;
    mutationRate: number;
    crossoverRate: number;
    mutationStrength: number;
    selectionMethod: 'tournament' | 'roulette' | 'rank' | 'steady_state';
    tournamentSize: number;
    selectionPressure: number;
    convergenceThreshold: number;
    maxIterations: number;
    stagnationGenerations: number;
    diversityThreshold: number;
    nichingEnabled: boolean;
    crowdingDistance: number;
    adaptiveMutation: boolean;
    adaptiveCrossover: boolean;
    multiObjectiveOptimization: boolean;
    paretoFrontSize: number;
    penaltyFactor: number;
    constraintHandling: 'death' | 'penalty' | 'repair';
    parallelEvaluation: boolean;
    evaluationBatchSize: number;
    asyncEvaluation: boolean;
}
interface EvolutionConfigPanelProps {
    config: EvolutionConfig;
    onConfigChange: (config: EvolutionConfig) => void;
}
export declare const EvolutionConfigPanel: React.FC<EvolutionConfigPanelProps>;
export default EvolutionConfigPanel;
