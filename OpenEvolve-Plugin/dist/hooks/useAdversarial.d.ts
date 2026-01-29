import { AdversarialTest } from '../stores/evolutionStore';
/**
 * Adversarial testing parameters
 */
export interface AdversarialParams {
    content: string;
    attack_modes: string[];
    parameters: {
        num_rounds: number;
        red_team_models: Array<{
            provider: string;
            model: string;
        }>;
        blue_team_models: Array<{
            provider: string;
            model: string;
        }>;
    };
}
/**
 * Adversarial state
 */
export interface AdversarialState {
    data: AdversarialTest | null;
    loading: boolean;
    error: Error | null;
    progress: number;
    currentRound: number;
    totalRounds: number;
}
/**
 * Custom hook for adversarial testing
 * Manages red team vs blue team security testing workflows
 */
export declare function useAdversarial(testId?: string): {
    execute: (params: AdversarialParams) => Promise<void>;
    getStatus: () => Promise<AdversarialTest | null>;
    getResults: () => AdversarialTest | null;
    cancel: () => Promise<void>;
    approvePatch: (round: number, approved: boolean, feedback?: string) => Promise<void>;
    reset: () => void;
    data: AdversarialTest | null;
    loading: boolean;
    error: Error | null;
    progress: number;
    currentRound: number;
    totalRounds: number;
};
/**
 * Adversarial tests list hook
 */
export declare function useAdversarialTests(params?: {
    status?: string;
    limit?: number;
    offset?: number;
}): {
    refetch: () => Promise<void>;
    data: AdversarialTest[] | null;
    loading: boolean;
    error: Error | null;
};
