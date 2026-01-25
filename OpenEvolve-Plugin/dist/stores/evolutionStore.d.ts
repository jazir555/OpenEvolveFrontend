import { WorkflowExecution, WorkflowStatus } from './workflowStore';
/**
 * Adversarial testing round result
 */
export interface AdversarialRound {
    round: number;
    attack_mode: string;
    success: boolean;
    vulnerability?: string;
    payload: string;
    patch?: string;
    patch_approved?: boolean;
}
/**
 * Adversarial testing state
 */
export interface AdversarialTest {
    test_id: string;
    status: WorkflowStatus;
    current_round: number;
    total_rounds: number;
    red_team_results: AdversarialRound[];
    blue_team_results: AdversarialRound[];
    vulnerabilities_found: number;
    patches_generated: number;
    patches_approved: number;
    created_at: string;
    updated_at: string;
    websocket_url?: string;
}
/**
 * Evolution store state
 */
interface EvolutionState {
    evolution: WorkflowExecution | null;
    evolutions: WorkflowExecution[];
    adversarialTest: AdversarialTest | null;
    adversarialTests: AdversarialTest[];
    content: string;
    attackModes: string[];
    isLoading: boolean;
    isEvolutionRunning: boolean;
    isAdversarialRunning: boolean;
    error: string | null;
    setEvolution: (evolution: WorkflowExecution | null) => void;
    setEvolutions: (evolutions: WorkflowExecution[]) => void;
    addEvolution: (evolution: WorkflowExecution) => void;
    updateEvolution: (id: string, updates: Partial<WorkflowExecution>) => void;
    setAdversarialTest: (test: AdversarialTest | null) => void;
    setAdversarialTests: (tests: AdversarialTest[]) => void;
    addAdversarialTest: (test: AdversarialTest) => void;
    updateAdversarialTest: (id: string, updates: Partial<AdversarialTest>) => void;
    approvePatch: (testId: string, round: number, approved: boolean, feedback?: string) => void;
    setContent: (content: string) => void;
    setAttackModes: (modes: string[]) => void;
    setIsEvolutionRunning: (running: boolean) => void;
    setIsAdversarialRunning: (running: boolean) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    clearEvolution: () => void;
    clearAdversarial: () => void;
    reset: () => void;
}
/**
 * Evolution store
 */
export declare const useEvolutionStore: import('zustand').UseBoundStore<Omit<import('zustand').StoreApi<EvolutionState>, "setState"> & {
    setState<A extends string | {
        type: string;
    }>(partial: EvolutionState | Partial<EvolutionState> | ((state: EvolutionState) => EvolutionState | Partial<EvolutionState>), replace?: boolean, action?: A): void;
}>;
export {};
