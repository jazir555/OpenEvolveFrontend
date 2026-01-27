import React from 'react';
import { OpenEvolveClient } from '../api/client';
import { IntegrationName } from '../api/client';
import { DecompositionInputs, DecompositionResult, LeanAideInputs, LeanAideResult, EvolutionInputs, EvolutionResult, KnowledgeInputs, KnowledgeResult, MakerInputs, MakerResult, HephaestusInputs, HephaestusResult, SolutionInputs, SolutionResult, VerificationInputs, VerificationResult, AssemblyInputs, AssemblyResult } from '../integrations';
import { IntegrationError } from '../api/errors';
export declare const OpenEvolveProvider: React.FC<{
    client: OpenEvolveClient;
    children: React.ReactNode;
}>;
export declare const useOpenEvolveClient: () => OpenEvolveClient;
export declare function useIntegration<TInputs, TResult>(integrationName: IntegrationName | string): {
    data: TResult | null;
    loading: boolean;
    error: IntegrationError | null;
    execute: (inputs: TInputs) => Promise<TResult | null>;
    reset: () => void;
};
export declare function useDecomposition(): {
    data: DecompositionResult | null;
    loading: boolean;
    error: IntegrationError | null;
    execute: (inputs: DecompositionInputs) => Promise<DecompositionResult | null>;
    reset: () => void;
};
export declare function useLeanAide(): {
    data: LeanAideResult | null;
    loading: boolean;
    error: IntegrationError | null;
    execute: (inputs: LeanAideInputs) => Promise<LeanAideResult | null>;
    reset: () => void;
};
export declare function useEvolution(): {
    data: EvolutionResult | null;
    loading: boolean;
    error: IntegrationError | null;
    execute: (inputs: EvolutionInputs) => Promise<EvolutionResult | null>;
    reset: () => void;
};
export declare function useKnowledgeEngine(): {
    data: KnowledgeResult | null;
    loading: boolean;
    error: IntegrationError | null;
    execute: (inputs: KnowledgeInputs) => Promise<KnowledgeResult | null>;
    reset: () => void;
};
export declare function useMaker(): {
    data: MakerResult | null;
    loading: boolean;
    error: IntegrationError | null;
    execute: (inputs: MakerInputs) => Promise<MakerResult | null>;
    reset: () => void;
};
export declare function useHephaestus(): {
    data: HephaestusResult | null;
    loading: boolean;
    error: IntegrationError | null;
    execute: (inputs: HephaestusInputs) => Promise<HephaestusResult | null>;
    reset: () => void;
};
export declare function useSolution(): {
    data: SolutionResult | null;
    loading: boolean;
    error: IntegrationError | null;
    execute: (inputs: SolutionInputs) => Promise<SolutionResult | null>;
    reset: () => void;
};
export declare function useVerification(): {
    data: VerificationResult | null;
    loading: boolean;
    error: IntegrationError | null;
    execute: (inputs: VerificationInputs) => Promise<VerificationResult | null>;
    reset: () => void;
};
export declare function useAssembly(): {
    data: AssemblyResult | null;
    loading: boolean;
    error: IntegrationError | null;
    execute: (inputs: AssemblyInputs) => Promise<AssemblyResult | null>;
    reset: () => void;
};
//# sourceMappingURL=index.d.ts.map