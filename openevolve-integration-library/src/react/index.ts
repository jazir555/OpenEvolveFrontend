import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import { OpenEvolveClient } from '../api/client';
import { IntegrationName } from '../api/client';
import {
  DecompositionInputs, 
  DecompositionResult,
  LeanAideInputs, 
  LeanAideResult,
  EvolutionInputs, 
  EvolutionResult,
  KnowledgeInputs,
  KnowledgeResult,
  MakerInputs,
  MakerResult,
  CrewAIInputs,
  CrewAIResult,
  SolutionInputs,
  SolutionResult,
  VerificationInputs,
  VerificationResult,
  AssemblyInputs,
  AssemblyResult
} from '../integrations';

import {
  IntegrationError,
  createIntegrationError,
} from '../api/errors';

// Context
const OpenEvolveContext = createContext<OpenEvolveClient | null>(null);

export const OpenEvolveProvider: React.FC<{
  client: OpenEvolveClient; 
  children: React.ReactNode 
}> = ({ client, children }) => {
  const value = React.useMemo(() => client, [client]);
  
  return React.createElement(
    OpenEvolveContext.Provider,
    { value },
    children
  );
};

export const useOpenEvolveClient = () => {
  const client = useContext(OpenEvolveContext);
  if (!client) {
    throw new Error('useOpenEvolveClient must be used within an OpenEvolveProvider');
  }
  return client;
};

/**
 * Generic Integration Hook
 * Handles execution, loading state, errors, and cancellation.
 */
export function useIntegration<TInputs, TResult>(
  integrationName: IntegrationName | string
) {
  const client = useOpenEvolveClient();
  const [data, setData] = useState<TResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<IntegrationError | null>(null);
  
  // Track the current execution to handle cancellation and race conditions
  const abortControllerRef = useRef<AbortController | null>(null);
  const executionCountRef = useRef(0);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const reset = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setData(null);
    setLoading(false);
    setError(null);
  }, []);

  const execute = useCallback(async (inputs: TInputs) => {
    // Abort previous execution if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const executionId = ++executionCountRef.current;
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const result = await client.execute<string, TInputs, TResult>(
        integrationName as string, 
        inputs,
        { signal: controller.signal }
      );

      // Only update state if this is still the current execution
      if (executionId === executionCountRef.current) {
        setData(result);
        setLoading(false);
        return result;
      }
      return null;
    } catch (err: any) {
      // Don't set error if the request was intentionally aborted
      if (err.name === 'AbortError' || err.code === 'CANCELLATION_ERROR' || (err instanceof IntegrationError && err.code === 'CANCELLATION_ERROR')) {
        if (executionId === executionCountRef.current) {
          setLoading(false);
        }
        return null;
      }

      const integrationError = createIntegrationError(integrationName as string, err);

      if (executionId === executionCountRef.current) {
        setError(integrationError);
        setLoading(false);
        throw integrationError;
      }
      return null;
    }
  }, [client, integrationName]);

  return { data, loading, error, execute, reset };
}


// Specific Hooks
export function useDecomposition() {
  return useIntegration<DecompositionInputs, DecompositionResult>(IntegrationName.DECOMPOSITION);
}

export function useLeanAide() {
  return useIntegration<LeanAideInputs, LeanAideResult>(IntegrationName.LEANAIDE);
}

export function useEvolution() {
  return useIntegration<EvolutionInputs, EvolutionResult>(IntegrationName.EVOLUTION);
}

export function useKnowledgeEngine() {
  return useIntegration<KnowledgeInputs, KnowledgeResult>(IntegrationName.KNOWLEDGE);
}

export function useMaker() {
  return useIntegration<MakerInputs, MakerResult>(IntegrationName.MAKER);
}

export function useCrewAI() {
  return useIntegration<CrewAIInputs, CrewAIResult>(IntegrationName.CREWAI);
}

export function useSolution() {
  return useIntegration<SolutionInputs, SolutionResult>(IntegrationName.SOLUTION);
}

export function useVerification() {
  return useIntegration<VerificationInputs, VerificationResult>(IntegrationName.VERIFICATION);
}

export function useAssembly() {
  return useIntegration<AssemblyInputs, AssemblyResult>(IntegrationName.ASSEMBLY);
}
