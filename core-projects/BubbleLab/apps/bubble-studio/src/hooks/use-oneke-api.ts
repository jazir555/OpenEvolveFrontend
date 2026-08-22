/**
 * OneKE API Hooks
 * React hooks (TanStack Query) for driving knowledge extraction via onekeApi.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { onekeApi, OneKEExtractPayload, OneKEResult } from '../services/onekeApi';

// ============================================================================
// Query Keys
// ============================================================================

export const onekeKeys = {
  all: ['oneke'] as const,
  health: () => [...onekeKeys.all, 'health'] as const,
  schemas: () => [...onekeKeys.all, 'schemas'] as const,
  cases: () => [...onekeKeys.all, 'cases'] as const,
  result: (id: string) => [...onekeKeys.all, 'result', id] as const,
};

// ============================================================================
// Health
// ============================================================================

export function useOneKEHealth() {
  return useQuery({
    queryKey: onekeKeys.health(),
    queryFn: () => onekeApi.healthz(),
    retry: false,
    refetchInterval: false,
  });
}

// ============================================================================
// Schemas
// ============================================================================

export function useOneKESchemas() {
  return useQuery({
    queryKey: onekeKeys.schemas(),
    queryFn: () => onekeApi.listSchemas(),
  });
}

// ============================================================================
// Cases
// ============================================================================

export function useOneKECases() {
  return useQuery({
    queryKey: onekeKeys.cases(),
    queryFn: () => onekeApi.listCases(),
  });
}

// ============================================================================
// Extract (mutation)
// ============================================================================

export function useOneKEExtract() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (payload: OneKEExtractPayload): Promise<OneKEResult> => {
      const result = await onekeApi.extract(payload);
      if (result.status === 'success') {
        queryClient.setQueryData(onekeKeys.result(result.id), result);
      }
      return result;
    },
  });
}

// ============================================================================
// Result by id
// ============================================================================

export function useOneKEResult(id: string | undefined) {
  return useQuery({
    queryKey: id ? onekeKeys.result(id) : onekeKeys.all,
    queryFn: () => onekeApi.getResult(id as string),
    enabled: !!id,
  });
}
