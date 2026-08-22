/**
 * GKET API Hooks
 * TanStack Query hooks wrapping `gketApi`.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  gketApi,
  GketExportFormat,
  GketExportResponse,
  GketExtractPayload,
  GketExtractResponse,
  GketGenerateModelsPayload,
  GketHealthResponse,
  GketParsePayload,
} from '@/services/gketApi';

// ==================== Query Keys ====================

export const gketKeys = {
  all: ['gket'] as const,
  health: () => [...gketKeys.all, 'health'] as const,
  parse: () => [...gketKeys.all, 'parse'] as const,
  models: () => [...gketKeys.all, 'generate-models'] as const,
  extractions: () => [...gketKeys.all, 'extract'] as const,
  export: (id?: string, format?: GketExportFormat) =>
    [...gketKeys.all, 'export', id ?? '', format ?? 'json'] as const,
};

// ==================== Queries ====================

/**
 * Poll the GKET server liveness endpoint.
 */
export function useGketHealth() {
  return useQuery<GketHealthResponse>({
    queryKey: gketKeys.health(),
    queryFn: () => gketApi.healthz(),
    refetchInterval: 30000,
    retry: false,
  });
}

/**
 * Read back a stored extraction result. Disabled until an id is provided.
 */
export function useGketExport(id?: string | null, format: GketExportFormat = 'json') {
  return useQuery<GketExportResponse>({
    queryKey: gketKeys.export(id ?? undefined, format),
    queryFn: () => gketApi.getExport(id as string, format),
    enabled: !!id,
  });
}

// ==================== Mutations ====================

export function useGketParseDoc() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: GketParsePayload) => gketApi.parseDoc(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: gketKeys.parse() });
    },
  });
}

export function useGketGenerateModels() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: GketGenerateModelsPayload) =>
      gketApi.generateModels(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: gketKeys.models() });
    },
  });
}

export function useGketExtract() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: GketExtractPayload): Promise<GketExtractResponse> =>
      gketApi.extract(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: gketKeys.extractions() });
    },
  });
}

/**
 * Imperatively fetch an export (used by the download buttons, which need the
 * payload once rather than as cached state).
 */
export function useGketFetchExport() {
  return useMutation({
    mutationFn: ({ id, format }: { id: string; format: GketExportFormat }) =>
      gketApi.getExport(id, format),
  });
}
