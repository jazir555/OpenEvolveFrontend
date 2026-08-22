/**
 * LeanAide API Hooks
 * React Query hooks wrapping `leanaideApi`.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  leanaideApi,
  LeanAideAnyPayload,
  LeanAideSimSearchPayload,
  LeanAideBenchmarkStartResponse,
  LeanAideBenchmarkResult,
} from '@/services/leanaideApi';

// ==================== Query Keys ====================

export const leanaideKeys = {
  all: ['leanaide'] as const,
  generate: () => [...leanaideKeys.all, 'generate'] as const,
  verify: () => [...leanaideKeys.all, 'verify'] as const,
  raw: () => [...leanaideKeys.all, 'raw'] as const,
  simSearch: () => [...leanaideKeys.all, 'sim-search'] as const,
  benchmark: (id?: string) =>
    [...leanaideKeys.all, 'benchmark', id ?? ''] as const,
};

// ==================== Mutations ====================

export function useLeanAideGenerate() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: LeanAideAnyPayload) => leanaideApi.generate(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: leanaideKeys.generate() });
    },
  });
}

export function useLeanAideVerify() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: LeanAideAnyPayload) => leanaideApi.verify(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: leanaideKeys.verify() });
    },
  });
}

export function useLeanAideRawResponse() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: LeanAideAnyPayload) => leanaideApi.rawResponse(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: leanaideKeys.raw() });
    },
  });
}

export function useLeanAideSimSearch() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: LeanAideSimSearchPayload) =>
      leanaideApi.simSearch(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: leanaideKeys.simSearch() });
    },
  });
}

// ==================== Benchmark ====================

/**
 * Start a benchmark run (returns a `benchmark_id` to poll with).
 */
export function useLeanAideBenchmarkStart() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (): Promise<LeanAideBenchmarkStartResponse> =>
      leanaideApi.runBenchmark(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: leanaideKeys.benchmark() });
    },
  });
}

/**
 * Poll benchmark results for a given id. Disabled until an id is provided.
 */
export function useLeanAideBenchmark(benchmarkId?: string | null) {
  return useQuery({
    queryKey: leanaideKeys.benchmark(benchmarkId ?? undefined),
    queryFn: () => leanaideApi.getBenchmark(benchmarkId as string),
    enabled: !!benchmarkId,
    refetchInterval: (query) => {
      const data = query.state.data as LeanAideBenchmarkResult | undefined;
      // Stop polling once the backend reports a terminal status.
      if (data && data.status !== 'pending' && data.status !== 'running') {
        return false;
      }
      return 2000;
    },
  });
}
