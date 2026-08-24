import { useQuery } from '@tanstack/react-query';

export interface NvidiaNimModel {
  id: string;
  name: string;
  owned_by: string;
}

async function fetchNvidiaNimModels(): Promise<NvidiaNimModel[]> {
  const res = await fetch('/api/nvidia-nim/models');
  if (!res.ok) {
    throw new Error(`Failed to load NVIDIA NIM models: ${res.status}`);
  }
  const data = (await res.json()) as { models?: NvidiaNimModel[] };
  return data.models ?? [];
}

/**
 * Live list of NVIDIA NIM catalog models for the model selector.
 * Auto-populates from the API — no static per-model file or manual
 * regeneration required. Cached for 5 minutes.
 */
export function useNvidiaNimModels() {
  return useQuery({
    queryKey: ['nvidia-nim', 'models'],
    queryFn: fetchNvidiaNimModels,
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
    retry: 1,
  });
}
