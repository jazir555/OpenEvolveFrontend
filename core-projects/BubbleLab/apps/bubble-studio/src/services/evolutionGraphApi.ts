import { api } from '@/lib/api';

export interface EvolutionRunRecord {
  id: number;
  evolutionId: string;
  status: string;
  name?: string;
  config?: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
}

export interface EvolutionNodeRecord {
  id: number;
  runId: number;
  nodeId: string;
  parentNodeId?: string | null;
  generation: number;
  status: string;
  fitness?: number | null;
  score?: number | null;
  label?: string | null;
  htmlAssetId?: number | null;
  thumbnailAssetId?: number | null;
  metadata?: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
}

export interface EvolutionAssetResponse {
  id: number;
  url: string;
  contentType: string;
  size: number;
}

export interface EvolutionRunUsage {
  totalBytes: number;
  totalAssets: number;
  htmlBytes: number;
  htmlCount: number;
  thumbnailBytes: number;
  thumbnailCount: number;
}

export interface EvolutionThumbnailCleanupResponse {
  message: string;
  removedCount: number;
  freedBytes: number;
}

export interface CreateEvolutionRunRequest {
  evolutionId: string;
  name?: string;
  status?: string;
  config?: Record<string, unknown> | null;
}

export interface UpsertEvolutionNodeRequest {
  runId: number;
  nodeId: string;
  parentNodeId?: string | null;
  generation: number;
  status: string;
  fitness?: number | null;
  score?: number | null;
  label?: string | null;
  htmlAssetId?: number | null;
  thumbnailAssetId?: number | null;
  metadata?: Record<string, unknown> | null;
}

export interface CreateEvolutionAssetRequest {
  runId: number;
  kind: 'html' | 'thumbnail';
  contentType: string;
  dataBase64: string;
  filename?: string;
}

export const evolutionGraphApi = {
  createRun: async (
    payload: CreateEvolutionRunRequest
  ): Promise<EvolutionRunRecord> => {
    return api.post<EvolutionRunRecord>('/evolution-graph/runs', payload);
  },
  listRuns: async (): Promise<EvolutionRunRecord[]> => {
    return api.get<EvolutionRunRecord[]>('/evolution-graph/runs');
  },
  listNodes: async (runId: number): Promise<EvolutionNodeRecord[]> => {
    return api.get<EvolutionNodeRecord[]>(
      `/evolution-graph/runs/${runId}/nodes`
    );
  },
  getRunUsage: async (runId: number): Promise<EvolutionRunUsage> => {
    return api.get<EvolutionRunUsage>(
      `/evolution-graph/runs/${runId}/usage`
    );
  },
  clearRunNodes: async (runId: number): Promise<{ message: string }> => {
    return api.delete<{ message: string }>(
      `/evolution-graph/runs/${runId}/nodes`
    );
  },
  deleteRun: async (runId: number): Promise<{ message: string }> => {
    return api.delete<{ message: string }>(`/evolution-graph/runs/${runId}`);
  },
  clearRunThumbnails: async (
    runId: number
  ): Promise<EvolutionThumbnailCleanupResponse> => {
    return api.delete<EvolutionThumbnailCleanupResponse>(
      `/evolution-graph/runs/${runId}/thumbnails`
    );
  },
  upsertNode: async (
    payload: UpsertEvolutionNodeRequest
  ): Promise<EvolutionNodeRecord> => {
    return api.post<EvolutionNodeRecord>('/evolution-graph/nodes', payload);
  },
  createAsset: async (
    payload: CreateEvolutionAssetRequest
  ): Promise<EvolutionAssetResponse> => {
    return api.post<EvolutionAssetResponse>('/evolution-graph/assets', payload);
  },
};
