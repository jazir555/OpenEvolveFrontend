/**
 * Backend-launch control panel client.
 *
 * Talks to the BubbleLab Hono API's `/api/backends` control plane
 * (apps/bubblelab-api/src/services/backends.ts), which spawns / stops / health
 * checks the product's backend servers (LeanAide, OneKE, GKET, OpenEvolve API,
 * OpenEvolve Engine).
 */

import { api } from '@/lib/api';

export interface BackendStatus {
  name: string;
  label: string;
  description: string;
  port: number;
  running: boolean;
  pid?: number;
  startedAt?: number;
  healthPath: string;
  error?: string;
}

export interface BackendsListResponse {
  backends: BackendStatus[];
}

export interface StartResponse {
  started: boolean;
  alreadyRunning?: boolean;
  name?: string;
  pid?: number;
  port?: number;
  cwd?: string;
  cmd?: string[];
  error?: string;
}

export interface StopResponse {
  stopped: boolean;
  wasRunning: boolean;
  name: string;
}

export const backendsApi = {
  /**
   * List all backends with their current status.
   */
  list: async (): Promise<BackendsListResponse> => {
    return api.get<BackendsListResponse>('/api/backends');
  },

  /**
   * Health-check a single backend by name.
   */
  status: async (name: string): Promise<BackendStatus> => {
    return api.get<BackendStatus>(`/api/backends/${encodeURIComponent(name)}/status`);
  },

  /**
   * Start a backend by name. Avoids spawning duplicates.
   */
  start: async (name: string): Promise<StartResponse> => {
    return api.post<StartResponse>(`/api/backends/${encodeURIComponent(name)}/start`);
  },

  /**
   * Stop a backend by name (kills the tracked PID).
   */
  stop: async (name: string): Promise<StopResponse> => {
    return api.post<StopResponse>(`/api/backends/${encodeURIComponent(name)}/stop`);
  },
};
