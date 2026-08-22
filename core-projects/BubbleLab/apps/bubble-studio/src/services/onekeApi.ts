/**
 * OneKE API Client
 *
 * Connects the BubbleLab frontend to the OneKE FastAPI extraction service
 * (core-projects/OneKE/server.py). The backend wraps src/run.py and exposes a
 * small REST surface: health, schema/case discovery, synchronous extraction, and
 * result retrieval by id.
 *
 * Contract (server.py):
 *   GET  /healthz            -> { status: "ok" }
 *   GET  /schemas            -> string[]            (schema class names)
 *   GET  /cases              -> string[]            (case-kb task names)
 *   POST /extract            -> ExtractResult
 *   GET  /result/{id}        -> ExtractResult
 */

import { ApiClient, ApiClientConfig } from '@/lib/api';
import { ONEKE_API_BASE_URL } from '@/env';

const onekeClientConfig: ApiClientConfig = {
  baseURL: ONEKE_API_BASE_URL,
  timeout: 600000, // extraction can take several minutes
  enableRetry: false,
};

const onekeClient = new ApiClient(onekeClientConfig);

// ==================== Types ====================

export type OneKETask = 'NER' | 'RE' | 'EE' | 'Triple' | 'Base';
export type OneKEMode = 'quick' | 'agent' | 'customized';

export interface OneKEExtractPayload {
  task?: OneKETask;
  mode?: OneKEMode;
  config_yaml?: string;
  text?: string;
  file_ref?: string;
  instruction?: string;
  constraint?: string;
  model?: Record<string, unknown>;
  api_key?: string;
  base_url?: string;
  model_name?: string;
  construct?: Record<string, unknown>;
}

export interface OneKEResult {
  id: string;
  answer_json: unknown;
  schema: string;
  triples: unknown[];
  status: 'success' | 'error';
  error?: string | null;
}

// ==================== API Client ====================

export const onekeApi = {
  /**
   * Check API health.
   */
  healthz: async (): Promise<{ status: string }> => {
    return onekeClient.get<{ status: string }>('/healthz');
  },

  /**
   * List available schema names.
   */
  listSchemas: async (): Promise<string[]> => {
    return onekeClient.get<string[]>('/schemas');
  },

  /**
   * List available case / example knowledge bases.
   */
  listCases: async (): Promise<string[]> => {
    return onekeClient.get<string[]>('/cases');
  },

  /**
   * Run knowledge extraction.
   */
  extract: async (payload: OneKEExtractPayload): Promise<OneKEResult> => {
    return onekeClient.post<OneKEResult>('/extract', payload);
  },

  /**
   * Fetch a previously stored extraction result by id.
   */
  getResult: async (id: string): Promise<OneKEResult> => {
    return onekeClient.get<OneKEResult>(`/result/${encodeURIComponent(id)}`);
  },
};
