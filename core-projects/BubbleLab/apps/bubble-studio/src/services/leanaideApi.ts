/**
 * LeanAide API Client
 *
 * Connects BubbleLab frontend to the LeanAide Python backend
 * (`core-projects/LeanAide/server/api_server.py`), a `http.server` JSON API
 * listening on `http://localhost:7654` by default.
 *
 * Contract:
 * - `POST /`                  dispatches by a `task` field. Known tasks:
 *                              - `prove_for_formalization` (generation)
 *                              - `elaborate`                (verification)
 *                              - `lean_from_json_structured` (structured proof)
 * - `POST /run-sim-search`    embedding similarity search `{ num, query, descField }`
 *
 * Benchmark routes are NOT implemented by the Python server. They are proxied
 * through the BubbleLab API (`apps/bubblelab-api`, Hono) which mocks them:
 * - `POST /leanaide/benchmark/start`        -> `{ benchmark_id, status, message }`
 * - `GET  /leanaide/benchmark/:id/results` -> mock benchmark result
 * The benchmark calls therefore target `BUBBLELAB_API_BASE_URL`, while every
 * other call targets the LeanAide server directly via `LEANAIDE_API_URL`.
 */

import { ApiClient, ApiClientConfig } from '@/lib/api';
import { LEANAIDE_API_URL, BUBBLELAB_API_BASE_URL } from '@/env';

// ==================== Client Instances ====================

const leanaideClientConfig: ApiClientConfig = {
  baseURL: LEANAIDE_API_URL,
  timeout: 120000,
  enableRetry: true,
  maxRetries: 2,
  retryDelay: 2000,
};
const leanaideClient = new ApiClient(leanaideClientConfig);

const bubblelabClientConfig: ApiClientConfig = {
  baseURL: BUBBLELAB_API_BASE_URL,
  timeout: 60000,
  enableRetry: true,
  maxRetries: 2,
  retryDelay: 2000,
};
const bubblelabClient = new ApiClient(bubblelabClientConfig);

// ==================== Types ====================

export type LeanAideTask =
  | 'prove_for_formalization'
  | 'elaborate'
  | 'lean_from_json_structured';

export interface LeanAideGeneratePayload {
  task: 'prove_for_formalization';
  theorem: string;
  proof_attempt?: string;
  model?: string;
  temperature?: number;
  [key: string]: unknown;
}

export interface LeanAideVerifyPayload {
  task: 'elaborate';
  code: string;
  [key: string]: unknown;
}

export interface LeanAideStructuredPayload {
  task: 'lean_from_json_structured';
  [key: string]: unknown;
}

/**
 * Fully open payload shape used by the raw / generic `POST /` dispatcher.
 * The LeanAide server accepts any JSON with a `task` field, so callers may
 * attach arbitrary extra keys.
 */
export type LeanAideAnyPayload = Record<string, unknown> & {
  task?: string;
};

export interface LeanAideSimSearchPayload {
  num: number;
  query: string;
  descField: string;
}

export interface LeanAideBenchmarkStartResponse {
  benchmark_id: string;
  status: string;
  message: string;
}

export interface LeanAideBenchmarkResult {
  benchmark_id: string;
  status: string;
  results: unknown[];
  total: number;
  successful: number;
  failed: number;
  avg_time: number;
}

/** A generic raw LeanAide `POST /` response (shape varies by task). */
export type LeanAideRawResponse = Record<string, unknown>;

// ==================== API Client ====================

export const leanaideApi = {
  /**
   * Generation task (`prove_for_formalization`). POST `/`.
   */
  generate: (payload: LeanAideAnyPayload): Promise<LeanAideRawResponse> =>
    leanaideClient.post<LeanAideRawResponse>('/', payload),

  /**
   * Verification task (`elaborate`). POST `/`.
   */
  verify: (payload: LeanAideAnyPayload): Promise<LeanAideRawResponse> =>
    leanaideClient.post<LeanAideRawResponse>('/', payload),

  /**
   * Send a raw, unmodified task payload to `POST /` and return the raw JSON.
   */
  rawResponse: (payload: LeanAideAnyPayload): Promise<LeanAideRawResponse> =>
    leanaideClient.post<LeanAideRawResponse>('/', payload),

  /**
   * Embedding similarity search. POST `/run-sim-search`.
   */
  simSearch: (
    payload: LeanAideSimSearchPayload
  ): Promise<LeanAideRawResponse> =>
    leanaideClient.post<LeanAideRawResponse>('/run-sim-search', payload),

  /**
   * Start a benchmark (proxied + mocked by the BubbleLab API).
   */
  runBenchmark: (): Promise<LeanAideBenchmarkStartResponse> =>
    bubblelabClient.post<LeanAideBenchmarkStartResponse>(
      '/leanaide/benchmark/start',
      {}
    ),

  /**
   * Poll benchmark results by id (proxied + mocked by the BubbleLab API).
   */
  getBenchmark: (benchmarkId: string): Promise<LeanAideBenchmarkResult> =>
    bubblelabClient.get<LeanAideBenchmarkResult>(
      `/leanaide/benchmark/${encodeURIComponent(benchmarkId)}/results`
    ),
};
