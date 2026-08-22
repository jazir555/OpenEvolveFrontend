/**
 * GKET (Generic Knowledge Extraction Tool) API Client
 *
 * Connects BubbleLab frontend to the GKET FastAPI wrapper
 * (`core-projects/Generic-Knowledge-Extraction-Tool/server.py`), listening on
 * `http://localhost:8766` by default (`VITE_GKET_API_URL`).
 *
 * Contract (all routes unprefixed — the server mounts them at the root):
 * - `GET  /healthz`               -> `{ status: 'ok' }`
 * - `POST /parse`                 -> `{ file_ref, parser: 'fast' | 'docling' }`
 * - `POST /generate-models`       -> `{ text_description }` -> field config + JSON schema
 * - `POST /extract`               -> `{ case: 0 | 1 | 2, llm, text_or_file_ref, ... }`
 * - `GET  /export/{id}?format=`   -> stored `/extract` result as json | csv | xlsx
 *
 * The backend never throws: recoverable failures come back as
 * `{ status: 'error', error, detail }` with HTTP 200, and partial successes carry
 * a `notes` array explaining what degraded (missing API key, absent docling,
 * missing `core/model_generator.py`, ...). Always surface `notes` in the UI.
 */

import { ApiClient, ApiClientConfig } from '@/lib/api';
import { GKET_API_URL } from '@/env';

// ==================== Client Instance ====================

const gketClientConfig: ApiClientConfig = {
  baseURL: GKET_API_URL,
  timeout: 180000, // document parsing + LLM extraction can be slow
  enableRetry: true,
  maxRetries: 2,
  retryDelay: 2000,
};

const gketClient = new ApiClient(gketClientConfig);

// ==================== Types ====================

export type GketParser = 'fast' | 'docling';
export type GketCase = 0 | 1 | 2;
export type GketLlm = 'openai' | 'claude';
export type GketExportFormat = 'json' | 'csv' | 'xlsx';

/** Shape returned by every endpoint when the backend degrades gracefully. */
export interface GketErrorResponse {
  status: 'error';
  error: string;
  detail?: string;
}

export interface GketHealthResponse {
  status: string;
}

export interface GketParsePayload {
  file_ref: string;
  parser?: GketParser;
  use_markdown?: boolean;
}

export interface GketParseResponse {
  status: string;
  parser?: GketParser;
  file_name?: string;
  text?: string;
  content_length?: number;
  word_count?: number;
  metadata?: Record<string, unknown>;
  notes?: string[];
  error?: string;
  detail?: string;
}

export interface GketGenerateModelsPayload {
  text_description: string;
  use_case?: string;
  context?: string;
  llm?: GketLlm;
}

/** One parsed field of a generated model (mirrors `core/text_description_parser.py`). */
export interface GketParsedField {
  field_name: string;
  field_type: string;
  description?: string;
  required?: boolean;
  enum_values?: string[] | null;
}

export interface GketGenerateModelsResponse {
  status: string;
  model_name?: string;
  fields?: GketParsedField[];
  json_schema?: Record<string, unknown>;
  model_code?: string;
  config?: Record<string, unknown>;
  notes?: string[];
  error?: string;
  detail?: string;
}

export interface GketExtractPayload {
  case: GketCase;
  llm?: GketLlm;
  text_or_file_ref: string;
  /** Field config or JSON schema for case 0. */
  model_schema?: Record<string, unknown> | null;
  instruction?: string;
  use_case?: string;
}

export interface GketExtractResponse {
  id?: string;
  status: string;
  records?: Array<Record<string, unknown>>;
  case?: GketCase;
  notes?: string[];
  error?: string;
  detail?: string;
}

export interface GketExportJsonResponse {
  id: string;
  status: string;
  case?: GketCase;
  records: Array<Record<string, unknown>>;
  raw?: unknown;
  notes?: string[];
}

/**
 * `csv` and `xlsx` come back as non-JSON bodies, which `ApiClient` returns as
 * text; `json` comes back parsed.
 */
export type GketExportResponse = GketExportJsonResponse | GketErrorResponse | string;

// ==================== API Client ====================

export const gketApi = {
  /**
   * Liveness probe. `GET /healthz`.
   */
  healthz: (): Promise<GketHealthResponse> =>
    gketClient.get<GketHealthResponse>('/healthz'),

  /**
   * Parse a document by server-side path. `POST /parse`.
   */
  parseDoc: (payload: GketParsePayload): Promise<GketParseResponse> =>
    gketClient.post<GketParseResponse>('/parse', {
      parser: 'fast',
      use_markdown: true,
      ...payload,
    }),

  /**
   * Turn a natural-language description into a Pydantic model schema.
   * `POST /generate-models`.
   */
  generateModels: (
    payload: GketGenerateModelsPayload
  ): Promise<GketGenerateModelsResponse> =>
    gketClient.post<GketGenerateModelsResponse>('/generate-models', {
      llm: 'openai',
      ...payload,
    }),

  /**
   * Run case 0 (single type), 1 (classification routing) or 2 (hierarchical).
   * `POST /extract`.
   */
  extract: (payload: GketExtractPayload): Promise<GketExtractResponse> =>
    gketClient.post<GketExtractResponse>('/extract', {
      llm: 'openai',
      ...payload,
    }),

  /**
   * Fetch a stored extraction result. `GET /export/{id}?format=`.
   */
  getExport: (
    id: string,
    format: GketExportFormat = 'json'
  ): Promise<GketExportResponse> =>
    gketClient.get<GketExportResponse>(
      `/export/${encodeURIComponent(id)}?format=${encodeURIComponent(format)}`
    ),
};
