export enum VectorDBType {
  QDRANT = 'qdrant',
  PINECONE = 'pinecone',
  CHROMA = 'chroma',
  PGVECTOR = 'pgvector',
}

export interface VectorEntry {
  id: string;
  vector: number[];
  metadata?: Record<string, any>;
  payload?: Record<string, any>;
  text?: string;
  created_at?: string;
}

export interface CollectionConfig {
  name: string;
  dimension: number;
  distance_metric: 'cosine' | 'euclidean' | 'dot_product';
}

export interface SearchQuery {
  vector: number[];
  k: number;
  score_threshold?: number;
  filter?: Record<string, any>;
  distance_metric?: 'cosine' | 'euclidean' | 'dot_product';
}

export interface SearchResult {
  entry: VectorEntry;
  score: number;
  distance: number;
}

export interface UpsertRequest {
  collection_name: string;
  entries: VectorEntry[];
  namespace?: string;
}

export interface UpsertResponse {
  upserted_count: number;
  collection_name: string;
  timestamp: string;
}

export interface DeleteRequest {
  collection_name: string;
  ids: string[];
  delete_all?: boolean;
  namespace?: string;
}

export interface DeleteResponse {
  deleted_count: number;
  collection_name: string;
  timestamp: string;
}

export interface CollectionInfo {
  name: string;
  dimension: number;
  vector_count: number;
  distance_metric: string;
  created_at: string;
  updated_at: string;
}

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy';
  backend_type: VectorDBType;
  connected: boolean;
  latency_ms?: number;
  error?: string;
  timestamp: string;
}

export function transformCanonicalToQdrant(entry: VectorEntry): { id: string; vector: number[]; payload?: Record<string, any> } {
  return { id: entry.id, vector: entry.vector, payload: entry.metadata ?? entry.payload };
}

export function transformQdrantToCanonical(point: { id: string; vector: number[]; payload?: Record<string, any> }): VectorEntry {
  return { id: point.id, vector: point.vector, metadata: point.payload, payload: point.payload };
}

export function transformCanonicalToPinecone(entry: VectorEntry): { id: string; values: number[]; metadata?: Record<string, any> } {
  return { id: entry.id, values: entry.vector, metadata: entry.metadata };
}

export function transformPineconeToCanonical(match: { id: string; values?: number[]; vector?: number[]; metadata?: Record<string, any> }): VectorEntry {
  return { id: match.id, vector: match.values ?? match.vector ?? [], metadata: match.metadata };
}

export function transformCanonicalToChroma(entry: VectorEntry): { id: string; vector: number[]; metadata?: Record<string, any> } {
  return { id: entry.id, vector: entry.vector, metadata: entry.metadata };
}

export function transformChromaToCanonical(embedding: { id: string; vector?: number[]; embedding?: number[]; metadata?: Record<string, any> }): VectorEntry {
  return { id: embedding.id, vector: embedding.vector ?? embedding.embedding ?? [], metadata: embedding.metadata };
}

export function transformCanonicalToPgvector(entry: VectorEntry): { id: string; vector: number[]; text?: string; metadata?: Record<string, any>; created_at?: string } {
  return { id: entry.id, vector: entry.vector, text: entry.text, metadata: entry.metadata, created_at: entry.created_at };
}

export function transformPgvectorToCanonical(row: { id: string; vector: number[]; text?: string; metadata?: Record<string, any>; created_at?: string }): VectorEntry {
  return { id: row.id, vector: row.vector, text: row.text, metadata: row.metadata, created_at: row.created_at };
}

export function validateVectorDimension(vector: number[], dimension: number): { valid: boolean; error?: string } {
  if (!Array.isArray(vector) || vector.length !== dimension) {
    return { valid: false, error: `Vector dimension ${Array.isArray(vector) ? vector.length : 'invalid'} does not match required ${dimension}` };
  }
  return { valid: true };
}

function ok(): { success: boolean; error?: { issues: any } } {
  return { success: true };
}

function fail(issue: string): { success: boolean; error?: { issues: any } } {
  return { success: false, error: { issues: [{ message: issue }] } };
}

export function validateVectorEntry(entry: VectorEntry): { success: boolean; error?: { issues: any } } {
  if (!entry || typeof entry.id !== 'string') return fail('entry.id must be a string');
  if (!Array.isArray(entry.vector)) return fail('entry.vector must be an array of numbers');
  return ok();
}

export function validateSearchQuery(query: SearchQuery): { success: boolean; error?: { issues: any } } {
  if (!Array.isArray(query.vector)) return fail('query.vector must be an array of numbers');
  if (typeof query.k !== 'number' || query.k <= 0) return fail('query.k must be a positive number');
  return ok();
}

export function validateUpsertRequest(request: UpsertRequest): { success: boolean; error?: { issues: any } } {
  if (!request || typeof request.collection_name !== 'string') return fail('request.collection_name must be a string');
  if (!Array.isArray(request.entries) || request.entries.length === 0) return fail('request.entries must be a non-empty array');
  return ok();
}

export function validateDeleteRequest(request: DeleteRequest): { success: boolean; error?: { issues: any } } {
  if (!request || typeof request.collection_name !== 'string') return fail('request.collection_name must be a string');
  if (request.delete_all !== true && (!Array.isArray(request.ids) || request.ids.length === 0)) return fail('request.ids must be a non-empty array when delete_all is false');
  return ok();
}

export function validateCollectionConfig(config: CollectionConfig): { success: boolean; error?: { issues: any } } {
  if (!config || typeof config.name !== 'string') return fail('config.name must be a string');
  if (typeof config.dimension !== 'number' || config.dimension <= 0) return fail('config.dimension must be a positive number');
  return ok();
}
