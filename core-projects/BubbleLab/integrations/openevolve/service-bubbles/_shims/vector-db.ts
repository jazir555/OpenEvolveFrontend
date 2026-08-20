/**
 * Local shim for the vector DB service.
 *
 * The real implementation lives outside this package
 * (../../../services/vector-db). This placeholder keeps the integration
 * typecheckable in isolation. Swap for the real import when wiring the
 * full monorepo build.
 */
export interface VectorDBServiceConfig {
  type: string;
  url?: string;
  apiKey?: string;
  collectionName?: string;
  index?: string;
  [key: string]: unknown;
}

export default class VectorDBService {
  constructor(_config: VectorDBServiceConfig) {}

  async search(_query: string, _limit: number, _filters?: unknown): Promise<any[]> {
    return [];
  }
}
