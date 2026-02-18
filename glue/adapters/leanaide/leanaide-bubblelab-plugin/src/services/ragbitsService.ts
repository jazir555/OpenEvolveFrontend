import {
  RagbitsClient,
  RagbitsSearchRequest,
  RagbitsSearchResponse,
  RagbitsIngestRequest,
  RagbitsIngestResponse,
} from '../lib/ragbitsClient';

const DEFAULT_RAGBITS_CONFIG = {
  serverUrl: import.meta.env.VITE_RAGBITS_SERVER_URL || 'http://localhost:3000/ragbits',
};

let ragbitsClient: RagbitsClient | null = null;

export function getRagbitsClient(): RagbitsClient {
  if (!ragbitsClient) {
    ragbitsClient = new RagbitsClient(DEFAULT_RAGBITS_CONFIG);
  }
  return ragbitsClient;
}

export function initializeRagbitsClient(config: { serverUrl?: string; apiKey?: string }) {
  const clientConfig = {
    serverUrl: config.serverUrl || DEFAULT_RAGBITS_CONFIG.serverUrl,
    apiKey: config.apiKey,
  };
  ragbitsClient = new RagbitsClient(clientConfig);
}

export async function searchKnowledge(
  request: RagbitsSearchRequest
): Promise<RagbitsSearchResponse> {
  const client = getRagbitsClient();
  return client.search(request);
}

export async function ingestArtifact(
  request: RagbitsIngestRequest
): Promise<RagbitsIngestResponse> {
  const client = getRagbitsClient();
  return client.ingest(request);
}

export function isRagbitsAvailable(): boolean {
  return !!ragbitsClient;
}
