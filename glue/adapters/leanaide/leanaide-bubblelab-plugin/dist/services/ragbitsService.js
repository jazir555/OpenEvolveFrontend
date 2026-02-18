import { RagbitsClient, } from '../lib/ragbitsClient';
const DEFAULT_RAGBITS_CONFIG = {
    serverUrl: import.meta.env.VITE_RAGBITS_SERVER_URL || 'http://localhost:3000/ragbits',
};
let ragbitsClient = null;
export function getRagbitsClient() {
    if (!ragbitsClient) {
        ragbitsClient = new RagbitsClient(DEFAULT_RAGBITS_CONFIG);
    }
    return ragbitsClient;
}
export function initializeRagbitsClient(config) {
    const clientConfig = {
        serverUrl: config.serverUrl || DEFAULT_RAGBITS_CONFIG.serverUrl,
        apiKey: config.apiKey,
    };
    ragbitsClient = new RagbitsClient(clientConfig);
}
export async function searchKnowledge(request) {
    const client = getRagbitsClient();
    return client.search(request);
}
export async function ingestArtifact(request) {
    const client = getRagbitsClient();
    return client.ingest(request);
}
export function isRagbitsAvailable() {
    return !!ragbitsClient;
}
//# sourceMappingURL=ragbitsService.js.map