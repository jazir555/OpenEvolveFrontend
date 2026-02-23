"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getRagbitsClient = getRagbitsClient;
exports.initializeRagbitsClient = initializeRagbitsClient;
exports.searchKnowledge = searchKnowledge;
exports.ingestArtifact = ingestArtifact;
exports.isRagbitsAvailable = isRagbitsAvailable;
const ragbitsClient_1 = require("../lib/ragbitsClient");
const DEFAULT_RAGBITS_CONFIG = {
    serverUrl: import.meta.env.VITE_RAGBITS_SERVER_URL || 'http://localhost:3000/ragbits',
};
let ragbitsClient = null;
function getRagbitsClient() {
    if (!ragbitsClient) {
        ragbitsClient = new ragbitsClient_1.RagbitsClient(DEFAULT_RAGBITS_CONFIG);
    }
    return ragbitsClient;
}
function initializeRagbitsClient(config) {
    const clientConfig = {
        serverUrl: config.serverUrl || DEFAULT_RAGBITS_CONFIG.serverUrl,
        apiKey: config.apiKey,
    };
    ragbitsClient = new ragbitsClient_1.RagbitsClient(clientConfig);
}
async function searchKnowledge(request) {
    const client = getRagbitsClient();
    return client.search(request);
}
async function ingestArtifact(request) {
    const client = getRagbitsClient();
    return client.ingest(request);
}
function isRagbitsAvailable() {
    return !!ragbitsClient;
}
//# sourceMappingURL=ragbitsService.js.map