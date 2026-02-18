import { RagbitsClient, RagbitsSearchRequest, RagbitsSearchResponse, RagbitsIngestRequest, RagbitsIngestResponse } from '../lib/ragbitsClient';
export declare function getRagbitsClient(): RagbitsClient;
export declare function initializeRagbitsClient(config: {
    serverUrl?: string;
    apiKey?: string;
}): void;
export declare function searchKnowledge(request: RagbitsSearchRequest): Promise<RagbitsSearchResponse>;
export declare function ingestArtifact(request: RagbitsIngestRequest): Promise<RagbitsIngestResponse>;
export declare function isRagbitsAvailable(): boolean;
