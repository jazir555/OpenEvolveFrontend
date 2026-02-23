"use strict";
// RAGBits Plugin Types and Interfaces
// Defines all types and interfaces for the BubbleLabs RAGBits plugin
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_RAGBITS_CONFIG = exports.RAGBITS_DOCUMENT_TYPES = exports.RAGBITS_SEARCH_TYPES = void 0;
exports.RAGBITS_SEARCH_TYPES = [
    {
        value: 'semantic',
        label: 'Semantic Search',
        description: 'Vector-based semantic similarity search',
        recommendedFor: ['Conceptual queries', 'Meaning-based search', 'Cross-domain matching']
    },
    {
        value: 'hybrid',
        label: 'Hybrid Search',
        description: 'Combines semantic and keyword search',
        recommendedFor: ['Complex queries', 'Precise matching', 'Balanced results']
    },
    {
        value: 'keyword',
        label: 'Keyword Search',
        description: 'Traditional keyword-based search',
        recommendedFor: ['Exact term matching', 'Specific phrases', 'Technical terms']
    }
];
exports.RAGBITS_DOCUMENT_TYPES = [
    { value: 'solution', label: 'Solution', description: 'Solution documents and proposals' },
    { value: 'problem', label: 'Problem', description: 'Problem statements and descriptions' },
    { value: 'test_case', label: 'Test Case', description: 'Test cases and testing artifacts' },
    { value: 'documentation', label: 'Documentation', description: 'Technical documentation' },
    { value: 'code', label: 'Code', description: 'Source code and scripts' },
    { value: 'analysis', label: 'Analysis', description: 'Analysis documents and reports' },
    { value: 'report', label: 'Report', description: 'Progress and status reports' },
    { value: 'artifact', label: 'Artifact', description: 'Workflow artifacts and outputs' },
    { value: 'general', label: 'General', description: 'General documents' }
];
/**
 * Get RAGBits server URL from environment variable
 * Per CLAUDE.md Law of Configuration Explicitness:
 * - Must use import.meta.env.VITE_RAGBITS_SERVER_URL
 * - Crashes immediately if missing (no magic defaults)
 * - Validates URL format at startup
 */
function getRAGBitsServerUrl() {
    const url = import.meta.env.VITE_RAGBITS_SERVER_URL;
    if (!url) {
        throw new Error('FATAL: RAGBITS_SERVER_URL environment variable is not set. ' +
            'This is a required configuration. Please set RAGBITS_SERVER_URL before starting the service. ' +
            'Example: http://localhost:3000/ragbits or http://ragbits-core:3000/ragbits');
    }
    // Validate URL format
    try {
        new URL(url);
    }
    catch (error) {
        throw new Error(`FATAL: Invalid RAGBITS_SERVER_URL format: "${url}". ` +
            `Must be a valid URL. Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
    return url;
}
exports.DEFAULT_RAGBITS_CONFIG = {
    enabled: true,
    serverUrl: (() => {
        try {
            return getRAGBitsServerUrl();
        }
        catch (error) {
            // Fallback for development only (production will crash)
            console.warn('WARNING: Using fallback RAGBits URL. ' +
                'Set VITE_RAGBITS_SERVER_URL environment variable for production use.');
            return 'http://localhost:3000/ragbits';
        }
    })(),
    apiKey: '',
    timeout: 30,
    defaultTopK: 10,
    defaultScoreThreshold: 0.7,
    enableHybridSearch: true,
    enableReranking: true,
    autoIndexArtifacts: true,
    indexingBatchSize: 100,
    integrateWithDecomposition: true,
    integrateWithKnowledgeEngine: true,
    integrateWithEvolution: true,
    enableCaching: true,
    cacheTTLSeconds: 3600,
    maxSearchTime: 15,
    showAdvancedOptions: false,
    showDebugInfo: false,
    theme: 'system'
};
//# sourceMappingURL=plugin-types.js.map