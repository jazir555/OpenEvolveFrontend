import type { RAGBitsPlugin, RAGBitsPluginConfig } from './types/plugin-types';
export type { RAGBitsPluginConfig, RAGBitsPluginState, RAGBitsSearchRequest, RAGBitsSearchResponse, RAGBitsSearchResult, RAGBitsIngestRequest, RAGBitsIngestResponse, RAGBitsIndexStats, RAGBitsPluginContext, RAGBitsPluginMethods, RAGBitsPlugin, RAGBitsPluginProps, RAGBitsConfigPanelProps, RAGBitsSearchPanelProps, RAGBitsIngestPanelProps, RAGBitsStatusIndicatorProps, RAGBitsSearchResultsProps, RAGBitsSearchType, RAGBitsDocumentType } from './types/plugin-types';
export { RAGBITS_SEARCH_TYPES, RAGBITS_DOCUMENT_TYPES, DEFAULT_RAGBITS_CONFIG } from './types/plugin-types';
export { RAGBitsConfigPanel } from './components/RAGBitsConfigPanel';
export { RAGBitsSearchPanel } from './components/RAGBitsSearchPanel';
export { RAGBitsIngestPanel } from './components/RAGBitsIngestPanel';
export { RAGBitsStatusIndicator } from './components/RAGBitsStatusIndicator';
export { RAGBitsSearchResults } from './components/RAGBitsSearchResults';
export { useRAGBitsConfig } from './hooks/useRAGBitsConfig';
export { useRAGBitsState } from './hooks/useRAGBitsState';
export { useRAGBitsSearch } from './hooks/useRAGBitsSearch';
export { useRAGBitsIngest } from './hooks/useRAGBitsIngest';
export { RagbitsClient } from './lib/ragbitsClient';
export { RagbitsService } from './services/ragbitsService';
export { createRAGBitsPlugin, getRAGBitsPlugin, useRAGBitsPlugin } from './utils/createRAGBitsPlugin';
/**
 * Create a new RAGBits plugin instance
 * @param config Optional initial configuration
 * @returns RAGBitsPlugin instance
 */
export declare function createPlugin(config?: Partial<RAGBitsPluginConfig>): RAGBitsPlugin;
/**
 * Default plugin instance
 */
export declare const ragbitsPlugin: RAGBitsPlugin;
export default ragbitsPlugin;
//# sourceMappingURL=index.d.ts.map