// BubbleLabs RAGBits Plugin - Main Export
// Standalone plugin for semantic document search and knowledge retrieval

import { createRAGBitsPlugin } from './utils/createRAGBitsPlugin';
import type { RAGBitsPlugin, RAGBitsPluginConfig } from './types/plugin-types';

// Export types
export type {
  RAGBitsPluginConfig,
  RAGBitsPluginState,
  RAGBitsSearchRequest,
  RAGBitsSearchResponse,
  RAGBitsSearchResult,
  RAGBitsIngestRequest,
  RAGBitsIngestResponse,
  RAGBitsIndexStats,
  RAGBitsPluginContext,
  RAGBitsPluginMethods,
  RAGBitsPlugin,
  RAGBitsPluginProps,
  RAGBitsConfigPanelProps,
  RAGBitsSearchPanelProps,
  RAGBitsIngestPanelProps,
  RAGBitsStatusIndicatorProps,
  RAGBitsSearchResultsProps,
  RAGBitsSearchType,
  RAGBitsDocumentType
} from './types/plugin-types';

export {
  RAGBITS_SEARCH_TYPES,
  RAGBITS_DOCUMENT_TYPES,
  DEFAULT_RAGBITS_CONFIG
} from './types/plugin-types';

// Export components
export { RAGBitsConfigPanel } from './components/RAGBitsConfigPanel';
export { RAGBitsSearchPanel } from './components/RAGBitsSearchPanel';
export { RAGBitsIngestPanel } from './components/RAGBitsIngestPanel';
export { RAGBitsStatusIndicator } from './components/RAGBitsStatusIndicator';
export { RAGBitsSearchResults } from './components/RAGBitsSearchResults';

// Export hooks
export { useRAGBitsConfig } from './hooks/useRAGBitsConfig';
export { useRAGBitsState } from './hooks/useRAGBitsState';
export { useRAGBitsSearch } from './hooks/useRAGBitsSearch';
export { useRAGBitsIngest } from './hooks/useRAGBitsIngest';

// Export services
export { RagbitsClient } from './lib/ragbitsClient';
export { RagbitsService } from './services/ragbitsService';

// Export utilities
export { createRAGBitsPlugin, getRAGBitsPlugin, useRAGBitsPlugin } from './utils/createRAGBitsPlugin';

/**
 * Create a new RAGBits plugin instance
 * @param config Optional initial configuration
 * @returns RAGBitsPlugin instance
 */
export function createPlugin(config?: Partial<RAGBitsPluginConfig>): RAGBitsPlugin {
  return createRAGBitsPlugin(config);
}

/**
 * Default plugin instance
 */
export const ragbitsPlugin = createRAGBitsPlugin();

export default ragbitsPlugin;
