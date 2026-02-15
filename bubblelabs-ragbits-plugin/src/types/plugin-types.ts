// RAGBits Plugin Types and Interfaces
// Defines all types and interfaces for the BubbleLabs RAGBits plugin

export interface RAGBitsPluginConfig {
  /** Enable/disable the plugin */
  enabled: boolean;

  /** RAGBits server configuration */
  serverUrl: string;
  apiKey?: string;
  timeout?: number;

  /** Search settings */
  defaultTopK: number;
  defaultScoreThreshold: number;
  enableHybridSearch: boolean;
  enableReranking: boolean;

  /** Indexing settings */
  autoIndexArtifacts: boolean;
  indexingBatchSize: number;

  /** Integration settings */
  integrateWithDecomposition: boolean;
  integrateWithKnowledgeEngine: boolean;
  integrateWithEvolution: boolean;

  /** Performance settings */
  enableCaching: boolean;
  cacheTTLSeconds: number;
  maxSearchTime: number;

  /** UI settings */
  showAdvancedOptions: boolean;
  showDebugInfo: boolean;
  theme: 'light' | 'dark' | 'system';
}

export interface RAGBitsPluginState extends RAGBitsPluginConfig {
  /** Current plugin status */
  status: 'idle' | 'initializing' | 'ready' | 'error' | 'busy';

  /** Current operation */
  currentOperation?: {
    type: 'search' | 'ingest' | 'index' | 'configuration';
    startedAt: Date;
    progress?: number;
    message?: string;
  };

  /** Recent operations history */
  operationHistory: Array<{
    id: string;
    type: string;
    timestamp: Date;
    success: boolean;
    message: string;
    details?: any;
  }>;

  /** Statistics */
  statistics: {
    totalSearches: number;
    successfulSearches: number;
    failedSearches: number;
    totalDocumentsIndexed: number;
    averageSearchTime: number;
    averageRelevanceScore: number;
    lastOperationTime?: Date;
  };
}

export interface RAGBitsSearchRequest {
  query: string;
  topK?: number;
  scoreThreshold?: number;
  filter?: {
    documentType?: string;
    stage?: string;
    team?: string;
    tags?: string[];
  };
  enableHybridSearch?: boolean;
  enableReranking?: boolean;
}

export interface RAGBitsSearchResult {
  documentId: string;
  content: string;
  metadata: {
    documentType: string;
    source?: string;
    stage?: string;
    team?: string;
    tags?: string[];
    timestamp?: Date;
    [key: string]: any;
  };
  score: number;
  relevanceScore: number;
}

export interface RAGBitsSearchResponse {
  success: boolean;
  query: string;
  results: RAGBitsSearchResult[];
  totalResults: number;
  executionTime: number;
  metadata: {
    searchType: 'semantic' | 'hybrid' | 'keyword';
    vectorStoreUsed: string;
    rerankingApplied: boolean;
    cacheHit: boolean;
  };
  errors: string[];
  warnings: string[];
  timestamp: Date;
}

export interface RAGBitsIngestRequest {
  content: string;
  metadata: {
    documentType: string;
    source?: string;
    stage?: string;
    team?: string;
    tags?: string[];
    [key: string]: any;
  };
}

export interface RAGBitsIngestResponse {
  success: boolean;
  documentId: string;
  message: string;
  executionTime: number;
  errors: string[];
  warnings: string[];
  timestamp: Date;
}

export interface RAGBitsIndexStats {
  totalDocuments: number;
  documentsByType: Record<string, number>;
  documentsByStage: Record<string, number>;
  documentsByTeam: Record<string, number>;
  indexSize: number;
  lastUpdated: Date;
}

export interface RAGBitsPluginContext {
  /** Plugin configuration */
  config: RAGBitsPluginConfig;

  /** Plugin state */
  state: RAGBitsPluginState;

  /** Available search types */
  searchTypes: Array<{
    value: string;
    label: string;
    description: string;
    recommendedFor: string[];
  }>;

  /** Supported document types */
  documentTypes: Array<{
    value: string;
    label: string;
    description: string;
  }>;

  /** Plugin capabilities */
  capabilities: {
    semanticSearch: boolean;
    hybridSearch: boolean;
    keywordSearch: boolean;
    reranking: boolean;
    caching: boolean;
    indexing: boolean;
    monitoring: boolean;
    reporting: boolean;
  };
}

export interface RAGBitsPluginMethods {
  /** Initialize the plugin */
  initialize: (config?: Partial<RAGBitsPluginConfig>) => Promise<void>;

  /** Update plugin configuration */
  updateConfig: (config: Partial<RAGBitsPluginConfig>) => Promise<void>;

  /** Reset plugin configuration */
  resetConfig: () => Promise<void>;

  /** Search for documents */
  search: (request: RAGBitsSearchRequest) => Promise<RAGBitsSearchResponse>;

  /** Ingest a document */
  ingest: (request: RAGBitsIngestRequest) => Promise<RAGBitsIngestResponse>;

  /** Batch ingest documents */
  batchIngest: (requests: RAGBitsIngestRequest[]) => Promise<RAGBitsIngestResponse[]>;

  /** Get index statistics */
  getIndexStats: () => Promise<RAGBitsIndexStats>;

  /** Clear cache */
  clearCache: () => Promise<void>;

  /** Get plugin statistics */
  getStatistics: () => RAGBitsPluginState['statistics'];

  /** Get operation history */
  getOperationHistory: () => RAGBitsPluginState['operationHistory'];

  /** Clear operation history */
  clearOperationHistory: () => void;

  /** Get plugin status */
  getStatus: () => RAGBitsPluginState['status'];

  /** Get full plugin context */
  getContext: () => RAGBitsPluginContext;
}

export interface RAGBitsPlugin extends RAGBitsPluginMethods {
  /** Plugin metadata */
  metadata: {
    name: string;
    version: string;
    description: string;
    author: string;
    website: string;
  };

  /** React components */
  components: {
    ConfigPanel: React.ComponentType<{ onClose: () => void }>;
    SearchPanel: React.ComponentType<{ onResult: (result: any) => void }>;
    IngestPanel: React.ComponentType<{ onSuccess: (response: any) => void }>;
    StatusIndicator: React.ComponentType<{}>;
    SearchResults: React.ComponentType<{ results: RAGBitsSearchResult[] }>;
  };

  /** React hooks */
  hooks: {
    useRAGBitsConfig: () => [RAGBitsPluginConfig, (config: Partial<RAGBitsPluginConfig>) => void];
    useRAGBitsState: () => RAGBitsPluginState;
    useRAGBitsSearch: () => (request: RAGBitsSearchRequest) => Promise<RAGBitsSearchResponse>;
    useRAGBitsIngest: () => (request: RAGBitsIngestRequest) => Promise<RAGBitsIngestResponse>;
  };
}

export interface RAGBitsPluginProps {
  /** Plugin configuration */
  config?: Partial<RAGBitsPluginConfig>;

  /** Callback for configuration changes */
  onConfigChange?: (config: RAGBitsPluginConfig) => void;

  /** Callback for operation results */
  onOperationResult?: (operation: 'search' | 'ingest' | 'index', result: any) => void;

  /** Callback for errors */
  onError?: (error: Error) => void;

  /** Callback for status changes */
  onStatusChange?: (status: RAGBitsPluginState['status']) => void;

  /** Children components */
  children?: React.ReactNode;
}

export interface RAGBitsConfigPanelProps {
  /** Initial configuration */
  initialConfig?: Partial<RAGBitsPluginConfig>;

  /** Callback when configuration is saved */
  onSave: (config: RAGBitsPluginConfig) => void;

  /** Callback when configuration is cancelled */
  onCancel: () => void;

  /** Show advanced options */
  showAdvanced?: boolean;
}

export interface RAGBitsSearchPanelProps {
  /** Initial query */
  initialQuery?: string;

  /** Callback with search result */
  onResult: (result: RAGBitsSearchResponse) => void;

  /** Callback when panel is closed */
  onClose: () => void;

  /** Show debug information */
  showDebug?: boolean;
}

export interface RAGBitsIngestPanelProps {
  /** Callback with ingest result */
  onSuccess: (response: RAGBitsIngestResponse) => void;

  /** Callback when panel is closed */
  onClose: () => void;

  /** Show debug information */
  showDebug?: boolean;
}

export interface RAGBitsStatusIndicatorProps {
  /** Custom class name */
  className?: string;

  /** Show detailed status */
  showDetails?: boolean;
}

export interface RAGBitsSearchResultsProps {
  /** Search results to display */
  results: RAGBitsSearchResult[];

  /** On result click handler */
  onResultClick?: (result: RAGBitsSearchResult) => void;

  /** Show metadata */
  showMetadata?: boolean;

  /** Show scores */
  showScores?: boolean;
}

export type RAGBitsSearchType = 'semantic' | 'hybrid' | 'keyword';

export type RAGBitsDocumentType =
  'solution' | 'problem' | 'test_case' | 'documentation' |
  'code' | 'analysis' | 'report' | 'artifact' | 'general';

export const RAGBITS_SEARCH_TYPES: Array<{
  value: RAGBitsSearchType;
  label: string;
  description: string;
  recommendedFor: string[];
}> = [
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

export const RAGBITS_DOCUMENT_TYPES: Array<{
  value: RAGBitsDocumentType;
  label: string;
  description: string;
}> = [
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
 * - Must use process.env.RAGBITS_SERVER_URL
 * - Crashes immediately if missing (no magic defaults)
 * - Validates URL format at startup
 */
function getRAGBitsServerUrl(): string {
  const url = process.env.RAGBITS_SERVER_URL;

  if (!url) {
    throw new Error(
      'FATAL: RAGBITS_SERVER_URL environment variable is not set. ' +
      'This is a required configuration. Please set RAGBITS_SERVER_URL before starting the service. ' +
      'Example: http://localhost:3000/ragbits or http://ragbits-core:3000/ragbits'
    );
  }

  // Validate URL format
  try {
    new URL(url);
  } catch (error) {
    throw new Error(
      `FATAL: Invalid RAGBITS_SERVER_URL format: "${url}". ` +
      `Must be a valid URL. Error: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }

  return url;
}

export const DEFAULT_RAGBITS_CONFIG: RAGBitsPluginConfig = {
  enabled: true,
  serverUrl: (() => {
    try {
      return getRAGBitsServerUrl();
    } catch (error) {
      // Fallback for development only (production will crash)
      console.warn(
        'WARNING: Using fallback RAGBits URL. ' +
        'Set RAGBITS_SERVER_URL environment variable for production use.'
      );
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
