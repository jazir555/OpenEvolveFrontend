// Datapizza Plugin Types and Interfaces
// Defines all types and interfaces for the BubbleLabs Datapizza plugin

import type { ComponentType, ReactNode } from 'react';

export interface DatapizzaPluginConfig {
  /** Enable/disable the plugin */
  enabled: boolean;

  /** Datapizza server configuration */
  serverUrl: string;
  apiKey?: string;
  timeout: number;

  /** Pipeline settings */
  pipelineEnabled: boolean;
  autoDetectDataSources: boolean;
  defaultPipelineType: 'standard' | 'advanced' | 'custom';

  /** Data processing configurations */
  dataProcessingConfig: {
    chunkSize?: number;
    overlapSize?: number;
    embeddingModel?: string;
    vectorStoreType?: string;
    maxParallelProcesses?: number;
  };

  /** Agent configurations */
  agentConfigurations: {
    agent1?: {
      enabled: boolean;
      maxTasks?: number;
      timeout?: number;
    };
    agent2?: {
      enabled: boolean;
      parallelExecution?: boolean;
      maxWorkers?: number;
    };
    agent3?: {
      enabled: boolean;
      critiqueLevel?: 'basic' | 'standard' | 'advanced';
    };
  };

  /** Integration settings */
  integrateWithWorkflow: boolean;
  integrateWithKnowledgeGraph: boolean;
  integrateWithExternalSources: boolean;

  /** Performance settings */
  enableCaching: boolean;
  cacheTTLSeconds: number;
  maxProcessingTime: number;

  /** UI settings */
  showAdvancedOptions: boolean;
  showDebugInfo: boolean;
  theme: 'light' | 'dark' | 'system';
}

export interface DatapizzaPluginState extends DatapizzaPluginConfig {
  /** Current plugin status */
  status: 'idle' | 'initializing' | 'ready' | 'error' | 'busy';

  /** Current operation */
  currentOperation?: {
    type: 'pipeline' | 'processing' | 'configuration' | 'query';
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
    totalOperations: number;
    successfulOperations: number;
    failedOperations: number;
    averageProcessingTime: number;
    lastOperationTime?: Date;
  };
}

export interface DatapizzaPipelineResult {
  success: boolean;
  pipelineId: string;
  dataSource?: string;
  processedData?: any;
  confidenceScore: number;
  pipelineType?: string;
  dataDomain?: string;
  errors: string[];
  warnings: string[];
  executionTime: number;
  metadata: Record<string, any>;
  timestamp: Date;
}

export interface DatapizzaProcessingResult {
  success: boolean;
  dataId: string;
  processedData: any;
  confidenceScore: number;
  processingType?: string;
  errors: string[];
  warnings: string[];
  executionTime: number;
  metadata: Record<string, any>;
  timestamp: Date;
}

export interface DatapizzaQueryResult {
  success: boolean;
  query: string;
  results: any[];
  confidenceScore: number;
  processingTime: number;
  errors: string[];
  warnings: string[];
  metadata: Record<string, any>;
  timestamp: Date;
}

export interface DatapizzaPluginContext {
  /** Plugin configuration */
  config: DatapizzaPluginConfig;

  /** Plugin state */
  state: DatapizzaPluginState;

  /** Available pipeline types */
  availablePipelineTypes: Array<{
    value: string;
    label: string;
    description: string;
    recommendedFor: string[];
  }>;

  /** Data domains */
  dataDomains: Array<{
    value: string;
    label: string;
    description: string;
  }>;

  /** Plugin capabilities */
  capabilities: {
    pipelineProcessing: boolean;
    dataQuerying: boolean;
    caching: boolean;
    monitoring: boolean;
    reporting: boolean;
    externalIntegration: boolean;
  };
}

export interface DatapizzaPluginMethods {
  /** Initialize the plugin */
  initialize: (config?: Partial<DatapizzaPluginConfig>) => Promise<void>;

  /** Update plugin configuration */
  updateConfig: (config: Partial<DatapizzaPluginConfig>) => Promise<void>;

  /** Reset plugin configuration */
  resetConfig: () => Promise<void>;

  /** Run a data pipeline */
  runPipeline: (dataSource: string, pipelineType?: string) => Promise<DatapizzaPipelineResult>;

  /** Process data */
  processData: (data: any, processingType?: string) => Promise<DatapizzaProcessingResult>;

  /** Query data */
  queryData: (query: string, dataSource?: string) => Promise<DatapizzaQueryResult>;

  /** Get pipeline recommendation */
  getPipelineRecommendation: (dataSource: string, context?: string) => Promise<string>;

  /** Detect data domain */
  detectDataDomain: (data: any) => Promise<string | null>;

  /** Check if data is processable */
  isProcessableData: (data: any) => Promise<boolean>;

  /** Clear cache */
  clearCache: () => Promise<void>;

  /** Get plugin statistics */
  getStatistics: () => DatapizzaPluginState['statistics'];

  /** Get operation history */
  getOperationHistory: () => DatapizzaPluginState['operationHistory'];

  /** Clear operation history */
  clearOperationHistory: () => void;

  /** Get plugin status */
  getStatus: () => DatapizzaPluginState['status'];

  /** Get full plugin context */
  getContext: () => DatapizzaPluginContext;
}

export interface DatapizzaPlugin extends DatapizzaPluginMethods {
  /** Plugin metadata */
  metadata: {
    name: string;
    version: string;
    description: string;
    author: string;
    website: string;
  };

  /** React components */
  components?: Partial<{
    ConfigPanel: ComponentType<{ onClose: () => void }>;
    PipelinePanel: ComponentType<{ dataSource: string; onResult: (result: any) => void }>;
    ProcessingPanel: ComponentType<{ data: any; onResult: (result: any) => void }>;
    QueryPanel: ComponentType<{ query: string; onResult: (result: any) => void }>;
    StatusIndicator: ComponentType<{}>;
    PipelineSelector: ComponentType<{ onSelect: (pipeline: string) => void }>;
  }>;

  /** React hooks */
  hooks?: Partial<{
    useDatapizzaConfig: () => [DatapizzaPluginConfig, (config: Partial<DatapizzaPluginConfig>) => void];
    useDatapizzaState: () => DatapizzaPluginState;
    useDatapizzaPipeline: () => (dataSource: string) => Promise<DatapizzaPipelineResult>;
    useDatapizzaProcessing: () => (data: any) => Promise<DatapizzaProcessingResult>;
    useDatapizzaQuery: () => (query: string) => Promise<DatapizzaQueryResult>;
  }>;
}

export interface DatapizzaPluginProps {
  /** Plugin configuration */
  config?: Partial<DatapizzaPluginConfig>;

  /** Callback for configuration changes */
  onConfigChange?: (config: DatapizzaPluginConfig) => void;

  /** Callback for operation results */
  onOperationResult?: (operation: 'pipeline' | 'processing' | 'query', result: any) => void;

  /** Callback for errors */
  onError?: (error: Error) => void;

  /** Callback for status changes */
  onStatusChange?: (status: DatapizzaPluginState['status']) => void;

  /** Children components */
  children?: ReactNode;
}

function getDefaultDatapizzaServerUrl(): string {
  return import.meta.env.VITE_DATAPIZZA_SERVER_URL || 'http://localhost:3000/datapizza';
}

export interface DatapizzaConfigPanelProps {
  /** Initial configuration */
  initialConfig?: Partial<DatapizzaPluginConfig>;

  /** Callback when configuration is saved */
  onSave: (config: DatapizzaPluginConfig) => void;

  /** Callback when configuration is cancelled */
  onCancel: () => void;

  /** Show advanced options */
  showAdvanced?: boolean;
}

export interface DatapizzaPipelinePanelProps {
  /** Data source to process */
  dataSource: string;

  /** Optional initial pipeline type */
  initialPipelineType?: string;

  /** Callback with pipeline result */
  onResult: (result: DatapizzaPipelineResult) => void;

  /** Callback when panel is closed */
  onClose: () => void;

  /** Show debug information */
  showDebug?: boolean;
}

export interface DatapizzaProcessingPanelProps {
  /** Data to process */
  data: any;

  /** Optional processing type */
  processingType?: string;

  /** Callback with processing result */
  onResult: (result: DatapizzaProcessingResult) => void;

  /** Callback when panel is closed */
  onClose: () => void;

  /** Show debug information */
  showDebug?: boolean;
}

export interface DatapizzaQueryPanelProps {
  /** Query to execute */
  query: string;

  /** Optional data source */
  dataSource?: string;

  /** Callback with query result */
  onResult: (result: DatapizzaQueryResult) => void;

  /** Callback when panel is closed */
  onClose: () => void;

  /** Show debug information */
  showDebug?: boolean;
}

export type DatapizzaPipelineType = 'standard' | 'advanced' | 'custom';

export type DatapizzaDataDomain =
  'structured' | 'unstructured' | 'semi_structured' |
  'relational' | 'document' | 'time_series' |
  'graph' | 'geospatial' | 'multimedia' | 'general';

export const DATAPIZZA_PIPELINE_TYPES: Array<{
  value: DatapizzaPipelineType;
  label: string;
  description: string;
  recommendedFor: string[];
}> = [
  {
    value: 'standard',
    label: 'Standard Pipeline',
    description: 'Standard data processing pipeline with basic transformations',
    recommendedFor: ['Simple data', 'Basic processing', 'Quick results']
  },
  {
    value: 'advanced',
    label: 'Advanced Pipeline',
    description: 'Advanced pipeline with complex transformations and optimizations',
    recommendedFor: ['Complex data', 'Large datasets', 'High performance needs']
  },
  {
    value: 'custom',
    label: 'Custom Pipeline',
    description: 'Customizable pipeline for specific use cases',
    recommendedFor: ['Specialized processing', 'Unique requirements', 'Custom workflows']
  }
];

export const DATAPIZZA_DATA_DOMAINS: Array<{
  value: DatapizzaDataDomain;
  label: string;
  description: string;
}> = [
  { value: 'structured', label: 'Structured Data', description: 'Relational databases, CSV files, spreadsheets' },
  { value: 'unstructured', label: 'Unstructured Data', description: 'Text documents, emails, social media posts' },
  { value: 'semi_structured', label: 'Semi-Structured Data', description: 'JSON, XML, HTML documents' },
  { value: 'relational', label: 'Relational Data', description: 'SQL databases, normalized data structures' },
  { value: 'document', label: 'Document Data', description: 'PDFs, Word documents, scanned text' },
  { value: 'time_series', label: 'Time Series Data', description: 'Temporal data, sensor readings, financial data' },
  { value: 'graph', label: 'Graph Data', description: 'Network data, social graphs, knowledge graphs' },
  { value: 'geospatial', label: 'Geospatial Data', description: 'GIS data, location-based information' },
  { value: 'multimedia', label: 'Multimedia Data', description: 'Images, audio, video files' },
  { value: 'general', label: 'General Data', description: 'Mixed or unspecified data types' }
];

export const DEFAULT_DATAPIZZA_CONFIG: DatapizzaPluginConfig = {
  enabled: true,
  serverUrl: getDefaultDatapizzaServerUrl(),
  apiKey: '',
  timeout: 30000,
  pipelineEnabled: true,
  autoDetectDataSources: true,
  defaultPipelineType: 'standard',
  dataProcessingConfig: {
    chunkSize: 1000,
    overlapSize: 200,
    embeddingModel: 'text-embedding-ada-002',
    vectorStoreType: 'qdrant',
    maxParallelProcesses: 4
  },
  agentConfigurations: {
    agent1: {
      enabled: true,
      maxTasks: 10,
      timeout: 60
    },
    agent2: {
      enabled: true,
      parallelExecution: true,
      maxWorkers: 4
    },
    agent3: {
      enabled: true,
      critiqueLevel: 'standard'
    }
  },
  integrateWithWorkflow: true,
  integrateWithKnowledgeGraph: true,
  integrateWithExternalSources: true,
  enableCaching: true,
  cacheTTLSeconds: 3600,
  maxProcessingTime: 300,
  showAdvancedOptions: false,
  showDebugInfo: false,
  theme: 'system'
};
