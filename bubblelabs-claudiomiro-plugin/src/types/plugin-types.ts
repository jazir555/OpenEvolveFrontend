// ClaudieMiro Plugin Types and Interfaces
// Defines all types and interfaces for the BubbleLabs ClaudieMiro plugin

export interface ClaudieMiroPluginConfig {
  /** Enable/disable the plugin */
  enabled: boolean;
  
  /** ClaudieMiro server configuration */
  serverUrl: string;
  apiKey?: string;
  timeout?: number;
  
  /** Autonomous development settings */
  autonomousDevelopmentEnabled: boolean;
  autoDetectDevelopmentTasks: boolean;
  defaultWorkflow: 'standard' | 'advanced' | 'custom';
  
  /** Phase-specific configurations */
  phaseConfigurations: {
    phase1?: {
      enabled: boolean;
      maxTasks?: number;
      timeout?: number;
    };
    phase2?: {
      enabled: boolean;
      parallelExecution?: boolean;
      maxWorkers?: number;
    };
    phase3?: {
      enabled: boolean;
      critiqueLevel?: 'basic' | 'standard' | 'advanced';
    };
    phase4?: {
      enabled: boolean;
      testCoverageThreshold?: number;
    };
    phase5?: {
      enabled: boolean;
      reassemblyStrategy?: 'automatic' | 'manual' | 'hybrid';
    };
    phase6?: {
      enabled: boolean;
      validationLevel?: 'basic' | 'standard' | 'strict';
    };
  };
  
  /** Integration settings */
  integrateWithWorkflow: boolean;
  integrateWithHephaestus: boolean;
  integrateWithMCP: boolean;
  
  /** Performance settings */
  enableCaching: boolean;
  cacheTTLSeconds: number;
  maxOperationTime: number;
  
  /** UI settings */
  showAdvancedOptions: boolean;
  showDebugInfo: boolean;
  theme: 'light' | 'dark' | 'system';
}

export interface ClaudieMiroPluginState extends ClaudieMiroPluginConfig {
  /** Current plugin status */
  status: 'idle' | 'initializing' | 'ready' | 'error' | 'busy';
  
  /** Current operation */
  currentOperation?: {
    type: 'development' | 'critique' | 'testing' | 'validation' | 'configuration';
    startedAt: Date;
    phase?: number;
    progress?: number;
    message?: string;
  };
  
  /** Recent operations history */
  operationHistory: Array<{
    id: string;
    type: string;
    phase?: number;
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
    averageCompletionTime: number;
    lastOperationTime?: Date;
    phasesCompleted: Record<number, number>;
  };
}

export interface ClaudieMiroDevelopmentResult {
  success: boolean;
  taskId: string;
  phase: number;
  description: string;
  artifacts?: Array<{
    type: 'code' | 'documentation' | 'test' | 'configuration';
    path: string;
    content?: string;
    size: number;
  }>;
  confidenceScore: number;
  errors: string[];
  warnings: string[];
  executionTime: number;
  metadata: Record<string, any>;
  timestamp: Date;
}

export interface ClaudieMiroCritiqueResult {
  success: boolean;
  taskId: string;
  phase: number;
  issuesFound: number;
  issuesResolved: number;
  suggestions: string[];
  confidenceScore: number;
  errors: string[];
  warnings: string[];
  executionTime: number;
  metadata: Record<string, any>;
  timestamp: Date;
}

export interface ClaudieMiroTestResult {
  success: boolean;
  taskId: string;
  phase: number;
  testsPassed: number;
  testsFailed: number;
  testCoverage: number;
  confidenceScore: number;
  errors: string[];
  warnings: string[];
  executionTime: number;
  metadata: Record<string, any>;
  timestamp: Date;
}

export interface ClaudieMiroValidationResult {
  success: boolean;
  taskId: string;
  phase: number;
  validationScore: number;
  qualityMetrics: Record<string, number>;
  confidenceScore: number;
  errors: string[];
  warnings: string[];
  executionTime: number;
  metadata: Record<string, any>;
  timestamp: Date;
}

export interface ClaudieMiroPluginContext {
  /** Plugin configuration */
  config: ClaudieMiroPluginConfig;
  
  /** Plugin state */
  state: ClaudieMiroPluginState;
  
  /** Available workflows */
  availableWorkflows: Array<{
    value: string;
    label: string;
    description: string;
    phases: number;
  }>;
  
  /** Phase descriptions */
  phases: Array<{
    phase: number;
    name: string;
    description: string;
    enabled: boolean;
  }>;
  
  /** Plugin capabilities */
  capabilities: {
    autonomousDevelopment: boolean;
    multiPhaseWorkflow: boolean;
    parallelExecution: boolean;
    automatedTesting: boolean;
    qualityValidation: boolean;
    caching: boolean;
    monitoring: boolean;
    reporting: boolean;
  };
}

export interface ClaudieMiroPluginMethods {
  /** Initialize the plugin */
  initialize: (config?: Partial<ClaudieMiroPluginConfig>) => Promise<void>;
  
  /** Update plugin configuration */
  updateConfig: (config: Partial<ClaudieMiroPluginConfig>) => Promise<void>;
  
  /** Reset plugin configuration */
  resetConfig: () => Promise<void>;
  
  /** Run autonomous development workflow */
  runDevelopmentWorkflow: (taskDescription: string, workflow?: string) => Promise<ClaudieMiroDevelopmentResult>;
  
  /** Run specific phase */
  runPhase: (taskId: string, phase: number, context?: any) => Promise<any>;
  
  /** Get phase status */
  getPhaseStatus: (taskId: string, phase: number) => Promise<{
    status: string;
    progress: number;
    result?: any;
    errors?: string[];
  }>;
  
  /** Get task status */
  getTaskStatus: (taskId: string) => Promise<{
    status: string;
    currentPhase: number;
    phasesCompleted: number;
    overallProgress: number;
    results: Record<number, any>;
  }>;
  
  /** Cancel task */
  cancelTask: (taskId: string) => Promise<boolean>;
  
  /** Clear cache */
  clearCache: () => Promise<void>;
  
  /** Get plugin statistics */
  getStatistics: () => ClaudieMiroPluginState['statistics'];
  
  /** Get operation history */
  getOperationHistory: () => ClaudieMiroPluginState['operationHistory'];
  
  /** Clear operation history */
  clearOperationHistory: () => void;
  
  /** Get plugin status */
  getStatus: () => ClaudieMiroPluginState['status'];
  
  /** Get full plugin context */
  getContext: () => ClaudieMiroPluginContext;
}

export interface ClaudieMiroPlugin extends ClaudieMiroPluginMethods {
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
    DevelopmentPanel: React.ComponentType<{ taskDescription: string; onResult: (result: any) => void }>;
    PhaseMonitor: React.ComponentType<{ taskId: string; onClose: () => void }>;
    StatusIndicator: React.ComponentType<{}>;
    WorkflowSelector: React.ComponentType<{ onSelect: (workflow: string) => void }>;
  };
  
  /** React hooks */
  hooks: {
    useClaudieMiroConfig: () => [ClaudieMiroPluginConfig, (config: Partial<ClaudieMiroPluginConfig>) => void];
    useClaudieMiroState: () => ClaudieMiroPluginState;
    useClaudieMiroDevelopment: () => (taskDescription: string, workflow?: string) => Promise<ClaudieMiroDevelopmentResult>;
    useClaudieMiroPhase: () => (taskId: string, phase: number) => Promise<any>;
  };
}

export interface ClaudieMiroPluginProps {
  /** Plugin configuration */
  config?: Partial<ClaudieMiroPluginConfig>;
  
  /** Callback for configuration changes */
  onConfigChange?: (config: ClaudieMiroPluginConfig) => void;
  
  /** Callback for operation results */
  onOperationResult?: (operation: 'development' | 'critique' | 'testing' | 'validation', result: any) => void;
  
  /** Callback for errors */
  onError?: (error: Error) => void;
  
  /** Callback for status changes */
  onStatusChange?: (status: ClaudieMiroPluginState['status']) => void;
  
  /** Children components */
  children?: React.ReactNode;
}

export interface ClaudieMiroConfigPanelProps {
  /** Initial configuration */
  initialConfig?: Partial<ClaudieMiroPluginConfig>;
  
  /** Callback when configuration is saved */
  onSave: (config: ClaudieMiroPluginConfig) => void;
  
  /** Callback when configuration is cancelled */
  onCancel: () => void;
  
  /** Show advanced options */
  showAdvanced?: boolean;
}

export interface ClaudieMiroDevelopmentPanelProps {
  /** Task description for autonomous development */
  taskDescription: string;
  
  /** Optional workflow type */
  workflow?: string;
  
  /** Callback with development result */
  onResult: (result: ClaudieMiroDevelopmentResult) => void;
  
  /** Callback when panel is closed */
  onClose: () => void;
  
  /** Show debug information */
  showDebug?: boolean;
}

export interface ClaudieMiroPhaseMonitorProps {
  /** Task ID to monitor */
  taskId: string;
  
  /** Callback when monitoring is closed */
  onClose: () => void;
  
  /** Show detailed information */
  showDetails?: boolean;
}

export interface ClaudieMiroStatusIndicatorProps {
  /** Custom class name */
  className?: string;
  
  /** Show detailed status */
  showDetails?: boolean;
}

export interface ClaudieMiroWorkflowSelectorProps {
  /** Currently selected workflow */
  selectedWorkflow?: string;
  
  /** Callback when workflow is selected */
  onSelect: (workflow: string) => void;
  
  /** Show descriptions */
  showDescriptions?: boolean;
}

export type ClaudieMiroWorkflow = 'standard' | 'advanced' | 'custom';

export type ClaudieMiroPhase = 1 | 2 | 3 | 4 | 5 | 6;

export const CLAUDIEMIRO_WORKFLOWS: Array<{ 
  value: ClaudieMiroWorkflow; 
  label: string; 
  description: string; 
  phases: number; 
}> = [
  {
    value: 'standard',
    label: 'Standard Workflow',
    description: 'Standard 6-phase autonomous development workflow',
    phases: 6
  },
  {
    value: 'advanced',
    label: 'Advanced Workflow',
    description: 'Advanced workflow with enhanced critique and testing',
    phases: 6
  },
  {
    value: 'custom',
    label: 'Custom Workflow',
    description: 'Customizable workflow with phase-specific configuration',
    phases: 6
  }
];

export const CLAUDIEMIRO_PHASES: Array<{ 
  phase: ClaudieMiroPhase; 
  name: string; 
  description: string; 
}> = [
  {
    phase: 1,
    name: 'Problem Setup',
    description: 'Task decomposition and environment setup'
  },
  {
    phase: 2,
    name: 'Solution Generation',
    description: 'Parallel code generation and implementation'
  },
  {
    phase: 3,
    name: 'Adversarial Critique',
    description: 'Multi-agent critique and refinement'
  },
  {
    phase: 4,
    name: 'Testing & Validation',
    description: 'Automated test generation and execution'
  },
  {
    phase: 5,
    name: 'Reassembly',
    description: 'Component integration and finalization'
  },
  {
    phase: 6,
    name: 'Final Validation',
    description: 'Production-ready validation and deployment'
  }
];

export const DEFAULT_CLAUDIEMIRO_CONFIG: ClaudieMiroPluginConfig = {
  enabled: true,
  serverUrl: 'http://localhost:3000/claudiomiro',
  apiKey: '',
  timeout: 600,
  autonomousDevelopmentEnabled: true,
  autoDetectDevelopmentTasks: true,
  defaultWorkflow: 'standard',
  phaseConfigurations: {
    phase1: { enabled: true, maxTasks: 10, timeout: 300 },
    phase2: { enabled: true, parallelExecution: true, maxWorkers: 4 },
    phase3: { enabled: true, critiqueLevel: 'standard' },
    phase4: { enabled: true, testCoverageThreshold: 80 },
    phase5: { enabled: true, reassemblyStrategy: 'automatic' },
    phase6: { enabled: true, validationLevel: 'standard' }
  },
  integrateWithWorkflow: true,
  integrateWithHephaestus: true,
  integrateWithMCP: true,
  enableCaching: true,
  cacheTTLSeconds: 3600,
  maxOperationTime: 300,
  showAdvancedOptions: false,
  showDebugInfo: false,
  theme: 'system'
};