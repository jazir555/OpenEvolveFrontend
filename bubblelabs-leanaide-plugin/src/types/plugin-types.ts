// LeanAIDE Plugin Types and Interfaces
// Defines all types and interfaces for the BubbleLabs LeanAIDE plugin

export interface LeanAIDEPluginConfig {
  /** Enable/disable the plugin */
  enabled: boolean;
  
  /** LeanAIDE server configuration */
  serverUrl: string;
  apiKey?: string;
  timeout?: number;
  
  /** Autoformalization settings */
  autoformalizationEnabled: boolean;
  autoDetectMathProblems: boolean;
  defaultStrategy: 'DIRECT' | 'MDAP' | 'MAKER' | 'HYBRID' | 'ADAPTIVE';
  
  /** Confidence thresholds */
  minConfidenceForAutoformalization: number;
  minConfidenceForVerification: number;
  
  /** Integration settings */
  integrateWithDecomposition: boolean;
  integrateWithEvolution: boolean;
  integrateWithVerification: boolean;
  
  /** Performance settings */
  enableCaching: boolean;
  cacheTTLSeconds: number;
  maxAutoformalizationTime: number;
  
  /** UI settings */
  showAdvancedOptions: boolean;
  showDebugInfo: boolean;
  theme: 'light' | 'dark' | 'system';
}

export interface LeanAIDEPluginState extends LeanAIDEPluginConfig {
  /** Current plugin status */
  status: 'idle' | 'initializing' | 'ready' | 'error' | 'busy';
  
  /** Current operation */
  currentOperation?: {
    type: 'autoformalization' | 'verification' | 'configuration';
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
    averageConfidence: number;
    lastOperationTime?: Date;
  };
}

export interface LeanAIDEAutoformalizationResult {
  success: boolean;
  originalProblem: string;
  formalizedProblem?: string;
  leanCode?: string;
  confidenceScore: number;
  strategyUsed?: string;
  mathematicalDomain?: string;
  errors: string[];
  warnings: string[];
  executionTime: number;
  metadata: Record<string, any>;
  timestamp: Date;
}

export interface LeanAIDEVerificationResult {
  success: boolean;
  problem: string;
  leanCode: string;
  confidenceScore: number;
  formalProof?: string;
  errors: string[];
  warnings: string[];
  executionTime: number;
  metadata: Record<string, any>;
  timestamp: Date;
}

export interface LeanAIDEPluginContext {
  /** Plugin configuration */
  config: LeanAIDEPluginConfig;
  
  /** Plugin state */
  state: LeanAIDEPluginState;
  
  /** Available strategies */
  availableStrategies: Array<{
    value: string;
    label: string;
    description: string;
    recommendedFor: string[];
  }>;
  
  /** Mathematical domains */
  mathematicalDomains: Array<{
    value: string;
    label: string;
    description: string;
  }>;
  
  /** Plugin capabilities */
  capabilities: {
    autoformalization: boolean;
    verification: boolean;
    caching: boolean;
    monitoring: boolean;
    reporting: boolean;
  };
}

export interface LeanAIDEPluginMethods {
  /** Initialize the plugin */
  initialize: (config?: Partial<LeanAIDEPluginConfig>) => Promise<void>;
  
  /** Update plugin configuration */
  updateConfig: (config: Partial<LeanAIDEPluginConfig>) => Promise<void>;
  
  /** Reset plugin configuration */
  resetConfig: () => Promise<void>;
  
  /** Autoformalize a problem */
  autoformalize: (problem: string, strategy?: string) => Promise<LeanAIDEAutoformalizationResult>;
  
  /** Verify a formalized solution */
  verify: (problem: string, leanCode: string) => Promise<LeanAIDEVerificationResult>;
  
  /** Get strategy recommendation */
  getStrategyRecommendation: (problem: string, context?: string) => Promise<string>;
  
  /** Detect mathematical domain */
  detectMathematicalDomain: (problem: string) => Promise<string | null>;
  
  /** Check if problem is mathematical */
  isMathematicalProblem: (problem: string) => Promise<boolean>;
  
  /** Clear cache */
  clearCache: () => Promise<void>;
  
  /** Get plugin statistics */
  getStatistics: () => LeanAIDEPluginState['statistics'];
  
  /** Get operation history */
  getOperationHistory: () => LeanAIDEPluginState['operationHistory'];
  
  /** Clear operation history */
  clearOperationHistory: () => void;
  
  /** Get plugin status */
  getStatus: () => LeanAIDEPluginState['status'];
  
  /** Get full plugin context */
  getContext: () => LeanAIDEPluginContext;
}

export interface LeanAIDEPlugin extends LeanAIDEPluginMethods {
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
    AutoformalizationPanel: React.ComponentType<{ problem: string; onResult: (result: any) => void }>;
    VerificationPanel: React.ComponentType<{ problem: string; leanCode: string; onResult: (result: any) => void }>;
    StatusIndicator: React.ComponentType<{}>;
    StrategySelector: React.ComponentType<{ onSelect: (strategy: string) => void }>;
  };
  
  /** React hooks */
  hooks: {
    useLeanAIDEConfig: () => [LeanAIDEPluginConfig, (config: Partial<LeanAIDEPluginConfig>) => void];
    useLeanAIDEState: () => LeanAIDEPluginState;
    useLeanAIDEAutoformalization: () => (problem: string) => Promise<LeanAIDEAutoformalizationResult>;
    useLeanAIDEVerification: () => (problem: string, leanCode: string) => Promise<LeanAIDEVerificationResult>;
  };
}

export interface LeanAIDEPluginProps {
  /** Plugin configuration */
  config?: Partial<LeanAIDEPluginConfig>;
  
  /** Callback for configuration changes */
  onConfigChange?: (config: LeanAIDEPluginConfig) => void;
  
  /** Callback for operation results */
  onOperationResult?: (operation: 'autoformalization' | 'verification', result: any) => void;
  
  /** Callback for errors */
  onError?: (error: Error) => void;
  
  /** Callback for status changes */
  onStatusChange?: (status: LeanAIDEPluginState['status']) => void;
  
  /** Children components */
  children?: React.ReactNode;
}

export interface LeanAIDEConfigPanelProps {
  /** Initial configuration */
  initialConfig?: Partial<LeanAIDEPluginConfig>;
  
  /** Callback when configuration is saved */
  onSave: (config: LeanAIDEPluginConfig) => void;
  
  /** Callback when configuration is cancelled */
  onCancel: () => void;
  
  /** Show advanced options */
  showAdvanced?: boolean;
}

export interface LeanAIDEAutoformalizationPanelProps {
  /** Problem statement to autoformalize */
  problem: string;
  
  /** Optional initial strategy */
  initialStrategy?: string;
  
  /** Callback with autoformalization result */
  onResult: (result: LeanAIDEAutoformalizationResult) => void;
  
  /** Callback when panel is closed */
  onClose: () => void;
  
  /** Show debug information */
  showDebug?: boolean;
}

export interface LeanAIDEVerificationPanelProps {
  /** Original problem statement */
  problem: string;
  
  /** Lean code to verify */
  leanCode: string;
  
  /** Callback with verification result */
  onResult: (result: LeanAIDEVerificationResult) => void;
  
  /** Callback when panel is closed */
  onClose: () => void;
  
  /** Show debug information */
  showDebug?: boolean;
}

export interface LeanAIDEStrategySelectorProps {
  /** Currently selected strategy */
  selectedStrategy?: string;
  
  /** Callback when strategy is selected */
  onSelect: (strategy: string) => void;
  
  /** Problem context for recommendation */
  problemContext?: string;
  
  /** Show descriptions */
  showDescriptions?: boolean;
}

export interface LeanAIDEStatusIndicatorProps {
  /** Custom class name */
  className?: string;
  
  /** Show detailed status */
  showDetails?: boolean;
}

export type LeanAIDEStrategy = 'DIRECT' | 'MDAP' | 'MAKER' | 'HYBRID' | 'ADAPTIVE';

export type LeanAIDEMathematicalDomain = 
  'algebra' | 'analysis' | 'logic' | 'number_theory' | 
  'combinatorics' | 'geometry' | 'topology' | 'category_theory' | 
  'linear_algebra' | 'calculus' | 'probability' | 'set_theory' | 'general';

export const LEANAIDE_STRATEGIES: Array<{ 
  value: LeanAIDEStrategy; 
  label: string; 
  description: string; 
  recommendedFor: string[]; 
}> = [
  {
    value: 'DIRECT',
    label: 'Direct Translation',
    description: 'Direct translation using LeanAIDE core capabilities',
    recommendedFor: ['Simple problems', 'Logic', 'Basic algebra']
  },
  {
    value: 'MDAP',
    label: 'Multi-Agent Generation',
    description: 'Multi-agent generation with aggregation (MDAP)',
    recommendedFor: ['Complex problems', 'Algebra', 'Analysis']
  },
  {
    value: 'MAKER',
    label: 'Voting-Based Refinement',
    description: 'Voting-based refinement (MAKER)',
    recommendedFor: ['Proof refinement', 'Verification', 'High confidence needed']
  },
  {
    value: 'HYBRID',
    label: 'Hybrid Approach',
    description: 'Combines MDAP and MAKER for optimal results',
    recommendedFor: ['Complex proofs', 'High importance problems']
  },
  {
    value: 'ADAPTIVE',
    label: 'Adaptive Selection',
    description: 'Automatically selects the best strategy',
    recommendedFor: ['General use', 'Unknown problem types']
  }
];

export const MATHEMATICAL_DOMAINS: Array<{ 
  value: LeanAIDEMathematicalDomain; 
  label: string; 
  description: string; 
}> = [
  { value: 'algebra', label: 'Algebra', description: 'Abstract algebra, group theory, ring theory' },
  { value: 'analysis', label: 'Analysis', description: 'Real analysis, complex analysis, measure theory' },
  { value: 'logic', label: 'Logic', description: 'Mathematical logic, proof theory, model theory' },
  { value: 'number_theory', label: 'Number Theory', description: 'Elementary and algebraic number theory' },
  { value: 'combinatorics', label: 'Combinatorics', description: 'Enumerative combinatorics, graph theory' },
  { value: 'geometry', label: 'Geometry', description: 'Euclidean geometry, differential geometry' },
  { value: 'topology', label: 'Topology', description: 'Point-set topology, algebraic topology' },
  { value: 'category_theory', label: 'Category Theory', description: 'Categories, functors, natural transformations' },
  { value: 'linear_algebra', label: 'Linear Algebra', description: 'Vector spaces, matrices, linear transformations' },
  { value: 'calculus', label: 'Calculus', description: 'Differential and integral calculus' },
  { value: 'probability', label: 'Probability', description: 'Probability theory, stochastic processes' },
  { value: 'set_theory', label: 'Set Theory', description: 'ZFC, axiomatic set theory' },
  { value: 'general', label: 'General', description: 'General or cross-domain mathematics' }
];

export const DEFAULT_LEANAIDE_CONFIG: LeanAIDEPluginConfig = {
  enabled: true,
  serverUrl: 'http://localhost:3000/leanaide',
  apiKey: '',
  timeout: 300,
  autoformalizationEnabled: true,
  autoDetectMathProblems: true,
  defaultStrategy: 'ADAPTIVE',
  minConfidenceForAutoformalization: 0.6,
  minConfidenceForVerification: 0.8,
  integrateWithDecomposition: true,
  integrateWithEvolution: true,
  integrateWithVerification: true,
  enableCaching: true,
  cacheTTLSeconds: 3600,
  maxAutoformalizationTime: 120,
  showAdvancedOptions: false,
  showDebugInfo: false,
  theme: 'system'
};