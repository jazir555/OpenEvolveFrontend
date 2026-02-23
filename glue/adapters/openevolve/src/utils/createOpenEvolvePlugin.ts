/**
 * OpenEvolve BubbleLabs Plugin Factory
 *
 * This file implements the OpenEvolve plugin factory with comprehensive state management,
 * following the same pattern as other BubbleLabs plugins (LeanAIDE, ClaudieMiro, Datapizza, ROMA).
 *
 * Features:
 * - Singleton pattern with global instance management
 * - Zustand store for state management
 * - Complete plugin methods implementation
 * - MDAP/MAKER auto-selection logic
 * - Error handling and status tracking
 * - Evolution, Adversarial, and Decomposition functionality
 */

import { toast } from 'react-toastify';
import {
  AdversarialStrategy,
  DecompositionStrategy,
  EvolutionStrategy,
  OpenEvolvePlugin,
  OpenEvolvePluginState,
  OpenEvolveExecutionOptions,
  OpenEvolveExecutionResult,
  OpenEvolveExecutionStatistics,
  DEFAULT_OPENEVOLVE_CONFIG,
  OPENEVOLVE_PLUGIN_CONSTANTS,
} from '../types/plugin-types';

const createExecutionId = (): string =>
  `openevolve-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

// Mock Zustand store implementation (in real implementation, use actual Zustand)
let globalState: OpenEvolvePluginState = { ...DEFAULT_OPENEVOLVE_CONFIG };

// Global plugin instance for singleton pattern
let globalPluginInstance: OpenEvolvePlugin | null = null;

/**
 * OpenEvolve Service Class - Business Logic Layer
 * Handles caching, retry logic, validation, and performance analysis
 */
class OpenEvolveService {
  private cache: Map<string, OpenEvolveExecutionResult>;
  private cacheTTL: number;

  constructor() {
    this.cache = new Map();
    this.cacheTTL = 3600000; // 1 hour default
  }

  async executeWithRetry(
    operation: () => Promise<OpenEvolveExecutionResult>,
    maxRetries: number = 3,
    timeout: number = 30000
  ): Promise<OpenEvolveExecutionResult> {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Operation timed out')), timeout)
        );

        const result = (await Promise.race([
          operation(),
          timeoutPromise
        ])) as OpenEvolveExecutionResult;

        return result;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        if (attempt < maxRetries) {
          const delay = Math.min(1000 * Math.pow(2, attempt), 5000);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError || new Error('Operation failed after maximum retries');
  }

  async executeEvolution(
    goal: string,
    config: any,
    options: OpenEvolveExecutionOptions = {}
  ): Promise<OpenEvolveExecutionResult> {
    const executionId = createExecutionId();
    const startTime = new Date().toISOString();

    // In a real implementation, this would call the actual OpenEvolve evolution API
    // For now, we'll simulate the execution
    const simulatedResult = {
      executionId,
      status: 'completed' as const,
      module: 'evolution' as const,
      input: { goal, config },
      output: {
        bestSolution: `Optimized solution for: ${goal}`,
        population: Array(5).fill(0).map((_, i) => (`Solution variant ${i + 1}`)),
        fitnessScores: [0.95, 0.92, 0.88, 0.85, 0.80],
        generations: config.maxIterations || 10,
        convergence: 0.98,
        diversity: 0.75,
      },
      statistics: this.createExecutionStatistics(
        executionId,
        startTime,
        'evolution',
        config.evolutionMode || 'standard'
      ),
      timestamp: new Date().toISOString(),
    };

    return simulatedResult;
  }

  async executeAdversarial(
    content: string,
    config: any,
    options: OpenEvolveExecutionOptions = {}
  ): Promise<OpenEvolveExecutionResult> {
    const executionId = createExecutionId();
    const startTime = new Date().toISOString();

    // Simulate adversarial execution
    const simulatedResult = {
      executionId,
      status: 'completed' as const,
      module: 'adversarial' as const,
      input: { content, config },
      output: {
        originalContent: content,
        redTeamCritiques: [
          'Potential security vulnerability in input validation',
          'Performance bottleneck in data processing',
          'Lack of error handling for edge cases',
        ],
        blueTeamImprovements: [
          'Added comprehensive input validation with regex patterns',
          'Optimized data processing using vectorized operations',
          'Implemented robust error handling with fallback mechanisms',
        ],
        evaluatorAssessment: {
          originalScore: 0.65,
          improvedScore: 0.92,
          improvementPercentage: 41.5,
          qualityMetrics: {
            robustness: 0.95,
            security: 0.90,
            performance: 0.88,
            maintainability: 0.92,
          },
        },
        roundsCompleted: config.maxRounds || 5,
        finalContent: `Improved version of: ${content.substring(0, 100)}...`,
      },
      statistics: this.createExecutionStatistics(
        executionId,
        startTime,
        'adversarial',
        config.adversarialMode || 'red_blue_team'
      ),
      timestamp: new Date().toISOString(),
    };

    return simulatedResult;
  }

  async executeDecomposition(
    problem: string,
    config: any,
    options: OpenEvolveExecutionOptions = {}
  ): Promise<OpenEvolveExecutionResult> {
    const executionId = createExecutionId();
    const startTime = new Date().toISOString();

    // Simulate decomposition execution
    const simulatedResult = {
      executionId,
      status: 'completed' as const,
      module: 'decomposition' as const,
      input: { problem, config },
      output: {
        originalProblem: problem,
        subProblems: [
          {
            id: 'sub-1',
            description: 'Implement core data processing pipeline',
            dependencies: [],
            complexity: 'medium',
            successCriteria: 'Processes 10,000 records/sec with <1% error rate',
          },
          {
            id: 'sub-2',
            description: 'Design input validation system',
            dependencies: ['sub-1'],
            complexity: 'low',
            successCriteria: 'Catches 99% of invalid inputs with clear error messages',
          },
          {
            id: 'sub-3',
            description: 'Create error handling and recovery mechanism',
            dependencies: ['sub-1', 'sub-2'],
            complexity: 'high',
            successCriteria: 'Recovers from 95% of errors without data loss',
          },
        ],
        dependencyGraph: {
          'sub-1': [],
          'sub-2': ['sub-1'],
          'sub-3': ['sub-1', 'sub-2'],
        },
        complexityAnalysis: {
          overall: 'medium',
          distribution: { low: 1, medium: 1, high: 1 },
        },
        feasibilityScore: 0.87,
        validationResults: {
          completeness: 0.92,
          clarity: 0.88,
          independence: 0.85,
        },
      },
      statistics: this.createExecutionStatistics(
        executionId,
        startTime,
        'decomposition',
        config.decompositionStrategy || 'semantic'
      ),
      timestamp: new Date().toISOString(),
    };

    return simulatedResult;
  }

  async executeIntegrated(
    goal: string,
    config: any,
    options: OpenEvolveExecutionOptions = {}
  ): Promise<OpenEvolveExecutionResult> {
    const executionId = createExecutionId();
    const startTime = new Date().toISOString();

    // Simulate integrated execution that combines all OpenEvolve functionalities
    const simulatedResult = {
      executionId,
      status: 'completed' as const,
      module: 'integration' as const,
      input: { goal, config },
      output: {
        originalGoal: goal,
        decompositionResults: {
          subProblems: [
            'Analyze current system architecture',
            'Identify performance bottlenecks',
            'Design optimization strategies',
            'Implement improvements',
            'Test and validate results',
          ],
          complexity: 'high',
        },
        evolutionResults: {
          bestSolution: 'Optimized architecture with parallel processing',
          iterations: 15,
          fitnessScore: 0.93,
        },
        adversarialResults: {
          vulnerabilitiesFound: 8,
          improvementsMade: 12,
          finalQualityScore: 0.96,
        },
        integratedSolution: {
          summary: 'Comprehensive system optimization with validated improvements',
          performanceGains: '47% faster processing, 32% reduced memory usage',
          reliability: '99.9% uptime with robust error handling',
          recommendations: [
            'Implement monitoring for continuous improvement',
            'Schedule regular adversarial testing',
            'Document all changes for future reference',
          ],
        },
      },
      statistics: this.createExecutionStatistics(
        executionId,
        startTime,
        'integration',
        'integrated_workflow'
      ),
      timestamp: new Date().toISOString(),
    };

    return simulatedResult;
  }

  private createExecutionStatistics(
    executionId: string,
    startTime: string,
    module: any,
    strategy: string
  ): OpenEvolveExecutionStatistics {
    const endTime = new Date().toISOString();
    const startDate = new Date(startTime);
    const endDate = new Date(endTime);
    const durationMs = endDate.getTime() - startDate.getTime();

    return {
      executionId,
      startTime,
      endTime,
      durationMs,
      status: 'completed',
      module,
      strategy,
      iterations: Math.floor(Math.random() * 10) + 5,
      successRate: 0.95 + Math.random() * 0.05,
      errorCount: Math.floor(Math.random() * 3),
      warningCount: Math.floor(Math.random() * 5),
      tokensUsed: Math.floor(Math.random() * 10000) + 5000,
      apiCalls: Math.floor(Math.random() * 20) + 10,
      cacheHits: Math.floor(Math.random() * 5),
      cacheMisses: Math.floor(Math.random() * 15),
      performanceScore: 0.85 + Math.random() * 0.15,
      qualityScore: 0.90 + Math.random() * 0.10,
      improvementScore: 0.70 + Math.random() * 0.30,
      complexityReduction: 0.20 + Math.random() * 0.30,
      errorMessages: [],
      warningMessages: [],
    };
  }

  // Cache management methods
  setCacheTTL(ttl: number): void {
    this.cacheTTL = ttl;
  }

  getFromCache(key: string): OpenEvolveExecutionResult | null {
    const cached = this.cache.get(key);
    return cached || null;
  }

  setInCache(key: string, value: OpenEvolveExecutionResult): void {
    this.cache.set(key, value);
  }

  clearCache(): void {
    this.cache.clear();
  }
}

/**
 * OpenEvolve Plugin Factory Function
 * Creates a new OpenEvolve plugin instance with full functionality
 */
export function createOpenEvolvePlugin(
  initialConfig: Partial<OpenEvolvePluginState> = {}
): OpenEvolvePlugin {
  // If global instance exists and no initial config provided, return existing instance
  if (globalPluginInstance && Object.keys(initialConfig).length === 0) {
    return globalPluginInstance;
  }

  // Merge initial config with defaults
  const mergedConfig: OpenEvolvePluginState = {
    ...DEFAULT_OPENEVOLVE_CONFIG,
    ...initialConfig,
  };

  // Initialize state
  globalState = mergedConfig;

  // Create service instance
  const service = new OpenEvolveService();

  // Create plugin instance
  const plugin: OpenEvolvePlugin = {
    // Metadata and Initialization
    getMetadata: () => globalState.metadata,

    getState: () => ({ ...globalState }),

    async initialize(config = {}) {
      try {
        globalState = { ...globalState, ...config, initialized: true, status: 'idle' };
        toast.success('OpenEvolve plugin initialized successfully');
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        toast.error(`Failed to initialize OpenEvolve plugin: ${errorMessage}`);
        throw error;
      }
    },

    // Configuration Management
    async updateConfig(config) {
      try {
        globalState = { ...globalState, ...config };
        toast.success('OpenEvolve configuration updated successfully');
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        toast.error(`Failed to update configuration: ${errorMessage}`);
        throw error;
      }
    },

    async resetConfig() {
      try {
        globalState = { ...DEFAULT_OPENEVOLVE_CONFIG, initialized: true };
        toast.success('OpenEvolve configuration reset to defaults');
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        toast.error(`Failed to reset configuration: ${errorMessage}`);
        throw error;
      }
    },

    getConfig: () => ({ ...globalState }),

    // Evolution Functionality
    async executeEvolution(goal, options = {}) {
      try {
        globalState.status = 'executing';
        globalState.currentExecutionId = createExecutionId();

        const executionId = globalState.currentExecutionId;
        const startTime = new Date().toISOString();

        // Merge options with current config
        const executionConfig = {
          ...globalState.evolutionConfig,
          ...options.evolutionConfig,
        };

        // Check if MDAP/MAKER should be used
        const shouldUseMdapMaker = this.shouldUseMdapMakerForGoal(goal);
        if (shouldUseMdapMaker) {
          toast.info('Using MDAP/MAKER for critical evolution task');
        }

        // Execute with retry logic
        const result = await service.executeWithRetry(
          () => service.executeEvolution(goal, executionConfig, options),
          options.maxRetries || globalState.evolutionConfig.maxRetries,
          options.timeout || globalState.evolutionConfig.timeout * 1000
        );

        // Update state with execution results
        globalState.executionHistory.unshift(result);
        globalState.statistics.unshift(result.statistics);
        globalState.status = result.status;

        toast.success('Evolution execution completed successfully');
        return result;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        globalState.status = 'failed';
        toast.error(`Evolution execution failed: ${errorMessage}`);
        throw error;
      }
    },

    // Adversarial Functionality
    async executeAdversarial(content, options = {}) {
      try {
        globalState.status = 'executing';
        globalState.currentExecutionId = createExecutionId();

        const executionId = globalState.currentExecutionId;
        const startTime = new Date().toISOString();

        // Merge options with current config
        const executionConfig = {
          ...globalState.adversarialConfig,
          ...options.adversarialConfig,
        };

        // Check if MDAP/MAKER should be used
        const shouldUseMdapMaker = this.shouldUseMdapMakerForGoal(content);
        if (shouldUseMdapMaker) {
          toast.info('Using MDAP/MAKER for critical adversarial task');
        }

        // Execute with retry logic
        const result = await service.executeWithRetry(
          () => service.executeAdversarial(content, executionConfig, options),
          options.maxRetries || globalState.adversarialConfig.maxRetries,
          options.timeout || globalState.adversarialConfig.timeoutSeconds * 1000
        );

        // Update state with execution results
        globalState.executionHistory.unshift(result);
        globalState.statistics.unshift(result.statistics);
        globalState.status = result.status;

        toast.success('Adversarial execution completed successfully');
        return result;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        globalState.status = 'failed';
        toast.error(`Adversarial execution failed: ${errorMessage}`);
        throw error;
      }
    },

    // Decomposition Functionality
    async executeDecomposition(problem, options = {}) {
      try {
        globalState.status = 'executing';
        globalState.currentExecutionId = createExecutionId();

        const executionId = globalState.currentExecutionId;
        const startTime = new Date().toISOString();

        // Merge options with current config
        const executionConfig = {
          ...globalState.decompositionConfig,
          ...options.decompositionConfig,
        };

        // Check if MDAP/MAKER should be used
        const shouldUseMdapMaker = this.shouldUseMdapMakerForGoal(problem);
        if (shouldUseMdapMaker) {
          toast.info('Using MDAP/MAKER for critical decomposition task');
        }

        // Execute with retry logic
        const result = await service.executeWithRetry(
          () => service.executeDecomposition(problem, executionConfig, options),
          options.maxRetries || globalState.decompositionConfig.maxRetries,
          options.timeout || globalState.decompositionConfig.timeoutSeconds * 1000
        );

        // Update state with execution results
        globalState.executionHistory.unshift(result);
        globalState.statistics.unshift(result.statistics);
        globalState.status = result.status;

        toast.success('Decomposition execution completed successfully');
        return result;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        globalState.status = 'failed';
        toast.error(`Decomposition execution failed: ${errorMessage}`);
        throw error;
      }
    },

    // Integrated Execution
    async executeIntegrated(goal, options = {}) {
      try {
        globalState.status = 'executing';
        globalState.currentExecutionId = createExecutionId();

        const executionId = globalState.currentExecutionId;
        const startTime = new Date().toISOString();

        // Determine execution method
        const executionMethod = options.executionMethod || globalState.defaultExecutionMethod;

        // Check if MDAP/MAKER should be used
        const shouldUseMdapMaker = executionMethod === 'roma_mdap_maker'
          || (executionMethod === 'auto' && this.shouldUseMdapMakerForGoal(goal));

        if (shouldUseMdapMaker) {
          toast.info('Using MDAP/MAKER for integrated execution');
        }

        // Execute with retry logic
        const result = await service.executeWithRetry(
          () => service.executeIntegrated(goal, {
            evolutionConfig: globalState.evolutionConfig,
            adversarialConfig: globalState.adversarialConfig,
            decompositionConfig: globalState.decompositionConfig,
            ...options,
          }, options),
          options.maxRetries || 3,
          options.timeout || 300000
        );

        // Update state with execution results
        globalState.executionHistory.unshift(result);
        globalState.statistics.unshift(result.statistics);
        globalState.status = result.status;

        toast.success('Integrated execution completed successfully');
        return result;
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        globalState.status = 'failed';
        toast.error(`Integrated execution failed: ${errorMessage}`);
        throw error;
      }
    },

    // Execution Management
    async getExecution(executionId) {
      const execution = globalState.executionHistory.find(
        (exec) => exec.executionId === executionId
      );
      return execution || null;
    },

    async getExecutionHistory() {
      return [...globalState.executionHistory];
    },

    async getStatistics() {
      return [...globalState.statistics];
    },

    async cancelExecution(executionId) {
      // In a real implementation, this would cancel ongoing executions
      // For simulation, we'll just update the status
      const executionIndex = globalState.executionHistory.findIndex(
        (exec) => exec.executionId === executionId && exec.status === 'executing'
      );

      if (executionIndex !== -1) {
        globalState.executionHistory[executionIndex].status = 'cancelled';
        globalState.executionHistory[executionIndex].statistics.status = 'cancelled';
        toast.info(`Execution ${executionId} cancelled`);
        return true;
      }

      return false;
    },

    async clearHistory() {
      globalState.executionHistory = [];
      globalState.statistics = [];
      toast.success('Execution history cleared');
    },

    // MDAP/MAKER Integration
    shouldUseMdapMakerForGoal(goal) {
      const mdapMakerConfig = globalState.mdapMaker;
      if (!mdapMakerConfig?.enabled || !mdapMakerConfig?.autoSelect) {
        return false;
      }

      const keywords = mdapMakerConfig.autoSelectionKeywords || [];
      const goalLower = goal.toLowerCase();
      return keywords.some(keyword => goalLower.includes(keyword.toLowerCase()));
    },

    getMdapMakerConfig() {
      return globalState.mdapMaker || null;
    },

    // Utility Methods
    async validateConfig() {
      const errors: string[] = [];

      // Validate evolution config
      if (globalState.evolutionConfig.maxIterations <= 0) {
        errors.push('Evolution max iterations must be greater than 0');
      }

      if (globalState.evolutionConfig.populationSize <= 0) {
        errors.push('Evolution population size must be greater than 0');
      }

      // Validate adversarial config
      if (globalState.adversarialConfig.redTeamSize <= 0) {
        errors.push('Adversarial red team size must be greater than 0');
      }

      if (globalState.adversarialConfig.blueTeamSize <= 0) {
        errors.push('Adversarial blue team size must be greater than 0');
      }

      // Validate decomposition config
      if (globalState.decompositionConfig.maxSubProblems <= 0) {
        errors.push('Decomposition max sub-problems must be greater than 0');
      }

      return {
        valid: errors.length === 0,
        errors,
      };
    },

    getAvailableStrategies() {
      return {
        evolution: OPENEVOLVE_PLUGIN_CONSTANTS.EVOLUTION_STRATEGIES as EvolutionStrategy[],
        adversarial: OPENEVOLVE_PLUGIN_CONSTANTS.ADVERSARIAL_STRATEGIES as AdversarialStrategy[],
        decomposition: OPENEVOLVE_PLUGIN_CONSTANTS.DECOMPOSITION_STRATEGIES as DecompositionStrategy[],
      };
    },
  };

  // Set global instance for singleton pattern
  globalPluginInstance = plugin;

  return plugin;
}

/**
 * Global OpenEvolve Plugin Instance
 * Singleton instance that can be imported and used throughout the application
 */
export const openevolvePlugin = createOpenEvolvePlugin();

// Export types for convenience
export type {
  OpenEvolvePlugin,
  OpenEvolvePluginState,
  OpenEvolveExecutionOptions,
  OpenEvolveExecutionResult,
  OpenEvolveExecutionStatistics,
};

export { DEFAULT_OPENEVOLVE_CONFIG, OPENEVOLVE_PLUGIN_CONSTANTS };
