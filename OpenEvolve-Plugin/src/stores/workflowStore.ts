import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { errorLogger } from '@/utils';

/**
 * Workflow status types
 */
export type WorkflowStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped';

/**
 * Workflow configuration
 */
export interface WorkflowConfig {
  mode: 'standard' | 'quality_diversity' | 'island_model';
  max_iterations: number;
  population_size: number;
  temperature: number;
  top_p: number;
  mutation_rate: number;
  crossover_rate: number;
}

/**
 * Model configuration
 */
export interface ModelConfig {
  provider: string;
  model: string;
  api_key?: string;
  api_base?: string;
}

/**
 * Individual in population
 */
export interface Individual {
  id: string;
  fitness: number;
  content: string;
  generation: number;
}

/**
 * Evolution progress
 */
export interface EvolutionProgress {
  current_iteration: number;
  max_iterations: number;
  percentage: number;
}

/**
 * Workflow execution state
 */
export interface WorkflowExecution {
  evolution_id: string;
  status: WorkflowStatus;
  progress: EvolutionProgress;
  population: Individual[];
  best_individual: Individual | null;
  metrics: {
    average_fitness: number;
    diversity_score: number;
    convergence_rate: number;
  };
  started_at: string;
  updated_at: string;
  websocket_url?: string;
}

/**
 * Workflow state interface
 */
interface WorkflowState {
  // Current workflow
  currentWorkflow: WorkflowExecution | null;

  // Workflow history
  workflows: WorkflowExecution[];

  // Configuration
  config: WorkflowConfig;

  // Models
  models: ModelConfig[];

  // Content
  initialContent: string;

  // UI state
  isLoading: boolean;
  error: string | null;
  activeTab: string;

  // Actions
  setCurrentWorkflow: (workflow: WorkflowExecution | null) => void;
  setWorkflows: (workflows: WorkflowExecution[]) => void;
  addWorkflow: (workflow: WorkflowExecution) => void;
  updateWorkflow: (id: string, updates: Partial<WorkflowExecution>) => void;
  removeWorkflow: (id: string) => void;

  setConfig: (config: Partial<WorkflowConfig>) => void;
  resetConfig: () => void;

  setModels: (models: ModelConfig[]) => void;
  addModel: (model: ModelConfig) => void;
  removeModel: (index: number) => void;

  setInitialContent: (content: string) => void;

  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setActiveTab: (tab: string) => void;

  // Clear state
  clearCurrentWorkflow: () => void;
  reset: () => void;
}

/**
 * Default configuration
 */
const defaultConfig: WorkflowConfig = {
  mode: 'standard',
  max_iterations: 100,
  population_size: 50,
  temperature: 0.7,
  top_p: 0.9,
  mutation_rate: 0.1,
  crossover_rate: 0.8,
};

/**
 * Workflow store
 */
export const useWorkflowStore = create<WorkflowState>()(
  devtools(
    persist(
      (set, get) => ({
        currentWorkflow: null,
        workflows: [],
        config: defaultConfig,
        models: [],
        initialContent: '',
        isLoading: false,
        error: null,
        activeTab: 'overview',

        setCurrentWorkflow: (workflow) => {
          try {
            set({ currentWorkflow: workflow });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'setCurrentWorkflow', additionalData: { workflow } }
            );
          }
        },

        setWorkflows: (workflows) => {
          try {
            set({ workflows });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'setWorkflows', additionalData: { workflows } }
            );
          }
        },

        addWorkflow: (workflow) => {
          try {
            set((state) => ({
              workflows: [workflow, ...state.workflows],
            }));
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'addWorkflow', additionalData: { workflow } }
            );
          }
        },

        updateWorkflow: (id, updates) => {
          try {
            set((state) => ({
              workflows: state.workflows.map((w) =>
                w.evolution_id === id ? { ...w, ...updates } : w
              ),
              currentWorkflow: state.currentWorkflow?.evolution_id === id
                ? { ...state.currentWorkflow, ...updates }
                : state.currentWorkflow,
            }));
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'updateWorkflow', additionalData: { id, updates } }
            );
          }
        },

        removeWorkflow: (id) => {
          try {
            set((state) => ({
              workflows: state.workflows.filter((w) => w.evolution_id !== id),
              currentWorkflow: state.currentWorkflow?.evolution_id === id
                ? null
                : state.currentWorkflow,
            }));
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'removeWorkflow', additionalData: { id } }
            );
          }
        },

        setConfig: (config) => {
          try {
            set((state) => ({
              config: { ...state.config, ...config },
            }));
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'setConfig', additionalData: { config } }
            );
          }
        },

        resetConfig: () => {
          try {
            set({ config: defaultConfig });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'resetConfig' }
            );
          }
        },

        setModels: (models) => {
          try {
            set({ models });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'setModels', additionalData: { models } }
            );
          }
        },

        addModel: (model) => {
          try {
            set((state) => ({
              models: [...state.models, model],
            }));
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'addModel', additionalData: { model } }
            );
          }
        },

        removeModel: (index) => {
          try {
            set((state) => ({
              models: state.models.filter((_, i) => i !== index),
            }));
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'removeModel', additionalData: { index } }
            );
          }
        },

        setInitialContent: (content) => {
          try {
            set({ initialContent: content });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'setInitialContent', additionalData: { content } }
            );
          }
        },

        setLoading: (loading) => {
          try {
            set({ isLoading: loading });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'setLoading', additionalData: { loading } }
            );
          }
        },

        setError: (error) => {
          try {
            set({ error });
          } catch (setErrorError) {
            errorLogger.logError(
              setErrorError instanceof Error ? setErrorError : new Error(String(setErrorError)),
              'error',
              { component: 'WorkflowStore', function: 'setError', additionalData: { error } }
            );
          }
        },

        setActiveTab: (tab) => {
          try {
            set({ activeTab: tab });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'setActiveTab', additionalData: { tab } }
            );
          }
        },

        clearCurrentWorkflow: () => {
          try {
            set({ currentWorkflow: null });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'clearCurrentWorkflow' }
            );
          }
        },

        reset: () => {
          try {
            set({
              currentWorkflow: null,
              initialContent: '',
              error: null,
              isLoading: false,
            });
          } catch (error) {
            errorLogger.logError(
              error instanceof Error ? error : new Error(String(error)),
              'error',
              { component: 'WorkflowStore', function: 'reset' }
            );
          }
        },
      }),
      {
        name: 'workflow-storage',
        partialize: (state) => ({
          config: state.config,
          models: state.models,
          activeTab: state.activeTab,
        }),
      }
    ),
    { name: 'WorkflowStore' }
  )
);
