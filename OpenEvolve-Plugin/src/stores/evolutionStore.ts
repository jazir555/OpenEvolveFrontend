import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { WorkflowExecution, WorkflowStatus } from './workflowStore';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';
import { errorLogger } from '@/utils';

/**
 * Adversarial testing round result
 */
export interface AdversarialRound {
  round: number;
  attack_mode: string;
  success: boolean;
  vulnerability?: string;
  payload: string;
  patch?: string;
  patch_approved?: boolean;
}

/**
 * Adversarial testing state
 */
export interface AdversarialTest {
  test_id: string;
  status: WorkflowStatus;
  current_round: number;
  total_rounds: number;
  red_team_results: AdversarialRound[];
  blue_team_results: AdversarialRound[];
  vulnerabilities_found: number;
  patches_generated: number;
  patches_approved: number;
  created_at: string;
  updated_at: string;
  websocket_url?: string;
}

/**
 * Evolution store state
 */
interface EvolutionState {
  // Evolutionary optimization
  evolution: WorkflowExecution | null;
  evolutions: WorkflowExecution[];

  // Adversarial testing
  adversarialTest: AdversarialTest | null;
  adversarialTests: AdversarialTest[];

  // Content being optimized
  content: string;

  // Configuration
  attackModes: string[];

  // UI state
  isLoading: boolean;
  isEvolutionRunning: boolean;
  isAdversarialRunning: boolean;
  error: string | null;

  // Actions - Evolution
  setEvolution: (evolution: WorkflowExecution | null) => void;
  setEvolutions: (evolutions: WorkflowExecution[]) => void;
  addEvolution: (evolution: WorkflowExecution) => void;
  updateEvolution: (id: string, updates: Partial<WorkflowExecution>) => void;

  // Actions - Adversarial
  setAdversarialTest: (test: AdversarialTest | null) => void;
  setAdversarialTests: (tests: AdversarialTest[]) => void;
  addAdversarialTest: (test: AdversarialTest) => void;
  updateAdversarialTest: (id: string, updates: Partial<AdversarialTest>) => void;
  approvePatch: (testId: string, round: number, approved: boolean, feedback?: string) => void;

  // Actions - Content
  setContent: (content: string) => void;

  // Actions - Configuration
  setAttackModes: (modes: string[]) => void;

  // Actions - State management
  setIsEvolutionRunning: (running: boolean) => void;
  setIsAdversarialRunning: (running: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // Clear
  clearEvolution: () => void;
  clearAdversarial: () => void;
  reset: () => void;
}

/**
 * Evolution store
 */
export const useEvolutionStore = create<EvolutionState>()(
  devtools(
    (set, get) => ({
      evolution: null,
      evolutions: [],
      adversarialTest: null,
      adversarialTests: [],
      content: '',
      attackModes: [
        'prompt_injection',
        'jailbreak',
        'adversarial_example',
        'data_extraction',
        'model_inversion',
      ],
      isLoading: false,
      isEvolutionRunning: false,
      isAdversarialRunning: false,
      error: null,

      setEvolution: (evolution) => {
        gracefulErrorHandler.executeWithErrorHandling(() => {
          set({ evolution });
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'useEvolutionStore',
            function: 'setEvolution',
            operation: 'SET_EVOLUTION',
          }
        }).catch(error => {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'useEvolutionStore', function: 'setEvolution' }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set evolution' });
        });
      },

      setEvolutions: (evolutions) => {
        gracefulErrorHandler.executeWithErrorHandling(() => {
          set({ evolutions });
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'useEvolutionStore',
            function: 'setEvolutions',
            operation: 'SET_EVOLUTIONS',
          }
        }).catch(error => {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'useEvolutionStore', function: 'setEvolutions' }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set evolutions' });
        });
      },

      addEvolution: (evolution) => {
        gracefulErrorHandler.executeWithErrorHandling(() => {
          set((state) => ({
            evolutions: [evolution, ...state.evolutions],
          }));
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'useEvolutionStore',
            function: 'addEvolution',
            operation: 'ADD_EVOLUTION',
          }
        }).catch(error => {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'useEvolutionStore', function: 'addEvolution' }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to add evolution' });
        });
      },

      updateEvolution: (id, updates) => {
        gracefulErrorHandler.executeWithErrorHandling(() => {
          set((state) => ({
            evolutions: state.evolutions.map((e) =>
              e.evolution_id === id ? { ...e, ...updates } : e
            ),
            evolution: state.evolution?.evolution_id === id
              ? { ...state.evolution, ...updates } : state.evolution,
          }));
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'useEvolutionStore',
            function: 'updateEvolution',
            operation: 'UPDATE_EVOLUTION',
            additionalData: { id }
          }
        }).catch(error => {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'useEvolutionStore', function: 'updateEvolution' }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to update evolution' });
        });
      },

      setAdversarialTest: (test) => {
        gracefulErrorHandler.executeWithErrorHandling(() => {
          set({ adversarialTest: test });
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'useEvolutionStore',
            function: 'setAdversarialTest',
            operation: 'SET_ADVERSARIAL_TEST',
          }
        }).catch(error => {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'useEvolutionStore', function: 'setAdversarialTest' }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set adversarial test' });
        });
      },

      setAdversarialTests: (tests) => {
        try {
          set({ adversarialTests: tests });
        } catch (error) {
          errorLogger.logError('Error setting adversarial tests', 'error', {
            component: 'evolutionStore',
            function: 'setAdversarialTests',
            additionalData: { error, tests }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to set adversarial tests' });
        }
      },

      addAdversarialTest: (test) => {
        try {
          set((state) => ({
            adversarialTests: [test, ...state.adversarialTests],
          }));
        } catch (error) {
          errorLogger.logError('Error adding adversarial test', 'error', {
            component: 'evolutionStore',
            function: 'addAdversarialTest',
            additionalData: { error, test }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to add adversarial test' });
        }
      },

      updateAdversarialTest: (id, updates) => {
        try {
          set((state) => ({
            adversarialTests: state.adversarialTests.map((t) =>
              t.test_id === id ? { ...t, ...updates } : t
            ),
            adversarialTest: state.adversarialTest?.test_id === id
              ? { ...state.adversarialTest, ...updates } : state.adversarialTest,
          }));
        } catch (error) {
          errorLogger.logError('Error updating adversarial test', 'error', {
            component: 'evolutionStore',
            function: 'updateAdversarialTest',
            additionalData: { error, id, updates }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to update adversarial test' });
        }
      },

      approvePatch: (testId, round, approved, feedback) => {
        gracefulErrorHandler.executeWithErrorHandling(() => {
          set((state) => ({
            adversarialTests: state.adversarialTests.map((test) => {
              if (test.test_id === testId) {
                const updatedBlueTeam = test.blue_team_results.map((r) =>
                  r.round === round
                    ? { ...r, patch_approved: approved }
                    : r
                );
                return { ...test, blue_team_results: updatedBlueTeam };
              }
              return test;
            }),
            adversarialTest: state.adversarialTest?.test_id === testId
              ? {
                  ...state.adversarialTest,
                  blue_team_results: state.adversarialTest.blue_team_results.map((r) =>
                    r.round === round
                      ? { ...r, patch_approved: approved }
                      : r
                  ),
                  patches_approved: approved
                    ? state.adversarialTest.patches_approved + 1
                    : state.adversarialTest.patches_approved,
                }
              : state.adversarialTest,
          }));
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'useEvolutionStore',
            function: 'approvePatch',
            operation: 'APPROVE_PATCH',
            additionalData: { testId, round, approved }
          }
        }).catch(error => {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'useEvolutionStore', function: 'approvePatch' }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to approve patch' });
        });
      },

      setContent: (content) => {
        try {
          set({ content });
        } catch (error) {
          errorLogger.logError('Error setting content', 'error', {
            component: 'evolutionStore',
            function: 'setContent',
            additionalData: { error, content }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to set content' });
        }
      },

      setAttackModes: (modes) => {
        try {
          set({ attackModes: modes });
        } catch (error) {
          errorLogger.logError('Error setting attack modes', 'error', {
            component: 'evolutionStore',
            function: 'setAttackModes',
            additionalData: { error, modes }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to set attack modes' });
        }
      },

      setIsEvolutionRunning: (running) => {
        try {
          set({ isEvolutionRunning: running });
        } catch (error) {
          errorLogger.logError('Error setting evolution running state', 'error', {
            component: 'evolutionStore',
            function: 'setEvolutionRunning',
            additionalData: { error, running }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to set evolution running state' });
        }
      },

      setIsAdversarialRunning: (running) => {
        try {
          set({ isAdversarialRunning: running });
        } catch (error) {
          errorLogger.logError('Error setting adversarial running state', 'error', {
            component: 'evolutionStore',
            function: 'setAdversarialRunning',
            additionalData: { error, running }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to set adversarial running state' });
        }
      },

      setLoading: (loading) => {
        try {
          set({ isLoading: loading });
        } catch (error) {
          errorLogger.logError('Error setting loading state', 'error', {
            component: 'evolutionStore',
            function: 'setLoading',
            additionalData: { error, loading }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to set loading state' });
        }
      },

      setError: (error) => {
        gracefulErrorHandler.executeWithErrorHandling(() => {
          set({ error: error });
        }, {
          strategy: 'retry',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          context: {
            component: 'useEvolutionStore',
            function: 'setError',
            operation: 'SET_ERROR',
          }
        }).catch(setErrorError => {
          errorLogger.logError(
            setErrorError instanceof Error ? setErrorError : new Error(String(setErrorError)),
            'error',
            { component: 'useEvolutionStore', function: 'setError' }
          );
        });
      },

      clearEvolution: () => {
        try {
          set({
            evolution: null,
            isEvolutionRunning: false,
            error: null,
          });
        } catch (error) {
          errorLogger.logError('Error clearing evolution', 'error', {
            component: 'evolutionStore',
            function: 'clearEvolution',
            additionalData: { error }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to clear evolution' });
        }
      },

      clearAdversarial: () => {
        try {
          set({
            adversarialTest: null,
            isAdversarialRunning: false,
            error: null,
          });
        } catch (error) {
          errorLogger.logError('Error clearing adversarial', 'error', {
            component: 'evolutionStore',
            function: 'clearAdversarial',
            additionalData: { error }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to clear adversarial' });
        }
      },

      reset: () => {
        try {
          set({
            evolution: null,
            adversarialTest: null,
            content: '',
            isEvolutionRunning: false,
            isAdversarialRunning: false,
            error: null,
          });
        } catch (error) {
          errorLogger.logError('Error resetting store', 'error', {
            component: 'evolutionStore',
            function: 'resetStore',
            additionalData: { error }
          });
          set({ error: error instanceof Error ? error.message : 'Failed to reset store' });
        }
      },
    }),
    { name: 'EvolutionStore' }
  )
);
