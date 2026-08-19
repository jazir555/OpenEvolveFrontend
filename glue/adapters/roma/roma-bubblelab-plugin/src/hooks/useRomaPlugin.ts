/**
 * useRomaPlugin Hook
 *
 * Core hook for accessing ROMA plugin instance and methods.
 * This is the primary hook that all other ROMA hooks depend on.
 */

import { useMemo } from 'react';
import { romaPlugin } from '../utils/createRomaPlugin';
import { RomaPlugin, RomaPluginState, RomaPluginError } from '../types/plugin-types';

/**
 * Hook return type for useRomaPlugin
 */
export interface UseRomaPluginReturn {
  plugin: RomaPlugin;
  config: RomaPluginState;
  state: RomaPluginState;
  isReady: boolean;
  isExecuting: boolean;
  error?: string;
}

/**
 * useRomaPlugin Hook
 *
 * Provides access to the global ROMA plugin instance.
 * Uses singleton pattern - only one plugin instance exists per application.
 *
 * @returns Plugin instance, config, state, and status flags
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { plugin, isReady, isExecuting } = useRomaPlugin();
 *
 *   const handleExecute = async () => {
 *     if (!isReady) {
 *       console.error('Plugin not initialized');
 *       return;
 *     }
 *
 *     const result = await plugin.executeTask('What is 2+2?');
 *     console.log('Result:', result);
 *   };
 *
 *   return (
 *     <div>
 *       <p>Status: {isExecuting ? 'Executing...' : 'Idle'}</p>
 *       <button onClick={handleExecute} disabled={!isReady}>
 *         Execute Task
 *       </button>
 *     </div>
 *   );
 * }
 * ```
 */
export function useRomaPlugin(): UseRomaPluginReturn {
  const plugin = useMemo(() => romaPlugin, []);

  if (!plugin) {
    return {
      plugin: {} as RomaPlugin,
      config: {} as RomaPluginState,
      state: {} as RomaPluginState,
      isReady: false,
      isExecuting: false,
      error: 'ROMA plugin not initialized. Call createRomaPlugin() first.'
    };
  }

  const state = plugin.getState();
  const isReady = plugin.isReady();
  const isExecuting = state.status === 'executing';
  const error = state.initializationError;

  return {
    plugin,
    config: state,
    state,
    isReady,
    isExecuting,
    error
  };
}
