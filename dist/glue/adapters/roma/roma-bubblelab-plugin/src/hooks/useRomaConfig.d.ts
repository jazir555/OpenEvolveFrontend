/**
 * useRomaConfig Hook
 *
 * React hook for ROMA plugin configuration management.
 * Provides methods to update and access ROMA configuration.
 */
import { RomaPluginConfig } from '../types/plugin-types';
/**
 * Hook return type for useRomaConfig
 */
export interface UseRomaConfigReturn {
    config: RomaPluginConfig;
    updateConfig: (configUpdate: Partial<RomaPluginConfig>) => Promise<void>;
    isUpdating: boolean;
    error?: string;
}
/**
 * useRomaConfig Hook
 *
 * Provides configuration management for the ROMA plugin.
 * Uses the underlying useRomaPlugin hook for state access.
 *
 * @returns Configuration state and update methods
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { config, updateConfig } = useRomaConfig();
 *
 *   const handleUpdate = async () => {
 *     await updateConfig({ maxDepth: 5 });
 *   };
 *
 *   return (
 *     <div>
 *       <p>Max Depth: {config.maxDepth}</p>
 *       <button onClick={handleUpdate}>Update</button>
 *     </div>
 *   );
 * }
 * ```
 */
export declare function useRomaConfig(): UseRomaConfigReturn;
//# sourceMappingURL=useRomaConfig.d.ts.map