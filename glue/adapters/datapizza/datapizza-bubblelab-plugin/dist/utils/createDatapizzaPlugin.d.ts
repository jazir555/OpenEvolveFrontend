import { DatapizzaPluginConfig, DatapizzaPlugin } from '../types/plugin-types';
/**
 * Create a new Datapizza plugin instance
 * @param initialConfig Optional initial configuration
 * @returns DatapizzaPlugin instance
 */
export declare function createDatapizzaPlugin(initialConfig?: Partial<DatapizzaPluginConfig>): DatapizzaPlugin;
export declare function useDatapizzaPlugin(): DatapizzaPlugin;
