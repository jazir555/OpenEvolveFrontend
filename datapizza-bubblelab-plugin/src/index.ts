// BubbleLabs Datapizza Plugin - Main Export
// Standalone plugin for data pipeline processing and querying

// Export types
export type {
  DatapizzaPluginConfig,
  DatapizzaPluginState,
  DatapizzaPipelineResult,
  DatapizzaProcessingResult,
  DatapizzaQueryResult,
  DatapizzaPluginContext,
  DatapizzaPluginMethods,
  DatapizzaPlugin,
  DatapizzaPluginProps,
  DatapizzaConfigPanelProps,
  DatapizzaPipelinePanelProps,
  DatapizzaProcessingPanelProps,
  DatapizzaQueryPanelProps,
  DatapizzaPipelineType,
  DatapizzaDataDomain
} from './types/plugin-types';

export {
  DATAPIZZA_PIPELINE_TYPES,
  DATAPIZZA_DATA_DOMAINS,
  DEFAULT_DATAPIZZA_CONFIG
} from './types/plugin-types';

// Export components
export { DatapizzaConfigPanel } from './components/DatapizzaConfigPanel';
export { DatapizzaPipelinePanel } from './components/DatapizzaPipelinePanel';

// Export hooks (stubs for now)
export { useDatapizzaConfig } from './hooks/useDatapizzaConfig';
export { useDatapizzaState } from './hooks/useDatapizzaState';
export { useDatapizzaPipeline } from './hooks/useDatapizzaPipeline';
export { useDatapizzaProcessing } from './hooks/useDatapizzaProcessing';
export { useDatapizzaQuery } from './hooks/useDatapizzaQuery';

// Export services (stubs for now)
export { DatapizzaClient } from './services/DatapizzaClient';
export { DatapizzaService } from './services/DatapizzaService';

// Export utilities
export { createDatapizzaPlugin } from './utils/createDatapizzaPlugin';
export { useDatapizzaPlugin } from './utils/createDatapizzaPlugin';

// Export the main plugin factory
import { createDatapizzaPlugin } from './utils/createDatapizzaPlugin';

/**
 * Create a new Datapizza plugin instance
 * @param config Optional initial configuration
 * @returns DatapizzaPlugin instance
 */
export function createPlugin(config?: Partial<DatapizzaPluginConfig>): DatapizzaPlugin {
  return createDatapizzaPlugin(config);
}

/**
 * Default plugin instance
 */
export const DatapizzaPlugin = createDatapizzaPlugin();

export default DatapizzaPlugin;