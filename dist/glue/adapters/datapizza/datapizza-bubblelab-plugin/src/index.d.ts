export type { DatapizzaPluginConfig, DatapizzaPluginState, DatapizzaPipelineResult, DatapizzaProcessingResult, DatapizzaQueryResult, DatapizzaPluginContext, DatapizzaPluginMethods, DatapizzaPlugin, DatapizzaPluginProps, DatapizzaConfigPanelProps, DatapizzaPipelinePanelProps, DatapizzaProcessingPanelProps, DatapizzaQueryPanelProps, DatapizzaPipelineType, DatapizzaDataDomain } from './types/plugin-types';
export { DATAPIZZA_PIPELINE_TYPES, DATAPIZZA_DATA_DOMAINS, DEFAULT_DATAPIZZA_CONFIG } from './types/plugin-types';
export { DatapizzaConfigPanel } from './components/DatapizzaConfigPanel';
export { DatapizzaPipelinePanel } from './components/DatapizzaPipelinePanel';
export { useDatapizzaConfig } from './hooks/useDatapizzaConfig';
export { useDatapizzaState } from './hooks/useDatapizzaState';
export { useDatapizzaPipeline } from './hooks/useDatapizzaPipeline';
export { useDatapizzaProcessing } from './hooks/useDatapizzaProcessing';
export { useDatapizzaQuery } from './hooks/useDatapizzaQuery';
export { DatapizzaClient } from './services/DatapizzaClient';
export { DatapizzaService } from './services/DatapizzaService';
export { createDatapizzaPlugin } from './utils/createDatapizzaPlugin';
export { useDatapizzaPlugin } from './utils/createDatapizzaPlugin';
import type { DatapizzaPlugin as DatapizzaPluginInstance, DatapizzaPluginConfig } from './types/plugin-types';
/**
 * Create a new Datapizza plugin instance
 * @param config Optional initial configuration
 * @returns DatapizzaPlugin instance
 */
export declare function createPlugin(config?: Partial<DatapizzaPluginConfig>): DatapizzaPluginInstance;
/**
 * Default plugin instance
 */
export declare const datapizzaPlugin: DatapizzaPluginInstance;
export default datapizzaPlugin;
//# sourceMappingURL=index.d.ts.map