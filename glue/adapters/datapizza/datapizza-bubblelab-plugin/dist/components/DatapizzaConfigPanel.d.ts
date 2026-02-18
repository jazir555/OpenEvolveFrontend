import { DatapizzaPluginConfig } from '../types/plugin-types';
export interface DatapizzaConfigPanelProps {
    /** Initial configuration */
    initialConfig?: Partial<DatapizzaPluginConfig>;
    /** Callback when configuration is saved */
    onSave: (config: DatapizzaPluginConfig) => void;
    /** Callback when configuration is cancelled */
    onCancel: () => void;
    /** Show advanced options */
    showAdvanced?: boolean;
}
export declare function DatapizzaConfigPanel({ initialConfig, onSave, onCancel, showAdvanced }: DatapizzaConfigPanelProps): any;
