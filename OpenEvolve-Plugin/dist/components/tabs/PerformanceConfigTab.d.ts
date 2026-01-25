import { default as React } from 'react';
import { EnhancedOpenEvolvePluginState } from '../../types/enhanced-plugin-types';
/**
 * Performance Configuration Tab
 * UI for configuring performance optimization settings
 */
export declare const PerformanceConfigTab: React.FC<{
    config: EnhancedOpenEvolvePluginState;
    onConfigUpdate: (updates: Partial<EnhancedOpenEvolvePluginState>) => void;
    onValidate: () => void;
}>;
