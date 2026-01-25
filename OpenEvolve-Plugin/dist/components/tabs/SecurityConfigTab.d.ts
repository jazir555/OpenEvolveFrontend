import { default as React } from 'react';
import { EnhancedOpenEvolvePluginState } from '../../types/enhanced-plugin-types';
export declare const SecurityConfigTab: React.FC<{
    config: EnhancedOpenEvolvePluginState;
    onConfigUpdate: (updates: Partial<EnhancedOpenEvolvePluginState>) => void;
    onValidate: () => void;
}>;
