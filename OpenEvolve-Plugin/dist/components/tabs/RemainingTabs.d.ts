import { default as React } from 'react';
import { EnhancedOpenEvolvePluginState } from '../../types/enhanced-plugin-types';
export declare const MonitoringConfigTab: React.FC<{
    config: EnhancedOpenEvolvePluginState;
    onConfigUpdate: (updates: Partial<EnhancedOpenEvolvePluginState>) => void;
    onValidate: () => void;
}>;
export declare const IntegrationConfigTab: React.FC<{
    config: EnhancedOpenEvolvePluginState;
    onConfigUpdate: (updates: Partial<EnhancedOpenEvolvePluginState>) => void;
    onValidate: () => void;
}>;
export declare const ErrorHandlingConfigTab: React.FC<{
    config: EnhancedOpenEvolvePluginState;
    onConfigUpdate: (updates: Partial<EnhancedOpenEvolvePluginState>) => void;
    onValidate: () => void;
}>;
export declare const ProfilesTab: React.FC<{
    config: EnhancedOpenEvolvePluginState;
    onAddPerformanceProfile: () => void;
    onRemovePerformanceProfile: (profileName: string) => void;
    onAddSecurityProfile: () => void;
    onRemoveSecurityProfile: (profileName: string) => void;
}>;
export declare const StatisticsTab: React.FC<{
    config: EnhancedOpenEvolvePluginState;
    onClearValidationHistory: () => void;
}>;
