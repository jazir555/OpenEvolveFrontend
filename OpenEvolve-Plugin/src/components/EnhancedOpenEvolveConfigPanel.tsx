
import React, { useState, useEffect } from 'react';
import { EnhancedOpenEvolvePluginState } from '../types/enhanced-plugin-types';
import { getEnhancedOpenEvolvePlugin } from '../utils/createEnhancedOpenEvolvePlugin';
import { toast } from 'react-toastify';
import { v4 as uuidv4 } from 'uuid';
import { PerformanceConfigTab } from './tabs/PerformanceConfigTab';
import { SecurityConfigTab } from './tabs/SecurityConfigTab';
import { MonitoringConfigTab, IntegrationConfigTab, ErrorHandlingConfigTab, ProfilesTab, StatisticsTab } from './tabs/RemainingTabs';
import { BubbleButton, BubbleCard } from './bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

/**
 * Enhanced OpenEvolve Configuration Panel
 * Comprehensive UI for configuring all enhanced features
 */
const EnhancedOpenEvolveConfigPanelBase: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'performance' | 'security' | 'monitoring' | 'integration' | 'error_handling' | 'profiles' | 'statistics'>('performance');
  const [config, setConfig] = useState<EnhancedOpenEvolvePluginState | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Get enhanced plugin instance
  const enhancedPlugin = getEnhancedOpenEvolvePlugin();

  // Load initial configuration
  useEffect(() => {
    try {
      const initialConfig = enhancedPlugin.getEnhancedState();
      setConfig(initialConfig);
      setIsLoading(false);
    } catch (error) {
      toast.error(`Failed to load enhanced configuration: ${error instanceof Error ? error.message : String(error)}`);
      setIsLoading(false);
    }
  }, [enhancedPlugin]);

  // Subscribe to state changes
  useEffect(() => {
    const unsubscribe = enhancedPlugin.subscribeToEnhancedState((newState: EnhancedOpenEvolvePluginState) => {
      setConfig(newState);
    });

    return () => unsubscribe();
  }, [enhancedPlugin]);

  // Handle configuration updates
  const handleConfigUpdate = (updates: Partial<EnhancedOpenEvolvePluginState>) => {
    try {
      const success = enhancedPlugin.updateEnhancedConfig(updates);
      if (!success) {
        throw new Error('Failed to update configuration');
      }
    } catch (error) {
      toast.error(`Failed to update configuration: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  // Handle configuration reset
  const handleResetConfig = () => {
    try {
      const confirmed = window.confirm('Are you sure you want to reset all enhanced configuration to defaults?');
      if (confirmed) {
        const success = enhancedPlugin.resetEnhancedConfig();
        if (!success) {
          throw new Error('Failed to reset configuration');
        }
      }
    } catch (error) {
      toast.error(`Failed to reset configuration: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  // Handle validation
  const handleValidateConfig = (configType: 'performance' | 'security' | 'monitoring' | 'integration' | 'error_handling') => {
    try {
      let success = false;
      const validationId = uuidv4();
      const timestamp = Date.now();

      switch (configType) {
        case 'performance':
          success = enhancedPlugin.validatePerformanceConfig();
          break;
        case 'security':
          success = enhancedPlugin.validateSecurityConfig();
          break;
        case 'monitoring':
          success = enhancedPlugin.validateMonitoringConfig();
          break;
        case 'integration':
          success = enhancedPlugin.validateIntegrationConfig();
          break;
        case 'error_handling':
          success = enhancedPlugin.validateErrorHandlingConfig();
          break;
      }

      // Add validation result to history
      enhancedPlugin.addValidationResult({
        validationId,
        validationType: configType,
        success,
        errorMessage: success ? undefined : `Validation failed for ${configType}`,
        timestamp,
      });
    } catch (error) {
      toast.error(`Failed to validate ${configType} configuration: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  // Handle performance profile management
  const handleAddPerformanceProfile = () => {
    try {
      const profileName = window.prompt('Enter performance profile name:');
      if (!profileName) return;

      if (!config) return;

      const profileConfig = {
        ...config.performanceConfig,
        caching: {
          ...config.performanceConfig?.caching,
          enabled: true,
        },
        parallel_processing: {
          ...config.performanceConfig?.parallel_processing,
          enabled: true,
        },
      };

      const success = enhancedPlugin.addPerformanceProfile(profileName, profileConfig);
      if (!success) {
        throw new Error('Failed to add performance profile');
      }
    } catch (error) {
      toast.error(`Failed to add performance profile: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleRemovePerformanceProfile = (profileName: string) => {
    try {
      const confirmed = window.confirm(`Are you sure you want to remove performance profile "${profileName}"?`);
      if (confirmed) {
        const success = enhancedPlugin.removePerformanceProfile(profileName);
        if (!success) {
          throw new Error('Failed to remove performance profile');
        }
      }
    } catch (error) {
      toast.error(`Failed to remove performance profile: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  // Handle security profile management
  const handleAddSecurityProfile = () => {
    try {
      const profileName = window.prompt('Enter security profile name:');
      if (!profileName) return;

      if (!config) return;

      const profileConfig = {
        ...config.securityConfig,
        authentication: {
          ...config.securityConfig?.authentication,
          enabled: true,
        },
        data_protection: {
          ...config.securityConfig?.data_protection,
          encryption: {
            ...config.securityConfig?.data_protection?.encryption,
            enabled: true,
          },
        },
      };

      const success = enhancedPlugin.addSecurityProfile(profileName, profileConfig);
      if (!success) {
        throw new Error('Failed to add security profile');
      }
    } catch (error) {
      toast.error(`Failed to add security profile: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleRemoveSecurityProfile = (profileName: string) => {
    try {
      const confirmed = window.confirm(`Are you sure you want to remove security profile "${profileName}"?`);
      if (confirmed) {
        const success = enhancedPlugin.removeSecurityProfile(profileName);
        if (!success) {
          throw new Error('Failed to remove security profile');
        }
      }
    } catch (error) {
      toast.error(`Failed to remove security profile: ${error instanceof Error ? error.message : String(error)}`);
    }
  };
  // Handle validation history
  const handleClearValidationHistory = () => {
    try {
      const confirmed = window.confirm('Are you sure you want to clear all validation history?');
      if (confirmed) {
        const success = enhancedPlugin.clearValidationHistory();
        if (!success) {
          throw new Error('Failed to clear validation history');
        }
      }
    } catch (error) {
      toast.error(`Failed to clear validation history: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  // Handle execution with enhanced features
  const handleExecuteWithEnhancedFeatures = async () => {
    try {
      const goal = window.prompt('Enter evolution goal:');
      if (!goal) return;

      const performanceProfile = window.prompt('Enter performance profile name (leave blank for default):');
      const securityProfile = window.prompt('Enter security profile name (leave blank for default):');
      const monitoringEnabled = window.confirm('Enable monitoring?');
      const integrationMode = window.prompt('Enter integration mode (auto/manual/disabled):') as 'auto' | 'manual' | 'disabled' || 'auto';

      const result = await enhancedPlugin.executeEvolutionWithEnhancedFeatures(goal, {
        performanceProfile: performanceProfile || undefined,
        securityProfile: securityProfile || undefined,
        monitoringEnabled,
        integrationMode,
      });

      if (result.success) {
        toast.success('Evolution executed successfully with enhanced features');
        console.log('Execution result:', result);
      } else {
        toast.error(`Evolution failed: ${result.error?.message || 'Unknown error'}`);
        console.error('Execution error:', result.error);
      }
    } catch (error) {
      toast.error(`Failed to execute evolution: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  if (isLoading) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-6">
        <BubbleCard title="Enhanced OpenEvolve Configuration" description="Loading enhanced configuration...">
          <div className="text-sm text-slate-500">Loading enhanced configuration...</div>
        </BubbleCard>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-6">
        <BubbleCard title="Enhanced OpenEvolve Configuration" description="Failed to load enhanced configuration.">
          <div className="text-sm text-slate-500">Please retry or check logs for details.</div>
        </BubbleCard>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Enhanced OpenEvolve Configuration</h1>
          <p className="mt-1 text-sm text-slate-500">Tune performance, security, monitoring, and integrations.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <BubbleButton onClick={() => setIsDarkMode(!isDarkMode)} variant="secondary">
            {isDarkMode ? 'Light mode' : 'Dark mode'}
          </BubbleButton>
          <BubbleButton onClick={handleResetConfig} variant="ghost">
            Reset all config
          </BubbleButton>
          <BubbleButton onClick={handleExecuteWithEnhancedFeatures}>
            Execute with enhanced features
          </BubbleButton>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <BubbleButton
          variant={activeTab === 'performance' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('performance')}
        >
          Performance
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'security' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('security')}
        >
          Security
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'monitoring' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('monitoring')}
        >
          Monitoring
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'integration' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('integration')}
        >
          Integration
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'error_handling' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('error_handling')}
        >
          Error Handling
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'profiles' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('profiles')}
        >
          Profiles
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'statistics' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('statistics')}
        >
          Statistics
        </BubbleButton>
      </div>

      <div className="rounded-xl border border-slate-100 bg-slate-50 p-5">
        {activeTab === 'performance' && (
          <PerformanceConfigTab
            config={config}
            onConfigUpdate={handleConfigUpdate}
            onValidate={() => handleValidateConfig('performance')}
          />
        )}

        {activeTab === 'security' && (
          <SecurityConfigTab
            config={config}
            onConfigUpdate={handleConfigUpdate}
            onValidate={() => handleValidateConfig('security')}
          />
        )}

        {activeTab === 'monitoring' && (
          <MonitoringConfigTab
            config={config}
            onConfigUpdate={handleConfigUpdate}
            onValidate={() => handleValidateConfig('monitoring')}
          />
        )}

        {activeTab === 'integration' && (
          <IntegrationConfigTab
            config={config}
            onConfigUpdate={handleConfigUpdate}
            onValidate={() => handleValidateConfig('integration')}
          />
        )}

        {activeTab === 'error_handling' && (
          <ErrorHandlingConfigTab
            config={config}
            onConfigUpdate={handleConfigUpdate}
            onValidate={() => handleValidateConfig('error_handling')}
          />
        )}

        {activeTab === 'profiles' && (
          <ProfilesTab
            config={config}
            onAddPerformanceProfile={handleAddPerformanceProfile}
            onRemovePerformanceProfile={handleRemovePerformanceProfile}
            onAddSecurityProfile={handleAddSecurityProfile}
            onRemoveSecurityProfile={handleRemoveSecurityProfile}
          />
        )}

        {activeTab === 'statistics' && (
          <StatisticsTab
            config={config}
            onClearValidationHistory={handleClearValidationHistory}
          />
        )}
      </div>
    </div>
  );
};

export const EnhancedOpenEvolveConfigPanel = withComponentBoundary(
  EnhancedOpenEvolveConfigPanelBase,
  'EnhancedOpenEvolveConfigPanel'
);
