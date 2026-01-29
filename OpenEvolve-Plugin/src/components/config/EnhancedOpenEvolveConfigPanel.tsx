// @ts-nocheck
import React, { useState, useEffect } from 'react';
import { EnhancedOpenEvolvePluginState } from '../../types/enhanced-plugin-types';
import { createEnhancedOpenEvolvePlugin } from '../../utils/createEnhancedOpenEvolvePlugin';
import { toast } from 'react-toastify';
import { v4 as uuidv4 } from 'uuid';
import { PerformanceConfigTab } from '../tabs/PerformanceConfigTab';
import { SecurityConfigTab } from '../tabs/SecurityConfigTab';
import { MonitoringConfigTab, IntegrationConfigTab, ErrorHandlingConfigTab, ProfilesTab, StatisticsTab } from '../tabs/RemainingTabs';
import { BubbleButton, BubbleInput, BubbleSelect } from '@/components/bubblelab';
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
  const enhancedPlugin = createEnhancedOpenEvolvePlugin();

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
    const unsubscribe = enhancedPlugin.subscribeToEnhancedState((newState) => {
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

      enhancedPlugin.addPerformanceProfile(profileName, profileConfig);
    } catch (error) {
      toast.error(`Failed to add performance profile: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleRemovePerformanceProfile = (profileName: string) => {
    try {
      const confirmed = window.confirm(`Are you sure you want to remove performance profile "${profileName}"?`);
      if (confirmed) {
        enhancedPlugin.removePerformanceProfile(profileName);
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

      enhancedPlugin.addSecurityProfile(profileName, profileConfig);
    } catch (error) {
      toast.error(`Failed to add security profile: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const handleRemoveSecurityProfile = (profileName: string) => {
    try {
      const confirmed = window.confirm(`Are you sure you want to remove security profile "${profileName}"?`);
      if (confirmed) {
        enhancedPlugin.removeSecurityProfile(profileName);
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
        enhancedPlugin.clearValidationHistory();
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
      <div className={`enhanced-config-panel ${isDarkMode ? 'dark' : ''}`}>
        <div className="loading-spinner">Loading enhanced configuration...</div>
      </div>
    );
  }

  if (!config) {
    return (
      <div className={`enhanced-config-panel ${isDarkMode ? 'dark' : ''}`}>
        <div className="error-message">Failed to load enhanced configuration</div>
      </div>
    );
  }

  return (
    <div className={`enhanced-config-panel ${isDarkMode ? 'dark' : ''}`}>
      <div className="panel-header">
        <h1>Enhanced OpenEvolve Configuration</h1>
        <div className="header-actions">
          <BubbleButton onClick={() => setIsDarkMode(!isDarkMode)} variant="secondary">
            {isDarkMode ? '?? Light Mode' : '?? Dark Mode'}
          </BubbleButton>
          <BubbleButton onClick={handleResetConfig} variant="secondary">
            ?? Reset All Config
          </BubbleButton>
          <BubbleButton onClick={handleExecuteWithEnhancedFeatures}>
            ? Execute with Enhanced Features
          </BubbleButton>
        </div>
      </div>

      <div className="tabs-container">
        <BubbleButton
          variant={activeTab === 'performance' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('performance')}
        >
          ?? Performance
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'security' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('security')}
        >
          ?? Security
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'monitoring' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('monitoring')}
        >
          ?? Monitoring
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'integration' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('integration')}
        >
          ?? Integration
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'error_handling' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('error_handling')}
        >
          ?? Error Handling
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'profiles' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('profiles')}
        >
          ?? Profiles
        </BubbleButton>
        <BubbleButton
          variant={activeTab === 'statistics' ? 'primary' : 'secondary'}
          onClick={() => setActiveTab('statistics')}
        >
          ?? Statistics
        </BubbleButton>
      </div>

      <div className="tab-content">
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

      <style>{`
        .enhanced-config-panel {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
          border-radius: 12px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
          background-color: #ffffff;
          color: #333333;
        }

        .enhanced-config-panel.dark {
          background-color: #1a1a1a;
          color: #f0f0f0;
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          padding-bottom: 15px;
          border-bottom: 2px solid #e0e0e0;
        }

        .enhanced-config-panel.dark .panel-header {
          border-bottom-color: #333333;
        }

        .panel-header h1 {
          font-size: 28px;
          font-weight: 600;
          color: #2c3e50;
          margin: 0;
        }

        .enhanced-config-panel.dark .panel-header h1 {
          color: #f0f0f0;
        }

        .header-actions {
          display: flex;
          gap: 10px;
        }

        .tabs-container {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 20px;
          border-bottom: 1px solid #e0e0e0;
          padding-bottom: 10px;
        }

        .enhanced-config-panel.dark .tabs-container {
          border-bottom-color: #333333;
        }

        .tab-button {
          padding: 10px 20px;
          background-color: #f5f5f5;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: all 0.2s ease;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .enhanced-config-panel.dark .tab-button {
          background-color: #2a2a2a;
          color: #f0f0f0;
        }

        .tab-button:hover {
          background-color: #e0e0e0;
        }

        .enhanced-config-panel.dark .tab-button:hover {
          background-color: #3a3a3a;
        }

        .tab-button.active {
          background-color: #3498db;
          color: white;
        }

        .tab-content {
          padding: 20px;
          border-radius: 8px;
          background-color: #f9f9f9;
        }

        .enhanced-config-panel.dark .tab-content {
          background-color: #1e1e1e;
        }

        .loading-spinner {
          text-align: center;
          padding: 40px;
          font-size: 16px;
          color: #666666;
        }

        .enhanced-config-panel.dark .loading-spinner {
          color: #999999;
        }

        .error-message {
          text-align: center;
          padding: 40px;
          font-size: 16px;
          color: #e74c3c;
        }

        .enhanced-config-panel.dark .error-message {
          color: #ff6b6b;
        }
      `}</style>
    </div>
  );
};

export const EnhancedOpenEvolveConfigPanel = withComponentBoundary(
  EnhancedOpenEvolveConfigPanelBase,
  'EnhancedOpenEvolveConfigPanel'
);


