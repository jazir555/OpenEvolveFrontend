
// @ts-nocheck
import React from 'react';
import { EnhancedOpenEvolvePluginState } from '../../types/enhanced-plugin-types';
import {
  BubbleBadge,
  BubbleButton,
  BubbleCard,
  BubbleCheckbox,
  BubbleField,
  BubbleInput,
  BubbleSelect,
  BubbleTextArea,
} from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

// Monitoring Configuration Tab
const MonitoringConfigTabBase: React.FC<{
  config: EnhancedOpenEvolvePluginState;
  onConfigUpdate: (updates: Partial<EnhancedOpenEvolvePluginState>) => void;
  onValidate: () => void;
}> = ({ config, onConfigUpdate, onValidate }) => {
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ monitoringConfig: { ...config.monitoringConfig, [name]: type === 'checkbox' ? checked : value } });
  };

  const handleMetricsChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ monitoringConfig: { ...config.monitoringConfig, metrics: { ...config.monitoringConfig?.metrics, [name]: type === 'checkbox' ? checked : value } } });
  };

  const handleLoggingChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ monitoringConfig: { ...config.monitoringConfig, logging: { ...config.monitoringConfig?.logging, [name]: type === 'checkbox' ? checked : value } } });
  };

  const handleAlertingChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ monitoringConfig: { ...config.monitoringConfig, alerting: { ...config.monitoringConfig?.alerting, [name]: type === 'checkbox' ? checked : value } } });
  };

  return (
    <div className="space-y-6">
      <BubbleCard
        title="Monitoring Configuration"
        description="Enable monitoring and observability."
        actions={
          <BubbleBadge tone={config.monitoringConfig?.enabled ? 'success' : 'neutral'}>
            {config.monitoringConfig?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <BubbleCheckbox
          name="enabled"
          checked={config.monitoringConfig?.enabled || false}
          onChange={handleInputChange}
          label="Enable monitoring"
        />
      </BubbleCard>

      <BubbleCard
        title="Metrics Configuration"
        description="Configure metrics collection cadence."
        actions={
          <BubbleBadge tone={config.monitoringConfig?.metrics?.enabled ? 'success' : 'neutral'}>
            {config.monitoringConfig?.metrics?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.monitoringConfig?.metrics?.enabled || false}
            onChange={handleMetricsChange}
            label="Enable metrics collection"
          />
          {config.monitoringConfig?.metrics?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="Collection Interval (seconds)">
                <BubbleInput
                  type="number"
                  name="collection_interval"
                  min="1"
                  max="3600"
                  value={config.monitoringConfig?.metrics?.collection_interval || 60}
                  onChange={handleMetricsChange}
                />
              </BubbleField>
              <BubbleField label="Metrics to Collect">
                <BubbleTextArea
                  name="metrics_to_collect"
                  value={config.monitoringConfig?.metrics?.metrics_to_collect || ''}
                  onChange={handleMetricsChange}
                  placeholder="cpu, memory, network, disk, etc."
                />
              </BubbleField>
              <BubbleField label="Retention Days">
                <BubbleInput
                  type="number"
                  name="retention_days"
                  min="1"
                  max="365"
                  value={config.monitoringConfig?.metrics?.retention_days || 30}
                  onChange={handleMetricsChange}
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <BubbleCard
        title="Logging Configuration"
        description="Select logging level and format."
        actions={
          <BubbleBadge tone={config.monitoringConfig?.logging?.enabled ? 'success' : 'neutral'}>
            {config.monitoringConfig?.logging?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.monitoringConfig?.logging?.enabled || false}
            onChange={handleLoggingChange}
            label="Enable logging"
          />
          {config.monitoringConfig?.logging?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="Log Level">
                <BubbleSelect
                  name="level"
                  value={config.monitoringConfig?.logging?.level || 'info'}
                  onChange={handleLoggingChange}
                >
                  <option value="debug">Debug</option>
                  <option value="info">Info</option>
                  <option value="warn">Warn</option>
                  <option value="error">Error</option>
                  <option value="critical">Critical</option>
                </BubbleSelect>
              </BubbleField>
              <BubbleField label="Log Format">
                <BubbleSelect
                  name="format"
                  value={config.monitoringConfig?.logging?.format || 'json'}
                  onChange={handleLoggingChange}
                >
                  <option value="json">JSON</option>
                  <option value="text">Text</option>
                  <option value="structured">Structured</option>
                </BubbleSelect>
              </BubbleField>
              <BubbleField label="Max Log Size (MB)">
                <BubbleInput
                  type="number"
                  name="max_size_mb"
                  min="1"
                  max="1000"
                  value={config.monitoringConfig?.logging?.max_size_mb || 100}
                  onChange={handleLoggingChange}
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>
      <BubbleCard
        title="Alerting Configuration"
        description="Manage alert destinations and thresholds."
        actions={
          <BubbleBadge tone={config.monitoringConfig?.alerting?.enabled ? 'success' : 'neutral'}>
            {config.monitoringConfig?.alerting?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.monitoringConfig?.alerting?.enabled || false}
            onChange={handleAlertingChange}
            label="Enable alerting"
          />
          {config.monitoringConfig?.alerting?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="Alert Destinations">
                <BubbleTextArea
                  name="destinations"
                  value={config.monitoringConfig?.alerting?.destinations || ''}
                  onChange={handleAlertingChange}
                  placeholder="email, slack, pagerduty, etc."
                />
              </BubbleField>
              <BubbleField label="Alert Thresholds (JSON)">
                <BubbleTextArea
                  name="thresholds_json"
                  value={JSON.stringify(config.monitoringConfig?.alerting?.thresholds || {}, null, 2)}
                  onChange={(e) => {
                    try {
                      const thresholds = JSON.parse(e.target.value);
                      onConfigUpdate({ monitoringConfig: { ...config.monitoringConfig, alerting: { ...config.monitoringConfig?.alerting, thresholds } } });
                    } catch {}
                  }}
                  placeholder='{"cpu": 90, "memory": 85, "error_rate": 10}'
                />
              </BubbleField>
              <BubbleField label="Alert Cooldown (minutes)">
                <BubbleInput
                  type="number"
                  name="cooldown_minutes"
                  min="1"
                  max="1440"
                  value={config.monitoringConfig?.alerting?.cooldown_minutes || 5}
                  onChange={handleAlertingChange}
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <div className="flex flex-wrap gap-2">
        <BubbleButton onClick={onValidate} variant="secondary">
          Validate monitoring config
        </BubbleButton>
        <BubbleButton onClick={() => onConfigUpdate({ monitoringConfig: { ...config.monitoringConfig } })}>
          Save monitoring config
        </BubbleButton>
      </div>
    </div>
  );
};

// Integration Configuration Tab
const IntegrationConfigTabBase: React.FC<{
  config: EnhancedOpenEvolvePluginState;
  onConfigUpdate: (updates: Partial<EnhancedOpenEvolvePluginState>) => void;
  onValidate: () => void;
}> = ({ config, onConfigUpdate, onValidate }) => {
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ integrationConfig: { ...config.integrationConfig, [name]: type === 'checkbox' ? checked : value } });
  };

  const handleRestApiChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ integrationConfig: { ...config.integrationConfig, rest_api: { ...config.integrationConfig?.rest_api, [name]: type === 'checkbox' ? checked : value } } });
  };

  const handleGraphqlChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ integrationConfig: { ...config.integrationConfig, graphql: { ...config.integrationConfig?.graphql, [name]: type === 'checkbox' ? checked : value } } });
  };

  const handleWebsocketChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ integrationConfig: { ...config.integrationConfig, websocket: { ...config.integrationConfig?.websocket, [name]: type === 'checkbox' ? checked : value } } });
  };

  return (
    <div className="space-y-6">
      <BubbleCard
        title="Integration Configuration"
        description="Enable and manage integration features."
        actions={
          <BubbleBadge tone={config.integrationConfig?.enabled ? 'success' : 'neutral'}>
            {config.integrationConfig?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <BubbleCheckbox
          name="enabled"
          checked={config.integrationConfig?.enabled || false}
          onChange={handleInputChange}
          label="Enable integration features"
        />
      </BubbleCard>

      <BubbleCard
        title="REST API Configuration"
        description="Configure REST endpoints and limits."
        actions={
          <BubbleBadge tone={config.integrationConfig?.rest_api?.enabled ? 'success' : 'neutral'}>
            {config.integrationConfig?.rest_api?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.integrationConfig?.rest_api?.enabled || false}
            onChange={handleRestApiChange}
            label="Enable REST API integration"
          />
          {config.integrationConfig?.rest_api?.enabled && (
            <div className="grid gap-4 md:grid-cols-2">
              <BubbleField label="Base URL">
                <BubbleInput
                  type="text"
                  name="base_url"
                  value={config.integrationConfig?.rest_api?.base_url || ''}
                  onChange={handleRestApiChange}
                  placeholder="https://api.example.com"
                />
              </BubbleField>
              <BubbleField label="Timeout (ms)">
                <BubbleInput
                  type="number"
                  name="timeout"
                  min="1000"
                  max="60000"
                  value={config.integrationConfig?.rest_api?.timeout || 5000}
                  onChange={handleRestApiChange}
                />
              </BubbleField>
              <BubbleField label="Max Retries">
                <BubbleInput
                  type="number"
                  name="max_retries"
                  min="0"
                  max="10"
                  value={config.integrationConfig?.rest_api?.max_retries || 3}
                  onChange={handleRestApiChange}
                />
              </BubbleField>
              <BubbleField label="Endpoints (JSON)">
                <BubbleTextArea
                  name="endpoints_json"
                  value={JSON.stringify(config.integrationConfig?.rest_api?.endpoints || [], null, 2)}
                  onChange={(e) => {
                    try {
                      const endpoints = JSON.parse(e.target.value);
                      onConfigUpdate({ integrationConfig: { ...config.integrationConfig, rest_api: { ...config.integrationConfig?.rest_api, endpoints } } });
                    } catch {}
                  }}
                  placeholder='[{"path": "/users", "method": "GET"}]'
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>
      <BubbleCard
        title="GraphQL Configuration"
        description="Configure GraphQL batching and schema."
        actions={
          <BubbleBadge tone={config.integrationConfig?.graphql?.enabled ? 'success' : 'neutral'}>
            {config.integrationConfig?.graphql?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.integrationConfig?.graphql?.enabled || false}
            onChange={handleGraphqlChange}
            label="Enable GraphQL integration"
          />
          {config.integrationConfig?.graphql?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="Schema URL">
                <BubbleInput
                  type="text"
                  name="schema_url"
                  value={(config.integrationConfig?.graphql as any)?.schema_url || ''}
                  onChange={handleGraphqlChange}
                  placeholder="https://api.example.com/graphql"
                />
              </BubbleField>
              <BubbleField label="Max Batch Size">
                <BubbleInput
                  type="number"
                  name="max_batch_size"
                  min="1"
                  max="100"
                  value={config.integrationConfig?.graphql?.max_batch_size || 10}
                  onChange={handleGraphqlChange}
                />
              </BubbleField>
              <BubbleField label="Query Timeout (ms)">
                <BubbleInput
                  type="number"
                  name="query_timeout_ms"
                  min="1000"
                  max="30000"
                  value={(config.integrationConfig?.graphql as any)?.query_timeout_ms || 10000}
                  onChange={handleGraphqlChange}
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <BubbleCard
        title="WebSocket Configuration"
        description="Configure realtime connections."
        actions={
          <BubbleBadge tone={config.integrationConfig?.websocket?.enabled ? 'success' : 'neutral'}>
            {config.integrationConfig?.websocket?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.integrationConfig?.websocket?.enabled || false}
            onChange={handleWebsocketChange}
            label="Enable WebSocket integration"
          />
          {config.integrationConfig?.websocket?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="WebSocket URL">
                <BubbleInput
                  type="text"
                  name="url"
                  value={config.integrationConfig?.websocket?.url || ''}
                  onChange={handleWebsocketChange}
                  placeholder="wss://api.example.com/ws"
                />
              </BubbleField>
              <BubbleField label="Ping Interval (ms)">
                <BubbleInput
                  type="number"
                  name="ping_interval"
                  min="1000"
                  max="30000"
                  value={config.integrationConfig?.websocket?.ping_interval || 5000}
                  onChange={handleWebsocketChange}
                />
              </BubbleField>
              <BubbleField label="Reconnect Attempts">
                <BubbleInput
                  type="number"
                  name="reconnect_attempts"
                  min="0"
                  max="10"
                  value={config.integrationConfig?.websocket?.reconnect_attempts as any || 5}
                  onChange={handleWebsocketChange}
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <div className="flex flex-wrap gap-2">
        <BubbleButton onClick={onValidate} variant="secondary">
          Validate integration config
        </BubbleButton>
        <BubbleButton onClick={() => onConfigUpdate({ integrationConfig: { ...config.integrationConfig } })}>
          Save integration config
        </BubbleButton>
      </div>
    </div>
  );
};

// Error Handling Configuration Tab
const ErrorHandlingConfigTabBase: React.FC<{
  config: EnhancedOpenEvolvePluginState;
  onConfigUpdate: (updates: Partial<EnhancedOpenEvolvePluginState>) => void;
  onValidate: () => void;
}> = ({ config, onConfigUpdate, onValidate }) => {
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ errorHandlingConfig: { ...config.errorHandlingConfig, [name]: type === 'checkbox' ? checked : value } });
  };

  const handleErrorClassificationChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ errorHandlingConfig: { ...config.errorHandlingConfig, error_classification: { ...config.errorHandlingConfig?.error_classification, [name]: type === 'checkbox' ? checked : value } } });
  };

  const handleErrorRecoveryChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ errorHandlingConfig: { ...config.errorHandlingConfig, error_recovery: { ...config.errorHandlingConfig?.error_recovery, [name]: type === 'checkbox' ? checked : value } } });
  };

  const handleErrorReportingChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ errorHandlingConfig: { ...config.errorHandlingConfig, error_reporting: { ...config.errorHandlingConfig?.error_reporting, [name]: type === 'checkbox' ? checked : value } } });
  };

  return (
    <div className="space-y-6">
      <BubbleCard
        title="Error Handling Configuration"
        description="Enable error classification, recovery, and reporting."
        actions={
          <BubbleBadge tone={config.errorHandlingConfig?.enabled ? 'success' : 'neutral'}>
            {config.errorHandlingConfig?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <BubbleCheckbox
          name="enabled"
          checked={config.errorHandlingConfig?.enabled || false}
          onChange={handleInputChange}
          label="Enable error handling"
        />
      </BubbleCard>

      <BubbleCard
        title="Error Classification Configuration"
        description="Classify and categorize errors."
        actions={
          <BubbleBadge tone={config.errorHandlingConfig?.error_classification?.enabled ? 'success' : 'neutral'}>
            {config.errorHandlingConfig?.error_classification?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.errorHandlingConfig?.error_classification?.enabled || false}
            onChange={handleErrorClassificationChange}
            label="Enable error classification"
          />
          {config.errorHandlingConfig?.error_classification?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="Max History">
                <BubbleInput
                  type="number"
                  name="max_history"
                  min="1"
                  max="1000"
                  value={config.errorHandlingConfig?.error_classification?.max_history || 100}
                  onChange={handleErrorClassificationChange}
                />
              </BubbleField>
              <BubbleField label="Classification Method">
                <BubbleSelect
                  name="method"
                  value={config.errorHandlingConfig?.error_classification?.method || 'rule-based'}
                  onChange={handleErrorClassificationChange}
                >
                  <option value="rule-based">Rule-Based</option>
                  <option value="ml-based">ML-Based</option>
                  <option value="hybrid">Hybrid</option>
                </BubbleSelect>
              </BubbleField>
              <BubbleField label="Error Categories">
                <BubbleTextArea
                  name="categories"
                  value={config.errorHandlingConfig?.error_classification?.categories || ''}
                  onChange={handleErrorClassificationChange}
                  placeholder="network_error, validation_error, authentication_error, etc."
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>
      <BubbleCard
        title="Error Recovery Configuration"
        description="Set retries and recovery strategy."
        actions={
          <BubbleBadge tone={config.errorHandlingConfig?.error_recovery?.enabled ? 'success' : 'neutral'}>
            {config.errorHandlingConfig?.error_recovery?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.errorHandlingConfig?.error_recovery?.enabled || false}
            onChange={handleErrorRecoveryChange}
            label="Enable error recovery"
          />
          {config.errorHandlingConfig?.error_recovery?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="Max Attempts">
                <BubbleInput
                  type="number"
                  name="max_attempts"
                  min="1"
                  max="10"
                  value={config.errorHandlingConfig?.error_recovery?.max_attempts || 3}
                  onChange={handleErrorRecoveryChange}
                />
              </BubbleField>
              <BubbleField label="Retry Delay (ms)">
                <BubbleInput
                  type="number"
                  name="retry_delay"
                  min="100"
                  max="30000"
                  value={config.errorHandlingConfig?.error_recovery?.retry_delay || 1000}
                  onChange={handleErrorRecoveryChange}
                />
              </BubbleField>
              <BubbleField label="Recovery Strategies">
                <BubbleTextArea
                  name="strategies"
                  value={config.errorHandlingConfig?.error_recovery?.strategies || ''}
                  onChange={handleErrorRecoveryChange}
                  placeholder="retry, fallback, circuit_breaker, etc."
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <BubbleCard
        title="Error Reporting Configuration"
        description="Configure reporting destinations and format."
        actions={
          <BubbleBadge tone={config.errorHandlingConfig?.error_reporting?.enabled ? 'success' : 'neutral'}>
            {config.errorHandlingConfig?.error_reporting?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.errorHandlingConfig?.error_reporting?.enabled || false}
            onChange={handleErrorReportingChange}
            label="Enable error reporting"
          />
          {config.errorHandlingConfig?.error_reporting?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="Destinations">
                <BubbleTextArea
                  name="destinations"
                  value={config.errorHandlingConfig?.error_reporting?.destinations || ''}
                  onChange={handleErrorReportingChange}
                  placeholder="console, file, api, email, database"
                />
              </BubbleField>
              <BubbleField label="Report Format">
                <BubbleSelect
                  name="format"
                  value={config.errorHandlingConfig?.error_reporting?.format || 'json'}
                  onChange={handleErrorReportingChange}
                >
                  <option value="json">JSON</option>
                  <option value="text">Text</option>
                  <option value="structured">Structured</option>
                </BubbleSelect>
              </BubbleField>
              <BubbleField label="Report Interval (minutes)">
                <BubbleInput
                  type="number"
                  name="report_interval_minutes"
                  min="1"
                  max="1440"
                  value={config.errorHandlingConfig?.error_reporting?.report_interval_minutes || 5}
                  onChange={handleErrorReportingChange}
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <div className="flex flex-wrap gap-2">
        <BubbleButton onClick={onValidate} variant="secondary">
          Validate error handling config
        </BubbleButton>
        <BubbleButton onClick={() => onConfigUpdate({ errorHandlingConfig: { ...config.errorHandlingConfig } })}>
          Save error handling config
        </BubbleButton>
      </div>
    </div>
  );
};

// Profiles Tab
const ProfilesTabBase: React.FC<{
  config: EnhancedOpenEvolvePluginState;
  onAddPerformanceProfile: () => void;
  onRemovePerformanceProfile: (profileName: string) => void;
  onAddSecurityProfile: () => void;
  onRemoveSecurityProfile: (profileName: string) => void;
}> = ({ config, onAddPerformanceProfile, onRemovePerformanceProfile, onAddSecurityProfile, onRemoveSecurityProfile }) => {
  return (
    <div className="space-y-6">
      <BubbleCard
        title="Performance Profiles"
        description="Manage saved performance presets."
        actions={
          <BubbleButton onClick={onAddPerformanceProfile} variant="secondary">
            Add performance profile
          </BubbleButton>
        }
      >
        {config.performanceProfiles && Object.keys(config.performanceProfiles).length > 0 ? (
          <div className="space-y-3">
            {Object.entries(config.performanceProfiles).map(([profileName, profileConfig]) => {
              const settings = (profileConfig as any).settings || profileConfig;
              return (
              <div key={profileName} className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-100 bg-slate-50 px-4 py-3">
                <div>
                  <div className="text-sm font-semibold text-slate-900">{profileName}</div>
                  <div className="mt-1 flex flex-wrap gap-2 text-xs text-slate-500">
                    {settings.caching?.enabled && <BubbleBadge tone="info">Caching</BubbleBadge>}
                    {settings.parallel_processing?.enabled && <BubbleBadge tone="warning">Parallel</BubbleBadge>}
                    {settings.memory_management?.enabled && <BubbleBadge tone="neutral">Memory</BubbleBadge>}
                  </div>
                </div>
                <BubbleButton onClick={() => onRemovePerformanceProfile(profileName)} variant="ghost">
                  Remove
                </BubbleButton>
              </div>
            )})}
          </div>
        ) : (
          <p className="text-sm text-slate-500">No performance profiles defined.</p>
        )}
      </BubbleCard>

      <BubbleCard
        title="Security Profiles"
        description="Manage saved security presets."
        actions={
          <BubbleButton onClick={onAddSecurityProfile} variant="secondary">
            Add security profile
          </BubbleButton>
        }
      >
        {config.securityProfiles && Object.keys(config.securityProfiles).length > 0 ? (
          <div className="space-y-3">
            {Object.entries(config.securityProfiles).map(([profileName, profileConfig]) => {
              const settings = (profileConfig as any).settings || profileConfig;
              return (
              <div key={profileName} className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-100 bg-slate-50 px-4 py-3">
                <div>
                  <div className="text-sm font-semibold text-slate-900">{profileName}</div>
                  <div className="mt-1 flex flex-wrap gap-2 text-xs text-slate-500">
                    {settings.authentication?.enabled && <BubbleBadge tone="info">Auth</BubbleBadge>}
                    {settings.data_protection?.enabled && <BubbleBadge tone="warning">Protection</BubbleBadge>}
                    {settings.compliance?.enabled && <BubbleBadge tone="neutral">Compliance</BubbleBadge>}
                  </div>
                </div>
                <BubbleButton onClick={() => onRemoveSecurityProfile(profileName)} variant="ghost">
                  Remove
                </BubbleButton>
              </div>
            )})}
          </div>
        ) : (
          <p className="text-sm text-slate-500">No security profiles defined.</p>
        )}
      </BubbleCard>
    </div>
  );
};
// Statistics Tab
const StatisticsTabBase: React.FC<{
  config: EnhancedOpenEvolvePluginState;
  onClearValidationHistory: () => void;
}> = ({ config, onClearValidationHistory }) => {
  const totalExecutions = config.executionStatistics?.totalExecutions || 0;
  const averageExecutionTime = config.executionStatistics?.averageExecutionTime || 0;
  const totalExecutionTime = totalExecutions > 0 ? totalExecutions * averageExecutionTime : 0;

  const formatValidationDate = (value: number | Date | string) => {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString();
  };

  return (
    <div className="space-y-6">
      <BubbleCard title="Execution Statistics" description="Execution volume and timings.">
        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Total Executions</div>
            <div className="mt-2 text-2xl font-semibold text-slate-900">{totalExecutions}</div>
          </div>
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Successful Executions</div>
            <div className="mt-2 text-2xl font-semibold text-emerald-600">{config.executionStatistics?.successfulExecutions || 0}</div>
          </div>
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Failed Executions</div>
            <div className="mt-2 text-2xl font-semibold text-rose-600">{config.executionStatistics?.failedExecutions || 0}</div>
          </div>
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Total Execution Time</div>
            <div className="mt-2 text-lg font-semibold text-slate-900">{(totalExecutionTime / 1000).toFixed(2)} seconds</div>
          </div>
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Average Execution Time</div>
            <div className="mt-2 text-lg font-semibold text-slate-900">
              {totalExecutions > 0 ? `${(averageExecutionTime / 1000).toFixed(2)} seconds` : 'N/A'}
            </div>
          </div>
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Success Rate</div>
            <div className="mt-2 text-lg font-semibold text-slate-900">
              {totalExecutions > 0
                ? ((config.executionStatistics?.successfulExecutions || 0) / totalExecutions * 100).toFixed(2) + '%'
                : 'N/A'}
            </div>
          </div>
        </div>
      </BubbleCard>

      <BubbleCard title="Error Statistics" description="Error counts and types.">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Total Errors</div>
            <div className="mt-2 text-2xl font-semibold text-slate-900">{config.errorStatistics?.totalErrors || 0}</div>
          </div>
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Error Rate</div>
            <div className="mt-2 text-lg font-semibold text-slate-900">
              {config.executionStatistics?.totalExecutions && config.executionStatistics.totalExecutions > 0
                ? ((config.errorStatistics?.totalErrors || 0) / config.executionStatistics.totalExecutions * 100).toFixed(2) + '%'
                : 'N/A'}
            </div>
          </div>
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Most Common Error Type</div>
            <div className="mt-2 text-lg font-semibold text-slate-900">
              {config.errorStatistics?.errorsByType
                ? Object.entries(config.errorStatistics.errorsByType).sort((a, b) => b[1] - a[1])[0]?.[0] || 'None'
                : 'None'}
            </div>
          </div>
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
            <div className="text-xs uppercase tracking-wide text-slate-500">Last Error</div>
            <div className="mt-2 text-lg font-semibold text-slate-900">{config.errorStatistics?.lastError?.errorMessage || 'None'}</div>
          </div>
        </div>
      </BubbleCard>

      <BubbleCard
        title="Validation History"
        description="Recent validation outcomes."
        actions={
          <BubbleButton onClick={onClearValidationHistory} variant="secondary">
            Clear validation history
          </BubbleButton>
        }
      >
        {config.validationHistory && config.validationHistory.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-2">Timestamp</th>
                  <th className="py-2">Type</th>
                  <th className="py-2">Status</th>
                  <th className="py-2">Message</th>
                </tr>
              </thead>
              <tbody className="text-slate-700">
                {config.validationHistory.slice().reverse().map((validation) => (
                  <tr key={validation.validationId} className="border-t border-slate-100">
                    <td className="py-2">{formatValidationDate(validation.timestamp)}</td>
                    <td className="py-2">{validation.validationType}</td>
                    <td className="py-2">
                      <BubbleBadge tone={validation.success ? 'success' : 'danger'}>
                        {validation.success ? 'Success' : 'Failed'}
                      </BubbleBadge>
                    </td>
                    <td className="py-2">{validation.errorMessage || 'Valid'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-slate-500">No validation history available.</p>
        )}
      </BubbleCard>
    </div>
  );
};

export const MonitoringConfigTab = withComponentBoundary(
  MonitoringConfigTabBase,
  'MonitoringConfigTab'
);
export const IntegrationConfigTab = withComponentBoundary(
  IntegrationConfigTabBase,
  'IntegrationConfigTab'
);
export const ErrorHandlingConfigTab = withComponentBoundary(
  ErrorHandlingConfigTabBase,
  'ErrorHandlingConfigTab'
);
export const ProfilesTab = withComponentBoundary(ProfilesTabBase, 'ProfilesTab');
export const StatisticsTab = withComponentBoundary(StatisticsTabBase, 'StatisticsTab');
