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

const SecurityConfigTabBase: React.FC<{
  config: EnhancedOpenEvolvePluginState;
  onConfigUpdate: (updates: Partial<EnhancedOpenEvolvePluginState>) => void;
  onValidate: () => void;
}> = ({ config, onConfigUpdate, onValidate }) => {
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ securityConfig: { ...config.securityConfig, [name]: type === 'checkbox' ? checked : value } });
  };

  const handleAuthenticationChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ securityConfig: { ...config.securityConfig, authentication: { ...config.securityConfig?.authentication, [name]: type === 'checkbox' ? checked : value } } });
  };

  const handleDataProtectionChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ securityConfig: { ...config.securityConfig, data_protection: { ...config.securityConfig?.data_protection, [name]: type === 'checkbox' ? checked : value } } });
  };

  const handleEncryptionChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ securityConfig: { ...config.securityConfig, data_protection: { ...config.securityConfig?.data_protection, encryption: { ...config.securityConfig?.data_protection?.encryption, [name]: type === 'checkbox' ? checked : value } } } });
  };

  const handleComplianceChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ securityConfig: { ...config.securityConfig, compliance: { ...config.securityConfig?.compliance, [name]: type === 'checkbox' ? checked : value } } });
  };

  const handleAuditLoggingChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;
    onConfigUpdate({ securityConfig: { ...config.securityConfig, compliance: { ...config.securityConfig?.compliance, audit_logging: { ...config.securityConfig?.compliance?.audit_logging, [name]: type === 'checkbox' ? checked : value } } } });
  };

  return (
    <div className="space-y-6">
      <BubbleCard
        title="Security Configuration"
        description="Enable and manage core security features."
        actions={
          <BubbleBadge tone={config.securityConfig?.enabled ? 'success' : 'neutral'}>
            {config.securityConfig?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <BubbleCheckbox
          name="enabled"
          checked={config.securityConfig?.enabled || false}
          onChange={handleInputChange}
          label="Enable security features"
        />
      </BubbleCard>

      <BubbleCard
        title="Authentication Configuration"
        description="Select authentication methods and limits."
        actions={
          <BubbleBadge tone={config.securityConfig?.authentication?.enabled ? 'success' : 'neutral'}>
            {config.securityConfig?.authentication?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.securityConfig?.authentication?.enabled || false}
            onChange={handleAuthenticationChange}
            label="Enable authentication"
          />
          {config.securityConfig?.authentication?.enabled && (
            <div className="grid gap-4 md:grid-cols-3">
              <BubbleField label="Authentication Method">
                <BubbleSelect
                  name="method"
                  value={config.securityConfig?.authentication?.method || 'api-key'}
                  onChange={handleAuthenticationChange}
                >
                  <option value="api-key">API Key</option>
                  <option value="oauth2">OAuth 2.0</option>
                  <option value="jwt">JWT</option>
                  <option value="basic">Basic Auth</option>
                </BubbleSelect>
              </BubbleField>
              <BubbleField label="Session Timeout (minutes)">
                <BubbleInput
                  type="number"
                  name="session_timeout_minutes"
                  min="5"
                  max="1440"
                  value={config.securityConfig?.authentication(config.securityConfig?.authentication as any)?.session_timeout_minutes || 30}
                  onChange={handleAuthenticationChange}
                />
              </BubbleField>
              <BubbleField label="Max Failed Attempts">
                <BubbleInput
                  type="number"
                  name="max_failed_attempts"
                  min="1"
                  max="10"
                  value={config.securityConfig?.authentication(config.securityConfig?.authentication as any)?.max_failed_attempts || 5}
                  onChange={handleAuthenticationChange}
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <BubbleCard
        title="Data Protection Configuration"
        description="Manage encryption and retention policies."
        actions={
          <BubbleBadge tone={config.securityConfig?.data_protection?.enabled ? 'success' : 'neutral'}>
            {config.securityConfig?.data_protection?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.securityConfig?.data_protection?.enabled || false}
            onChange={handleDataProtectionChange}
            label="Enable data protection"
          />
          {config.securityConfig?.data_protection?.enabled && (
            <div className="space-y-4">
              <BubbleCard
                title="Encryption Configuration"
                description="Select encryption settings for data at rest and in transit."
                actions={
                  <BubbleBadge tone={config.securityConfig?.data_protection?.encryption?.enabled ? 'success' : 'neutral'}>
                    {config.securityConfig?.data_protection?.encryption?.enabled ? 'Enabled' : 'Disabled'}
                  </BubbleBadge>
                }
              >
                <div className="space-y-4">
                  <BubbleCheckbox
                    name="enabled"
                    checked={config.securityConfig?.data_protection?.encryption?.enabled || false}
                    onChange={handleEncryptionChange}
                    label="Enable encryption"
                  />
                  {config.securityConfig?.data_protection?.encryption?.enabled && (
                    <div className="grid gap-4 md:grid-cols-3">
                      <BubbleField label="Encryption Algorithm">
                        <BubbleSelect
                          name="algorithm"
                          value={config.securityConfig?.data_protection?.encryption?.algorithm || 'aes-256'}
                          onChange={handleEncryptionChange}
                        >
                          <option value="aes-256">AES-256</option>
                          <option value="rsa-2048">RSA-2048</option>
                          <option value="chacha20">ChaCha20</option>
                        </BubbleSelect>
                      </BubbleField>
                      <BubbleCheckbox
                        name="at_rest"
                        checked={config.securityConfig?.data_protection?.encryption?.at_rest || false}
                        onChange={handleEncryptionChange}
                        label="Encrypt data at rest"
                      />
                      <BubbleCheckbox
                        name="in_transit"
                        checked={config.securityConfig?.data_protection?.encryption?.in_transit || false}
                        onChange={handleEncryptionChange}
                        label="Encrypt data in transit"
                      />
                    </div>
                  )}
                </div>
              </BubbleCard>

              <div className="grid gap-4 md:grid-cols-2">
                <BubbleField label="Data Retention Days">
                  <BubbleInput
                    type="number"
                    name="data_retention_days"
                    min="30"
                    max="3650"
                    value={config.securityConfig?.data_protection(config.securityConfig?.encryption as any)?.data_retention_days || 365}
                    onChange={handleDataProtectionChange}
                  />
                </BubbleField>
              </div>
            </div>
          )}
        </div>
      </BubbleCard>

      <BubbleCard
        title="Compliance Configuration"
        description="Audit logging and compliance standards."
        actions={
          <BubbleBadge tone={config.securityConfig?.compliance?.enabled ? 'success' : 'neutral'}>
            {config.securityConfig?.compliance?.enabled ? 'Enabled' : 'Disabled'}
          </BubbleBadge>
        }
      >
        <div className="space-y-4">
          <BubbleCheckbox
            name="enabled"
            checked={config.securityConfig?.compliance?.enabled || false}
            onChange={handleComplianceChange}
            label="Enable compliance features"
          />
          {config.securityConfig?.compliance?.enabled && (
            <div className="space-y-4">
              <BubbleCard
                title="Audit Logging Configuration"
                description="Retention and logging levels."
                actions={
                  <BubbleBadge tone={config.securityConfig?.compliance?.audit_logging?.enabled ? 'success' : 'neutral'}>
                    {config.securityConfig?.compliance?.audit_logging?.enabled ? 'Enabled' : 'Disabled'}
                  </BubbleBadge>
                }
              >
                <div className="space-y-4">
                  <BubbleCheckbox
                    name="enabled"
                    checked={config.securityConfig?.compliance?.audit_logging?.enabled || false}
                    onChange={handleAuditLoggingChange}
                    label="Enable audit logging"
                  />
                  {config.securityConfig?.compliance?.audit_logging?.enabled && (
                    <div className="grid gap-4 md:grid-cols-2">
                      <BubbleField label="Audit Log Retention Days">
                        <BubbleInput
                          type="number"
                          name="retention_days"
                          min="30"
                          max="3650"
                          value={config.securityConfig?.compliance?.audit_logging?.retention_days || 365}
                          onChange={handleAuditLoggingChange}
                        />
                      </BubbleField>
                      <BubbleField label="Audit Log Level">
                        <BubbleSelect
                          name="log_level"
                          value={config.securityConfig?.compliance?.audit_logging?.log_level || 'info'}
                          onChange={handleAuditLoggingChange}
                        >
                          <option value="debug">Debug</option>
                          <option value="info">Info</option>
                          <option value="warn">Warn</option>
                          <option value="error">Error</option>
                        </BubbleSelect>
                      </BubbleField>
                    </div>
                  )}
                </div>
              </BubbleCard>

              <BubbleField label="Compliance Standards">
                <BubbleTextArea
                  name="standards"
                  value={config.securityConfig?.compliance(config.securityConfig?.compliance as any)?.standards || ''}
                  onChange={handleComplianceChange}
                  placeholder="GDPR, HIPAA, SOC2, etc."
                />
              </BubbleField>
            </div>
          )}
        </div>
      </BubbleCard>

      <div className="flex flex-wrap gap-2">
        <BubbleButton onClick={onValidate} variant="secondary">
          Validate security config
        </BubbleButton>
        <BubbleButton onClick={() => onConfigUpdate({ securityConfig: { ...config.securityConfig } })}>
          Save security config
        </BubbleButton>
      </div>
    </div>
  );
};

export const SecurityConfigTab = withComponentBoundary(SecurityConfigTabBase, 'SecurityConfigTab');
