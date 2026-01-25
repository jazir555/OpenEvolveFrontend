/**
 * Configuration Generator Types for Ragbits + BubbleLab Integration
 */

export interface GenerationOptions {
  includeComments?: boolean;
  format?: 'json' | 'yaml' | 'typescript';
  validate?: boolean;
  generateDeploymentFiles?: boolean;
  targetEnvironment?: 'development' | 'staging' | 'production';
}

export interface GeneratedConfig {
  ragbitsConfig: RagbitsConfig;
  deploymentManifest?: any;
  environmentConfig?: any;
  validationErrors?: string[];
}