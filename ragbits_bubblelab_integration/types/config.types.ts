/**
 * Configuration Generator Types for Ragbits + BubbleLab Integration
 */

/**
 * Options for configuring generation
 */
export interface GenerationOptions {
  /** Whether to include comments in the generated configuration */
  includeComments?: boolean;
  /** Format for the generated configuration */
  format?: 'json' | 'yaml' | 'typescript';
  /** Whether to validate the generated configuration */
  validate?: boolean;
  /** Whether to generate deployment files */
  generateDeploymentFiles?: boolean;
  /** Target environment for the configuration */
  targetEnvironment?: 'development' | 'staging' | 'production';
}

/**
 * Result of configuration generation
 */
export interface GeneratedConfig {
  /** The generated Ragbits configuration */
  ragbitsConfig: RagbitsConfig;
  /** Deployment manifest if requested */
  deploymentManifest?: any;
  /** Environment-specific configuration if applicable */
  environmentConfig?: any;
  /** Validation errors if validation was performed */
  validationErrors?: string[];
}