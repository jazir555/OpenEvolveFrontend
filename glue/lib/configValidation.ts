/**
 * Configuration Validation - Law of Configuration Explicitness
 * Per CLAUDE.md Section 1.5: Every configurable value must be validated at startup
 * If required env vars are missing, the service crashes immediately with loud error
 */

export interface EnvConfigSchema {
  [key: string]: {
    required: boolean;
    type: 'string' | 'number' | 'boolean' | 'url' | 'positive-int';
    defaultValue?: any;
    description: string;
  };
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  config: Record<string, any>;
}

class ConfigValidator {
  private schema: EnvConfigSchema;
  private serviceName: string;

  constructor(serviceName: string, schema: EnvConfigSchema) {
    this.serviceName = serviceName;
    this.schema = schema;
  }

  private validateType(value: string, type: string): any {
    switch (type) {
      case 'string':
        return value;
      case 'number':
        const num = Number(value);
        if (isNaN(num)) {
          throw new Error(`Invalid number: ${value}`);
        }
        return num;
      case 'positive-int':
        const int = parseInt(value, 10);
        if (isNaN(int) || int <= 0) {
          throw new Error(`Invalid positive integer: ${value}`);
        }
        return int;
      case 'boolean':
        if (value.toLowerCase() === 'true') return true;
        if (value.toLowerCase() === 'false') return false;
        throw new Error(`Invalid boolean: ${value}`);
      case 'url':
        try {
          new URL(value);
          return value;
        } catch {
          throw new Error(`Invalid URL: ${value}`);
        }
      default:
        return value;
    }
  }

  validate(): Record<string, any> {
    const errors: string[] = [];
    const warnings: string[] = [];
    const config: Record<string, any> = {};

    // Check each required config value
    for (const [key, spec] of Object.entries(this.schema)) {
      const envValue = process.env[key];

      if (spec.required && !envValue) {
        errors.push(`Missing required environment variable: ${key} - ${spec.description}`);
        continue;
      }

      if (!envValue) {
        if (spec.defaultValue !== undefined) {
          config[key] = spec.defaultValue;
          warnings.push(`Using default value for ${key}: ${spec.defaultValue}`);
        }
        continue;
      }

      try {
        config[key] = this.validateType(envValue, spec.type);
      } catch (err) {
        errors.push(`Invalid ${key}: ${(err as Error).message}`);
      }
    }

    // If there are errors, crash immediately with loud error
    if (errors.length > 0) {
      console.error(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'FATAL',
        msg: `Configuration validation failed for ${this.serviceName}`,
        service: this.serviceName,
        errors
      }));
      throw new Error(
        `Configuration validation failed for ${this.serviceName}:\n` +
        errors.map(e => `  - ${e}`).join('\n') +
        '\nService cannot start. Fix the configuration and retry.'
      );
    }

    // Log warnings but continue
    if (warnings.length > 0) {
      console.warn(JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'WARN',
        msg: `Configuration warnings for ${this.serviceName}`,
        service: this.serviceName,
        warnings
      }));
    }

    // Log successful validation
    console.info(JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      msg: `Configuration validated successfully for ${this.serviceName}`,
      service: this.serviceName,
      config_keys: Object.keys(config)
    }));

    return config;
  }
}

// Define schemas for different services

export const FRONTEND_CONFIG_SCHEMA: EnvConfigSchema = {
  // API Configuration
  OPENEVOLVE_API_BASE: {
    required: true,
    type: 'url',
    description: 'Base URL for OpenEvolve API (e.g., https://api.openevolve.com)'
  },
  OPENEVOLVE_API_KEY: {
    required: true,
    type: 'string',
    description: 'API key for OpenEvolve authentication'
  },

  // Plugin Configuration
  RAGBITS_API_URL: {
    required: false,
    type: 'url',
    description: 'RAGBits server URL for document search',
    defaultValue: 'http://localhost:8001'
  },
  RAGBITS_API_KEY: {
    required: false,
    type: 'string',
    description: 'RAGBits API key'
  },
  RAGBITS_TIMEOUT: {
    required: false,
    type: 'positive-int',
    description: 'RAGBits request timeout in milliseconds',
    defaultValue: 30000
  },

  LEANAIDE_API_URL: {
    required: false,
    type: 'url',
    description: 'LeanAide API URL for formal proofs',
    defaultValue: 'http://localhost:7654'
  },
  LEANAIDE_API_KEY: {
    required: false,
    type: 'string',
    description: 'LeanAide API key'
  },

  // Datapizza Configuration
  DATAPIZZA_API_URL: {
    required: false,
    type: 'url',
    description: 'Datapizza API URL for data processing',
    defaultValue: '/api/datapizza'
  },
  DATAPIZZA_API_KEY: {
    required: false,
    type: 'string',
    description: 'Datapizza API key'
  },
  DATAPIZZA_TIMEOUT: {
    required: false,
    type: 'positive-int',
    description: 'Datapizza request timeout in milliseconds',
    defaultValue: 30000
  },

  // timeouts
  DEFAULT_REQUEST_TIMEOUT: {
    required: false,
    type: 'positive-int',
    description: 'Default HTTP request timeout in milliseconds',
    defaultValue: 30000
  },

  // Feature flags
  ENABLE_DEBUG_LOGGING: {
    required: false,
    type: 'boolean',
    description: 'Enable debug logging',
    defaultValue: false
  },
  ENABLE_TELEMETRY: {
    required: false,
    type: 'boolean',
    description: 'Enable anonymous telemetry',
    defaultValue: true
  }
};

/**
 * Validate all frontend configuration at startup
 * Throws error if configuration is invalid, causing fail-fast
 */
export function validateFrontendConfig(): Record<string, any> {
  const validator = new ConfigValidator('frontend', FRONTEND_CONFIG_SCHEMA);
  return validator.validate();
}

/**
 * Validate configuration for a specific plugin
 */
export function validatePluginConfig(pluginName: string, schema: EnvConfigSchema): Record<string, any> {
  const validator = new ConfigValidator(pluginName, schema);
  return validator.validate();
}

// Auto-validate on import if in Node.js environment
if (typeof process !== 'undefined' && process.env) {
  try {
    const config = validateFrontendConfig();
    // Store validated config globally
    (global as any).__VALIDATED_CONFIG = config;
  } catch (err) {
    // Error already logged, re-throw to crash the process
    throw err;
  }
}

/**
 * Get validated config value
 */
export function getConfig(key: string): any {
  const config = (global as any).__VALIDATED_CONFIG;
  if (!config) {
    throw new Error('Configuration not validated. Call validateFrontendConfig() first.');
  }
  return config[key];
}

export { ConfigValidator };
