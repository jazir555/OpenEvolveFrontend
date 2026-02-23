"use strict";
/**
 * Startup Validation Example
 *
 * Demonstrates how to validate environment variables at adapter startup
 * Following the Federation Constitution - Law of Configuration Explicitness
 *
 * Usage:
 *   import { validateAdapterStartup } from './startup-validation';
 *
 *   // In your adapter's main/index.ts:
 *   try {
 *     const config = validateAdapterStartup('graphiti');
 *     // Start adapter with validated config
 *   } catch (error) {
 *     // Logger already logged the error
 *     // Crash immediately - don't start with invalid config
 *     process.exit(1);
 *   }
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateAdapterStartup = validateAdapterStartup;
exports.validateGlobalConfig = validateGlobalConfig;
exports.validateGraphitiAdapter = validateGraphitiAdapter;
exports.minimalAdapterStartup = minimalAdapterStartup;
exports.validateMultipleComponents = validateMultipleComponents;
const logger_1 = require("./logger");
const env_validator_1 = require("./env-validator");
const env_schema_1 = require("./env-schema");
/**
 * Validate adapter-specific environment variables at startup
 *
 * @param adapterName - Name of the adapter (e.g., 'graphiti', 'bubblelab')
 * @returns Validated configuration object
 * @throws Error if validation fails (crashes the application)
 */
function validateAdapterStartup(adapterName) {
    logger_1.logger.info(`Validating configuration for ${adapterName} adapter`, {
        component: adapterName,
        phase: 'startup_validation',
    });
    try {
        // Get schema for this specific adapter
        const schema = (0, env_schema_1.getSchemaForComponent)(adapterName);
        if (schema.length === 0) {
            throw new Error(`Unknown adapter: ${adapterName}`);
        }
        // Validate environment variables with type checking
        const envVars = (0, env_validator_1.validateEnvWithTypes)(schema);
        // Build and return typed configuration object
        const config = buildConfig(envVars, adapterName);
        logger_1.logger.info(`${adapterName} adapter configuration validated successfully`, {
            component: adapterName,
            config: {
                port: config.port,
                timeout: config.timeout,
                hasTargetUrl: !!config.targetUrl,
                hasApiKey: !!config.apiKey,
            },
        });
        return config;
    }
    catch (error) {
        // Error already logged by validateEnvWithTypes
        // Crash immediately with clear error message
        logger_1.logger.error(`Configuration validation failed for ${adapterName} adapter`, error, {
            component: adapterName,
            phase: 'startup_validation',
            fatal: true,
        });
        // Re-throw to allow caller to handle exit
        throw error;
    }
}
/**
 * Validate ALL environment variables (strict mode)
 *
 * Useful for testing or global validation tools
 */
function validateGlobalConfig() {
    logger_1.logger.info('Validating global configuration', {
        phase: 'global_validation',
    });
    try {
        const config = (0, env_validator_1.validateEnvWithTypes)(env_schema_1.ALL_ENV_VARS);
        logger_1.logger.info('Global configuration validated', {
            total_vars: Object.keys(config).length,
        });
        return config;
    }
    catch (error) {
        logger_1.logger.error('Global configuration validation failed', error, {
            phase: 'global_validation',
            fatal: true,
        });
        throw error;
    }
}
/**
 * Build typed configuration object from validated env vars
 */
function buildConfig(envVars, adapterName) {
    // Extract common configuration
    const config = {
        port: envVars[`${adapterName.toUpperCase()}_PORT`] || 3000,
        timeout: envVars[`${adapterName.toUpperCase()}_TIMEOUT_MS`] || 30000,
        maxRetries: envVars[`${adapterName.toUpperCase()}_MAX_RETRIES`] || 3,
        logLevel: envVars.LOG_LEVEL || 'INFO',
        enableTracing: envVars.ENABLE_TRACING || false,
        debugMode: envVars.DEBUG || false,
    };
    // Extract adapter-specific configuration
    const apiUrlKey = `${adapterName.toUpperCase()}_API_URL`;
    const apiKeyKey = `${adapterName.toUpperCase()}_API_KEY`;
    if (envVars[apiUrlKey]) {
        config.apiUrl = envVars[apiUrlKey];
        config.targetUrl = envVars[apiUrlKey];
    }
    if (envVars[apiKeyKey]) {
        config.apiKey = envVars[apiKeyKey];
    }
    return config;
}
/**
 * Example: Custom validation for specific adapter
 *
 * This shows how to add custom validation logic beyond type checking
 */
function validateGraphitiAdapter() {
    logger_1.logger.info('Validating Graphiti adapter configuration', {
        component: 'graphiti',
    });
    try {
        // First, validate with schema
        const envVars = (0, env_validator_1.validateEnvWithTypes)((0, env_schema_1.getSchemaForComponent)('graphiti'));
        // Custom validation: Check if Neo4j is accessible
        if (envVars.NEO4J_URI) {
            // Could add connection check here
            logger_1.logger.info('Neo4j URI validated', {
                uri: envVars.NEO4J_URI,
            });
        }
        // Custom validation: Warn if no LLM API key provided
        if (!envVars.OPENAI_API_KEY && !envVars.ANTHROPIC_API_KEY) {
            logger_1.logger.warn('No LLM API key provided - entity extraction will be disabled', {
                component: 'graphiti',
                has_openai_key: !!envVars.OPENAI_API_KEY,
                has_anthropic_key: !!envVars.ANTHROPIC_API_KEY,
            });
        }
        return {
            port: envVars.GRAPHITI_PORT || 3000,
            neo4j: {
                uri: envVars.NEO4J_URI,
                user: envVars.NEO4J_USER || 'neo4j',
                password: envVars.NEO4J_PASSWORD,
            },
            llm: {
                openaiApiKey: envVars.OPENAI_API_KEY,
                anthropicApiKey: envVars.ANTHROPIC_API_KEY,
            },
            timeout: envVars.GRAPHITI_TIMEOUT_MS || 30000,
            updateCommunities: envVars.UPDATE_COMMUNITIES || false,
            storeRawEpisodes: envVars.STORE_RAW_EPISODES !== false,
        };
    }
    catch (error) {
        logger_1.logger.error('Graphiti adapter configuration validation failed', error, {
            component: 'graphiti',
            fatal: true,
        });
        throw error;
    }
}
/**
 * Example: Minimal adapter startup
 *
 * This is the simplest way to validate env vars at startup
 */
function minimalAdapterStartup(adapterName) {
    try {
        const schema = (0, env_schema_1.getSchemaForComponent)(adapterName);
        (0, env_validator_1.validateEnvWithTypes)(schema);
        logger_1.logger.info(`${adapterName} adapter started with valid configuration`, {
            component: adapterName,
        });
        // Continue with adapter initialization...
    }
    catch (error) {
        // Application crashes immediately - Law of Configuration Explicitness
        process.exit(1);
    }
}
/**
 * Example: Validate multiple components
 *
 * Useful for orchestration services that depend on multiple adapters
 */
function validateMultipleComponents(components) {
    logger_1.logger.info('Validating configuration for multiple components', {
        components,
    });
    const configs = {};
    const schemas = components.map(env_schema_1.getSchemaForComponent);
    const allVars = schemas.flat();
    try {
        const envVars = (0, env_validator_1.validateEnvWithTypes)(allVars);
        for (const component of components) {
            configs[component] = buildConfig(envVars, component);
        }
        logger_1.logger.info('All component configurations validated', {
            components: Object.keys(configs),
        });
        return configs;
    }
    catch (error) {
        logger_1.logger.error('Multi-component configuration validation failed', error, {
            components,
            fatal: true,
        });
        throw error;
    }
}
/**
 * Usage Examples
 *
 * ```typescript
 * // Example 1: Simple adapter startup
 * import { validateAdapterStartup } from './startup-validation';
 *
 * try {
 *   const config = validateAdapterStartup('graphiti');
 *   // Start adapter with config.port, config.timeout, etc.
 * } catch (error) {
 *   process.exit(1);
 * }
 *
 * // Example 2: Custom Graphiti validation
 * import { validateGraphitiAdapter } from './startup-validation';
 *
 * try {
 *   const config = validateGraphitiAdapter();
 *   // Start Graphiti adapter with config.neo4j, config.llm, etc.
 * } catch (error) {
 *   process.exit(1);
 * }
 *
 * // Example 3: Multiple components
 * import { validateMultipleComponents } from './startup-validation';
 *
 * try {
 *   const configs = validateMultipleComponents(['graphiti', 'bubblelab', 'vectordb']);
 *   // Start orchestration service
 * } catch (error) {
 *   process.exit(1);
 * }
 *
 * // Example 4: Global validation
 * import { validateGlobalConfig } from './startup-validation';
 *
 * try {
 *   const config = validateGlobalConfig();
 *   // Access config.SECRET_KEY, config.DATABASE_URL, etc.
 * } catch (error) {
 *   process.exit(1);
 * }
 * ```
 */
//# sourceMappingURL=startup-validation.js.map