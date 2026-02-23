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
export interface AdapterConfig {
    port: number;
    timeout: number;
    maxRetries: number;
    logLevel: string;
    apiUrl?: string;
    targetUrl?: string;
    apiKey?: string;
    enableTracing: boolean;
    debugMode: boolean;
}
/**
 * Validate adapter-specific environment variables at startup
 *
 * @param adapterName - Name of the adapter (e.g., 'graphiti', 'bubblelab')
 * @returns Validated configuration object
 * @throws Error if validation fails (crashes the application)
 */
export declare function validateAdapterStartup(adapterName: string): AdapterConfig;
/**
 * Validate ALL environment variables (strict mode)
 *
 * Useful for testing or global validation tools
 */
export declare function validateGlobalConfig(): Record<string, any>;
/**
 * Example: Custom validation for specific adapter
 *
 * This shows how to add custom validation logic beyond type checking
 */
export declare function validateGraphitiAdapter(): GraphitiAdapterConfig;
export interface GraphitiAdapterConfig {
    port: number;
    neo4j: {
        uri: string;
        user: string;
        password: string;
    };
    llm: {
        openaiApiKey?: string;
        anthropicApiKey?: string;
    };
    timeout: number;
    updateCommunities: boolean;
    storeRawEpisodes: boolean;
}
/**
 * Example: Minimal adapter startup
 *
 * This is the simplest way to validate env vars at startup
 */
export declare function minimalAdapterStartup(adapterName: string): void;
/**
 * Example: Validate multiple components
 *
 * Useful for orchestration services that depend on multiple adapters
 */
export declare function validateMultipleComponents(components: string[]): Record<string, any>;
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
//# sourceMappingURL=startup-validation.d.ts.map