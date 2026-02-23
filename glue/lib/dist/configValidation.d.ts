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
declare class ConfigValidator {
    private schema;
    private serviceName;
    constructor(serviceName: string, schema: EnvConfigSchema);
    private validateType;
    validate(): Record<string, any>;
}
export declare const FRONTEND_CONFIG_SCHEMA: EnvConfigSchema;
/**
 * Validate all frontend configuration at startup
 * Throws error if configuration is invalid, causing fail-fast
 */
export declare function validateFrontendConfig(): Record<string, any>;
/**
 * Validate configuration for a specific plugin
 */
export declare function validatePluginConfig(pluginName: string, schema: EnvConfigSchema): Record<string, any>;
/**
 * Get validated config value
 */
export declare function getConfig(key: string): any;
export { ConfigValidator };
//# sourceMappingURL=configValidation.d.ts.map