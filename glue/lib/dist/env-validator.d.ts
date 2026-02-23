/**
 * Environment Validator
 *
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: No magic defaults
 * - All configurable values must be via environment variables
 * - Crashes immediately if required vars are missing
 */
export type EnvType = 'string' | 'number' | 'url' | 'port' | 'boolean';
export interface EnvVar {
    name: string;
    type: EnvType;
    required: boolean;
    default?: string | number | boolean;
}
export interface ValidationResult {
    valid: boolean;
    errors: string[];
}
/**
 * Validate environment variables
 *
 * Crashes immediately if missing required vars (Law of Configuration Explicitness)
 * Validates types (URLs, ports, numbers)
 *
 * @param required - Array of required environment variable names
 * @throws Error if validation fails
 */
export declare function validateEnv(required: string[]): void;
/**
 * Validate environment variables with type checking
 *
 * @param vars - Array of environment variable definitions
 * @returns Object with validated and parsed values
 * @throws Error if validation fails
 */
export declare function validateEnvWithTypes(vars: EnvVar[]): Record<string, any>;
/**
 * Get environment variable or throw error
 *
 * Convenience function to get a single required env var with type checking
 */
export declare function getEnv(name: string, type?: EnvType): any;
/**
 * Example usage:
 *
 * ```typescript
 * import { validateEnv, validateEnvWithTypes, getEnv } from './env-validator';
 *
 * // Simple validation (just checks presence)
 * validateEnv([
 *   'DATABASE_URL',
 *   'API_KEY',
 *   'SERVICE_PORT',
 * ]);
 *
 * // Validation with type checking
 * const config = validateEnvWithTypes([
 *   { name: 'DATABASE_URL', type: 'url', required: true },
 *   { name: 'SERVICE_PORT', type: 'port', required: true },
 *   { name: 'MAX_RETRIES', type: 'number', required: false, default: 3 },
 *   { name: 'DEBUG_MODE', type: 'boolean', required: false, default: false },
 *   { name: 'API_TIMEOUT_MS', type: 'number', required: false, default: 5000 },
 * ]);
 *
 * console.log(config.DATABASE_URL);  // "postgresql://..."
 * console.log(config.SERVICE_PORT);  // 8000 (number)
 * console.log(config.MAX_RETRIES);   // 3 (number)
 * console.log(config.DEBUG_MODE);    // false (boolean)
 *
 * // Get single variable
 * const dbUrl = getEnv('DATABASE_URL', 'url');
 * const port = getEnv('SERVICE_PORT', 'port');
 *
 * // In adapter startup (Law of Configuration Explicitness):
 * try {
 *   const config = validateEnvWithTypes([
 *     { name: 'TARGET_API_URL', type: 'url', required: true },
 *     { name: 'TIMEOUT_MS', type: 'number', required: false, default: 5000 },
 *   ]);
 *   // Start service
 * } catch (error) {
 *   // Service crashes immediately with clear error message
 *   process.exit(1);
 * }
 * ```
 */
