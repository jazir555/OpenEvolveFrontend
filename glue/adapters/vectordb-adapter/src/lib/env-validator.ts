/**
 * Environment variable validation - Law of Configuration Explicitness.
 */

export interface EnvValidationResult {
  valid: boolean;
  missing: string[];
}

export function validateEnvVars(
  required: string[],
  env: Record<string, string | undefined> | NodeJS.ProcessEnv
): EnvValidationResult {
  const missing = required.filter((key) => !env[key]);
  return {
    valid: missing.length === 0,
    missing,
  };
}
