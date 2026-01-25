import { CREDENTIAL_ENV_MAP } from '@bubblelab/shared-schemas';
import { CredentialType } from '@bubblelab/shared-schemas';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Utility for managing .env file entries
 * Allows adding and removing environment variables while preserving comments and formatting
 */
export class EnvFileManager {
  private envPath: string;

  constructor(envPath?: string) {
    // If no path provided, search for .env file in common locations
    if (envPath) {
      this.envPath = envPath;
    } else {
      this.envPath = this.findEnvFile() || '.env';
    }
  }

  /**
   * Find the .env file by searching in common locations
   * Matches the logic in env.ts
   */
  private findEnvFile(): string | null {
    const searchPaths = [
      path.join(process.cwd(), '.env'), // Current dir
      path.join(process.cwd(), '../.env'), // One up (apps level)
      path.join(process.cwd(), '../../.env'), // Two up (monorepo root)
    ];

    for (const searchPath of searchPaths) {
      if (fs.existsSync(searchPath)) {
        return searchPath;
      }
    }

    return null;
  }

  /**
   * Remove an environment variable from the .env file
   * @param envVarName - The name of the environment variable to remove (e.g., "DEEPSEEK_API_KEY")
   * @returns true if removed, false if not found
   */
  removeEnvVar(envVarName: string): boolean {
    if (!fs.existsSync(this.envPath)) {
      console.warn(`[EnvFileManager] .env file not found: ${this.envPath}`);
      return false;
    }

    try {
      const content = fs.readFileSync(this.envPath, 'utf-8');
      const lines = content.split('\n');

      // Find and remove the line containing this env var
      const newLines = lines.filter((line) => {
        const trimmed = line.trim();
        // Skip empty lines and comments
        if (!trimmed || trimmed.startsWith('#')) {
          return true;
        }
        // Remove the env var line
        return !trimmed.startsWith(`${envVarName}=`);
      });

      // Only write if something changed
      if (newLines.length !== lines.length) {
        fs.writeFileSync(this.envPath, newLines.join('\n'), 'utf-8');
        console.log(`[EnvFileManager] Removed ${envVarName} from .env file (${this.envPath})`);
        return true;
      }

      return false;
    } catch (error) {
      console.error(`[EnvFileManager] Failed to remove ${envVarName} from .env:`, error);
      return false;
    }
  }

  /**
   * Check if an environment variable exists in the .env file
   * @param envVarName - The name of the environment variable to check
   * @returns true if the env var exists in .env
   */
  hasEnvVar(envVarName: string): boolean {
    if (!fs.existsSync(this.envPath)) {
      return false;
    }

    try {
      const content = fs.readFileSync(this.envPath, 'utf-8');
      const lines = content.split('\n');

      return lines.some((line) => {
        const trimmed = line.trim();
        return trimmed.startsWith(`${envVarName}=`);
      });
    } catch {
      return false;
    }
  }

  /**
   * Get the path to the .env file for a credential type
   * @param credentialType - The credential type to get the env path for
   * @returns The environment variable name, or undefined if not mapped
   */
  static getEnvVarForCredential(credentialType: CredentialType): string | undefined {
    return CREDENTIAL_ENV_MAP[credentialType];
  }
}
