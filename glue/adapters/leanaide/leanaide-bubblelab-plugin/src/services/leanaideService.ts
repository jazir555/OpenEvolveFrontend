import { LeanAideClient, LeanAideTaskResponse } from '../lib/leanaideClient';

// Default configuration for LeanAIDE service
const DEFAULT_LEANAIDE_CONFIG = {
  serverUrl: import.meta.env.VITE_LEANAIDE_SERVER_URL || 'http://localhost:3000/leanaide',
  // apiKey will be set from environment or user configuration
};

// Global LeanAIDE client instance
let leanaideClient: LeanAideClient | null = null;

export function getLeanAideClient(): LeanAideClient {
  if (!leanaideClient) {
    leanaideClient = new LeanAideClient(DEFAULT_LEANAIDE_CONFIG);
  }
  return leanaideClient;
}

export function initializeLeanAideClient(config: { serverUrl?: string; apiKey?: string }) {
  const clientConfig = {
    serverUrl: config.serverUrl || DEFAULT_LEANAIDE_CONFIG.serverUrl,
    apiKey: config.apiKey,
  };
  leanaideClient = new LeanAideClient(clientConfig);
}

export async function translateTheorem(
  theoremStatement: string,
  context?: string
): Promise<LeanAideTaskResponse> {
  const client = getLeanAideClient();
  return client.translateTheorem(theoremStatement, context);
}

export async function translateDefinition(
  definitionStatement: string,
  context?: string
): Promise<LeanAideTaskResponse> {
  const client = getLeanAideClient();
  return client.translateDefinition(definitionStatement, context);
}

export async function verifySolution(
  problem: string,
  solution: string,
  context?: string
): Promise<LeanAideTaskResponse> {
  const client = getLeanAideClient();
  return client.verifySolution(problem, solution, context);
}

export async function elaborateCode(
  leanCode: string,
  context?: string
): Promise<LeanAideTaskResponse> {
  const client = getLeanAideClient();
  return client.elaborateCode(leanCode, context);
}

export async function mathQuery(
  query: string,
  context?: string
): Promise<LeanAideTaskResponse> {
  const client = getLeanAideClient();
  return client.mathQuery(query, context);
}

// Utility function to check if LeanAIDE is available
// This can be extended to check server connectivity
export function isLeanAideAvailable(): boolean {
  // For now, just check if we have a client configured
  return !!leanaideClient;
}
