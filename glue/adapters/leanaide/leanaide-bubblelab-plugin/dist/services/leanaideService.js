import { LeanAideClient } from '../lib/leanaideClient';
// Default configuration for LeanAIDE service
const DEFAULT_LEANAIDE_CONFIG = {
    serverUrl: import.meta.env.VITE_LEANAIDE_SERVER_URL || 'http://localhost:3000/leanaide',
    // apiKey will be set from environment or user configuration
};
// Global LeanAIDE client instance
let leanaideClient = null;
export function getLeanAideClient() {
    if (!leanaideClient) {
        leanaideClient = new LeanAideClient(DEFAULT_LEANAIDE_CONFIG);
    }
    return leanaideClient;
}
export function initializeLeanAideClient(config) {
    const clientConfig = {
        serverUrl: config.serverUrl || DEFAULT_LEANAIDE_CONFIG.serverUrl,
        apiKey: config.apiKey,
    };
    leanaideClient = new LeanAideClient(clientConfig);
}
export async function translateTheorem(theoremStatement, context) {
    const client = getLeanAideClient();
    return client.translateTheorem(theoremStatement, context);
}
export async function translateDefinition(definitionStatement, context) {
    const client = getLeanAideClient();
    return client.translateDefinition(definitionStatement, context);
}
export async function verifySolution(problem, solution, context) {
    const client = getLeanAideClient();
    return client.verifySolution(problem, solution, context);
}
export async function elaborateCode(leanCode, context) {
    const client = getLeanAideClient();
    return client.elaborateCode(leanCode, context);
}
export async function mathQuery(query, context) {
    const client = getLeanAideClient();
    return client.mathQuery(query, context);
}
// Utility function to check if LeanAIDE is available
// This can be extended to check server connectivity
export function isLeanAideAvailable() {
    // For now, just check if we have a client configured
    return !!leanaideClient;
}
//# sourceMappingURL=leanaideService.js.map