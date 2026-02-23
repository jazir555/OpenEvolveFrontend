"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getLeanAideClient = getLeanAideClient;
exports.initializeLeanAideClient = initializeLeanAideClient;
exports.translateTheorem = translateTheorem;
exports.translateDefinition = translateDefinition;
exports.verifySolution = verifySolution;
exports.elaborateCode = elaborateCode;
exports.mathQuery = mathQuery;
exports.isLeanAideAvailable = isLeanAideAvailable;
const leanaideClient_1 = require("../lib/leanaideClient");
// Default configuration for LeanAIDE service
const DEFAULT_LEANAIDE_CONFIG = {
    serverUrl: import.meta.env.VITE_LEANAIDE_SERVER_URL || 'http://localhost:3000/leanaide',
    // apiKey will be set from environment or user configuration
};
// Global LeanAIDE client instance
let leanaideClient = null;
function getLeanAideClient() {
    if (!leanaideClient) {
        leanaideClient = new leanaideClient_1.LeanAideClient(DEFAULT_LEANAIDE_CONFIG);
    }
    return leanaideClient;
}
function initializeLeanAideClient(config) {
    const clientConfig = {
        serverUrl: config.serverUrl || DEFAULT_LEANAIDE_CONFIG.serverUrl,
        apiKey: config.apiKey,
    };
    leanaideClient = new leanaideClient_1.LeanAideClient(clientConfig);
}
async function translateTheorem(theoremStatement, context) {
    const client = getLeanAideClient();
    return client.translateTheorem(theoremStatement, context);
}
async function translateDefinition(definitionStatement, context) {
    const client = getLeanAideClient();
    return client.translateDefinition(definitionStatement, context);
}
async function verifySolution(problem, solution, context) {
    const client = getLeanAideClient();
    return client.verifySolution(problem, solution, context);
}
async function elaborateCode(leanCode, context) {
    const client = getLeanAideClient();
    return client.elaborateCode(leanCode, context);
}
async function mathQuery(query, context) {
    const client = getLeanAideClient();
    return client.mathQuery(query, context);
}
// Utility function to check if LeanAIDE is available
// This can be extended to check server connectivity
function isLeanAideAvailable() {
    // For now, just check if we have a client configured
    return !!leanaideClient;
}
//# sourceMappingURL=leanaideService.js.map