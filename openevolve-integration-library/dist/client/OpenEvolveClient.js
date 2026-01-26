"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.OpenEvolveClient = void 0;
const client_1 = require("../api/client");
const decomposition_1 = require("../integrations/decomposition");
const leanaide_1 = require("../integrations/leanaide");
const evolution_1 = require("../integrations/evolution");
const knowledge_1 = require("../integrations/knowledge");
const maker_1 = require("../integrations/maker");
const hephaestus_1 = require("../integrations/hephaestus");
class OpenEvolveClient {
    constructor(config) {
        this.config = config;
        this.apiClient = new client_1.ApiClient(config);
        this.integrations = {
            decomposition: new decomposition_1.DecompositionIntegration(this.apiClient),
            leanaide: new leanaide_1.LeanAideIntegration(this.apiClient),
            evolution: new evolution_1.EvolutionIntegration(this.apiClient),
            knowledge: new knowledge_1.KnowledgeIntegration(this.apiClient),
            maker: new maker_1.MakerIntegration(this.apiClient),
            hephaestus: new hephaestus_1.HephaestusIntegration(this.apiClient)
        };
    }
    updateConfig(config) {
        this.apiClient.updateConfig(config);
        Object.assign(this.config, config);
    }
    getConfig() {
        return { ...this.config };
    }
    disconnect() {
        if (this.config.debug) {
            console.log('[OpenEvolve] Client disconnected');
        }
    }
    async healthCheck() {
        const results = {};
        for (const [name, integration] of Object.entries(this.integrations)) {
            try {
                await this.apiClient.get(`/api/v1/${name}/health`);
                results[name] = true;
            }
            catch (error) {
                results[name] = false;
                if (this.config.debug) {
                    console.error(`[OpenEvolve] Health check failed for ${name}:`, error);
                }
            }
        }
        return results;
    }
    async getVersions() {
        const versions = {};
        for (const [name, integration] of Object.entries(this.integrations)) {
            versions[name] = integration.version || 'unknown';
        }
        return versions;
    }
    async batchExecute(operations) {
        const results = await Promise.allSettled(operations.map(async ({ integration, inputs }) => {
            const integrationInstance = this.integrations[integration];
            return integrationInstance.execute(inputs);
        }));
        return results.map((result, index) => {
            if (result.status === 'fulfilled') {
                return result.value;
            }
            else {
                if (this.config.debug) {
                    console.error(`[OpenEvolve] Batch operation ${index} failed:`, result.reason);
                }
                throw result.reason;
            }
        });
    }
    static create(config) {
        return new OpenEvolveClient(config);
    }
}
exports.OpenEvolveClient = OpenEvolveClient;
//# sourceMappingURL=OpenEvolveClient.js.map