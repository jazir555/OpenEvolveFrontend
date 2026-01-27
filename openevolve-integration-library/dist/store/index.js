"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createOpenEvolveStore = void 0;
const zustand_1 = require("zustand");
const errors_1 = require("../api/errors");
const createOpenEvolveStore = () => (0, zustand_1.create)((set, get) => ({
    client: null,
    results: {},
    loading: {},
    errors: {},
    versions: {},
    initialize: (client) => set({ client }),
    execute: async (integration, inputs) => {
        const { client, versions } = get();
        const key = integration;
        if (!client) {
            const initError = (0, errors_1.createIntegrationError)(key, new Error('Client not initialized'));
            set((state) => ({
                errors: { ...state.errors, [key]: initError }
            }));
            throw initError;
        }
        const version = (versions[key] || 0) + 1;
        set((state) => ({
            loading: { ...state.loading, [key]: true },
            errors: { ...state.errors, [key]: null },
            versions: { ...state.versions, [key]: version }
        }));
        try {
            const result = await client.execute(key, inputs);
            if (get().versions[key] === version) {
                set((state) => ({
                    results: { ...state.results, [key]: result },
                    loading: { ...state.loading, [key]: false }
                }));
            }
            return result;
        }
        catch (error) {
            const integrationError = (0, errors_1.createIntegrationError)(key, error);
            if (get().versions[key] === version) {
                set((state) => ({
                    errors: { ...state.errors, [key]: integrationError },
                    loading: { ...state.loading, [key]: false }
                }));
            }
            throw integrationError;
        }
    },
    clearResult: (integration) => set((state) => {
        const results = { ...state.results };
        const loading = { ...state.loading };
        const errors = { ...state.errors };
        const versions = { ...state.versions };
        delete results[integration];
        delete loading[integration];
        delete errors[integration];
        delete versions[integration];
        return { results, loading, errors, versions };
    }),
    reset: () => set({
        results: {},
        loading: {},
        errors: {},
        versions: {}
    })
}));
exports.createOpenEvolveStore = createOpenEvolveStore;
//# sourceMappingURL=index.js.map