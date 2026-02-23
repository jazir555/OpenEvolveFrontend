"use strict";
/**
 * ROMA BubbleLabs Plugin - TypeScript Interfaces
 *
 * This file contains all TypeScript interfaces, types, and constants for the ROMA plugin.
 * The interfaces follow the same pattern as other BubbleLabs plugins (LeanAIDE, ClaudieMiro, Datapizza).
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.RomaPluginError = exports.DEFAULT_ROMA_CONFIG = exports.ROMA_PLUGIN_CONSTANTS = void 0;
/**
 * ROMA Plugin Constants
 */
exports.ROMA_PLUGIN_CONSTANTS = {
    DEFAULT_SERVER_URL: 'http://localhost:8000',
    DEFAULT_API_KEY: '',
    DEFAULT_PROFILE: 'general',
    DEFAULT_MAX_DEPTH: 3,
    DEFAULT_TIMEOUT: 30000,
    DEFAULT_CACHE_TTL: 3600000,
    DEFAULT_STORAGE_PATH: './roma-storage',
    DEFAULT_EXECUTION_METHOD: 'auto',
    DEFAULT_MDAP_MAKER_CONFIG: {
        enabled: true,
        autoSelect: true,
        maxDepth: 2,
        kAhead: 3,
        enableRedFlagging: true,
        enableAdaptiveK: true,
        provider: 'openai',
        model: 'gpt-4o-mini',
        autoSelectionKeywords: [
            'critical',
            'zero error',
            'flawless',
            'perfect',
            'mission-critical',
            'safety-critical',
            'high-reliability'
        ]
    },
    SUPPORTED_STRATEGIES: [
        'predict',
        'chain_of_thought',
        'react',
        'code_act',
        'best_of_n',
        'refine',
        'parallel',
        'majority'
    ],
    SUPPORTED_MODULES: [
        'atomizer',
        'planner',
        'executor',
        'aggregator',
        'verifier'
    ],
    SUPPORTED_TASK_TYPES: [
        'retrieve',
        'write',
        'think',
        'code_interpret',
        'image_generation'
    ],
    SUPPORTED_EXECUTION_METHODS: [
        'traditional',
        'claudiomiro',
        'datapizza',
        'roma',
        'hybrid',
        'roma_mdap_maker',
        'auto'
    ],
    DEFAULT_AGENT_CONFIG: {
        llm: {
            model: 'openrouter/google/gemini-2.5-flash',
            temperature: 0.6,
            cache: true
        },
        prediction_strategy: 'chain_of_thought',
        toolkits: [],
        context_defaults: {}
    }
};
/**
 * ROMA Plugin Default Configuration
 */
exports.DEFAULT_ROMA_CONFIG = {
    serverUrl: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_SERVER_URL,
    apiKey: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_API_KEY,
    defaultProfile: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_PROFILE,
    maxDepth: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_MAX_DEPTH,
    timeout: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_TIMEOUT,
    cacheTTL: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_CACHE_TTL,
    enableObservability: false,
    enableStorage: false,
    storageBasePath: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_STORAGE_PATH,
    defaultExecutionMethod: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_EXECUTION_METHOD,
    mdapMaker: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_MDAP_MAKER_CONFIG,
    agents: {
        atomizer: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_AGENT_CONFIG,
        planner: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_AGENT_CONFIG,
        executor: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_AGENT_CONFIG,
        aggregator: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_AGENT_CONFIG,
        verifier: exports.ROMA_PLUGIN_CONSTANTS.DEFAULT_AGENT_CONFIG
    },
    mcpServers: [],
    debugMode: false
};
/**
 * ROMA Plugin Error Types
 */
class RomaPluginError extends Error {
    constructor(message, code, details) {
        super(message);
        this.code = code;
        this.details = details;
        this.name = 'RomaPluginError';
    }
}
exports.RomaPluginError = RomaPluginError;
//# sourceMappingURL=plugin-types.js.map