"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.LeanAidePlugin = exports.DEFAULT_LEANAIDE_PLUGIN_CONFIG = void 0;
exports.registerLeanAidePlugin = registerLeanAidePlugin;
const react_1 = __importStar(require("react"));
const lucide_react_1 = require("lucide-react");
const autoformalizationAnalytics_1 = require("../integration/autoformalizationAnalytics");
exports.DEFAULT_LEANAIDE_PLUGIN_CONFIG = {
    serverUrl: 'http://localhost:3000/leanaide',
    ragbitsUrl: 'http://localhost:3000/ragbits',
    enableAnalytics: true,
    defaultDomain: 'general',
    defaultStrategy: 'auto',
    analyticsRefreshInterval: 5000,
    maxConcurrentRequests: 5,
    cacheEnabled: true,
    cacheTTL: 3600,
};
const LeanAidePlugin = ({ config, onConfigChange, className = '', }) => {
    const resolvedConfig = (0, react_1.useMemo)(() => ({ ...exports.DEFAULT_LEANAIDE_PLUGIN_CONFIG, ...config }), [config]);
    (0, react_1.useEffect)(() => {
        onConfigChange?.(resolvedConfig);
    }, [onConfigChange, resolvedConfig]);
    return <autoformalizationAnalytics_1.LeanAideBubbleLabIntegration className={className}/>;
};
exports.LeanAidePlugin = LeanAidePlugin;
function registerLeanAidePlugin() {
    return {
        id: 'leanaide-autoformalization',
        name: 'LeanAide Autoformalization',
        description: 'Convert natural language mathematical statements to formal Lean 4 code with analytics.',
        version: '1.0.0',
        category: 'formalization',
        component: exports.LeanAidePlugin,
        icon: <lucide_react_1.Brain className="h-4 w-4"/>,
        settingsSchema: {
            type: 'object',
            properties: {
                serverUrl: { type: 'string', default: exports.DEFAULT_LEANAIDE_PLUGIN_CONFIG.serverUrl },
                ragbitsUrl: { type: 'string', default: exports.DEFAULT_LEANAIDE_PLUGIN_CONFIG.ragbitsUrl },
                enableAnalytics: { type: 'boolean', default: true },
            },
        },
        permissions: ['network', 'storage'],
    };
}
exports.default = exports.LeanAidePlugin;
//# sourceMappingURL=LeanAidePlugin.js.map