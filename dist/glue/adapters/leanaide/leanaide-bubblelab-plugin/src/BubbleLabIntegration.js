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
exports.registerBubbleLabIntegration = exports.BubbleLabLeanAideIntegrationLazy = exports.BubbleLabLeanAideIntegration = void 0;
const react_1 = __importStar(require("react"));
const lucide_react_1 = require("lucide-react");
const autoformalizationAnalytics_1 = require("./integration/autoformalizationAnalytics");
const PluginInterface_1 = require("./PluginInterface");
const BubbleLabLeanAideIntegration = ({ className = '' }) => {
    return <autoformalizationAnalytics_1.LeanAideBubbleLabIntegration className={className}/>;
};
exports.BubbleLabLeanAideIntegration = BubbleLabLeanAideIntegration;
const LazyBubbleLabIntegration = react_1.default.lazy(async () => ({
    default: exports.BubbleLabLeanAideIntegration,
}));
const BubbleLabLeanAideIntegrationLazy = (props) => {
    return (<react_1.Suspense fallback={<div className="p-4 text-sm text-gray-500">Loading LeanAide integration...</div>}>
      <LazyBubbleLabIntegration {...props}/>
    </react_1.Suspense>);
};
exports.BubbleLabLeanAideIntegrationLazy = BubbleLabLeanAideIntegrationLazy;
const registerBubbleLabIntegration = () => {
    return {
        id: 'bubblelab-leanaide-integration',
        name: 'BubbleLab LeanAide Integration',
        description: 'LeanAide formalization workflows integrated into BubbleLab.',
        version: '1.0.0',
        category: 'integration',
        component: exports.BubbleLabLeanAideIntegration,
        icon: <lucide_react_1.Brain className="h-4 w-4"/>,
        settingsSchema: {
            type: 'object',
            properties: {
                serverUrl: { type: 'string', default: 'http://localhost:3000/leanaide' },
                ragbitsUrl: { type: 'string', default: 'http://localhost:3000/ragbits' },
                enableAnalytics: { type: 'boolean', default: true },
            },
        },
        permissions: ['network', 'storage'],
        dependencies: ['bubblelab-core'],
    };
};
exports.registerBubbleLabIntegration = registerBubbleLabIntegration;
if (!PluginInterface_1.pluginRegistry.getPlugin('bubblelab-leanaide-integration')) {
    PluginInterface_1.pluginRegistry.register((0, exports.registerBubbleLabIntegration)());
    void PluginInterface_1.pluginRegistry.activate('bubblelab-leanaide-integration');
}
exports.default = exports.BubbleLabLeanAideIntegration;
//# sourceMappingURL=BubbleLabIntegration.js.map