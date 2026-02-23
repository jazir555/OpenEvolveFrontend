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
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.registerLeanAidePlugin = exports.usePluginManager = exports.pluginRegistry = exports.PluginManagerProvider = exports.PluginManager = exports.LeanAidePlugin = exports.DEFAULT_ANALYTICS_CONFIG = exports.useAutoformalizationAnalytics = exports.KnowledgeGraphIntegration = exports.AnalyticsDashboard = exports.EnhancedLeanAideVerification = exports.autoformalize_with_mdap_maker = exports.create_leanaide_autoformalization_engine = exports.LeanAideAutoformalizationEngine = exports.registerBubbleLabIntegration = exports.BubbleLabLeanAideIntegrationLazy = exports.BubbleLabLeanAideIntegration = exports.LeanAideBubbleLabIntegration = void 0;
var BubbleLabIntegration_1 = require("./BubbleLabIntegration");
Object.defineProperty(exports, "LeanAideBubbleLabIntegration", { enumerable: true, get: function () { return __importDefault(BubbleLabIntegration_1).default; } });
Object.defineProperty(exports, "BubbleLabLeanAideIntegration", { enumerable: true, get: function () { return BubbleLabIntegration_1.BubbleLabLeanAideIntegration; } });
Object.defineProperty(exports, "BubbleLabLeanAideIntegrationLazy", { enumerable: true, get: function () { return BubbleLabIntegration_1.BubbleLabLeanAideIntegrationLazy; } });
Object.defineProperty(exports, "registerBubbleLabIntegration", { enumerable: true, get: function () { return BubbleLabIntegration_1.registerBubbleLabIntegration; } });
var autoformalizationAnalytics_1 = require("./integration/autoformalizationAnalytics");
Object.defineProperty(exports, "LeanAideAutoformalizationEngine", { enumerable: true, get: function () { return autoformalizationAnalytics_1.LeanAideAutoformalizationEngine; } });
Object.defineProperty(exports, "create_leanaide_autoformalization_engine", { enumerable: true, get: function () { return autoformalizationAnalytics_1.create_leanaide_autoformalization_engine; } });
Object.defineProperty(exports, "autoformalize_with_mdap_maker", { enumerable: true, get: function () { return autoformalizationAnalytics_1.autoformalize_with_mdap_maker; } });
Object.defineProperty(exports, "EnhancedLeanAideVerification", { enumerable: true, get: function () { return autoformalizationAnalytics_1.EnhancedLeanAideVerification; } });
Object.defineProperty(exports, "AnalyticsDashboard", { enumerable: true, get: function () { return autoformalizationAnalytics_1.AnalyticsDashboard; } });
Object.defineProperty(exports, "KnowledgeGraphIntegration", { enumerable: true, get: function () { return autoformalizationAnalytics_1.KnowledgeGraphIntegration; } });
Object.defineProperty(exports, "useAutoformalizationAnalytics", { enumerable: true, get: function () { return autoformalizationAnalytics_1.useAutoformalizationAnalytics; } });
Object.defineProperty(exports, "DEFAULT_ANALYTICS_CONFIG", { enumerable: true, get: function () { return autoformalizationAnalytics_1.DEFAULT_ANALYTICS_CONFIG; } });
var PluginSystem_1 = require("./PluginSystem");
Object.defineProperty(exports, "LeanAidePlugin", { enumerable: true, get: function () { return PluginSystem_1.LeanAidePlugin; } });
Object.defineProperty(exports, "PluginManager", { enumerable: true, get: function () { return PluginSystem_1.PluginManager; } });
Object.defineProperty(exports, "PluginManagerProvider", { enumerable: true, get: function () { return PluginSystem_1.PluginManagerProvider; } });
Object.defineProperty(exports, "pluginRegistry", { enumerable: true, get: function () { return PluginSystem_1.pluginRegistry; } });
Object.defineProperty(exports, "usePluginManager", { enumerable: true, get: function () { return PluginSystem_1.usePluginManager; } });
var LeanAidePlugin_1 = require("./plugins/LeanAidePlugin");
Object.defineProperty(exports, "registerLeanAidePlugin", { enumerable: true, get: function () { return LeanAidePlugin_1.registerLeanAidePlugin; } });
__exportStar(require("./services"), exports);
//# sourceMappingURL=index.js.map