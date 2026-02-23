"use strict";
/**
 * OpenEvolve Library Exports
 *
 * Central export point for all OpenEvolve libraries and integrations.
 */
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.useWorkflowOrchestrator = exports.usePluginRegistry = exports.useBubbleLabIntegrationInstance = exports.useBubbleLabIntegration = exports.openevolveApi = exports.resetBubbleLabIntegration = exports.getBubbleLabIntegration = exports.initializeBubbleLabIntegration = exports.BubbleLabIntegration = exports.resetWorkflowMonitor = exports.getWorkflowMonitor = exports.WorkflowMonitor = exports.resetPluginEventIntegration = exports.getPluginEventIntegration = exports.PluginEventIntegration = exports.getWorkflowTemplatesByCategory = exports.getAllWorkflowTemplates = exports.getWorkflowTemplate = exports.WORKFLOW_TEMPLATES = exports.PROBLEM_SOLVING_WORKFLOW = exports.KNOWLEDGE_EXTRACTION_WORKFLOW = exports.PROOF_VERIFICATION_WORKFLOW = exports.DATA_ANALYSIS_PIPELINE = exports.RESEARCH_ASSISTANT_WORKFLOW = exports.getWorkflowOrchestrator = exports.WorkflowOrchestrator = exports.OpenEvolveApiAdapter = exports.DatapizzaPluginAdapter = exports.RAGBitsPluginAdapter = exports.resetPluginRegistry = exports.getPluginRegistry = exports.PluginRegistry = void 0;
// Plugin System
var plugin_registry_1 = require("./plugin-registry");
Object.defineProperty(exports, "PluginRegistry", { enumerable: true, get: function () { return plugin_registry_1.PluginRegistry; } });
Object.defineProperty(exports, "getPluginRegistry", { enumerable: true, get: function () { return plugin_registry_1.getPluginRegistry; } });
Object.defineProperty(exports, "resetPluginRegistry", { enumerable: true, get: function () { return plugin_registry_1.resetPluginRegistry; } });
// Plugin Adapters
var plugin_adapters_1 = require("./plugin-adapters");
Object.defineProperty(exports, "RAGBitsPluginAdapter", { enumerable: true, get: function () { return plugin_adapters_1.RAGBitsPluginAdapter; } });
Object.defineProperty(exports, "DatapizzaPluginAdapter", { enumerable: true, get: function () { return plugin_adapters_1.DatapizzaPluginAdapter; } });
Object.defineProperty(exports, "OpenEvolveApiAdapter", { enumerable: true, get: function () { return plugin_adapters_1.OpenEvolveApiAdapter; } });
// Workflow System
var workflow_orchestrator_1 = require("./workflow-orchestrator");
Object.defineProperty(exports, "WorkflowOrchestrator", { enumerable: true, get: function () { return workflow_orchestrator_1.WorkflowOrchestrator; } });
Object.defineProperty(exports, "getWorkflowOrchestrator", { enumerable: true, get: function () { return workflow_orchestrator_1.getWorkflowOrchestrator; } });
// Workflow Templates
var workflow_templates_1 = require("./workflow-templates");
Object.defineProperty(exports, "RESEARCH_ASSISTANT_WORKFLOW", { enumerable: true, get: function () { return workflow_templates_1.RESEARCH_ASSISTANT_WORKFLOW; } });
Object.defineProperty(exports, "DATA_ANALYSIS_PIPELINE", { enumerable: true, get: function () { return workflow_templates_1.DATA_ANALYSIS_PIPELINE; } });
Object.defineProperty(exports, "PROOF_VERIFICATION_WORKFLOW", { enumerable: true, get: function () { return workflow_templates_1.PROOF_VERIFICATION_WORKFLOW; } });
Object.defineProperty(exports, "KNOWLEDGE_EXTRACTION_WORKFLOW", { enumerable: true, get: function () { return workflow_templates_1.KNOWLEDGE_EXTRACTION_WORKFLOW; } });
Object.defineProperty(exports, "PROBLEM_SOLVING_WORKFLOW", { enumerable: true, get: function () { return workflow_templates_1.PROBLEM_SOLVING_WORKFLOW; } });
Object.defineProperty(exports, "WORKFLOW_TEMPLATES", { enumerable: true, get: function () { return workflow_templates_1.WORKFLOW_TEMPLATES; } });
Object.defineProperty(exports, "getWorkflowTemplate", { enumerable: true, get: function () { return workflow_templates_1.getWorkflowTemplate; } });
Object.defineProperty(exports, "getAllWorkflowTemplates", { enumerable: true, get: function () { return workflow_templates_1.getAllWorkflowTemplates; } });
Object.defineProperty(exports, "getWorkflowTemplatesByCategory", { enumerable: true, get: function () { return workflow_templates_1.getWorkflowTemplatesByCategory; } });
// Event Integration
var plugin_events_1 = require("./plugin-events");
Object.defineProperty(exports, "PluginEventIntegration", { enumerable: true, get: function () { return plugin_events_1.PluginEventIntegration; } });
Object.defineProperty(exports, "getPluginEventIntegration", { enumerable: true, get: function () { return plugin_events_1.getPluginEventIntegration; } });
Object.defineProperty(exports, "resetPluginEventIntegration", { enumerable: true, get: function () { return plugin_events_1.resetPluginEventIntegration; } });
// Monitoring
var workflow_monitoring_1 = require("./workflow-monitoring");
Object.defineProperty(exports, "WorkflowMonitor", { enumerable: true, get: function () { return workflow_monitoring_1.WorkflowMonitor; } });
Object.defineProperty(exports, "getWorkflowMonitor", { enumerable: true, get: function () { return workflow_monitoring_1.getWorkflowMonitor; } });
Object.defineProperty(exports, "resetWorkflowMonitor", { enumerable: true, get: function () { return workflow_monitoring_1.resetWorkflowMonitor; } });
// Main Integration
var plugin_integration_1 = require("./plugin-integration");
Object.defineProperty(exports, "BubbleLabIntegration", { enumerable: true, get: function () { return plugin_integration_1.BubbleLabIntegration; } });
Object.defineProperty(exports, "initializeBubbleLabIntegration", { enumerable: true, get: function () { return plugin_integration_1.initializeBubbleLabIntegration; } });
Object.defineProperty(exports, "getBubbleLabIntegration", { enumerable: true, get: function () { return plugin_integration_1.getBubbleLabIntegration; } });
Object.defineProperty(exports, "resetBubbleLabIntegration", { enumerable: true, get: function () { return plugin_integration_1.resetBubbleLabIntegration; } });
// API Client
var openevolveApi_1 = require("./openevolveApi");
Object.defineProperty(exports, "openevolveApi", { enumerable: true, get: function () { return openevolveApi_1.openevolveApi; } });
// Types
__exportStar(require("./types"), exports);
// Re-export hooks for convenience
var useBubbleLabIntegration_1 = require("../hooks/useBubbleLabIntegration");
Object.defineProperty(exports, "useBubbleLabIntegration", { enumerable: true, get: function () { return useBubbleLabIntegration_1.useBubbleLabIntegration; } });
Object.defineProperty(exports, "useBubbleLabIntegrationInstance", { enumerable: true, get: function () { return useBubbleLabIntegration_1.useBubbleLabIntegrationInstance; } });
Object.defineProperty(exports, "usePluginRegistry", { enumerable: true, get: function () { return useBubbleLabIntegration_1.usePluginRegistry; } });
Object.defineProperty(exports, "useWorkflowOrchestrator", { enumerable: true, get: function () { return useBubbleLabIntegration_1.useWorkflowOrchestrator; } });
//# sourceMappingURL=index.js.map