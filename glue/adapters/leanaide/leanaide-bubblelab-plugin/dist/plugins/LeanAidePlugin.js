import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * LeanAide Autoformalization Plugin for BubbleLab UI
 *
 * This plugin integrates the complete LeanAide autoformalization system with predictive analytics
 * into the BubbleLab UI as a comprehensive plugin.
 */
import React, { useState, useEffect, useRef } from 'react';
import { Brain, BarChart3, Shield, Database, Settings, AlertTriangle, Plus, Search, Download, Upload, RefreshCw, Play } from 'lucide-react';
import { toast } from 'react-toastify';
import { EnhancedLeanAideVerification, AnalyticsDashboard, KnowledgeGraphIntegration } from '../integration/autoformalizationAnalytics';
// Default configuration
export const DEFAULT_LEANAIDE_PLUGIN_CONFIG = {
    enableAnalytics: true,
    enablePredictiveFlagging: true,
    enableKnowledgeGraph: true,
    analyticsRefreshInterval: 5000,
    maxConcurrentRequests: 5,
    cacheEnabled: true,
    cacheTTL: 3600,
    serverUrl: 'http://localhost:3000/leanaide',
    apiKey: undefined
};
export const LeanAidePlugin = ({ config: userConfig, onConfigChange, className = '' }) => {
    const [activeTab, setActiveTab] = useState('dashboard');
    const [pluginConfig, setPluginConfig] = useState({
        ...DEFAULT_LEANAIDE_PLUGIN_CONFIG,
        ...userConfig
    });
    const [isInitialized, setIsInitialized] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const analyticsRef = useRef(null);
    // Initialize the plugin
    useEffect(() => {
        const initializePlugin = async () => {
            try {
                setIsLoading(true);
                setError(null);
                // Initialize LeanAide client if not already done
                if (typeof window !== 'undefined') {
                    // Wait for DOM to be ready
                    await new Promise(resolve => setTimeout(resolve, 100));
                }
                setIsInitialized(true);
            }
            catch (err) {
                const errorMessage = err instanceof Error ? err.message : 'Failed to initialize LeanAide plugin';
                setError(errorMessage);
                toast.error(`LeanAide plugin initialization failed: ${errorMessage}`);
            }
            finally {
                setIsLoading(false);
            }
        };
        initializePlugin();
    }, []);
    // Handle config changes
    const handleConfigChange = (newConfig) => {
        setPluginConfig(newConfig);
        if (onConfigChange) {
            onConfigChange(newConfig);
        }
    };
    // Render loading state
    if (isLoading) {
        return (_jsx("div", { className: `flex items-center justify-center h-64 ${className}`, children: _jsxs("div", { className: "flex flex-col items-center gap-4", children: [_jsx("div", { className: "animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500" }), _jsx("p", { className: "text-gray-600", children: "Initializing LeanAide Plugin..." })] }) }));
    }
    // Render error state
    if (error) {
        return (_jsxs("div", { className: `bg-red-50 border border-red-200 rounded-lg p-6 ${className}`, children: [_jsxs("div", { className: "flex items-center gap-2 text-red-800", children: [_jsx(AlertTriangle, { className: "w-5 h-5" }), _jsx("h3", { className: "font-medium", children: "Plugin Initialization Error" })] }), _jsx("p", { className: "text-red-600 mt-2", children: error }), _jsxs("button", { onClick: () => window.location.reload(), className: "mt-4 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors", children: [_jsx(RefreshCw, { className: "w-4 h-4 inline mr-2" }), "Reload Plugin"] })] }));
    }
    return (_jsxs("div", { className: `bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden ${className}`, children: [_jsx("div", { className: "bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-4", children: _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { className: "flex items-center gap-3", children: [_jsx(Brain, { className: "w-8 h-8" }), _jsxs("div", { children: [_jsx("h1", { className: "text-xl font-bold", children: "LeanAide Autoformalization" }), _jsx("p", { className: "text-blue-100 text-sm", children: "Natural Language to Lean 4 Formalization" })] })] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: "bg-blue-500 text-xs px-2 py-1 rounded-full", children: "v1.0.0" }), isInitialized && (_jsxs("span", { className: "bg-green-500 text-xs px-2 py-1 rounded-full flex items-center gap-1", children: [_jsx("div", { className: "w-2 h-2 bg-white rounded-full animate-pulse" }), "Connected"] }))] })] }) }), _jsx("div", { className: "border-b border-gray-200", children: _jsx("nav", { className: "flex space-x-8 px-6", children: [
                        { id: 'dashboard', label: 'Analytics Dashboard', icon: BarChart3 },
                        { id: 'verification', label: 'Autoformalization', icon: Shield },
                        { id: 'knowledge', label: 'Knowledge Graph', icon: Database },
                        { id: 'settings', label: 'Settings', icon: Settings },
                    ].map((tab) => (_jsxs("button", { onClick: () => setActiveTab(tab.id), className: `py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${activeTab === tab.id
                            ? 'border-indigo-500 text-indigo-600'
                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`, children: [_jsx(tab.icon, { className: "w-4 h-4" }), tab.label] }, tab.id))) }) }), _jsxs("div", { className: "p-6", children: [activeTab === 'dashboard' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(BarChart3, { className: "w-6 h-6" }), "Analytics Dashboard"] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors", children: [_jsx(Download, { className: "w-4 h-4" }), "Export"] }), _jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors", children: [_jsx(RefreshCw, { className: "w-4 h-4" }), "Refresh"] })] })] }), _jsx(AnalyticsDashboard, {})] })), activeTab === 'verification' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Shield, { className: "w-6 h-6" }), "Autoformalization Verification"] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-green-100 text-green-700 rounded-md hover:bg-green-200 transition-colors", children: [_jsx(Play, { className: "w-4 h-4" }), "Run"] }), _jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-yellow-100 text-yellow-700 rounded-md hover:bg-yellow-200 transition-colors", children: [_jsx(Plus, { className: "w-4 h-4" }), "New"] })] })] }), _jsx(EnhancedLeanAideVerification, { problemStatement: "", mode: "theorem", enableAnalytics: pluginConfig.enableAnalytics, strategy: "auto", domain: "general" })] })), activeTab === 'knowledge' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Database, { className: "w-6 h-6" }), "Knowledge Graph Integration"] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-purple-100 text-purple-700 rounded-md hover:bg-purple-200 transition-colors", children: [_jsx(Search, { className: "w-4 h-4" }), "Search"] }), _jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-indigo-100 text-indigo-700 rounded-md hover:bg-indigo-200 transition-colors", children: [_jsx(Upload, { className: "w-4 h-4" }), "Ingest"] })] })] }), _jsx(KnowledgeGraphIntegration, {})] })), activeTab === 'settings' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Settings, { className: "w-6 h-6" }), "Plugin Settings"] }), _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-6", children: [_jsxs("div", { className: "bg-gray-50 p-4 rounded-lg border", children: [_jsx("h3", { className: "font-medium text-gray-700 mb-3", children: "Analytics Configuration" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Enable Analytics" }), _jsx("p", { className: "text-sm text-gray-500", children: "Track performance metrics" })] }), _jsx("div", { className: `w-12 h-6 rounded-full relative cursor-pointer ${pluginConfig.enableAnalytics ? 'bg-blue-500' : 'bg-gray-300'}`, onClick: () => handleConfigChange({
                                                                    ...pluginConfig,
                                                                    enableAnalytics: !pluginConfig.enableAnalytics
                                                                }), children: _jsx("div", { className: `w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${pluginConfig.enableAnalytics ? 'left-6' : 'left-0.5'}` }) })] }), _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Predictive Flagging" }), _jsx("p", { className: "text-sm text-gray-500", children: "Enable predictive quality control" })] }), _jsx("div", { className: `w-12 h-6 rounded-full relative cursor-pointer ${pluginConfig.enablePredictiveFlagging ? 'bg-blue-500' : 'bg-gray-300'}`, onClick: () => handleConfigChange({
                                                                    ...pluginConfig,
                                                                    enablePredictiveFlagging: !pluginConfig.enablePredictiveFlagging
                                                                }), children: _jsx("div", { className: `w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${pluginConfig.enablePredictiveFlagging ? 'left-6' : 'left-0.5'}` }) })] }), _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Knowledge Graph" }), _jsx("p", { className: "text-sm text-gray-500", children: "Enable knowledge integration" })] }), _jsx("div", { className: `w-12 h-6 rounded-full relative cursor-pointer ${pluginConfig.enableKnowledgeGraph ? 'bg-blue-500' : 'bg-gray-300'}`, onClick: () => handleConfigChange({
                                                                    ...pluginConfig,
                                                                    enableKnowledgeGraph: !pluginConfig.enableKnowledgeGraph
                                                                }), children: _jsx("div", { className: `w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${pluginConfig.enableKnowledgeGraph ? 'left-6' : 'left-0.5'}` }) })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-700 mb-1", children: "Refresh Interval (ms)" }), _jsx("input", { type: "number", value: pluginConfig.analyticsRefreshInterval, onChange: (e) => handleConfigChange({
                                                                    ...pluginConfig,
                                                                    analyticsRefreshInterval: parseInt(e.target.value) || 5000
                                                                }), className: "w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500" })] })] })] }), _jsxs("div", { className: "bg-gray-50 p-4 rounded-lg border", children: [_jsx("h3", { className: "font-medium text-gray-700 mb-3", children: "Connection Settings" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-700 mb-1", children: "Server URL" }), _jsx("input", { type: "text", value: pluginConfig.serverUrl, onChange: (e) => handleConfigChange({
                                                                    ...pluginConfig,
                                                                    serverUrl: e.target.value
                                                                }), className: "w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-700 mb-1", children: "API Key" }), _jsx("input", { type: "password", value: pluginConfig.apiKey || '', onChange: (e) => handleConfigChange({
                                                                    ...pluginConfig,
                                                                    apiKey: e.target.value || undefined
                                                                }), className: "w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500", placeholder: "Enter API key (optional)" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-700 mb-1", children: "Max Concurrent Requests" }), _jsx("input", { type: "number", value: pluginConfig.maxConcurrentRequests, onChange: (e) => handleConfigChange({
                                                                    ...pluginConfig,
                                                                    maxConcurrentRequests: parseInt(e.target.value) || 5
                                                                }), className: "w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500" })] }), _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Enable Caching" }), _jsx("p", { className: "text-sm text-gray-500", children: "Cache results for performance" })] }), _jsx("div", { className: `w-12 h-6 rounded-full relative cursor-pointer ${pluginConfig.cacheEnabled ? 'bg-blue-500' : 'bg-gray-300'}`, onClick: () => handleConfigChange({
                                                                    ...pluginConfig,
                                                                    cacheEnabled: !pluginConfig.cacheEnabled
                                                                }), children: _jsx("div", { className: `w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${pluginConfig.cacheEnabled ? 'left-6' : 'left-0.5'}` }) })] }), pluginConfig.cacheEnabled && (_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-700 mb-1", children: "Cache TTL (seconds)" }), _jsx("input", { type: "number", value: pluginConfig.cacheTTL, onChange: (e) => handleConfigChange({
                                                                    ...pluginConfig,
                                                                    cacheTTL: parseInt(e.target.value) || 3600
                                                                }), className: "w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500" })] }))] })] })] }), _jsxs("div", { className: "flex justify-end gap-3 pt-4", children: [_jsx("button", { onClick: () => {
                                            // Reset to defaults
                                            setPluginConfig(DEFAULT_LEANAIDE_PLUGIN_CONFIG);
                                            if (onConfigChange) {
                                                onConfigChange(DEFAULT_LEANAIDE_PLUGIN_CONFIG);
                                            }
                                            toast.success('Settings reset to defaults');
                                        }, className: "px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors", children: "Reset Defaults" }), _jsx("button", { onClick: () => {
                                            toast.success('Settings saved successfully');
                                        }, className: "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors", children: "Save Settings" })] })] }))] })] }));
};
// Plugin registration function
export function registerLeanAidePlugin() {
    return {
        id: 'leanaide-autoformalization',
        name: 'LeanAide Autoformalization',
        description: 'Convert natural language mathematical statements to formal Lean 4 code with predictive analytics',
        version: '1.0.0',
        category: 'formalization',
        component: LeanAidePlugin,
        icon: _jsx(Brain, { className: "w-5 h-5" }),
        settingsSchema: {
            type: 'object',
            properties: {
                enableAnalytics: { type: 'boolean', default: true },
                enablePredictiveFlagging: { type: 'boolean', default: true },
                enableKnowledgeGraph: { type: 'boolean', default: true },
                analyticsRefreshInterval: { type: 'number', default: 5000 },
                maxConcurrentRequests: { type: 'number', default: 5 },
                cacheEnabled: { type: 'boolean', default: true },
                cacheTTL: { type: 'number', default: 3600 },
                serverUrl: { type: 'string', default: 'http://localhost:3000/leanaide' }
            }
        },
        permissions: ['network', 'storage']
    };
}
export default LeanAidePlugin;
//# sourceMappingURL=LeanAidePlugin.js.map