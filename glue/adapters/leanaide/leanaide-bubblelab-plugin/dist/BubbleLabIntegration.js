import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * BubbleLab UI Integration for LeanAide Autoformalization System
 *
 * This module provides the complete integration of the LeanAide autoformalization system
 * with predictive analytics into the BubbleLab UI as a comprehensive plugin system.
 */
import React, { useState, useEffect, Suspense } from 'react';
import { Brain, BarChart3, Shield, Database, Settings, AlertTriangle, Puzzle, RefreshCw, Play, Download, Upload, Search } from 'lucide-react';
import { toast } from 'react-toastify';
import { EnhancedLeanAideVerification, AnalyticsDashboard, KnowledgeGraphIntegration } from './integration/autoformalizationAnalytics';
import { pluginRegistry, PluginManager } from './PluginInterface';
export const BubbleLabLeanAideIntegration = ({ serverUrl = 'http://localhost:3000/leanaide', apiKey, enableAnalytics = true, enablePredictiveFlagging = true, enableKnowledgeGraph = true, className = '' }) => {
    const [activeTab, setActiveTab] = useState('dashboard');
    const [isInitialized, setIsInitialized] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [config, setConfig] = useState({
        serverUrl,
        apiKey,
        enableAnalytics,
        enablePredictiveFlagging,
        enableKnowledgeGraph,
        analyticsRefreshInterval: 5000,
        maxConcurrentRequests: 5,
        cacheEnabled: true,
        cacheTTL: 3600
    });
    // Initialize the integration
    useEffect(() => {
        const initializeIntegration = async () => {
            try {
                setIsLoading(true);
                setError(null);
                // Initialize LeanAide client if needed
                // In a real implementation, this would connect to the server
                console.log('Initializing LeanAide integration with server:', serverUrl);
                // Initialize plugins
                await initializePlugins();
                setIsInitialized(true);
            }
            catch (err) {
                const errorMessage = err instanceof Error ? err.message : 'Failed to initialize LeanAide integration';
                setError(errorMessage);
                toast.error(`LeanAide integration initialization failed: ${errorMessage}`);
            }
            finally {
                setIsLoading(false);
            }
        };
        initializeIntegration();
    }, [serverUrl, apiKey]);
    const initializePlugins = async () => {
        // Initialize any required plugins
        // In a real implementation, this would initialize the plugin system
        console.log('Initializing plugins...');
    };
    const handleConfigChange = (newConfig) => {
        setConfig(prev => ({ ...prev, ...newConfig }));
    };
    if (isLoading) {
        return (_jsx("div", { className: `flex items-center justify-center h-96 ${className}`, children: _jsxs("div", { className: "flex flex-col items-center gap-4", children: [_jsx("div", { className: "animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500" }), _jsx("h3", { className: "text-xl font-medium text-gray-800", children: "Initializing LeanAide Integration" }), _jsx("p", { className: "text-gray-600", children: "Connecting to autoformalization services..." })] }) }));
    }
    if (error) {
        return (_jsxs("div", { className: `bg-red-50 border border-red-200 rounded-lg p-6 ${className}`, children: [_jsxs("div", { className: "flex items-center gap-2 text-red-800 mb-4", children: [_jsx(AlertTriangle, { className: "w-5 h-5" }), _jsx("h3", { className: "font-medium", children: "Integration Error" })] }), _jsx("p", { className: "text-red-600 mb-4", children: error }), _jsxs("button", { onClick: () => window.location.reload(), className: "px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors flex items-center gap-2", children: [_jsx(RefreshCw, { className: "w-4 h-4" }), "Reload Integration"] })] }));
    }
    return (_jsx("div", { className: `bg-gray-50 min-h-screen ${className}`, children: _jsxs("div", { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8", children: [_jsxs("div", { className: "mb-8", children: [_jsxs("div", { className: "flex items-center gap-3 mb-2", children: [_jsx("div", { className: "p-2 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-lg", children: _jsx(Brain, { className: "w-8 h-8 text-white" }) }), _jsxs("div", { children: [_jsx("h1", { className: "text-3xl font-bold text-gray-900", children: "LeanAide Autoformalization" }), _jsx("p", { className: "text-gray-600", children: "Natural Language to Lean 4 Formalization with Analytics" })] })] }), _jsxs("div", { className: "flex items-center gap-6 mt-4 text-sm text-gray-500", children: [_jsxs("div", { className: "flex items-center gap-1", children: [_jsx("div", { className: "w-2 h-2 bg-green-500 rounded-full" }), _jsx("span", { children: "Connected" })] }), _jsxs("div", { children: ["Server: ", config.serverUrl] }), _jsxs("div", { children: ["Analytics: ", config.enableAnalytics ? 'Enabled' : 'Disabled'] }), _jsxs("div", { children: ["Predictive: ", config.enablePredictiveFlagging ? 'Enabled' : 'Disabled'] })] })] }), _jsxs("div", { className: "bg-white rounded-xl shadow-lg overflow-hidden", children: [_jsx("div", { className: "border-b border-gray-200", children: _jsx("nav", { className: "flex space-x-8 px-6", children: [
                                    { id: 'dashboard', label: 'Analytics Dashboard', icon: BarChart3 },
                                    { id: 'verification', label: 'Autoformalization', icon: Shield },
                                    { id: 'knowledge', label: 'Knowledge Graph', icon: Database },
                                    { id: 'plugins', label: 'Plugin Manager', icon: Puzzle },
                                    { id: 'settings', label: 'Settings', icon: Settings },
                                ].map((tab) => (_jsxs("button", { onClick: () => setActiveTab(tab.id), className: `py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${activeTab === tab.id
                                        ? 'border-indigo-500 text-indigo-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`, children: [_jsx(tab.icon, { className: "w-4 h-4" }), tab.label] }, tab.id))) }) }), _jsxs("div", { className: "p-6", children: [activeTab === 'dashboard' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(BarChart3, { className: "w-6 h-6" }), "Analytics Dashboard"] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors", children: [_jsx(Download, { className: "w-4 h-4" }), "Export"] }), _jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors", children: [_jsx(RefreshCw, { className: "w-4 h-4" }), "Refresh"] })] })] }), _jsx(AnalyticsDashboard, {})] })), activeTab === 'verification' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Shield, { className: "w-6 h-6" }), "Autoformalization Verification"] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-green-100 text-green-700 rounded-md hover:bg-green-200 transition-colors", children: [_jsx(Play, { className: "w-4 h-4" }), "Run"] }), _jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-yellow-100 text-yellow-700 rounded-md hover:bg-yellow-200 transition-colors", children: [_jsx(Plus, { className: "w-4 h-4" }), "New"] })] })] }), _jsx(EnhancedLeanAideVerification, { problemStatement: "", mode: "theorem", enableAnalytics: config.enableAnalytics, strategy: "auto", domain: "general" })] })), activeTab === 'knowledge' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Database, { className: "w-6 h-6" }), "Knowledge Graph Integration"] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-purple-100 text-purple-700 rounded-md hover:bg-purple-200 transition-colors", children: [_jsx(Search, { className: "w-4 h-4" }), "Search"] }), _jsxs("button", { className: "flex items-center gap-2 px-3 py-2 bg-indigo-100 text-indigo-700 rounded-md hover:bg-indigo-200 transition-colors", children: [_jsx(Upload, { className: "w-4 h-4" }), "Ingest"] })] })] }), _jsx(KnowledgeGraphIntegration, {})] })), activeTab === 'plugins' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Puzzle, { className: "w-6 h-6" }), "Plugin Manager"] }), _jsx(PluginManager, {})] })), activeTab === 'settings' && (_jsxs("div", { className: "space-y-6", children: [_jsxs("h2", { className: "text-2xl font-bold text-gray-800 flex items-center gap-2", children: [_jsx(Settings, { className: "w-6 h-6" }), "Integration Settings"] }), _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-6", children: [_jsxs("div", { className: "bg-gray-50 p-4 rounded-lg border", children: [_jsx("h3", { className: "font-medium text-gray-700 mb-3", children: "Service Configuration" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-700 mb-1", children: "Server URL" }), _jsx("input", { type: "text", value: config.serverUrl, onChange: (e) => handleConfigChange({ serverUrl: e.target.value }), className: "w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-700 mb-1", children: "API Key" }), _jsx("input", { type: "password", value: config.apiKey || '', onChange: (e) => handleConfigChange({ apiKey: e.target.value || undefined }), className: "w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500", placeholder: "Enter API key (optional)" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-700 mb-1", children: "Max Concurrent Requests" }), _jsx("input", { type: "number", value: config.maxConcurrentRequests, onChange: (e) => handleConfigChange({ maxConcurrentRequests: parseInt(e.target.value) || 5 }), className: "w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500" })] })] })] }), _jsxs("div", { className: "bg-gray-50 p-4 rounded-lg border", children: [_jsx("h3", { className: "font-medium text-gray-700 mb-3", children: "Feature Configuration" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Analytics" }), _jsx("p", { className: "text-sm text-gray-500", children: "Enable real-time metrics" })] }), _jsx("div", { className: `w-12 h-6 rounded-full relative cursor-pointer ${config.enableAnalytics ? 'bg-blue-500' : 'bg-gray-300'}`, onClick: () => handleConfigChange({ enableAnalytics: !config.enableAnalytics }), children: _jsx("div", { className: `w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${config.enableAnalytics ? 'left-6' : 'left-0.5'}` }) })] }), _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Predictive Flagging" }), _jsx("p", { className: "text-sm text-gray-500", children: "Enable predictive quality control" })] }), _jsx("div", { className: `w-12 h-6 rounded-full relative cursor-pointer ${config.enablePredictiveFlagging ? 'bg-blue-500' : 'bg-gray-300'}`, onClick: () => handleConfigChange({ enablePredictiveFlagging: !config.enablePredictiveFlagging }), children: _jsx("div", { className: `w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${config.enablePredictiveFlagging ? 'left-6' : 'left-0.5'}` }) })] }), _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Knowledge Graph" }), _jsx("p", { className: "text-sm text-gray-500", children: "Enable knowledge integration" })] }), _jsx("div", { className: `w-12 h-6 rounded-full relative cursor-pointer ${config.enableKnowledgeGraph ? 'bg-blue-500' : 'bg-gray-300'}`, onClick: () => handleConfigChange({ enableKnowledgeGraph: !config.enableKnowledgeGraph }), children: _jsx("div", { className: `w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${config.enableKnowledgeGraph ? 'left-6' : 'left-0.5'}` }) })] }), _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("p", { className: "font-medium text-gray-800", children: "Caching" }), _jsx("p", { className: "text-sm text-gray-500", children: "Enable result caching" })] }), _jsx("div", { className: `w-12 h-6 rounded-full relative cursor-pointer ${config.cacheEnabled ? 'bg-blue-500' : 'bg-gray-300'}`, onClick: () => handleConfigChange({ cacheEnabled: !config.cacheEnabled }), children: _jsx("div", { className: `w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${config.cacheEnabled ? 'left-6' : 'left-0.5'}` }) })] }), config.cacheEnabled && (_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-700 mb-1", children: "Cache TTL (seconds)" }), _jsx("input", { type: "number", value: config.cacheTTL, onChange: (e) => handleConfigChange({ cacheTTL: parseInt(e.target.value) || 3600 }), className: "w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500" })] }))] })] })] }), _jsxs("div", { className: "flex justify-end gap-3 pt-4", children: [_jsx("button", { onClick: () => {
                                                        setConfig({
                                                            serverUrl: 'http://localhost:3000/leanaide',
                                                            apiKey: undefined,
                                                            enableAnalytics: true,
                                                            enablePredictiveFlagging: true,
                                                            enableKnowledgeGraph: true,
                                                            analyticsRefreshInterval: 5000,
                                                            maxConcurrentRequests: 5,
                                                            cacheEnabled: true,
                                                            cacheTTL: 3600
                                                        });
                                                        toast.success('Settings reset to defaults');
                                                    }, className: "px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors", children: "Reset Defaults" }), _jsx("button", { onClick: () => {
                                                        toast.success('Settings saved successfully');
                                                    }, className: "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors", children: "Save Settings" })] })] }))] })] })] }) }));
};
// Lazy-loaded component for better performance
const LazyBubbleLabIntegration = React.lazy(() => import('./BubbleLabIntegration').then(module => ({ default: module.BubbleLabLeanAideIntegration })));
export const BubbleLabLeanAideIntegrationLazy = (props) => (_jsx(Suspense, { fallback: _jsx("div", { className: "flex items-center justify-center h-96", children: _jsxs("div", { className: "flex flex-col items-center gap-4", children: [_jsx("div", { className: "animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500" }), _jsx("p", { children: "Loading LeanAide Integration..." })] }) }), children: _jsx(LazyBubbleLabIntegration, { ...props }) }));
// Plugin registration for BubbleLab
export const registerBubbleLabIntegration = () => {
    return {
        id: 'bubblelab-leanaide-integration',
        name: 'BubbleLab LeanAide Integration',
        description: 'Complete integration of LeanAide autoformalization with analytics into BubbleLab UI',
        version: '1.0.0',
        category: 'integration',
        component: BubbleLabLeanAideIntegration,
        icon: _jsx(Brain, { className: "w-5 h-5" }),
        settingsSchema: {
            type: 'object',
            properties: {
                serverUrl: { type: 'string', default: 'http://localhost:3000/leanaide' },
                apiKey: { type: 'string', default: '' },
                enableAnalytics: { type: 'boolean', default: true },
                enablePredictiveFlagging: { type: 'boolean', default: true },
                enableKnowledgeGraph: { type: 'boolean', default: true },
                analyticsRefreshInterval: { type: 'number', default: 5000 },
                maxConcurrentRequests: { type: 'number', default: 5 },
                cacheEnabled: { type: 'boolean', default: true },
                cacheTTL: { type: 'number', default: 3600 }
            }
        },
        permissions: ['network', 'storage'],
        dependencies: ['leanaide-core', 'bubblelab-core'],
        author: 'OpenEvolve',
        license: 'MIT',
        homepage: 'https://github.com/openevolve/leanaide',
        repository: 'https://github.com/openevolve/leanaide/leanaide-bubblelab-plugin',
        keywords: ['lean', 'theorem', 'prover', 'formalization', 'autoformalization', 'bubblelab', 'integration', 'analytics'],
        activationEvents: ['onView:leanaide-dashboard', 'onCommand:leanaide.open'],
        contributes: {
            views: [
                {
                    id: 'leanaide-dashboard',
                    name: 'LeanAide Dashboard',
                    when: 'leanaide.enabled'
                },
                {
                    id: 'leanaide-verification',
                    name: 'Autoformalization',
                    when: 'leanaide.enabled'
                },
                {
                    id: 'leanaide-knowledge',
                    name: 'Knowledge Graph',
                    when: 'leanaide.knowledgeGraphEnabled'
                }
            ],
            commands: [
                {
                    command: 'leanaide.convert',
                    title: 'Convert Natural Language to Lean',
                    category: 'LeanAide'
                },
                {
                    command: 'leanaide.verify',
                    title: 'Verify Lean Code',
                    category: 'LeanAide'
                },
                {
                    command: 'leanaide.searchKnowledge',
                    title: 'Search Mathematical Knowledge',
                    category: 'LeanAide'
                }
            ],
            configuration: {
                title: 'LeanAide Configuration',
                properties: {
                    'leanaide.serverUrl': {
                        type: 'string',
                        default: 'http://localhost:3000/leanaide',
                        description: 'URL of the LeanAide server'
                    },
                    'leanaide.apiKey': {
                        type: 'string',
                        default: '',
                        description: 'API key for LeanAide server'
                    },
                    'leanaide.enableAnalytics': {
                        type: 'boolean',
                        default: true,
                        description: 'Enable real-time analytics'
                    },
                    'leanaide.enablePredictiveFlagging': {
                        type: 'boolean',
                        default: true,
                        description: 'Enable predictive quality control'
                    }
                }
            }
        }
    };
};
// Register the integration plugin
const bubbleLabIntegrationPlugin = registerBubbleLabIntegration();
pluginRegistry.register(bubbleLabIntegrationPlugin);
// Auto-activate the integration plugin
pluginRegistry.activate('bubblelab-leanaide-integration').catch(console.error);
// Export the main integration component
export { BubbleLabLeanAideIntegration, BubbleLabLeanAideIntegrationLazy };
export default BubbleLabLeanAideIntegration;
//# sourceMappingURL=BubbleLabIntegration.js.map