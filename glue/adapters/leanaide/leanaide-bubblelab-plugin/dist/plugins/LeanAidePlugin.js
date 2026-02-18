import { jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useMemo } from 'react';
import { Brain } from 'lucide-react';
import { LeanAideBubbleLabIntegration, } from '../integration/autoformalizationAnalytics';
export const DEFAULT_LEANAIDE_PLUGIN_CONFIG = {
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
export const LeanAidePlugin = ({ config, onConfigChange, className = '', }) => {
    const resolvedConfig = useMemo(() => ({ ...DEFAULT_LEANAIDE_PLUGIN_CONFIG, ...config }), [config]);
    useEffect(() => {
        onConfigChange?.(resolvedConfig);
    }, [onConfigChange, resolvedConfig]);
    return _jsx(LeanAideBubbleLabIntegration, { className: className });
};
export function registerLeanAidePlugin() {
    return {
        id: 'leanaide-autoformalization',
        name: 'LeanAide Autoformalization',
        description: 'Convert natural language mathematical statements to formal Lean 4 code with analytics.',
        version: '1.0.0',
        category: 'formalization',
        component: LeanAidePlugin,
        icon: _jsx(Brain, { className: "h-4 w-4" }),
        settingsSchema: {
            type: 'object',
            properties: {
                serverUrl: { type: 'string', default: DEFAULT_LEANAIDE_PLUGIN_CONFIG.serverUrl },
                ragbitsUrl: { type: 'string', default: DEFAULT_LEANAIDE_PLUGIN_CONFIG.ragbitsUrl },
                enableAnalytics: { type: 'boolean', default: true },
            },
        },
        permissions: ['network', 'storage'],
    };
}
export default LeanAidePlugin;
//# sourceMappingURL=LeanAidePlugin.js.map