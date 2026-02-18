import { jsx as _jsx } from "react/jsx-runtime";
import React, { Suspense } from 'react';
import { Brain } from 'lucide-react';
import { LeanAideBubbleLabIntegration, } from './integration/autoformalizationAnalytics';
import { pluginRegistry } from './PluginInterface';
export const BubbleLabLeanAideIntegration = ({ className = '' }) => {
    return _jsx(LeanAideBubbleLabIntegration, { className: className });
};
const LazyBubbleLabIntegration = React.lazy(async () => ({
    default: BubbleLabLeanAideIntegration,
}));
export const BubbleLabLeanAideIntegrationLazy = (props) => {
    return (_jsx(Suspense, { fallback: _jsx("div", { className: "p-4 text-sm text-gray-500", children: "Loading LeanAide integration..." }), children: _jsx(LazyBubbleLabIntegration, { ...props }) }));
};
export const registerBubbleLabIntegration = () => {
    return {
        id: 'bubblelab-leanaide-integration',
        name: 'BubbleLab LeanAide Integration',
        description: 'LeanAide formalization workflows integrated into BubbleLab.',
        version: '1.0.0',
        category: 'integration',
        component: BubbleLabLeanAideIntegration,
        icon: _jsx(Brain, { className: "h-4 w-4" }),
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
if (!pluginRegistry.getPlugin('bubblelab-leanaide-integration')) {
    pluginRegistry.register(registerBubbleLabIntegration());
    void pluginRegistry.activate('bubblelab-leanaide-integration');
}
export default BubbleLabLeanAideIntegration;
//# sourceMappingURL=BubbleLabIntegration.js.map