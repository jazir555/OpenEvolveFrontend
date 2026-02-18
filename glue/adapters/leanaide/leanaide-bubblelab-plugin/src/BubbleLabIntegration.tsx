import React, { Suspense } from 'react';
import { Brain } from 'lucide-react';

import {
  LeanAideBubbleLabIntegration,
  type AutoformalizationConfig,
} from './integration/autoformalizationAnalytics';
import { pluginRegistry, type LeanAidePluginInterface } from './PluginInterface';

export interface BubbleLabIntegrationProps extends AutoformalizationConfig {
  className?: string;
}

export type LeanAideBubbleLabIntegrationProps = BubbleLabIntegrationProps;

export const BubbleLabLeanAideIntegration: React.FC<BubbleLabIntegrationProps> = ({ className = '' }) => {
  return <LeanAideBubbleLabIntegration className={className} />;
};

const LazyBubbleLabIntegration = React.lazy(async () => ({
  default: BubbleLabLeanAideIntegration,
}));

export const BubbleLabLeanAideIntegrationLazy: React.FC<BubbleLabIntegrationProps> = (
  props: BubbleLabIntegrationProps
) => {
  return (
    <Suspense fallback={<div className="p-4 text-sm text-gray-500">Loading LeanAide integration...</div>}>
      <LazyBubbleLabIntegration {...props} />
    </Suspense>
  );
};

export const registerBubbleLabIntegration = (): LeanAidePluginInterface => {
  return {
    id: 'bubblelab-leanaide-integration',
    name: 'BubbleLab LeanAide Integration',
    description: 'LeanAide formalization workflows integrated into BubbleLab.',
    version: '1.0.0',
    category: 'integration',
    component: BubbleLabLeanAideIntegration,
    icon: <Brain className="h-4 w-4" />,
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
