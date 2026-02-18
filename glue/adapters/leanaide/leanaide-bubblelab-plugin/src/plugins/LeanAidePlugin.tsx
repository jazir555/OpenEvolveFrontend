import React, { useEffect, useMemo } from 'react';
import { Brain } from 'lucide-react';

import {
  LeanAideBubbleLabIntegration,
  type AutoformalizationConfig,
} from '../integration/autoformalizationAnalytics';

export interface LeanAidePluginInterface {
  id: string;
  name: string;
  description: string;
  version: string;
  category: string;
  component: React.ComponentType<any>;
  icon: React.ReactNode;
  settingsSchema?: Record<string, unknown>;
  permissions?: string[];
}

export interface LeanAidePluginConfig extends AutoformalizationConfig {
  analyticsRefreshInterval?: number;
  maxConcurrentRequests?: number;
  cacheEnabled?: boolean;
  cacheTTL?: number;
}

export const DEFAULT_LEANAIDE_PLUGIN_CONFIG: LeanAidePluginConfig = {
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

export interface LeanAidePluginProps {
  config?: Partial<LeanAidePluginConfig>;
  onConfigChange?: (config: LeanAidePluginConfig) => void;
  className?: string;
}

export const LeanAidePlugin = ({
  config,
  onConfigChange,
  className = '',
}: LeanAidePluginProps) => {
  const resolvedConfig = useMemo(
    () => ({ ...DEFAULT_LEANAIDE_PLUGIN_CONFIG, ...config }),
    [config]
  );

  useEffect(() => {
    onConfigChange?.(resolvedConfig);
  }, [onConfigChange, resolvedConfig]);

  return <LeanAideBubbleLabIntegration className={className} />;
};

export function registerLeanAidePlugin(): LeanAidePluginInterface {
  return {
    id: 'leanaide-autoformalization',
    name: 'LeanAide Autoformalization',
    description:
      'Convert natural language mathematical statements to formal Lean 4 code with analytics.',
    version: '1.0.0',
    category: 'formalization',
    component: LeanAidePlugin,
    icon: <Brain className="h-4 w-4" />,
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
