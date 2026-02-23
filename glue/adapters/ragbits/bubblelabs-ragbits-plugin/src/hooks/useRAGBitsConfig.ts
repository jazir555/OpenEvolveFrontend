// RAGBits Configuration Hook

import { useState, useCallback } from 'react';
import type { RAGBitsPluginConfig } from '../types/plugin-types';
import { DEFAULT_RAGBITS_CONFIG } from '../types/plugin-types';
import { useRAGBitsPlugin } from '../utils/createRAGBitsPlugin';

export function useRAGBitsConfig(): [
  RAGBitsPluginConfig,
  (config: Partial<RAGBitsPluginConfig>) => void
  ] {
  const plugin = useRAGBitsPlugin();
  const [config, setConfig] = useState<RAGBitsPluginConfig>({
    ...DEFAULT_RAGBITS_CONFIG,
    ...(plugin.getContext().config as Partial<RAGBitsPluginConfig>)
  });

  const updateConfig = useCallback(async (configUpdate: Partial<RAGBitsPluginConfig>) => {
    try {
      await plugin.updateConfig(configUpdate);
      setConfig(prev => ({ ...prev, ...configUpdate }));
    } catch (error) {
      console.error('Failed to update config:', error);
      throw error;
    }
  }, [plugin]);

  return [config, updateConfig];
}
