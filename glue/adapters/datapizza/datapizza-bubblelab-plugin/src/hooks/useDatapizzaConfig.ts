// Datapizza Configuration Hook
// React hook for managing Datapizza plugin configuration

import { useState, useEffect } from 'react';
import { DatapizzaPluginConfig, DEFAULT_DATAPIZZA_CONFIG } from '../types/plugin-types';

export function useDatapizzaConfig(): [DatapizzaPluginConfig, (config: Partial<DatapizzaPluginConfig>) => void] {
  const [config, setConfig] = useState<DatapizzaPluginConfig>(DEFAULT_DATAPIZZA_CONFIG);

  const updateConfig = (newConfig: Partial<DatapizzaPluginConfig>) => {
    setConfig((prev: DatapizzaPluginConfig) => ({ ...prev, ...newConfig }));
  };

  return [config, updateConfig];
}
