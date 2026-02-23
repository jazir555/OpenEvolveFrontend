// Datapizza State Hook
// React hook for accessing Datapizza plugin state

import { useState } from 'react';
import { DatapizzaPluginState, DEFAULT_DATAPIZZA_CONFIG } from '../types/plugin-types';

export function useDatapizzaState(): DatapizzaPluginState {
  const [state] = useState<DatapizzaPluginState>({
    ...DEFAULT_DATAPIZZA_CONFIG,
    status: 'idle',
    operationHistory: [],
    statistics: {
      totalOperations: 0,
      successfulOperations: 0,
      failedOperations: 0,
      averageProcessingTime: 0
    }
  });

  return state;
}
