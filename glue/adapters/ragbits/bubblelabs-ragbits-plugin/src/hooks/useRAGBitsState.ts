// RAGBits State Hook

import { useState, useEffect } from 'react';
import type { RAGBitsPluginState } from '../types/plugin-types';
import { useRAGBitsPlugin } from '../utils/createRAGBitsPlugin';

export function useRAGBitsState(): RAGBitsPluginState {
  const plugin = useRAGBitsPlugin();
  const [state, setState] = useState<RAGBitsPluginState>(plugin.getContext().state);

  useEffect(() => {
    const interval = setInterval(() => {
      setState(plugin.getContext().state);
    }, 1000);

    return () => clearInterval(interval);
  }, [plugin]);

  return state;
}
