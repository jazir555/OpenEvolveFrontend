// RAGBits Search Hook

import { useCallback } from 'react';
import type { RAGBitsSearchRequest, RAGBitsSearchResponse } from '../types/plugin-types';
import { useRAGBitsPlugin } from '../utils/createRAGBitsPlugin';

export function useRAGBitsSearch(): (
  request: RAGBitsSearchRequest
) => Promise<RAGBitsSearchResponse> {
  const plugin = useRAGBitsPlugin();

  const search = useCallback(async (request: RAGBitsSearchRequest) => {
    return await plugin.search(request);
  }, [plugin]);

  return search;
}
