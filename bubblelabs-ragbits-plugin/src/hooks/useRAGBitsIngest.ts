// RAGBits Ingest Hook

import { useCallback } from 'react';
import type { RAGBitsIngestRequest, RAGBitsIngestResponse } from '../types/plugin-types';
import { useRAGBitsPlugin } from '../utils/createRAGBitsPlugin';

export function useRAGBitsIngest(): (
  request: RAGBitsIngestRequest
) => Promise<RAGBitsIngestResponse> {
  const plugin = useRAGBitsPlugin();

  const ingest = useCallback(async (request: RAGBitsIngestRequest) => {
    return await plugin.ingest(request);
  }, [plugin]);

  return ingest;
}
