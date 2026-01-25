// Export all bubble components

export * from './BaseBubble';
export * from './ingest/RAGBitsIngestBubble';
export * from './search/RAGBitsSearchBubble';
export * from './generation/RAGBitsGenerationBubble';
export * from './index/RAGBitsIndexBubble';

// Export bubble types
import { RAGBitsIngestBubble } from './ingest/RAGBitsIngestBubble';
import { RAGBitsSearchBubble } from './search/RAGBitsSearchBubble';
import { RAGBitsGenerationBubble } from './generation/RAGBitsGenerationBubble';
import { RAGBitsIndexBubble } from './index/RAGBitsIndexBubble';

export type { RAGBitsIngestBubble };
export type { RAGBitsSearchBubble };
export type { RAGBitsGenerationBubble };
export type { RAGBitsIndexBubble };

// Bubble factory function
export function createBubble(config: any): any {
  switch (config.type) {
    case 'ragbits-ingest':
      return new RAGBitsIngestBubble(config);
    case 'ragbits-search':
      return new RAGBitsSearchBubble(config);
    case 'ragbits-generation':
      return new RAGBitsGenerationBubble(config);
    case 'ragbits-index':
      return new RAGBitsIndexBubble(config);
    default:
      throw new Error(`Unknown bubble type: ${config.type}`);
  }
}