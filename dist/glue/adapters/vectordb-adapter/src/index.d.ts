/**
 * Vector DB Adapter - Exports
 *
 * Multi-backend vector database adapter for the OpenEvolve Federation.
 */
export { VectorDBAdapter, VectorDBAdapterConfig, createVectorDBAdapter, createVectorDBAdapterWithConfig, } from './adapter';
export { QdrantClient, QdrantClientConfig, } from './clients/qdrant-client';
export { PineconeClient, PineconeClientConfig, } from './clients/pinecone-client';
export { ChromaClient, ChromaClientConfig, } from './clients/chroma-client';
export { PgvectorClient, PgvectorClientConfig, } from './clients/pgvector-client';
//# sourceMappingURL=index.d.ts.map