import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { Pinecone } from '@pinecone-database/pinecone';

// Initialize embeddings
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});

// Get Pinecone index
const indexName = process.env.PINECONE_INDEX_NAME || 'knowledge-engine';
const pineconeIndex = pinecone.Index(indexName);

/**
 * Generate embeddings for text using OpenAI
 */
export async function generateEmbeddings(text: string | string[]) {
  try {
    if (Array.isArray(text)) {
      // Generate embeddings for multiple texts
      return await embeddings.embedDocuments(text);
    } else {
      // Generate embedding for single text
      return [await embeddings.embedQuery(text)];
    }
  } catch (error) {
    console.error('Embedding generation error:', error);
    throw new Error('Failed to generate embeddings');
  }
}

/**
 * Store embeddings in Pinecone vector database
 */
export async function storeEmbeddings(embeddingsData: {
  id: string;
  text: string;
  metadata: Record<string, any>;
}[]) {
  try {
    const vectorsToUpsert = await Promise.all(
      embeddingsData.map(async (item) => {
        const embedding = await embeddings.embedQuery(item.text);
        return {
          id: item.id,
          values: embedding,
          metadata: item.metadata,
        };
      })
    );

    const upsertResponse = await pineconeIndex.upsert({
      vectors: vectorsToUpsert,
      namespace: process.env.PINECONE_NAMESPACE || 'default',
    });

    return upsertResponse;
  } catch (error) {
    console.error('Embedding storage error:', error);
    throw new Error('Failed to store embeddings');
  }
}

/**
 * Search for similar embeddings in Pinecone
 */
export async function searchEmbeddings(
  query: string,
  topK: number = 10,
  filters: Record<string, any> = {}
) {
  try {
    // Generate embedding for the query
    const queryEmbedding = await embeddings.embedQuery(query);

    // Perform similarity search
    const queryResponse = await pineconeIndex.query({
      vector: queryEmbedding,
      topK: topK,
      includeMetadata: true,
      filter: filters,
      namespace: process.env.PINECONE_NAMESPACE || 'default',
    });

    return queryResponse.matches.map((match: any) => ({
      id: match.id,
      score: match.score,
      metadata: match.metadata,
      text: match.metadata?.text || '',
    }));
  } catch (error) {
    console.error('Embedding search error:', error);
    throw new Error('Failed to search embeddings');
  }
}

/**
 * Delete embeddings from Pinecone
 */
export async function deleteEmbeddings(ids: string[]) {
  try {
    const deleteResponse = await pineconeIndex.deleteMany({
      ids: ids,
      namespace: process.env.PINECONE_NAMESPACE || 'default',
    });

    return deleteResponse;
  } catch (error) {
    console.error('Embedding deletion error:', error);
    throw new Error('Failed to delete embeddings');
  }
}

/**
 * Get embeddings statistics
 */
export async function getEmbeddingsStats() {
  try {
    const describeIndexStatsResponse = await pineconeIndex.describeIndexStats();

    return {
      namespaces: describeIndexStatsResponse.namespaces,
      dimension: describeIndexStatsResponse.dimension,
      indexName: describeIndexStatsResponse.indexName,
    };
  } catch (error) {
    console.error('Embedding stats error:', error);
    throw new Error('Failed to get embeddings stats');
  }
}