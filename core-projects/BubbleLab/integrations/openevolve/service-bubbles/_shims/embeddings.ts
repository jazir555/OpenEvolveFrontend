/**
 * Local shim for the embeddings implementation.
 *
 * The real implementation lives outside this package
 * (../../../utils/embeddings). This placeholder keeps the integration
 * typecheckable in isolation. Swap for the real import when wiring the
 * full monorepo build.
 */
export async function generateEmbeddings(_text: string): Promise<number[][]> {
  // Placeholder: returns a zero vector. The caller already falls back to a
  // random vector if the real embeddings service is unavailable.
  return [Array(1536).fill(0)];
}
