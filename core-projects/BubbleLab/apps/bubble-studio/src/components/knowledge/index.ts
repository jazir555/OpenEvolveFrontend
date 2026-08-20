/**
 * Knowledge Base component barrel.
 *
 * Re-exports the public surface of the Knowledge Base interface so consumers
 * (and `App.tsx`/routing) can import from a single module, e.g.:
 *   import { KnowledgeBase } from '@/components/knowledge';
 */

export { KnowledgeBase } from './KnowledgeBase';
export { ArtifactList, artifactTitle } from './ArtifactList';
export { ArtifactDetail } from './ArtifactDetail';
export { KnowledgeGraphView } from './KnowledgeGraphView';
export { KnowledgeStatsView } from './KnowledgeStatsView';
