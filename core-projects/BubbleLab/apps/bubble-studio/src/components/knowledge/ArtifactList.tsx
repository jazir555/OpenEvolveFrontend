/**
 * ArtifactList
 * Renders the list of knowledge artifacts returned by `listKnowledgeArtifacts`
 * or `searchKnowledge`. Each row shows the artifact id, a human-readable
 * title/type and its effectiveness score. Clicking a row selects it.
 */

import type { KnowledgeArtifact } from '@/types/openevolve';

interface ArtifactListProps {
  artifacts: KnowledgeArtifact[];
  selectedId: string | null;
  onSelect: (artifact: KnowledgeArtifact) => void;
  loading: boolean;
}

/**
 * Best-effort human readable title for an artifact. Content may be a string
 * (a snippet) or a structured record that sometimes carries a `title`/`name`.
 */
export function artifactTitle(artifact: KnowledgeArtifact): string {
  if (typeof artifact.content === 'string') {
    const trimmed = artifact.content.trim();
    if (trimmed.length > 0) {
      return trimmed.length > 90 ? `${trimmed.slice(0, 90)}…` : trimmed;
    }
  } else {
    const record = artifact.content as Record<string, unknown>;
    const candidate =
      record['title'] ?? record['name'] ?? record['summary'] ?? record['description'];
    if (typeof candidate === 'string') {
      return candidate.length > 90 ? `${candidate.slice(0, 90)}…` : candidate;
    }
  }
  return `${artifact.artifact_type} artifact`;
}

export function ArtifactList({
  artifacts,
  selectedId,
  onSelect,
  loading,
}: ArtifactListProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-10 text-sm text-gray-500 dark:text-gray-400">
        Loading artifacts…
      </div>
    );
  }

  if (artifacts.length === 0) {
    return (
      <div className="flex items-center justify-center py-10 text-sm text-gray-500 dark:text-gray-400">
        No artifacts found.
      </div>
    );
  }

  return (
    <ul className="divide-y divide-gray-200 dark:divide-gray-700">
      {artifacts.map((artifact) => {
        const isSelected = artifact.id === selectedId;
        return (
          <li key={artifact.id}>
            <button
              type="button"
              onClick={() => onSelect(artifact)}
              className={`-mx-2 flex w-full flex-col items-start gap-1 rounded-md px-2 py-3 text-left transition-colors ${
                isSelected
                  ? 'bg-blue-50 dark:bg-blue-900/20'
                  : 'hover:bg-gray-50 dark:hover:bg-gray-800'
              }`}
            >
              <div className="flex w-full items-center justify-between gap-2">
                <span className="truncate font-mono text-xs text-gray-500 dark:text-gray-400">
                  {artifact.id}
                </span>
                <span className="shrink-0 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                  {artifact.artifact_type}
                </span>
              </div>
              <span className="line-clamp-2 text-sm font-medium text-gray-900 dark:text-white">
                {artifactTitle(artifact)}
              </span>
              <div className="flex w-full items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
                <span>score {artifact.effectiveness_score.toFixed(2)}</span>
                <span>used {artifact.usage_count}×</span>
                {artifact.domain && <span>· {artifact.domain}</span>}
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
