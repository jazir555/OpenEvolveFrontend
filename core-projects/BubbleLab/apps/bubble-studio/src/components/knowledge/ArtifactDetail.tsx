/**
 * ArtifactDetail
 * Detailed view of a single knowledge artifact fetched via
 * `getKnowledgeArtifact`. Shows the structured content, provenance and
 * related artifacts.
 */

import type { KnowledgeArtifact } from '@/types/openevolve';
import { artifactTitle } from './ArtifactList';

interface ArtifactDetailProps {
  artifact: KnowledgeArtifact | null;
  loading: boolean;
  error: string | null;
}

function formatTimestamp(value: string | number): string {
  const date = typeof value === 'number' ? new Date(value) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

export function ArtifactDetail({ artifact, loading, error }: ArtifactDetailProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-10 text-sm text-gray-500 dark:text-gray-400">
        Loading artifact…
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md border border-red-300 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
        {error}
      </div>
    );
  }

  if (!artifact) {
    return (
      <div className="flex items-center justify-center py-10 text-center text-sm text-gray-500 dark:text-gray-400">
        Select an artifact to view its details.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center gap-2">
          <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700 dark:bg-gray-700 dark:text-gray-300">
            {artifact.artifact_type}
          </span>
          {artifact.domain && (
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {artifact.domain}
            </span>
          )}
        </div>
        <h3 className="mt-2 text-base font-semibold text-gray-900 dark:text-white">
          {artifactTitle(artifact)}
        </h3>
        <p className="mt-1 font-mono text-xs text-gray-500 dark:text-gray-400">
          {artifact.id}
        </p>
      </div>

      <dl className="grid grid-cols-2 gap-3 text-sm">
        <Meta label="Source Workflow" value={artifact.source_workflow_id} />
        <Meta label="Extracted" value={formatTimestamp(artifact.extraction_timestamp)} />
        <Meta label="Effectiveness" value={artifact.effectiveness_score.toFixed(2)} />
        <Meta label="Usage Count" value={String(artifact.usage_count)} />
        {artifact.problem_type && (
          <Meta label="Problem Type" value={artifact.problem_type} />
        )}
      </dl>

      <div>
        <h4 className="mb-1 text-sm font-medium text-gray-700 dark:text-gray-300">
          Content
        </h4>
        <pre className="thin-scrollbar max-h-72 overflow-auto rounded-md border border-gray-200 bg-gray-50 p-3 text-xs text-gray-800 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200">
          {renderContent(artifact.content)}
        </pre>
      </div>

      {artifact.related_artifacts && artifact.related_artifacts.length > 0 && (
        <div>
          <h4 className="mb-1 text-sm font-medium text-gray-700 dark:text-gray-300">
            Related Artifacts
          </h4>
          <ul className="flex flex-wrap gap-2">
            {artifact.related_artifacts.map((rel) => (
              <li
                key={rel}
                className="rounded-full bg-gray-100 px-2 py-1 font-mono text-xs text-gray-600 dark:bg-gray-700 dark:text-gray-300"
              >
                {rel}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-gray-500 dark:text-gray-400">{label}</dt>
      <dd className="mt-0.5 break-words text-gray-900 dark:text-white">
        {value}
      </dd>
    </div>
  );
}

function renderContent(content: KnowledgeArtifact['content']): string {
  if (typeof content === 'string') return content;
  try {
    return JSON.stringify(content, null, 2);
  } catch {
    return String(content);
  }
}
