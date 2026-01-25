import { useState } from 'react';
import { cn } from '@/lib/utils';
import { BubbleBadge, BubbleButton, BubbleCard } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface ArtifactVersion {
  version: number;
  content: string;
  created: string;
  created_by: string;
  comment?: string;
}

export interface ArtifactDetail {
  id: string;
  name?: string;
  type?: string;
  description?: string;
  content?: string;
  tags?: string[];
  versions?: ArtifactVersion[];
  current_version?: number;
  created?: string;
  updated?: string;
  created_by?: string;
}

interface ArtifactDetailProps {
  artifact: ArtifactDetail;
  onEdit?: () => void;
  onDelete?: () => void;
  onVersionRestore?: (version: number) => void;
  className?: string;
}

function ArtifactDetailBase({
  artifact,
  onEdit,
  onDelete,
  onVersionRestore,
  className,
}: ArtifactDetailProps) {
  const [activeTab, setActiveTab] = useState<'content' | 'versions' | 'metadata'>('content');
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);

  const versions = artifact.versions || [];
  const currentVersion = artifact.current_version ?? versions[0]?.version;
  const currentVersionData = versions.find(
    (v) => v.version === (selectedVersion || currentVersion)
  );
  const tags = artifact.tags || [];
  const formatDate = (value?: string) => {
    if (!value) return 'Unknown';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString();
  };

  return (
    <div className={cn('artifact-detail', className)}>
      <BubbleCard
        title={artifact.name || 'Untitled Artifact'}
        description={artifact.description || 'No description provided.'}
        actions={
          <div className="flex flex-wrap gap-2">
            {onEdit && (
              <BubbleButton onClick={onEdit}>Edit</BubbleButton>
            )}
            {onDelete && (
              <BubbleButton onClick={onDelete} variant="secondary">
                Delete
              </BubbleButton>
            )}
          </div>
        }
      >
        <div className="flex flex-wrap gap-4 text-sm text-gray-600">
          <div>
            <span className="font-medium">Type:</span> {artifact.type || 'Unknown'}
          </div>
          <div>
            <span className="font-medium">Version:</span> {currentVersion ?? 'Unknown'}
          </div>
          <div>
            <span className="font-medium">Created:</span> {formatDate(artifact.created)}
          </div>
          <div>
            <span className="font-medium">Updated:</span> {formatDate(artifact.updated)}
          </div>
          <div>
            <span className="font-medium">By:</span> {artifact.created_by || 'Unknown'}
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mt-3">
          {tags.map((tag) => (
            <BubbleBadge key={tag} tone="neutral">
              {tag}
            </BubbleBadge>
          ))}
          {!tags.length && (
            <BubbleBadge tone="neutral">No tags</BubbleBadge>
          )}
        </div>
      </BubbleCard>

      <div className="mt-4 flex flex-wrap gap-2">
        {(['content', 'versions', 'metadata'] as const).map((tab) => (
          <BubbleButton
            key={tab}
            onClick={() => setActiveTab(tab)}
            variant={activeTab === tab ? 'primary' : 'secondary'}
          >
            {tab}
          </BubbleButton>
        ))}
      </div>

      <div className="mt-4 rounded-lg border border-gray-200 bg-white p-6">
        {activeTab === 'content' && (
          <div>
            {selectedVersion && (
              <div className="mb-4 bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-2 rounded-lg flex items-center justify-between">
                <span>Viewing version {selectedVersion}</span>
                <BubbleButton onClick={() => setSelectedVersion(null)} variant="ghost">
                  Back to latest
                </BubbleButton>
              </div>
            )}
            <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm font-mono whitespace-pre-wrap">
              {currentVersionData?.content || artifact.content || 'No content available.'}
            </pre>
          </div>
        )}

        {activeTab === 'versions' && (
          <div className="space-y-3">
            <h3 className="text-lg font-semibold text-gray-900">Version History</h3>
            {versions.map((version) => (
              <div
                key={version.version}
                className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50"
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="font-medium text-gray-900">
                      Version {version.version}
                    </div>
                    <div className="text-sm text-gray-600">
                      {formatDate(version.created)} by {version.created_by}
                    </div>
                    {version.comment && (
                      <div className="text-sm text-gray-500 mt-1">{version.comment}</div>
                    )}
                  </div>
                  {version.version !== currentVersion && (
                    <BubbleButton
                      onClick={() => setSelectedVersion(version.version)}
                      variant="ghost"
                    >
                      View
                    </BubbleButton>
                  )}
                </div>
                {onVersionRestore && version.version !== currentVersion && (
                  <BubbleButton onClick={() => onVersionRestore(version.version)} variant="secondary">
                    Restore
                  </BubbleButton>
                )}
              </div>
            ))}
            {!versions.length && (
              <p className="text-sm text-gray-500">No versions available.</p>
            )}
          </div>
        )}

        {activeTab === 'metadata' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">Metadata</h3>
            <dl className="grid grid-cols-2 gap-4">
              <div>
                <dt className="text-sm font-medium text-gray-600">ID</dt>
                <dd className="text-sm text-gray-900 font-mono">{artifact.id}</dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-600">Type</dt>
                <dd className="text-sm text-gray-900">{artifact.type || 'Unknown'}</dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-600">Current Version</dt>
                <dd className="text-sm text-gray-900">{currentVersion ?? 'Unknown'}</dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-600">Total Versions</dt>
                <dd className="text-sm text-gray-900">{versions.length}</dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-600">Created</dt>
                <dd className="text-sm text-gray-900">
                  {formatDate(artifact.created)}
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-600">Updated</dt>
                <dd className="text-sm text-gray-900">
                  {formatDate(artifact.updated)}
                </dd>
              </div>
            </dl>
          </div>
        )}
      </div>
    </div>
  );
}

export const ArtifactDetail = withComponentBoundary(ArtifactDetailBase, 'ArtifactDetail');
