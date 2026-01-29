import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { BubbleBadge, BubbleButton, BubbleField, BubbleInput, BubbleSelect } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

export interface Artifact {
  id: string;
  name: string;
  type: string;
  description?: string;
  created: string;
  updated: string;
  version: number;
  tags: string[];
}

interface ArtifactListProps {
  artifacts: Artifact[];
  onArtifactSelect?: (artifact: Artifact) => void;
  onArtifactDelete?: (artifactId: string) => void;
  onVerify?: (text: string) => void;
  onDistill?: (id: string) => void;
  className?: string;
}

function ArtifactListBase({
  artifacts = [],
  onArtifactSelect,
  onArtifactDelete,
  onVerify,
  onDistill,
  className,
}: ArtifactListProps) {
  const [search, setSearch] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [selectedTag, setSelectedTag] = useState<string>('all');

  const types = Array.from(new Set(artifacts.map((a) => a.type || 'Unknown')));
  const tags = Array.from(new Set(artifacts.flatMap((a) => a.tags || [])));

  useEffect(() => {
    if (selectedType !== 'all' && !types.includes(selectedType)) {
      setSelectedType('all');
    }
    if (selectedTag !== 'all' && !tags.includes(selectedTag)) {
      setSelectedTag('all');
    }
  }, [selectedType, selectedTag, types, tags]);

  const filtered = artifacts.filter((artifact) => {
    const name = artifact.name || '';
    const description = artifact.description || '';
    const artifactTags = artifact.tags || [];
    const matchesSearch =
      name.toLowerCase().includes(search.toLowerCase()) ||
      description.toLowerCase().includes(search.toLowerCase());

    const matchesType = selectedType === 'all' || (artifact.type || 'Unknown') === selectedType;
    const matchesTag = selectedTag === 'all' || artifactTags.includes(selectedTag);

    return matchesSearch && matchesType && matchesTag;
  });

  return (
    <div className={cn('artifact-list', className)}>
      <div className="mb-6 space-y-4">
        <BubbleInput
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search artifacts..."
        />

        <div className="grid gap-4 md:grid-cols-2">
          <BubbleField label="Type">
            <BubbleSelect
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
            >
              <option value="all">All Types</option>
              {types.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </BubbleSelect>
          </BubbleField>

          <BubbleField label="Tag">
            <BubbleSelect
              value={selectedTag}
              onChange={(e) => setSelectedTag(e.target.value)}
            >
              <option value="all">All Tags</option>
              {tags.map((tag) => (
                <option key={tag} value={tag}>
                  {tag}
                </option>
              ))}
            </BubbleSelect>
          </BubbleField>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((artifact) => (
          <div
            key={artifact.id}
            className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-lg transition-all cursor-pointer group flex flex-col"
            onClick={() => onArtifactSelect?.(artifact)}
          >
            <div className="flex items-start justify-between mb-2">
              <h3 className="text-lg font-semibold text-gray-900 line-clamp-1">{artifact.name || 'Untitled Artifact'}</h3>
              <BubbleButton
                onClick={(e) => {
                  e.stopPropagation();
                  onArtifactDelete?.(artifact.id);
                }}
                variant="ghost"
                className="px-2 py-1 text-red-600 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                Delete
              </BubbleButton>
            </div>

              {artifact.description && (
                <p className="text-sm text-gray-600 mb-3 line-clamp-2 flex-grow">{artifact.description}</p>
              )}

            <div className="flex items-center gap-2 mb-2">
              <BubbleBadge tone="info">{artifact.type || 'Unknown'}</BubbleBadge>
              <span className="text-xs text-gray-500">v{artifact.version ?? 1}</span>
            </div>

            <div className="flex flex-wrap gap-1 mb-4">
              {(artifact.tags || []).slice(0, 3).map((tag) => (
                <BubbleBadge key={tag} tone="neutral">
                  {tag}
                </BubbleBadge>
              ))}
              {(artifact.tags || []).length > 3 && (
                <span className="text-xs text-gray-500">+{(artifact.tags || []).length - 3}</span>
              )}
              {!(artifact.tags || []).length && (
                <BubbleBadge tone="neutral">No tags</BubbleBadge>
              )}
            </div>

            <div className="flex gap-2 mb-3">
              <BubbleButton 
                size="sm" 
                variant="secondary" 
                onClick={(e) => { e.stopPropagation(); onDistill?.(artifact.id); }}
                className="flex-1 text-[10px]"
              >
                Distill Skill
              </BubbleButton>
              <BubbleButton 
                size="sm" 
                variant="secondary" 
                onClick={(e) => { e.stopPropagation(); onVerify?.(artifact.description || artifact.name); }}
                className="flex-1 text-[10px]"
              >
                Verify
              </BubbleButton>
            </div>

            <div className="text-[10px] text-gray-400 mt-auto pt-2 border-t border-gray-50">
              Updated {new Date(artifact.updated || artifact.created || new Date().toISOString()).toLocaleDateString()}
            </div>
          </div>
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <p>No artifacts found</p>
        </div>
      )}
    </div>
  );
}

export const ArtifactList = withComponentBoundary(ArtifactListBase, 'ArtifactList');
