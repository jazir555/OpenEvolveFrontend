import { useState } from 'react';
import { cn } from '@/lib/utils';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';
import { BubbleInput } from '../bubblelab';

export interface Artifact {
  id: string;
  name: string;
  type: string;
  created: string;
  status: string;
  fitness?: number;
  generation?: number;
}

interface ArtifactTableProps {
  artifacts: Artifact[];
  onRowClick?: (artifact: Artifact) => void;
  className?: string;
}

function ArtifactTableBase({ artifacts, onRowClick, className }: ArtifactTableProps) {
  const [sort, setSort] = useState<{ key: string; dir: 'asc' | 'desc' }>({
    key: 'created',
    dir: 'desc',
  });
  const [filter, setFilter] = useState('');
  const formatDate = (value: string) => {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString();
  };

  const handleSort = (key: string) => {
    setSort((prev) => ({
      key,
      dir: prev.key === key && prev.dir === 'asc' ? 'desc' : 'asc',
    }));
  };

  const filtered = artifacts.filter((artifact) => {
    const name = (artifact.name || '').toLowerCase();
    const type = (artifact.type || '').toLowerCase();
    const query = filter.toLowerCase();
    return name.includes(query) || type.includes(query);
  });

  const sorted = [...filtered].sort((a, b) => {
    const aValue = a[sort.key as keyof Artifact];
    const bValue = b[sort.key as keyof Artifact];

    if (typeof aValue === 'string' && typeof bValue === 'string') {
      const cmp = aValue.localeCompare(bValue);
      return sort.dir === 'asc' ? cmp : -cmp;
    }

    if (typeof aValue === 'number' && typeof bValue === 'number') {
      const cmp = aValue - bValue;
      return sort.dir === 'asc' ? cmp : -cmp;
    }

    return 0;
  });

  const renderSortIcon = (key: string) => {
    if (sort.key !== key) return null;
    return sort.dir === 'asc' ? '^' : 'v';
  };

  return (
    <div className={cn('artifact-table', className)}>
      <div className="mb-4">
        <BubbleInput
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter artifacts..."
        />
      </div>

      <div className="overflow-x-auto border border-gray-200 rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th
                onClick={() => handleSort('name')}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              >
                Name {renderSortIcon('name')}
              </th>
              <th
                onClick={() => handleSort('type')}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              >
                Type {renderSortIcon('type')}
              </th>
              <th
                onClick={() => handleSort('status')}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              >
                Status {renderSortIcon('status')}
              </th>
              <th
                onClick={() => handleSort('fitness')}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              >
                Fitness {renderSortIcon('fitness')}
              </th>
              <th
                onClick={() => handleSort('generation')}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              >
                Generation {renderSortIcon('generation')}
              </th>
              <th
                onClick={() => handleSort('created')}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              >
                Created {renderSortIcon('created')}
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sorted.map((artifact) => (
              <tr
                key={artifact.id}
                onClick={() => onRowClick?.(artifact)}
                className={cn(
                  'hover:bg-gray-50 transition-colors',
                  onRowClick && 'cursor-pointer'
                )}
              >
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">{artifact.name}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-500">{artifact.type}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={cn(
                    'inline-flex px-2 py-1 text-xs font-semibold rounded-full',
                    artifact.status === 'success' && 'bg-green-100 text-green-800',
                    artifact.status === 'pending' && 'bg-yellow-100 text-yellow-800',
                    artifact.status === 'failed' && 'bg-red-100 text-red-800'
                  )}>
                    {artifact.status}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">
                    {Number.isFinite(Number(artifact.fitness))
                      ? Number(artifact.fitness).toFixed(4)
                      : 'N/A'}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-500">{artifact.generation || 'N/A'}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-500">
                    {formatDate(artifact.created)}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {sorted.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            <p>No artifacts found</p>
          </div>
        )}
      </div>
    </div>
  );
}

export const ArtifactTable = withComponentBoundary(ArtifactTableBase, 'ArtifactTable');
