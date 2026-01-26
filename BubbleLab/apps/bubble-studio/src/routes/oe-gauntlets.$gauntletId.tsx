/**
 * Gauntlet Detail Route
 * View and edit individual gauntlet configuration
 */

import { useParams, useNavigate } from '@tanstack/react-router';
import { useGauntlet } from '../hooks/use-gauntlets-api';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge } from '../components/common/Badge';
import { LoadingSpinner } from '../components/common/Spinner';
import { Alert } from '../components/common/Alert';

export default function GauntletDetailPage() {
  const { gauntletId } = useParams({ from: '/gauntlets/$gauntletId' });
  const navigate = useNavigate();
  const { data: gauntlet, isLoading, error } = useGauntlet(gauntletId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error || !gauntlet) {
    return (
      <Alert variant="error" title="Error loading gauntlet">
        {error instanceof Error ? error.message : 'Gauntlet not found'}
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              {gauntlet.name}
            </h1>
            <Badge variant={gauntlet.is_active ? 'success' : 'secondary'}>
              {gauntlet.is_active ? 'Active' : 'Inactive'}
            </Badge>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            {gauntlet.description || 'No description'}
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="secondary" onClick={() => navigate({ to: '/gauntlets' })}>
            Back to Gauntlets
          </Button>
          <Button>Edit Gauntlet</Button>
        </div>
      </div>

      {/* Gauntlet Details */}
      <Card>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Gauntlet Configuration
        </h2>
        <dl className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-2">
          <div>
            <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Gauntlet Name
            </dt>
            <dd className="mt-1 text-sm text-gray-900 dark:text-white">
              {gauntlet.name}
            </dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Status
            </dt>
            <dd className="mt-1">
              <Badge variant={gauntlet.is_active ? 'success' : 'secondary'}>
                {gauntlet.is_active ? 'Active' : 'Inactive'}
              </Badge>
            </dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Rounds
            </dt>
            <dd className="mt-1 text-sm text-gray-900 dark:text-white">
              {gauntlet.rounds.length}
            </dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Created
            </dt>
            <dd className="mt-1 text-sm text-gray-900 dark:text-white">
              {new Date(gauntlet.created_at).toLocaleDateString()}
            </dd>
          </div>
        </dl>
      </Card>

      {/* Gauntlet Rounds */}
      <Card>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Gauntlet Rounds ({gauntlet.rounds.length})
        </h2>
        <div className="space-y-4">
          {gauntlet.rounds.map((round, index) => (
            <div
              key={index}
              className="border border-gray-300 dark:border-gray-700 rounded-lg p-4"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="text-base font-medium text-gray-900 dark:text-white">
                    Round {index + 1}: {round.name}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {round.description || 'No description'}
                  </p>
                </div>
                <Badge variant="blue">{round.round_type}</Badge>
              </div>
              <dl className="grid grid-cols-1 gap-x-4 gap-y-2 sm:grid-cols-3">
                <div>
                  <dt className="text-xs font-medium text-gray-500 dark:text-gray-400">
                    Quorum
                  </dt>
                  <dd className="mt-1 text-sm text-gray-900 dark:text-white">
                    {round.quorum}%
                  </dd>
                </div>
                <div>
                  <dt className="text-xs font-medium text-gray-500 dark:text-gray-400">
                    Confidence Threshold
                  </dt>
                  <dd className="mt-1 text-sm text-gray-900 dark:text-white">
                    {(round.confidence_threshold * 100).toFixed(0)}%
                  </dd>
                </div>
                <div>
                  <dt className="text-xs font-medium text-gray-500 dark:text-gray-400">
                    Max Iterations
                  </dt>
                  <dd className="mt-1 text-sm text-gray-900 dark:text-white">
                    {round.max_iterations}
                  </dd>
                </div>
              </dl>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
